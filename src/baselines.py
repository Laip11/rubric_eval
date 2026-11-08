"""Offline adapters for the original CheckEval, DnA-Eval and RubricHub algorithms."""
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from async_vllm import AsyncLLMPool
from paor_pipeline import load_json, parse_winner, save_json, truncate_text
from paor_prompts import (DIRECT_PAIR_SYSTEM, DIRECT_PAIR_USER, DIRECT_POINT_SYSTEM,
                          DIRECT_POINT_USER, DYNAMIC_PAIR_SYSTEM, DYNAMIC_PAIR_USER,
                          DYNAMIC_POINT_SYSTEM, DYNAMIC_POINT_USER, DYNAMIC_RUBRIC_SYSTEM,
                          DYNAMIC_RUBRIC_USER)

ORIGINAL_SRC = Path(__file__).resolve().parents[1] / "generating_dynamic_rubric" / "src"
if str(ORIGINAL_SRC) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_SRC))

from eval_checkeval import (BENCHMARK_DESCRIPTIONS, CHECKEVAL_SYSTEM_PROMPT,
                            CHECKEVAL_USER_TEMPLATE, CHECKLIST_GEN_SYSTEM_PROMPT,
                            CHECKLIST_GEN_USER_PROMPT, compute_checkeval_score,
                            parse_checklist, parse_checkeval_answers)
from eval_dna import (ASPECT_GEN_SYSTEM, ASPECT_GEN_USER, EVAL_PAIRWISE_SYSTEM,
                      EVAL_PAIRWISE_USER, EVAL_POINTWISE_SYSTEM, EVAL_POINTWISE_USER,
                      WEIGHT_GEN_SYSTEM, WEIGHT_GEN_USER, aggregate_pairwise,
                      parse_aspects, parse_pairwise_scores, parse_pointwise_score,
                      parse_weights)
from eval_rubrichub import (GRADER_SYSTEM, GRADER_USER, REFERENCE_GEN_SYSTEM,
                            REFERENCE_GEN_USER, RUBRIC_GEN_SYSTEM, RUBRIC_GEN_USER,
                            compute_rubrichub_score, parse_grader_output, parse_rubric)

BASELINE_DESCRIPTIONS = {
    **BENCHMARK_DESCRIPTIONS,
    "rewardbench2": "unseen human prompts covering factual accuracy, precise instruction following, mathematics, safe compliance or refusal, and on-topic helpfulness",
    "helpsteer3": "general, STEM, code, and multilingual assistant responses in real single-turn and multi-turn conversations",
    "rewardbench": "prompt-response pairs spanning general chat, adversarial instruction following, safety and appropriate refusals, mathematics, and code",
    "judgebench_gpt": "challenging objective general knowledge, reasoning, mathematics, and coding responses",
    "judgebench_claude": "challenging objective general knowledge, reasoning, mathematics, and coding responses",
}


def parse_point_score(text: str) -> float | None:
    match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
    if not match:
        match = re.search(r"\b([1-5](?:\.\d+)?)\b", text)
    if not match:
        return None
    value = float(match.group(1))
    return value if 1 <= value <= 5 else None


def tie_prediction(pair_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).digest()
    return "A" if digest[0] % 2 == 0 else "B"


class BaselineRunner:
    def __init__(self, pool: AsyncLLMPool, cache_root: str | Path, *, tokenizer: Any = None,
                 judge_max_tokens: int = 512, query_max_tokens: int = 4096,
                 response_max_tokens: int = 4096) -> None:
        self.pool = pool
        self.cache_root = Path(cache_root)
        self.tokenizer = tokenizer
        self.judge_max_tokens = judge_max_tokens
        self.query_max_tokens = query_max_tokens
        self.response_max_tokens = response_max_tokens
        self.benchmark = ""

    def _tag(self, method: str, seed: int, qid: str) -> str:
        return f"{method}:{self.benchmark}:seed{seed}:{qid}"

    def _request(self, system: str, user: str, tag: str, *, temperature: float = 0.0,
                 max_tokens: int | None = None, seed: int | None = None, role: str = "strong") -> dict[str, Any]:
        return self.pool.request(system, user, role=role, temperature=temperature,
                                 max_tokens=max_tokens or self.judge_max_tokens, seed=seed, tag=tag)

    def _query(self, group: dict[str, Any]) -> str:
        return truncate_text(group["query"], self.tokenizer, self.query_max_tokens)

    def _response(self, text: str) -> str:
        return truncate_text(text, self.tokenizer, self.response_max_tokens)

    def _result(self, group: dict[str, Any], scores: dict[str, float | None], *, method: str,
                seed: int, diagnostics: dict[str, Any] | None = None,
                pair_predictions: dict[str, str | None] | None = None,
                pair_details: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        pair_predictions = pair_predictions or {}
        pair_details = pair_details or {}
        pairs = []
        for pair in group["pairs"]:
            details = dict(pair_details.get(pair["id"], {}))
            if pair["id"] in pair_predictions:
                prediction = pair_predictions[pair["id"]]
            elif pair["id"] in details:
                prediction = details["prediction"]
            else:
                a, b = scores.get(pair["a"]), scores.get(pair["b"])
                if a is None or b is None:
                    prediction = None
                elif a > b:
                    prediction = "A"
                elif b > a:
                    prediction = "B"
                else:
                    prediction = tie_prediction(pair["id"], seed)
                    details["model_score_tie"] = True
            details.setdefault("model_score_tie", False)
            details.setdefault("order_disagreement", False)
            details.setdefault("parse_error", prediction is None and not details["order_disagreement"])
            pairs.append({"id": pair["id"], "gold": pair["gold"], "prediction": prediction,
                          "fallback": details.pop("fallback", False), "a": pair["a"], "b": pair["b"],
                          **details})
        items = [{"id": item["id"], "response_id": item["response_id"], "score": scores.get(item["response_id"]),
                  "human_score": item["human_score"], "model": item.get("model", "unknown")} for item in group["items"]]
        return {"qid": group["qid"], "kind": group["kind"], "scores": scores, "pairs": pairs, "items": items,
                "diagnostic": diagnostics or {}, "cost": self.pool.tag_snapshot(self._tag(method, seed, group["qid"]))}

    def _finish_cost(self, result: dict[str, Any], method: str, seed: int) -> None:
        result["cost"] = self.pool.tag_snapshot(self._tag(method, seed, result["qid"]))

    async def _direct(self, benchmark: str, groups: list[dict[str, Any]], seed: int,
                      repeats: int | dict[str, int] = 2, method: str = "direct") -> list[dict[str, Any]]:
        requests = []
        metadata = []
        sampling_temperature = 0.7 if method == "direct_k" else 0.0
        repeat_for = lambda group: repeats.get(group["qid"], 1) if isinstance(repeats, dict) else repeats
        for group in groups:
            tag = self._tag(method, seed, group["qid"])
            if group["kind"] == "pointwise":
                for response in group["responses"]:
                    for repeat in range(1 if method == "direct" else repeat_for(group)):
                        user = DIRECT_POINT_USER.format(query=self._query(group), response=self._response(response["text"]))
                        requests.append(self._request(DIRECT_POINT_SYSTEM, user, tag, temperature=sampling_temperature,
                                                      max_tokens=256, seed=seed * 1000 + repeat))
                        metadata.append((group["qid"], response["id"], repeat, False, True))
            else:
                response_map = {x["id"]: x["text"] for x in group["responses"]}
                for pair in group["pairs"]:
                    for repeat in range(repeat_for(group)):
                        reverse = repeat % 2 == 1
                        a = response_map[pair["b"] if reverse else pair["a"]]
                        b = response_map[pair["a"] if reverse else pair["b"]]
                        user = DIRECT_PAIR_USER.format(query=self._query(group), response_a=self._response(a),
                                                       response_b=self._response(b))
                        requests.append(self._request(DIRECT_PAIR_SYSTEM, user, tag, temperature=sampling_temperature,
                                                      max_tokens=256, seed=seed * 1000 + repeat))
                        metadata.append((group["qid"], pair["id"], repeat, reverse, False))
        raw = await self.pool.complete_many(requests)
        point_votes: dict[tuple[str, str], list[float]] = {}
        pair_votes: dict[tuple[str, str], list[float]] = {}
        pair_raw: dict[tuple[str, str], list[str]] = {}
        for (qid, identifier, repeat, reverse, is_pointwise), result in zip(metadata, raw):
            if is_pointwise:
                score = parse_point_score(result.text)
                if score is not None:
                    point_votes.setdefault((qid, identifier), []).append(score)
            else:
                winner = parse_winner(result.text)
                if winner is not None:
                    value = (winner == "A") if not reverse else (winner == "B")
                    pair_votes.setdefault((qid, identifier), []).append(float(value))
                if method == "direct":
                    pair_raw.setdefault((qid, identifier), []).append(result.text)
        output = []
        for group in groups:
            if group["kind"] == "pointwise":
                scores = {r["id"]: (float(np.mean(point_votes[(group["qid"], r["id"])]))
                                    if point_votes.get((group["qid"], r["id"])) else None)
                          for r in group["responses"]}
                result = self._result(group, scores, method=method, seed=seed)
            else:
                predictions = {}
                details = {}
                for pair in group["pairs"]:
                    votes = pair_votes.get((group["qid"], pair["id"]), [])
                    expected = repeat_for(group)
                    detail: dict[str, Any] = {"order_votes": votes, "expected_votes": expected,
                                              "parse_error": len(votes) < expected if method == "direct" else not votes}
                    if method == "direct" and len(votes) == 2 and votes[0] != votes[1]:
                        prediction = None
                        detail["order_disagreement"] = True
                        detail["parse_error"] = False
                    elif not votes or (method == "direct" and len(votes) != 2):
                        prediction = None
                    elif np.mean(votes) > 0.5:
                        prediction = "A"
                    elif np.mean(votes) < 0.5:
                        prediction = "B"
                    else:
                        prediction = tie_prediction(pair["id"], seed)
                        detail["model_score_tie"] = True
                    if method == "direct":
                        detail["raw_orders"] = pair_raw.get((group["qid"], pair["id"]), [])
                    predictions[pair["id"]] = prediction
                    details[pair["id"]] = detail
                result = self._result(group, {}, method=method, seed=seed,
                                      pair_predictions=predictions, pair_details=details)
            self._finish_cost(result, method, seed)
            output.append(result)
        return output

    async def _dynamic(self, benchmark: str, groups: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
        requests = [self._request(DYNAMIC_RUBRIC_SYSTEM, DYNAMIC_RUBRIC_USER.format(query=self._query(group)),
                                  self._tag("dynamic", seed, group["qid"]), max_tokens=1024,
                                  seed=seed, role="generator") for group in groups]
        raw = await self.pool.complete_many(requests)
        rubrics = {group["qid"]: result.text.strip() for group, result in zip(groups, raw)}
        requests = []
        metadata = []
        for group in groups:
            tag = self._tag("dynamic", seed, group["qid"])
            if group["kind"] == "pointwise":
                for response in group["responses"]:
                    user = DYNAMIC_POINT_USER.format(query=self._query(group), rubric=rubrics[group["qid"]],
                                                     response=self._response(response["text"]))
                    requests.append(self._request(DYNAMIC_POINT_SYSTEM, user, tag, max_tokens=256, seed=seed))
                    metadata.append((group["qid"], response["id"], False, True))
            else:
                response_map = {x["id"]: x["text"] for x in group["responses"]}
                for pair in group["pairs"]:
                    for reverse in (False, True):
                        a = response_map[pair["b"] if reverse else pair["a"]]
                        b = response_map[pair["a"] if reverse else pair["b"]]
                        user = DYNAMIC_PAIR_USER.format(query=self._query(group), rubric=rubrics[group["qid"]],
                                                        response_a=self._response(a), response_b=self._response(b))
                        requests.append(self._request(DYNAMIC_PAIR_SYSTEM, user, tag, max_tokens=256,
                                                      seed=seed + int(reverse)))
                        metadata.append((group["qid"], pair["id"], reverse, False))
        raw = await self.pool.complete_many(requests)
        points = {}
        votes: dict[tuple[str, str], list[float]] = {}
        raw_orders: dict[tuple[str, str], list[str]] = {}
        for (qid, identifier, reverse, is_pointwise), result in zip(metadata, raw):
            if is_pointwise:
                score = parse_point_score(result.text)
                if score is not None:
                    points[(qid, identifier)] = score
            else:
                winner = parse_winner(result.text)
                if winner is not None:
                    votes.setdefault((qid, identifier), []).append(float((winner == "A") if not reverse else (winner == "B")))
                raw_orders.setdefault((qid, identifier), []).append(result.text)
        output = []
        for group in groups:
            if group["kind"] == "pointwise":
                scores = {r["id"]: points.get((group["qid"], r["id"])) for r in group["responses"]}
                result = self._result(group, scores, method="dynamic", seed=seed,
                                      diagnostics={"rubric": rubrics[group["qid"]]})
            else:
                predictions = {}
                details = {}
                for pair in group["pairs"]:
                    values = votes.get((group["qid"], pair["id"]), [])
                    detail: dict[str, Any] = {"order_votes": values, "raw_orders": raw_orders.get((group["qid"], pair["id"]), []),
                                              "parse_error": len(values) != 2}
                    if len(values) == 2 and values[0] != values[1]:
                        prediction = None
                        detail["order_disagreement"] = True
                        detail["parse_error"] = False
                    elif len(values) != 2:
                        prediction = None
                    else:
                        prediction = "A" if values[0] == 1.0 else "B"
                    predictions[pair["id"]] = prediction
                    details[pair["id"]] = detail
                result = self._result(group, {}, method="dynamic", seed=seed,
                                      diagnostics={"rubric": rubrics[group["qid"]]},
                                      pair_predictions=predictions, pair_details=details)
            self._finish_cost(result, "dynamic", seed)
            output.append(result)
        return output

    async def _checklist(self, benchmark: str, groups: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
        checklist_path = self.cache_root / "baselines" / "checkeval" / f"{benchmark}.json"
        questions = load_json(checklist_path, None)
        if not questions:
            description = BASELINE_DESCRIPTIONS[benchmark]
            system = CHECKLIST_GEN_SYSTEM_PROMPT.format(num_questions=20, benchmark_description=description)
            user = CHECKLIST_GEN_USER_PROMPT.format(num_questions=20, benchmark_description=description)
            request = self._request(system, user, f"checkeval:checklist:{benchmark}", max_tokens=1024,
                                    seed=seed, role="generator")
            raw = (await self.pool.complete_many([request]))[0]
            questions = parse_checklist(raw.text)
            if not questions:
                questions = [line.strip() for line in raw.text.strip().split("\n") if line.strip() and "?" in line]
            if not questions:
                raise RuntimeError(f"CheckEval produced no checklist questions for {benchmark}")
            save_json(checklist_path, questions)
        question_text = "\n".join(f"Q{i + 1}: {q}" for i, q in enumerate(questions))
        requests = []
        metadata = []
        for group in groups:
            for response in group["responses"]:
                user = CHECKEVAL_USER_TEMPLATE.format(prompt=self._query(group), response=self._response(response["text"]),
                                                      questions=question_text)
                requests.append(self._request(CHECKEVAL_SYSTEM_PROMPT, user, self._tag("checkeval", seed, group["qid"]),
                                              max_tokens=256, seed=seed))
                metadata.append((group["qid"], response["id"]))
        raw = await self.pool.complete_many(requests)
        answers = {}
        scores = {}
        raw_outputs = {}
        for (qid, rid), result in zip(metadata, raw):
            values = parse_checkeval_answers(result.text, len(questions))
            answers[(qid, rid)] = values
            scores[(qid, rid)] = compute_checkeval_score(values)
            raw_outputs[(qid, rid)] = result.text
        output = []
        for group in groups:
            values = {r["id"]: scores.get((group["qid"], r["id"])) for r in group["responses"]}
            diagnostics = {"questions": questions,
                           "answers": {r["id"]: answers[(group["qid"], r["id"])] for r in group["responses"]},
                           "raw_answers": {r["id"]: raw_outputs[(group["qid"], r["id"])] for r in group["responses"]}}
            result = self._result(group, values, method="checkeval", seed=seed, diagnostics=diagnostics)
            self._finish_cost(result, "checkeval", seed)
            output.append(result)
        return output

    async def _dna(self, benchmark: str, groups: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
        requests = [self._request(ASPECT_GEN_SYSTEM, ASPECT_GEN_USER.format(num_aspects=3, instruction=self._query(group)),
                                  self._tag("dna", seed, group["qid"]), max_tokens=512,
                                  seed=seed, role="generator") for group in groups]
        raw = await self.pool.complete_many(requests)
        raw_aspects = {group["qid"]: result.text for group, result in zip(groups, raw)}
        aspects = {group["qid"]: parse_aspects(result.text, 3) for group, result in zip(groups, raw)}
        requests = []
        weight_groups = []
        for group in groups:
            values = aspects[group["qid"]]
            if not values:
                continue
            aspect_text = "\n".join(f"{i + 1}. {aspect}" for i, aspect in enumerate(values))
            requests.append(self._request(WEIGHT_GEN_SYSTEM,
                                          WEIGHT_GEN_USER.format(num_aspects=len(values), instruction=self._query(group),
                                                                 aspects=aspect_text),
                                          self._tag("dna", seed, group["qid"]), max_tokens=256,
                                          seed=seed, role="generator"))
            weight_groups.append(group)
        raw = await self.pool.complete_many(requests)
        raw_weights = {group["qid"]: result.text for group, result in zip(weight_groups, raw)}
        weights = {}
        for group in groups:
            n = len(aspects[group["qid"]])
            if not n:
                weights[group["qid"]] = []
                continue
            values = parse_weights(raw_weights[group["qid"]], n)
            weights[group["qid"]] = values or [1.0 / n] * n
        requests = []
        metadata = []
        for group in groups:
            tag = self._tag("dna", seed, group["qid"])
            for aspect_index, aspect in enumerate(aspects[group["qid"]]):
                if group["kind"] == "pointwise":
                    for response in group["responses"]:
                        user = EVAL_POINTWISE_USER.format(instruction=self._query(group),
                                                          response=self._response(response["text"]), aspect=aspect)
                        requests.append(self._request(EVAL_POINTWISE_SYSTEM, user, tag, max_tokens=64, seed=seed))
                        metadata.append((group["qid"], response["id"], None, aspect_index))
                else:
                    response_map = {x["id"]: x["text"] for x in group["responses"]}
                    for pair in group["pairs"]:
                        user = EVAL_PAIRWISE_USER.format(instruction=self._query(group),
                                                         output_1=self._response(response_map[pair["a"]]),
                                                         output_2=self._response(response_map[pair["b"]]), aspect=aspect)
                        requests.append(self._request(EVAL_PAIRWISE_SYSTEM, user, tag, max_tokens=64, seed=seed))
                        metadata.append((group["qid"], pair["id"], pair, aspect_index))
        raw = await self.pool.complete_many(requests)
        point_scores: dict[tuple[str, str], list[float | None]] = {}
        pair_scores: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}
        raw_scores: dict[tuple[str, str], list[str]] = {}
        for (qid, identifier, pair, aspect_index), result in zip(metadata, raw):
            if pair is None:
                values = point_scores.setdefault((qid, identifier), [None] * len(aspects[qid]))
                values[aspect_index] = parse_pointwise_score(result.text)
            else:
                values = pair_scores.setdefault((qid, identifier), [(None, None)] * len(aspects[qid]))
                values[aspect_index] = parse_pairwise_scores(result.text)
            raw_scores.setdefault((qid, identifier), [""] * len(aspects[qid]))[aspect_index] = result.text
        output = []
        for group in groups:
            n = len(aspects[group["qid"]])
            if group["kind"] == "pointwise":
                scores = {}
                for response in group["responses"]:
                    values = point_scores.get((group["qid"], response["id"]), [None] * n)
                    scores[response["id"]] = (float(sum(w * s for w, s in zip(weights[group["qid"]], values)))
                                               if n and all(s is not None for s in values) else None)
                result = self._result(group, scores, method="dna", seed=seed,
                                      diagnostics={"aspects": aspects[group["qid"]], "weights": weights[group["qid"]]})
            else:
                predictions = {}
                details = {}
                evaluations = {}
                for pair in group["pairs"]:
                    values = pair_scores.get((group["qid"], pair["id"]), [(None, None)] * n)
                    valid = n > 0 and all(a is not None and b is not None for a, b in values)
                    detail: dict[str, Any] = {"parse_error": not valid}
                    if valid:
                        ws1, ws2, preference = aggregate_pairwise(weights[group["qid"]],
                                                                  [x[0] for x in values], [x[1] for x in values])
                        prediction = "A" if preference == 1 else "B" if preference == 2 else tie_prediction(pair["id"], seed)
                        detail.update({"weighted_score_a": ws1, "weighted_score_b": ws2,
                                       "model_score_tie": preference is None})
                    else:
                        prediction = None
                    predictions[pair["id"]] = prediction
                    details[pair["id"]] = detail
                    evaluations[pair["id"]] = {"aspect_scores": values,
                                               "raw": raw_scores.get((group["qid"], pair["id"]), [])}
                result = self._result(group, {}, method="dna", seed=seed,
                                      diagnostics={"aspects": aspects[group["qid"]], "weights": weights[group["qid"]],
                                                   "raw_aspects": raw_aspects[group["qid"]],
                                                   "raw_weights": raw_weights.get(group["qid"], ""),
                                                   "pair_evaluations": evaluations},
                                      pair_predictions=predictions, pair_details=details)
            self._finish_cost(result, "dna", seed)
            output.append(result)
        return output

    async def _rubrichub(self, benchmark: str, groups: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
        requests = [self._request(REFERENCE_GEN_SYSTEM, REFERENCE_GEN_USER.format(instruction=self._query(group)),
                                  self._tag("rubrichub", seed, group["qid"]), max_tokens=1024,
                                  seed=seed, role="generator") for group in groups]
        raw = await self.pool.complete_many(requests)
        references = {group["qid"]: result.text for group, result in zip(groups, raw)}
        requests = [self._request(RUBRIC_GEN_SYSTEM,
                                  RUBRIC_GEN_USER.format(instruction=self._query(group), reference=references[group["qid"]]),
                                  self._tag("rubrichub", seed, group["qid"]), max_tokens=2048,
                                  seed=seed, role="generator") for group in groups]
        raw = await self.pool.complete_many(requests)
        raw_criteria = {group["qid"]: result.text for group, result in zip(groups, raw)}
        criteria = {}
        for group in groups:
            values = parse_rubric(raw_criteria[group["qid"]])
            criteria[group["qid"]] = values or [{
                "title": "Overall Quality",
                "description": "The response adequately addresses the user's request with accurate and helpful information.",
                "weight": 5,
            }]
        requests = []
        metadata = []
        for group in groups:
            for response in group["responses"]:
                for criterion_index, criterion in enumerate(criteria[group["qid"]]):
                    user = GRADER_USER.format(instruction=self._query(group), response=self._response(response["text"]),
                                              title=criterion["title"], description=criterion["description"])
                    requests.append(self._request(GRADER_SYSTEM, user, self._tag("rubrichub", seed, group["qid"]),
                                                  max_tokens=256, seed=seed))
                    metadata.append((group["qid"], response["id"], criterion_index))
        raw = await self.pool.complete_many(requests)
        grades: dict[tuple[str, str], dict[int, tuple[bool | None, str]]] = {}
        for (qid, rid, criterion_index), result in zip(metadata, raw):
            grades.setdefault((qid, rid), {})[criterion_index] = (parse_grader_output(result.text), result.text)
        output = []
        for group in groups:
            scores = {}
            diagnostics = {"reference": references[group["qid"]], "criteria": criteria[group["qid"]],
                           "raw_criteria": raw_criteria[group["qid"]], "grades": {}}
            for response in group["responses"]:
                entries = grades.get((group["qid"], response["id"]), {})
                values = [entries.get(i, (None, ""))[0] for i in range(len(criteria[group["qid"]]))]
                scores[response["id"]] = compute_rubrichub_score(criteria[group["qid"]], values)
                diagnostics["grades"][response["id"]] = [
                    {"criterion": criterion, "met": entries.get(i, (None, ""))[0],
                     "raw": entries.get(i, (None, ""))[1]}
                    for i, criterion in enumerate(criteria[group["qid"]])]
            result = self._result(group, scores, method="rubrichub", seed=seed, diagnostics=diagnostics)
            self._finish_cost(result, "rubrichub", seed)
            output.append(result)
        return output

    async def run(self, method: str, benchmark: str, groups: list[dict[str, Any]], *, seed: int = 0,
                  direct_k: int | None = None, direct_budget: int = 14) -> list[dict[str, Any]]:
        self.benchmark = benchmark
        started = time.perf_counter()
        if method == "direct":
            output = await self._direct(benchmark, groups, seed)
        elif method == "direct_k":
            if direct_k is None:
                budgets = {}
                for group in groups:
                    if group["kind"] == "pairwise":
                        budgets[group["qid"]] = max(2, round((direct_budget + len(group["responses"])) / max(1, len(group["pairs"]))))
                    else:
                        budgets[group["qid"]] = max(1, round((direct_budget + len(group["responses"])) / max(1, len(group["responses"]))))
                direct_k = budgets
            output = await self._direct(benchmark, groups, seed, repeats=direct_k, method="direct_k")
        elif method == "dynamic":
            output = await self._dynamic(benchmark, groups, seed)
        elif method == "checkeval":
            output = await self._checklist(benchmark, groups, seed)
        elif method == "dna":
            output = await self._dna(benchmark, groups, seed)
        elif method == "rubrichub":
            output = await self._rubrichub(benchmark, groups, seed)
        else:
            raise ValueError(f"Unknown baseline {method}")
        elapsed = time.perf_counter() - started
        for result in output:
            result["wall_seconds"] = elapsed / max(1, len(output))
            self._finish_cost(result, method, seed)
        return output
