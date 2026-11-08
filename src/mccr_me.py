"""MCCR-ME: rubric construction as a minimal evidence set (new_our_method.md §5).

Target: for a query q, a criterion set E that is
  * grounded   — every criterion is stated in q (explicit, verified evidence span) or is an implicit
                 requirement of q that survives a necessity test;
  * necessary  — for every e in E there is a witness response that satisfies E \\ {e}, violates e and is
                 worse (checked empirically with a single-violation witness and an order-swapped judge);
  * irredundant— no criterion is implied by the others (deletion-based minimisation);
  * sufficient — one round of counterexample search: an adversarial response that satisfies all of E but is
                 worse reveals a missing requirement, which is added after its own necessity test.

Pipeline: query view (with grounding) → shared probes → response views → clustering → classes
(explicit / both / query_implicit / response_only) → necessity tests → deduplication → sufficiency →
uniform scoring of targets on E → shared direct fallback for ties. Everything is training-free.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import numpy as np

from mccr_core import (classify_clusters, deletion_minimize, greedy_hitting_set,
                       semantic_consensus_clusters)
from mccr_pipeline import ARCHITECT_SYSTEM, JUDGE_SYSTEM, RUBRIC_SAFETY_CAP, MCCRPipeline
from paor_core import extract_json
from paor_pipeline import (decide_pair, format_criteria, load_json, parse_winner, probe_diagnostics,
                           save_json, truncate_text, valid_probes)
from paor_prompts import (DIRECT_PAIR_SYSTEM, DIRECT_PAIR_USER, PROMPT_ADVERSARY, PROMPT_ATOMIC_JUDGE,
                          PROMPT_ENTAILMENT, PROMPT_ME_PRIOR, PROMPT_MISSING_REQUIREMENT,
                          PROMPT_NECESSITY_JUDGMENT, PROMPT_WITNESS)

NECESSITY_MODES = {"witness", "judgment", "hitting_set", "none"}
# Classes accepted without a necessity test: stated in the query, or proposed by both the query view
# and at least one response view. The others must earn their place.
UNTESTED_CLASSES = {"explicit", "both"}


class MinimalEvidencePipeline(MCCRPipeline):
    def __init__(self, *args: Any, necessity: str = "witness", max_tests: int = 12,
                 sufficiency: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("method_name", "mccr_me")
        kwargs.setdefault("variant", "v2")
        super().__init__(*args, **kwargs)
        if necessity not in NECESSITY_MODES:
            raise ValueError(f"Unknown necessity mode {necessity}; choose from {sorted(NECESSITY_MODES)}")
        self.necessity = necessity
        self.max_tests = max_tests
        self.sufficiency = sufficiency
        # Grounded query view; response views use the coverage prompt inherited from MCCR.
        self._prior_prompt = PROMPT_ME_PRIOR
        self._prior_limits = (1, RUBRIC_SAFETY_CAP)
        self._prior_tokens = 2048

    def _uses_counterfactuals(self) -> bool:
        # Generic counterfactuals are only needed by the hitting-set estimator.
        return self.necessity == "hitting_set" and self.n_counterfactual > 0

    # ------------------------------------------------------------- helpers
    def _q(self, group: dict[str, Any]) -> str:
        return truncate_text(group["query"], self.tokenizer, self.query_max_tokens)

    def _r(self, text: str) -> str:
        return truncate_text(text, self.tokenizer, self.response_max_tokens)

    @staticmethod
    def _base_probe(cache: dict[str, Any]) -> str:
        """The first complete strong-generator probe (fallback: any probe) is the base for witnesses."""
        roles = cache.get("probe_roles") or []
        probes = valid_probes(cache)
        return next((text for slot, text in probes if slot < len(roles) and roles[slot] == "generator"), probes[0][1])

    async def _pairwise_base_wins(self, jobs: list[tuple[int, str, str]], groups: list[dict[str, Any]],
                                  seed: int) -> list[dict[str, Any]]:
        """jobs: (group index, base text, other text) → base preferred in BOTH orders?"""
        requests, keys = [], []
        for j, (i, base, other) in enumerate(jobs):
            for reverse in (False, True):
                a, b = (other, base) if reverse else (base, other)
                user = DIRECT_PAIR_USER.format(query=self._q(groups[i]), response_a=self._r(a), response_b=self._r(b))
                requests.append(self._request(DIRECT_PAIR_SYSTEM, user, self._tag(seed, groups[i]["qid"], "none"),
                                              temperature=0.0, max_tokens=256, seed=seed + int(reverse)))
                keys.append((j, reverse))
        raw = await self.pool.complete_many(requests) if requests else []
        votes: dict[int, list[bool]] = {}
        for (j, reverse), result in zip(keys, raw):
            winner = parse_winner(result.text)
            votes.setdefault(j, []).append(winner == ("B" if reverse else "A"))
        return [{"base_wins_both": len(votes.get(j, [])) == 2 and all(votes[j]), "votes": votes.get(j, [])}
                for j in range(len(jobs))]

    # ------------------------------------------------------------- stages
    def _classify(self, groups: list[dict[str, Any]], caches: list[dict[str, Any]], embeddings_needed: list[int],
                  embeddings: np.ndarray) -> None:
        cursor = 0
        for i in embeddings_needed:
            cache = caches[i]
            n = len(cache["U"])
            views = np.asarray([x["view"] for x in cache["U"]], dtype=int)
            cache["clusters"] = semantic_consensus_clusters(embeddings[cursor:cursor + n], views,
                                                           theta_sim=self.theta_sim, beta=self.beta)
            cursor += n
            cache["me_candidates"] = classify_clusters(cache["clusters"], cache["U"], groups[i]["query"])

    async def _cluster(self, benchmark, groups, caches, seed) -> None:
        todo, texts = [], []
        for i, cache in enumerate(caches):
            if cache.get("me_candidates") is not None:
                continue
            flattened = []
            for key in sorted(cache["R_views"], key=int):
                for item in cache["R_views"][key]:
                    flattened.append({"id": f"R{len(flattened) + 1}", "criterion": item["criterion"], "view": int(key),
                                      "type": item.get("type"), "evidence": item.get("evidence")})
            cache["U"] = flattened
            todo.append(i)
            texts.extend(x["criterion"] for x in flattened)
        if todo:
            embeddings = await asyncio.to_thread(self.embedding_fn, texts)
            self._classify(groups, caches, todo, embeddings)
            for i in todo:
                self._save(benchmark, groups[i], seed, caches[i])

    def _tested(self, cache: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Split candidates into (accepted without test, to test, dropped by the test budget)."""
        untested = [x for x in cache["me_candidates"] if x["class"] in UNTESTED_CLASSES]
        pending = [x for x in cache["me_candidates"] if x["class"] not in UNTESTED_CLASSES]
        # Query-grounded candidates first, then response-only ones by how many views proposed them.
        pending.sort(key=lambda x: (x["class"] != "query_implicit", -x["n_response_views"], x["id"]))
        return untested, pending[:self.max_tests], pending[self.max_tests:]

    async def _witness_tests(self, groups, caches, seed, targets: dict[int, list[dict[str, Any]]],
                             context: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, dict[str, Any]]]:
        """Single-violation witness + order-swapped pairwise judgement for every target criterion."""
        requests, owners = [], []
        for i, items in targets.items():
            base = self._base_probe(caches[i])
            for item in items:
                others = [x for x in context[i] if x["id"] != item["id"]]
                user = PROMPT_WITNESS.format(query=self._q(groups[i]), criterion=f"{item['id']}: {item['criterion']}",
                                             others=format_criteria(others) or "(none)", response=base)
                requests.append(self._request("You create controlled evaluation test cases. Follow the output format exactly.",
                                              user, self._tag(seed, groups[i]["qid"], "none"), role="generator",
                                              temperature=0.3, max_tokens=self.probe_max_tokens, seed=seed + 31))
                owners.append((i, item, base))
        raw = await self.pool.complete_many(requests) if requests else []
        results: dict[int, dict[str, dict[str, Any]]] = {i: {} for i in targets}
        jobs, job_owner = [], []
        for (i, item, base), result in zip(owners, raw):
            text = result.text.strip()
            impossible = re.match(r"^\s*IMPOSSIBLE\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
            if impossible:
                implied = re.findall(r"[A-Za-z]+\d+", impossible.group(1))
                results[i][item["id"]] = {"necessary": False, "reason": "implied", "implied_by": implied}
            elif result.finish_reason == "length" or not text:
                results[i][item["id"]] = {"necessary": False, "reason": "witness_failed"}
            else:
                jobs.append((i, base, text))
                job_owner.append((i, item, text))
        for (i, item, text), verdict in zip(job_owner, await self._pairwise_base_wins(jobs, groups, seed)):
            results[i][item["id"]] = {"necessary": verdict["base_wins_both"], "votes": verdict["votes"],
                                      "reason": "witness_worse" if verdict["base_wins_both"] else "witness_not_worse",
                                      "witness": text}
        return results

    async def _judgment_tests(self, groups, caches, seed, targets) -> dict[int, dict[str, dict[str, Any]]]:
        requests, owners = [], []
        for i, items in targets.items():
            user = PROMPT_NECESSITY_JUDGMENT.format(query=self._q(groups[i]), criteria=format_criteria(items))
            requests.append(self._request(JUDGE_SYSTEM, user, self._tag(seed, groups[i]["qid"], "none"),
                                          temperature=0.0, max_tokens=1024))
            owners.append((i, items))
        raw = await self.pool.complete_many(requests) if requests else []
        results = {}
        for (i, items), result in zip(owners, raw):
            try:
                data = extract_json(result.text)
                verdicts = data.get("necessary", data) if isinstance(data, dict) else {}
            except ValueError:
                verdicts = {}
            results[i] = {x["id"]: {"necessary": bool(verdicts.get(x["id"], False)), "reason": "judgment"} for x in items}
        return results

    async def _hitting_set_tests(self, groups, caches, seed, targets) -> dict[int, dict[str, dict[str, Any]]]:
        jobs = []
        for i, items in targets.items():
            texts, meta, pairs = self._score_rows(caches[i])
            caches[i]["score_rows"], caches[i]["known_pairs"] = meta, pairs
            for r, text in enumerate(texts):
                user = PROMPT_ATOMIC_JUDGE.format(query=self._q(groups[i]), response=text,
                                                  criteria_list_formatted=format_criteria(items))
                jobs.append((i, r, [x["id"] for x in items],
                             self._request(JUDGE_SYSTEM, user, self._tag(seed, groups[i]["qid"], "none"),
                                           temperature=0.0, max_tokens=self.judge_max_tokens)))
        requests = [job[3] for job in jobs]
        raw = await self.pool.complete_many(requests) if requests else []
        raw = await self._retry_atomic(requests, raw, [job[2] for job in jobs])
        rows: dict[int, list[list[float]]] = {}
        for (i, _, ids, _), result in zip(jobs, raw):
            values, _ = self._score_result(result, ids)
            rows.setdefault(i, []).append(values)
        results = {}
        for i, items in targets.items():
            F = np.asarray(rows.get(i, []), dtype=float).reshape(-1, len(items))
            selected, info = greedy_hitting_set(F, [tuple(p) for p in caches[i].get("known_pairs", [])],
                                                tau_align=self.tau_align)
            caches[i]["hitting_set_info"] = info
            results[i] = {x["id"]: {"necessary": j in selected, "reason": "hitting_set"} for j, x in enumerate(items)}
        return results

    async def _test_necessity(self, groups, caches, seed, targets, context) -> dict[int, dict[str, dict[str, Any]]]:
        targets = {i: items for i, items in targets.items() if items}
        if not targets:
            return {}
        if self.necessity == "witness":
            return await self._witness_tests(groups, caches, seed, targets, context)
        if self.necessity == "judgment":
            return await self._judgment_tests(groups, caches, seed, targets)
        if self.necessity == "hitting_set":
            return await self._hitting_set_tests(groups, caches, seed, targets)
        return {i: {x["id"]: {"necessary": True, "reason": "untested"} for x in items} for i, items in targets.items()}

    async def _necessity_stage(self, benchmark, groups, caches, seed) -> None:
        targets, context = {}, {}
        for i, cache in enumerate(caches):
            if cache.get("me_tests") is not None:
                continue
            untested, tested, dropped = self._tested(cache)
            cache["me_budget_dropped"] = [x["id"] for x in dropped]
            targets[i] = tested
            context[i] = untested + tested
        if not targets:
            return
        results = await self._test_necessity(groups, caches, seed, targets, context)
        for i in targets:
            cache = caches[i]
            cache["me_tests"] = results.get(i, {})
            untested, tested, _ = self._tested(cache)
            cache["me_accepted"] = untested + [x for x in tested if cache["me_tests"].get(x["id"], {}).get("necessary")]
            cache["me_fallback"] = not cache["me_accepted"]
            if cache["me_fallback"]:
                # Nothing survived: keep the best-supported candidate so the response can still be scored.
                cache["me_accepted"] = sorted(cache["me_candidates"], key=lambda x: (-x["rho"], x["id"]))[:1]
            self._save(benchmark, groups[i], seed, cache)

    async def _dedup_stage(self, benchmark, groups, caches, seed, *, source: str = "me_accepted", target: str = "me_dedup") -> None:
        requests, owners, changed = [], [], []
        for i, (group, cache) in enumerate(zip(groups, caches)):
            if cache.get(target) is not None:
                continue
            items = cache[source]
            changed.append(i)
            if len(items) < 2:
                cache[target] = {"kept": items, "removed": []}
                continue
            user = PROMPT_ENTAILMENT.format(query=self._q(group), criteria=format_criteria(items))
            requests.append(self._request(JUDGE_SYSTEM, user, self._tag(seed, group["qid"], "none"),
                                          temperature=0.0, max_tokens=1024))
            owners.append(i)
        raw = await self.pool.complete_many(requests) if requests else []
        for i, result in zip(owners, raw):
            cache = caches[i]
            implied: dict[str, list[str]] = {}
            try:
                data = extract_json(result.text)
                for entry in (data.get("redundant", []) if isinstance(data, dict) else []):
                    if isinstance(entry, dict) and entry.get("id"):
                        implied[str(entry["id"])] = [str(x) for x in entry.get("implied_by", [])]
            except ValueError:
                pass
            # Witnesses that could not violate a criterion alone are evidence of implication as well.
            for cid, test in (cache.get("me_tests") or {}).items():
                if test.get("reason") == "implied" and cid not in implied:
                    implied[cid] = test.get("implied_by", [])
            kept, removed = deletion_minimize(cache[source], implied)
            if not kept:
                kept, removed = cache[source][:1], removed
            cache[target] = {"kept": kept, "removed": removed}
        for i in changed:
            self._save(benchmark, groups[i], seed, caches[i])

    async def _sufficiency_stage(self, benchmark, groups, caches, seed) -> None:
        todo = [i for i, cache in enumerate(caches) if cache.get("me_sufficiency") is None]
        if not todo:
            return
        if not self.sufficiency:
            for i in todo:
                caches[i]["me_sufficiency"] = {"enabled": False}
                caches[i]["E"] = caches[i]["me_dedup"]["kept"]
                self._save(benchmark, groups[i], seed, caches[i])
            return
        # 1) adversary that satisfies all of E but should be worse than the base probe.
        requests = []
        for i in todo:
            cache = caches[i]
            user = PROMPT_ADVERSARY.format(query=self._q(groups[i]), criteria=format_criteria(cache["me_dedup"]["kept"]),
                                           response=self._base_probe(cache))
            requests.append(self._request("You stress-test evaluation rubrics. Output only the response.", user,
                                          self._tag(seed, groups[i]["qid"], "none"), role="generator",
                                          temperature=0.7, max_tokens=self.probe_max_tokens, seed=seed + 53))
        adversaries = await self.pool.complete_many(requests)
        # 2) does it satisfy every criterion of E, and is it worse in both orders?
        score_requests, pair_jobs, valid = [], [], []
        for i, result in zip(todo, adversaries):
            text = result.text.strip()
            if not text or result.finish_reason == "length":
                caches[i]["me_sufficiency"] = {"enabled": True, "counterexample": False, "reason": "adversary_failed"}
                continue
            criteria = caches[i]["me_dedup"]["kept"]
            user = PROMPT_ATOMIC_JUDGE.format(query=self._q(groups[i]), response=self._r(text),
                                              criteria_list_formatted=format_criteria(criteria))
            score_requests.append(self._request(JUDGE_SYSTEM, user, self._tag(seed, groups[i]["qid"], "none"),
                                                temperature=0.0, max_tokens=self.judge_max_tokens))
            pair_jobs.append((i, self._base_probe(caches[i]), text))
            valid.append((i, text, [x["id"] for x in criteria]))
        scored = await self.pool.complete_many(score_requests) if score_requests else []
        scored = await self._retry_atomic(score_requests, scored, [v[2] for v in valid]) if score_requests else []
        verdicts = await self._pairwise_base_wins(pair_jobs, groups, seed)
        found = []
        for (i, text, ids), result, verdict in zip(valid, scored, verdicts):
            values, _ = self._score_values(result, ids)
            satisfies_all = all(v is not None and v >= 0.999 for v in values)
            counterexample = satisfies_all and verdict["base_wins_both"]
            caches[i]["me_sufficiency"] = {"enabled": True, "counterexample": counterexample, "adversary": text,
                                           "satisfies_all": satisfies_all, "votes": verdict["votes"]}
            if counterexample:
                found.append((i, text))
        # 3) extract the missing requirement(s), test their necessity, and add the survivors.
        additions: dict[int, list[dict[str, Any]]] = {}
        if found:
            requests = [self._request(ARCHITECT_SYSTEM, PROMPT_MISSING_REQUIREMENT.format(
                query=self._q(groups[i]), criteria=format_criteria(caches[i]["me_dedup"]["kept"]),
                reference=self._base_probe(caches[i]), adversarial=text), self._tag(seed, groups[i]["qid"], "none"),
                role="generator", temperature=0.3, max_tokens=1024, seed=seed + 59) for i, text in found]
            missing, _ = await self._rubric_batch(requests, "missing_requirements", 1, 3, [], "missing")
            targets, context = {}, {}
            for (i, _), values in zip(found, missing):
                items = [{"id": f"S{j + 1}", "criterion": x["criterion"], "class": "sufficiency", "response_support": 0.0,
                          "n_response_views": 0} for j, x in enumerate(values)]
                caches[i]["me_sufficiency"]["proposed"] = items
                targets[i] = items
                context[i] = caches[i]["me_dedup"]["kept"] + items
            tests = await self._test_necessity(groups, caches, seed, targets, context)
            for i, items in targets.items():
                caches[i]["me_sufficiency"]["tests"] = tests.get(i, {})
                additions[i] = [x for x in items if tests.get(i, {}).get(x["id"], {}).get("necessary")]
        for i in todo:
            caches[i]["E"] = caches[i]["me_dedup"]["kept"] + additions.get(i, [])
            self._save(benchmark, groups[i], seed, caches[i])

    async def _prepare(self, benchmark: str, groups: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
        caches = [self._cache(benchmark, group, seed) for group in groups]
        await self._prior_views(benchmark, groups, caches, seed)
        caches = await self._sample_probes(benchmark, groups, seed, caches)
        await self._response_views(benchmark, groups, caches, seed)
        await self._counterfactuals(benchmark, groups, caches, seed)
        await self._cluster(benchmark, groups, caches, seed)
        await self._necessity_stage(benchmark, groups, caches, seed)
        await self._dedup_stage(benchmark, groups, caches, seed)
        await self._sufficiency_stage(benchmark, groups, caches, seed)
        return caches

    async def _target_scores(self, benchmark, groups, caches, seed) -> None:
        requests, indices = [], []
        for i, (group, cache) in enumerate(zip(groups, caches)):
            target = cache.get("target_scores", {})
            if target and all(r["id"] in target for r in group["responses"]):
                continue
            for response in group["responses"]:
                user = PROMPT_ATOMIC_JUDGE.format(query=self._q(group), response=self._r(response["text"]),
                                                  criteria_list_formatted=format_criteria(cache["E"]))
                requests.append(self._request(JUDGE_SYSTEM, user, self._tag(seed, group["qid"], "none"),
                                              temperature=0.0, max_tokens=self.judge_max_tokens))
                indices.append((i, response["id"], [x["id"] for x in cache["E"]]))
        if not requests:
            return
        raw = await self.pool.complete_many(requests)
        raw = await self._retry_atomic(requests, raw, [x[2] for x in indices])
        for (i, rid, ids), result in zip(indices, raw):
            values, missing = self._score_values(result, ids)
            caches[i].setdefault("target_scores", {})[rid] = values
            caches[i].setdefault("target_raw", {})[rid] = result.text
            caches[i]["parse_fail"] = caches[i].get("parse_fail", 0) + missing
        for i in {x[0] for x in indices}:
            self._save(benchmark, groups[i], seed, caches[i])

    async def run(self, benchmark: str, groups: list[dict[str, Any]], *, seed: int = 0,
                  ablation: str = "none") -> list[dict[str, Any]]:
        self.benchmark = benchmark
        self._ablation = "none"
        started = time.perf_counter()
        self._run_started = started
        self._wall_offset = float(load_json(self._clock_path(benchmark, "none"), {}).get("seconds", 0.0))
        caches = await self._prepare(benchmark, groups, seed)
        await self._target_scores(benchmark, groups, caches, seed)
        output, fallback_requests = [], []
        for i, (group, cache) in enumerate(zip(groups, caches)):
            scores: dict[str, float | None] = {}
            for response in group["responses"]:
                resolved = self._resolve_missing(list(cache["target_scores"][response["id"]]))
                # Every criterion of E is a necessary condition, so all carry equal weight.
                scores[response["id"]] = None if resolved is None else float(np.mean(resolved))
            pairs = []
            for pair in group["pairs"]:
                item = decide_pair(pair, scores)
                pairs.append(item)
                if item["fallback"]:
                    fallback_requests.append((i, pair))
            counts = {label: sum(1 for x in cache["me_candidates"] if x["class"] == label)
                      for label in ["explicit", "both", "query_implicit", "response_only"]}
            output.append({"qid": group["qid"], "kind": group["kind"], "scores": scores, "pairs": pairs, "items": [],
                           "diagnostic": {"method": "mccr_me", "necessity_mode": self.necessity,
                                          "E": cache["E"], "candidates": cache["me_candidates"], "class_counts": counts,
                                          "tests": cache.get("me_tests", {}),
                                          "budget_dropped": cache.get("me_budget_dropped", []),
                                          "me_fallback": cache.get("me_fallback", False),
                                          "dedup_removed": cache["me_dedup"]["removed"],
                                          "sufficiency": cache.get("me_sufficiency", {}),
                                          "hitting_set_info": cache.get("hitting_set_info"),
                                          "U": cache["U"], "clusters": cache["clusters"],
                                          "parse_fail": cache.get("parse_fail", 0), **probe_diagnostics(cache)}})
        fallback_values = await self._fallbacks(groups, fallback_requests, seed, "none")
        for result, cache, group in zip(output, caches, groups):
            for pair in result["pairs"]:
                if pair["fallback"]:
                    fallback = fallback_values[pair["id"]]
                    pair["prediction"] = fallback["prediction"]
                    pair.update({k: v for k, v in fallback.items() if k not in ("prediction", "model_score_tie")})
            result["cost"] = self._cost(cache, seed, group["qid"], "none")
        elapsed = self._wall_offset + time.perf_counter() - started
        save_json(self._clock_path(benchmark, "none"), {"seconds": elapsed})
        for result in output:
            result["wall_seconds"] = elapsed / max(1, len(output))
        return output
