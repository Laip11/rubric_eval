"""Benchmark adapters and query-level splits for the PAOR experiments."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PAIRWISE = {"rewardbench2", "helpsteer3", "rewardbench", "judgebench_gpt", "judgebench_claude", "alpacaeval", "mtbench", "biggen"}
POINTWISE = {"helpsteer2", "biggen_pointwise", "healthbench"}
BENCHMARKS = [*sorted(PAIRWISE), "helpsteer2", "biggen_pointwise", "healthbench"]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, np.ndarray)):
        parts = []
        for message in value:
            if isinstance(message, dict):
                role = str(message.get("role", "user")).capitalize()
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    return str(value)


def _messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _check_payload(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {path}")
    if path.read_bytes().startswith(b"version https://git-lfs"):
        raise RuntimeError(f"Git LFS payload is not downloaded: {path}")


def _majority(values: list[Any], rng: random.Random | None = None) -> Any:
    counts = Counter(values)
    modes = [key for key, count in counts.items() if count == max(counts.values())]
    return (rng or random.Random(42)).choice(modes)


def _pair(qid: str, query: str, a: str, b: str, gold: str, **metadata: Any) -> dict[str, Any]:
    return {"qid": qid, "query": query, "response_a": a, "response_b": b, "gold": gold, **metadata}


def load_rewardbench2(root: Path) -> list[dict[str, Any]]:
    path = root / "rewardbench2" / "test.parquet"
    _check_payload(path)
    df = pd.read_parquet(path)
    records = []
    for _, row in df[df["subset"] != "Ties"].iterrows():
        chosen = [str(x) for x in row["chosen"]]
        rejected = [str(x) for x in row["rejected"]]
        if len(chosen) != 1 or len(rejected) != 3:
            raise ValueError(f"Unexpected RewardBench 2 non-Ties shape for id={row['id']}")
        qid = f"rewardbench2:{row['subset']}:{row['id']}"
        rejected_entries = []
        seen_rejected = set()
        for index, bad in enumerate(rejected):
            if bad not in seen_rejected:
                rejected_entries.append((index, bad))
                seen_rejected.add(bad)
        for index, bad in rejected_entries:
            pair_key = f"{qid}:{index}"
            swapped = int(_digest(pair_key), 16) % 2 == 1
            records.append(_pair(
                qid, str(row["prompt"]), bad if swapped else chosen[0], chosen[0] if swapped else bad,
                "B" if swapped else "A", subset=str(row["subset"]), source_id=str(row["id"]),
                choice_index=index, orientation_swapped=swapped,
                evaluation_protocol="derived_correct_incorrect_pairwise",
                model_a=str(row["models"][index + 1] if swapped else row["models"][0]),
                model_b=str(row["models"][0] if swapped else row["models"][index + 1]),
            ))
    return records


def load_helpsteer3(root: Path) -> list[dict[str, Any]]:
    import gzip
    path = root / "helpsteer3" / "preference_validation.jsonl.gz"
    _check_payload(path)
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for index, row in enumerate(rows):
        if row["response1"] == row["response2"]:
            continue
        preference = int(row["overall_preference"])
        records.append(_pair(
            f"helpsteer3:{index:05d}", _text(row["context"]), str(row["response1"]), str(row["response2"]),
            "tie" if preference == 0 else ("A" if preference < 0 else "B"),
            subset=str(row["domain"]), language=str(row["language"]),
            overall_preference=preference, evaluation_protocol="official_validation_pairwise",
        ))
    return records


def load_rewardbench(root: Path) -> list[dict[str, Any]]:
    path = root / "rewardbench" / "filtered.parquet"
    _check_payload(path)
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        qid = f"rewardbench:{row['subset']}:{int(row['id'])}"
        swapped = int(_digest(qid), 16) % 2 == 1
        records.append(_pair(
            qid, str(row["prompt"]),
            str(row["rejected"] if swapped else row["chosen"]),
            str(row["chosen"] if swapped else row["rejected"]),
            "B" if swapped else "A",
            subset=str(row["subset"]), source_id=int(row["id"]),
            model_a=str(row["rejected_model"] if swapped else row["chosen_model"]),
            model_b=str(row["chosen_model"] if swapped else row["rejected_model"]),
            orientation_swapped=swapped,
        ))
    return records


def _load_judgebench(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / "judgebench" / f"{split}.jsonl"
    _check_payload(path)
    records = []
    for row in _read_jsonl(path):
        if row["label"] not in {"A>B", "B>A"}:
            raise ValueError(f"Unexpected JudgeBench label: {row['label']}")
        records.append(_pair(
            f"judgebench:{split}:{row['pair_id']}", str(row["question"]),
            str(row["response_A"]), str(row["response_B"]),
            "A" if row["label"] == "A>B" else "B",
            source=str(row["source"]), pair_id=str(row["pair_id"]),
            original_id=str(row["original_id"]), response_model=str(row["response_model"]),
            split=split,
        ))
    return records


def load_judgebench_gpt(root: Path) -> list[dict[str, Any]]:
    return _load_judgebench(root, "gpt")


def load_judgebench_claude(root: Path) -> list[dict[str, Any]]:
    return _load_judgebench(root, "claude")


def load_alpacaeval(root: Path) -> list[dict[str, Any]]:
    directory = root / "alpacaeval"
    cross_path = directory / "alpaca_farm_human_crossannotations.json"
    annotations_path = directory / "alpaca_farm_human_annotations.json"
    _check_payload(cross_path)
    cross = _read_json(cross_path)
    # The cross-annotation file contains all labels needed for accuracy; the
    # optional annotation file only adds generator metadata.
    annotations = _read_json(annotations_path) if annotations_path.exists() and not annotations_path.read_bytes().startswith(b"version https://git-lfs") else []
    generators = {(x["instruction"], x["output_1"], x["output_2"]): x.get("generator", "unknown") for x in annotations}
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for item in cross:
        if item.get("datasplit") == "eval":
            grouped[(item["instruction"], item["output_1"], item["output_2"])].append(int(item["preference"]))
    pairs = []
    rng = random.Random(42)
    for index, ((query, a, b), votes) in enumerate(grouped.items()):
        winner = int(_majority(votes, rng))
        pairs.append(_pair(f"alpacaeval:{index:05d}", query, a, b, "A" if winner == 1 else "B",
                           generator=generators.get((query, a, b), "unknown"), human_preferences=votes))
    return pairs


def _mt_turn(conversation: Any, turn: int) -> tuple[str, str, str]:
    messages = _messages(conversation)
    users = [str(x.get("content", "")) for x in messages if x.get("role") == "user"]
    assistants = [str(x.get("content", "")) for x in messages if x.get("role") == "assistant"]
    if not users:
        return _text(conversation), _text(conversation), ""
    turn_index = max(0, min(int(turn) - 1, len(users) - 1))
    query_parts = []
    for i in range(turn_index + 1):
        query_parts.append(f"User: {users[i]}")
        if i < turn_index and i < len(assistants):
            query_parts.append(f"Assistant: {assistants[i]}")
    response = assistants[turn_index] if turn_index < len(assistants) else (assistants[-1] if assistants else "")
    first_response = assistants[0] if assistants else ""
    return "\n\n".join(query_parts), response, first_response


def load_mtbench(root: Path) -> list[dict[str, Any]]:
    path = root / "mtbench" / "lmsys_mt_bench_human_judgments_human.parquet"
    _check_payload(path)
    df = pd.read_parquet(path)
    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    rows: dict[tuple[Any, ...], Any] = {}
    for _, row in df.iterrows():
        turn = int(row.get("turn", 1))
        key = (row["question_id"], row.get("model_a", "model_a"), row.get("model_b", "model_b"), turn)
        grouped[key].append(str(row["winner"]))
        rows.setdefault(key, row)
    pairs = []
    rng = random.Random(42)
    for index, (key, votes) in enumerate(grouped.items()):
        winner = _majority(votes, rng)
        question_id, model_a, model_b, turn = key
        row = rows[key]
        query_a, response_a, first_a = _mt_turn(row["conversation_a"], turn)
        query_b, response_b, first_b = _mt_turn(row["conversation_b"], turn)
        query = query_a if turn == 1 else query_a + "\n\n[The other response may have a different first-turn answer; judge only the requested turn.]"
        pairs.append(_pair(f"mtbench:{question_id}:turn{turn}", query, response_a, response_b,
                           "tie" if winner == "tie" else ("A" if winner == "model_a" else "B"), question_id=int(question_id), turn=turn,
                           model_a=str(model_a), model_b=str(model_b), generator=str(model_b),
                           first_response_a=first_a, first_response_b=first_b, human_preferences=votes))
    return pairs


def _biggen_frame(root: Path) -> pd.DataFrame:
    path = root / "biggen" / "prometheus-eval_BiGGen-Bench-Results_human_eval.parquet"
    _check_payload(path)
    return pd.read_parquet(path)


def _biggen_rows(root: Path) -> list[dict[str, Any]]:
    df = _biggen_frame(root)
    rows = []
    for index, row in df.iterrows():
        query = str(row.get("input", row.get("prompt", "")))
        response = str(row.get("response", row.get("output", "")))
        score = row.get("human_score", row.get("score", row.get("human_eval", 0)))
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        rubric = row.get("score_rubric")
        if isinstance(rubric, str):
            try:
                rubric = json.loads(rubric)
            except json.JSONDecodeError:
                rubric = None
        rows.append({"qid": f"biggen:{_digest(query)}", "query": query, "response": response,
                     "human_score": score, "model": str(row.get("model_name", row.get("model", row.get("generator", "unknown")))),
                     "score_rubric": rubric, "source_index": int(index)})
    return rows


def load_biggen(root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _biggen_rows(root):
        grouped[row["qid"]].append(row)
    pairs = []
    for qid, rows in grouped.items():
        for first, second in combinations(rows, 2):
            pairs.append(_pair(qid, first["query"], first["response"], second["response"],
                               "tie" if first["human_score"] == second["human_score"] else ("A" if first["human_score"] > second["human_score"] else "B"),
                               model_a=first["model"], model_b=second["model"], generator=second["model"],
                               score_a=first["human_score"], score_b=second["human_score"], score_rubric=first.get("score_rubric")))
    return pairs


def load_biggen_pointwise(root: Path) -> list[dict[str, Any]]:
    return [{"qid": row["qid"], "query": row["query"], "response": row["response"],
             "human_score": row["human_score"], "model": row["model"], "score_rubric": row.get("score_rubric")}
            for row in _biggen_rows(root) if row["human_score"] > 0]


def load_helpsteer2(root: Path) -> list[dict[str, Any]]:
    path = root / "helpsteer2" / "nvidia_HelpSteer2_validation.parquet"
    _check_payload(path)
    df = pd.read_parquet(path)
    return [{"qid": f"helpsteer2:{_digest(str(row['prompt']))}", "query": str(row["prompt"]),
             "response": str(row["response"]), "human_score": float(row["helpfulness"]),
             "model": str(row.get("model", "unknown")), "helpfulness": float(row["helpfulness"])}
            for _, row in df.iterrows()]


def load_healthbench(root: Path) -> list[dict[str, Any]]:
    directory = root / "healthbench"
    eval_path = directory / "2025-05-07-06-14-12_oss_eval.jsonl"
    meta_path = directory / "2025-05-07-06-14-12_oss_meta_eval.jsonl"
    _check_payload(eval_path)
    _check_payload(meta_path)
    rubrics = {x["prompt_id"]: x.get("rubrics", []) for x in _read_jsonl(eval_path)}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(meta_path):
        key = (str(row["prompt_id"]), str(row["completion_id"]))
        entry = grouped.setdefault(key, {"labels": [], "row": row})
        labels = row.get("binary_labels", [])
        entry["labels"].append(bool(sum(bool(x) for x in labels) > len(labels) / 2) if labels else False)
    items = []
    for (prompt_id, completion_id), entry in grouped.items():
        row = entry["row"]
        labels = entry["labels"]
        items.append({"qid": f"healthbench:{prompt_id}", "query": _text(row.get("prompt", "")),
                      "response": _text(row.get("completion", "")),
                      "human_score": float(np.mean(labels)) if labels else 0.0,
                      "model": str(row.get("model", row.get("completion_model", "unknown"))),
                      "rubric_criteria": rubrics.get(prompt_id, []), "completion_id": completion_id})
    return items


def load_records(benchmark: str, data_root: str | Path) -> list[dict[str, Any]]:
    root = Path(data_root)
    loaders = {"rewardbench2": load_rewardbench2, "helpsteer3": load_helpsteer3,
               "rewardbench": load_rewardbench, "judgebench_gpt": load_judgebench_gpt,
               "judgebench_claude": load_judgebench_claude,
               "alpacaeval": load_alpacaeval, "mtbench": load_mtbench, "biggen": load_biggen,
               "biggen_pointwise": load_biggen_pointwise, "helpsteer2": load_helpsteer2,
               "healthbench": load_healthbench}
    if benchmark not in loaders:
        raise ValueError(f"Unknown benchmark {benchmark}; choose from {sorted(loaders)}")
    return loaders[benchmark](root)


def make_groups(records: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    """Deduplicate responses within a query while retaining every evaluated pair/item."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["qid"]].append(record)
    groups = []
    for qid, items in grouped.items():
        first = items[0]
        group: dict[str, Any] = {"qid": qid, "query": first["query"], "benchmark": benchmark,
                                 "kind": "pairwise" if benchmark in PAIRWISE else "pointwise",
                                 "responses": [], "pairs": [], "items": []}
        response_ids: dict[str, str] = {}
        def response_id(text: str) -> str:
            if text not in response_ids:
                response_ids[text] = f"r{len(response_ids)}"
                group["responses"].append({"id": response_ids[text], "text": text})
            return response_ids[text]
        if group["kind"] == "pairwise":
            for index, item in enumerate(items):
                group["pairs"].append({"id": f"{qid}:pair{index}", "a": response_id(item["response_a"]),
                                       "b": response_id(item["response_b"]), "gold": item["gold"],
                                       "metadata": {k: v for k, v in item.items() if k not in {"query", "response_a", "response_b", "gold"}}})
        else:
            for index, item in enumerate(items):
                rid = response_id(item["response"])
                group["items"].append({"id": f"{qid}:item{index}", "response_id": rid,
                                       "human_score": float(item["human_score"]), "model": item.get("model", "unknown"),
                                       "metadata": {k: v for k, v in item.items() if k not in {"query", "response", "human_score"}}})
        groups.append(group)
    return sorted(groups, key=lambda x: x["qid"])


def split_qids(groups: list[dict[str, Any]], output_dir: str | Path, seed: int = 2026) -> dict[str, list[str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = groups[0]["benchmark"] if groups else "unknown"
    path = output_dir / f"{benchmark}.json"
    if path.exists():
        return _read_json(path)
    qids = sorted(group["qid"] for group in groups)
    n_dev = min(100, max(1, int(round(len(qids) * 0.1)))) if qids else 0
    strata: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        metadata = group["pairs"][0].get("metadata", {}) if group.get("pairs") else {}
        label = str(metadata.get("subset") or metadata.get("source") or "all")
        strata[label].append(group["qid"])
    dev = []
    if len(strata) > 1 and n_dev:
        counts = {label: int(n_dev * len(values) / len(groups)) for label, values in strata.items()}
        for label in strata:
            if counts[label] == 0:
                counts[label] = 1
        while sum(counts.values()) > n_dev:
            candidates = [label for label in strata if counts[label] > 1]
            label = min(candidates, key=lambda x: counts[x] - n_dev * len(strata[x]) / len(groups))
            counts[label] -= 1
        while sum(counts.values()) < n_dev:
            label = max(strata, key=lambda x: n_dev * len(strata[x]) / len(groups) - counts[x])
            counts[label] += 1
        for label, values in strata.items():
            values = sorted(values)
            random.Random(seed + sum(map(ord, label))).shuffle(values)
            dev.extend(values[:counts[label]])
    else:
        shuffled = list(qids)
        random.Random(seed).shuffle(shuffled)
        dev = shuffled[:n_dev]
    dev_set = set(dev)
    splits = {"seed": seed, "dev": sorted(dev), "test": sorted(x for x in qids if x not in dev_set),
              "protocol": "stratified_by_subset_or_source" if len(strata) > 1 else "random"}
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    return splits


def select_groups(groups: list[dict[str, Any]], splits: dict[str, list[str]], split: str = "test", max_queries: int | None = None) -> list[dict[str, Any]]:
    allowed = set(splits[split])
    selected = [group for group in groups if group["qid"] in allowed]
    if max_queries is not None:
        selected = selected[:max_queries]
    return selected


def drop_human_ties(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for group in groups:
        if group["kind"] != "pairwise":
            filtered.append(group)
            continue
        pairs = [pair for pair in group["pairs"] if pair["gold"] in ("A", "B")]
        if not pairs:
            continue
        used = {response for pair in pairs for response in (pair["a"], pair["b"])}
        value = {**group, "pairs": pairs}
        if "responses" in group:
            value["responses"] = [response for response in group["responses"] if response["id"] in used]
        filtered.append(value)
    return filtered


def group_from_pairs_for_mock(qid: str, query: str, pairs: list[tuple[str, str, str]]) -> dict[str, Any]:
    records = [_pair(qid, query, a, b, gold) for a, b, gold in pairs]
    return make_groups(records, "alpacaeval")[0]
