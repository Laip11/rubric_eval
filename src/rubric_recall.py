#!/usr/bin/env python3
"""Rubric-quality analysis against expert rubrics (RubricBench; report hypothesis H1).

For every method's seed-0 result on ``rubricbench``, extract the final rubric of each query
and measure how well it covers the expert atomic rubric:

* ``recall``     share of expert items matched by at least one generated criterion;
* ``precision``  share of generated criteria matching at least one expert item;
* ``redundancy`` share of generated criteria that have a near-duplicate inside the same rubric.

Matching is either embedding cosine ≥ ``--threshold`` (cheap, default) or an LLM matcher
(``--matcher llm``, one judge call per query through an OpenAI-compatible endpoint).

Usage:
    python src/rubric_recall.py --output-root outputs/mccr_pairwise_v2 \
        --embedding-model /path/Qwen3-Embedding-0.6B --methods mccr mccr_v2 rrd rubric_resp checkeval rubrichub
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paor_data import load_records  # noqa: E402

MATCH_PROMPT = '''Below are an EXPERT rubric and a GENERATED rubric for the same user query.
For each expert item, decide whether some generated item checks essentially the same requirement
(same intent; wording may differ; a generated item that is strictly broader still counts only if a
response satisfying it would very likely satisfy the expert item).

### User Query:
{query}

### Expert rubric:
{expert}

### Generated rubric:
{generated}

Output ONLY JSON: {{"covered": ["X1", ...], "matched_generated": ["G2", ...]}}'''


def criteria_from_diagnostic(method: str, diagnostic: dict[str, Any]) -> list[str]:
    if not diagnostic:
        return []
    if diagnostic.get("method") == "mccr_me":  # minimal evidence set
        return [x["criterion"] for x in diagnostic.get("E", [])]
    if "R_core" in diagnostic:  # MCCR v1/v2
        return [x["criterion"] for x in diagnostic.get("R_core", []) + diagnostic.get("R_ext", [])]
    if "criteria" in diagnostic and isinstance(diagnostic["criteria"], dict):  # probe rubric runners
        return [x["criterion"] for x in diagnostic["criteria"].get("core", []) + diagnostic["criteria"].get("ext", [])]
    if "R0" in diagnostic:  # PAOR
        selected = set(diagnostic.get("selected", []))
        candidates = diagnostic.get("R_candidates", [])
        return [x["criterion"] for x in diagnostic["R0"]] + [x["criterion"] for j, x in enumerate(candidates) if j in selected]
    if "questions" in diagnostic:  # CheckEval
        return list(diagnostic["questions"])
    if "criteria" in diagnostic and isinstance(diagnostic["criteria"], list):  # RubricHub
        return [f"{x.get('title', '')}: {x.get('description', '')}" for x in diagnostic["criteria"]]
    if "aspects" in diagnostic:  # DnA
        return list(diagnostic["aspects"])
    return []


def embedding_match(expert: np.ndarray, generated: np.ndarray, threshold: float) -> tuple[float, float, float]:
    if len(expert) == 0 or len(generated) == 0:
        return 0.0, 0.0, 0.0
    sim = expert @ generated.T
    recall = float(np.mean(sim.max(axis=1) >= threshold))
    precision = float(np.mean(sim.max(axis=0) >= threshold))
    inner = generated @ generated.T
    np.fill_diagonal(inner, -1.0)
    redundancy = float(np.mean(inner.max(axis=1) >= 0.9)) if len(generated) > 1 else 0.0
    return recall, precision, redundancy


async def llm_match(items: list[dict[str, Any]], endpoint: str, model: str) -> list[tuple[float, float]]:
    from async_vllm import AsyncLLMPool
    pool = AsyncLLMPool([endpoint], strong_model=model)
    requests = []
    for item in items:
        expert = "\n".join(f"X{j + 1}: {x}" for j, x in enumerate(item["expert"]))
        generated = "\n".join(f"G{j + 1}: {x}" for j, x in enumerate(item["generated"]))
        requests.append(pool.request("You are a careful annotator. Output only JSON.",
                                     MATCH_PROMPT.format(query=item["query"], expert=expert, generated=generated),
                                     max_tokens=512, tag="rubric_recall"))
    results = await pool.complete_many(requests)
    await pool.close()
    output = []
    for item, result in zip(items, results):
        covered = set(re.findall(r'"X(\d+)"', result.text.split("matched_generated")[0]))
        matched = set(re.findall(r'"G(\d+)"', result.text.split("matched_generated")[-1]))
        output.append((len(covered) / max(1, len(item["expert"])), len(matched) / max(1, len(item["generated"]))))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=HERE.parent / "generating_dynamic_rubric" / "data")
    parser.add_argument("--benchmark", default="rubricbench")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--matcher", choices=["embedding", "llm"], default="embedding")
    parser.add_argument("--embedding-model", default="/data/project/sustech/model/Qwen3-Embedding-0.6B")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="judge")
    args = parser.parse_args()
    records = load_records(args.benchmark, args.data_root)
    expert = {r["qid"]: r.get("expert_rubrics") or [] for r in records}
    queries = {r["qid"]: r["query"] for r in records}
    encoder = None
    if args.matcher == "embedding":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embedding_model, device="cpu", local_files_only=Path(args.embedding_model).exists())
        encoder = lambda texts: model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    report: dict[str, Any] = {}
    for method in args.methods:
        path = args.output_root / args.benchmark / method / "seed0.json"
        if not path.exists():
            print(f"skip {method}: {path} missing")
            continue
        groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
        items = [{"qid": g["qid"], "query": queries.get(g["qid"], ""), "expert": expert.get(g["qid"], []),
                  "generated": criteria_from_diagnostic(method, g.get("diagnostic", {}))} for g in groups]
        items = [x for x in items if x["expert"] and x["generated"]]
        if not items:
            print(f"skip {method}: no rubric found in diagnostics")
            continue
        if args.matcher == "embedding":
            recalls, precisions, redundancies = [], [], []
            for item in items:
                r, p, d = embedding_match(np.asarray(encoder(item["expert"])), np.asarray(encoder(item["generated"])), args.threshold)
                recalls.append(r); precisions.append(p); redundancies.append(d)
            report[method] = {"recall": float(np.mean(recalls)), "precision": float(np.mean(precisions)),
                              "redundancy": float(np.mean(redundancies))}
        else:
            values = asyncio.run(llm_match(items, args.endpoint, args.model))
            report[method] = {"recall": float(np.mean([v[0] for v in values])),
                              "precision": float(np.mean([v[1] for v in values]))}
        report[method].update({"n_queries": len(items), "mean_rubric_size": float(np.mean([len(x["generated"]) for x in items])),
                               "mean_expert_size": float(np.mean([len(x["expert"]) for x in items]))})
    out = args.output_root / f"rubric_recall_{args.benchmark}_{args.matcher}.json"
    out.write_text(json.dumps({"matcher": args.matcher, "threshold": args.threshold, "methods": report}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(out)


if __name__ == "__main__":
    main()
