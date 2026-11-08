#!/usr/bin/env python3
"""Mechanism diagnostics for MCCR runs (report §1.2, P2–P4).

Reads ``<output-root>/diagnostics/<benchmark>.<method>.seed<k>.jsonl`` and the matching
``<benchmark>/<method>/seed<k>.json`` and reports, per benchmark and method:

* P2 — residual selection: distribution of L′ (number of supplementary criteria), share of
  queries with L′ = 0, core rank vs. number of probe rows (rank saturation);
* P3 — core composition: share of core medoids whose cluster contains a prior-view criterion,
  share that are "prior echoed by exactly one response view", share supported only by response
  views; share of prior-view criteria that end up in the core;
* P4 — ties: fallback rate and accuracy on fallback vs. non-fallback pairs;
* probes — valid/dropped probes and role recoveries; v2 — misaligned core criteria.

Usage:
    python src/mccr_diagnostics.py --output-root outputs/mccr_pairwise_v2 --methods mccr mccr_v2
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def query_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    clusters = row.get("clusters", [])
    core = row.get("R_core_initial") or row.get("R_core", [])
    core_ids = {item.get("cluster_id") for item in core}
    response_views = sorted({int(x["view"]) for x in row.get("U", []) if int(x["view"]) > 0})
    m = max(1, len(response_views))
    core_clusters = [clusters[c] for c in core_ids if c is not None and c < len(clusters)]
    with_prior = [c for c in core_clusters if c.get("prior_present")]
    echoed_once = [c for c in with_prior if c.get("response_support", 0.0) * m <= 1 + 1e-9]
    prior_clusters = [c for c in clusters if c.get("prior_present")]
    selection = row.get("selection", {}) or {}
    rows = len(row.get("F_core", []))
    return {
        "K": len(row.get("R_core", [])), "K_initial": len(core), "L_prime": int(row.get("L_prime", 0)),
        "candidates": len(row.get("R_ext_candidates", [])), "raw_criteria": len(row.get("U", [])),
        "clusters": len(clusters), "score_rows": rows,
        "core_rank": selection.get("core_rank"), "final_rank": selection.get("final_rank"),
        "rank_saturated": (selection.get("core_rank") is not None and rows > 0 and selection["core_rank"] >= rows - 1),
        "core_share_with_prior": len(with_prior) / len(core_clusters) if core_clusters else None,
        "core_share_prior_echoed_once": len(echoed_once) / len(core_clusters) if core_clusters else None,
        "core_share_response_only": (len(core_clusters) - len(with_prior)) / len(core_clusters) if core_clusters else None,
        "prior_clusters_in_core": (sum(1 for c in prior_clusters if c in core_clusters) / len(prior_clusters)
                                   if prior_clusters else None),
        "core_fallback": bool(row.get("core_fallback")),
        "core_misaligned": len(row.get("core_misaligned", []) or []),
        "n_valid_probes": row.get("n_valid_probes"), "probe_dropped": row.get("probe_dropped", 0),
        "probe_role_recoveries": len(row.get("probe_role_recoveries", []) or []),
        "n_counterfactual": row.get("n_counterfactual", 0),
    }


def summarize(rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    per_query = [query_diagnostics(row) for row in rows]
    def collect(key: str) -> list[float]:
        return [float(x[key]) for x in per_query if x.get(key) is not None]
    pairs = [p for g in groups for p in g.get("pairs", []) if p.get("gold") in ("A", "B")]
    fallback = [p for p in pairs if p.get("fallback")]
    other = [p for p in pairs if not p.get("fallback")]
    accuracy = lambda values: mean([float(p.get("prediction") == p["gold"]) for p in values])
    return {
        "n_queries": len(per_query),
        "L_prime_distribution": dict(sorted(Counter(int(x["L_prime"]) for x in per_query).items())),
        "share_L_prime_zero": mean([float(x["L_prime"] == 0) for x in per_query]),
        "mean_K": mean(collect("K")), "mean_K_initial": mean(collect("K_initial")),
        "mean_candidates": mean(collect("candidates")), "mean_raw_criteria": mean(collect("raw_criteria")),
        "mean_clusters": mean(collect("clusters")), "mean_score_rows": mean(collect("score_rows")),
        "core_rank_distribution": dict(sorted(Counter(x["core_rank"] for x in per_query if x["core_rank"] is not None).items())),
        "share_rank_saturated": mean([float(x["rank_saturated"]) for x in per_query]),
        "core_share_with_prior": mean(collect("core_share_with_prior")),
        "core_share_prior_echoed_once": mean(collect("core_share_prior_echoed_once")),
        "core_share_response_only": mean(collect("core_share_response_only")),
        "prior_clusters_in_core": mean(collect("prior_clusters_in_core")),
        "share_core_fallback": mean([float(x["core_fallback"]) for x in per_query]),
        "mean_core_misaligned": mean(collect("core_misaligned")),
        "mean_valid_probes": mean(collect("n_valid_probes")),
        "total_probe_dropped": sum(int(x["probe_dropped"] or 0) for x in per_query),
        "total_role_recoveries": sum(int(x["probe_role_recoveries"]) for x in per_query),
        "fallback_rate": len(fallback) / len(pairs) if pairs else None,
        "fallback_accuracy": accuracy(fallback), "non_fallback_accuracy": accuracy(other),
        "accuracy": accuracy(pairs),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["mccr", "mccr_v2"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report: dict[str, Any] = {}
    for path in sorted((args.output_root / "diagnostics").glob(f"*.seed{args.seed}.jsonl")):
        benchmark, method = path.name.split(".")[:2]
        if method not in args.methods:
            continue
        result_path = args.output_root / benchmark / method / f"seed{args.seed}.json"
        groups = json.loads(result_path.read_text(encoding="utf-8"))["groups"] if result_path.exists() else []
        report.setdefault(benchmark, {})[method] = summarize(load_jsonl(path), groups)
    keys = ["n_queries", "accuracy", "share_L_prime_zero", "mean_K", "mean_candidates", "share_rank_saturated",
            "core_share_with_prior", "core_share_prior_echoed_once", "core_share_response_only",
            "prior_clusters_in_core", "fallback_rate", "fallback_accuracy", "non_fallback_accuracy",
            "mean_core_misaligned", "mean_valid_probes", "total_probe_dropped", "total_role_recoveries"]
    lines = ["# MCCR mechanism diagnostics", "",
             "See docs/MCCR_REVIEW.md §1.2: P2 (rank saturation / L′=0), P3 (core ≈ prior view), P4 (fallback share).", ""]
    for benchmark, methods in report.items():
        names = list(methods)
        lines += [f"## {benchmark}", "", "| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for key in keys:
            lines.append(f"| {key} | " + " | ".join(fmt(methods[n].get(key)) for n in names) + " |")
        for name in names:
            lines.append(f"\n`{name}` L′ distribution: {methods[name]['L_prime_distribution']}; "
                         f"core-rank distribution: {methods[name]['core_rank_distribution']}")
        lines.append("")
    (args.output_root / "MCCR_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_root / "mccr_diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output_root / "MCCR_DIAGNOSTICS.md")


if __name__ == "__main__":
    main()
