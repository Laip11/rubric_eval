"""Metrics and resampling procedures specified in method.md."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import binomtest, kendalltau, pearsonr, spearmanr


def _corr(fn, x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = fn(x, y)
    value = getattr(value, "statistic", value[0] if isinstance(value, tuple) else value)
    return None if not np.isfinite(value) else float(value)


def _rows(groups: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "pairwise":
        return [dict(pair, qid=group["qid"]) for group in groups for pair in group.get("pairs", []) if pair.get("gold") in ("A", "B")]
    return [dict(item, qid=group["qid"]) for group in groups for item in group.get("items", [])]


def compute_metrics(groups: list[dict[str, Any]], *, bootstrap: int = 0, seed: int = 2026) -> dict[str, Any]:
    kind = groups[0]["kind"] if groups else "pairwise"
    rows = _rows(groups, kind)
    costs = [group.get("cost", {}) for group in groups]
    n_samples = len(rows)
    n_ties = sum(pair.get("gold") == "tie" for group in groups for pair in group.get("pairs", []))
    result: dict[str, Any] = {"kind": kind, "n_queries": len(groups), "n_samples": n_samples,
                              "n_ties_excluded": n_ties, "calls": float(sum(x.get("calls", 0) for x in costs)),
                              "prompt_tokens": float(sum(x.get("prompt_tokens", 0) for x in costs)),
                              "completion_tokens": float(sum(x.get("completion_tokens", 0) for x in costs)),
                              "wall_seconds": float(sum(x.get("wall_seconds", 0) for x in groups))}
    result["avg_calls_per_sample"] = result["calls"] / n_samples if n_samples else 0.0
    result["avg_input_tokens_per_sample"] = result["prompt_tokens"] / n_samples if n_samples else 0.0
    result["avg_output_tokens_per_sample"] = result["completion_tokens"] / n_samples if n_samples else 0.0
    if kind == "pairwise":
        def parse_failed(row: dict[str, Any]) -> bool:
            return bool(row.get("parse_error", row.get("prediction") not in ("A", "B") and not row.get("order_disagreement", False)))
        result.update({"accuracy": float(np.mean([row.get("prediction") == row["gold"] for row in rows])) if rows else None,
                       "n_parse_errors": sum(parse_failed(row) for row in rows),
                       "n_order_disagreements": sum(bool(row.get("order_disagreement")) for row in rows),
                       "n_model_score_ties": sum(bool(row.get("model_score_tie")) for row in rows),
                       "fallback_rate": float(np.mean([bool(row.get("fallback")) for row in rows])) if rows else 0.0})
    else:
        valid = [row for row in rows if row.get("score") is not None and row.get("human_score") is not None]
        predicted = [float(row["score"]) for row in valid]
        human = [float(row["human_score"]) for row in valid]
        group_taus = []
        for group in groups:
            items = [x for x in group.get("items", []) if x.get("score") is not None]
            if len(items) >= 3:
                tau = _corr(kendalltau, [float(x["score"]) for x in items], [float(x["human_score"]) for x in items])
                if tau is not None:
                    group_taus.append(tau)
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid:
            by_model[str(row.get("model", "unknown"))].append(row)
        model_pred = [float(np.mean([x["score"] for x in values])) for values in by_model.values()]
        model_human = [float(np.mean([x["human_score"] for x in values])) for values in by_model.values()]
        result.update({"pearson": _corr(pearsonr, predicted, human), "spearman": _corr(spearmanr, predicted, human),
                       "kendall": _corr(kendalltau, predicted, human),
                       "within_query_kendall": float(np.mean(group_taus)) if group_taus else None,
                       "system_spearman": _corr(spearmanr, model_pred, model_human),
                       "n_valid": len(valid), "n_parse_errors": n_samples - len(valid),
                       "n_groups_for_within": len(group_taus), "n_models": len(by_model)})
    if bootstrap > 0 and rows:
        result["ci95"] = bootstrap_ci(groups, kind, n_boot=bootstrap, seed=seed)
    return result


def bootstrap_ci(groups: list[dict[str, Any]], kind: str, *, n_boot: int = 1000, seed: int = 2026) -> dict[str, list[float | None]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        indices = rng.integers(0, len(groups), len(groups))
        sampled = [groups[int(i)] for i in indices]
        metric = compute_metrics(sampled, bootstrap=0)
        keys = ["accuracy"] if kind == "pairwise" else ["pearson", "spearman", "kendall", "within_query_kendall", "system_spearman"]
        for key in keys:
            value = metric.get(key)
            if value is not None and np.isfinite(value):
                values[key].append(float(value))
    return {key: [float(np.percentile(values[key], 2.5)), float(np.percentile(values[key], 97.5))] if values[key] else [None, None]
            for key in (["accuracy"] if kind == "pairwise" else ["pearson", "spearman", "kendall", "within_query_kendall", "system_spearman"])}


def mcnemar(paor_groups: list[dict[str, Any]], baseline_groups: list[dict[str, Any]]) -> dict[str, Any]:
    def correctness(groups: list[dict[str, Any]]) -> dict[str, bool]:
        return {pair["id"]: pair.get("prediction") == pair.get("gold")
                for group in groups for pair in group.get("pairs", []) if pair.get("gold") in ("A", "B")}
    a, b = correctness(paor_groups), correctness(baseline_groups)
    if set(a) != set(b):
        raise ValueError("McNemar requires identical non-tie pair IDs for both methods")
    common = sorted(a)
    a_only = sum(a[x] and not b[x] for x in common)
    b_only = sum(b[x] and not a[x] for x in common)
    p = float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0
    return {"n": len(common), "method_correct_baseline_wrong": a_only, "baseline_correct_method_wrong": b_only, "p_value": p}


def paired_bootstrap_difference(paor_groups: list[dict[str, Any]], baseline_groups: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 2026) -> dict[str, Any]:
    if not paor_groups or paor_groups[0]["kind"] != "pointwise":
        return {}
    base_by_qid = {x["qid"]: x for x in baseline_groups}
    common = [x for x in paor_groups if x["qid"] in base_by_qid]
    rng = np.random.default_rng(seed)
    differences: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sampled = [common[int(i)] for i in rng.integers(0, len(common), len(common))]
        other = [base_by_qid[x["qid"]] for x in sampled]
        a, b = compute_metrics(sampled), compute_metrics(other)
        for key in ["pearson", "spearman", "kendall", "within_query_kendall", "system_spearman"]:
            if a.get(key) is not None and b.get(key) is not None:
                differences[key].append(a[key] - b[key])
    output = {}
    for key, values in differences.items():
        if values:
            output[key] = {"mean_difference": float(np.mean(values)), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
                           "p_value": float(2 * min(np.mean(np.asarray(values) <= 0), np.mean(np.asarray(values) >= 0)))}
    return output


def aggregate_seed_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_seed:
        return {}
    output: dict[str, Any] = {"n_seeds": len(per_seed), "kind": per_seed[0].get("kind"),
                               "ci95": per_seed[0].get("ci95", {})}
    keys = sorted({key for metric in per_seed for key, value in metric.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    for key in keys:
        values = [float(metric[key]) for metric in per_seed if isinstance(metric.get(key), (int, float)) and np.isfinite(metric[key])]
        if values:
            output[key] = float(np.mean(values))
            output[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    output["per_seed"] = per_seed
    return output
