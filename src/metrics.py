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


def pair_metadata(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map pair ID → loader metadata (subset, style indices, ...) from the evaluated data groups."""
    return {pair["id"]: pair.get("metadata", {}) for group in groups for pair in group.get("pairs", [])}


def lenient_credit(row: dict[str, Any]) -> float:
    """1 for correct, 0.5 for an unresolved or randomly broken tie, 0 for wrong or unparsable."""
    if row.get("order_disagreement") or (row.get("model_score_tie") and not row.get("fallback")):
        return 0.5
    return float(row.get("prediction") == row["gold"])


def param_weighted_tokens(cost: dict[str, Any], role_params: dict[str, float] | None) -> float | None:
    """Σ_role params(role) × tokens(role): a FLOPs proxy (billion-parameter tokens)."""
    if not role_params:
        return None
    return float(sum(float(role_params.get(key.split(":", 1)[1], 0.0)) * float(value)
                     for key, value in cost.items() if key.startswith("tokens:")))


def rmbench_breakdown(rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Official RM-Bench style-matrix accuracies: hard (chosen less styled than rejected),
    normal (same style) and easy, each averaged per domain and then across domains."""
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        info = meta.get(row["id"], {})
        if "style_chosen" not in info:
            continue
        i, j = int(info["style_chosen"]), int(info["style_rejected"])
        level = "hard" if i < j else "normal" if i == j else "easy"
        correct = float(row.get("prediction") == row["gold"])
        cells[str(info.get("subset", "all"))][level].append(correct)
        cells[str(info.get("subset", "all"))]["total"].append(correct)
    if not cells:
        return {}
    per_domain = {domain: {level: float(np.mean(values)) for level, values in levels.items()} for domain, levels in cells.items()}
    output = {"per_domain": per_domain}
    for level in ["hard", "normal", "easy", "total"]:
        values = [levels[level] for levels in per_domain.values() if level in levels]
        output[level] = float(np.mean(values)) if values else None
    return output


def compute_metrics(groups: list[dict[str, Any]], *, bootstrap: int = 0, seed: int = 2026,
                    meta: dict[str, dict[str, Any]] | None = None,
                    role_params: dict[str, float] | None = None) -> dict[str, Any]:
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
        result["accuracy_lenient"] = float(np.mean([lenient_credit(row) for row in rows])) if rows else None
        two_order = [row for row in rows if len(row.get("order_votes") or []) == 2]
        result["position_consistency"] = (float(np.mean([row["order_votes"][0] == row["order_votes"][1] for row in two_order]))
                                          if two_order else None)
        fallback_rows = [row for row in rows if row.get("fallback")]
        result["fallback_accuracy"] = (float(np.mean([row.get("prediction") == row["gold"] for row in fallback_rows]))
                                       if fallback_rows else None)
        non_fallback = [row for row in rows if not row.get("fallback")]
        result["non_fallback_accuracy"] = (float(np.mean([row.get("prediction") == row["gold"] for row in non_fallback]))
                                           if non_fallback else None)
        # Query-level "all pairs correct" = official best-of-N accuracy for RewardBench 2 (1 chosen vs 3 rejected).
        multi = [group for group in groups if sum(p.get("gold") in ("A", "B") for p in group.get("pairs", [])) > 1]
        if multi:
            result["query_all_correct"] = float(np.mean([all(p.get("prediction") == p["gold"] for p in group["pairs"]
                                                             if p.get("gold") in ("A", "B")) for group in multi]))
        if meta:
            by_subset: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                subset = meta.get(row["id"], {}).get("subset")
                if subset is not None:
                    by_subset[str(subset)].append(float(row.get("prediction") == row["gold"]))
            if by_subset:
                result["accuracy_by_subset"] = {key: float(np.mean(values)) for key, values in sorted(by_subset.items())}
                result["n_by_subset"] = {key: len(values) for key, values in sorted(by_subset.items())}
                result["accuracy_subset_macro"] = float(np.mean(list(result["accuracy_by_subset"].values())))
                if multi:
                    by_query: dict[str, list[float]] = defaultdict(list)
                    for group in multi:
                        valid = [p for p in group["pairs"] if p.get("gold") in ("A", "B")]
                        subset = meta.get(valid[0]["id"], {}).get("subset")
                        by_query[str(subset)].append(float(all(p.get("prediction") == p["gold"] for p in valid)))
                    result["query_all_correct_by_subset"] = {k: float(np.mean(v)) for k, v in sorted(by_query.items())}
            rmbench = rmbench_breakdown(rows, meta)
            if rmbench:
                result["rmbench"] = rmbench
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
    flops = [param_weighted_tokens(cost, role_params) for cost in costs]
    if role_params and n_samples:
        result["param_tokens_per_sample"] = float(sum(x or 0.0 for x in flops)) / n_samples
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
        keys = ["accuracy", "accuracy_lenient"] if kind == "pairwise" else ["pearson", "spearman", "kendall", "within_query_kendall", "system_spearman"]
        for key in keys:
            value = metric.get(key)
            if value is not None and np.isfinite(value):
                values[key].append(float(value))
    return {key: [float(np.percentile(values[key], 2.5)), float(np.percentile(values[key], 97.5))] if values[key] else [None, None]
            for key in (["accuracy", "accuracy_lenient"] if kind == "pairwise" else ["pearson", "spearman", "kendall", "within_query_kendall", "system_spearman"])}


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


def paired_accuracy_bootstrap(method_groups: list[dict[str, Any]], baseline_groups: list[dict[str, Any]], *,
                              n_boot: int = 1000, seed: int = 2026) -> dict[str, Any]:
    """Query-level paired bootstrap of the accuracy difference (method − baseline)."""
    def per_query(groups: list[dict[str, Any]]) -> dict[str, list[float]]:
        return {group["qid"]: [float(p.get("prediction") == p["gold"]) for p in group.get("pairs", []) if p.get("gold") in ("A", "B")]
                for group in groups}
    a, b = per_query(method_groups), per_query(baseline_groups)
    qids = sorted(q for q in a if q in b and a[q])
    if not qids:
        return {}
    a_sum = np.asarray([sum(a[q]) for q in qids]); b_sum = np.asarray([sum(b[q]) for q in qids])
    counts = np.asarray([len(a[q]) for q in qids], dtype=float)
    observed = float((a_sum.sum() - b_sum.sum()) / counts.sum())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(qids), len(qids))
        values.append(float((a_sum[idx].sum() - b_sum[idx].sum()) / counts[idx].sum()))
    values = np.asarray(values)
    return {"difference": observed, "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "p_value": float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0))))}


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of comparisons."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[key] = running
    return adjusted


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
