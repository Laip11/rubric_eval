#!/usr/bin/env python3
"""Run dev-only MCCR coverage ablations from cached shared stages."""
from __future__ import annotations

import argparse
import asyncio
import copy
import html
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mccr_core import orthogonal_residual_select
from mccr_pipeline import MCCRPipeline
from metrics import compute_metrics, mcnemar
from offline_vllm import OfflineLLMPool
from paor_data import drop_human_ties, load_records, make_groups, select_groups, split_qids
from paor_pipeline import load_json, safe_name, save_json

VARIANTS = {
    "core_030": {"tau_core": 0.30, "tau_var": 0.04, "L": 4},
    "residual_002_L8": {"tau_core": 0.50, "tau_var": 0.02, "L": 8},
    "coverage_030_002_L8": {"tau_core": 0.30, "tau_var": 0.02, "L": 8},
}


def build_variant_cache(base: dict[str, Any], *, tau_core: float, tau_var: float, L: int,
                        rho_min: float, N: int) -> dict[str, Any]:
    cache = copy.deepcopy(base)
    for key in ("target_scores", "cost_checkpoint", "_cost_offset"):
        cache.pop(key, None)
    criteria = cache.get("R_core_initial", cache["R_core"]) + cache["R_ext_candidates"]  # F_core columns follow R_core_initial
    rows = len(cache["F_core"])  # valid probes (+ counterfactuals); may be < N when probes were dropped
    core_scores = np.asarray(cache["F_core"], dtype=float).reshape(rows, -1)
    ext_scores = np.asarray(cache["F_ext"], dtype=float).reshape(rows, -1)
    scores = np.concatenate([core_scores, ext_scores], axis=1)
    if scores.shape != (rows, len(criteria)):
        raise ValueError(f"Shared probe score shape {scores.shape} does not match {len(criteria)} criteria")
    core_indices = [i for i, item in enumerate(criteria) if item["rho"] >= tau_core]
    core_fallback = False
    if not core_indices:
        ordered = sorted(range(len(criteria)), key=lambda i: criteria[i]["rho"], reverse=True)
        if not ordered:
            raise ValueError("MCCR cache has no medoids")
        core_indices = [ordered[0]]
        candidate_indices = ordered[1:]
        core_fallback = True
    else:
        candidate_indices = [i for i in range(len(criteria)) if i not in set(core_indices)]
    new_core_scores = scores[:, core_indices]
    new_candidate_scores = scores[:, candidate_indices]
    supports = np.asarray([criteria[i]["rho"] for i in candidate_indices], dtype=float)
    selection = orthogonal_residual_select(new_core_scores, new_candidate_scores, supports,
                                           max_L=L, tau_var=tau_var, rho_min=rho_min)
    cache["R_core"] = [criteria[i] for i in core_indices]
    cache["R_core_initial"] = cache["R_core"]
    cache["core_weights"] = (np.asarray([criteria[i]["rho"] for i in core_indices]) /
                             max(sum(criteria[i]["rho"] for i in core_indices), 1e-12)).tolist()
    cache["R_ext_candidates"] = [criteria[i] for i in candidate_indices]
    cache["R_ext"] = [cache["R_ext_candidates"][i] for i in selection["selected"]]
    cache["core_fallback"] = core_fallback
    cache["F_core"] = new_core_scores.tolist()
    cache["F_ext"] = new_candidate_scores.tolist()
    cache["selection"] = selection
    cache["cost_checkpoint"] = {}
    return cache


def rubric_stats(groups: list[dict[str, Any]]) -> dict[str, float]:
    final = []
    core = []
    ext = []
    raw = []
    for group in groups:
        diagnostic = group["diagnostic"]
        core.append(len(diagnostic["R_core"]))
        ext.append(len(diagnostic["R_ext"]))
        final.append(core[-1] + ext[-1])
        raw.append(len(diagnostic["U"]))
    return {
        "raw_mean": float(statistics.mean(raw)),
        "core_mean": float(statistics.mean(core)),
        "ext_mean": float(statistics.mean(ext)),
        "final_mean": float(statistics.mean(final)),
        "final_median": float(statistics.median(final)),
        "final_min": min(final),
        "final_max": max(final),
    }


def make_pipeline(pool: OfflineLLMPool, cache_root: Path, config: dict[str, Any], name: str,
                  variant: dict[str, Any], tokenizer: Any) -> MCCRPipeline:
    pipeline = MCCRPipeline(
        pool, cache_root, embedding_fn=lambda texts: np.empty((0, 384)), tokenizer=tokenizer,
        N=config["N"], L=variant["L"], probe_max_tokens=config["probe_max_tokens"],
        query_max_tokens=config["query_max_tokens"], response_max_tokens=config["response_max_tokens"],
        probe_max_attempts=config["probe_max_attempts"], theta_sim=config["theta_sim"],
        tau_core=variant["tau_core"], beta=config["beta"], tau_var=variant["tau_var"],
        rho_min=config["rho_min"], gamma=config["gamma"], eta=config["eta"],
        cache_signature=json.dumps({"coverage_ablation": name, **variant}, sort_keys=True),
    )
    pipeline.method_name = f"mccr_{name}"
    return pipeline


def diagnostic_line(result: dict[str, Any]) -> str:
    return json.dumps({"qid": result["qid"], **result["diagnostic"]}, ensure_ascii=False,
                      default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n"


def rubric_lines(diagnostic: dict[str, Any]) -> list[str]:
    lines = []
    for item in diagnostic["R_core"]:
        lines.append(f"- **[核心]** `{item['id']}`，支持度 `rho={item['rho']:.4f}`：{item['criterion']}")
    sigmas = diagnostic["selection"]["sigmas"]
    for index, item in enumerate(diagnostic["R_ext"]):
        sigma = sigmas[index] if index < len(sigmas) else 0.0
        lines.append(f"- **[补充]** `{item['id']}`，支持度 `rho={item['rho']:.4f}`，残差 `sigma={sigma:.4f}`：{item['criterion']}")
    return lines


def example_sections(benchmark: str, raw_groups: list[dict[str, Any]], default_groups: list[dict[str, Any]],
                     coverage_groups: list[dict[str, Any]], count: int) -> list[str]:
    queries = {group["qid"]: group for group in raw_groups}
    default = {group["qid"]: group["diagnostic"] for group in default_groups}
    coverage = {group["qid"]: group["diagnostic"] for group in coverage_groups}
    candidates = []
    for qid in sorted(default):
        if qid not in coverage or qid not in queries:
            continue
        query = queries[qid]["query"]
        if len(query) > 6000:
            continue
        old_count = len(default[qid]["R_core"]) + len(default[qid]["R_ext"])
        new_count = len(coverage[qid]["R_core"]) + len(coverage[qid]["R_ext"])
        metadata = queries[qid]["pairs"][0].get("metadata", {}) if queries[qid]["pairs"] else {}
        subset = str(metadata.get("subset") or metadata.get("source") or "unknown")
        candidates.append((new_count - old_count, -len(query), qid, subset))
    candidates.sort(reverse=True)
    selected = []
    used_subsets = set()
    for _, _, qid, subset in candidates:
        if subset in used_subsets:
            continue
        selected.append(qid)
        used_subsets.add(subset)
        if len(selected) == count:
            break
    if len(selected) < count:
        for _, _, qid, _ in candidates:
            if qid not in selected:
                selected.append(qid)
            if len(selected) == count:
                break
    lines = []
    for qid in selected:
        group = queries[qid]
        metadata = group["pairs"][0].get("metadata", {}) if group["pairs"] else {}
        subset = metadata.get("subset") or metadata.get("source") or "unknown"
        old = default[qid]
        new = coverage[qid]
        lines += ["", f"### {benchmark} / `{qid}`", "", f"- 子集：`{subset}`",
                  f"- 默认配置：原始 {len(old['U'])} 条，聚类 {len(old['clusters'])} 个，最终 {len(old['R_core']) + len(old['R_ext'])} 条。",
                  f"- 组合覆盖配置：原始 {len(new['U'])} 条，聚类 {len(new['clusters'])} 个，最终 {len(new['R_core']) + len(new['R_ext'])} 条。",
                  "", "#### 完整 Query", "", f"<pre>{html.escape(group['query'])}</pre>", "",
                  "#### 默认配置最终 Rubric", "", *rubric_lines(old), "",
                  "#### 组合覆盖配置最终 Rubric", "", *rubric_lines(new), ""]
    return lines


def write_report(output_root: Path, raw_groups: dict[str, list[dict[str, Any]]],
                 results: dict[str, dict[str, list[dict[str, Any]]]],
                 metrics: dict[str, dict[str, dict[str, Any]]],
                 comparisons: dict[str, dict[str, dict[str, Any]]]) -> None:
    lines = ["# MCCR Dev 覆盖率消融与完整 Rubric 示例", "",
             "> 本实验只使用固定 dev 集。未根据 test 结果修改正式 test 超参数，也不把 dev 最优配置解释为 test 改进。", "",
             "## 协议", "",
             "- 默认配置：`tau_core=0.50`、`tau_var=0.04`、`L=4`。",
             "- 人类 gold tie 在固定划分后过滤；所有方法使用相同 pair ID；无法区分和解析失败保留在分母中计错。",
             "- 变体复用探测、原始候选、聚类和全部 medoid 的探测评分，只重新执行选择、目标 response 评分与必要 fallback。",
             "- 变体成本列是**新增目标评分/fallback 成本**，不是冷启动总成本；默认 MCCR 的成本列是完整冷启动成本。",
             "- CI 为 query-level bootstrap 1,000 次；dev 样本较小，显著性仅用于探索。", ""]
    for benchmark in ("rewardbench2", "helpsteer3"):
        lines += ["", f"## {benchmark} 消融结果", "",
                  f"正式 dev pair 数：**{metrics[benchmark]['mccr']['n_samples']}**。",
                  "", "| 配置 | tau_core | tau_var | L | 最终 rubric 均值 | 中位数 | 范围 | Accuracy | 95% CI | Fallback | 调用/pair | p vs 默认 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        configs = {"mccr": {"tau_core": 0.5, "tau_var": 0.04, "L": 4}, **VARIANTS}
        for name, config in configs.items():
            metric = metrics[benchmark][name]
            stats = rubric_stats(results[benchmark][name])
            ci = metric["ci95"]["accuracy"]
            comparison = comparisons[benchmark].get(name, {})
            p_value = comparison.get("mccr", {}).get("significance", {}).get("p_value")
            p_text = "—" if p_value is None else f"{p_value:.4g}"
            lines.append(f"| {name} | {config['tau_core']:.2f} | {config['tau_var']:.2f} | {config['L']} | "
                         f"{stats['final_mean']:.2f} | {stats['final_median']:.0f} | {stats['final_min']}–{stats['final_max']} | "
                         f"{metric['accuracy']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {metric['fallback_rate']:.4f} | "
                         f"{metric['avg_calls_per_sample']:.2f} | {p_text} |")
        lines += ["", "Direct 对照：", "", "| 方法 | Accuracy | 95% CI | 调用/pair |", "|---|---:|---:|---:|"]
        for name in ("direct", "direct_k"):
            metric = metrics[benchmark][name]
            ci = metric["ci95"]["accuracy"]
            lines.append(f"| {name} | {metric['accuracy']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {metric['avg_calls_per_sample']:.2f} |")
    lines += ["", "# 完整 Query 与筛选 Rubric 示例", "",
              "以下示例来自 dev，固定比较默认配置与组合覆盖配置 `coverage_030_002_L8`；每条准则均完整列出，未省略。"]
    lines += example_sections("rewardbench2", raw_groups["rewardbench2"], results["rewardbench2"]["mccr"],
                              results["rewardbench2"]["coverage_030_002_L8"], 2)
    lines += example_sections("helpsteer3", raw_groups["helpsteer3"], results["helpsteer3"]["mccr"],
                              results["helpsteer3"]["coverage_030_002_L8"], 1)
    lines += ["", "## 解释限制", "",
              "变体间 Accuracy 差异来自 dev，不能据此承诺 test 提升。最终是否采用某一配置，还需要独立确认实验，并同时评估冷启动总成本、fallback、顺序不一致和准则质量。", ""]
    (output_root / "DEV_COVERAGE_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    output_root = args.output_root
    cache_root = args.cache_root
    manifest = load_json(output_root / "manifest.json", {})
    config = manifest["config"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    role_configs = {"strong": {"model_path": config["model_path"], "model_name": config["strong_model"],
                               "gpus": config["judge_gpus"], "memory": config["offline_memory"],
                               "max_num_seqs": config["judge_max_num_seqs"],
                               "max_model_len": config["judge_max_model_len"], "replica_per_gpu": True}}
    pool = OfflineLLMPool(role_configs, wait_seconds=config["offline_wait_seconds"],
                          dynamic_gpus=manifest["execution"]["gpus"])
    raw_groups: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        for benchmark in args.benchmarks:
            records = load_records(benchmark, args.data_root)
            groups = make_groups(records, benchmark)
            splits = split_qids(groups, cache_root / "splits")
            selected = drop_human_ties(select_groups(groups, splits, "dev", None))
            raw_groups[benchmark] = selected
            results[benchmark] = {}
            metrics[benchmark] = {}
            comparisons[benchmark] = {}
            for method in ("mccr", "direct", "direct_k"):
                payload = load_json(output_root / benchmark / method / "seed0.json", None)
                if payload is None:
                    raise FileNotFoundError(f"Run the default dev baseline first: {benchmark}/{method}")
                results[benchmark][method] = payload["groups"]
                metrics[benchmark][method] = compute_metrics(payload["groups"], bootstrap=args.bootstrap, seed=2026)
            for name, variant in VARIANTS.items():
                pipeline = make_pipeline(pool, cache_root, config, name, variant, tokenizer)
                for group in selected:
                    existing = pipeline._cache(benchmark, group, 0)
                    if existing:
                        continue
                    base_path = cache_root / "mccr" / benchmark / "seed0" / f"{safe_name(group['qid'])}.json"
                    base = load_json(base_path, None)
                    if base is None:
                        raise FileNotFoundError(f"Missing shared MCCR cache: {base_path}")
                    cache = build_variant_cache(base, tau_core=variant["tau_core"], tau_var=variant["tau_var"],
                                                L=variant["L"], rho_min=config["rho_min"], N=config["N"])
                    cache.update({"qid": group["qid"], "query": group["query"], "seed": 0,
                                  "cache_signature": pipeline.cache_signature})
                    save_json(pipeline._path(benchmark, group["qid"], 0), cache)
                method_name = pipeline.method_name
                output_path = output_root / benchmark / method_name / "seed0.json"
                if output_path.exists():
                    variant_results = load_json(output_path, {})["groups"]
                else:
                    started = time.perf_counter()
                    variant_results = await pipeline.run(benchmark, selected, seed=0)
                    save_json(output_path, {"benchmark": benchmark, "method": method_name, "seed": 0, "split": "dev",
                                            "ablation": variant, "cost_protocol": "marginal_target_and_fallback",
                                            "groups": variant_results})
                    diagnostic_path = output_root / "diagnostics" / f"{benchmark}.{method_name}.seed0.jsonl"
                    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                    diagnostic_path.write_text("".join(diagnostic_line(result) for result in variant_results), encoding="utf-8")
                    print(f"{benchmark}/{name} finished in {time.perf_counter() - started:.1f}s", flush=True)
                metric = compute_metrics(variant_results, bootstrap=args.bootstrap, seed=2026)
                save_json(output_root / benchmark / method_name / "seed0.metrics.json", metric)
                save_json(output_root / benchmark / method_name / "rubric_stats.json", rubric_stats(variant_results))
                results[benchmark][name] = variant_results
                metrics[benchmark][name] = metric
                for baseline in ("mccr", "direct_k"):
                    comparison = {"method": method_name, "baseline": baseline,
                                  "significance": mcnemar(variant_results, results[benchmark][baseline])}
                    comparisons[benchmark].setdefault(name, {})[baseline] = comparison
                    save_json(output_root / benchmark / method_name / f"vs_{baseline}.json", comparison)
            save_json(output_root / f"{benchmark}_ablation_metrics.json", metrics[benchmark])
        write_report(output_root, raw_groups, results, metrics, comparisons)
        save_json(output_root / "ablation_config.json", {"split": "dev", "variants": VARIANTS,
                  "shared_stages": ["probes", "raw_rubrics", "clusters", "all_medoid_probe_scores"],
                  "cost_protocol": "variants_report_marginal_target_and_fallback"})
        print(output_root / "DEV_COVERAGE_ABLATION.md", flush=True)
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = HERE.parent
    parser.add_argument("--data-root", type=Path, default=root / "generating_dynamic_rubric" / "data")
    parser.add_argument("--output-root", type=Path, default=root / "outputs" / "mccr_dev_ablation_v1")
    parser.add_argument("--cache-root", type=Path, default=root / "cache" / "mccr_dev_ablation_v1")
    parser.add_argument("--benchmarks", nargs="+", default=["rewardbench2", "helpsteer3"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
