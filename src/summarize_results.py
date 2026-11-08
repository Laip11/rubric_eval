#!/usr/bin/env python3
"""Aggregate raw E1 outputs into machine-readable and Markdown result tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from metrics import aggregate_seed_metrics, compute_metrics
from paor_data import PAIRWISE


def load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def confidence_interval(summary: dict[str, Any], key: str) -> str:
    values = summary.get("ci95", {}).get(key)
    return f"[{values[0]:.4f}, {values[1]:.4f}]" if values and values[0] is not None else "—"


def mean_std(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key)
    if value is None:
        return "—"
    if summary.get("n_seeds") == 1:
        return f"{value:.4f}"
    std = summary.get(key + "_std", 0.0)
    return f"{value:.4f} ± {std:.4f}"


def cold_cost(summary: dict[str, Any], key: str) -> float:
    per_seed = summary.get("per_seed") or []
    return float(per_seed[0].get(key, 0.0)) if per_seed else 0.0


def collect(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output: dict[str, Any] = {"benchmarks": {}, "macro": {}}
    rows = []
    for benchmark_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        benchmark = benchmark_dir.name
        comparisons = {path.name: load(path) for path in sorted(benchmark_dir.glob("*_vs_*.json"))}
        primary_significance = (comparisons.get("mccr_vs_strongest.json") or
                                comparisons.get("paor_vs_strongest.json") or {})
        benchmark_result = {"methods": {}, "significance": primary_significance, "comparisons": comparisons}
        for method_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            seed_metrics = []
            for path in sorted(method_dir.glob("seed*.metrics.json")):
                metric = load(path)
                if metric:
                    seed_metrics.append(metric)
            if not seed_metrics:
                raw_paths = sorted(method_dir.glob("seed*.json"))
                for path in raw_paths:
                    if path.name.endswith(".metrics.json"):
                        continue
                    payload = load(path)
                    if payload and "groups" in payload:
                        seed_metrics.append(compute_metrics(payload["groups"]))
            if not seed_metrics:
                continue
            summary = aggregate_seed_metrics(seed_metrics)
            benchmark_result["methods"][method_dir.name] = summary
            row = {"benchmark": benchmark, "method": method_dir.name}
            cost_keys = {"avg_calls_per_sample", "avg_input_tokens_per_sample", "avg_output_tokens_per_sample", "wall_seconds"}
            for key in ["kind", "n_queries", "n_samples", "accuracy", "accuracy_std", "fallback_rate", "pearson", "pearson_std",
                        "spearman", "spearman_std", "kendall", "kendall_std", "within_query_kendall", "within_query_kendall_std",
                        "system_spearman", "system_spearman_std", "avg_calls_per_sample", "avg_input_tokens_per_sample",
                        "avg_output_tokens_per_sample", "wall_seconds", "n_parse_errors"]:
                source = seed_metrics[0] if key in cost_keys else summary
                if key in source:
                    row[key] = source[key]
            row["cost_protocol"] = "seed0_cold_cache"
            rows.append(row)
        if benchmark == "mtbench":
            turn_results = {}
            for turn in [1, 2]:
                turn_methods = {}
                for method_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
                    seed_metrics = []
                    for raw_path in sorted(method_dir.glob("seed*.json")):
                        if raw_path.name.endswith(".metrics.json"):
                            continue
                        payload = load(raw_path)
                        if payload and "groups" in payload:
                            selected = [g for g in payload["groups"] if f":turn{turn}" in g["qid"]]
                            if selected:
                                seed_metrics.append(compute_metrics(selected))
                    if seed_metrics:
                        turn_methods[method_dir.name] = aggregate_seed_metrics(seed_metrics)
                if turn_methods:
                    turn_results[f"turn{turn}"] = turn_methods
            benchmark_result["turns"] = turn_results
        if benchmark_result["methods"]:
            output["benchmarks"][benchmark] = benchmark_result
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for kind in ["pairwise", "pointwise"]:
            selected = [row for row in method_rows if row.get("kind") == kind]
            if not selected:
                continue
            primary = "accuracy" if kind == "pairwise" else "spearman"
            valid_selected = [row for row in selected if isinstance(row.get(primary), (int, float))]
            values = [row[primary] for row in valid_selected]
            if values:
                output["macro"][f"{method}:{kind}"] = {"n_benchmarks": len(values), primary: sum(values) / len(values),
                                                          "benchmarks": [row["benchmark"] for row in valid_selected]}
    return output, rows


def markdown(report: dict[str, Any], rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    config = manifest.get("config", {})
    seeds = config.get("seeds", [0])
    seed_description = "seed0 单次结果" if seeds == [0] else f"{len(seeds)} 个 seed 的均值 ± 标准差"
    generator_name = config.get("generator_model") or config.get("strong_model", "unknown")
    has_pointwise = any(row.get("kind") == "pointwise" for row in rows)
    metric_description = "Pairwise 在固定非人类 tie 样本集上计算一致率；无法区分、正反顺序不一致和解析失败均保留在分母中并计错，独立评分同分按固定种子分配"
    if has_pointwise:
        metric_description += "；Pointwise 报告实例级 Pearson/Spearman/Kendall、组内 Kendall 和系统级 Spearman"
    max_queries = config.get("max_queries")
    query_description = "固定 test 划分后按协议过滤人类 gold tie（未设置 query 上限）" if max_queries is None else f"每个数据集最多选择 {max_queries} 条 query，再按协议过滤人类 gold tie"
    lines = ["# MCCR / PAOR Pairwise 结果汇总", "", "## 运行范围", "",
             f"- 裁判模型：`{config.get('strong_model', 'unknown')}`；生成模型：`{generator_name}`；弱探测模型：`{config.get('weak_model', 'unknown')}`；后端：`{config.get('backend', 'http')}`。",
             f"- 嵌入模型：`{manifest.get('config', {}).get('embedding_model', 'unknown')}`。",
             f"- 指标：{metric_description}。",
             f"- 指标为 {seed_description}；CI 来自 seed0 的 query 级 bootstrap {config.get('bootstrap', 1000)} 次；成本列为 seed0 冷缓存运行。",
             f"- 数据划分：`{config.get('split', 'test')}`；{query_description}。",
             f"- 确定性方法的非零 seed 复用 seed0：{', '.join(manifest.get('deterministic_reuse', []))}。" if seeds != [0] else "- 当前仅运行 seed0，无跨 seed 复用。",
             f"- HelpSteer2 人类标签：`{manifest.get('label_protocol', {}).get('helpsteer2', 'helpfulness')}`。" if has_pointwise else "- Pointwise 已暂停，不进入本轮主表。", ""]
    lines += ["| 数据集 | 类型 | 正式评估 query 数 | 非 tie/Pointwise 样本数 | 推理前过滤的人类 tie |", "|---|---|---:|---:|---:|"]
    for benchmark, info in manifest.get("benchmarks", {}).items():
        kind = "Pairwise" if benchmark in PAIRWISE else "Pointwise"
        method_summaries = report["benchmarks"].get(benchmark, {}).get("methods", {})
        selected_samples = next((x.get("n_samples") for x in method_summaries.values() if x.get("n_samples") is not None), info.get("samples_selected", "—"))
        ties = info.get("tie_pairs_removed", next((x.get("n_ties_excluded") for x in method_summaries.values() if x.get("n_ties_excluded") is not None), 0))
        sample_text = f"{selected_samples:.0f}" if isinstance(selected_samples, (int, float)) else str(selected_samples)
        lines.append(f"| {benchmark} | {kind} | {info.get('queries_selected', '—')} | {sample_text} | {ties:.0f} |")
    for benchmark, result in report["benchmarks"].items():
        methods = result["methods"]
        kind = next(iter(methods.values())).get("kind", "pairwise")
        lines += ["", f"## {benchmark}", ""]
        if kind == "pairwise":
            lines += ["| 方法 | Accuracy | 95% CI（seed0） | Fallback | 顺序不一致 | 模型同分 | 解析失败 | 平均调用/样本 | 平均输入 token/样本 | 平均输出 token/样本 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for method, summary in methods.items():
                lines.append(f"| {method} | {mean_std(summary, 'accuracy')} | {confidence_interval(summary, 'accuracy')} | {summary.get('fallback_rate', 0):.4f} | {summary.get('n_order_disagreements', 0):.0f} | {summary.get('n_model_score_ties', 0):.0f} | {summary.get('n_parse_errors', 0):.2f} | {cold_cost(summary, 'avg_calls_per_sample'):.2f} | {cold_cost(summary, 'avg_input_tokens_per_sample'):.1f} | {cold_cost(summary, 'avg_output_tokens_per_sample'):.1f} |")
        else:
            lines += ["| 方法 | Pearson | Spearman | 95% CI（seed0） | Kendall | 组内 Kendall | 系统 Spearman | 解析失败 | 平均调用/样本 | 平均输出 token/样本 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for method, summary in methods.items():
                lines.append(f"| {method} | {mean_std(summary, 'pearson')} | {mean_std(summary, 'spearman')} | {confidence_interval(summary, 'spearman')} | {mean_std(summary, 'kendall')} | {mean_std(summary, 'within_query_kendall')} | {mean_std(summary, 'system_spearman')} | {summary.get('n_parse_errors', 0):.2f} | {cold_cost(summary, 'avg_calls_per_sample'):.2f} | {cold_cost(summary, 'avg_output_tokens_per_sample'):.1f} |")
        significance = result.get("significance") or {}
        if significance:
            method_name = significance.get("method", "paor").upper()
            lines += ["", f"{method_name} 对最强 baseline `{significance.get('baseline', 'unknown')}` 的显著性检验：`{json.dumps(significance.get('significance', {}), ensure_ascii=False)}`。"]
        comparison = result.get("comparisons", {}).get("mccr_vs_paor.json")
        if comparison:
            lines += ["", f"MCCR 对旧 PAOR 的显著性检验：`{json.dumps(comparison.get('significance', {}), ensure_ascii=False)}`。"]
        for turn, turn_methods in result.get("turns", {}).items():
            lines += ["", f"### MT-Bench {turn}", "", "| 方法 | Accuracy | 样本数 |", "|---|---:|---:|"]
            for method, summary in turn_methods.items():
                lines.append(f"| {method} | {mean_std(summary, 'accuracy')} | {summary.get('n_samples', 0):.0f} |")
    if report["macro"]:
        lines += ["", "## 宏平均（按数据集等权）", "", "| 方法 | 类型 | 主指标 | 数据集数 |", "|---|---|---:|---:|"]
        for key, value in report["macro"].items():
            method, kind = key.rsplit(":", 1)
            primary = "accuracy" if kind == "pairwise" else "spearman"
            lines.append(f"| {method} | {kind} | {value[primary]:.4f} | {value['n_benchmarks']} |")
    lines += ["", "## 可复现性说明", "",
              "原始逐 query 输出、诊断 JSONL、每个 seed 的 metrics 和运行 manifest 均保存在本目录；本表不对未运行的数据集或方法做插值。",
              "人类 gold tie 在固定划分后、推理前统一过滤，不进入 Accuracy、显著性、bootstrap 或成本；模型无法区分和解析失败从不从分母剔除。`summary.csv` 的 `wall_seconds` 未换算成单张 A100；本次 HelpSteer3 离线过滤迁移按逐 query 均值比例估算墙钟，不冒充新的实测运行。",
              "CheckEval 的全局 checklist 生成每个数据集仅 1 次，未摊入逐 query 成本。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "main")
    args = parser.parse_args()
    report, rows = collect(args.output_root)
    manifest = load(args.output_root / "manifest.json", {})
    (args.output_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_root / "SUMMARY.md").write_text(markdown(report, rows, manifest), encoding="utf-8")
    print(json.dumps(report["macro"], ensure_ascii=False, indent=2))
    print(args.output_root / "SUMMARY.md")


if __name__ == "__main__":
    main()
