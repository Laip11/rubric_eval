#!/usr/bin/env python3
"""Run MCCR/PAOR and original rubric baselines with offline or HTTP vLLM."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from async_vllm import AsyncLLMPool, check_endpoints
from baselines import BaselineRunner
from mccr_pipeline import MCCRPipeline
from metrics import aggregate_seed_metrics, compute_metrics, mcnemar, paired_bootstrap_difference
from paor_data import BENCHMARKS, drop_human_ties, load_records, make_groups, select_groups, split_qids
from paor_pipeline import PAORPipeline, save_json

LOG = logging.getLogger("paor-main")
DEFAULT_METHODS = ["direct", "checkeval", "dna", "rubrichub", "mccr", "direct_k"]
METHOD_CHOICES = [*DEFAULT_METHODS, "paor", "dynamic"]


def endpoints(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [part.strip() for item in values for part in item.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = HERE.parent
    parser.add_argument("--data-root", type=Path, default=root / "generating_dynamic_rubric" / "data")
    parser.add_argument("--backend", choices=["offline", "http"], default="offline")
    parser.add_argument("--output-root", type=Path, default=root / "outputs" / "main")
    parser.add_argument("--cache-root", type=Path, default=root / "cache")
    parser.add_argument("--benchmarks", nargs="+", default=["rewardbench2", "helpsteer3"], choices=BENCHMARKS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=METHOD_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-pairs-per-query", type=int, default=None)
    parser.add_argument("--strong-endpoints", nargs="*", default=endpoints(os.environ.get("STRONG_ENDPOINTS")))
    parser.add_argument("--weak-endpoints", nargs="*", default=endpoints(os.environ.get("WEAK_ENDPOINTS")))
    parser.add_argument("--generator-endpoints", nargs="*", default=endpoints(os.environ.get("GENERATOR_ENDPOINTS")))
    parser.add_argument("--strong-model", default=os.environ.get("STRONG_MODEL", "qwen25-7b"))
    parser.add_argument("--weak-model", default=os.environ.get("WEAK_MODEL", "qwen25-1_5b"))
    parser.add_argument("--generator-model", default=os.environ.get("GENERATOR_MODEL"))
    parser.add_argument("--model-path", type=Path, default=Path("/data/project/sustech/model/Qwen2.5-7B-Instruct"))
    parser.add_argument("--generator-model-path", type=Path)
    parser.add_argument("--weak-model-path", type=Path)
    parser.add_argument("--judge-gpus", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--generator-gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--weak-gpus", nargs="+", type=int, default=[5])
    parser.add_argument("--judge-max-num-seqs", type=int, default=1024)
    parser.add_argument("--generator-max-num-seqs", type=int, default=512)
    parser.add_argument("--weak-max-num-seqs", type=int, default=1024)
    parser.add_argument("--offline-memory", type=float, default=0.85)
    parser.add_argument("--offline-wait-seconds", type=int, default=1800)
    parser.add_argument("--judge-max-model-len", type=int, default=32768)
    parser.add_argument("--generator-max-model-len", type=int, default=40960)
    parser.add_argument("--weak-max-model-len", type=int, default=16384)
    parser.add_argument("--embedding-model", default="/data/project/sustech/model/Qwen3-Embedding-0.6B")
    parser.add_argument("--query-max-tokens", type=int, default=8192)
    parser.add_argument("--response-max-tokens", type=int, default=4096)
    parser.add_argument("--probe-max-tokens", type=int, default=4096)
    parser.add_argument("--probe-max-attempts", type=int, default=7)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--paor-c", type=float, default=0.8)
    parser.add_argument("--tau-d", type=float, default=0.15)
    parser.add_argument("--tau-rho", type=float, default=0.85)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--theta-sim", type=float, default=0.8)
    parser.add_argument("--tau-core", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--tau-var", type=float, default=0.04)
    parser.add_argument("--rho-min", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--ablation", default="none", choices=["none", "prior_only", "post_only", "no_selection", "random_select", "disc_only", "flat_mean", "discrete_score"])
    parser.add_argument("--max-concurrency", type=int, default=1536)
    parser.add_argument("--per-replica-concurrency", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-health-check", action="store_true")
    return parser.parse_args()


def make_embedding(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu", local_files_only=Path(model_name).exists())
    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return encode


def trim_pairs(groups: list[dict], max_pairs: int | None) -> None:
    if max_pairs is None:
        return
    for group in groups:
        if len(group["pairs"]) > max_pairs:
            group["pairs"] = group["pairs"][:max_pairs]
            used = {x for pair in group["pairs"] for x in (pair["a"], pair["b"])}
            group["responses"] = [x for x in group["responses"] if x["id"] in used]


async def run(args: argparse.Namespace) -> None:
    if "paor" in args.methods and args.generator_max_model_len < args.query_max_tokens + args.N * args.probe_max_tokens + 2048:
        raise ValueError("generator-max-model-len is too small for PAOR posterior expansion")
    if args.judge_max_model_len < args.query_max_tokens + 2 * args.response_max_tokens + 4096:
        raise ValueError("judge-max-model-len is too small for pairwise and atomic judging")
    if any(method in {"paor", "mccr"} for method in args.methods) and args.weak_max_model_len < args.query_max_tokens + args.probe_max_tokens + 512:
        raise ValueError("weak-max-model-len is too small for complete probe generation")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    if args.backend == "offline":
        from offline_vllm import OfflineLLMPool

        needs_generator = any(method in {"dynamic", "checkeval", "dna", "rubrichub", "paor", "mccr"} for method in args.methods)
        needs_weak = any(method in {"paor", "mccr"} for method in args.methods)
        if needs_generator and args.generator_model_path is None:
            raise ValueError("--generator-model-path is required for generation-based methods in offline mode")
        if needs_weak and args.weak_model_path is None:
            raise ValueError("--weak-model-path is required for PAOR in offline mode")
        role_configs = {"strong": {"model_path": args.model_path, "model_name": args.strong_model,
                                   "gpus": args.judge_gpus, "memory": args.offline_memory,
                                   "max_num_seqs": args.judge_max_num_seqs,
                                   "max_model_len": args.judge_max_model_len,
                                   "replica_per_gpu": True}}
        if needs_generator:
            role_configs["generator"] = {"model_path": args.generator_model_path,
                                         "model_name": args.generator_model or args.generator_model_path.name,
                                         "gpus": args.generator_gpus, "memory": args.offline_memory,
                                         "max_num_seqs": args.generator_max_num_seqs,
                                         "max_model_len": args.generator_max_model_len,
                                         "replica_per_gpu": True}
        if needs_weak:
            role_configs["weak"] = {"model_path": args.weak_model_path,
                                    "model_name": args.weak_model, "gpus": args.weak_gpus,
                                    "memory": args.offline_memory, "max_num_seqs": args.weak_max_num_seqs,
                                    "max_model_len": args.weak_max_model_len}
        dynamic_gpus = sorted(set(args.judge_gpus) | set(args.generator_gpus) | set(args.weak_gpus))
        pool = OfflineLLMPool(role_configs, wait_seconds=args.offline_wait_seconds, dynamic_gpus=dynamic_gpus)
    else:
        strong = endpoints(args.strong_endpoints)
        weak = endpoints(args.weak_endpoints)
        generator = endpoints(args.generator_endpoints)
        if not strong:
            raise ValueError("Provide --strong-endpoints or STRONG_ENDPOINTS in HTTP mode")
        if not weak:
            weak = strong
        if not args.no_health_check:
            health = await check_endpoints([*strong, *weak, *generator])
            LOG.info("Endpoint health: %s", health)
            unavailable = [x for x, ok in health.items() if not ok]
            if unavailable:
                raise ConnectionError(f"Unavailable vLLM endpoints: {unavailable}")
        pool = AsyncLLMPool(strong, weak, generator, strong_model=args.strong_model, weak_model=args.weak_model,
                            generator_model=args.generator_model, max_concurrency=args.max_concurrency,
                            per_replica_concurrency=args.per_replica_concurrency)
    needs_embeddings = any(method in {"paor", "mccr"} for method in args.methods)
    embedding_fn = make_embedding(args.embedding_model) if needs_embeddings else None
    baseline = BaselineRunner(pool, args.cache_root, tokenizer=tokenizer,
                              query_max_tokens=args.query_max_tokens,
                              response_max_tokens=args.response_max_tokens)
    manifest = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "execution": {"schedule": "dynamic_per_batch" if args.backend == "offline" else "http",
                              "gpus": dynamic_gpus if args.backend == "offline" else []},
                "deterministic_reuse": ["direct", "dynamic", "checkeval", "dna"],
                "label_protocol": {"helpsteer2": "helpfulness", "biggen_pointwise": "human_score",
                                   "healthbench": "physician-majority rubric pass fraction"},
                "benchmarks": {}, "started_unix": time.time()}
    save_json(args.output_root / "manifest.json", manifest)
    try:
        for benchmark in args.benchmarks:
            LOG.info("Loading %s", benchmark)
            records = load_records(benchmark, args.data_root)
            groups = make_groups(records, benchmark)
            splits = split_qids(groups, args.cache_root / "splits")
            selected = select_groups(groups, splits, args.split, args.max_queries)
            tie_pairs_removed = sum(pair["gold"] == "tie" for group in selected for pair in group["pairs"])
            selected = drop_human_ties(selected)
            trim_pairs(selected, args.max_pairs_per_query)
            rubric_methods = {"mccr", "paor"} & set(args.methods)
            if selected and selected[0]["kind"] == "pointwise" and rubric_methods and set(args.methods) - rubric_methods:
                raise ValueError("Pointwise rubric/baseline comparisons require one shared logprob protocol for every method")
            if not selected:
                raise ValueError(f"No evaluation samples remain for {benchmark} after applying the label protocol")
            LOG.info("%s: %d records, %d selected queries (%d responses, %d pairs)", benchmark, len(records), len(selected),
                     sum(len(x["responses"]) for x in selected), sum(len(x["pairs"]) for x in selected))
            manifest["benchmarks"][benchmark] = {"records": len(records), "queries_total": len(groups),
                                                  "queries_selected": len(selected), "dev_queries": len(splits["dev"]),
                                                  "test_queries": len(splits["test"]), "split": args.split,
                                                  "samples_selected": sum(len(x["pairs"]) if x["kind"] == "pairwise" else len(x["items"]) for x in selected),
                                                  "responses_selected": sum(len(x["responses"]) for x in selected),
                                                  "tie_pairs_removed": tie_pairs_removed,
                                                  "evaluation_protocol": "fixed_non_tie_denominator"}
            save_json(args.output_root / "manifest.json", manifest)
            cache_common = {"version": "mccr-v6-audited-role-recovery", "generator": args.generator_model or str(args.generator_model_path),
                            "judge": args.strong_model, "weak": args.weak_model, "embedding": args.embedding_model,
                            "N": args.N, "L": args.L, "probe_max_tokens": args.probe_max_tokens,
                            "query_max_tokens": args.query_max_tokens, "response_max_tokens": args.response_max_tokens}
            paor = PAORPipeline(pool, args.cache_root, embedding_fn=embedding_fn, tokenizer=tokenizer,
                                N=args.N, L=args.L, c=args.paor_c, tau_d=args.tau_d, tau_rho=args.tau_rho, lam=args.lam,
                                probe_max_tokens=args.probe_max_tokens, query_max_tokens=args.query_max_tokens,
                                response_max_tokens=args.response_max_tokens, probe_max_attempts=args.probe_max_attempts,
                                cache_signature=json.dumps({**cache_common, "method": "paor", "c": args.paor_c,
                                                            "tau_d": args.tau_d, "tau_rho": args.tau_rho, "lam": args.lam}, sort_keys=True)
                                ) if "paor" in args.methods else None
            mccr = MCCRPipeline(pool, args.cache_root, embedding_fn=embedding_fn, tokenizer=tokenizer,
                                N=args.N, L=args.L, probe_max_tokens=args.probe_max_tokens,
                                query_max_tokens=args.query_max_tokens, response_max_tokens=args.response_max_tokens,
                                probe_max_attempts=args.probe_max_attempts, theta_sim=args.theta_sim,
                                tau_core=args.tau_core, beta=args.beta, tau_var=args.tau_var,
                                rho_min=args.rho_min, gamma=args.gamma, eta=args.eta,
                                cache_signature=json.dumps({**cache_common, "method": "mccr", "theta_sim": args.theta_sim,
                                                            "tau_core": args.tau_core, "beta": args.beta,
                                                            "tau_var": args.tau_var, "rho_min": args.rho_min,
                                                            "gamma": args.gamma, "eta": args.eta}, sort_keys=True)
                                ) if "mccr" in args.methods else None
            def output_name(method: str) -> str:
                return f"paor_{args.ablation}" if method == "paor" and args.ablation != "none" else method

            per_seed_metrics = {output_name(method): [] for method in args.methods}
            for seed in args.seeds:
                async def execute_method(method: str) -> tuple[str, dict[str, Any]]:
                    method_name = output_name(method)
                    output_path = args.output_root / benchmark / method_name / f"seed{seed}.json"
                    deterministic_reuse = method in {"direct", "dynamic", "checkeval", "dna"} and seed != 0
                    reuse_path = args.output_root / benchmark / method_name / "seed0.json"
                    if deterministic_reuse and reuse_path.exists():
                        payload = json.loads(reuse_path.read_text(encoding="utf-8"))
                        results = payload["groups"]
                        save_json(output_path, {"benchmark": benchmark, "method": method_name, "seed": seed,
                                                "split": args.split, "reused_from_seed": 0, "groups": results})
                    elif args.skip_existing and output_path.exists():
                        payload = json.loads(output_path.read_text(encoding="utf-8"))
                        results = payload["groups"]
                    else:
                        LOG.info("Running %s / %s / seed=%d", benchmark, method_name, seed)
                        if method == "paor":
                            results = await paor.run(benchmark, selected, seed=seed, ablation=args.ablation)
                        elif method == "mccr":
                            results = await mccr.run(benchmark, selected, seed=seed)
                        else:
                            direct_budget = (1 + 3 * args.N) if "mccr" in args.methods else (2 + 2 * args.N)
                            results = await baseline.run(method, benchmark, selected, seed=seed,
                                                         direct_budget=direct_budget)
                        save_json(output_path, {"benchmark": benchmark, "method": method_name, "seed": seed,
                                                "split": args.split, "groups": results})
                        if method in {"paor", "mccr"}:
                            diagnostic_path = args.output_root / "diagnostics" / f"{benchmark}.{method_name}.seed{seed}.jsonl"
                            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                            with diagnostic_path.open("w", encoding="utf-8") as f:
                                for result in results:
                                    f.write(json.dumps({"qid": result["qid"], **result["diagnostic"]}, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")
                    metric = compute_metrics(results, bootstrap=args.bootstrap, seed=2026 + seed)
                    save_json(args.output_root / benchmark / method_name / f"seed{seed}.metrics.json", metric)
                    LOG.info("%s / %s / seed=%d: %s", benchmark, method_name, seed,
                             {key: value for key, value in metric.items() if key in ["accuracy", "spearman", "pearson", "within_query_kendall", "fallback_rate", "n_samples", "avg_calls_per_sample"]})
                    return method_name, metric

                outcomes = await asyncio.gather(*(execute_method(method) for method in args.methods), return_exceptions=True)
                failures = []
                for method, outcome in zip(args.methods, outcomes):
                    if isinstance(outcome, Exception):
                        LOG.error("Concurrent method failed: %s / %s / seed=%d: %s", benchmark, method, seed, outcome,
                                  exc_info=(type(outcome), outcome, outcome.__traceback__))
                        failures.append((method, outcome))
                    else:
                        method_name, metric = outcome
                        per_seed_metrics[method_name].append(metric)
                if failures:
                    raise RuntimeError(f"Concurrent methods failed for {benchmark} seed{seed}: {[x[0] for x in failures]}")
            for method in args.methods:
                method_name = output_name(method)
                summary = aggregate_seed_metrics(per_seed_metrics[method_name])
                save_json(args.output_root / benchmark / method_name / "summary.json", summary)
            primary_method = "mccr" if "mccr" in args.methods else "paor" if "paor" in args.methods else None
            if primary_method:
                method_dir = primary_method if primary_method == "mccr" or args.ablation == "none" else f"paor_{args.ablation}"
                primary_path = args.output_root / benchmark / method_dir / "seed0.json"
            else:
                primary_path = Path("__missing__")
            if primary_path.exists():
                primary_groups = json.loads(primary_path.read_text(encoding="utf-8"))["groups"]
                candidates = []
                for method in args.methods:
                    if method in {"paor", "mccr"}:
                        continue
                    path = args.output_root / benchmark / method / "seed0.metrics.json"
                    if path.exists():
                        metric = json.loads(path.read_text(encoding="utf-8"))
                        primary = metric.get("accuracy") if metric.get("kind") == "pairwise" else metric.get("spearman")
                        if primary is not None:
                            candidates.append((primary, method))
                if candidates:
                    best_method = max(candidates)[1]
                    baseline_groups = json.loads((args.output_root / benchmark / best_method / "seed0.json").read_text(encoding="utf-8"))["groups"]
                    significance = (mcnemar(primary_groups, baseline_groups) if primary_groups[0]["kind"] == "pairwise"
                                    else paired_bootstrap_difference(primary_groups, baseline_groups, n_boot=args.bootstrap))
                    save_json(args.output_root / benchmark / f"{primary_method}_vs_strongest.json",
                              {"method": primary_method, "baseline": best_method, "significance": significance})
            if "mccr" in args.methods and "paor" in args.methods:
                mccr_path = args.output_root / benchmark / "mccr" / "seed0.json"
                paor_path = args.output_root / benchmark / ("paor" if args.ablation == "none" else f"paor_{args.ablation}") / "seed0.json"
                if mccr_path.exists() and paor_path.exists():
                    mccr_groups = json.loads(mccr_path.read_text(encoding="utf-8"))["groups"]
                    paor_groups = json.loads(paor_path.read_text(encoding="utf-8"))["groups"]
                    save_json(args.output_root / benchmark / "mccr_vs_paor.json",
                              {"method": "mccr", "baseline": "paor",
                               "significance": mcnemar(mccr_groups, paor_groups)})
    finally:
        await pool.close()
    manifest["finished_unix"] = time.time()
    save_json(args.output_root / "manifest.json", manifest)
    LOG.info("Experiment complete. Results: %s", args.output_root)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    args = parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        LOG.exception("Experiment failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
