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
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from async_vllm import AsyncLLMPool, check_endpoints
from baselines import BaselineRunner
from mccr_pipeline import MCCRPipeline
from metrics import (aggregate_seed_metrics, compute_metrics, holm_correction, mcnemar, pair_metadata,
                     paired_accuracy_bootstrap, paired_bootstrap_difference)
from paor_data import BENCHMARKS, drop_human_ties, load_records, make_groups, select_groups, split_qids
from paor_pipeline import PAORPipeline, ProbeStore, save_json
from rubric_baselines import PROBE_RUBRIC_METHODS, ProbeRubricRunner

LOG = logging.getLogger("paor-main")
DEFAULT_METHODS = ["direct", "checkeval", "dna", "rubrichub", "mccr", "direct_k"]
RUBRIC_METHODS = [*PROBE_RUBRIC_METHODS, "rrd_uniform"]
METHOD_CHOICES = [*DEFAULT_METHODS, "mccr_v2", "paor", "dynamic", "direct_point", "sc_principles", *RUBRIC_METHODS]
# Methods that read probes (always through the shared ProbeStore so they see identical probes).
PROBE_METHODS = {"paor", "mccr", "mccr_v2", "rubric_resp", "rrd", "rrd_uniform", "crg", "rtd", "direct_ref"}
GENERATOR_METHODS = PROBE_METHODS | {"dynamic", "checkeval", "dna", "rubrichub", "rubric_query"}
# expert_rubric needs neither generator nor probes: it reads the benchmark's human rubric.
PRIMARY_ORDER = ["mccr_v2", "mccr", "paor"]


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
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test",
                        help="'all' evaluates every query; use it for benchmarks never used for tuning")
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
    # Protocol (fairness) settings shared by every rubric-scoring method.
    parser.add_argument("--score-mode", choices=["discrete", "logprob"], default="discrete",
                        help="criterion scores for ALL rubric methods; baselines are discrete, so keep discrete for main tables")
    parser.add_argument("--missing-policy", choices=["fail", "neutral"], default="fail",
                        help="'fail' counts a target response with unparsable criteria as a parse error (like every baseline)")
    parser.add_argument("--probe-recovery", choices=["full", "simple"], default="simple",
                        help="'simple': same model, one compact retry, drop exhausted slots; 'full': legacy 7-step role-switching chain")
    parser.add_argument("--min-probes", type=int, default=2)
    parser.add_argument("--no-share-probes", action="store_true", help="let each method sample its own probes (not recommended)")
    parser.add_argument("--role-params", nargs="*", default=[],
                        help="billions of parameters per role for the FLOPs proxy, e.g. strong=9 generator=27 weak=7")
    # MCCR v2 options (also usable to ablate v1 components).
    parser.add_argument("--mccr-selection", choices=["residual", "contrastive", "none"])
    parser.add_argument("--mccr-core-weighting", choices=["support", "uniform", "whiten"])
    parser.add_argument("--mccr-aggregation", choices=["gated", "lexicographic", "linear"])
    parser.add_argument("--mccr-contrast", choices=["none", "roles", "counterfactual", "both"])
    parser.add_argument("--mccr-filter-core", choices=["on", "off"])
    parser.add_argument("--n-counterfactual", type=int, default=3)
    parser.add_argument("--tau-align", type=float, default=0.05)
    parser.add_argument("--max-collinearity", type=float, default=0.9)
    parser.add_argument("--lex-c", type=float, default=0.8)
    parser.add_argument("--method-suffix", default="", help="appended to MCCR output names for ablation runs")
    # Baseline options.
    parser.add_argument("--sc-k", type=int, default=8, help="samples for sc_principles (SPCT-style voting@k)")
    parser.add_argument("--direct-k", type=int, default=None, help="explicit k for direct_k (output name direct_k<k>)")
    parser.add_argument("--rrd-rounds", type=int, default=2)
    parser.add_argument("--rrd-satisfied-threshold", type=int, default=2)
    parser.add_argument("--rrd-redundancy", type=float, default=0.9)
    parser.add_argument("--rtd-rounds", type=int, default=2)
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
    multi_probe = {"paor", "rubric_resp", "rrd", "rrd_uniform"} & set(args.methods)
    if multi_probe and args.generator_max_model_len < args.query_max_tokens + args.N * args.probe_max_tokens + 2048:
        raise ValueError(f"generator-max-model-len is too small for prompts holding all probes ({sorted(multi_probe)})")
    role_params = {}
    for item in args.role_params:
        role, _, value = item.partition("=")
        role_params[role.strip()] = float(value)
    if args.judge_max_model_len < args.query_max_tokens + 2 * args.response_max_tokens + 4096:
        raise ValueError("judge-max-model-len is too small for pairwise and atomic judging")
    if PROBE_METHODS & set(args.methods) and args.weak_max_model_len < args.query_max_tokens + args.probe_max_tokens + 512:
        raise ValueError("weak-max-model-len is too small for complete probe generation")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    if args.backend == "offline":
        from offline_vllm import OfflineLLMPool

        needs_generator = bool(GENERATOR_METHODS & set(args.methods))
        needs_weak = bool(PROBE_METHODS & set(args.methods))
        if needs_generator and args.generator_model_path is None:
            raise ValueError("--generator-model-path is required for generation-based methods in offline mode")
        if needs_weak and args.weak_model_path is None:
            raise ValueError("--weak-model-path is required for probe-based methods in offline mode")
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
    needs_embeddings = bool({"paor", "mccr", "mccr_v2", "rrd", "rrd_uniform"} & set(args.methods))
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
            rubric_methods = {"mccr", "mccr_v2", "paor", *RUBRIC_METHODS} & set(args.methods)
            if (args.score_mode == "logprob" and selected and selected[0]["kind"] == "pointwise" and rubric_methods
                    and set(args.methods) - rubric_methods):
                raise ValueError("logprob scores are only comparable when every method uses them; use --score-mode discrete")
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
            protocol = {"score_mode": args.score_mode, "missing_policy": args.missing_policy,
                        "probe_recovery": args.probe_recovery, "min_probes": args.min_probes}
            manifest["protocol"] = {**protocol, "shared_probes": not args.no_share_probes, "role_params": role_params}
            probe_kwargs = dict(N=args.N, probe_max_tokens=args.probe_max_tokens, query_max_tokens=args.query_max_tokens,
                                response_max_tokens=args.response_max_tokens, probe_max_attempts=args.probe_max_attempts,
                                tokenizer=tokenizer, embedding_fn=embedding_fn, **protocol)
            probe_signature = json.dumps({"version": "shared-probes-v1", "generator": args.generator_model or str(args.generator_model_path),
                                          "weak": args.weak_model, "N": args.N, "probe_max_tokens": args.probe_max_tokens,
                                          "query_max_tokens": args.query_max_tokens, "probe_recovery": args.probe_recovery,
                                          "probe_max_attempts": args.probe_max_attempts}, sort_keys=True)
            store = (ProbeStore(pool, args.cache_root, cache_signature=probe_signature, **probe_kwargs)
                     if PROBE_METHODS & set(args.methods) and not args.no_share_probes else None)
            cache_common = {"version": "mccr-v7-fair-protocol", **protocol, "shared_probes": store is not None, "generator": args.generator_model or str(args.generator_model_path),
                            "judge": args.strong_model, "weak": args.weak_model, "embedding": args.embedding_model,
                            "N": args.N, "L": args.L, "probe_max_tokens": args.probe_max_tokens,
                            "query_max_tokens": args.query_max_tokens, "response_max_tokens": args.response_max_tokens}
            paor = PAORPipeline(pool, args.cache_root, embedding_fn=embedding_fn, tokenizer=tokenizer,
                                N=args.N, L=args.L, c=args.paor_c, tau_d=args.tau_d, tau_rho=args.tau_rho, lam=args.lam,
                                probe_max_tokens=args.probe_max_tokens, query_max_tokens=args.query_max_tokens,
                                response_max_tokens=args.response_max_tokens, probe_max_attempts=args.probe_max_attempts,
                                probe_store=store, **protocol,
                                cache_signature=json.dumps({**cache_common, "method": "paor", "c": args.paor_c,
                                                            "tau_d": args.tau_d, "tau_rho": args.tau_rho, "lam": args.lam}, sort_keys=True)
                                ) if "paor" in args.methods else None
            mccr_options = {"selection": args.mccr_selection, "core_weighting": args.mccr_core_weighting,
                            "aggregation": args.mccr_aggregation, "contrast": args.mccr_contrast,
                            "filter_core": None if args.mccr_filter_core is None else args.mccr_filter_core == "on",
                            "n_counterfactual": args.n_counterfactual, "tau_align": args.tau_align,
                            "max_collinearity": args.max_collinearity, "lex_c": args.lex_c}

            def make_mccr(variant: str) -> MCCRPipeline:
                name = ("mccr" if variant == "v1" else "mccr_v2") + args.method_suffix
                options = mccr_options if variant == "v2" or args.method_suffix else {
                    key: value for key, value in mccr_options.items() if key in {"n_counterfactual", "tau_align", "max_collinearity", "lex_c"}}
                return MCCRPipeline(pool, args.cache_root, embedding_fn=embedding_fn, tokenizer=tokenizer,
                                    N=args.N, L=args.L, probe_max_tokens=args.probe_max_tokens,
                                    query_max_tokens=args.query_max_tokens, response_max_tokens=args.response_max_tokens,
                                    probe_max_attempts=args.probe_max_attempts, theta_sim=args.theta_sim,
                                    tau_core=args.tau_core, beta=args.beta, tau_var=args.tau_var,
                                    rho_min=args.rho_min, gamma=args.gamma, eta=args.eta, variant=variant,
                                    method_name=name, probe_store=store, **protocol, **options,
                                    cache_signature=json.dumps({**cache_common, "method": name, "variant": variant,
                                                                "theta_sim": args.theta_sim, "tau_core": args.tau_core,
                                                                "beta": args.beta, "tau_var": args.tau_var,
                                                                "rho_min": args.rho_min, "gamma": args.gamma,
                                                                "eta": args.eta, **options}, sort_keys=True))

            mccr = make_mccr("v1") if "mccr" in args.methods else None
            mccr_v2 = make_mccr("v2") if "mccr_v2" in args.methods else None
            rubric_runners = {
                method: ProbeRubricRunner(pool, args.cache_root, method="rrd" if method == "rrd_uniform" else method,
                                          rrd_weighting="uniform" if method == "rrd_uniform" else "whiten",
                                          rrd_rounds=args.rrd_rounds, rrd_satisfied_threshold=args.rrd_satisfied_threshold,
                                          rrd_redundancy=args.rrd_redundancy, rtd_rounds=args.rtd_rounds,
                                          lex_c=args.lex_c, probe_store=store, **probe_kwargs,
                                          cache_signature=json.dumps({**cache_common, "method": method,
                                                                      "rrd": [args.rrd_rounds, args.rrd_satisfied_threshold, args.rrd_redundancy],
                                                                      "rtd_rounds": args.rtd_rounds, "lex_c": args.lex_c}, sort_keys=True))
                for method in RUBRIC_METHODS if method in args.methods}
            meta = pair_metadata(selected)

            def output_name(method: str) -> str:
                if method == "paor" and args.ablation != "none":
                    return f"paor_{args.ablation}"
                if method in {"mccr", "mccr_v2"}:
                    return method + args.method_suffix
                if method == "direct_k" and args.direct_k:
                    return f"direct_k{args.direct_k}"
                if method == "sc_principles":
                    return f"sc_principles_k{args.sc_k}"
                return method

            per_seed_metrics = {output_name(method): [] for method in args.methods}
            for seed in args.seeds:
                if store is not None:
                    # Generate the shared probes once, before methods run concurrently.
                    LOG.info("Sampling shared probes for %s / seed=%d", benchmark, seed)
                    await store.ensure(benchmark, selected, seed)

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
                        elif method == "mccr_v2":
                            results = await mccr_v2.run(benchmark, selected, seed=seed)
                        elif method in rubric_runners:
                            results = await rubric_runners[method].run(benchmark, selected, seed=seed)
                        else:
                            direct_budget = (1 + 3 * args.N) if {"mccr", "mccr_v2"} & set(args.methods) else (2 + 2 * args.N)
                            results = await baseline.run(method, benchmark, selected, seed=seed,
                                                         direct_budget=direct_budget, direct_k=args.direct_k,
                                                         sc_k=args.sc_k)
                        save_json(output_path, {"benchmark": benchmark, "method": method_name, "seed": seed,
                                                "split": args.split, "groups": results})
                        if method in {"paor", "mccr", "mccr_v2"} or method in rubric_runners:
                            diagnostic_path = args.output_root / "diagnostics" / f"{benchmark}.{method_name}.seed{seed}.jsonl"
                            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                            with diagnostic_path.open("w", encoding="utf-8") as f:
                                for result in results:
                                    f.write(json.dumps({"qid": result["qid"], **result["diagnostic"]}, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")
                    metric = compute_metrics(results, bootstrap=args.bootstrap, seed=2026 + seed, meta=meta,
                                             role_params=role_params)
                    save_json(args.output_root / benchmark / method_name / f"seed{seed}.metrics.json", metric)
                    LOG.info("%s / %s / seed=%d: %s", benchmark, method_name, seed,
                             {key: value for key, value in metric.items() if key in ["accuracy", "accuracy_lenient", "query_all_correct", "spearman", "pearson", "within_query_kendall", "fallback_rate", "n_samples", "avg_calls_per_sample", "param_tokens_per_sample"]})
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
            names = [output_name(method) for method in args.methods]
            primary = next((output_name(m) for m in PRIMARY_ORDER if m in args.methods), None)
            if primary:
                comparisons = significance_against_all(args.output_root / benchmark, primary, names, args.bootstrap)
                if comparisons:
                    save_json(args.output_root / benchmark / f"{primary}_vs_all.json", comparisons)
                    strongest = comparisons.get("strongest_baseline")
                    if strongest:
                        save_json(args.output_root / benchmark / f"{primary}_vs_strongest.json",
                                  {"method": primary, "baseline": strongest,
                                   "significance": comparisons["comparisons"][strongest]})
            for first, second in [("mccr_v2", "mccr"), ("mccr", "paor"), ("mccr_v2", "rrd")]:
                a = args.output_root / benchmark / output_name(first) / "seed0.json"
                b = args.output_root / benchmark / output_name(second) / "seed0.json"
                if first in args.methods and second in args.methods and a.exists() and b.exists():
                    a_groups = json.loads(a.read_text(encoding="utf-8"))["groups"]
                    b_groups = json.loads(b.read_text(encoding="utf-8"))["groups"]
                    if a_groups and a_groups[0]["kind"] == "pairwise":
                        save_json(args.output_root / benchmark / f"{output_name(first)}_vs_{output_name(second)}.json",
                                  {"method": output_name(first), "baseline": output_name(second),
                                   "significance": mcnemar(a_groups, b_groups),
                                   "paired_bootstrap": paired_accuracy_bootstrap(a_groups, b_groups, n_boot=args.bootstrap)})
    finally:
        await pool.close()
    manifest["finished_unix"] = time.time()
    save_json(args.output_root / "manifest.json", manifest)
    LOG.info("Experiment complete. Results: %s", args.output_root)


def significance_against_all(benchmark_dir: Path, primary: str, names: list[str], n_boot: int) -> dict[str, Any]:
    """Seed-0 paired tests of the primary method against every other method, Holm-corrected."""
    primary_path = benchmark_dir / primary / "seed0.json"
    if not primary_path.exists():
        return {}
    primary_groups = json.loads(primary_path.read_text(encoding="utf-8"))["groups"]
    if not primary_groups:
        return {}
    pairwise = primary_groups[0]["kind"] == "pairwise"
    comparisons, accuracies = {}, {}
    for name in names:
        path = benchmark_dir / name / "seed0.json"
        if name == primary or not path.exists():
            continue
        groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
        if pairwise:
            comparisons[name] = {**mcnemar(primary_groups, groups),
                                 "paired_bootstrap": paired_accuracy_bootstrap(primary_groups, groups, n_boot=n_boot)}
            accuracies[name] = compute_metrics(groups).get("accuracy") or 0.0
        else:
            comparisons[name] = paired_bootstrap_difference(primary_groups, groups, n_boot=n_boot)
    if pairwise and comparisons:
        adjusted = holm_correction({name: value["p_value"] for name, value in comparisons.items()})
        for name, value in adjusted.items():
            comparisons[name]["p_value_holm"] = value
    strongest = max(accuracies, key=accuracies.get) if accuracies else None
    return {"method": primary, "comparisons": comparisons, "strongest_baseline": strongest,
            "correction": "holm" if pairwise else None}


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
