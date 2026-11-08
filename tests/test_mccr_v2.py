"""Tests for the fair protocol, MCCR v2, shared probes, new baselines, metrics and loaders."""
import asyncio
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from async_vllm import LLMResult  # noqa: E402
from baselines import BaselineRunner  # noqa: E402
from mccr_core import (contrastive_select, directional_alignment, hierarchical_aggregate,  # noqa: E402
                       ledoit_wolf_covariance, whitened_weights)
from mccr_pipeline import MCCRPipeline  # noqa: E402
from metrics import compute_metrics, holm_correction, paired_accuracy_bootstrap, pair_metadata  # noqa: E402
from offline_vllm import OfflineLLMPool  # noqa: E402
from paor_data import group_from_pairs_for_mock, load_records, make_groups  # noqa: E402
from paor_pipeline import PAORPipeline, ProbeStore, decide_pair  # noqa: E402
from rubric_baselines import ProbeRubricRunner  # noqa: E402


def embed(texts):
    """Deterministic fake embedding: identical texts map to identical unit vectors."""
    vectors = []
    for text in texts:
        seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
        vector = np.random.default_rng(seed).normal(size=16)
        vectors.append(vector / np.linalg.norm(vector))
    return np.asarray(vectors)


class ScriptedPool:
    """Answers each prompt type deterministically.

    Generator probes contain "GOOD", weak probes contain "BAD"; counterfactuals contain "BAD".
    The judge gives score 2 to every criterion containing "quality" when the response contains
    "GOOD", and 0 otherwise; criteria containing "verbose" prefer BAD responses (misaligned).
    """
    request = staticmethod(OfflineLLMPool.request)

    def __init__(self):
        self.requests = []
        self.stats = {}

    def _answer(self, request):
        user, system, role = request["user"], request["system"], request["role"]
        if "Return ONLY one JSON object" in user or "structured data" in system:
            return '{"answer": "x", "explanation": "y", "confidence": "high"}'
        if "controlled evaluation test cases" in system:
            return "BAD degraded copy"
        if "### Criteria:" in user:
            ids = re.findall(r"^(\w+): ", user.split("### Criteria:")[1], flags=re.M)
            response = user.split("### Response:")[1].split("### Criteria:")[0]
            good = "GOOD" in response
            out = {}
            criteria_block = user.split("### Criteria:")[1]
            for cid in ids:
                line = re.search(rf"^{cid}: (.*)$", criteria_block, flags=re.M).group(1)
                if "verbose" in line:
                    score = 0 if good else 2
                elif "always" in line:
                    score = 2
                else:
                    score = 2 if good else 0
                out[cid] = {"reason": "r", "score": score}
            return json.dumps(out)
        if "[Response A]" in user:
            a = user.split("[Response A]\n", 1)[1].split("\n\n[Response B]", 1)[0]
            return '{"winner":"A"}' if "GOOD" in a else '{"winner":"B"}'
        if '"prior_rubrics"' in user:
            return '{"prior_rubrics": [{"id": "P1", "criterion": "answer quality is high"}]}'
        if '"response_rubrics"' in user:
            return ('{"response_rubrics": [{"id": "R1", "criterion": "answer quality is high"}, '
                    '{"id": "R2", "criterion": "verbose style"}, {"id": "R3", "criterion": "detail quality"}]}')
        if '"hard_rules"' in user:
            return ('{"hard_rules": [{"id": "H1", "criterion": "answer quality is high"}], '
                    '"principles": [{"id": "Q1", "criterion": "detail quality"}]}')
        if '"rubrics"' in user:
            return ('{"rubrics": [{"id": "C1", "criterion": "answer quality is high"}, '
                    '{"id": "C2", "criterion": "verbose style"}, {"id": "C3", "criterion": "always satisfied"}]}')
        if role == "generator":
            return "GOOD probe"
        if role == "weak":
            return "BAD probe"
        return "probe"

    async def complete_many(self, requests):
        self.requests.extend(requests)
        out = []
        for request in requests:
            tag = request.get("tag", "global")
            entry = self.stats.setdefault(tag, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0})
            entry["calls"] += 1
            entry["prompt_tokens"] += 10
            entry["completion_tokens"] += 5
            key = f"tokens:{request['role']}"
            entry[key] = entry.get(key, 0) + 15
            out.append(LLMResult(self._answer(request), [], 10, 5, 0.0, "fake", finish_reason="stop"))
        return out

    def tag_snapshot(self, tag):
        return dict(self.stats.get(tag, {}))


def pair_group():
    return group_from_pairs_for_mock("rewardbench2:Math:1", "question", [("GOOD target", "BAD target", "A")])


class CorePrimitiveTest(unittest.TestCase):
    def test_directional_alignment_sign(self):
        F = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        alignment = directional_alignment(F, [(0, 1)])
        np.testing.assert_allclose(alignment, [1.0, -1.0])
        np.testing.assert_allclose(directional_alignment(F, []), [0.0, 0.0])

    def test_ledoit_wolf_is_positive_definite_for_tiny_samples(self):
        X = np.random.default_rng(0).integers(0, 3, size=(4, 8)) / 2
        cov, shrink = ledoit_wolf_covariance(X)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)
        self.assertGreaterEqual(shrink, 0.0)
        self.assertLessEqual(shrink, 1.0)

    def test_whitening_downweights_duplicated_criteria(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=40)
        b = rng.normal(size=40)
        F = np.column_stack([a, a + 1e-3 * rng.normal(size=40), b])
        weights = whitened_weights(F)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[2], weights[0])
        self.assertGreater(weights[2], weights[1])

    def test_contrastive_select_skips_misaligned_and_collinear(self):
        core = np.asarray([[1.0], [0.0], [1.0], [0.0]])
        ext = np.asarray([[1.0, 0.0, 1.0, 1.0],
                          [0.0, 1.0, 0.0, 1.0],
                          [1.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]])
        alignment = np.asarray([1.0, -1.0, 1.0, 0.5])
        result = contrastive_select(core, ext, alignment, max_L=4, tau_align=0.05, max_collinearity=0.9)
        self.assertNotIn(1, result["selected"])  # misaligned
        self.assertIn(0, result["skipped_collinear"])  # identical to the core profile
        self.assertEqual(result["selected"], [3])

    def test_lexicographic_band_never_overturns_core(self):
        weights = np.asarray([0.5, 0.3, 0.2])
        core = np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 0.5]])  # second has the smallest possible core defect
        ext = np.asarray([[0.0], [1.0]])
        result = hierarchical_aggregate(core, ext, weights, np.ones(1), mode="lexicographic", c=0.8)
        self.assertGreater(result["U"][0], result["U"][1])
        tied = hierarchical_aggregate(core[[0, 0]], ext, weights, np.ones(1), mode="lexicographic")
        self.assertGreater(tied["U"][1], tied["U"][0])


class ProtocolTest(unittest.TestCase):
    def test_missing_score_is_a_parse_error_not_a_neutral_score(self):
        pair = {"id": "p", "a": "r0", "b": "r1", "gold": "A"}
        self.assertTrue(decide_pair(pair, {"r0": None, "r1": 0.3})["parse_error"])
        tie = decide_pair(pair, {"r0": 0.5, "r1": 0.5})
        self.assertTrue(tie["fallback"])
        self.assertEqual(decide_pair(pair, {"r0": 0.6, "r1": 0.5})["prediction"], "A")

    def test_discrete_score_mode_ignores_logprobs_and_resolves_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PAORPipeline(ScriptedPool(), tmp, embedding_fn=embed, N=1)
            text = '{"C1": {"reason": "r", "score": 1}}'
            entries = [{"token": '{"C1": {"reason": "r", "score": ', "logprob": 0, "top_logprobs": []},
                       {"token": "1", "logprob": -0.1, "top_logprobs": [{"token": "2", "logprob": -0.2}, {"token": "1", "logprob": -0.1}]}]
            values, missing = pipeline._score_values(LLMResult(text, entries, 1, 1, 0.0, "x"), ["C1", "C2"])
            self.assertEqual(values, [0.5, None])
            self.assertEqual(missing, 1)
            self.assertIsNone(pipeline._resolve_missing(values))
            neutral = PAORPipeline(ScriptedPool(), tmp, embedding_fn=embed, N=1, missing_policy="neutral")
            self.assertEqual(neutral._resolve_missing(values), [0.5, 0.5])
            soft = PAORPipeline(ScriptedPool(), tmp, embedding_fn=embed, N=1, score_mode="logprob")
            self.assertGreater(soft._score_values(LLMResult(text, entries, 1, 1, 0.0, "x"), ["C1"])[0][0], 0.5)

    def test_simple_recovery_drops_exhausted_probe_slot(self):
        class TruncatingPool(ScriptedPool):
            async def complete_many(self, requests):
                self.requests.extend(requests)
                return [LLMResult("x" if r["role"] == "weak" else "GOOD", [], 1, 4096, 0.0, "f",
                                  finish_reason="length" if r["role"] == "weak" else "stop") for r in requests]
        with tempfile.TemporaryDirectory() as tmp:
            pool = TruncatingPool()
            pipeline = PAORPipeline(pool, tmp, embedding_fn=embed, N=3, probe_recovery="simple", probe_max_attempts=2)
            cache = asyncio.run(pipeline._sample_probes("mock", [{"qid": "q", "query": "q"}], 0, [{}]))[0]
            self.assertEqual(cache["probe_responses"], ["GOOD", None, "GOOD"])
            self.assertEqual(cache["probe_finish_reasons"][1], "dropped")
            self.assertTrue(all(r["role"] in {"generator", "weak"} for r in pool.requests))
            self.assertFalse(any(r["structured_json"] for r in pool.requests))
            strict = PAORPipeline(TruncatingPool(), tmp + "/b", embedding_fn=embed, N=2, probe_recovery="simple",
                                  probe_max_attempts=1, min_probes=2)
            with self.assertRaises(RuntimeError):
                asyncio.run(strict._sample_probes("mock", [{"qid": "q", "query": "q"}], 0, [{}]))

    def test_shared_probes_are_generated_once_and_charged_to_each_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = ScriptedPool()
            store = ProbeStore(pool, tmp, embedding_fn=embed, N=2, cache_signature="s")
            first = PAORPipeline(pool, tmp, embedding_fn=embed, N=2, probe_store=store)
            first.method_name = "m1"
            second = PAORPipeline(pool, tmp, embedding_fn=embed, N=2, probe_store=store)
            second.method_name = "m2"
            group = {"qid": "q", "query": "q"}
            a = asyncio.run(first._sample_probes("b", [group], 0, [{}]))[0]
            b = asyncio.run(second._sample_probes("b", [group], 0, [{}]))[0]
            self.assertEqual(a["probe_responses"], b["probe_responses"])
            self.assertEqual(sum(1 for r in pool.requests if r["role"] in {"generator", "weak"}), 2)
            self.assertEqual(a["_cost_offset"]["calls"], 2)
            self.assertEqual(b["_cost_offset"]["calls"], 2)


class MCCRV2Test(unittest.TestCase):
    def run_mccr(self, variant):
        with tempfile.TemporaryDirectory() as tmp:
            pool = ScriptedPool()
            store = ProbeStore(pool, tmp, embedding_fn=embed, N=4, cache_signature="s")
            pipeline = MCCRPipeline(pool, tmp, embedding_fn=embed, N=4, variant=variant, probe_store=store,
                                    n_counterfactual=2)
            asyncio.run(store.ensure("rewardbench2", [pair_group()], 0))
            result = asyncio.run(pipeline.run("rewardbench2", [pair_group()], seed=0))[0]
            return result, pool

    def test_v2_filters_misaligned_criteria_and_uses_counterfactuals(self):
        result, pool = self.run_mccr("v2")
        diagnostic = result["diagnostic"]
        self.assertEqual(diagnostic["n_counterfactual"], 2)
        self.assertEqual(len(diagnostic["score_rows"]), 6)  # 4 probes + 2 counterfactuals
        self.assertEqual(len(diagnostic["known_pairs"]), 2 * 2 + 2)
        kept = [x["criterion"] for x in diagnostic["R_core"] + diagnostic["R_ext"]]
        self.assertNotIn("verbose style", kept)
        self.assertEqual(result["pairs"][0]["prediction"], "A")
        self.assertGreater(result["cost"]["calls"], 0)
        self.assertIn("tokens:generator", result["cost"])

    def test_v1_runs_on_the_fixed_protocol(self):
        result, _ = self.run_mccr("v1")
        self.assertEqual(result["diagnostic"]["variant"], "v1")
        self.assertEqual(result["diagnostic"]["n_counterfactual"], 0)
        self.assertIn(result["pairs"][0]["prediction"], {"A", "B", None})


class RubricBaselineTest(unittest.TestCase):
    def run_method(self, method, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            pool = ScriptedPool()
            store = ProbeStore(pool, tmp, embedding_fn=embed, N=4, cache_signature="s")
            runner = ProbeRubricRunner(pool, tmp, method=method, embedding_fn=embed, N=4, probe_store=store, **kwargs)
            return asyncio.run(runner.run("rewardbench2", [pair_group()], seed=0))[0]

    def test_rrd_filters_misaligned_and_decomposes_oversatisfied(self):
        result = self.run_method("rrd", rrd_rounds=1)
        kept = [x["criterion"] for x in result["diagnostic"]["criteria"]["core"]]
        self.assertNotIn("verbose style", kept)
        reasons = {x["rejected"] for x in result["diagnostic"]["rrd_log"]}
        self.assertIn("misaligned", reasons)
        self.assertAlmostEqual(sum(result["diagnostic"]["weights"]["core"]), 1.0)
        self.assertEqual(result["pairs"][0]["prediction"], "A")

    def test_single_call_crg_rtd_and_reference_direct(self):
        for method in ["rubric_query", "rubric_resp", "crg", "rtd", "direct_ref"]:
            result = self.run_method(method)
            self.assertEqual(result["pairs"][0]["prediction"], "A", method)
        crg = self.run_method("crg")
        self.assertEqual([x["id"] for x in crg["diagnostic"]["criteria"]["core"]], ["H1"])

    def test_expert_rubric_uses_data_rubrics(self):
        group = pair_group()
        group["pairs"][0]["metadata"] = {"expert_rubrics": ["answer quality is high"]}
        with tempfile.TemporaryDirectory() as tmp:
            runner = ProbeRubricRunner(ScriptedPool(), tmp, method="expert_rubric", embedding_fn=embed, N=2)
            result = asyncio.run(runner.run("rubricbench", [group], seed=0))[0]
        self.assertEqual(result["diagnostic"]["criteria"]["core"][0]["criterion"], "answer quality is high")
        self.assertEqual(result["pairs"][0]["prediction"], "A")


class NewBaselineTest(unittest.TestCase):
    def test_sc_principles_votes_with_alternating_order(self):
        class SPCTPool(ScriptedPool):
            def _answer(self, request):
                first = request["user"].split("[Response 1]\n", 1)[1].split("\n\n[Response 2]", 1)[0]
                return "Scores: \\boxed{8, 3}" if "GOOD" in first else "Scores: \\boxed{3, 8}"
        runner = BaselineRunner(SPCTPool(), ".")
        result = asyncio.run(runner.run("sc_principles", "rewardbench2", [pair_group()], seed=0, sc_k=4))[0]
        self.assertEqual(result["pairs"][0]["prediction"], "A")
        self.assertEqual(result["pairs"][0]["n_parsed_samples"], 4)

    def test_direct_point_scores_independently(self):
        class PointPool(ScriptedPool):
            def _answer(self, request):
                return '{"score": 9}' if "GOOD" in request["user"] else '{"score": 2}'
        runner = BaselineRunner(PointPool(), ".")
        result = asyncio.run(runner.run("direct_point", "rewardbench2", [pair_group()], seed=0))[0]
        self.assertEqual(result["pairs"][0]["prediction"], "A")


class MetricsV2Test(unittest.TestCase):
    def test_lenient_best_of_n_subset_and_flops(self):
        groups = [{"qid": "q1", "kind": "pairwise", "cost": {"tokens:strong": 100, "tokens:generator": 10}, "pairs": [
                       {"id": "a", "gold": "A", "prediction": "A", "order_votes": [1.0, 1.0]},
                       {"id": "b", "gold": "A", "prediction": None, "order_disagreement": True, "order_votes": [1.0, 0.0]}]},
                  {"qid": "q2", "kind": "pairwise", "cost": {}, "pairs": [
                       {"id": "c", "gold": "B", "prediction": "B"}, {"id": "d", "gold": "A", "prediction": "A"}]}]
        meta = {"a": {"subset": "x"}, "b": {"subset": "x"}, "c": {"subset": "y"}, "d": {"subset": "y"}}
        metric = compute_metrics(groups, meta=meta, role_params={"strong": 9, "generator": 27})
        self.assertAlmostEqual(metric["accuracy"], 0.75)
        self.assertAlmostEqual(metric["accuracy_lenient"], 0.875)
        self.assertAlmostEqual(metric["query_all_correct"], 0.5)
        self.assertAlmostEqual(metric["position_consistency"], 0.5)
        self.assertEqual(metric["accuracy_by_subset"], {"x": 0.5, "y": 1.0})
        self.assertAlmostEqual(metric["param_tokens_per_sample"], (9 * 100 + 27 * 10) / 4)

    def test_rmbench_style_matrix(self):
        rows, meta = [], {}
        for i in range(3):
            for j in range(3):
                pid = f"p{i}{j}"
                rows.append({"id": pid, "gold": "A", "prediction": "A" if i >= j else "B"})
                meta[pid] = {"style_chosen": i, "style_rejected": j, "subset": "chat"}
        metric = compute_metrics([{"qid": "q", "kind": "pairwise", "pairs": rows}], meta=meta)
        self.assertAlmostEqual(metric["rmbench"]["hard"], 0.0)
        self.assertAlmostEqual(metric["rmbench"]["normal"], 1.0)
        self.assertAlmostEqual(metric["rmbench"]["easy"], 1.0)

    def test_holm_and_paired_bootstrap(self):
        adjusted = holm_correction({"a": 0.01, "b": 0.04})
        self.assertAlmostEqual(adjusted["a"], 0.02)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        better = [{"qid": f"q{i}", "pairs": [{"id": f"p{i}", "gold": "A", "prediction": "A"}]} for i in range(30)]
        worse = [{"qid": f"q{i}", "pairs": [{"id": f"p{i}", "gold": "A", "prediction": "B" if i % 2 else "A"}]} for i in range(30)]
        result = paired_accuracy_bootstrap(better, worse, n_boot=200)
        self.assertAlmostEqual(result["difference"], 0.5)
        self.assertGreater(result["ci95"][0], 0.0)


class NewDataTest(unittest.TestCase):
    DATA = ROOT / "generating_dynamic_rubric" / "data"

    def test_new_benchmark_shapes(self):
        expected = {"rubricbench": (1147, 1147), "llmbar": (419, 419), "ppe": (10211, 10211)}
        for benchmark, (n_records, n_groups) in expected.items():
            records = load_records(benchmark, self.DATA)
            self.assertEqual(len(records), n_records, benchmark)
            self.assertEqual(len(make_groups(records, benchmark)), n_groups, benchmark)
        rubricbench = load_records("rubricbench", self.DATA)
        self.assertTrue(all(r["expert_rubrics"] for r in rubricbench))
        self.assertEqual({r["subset"] for r in rubricbench}, {"chat", "if", "stem", "code", "safety"})

    def test_rmbench_keeps_style_matrix(self):
        records = load_records("rmbench", self.DATA)
        groups = make_groups(records, "rmbench")
        self.assertEqual(len(groups), 1327)
        meta = pair_metadata(groups)
        self.assertEqual({(m["style_chosen"], m["style_rejected"]) for m in meta.values()},
                         {(i, j) for i in range(3) for j in range(3)})
        self.assertTrue(all(len(g["responses"]) <= 6 for g in groups))


if __name__ == "__main__":
    unittest.main()
