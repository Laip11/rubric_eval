import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paor_core import (aggregate, expected_score, pairwise_decide, parse_atomic_scores, parse_rubrics, select_posterior,
                       softmax_expected_from_entries, unit_defect_band)
from mccr_core import gated_aggregate, orthogonal_residual_select, semantic_consensus_clusters
from mccr_coverage_ablation import build_variant_cache
from paor_data import drop_human_ties, group_from_pairs_for_mock, load_records, make_groups
from paor_pipeline import PAORPipeline
from metrics import compute_metrics, mcnemar
from baselines import BaselineRunner
from async_vllm import LLMResult
from offline_vllm import OfflineLLMPool, _serialize_output


class PAORCoreTest(unittest.TestCase):
    def test_zero_discrimination_is_not_selected(self):
        V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        E = np.eye(2)
        selected, _ = select_posterior(V, E, np.empty((0, 2)), max_L=2, tau_d=0.15)
        self.assertEqual(selected, [0])

    def test_duplicate_is_rejected(self):
        V = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        E = np.array([[1.0, 0.0], [1.0, 0.0]])
        selected, _ = select_posterior(V, E, np.empty((0, 2)), max_L=2, tau_d=0.0, tau_rho=0.85)
        self.assertEqual(selected, [0])

    def test_unit_defect_band(self):
        prior_a = np.array([[1.0, 1.0, 1.0, 1.0]])
        prior_b = np.array([[1.0, 1.0, 1.0, 0.5]])
        post_a = np.array([[0.0]])
        post_b = np.array([[1.0]])
        ua, *_ = aggregate(prior_a, post_a, c=0.8)
        ub, *_ = aggregate(prior_b, post_b, c=0.8)
        self.assertGreater(ua[0], ub[0])
        eta, delta = unit_defect_band(4)
        self.assertLess(eta, delta)

    def test_expected_score(self):
        self.assertAlmostEqual(expected_score({0: 0.0, 1: 0.0, 2: 0.0}), 0.5)
        self.assertAlmostEqual(expected_score(None, 2), 1.0)

    def test_logprob_extraction(self):
        text = '{"P1": {"reason": "ok", "score": 2}}'
        entries = [
            {"token": '{"P1": {"reason": "ok", "score": ', "logprob": -0.1, "top_logprobs": []},
            {"token": "2", "logprob": -0.1, "top_logprobs": [{"token": "0", "logprob": -5}, {"token": "1", "logprob": -4}, {"token": "2", "logprob": -0.1}]},
            {"token": "}}", "logprob": -0.1, "top_logprobs": []},
        ]
        values, logprobs = softmax_expected_from_entries(text, entries, ["P1"], {"P1": 2})
        self.assertIn("P1", logprobs)
        self.assertGreater(values["P1"], 0.9)

    def test_score_id_aliases_and_multiple_objects(self):
        text = '{"P1": {"reason": "ok", "score": 2}}\n{"P2": {"reason": "partial", "score": 1}}'
        scores, missing = parse_atomic_scores(text, ["E1", "E2", "E3"])
        self.assertEqual(scores, {"E1": 2, "E2": 1})
        self.assertEqual(missing, 1)

    def test_pairwise_decision(self):
        self.assertEqual(pairwise_decide(0.8, 0.7), "A")
        self.assertEqual(pairwise_decide(0.7, 0.8), "B")
        self.assertEqual(pairwise_decide(0.8, 0.8), "fallback")

    def test_single_rubric_and_case_insensitive_criterion(self):
        text = '{"prior_rubrics":[{"id":"P1","Criterion":"Identifies the correct architectural style."}]}'
        values = parse_rubrics(text, "prior_rubrics", 1, 5)
        self.assertEqual(values, [{"id": "P1", "criterion": "Identifies the correct architectural style."}])

    def test_recovers_only_complete_rubrics_from_malformed_json(self):
        malformed = '{"prior_rubrics":[{"id":"P1","criterion":"first"},{{"id":"P2","criterion":"second"}]}'
        truncated = '{"id":"P1","criterion":"first"},{"id":"P2","criterion":"second"},{"id":"P3","criterion":"unfinished'
        self.assertEqual([x["criterion"] for x in parse_rubrics(malformed, "prior_rubrics", 1, 5)], ["first", "second"])
        self.assertEqual([x["criterion"] for x in parse_rubrics(truncated, "prior_rubrics", 1, 5)], ["first", "second"])


class OfflineBackendTest(unittest.TestCase):
    def test_output_order_across_roles(self):
        pool = object.__new__(OfflineLLMPool)
        def fake_call(role, requests):
            return [LLMResult(request["user"], [], 1, 1, 0.0, f"offline:{role}") for request in requests]
        pool._call_sync = fake_call
        requests = [OfflineLLMPool.request("s", "a", role="strong"),
                    OfflineLLMPool.request("s", "b", role="weak"),
                    OfflineLLMPool.request("s", "c", role="generator")]
        results = asyncio.run(pool.complete_many(requests))
        self.assertEqual([result.text for result in results], ["a", "b", "c"])

    def test_replica_batches_split_and_preserve_order(self):
        pool = object.__new__(OfflineLLMPool)
        pool._workers = {"strong": [{"replica": 0}, {"replica": 1}]}
        seen = []
        def fake_call(role, worker, values):
            seen.append((worker["replica"], [request["user"] for _, request in values]))
            return [(index, LLMResult(request["user"], [], 1, 1, 0.0, "fake")) for index, request in values]
        pool._call_worker = fake_call
        requests = [OfflineLLMPool.request("s", text, role="strong") for text in ["a", "b", "c", "d"]]
        results = pool._call_sync("strong", requests)
        self.assertEqual([result.text for result in results], ["a", "b", "c", "d"])
        self.assertCountEqual(seen, [(0, ["a", "c"]), (1, ["b", "d"])])

    def test_native_logprob_conversion(self):
        sampled = SimpleNamespace(decoded_token="2", logprob=-0.1)
        alternative = SimpleNamespace(decoded_token="1", logprob=-3.0)
        completion = SimpleNamespace(text="2", token_ids=[7], logprobs=[{7: sampled, 8: alternative}])
        output = SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2, 3])
        result = _serialize_output(output)
        self.assertEqual(result["text"], "2")
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["logprobs"][0]["token"], "2")
        self.assertEqual(len(result["logprobs"][0]["top_logprobs"]), 2)
        completion.logprobs = None
        self.assertEqual(_serialize_output(output)["logprobs"], [])


class MCCRCoreTest(unittest.TestCase):
    def test_consensus_support_and_medoid(self):
        embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.99, 0.01], [0.0, 1.0]])
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        clusters = semantic_consensus_clusters(embeddings, np.asarray([0, 1, 2, 3]), theta_sim=0.8, beta=0.4)
        self.assertEqual(len(clusters), 2)
        self.assertAlmostEqual(clusters[0]["rho"], 0.8)
        self.assertIn(clusters[0]["medoid"], [0, 1, 2])
        self.assertAlmostEqual(clusters[1]["rho"], 0.2)

    def test_orthogonal_selection_rejects_collinear_criterion(self):
        core = np.asarray([[0.0], [0.0], [1.0], [1.0]])
        ext = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        result = orthogonal_residual_select(core, ext, np.asarray([0.5, 0.5]), max_L=1, tau_var=0.04)
        self.assertEqual(result["selected"], [1])
        self.assertEqual(result["core_rank"], 1)
        self.assertAlmostEqual(result["sigmas"][0], 0.5)

    def test_core_gate_blocks_low_coverage_response(self):
        result = gated_aggregate(np.asarray([[1.0], [0.5]]), np.asarray([[0.0], [1.0]]),
                                 np.asarray([1.0]), np.asarray([1.0]), gamma=2.0, eta=0.5)
        self.assertAlmostEqual(result["U"][0], 1.0)
        self.assertAlmostEqual(result["U"][1], 0.625)

    def test_coverage_variant_reuses_probe_scores_and_reselects_criteria(self):
        criteria = [{"id": f"R{i + 1}", "criterion": f"c{i}", "rho": rho}
                    for i, rho in enumerate((0.6, 0.4, 0.2))]
        base = {"R_core": [criteria[0]], "R_ext_candidates": criteria[1:],
                "F_core": [[0], [0], [0], [1], [1], [1]],
                "F_ext": [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 1]],
                "target_scores": {"old": [1.0]}, "cost_checkpoint": {"calls": 3}}
        variant = build_variant_cache(base, tau_core=0.4, tau_var=0.0, L=1, rho_min=0.2, N=6)
        self.assertEqual([x["id"] for x in variant["R_core"]], ["R1", "R2"])
        self.assertEqual([x["id"] for x in variant["R_ext"]], ["R3"])
        self.assertEqual(len(variant["F_core"][0]), 2)
        self.assertNotIn("target_scores", variant)
        self.assertIn("target_scores", base)


class CompleteProbeTest(unittest.TestCase):
    def test_length_limited_probe_is_replaced(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            def __init__(self):
                self.round = 0
            async def complete_many(self, requests):
                self.round += 1
                if self.round == 1:
                    return [LLMResult("truncated", [], 10, 4096, 0.0, "fake", finish_reason="length")]
                return [LLMResult("complete", [], 10, 12, 0.0, "fake", finish_reason="stop")]
            def tag_snapshot(self, tag):
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PAORPipeline(FakePool(), tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1,
                                    probe_max_tokens=4096, probe_max_attempts=2)
            groups = [{"qid": "q", "query": "question", "responses": []}]
            caches = asyncio.run(pipeline._sample_probes("mock", groups, 0, [{}]))
            self.assertEqual(caches[0]["probe_responses"], ["complete"])
            self.assertEqual(caches[0]["probe_attempts"], [2])
            self.assertEqual(caches[0]["probe_truncated"], 1)

    def test_probe_roles_are_recorded_per_slot(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            def __init__(self):
                self.requests = []
            async def complete_many(self, requests):
                self.requests.extend(requests)
                return [LLMResult("complete", [], 10, 12, 0.0, "fake", finish_reason="stop") for _ in requests]
            def tag_snapshot(self, tag):
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            pipeline = PAORPipeline(pool, tmp, embedding_fn=lambda x: np.empty((0, 2)), N=2)
            groups = [{"qid": "q", "query": "question", "responses": []}]
            cache = asyncio.run(pipeline._sample_probes("mock", groups, 0, [{}]))[0]
            self.assertEqual([request["role"] for request in pool.requests], ["generator", "weak"])
            self.assertEqual(cache["probe_roles"], ["generator", "weak"])

    def test_third_attempt_enforces_compact_final_response(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            async def complete_many(self, requests):
                self.requests = requests
                return [LLMResult("complete", [], 10, 12, 0.0, "fake", finish_reason="stop")]
            def tag_snapshot(self, tag):
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            pipeline = PAORPipeline(pool, tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1,
                                    probe_max_tokens=4096, probe_max_attempts=4)
            groups = [{"qid": "q", "query": "count to 1000", "responses": []}]
            cache = {"probe_responses": [None], "probe_finish_reasons": ["length"],
                     "probe_attempts": [2], "probe_truncated": 2}
            asyncio.run(pipeline._sample_probes("mock", groups, 0, [cache]))
            self.assertEqual(pool.requests[0]["temperature"], 0.2)
            self.assertIn("ranges or ellipses", pool.requests[0]["user"])
            self.assertIn("final answer", pool.requests[0]["user"])

    def test_final_attempt_switches_probe_role(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            async def complete_many(self, requests):
                self.requests = requests
                return [LLMResult("complete", [], 10, 12, 0.0, "fake", finish_reason="stop")]
            def tag_snapshot(self, tag):
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            pipeline = PAORPipeline(pool, tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1,
                                    probe_max_tokens=4096, probe_max_attempts=4)
            groups = [{"qid": "q", "query": "question", "responses": []}]
            cache = {"probe_responses": [None], "probe_finish_reasons": ["length"], "probe_attempts": [3]}
            result = asyncio.run(pipeline._sample_probes("mock", groups, 0, [cache]))[0]
            self.assertEqual(pool.requests[0]["role"], "weak")
            self.assertEqual(result["probe_roles"], ["weak"])
            self.assertEqual(result["probe_attempts"], [4])

    def test_fifth_attempt_uses_judge_with_bounded_final_answer(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            async def complete_many(self, requests):
                self.requests = requests
                return [LLMResult("complete", [], 10, 12, 0.0, "fake", finish_reason="stop")]
            def tag_snapshot(self, tag):
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            pipeline = PAORPipeline(pool, tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1,
                                    probe_max_tokens=4096, probe_max_attempts=5)
            groups = [{"qid": "q", "query": "question", "responses": []}]
            cache = {"probe_responses": [None], "probe_finish_reasons": ["length"], "probe_attempts": [4]}
            result = asyncio.run(pipeline._sample_probes("mock", groups, 0, [cache]))[0]
            self.assertEqual(pool.requests[0]["role"], "strong")
            self.assertIn("uncertainty", pool.requests[0]["user"])
            self.assertIn("800 tokens", pool.requests[0]["user"])
            self.assertEqual(result["probe_roles"], ["strong"])

    def test_sixth_attempt_reframes_probe_as_structured_task(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            async def complete_many(self, requests):
                self.requests = requests
                return [LLMResult('{"answer":"ok","explanation":"brief","confidence":"low"}', [], 10, 12, 0.0, "fake", finish_reason="stop")]
            def tag_snapshot(self, tag):
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            pipeline = PAORPipeline(pool, tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1,
                                    probe_max_tokens=4096, probe_max_attempts=6)
            groups = [{"qid": "q", "query": "question", "responses": []}]
            cache = {"probe_responses": [None], "probe_finish_reasons": ["length"], "probe_attempts": [5]}
            result = asyncio.run(pipeline._sample_probes("mock", groups, 0, [cache]))[0]
            self.assertEqual(pool.requests[0]["role"], "strong")
            self.assertEqual(pool.requests[0]["max_tokens"], 1024)
            self.assertIn("[Conversation]", pool.requests[0]["user"])
            self.assertIn("at most 80 words", pool.requests[0]["user"])
            self.assertEqual(pool.requests[0]["structured_json"]["properties"]["answer"]["maxLength"], 200)
            self.assertEqual(result["probe_finish_reasons"], ["stop"])


class ExhaustedProbeTest(unittest.TestCase):
    def test_exhausted_probe_fails_explicitly(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            async def complete_many(self, requests):
                return [LLMResult("truncated", [], 10, 4096, 0.0, "fake", finish_reason="length")]
            def tag_snapshot(self, tag):
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PAORPipeline(FakePool(), tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1,
                                    probe_max_tokens=4096, probe_max_attempts=1)
            groups = [{"qid": "q", "query": "question", "responses": []}]
            with self.assertRaises(RuntimeError):
                asyncio.run(pipeline._sample_probes("mock", groups, 0, [{}]))

    def test_cached_exhaustion_does_not_retry_again(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            async def complete_many(self, requests):
                raise AssertionError("exhausted cache must not issue another request")
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PAORPipeline(FakePool(), tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1,
                                    probe_max_tokens=4096, probe_max_attempts=1)
            groups = [{"qid": "q", "query": "question", "responses": []}]
            cache = {"probe_responses": [None], "probe_finish_reasons": ["length"],
                     "probe_attempts": [1], "probe_truncated": 1}
            with self.assertRaises(RuntimeError):
                asyncio.run(pipeline._sample_probes("mock", groups, 0, [cache]))

    def test_partial_cached_attempts_cannot_exceed_limit(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            def __init__(self):
                self.rounds = 0
            async def complete_many(self, requests):
                self.rounds += 1
                return [LLMResult("truncated", [], 10, 4096, 0.0, "fake", finish_reason="length")]
            def tag_snapshot(self, tag):
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            pipeline = PAORPipeline(pool, tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1,
                                    probe_max_tokens=4096, probe_max_attempts=4)
            groups = [{"qid": "q", "query": "question", "responses": []}]
            cache = {"probe_responses": [None], "probe_finish_reasons": ["length"], "probe_attempts": [3]}
            with self.assertRaises(RuntimeError):
                asyncio.run(pipeline._sample_probes("mock", groups, 0, [cache]))
            self.assertEqual(pool.rounds, 1)
            self.assertEqual(cache["probe_attempts"], [4])

    def test_response_budget_cannot_truncate_complete_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                PAORPipeline(None, tmp, embedding_fn=lambda x: np.empty((0, 2)),
                             probe_max_tokens=4096, response_max_tokens=2048)


class RubricBatchRetryTest(unittest.TestCase):
    def test_two_deterministic_json_retries(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            def __init__(self):
                self.round = 0
            async def complete_many(self, requests):
                self.round += 1
                text = ('{"prior_rubrics":[{"id":"P1","criterion":"first requirement"},'
                        '{"id":"P2","criterion":"second requirement"}]}' if self.round == 3 else "invalid")
                return [LLMResult(text, [], 10, 5, 0.0, "fake") for _ in requests]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PAORPipeline(FakePool(), tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1)
            request = pipeline._request("system", "question", "tag", role="generator", max_tokens=512)
            values, failures = asyncio.run(pipeline._rubric_batch([request], "prior_rubrics", 2, 5, [], "prior"))
            self.assertEqual(failures, 0)
            self.assertEqual(len(values[0]), 2)
            self.assertEqual(pipeline.pool.round, 3)


class ResumableCostTest(unittest.TestCase):
    def test_cost_checkpoint_adds_only_new_process_delta(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            def __init__(self):
                self.stats = {"calls": 2.0, "prompt_tokens": 20.0, "completion_tokens": 10.0}
            def tag_snapshot(self, tag):
                return dict(self.stats)
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            pipeline = PAORPipeline(pool, tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1)
            pipeline.benchmark = "mock"
            pipeline._ablation = "none"
            group = {"qid": "q", "query": "question", "responses": []}
            cache = {"_cost_offset": {"calls": 3.0, "prompt_tokens": 30.0, "completion_tokens": 15.0}}
            pipeline._save("mock", group, 0, cache)
            self.assertEqual(cache["cost_checkpoint"]["calls"], 5.0)
            pool.stats = {"calls": 1.0, "prompt_tokens": 10.0, "completion_tokens": 5.0}
            restored = pipeline._cache("mock", group, 0)
            cost = pipeline._cost(restored, 0, "q", "none")
            self.assertEqual(cost["calls"], 6.0)
            self.assertEqual(cost["prompt_tokens"], 60.0)
            self.assertEqual(cost["completion_tokens"], 30.0)


class ReliableDataTest(unittest.TestCase):
    def test_human_ties_are_removed_after_split_without_touching_other_pairs(self):
        mixed = group_from_pairs_for_mock("mixed", "question", [("good", "bad", "A"), ("same1", "same2", "tie")])
        tie_only = group_from_pairs_for_mock("tie-only", "question", [("same1", "same2", "tie")])
        filtered = drop_human_ties([mixed, tie_only])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(filtered[0]["pairs"]), 1)
        self.assertEqual(filtered[0]["pairs"][0]["gold"], "A")
        self.assertEqual(len(filtered[0]["responses"]), 2)

    def test_new_dataset_counts_and_deduplication(self):
        root = Path(__file__).resolve().parents[1] / "generating_dynamic_rubric" / "data"
        reward = make_groups(load_records("rewardbench2", root), "rewardbench2")
        self.assertEqual(len(reward), 1763)
        self.assertEqual(sum(len(g["pairs"]) for g in reward), 5286)
        self.assertTrue(all(len(g["responses"]) == len(g["pairs"]) + 1 for g in reward))
        helpsteer = make_groups(load_records("helpsteer3", root), "helpsteer3")
        self.assertEqual(len(helpsteer), 2016)
        self.assertTrue(all(len(g["responses"]) == 2 for g in helpsteer))


class DirectPairParsingTest(unittest.TestCase):
    def test_pair_id_starting_with_r_uses_winner_parser(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            async def complete_many(self, requests):
                results = []
                for request in requests:
                    response_a = request["user"].split("[Response A]\n", 1)[1].split("\n\n[Response B]", 1)[0]
                    winner = "A" if response_a == "good" else "B"
                    results.append(LLMResult(f'{{"winner":"{winner}"}}', [], 10, 5, 0.0, "fake"))
                return results
            def tag_snapshot(self, tag):
                return {}
        group = group_from_pairs_for_mock("q", "question", [("good", "bad", "A")])
        group["pairs"][0]["id"] = "rewardbench2:Factuality:1:pair0"
        runner = BaselineRunner(FakePool(), ".")
        result = asyncio.run(runner._direct("rewardbench2", [group], 0))[0]
        self.assertEqual(result["pairs"][0]["prediction"], "A")
        self.assertEqual(result["pairs"][0]["order_votes"], [1.0, 1.0])
        self.assertFalse(result["pairs"][0]["parse_error"])


class HumanTieFallbackTest(unittest.TestCase):
    def test_model_tie_on_human_tie_runs_fallback_without_scoring_it(self):
        class FakePool:
            request = staticmethod(OfflineLLMPool.request)
            def __init__(self):
                self.requests = []
            async def complete_many(self, requests):
                self.requests.extend(requests)
                results = []
                for request in requests:
                    response_a = request["user"].split("[Response A]\n", 1)[1].split("\n\n[Response B]", 1)[0]
                    winner = "A" if response_a == "good" else "B"
                    results.append(LLMResult(f'{{"winner":"{winner}"}}', [], 10, 5, 0.0, "fake"))
                return results
            def tag_snapshot(self, tag):
                return {"calls": len(self.requests), "prompt_tokens": 10 * len(self.requests),
                        "completion_tokens": 5 * len(self.requests), "seconds": 0.0, "errors": 0}

        class CachedPAOR(PAORPipeline):
            async def _prepare(self, benchmark, groups, seed, ablation):
                return [{"target_scores:none": {"prior": {"r0": [1.0], "r1": [1.0]},
                                                  "post": {"r0": [], "r1": []}},
                         "selected_variants": {"none": []}}]
            async def _target_scores(self, benchmark, groups, caches, seed, ablation):
                pass

        group = group_from_pairs_for_mock("helpsteer3:00484", "question", [("good", "bad", "tie")])
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            pipeline = CachedPAOR(pool, tmp, embedding_fn=lambda x: np.empty((0, 2)), N=1)
            result = asyncio.run(pipeline.run("helpsteer3", [group], seed=0))
        pair = result[0]["pairs"][0]
        self.assertEqual(pair["prediction"], "A")
        self.assertEqual(pair["order_votes"], [1.0, 1.0])
        self.assertTrue(pair["fallback"])
        self.assertEqual(len(pool.requests), 2)
        metric = compute_metrics(result)
        self.assertEqual(metric["n_samples"], 0)
        self.assertEqual(metric["n_ties_excluded"], 1)
        self.assertIsNone(metric["accuracy"])
        self.assertEqual(metric["calls"], 2)


class CostNamespaceTest(unittest.TestCase):
    def test_benchmarks_do_not_share_cost_tags(self):
        runner = BaselineRunner(None, ".")
        runner.benchmark = "biggen"
        pairwise = runner._tag("direct", 0, "shared-qid")
        runner.benchmark = "biggen_pointwise"
        pointwise = runner._tag("direct", 0, "shared-qid")
        self.assertNotEqual(pairwise, pointwise)


class MetricsTest(unittest.TestCase):
    def test_pairwise_metrics(self):
        group = group_from_pairs_for_mock("q", "question", [("good", "bad", "A")])
        result = [{"qid": "q", "kind": "pairwise", "pairs": [{"id": "p", "gold": "A", "prediction": "A", "fallback": False}],
                   "items": [], "cost": {"calls": 2}, "wall_seconds": 1}]
        metric = compute_metrics(result)
        self.assertEqual(metric["accuracy"], 1.0)
        self.assertEqual(metric["avg_calls_per_sample"], 2)

    def test_order_disagreement_is_wrong_not_parse_failure(self):
        groups = [{"qid": "q", "kind": "pairwise", "pairs": [{"id": "p", "gold": "A", "prediction": None,
                                                             "order_disagreement": True, "parse_error": False}],
                   "items": [], "cost": {}, "wall_seconds": 0}]
        metric = compute_metrics(groups)
        self.assertEqual(metric["accuracy"], 0.0)
        self.assertEqual(metric["n_parse_errors"], 0)
        self.assertEqual(metric["n_order_disagreements"], 1)

    def test_unresolved_pairs_stay_in_the_fixed_accuracy_denominator(self):
        groups = [{"qid": "q", "kind": "pairwise", "pairs": [
            {"id": "p1", "gold": "A", "prediction": "A"},
            {"id": "p2", "gold": "B", "prediction": None, "order_disagreement": True, "parse_error": False},
            {"id": "p3", "gold": "A", "prediction": None, "order_disagreement": False, "parse_error": True}],
            "items": [], "cost": {}, "wall_seconds": 0}]
        metric = compute_metrics(groups)
        self.assertEqual(metric["n_samples"], 3)
        self.assertAlmostEqual(metric["accuracy"], 1 / 3)
        self.assertEqual(metric["n_order_disagreements"], 1)
        self.assertEqual(metric["n_parse_errors"], 1)
        self.assertNotIn("accuracy_valid", metric)

    def test_mcnemar_rejects_different_evaluation_samples(self):
        first = [{"qid": "q", "pairs": [{"id": "p1", "gold": "A", "prediction": "A"}]}]
        second = [{"qid": "q", "pairs": [{"id": "p2", "gold": "A", "prediction": "A"}]}]
        with self.assertRaises(ValueError):
            mcnemar(first, second)

    def test_mcnemar_excludes_human_ties(self):
        paor = [{"qid": "q", "pairs": [{"id": "p1", "gold": "A", "prediction": "A"},
                                          {"id": "p2", "gold": "tie", "prediction": "A"}]}]
        baseline = [{"qid": "q", "pairs": [{"id": "p1", "gold": "A", "prediction": "B"},
                                              {"id": "p2", "gold": "tie", "prediction": "B"}]}]
        result = mcnemar(paor, baseline)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["method_correct_baseline_wrong"], 1)


if __name__ == "__main__":
    unittest.main()
