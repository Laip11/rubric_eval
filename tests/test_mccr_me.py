"""Tests for MCCR-ME (minimal evidence set)."""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from mccr_core import classify_clusters, deletion_minimize, greedy_hitting_set, span_in_query  # noqa: E402
from mccr_me import MinimalEvidencePipeline  # noqa: E402
from paor_core import parse_rubrics  # noqa: E402
from paor_pipeline import ProbeStore  # noqa: E402
from test_mccr_v2 import ScriptedPool, embed, pair_group  # noqa: E402


class MEPool(ScriptedPool):
    """Scripted answers for the minimal-evidence prompts.

    Witnesses for criteria mentioning "optional" or "detail" stay GOOD (violating them does not make the
    response worse → not necessary); "duplicate" cannot be violated alone (IMPOSSIBLE → implied);
    everything else yields a BAD witness (→ necessary). The adversary is GOOD but "sneaky": it satisfies
    every criterion yet loses pairwise comparisons, so one missing requirement is discovered.
    """

    def _answer(self, request):
        user, system = request["user"], request["system"]
        if "Grounding —" in user:
            return ('{"prior_rubrics": ['
                    '{"id": "P1", "criterion": "answer quality is high", "type": "explicit", "evidence": "question"},'
                    '{"id": "P2", "criterion": "optional flourish quality", "type": "implicit", "evidence": ""},'
                    '{"id": "P3", "criterion": "duplicate answer quality", "type": "explicit", "evidence": "not in query"}]}')
        if '"response_rubrics"' in user:
            return ('{"response_rubrics": [{"id": "R1", "criterion": "answer quality is high"}, '
                    '{"id": "R2", "criterion": "detail quality"}, {"id": "R3", "criterion": "core correctness quality"}]}')
        if "### Criterion to violate:" in user:
            target = user.split("### Criterion to violate:")[1].split("###")[0]
            if "optional" in target or "detail" in target:
                return "GOOD witness still fine"
            if "duplicate" in target:
                return "IMPOSSIBLE: E1"
            return "BAD witness"
        if '"redundant"' in user:
            return '{"redundant": []}'
        if '"necessary"' in user:
            ids = [line.split(":")[0] for line in user.split("### Criteria list:")[1].strip().splitlines() if ":" in line]
            return "{" + '"necessary": {' + ", ".join(f'"{i}": true' for i in ids if not i.startswith("Output")) + "}}"
        if "stress-test" in system:
            return "GOOD sneaky adversary"
        if '"missing_requirements"' in user:
            return '{"missing_requirements": [{"id": "M1", "criterion": "no hidden flaw quality"}]}'
        if "[Response A]" in user and "### Criteria:" not in user:
            a = user.split("[Response A]\n", 1)[1].split("\n\n[Response B]", 1)[0]
            b = user.split("[Response B]\n", 1)[1].split("\n\nWhich response", 1)[0]
            if "sneaky" in a or "sneaky" in b:
                return '{"winner":"B"}' if "sneaky" in a else '{"winner":"A"}'
        return super()._answer(request)


def run(necessity="witness", sufficiency=True, max_tests=12, pool_cls=MEPool):
    with tempfile.TemporaryDirectory() as tmp:
        pool = pool_cls()
        store = ProbeStore(pool, tmp, embedding_fn=embed, N=4, cache_signature="s")
        pipeline = MinimalEvidencePipeline(pool, tmp, embedding_fn=embed, N=4, probe_store=store,
                                           necessity=necessity, sufficiency=sufficiency, max_tests=max_tests)
        result = asyncio.run(pipeline.run("rewardbench2", [pair_group()], seed=0))[0]
        return result, pool


class PrimitiveTest(unittest.TestCase):
    def test_parse_rubrics_keeps_grounding_fields(self):
        text = '{"prior_rubrics": [{"id": "P1", "criterion": "c", "type": "Explicit", "evidence": "in English"}]}'
        self.assertEqual(parse_rubrics(text, "prior_rubrics", 1, 5),
                         [{"id": "P1", "criterion": "c", "type": "explicit", "evidence": "in English"}])

    def test_span_in_query(self):
        query = "Write a haiku about autumn, in English."
        self.assertTrue(span_in_query("in   english", query))
        self.assertTrue(span_in_query('"a haiku about autumn"', query))
        self.assertFalse(span_in_query("in French", query))
        self.assertFalse(span_in_query("", query))

    def test_classify_clusters(self):
        U = [{"criterion": "stated", "view": 0, "type": "explicit", "evidence": "haiku"},
             {"criterion": "stated again", "view": 1},
             {"criterion": "implicit both", "view": 0, "type": "implicit"},
             {"criterion": "implicit both", "view": 2},
             {"criterion": "query only", "view": 0, "type": "explicit", "evidence": "not there"},
             {"criterion": "response only", "view": 2}]
        clusters = [{"members": [0, 1], "medoid": 1}, {"members": [2, 3], "medoid": 3},
                    {"members": [4], "medoid": 4}, {"members": [5], "medoid": 5}]
        classes = classify_clusters(clusters, U, "Write a haiku.")
        self.assertEqual([x["class"] for x in classes], ["explicit", "both", "query_implicit", "response_only"])
        self.assertEqual(classes[0]["criterion"], "stated")  # query-grounded wording preferred
        self.assertEqual(classes[1]["criterion"], "implicit both")

    def test_greedy_hitting_set_is_minimal_and_drops_misaligned(self):
        # 3 pairs; criterion 0 explains pairs 0,1; criterion 1 explains pair 2; criterion 2 duplicates 0;
        # criterion 3 prefers the worse response.
        F = np.asarray([[1, 0, 1, 0], [0, 0, 0, 1],    # pair 0: rows (0, 1)
                        [1, 0, 1, 0], [0, 0, 0, 1],    # pair 1: rows (2, 3)
                        [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)  # pair 2: rows (4, 5)
        selected, info = greedy_hitting_set(F, [(0, 1), (2, 3), (4, 5)])
        self.assertEqual(sorted(selected), [0, 1])
        self.assertIn(3, info["misaligned"])
        self.assertEqual(info["unexplained_pairs"], 0)

    def test_deletion_minimize_respects_priority_and_cycles(self):
        items = [{"id": "A", "criterion": "a", "class": "explicit"},
                 {"id": "B", "criterion": "b", "class": "response_only"},
                 {"id": "C", "criterion": "c", "class": "both"}]
        kept, removed = deletion_minimize(items, {"A": ["B"], "B": ["A"], "C": ["Z"]})
        self.assertEqual([x["id"] for x in kept], ["A", "C"])  # lower-priority B removed; A survives the cycle
        self.assertEqual([x["id"] for x in removed], ["B"])


class PipelineTest(unittest.TestCase):
    def test_witness_mode_builds_minimal_sufficient_set(self):
        result, _ = run()
        d = result["diagnostic"]
        classes = {x["criterion"]: x["class"] for x in d["candidates"]}
        self.assertEqual(classes["answer quality is high"], "explicit")
        self.assertEqual(classes["optional flourish quality"], "query_implicit")
        self.assertEqual(classes["duplicate answer quality"], "query_implicit")  # evidence not in query
        self.assertEqual(classes["core correctness quality"], "response_only")
        kept = {x["criterion"] for x in d["E"]}
        self.assertIn("answer quality is high", kept)          # explicit: no test needed
        self.assertIn("core correctness quality", kept)        # response-only, witness is worse
        self.assertNotIn("optional flourish quality", kept)    # witness not worse → not necessary
        self.assertNotIn("detail quality", kept)
        self.assertNotIn("duplicate answer quality", kept)     # cannot be violated alone → implied
        reasons = {x["criterion"]: d["tests"][x["id"]]["reason"] for x in d["candidates"] if x["id"] in d["tests"]}
        self.assertEqual(reasons["duplicate answer quality"], "implied")
        self.assertTrue(d["sufficiency"]["counterexample"])
        self.assertIn("no hidden flaw quality", kept)          # added by the counterexample, passed necessity
        self.assertEqual(result["pairs"][0]["prediction"], "A")

    def test_sufficiency_off_makes_no_adversary_calls(self):
        result, pool = run(sufficiency=False)
        self.assertFalse(any("stress-test" in r["system"] for r in pool.requests))
        self.assertFalse(result["diagnostic"]["sufficiency"]["enabled"])

    def test_alternative_necessity_estimators_run(self):
        for mode in ["judgment", "hitting_set", "none"]:
            result, pool = run(necessity=mode)
            self.assertEqual(result["diagnostic"]["necessity_mode"], mode)
            self.assertTrue(result["diagnostic"]["E"], mode)
            self.assertEqual(result["pairs"][0]["prediction"], "A", mode)
            witness_calls = sum("### Criterion to violate:" in r["user"] for r in pool.requests)
            if mode != "witness":
                self.assertEqual(witness_calls, 0, mode)
        result, _ = run(necessity="none")
        self.assertIn("optional flourish quality", {x["criterion"] for x in result["diagnostic"]["E"]})

    def test_test_budget_drops_lowest_priority_candidates(self):
        result, _ = run(max_tests=1, sufficiency=False)
        d = result["diagnostic"]
        self.assertEqual(len(d["tests"]), 1)
        self.assertTrue(d["budget_dropped"])
        tested_class = next(x["class"] for x in d["candidates"] if x["id"] in d["tests"])
        self.assertEqual(tested_class, "query_implicit")  # query-grounded candidates are tested first


if __name__ == "__main__":
    unittest.main()
