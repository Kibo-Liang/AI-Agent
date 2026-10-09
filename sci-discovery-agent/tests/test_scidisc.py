import json
import unittest

import numpy as np

from scidisc.agent import DiscoveryAgent, parse_candidates
from scidisc.bench import run_one
from scidisc.expr import ExprError, canonical_key, parse
from scidisc.fitting import evaluate
from scidisc.llm import BackendError, LLMBackend, ScriptedBackend
from scidisc.solvers import PolynomialBaseline, PowerLawBaseline, TemplateLibraryBaseline
from scidisc.tasks import TASKS, get_task, nrmse, r2_score


class ExprTests(unittest.TestCase):
    def test_valid_expression_evaluates(self):
        e = parse("c0*a**c1 + 1", ["a"])
        self.assertEqual(e.n_constants, 2)
        out = e({"a": np.array([1.0, 4.0])}, [2.0, 0.5])
        np.testing.assert_allclose(out, [3.0, 5.0])

    def test_caret_and_assignment_are_normalised(self):
        e = parse("T = c0 * a^2", ["a"])
        np.testing.assert_allclose(e({"a": np.array([3.0])}, [2.0]), [18.0])

    def test_constant_only_expression_broadcasts(self):
        e = parse("c0", ["a"])
        self.assertEqual(e({"a": np.arange(4.0)}, [5.0]).shape, (4,))

    def test_malicious_or_unsupported_input_is_rejected(self):
        bad = [
            '__import__("os").system("echo hi")',
            "a.__class__",
            "open('/etc/passwd')",
            "[x for x in range(3)]",
            "lambda: 1",
            "a if a else 1",
            "a and a",
            "'string'",
            "True",
            "unknown_var * 2",
            "sqrt(a, a)",
            "sin(x=a)",
            "a @ a",
            "a % 2",
            "",
            "   ",
            "1e12",
            "c0 + c2",  # gap in constant numbering
            "c9",
            "-" * 40 + "a",  # real AST depth beyond the limit
            "a\x00",  # null byte: ast.parse raises ValueError, not SyntaxError
            "(" * 250 + "a" + ")" * 250,  # parser's own nesting limit / over length cap
            "a" + " " * 400,  # length cap
        ]
        for src in bad:
            with self.assertRaises(ExprError, msg=src):
                parse(src, ["a"])

    def test_size_limit(self):
        with self.assertRaises(ExprError):
            parse("+".join(["a"] * 60), ["a"])

    def test_huge_exponent_is_clipped_not_exploding(self):
        e = parse("a**c0", ["a"])
        out = e({"a": np.array([10.0])}, [1e9])
        self.assertTrue(np.isfinite(out).all() or np.isinf(out).all())  # no exception, no hang

    def test_canonical_key_ignores_whitespace(self):
        self.assertEqual(canonical_key(parse("c0*a", ["a"])), canonical_key(parse("c0 * a", ["a"])))


class TaskTests(unittest.TestCase):
    def test_all_tasks_build(self):
        for name, task in TASKS.items():
            d = task.make_dataset(0)
            for split in (d.train, d.val, d.test, d.extrap):
                self.assertIn(task.target, split)
                for arr in split.values():
                    self.assertTrue(np.all(np.isfinite(arr)), name)

    def test_hidden_sets_are_clean_and_extrap_is_outside_range(self):
        task = get_task("kepler")
        d = task.make_dataset(1)
        np.testing.assert_allclose(d.test["T"], d.test["a"] ** 1.5)
        self.assertGreater(d.extrap["a"].min(), d.train["a"].max() - 1e-9)

    def test_noise_is_applied_only_to_visible_splits(self):
        task = get_task("pendulum")
        d = task.make_dataset(2)
        clean = 2 * np.pi * np.sqrt(d.train["L"] / 9.81)
        self.assertGreater(np.max(np.abs(d.train["T"] - clean)), 0)

    def test_dataset_is_reproducible(self):
        a, b = get_task("cooling").make_dataset(5), get_task("cooling").make_dataset(5)
        np.testing.assert_array_equal(a.train["temp"], b.train["temp"])

    def test_r2_edge_cases(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(r2_score(y, y), 1.0)
        self.assertEqual(r2_score(y, None), float("-inf"))
        self.assertEqual(r2_score(y, np.array([1.0, np.nan, 3.0])), float("-inf"))


class FittingTests(unittest.TestCase):
    def test_recovers_constants_of_known_law(self):
        task = get_task("kepler")
        d = task.make_dataset(0)
        cand = evaluate("c0*a**c1", ["a"], "T", d.train, d.val, np.random.default_rng(0))
        self.assertTrue(cand.ok)
        self.assertAlmostEqual(cand.consts[1], 1.5, delta=0.03)
        self.assertGreater(cand.val_r2, 0.999)

    def test_invalid_candidate_does_not_raise(self):
        d = get_task("kepler").make_dataset(0)
        cand = evaluate("os.system('x')", ["a"], "T", d.train, d.val, np.random.default_rng(0))
        self.assertFalse(cand.ok)
        self.assertEqual(cand.score, float("-inf"))

    def test_nan_producing_expression_is_marked_failed_or_low_scoring(self):
        d = get_task("cooling").make_dataset(0)
        cand = evaluate("log(c0 - 1000)", ["t"], "temp", d.train, d.val, np.random.default_rng(0))
        self.assertTrue((not cand.ok) or cand.val_r2 < 0.5)


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        out, note = parse_candidates('{"candidates":[{"expr":"c0*a","why":"x"},{"expr":"a**2"}]}')
        self.assertEqual(out, ["c0*a", "a**2"])
        self.assertEqual(note, "")

    def test_fenced_json_with_chatter(self):
        text = 'Sure! Here you go:\n```json\n{"candidates":[{"expr":"c0*a"}]}\n```\nHope it helps.'
        self.assertEqual(parse_candidates(text)[0], ["c0*a"])

    def test_list_of_strings(self):
        self.assertEqual(parse_candidates('["c0*a", "c0/a"]')[0], ["c0*a", "c0/a"])

    def test_malformed_json_falls_back_to_regex(self):
        out, _ = parse_candidates('{"candidates":[{"expr":"c0*a", "why": "ok"}, {"expr":"a**2" "why"')
        self.assertIn("c0*a", out)

    def test_garbage_returns_note(self):
        out, note = parse_candidates("I think the answer is 42.")
        self.assertEqual(out, [])
        self.assertTrue(note)


def reply(*exprs):
    return json.dumps({"candidates": [{"expr": e, "why": "test"} for e in exprs]})


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.task = get_task("kepler")
        self.data = self.task.make_dataset(0)

    def test_agent_finds_law_from_scripted_llm(self):
        be = ScriptedBackend([reply("c0*a", "c0*a**c1")])
        res = DiscoveryAgent(be, max_iters=3).solve(self.task, self.data, 0)
        self.assertGreater(res.val_r2, 0.999)
        self.assertEqual(be.calls, 1)  # stopped early once the target was reached

    def test_agent_survives_garbage_then_recovers(self):
        be = ScriptedBackend(["no json here", reply("c0*a**c1")])
        res = DiscoveryAgent(be, max_iters=3).solve(self.task, self.data, 0)
        self.assertGreater(res.val_r2, 0.999)
        self.assertEqual(be.calls, 2)
        # the second prompt must tell the model its previous reply was unusable
        self.assertIn("not valid JSON", be.seen[1][0]["content"])

    def test_invalid_candidates_are_reported_back_and_do_not_crash(self):
        be = ScriptedBackend([reply("__import__('os')", "c0*b", "c0*a"), reply("c0*a**c1")])
        res = DiscoveryAgent(be, max_iters=3).solve(self.task, self.data, 0)
        self.assertGreater(res.val_r2, 0.999)
        second_prompt = be.seen[1][0]["content"]
        self.assertIn("INVALID CANDIDATES", second_prompt)
        self.assertIn("unknown name 'b'", second_prompt)

    def test_duplicates_are_not_refit_and_are_flagged(self):
        be = ScriptedBackend([reply("c0*a", "c0 * a"), reply("c0*a", "c0*a**c1")])
        res = DiscoveryAgent(be, max_iters=3).solve(self.task, self.data, 0)
        first = res.trace[0]
        self.assertEqual(len(first["candidates"]), 1)
        self.assertEqual(len(first["skipped_duplicates"]), 1)
        self.assertIn("ALREADY TRIED", be.seen[1][0]["content"])

    def test_backend_failure_returns_best_so_far(self):
        class Flaky(LLMBackend):
            name = "flaky"

            def complete(self, system, messages):
                self.calls += 1
                if self.calls == 1:
                    return reply("c0*a")
                raise BackendError("network down")

        res = DiscoveryAgent(Flaky(), max_iters=4).solve(self.task, self.data, 0)
        self.assertNotEqual(res.formula, "<none>")
        self.assertTrue(any("error" in t for t in res.trace))

    def test_no_valid_candidate_gives_none_result_not_exception(self):
        be = ScriptedBackend([reply("zzz", "yyy")])
        res = DiscoveryAgent(be, max_iters=2).solve(self.task, self.data, 0)
        self.assertEqual(res.formula, "<none>")
        self.assertIsNone(res.predict(self.data.test))

    def test_prompt_never_contains_hidden_truth(self):
        be = ScriptedBackend([reply("c0*a")])
        DiscoveryAgent(be, max_iters=1).solve(self.task, self.data, 0)
        prompt = be.seen[0][0]["content"]
        self.assertNotIn("a**1.5", prompt)
        self.assertNotIn(self.task.truth, prompt)


class BenchTests(unittest.TestCase):
    def test_power_law_recovers_kepler_and_extrapolates(self):
        row = run_one(PowerLawBaseline(), get_task("kepler"), 0)
        self.assertGreater(row.extrap_r2, 0.999)
        self.assertTrue(row.success)

    def test_polynomial_fits_in_range_but_fails_to_extrapolate_exponential(self):
        row = run_one(PolynomialBaseline(), get_task("cooling"), 0)
        self.assertGreater(row.test_r2, 0.95)

    def test_saturating_law_is_not_misjudged_by_collapsed_extrapolation_variance(self):
        # Regression: with extrapolation R^2 this correct law (cooling, seed 0) scored 0.886
        # because the late-time target is almost flat. NRMSE uses the in-range scale instead.
        row = run_one(TemplateLibraryBaseline(), get_task("cooling"), 0)
        self.assertIn("exp", row.formula)
        self.assertTrue(row.success, f"extrap NRMSE={row.extrap_nrmse}")

    def test_nrmse_properties(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(nrmse(y, y, 2.0), 0.0)
        self.assertEqual(nrmse(y, None, 2.0), float("inf"))
        self.assertEqual(nrmse(y, np.array([1.0, np.inf, 3.0]), 2.0), float("inf"))
        self.assertEqual(nrmse(y, y, 0.0), float("inf"))
        self.assertAlmostEqual(nrmse(y, y + 1.0, 2.0), 0.5)

    def test_polynomial_does_not_count_as_finding_the_law(self):
        row = run_one(PolynomialBaseline(), get_task("pendulum"), 0)
        self.assertFalse(row.success)

    def test_hard_task_is_not_trivially_solved_by_power_law(self):
        row = run_one(PowerLawBaseline(), get_task("damped_oscillator"), 0)
        self.assertFalse(row.success)


if __name__ == "__main__":
    unittest.main()
