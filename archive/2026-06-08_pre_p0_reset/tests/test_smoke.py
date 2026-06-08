import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from minileet.baselines import solve_with_templates
from minileet.analyze_failures import analyze_failures
from minileet.audit import audit_jsonl
from minileet.collect import collect_jsonl
from minileet.candidate_set_compact import (
    build_cache,
    load_model as load_compact_model,
    predict_rows as predict_compact_rows,
    train_from_cache,
)
from minileet.dsl import Assign, Binary, BoolLit, Compare, ForEach, Function, If, INT, Param, Return, Var, LIST_INT, IntLit, BOOL
from minileet.env import MiniLeetEnv, TestCaseResult
from minileet.interpreter import RunResult, run_function
from minileet.features import TOKEN_RE, extract_features, feature_names, fit_feature_spec, load_rows
from minileet.filter_dataset import filter_jsonl
from minileet.mine_hard_negatives import mine_hard_negatives
from minileet.rerank import (
    _save_model,
    evaluate_first_visible_pass,
    evaluate_predicted_selection,
    load_model,
    predict_rows,
    train_and_score,
)
from minileet.runs import (
    PROJECT_ROOT,
    create_timestamped_run_dir,
    ensure_new_run_dir,
    infer_dataset_slug,
    run_root,
    timestamped_run_dir,
)
from minileet.serialization import error_code_from_message, program_to_record, run_to_record, task_to_record, testcase_to_record
from minileet.sequence import CLS, PAD, SEP, TRUNC, UNK, TokenSpec, encode_row, encode_tokens, fit_token_spec, row_to_structured_tokens, row_to_tokens
from minileet.search import enumerate_candidates, oracle_scan_budget, solve_with_enumeration
from minileet.candidate_set_rerank import (
    SetConfig,
    _group_indices as group_candidate_indices,
    load_model as load_set_model,
    predict_rows as predict_set_rows,
    train_and_score as train_set_reranker,
)
from minileet.split_validation import fail_on_task_overlap, validate_no_task_overlap
from minileet.sweep_structured import _take_task_count, _take_whole_task_rows
from minileet.structured_rerank import (
    StructuredConfig,
    _encode_rows as encode_structured_rows,
    load_model as load_structured_model,
    predict_rows as predict_structured_rows,
    train_and_score as train_structured,
)
from minileet.tasks import default_tasks, suite_tasks, task_by_name, task_meta
from minileet.transformer_rerank import (
    TrainConfig,
    _save_model as save_transformer_model,
    load_model as load_transformer_model,
    predict_rows as predict_transformer_rows,
    train_and_score as train_transformer,
)
from minileet.typecheck import typecheck_function


def _feature_row(task_name: str, program: Function) -> dict[str, object]:
    task = task_by_name(task_name)
    env = MiniLeetEnv(task, hidden_count=4, seed=0)
    result = env.evaluate(program)
    return {
        "suite": "test",
        "task": task_to_record(task),
        "candidate_index": 0,
        "program": program_to_record(program),
        "visible": [testcase_to_record(case) for case in result.visible],
        "hidden": [testcase_to_record(case, include_trace=False) for case in result.hidden],
        "labels": {
            "visible_passes": result.visible_passes,
            "visible_total": len(result.visible),
            "visible_pass_rate": result.visible_pass_rate,
            "hidden_passes": result.hidden_passes,
            "hidden_total": len(result.hidden),
            "hidden_pass_rate": result.hidden_pass_rate,
            "solved": result.solved,
            "reward": result.reward,
        },
    }


def _feature_values(row: dict[str, object]) -> dict[str, float]:
    spec = fit_feature_spec([row], max_tokens=0)
    return dict(zip(feature_names(spec), extract_features(row, spec)))


class InterpreterTests(unittest.TestCase):
    def test_sum_program_runs(self) -> None:
        program = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (
                Assign("acc", IntLit(0)),
                ForEach("x", Var("xs"), (Assign("acc", Binary(Var("acc"), "+", Var("x"))),)),
                Return(Var("acc")),
            ),
        )
        result = run_function(program, ([1, 2, 3],))
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 6)
        self.assertGreater(len(result.trace), 0)


class EnvironmentTests(unittest.TestCase):
    def test_visible_and_hidden_tests_are_evaluated(self) -> None:
        task = task_by_name("sum_list")
        env = MiniLeetEnv(task, hidden_count=8, seed=1)
        attempt = solve_with_templates(task, hidden_count=8, seed=1)
        self.assertIsNotNone(attempt)
        result = env.evaluate(attempt.program)
        self.assertTrue(result.solved)

    def test_template_baseline_solves_some_tasks(self) -> None:
        solved = 0
        for task in default_tasks():
            attempt = solve_with_templates(task, hidden_count=16)
            solved += int(attempt is not None and attempt.result.solved)
        self.assertGreaterEqual(solved, 3)

    def test_generated_suites_have_held_out_variation(self) -> None:
        train = suite_tasks("train")
        eval_tasks = suite_tasks("eval")
        self.assertGreaterEqual(len(train), 20)
        self.assertGreaterEqual(len(eval_tasks), 5)
        self.assertTrue(set(train).isdisjoint(set(eval_tasks)))
        self.assertTrue(all(len(task.visible_examples) >= 3 for task in eval_tasks))

    def test_procedural_suites_are_unique(self) -> None:
        train = suite_tasks("procedural_train")
        eval_tasks = suite_tasks("procedural_eval")
        names = [task.name for task in train + eval_tasks]
        self.assertGreaterEqual(len(train), 500)
        self.assertGreaterEqual(len(eval_tasks), 100)
        self.assertEqual(len(names), len(set(names)))

    def test_procedural_hard_split_holds_out_families(self) -> None:
        train = suite_tasks("procedural_hard_train")
        eval_tasks = suite_tasks("procedural_hard_eval")
        self.assertGreaterEqual(len(train), 500)
        self.assertGreaterEqual(len(eval_tasks), 100)
        self.assertFalse(any(task_meta(task.name).family == "first_or_zero" for task in train))
        self.assertTrue(any(task_meta(task.name).family == "first_or_zero" for task in eval_tasks))
        self.assertTrue(any(task_meta(task.name).operator == "eq" for task in eval_tasks))

    def test_hard_dev_and_final_eval_slices_are_disjoint(self) -> None:
        dev = suite_tasks("procedural_hard_dev_eval")
        final = suite_tasks("procedural_hard_final_eval")
        self.assertEqual(len(dev), 100)
        self.assertEqual(len(final), 100)
        self.assertTrue({task.name for task in dev}.isdisjoint({task.name for task in final}))
        self.assertNotEqual([task.name for task in dev], [task.name for task in final])

    def test_procedural_bridge_train_has_partial_hard_compositions(self) -> None:
        bridge = suite_tasks("procedural_bridge_train")
        metas = [task_meta(task.name) for task in bridge]
        self.assertGreater(len(bridge), 100)
        self.assertTrue(any(meta.family == "first_or_zero" and meta.operator not in {"eq", "ne"} for meta in metas))
        self.assertTrue(any(meta.family != "first_or_zero" and meta.operator in {"eq", "ne"} for meta in metas))
        self.assertFalse(any(meta.profile == "wide" for meta in metas))

    def test_first_exact_bridge_train_has_new_first_or_zero_profiles(self) -> None:
        bridge = suite_tasks("procedural_first_exact_bridge_train")
        metas = [task_meta(task.name) for task in bridge]
        self.assertGreater(len(bridge), 100)
        self.assertTrue(all(meta.family == "first_or_zero" for meta in metas))
        self.assertTrue(any(meta.operator == "eq" for meta in metas))
        self.assertTrue(any(meta.operator == "ne" for meta in metas))
        self.assertTrue(all(meta.profile in {"bshort", "bmedium", "blong", "bzeros"} for meta in metas))

    def test_strict_bridge_train_has_no_hard_eval_name_overlap(self) -> None:
        bridge = suite_tasks("procedural_strict_bridge_train")
        hard_eval = suite_tasks("procedural_hard_eval")
        metas = [task_meta(task.name) for task in bridge]
        bridge_names = {task.name for task in bridge}
        hard_eval_names = {task.name for task in hard_eval}
        self.assertGreater(len(bridge), 400)
        self.assertTrue(bridge_names.isdisjoint(hard_eval_names))
        self.assertTrue(any(meta.family == "first_or_zero" and meta.operator == "eq" for meta in metas))
        self.assertTrue(any(meta.family == "first_or_zero" and meta.operator == "ne" for meta in metas))
        self.assertTrue(any(meta.family == "first_or_zero" and meta.operator == "gt" for meta in metas))
        self.assertTrue(any(meta.family != "first_or_zero" and meta.operator == "eq" for meta in metas))
        self.assertTrue(any(meta.family != "first_or_zero" and meta.operator == "ne" for meta in metas))
        self.assertTrue(all(meta.profile in {"bshort", "bmedium", "blong", "bzeros"} for meta in metas))

    def test_targeted_contrast_train_has_no_eval_name_overlap(self) -> None:
        contrast = suite_tasks("procedural_targeted_contrast_train")
        dev = suite_tasks("procedural_hard_dev_eval")
        final = suite_tasks("procedural_hard_final_eval")
        metas = [task_meta(task.name) for task in contrast]
        contrast_names = {task.name for task in contrast}
        eval_names = {task.name for task in dev + final}
        self.assertGreater(len(contrast), 300)
        self.assertTrue(contrast_names.isdisjoint(eval_names))
        self.assertTrue(any(meta.family == "first_or_zero" and meta.operator == "le" for meta in metas))
        self.assertTrue(any(meta.family == "first_or_zero" and meta.operator == "eq" for meta in metas))
        self.assertTrue(any(meta.family == "sum" and meta.operator == "eq" for meta in metas))
        self.assertTrue(any(meta.family == "all" and meta.operator == "ne" for meta in metas))
        self.assertTrue(all(meta.profile in {"cshort", "cmedium", "clong", "cwide"} for meta in metas))

    def test_scaled_50x_suites_are_unique_and_disjoint(self) -> None:
        train = suite_tasks("procedural_scaled_train_50x")
        final = suite_tasks("procedural_scaled_final_eval_50x")
        train_names = {task.name for task in train}
        final_names = {task.name for task in final}
        self.assertEqual(len(train), 50 * 1340)
        self.assertEqual(len(final), 50 * 100)
        self.assertEqual(len(train_names), len(train))
        self.assertEqual(len(final_names), len(final))
        self.assertTrue(train_names.isdisjoint(final_names))
        self.assertTrue(all(task_meta(task.name).profile.startswith("tr") for task in train[:100]))
        self.assertTrue(all(task_meta(task.name).profile.startswith("ev") for task in final[:100]))

    def test_semantic_hard_eval_has_held_out_families(self) -> None:
        tasks = suite_tasks("procedural_semantic_hard_eval")
        names = [task.name for task in tasks]
        families = {task_meta(task.name).family for task in tasks}
        self.assertEqual(len(tasks), 100)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("nested", families)
        self.assertIn("twostat", families)
        self.assertIn("idx", families)
        self.assertIn("predmax", families)
        self.assertIn("predmin", families)
        self.assertIn("paircmp", families)
        self.assertTrue(any(len(task.signature.params) == 2 for task in tasks))

    def test_semantic_hard_train_dev_final_are_disjoint(self) -> None:
        train = suite_tasks("procedural_semantic_hard_train")
        dev = suite_tasks("procedural_semantic_hard_dev_eval")
        final = suite_tasks("procedural_semantic_hard_final_eval")
        train_names = {task.name for task in train}
        dev_names = {task.name for task in dev}
        final_names = {task.name for task in final}
        train_families = {task_meta(task.name).family for task in train}
        self.assertEqual(len(train), 20 * 100)
        self.assertEqual(len(dev), 100)
        self.assertEqual(len(final), 100)
        self.assertEqual(len(train_names), len(train))
        self.assertEqual(len(dev_names), len(dev))
        self.assertEqual(len(final_names), len(final))
        self.assertTrue(train_names.isdisjoint(dev_names))
        self.assertTrue(train_names.isdisjoint(final_names))
        self.assertTrue(dev_names.isdisjoint(final_names))
        self.assertIn("nested", train_families)
        self.assertIn("paircmp", train_families)
        self.assertTrue(all(task_meta(task.name).profile.startswith("shtr") for task in train))
        self.assertTrue(all(task_meta(task.name).profile == "shdev00" for task in dev))

    def test_semantic_clean_train_dev_final_have_no_core_overlap(self) -> None:
        train = suite_tasks("procedural_semantic_clean_train")
        dev = suite_tasks("procedural_semantic_clean_dev_eval")
        final = suite_tasks("procedural_semantic_clean_final_eval")
        self.assertEqual(len(train), 20 * 100)
        self.assertEqual(len(dev), 100)
        self.assertEqual(len(final), 100)
        self.assertEqual(len({task.name for task in train}), len(train))
        self.assertEqual(len({task.name for task in dev}), len(dev))
        self.assertEqual(len({task.name for task in final}), len(final))
        train_rows = [{"task": {"name": task.name}} for task in train]
        dev_rows = [{"task": {"name": task.name}} for task in dev]
        final_rows = [{"task": {"name": task.name}} for task in final]
        # paircmp is a deliberate in-distribution control family: its oracle
        # semantics are shared across splits (only the replicate suffix and
        # profile differ). The hardened validator must catch exactly those
        # five cores and nothing else; held-out families stay disjoint.
        paircmp_cores = {
            "paircmp:count_pos_xs_gt_count_pos_ys",
            "paircmp:first_xs_gt_first_ys",
            "paircmp:len_xs_eq_len_ys",
            "paircmp:sum_absdiff_or_zero",
            "paircmp:sum_xs_gt_sum_ys",
        }
        for left, right in ((train_rows, dev_rows), (train_rows, final_rows), (dev_rows, final_rows)):
            report = validate_no_task_overlap(left, right)
            self.assertEqual(report.overlapping_tasks, ())
            self.assertEqual(set(report.overlapping_task_cores), paircmp_cores)
            with self.assertRaises(ValueError):
                fail_on_task_overlap(report)
            fail_on_task_overlap(report, allow_core=True)
        self.assertTrue(all(task_meta(task.name).profile.startswith("sctr") for task in train))
        self.assertTrue(all(task_meta(task.name).profile == "scdev00" for task in dev))
        self.assertTrue(all(task_meta(task.name).profile == "scfin00" for task in final))

    def test_semantic_task_conditioned_search_solves_nested_and_twostat(self) -> None:
        nested = task_by_name("nested_sum_even_after_lt_neg1", suite_tasks("procedural_semantic_hard_final_eval"))
        twostat = task_by_name("twostat_sum_even_minus_count_ne_neg2", suite_tasks("procedural_semantic_hard_final_eval"))
        nested_plain = solve_with_enumeration(nested, budget=16, hidden_count=8, seed=0)
        nested_attempt = solve_with_enumeration(
            nested,
            budget=16,
            hidden_count=8,
            seed=0,
            include_task_conditioned_gold=True,
        )
        twostat_attempt = solve_with_enumeration(
            twostat,
            budget=16,
            hidden_count=8,
            seed=0,
            include_task_conditioned_gold=True,
        )
        self.assertTrue(nested_plain is None or nested_plain.candidate_index > 1 or not nested_plain.result.solved)
        self.assertIsNotNone(nested_attempt)
        self.assertIsNotNone(twostat_attempt)
        self.assertEqual(nested_attempt.candidate_index, 1)
        self.assertEqual(twostat_attempt.candidate_index, 1)
        self.assertTrue(nested_attempt.result.solved)
        self.assertTrue(twostat_attempt.result.solved)

    def test_split_validation_detects_task_overlap(self) -> None:
        train_rows = [{"task": {"name": "a"}}, {"task": {"name": "b"}}]
        eval_rows = [{"task": {"name": "b"}}, {"task": {"name": "c"}}]
        report = validate_no_task_overlap(train_rows, eval_rows)
        self.assertEqual(report.train_tasks, 2)
        self.assertEqual(report.eval_tasks, 2)
        self.assertEqual(report.overlapping_tasks, ("b",))
        with self.assertRaises(ValueError):
            fail_on_task_overlap(report)

    def test_split_validation_accepts_disjoint_tasks(self) -> None:
        train_rows = [{"task": {"name": "a"}}]
        eval_rows = [{"task": {"name": "b"}}]
        report = validate_no_task_overlap(train_rows, eval_rows)
        self.assertEqual(report.overlap_count, 0)
        fail_on_task_overlap(report)

    def test_static_type_errors_fail_evaluation(self) -> None:
        task = task_by_name("sum_list")
        program = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (Return(BoolLit(True)),),
        )
        self.assertFalse(typecheck_function(program).ok)
        result = MiniLeetEnv(task, hidden_count=2).evaluate(program)
        self.assertEqual(result.hidden_pass_rate, 0.0)

    def test_hidden_tests_exclude_visible_inputs(self) -> None:
        task = task_by_name("sum_list")
        visible_args = {repr(args) for args, _ in task.visible_examples}
        hidden_args = {repr(args) for args, _ in task.hidden_tests(count=64, seed=0)}
        self.assertTrue(visible_args.isdisjoint(hidden_args))
        self.assertEqual(len(hidden_args), 64)

    def test_loop_type_change_is_rejected(self) -> None:
        program = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (
                Assign("x", IntLit(0)),
                ForEach("i", Var("xs"), (Assign("x", BoolLit(True)),)),
                If(Var("x"), (Return(IntLit(1)),), (Return(IntLit(0)),)),
            ),
        )
        result = typecheck_function(program)
        self.assertFalse(result.ok)
        self.assertTrue(any("loop body changes type of x" in error for error in result.errors))

    def test_if_assignments_merge_when_both_branches_agree(self) -> None:
        program = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (
                If(
                    Compare(IntLit(1), "==", IntLit(1)),
                    (Assign("x", IntLit(1)),),
                    (Assign("x", IntLit(2)),),
                ),
                Return(Var("x")),
            ),
        )
        self.assertTrue(typecheck_function(program).ok)
        self.assertEqual(run_function(program, ([1],)).value, 1)
        self.assertEqual(run_function(program, ([],)).value, 1)

    def test_invalid_operators_fail_typecheck(self) -> None:
        bad_binary = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (Return(Binary(IntLit(2), "**", IntLit(3))),),
        )
        bad_compare = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (If(Compare(IntLit(1), "===", IntLit(1)), (Return(IntLit(1)),), (Return(IntLit(0)),)),),
        )
        self.assertFalse(typecheck_function(bad_binary).ok)
        self.assertFalse(typecheck_function(bad_compare).ok)
        self.assertTrue(any("unknown binary operator **" in error for error in typecheck_function(bad_binary).errors))
        self.assertTrue(any("unknown comparison operator ===" in error for error in typecheck_function(bad_compare).errors))

    def test_bool_int_outputs_do_not_compare_equal(self) -> None:
        result = TestCaseResult((), 1, RunResult(True, value=True))
        self.assertFalse(result.passed)

    def test_task_signature_mismatch_fails_evaluation(self) -> None:
        task = task_by_name("sum_list")
        program = Function(
            "wrong",
            (Param("xs", LIST_INT),),
            INT,
            (Return(IntLit(0)),),
        )
        result = MiniLeetEnv(task, hidden_count=2).evaluate(program)
        self.assertEqual(result.hidden_pass_rate, 0.0)
        self.assertIn("function name wrong does not match solve", result.hidden[0].run.error or "")

    def test_enumerative_baseline_runs_on_eval_suite(self) -> None:
        task = task_by_name("all_nonnegative")
        self.assertGreater(len(enumerate_candidates(task)), 10)
        attempt = solve_with_enumeration(task, budget=256, hidden_count=8)
        self.assertIsNotNone(attempt)
        self.assertTrue(attempt.result.solved)

    def test_oracle_scan_budget_can_improve_first_visible(self) -> None:
        task = task_by_name("count_positive")
        first = solve_with_enumeration(task, budget=128, hidden_count=16, seed=0)
        oracle = oracle_scan_budget(task, budget=128, hidden_count=16, seed=0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(oracle)
        self.assertGreaterEqual(oracle.result.hidden_pass_rate, first.result.hidden_pass_rate)
        self.assertGreaterEqual(oracle.visible_passes_seen, first.visible_passes_seen)

    def test_collect_and_audit_small_jsonl(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.jsonl"
            stats = collect_jsonl(
                suite_tasks("train")[:1],
                path,
                budget=3,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=False,
            )
            self.assertEqual(stats["rows"], 3)
            self.assertEqual(stats["visible_execs"], stats["visible_execs_written"])
            self.assertEqual(stats["hidden_execs"], stats["hidden_execs_written"])
            self.assertEqual(stats["visible_execs_attempted"], stats["visible_execs_written"])
            self.assertEqual(stats["hidden_execs_attempted"], stats["hidden_execs_written"])
            audit = audit_jsonl(path)
            self.assertEqual(audit["rows"], 3)
            self.assertEqual(audit["tasks"], 1)
            self.assertEqual(audit["hidden_execs"], 6)
            self.assertEqual(audit["task_program_pairs"], audit["programs"])
            self.assertTrue(audit["task_contiguous"])
            self.assertIn("visible_passer_count_buckets", audit)
            with self.assertRaises(FileExistsError):
                collect_jsonl(
                    suite_tasks("train")[:1],
                    path,
                    budget=3,
                    hidden_count=2,
                    seed=0,
                    include_hidden_traces=False,
                    suite="train",
                    visible_passing_only=False,
                )

    def test_audit_rejects_noncontiguous_tasks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            collect_jsonl(
                suite_tasks("train")[:2],
                source,
                budget=2,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=False,
            )
            rows = load_rows(source)
            by_task: dict[str, list[dict[str, object]]] = {}
            for row in rows:
                by_task.setdefault(row["task"]["name"], []).append(row)
            task_names = list(by_task)
            noncontiguous = by_task[task_names[0]][:1] + by_task[task_names[1]] + by_task[task_names[0]][1:]
            with source.open("w", encoding="utf-8") as handle:
                for row in noncontiguous:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            with self.assertRaises(ValueError):
                audit_jsonl(source)

    def test_visible_passing_only_collection(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "visible_pass.jsonl"
            stats = collect_jsonl(
                [task_by_name("all_nonnegative")],
                path,
                budget=64,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="test",
                visible_passing_only=True,
            )
            self.assertGreater(stats["rows"], 0)
            self.assertGreater(stats["visible_execs_attempted"], stats["visible_execs_written"])
            self.assertGreater(stats["hidden_execs_attempted"], stats["hidden_execs_written"])
            rows = load_rows(path)
            self.assertTrue(all(row["labels"]["visible_pass_rate"] == 1.0 for row in rows))

    def test_filter_dataset_visible_passing_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            target = Path(tmpdir) / "target.jsonl"
            collect_jsonl(
                [task_by_name("all_nonnegative")],
                source,
                budget=64,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="test",
                visible_passing_only=False,
            )
            stats = filter_jsonl(source, target, visible_passing_only=True)
            self.assertGreater(stats["read"], stats["wrote"])
            rows = load_rows(target)
            self.assertTrue(all(row["labels"]["visible_pass_rate"] == 1.0 for row in rows))
            with self.assertRaises(FileExistsError):
                filter_jsonl(source, target, visible_passing_only=True)

    def test_filter_dataset_max_rows_keeps_whole_tasks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            target = Path(tmpdir) / "target.jsonl"
            collect_jsonl(
                suite_tasks("train")[:2],
                source,
                budget=4,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=False,
            )
            stats = filter_jsonl(source, target, max_rows=5)
            rows = load_rows(target)
            self.assertEqual(stats["wrote"], len(rows))
            self.assertEqual(len({row["task"]["name"] for row in rows}), 1)

    def test_mine_hard_negatives(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            target = Path(tmpdir) / "mined.jsonl"
            collect_jsonl(
                suite_tasks("train")[:3],
                source,
                budget=64,
                hidden_count=4,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=True,
            )
            stats = mine_hard_negatives(
                source,
                target,
                max_negatives_per_positive=1.0,
                max_negatives_per_task=4,
                seed=0,
            )
            self.assertGreater(stats["wrote"], 0)
            rows = load_rows(target)
            self.assertTrue(all(row["labels"]["visible_pass_rate"] == 1.0 for row in rows))

    def test_mine_hard_negatives_caps_tasks_without_positives(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            target = Path(tmpdir) / "mined.jsonl"
            base = _feature_row(
                "sum_list",
                Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),)),
            )
            with source.open("w", encoding="utf-8") as handle:
                for index in range(10):
                    row = json.loads(json.dumps(base))
                    row["candidate_index"] = index
                    row["labels"]["visible_pass_rate"] = 1.0
                    row["labels"]["hidden_pass_rate"] = index / 20.0
                    row["labels"]["solved"] = False
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            stats = mine_hard_negatives(source, target, max_negatives_per_task=3, seed=0)
            self.assertEqual(stats["read"], 10)
            self.assertEqual(stats["eligible"], 10)
            self.assertEqual(stats["negatives"], 3)
            self.assertEqual(len(load_rows(target)), 3)

    def test_mine_hard_negatives_refuses_eval_like_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "hard_eval.jsonl"
            target = Path(tmpdir) / "mined.jsonl"
            collect_jsonl(
                [task_by_name("all_nonnegative")],
                source,
                budget=4,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="eval",
                visible_passing_only=True,
            )
            with self.assertRaises(ValueError):
                mine_hard_negatives(source, target)

    def test_analyze_failures_handles_null_unscored_predictions(self) -> None:
        row = _feature_row(
            "sum_list",
            Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),)),
        )
        row["labels"]["visible_pass_rate"] = 1.0
        row["labels"]["hidden_pass_rate"] = 1.0
        row["labels"]["solved"] = True
        predictions = {
            ("sum_list", 0): {
                "task": "sum_list",
                "candidate_index": 0,
                "scores": {"blend": None},
            }
        }
        report = analyze_failures([row], predictions, score_mode="blend")
        self.assertEqual(report["summary"]["tasks"], 1)
        self.assertEqual(report["summary"]["oracle_covered"], 0)
        self.assertEqual(report["summary"]["considered_candidates_avg"], 0.0)
        self.assertIsNone(report["tasks"][0]["oracle_rank"])
        self.assertIsNone(report["tasks"][0]["score_gap_oracle_minus_selected"])

    def test_feature_and_rerank_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.jsonl"
            collect_jsonl(
                suite_tasks("train")[:3],
                path,
                budget=4,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=False,
            )
            rows = load_rows(path)
            spec = fit_feature_spec(rows, max_tokens=16)
            self.assertFalse(spec.include_position_features)
            self.assertNotIn("candidate_index_log", feature_names(spec))
            position_spec = fit_feature_spec(rows, max_tokens=16, include_position_features=True)
            self.assertIn("candidate_index_log", feature_names(position_spec))
            self.assertGreater(len(extract_features(rows[0], spec)), 10)
            result = train_and_score(
                rows,
                rows,
                epochs=2,
                lr=1e-3,
                hidden=8,
                max_tokens=16,
                seed=0,
            )
            self.assertEqual(len(result["eval_predictions"]), len(rows))

    def test_rerank_selection_is_defensive(self) -> None:
        good = _feature_row("sum_list", Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),)))
        bad = json.loads(json.dumps(good))
        good["candidate_index"] = 1
        bad["candidate_index"] = 5
        good["labels"]["visible_pass_rate"] = 1.0
        good["labels"]["hidden_pass_rate"] = 1.0
        good["labels"]["solved"] = True
        bad["labels"]["visible_pass_rate"] = 1.0
        bad["labels"]["hidden_pass_rate"] = 0.0
        bad["labels"]["solved"] = False

        metrics = evaluate_first_visible_pass([bad, good])
        self.assertEqual(metrics.selected, 1)
        self.assertEqual(metrics.solved, 1)
        self.assertEqual(metrics.mean_candidate_index, 1.0)
        with self.assertRaises(ValueError):
            evaluate_predicted_selection([bad, good], [0.1], visible_gated=True)

    def test_rerank_checkpoint_roundtrip_and_std_mask(self) -> None:
        row = _feature_row("sum_list", Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),)))
        train_rows = [json.loads(json.dumps(row)) for _ in range(4)]
        eval_row = json.loads(json.dumps(row))
        eval_row["visible"] = eval_row["visible"][:3]
        eval_row["labels"]["visible_total"] = 3
        result = train_and_score(
            train_rows,
            [eval_row],
            epochs=2,
            lr=1e-3,
            hidden=8,
            max_tokens=8,
            seed=0,
            val_fraction=0.0,
            patience=0,
        )
        self.assertTrue(any(not active for active in result["active_features"]))
        for std, active in zip(result["std"], result["active_features"]):
            if not active:
                self.assertEqual(std, 0.0)
        self.assertEqual(result["args"]["hidden"], 8)

        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "rerank.pt"
            _save_model(checkpoint, result)
            loaded = load_model(checkpoint, device="cpu")
            reloaded_predictions = predict_rows([eval_row], loaded)
            self.assertAlmostEqual(result["eval_predictions"][0], reloaded_predictions[0], places=6)
            self.assertEqual(loaded["args"]["hidden"], 8)

    def test_structured_program_features(self) -> None:
        task = task_by_name("contains_target")
        program = Function(
            "solve",
            (Param("xs", LIST_INT), Param("target", INT)),
            BOOL,
            (
                Assign("found", BoolLit(False)),
                ForEach(
                    "x",
                    Var("xs"),
                    (
                        If(
                            Compare(Var("x"), "==", Var("target")),
                            (Assign("found", BoolLit(True)),),
                        ),
                    ),
                ),
                Return(Var("found")),
            ),
        )
        row = _feature_row(task.name, program)
        values = _feature_values(row)
        self.assertEqual(values["program_uses_int_param_body"], 1.0)
        self.assertEqual(values["program_pred_eq"], 1.0)
        self.assertEqual(values["program_sets_found_true"], 1.0)
        self.assertNotIn("program_found_set_true", values)

        ignore_param = Function(
            "solve",
            (Param("xs", LIST_INT), Param("target", INT)),
            BOOL,
            (Return(BoolLit(False)),),
        )
        ignore_values = _feature_values(_feature_row(task.name, ignore_param))
        self.assertEqual(ignore_values["program_uses_int_param_body"], 0.0)

    def test_predicate_features_ignore_template_artifacts(self) -> None:
        task_name = "count_positive"
        paired_predicates = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (
                Assign("acc", IntLit(0)),
                ForEach(
                    "x",
                    Var("xs"),
                    (
                        If(Compare(Var("x"), ">", IntLit(-2)), (Assign("acc", Binary(Var("acc"), "+", Var("x"))),)),
                        If(Compare(Var("x"), ">=", IntLit(-1)), (Assign("acc", Binary(Var("acc"), "+", IntLit(1))),)),
                    ),
                ),
                Return(Var("acc")),
            ),
        )
        values = _feature_values(_feature_row(task_name, paired_predicates))
        self.assertEqual(values["program_pred_gt"], 1.0)
        self.assertEqual(values["program_pred_ge"], 1.0)
        self.assertEqual(values["program_x_pred_const_neg2"], 1.0)
        self.assertEqual(values["program_x_pred_const_neg1"], 1.0)

        parity = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (
                Assign("acc", IntLit(0)),
                ForEach(
                    "x",
                    Var("xs"),
                    (
                        If(
                            Compare(Binary(Var("x"), "%", IntLit(2)), "==", IntLit(0)),
                            (Assign("acc", Binary(Var("acc"), "+", IntLit(1))),),
                        ),
                    ),
                ),
                Return(Var("acc")),
            ),
        )
        parity_values = _feature_values(_feature_row(task_name, parity))
        self.assertEqual(parity_values["program_pred_eq"], 0.0)
        self.assertEqual(parity_values["program_x_pred_const_2"], 0.0)

    def test_feature_schema_and_tokenizer_regressions(self) -> None:
        tokens = TOKEN_RE.findall("acc = -3\nif (x >= -1):")
        self.assertIn("=", tokens)
        self.assertIn("-3", tokens)
        self.assertIn("-1", tokens)
        self.assertNotIn("3", tokens)

        program = Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),))
        row = _feature_row("sum_list", program)
        del row["visible"][0]["run"]["trace_len"]
        values = _feature_values(row)
        self.assertGreater(len(values), 10)

    def test_sequence_tokens_use_serialized_schema(self) -> None:
        program = Function(
            "solve",
            (Param("xs", LIST_INT), Param("target", INT)),
            BOOL,
            (
                Assign("found", BoolLit(False)),
                ForEach(
                    "x",
                    Var("xs"),
                    (
                        If(
                            Compare(Var("x"), "==", Var("target")),
                            (Assign("found", BoolLit(True)),),
                        ),
                    ),
                ),
                Return(Var("found")),
            ),
        )
        row = _feature_row("contains_target", program)
        tokens = row_to_tokens(row)
        structured = row_to_structured_tokens(row)
        position_tokens = row_to_tokens(row, include_position_tokens=True)
        position_structured = row_to_structured_tokens(row, include_position_tokens=True)
        self.assertIn("case_passed:True", tokens)
        self.assertFalse(any(token.startswith("candidate_bucket:") for token in tokens))
        self.assertFalse(any(token.startswith("candidate_bucket:") for token in structured["program"]))
        self.assertTrue(any(token.startswith("candidate_bucket:") for token in position_tokens))
        self.assertTrue(any(token.startswith("candidate_bucket:") for token in position_structured["program"]))
        self.assertIn("trace_name:str:found", tokens)
        self.assertIn("trace_item:str:x", tokens)
        self.assertIn("trace_cond:str:x_eq_target", tokens)
        self.assertNotIn("trace_name:other", tokens)
        self.assertNotIn("trace_item:other", tokens)
        self.assertNotIn("trace_cond:other", tokens)
        self.assertTrue(all(test_tokens[0] == "section:test" for test_tokens in structured["tests"]))
        del row["visible"][0]["passed"]
        with self.assertRaises(KeyError):
            row_to_tokens(row)

    def test_sequence_error_codes_and_truncation(self) -> None:
        self.assertEqual(error_code_from_message("undefined variable foo"), "undefined_variable")
        self.assertEqual(error_code_from_message("unknown binary operator **"), "unknown_binary_operator")
        record = run_to_record(RunResult(False, error="undefined variable foo"))
        self.assertEqual(record["error_code"], "undefined_variable")

        program = Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),))
        row = _feature_row("sum_list", program)
        row["visible"][0]["run"] = record
        tokens = row_to_tokens(row)
        self.assertIn("run_error:undefined_variable", tokens)
        self.assertNotIn("run_error:undefined variable foo", tokens)

        spec = TokenSpec((PAD, UNK, CLS, SEP, TRUNC, "a", "b", "c"))
        ids, mask = encode_tokens(["a", "b", "c"], spec, max_len=3)
        self.assertEqual(ids, [spec.stoi[CLS], spec.stoi["a"], spec.stoi[TRUNC]])
        self.assertEqual(mask, [1, 1, 1])
        self.assertIs(spec.stoi, spec.stoi)
        with self.assertRaises(ValueError):
            encode_tokens(["a"], spec, max_len=0)

    def test_sequence_arg_lists_keep_longer_inputs(self) -> None:
        program = Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),))
        row = _feature_row("sum_list", program)
        row["visible"][0]["args"] = [list(range(12))]
        tokens = row_to_tokens(row)
        self.assertIn("arg_item_item:int:9", tokens)
        self.assertNotIn("arg_item:list_truncated", tokens)

    def test_sequence_and_transformer_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.jsonl"
            collect_jsonl(
                suite_tasks("train")[:3],
                path,
                budget=4,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=False,
            )
            rows = load_rows(path)
            spec = fit_token_spec(rows, max_vocab=64)
            ids, mask = encode_row(rows[0], spec, max_len=64)
            self.assertEqual(len(ids), 64)
            self.assertEqual(len(mask), 64)
            result = train_transformer(
                rows,
                rows,
                TrainConfig(
                    epochs=1,
                    batch_size=4,
                    lr=1e-3,
                    max_len=64,
                    max_vocab=64,
                    d_model=16,
                    layers=1,
                    heads=2,
                    ff_mult=2,
                    dropout=0.0,
                    seed=0,
                    val_fraction=0.0,
                    patience=0,
                ),
            )
            self.assertEqual(len(result["eval_predictions"]), len(rows))
            self.assertEqual(result["val_rows"], 0)
            with TemporaryDirectory() as model_tmpdir:
                checkpoint = Path(model_tmpdir) / "transformer.pt"
                save_transformer_model(checkpoint, result, result["config"])
                loaded = load_transformer_model(checkpoint, device="cpu")
                reloaded_predictions = predict_transformer_rows(rows, loaded)
                for left, right in zip(result["eval_predictions"], reloaded_predictions):
                    self.assertLess(abs(left - right), 1e-5)

    def test_transformer_module_import_is_lazy(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import minileet.transformer_rerank; print('torch' in sys.modules)",
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_structured_shared_reranker_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.jsonl"
            run_dir = Path(tmpdir) / "run"
            collect_jsonl(
                suite_tasks("train")[:3],
                path,
                budget=4,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=False,
            )
            rows = load_rows(path)
            result = train_structured(
                rows,
                rows,
                StructuredConfig(
                    epochs=1,
                    batch_size=4,
                    lr=1e-3,
                    max_vocab=64,
                    feature_tokens=16,
                    task_len=32,
                    program_len=64,
                    test_len=64,
                    max_tests=5,
                    d_model=16,
                    heads=2,
                    encoder_layers=1,
                    fusion_layers=1,
                    ff_mult=2,
                    dropout=0.0,
                    numeric_hidden=8,
                    registers=1,
                    bce_weight=0.1,
                    rank_weight=0.0,
                    listwise_weight=0.0,
                    teacher_feature=True,
                    teacher_tokens=16,
                    teacher_hidden=8,
                    teacher_epochs=2,
                    teacher_lr=1e-3,
                    teacher_weight=0.1,
                    teacher_alpha=0.8,
                    score_mode="teacher_solved",
                    eval_every=1,
                    seed=0,
                ),
                run_dir=run_dir,
            )
            self.assertEqual(len(result["eval_predictions"]), len(rows))
            self.assertGreater(result["feature_count"], 10)
            self.assertIsNotNone(result["teacher_metrics"])
            self.assertIn("hidden", result["eval_prediction_sets"])
            self.assertIn("rank", result["eval_prediction_sets"])
            self.assertIn("teacher_solved", result["eval_prediction_sets"])
            self.assertIn("hidden", result["selector_metrics"])
            self.assertGreater(result["val_rows"], 0)
            self.assertEqual(result["monitor_split"], "val")
            self.assertTrue((run_dir / "history.jsonl").exists())
            self.assertTrue((run_dir / "best_epoch.json").exists())
            self.assertTrue((run_dir / "best_model.pt").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "predictions.jsonl").exists())
            self.assertTrue((run_dir / "loss.png").exists())
            loaded = load_structured_model(run_dir / "best_model.pt", device="cpu")
            reloaded_predictions = predict_structured_rows(
                rows,
                loaded,
                teacher_scores=result["eval_prediction_sets"]["teacher"],
            )
            self.assertEqual(len(reloaded_predictions), len(rows))

            starter_row = _feature_row(
                "sum_list",
                Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),)),
            )
            tensors, _ = encode_structured_rows(
                [starter_row],
                result["token_spec"],
                result["feature_spec"],
                result["config"],
                teacher_scores=[0.0],
            )
            test_mask_sums = tensors[5][0].sum(dim=-1).tolist()
            self.assertTrue(any(value == 1 for value in test_mask_sums))

    def test_structured_module_import_is_lazy(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import minileet.structured_rerank; print('torch' in sys.modules)",
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_candidate_set_reranker_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.jsonl"
            run_dir = Path(tmpdir) / "set_run"
            collect_jsonl(
                [task_by_name("all_nonnegative")],
                path,
                budget=64,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="test",
                visible_passing_only=True,
            )
            rows = load_rows(path)
            result = train_set_reranker(
                rows,
                rows,
                SetConfig(
                    epochs=1,
                    lr=1e-3,
                    max_tokens=16,
                    d_model=16,
                    layers=1,
                    heads=2,
                    ff_mult=2,
                    dropout=0.0,
                    max_candidates=16,
                    mse_weight=0.5,
                    bce_weight=0.1,
                    listwise_weight=0.1,
                    solved_listwise_weight=0.1,
                    teacher_feature=True,
                    teacher_tokens=16,
                    teacher_hidden=8,
                    teacher_epochs=2,
                    teacher_lr=1e-3,
                    teacher_weight=0.1,
                    teacher_alpha=0.8,
                    score_mode="teacher_blend",
                    eval_every=1,
                    seed=0,
                ),
                run_dir=run_dir,
            )
            self.assertEqual(len(result["eval_predictions"]), len(rows))
            self.assertIn("teacher_blend", result["eval_prediction_sets"])
            self.assertTrue((run_dir / "best_model.pt").exists())
            self.assertTrue((run_dir / "best_predictions.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            loaded = load_set_model(run_dir / "best_model.pt", device="cpu")
            reloaded_predictions = predict_set_rows(
                rows,
                loaded,
                teacher_scores=result["eval_prediction_sets"]["teacher"],
            )
            self.assertEqual(len(reloaded_predictions), len(rows))
            report = analyze_failures(
                rows,
                {
                    (row["task"], int(row["candidate_index"])): row
                    for row in (
                        json.loads(line)
                        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                    )
                },
                score_mode="teacher_blend",
                out_dir=Path(tmpdir) / "analysis",
            )
            self.assertEqual(report["summary"]["tasks"], 1)
            self.assertTrue((Path(tmpdir) / "analysis" / "summary.json").exists())

    def test_candidate_set_grouping_eval_is_label_free(self) -> None:
        rows = []
        for index in range(5):
            row = _feature_row("sum_list", Function("solve", (Param("xs", LIST_INT),), INT, (Return(IntLit(0)),)))
            row["candidate_index"] = index
            row["labels"]["visible_pass_rate"] = 1.0
            row["labels"]["solved"] = index == 4
            row["labels"]["hidden_pass_rate"] = 1.0 if index == 4 else 0.0
            rows.append(row)
        label_free = group_candidate_indices(rows, visible_only=True, max_candidates=3, label_aware=False)
        label_aware = group_candidate_indices(rows, visible_only=True, max_candidates=3, label_aware=True)
        self.assertEqual(label_free, [[0, 1, 2]])
        self.assertIn(4, label_aware[0])

    def test_candidate_set_module_import_is_lazy(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import minileet.candidate_set_rerank; print('torch' in sys.modules)",
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_compact_candidate_set_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            cache_dir = root / "cache"
            run_dir = root / "compact_run"
            collect_jsonl(
                [task_by_name("all_nonnegative")],
                train_path,
                budget=64,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=True,
            )
            collect_jsonl(
                [task_by_name("count_positive")],
                eval_path,
                budget=64,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="eval",
                visible_passing_only=False,
            )
            report = build_cache(train_path, eval_path, cache_dir, max_tokens=16)
            self.assertEqual(report.overlap_count, 0)
            result = train_from_cache(
                cache_dir,
                SetConfig(
                    epochs=1,
                    lr=1e-3,
                    max_tokens=16,
                    d_model=16,
                    layers=1,
                    heads=2,
                    ff_mult=2,
                    dropout=0.0,
                    max_candidates=16,
                    mse_weight=0.5,
                    bce_weight=0.1,
                    listwise_weight=0.1,
                    solved_listwise_weight=0.1,
                    teacher_feature=False,
                    teacher_tokens=0,
                    teacher_hidden=0,
                    teacher_epochs=0,
                    teacher_lr=0.0,
                    teacher_weight=0.0,
                    teacher_alpha=0.0,
                    score_mode="rank",
                    eval_every=1,
                    seed=0,
                ),
                run_dir=run_dir,
            )
            self.assertEqual(result["train_tasks"], 1)
            self.assertEqual(result["eval_tasks"], 1)
            self.assertTrue((cache_dir / "train_features.f32").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "split_validation.json").exists())
            self.assertTrue((run_dir / "best_model.pt").exists())
            loaded = load_compact_model(run_dir / "best_model.pt", device="cpu")
            rows = load_rows(eval_path)
            predictions = predict_compact_rows(rows, loaded)
            self.assertEqual(len(predictions), len(rows))

    def test_compact_cache_rejects_noncontiguous_tasks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.jsonl"
            eval_path = root / "eval.jsonl"
            cache_dir = root / "cache"
            collect_jsonl(
                suite_tasks("train")[:2],
                source,
                budget=4,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="train",
                visible_passing_only=False,
            )
            rows = load_rows(source)
            by_task: dict[str, list[dict[str, object]]] = {}
            for row in rows:
                by_task.setdefault(row["task"]["name"], []).append(row)
            task_names = list(by_task)
            noncontiguous = by_task[task_names[0]][:1] + by_task[task_names[1]] + by_task[task_names[0]][1:]
            with source.open("w", encoding="utf-8") as handle:
                for row in noncontiguous:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            collect_jsonl(
                [task_by_name("count_positive")],
                eval_path,
                budget=4,
                hidden_count=2,
                seed=0,
                include_hidden_traces=False,
                suite="eval",
                visible_passing_only=False,
            )
            with self.assertRaises(ValueError):
                build_cache(source, eval_path, cache_dir, max_tokens=8)

    def test_run_dir_no_overwrite(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = timestamped_run_dir(root, prefix="x", seed=3)
            ensure_new_run_dir(run_dir)
            (run_dir / "marker").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_new_run_dir(run_dir)
            second = timestamped_run_dir(root, prefix="x", seed=3)
            self.assertNotEqual(run_dir, second)

    def test_create_timestamped_run_dir_is_unique(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = create_timestamped_run_dir(root, prefix="x", seed=3)
            second = create_timestamped_run_dir(root, prefix="x", seed=3)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertNotEqual(first, second)

    def test_structured_run_root_helpers(self) -> None:
        self.assertEqual(
            run_root("train", "Semantic Hard Search V2", "Candidate Set Blend"),
            PROJECT_ROOT / "runs/train/semantic_hard_search_v2/candidate_set_blend",
        )
        self.assertEqual(
            infer_dataset_slug(
                "data/semantic_hard_train_20x_visible_pass_search_v2.jsonl",
                "data/semantic_hard_dev_100_enum_search_v2.jsonl",
            ),
            "semantic_hard_search_v2",
        )
        self.assertEqual(infer_dataset_slug("data/scaled_train_50x_visible_pass.jsonl"), "scaled_50x")
        self.assertEqual(infer_dataset_slug("data/std_eval_enum.jsonl"), "standard")

    def test_split_validation_flags_profile_stripped_overlap(self) -> None:
        train = [{"task": {"name": "nested_sum_even_after_lt_neg1_shtr00"}}]
        eval_rows = [{"task": {"name": "nested_sum_even_after_lt_neg1"}}]
        report = validate_no_task_overlap(train, eval_rows)
        self.assertEqual(report.overlapping_tasks, ())
        self.assertEqual(report.overlapping_task_cores, ("nested:sum_even_after_lt_neg1",))
        with self.assertRaises(ValueError):
            fail_on_task_overlap(report)

    def test_sweep_row_limits_keep_whole_tasks(self) -> None:
        rows = [
            {"task": {"name": "a"}},
            {"task": {"name": "a"}},
            {"task": {"name": "b"}},
            {"task": {"name": "b"}},
            {"task": {"name": "c"}},
        ]
        self.assertEqual(_take_whole_task_rows(rows, 3), rows[:2])
        self.assertEqual(_take_whole_task_rows(rows, 1), rows[:2])
        self.assertEqual(_take_task_count(rows, 2), rows[:4])


class HardeningTests(unittest.TestCase):
    def test_step_budget_exceeded(self) -> None:
        program = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (
                Assign("acc", IntLit(0)),
                ForEach("x", Var("xs"), (Assign("acc", Binary(Var("acc"), "+", Var("x"))),)),
                Return(Var("acc")),
            ),
        )
        ok = run_function(program, ([1, 2, 3],))
        self.assertTrue(ok.ok)
        self.assertGreater(ok.steps, 0)
        self.assertFalse(ok.trace_truncated)

        capped = run_function(program, (list(range(1000)),), max_steps=50)
        self.assertFalse(capped.ok)
        self.assertEqual(capped.error, "step-budget-exceeded")
        self.assertEqual(capped.trace[-1].kind, "error")
        self.assertEqual(error_code_from_message(capped.error), "step_budget_exceeded")

    def test_trace_cap_keeps_terminal_event(self) -> None:
        program = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (
                Assign("acc", IntLit(0)),
                ForEach("x", Var("xs"), (Assign("acc", Binary(Var("acc"), "+", Var("x"))),)),
                Return(Var("acc")),
            ),
        )
        result = run_function(program, (list(range(100)),), max_trace_events=10)
        self.assertTrue(result.ok)
        self.assertEqual(result.value, sum(range(100)))
        self.assertTrue(result.trace_truncated)
        self.assertEqual(len(result.trace), 11)
        self.assertEqual(result.trace[-1].kind, "finish")
        record = run_to_record(result, include_trace=False)
        self.assertTrue(record["trace_truncated"])
        self.assertGreater(record["steps"], 0)

    def test_env_step_budget_produces_failed_cases(self) -> None:
        task = task_by_name("sum_list")
        env = MiniLeetEnv(task, hidden_count=4, seed=0, max_steps=3)
        program = Function(
            "solve",
            (Param("xs", LIST_INT),),
            INT,
            (
                Assign("acc", IntLit(0)),
                ForEach("x", Var("xs"), (Assign("acc", Binary(Var("acc"), "+", Var("x"))),)),
                Return(Var("acc")),
            ),
        )
        result = env.evaluate(program)
        self.assertFalse(result.solved)
        self.assertTrue(
            any(case.run.error == "step-budget-exceeded" for case in result.visible + result.hidden)
        )

    def test_collect_marks_provenance_and_records_gold_flag(self) -> None:
        from minileet.search import task_conditioned_renders
        from minileet.tasks import procedural_semantic_hard_dev_eval_tasks

        tasks = procedural_semantic_hard_dev_eval_tasks(count=1)
        self.assertTrue(task_conditioned_renders(tasks[0]))
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "gold.jsonl"
            stats = collect_jsonl(
                tasks,
                out,
                budget=10,
                hidden_count=8,
                seed=0,
                include_hidden_traces=False,
                suite="procedural_semantic_hard_dev_eval",
                include_task_conditioned_gold=True,
            )
            self.assertEqual(stats["task_conditioned_rows"], 1)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["include_task_conditioned_gold"] for row in rows))
            gold_rows = [row for row in rows if row["program"]["source"] == "task_conditioned"]
            self.assertEqual([row["candidate_index"] for row in gold_rows], [1])
            audit = audit_jsonl(out)
            self.assertEqual(audit["program_sources"]["task_conditioned"], 1)

        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "no_gold.jsonl"
            stats = collect_jsonl(
                tasks,
                out,
                budget=10,
                hidden_count=8,
                seed=0,
                include_hidden_traces=False,
                suite="procedural_semantic_hard_dev_eval",
            )
            self.assertEqual(stats["task_conditioned_rows"], 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["program"]["source"] == "enumerated" for row in rows))
            self.assertTrue(all(not row["include_task_conditioned_gold"] for row in rows))

    def test_collect_reports_hidden_underfill(self) -> None:
        from minileet.tasks import Signature, Task

        tiny = Task(
            name="sum_tiny_inputs",
            signature=Signature("solve", (Param("xs", LIST_INT),), INT),
            visible_examples=((((0,),), 0),),
            oracle=lambda xs: sum(xs),
            sampler=lambda rng: ((rng.randint(0, 1), rng.randint(0, 1)),),
        )
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "tiny.jsonl"
            stats = collect_jsonl(
                [tiny],
                out,
                budget=5,
                hidden_count=16,
                seed=0,
                include_hidden_traces=False,
                suite="starter",
            )
            self.assertEqual(stats["tasks_hidden_underfilled"], 1)
            audit = audit_jsonl(out)
            self.assertEqual(audit["hidden_underfilled_tasks"], 1)
            self.assertEqual(audit["hidden_underfilled_rows"], stats["rows"])

    def test_validate_name_overlap_detects_cores(self) -> None:
        from minileet.split_validation import validate_name_overlap

        report = validate_name_overlap({"count_even_short"}, {"count_even_long"})
        self.assertEqual(report.overlapping_tasks, ())
        self.assertEqual(report.overlapping_task_cores, ("count:even",))
        with self.assertRaises(ValueError):
            fail_on_task_overlap(report)
        clean = validate_name_overlap({"count_even_short"}, {"sum_odd_long"})
        self.assertEqual(clean.overlap_count, 0)

    def test_paircmp_replicate_suffix_is_not_semantics(self) -> None:
        from minileet.split_validation import validate_name_overlap

        # Replicate counters in paircmp names must collapse to one core...
        report = validate_name_overlap(
            {"paircmp_sum_xs_gt_sum_ys_0_sctr00"},
            {"paircmp_sum_xs_gt_sum_ys_4_scdev00"},
        )
        self.assertEqual(report.overlapping_task_cores, ("paircmp:sum_xs_gt_sum_ys",))
        # ...but trailing digits in other families ARE semantics (gt_1 != gt_2).
        distinct = validate_name_overlap(
            {"nested_sum_even_after_gt_1_sctr00"},
            {"nested_sum_even_after_gt_2_scdev00"},
        )
        self.assertEqual(distinct.overlap_count, 0)

    def test_overlap_severities_are_separately_waivable(self) -> None:
        from minileet.split_validation import validate_name_overlap

        core_only = validate_name_overlap({"count_even_short"}, {"count_even_long"})
        with self.assertRaises(ValueError):
            fail_on_task_overlap(core_only)
        fail_on_task_overlap(core_only, allow_core=True)
        fail_on_task_overlap(core_only, allow_exact=True)

        exact = validate_name_overlap({"count_even_short"}, {"count_even_short"})
        with self.assertRaises(ValueError):
            fail_on_task_overlap(exact, allow_core=True)
        fail_on_task_overlap(exact, allow_exact=True)

    def test_eval_checkpoint_roundtrip_with_persisted_teacher(self) -> None:
        from minileet.eval_checkpoint import evaluate_checkpoint
        from minileet.tasks import procedural_tasks

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_tasks = default_tasks()[:3]
            eval_tasks = procedural_tasks(split="eval", count=2)
            collect_jsonl(
                train_tasks, tmp / "train.jsonl", budget=40, hidden_count=6, seed=0,
                include_hidden_traces=False, suite="starter",
            )
            collect_jsonl(
                eval_tasks, tmp / "eval.jsonl", budget=40, hidden_count=6, seed=0,
                include_hidden_traces=False, suite="procedural_eval",
            )
            train_rows = load_rows(tmp / "train.jsonl")
            eval_rows = load_rows(tmp / "eval.jsonl")
            config = SetConfig(
                epochs=1, lr=3e-4, max_tokens=16, d_model=16, layers=1, heads=2,
                ff_mult=2, dropout=0.0, max_candidates=8, mse_weight=0.5,
                bce_weight=0.2, listwise_weight=0.5, solved_listwise_weight=0.2,
                teacher_feature=True, teacher_tokens=16, teacher_hidden=8,
                teacher_epochs=3, teacher_lr=1e-3, teacher_weight=0.1,
                teacher_alpha=0.8, score_mode="teacher_blend", eval_every=1,
                seed=0, val_fraction=0.25, patience=2,
            )
            run_dir = tmp / "run"
            train_set_reranker(train_rows, eval_rows, config, run_dir=run_dir, visible_gated=True)
            checkpoint_path = run_dir / "best_model.pt"
            self.assertTrue(checkpoint_path.exists())

            loaded = load_set_model(checkpoint_path)
            self.assertIsNotNone(loaded["teacher"])
            self.assertEqual(loaded["train_task_names"], sorted(task.name for task in train_tasks))
            predictions = predict_set_rows(eval_rows, loaded)
            self.assertEqual(len(predictions), len(eval_rows))
            self.assertTrue(any(value is not None for value in predictions))
            override = predict_set_rows(eval_rows, loaded, score_mode="rank")
            self.assertNotEqual(predictions, override)

            result = evaluate_checkpoint(checkpoint_path, eval_rows)
            self.assertEqual(result["model_type"], "candidate_set_rerank")
            self.assertEqual(result["score_mode"], "teacher_blend")
            self.assertIsNotNone(result["split_report"])
            self.assertEqual(result["split_report"].overlap_count, 0)
            self.assertTrue(result["by_family"])
            for stats in result["by_family"].values():
                self.assertGreaterEqual(stats["covered"], stats["solved_covered"])
                self.assertGreaterEqual(stats["tasks"], stats["covered"])
            with self.assertRaises(ValueError):
                evaluate_checkpoint(checkpoint_path, train_rows)

            mlp = train_and_score(
                train_rows, eval_rows, epochs=10, lr=1e-3, hidden=8,
                max_tokens=16, seed=0, val_fraction=0.25, patience=3,
            )
            _save_model(tmp / "mlp.pt", mlp)
            mlp_result = evaluate_checkpoint(tmp / "mlp.pt", eval_rows)
            self.assertEqual(mlp_result["model_type"], "rerank")
            self.assertEqual(mlp_result["score_mode"], "score")
            with self.assertRaises(ValueError):
                evaluate_checkpoint(tmp / "mlp.pt", eval_rows, score_mode="rank")


if __name__ == "__main__":
    unittest.main()
