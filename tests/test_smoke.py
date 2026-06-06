import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from minileet.baselines import solve_with_templates
from minileet.audit import audit_jsonl
from minileet.collect import collect_jsonl
from minileet.dsl import Assign, Binary, BoolLit, ForEach, Function, INT, Param, Return, Var, LIST_INT, IntLit
from minileet.env import MiniLeetEnv
from minileet.interpreter import run_function
from minileet.features import extract_features, fit_feature_spec, load_rows
from minileet.filter_dataset import filter_jsonl
from minileet.rerank import train_and_score
from minileet.sequence import encode_row, fit_token_spec
from minileet.search import enumerate_candidates, solve_with_enumeration
from minileet.structured_rerank import StructuredConfig, train_and_score as train_structured
from minileet.tasks import default_tasks, suite_tasks, task_by_name, task_meta
from minileet.transformer_rerank import TrainConfig, train_and_score as train_transformer
from minileet.typecheck import typecheck_function


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

    def test_enumerative_baseline_runs_on_eval_suite(self) -> None:
        task = task_by_name("all_nonnegative")
        self.assertGreater(len(enumerate_candidates(task)), 10)
        attempt = solve_with_enumeration(task, budget=256, hidden_count=8)
        self.assertIsNotNone(attempt)
        self.assertTrue(attempt.result.solved)

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
            audit = audit_jsonl(path)
            self.assertEqual(audit["rows"], 3)
            self.assertEqual(audit["tasks"], 1)
            self.assertEqual(audit["hidden_execs"], 6)

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

    def test_feature_and_rerank_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.jsonl"
            collect_jsonl(
                suite_tasks("train")[:2],
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

    def test_sequence_and_transformer_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.jsonl"
            collect_jsonl(
                suite_tasks("train")[:2],
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
                ),
            )
            self.assertEqual(len(result["eval_predictions"]), len(rows))

    def test_structured_shared_reranker_smoke(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small.jsonl"
            run_dir = Path(tmpdir) / "run"
            collect_jsonl(
                suite_tasks("train")[:2],
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
                    eval_every=1,
                    seed=0,
                ),
                run_dir=run_dir,
            )
            self.assertEqual(len(result["eval_predictions"]), len(rows))
            self.assertTrue((run_dir / "history.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "loss.png").exists())


if __name__ == "__main__":
    unittest.main()
