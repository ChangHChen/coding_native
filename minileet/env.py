from __future__ import annotations

from dataclasses import dataclass

from minileet.dsl import Function
from minileet.interpreter import RunResult, run_function
from minileet.tasks import Example, Task
from minileet.typecheck import typecheck_function


@dataclass(frozen=True)
class TestCaseResult:
    args: tuple[object, ...]
    expected: object
    run: RunResult

    @property
    def passed(self) -> bool:
        return self.run.ok and self.run.value == self.expected


@dataclass(frozen=True)
class EvaluationResult:
    task_name: str
    visible: tuple[TestCaseResult, ...]
    hidden: tuple[TestCaseResult, ...]

    @property
    def visible_passes(self) -> int:
        return sum(result.passed for result in self.visible)

    @property
    def hidden_passes(self) -> int:
        return sum(result.passed for result in self.hidden)

    @property
    def visible_pass_rate(self) -> float:
        return self.visible_passes / len(self.visible) if self.visible else 0.0

    @property
    def hidden_pass_rate(self) -> float:
        return self.hidden_passes / len(self.hidden) if self.hidden else 0.0

    @property
    def solved(self) -> bool:
        return bool(self.hidden) and self.hidden_passes == len(self.hidden)

    @property
    def reward(self) -> float:
        if self.solved:
            return 1.0
        return 0.2 * self.visible_pass_rate + 0.8 * self.hidden_pass_rate


class MiniLeetEnv:
    def __init__(self, task: Task, hidden_count: int = 64, seed: int = 0) -> None:
        self.task = task
        self.hidden_tests = task.hidden_tests(hidden_count, seed)

    def run_visible(self, program: Function) -> tuple[TestCaseResult, ...]:
        type_errors = typecheck_function(program).errors
        if type_errors:
            return self._failed_tests(self.task.visible_examples, type_errors)
        return self._run_tests(program, self.task.visible_examples)

    def evaluate(self, program: Function) -> EvaluationResult:
        type_errors = typecheck_function(program).errors
        if type_errors:
            visible = self._failed_tests(self.task.visible_examples, type_errors)
            hidden = self._failed_tests(self.hidden_tests, type_errors)
            return EvaluationResult(self.task.name, visible, hidden)
        visible = self.run_visible(program)
        hidden = self._run_tests(program, self.hidden_tests)
        return EvaluationResult(self.task.name, visible, hidden)

    def _run_tests(self, program: Function, tests: tuple[Example, ...]) -> tuple[TestCaseResult, ...]:
        results: list[TestCaseResult] = []
        for args, expected in tests:
            normalized_args = tuple(_normalize_arg(arg) for arg in args)
            run = run_function(program, normalized_args)
            results.append(TestCaseResult(normalized_args, expected, run))
        return tuple(results)

    def _failed_tests(
        self,
        tests: tuple[Example, ...],
        type_errors: tuple[str, ...],
    ) -> tuple[TestCaseResult, ...]:
        error = "type-error: " + "; ".join(type_errors)
        return tuple(
            TestCaseResult(
                tuple(_normalize_arg(arg) for arg in args),
                expected,
                RunResult(False, error=error),
            )
            for args, expected in tests
        )


def _normalize_arg(arg: object) -> object:
    if isinstance(arg, tuple):
        return list(arg)
    return arg
