from __future__ import annotations

from dataclasses import dataclass

from minileet.dsl import Function
from minileet.interpreter import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TRACE_EVENTS,
    RunResult,
    run_function,
)
from minileet.tasks import Example, Task
from minileet.typecheck import typecheck_function


@dataclass(frozen=True)
class TestCaseResult:
    args: tuple[object, ...]
    expected: object
    run: RunResult

    @property
    def passed(self) -> bool:
        return self.run.ok and type(self.run.value) is type(self.expected) and self.run.value == self.expected


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
    def __init__(
        self,
        task: Task,
        hidden_count: int = 64,
        seed: int = 0,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
    ) -> None:
        self.task = task
        self.hidden_tests = task.hidden_tests(hidden_count, seed)
        self.max_steps = max_steps
        self.max_trace_events = max_trace_events

    def run_visible(self, program: Function) -> tuple[TestCaseResult, ...]:
        errors = self._program_errors(program)
        if errors:
            return self._failed_tests(self.task.visible_examples, errors)
        return self._run_tests(program, self.task.visible_examples)

    def evaluate(self, program: Function, visible: tuple[TestCaseResult, ...] | None = None) -> EvaluationResult:
        if visible is not None:
            type_error = _visible_type_error(visible)
            if type_error is not None:
                hidden = self._failed_tests(self.hidden_tests, (type_error,))
            else:
                hidden = self._run_tests(program, self.hidden_tests)
            return EvaluationResult(self.task.name, visible, hidden)

        errors = self._program_errors(program)
        if errors:
            visible = self._failed_tests(self.task.visible_examples, errors)
            hidden = self._failed_tests(self.hidden_tests, errors)
            return EvaluationResult(self.task.name, visible, hidden)
        visible = self._run_tests(program, self.task.visible_examples)
        hidden = self._run_tests(program, self.hidden_tests)
        return EvaluationResult(self.task.name, visible, hidden)

    def _program_errors(self, program: Function) -> tuple[str, ...]:
        return _signature_errors(program, self.task) + typecheck_function(program).errors

    def _run_tests(self, program: Function, tests: tuple[Example, ...]) -> tuple[TestCaseResult, ...]:
        results: list[TestCaseResult] = []
        for args, expected in tests:
            normalized_args = tuple(_normalize_arg(arg) for arg in args)
            run = run_function(
                program,
                normalized_args,
                max_steps=self.max_steps,
                max_trace_events=self.max_trace_events,
            )
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


def _visible_type_error(visible: tuple[TestCaseResult, ...]) -> str | None:
    if not visible:
        return None
    errors = [case.run.error for case in visible]
    if all(error is not None and error.startswith("type-error: ") for error in errors):
        return str(errors[0]).removeprefix("type-error: ")
    return None


def _signature_errors(program: Function, task: Task) -> tuple[str, ...]:
    errors: list[str] = []
    expected = task.signature
    if program.name != expected.name:
        errors.append(f"function name {program.name} does not match {expected.name}")
    if len(program.params) != len(expected.params):
        errors.append(f"arity {len(program.params)} does not match {len(expected.params)}")
    for index, (actual_param, expected_param) in enumerate(zip(program.params, expected.params)):
        if actual_param.typ != expected_param.typ:
            errors.append(f"param {index} type {actual_param.typ} does not match {expected_param.typ}")
    if program.return_type != expected.return_type:
        errors.append(f"return type {program.return_type} does not match task return type {expected.return_type}")
    return tuple(errors)
