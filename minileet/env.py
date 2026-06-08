from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minileet.dsl import Program, TypeName
from minileet.interpreter import ExecResult, run, value_type


@dataclass(frozen=True)
class TestCase:
    args: dict[str, Any]
    expected: Any


@dataclass(frozen=True)
class Task:
    name: str
    signature: tuple[tuple[str, TypeName], ...]
    return_type: TypeName
    visible: tuple[TestCase, ...]
    hidden: tuple[TestCase, ...]


@dataclass(frozen=True)
class CaseGrade:
    passed: bool
    expected: Any
    result: ExecResult


@dataclass(frozen=True)
class EvalResult:
    visible: tuple[CaseGrade, ...]
    hidden: tuple[CaseGrade, ...]

    @property
    def solved(self) -> bool:
        return all(case.passed for case in self.visible + self.hidden)

    @property
    def visible_pass_rate(self) -> float:
        return sum(c.passed for c in self.visible) / max(1, len(self.visible))

    @property
    def hidden_pass_rate(self) -> float:
        return sum(c.passed for c in self.hidden) / max(1, len(self.hidden))


def canonical_args(args: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, repr(value)) for key, value in args.items()))


def validate_task(task: Task, entropy_floor: int = 2) -> None:
    visible_inputs = {canonical_args(case.args) for case in task.visible}
    hidden_inputs = {canonical_args(case.args) for case in task.hidden}
    overlap = visible_inputs & hidden_inputs
    if overlap:
        raise ValueError("visible/hidden input overlap")
    hidden_outputs = {repr(case.expected) for case in task.hidden}
    if len(task.hidden) > 1 and len(hidden_outputs) < entropy_floor:
        raise ValueError("degenerate hidden outputs")


def grade_case(program: Program, case: TestCase) -> CaseGrade:
    result = run(program, case.args)
    passed = (
        result.error is None
        and value_type(result.output) == program.return_type
        and value_type(case.expected) == program.return_type
        and result.output == case.expected
    )
    return CaseGrade(passed, case.expected, result)


def evaluate(task: Task, program: Program) -> EvalResult:
    if program.args != task.signature or program.return_type != task.return_type:
        raise ValueError("program signature does not match task")
    visible = tuple(grade_case(program, case) for case in task.visible)
    hidden = tuple(grade_case(program, case) for case in task.hidden)
    return EvalResult(visible, hidden)
