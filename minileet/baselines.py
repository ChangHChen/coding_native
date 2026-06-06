from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from minileet.dsl import (
    BOOL,
    INT,
    LIST_INT,
    Assign,
    Binary,
    BoolLit,
    Compare,
    ForEach,
    Function,
    If,
    IntLit,
    Len,
    Param,
    Return,
    Var,
)
from minileet.env import EvaluationResult, MiniLeetEnv
from minileet.tasks import Task


@dataclass(frozen=True)
class SolveAttempt:
    program: Function
    result: EvaluationResult
    candidate_index: int


def solve_with_templates(task: Task, budget: int = 64, hidden_count: int = 64, seed: int = 0) -> SolveAttempt | None:
    env = MiniLeetEnv(task, hidden_count=hidden_count, seed=seed)
    for index, program in enumerate(template_candidates(task), start=1):
        if index > budget:
            break
        visible = env.run_visible(program)
        if visible and all(case.passed for case in visible):
            result = env.evaluate(program)
            return SolveAttempt(program, result, index)
    return None


def template_candidates(task: Task) -> tuple[Function, ...]:
    sig = task.signature
    params = sig.params
    name = sig.name
    candidates: list[Function] = []

    if sig.return_type == INT:
        candidates.extend(
            [
                Function(name, params, INT, (Return(IntLit(0)),)),
                Function(name, params, INT, (Return(Len(Var(_first_list_param(params)))),)),
            ]
        )
        candidates.extend(_int_list_templates(name, params))

    if sig.return_type == BOOL:
        candidates.extend(
            [
                Function(name, params, BOOL, (Return(BoolLit(False)),)),
                Function(name, params, BOOL, (Return(BoolLit(True)),)),
            ]
        )
        candidates.extend(_bool_list_templates(name, params))

    return tuple(_dedupe(candidates))


def _int_list_templates(name: str, params: tuple[Param, ...]) -> Iterable[Function]:
    xs = _first_list_param(params)
    x = "x"
    acc = "acc"
    yield Function(
        name,
        params,
        INT,
        (
            Assign(acc, IntLit(0)),
            ForEach(x, Var(xs), (Assign(acc, Binary(Var(acc), "+", Var(x))),)),
            Return(Var(acc)),
        ),
    )
    yield Function(
        name,
        params,
        INT,
        (
            Assign(acc, IntLit(0)),
            ForEach(
                x,
                Var(xs),
                (
                    If(
                        Compare(Var(x), ">", IntLit(0)),
                        (Assign(acc, Binary(Var(acc), "+", IntLit(1))),),
                    ),
                ),
            ),
            Return(Var(acc)),
        ),
    )
    yield Function(
        name,
        params,
        INT,
        (
            Assign(acc, IntLit(0)),
            ForEach(
                x,
                Var(xs),
                (
                    If(
                        Compare(Var(x), ">", Var(acc)),
                        (Assign(acc, Var(x)),),
                    ),
                ),
            ),
            Return(Var(acc)),
        ),
    )
    yield Function(
        name,
        params,
        INT,
        (
            Assign("seen", BoolLit(False)),
            Assign(acc, IntLit(0)),
            ForEach(
                x,
                Var(xs),
                (
                    If(
                        Compare(Var("seen"), "==", BoolLit(False)),
                        (Assign(acc, Var(x)), Assign("seen", BoolLit(True))),
                    ),
                    If(
                        Compare(Var(x), ">", Var(acc)),
                        (Assign(acc, Var(x)),),
                    ),
                ),
            ),
            Return(Var(acc)),
        ),
    )


def _bool_list_templates(name: str, params: tuple[Param, ...]) -> Iterable[Function]:
    xs = _first_list_param(params)
    int_params = [param.name for param in params if param.typ == INT]
    x = "x"
    yield Function(
        name,
        params,
        BOOL,
        (
            Assign("found", BoolLit(False)),
            ForEach(
                x,
                Var(xs),
                (
                    If(
                        Compare(Var(x), "==", Var(int_params[0])),
                        (Assign("found", BoolLit(True)),),
                    ),
                ),
            ),
            Return(Var("found")),
        ),
    ) if int_params else None
    yield Function(
        name,
        params,
        BOOL,
        (
            Assign("ok", BoolLit(True)),
            ForEach(
                x,
                Var(xs),
                (
                    If(
                        Compare(Var(x), "<", IntLit(0)),
                        (Assign("ok", BoolLit(False)),),
                    ),
                ),
            ),
            Return(Var("ok")),
        ),
    )


def _first_list_param(params: tuple[Param, ...]) -> str:
    for param in params:
        if param.typ == LIST_INT:
            return param.name
    raise ValueError("task has no list[int] parameter")


def _dedupe(programs: Iterable[Function | None]) -> Iterable[Function]:
    seen: set[str] = set()
    for program in programs:
        if program is None:
            continue
        key = program.render()
        if key in seen:
            continue
        seen.add(key)
        yield program
