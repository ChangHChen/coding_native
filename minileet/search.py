from __future__ import annotations

import random
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
    Expr,
    ForEach,
    Function,
    If,
    IntLit,
    Len,
    Param,
    Return,
    Stmt,
    Var,
)
from minileet.env import EvaluationResult, MiniLeetEnv
from minileet.tasks import Task


@dataclass(frozen=True)
class SearchAttempt:
    program: Function
    result: EvaluationResult
    candidate_index: int
    visible_passes_seen: int


def solve_with_enumeration(
    task: Task,
    budget: int = 512,
    hidden_count: int = 64,
    seed: int = 0,
) -> SearchAttempt | None:
    return _solve_from_candidates(
        task,
        enumerate_candidates(task),
        budget=budget,
        hidden_count=hidden_count,
        seed=seed,
    )


def solve_with_random_search(
    task: Task,
    budget: int = 512,
    hidden_count: int = 64,
    seed: int = 0,
) -> SearchAttempt | None:
    candidates = list(enumerate_candidates(task))
    random.Random(seed).shuffle(candidates)
    return _solve_from_candidates(
        task,
        candidates,
        budget=budget,
        hidden_count=hidden_count,
        seed=seed,
    )


def enumerate_candidates(task: Task) -> tuple[Function, ...]:
    params = task.signature.params
    name = task.signature.name
    xs = _first_list_param(params)

    if task.signature.return_type == INT:
        return tuple(_dedupe(_int_candidates(name, params, xs)))
    if task.signature.return_type == BOOL:
        return tuple(_dedupe(_bool_candidates(name, params, xs)))
    return ()


def _solve_from_candidates(
    task: Task,
    candidates: Iterable[Function],
    budget: int,
    hidden_count: int,
    seed: int,
) -> SearchAttempt | None:
    env = MiniLeetEnv(task, hidden_count=hidden_count, seed=seed)
    visible_passes_seen = 0
    for index, program in enumerate(candidates, start=1):
        if index > budget:
            break
        visible = env.run_visible(program)
        if visible and all(case.passed for case in visible):
            visible_passes_seen += 1
            result = env.evaluate(program)
            return SearchAttempt(program, result, index, visible_passes_seen)
    return None


def _int_candidates(name: str, params: tuple[Param, ...], xs: str) -> Iterable[Function]:
    for expr in _return_int_exprs(params, xs):
        yield Function(name, params, INT, (Return(expr),))

    for init in _int_inits():
        for update in _unconditional_int_updates():
            yield Function(
                name,
                params,
                INT,
                (
                    Assign("acc", init),
                    ForEach("x", Var(xs), (Assign("acc", update),)),
                    Return(Var("acc")),
                ),
            )

    for pred in _predicate_exprs(params):
        for update in _conditional_int_updates():
            yield Function(
                name,
                params,
                INT,
                (
                    Assign("acc", IntLit(0)),
                    ForEach("x", Var(xs), (If(pred, (Assign("acc", update),)),)),
                    Return(Var("acc")),
                ),
            )
        yield _first_or_zero_program(name, params, xs, pred)

    for init in _int_inits():
        for op in (">", "<"):
            yield Function(
                name,
                params,
                INT,
                (
                    Assign("acc", init),
                    ForEach(
                        "x",
                        Var(xs),
                        (If(Compare(Var("x"), op, Var("acc")), (Assign("acc", Var("x")),)),),
                    ),
                    Return(Var("acc")),
                ),
            )

    yield Function(
        name,
        params,
        INT,
        (
            Assign("seen", BoolLit(False)),
            Assign("acc", IntLit(0)),
            ForEach(
                "x",
                Var(xs),
                (
                    If(
                        Compare(Var("seen"), "==", BoolLit(False)),
                        (Assign("acc", Var("x")), Assign("seen", BoolLit(True))),
                    ),
                    If(Compare(Var("x"), ">", Var("acc")), (Assign("acc", Var("x")),)),
                ),
            ),
            Return(Var("acc")),
        ),
    )


def _bool_candidates(name: str, params: tuple[Param, ...], xs: str) -> Iterable[Function]:
    yield Function(name, params, BOOL, (Return(BoolLit(False)),))
    yield Function(name, params, BOOL, (Return(BoolLit(True)),))

    for pred in _predicate_exprs(params):
        yield Function(
            name,
            params,
            BOOL,
            (
                Assign("found", BoolLit(False)),
                ForEach("x", Var(xs), (If(pred, (Assign("found", BoolLit(True)),)),)),
                Return(Var("found")),
            ),
        )
        yield Function(
            name,
            params,
            BOOL,
            (
                Assign("ok", BoolLit(True)),
                ForEach(
                    "x",
                    Var(xs),
                    (If(Compare(pred, "==", BoolLit(False)), (Assign("ok", BoolLit(False)),)),),
                ),
                Return(Var("ok")),
            ),
        )

    for agg_name, body in _aggregate_bodies(xs):
        for cmp_expr in _aggregate_comparisons(agg_name, params):
            yield Function(name, params, BOOL, body + (Return(cmp_expr),))


def _return_int_exprs(params: tuple[Param, ...], xs: str) -> tuple[Expr, ...]:
    exprs: list[Expr] = [IntLit(-1), IntLit(0), IntLit(1), Len(Var(xs))]
    exprs.extend(Var(param.name) for param in params if param.typ == INT)
    return tuple(exprs)


def _int_inits() -> tuple[Expr, ...]:
    return (IntLit(-1), IntLit(0), IntLit(1))


def _unconditional_int_updates() -> tuple[Expr, ...]:
    return (
        Binary(Var("acc"), "+", Var("x")),
        Binary(Var("acc"), "-", Var("x")),
        Binary(Var("acc"), "+", IntLit(1)),
        Binary(Var("acc"), "-", IntLit(1)),
    )


def _conditional_int_updates() -> tuple[Expr, ...]:
    return (
        Binary(Var("acc"), "+", IntLit(1)),
        Binary(Var("acc"), "+", Var("x")),
        Var("x"),
    )


def _predicate_exprs(params: tuple[Param, ...]) -> tuple[Expr, ...]:
    x = Var("x")
    predicates: list[Expr] = [
        Compare(Binary(x, "%", IntLit(2)), "==", IntLit(0)),
        Compare(Binary(x, "%", IntLit(2)), "!=", IntLit(0)),
    ]
    for const in (-3, -2, -1, 0, 1, 2, 3):
        c = IntLit(const)
        predicates.extend(
            [
                Compare(x, ">", c),
                Compare(x, ">=", c),
                Compare(x, "<", c),
                Compare(x, "<=", c),
                Compare(x, "==", c),
                Compare(x, "!=", c),
            ]
        )
    for param in params:
        if param.typ != INT:
            continue
        k = Var(param.name)
        predicates.extend(
            [
                Compare(x, "==", k),
                Compare(x, "!=", k),
                Compare(x, ">", k),
                Compare(x, ">=", k),
                Compare(x, "<", k),
                Compare(x, "<=", k),
            ]
        )
    return tuple(predicates)


def _first_or_zero_program(name: str, params: tuple[Param, ...], xs: str, pred: Expr) -> Function:
    return Function(
        name,
        params,
        INT,
        (
            Assign("found", BoolLit(False)),
            Assign("acc", IntLit(0)),
            ForEach(
                "x",
                Var(xs),
                (
                    If(
                        Compare(Var("found"), "==", BoolLit(False)),
                        (If(pred, (Assign("acc", Var("x")), Assign("found", BoolLit(True)))),),
                    ),
                ),
            ),
            Return(Var("acc")),
        ),
    )


def _aggregate_bodies(xs: str) -> tuple[tuple[str, tuple[Stmt, ...]], ...]:
    return (
        (
            "sum",
            (
                Assign("acc", IntLit(0)),
                ForEach("x", Var(xs), (Assign("acc", Binary(Var("acc"), "+", Var("x"))),)),
            ),
        ),
        (
            "count_positive",
            (
                Assign("acc", IntLit(0)),
                ForEach(
                    "x",
                    Var(xs),
                    (
                        If(
                            Compare(Var("x"), ">", IntLit(0)),
                            (Assign("acc", Binary(Var("acc"), "+", IntLit(1))),),
                        ),
                    ),
                ),
            ),
        ),
        (
            "count_negative",
            (
                Assign("acc", IntLit(0)),
                ForEach(
                    "x",
                    Var(xs),
                    (
                        If(
                            Compare(Var("x"), "<", IntLit(0)),
                            (Assign("acc", Binary(Var("acc"), "+", IntLit(1))),),
                        ),
                    ),
                ),
            ),
        ),
        ("length", (Assign("acc", Len(Var(xs))),)),
    )


def _aggregate_comparisons(acc_name: str, params: tuple[Param, ...]) -> tuple[Expr, ...]:
    del acc_name
    right_exprs: list[Expr] = [IntLit(-1), IntLit(0), IntLit(1)]
    right_exprs.extend(Var(param.name) for param in params if param.typ == INT)
    return tuple(
        Compare(Var("acc"), op, right)
        for right in right_exprs
        for op in ("==", "!=", ">", ">=", "<", "<=")
    )


def _first_list_param(params: tuple[Param, ...]) -> str:
    for param in params:
        if param.typ == LIST_INT:
            return param.name
    raise ValueError("task has no list[int] parameter")


def _dedupe(programs: Iterable[Function]) -> Iterable[Function]:
    seen: set[str] = set()
    for program in programs:
        key = program.render()
        if key in seen:
            continue
        seen.add(key)
        yield program
