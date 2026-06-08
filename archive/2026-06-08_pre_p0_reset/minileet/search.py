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
    Index,
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
    include_task_conditioned_gold: bool = False,
) -> SearchAttempt | None:
    """Return the first candidate that passes visible tests.

    This simulates a visible-test-only agent and is not a solvability oracle.
    Use oracle_scan_budget for measurement-only hidden-test scanning.
    """
    return _solve_from_candidates(
        task,
        enumerate_candidates(task, include_task_conditioned_gold=include_task_conditioned_gold),
        budget=budget,
        hidden_count=hidden_count,
        seed=seed,
    )


def solve_with_random_search(
    task: Task,
    budget: int = 512,
    hidden_count: int = 64,
    seed: int = 0,
    include_task_conditioned_gold: bool = False,
) -> SearchAttempt | None:
    """Return the first shuffled candidate that passes visible tests."""
    candidates = list(enumerate_candidates(task, include_task_conditioned_gold=include_task_conditioned_gold))
    random.Random(seed).shuffle(candidates)
    return _solve_from_candidates(
        task,
        candidates,
        budget=budget,
        hidden_count=hidden_count,
        seed=seed,
    )


def oracle_scan_budget(
    task: Task,
    budget: int = 512,
    hidden_count: int = 64,
    seed: int = 0,
    include_task_conditioned_gold: bool = False,
) -> SearchAttempt | None:
    """Scan the full budget and return the best visible-passing candidate.

    This uses hidden labels to choose the result, so it is for evaluation and
    coverage audits only, not for agent behavior.
    """
    env = MiniLeetEnv(task, hidden_count=hidden_count, seed=seed)
    visible_passes_seen = 0
    best: SearchAttempt | None = None
    for index, program in enumerate(
        enumerate_candidates(task, include_task_conditioned_gold=include_task_conditioned_gold),
        start=1,
    ):
        if index > budget:
            break
        visible = env.run_visible(program)
        if not visible or not all(case.passed for case in visible):
            continue
        visible_passes_seen += 1
        result = env.evaluate(program, visible=visible)
        attempt = SearchAttempt(program, result, index, visible_passes_seen)
        if best is None or _attempt_is_better(attempt, best):
            best = attempt
    return best


def task_conditioned_renders(task: Task) -> frozenset[str]:
    """Rendered sources of the name-conditioned gold programs for this task.

    Used to mark candidate provenance at collection time: a candidate whose
    render appears in this set was injected from the task name, not reached
    by the generic enumerator.
    """
    if task.signature.return_type != INT:
        return frozenset()
    try:
        xs = _first_list_param(task.signature.params)
    except ValueError:
        return frozenset()
    return frozenset(
        program.render()
        for program in _task_conditioned_int_candidates(
            task.signature.name,
            task.signature.params,
            xs,
            task.name,
        )
    )


def enumerate_candidates(task: Task, include_task_conditioned_gold: bool = False) -> tuple[Function, ...]:
    params = task.signature.params
    name = task.signature.name
    xs = _first_list_param(params)

    if task.signature.return_type == INT:
        return tuple(_dedupe(_int_candidates(name, params, xs, task.name, include_task_conditioned_gold)))
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
            result = env.evaluate(program, visible=visible)
            return SearchAttempt(program, result, index, visible_passes_seen)
    return None


def _attempt_is_better(candidate: SearchAttempt, current: SearchAttempt) -> bool:
    return (
        int(candidate.result.solved),
        candidate.result.hidden_pass_rate,
        -candidate.candidate_index,
    ) > (
        int(current.result.solved),
        current.result.hidden_pass_rate,
        -current.candidate_index,
    )


def _int_candidates(
    name: str,
    params: tuple[Param, ...],
    xs: str,
    task_name: str = "",
    include_task_conditioned_gold: bool = False,
) -> Iterable[Function]:
    if include_task_conditioned_gold:
        yield from _task_conditioned_int_candidates(name, params, xs, task_name)

    list_params = _list_params(params)
    if len(list_params) >= 2:
        yield from _pair_int_candidates(name, params, list_params[0], list_params[1])

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
        yield _index_program(name, params, xs, pred, first=True)
        yield _index_program(name, params, xs, pred, first=False)
        yield _predicate_minmax_program(name, params, xs, pred, op=">")
        yield _predicate_minmax_program(name, params, xs, pred, op="<")

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
    list_params = _list_params(params)
    if len(list_params) >= 2:
        yield from _pair_bool_candidates(name, params, list_params[0], list_params[1])

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


def _task_conditioned_int_candidates(
    name: str,
    params: tuple[Param, ...],
    xs: str,
    task_name: str,
) -> Iterable[Function]:
    if not task_name:
        return
    core_name = _strip_profile_suffix(task_name)
    if core_name.startswith("nested_sum_") and "_after_" in core_name:
        left, gate_name = core_name.removeprefix("nested_sum_").split("_after_", maxsplit=1)
        value_pred = _predicate_expr_by_name(left)
        gate_pred = _predicate_expr_by_name(gate_name)
        if value_pred is not None and gate_pred is not None:
            yield _nested_sum_after_program(name, params, xs, value_pred, gate_pred)
        return
    if core_name.startswith("nested_count_") and "_before_" in core_name:
        left, stop_name = core_name.removeprefix("nested_count_").split("_before_", maxsplit=1)
        value_pred = _predicate_expr_by_name(left)
        stop_pred = _predicate_expr_by_name(stop_name)
        if value_pred is not None and stop_pred is not None:
            yield _nested_count_before_program(name, params, xs, value_pred, stop_pred)
        return
    if core_name.startswith("twostat_sum_") and "_minus_count_" in core_name:
        left, count_name = core_name.removeprefix("twostat_sum_").split("_minus_count_", maxsplit=1)
        sum_pred = _predicate_expr_by_name(left)
        count_pred = _predicate_expr_by_name(count_name)
        if sum_pred is not None and count_pred is not None:
            yield _twostat_sum_minus_count_program(name, params, xs, sum_pred, count_pred)


def _predicate_expr_by_name(predicate_name: str) -> Expr | None:
    x = Var("x")
    if predicate_name == "even":
        return Compare(Binary(x, "%", IntLit(2)), "==", IntLit(0))
    if predicate_name == "odd":
        return Compare(Binary(x, "%", IntLit(2)), "!=", IntLit(0))
    for prefix, op in (("gt_", ">"), ("ge_", ">="), ("lt_", "<"), ("le_", "<="), ("eq_", "=="), ("ne_", "!=")):
        if predicate_name.startswith(prefix):
            value = _parse_const_token(predicate_name.removeprefix(prefix))
            if value is None:
                return None
            return Compare(x, op, IntLit(value))
    return None


def _parse_const_token(token: str) -> int | None:
    if token.startswith("neg") and token[3:].isdigit():
        return -int(token[3:])
    if token.isdigit():
        return int(token)
    return None


def _strip_profile_suffix(task_name: str) -> str:
    parts = task_name.split("_")
    if parts and (parts[-1].startswith("shtr") or parts[-1].startswith("shdev")):
        return "_".join(parts[:-1])
    return task_name


def _nested_sum_after_program(name: str, params: tuple[Param, ...], xs: str, value_pred: Expr, gate_pred: Expr) -> Function:
    return Function(
        name,
        params,
        INT,
        (
            Assign("acc", IntLit(0)),
            Assign("enabled", BoolLit(False)),
            ForEach(
                "x",
                Var(xs),
                (
                    If(gate_pred, (Assign("enabled", BoolLit(True)),)),
                    If(
                        Compare(Var("enabled"), "==", BoolLit(True)),
                        (If(value_pred, (Assign("acc", Binary(Var("acc"), "+", Var("x"))),)),),
                    ),
                ),
            ),
            Return(Var("acc")),
        ),
    )


def _nested_count_before_program(name: str, params: tuple[Param, ...], xs: str, value_pred: Expr, stop_pred: Expr) -> Function:
    return Function(
        name,
        params,
        INT,
        (
            Assign("acc", IntLit(0)),
            Assign("stopped", BoolLit(False)),
            ForEach(
                "x",
                Var(xs),
                (
                    If(
                        Compare(Var("stopped"), "==", BoolLit(False)),
                        (
                            If(
                                stop_pred,
                                (Assign("stopped", BoolLit(True)),),
                                (If(value_pred, (Assign("acc", Binary(Var("acc"), "+", IntLit(1))),)),),
                            ),
                        ),
                    ),
                ),
            ),
            Return(Var("acc")),
        ),
    )


def _twostat_sum_minus_count_program(name: str, params: tuple[Param, ...], xs: str, sum_pred: Expr, count_pred: Expr) -> Function:
    return Function(
        name,
        params,
        INT,
        (
            Assign("total", IntLit(0)),
            Assign("count", IntLit(0)),
            ForEach(
                "x",
                Var(xs),
                (
                    If(sum_pred, (Assign("total", Binary(Var("total"), "+", Var("x"))),)),
                    If(count_pred, (Assign("count", Binary(Var("count"), "+", IntLit(1))),)),
                ),
            ),
            Return(Binary(Var("total"), "-", Var("count"))),
        ),
    )


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


def _index_program(name: str, params: tuple[Param, ...], xs: str, pred: Expr, first: bool) -> Function:
    body: tuple[Stmt, ...]
    if first:
        body = (
            Assign("i", IntLit(0)),
            Assign("found", BoolLit(False)),
            Assign("acc", IntLit(-1)),
            ForEach(
                "x",
                Var(xs),
                (
                    If(
                        Compare(Var("found"), "==", BoolLit(False)),
                        (If(pred, (Assign("acc", Var("i")), Assign("found", BoolLit(True)))),),
                    ),
                    Assign("i", Binary(Var("i"), "+", IntLit(1))),
                ),
            ),
            Return(Var("acc")),
        )
    else:
        body = (
            Assign("i", IntLit(0)),
            Assign("acc", IntLit(-1)),
            ForEach(
                "x",
                Var(xs),
                (
                    If(pred, (Assign("acc", Var("i")),)),
                    Assign("i", Binary(Var("i"), "+", IntLit(1))),
                ),
            ),
            Return(Var("acc")),
        )
    return Function(name, params, INT, body)


def _predicate_minmax_program(name: str, params: tuple[Param, ...], xs: str, pred: Expr, op: str) -> Function:
    return Function(
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
                        pred,
                        (
                            If(
                                Compare(Var("seen"), "==", BoolLit(False)),
                                (Assign("acc", Var("x")), Assign("seen", BoolLit(True))),
                            ),
                            If(Compare(Var("x"), op, Var("acc")), (Assign("acc", Var("x")),)),
                        ),
                    ),
                ),
            ),
            Return(Var("acc")),
        ),
    )


def _pair_int_candidates(name: str, params: tuple[Param, ...], xs: str, ys: str) -> Iterable[Function]:
    yield Function(
        name,
        params,
        INT,
        (
            Assign("i", IntLit(0)),
            Assign("acc", IntLit(0)),
            ForEach(
                "x",
                Var(xs),
                (
                    If(
                        Compare(Var("i"), "<", Len(Var(ys))),
                        (
                            Assign("y", Index(Var(ys), Var("i"))),
                            If(
                                Compare(Var("x"), ">", Var("y")),
                                (Assign("acc", Binary(Var("acc"), "+", Binary(Var("x"), "-", Var("y")))),),
                                (Assign("acc", Binary(Var("acc"), "+", Binary(Var("y"), "-", Var("x")))),),
                            ),
                        ),
                    ),
                    Assign("i", Binary(Var("i"), "+", IntLit(1))),
                ),
            ),
            Return(Var("acc")),
        ),
    )


def _pair_bool_candidates(name: str, params: tuple[Param, ...], xs: str, ys: str) -> Iterable[Function]:
    for op in ("==", "!=", ">", ">=", "<", "<="):
        yield Function(
            name,
            params,
            BOOL,
            (
                Assign("left", IntLit(0)),
                ForEach("x", Var(xs), (Assign("left", Binary(Var("left"), "+", Var("x"))),)),
                Assign("right", IntLit(0)),
                ForEach("y", Var(ys), (Assign("right", Binary(Var("right"), "+", Var("y"))),)),
                Return(Compare(Var("left"), op, Var("right"))),
            ),
        )
        yield Function(
            name,
            params,
            BOOL,
            (Return(Compare(Len(Var(xs)), op, Len(Var(ys)))),),
        )
        yield Function(
            name,
            params,
            BOOL,
            (
                Assign("left", IntLit(0)),
                ForEach(
                    "x",
                    Var(xs),
                    (If(Compare(Var("x"), ">", IntLit(0)), (Assign("left", Binary(Var("left"), "+", IntLit(1))),)),),
                ),
                Assign("right", IntLit(0)),
                ForEach(
                    "y",
                    Var(ys),
                    (If(Compare(Var("y"), ">", IntLit(0)), (Assign("right", Binary(Var("right"), "+", IntLit(1))),)),),
                ),
                Return(Compare(Var("left"), op, Var("right"))),
            ),
        )
        yield Function(
            name,
            params,
            BOOL,
            (
                Assign("left_seen", BoolLit(False)),
                Assign("left", IntLit(0)),
                ForEach(
                    "x",
                    Var(xs),
                    (
                        If(
                            Compare(Var("left_seen"), "==", BoolLit(False)),
                            (Assign("left", Var("x")), Assign("left_seen", BoolLit(True))),
                        ),
                    ),
                ),
                Assign("right_seen", BoolLit(False)),
                Assign("right", IntLit(0)),
                ForEach(
                    "y",
                    Var(ys),
                    (
                        If(
                            Compare(Var("right_seen"), "==", BoolLit(False)),
                            (Assign("right", Var("y")), Assign("right_seen", BoolLit(True))),
                        ),
                    ),
                ),
                Return(Compare(Var("left"), op, Var("right"))),
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


def _list_params(params: tuple[Param, ...]) -> tuple[str, ...]:
    return tuple(param.name for param in params if param.typ == LIST_INT)


def _dedupe(programs: Iterable[Function]) -> Iterable[Function]:
    seen: set[str] = set()
    for program in programs:
        key = program.render()
        if key in seen:
            continue
        seen.add(key)
        yield program
