from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minileet.dsl import (
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
    Return,
    Stmt,
    Var,
)


class MiniLeetRuntimeError(Exception):
    pass


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    data: dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    ok: bool
    value: Any = None
    error: str | None = None
    trace: tuple[TraceEvent, ...] = ()


class _Returned(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


def run_function(function: Function, args: tuple[Any, ...]) -> RunResult:
    trace: list[TraceEvent] = []
    if len(args) != len(function.params):
        return RunResult(False, error="wrong-arity")

    env = dict(zip(function.param_names, args))
    try:
        try:
            _exec_block(function.body, env, trace)
        except _Returned as returned:
            trace.append(TraceEvent("finish", {"value": returned.value}))
            return RunResult(True, value=returned.value, trace=tuple(trace))
        return RunResult(False, error="missing-return", trace=tuple(trace))
    except MiniLeetRuntimeError as exc:
        trace.append(TraceEvent("error", {"message": str(exc)}))
        return RunResult(False, error=str(exc), trace=tuple(trace))


def _exec_block(stmts: tuple[Stmt, ...], env: dict[str, Any], trace: list[TraceEvent]) -> None:
    for stmt in stmts:
        _exec_stmt(stmt, env, trace)


def _exec_stmt(stmt: Stmt, env: dict[str, Any], trace: list[TraceEvent]) -> None:
    if isinstance(stmt, Assign):
        value = _eval_expr(stmt.expr, env)
        env[stmt.name] = value
        trace.append(TraceEvent("assign", {"name": stmt.name, "value": value}))
        return

    if isinstance(stmt, Return):
        value = _eval_expr(stmt.expr, env)
        trace.append(TraceEvent("return", {"value": value}))
        raise _Returned(value)

    if isinstance(stmt, If):
        cond = _eval_expr(stmt.cond, env)
        if not isinstance(cond, bool):
            raise MiniLeetRuntimeError("if condition is not bool")
        trace.append(TraceEvent("branch", {"cond": stmt.cond.render(), "value": cond}))
        _exec_block(stmt.then_body if cond else stmt.else_body, env, trace)
        return

    if isinstance(stmt, ForEach):
        seq = _eval_expr(stmt.seq, env)
        if not isinstance(seq, list):
            raise MiniLeetRuntimeError("for target is not list")
        for index, item in enumerate(seq):
            env[stmt.item_name] = item
            trace.append(
                TraceEvent(
                    "loop",
                    {"item": stmt.item_name, "index": index, "value": item},
                )
            )
            _exec_block(stmt.body, env, trace)
        return

    raise MiniLeetRuntimeError(f"unknown statement {type(stmt).__name__}")


def _eval_expr(expr: Expr, env: dict[str, Any]) -> Any:
    if isinstance(expr, IntLit):
        return expr.value
    if isinstance(expr, BoolLit):
        return expr.value
    if isinstance(expr, Var):
        if expr.name not in env:
            raise MiniLeetRuntimeError(f"undefined variable {expr.name}")
        return env[expr.name]
    if isinstance(expr, Len):
        seq = _eval_expr(expr.seq, env)
        if not isinstance(seq, list):
            raise MiniLeetRuntimeError("len target is not list")
        return len(seq)
    if isinstance(expr, Index):
        seq = _eval_expr(expr.seq, env)
        index = _eval_expr(expr.index, env)
        if not isinstance(seq, list):
            raise MiniLeetRuntimeError("index target is not list")
        if not isinstance(index, int):
            raise MiniLeetRuntimeError("index is not int")
        try:
            return seq[index]
        except IndexError as exc:
            raise MiniLeetRuntimeError("index out of range") from exc
    if isinstance(expr, Binary):
        return _eval_binary(_eval_expr(expr.left, env), expr.op, _eval_expr(expr.right, env))
    if isinstance(expr, Compare):
        return _eval_compare(_eval_expr(expr.left, env), expr.op, _eval_expr(expr.right, env))

    raise MiniLeetRuntimeError(f"unknown expression {type(expr).__name__}")


def _eval_binary(left: Any, op: str, right: Any) -> Any:
    if not isinstance(left, int) or not isinstance(right, int):
        raise MiniLeetRuntimeError(f"binary operator {op} requires ints")
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "//":
        if right == 0:
            raise MiniLeetRuntimeError("division by zero")
        return left // right
    if op == "%":
        if right == 0:
            raise MiniLeetRuntimeError("modulo by zero")
        return left % right
    raise MiniLeetRuntimeError(f"unknown binary operator {op}")


def _eval_compare(left: Any, op: str, right: Any) -> bool:
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if not isinstance(left, int) or not isinstance(right, int):
        raise MiniLeetRuntimeError(f"comparison {op} requires ints")
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise MiniLeetRuntimeError(f"unknown comparison operator {op}")
