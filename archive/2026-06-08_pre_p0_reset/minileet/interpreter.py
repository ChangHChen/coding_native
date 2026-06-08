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


DEFAULT_MAX_STEPS = 100_000
DEFAULT_MAX_TRACE_EVENTS = 10_000


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
    steps: int = 0
    trace_truncated: bool = False


class _Returned(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _RunState:
    """Step budget and bounded trace shared across one program execution.

    The step budget guards against unbounded execution once programs are no
    longer guaranteed to be finite-loop templates (e.g. agent-written code).
    The trace cap bounds memory and downstream serialization; terminal events
    (finish/error) are always recorded even when the cap is hit.
    """

    __slots__ = ("steps", "max_steps", "events", "max_trace_events", "trace_truncated")

    def __init__(self, max_steps: int, max_trace_events: int) -> None:
        self.steps = 0
        self.max_steps = max_steps
        self.events: list[TraceEvent] = []
        self.max_trace_events = max_trace_events
        self.trace_truncated = False

    def tick(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise MiniLeetRuntimeError("step-budget-exceeded")

    def record(self, event: TraceEvent) -> None:
        if len(self.events) >= self.max_trace_events:
            self.trace_truncated = True
            return
        self.events.append(event)

    def record_terminal(self, event: TraceEvent) -> None:
        self.events.append(event)


def run_function(
    function: Function,
    args: tuple[Any, ...],
    max_steps: int = DEFAULT_MAX_STEPS,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
) -> RunResult:
    state = _RunState(max_steps, max_trace_events)
    if len(args) != len(function.params):
        return RunResult(False, error="wrong-arity")

    env = dict(zip(function.param_names, args))
    try:
        try:
            _exec_block(function.body, env, state)
        except _Returned as returned:
            state.record_terminal(TraceEvent("finish", {"value": returned.value}))
            return RunResult(
                True,
                value=returned.value,
                trace=tuple(state.events),
                steps=state.steps,
                trace_truncated=state.trace_truncated,
            )
        return RunResult(
            False,
            error="missing-return",
            trace=tuple(state.events),
            steps=state.steps,
            trace_truncated=state.trace_truncated,
        )
    except MiniLeetRuntimeError as exc:
        state.record_terminal(TraceEvent("error", {"message": str(exc)}))
        return RunResult(
            False,
            error=str(exc),
            trace=tuple(state.events),
            steps=state.steps,
            trace_truncated=state.trace_truncated,
        )


def _exec_block(stmts: tuple[Stmt, ...], env: dict[str, Any], state: _RunState) -> None:
    for stmt in stmts:
        _exec_stmt(stmt, env, state)


def _exec_stmt(stmt: Stmt, env: dict[str, Any], state: _RunState) -> None:
    state.tick()
    if isinstance(stmt, Assign):
        value = _eval_expr(stmt.expr, env, state)
        env[stmt.name] = value
        state.record(TraceEvent("assign", {"name": stmt.name, "value": value}))
        return

    if isinstance(stmt, Return):
        value = _eval_expr(stmt.expr, env, state)
        state.record(TraceEvent("return", {"value": value}))
        raise _Returned(value)

    if isinstance(stmt, If):
        cond = _eval_expr(stmt.cond, env, state)
        if not _is_bool(cond):
            raise MiniLeetRuntimeError("if condition is not bool")
        state.record(TraceEvent("branch", {"cond": stmt.cond.render(), "value": cond}))
        _exec_block(stmt.then_body if cond else stmt.else_body, env, state)
        return

    if isinstance(stmt, ForEach):
        seq = _eval_expr(stmt.seq, env, state)
        if not _is_list(seq):
            raise MiniLeetRuntimeError("for target is not list")
        had_item = stmt.item_name in env
        previous_item = env.get(stmt.item_name)
        for index, item in enumerate(seq):
            env[stmt.item_name] = item
            state.record(
                TraceEvent(
                    "loop",
                    {"item": stmt.item_name, "index": index, "value": item},
                )
            )
            _exec_block(stmt.body, env, state)
        if had_item:
            env[stmt.item_name] = previous_item
        else:
            env.pop(stmt.item_name, None)
        return

    raise MiniLeetRuntimeError(f"unknown statement {type(stmt).__name__}")


def _eval_expr(expr: Expr, env: dict[str, Any], state: _RunState) -> Any:
    state.tick()
    if isinstance(expr, IntLit):
        return expr.value
    if isinstance(expr, BoolLit):
        return expr.value
    if isinstance(expr, Var):
        if expr.name not in env:
            raise MiniLeetRuntimeError(f"undefined variable {expr.name}")
        return env[expr.name]
    if isinstance(expr, Len):
        seq = _eval_expr(expr.seq, env, state)
        if not _is_list(seq):
            raise MiniLeetRuntimeError("len target is not list")
        return len(seq)
    if isinstance(expr, Index):
        seq = _eval_expr(expr.seq, env, state)
        index = _eval_expr(expr.index, env, state)
        if not _is_list(seq):
            raise MiniLeetRuntimeError("index target is not list")
        if not _is_int(index):
            raise MiniLeetRuntimeError("index is not int")
        try:
            return seq[index]
        except IndexError as exc:
            raise MiniLeetRuntimeError("index out of range") from exc
    if isinstance(expr, Binary):
        return _eval_binary(_eval_expr(expr.left, env, state), expr.op, _eval_expr(expr.right, env, state))
    if isinstance(expr, Compare):
        return _eval_compare(_eval_expr(expr.left, env, state), expr.op, _eval_expr(expr.right, env, state))

    raise MiniLeetRuntimeError(f"unknown expression {type(expr).__name__}")


def _eval_binary(left: Any, op: str, right: Any) -> Any:
    if not _is_int(left) or not _is_int(right):
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
        return type(left) is type(right) and left == right
    if op == "!=":
        return type(left) is not type(right) or left != right
    if not _is_int(left) or not _is_int(right):
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


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _is_list(value: Any) -> bool:
    return type(value) is list
