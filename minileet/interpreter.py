from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minileet.dsl import (
    Assign,
    BinOp,
    BoolLit,
    Expr,
    ForEach,
    If,
    Index,
    IntLit,
    Len,
    Program,
    Return,
    Stmt,
    TraceEvent,
    Var,
)
from minileet.typecheck import typecheck


class RuntimeErrorDSL(Exception):
    pass


class _Returned(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


@dataclass(frozen=True)
class ExecResult:
    output: Any | None
    error: str | None
    trace: tuple[TraceEvent, ...]
    steps: int
    trace_truncated: bool = False


def value_type(value: Any) -> str:
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if isinstance(value, list) and all(type(v) is int for v in value):
        return "list[int]"
    return "invalid"


class Interpreter:
    def __init__(self, step_budget: int = 1000, trace_limit: int = 1000) -> None:
        self.step_budget = step_budget
        self.trace_limit = trace_limit
        self.steps = 0
        self.trace: list[TraceEvent] = []
        self.trace_truncated = False

    def tick(self) -> None:
        self.steps += 1
        if self.steps > self.step_budget:
            raise RuntimeErrorDSL("step-budget-exceeded")

    def emit(self, kind: str, **data: Any) -> None:
        if len(self.trace) < self.trace_limit:
            self.trace.append(TraceEvent(kind, data))
        else:
            self.trace_truncated = True

    def eval_expr(self, expr: Expr, env: dict[str, Any]) -> Any:
        self.tick()
        if isinstance(expr, IntLit):
            return expr.value
        if isinstance(expr, BoolLit):
            return expr.value
        if isinstance(expr, Var):
            if expr.name not in env:
                raise RuntimeErrorDSL(f"unknown variable {expr.name}")
            return env[expr.name]
        if isinstance(expr, Len):
            seq = self.eval_expr(expr.seq, env)
            if not isinstance(seq, list):
                raise RuntimeErrorDSL("len expects list")
            return len(seq)
        if isinstance(expr, Index):
            seq = self.eval_expr(expr.seq, env)
            index = self.eval_expr(expr.index, env)
            if not isinstance(seq, list) or type(index) is not int:
                raise RuntimeErrorDSL("bad index")
            try:
                return seq[index]
            except IndexError as exc:
                raise RuntimeErrorDSL("index-out-of-range") from exc
        if isinstance(expr, BinOp):
            left = self.eval_expr(expr.left, env)
            right = self.eval_expr(expr.right, env)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "==":
                return left == right
            if expr.op == "!=":
                return left != right
            if expr.op == "<":
                return left < right
            if expr.op == "<=":
                return left <= right
            if expr.op == ">":
                return left > right
            if expr.op == ">=":
                return left >= right
            if expr.op == "and":
                return left and right
            if expr.op == "or":
                return left or right
        raise RuntimeErrorDSL(f"unknown expression {type(expr).__name__}")

    def exec_stmt(self, stmt: Stmt, env: dict[str, Any]) -> None:
        self.tick()
        if isinstance(stmt, Assign):
            value = self.eval_expr(stmt.expr, env)
            env[stmt.name] = value
            self.emit("assign", name=stmt.name, value=value)
            return
        if isinstance(stmt, Return):
            value = self.eval_expr(stmt.expr, env)
            self.emit("return", value=value)
            raise _Returned(value)
        if isinstance(stmt, If):
            cond = self.eval_expr(stmt.cond, env)
            if type(cond) is not bool:
                raise RuntimeErrorDSL("if condition not bool")
            self.emit("branch", cond=cond)
            body = stmt.then_body if cond else stmt.else_body
            for inner in body:
                self.exec_stmt(inner, env)
            return
        if isinstance(stmt, ForEach):
            seq = self.eval_expr(stmt.seq, env)
            if not isinstance(seq, list):
                raise RuntimeErrorDSL("for-each expects list")
            old = env.get(stmt.item, None)
            had_old = stmt.item in env
            for i, item in enumerate(seq):
                env[stmt.item] = item
                self.emit("loop-step", item=stmt.item, index=i, value=item)
                for inner in stmt.body:
                    self.exec_stmt(inner, env)
            if had_old:
                env[stmt.item] = old
            else:
                env.pop(stmt.item, None)
            return
        raise RuntimeErrorDSL(f"unknown statement {type(stmt).__name__}")


def run(program: Program, args: dict[str, Any], step_budget: int = 1000, trace_limit: int = 1000) -> ExecResult:
    try:
        typecheck(program)
        interp = Interpreter(step_budget=step_budget, trace_limit=trace_limit)
        env = dict(args)
        for stmt in program.body:
            interp.exec_stmt(stmt, env)
        raise RuntimeErrorDSL("missing-return")
    except _Returned as ret:
        return ExecResult(ret.value, None, tuple(interp.trace), interp.steps, interp.trace_truncated)
    except Exception as exc:
        trace = tuple(interp.trace) if "interp" in locals() else ()
        steps = interp.steps if "interp" in locals() else 0
        truncated = interp.trace_truncated if "interp" in locals() else False
        return ExecResult(None, str(exc), trace, steps, truncated)
