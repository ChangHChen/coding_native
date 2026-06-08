from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TypeName = Literal["int", "bool", "list[int]"]


@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class IntLit(Expr):
    value: int


@dataclass(frozen=True)
class BoolLit(Expr):
    value: bool


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Len(Expr):
    seq: Expr


@dataclass(frozen=True)
class Index(Expr):
    seq: Expr
    index: Expr


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Stmt:
    pass


@dataclass(frozen=True)
class Assign(Stmt):
    name: str
    expr: Expr


@dataclass(frozen=True)
class If(Stmt):
    cond: Expr
    then_body: tuple[Stmt, ...]
    else_body: tuple[Stmt, ...]


@dataclass(frozen=True)
class ForEach(Stmt):
    item: str
    seq: Expr
    body: tuple[Stmt, ...]


@dataclass(frozen=True)
class Return(Stmt):
    expr: Expr


@dataclass(frozen=True)
class Program:
    name: str
    args: tuple[tuple[str, TypeName], ...]
    return_type: TypeName
    body: tuple[Stmt, ...]
    source: str = "enumerated"


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    data: dict[str, Any]


VALID_ARITH = {"+", "-", "*"}
VALID_CMP = {"==", "!=", "<", "<=", ">", ">="}
VALID_BOOL = {"and", "or"}
VALID_OPS = VALID_ARITH | VALID_CMP | VALID_BOOL
