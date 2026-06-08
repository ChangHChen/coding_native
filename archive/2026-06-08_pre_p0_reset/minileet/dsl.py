from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Type:
    name: str

    def __str__(self) -> str:
        return self.name


INT = Type("int")
BOOL = Type("bool")
LIST_INT = Type("list[int]")


class Expr:
    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class IntLit(Expr):
    value: int

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class BoolLit(Expr):
    value: bool

    def render(self) -> str:
        return "true" if self.value else "false"


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def render(self) -> str:
        return self.name


@dataclass(frozen=True)
class Len(Expr):
    seq: Expr

    def render(self) -> str:
        return f"len({self.seq.render()})"


@dataclass(frozen=True)
class Index(Expr):
    seq: Expr
    index: Expr

    def render(self) -> str:
        return f"{self.seq.render()}[{self.index.render()}]"


@dataclass(frozen=True)
class Binary(Expr):
    left: Expr
    op: str
    right: Expr

    def render(self) -> str:
        return f"({self.left.render()} {self.op} {self.right.render()})"


@dataclass(frozen=True)
class Compare(Expr):
    left: Expr
    op: str
    right: Expr

    def render(self) -> str:
        return f"({self.left.render()} {self.op} {self.right.render()})"


class Stmt:
    def render(self, indent: int = 0) -> str:
        raise NotImplementedError


def _pad(indent: int) -> str:
    return " " * indent


def _render_block(stmts: Sequence[Stmt], indent: int) -> str:
    if not stmts:
        return _pad(indent) + "pass"
    return "\n".join(stmt.render(indent) for stmt in stmts)


@dataclass(frozen=True)
class Assign(Stmt):
    name: str
    expr: Expr

    def render(self, indent: int = 0) -> str:
        return f"{_pad(indent)}{self.name} = {self.expr.render()}"


@dataclass(frozen=True)
class Return(Stmt):
    expr: Expr

    def render(self, indent: int = 0) -> str:
        return f"{_pad(indent)}return {self.expr.render()}"


@dataclass(frozen=True)
class If(Stmt):
    cond: Expr
    then_body: tuple[Stmt, ...]
    else_body: tuple[Stmt, ...] = ()

    def render(self, indent: int = 0) -> str:
        text = f"{_pad(indent)}if {self.cond.render()}:\n"
        text += _render_block(self.then_body, indent + 2)
        if self.else_body:
            text += f"\n{_pad(indent)}else:\n"
            text += _render_block(self.else_body, indent + 2)
        return text


@dataclass(frozen=True)
class ForEach(Stmt):
    item_name: str
    seq: Expr
    body: tuple[Stmt, ...]

    def render(self, indent: int = 0) -> str:
        return (
            f"{_pad(indent)}for {self.item_name} in {self.seq.render()}:\n"
            f"{_render_block(self.body, indent + 2)}"
        )


@dataclass(frozen=True)
class Param:
    name: str
    typ: Type


@dataclass(frozen=True)
class Function:
    name: str
    params: tuple[Param, ...]
    return_type: Type
    body: tuple[Stmt, ...]

    def render(self) -> str:
        params = ", ".join(f"{param.name}: {param.typ}" for param in self.params)
        text = f"def {self.name}({params}) -> {self.return_type}:\n"
        text += _render_block(self.body, 2)
        return text

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(param.name for param in self.params)


def ensure_tuple(items: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(items)

