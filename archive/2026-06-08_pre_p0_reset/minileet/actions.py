from __future__ import annotations

from dataclasses import dataclass, replace

from minileet.dsl import Function, Stmt


Path = tuple[int, ...]


@dataclass(frozen=True)
class ReplaceStmt:
    path: Path
    stmt: Stmt


@dataclass(frozen=True)
class InsertStmt:
    path: Path
    index: int
    stmt: Stmt


@dataclass(frozen=True)
class DeleteStmt:
    path: Path
    index: int


Action = ReplaceStmt | InsertStmt | DeleteStmt


def apply_action(function: Function, action: Action) -> Function:
    if isinstance(action, ReplaceStmt):
        return replace(function, body=_replace_in_block(function.body, action.path, action.stmt))
    if isinstance(action, InsertStmt):
        return replace(function, body=_insert_in_block(function.body, action.path, action.index, action.stmt))
    if isinstance(action, DeleteStmt):
        return replace(function, body=_delete_in_block(function.body, action.path, action.index))
    raise TypeError(type(action).__name__)


def _replace_in_block(block: tuple[Stmt, ...], path: Path, stmt: Stmt) -> tuple[Stmt, ...]:
    if len(path) != 1:
        raise NotImplementedError("nested statement paths are not implemented yet")
    items = list(block)
    items[path[0]] = stmt
    return tuple(items)


def _insert_in_block(block: tuple[Stmt, ...], path: Path, index: int, stmt: Stmt) -> tuple[Stmt, ...]:
    if path:
        raise NotImplementedError("nested statement paths are not implemented yet")
    items = list(block)
    items.insert(index, stmt)
    return tuple(items)


def _delete_in_block(block: tuple[Stmt, ...], path: Path, index: int) -> tuple[Stmt, ...]:
    if path:
        raise NotImplementedError("nested statement paths are not implemented yet")
    items = list(block)
    del items[index]
    return tuple(items)

