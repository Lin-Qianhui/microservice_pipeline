"""Shared helpers for process-tree coupling heuristics."""

from __future__ import annotations

import ast
from typing import Optional


ADD_SUBPROCESS_METHOD = "add_subprocess"


def is_add_subprocess_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == ADD_SUBPROCESS_METHOD
    )


def add_subprocess_name(node: ast.Call) -> str:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    for keyword in node.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return ""


def add_subprocess_child_expr(node: ast.Call) -> Optional[ast.AST]:
    if len(node.args) >= 2:
        return node.args[1]
    for keyword in node.keywords:
        if keyword.arg in {"proc", "process", "child"}:
            return keyword.value
    return None
