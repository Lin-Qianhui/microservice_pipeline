"""Parsing helpers and small AST utilities.

Every pass in this package reads source code the same way: parse it without
importing or executing it, then walk the resulting tree. These helpers centralise
that, including the ``SyntaxWarning`` suppression that keeps a noisy analyzed
project from spamming the console, and the parent links that Python's
downward-only AST does not provide.
"""


from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Dict, Optional


def attribute_to_name(node: ast.AST) -> Optional[str]:
    """Flatten a name/attribute AST into text such as ``package.module.func``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = attribute_to_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def unwrap_passthrough(node: ast.AST) -> ast.AST:
    """Strip wrappers whose value is exactly their inner expression's.

    ``await f()`` and ``(x := f())`` evaluate to whatever ``f()`` produced, so
    every shape question the inference asks -- what class is this, what is
    inside it, what sits at position 2 -- has the same answer for the wrapper
    as for the expression underneath. Peeling them here means each inference
    helper matches on the shape that actually matters instead of repeating
    these cases.
    """
    while isinstance(node, (ast.Await, ast.NamedExpr)):
        node = node.value
    return node


def parse_python_file(py_file: Path) -> ast.Module:
    """Parse a UTF-8 file without executing it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))


def parse_python_source(source: str, filename: str = "<source>") -> ast.Module:
    """Parse an in-memory snippet; ``filename`` is used in syntax errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source, filename=filename)


def attach_parents(node: ast.AST) -> None:
    """Add parent links absent from Python's standard downward-only AST.

    ``DefinitionCollector`` uses these links to distinguish a class method from
    a nested or module-level function.
    """
    for child in ast.iter_child_nodes(node):
        setattr(child, "parent", node)
        attach_parents(child)


class ParsedFileCache:
    """Parse each source file once and hand the same tree to every pass.

    A full run walks the same files four times over (definitions, returns,
    types, edges), and the two summary passes are themselves loops, so parsing
    per pass costs six to ten parses per file. Nothing gains from re-reading the
    file: the source cannot change mid-run.

    Sharing one tree is safe because no pass rewrites the AST. The single
    mutation anywhere in this package is ``attach_parents``, which only *adds* a
    ``parent`` attribute and is idempotent, so doing it once at parse time
    leaves every later pass with the links already in place.

    Instances are meant to live for one analysis run and then be dropped, which
    bounds the memory held to one file set's worth of trees.
    """

    def __init__(self) -> None:
        self._trees: Dict[Path, ast.Module] = {}

    def get(self, py_file: Path) -> ast.Module:
        """Return the parsed tree for ``py_file``, parsing it on first request.

        Callers key on the resolved path that ``iter_analysis_files`` yields, so
        two spellings of the same file cannot produce two entries.
        """
        tree = self._trees.get(py_file)
        if tree is None:
            tree = parse_python_file(py_file)
            attach_parents(tree)
            self._trees[py_file] = tree
        return tree

    def __len__(self) -> int:
        return len(self._trees)
