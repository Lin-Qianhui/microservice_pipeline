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
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar


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
    """Parse a Python file without executing it.

    ``read_bytes`` rather than ``read_text(encoding="utf-8")``: handing
    ``ast.parse`` the raw bytes lets it honour a PEP-263 coding cookie, so a
    latin-1 source with ``# -*- coding: latin-1 -*-`` parses instead of raising
    ``UnicodeDecodeError``. That is half of data-access review 1.12; the other
    half is that a failure here must not end the run, which is the caller's job.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(py_file.read_bytes(), filename=str(py_file))


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


_HasPath = TypeVar("_HasPath")


def partition_parseable(
    files: Sequence[_HasPath], cache: "ParsedFileCache"
) -> Tuple[List[_HasPath], List[Tuple[Path, str]]]:
    """Split analysed files into the ones that parse and the ones that do not.

    One unparseable file -- a vendored or generated source with a syntax error,
    or one this build cannot decode -- used to raise out through every caller
    and end the whole run. That is data-access review 1.12 and call-graph review
    items 14 and 15, which are the same defect seen from two packages.

    Doing the partition **once, up front** rather than catching at each parse
    site is the point. Every later pass then sees the same file set by
    construction, so a file cannot be present for one pass and missing from
    another -- which is the shape of the fixpoint defect Step 2 found, where a
    loop converged on a smaller question than the final pass asked.

    Parsing here costs nothing: it warms the shared cache that every pass would
    have filled anyway. Failures are returned rather than logged, because only
    the caller knows whether this run should report them or refuse.
    """
    parseable: List[_HasPath] = []
    failures: List[Tuple[Path, str]] = []
    for entry in files:
        path = getattr(entry, "path", entry)
        try:
            cache.get(path)
        except (SyntaxError, ValueError, UnicodeDecodeError, OSError) as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        parseable.append(entry)
    return parseable, failures
