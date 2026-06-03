"""Helpers for normalizing absolute and relative Python imports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def is_package_file(path: Path) -> bool:
    """Return whether a source file represents a package module."""
    return path.name == "__init__.py"


def resolve_import_from_module(
    current_module: str,
    imported_module: Optional[str],
    level: int,
    *,
    current_is_package: bool = False,
) -> Optional[str]:
    """Resolve the module part of an ``ast.ImportFrom`` node.

    ``level=0`` means an absolute import. For relative imports, ``current_module``
    is interpreted as either a normal module or a package ``__init__`` module.
    """
    if level <= 0:
        return imported_module or ""

    package = current_module if current_is_package else current_module.rpartition(".")[0]
    package_parts = package.split(".") if package else []
    parents_to_drop = level - 1
    if parents_to_drop > len(package_parts):
        return None
    if parents_to_drop:
        package_parts = package_parts[:-parents_to_drop]

    base = ".".join(part for part in package_parts if part)
    if imported_module:
        return f"{base}.{imported_module}" if base else imported_module
    return base


def resolve_import_from_target(
    current_module: str,
    imported_module: Optional[str],
    level: int,
    alias_name: str,
    *,
    current_is_package: bool = False,
) -> Optional[str]:
    """Resolve one imported name from an ``ast.ImportFrom`` node."""
    base = resolve_import_from_module(
        current_module,
        imported_module,
        level,
        current_is_package=current_is_package,
    )
    if base is None:
        return None
    if alias_name == "*":
        return base
    return f"{base}.{alias_name}" if base else alias_name

