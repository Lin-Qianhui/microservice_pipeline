"""Deciding which files to analyze and what to call them.

Two jobs live here. The first is selection: walk a source root, apply the
include/exclude globs, and add any explicitly requested entrypoints. The second
is naming: turn a filesystem path into the dotted module name that callable IDs
are built from, so ``src/shop/orders.py`` becomes ``shop.orders``.

Paths yielded by ``iter_analysis_files`` are always resolved and de-duplicated,
which is what makes them safe to use as cache keys further down the pipeline.
"""


from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from .models import MODULE_CALLABLE_QUALNAME, AnalysisFile

try:
    from microservice_pipeline.config import SourceRootConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import SourceRootConfig  # type: ignore


def path_to_module(py_file: Path, src_root: Path, module_prefix: Optional[str] = None) -> str:
    """Convert a file path under ``src_root`` to its Python module name.

    ``pkg/orders.py`` becomes ``pkg.orders`` and ``pkg/__init__.py`` becomes
    ``pkg``. ``module_prefix`` supports layouts where the scan root itself is a
    package directory and therefore is not present in the relative path.
    """
    rel = py_file.relative_to(src_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    module_name = ".".join(parts)
    if module_prefix:
        return f"{module_prefix}.{module_name}" if module_name else module_prefix
    return module_name


def module_callable_id(module: str) -> str:
    """Return the synthetic callable ID used for module-level execution."""
    if module:
        return f"{module}.{MODULE_CALLABLE_QUALNAME}"
    return MODULE_CALLABLE_QUALNAME


def _posix_relative(path: Path, base: Path) -> Optional[str]:
    """Return a portable relative path, or ``None`` when outside ``base``."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return None


def _glob_matches(value: str, pattern: str) -> bool:
    """Match a normalized path, including ``**/x`` matching top-level ``x``."""
    normalized = pattern.replace("\\", "/")
    return fnmatch.fnmatchcase(value, normalized) or (
        normalized.startswith("**/") and fnmatch.fnmatchcase(value, normalized[3:])
    )


def _matches_any(path: Path, patterns: Sequence[str], root: Path, project_root: Path) -> bool:
    """Check globs against project-relative, scan-relative, and base names."""
    if not patterns:
        return False
    candidates = [
        candidate
        for candidate in (
            _posix_relative(path, project_root),
            _posix_relative(path, root),
            path.name,
        )
        if candidate
    ]
    return any(_glob_matches(candidate, pattern) for candidate in candidates for pattern in patterns)


def iter_python_files(
    root: Path,
    *,
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
) -> Iterable[Path]:
    """Yield included Python files in stable order.

    Hidden directories are always ignored. Include patterns form an allow-list
    when supplied; exclude patterns are then applied as a deny-list.
    """
    project_root = project_root.resolve() if project_root is not None else infer_project_root(root)
    for path in sorted(root.rglob("*.py")):
        if any(part.startswith(".") for part in path.parts):
            continue
        if include_globs and not _matches_any(path, include_globs, root, project_root):
            continue
        if exclude_globs and _matches_any(path, exclude_globs, root, project_root):
            continue
        yield path


def infer_project_root(src_root: Path) -> Path:
    """Find the nearest repository/project root used to name entrypoint modules."""
    resolved = src_root.resolve()
    for candidate in [resolved, *resolved.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    if resolved.parent.name == "src":
        return resolved.parent.parent
    return resolved


def module_for_analysis_file(
    py_file: Path,
    src_root: Path,
    module_prefix: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> str:
    """Choose a module name for either an in-root file or extra entrypoint."""
    py_file = py_file.resolve()
    src_root = src_root.resolve()
    try:
        py_file.relative_to(src_root)
    except ValueError:
        root = project_root.resolve() if project_root is not None else infer_project_root(src_root)
        return path_to_module(py_file, root)
    return path_to_module(py_file, src_root, module_prefix=module_prefix)


def iter_analysis_files(
    src_root: Path,
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
) -> Iterable[AnalysisFile]:
    """Yield each source file once, followed by any additional entrypoints.

    Entrypoints allow scripts outside the normal source root to participate in
    the graph. They are validated eagerly because silently skipping a misspelled
    path would produce a plausible but incomplete graph.
    """
    src_root = src_root.resolve()
    project_root = project_root.resolve() if project_root is not None else infer_project_root(src_root)
    seen: Set[Path] = set()

    for py_file in iter_python_files(
        src_root,
        project_root=project_root,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    ):
        resolved = py_file.resolve()
        seen.add(resolved)
        yield AnalysisFile(
            path=resolved,
            module=module_for_analysis_file(
                resolved,
                src_root,
                module_prefix=module_prefix,
                project_root=project_root,
            ),
        )

    for entrypoint in entrypoints:
        resolved = entrypoint.resolve()
        if resolved in seen:
            continue
        if resolved.suffix != ".py":
            raise ValueError(f"Entrypoint must be a Python file: {entrypoint}")
        if not resolved.exists():
            raise FileNotFoundError(f"Entrypoint not found: {entrypoint}")
        seen.add(resolved)
        yield AnalysisFile(
            path=resolved,
            module=module_for_analysis_file(
                resolved,
                src_root,
                module_prefix=module_prefix,
                project_root=project_root,
            ),
        )


def iter_analysis_files_for_source_roots(
    source_roots: Sequence[SourceRootConfig],
    *,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
) -> List[AnalysisFile]:
    """Combine configured source roots and entrypoints without duplicate files."""
    files: List[AnalysisFile] = []
    seen: Set[Path] = set()
    resolved_project_root = (
        project_root.resolve()
        if project_root is not None
        else infer_project_root(source_roots[0].path)
    )
    for source_root in source_roots:
        for analysis_file in iter_analysis_files(
            source_root.path,
            module_prefix=source_root.module_prefix,
            entrypoints=(),
            project_root=resolved_project_root,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        ):
            if analysis_file.path in seen:
                continue
            seen.add(analysis_file.path)
            files.append(analysis_file)

    if entrypoints:
        first_root = source_roots[0]
        for analysis_file in iter_analysis_files(
            first_root.path,
            module_prefix=first_root.module_prefix,
            entrypoints=entrypoints,
            project_root=resolved_project_root,
        ):
            if analysis_file.path in seen:
                continue
            seen.add(analysis_file.path)
            files.append(analysis_file)
    return files
