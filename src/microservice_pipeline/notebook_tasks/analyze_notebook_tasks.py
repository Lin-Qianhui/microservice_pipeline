#!/usr/bin/env python3
"""Analyze tutorial notebook tasks against structural microservice candidates.

The notebook view is intentionally kept separate from the structural graph in
this first implementation. It produces overlays, diagnostics, and a proposed
task-aware refinement while preserving the baseline structural clustering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from microservice_pipeline.jsonc_config import load_jsonc
    from microservice_pipeline.call_graph.generate_call_graph_ast import (
        SourceCallResolver,
        build_indices,
        build_return_summaries,
        build_type_summaries,
    )
    from microservice_pipeline.notebook_tasks.heading_classification import (
        DEFAULT_IGNORED_PATTERNS,
        DEFAULT_SUPPORT_PATTERNS,
        HeadingClassification,
        classify_heading,
        heading_classification_from_config,
    )
    from microservice_pipeline.notebook_tasks.outputs import (
        CLUSTER_TASK_DIAGNOSTIC_FIELDS,
        NOTEBOOK_TASK_FIELDS,
        REFINEMENT_RECOMMENDATION_FIELDS,
        TASK_CALLABLE_USAGE_FIELDS,
        TASK_CLUSTER_OVERLAY_FIELDS,
        TASK_SCATTER_FIELDS,
        write_analysis_outputs,
        write_report,
    )
    from microservice_pipeline.notebook_tasks.pruning import (
        NotebookPruningResult,
        prune_notebook_unobserved_assignments,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from jsonc_config import load_jsonc  # type: ignore
    from call_graph.generate_call_graph_ast import (  # type: ignore
        SourceCallResolver,
        build_indices,
        build_return_summaries,
        build_type_summaries,
    )
    from notebook_tasks.heading_classification import (  # type: ignore
        DEFAULT_IGNORED_PATTERNS,
        DEFAULT_SUPPORT_PATTERNS,
        HeadingClassification,
        classify_heading,
        heading_classification_from_config,
    )
    from notebook_tasks.outputs import (  # type: ignore
        CLUSTER_TASK_DIAGNOSTIC_FIELDS,
        NOTEBOOK_TASK_FIELDS,
        REFINEMENT_RECOMMENDATION_FIELDS,
        TASK_CALLABLE_USAGE_FIELDS,
        TASK_CLUSTER_OVERLAY_FIELDS,
        TASK_SCATTER_FIELDS,
        write_analysis_outputs,
        write_report,
    )
    from notebook_tasks.pruning import (  # type: ignore
        NotebookPruningResult,
        prune_notebook_unobserved_assignments,
    )


TASK_GRANULARITIES = ("leaf-heading", "major-heading", "notebook")
REFINEMENT_ACCEPTANCE_MODES = ("all", "none", "selected")
DEFAULT_NOTEBOOKS: tuple[str, ...] = tuple()
DEFAULT_NOTEBOOK_GLOBS = (
    "docs/**/*.ipynb",
    "notebooks/**/*.ipynb",
)
DEFAULT_EXCLUDE_NOTEBOOKS: tuple[str, ...] = tuple()
SPLIT_DOMINANCE_THRESHOLD = 0.80
MIN_SECONDARY_CALLABLES = 2
MIN_SECONDARY_OCCURRENCES = 3
MERGE_PURITY_THRESHOLD = 0.80
TASK_EXTRACT_MIN_CALLABLES = 1
TASK_EXTRACT_MAX_CALLABLES = 8
TASK_EXTRACT_MAX_CLUSTERS = 3
TASK_EXTRACT_MIN_CLUSTER_SIZE = 20
TASK_EXTRACT_MIN_CLUSTER_SIZE_RATIO = 4.0
DEFAULT_REFINEMENT_HEADING_LEVEL = 4
DEFAULT_TASK_EXTRACT_CALL_DEPTH = 2
TASK_EXTRACT_MAX_EXPANDED_CALLABLES = 12
TASK_EXTRACT_EXPAND_RELATIONS = frozenset(
    {
        "direct",
        "imported",
        "self_method",
        "class_method",
    }
)
DATA_OWNERSHIP_ACCESSES = frozenset({"create", "write", "read_write"})
DATA_CREATOR_ACCESSES = frozenset({"create"})
DEFAULT_REFINEMENT_ACCEPTANCE_MODE = "all"
PREVIEW_LIMIT = 8


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    task_title: str
    task_classification: str
    scenario_task_id: str
    major_task_id: str
    major_task_title: str
    leaf_task_id: str
    leaf_task_title: str


@dataclass(frozen=True)
class RefinementAcceptance:
    mode: str = DEFAULT_REFINEMENT_ACCEPTANCE_MODE
    refined_groups: frozenset[str] = frozenset()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _slug(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or fallback


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _preview(values: Iterable[str], limit: int = PREVIEW_LIMIT) -> str:
    cleaned = sorted({value for value in values if value})
    return ";".join(cleaned[:limit])


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _has_glob_chars(value: str) -> bool:
    return any(char in value for char in "*?[")


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _resolve_optional_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(root, _text(value))


def _expand_notebook_specs(root: Path, specs: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for spec in specs:
        if _has_glob_chars(spec):
            base = root
            pattern = spec
            if Path(spec).is_absolute():
                base = Path("/")
                pattern = str(Path(spec).relative_to("/"))
            for path in base.glob(pattern):
                if path.is_file():
                    paths.add(path.resolve())
        else:
            path = _resolve_path(root, spec)
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_jsonc(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Notebook task config must be a JSON object: {path}")
    return payload


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Notebook task config section must be an object: {name}")
    return value


def _config_string_list(value: Any, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_text(value)] if _text(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_text(item) for item in value if _text(item)]
    raise ValueError(f"Notebook task config value must be a string or array: {key}")


def _set_config_default(
    defaults: dict[str, Any],
    section: Mapping[str, Any],
    key: str,
    dest: str | None = None,
    *,
    path: bool = False,
    string_list: bool = False,
    project_root: Path | None = None,
) -> None:
    if key not in section:
        return
    value = section[key]
    if path:
        if project_root is None:
            raise ValueError("project_root is required for path config values")
        path_value = _resolve_optional_path(project_root, value)
        value = str(path_value) if path_value is not None else None
    elif string_list:
        value = _config_string_list(value, key)
    defaults[dest or key] = value


def notebook_task_config_defaults(
    config: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    if not config:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("Notebook task config must be a JSON object")

    defaults: dict[str, Any] = {}
    paths = _config_section(config, "paths")
    source = _config_section(config, "source")
    tasks = _config_section(config, "tasks")
    refinement = _config_section(config, "refinement")
    pruning = _config_section(config, "pruning")
    notebooks = _config_section(config, "notebooks")

    for key in (
        "call_graph",
        "structural_nodes",
        "structural_edges",
        "clusters",
        "source_root",
        "outdir",
        "reusable_outdir",
    ):
        _set_config_default(defaults, paths, key, path=True, project_root=project_root)
    _set_config_default(defaults, paths, "output_dir", "outdir", path=True, project_root=project_root)
    _set_config_default(
        defaults,
        paths,
        "reusable_output_dir",
        "reusable_outdir",
        path=True,
        project_root=project_root,
    )

    _set_config_default(defaults, source, "source_root", path=True, project_root=project_root)
    _set_config_default(defaults, source, "module_prefix")
    _set_config_default(defaults, source, "package")

    _set_config_default(defaults, tasks, "task_granularity")
    _set_config_default(defaults, tasks, "granularity", "task_granularity")
    _set_config_default(defaults, tasks, "refinement_heading_level")
    _set_config_default(defaults, tasks, "task_extract_call_depth")

    _set_config_default(defaults, refinement, "accept_refinements")
    _set_config_default(defaults, refinement, "mode", "accept_refinements")
    _set_config_default(defaults, refinement, "accept_refinement", string_list=True)
    _set_config_default(defaults, refinement, "accepted_refinements", "accept_refinement", string_list=True)

    _set_config_default(defaults, pruning, "prune_notebook_unobserved")
    _set_config_default(defaults, pruning, "notebook_unobserved", "prune_notebook_unobserved")

    _set_config_default(defaults, notebooks, "notebook", string_list=True)
    _set_config_default(defaults, notebooks, "include", "notebook", string_list=True)
    _set_config_default(defaults, notebooks, "include_notebooks", "notebook", string_list=True)
    _set_config_default(defaults, notebooks, "notebook_glob", string_list=True)
    _set_config_default(defaults, notebooks, "include_globs", "notebook_glob", string_list=True)
    _set_config_default(defaults, notebooks, "exclude_notebook", string_list=True)
    _set_config_default(defaults, notebooks, "exclude", "exclude_notebook", string_list=True)
    _set_config_default(defaults, notebooks, "exclude_notebooks", "exclude_notebook", string_list=True)

    # Keep backwards compatibility with the original flat config shape.
    for key in (
        "task_granularity",
        "refinement_heading_level",
        "task_extract_call_depth",
        "prune_notebook_unobserved",
    ):
        _set_config_default(defaults, config, key)
    _set_config_default(defaults, config, "include_notebooks", "notebook", string_list=True)
    _set_config_default(defaults, config, "exclude_notebooks", "exclude_notebook", string_list=True)

    return defaults


def load_notebook_task_config_defaults(
    config_path: Path | None,
    project_root: Path,
) -> dict[str, Any]:
    if config_path is None:
        return {}
    return notebook_task_config_defaults(load_config(config_path), project_root)


def _int_config_value(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Config value {key!r} must be an integer") from None


def _bool_config_value(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Config value {key!r} must be a boolean")


def _string_sequence(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_text(value)] if _text(value) else []
    if isinstance(value, Sequence):
        return [_text(item) for item in value if _text(item)]
    return []


def refinement_acceptance_from_config(
    config: Mapping[str, Any],
    *,
    mode: str | None,
    accepted_refinements: Sequence[str],
) -> RefinementAcceptance:
    acceptance_config = config.get("refinement_acceptance")
    if not isinstance(acceptance_config, Mapping):
        acceptance_config = {}

    configured_groups = [
        *(_string_sequence(acceptance_config.get("accepted_refinements"))),
        *(_string_sequence(acceptance_config.get("accepted_refined_groups"))),
        *(_string_sequence(acceptance_config.get("accepted_groups"))),
    ]
    refined_groups = frozenset([*configured_groups, *accepted_refinements])

    selected_mode = (
        mode
        or _text(acceptance_config.get("mode"))
        or _text(config.get("refinement_acceptance_mode"))
        or (("selected" if refined_groups else DEFAULT_REFINEMENT_ACCEPTANCE_MODE))
    )
    if selected_mode not in REFINEMENT_ACCEPTANCE_MODES:
        raise ValueError(
            f"Unsupported refinement acceptance mode: {selected_mode}. "
            f"Expected one of {', '.join(REFINEMENT_ACCEPTANCE_MODES)}"
        )
    return RefinementAcceptance(mode=selected_mode, refined_groups=refined_groups)


def select_notebooks(
    root: Path,
    *,
    config: Mapping[str, Any],
    notebooks: Sequence[str],
    notebook_globs: Sequence[str],
    exclude_notebooks: Sequence[str],
) -> list[Path]:
    config_includes = [
        _text(value)
        for value in config.get("include_notebooks", [])
        if _text(value)
    ]
    config_excludes = [
        _text(value)
        for value in config.get("exclude_notebooks", [])
        if _text(value)
    ]
    include_specs = list(notebooks) + list(notebook_globs)
    if not include_specs:
        include_specs = config_includes or [*DEFAULT_NOTEBOOKS, *DEFAULT_NOTEBOOK_GLOBS]
    exclude_specs = [*DEFAULT_EXCLUDE_NOTEBOOKS, *config_excludes, *exclude_notebooks]
    included = _expand_notebook_specs(root, include_specs)
    excluded = set(_expand_notebook_specs(root, exclude_specs))
    return [path for path in included if path not in excluded]


def notebook_config_for_runtime(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose nested notebook config using the existing flat runtime keys."""

    if not isinstance(config, Mapping):
        return {}
    runtime = dict(config)
    notebooks = _config_section(config, "notebooks")
    heading = _config_section(config, "heading_classification")
    if "include_notebooks" not in runtime:
        include = notebooks.get("include_notebooks", notebooks.get("include"))
        if include is not None:
            runtime["include_notebooks"] = include
    if "exclude_notebooks" not in runtime:
        exclude = notebooks.get("exclude_notebooks", notebooks.get("exclude"))
        if exclude is not None:
            runtime["exclude_notebooks"] = exclude
    if heading:
        runtime["heading_classification"] = heading
    return runtime


def _markdown_headings(source: str) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for line in source.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line.strip())
        if not match:
            continue
        level = len(match.group(1))
        title = re.sub(r"\s+", " ", match.group(2)).strip()
        title = re.sub(r"<[^>]+>", "", title).strip()
        if title:
            headings.append((level, title))
    return headings


def _sanitize_code_cell(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")):
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines)


def _source(cell: Mapping[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(_text(part) for part in source)
    return _text(source)


def _task_row(
    *,
    task_id: str,
    notebook: str,
    notebook_path: str,
    cell_index: int,
    heading_level: int,
    heading_text: str,
    heading_path: str,
    parent_task_id: str,
    major_task_id: str,
    major_task_title: str,
    classification: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "notebook": notebook,
        "notebook_path": notebook_path,
        "cell_index": cell_index,
        "heading_level": heading_level,
        "heading_text": heading_text,
        "heading_path": heading_path,
        "parent_task_id": parent_task_id,
        "major_task_id": major_task_id,
        "major_task_title": major_task_title,
        "classification": classification,
        "normalized_label": _normalize_label(heading_text),
    }


def _selected_context(
    *,
    granularity: str,
    notebook_row: Mapping[str, Any],
    active_headings: Sequence[Mapping[str, Any]],
) -> TaskContext:
    major = next(
        (heading for heading in reversed(active_headings) if int(heading["heading_level"]) <= 3),
        notebook_row,
    )
    leaf = active_headings[-1] if active_headings else major
    selected = notebook_row
    if granularity == "major-heading":
        selected = major
    elif granularity == "leaf-heading":
        selected = leaf
    elif granularity == "notebook":
        selected = notebook_row
    else:
        raise ValueError(f"Unsupported task granularity: {granularity}")

    return TaskContext(
        task_id=_text(selected["task_id"]),
        task_title=_text(selected["heading_text"]),
        task_classification=_text(selected["classification"]),
        scenario_task_id=_text(notebook_row["task_id"]),
        major_task_id=_text(major["task_id"]),
        major_task_title=_text(major["heading_text"]),
        leaf_task_id=_text(leaf["task_id"]),
        leaf_task_title=_text(leaf["heading_text"]),
    )


def extract_notebook_tasks_and_usage(
    *,
    notebook_path: Path,
    project_root: Path,
    granularity: str,
    heading_rules: HeadingClassification,
    resolver: SourceCallResolver,
    structural_nodes: set[str],
    usage_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    notebook_payload = _load_json(notebook_path)
    cells = notebook_payload.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"Notebook cells must be a list: {notebook_path}")

    notebook_rel = _repo_relative(notebook_path, project_root)
    notebook_name = notebook_path.stem
    notebook_slug = _slug(notebook_rel)
    notebook_row = _task_row(
        task_id=f"notebook:{notebook_slug}",
        notebook=notebook_name,
        notebook_path=notebook_rel,
        cell_index=0,
        heading_level=0,
        heading_text=notebook_name,
        heading_path=notebook_name,
        parent_task_id="",
        major_task_id=f"notebook:{notebook_slug}",
        major_task_title=notebook_name,
        classification="domain",
    )
    task_rows: list[dict[str, Any]] = [notebook_row]
    usage_rows: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    usage_index = usage_start

    for cell_index, cell in enumerate(cells):
        cell_type = _text(cell.get("cell_type"))
        source = _source(cell)
        if cell_type == "markdown":
            for level, title in _markdown_headings(source):
                active = [
                    heading
                    for heading in active
                    if int(heading["heading_level"]) < level
                ]
                parent_task_id = _text(active[-1]["task_id"]) if active else notebook_row["task_id"]
                heading_path = " / ".join(
                    [*[_text(heading["heading_text"]) for heading in active], title]
                )
                classification = classify_heading(title, heading_rules)
                task_id = f"task:{notebook_slug}:c{cell_index}:h{level}:{_slug(title)}"
                major = next(
                    (heading for heading in reversed(active) if int(heading["heading_level"]) <= 3),
                    None,
                )
                if level <= 3:
                    major_task_id = task_id
                    major_task_title = title
                elif major:
                    major_task_id = _text(major["task_id"])
                    major_task_title = _text(major["heading_text"])
                else:
                    major_task_id = _text(notebook_row["task_id"])
                    major_task_title = _text(notebook_row["heading_text"])
                row = _task_row(
                    task_id=task_id,
                    notebook=notebook_name,
                    notebook_path=notebook_rel,
                    cell_index=cell_index,
                    heading_level=level,
                    heading_text=title,
                    heading_path=heading_path,
                    parent_task_id=parent_task_id,
                    major_task_id=major_task_id,
                    major_task_title=major_task_title,
                    classification=classification,
                )
                task_rows.append(row)
                active.append(row)
            continue

        if cell_type != "code":
            continue
        sanitized = _sanitize_code_cell(source)
        if not sanitized.strip():
            continue
        context = _selected_context(
            granularity=granularity,
            notebook_row=notebook_row,
            active_headings=active,
        )
        filename = f"{notebook_path}#cell{cell_index}"
        try:
            edges = resolver.resolve_source(sanitized, filename=filename)
        except SyntaxError as exc:
            usage_index += 1
            usage_rows.append(
                {
                    "usage_id": f"U{usage_index:06d}",
                    "notebook": notebook_name,
                    "notebook_path": notebook_rel,
                    "cell_index": cell_index,
                    "task_granularity": granularity,
                    "task_id": context.task_id,
                    "task_title": context.task_title,
                    "task_classification": context.task_classification,
                    "scenario_task_id": context.scenario_task_id,
                    "major_task_id": context.major_task_id,
                    "major_task_title": context.major_task_title,
                    "leaf_task_id": context.leaf_task_id,
                    "leaf_task_title": context.leaf_task_title,
                    "caller": "",
                    "callee": f"SyntaxError: {exc.msg}",
                    "callable_node": "",
                    "resolved": "0",
                    "relation": "syntax_error",
                    "lineno": exc.lineno or "",
                }
            )
            continue

        for edge in edges:
            callable_node = f"callable:{edge.callee}"
            usage_index += 1
            usage_rows.append(
                {
                    "usage_id": f"U{usage_index:06d}",
                    "notebook": notebook_name,
                    "notebook_path": notebook_rel,
                    "cell_index": cell_index,
                    "task_granularity": granularity,
                    "task_id": context.task_id,
                    "task_title": context.task_title,
                    "task_classification": context.task_classification,
                    "scenario_task_id": context.scenario_task_id,
                    "major_task_id": context.major_task_id,
                    "major_task_title": context.major_task_title,
                    "leaf_task_id": context.leaf_task_id,
                    "leaf_task_title": context.leaf_task_title,
                    "caller": edge.caller,
                    "callee": edge.callee,
                    "callable_node": callable_node if callable_node in structural_nodes else "",
                    "resolved": "1" if edge.resolved else "0",
                    "relation": edge.relation,
                    "lineno": edge.lineno,
                }
            )

    return task_rows, usage_rows, usage_index


def _task_heading_level(task: Mapping[str, Any]) -> int:
    try:
        return int(task.get("heading_level") or 0)
    except (TypeError, ValueError):
        return 0


def _refinement_task_for_usage(
    row: Mapping[str, Any],
    task_by_id: Mapping[str, Mapping[str, Any]],
    max_heading_level: int,
) -> Mapping[str, Any] | None:
    task_id = _text(row.get("task_id"))
    selected = task_by_id.get(task_id)
    if selected is None or max_heading_level <= 0:
        return selected
    if _text(selected.get("classification")) != "domain":
        return selected
    if _task_heading_level(selected) <= max_heading_level:
        return selected

    current = selected
    candidate: Mapping[str, Any] | None = None
    visited: set[str] = set()
    while True:
        parent_id = _text(current.get("parent_task_id"))
        if not parent_id or parent_id in visited:
            break
        visited.add(parent_id)
        parent = task_by_id.get(parent_id)
        if parent is None:
            break
        if (
            _task_heading_level(parent) <= max_heading_level
            and _task_heading_level(parent) > 0
            and _text(parent.get("classification")) == "domain"
        ):
            candidate = parent
            break
        current = parent
    return candidate or selected


def annotate_and_roll_up_usage_rows(
    usage_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    max_heading_level: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return raw usage with refinement metadata plus rows keyed by refinement task."""

    task_by_id = {_text(row.get("task_id")): row for row in task_rows}
    annotated_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    for row in usage_rows:
        raw_row = dict(row)
        refinement_task = _refinement_task_for_usage(raw_row, task_by_id, max_heading_level)
        if refinement_task is None:
            refinement_task = task_by_id.get(_text(raw_row.get("task_id")), {})
        refinement_task_id = _text(refinement_task.get("task_id")) or _text(raw_row.get("task_id"))
        refinement_task_title = _text(refinement_task.get("heading_text")) or _text(raw_row.get("task_title"))
        refinement_classification = _text(refinement_task.get("classification")) or _text(
            raw_row.get("task_classification")
        )

        raw_row["refinement_task_id"] = refinement_task_id
        raw_row["refinement_task_title"] = refinement_task_title
        raw_row["refinement_task_classification"] = refinement_classification
        annotated_rows.append(raw_row)

        refinement_row = dict(raw_row)
        refinement_row["task_id"] = refinement_task_id
        refinement_row["task_title"] = refinement_task_title
        refinement_row["task_classification"] = refinement_classification
        refinement_rows.append(refinement_row)
    return annotated_rows, refinement_rows


def usable_usage_rows(
    usage_rows: Sequence[Mapping[str, Any]],
    cluster_of: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for row in usage_rows:
        node = _text(row.get("callable_node"))
        if not node or node not in cluster_of:
            continue
        if _text(row.get("resolved")) not in {"1", "True", "true"}:
            continue
        if _text(row.get("relation")) == "import":
            continue
        if _text(row.get("task_classification")) != "domain":
            continue
        rows.append(row)
    return rows


def compute_task_cluster_overlay(
    usage_rows: Sequence[Mapping[str, Any]],
    cluster_of: Mapping[str, str],
    cluster_size: Mapping[str, int],
) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    all_rows = [
        row
        for row in usage_rows
        if _text(row.get("callable_node")) in cluster_of and _text(row.get("relation")) != "import"
    ]
    for row in all_rows:
        node = _text(row.get("callable_node"))
        cluster_id = cluster_of[node]
        key = (_text(row.get("task_id")), cluster_id)
        bucket = aggregate.setdefault(
            key,
            {
                "task_id": _text(row.get("task_id")),
                "task_title": _text(row.get("task_title")),
                "task_classification": _text(row.get("task_classification")),
                "task_granularity": _text(row.get("task_granularity")),
                "notebooks": set(),
                "cluster_id": cluster_id,
                "cluster_size": cluster_size.get(cluster_id, 0),
                "occurrences": 0,
                "callables": set(),
            },
        )
        bucket["notebooks"].add(_text(row.get("notebook_path")))
        bucket["occurrences"] += 1
        bucket["callables"].add(node)

    rows: list[dict[str, Any]] = []
    for bucket in aggregate.values():
        callables = bucket["callables"]
        rows.append(
            {
                "task_id": bucket["task_id"],
                "task_title": bucket["task_title"],
                "task_classification": bucket["task_classification"],
                "task_granularity": bucket["task_granularity"],
                "notebooks": _preview(bucket["notebooks"]),
                "cluster_id": bucket["cluster_id"],
                "cluster_size": bucket["cluster_size"],
                "occurrences": bucket["occurrences"],
                "callable_count": len(callables),
                "callables_preview": _preview(callables),
            }
        )
    rows.sort(key=lambda row: (row["task_id"], row["cluster_id"]))
    return rows


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(counts))


def cluster_usage_stats(
    usage_rows: Sequence[Mapping[str, Any]],
    cluster_of: Mapping[str, str],
) -> tuple[
    dict[str, Counter[str]],
    dict[str, dict[str, set[str]]],
    dict[str, Counter[str]],
]:
    cluster_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_task_callables: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    classification_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in usage_rows:
        node = _text(row.get("callable_node"))
        if not node or node not in cluster_of or _text(row.get("relation")) == "import":
            continue
        cluster_id = cluster_of[node]
        classification = _text(row.get("task_classification"))
        classification_counts[cluster_id][classification] += 1
        if classification != "domain":
            continue
        task_id = _text(row.get("task_id"))
        cluster_task_counts[cluster_id][task_id] += 1
        cluster_task_callables[cluster_id][task_id].add(node)
    return cluster_task_counts, cluster_task_callables, classification_counts


def compute_cluster_diagnostics(
    *,
    usage_rows: Sequence[Mapping[str, Any]],
    cluster_of: Mapping[str, str],
    cluster_size: Mapping[str, int],
    task_titles: Mapping[str, str],
) -> list[dict[str, Any]]:
    task_counts, task_callables, classification_counts = cluster_usage_stats(
        usage_rows, cluster_of
    )
    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(cluster_size):
        counts = task_counts.get(cluster_id, Counter())
        total = sum(counts.values())
        dominant_task_id = ""
        dominant_task_share = 0.0
        if counts:
            dominant_task_id, dominant_count = sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            dominant_task_share = dominant_count / total if total else 0.0
        task_count = len(counts)
        domain_callables = {
            node
            for callable_set in task_callables.get(cluster_id, {}).values()
            for node in callable_set
        }
        warning_values: list[str] = []
        action = "keep"
        if task_count >= 2 and dominant_task_share < SPLIT_DOMINANCE_THRESHOLD:
            eligible_tasks = [
                task_id
                for task_id, occurrences in counts.items()
                if len(task_callables[cluster_id][task_id]) >= MIN_SECONDARY_CALLABLES
                or occurrences >= MIN_SECONDARY_OCCURRENCES
            ]
            if len(eligible_tasks) >= 2:
                warning_values.append("mixed_domain_tasks")
                action = "split_candidate"
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": cluster_size.get(cluster_id, 0),
                "domain_occurrences": total,
                "domain_callable_count": len(domain_callables),
                "domain_task_count": task_count,
                "dominant_task_id": dominant_task_id,
                "dominant_task_title": task_titles.get(dominant_task_id, ""),
                "dominant_task_share": f"{dominant_task_share:.6f}",
                "task_entropy": f"{_entropy(list(counts.values())):.6f}",
                "support_occurrences": classification_counts.get(cluster_id, Counter()).get("support", 0),
                "ignored_occurrences": classification_counts.get(cluster_id, Counter()).get("ignored", 0),
                "warnings": ";".join(warning_values),
                "recommended_action": action,
            }
        )
    return rows


def compute_task_scatter(
    *,
    usage_rows: Sequence[Mapping[str, Any]],
    cluster_of: Mapping[str, str],
) -> list[dict[str, Any]]:
    task_clusters: dict[str, Counter[str]] = defaultdict(Counter)
    task_callables: dict[str, set[str]] = defaultdict(set)
    task_meta: dict[str, dict[str, str]] = {}
    for row in usage_rows:
        node = _text(row.get("callable_node"))
        if not node or node not in cluster_of or _text(row.get("relation")) == "import":
            continue
        task_id = _text(row.get("task_id"))
        cluster_id = cluster_of[node]
        task_clusters[task_id][cluster_id] += 1
        task_callables[task_id].add(node)
        meta = task_meta.setdefault(task_id, defaultdict(str))  # type: ignore[arg-type]
        meta["task_title"] = _text(row.get("task_title"))
        meta["task_classification"] = _text(row.get("task_classification"))
        meta["task_granularity"] = _text(row.get("task_granularity"))
        notebooks = set(filter(None, meta.get("notebooks", "").split(";")))
        notebooks.add(_text(row.get("notebook_path")))
        meta["notebooks"] = ";".join(sorted(notebooks))

    rows: list[dict[str, Any]] = []
    for task_id, counts in task_clusters.items():
        total = sum(counts.values())
        dominant_cluster_id, dominant_count = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        cluster_count = len(counts)
        warnings = "scattered_task" if cluster_count > 1 else ""
        meta = task_meta.get(task_id, {})
        rows.append(
            {
                "task_id": task_id,
                "task_title": meta.get("task_title", ""),
                "task_classification": meta.get("task_classification", ""),
                "task_granularity": meta.get("task_granularity", ""),
                "notebooks": meta.get("notebooks", ""),
                "cluster_count": cluster_count,
                "occurrences": total,
                "callable_count": len(task_callables.get(task_id, set())),
                "dominant_cluster_id": dominant_cluster_id,
                "dominant_cluster_share": f"{(dominant_count / total if total else 0.0):.6f}",
                "clusters_preview": _preview(counts.keys()),
                "warnings": warnings,
            }
        )
    rows.sort(key=lambda row: (-int(row["cluster_count"]), row["task_id"]))
    return rows


def identify_task_extraction_candidates(
    *,
    usage_rows: Sequence[Mapping[str, Any]],
    cluster_of: Mapping[str, str],
    cluster_size: Mapping[str, int],
    task_classification: Mapping[str, str],
) -> dict[str, set[str]]:
    task_callables: dict[str, set[str]] = defaultdict(set)
    task_clusters: dict[str, set[str]] = defaultdict(set)
    for row in usable_usage_rows(usage_rows, cluster_of):
        task_id = _text(row.get("task_id"))
        if task_classification.get(task_id, _text(row.get("task_classification"))) != "domain":
            continue
        node = _text(row.get("callable_node"))
        if not node:
            continue
        task_callables[task_id].add(node)
        task_clusters[task_id].add(cluster_of[node])

    candidates: dict[str, set[str]] = {}
    for task_id, callables in task_callables.items():
        callable_count = len(callables)
        if not (TASK_EXTRACT_MIN_CALLABLES <= callable_count <= TASK_EXTRACT_MAX_CALLABLES):
            continue
        clusters = task_clusters.get(task_id, set())
        if not clusters or len(clusters) > TASK_EXTRACT_MAX_CLUSTERS:
            continue
        oversized_source = any(
            cluster_size.get(cluster_id, 0) >= TASK_EXTRACT_MIN_CLUSTER_SIZE
            and cluster_size.get(cluster_id, 0) >= callable_count * TASK_EXTRACT_MIN_CLUSTER_SIZE_RATIO
            for cluster_id in clusters
        )
        if oversized_source:
            candidates[task_id] = set(callables)
    return candidates


def _call_expansion_adjacency(
    structural_edges: Sequence[Mapping[str, str]],
) -> dict[str, set[tuple[str, str]]]:
    adjacency: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for edge in structural_edges:
        if _text(edge.get("edge_type")) != "call":
            continue
        relation = _text(edge.get("relation"))
        if relation not in TASK_EXTRACT_EXPAND_RELATIONS:
            continue
        src = _text(edge.get("src"))
        dst = _text(edge.get("dst"))
        if src.startswith("callable:") and dst.startswith("callable:"):
            adjacency[src].add((dst, relation))
    return adjacency


def expand_task_extraction_candidates(
    *,
    task_extraction_candidates: Mapping[str, set[str]],
    structural_edges: Sequence[Mapping[str, str]],
    cluster_of: Mapping[str, str],
    max_depth: int,
    max_expanded_callables: int = TASK_EXTRACT_MAX_EXPANDED_CALLABLES,
) -> dict[str, set[str]]:
    if max_depth <= 0:
        return {task_id: set(nodes) for task_id, nodes in task_extraction_candidates.items()}

    adjacency = _call_expansion_adjacency(structural_edges)
    expanded_candidates: dict[str, set[str]] = {}
    for task_id, direct_nodes in task_extraction_candidates.items():
        direct = {node for node in direct_nodes if node in cluster_of and node.startswith("callable:")}
        if not direct:
            expanded_candidates[task_id] = set(direct_nodes)
            continue
        source_clusters = {cluster_of[node] for node in direct}
        reachable_clusters = set(source_clusters)
        reachable = set(direct)
        frontier = set(direct)
        exceeded_limit = False
        for _depth in range(max_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for callee, relation in adjacency.get(node, set()):
                    if callee not in cluster_of:
                        continue
                    callee_cluster = cluster_of[callee]
                    if callee_cluster not in source_clusters and relation != "imported":
                        continue
                    if (
                        callee_cluster not in reachable_clusters
                        and len(reachable_clusters | {callee_cluster}) > TASK_EXTRACT_MAX_CLUSTERS
                    ):
                        continue
                    if callee not in reachable:
                        next_frontier.add(callee)
            if not next_frontier:
                break
            if len(reachable | next_frontier) > max_expanded_callables:
                exceeded_limit = True
                break
            reachable.update(next_frontier)
            reachable_clusters.update(cluster_of[node] for node in next_frontier)
            frontier = next_frontier
        expanded_candidates[task_id] = set(direct_nodes) if exceeded_limit else reachable
    return expanded_candidates


def _edge_weight(row: Mapping[str, str]) -> float:
    try:
        return float(row.get("weight") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _incident_weights_by_node(edges: Sequence[Mapping[str, str]]) -> dict[str, list[tuple[str, float]]]:
    incident: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in edges:
        src = _text(edge.get("src"))
        dst = _text(edge.get("dst"))
        if not src or not dst:
            continue
        weight = _edge_weight(edge)
        incident[src].append((dst, weight))
        incident[dst].append((src, weight))
    return incident


def _data_access_mode(edge: Mapping[str, str]) -> str:
    access = _text(edge.get("access"))
    if access:
        return access
    return _text(edge.get("operation"))


def _data_access_edges_by_data(
    structural_edges: Sequence[Mapping[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    edges_by_data: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in structural_edges:
        if _text(edge.get("edge_type")) != "data_access":
            continue
        src = _text(edge.get("src"))
        dst = _text(edge.get("dst"))
        if src.startswith("callable:") and dst.startswith("data:"):
            edges_by_data[dst].append((src, _data_access_mode(edge)))
    return edges_by_data


def _attach_owned_data_to_refined_groups(
    *,
    structural_edges: Sequence[Mapping[str, str]],
    rows_by_node: Mapping[str, Mapping[str, str]],
    preliminary_group_of: dict[str, str],
    preliminary_action_of: dict[str, str],
    preliminary_task_of: dict[str, str],
) -> None:
    edges_by_data = _data_access_edges_by_data(structural_edges)
    for data_node, access_edges in edges_by_data.items():
        if data_node not in rows_by_node:
            continue
        creator_groups = {
            preliminary_group_of[callable_node]
            for callable_node, access in access_edges
            if callable_node in preliminary_group_of and access in DATA_CREATOR_ACCESSES
        }
        candidate_groups: dict[str, set[str]] = defaultdict(set)
        for callable_node, access in access_edges:
            if callable_node not in preliminary_group_of:
                continue
            if access not in DATA_OWNERSHIP_ACCESSES:
                continue
            task_id = preliminary_task_of.get(callable_node, "")
            if not task_id:
                continue
            group = preliminary_group_of[callable_node]
            if creator_groups and creator_groups != {group}:
                continue
            candidate_groups[group].add(callable_node)

        if len(candidate_groups) != 1:
            continue
        group = next(iter(candidate_groups))
        current_group = preliminary_group_of.get(data_node)
        if current_group == group:
            continue
        preliminary_group_of[data_node] = group
        preliminary_action_of[data_node] = "attached_owned_data"
        preliminary_task_of[data_node] = preliminary_task_of.get(
            next(iter(candidate_groups[group])),
            "",
        )


def _recommendation_accepted(
    refined_group: str,
    acceptance: RefinementAcceptance,
) -> bool:
    if acceptance.mode == "all":
        return True
    if acceptance.mode == "none":
        return False
    return refined_group in acceptance.refined_groups


def _applied_group_key(
    *,
    original_cluster_id: str,
    preliminary_group: str,
    final_group: str,
    acceptance: RefinementAcceptance,
) -> str:
    if acceptance.mode == "all":
        return final_group
    if acceptance.mode == "none":
        return original_cluster_id
    if final_group in acceptance.refined_groups:
        return final_group
    if preliminary_group in acceptance.refined_groups:
        return preliminary_group
    return original_cluster_id


def refine_assignments(
    *,
    cluster_rows: Sequence[Mapping[str, str]],
    structural_edges: Sequence[Mapping[str, str]],
    usage_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    task_titles: Mapping[str, str],
    task_classification: Mapping[str, str],
    task_granularity: str,
    task_extraction_candidates: Mapping[str, set[str]] | None = None,
    refinement_acceptance: RefinementAcceptance | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    acceptance = refinement_acceptance or RefinementAcceptance()
    rows_by_node = {_text(row.get("node")): row for row in cluster_rows if _text(row.get("node"))}
    cluster_members: dict[str, list[str]] = defaultdict(list)
    for row in cluster_rows:
        cluster_members[_text(row.get("cluster_id"))].append(_text(row.get("node")))

    split_clusters = {
        _text(row.get("cluster_id"))
        for row in diagnostics
        if _text(row.get("recommended_action")) == "split_candidate"
    }

    node_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_task_callables: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in usable_usage_rows(
        usage_rows,
        {_text(cluster_row.get("node")): _text(cluster_row.get("cluster_id")) for cluster_row in cluster_rows},
    ):
        node = _text(row.get("callable_node"))
        task_id = _text(row.get("task_id"))
        cluster_id = _text(rows_by_node[node].get("cluster_id"))
        node_task_counts[node][task_id] += 1
        cluster_task_counts[cluster_id][task_id] += 1
        cluster_task_callables[cluster_id][task_id].add(node)

    extraction_candidates = {
        task_id: set(nodes)
        for task_id, nodes in (task_extraction_candidates or {}).items()
        if task_classification.get(task_id, "domain") == "domain"
    }
    extract_task_votes_by_node: dict[str, Counter[str]] = defaultdict(Counter)
    for task_id, nodes in extraction_candidates.items():
        for node in nodes:
            if node in rows_by_node:
                extract_task_votes_by_node[node][task_id] = (
                    node_task_counts.get(node, Counter()).get(task_id, 0) or 1
                )
    extract_task_by_node: dict[str, str] = {}
    for node, votes in extract_task_votes_by_node.items():
        extract_task_by_node[node] = sorted(
            votes,
            key=lambda task_id: (
                -votes[task_id],
                len(extraction_candidates.get(task_id, set())),
                task_id,
            ),
        )[0]

    incident = _incident_weights_by_node(structural_edges)
    preliminary_group_of: dict[str, str] = {}
    preliminary_action_of: dict[str, str] = {}
    preliminary_task_of: dict[str, str] = {}
    recommendations: list[dict[str, Any]] = []

    extracted_nodes_by_task: dict[str, set[str]] = defaultdict(set)
    for node, task_id in extract_task_by_node.items():
        extracted_nodes_by_task[task_id].add(node)
    for task_id, nodes in sorted(extracted_nodes_by_task.items()):
        source_clusters = {
            _text(rows_by_node[node].get("cluster_id"))
            for node in nodes
            if node in rows_by_node
        }
        recommendations.append(
            {
                "kind": "extract_task",
                "cluster_id": _preview(source_clusters),
                "refined_group": f"extract::{task_id}",
                "task_id": task_id,
                "task_label": task_titles.get(task_id, ""),
                "reason": "small domain task extracted from oversized structural cluster(s)",
                "node_count": len(nodes),
                "occurrences": sum(
                    node_task_counts.get(node, Counter()).get(task_id, 0)
                    for node in nodes
                ),
                "callable_count": len(nodes),
            }
        )

    for cluster_id, members in sorted(cluster_members.items()):
        active_members: list[str] = []
        for node in members:
            task_id = extract_task_by_node.get(node)
            if task_id:
                preliminary_group_of[node] = f"extract::{task_id}"
                if node_task_counts.get(node, Counter()).get(task_id, 0):
                    preliminary_action_of[node] = "extracted_task_usage"
                else:
                    preliminary_action_of[node] = "extracted_task_call_expansion"
                preliminary_task_of[node] = task_id
            else:
                active_members.append(node)

        if cluster_id not in split_clusters:
            for node in active_members:
                preliminary_group_of[node] = cluster_id
                preliminary_action_of[node] = "unchanged"
                preliminary_task_of[node] = ""
            continue

        counts = cluster_task_counts.get(cluster_id, Counter())
        eligible_tasks = {
            task_id
            for task_id, occurrences in counts.items()
            if len(cluster_task_callables[cluster_id][task_id]) >= MIN_SECONDARY_CALLABLES
            or occurrences >= MIN_SECONDARY_OCCURRENCES
        }
        for task_id in sorted(eligible_tasks):
            recommendations.append(
                {
                    "kind": "split",
                    "cluster_id": cluster_id,
                    "refined_group": f"{cluster_id}::{task_id}",
                    "task_id": task_id,
                    "task_label": task_titles.get(task_id, ""),
                    "reason": "domain task mixture below dominance threshold",
                    "node_count": len(cluster_task_callables[cluster_id][task_id]),
                    "occurrences": counts[task_id],
                    "callable_count": len(cluster_task_callables[cluster_id][task_id]),
                }
            )

        assigned_members_by_group: dict[str, set[str]] = defaultdict(set)
        unassigned: list[str] = []
        for node in active_members:
            votes = Counter(
                {
                    task_id: count
                    for task_id, count in node_task_counts.get(node, Counter()).items()
                    if task_id in eligible_tasks
                }
            )
            if votes:
                top_count = max(votes.values())
                winners = sorted(task_id for task_id, count in votes.items() if count == top_count)
                if len(winners) == 1:
                    task_id = winners[0]
                    group = f"{cluster_id}::{task_id}"
                    preliminary_group_of[node] = group
                    preliminary_action_of[node] = "split_by_task_usage"
                    preliminary_task_of[node] = task_id
                    assigned_members_by_group[group].add(node)
                    continue
            unassigned.append(node)

        for node in unassigned:
            scores: dict[str, float] = defaultdict(float)
            for other, weight in incident.get(node, []):
                if other not in rows_by_node:
                    continue
                if _text(rows_by_node[other].get("cluster_id")) != cluster_id:
                    continue
                other_group = preliminary_group_of.get(other)
                if other_group:
                    scores[other_group] += weight
            if scores:
                best_score = max(scores.values())
                winners = sorted(group for group, score in scores.items() if score == best_score)
                if len(winners) == 1 and best_score > 0:
                    group = winners[0]
                    preliminary_group_of[node] = group
                    preliminary_action_of[node] = "split_by_task_structural_attach"
                    preliminary_task_of[node] = group.split("::", 1)[1]
                    assigned_members_by_group[group].add(node)
                    continue
            group = f"{cluster_id}::residual"
            preliminary_group_of[node] = group
            preliminary_action_of[node] = "split_residual"
            preliminary_task_of[node] = ""

    _attach_owned_data_to_refined_groups(
        structural_edges=structural_edges,
        rows_by_node=rows_by_node,
        preliminary_group_of=preliminary_group_of,
        preliminary_action_of=preliminary_action_of,
        preliminary_task_of=preliminary_task_of,
    )

    group_members: dict[str, set[str]] = defaultdict(set)
    for node, group in preliminary_group_of.items():
        group_members[group].add(node)

    group_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for node, counts in node_task_counts.items():
        group = preliminary_group_of.get(node)
        if group:
            group_task_counts[group].update(counts)

    for recommendation in recommendations:
        members = group_members.get(_text(recommendation.get("refined_group")))
        if not members:
            continue
        recommendation["node_count"] = len(members)
        recommendation["callable_count"] = sum(
            1 for node in members if node.startswith("callable:")
        )

    merge_key_of_group: dict[str, str] = {}
    for group, counts in group_task_counts.items():
        if not counts or task_granularity == "notebook":
            continue
        task_id, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        purity = count / sum(counts.values())
        classification = task_classification.get(task_id, "domain")
        if purity < MERGE_PURITY_THRESHOLD or classification != "domain":
            continue
        normalized_label = _normalize_label(task_titles.get(task_id, task_id))
        if not normalized_label:
            continue
        merge_key_of_group[group] = normalized_label

    grouped_by_merge_key: dict[str, list[str]] = defaultdict(list)
    for group, merge_key in merge_key_of_group.items():
        grouped_by_merge_key[merge_key].append(group)

    final_group_key: dict[str, str] = {}
    for group in group_members:
        merge_key = merge_key_of_group.get(group)
        if merge_key and len(grouped_by_merge_key[merge_key]) > 1:
            final_group_key[group] = f"merge::{merge_key}"
            recommendations.append(
                {
                    "kind": "merge",
                    "cluster_id": group.split("::", 1)[0],
                    "refined_group": f"merge::{merge_key}",
                    "task_id": "",
                    "task_label": merge_key,
                    "reason": "same normalized high-purity domain task",
                    "node_count": len(group_members[group]),
                    "occurrences": sum(group_task_counts.get(group, Counter()).values()),
                    "callable_count": sum(1 for node in group_members[group] if node.startswith("callable:")),
                }
            )
        else:
            final_group_key[group] = group

    for recommendation in recommendations:
        recommendation["accepted"] = (
            "1"
            if _recommendation_accepted(_text(recommendation.get("refined_group")), acceptance)
            else "0"
        )

    applied_group_key_by_node: dict[str, str] = {}
    applied_action_by_node: dict[str, str] = {}
    applied_task_by_node: dict[str, str] = {}
    for row in cluster_rows:
        node = _text(row.get("node"))
        original_cluster_id = _text(row.get("cluster_id"))
        prelim_group = preliminary_group_of[node]
        final_group = final_group_key[prelim_group]
        applied_group = _applied_group_key(
            original_cluster_id=original_cluster_id,
            preliminary_group=prelim_group,
            final_group=final_group,
            acceptance=acceptance,
        )
        accepted = applied_group != original_cluster_id
        action = preliminary_action_of[node] if accepted else "unchanged"
        if accepted and applied_group.startswith("merge::"):
            action = "merged_same_task" if action == "unchanged" else f"{action};merged_same_task"
        applied_group_key_by_node[node] = applied_group
        applied_action_by_node[node] = action
        applied_task_by_node[node] = preliminary_task_of.get(node, "") if accepted else ""

    ordered_final_keys = sorted(set(applied_group_key_by_node.values()))
    refined_id_by_key = {
        key: f"R{index:03d}" for index, key in enumerate(ordered_final_keys, start=1)
    }

    refined_rows: list[dict[str, Any]] = []
    for row in cluster_rows:
        node = _text(row.get("node"))
        applied_group = applied_group_key_by_node[node]
        action = applied_action_by_node[node]
        task_id = applied_task_by_node[node]
        refined_row = dict(row)
        refined_row.update(
            {
                "original_cluster_id": _text(row.get("cluster_id")),
                "original_cluster_size": _text(row.get("cluster_size")),
                "refined_cluster_id": refined_id_by_key[applied_group],
                "refinement_action": action,
                "task_id": task_id,
                "task_label": task_titles.get(task_id, ""),
            }
        )
        refined_rows.append(refined_row)

    refined_sizes = Counter(row["refined_cluster_id"] for row in refined_rows)
    for row in refined_rows:
        refined_size = refined_sizes[row["refined_cluster_id"]]
        row["cluster_id"] = row["refined_cluster_id"]
        row["cluster_size"] = refined_size
        row["refined_cluster_size"] = refined_size

    return refined_rows, recommendations


def build_resolver(
    *,
    source_root: Path,
    module_prefix: str | None,
    package: str | None,
    module: str,
    file: Path,
) -> SourceCallResolver:
    nodes, module_map, known_classes = build_indices(source_root, module_prefix=module_prefix)
    return_summaries = build_return_summaries(
        source_root,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        module_prefix=module_prefix,
    )
    param_summaries, class_attr_types = build_type_summaries(
        source_root,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
        module_prefix=module_prefix,
    )
    return SourceCallResolver(
        module=module,
        file=file,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        include_external=True,
        package_prefix=package,
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
    )


def _preparse_project_and_config(argv: Sequence[str] | None) -> tuple[Path, Path | None, dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=None)
    args, _unknown = parser.parse_known_args(argv)
    project_root = Path(args.project_root).resolve()
    config_path = _resolve_optional_path(project_root, args.config)
    defaults = load_notebook_task_config_defaults(config_path, project_root)
    defaults["project_root"] = str(project_root)
    defaults["config"] = str(config_path) if config_path else None
    return project_root, config_path, defaults


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    _project_root, _config_path, config_defaults = _preparse_project_and_config(argv)
    parser = argparse.ArgumentParser(description="Analyze notebook task usage against structural clusters")
    parser.add_argument("--project-root", default=".", help="Project root used to resolve relative paths")
    parser.add_argument("--config", default=None, help="Optional notebook task analysis JSON/JSONC config")
    parser.add_argument("--call-graph", default="artifacts/call_graph/call_graph.json")
    parser.add_argument("--structural-nodes", default="artifacts/structural_dependency_graph/nodes.csv")
    parser.add_argument("--structural-edges", default="artifacts/structural_dependency_graph/edges.csv")
    parser.add_argument("--clusters", default="artifacts/structural_microservice_candidates/cluster_assignments.csv")
    parser.add_argument("--source-root", default="src", help="Python source root for call resolver indices")
    parser.add_argument("--module-prefix", default=None, help="Module prefix for source-root modules")
    parser.add_argument("--package", default=None, help="Internal package prefix for resolved calls")
    parser.add_argument("--notebook", action="append", default=[], help="Notebook path to include; may be repeated")
    parser.add_argument("--notebook-glob", action="append", default=[], help="Notebook glob to include; may be repeated")
    parser.add_argument("--exclude-notebook", action="append", default=[], help="Notebook path or glob to exclude; may be repeated")
    parser.add_argument("--task-granularity", choices=TASK_GRANULARITIES, default=None)
    parser.add_argument(
        "--refinement-heading-level",
        type=int,
        default=None,
        help=(
            "Roll deeper domain headings up to this heading level for overlay/refinement; "
            "use 0 to disable"
        ),
    )
    parser.add_argument(
        "--task-extract-call-depth",
        type=int,
        default=None,
        help="Bounded outgoing call depth used when extracting small notebook task candidates",
    )
    parser.add_argument(
        "--accept-refinements",
        choices=REFINEMENT_ACCEPTANCE_MODES,
        default=None,
        help=(
            "Which recommendations to apply to refined_cluster_assignments.csv: "
            "all, none, or selected refined_group IDs"
        ),
    )
    parser.add_argument(
        "--accept-refinement",
        action="append",
        default=[],
        help=(
            "refined_group ID from refinement_recommendations.csv to apply when "
            "--accept-refinements selected is used; may be repeated"
        ),
    )
    parser.add_argument(
        "--prune-notebook-unobserved",
        dest="prune_notebook_unobserved",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prune notebook-unobserved zero-in-degree callables from refined assignments",
    )
    parser.add_argument("--outdir", default="artifacts/notebook_task_analysis")
    parser.add_argument(
        "--reusable-outdir",
        default=None,
        help=(
            "Output directory for reusable notebook task extraction artifacts. "
            "Defaults to --outdir for backwards compatibility."
        ),
    )
    parser.set_defaults(**config_defaults)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    config = load_config(_resolve_path(project_root, args.config) if args.config else None)
    runtime_config = notebook_config_for_runtime(config)
    granularity = args.task_granularity or _text(runtime_config.get("task_granularity")) or "leaf-heading"
    if granularity not in TASK_GRANULARITIES:
        raise ValueError(f"Unsupported task granularity: {granularity}")
    refinement_heading_level = (
        args.refinement_heading_level
        if args.refinement_heading_level is not None
        else _int_config_value(runtime_config, "refinement_heading_level", DEFAULT_REFINEMENT_HEADING_LEVEL)
    )
    task_extract_call_depth = (
        args.task_extract_call_depth
        if args.task_extract_call_depth is not None
        else _int_config_value(runtime_config, "task_extract_call_depth", DEFAULT_TASK_EXTRACT_CALL_DEPTH)
    )
    refinement_acceptance = refinement_acceptance_from_config(
        runtime_config,
        mode=args.accept_refinements,
        accepted_refinements=args.accept_refinement,
    )
    prune_notebook_unobserved = (
        args.prune_notebook_unobserved
        if args.prune_notebook_unobserved is not None
        else _bool_config_value(runtime_config, "prune_notebook_unobserved", True)
    )

    call_graph_path = _resolve_path(project_root, args.call_graph)
    _load_json(call_graph_path)
    structural_nodes_path = _resolve_path(project_root, args.structural_nodes)
    structural_edges_path = _resolve_path(project_root, args.structural_edges)
    clusters_path = _resolve_path(project_root, args.clusters)
    source_root = _resolve_path(project_root, args.source_root)
    outdir = _resolve_path(project_root, args.outdir)
    reusable_outdir = _resolve_optional_path(project_root, args.reusable_outdir) or outdir

    notebooks = select_notebooks(
        project_root,
        config=runtime_config,
        notebooks=args.notebook,
        notebook_globs=args.notebook_glob,
        exclude_notebooks=args.exclude_notebook,
    )
    if not notebooks:
        raise FileNotFoundError("No notebooks matched the configured include/exclude rules")

    structural_node_rows, _structural_node_fields = _read_csv(structural_nodes_path)
    structural_nodes = {_text(row.get("id")) for row in structural_node_rows}
    cluster_rows, cluster_fields = _read_csv(clusters_path)
    structural_edges, _structural_edge_fields = _read_csv(structural_edges_path)
    cluster_of = {_text(row.get("node")): _text(row.get("cluster_id")) for row in cluster_rows}
    cluster_size = Counter(cluster_of.values())
    heading_rules = heading_classification_from_config(runtime_config)

    all_task_rows: list[dict[str, Any]] = []
    all_usage_rows: list[dict[str, Any]] = []
    usage_index = 0
    for notebook_path in notebooks:
        resolver = build_resolver(
            source_root=source_root,
            module_prefix=args.module_prefix or None,
            package=args.package or None,
            module=f"notebook_tasks.{_slug(notebook_path.stem)}",
            file=notebook_path,
        )
        task_rows, usage_rows, usage_index = extract_notebook_tasks_and_usage(
            notebook_path=notebook_path,
            project_root=project_root,
            granularity=granularity,
            heading_rules=heading_rules,
            resolver=resolver,
            structural_nodes=structural_nodes,
            usage_start=usage_index,
        )
        all_task_rows.extend(task_rows)
        all_usage_rows.extend(usage_rows)

    all_usage_rows, refinement_usage_rows = annotate_and_roll_up_usage_rows(
        all_usage_rows,
        all_task_rows,
        refinement_heading_level,
    )
    task_titles = {_text(row.get("task_id")): _text(row.get("heading_text")) for row in all_task_rows}
    task_classification = {
        _text(row.get("task_id")): _text(row.get("classification")) for row in all_task_rows
    }
    overlay_rows = compute_task_cluster_overlay(refinement_usage_rows, cluster_of, cluster_size)
    diagnostics = compute_cluster_diagnostics(
        usage_rows=refinement_usage_rows,
        cluster_of=cluster_of,
        cluster_size=cluster_size,
        task_titles=task_titles,
    )
    scatter_rows = compute_task_scatter(usage_rows=refinement_usage_rows, cluster_of=cluster_of)
    task_extraction_candidates = identify_task_extraction_candidates(
        usage_rows=refinement_usage_rows,
        cluster_of=cluster_of,
        cluster_size=cluster_size,
        task_classification=task_classification,
    )
    expanded_task_extraction_candidates = expand_task_extraction_candidates(
        task_extraction_candidates=task_extraction_candidates,
        structural_edges=structural_edges,
        cluster_of=cluster_of,
        max_depth=task_extract_call_depth,
    )
    refined_rows, recommendations = refine_assignments(
        cluster_rows=cluster_rows,
        structural_edges=structural_edges,
        usage_rows=refinement_usage_rows,
        diagnostics=diagnostics,
        task_titles=task_titles,
        task_classification=task_classification,
        task_granularity=granularity,
        task_extraction_candidates=expanded_task_extraction_candidates,
        refinement_acceptance=refinement_acceptance,
    )
    pruning_result = (
        prune_notebook_unobserved_assignments(
            refined_rows=refined_rows,
            structural_edges=structural_edges,
            usage_rows=all_usage_rows,
        )
        if prune_notebook_unobserved
        else NotebookPruningResult(
            refined_rows=[dict(row) for row in refined_rows],
            excluded_rows=[],
            pruned_callable_count=0,
            pruned_data_count=0,
        )
    )
    refined_rows = pruning_result.refined_rows

    analysis_metadata = {
        "schema": "notebook_task_analysis.v1",
        "task_granularity": granularity,
        "refinement_heading_level": refinement_heading_level,
        "task_extract_call_depth": task_extract_call_depth,
        "refinement_acceptance": {
            "mode": refinement_acceptance.mode,
            "accepted_refinements": sorted(refinement_acceptance.refined_groups),
        },
        "notebooks": [_repo_relative(path, project_root) for path in notebooks],
        "counts": {
            "tasks": len(all_task_rows),
            "usage_rows": len(all_usage_rows),
            "overlay_rows": len(overlay_rows),
            "diagnostics": len(diagnostics),
            "task_extraction_candidates": len(task_extraction_candidates),
            "expanded_task_extraction_candidates": len(expanded_task_extraction_candidates),
            "recommendations": len(recommendations),
            "refined_assignments": len(refined_rows),
            "notebook_unobserved_pruned_callables": pruning_result.pruned_callable_count,
            "notebook_unobserved_pruned_data_rows": pruning_result.pruned_data_count,
            "notebook_unobserved_excluded_nodes": len(pruning_result.excluded_rows),
        },
        "prune_notebook_unobserved": prune_notebook_unobserved,
        "heading_classification": {
            "ignored_patterns": list(heading_rules.ignored_patterns),
            "support_patterns": list(heading_rules.support_patterns),
        },
    }
    write_analysis_outputs(
        outdir=outdir,
        reusable_outdir=reusable_outdir,
        cluster_fields=cluster_fields,
        notebooks=notebooks,
        task_rows=all_task_rows,
        usage_rows=all_usage_rows,
        overlay_rows=overlay_rows,
        diagnostics=diagnostics,
        scatter_rows=scatter_rows,
        recommendations=recommendations,
        refined_rows=refined_rows,
        pruning_result=pruning_result,
        metadata=analysis_metadata,
        prune_notebook_unobserved=prune_notebook_unobserved,
    )

    print(f"Notebook task analysis written to {outdir}")
    print(f"Reusable notebook task artifacts written to {reusable_outdir}")
    print(f"Notebooks: {len(notebooks)}")
    print(f"Tasks/headings: {len(all_task_rows)}")
    print(f"Usage rows: {len(all_usage_rows)}")
    print(f"Refinement recommendations: {len(recommendations)}")
    print(f"Refinement acceptance: {refinement_acceptance.mode}")
    print(
        "Notebook-unobserved pruning: "
        f"{'enabled' if prune_notebook_unobserved else 'disabled'} "
        f"({pruning_result.pruned_callable_count} callables, "
        f"{pruning_result.pruned_data_count} data rows)"
    )


if __name__ == "__main__":
    main()
