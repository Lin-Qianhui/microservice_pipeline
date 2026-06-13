#!/usr/bin/env python3
"""Shared Pyright probing helpers for static-analysis commands."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from microservice_pipeline.call_graph.generate_call_graph_ast import attach_parents, parse_python_source
except ImportError:  # pragma: no cover - supports direct script execution
    from microservice_pipeline.call_graph.generate_call_graph_ast import attach_parents, parse_python_source  # type: ignore


PROBE_NAME_PREFIX = "__msp_probe_"
PROBE_RE = re.compile(r'Type of "(__msp_probe_\d+)" is "([^"]+)"')
TYPE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FAMILY_DATAFRAME = "dataframe"
FAMILY_DICT = "dict"
FAMILY_LIST = "list"
FAMILY_SET = "set"
FAMILY_FILE = "file"
FAMILY_PATH = "path"
FAMILY_FIELD = "field"
FAMILY_XARRAY = "xarray"
FAMILY_OBJECT = "object"
FAMILY_UNKNOWN = "unknown"
PYRIGHT_FAMILIES = {
    FAMILY_DATAFRAME,
    FAMILY_DICT,
    FAMILY_LIST,
    FAMILY_SET,
    FAMILY_FILE,
    FAMILY_PATH,
    FAMILY_FIELD,
    FAMILY_XARRAY,
    FAMILY_OBJECT,
    FAMILY_UNKNOWN,
}

NON_CONTAINER_FIELD_QUALIFIERS = {
    "dataclasses",
    "django.",
    "pydantic",
    "sqlalchemy",
    "marshmallow",
}


@dataclass(frozen=True)
class PyrightProbeTarget:
    target_id: str
    expression: str
    file: Path
    module: str
    mode: str
    callable_id: str = ""
    lineno: int = 0
    insert_lineno: int = 0


def discover_project_root(root: Path) -> Path:
    current = root.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "pyrightconfig.json").exists():
            return candidate
    return root.resolve()


def pyright_family_from_type_text(type_text: str) -> str:
    text = " ".join(type_text.strip().split())
    if not text:
        return FAMILY_UNKNOWN
    lowered = text.lower()
    type_tokens = set(TYPE_NAME_RE.findall(text))
    lowered_tokens = {token.lower() for token in type_tokens}

    unknown_markers = {"unknown", "any", "typing.any", "none", "nonetype", "never"}
    if lowered in unknown_markers:
        return FAMILY_UNKNOWN

    has_unknown = any(marker in lowered for marker in {"unknown", "typing.any", " any", "any |", "| any"})
    families = set()

    if "dataframe" in lowered or "pandas.core.frame" in lowered:
        families.add(FAMILY_DATAFRAME)
    if any(token in lowered for token in {"dict[", "mapping[", "mutablemapping[", "collections.abc.mapping"}):
        families.add(FAMILY_DICT)
    if "attrdict" in lowered or "mutableattr" in lowered:
        families.add(FAMILY_DICT)
    if any(token in lowered for token in {"list[", "sequence[", "mutablesequence[", "collections.abc.sequence"}):
        families.add(FAMILY_LIST)
    if any(token in lowered for token in {"set[", "frozenset[", "abstractset[", "collections.abc.set"}):
        families.add(FAMILY_SET)
    if lowered in {"path", "pathlib.path"} or any(
        token in lowered for token in {"pathlib.", " path", "path[", "purepath", "pathlike", "os.pathlike"}
    ):
        families.add(FAMILY_PATH)
    if any(
        token in lowered
        for token in {
            "textio",
            "binaryio",
            "typing.io",
            "io[",
            "textiowrapper",
            "iobase",
            "bufferedreader",
            "bufferedwriter",
        }
    ):
        families.add(FAMILY_FILE)
    if any(
        token in lowered
        for token in {
            "numpy.ndarray",
            "np.ndarray",
            "numpy.typing.ndarray",
            "npt.ndarray",
        }
    ) or "ndarray" in lowered_tokens:
        families.add(FAMILY_FIELD)
    if (
        "field" in lowered_tokens
        and not any(token in lowered for token in NON_CONTAINER_FIELD_QUALIFIERS)
        and (
            lowered == "field"
            or lowered.startswith("field[")
            or lowered.endswith(".field")
            or ".field[" in lowered
            or "climlab" in lowered
        )
    ):
        families.add(FAMILY_FIELD)
    if any(
        token in lowered
        for token in {
            "xarray.dataset",
            "xarray.core.dataset",
            "xarray.dataarray",
            "xarray.core.dataarray",
            "xr.dataset",
            "xr.dataarray",
        }
    ):
        families.add(FAMILY_XARRAY)

    if has_unknown:
        return FAMILY_UNKNOWN
    if len(families) == 1:
        return next(iter(families))
    if len(families) > 1:
        return FAMILY_UNKNOWN

    if any(char.isalpha() for char in text):
        return FAMILY_OBJECT
    return FAMILY_UNKNOWN


def parse_pyright_probe_output(output: str, probe_to_target: Dict[str, str]) -> Dict[str, str]:
    families: Dict[str, str] = {}
    for probe_name, type_text in PROBE_RE.findall(output):
        target_id = probe_to_target.get(probe_name)
        if not target_id:
            continue
        families[target_id] = pyright_family_from_type_text(type_text)
    return families


def _resolve_pyright_bin(pyright_bin: str) -> str:
    candidate = Path(pyright_bin)
    if candidate.exists():
        return str(candidate.resolve())
    resolved = shutil.which(pyright_bin)
    if resolved:
        return resolved
    raise RuntimeError(f"Pyright binary not found: {pyright_bin}")


def _callable_node_map(tree: ast.Module, module: str) -> Dict[str, ast.AST]:
    mapping: Dict[str, ast.AST] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.current_class: Optional[str] = None
            self.current_callable: Optional[str] = None

        def _callable_id(self, name: str) -> str:
            if self.current_callable is not None:
                return f"{self.current_callable}.<locals>.{name}"
            if self.current_class:
                return f"{module}.{self.current_class}.{name}"
            return f"{module}.{name}"

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            prev_class = self.current_class
            self.current_class = node.name
            for stmt in node.body:
                self.visit(stmt)
            self.current_class = prev_class

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_callable(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_callable(node, node.name)

        def _visit_callable(self, node: ast.AST, name: str) -> None:
            prev_callable = self.current_callable
            self.current_callable = self._callable_id(name)
            mapping[self.current_callable] = node
            for stmt in getattr(node, "body", []):
                self.visit(stmt)
            self.current_callable = prev_callable

    Visitor().visit(tree)
    return mapping


def _local_insert_line(source_lines: List[str], lineno: int) -> int:
    return min(max(1, lineno + 1), len(source_lines) + 1)


def _indent_for_line(source_lines: List[str], lineno: int, fallback: int = 4) -> str:
    if 1 <= lineno <= len(source_lines):
        line = source_lines[lineno - 1]
        return line[: len(line) - len(line.lstrip(" "))]
    return " " * fallback


def _param_probe_location(node: ast.AST) -> Tuple[int, str]:
    body = list(getattr(node, "body", []))
    if body and isinstance(body[0], ast.Expr):
        expr_value = body[0].value
        if isinstance(expr_value, ast.Constant) and isinstance(expr_value.value, str):
            body = body[1:]
    if body:
        first_stmt = body[0]
        return getattr(first_stmt, "lineno", getattr(node, "lineno", 1)), " " * getattr(first_stmt, "col_offset", 4)
    return getattr(node, "lineno", 1) + 1, " " * (getattr(node, "col_offset", 0) + 4)


def _apply_probes_to_source(
    source_text: str,
    probes: Sequence[Tuple[str, PyrightProbeTarget]],
    callable_nodes: Dict[str, ast.AST],
) -> str:
    lines = source_text.splitlines()
    inserts_before: Dict[int, List[str]] = defaultdict(list)

    for probe_name, probe in probes:
        if probe.mode == "callable_entry":
            node = callable_nodes.get(probe.callable_id)
            if node is None:
                continue
            lineno, indent = _param_probe_location(node)
            inserts_before[lineno].extend(
                [
                    f"{indent}{probe_name} = {probe.expression}",
                    f"{indent}reveal_type({probe_name})",
                ]
            )
        else:
            insert_lineno = probe.insert_lineno or probe.lineno
            insert_line = _local_insert_line(lines, insert_lineno)
            indent = _indent_for_line(lines, insert_lineno)
            inserts_before[insert_line].extend(
                [
                    f"{indent}{probe_name} = {probe.expression}",
                    f"{indent}reveal_type({probe_name})",
                ]
            )

    output: List[str] = []
    for idx in range(1, len(lines) + 1):
        output.extend(inserts_before.get(idx, []))
        output.append(lines[idx - 1])
    output.extend(inserts_before.get(len(lines) + 1, []))
    return "\n".join(output) + ("\n" if source_text.endswith("\n") else "")


def _write_probe_config(temp_root: Path, include_paths: Sequence[Path]) -> Path:
    config_path = temp_root / "pyrightconfig.json"
    include = sorted({path.as_posix() for path in include_paths})
    config = {
        "include": include,
        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
        "typeCheckingMode": "basic",
        "reportMissingImports": "none",
        "reportMissingModuleSource": "none",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def probe_pyright_targets(
    project_root: Path,
    targets: Iterable[PyrightProbeTarget],
    pyright_bin: str,
) -> Dict[str, str]:
    project_root = project_root.resolve()
    probe_targets = list(targets)
    if not probe_targets:
        return {}

    pyright_path = _resolve_pyright_bin(pyright_bin)
    numbered_targets = [(f"{PROBE_NAME_PREFIX}{idx}", target) for idx, target in enumerate(probe_targets, start=1)]
    probe_to_target = {probe_name: target.target_id for probe_name, target in numbered_targets}

    targets_by_file: Dict[Path, List[Tuple[str, PyrightProbeTarget]]] = defaultdict(list)
    for probe_name, target in numbered_targets:
        targets_by_file[target.file.resolve()].append((probe_name, target))

    with tempfile.TemporaryDirectory(prefix="microservice_pipeline_pyright_") as tmpdir:
        temp_root = Path(tmpdir)
        for folder_name in ("src", "scripts", "tests"):
            src_dir = project_root / folder_name
            if src_dir.exists():
                shutil.copytree(src_dir, temp_root / folder_name)

        include_paths: List[Path] = []
        for file_path, file_targets in targets_by_file.items():
            rel_path = file_path.relative_to(project_root)
            include_paths.append(rel_path)
            temp_file = temp_root / rel_path
            source_text = file_path.read_text(encoding="utf-8")
            tree = parse_python_source(source_text, filename=str(file_path))
            attach_parents(tree)
            module = file_targets[0][1].module
            callable_nodes = _callable_node_map(tree, module)
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(
                _apply_probes_to_source(source_text, file_targets, callable_nodes),
                encoding="utf-8",
            )

        config_path = _write_probe_config(temp_root, include_paths)
        result = subprocess.run(
            [pyright_path, "--project", str(config_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode not in {0, 1}:
            raise RuntimeError(output or "Pyright failed without output")
    return parse_pyright_probe_output(output, probe_to_target)


__all__ = [
    "FAMILY_DATAFRAME",
    "FAMILY_DICT",
    "FAMILY_FIELD",
    "FAMILY_FILE",
    "FAMILY_LIST",
    "FAMILY_OBJECT",
    "FAMILY_PATH",
    "FAMILY_SET",
    "FAMILY_UNKNOWN",
    "FAMILY_XARRAY",
    "PROBE_RE",
    "PYRIGHT_FAMILIES",
    "PyrightProbeTarget",
    "discover_project_root",
    "parse_pyright_probe_output",
    "probe_pyright_targets",
    "pyright_family_from_type_text",
]
