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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from microservice_pipeline.call_graph.generate_call_graph_ast import attach_parents, parse_python_source
except ImportError:  # pragma: no cover - supports direct script execution
    from microservice_pipeline.call_graph.generate_call_graph_ast import attach_parents, parse_python_source  # type: ignore


PROBE_NAME_PREFIX = "__msp_probe_"
PROBE_RE = re.compile(r'Type of "(__msp_probe_\d+)" is "([^"]+)"')
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

# A type text is a nesting, and the family wanted is the outermost constructor,
# so these are keyed on the *head* -- the name before the first "[" -- and never
# matched as a substring of the whole text. Both the bare spelling and the
# qualified ones are listed, because pyright prints whichever the source used.
FAMILY_BY_TYPE_HEAD = {
    "dataframe": FAMILY_DATAFRAME,
    "pandas.core.frame.dataframe": FAMILY_DATAFRAME,
    "dict": FAMILY_DICT,
    "defaultdict": FAMILY_DICT,
    "ordereddict": FAMILY_DICT,
    "counter": FAMILY_DICT,
    "chainmap": FAMILY_DICT,
    "mapping": FAMILY_DICT,
    "mutablemapping": FAMILY_DICT,
    "typeddict": FAMILY_DICT,
    "attrdict": FAMILY_DICT,
    "mutableattr": FAMILY_DICT,
    "list": FAMILY_LIST,
    "sequence": FAMILY_LIST,
    "mutablesequence": FAMILY_LIST,
    "deque": FAMILY_LIST,
    "set": FAMILY_SET,
    "frozenset": FAMILY_SET,
    "abstractset": FAMILY_SET,
    "mutableset": FAMILY_SET,
    "path": FAMILY_PATH,
    "purepath": FAMILY_PATH,
    "posixpath": FAMILY_PATH,
    "pureposixpath": FAMILY_PATH,
    "windowspath": FAMILY_PATH,
    "purewindowspath": FAMILY_PATH,
    "pathlike": FAMILY_PATH,
    "io": FAMILY_FILE,
    "textio": FAMILY_FILE,
    "binaryio": FAMILY_FILE,
    "iobase": FAMILY_FILE,
    "textiowrapper": FAMILY_FILE,
    "bufferedreader": FAMILY_FILE,
    "bufferedwriter": FAMILY_FILE,
    "bufferedrandom": FAMILY_FILE,
    "stringio": FAMILY_FILE,
    "bytesio": FAMILY_FILE,
    "ndarray": FAMILY_FIELD,
    "field": FAMILY_FIELD,
    # xarray only by its qualified spelling. A bare ``Dataset`` is whatever the
    # module imported under that name, which is what the import map settles.
    "xarray.dataset": FAMILY_XARRAY,
    "xarray.dataarray": FAMILY_XARRAY,
    "xarray.core.dataset.dataset": FAMILY_XARRAY,
    "xarray.core.dataarray.dataarray": FAMILY_XARRAY,
    "xr.dataset": FAMILY_XARRAY,
    "xr.dataarray": FAMILY_XARRAY,
}

# Members of a union that say nothing. Dropped before the members are compared,
# so ``Unknown | Field | None`` is a Field rather than a refusal to answer.
UNINFORMATIVE_UNION_MEMBERS = {
    "unknown",
    "any",
    "typing.any",
    "none",
    "nonetype",
    "never",
    "...",
    "",
}


@dataclass(frozen=True)
class PyrightProbeTarget:
    """One expression to ask pyright the type of, and where to ask.

    ``mode`` says where the probe goes: ``callable_entry`` at the top of a
    function body, ``after_line`` after the statement containing the expression,
    and ``before_line`` in front of it -- the only workable place for a
    ``return`` or a ``raise``, since pyright does not answer for code after one
    of those at all.
    ``insert_col`` is where that statement starts on its line, so a probe is
    never wedged into the middle of ``if x: return y``.
    """

    target_id: str
    expression: str
    file: Path
    module: str
    mode: str
    callable_id: str = ""
    lineno: int = 0
    insert_lineno: int = 0
    insert_col: int = 0


def discover_project_root(root: Path) -> Path:
    current = root.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "pyrightconfig.json").exists():
            return candidate
    return root.resolve()


def _split_top_level_union(text: str) -> List[str]:
    """The members of a union, ignoring any ``|`` nested inside brackets."""
    members: List[str] = []
    depth = 0
    current: List[str] = []
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth = max(0, depth - 1)
        if char == "|" and depth == 0:
            members.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    members.append("".join(current).strip())
    return [member for member in members if member]


def _type_head(member: str) -> str:
    """The outermost constructor of one union member.

    ``dict[str, list[int]]`` is a dict; the inner ``list`` is what it holds, not
    what it is. Taking the text before the first bracket is the whole rule.

    The one wrinkle is pyright's own notation for a method's own class,
    ``Self@AttrDict``, which is that class and has to be read as one.
    """
    head = member.split("[", 1)[0].strip()
    if head.startswith("Self@"):
        head = head.split("@", 1)[1].strip()
    return head.rstrip("*").strip()


def _is_callable_type_text(member: str) -> bool:
    """Whether pyright described a function rather than a value.

    ``Overload[...]`` and ``(a: int) -> None`` are what comes back when a probe
    lands on a method reference. A function is never a container, and reading
    the names inside its signature is how ``(name: Unknown, value: ...) -> None``
    used to be answered ``unknown``.
    """
    return member.startswith("(") or member.lower().startswith("overload[")


def _family_for_union_member(member: str, name_qualifiers: Optional[Mapping[str, str]] = None) -> str:
    if _is_callable_type_text(member):
        return FAMILY_OBJECT
    head = _type_head(member)
    if not head:
        return FAMILY_UNKNOWN
    if name_qualifiers and "." not in head:
        head = name_qualifiers.get(head, head)
    lowered_head = head.lower()
    family = FAMILY_BY_TYPE_HEAD.get(lowered_head)
    if family is None:
        family = FAMILY_BY_TYPE_HEAD.get(lowered_head.rsplit(".", 1)[-1])
    if family == FAMILY_FIELD and any(
        qualifier in lowered_head for qualifier in NON_CONTAINER_FIELD_QUALIFIERS
    ):
        # ``dataclasses.Field`` and friends are schema declarations, not data.
        family = None
    if family is not None:
        return family
    if any(char.isalpha() for char in head):
        return FAMILY_OBJECT
    return FAMILY_UNKNOWN


def pyright_family_from_type_text(
    type_text: str, name_qualifiers: Optional[Mapping[str, str]] = None
) -> str:
    """The container family of one pyright ``reveal_type`` answer.

    The answer is parsed as what it is -- a union of nestings -- rather than
    scanned for substrings. Substring scanning made every parameterised generic
    whose value type was ``Any`` report ``unknown``, so ``dict[str, Any]`` was
    not a dict, and any text that mentioned two families at all collapsed to
    ``unknown``, so a dict of lists was neither.

    ``name_qualifiers`` maps names as written in the probed module to what they
    were imported from, so a bare ``Dataset`` can be recognised as xarray's only
    in the modules that actually imported it from there.
    """
    text = " ".join(type_text.strip().split())
    if not text:
        return FAMILY_UNKNOWN

    families = set()
    for member in _split_top_level_union(text):
        if member.lower() in UNINFORMATIVE_UNION_MEMBERS:
            continue
        families.add(_family_for_union_member(member, name_qualifiers))

    if len(families) == 1:
        return next(iter(families))
    # Nothing informative left, or the members genuinely disagree -- a value that
    # is a str or an ndarray depending on the branch has no one family.
    return FAMILY_UNKNOWN


UNRESOLVED_TYPE_TEXTS = {"unknown", "any", "typing.any", "none", "nonetype", "never"}


def _is_unresolved_type_text(type_text: str) -> bool:
    """Whether pyright had nothing to say, as opposed to saying something broad.

    ``dict[str, Unknown]`` is a dictionary and is not counted here; a bare
    ``Unknown`` means the probe's own type could not be worked out, which is
    almost always a failed import in the sandbox rather than a fact about the
    code.
    """
    return " ".join(type_text.strip().split()).lower() in UNRESOLVED_TYPE_TEXTS


def _parse_pyright_probe_answers(
    output: str,
    probe_to_target: Dict[str, str],
    qualifiers_by_probe: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Families by target id, and the raw type text each one came from."""
    families: Dict[str, str] = {}
    answers: Dict[str, str] = {}
    for probe_name, type_text in PROBE_RE.findall(output):
        target_id = probe_to_target.get(probe_name)
        if not target_id:
            continue
        qualifiers = qualifiers_by_probe.get(probe_name) if qualifiers_by_probe else None
        answers[target_id] = type_text
        families[target_id] = pyright_family_from_type_text(type_text, qualifiers)
    return families, answers


def parse_pyright_probe_output(output: str, probe_to_target: Dict[str, str]) -> Dict[str, str]:
    families, _ = _parse_pyright_probe_answers(output, probe_to_target)
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
        elif probe.mode == "before_line":
            insert_lineno = probe.insert_lineno or probe.lineno
            if not 1 <= insert_lineno <= len(lines):
                continue
            prefix = lines[insert_lineno - 1][: probe.insert_col]
            if prefix.strip():
                # The statement shares its line with something else, as in
                # ``if x: return y``. Inserting in front of it would not parse,
                # and a probe that cannot be placed correctly is better dropped
                # than placed somewhere it reports the wrong thing.
                continue
            inserts_before[insert_lineno].extend(
                [
                    f"{prefix}{probe_name} = {probe.expression}",
                    f"{prefix}reveal_type({probe_name})",
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


def _write_probe_config(
    temp_root: Path,
    include_paths: Sequence[Path],
    extra_paths: Sequence[Path] = (),
    venv_path: Optional[Path] = None,
    venv_name: str = "",
) -> Path:
    config_path = temp_root / "pyrightconfig.json"
    include = sorted({path.as_posix() for path in include_paths})
    config: Dict[str, object] = {
        "include": include,
        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
        "typeCheckingMode": "basic",
        "reportMissingImports": "none",
        "reportMissingModuleSource": "none",
    }
    # Without these the sandbox is a bare directory with no relationship to the
    # analyzed project's environment, so every third-party type -- which is
    # most of what this analysis wants to read -- resolves only by accident.
    if extra_paths:
        config["extraPaths"] = sorted({path.as_posix() for path in extra_paths})
    if venv_path is not None and venv_name:
        config["venvPath"] = str(venv_path)
        config["venv"] = venv_name
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def _discover_project_venv(project_root: Path) -> Tuple[Optional[Path], str]:
    """The analyzed project's own virtualenv, if it keeps one beside its source.

    Returned as ``(parent directory, name)`` because that is the shape pyright's
    ``venvPath``/``venv`` pair wants. The sandbox is a temp directory, so
    pyright's own discovery has nothing to find and must be told.
    """
    for name in (".venv", "venv"):
        candidate = project_root / name
        if (candidate / "bin" / "python").exists() or (candidate / "Scripts" / "python.exe").exists():
            return project_root, name
    return None, ""


@dataclass(frozen=True)
class PyrightProbeReport:
    """Families, plus enough counting to tell a failed probe from a typed unknown.

    Everything in this subsystem degrades to ``FAMILY_UNKNOWN`` rather than
    raising, and ``FAMILY_UNKNOWN`` is also a legitimate answer, so a total
    failure and a correctly-typed unknown are indistinguishable in the output.
    These counts are the difference.
    """

    families: Dict[str, str]
    probes_emitted: int = 0
    probes_answered: int = 0
    answers_unknown: int = 0
    support_files_copied: int = 0
    files_outside_project_root: int = 0

    @property
    def probes_resolved(self) -> int:
        """Answers that were an actual type rather than pyright's ``Unknown``."""
        return self.probes_answered - self.answers_unknown


def _copy_support_file(project_root: Path, temp_root: Path, file_path: Path) -> bool:
    """Copy one analyzed file into the sandbox at its project-relative path.

    Returns False for a file outside ``project_root``. ``config.py`` accepts
    absolute source roots with no requirement that they sit under the project
    root, and this used to raise ``ValueError`` out of the whole run -- the one
    place in this subsystem that failed loudly instead of degrading.
    """
    try:
        rel_path = file_path.relative_to(project_root)
    except ValueError:
        return False
    temp_file = temp_root / rel_path
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file_path, temp_file)
    return True


def probe_pyright_targets(
    project_root: Path,
    targets: Iterable[PyrightProbeTarget],
    pyright_bin: str,
    *,
    support_files: Sequence[Path] = (),
    extra_paths: Sequence[Path] = (),
    module_imports: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> PyrightProbeReport:
    """Ask pyright for the type of every probe target.

    ``support_files`` are copied into the sandbox unmodified so that imports
    resolve. They matter more than they sound: a module with no probe target of
    its own -- a package ``__init__.py`` of pure re-exports is the common case --
    used never to reach the sandbox at all, so every import routed through it
    failed, so every type behind it came back ``Unknown``. ``reportMissingImports``
    is off, so nothing said so.

    ``module_imports`` maps each probed module to its ``{name: what it was
    imported from}``. Pyright prints a class by the bare name the source used, so
    without it ``Dataset`` cannot be told apart from any other project's class of
    that name.
    """
    project_root = project_root.resolve()
    probe_targets = list(targets)
    if not probe_targets:
        return PyrightProbeReport(families={})

    pyright_path = _resolve_pyright_bin(pyright_bin)
    numbered_targets = [(f"{PROBE_NAME_PREFIX}{idx}", target) for idx, target in enumerate(probe_targets, start=1)]
    probe_to_target = {probe_name: target.target_id for probe_name, target in numbered_targets}
    qualifiers_by_probe = {
        probe_name: module_imports[target.module]
        for probe_name, target in numbered_targets
        if module_imports and target.module in module_imports
    }

    targets_by_file: Dict[Path, List[Tuple[str, PyrightProbeTarget]]] = defaultdict(list)
    for probe_name, target in numbered_targets:
        targets_by_file[target.file.resolve()].append((probe_name, target))

    copied = 0
    outside = 0
    with tempfile.TemporaryDirectory(prefix="microservice_pipeline_pyright_") as tmpdir:
        temp_root = Path(tmpdir)
        for support_file in support_files:
            if _copy_support_file(project_root, temp_root, support_file.resolve()):
                copied += 1
            else:
                outside += 1

        include_paths: List[Path] = []
        for file_path, file_targets in targets_by_file.items():
            try:
                rel_path = file_path.relative_to(project_root)
            except ValueError:
                outside += 1
                continue
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

        venv_root, venv_name = _discover_project_venv(project_root)
        config_path = _write_probe_config(
            temp_root, include_paths, extra_paths, venv_root, venv_name
        )
        command = [pyright_path, "--project", str(config_path)]
        if venv_root is None:
            # No project virtualenv to point at, so name the interpreter running
            # this pipeline explicitly. That is the one third-party environment
            # already known to exist; relying on pyright to find it by itself
            # from a temp directory is what made this accidental.
            command.extend(["--pythonpath", sys.executable])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode not in {0, 1}:
            raise RuntimeError(output or "Pyright failed without output")

    families, answers = _parse_pyright_probe_answers(output, probe_to_target, qualifiers_by_probe)
    return PyrightProbeReport(
        families=families,
        probes_emitted=len(probe_targets),
        probes_answered=len(answers),
        answers_unknown=sum(1 for text in answers.values() if _is_unresolved_type_text(text)),
        support_files_copied=copied,
        files_outside_project_root=outside,
    )


__all__ = [
    "PyrightProbeReport",
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
