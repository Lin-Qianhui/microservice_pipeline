#!/usr/bin/env python3
"""Generate an AST-based call graph for a Python project.

Outputs:
- nodes.csv: callable definitions discovered in scanned files
- edges.csv: caller -> callee edges with resolution metadata
- call_graph.json: combined nodes/edges payload

This extractor is static and conservative:
- It resolves direct function calls, class/static-like calls, and self/cls method calls.
- It attempts simple type inference for ``obj = ClassName(...)`` assignments.
- It does not execute code and cannot fully resolve dynamic dispatch.
"""


from __future__ import annotations

import argparse
import ast
import fnmatch
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from microservice_pipeline.call_graph.call_graph_outputs import write_outputs
    from microservice_pipeline.config import (
        ExtractionConfig,
        SourceRootConfig,
        load_extraction_config,
    )
    from microservice_pipeline.import_resolution import (
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from microservice_pipeline.call_graph.call_graph_outputs import write_outputs  # type: ignore
    from microservice_pipeline.config import (  # type: ignore
        ExtractionConfig,
        SourceRootConfig,
        load_extraction_config,
    )
    from microservice_pipeline.import_resolution import (  # type: ignore
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )

MODULE_CALLABLE_QUALNAME = "<module>"

PROPERTY_DECORATOR_NAMES = {
    "property",
    "cached_property",
    "functools.cached_property",
}

STATICMETHOD_DECORATOR_NAMES = {"staticmethod"}

CLASSMETHOD_DECORATOR_NAMES = {"classmethod"}

BINOP_DUNDER_METHODS = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.MatMult: "__matmul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitOr: "__or__",
    ast.BitXor: "__xor__",
    ast.BitAnd: "__and__",
}

REVERSE_BINOP_DUNDER_METHODS = {
    ast.Add: "__radd__",
    ast.Sub: "__rsub__",
    ast.Mult: "__rmul__",
    ast.MatMult: "__rmatmul__",
    ast.Div: "__rtruediv__",
    ast.FloorDiv: "__rfloordiv__",
    ast.Mod: "__rmod__",
    ast.Pow: "__rpow__",
    ast.LShift: "__rlshift__",
    ast.RShift: "__rrshift__",
    ast.BitOr: "__ror__",
    ast.BitXor: "__rxor__",
    ast.BitAnd: "__rand__",
}

AUGASSIGN_DUNDER_METHODS = {
    ast.Add: "__iadd__",
    ast.Sub: "__isub__",
    ast.Mult: "__imul__",
    ast.MatMult: "__imatmul__",
    ast.Div: "__itruediv__",
    ast.FloorDiv: "__ifloordiv__",
    ast.Mod: "__imod__",
    ast.Pow: "__ipow__",
    ast.LShift: "__ilshift__",
    ast.RShift: "__irshift__",
    ast.BitOr: "__ior__",
    ast.BitXor: "__ixor__",
    ast.BitAnd: "__iand__",
}

COMPARE_DUNDER_METHODS = {
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
}

UNARY_DUNDER_METHODS = {
    ast.UAdd: "__pos__",
    ast.USub: "__neg__",
    ast.Invert: "__invert__",
}


@dataclass
class CallableDef:
    id: str
    module: str
    qualname: str
    file: str
    lineno: int
    kind: str  # module | function | method
    class_name: Optional[str] = None


@dataclass
class ModuleIndex:
    module: str
    path: Path
    imports: Dict[str, str] = field(default_factory=dict)  # alias -> fully qualified target
    classes: Dict[str, str] = field(default_factory=dict)  # class name -> callable id namespace
    class_bases: Dict[str, List[str]] = field(default_factory=dict)  # class id -> base class ids
    properties: Set[str] = field(default_factory=set)  # property getter callable ids
    static_methods: Set[str] = field(default_factory=set)
    class_methods: Set[str] = field(default_factory=set)
    star_imports: List[str] = field(default_factory=list)  # modules imported via from x import *


@dataclass
class Edge:
    caller: str
    callee: str
    file: str
    lineno: int
    resolved: bool
    # direct | self_method | super_method | imported | inferred_type
    # attribute | dynamic_getattr | constructor | import
    # property_getter | dunder_getitem | dunder_setitem | dunder_delitem
    # dunder_contains | dunder_iter | dunder_operator
    relation: str


@dataclass(frozen=True)
class AnalysisFile:
    path: Path
    module: str


@dataclass
class FunctionReturnSummary:
    class_types: Set[str] = field(default_factory=set)
    element_types: Set[str] = field(default_factory=set)
    slot_types: Dict[int, Set[str]] = field(default_factory=dict)
    slot_element_types: Dict[int, Set[str]] = field(default_factory=dict)


@dataclass
class FunctionParamSummary:
    positional_types: Dict[int, Set[str]] = field(default_factory=dict)
    named_types: Dict[str, Set[str]] = field(default_factory=dict)


def path_to_module(py_file: Path, src_root: Path, module_prefix: Optional[str] = None) -> str:
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
    if module:
        return f"{module}.{MODULE_CALLABLE_QUALNAME}"
    return MODULE_CALLABLE_QUALNAME


def _posix_relative(path: Path, base: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return None


def _glob_matches(value: str, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/")
    return fnmatch.fnmatchcase(value, normalized) or (
        normalized.startswith("**/") and fnmatch.fnmatchcase(value, normalized[3:])
    )


def _matches_any(path: Path, patterns: Sequence[str], root: Path, project_root: Path) -> bool:
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


def attribute_to_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = attribute_to_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def parse_python_file(py_file: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))


def parse_python_source(source: str, filename: str = "<source>") -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source, filename=filename)


def add_types(target: Dict[str, Set[str]], key: str, types: Set[str]) -> None:
    if not types:
        return
    target.setdefault(key, set()).update(types)


def add_indexed_types(target: Dict[int, Set[str]], key: int, types: Set[str]) -> None:
    if not types:
        return
    target.setdefault(key, set()).update(types)


def copy_type_map(source: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    return {key: set(values) for key, values in source.items()}


def copy_indexed_type_map(source: Dict[int, Set[str]]) -> Dict[int, Set[str]]:
    return {key: set(values) for key, values in source.items()}


def copy_return_summaries(
    summaries: Dict[str, FunctionReturnSummary],
) -> Dict[str, FunctionReturnSummary]:
    return {
        key: FunctionReturnSummary(
            class_types=set(value.class_types),
            element_types=set(value.element_types),
            slot_types=copy_indexed_type_map(value.slot_types),
            slot_element_types=copy_indexed_type_map(value.slot_element_types),
        )
        for key, value in summaries.items()
    }


def copy_param_summaries(
    summaries: Dict[str, FunctionParamSummary],
) -> Dict[str, FunctionParamSummary]:
    return {
        key: FunctionParamSummary(
            positional_types=copy_indexed_type_map(value.positional_types),
            named_types=copy_type_map(value.named_types),
        )
        for key, value in summaries.items()
    }


def copy_class_attr_types(
    class_attr_types: Dict[Tuple[str, str], Set[str]],
) -> Dict[Tuple[str, str], Set[str]]:
    return {key: set(values) for key, values in class_attr_types.items()}


class DefinitionCollector(ast.NodeVisitor):
    def __init__(self, module: str, file: Path):
        self.module = module
        self.file = file
        self.callables: List[CallableDef] = [
            CallableDef(
                id=module_callable_id(module),
                module=module,
                qualname=MODULE_CALLABLE_QUALNAME,
                file=str(file),
                lineno=1,
                kind="module",
                class_name="",
            )
        ]
        self.module_index = ModuleIndex(module=module, path=file)
        self.current_class: Optional[str] = None
        self.current_callable: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            asname = alias.asname or alias.name.split(".")[0]
            self.module_index.imports[asname] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported_module = resolve_import_from_module(
            self.module,
            node.module,
            node.level,
            current_is_package=is_package_file(self.file),
        )
        if imported_module is None:
            return
        for alias in node.names:
            if alias.name == "*":
                if imported_module:
                    self.module_index.star_imports.append(imported_module)
                continue
            asname = alias.asname or alias.name
            target = resolve_import_from_target(
                self.module,
                node.module,
                node.level,
                alias.name,
                current_is_package=is_package_file(self.file),
            )
            if target is not None:
                self.module_index.imports[asname] = target

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = f"{self.module}.{node.name}"
        self.module_index.classes[node.name] = class_id
        bases = [
            base_id
            for base in node.bases
            if (base_id := self._resolve_class_reference(base)) is not None
        ]
        self.module_index.class_bases[class_id] = bases
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def _resolve_class_reference(self, node: ast.AST) -> Optional[str]:
        name = attribute_to_name(node)
        if not name:
            return None

        if "." not in name:
            local_class = self.module_index.classes.get(name)
            if local_class:
                return local_class
            imported = self.module_index.imports.get(name)
            if imported:
                return imported
            return f"{self.module}.{name}"

        base, _rest = name.split(".", 1)
        imported_base = self.module_index.imports.get(base)
        if imported_base:
            return name.replace(base, imported_base, 1)
        return name

    def _add_callable(
        self, name: str, lineno: int, kind: str, class_name: Optional[str] = None
    ) -> str:
        if self.current_callable is not None:
            call_id = f"{self.current_callable}.<locals>.{name}"
        elif kind == "method" and class_name:
            call_id = f"{self.module}.{class_name}.{name}"
        else:
            call_id = f"{self.module}.{name}"

        qualname = call_id[len(self.module) + 1 :]
        self.callables.append(
            CallableDef(
                id=call_id,
                module=self.module,
                qualname=qualname,
                file=str(self.file),
                lineno=lineno,
                kind=kind,
                class_name=class_name or "",
            )
        )
        return call_id

    def _visit_callable_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> None:
        parent = getattr(node, "parent", None)
        prev_callable = self.current_callable

        if self.current_callable is not None:
            next_callable = self._add_callable(
                name=name,
                lineno=node.lineno,
                kind="function",
                class_name=self.current_class,
            )
        elif isinstance(parent, ast.ClassDef):
            next_callable = self._add_callable(
                name=name,
                lineno=node.lineno,
                kind="method",
                class_name=self.current_class,
            )
        else:
            next_callable = self._add_callable(
                name=name,
                lineno=node.lineno,
                kind="function",
                class_name=None,
            )

        if isinstance(parent, ast.ClassDef):
            if self._has_decorator(node, PROPERTY_DECORATOR_NAMES):
                self.module_index.properties.add(next_callable)
            if self._has_decorator(node, STATICMETHOD_DECORATOR_NAMES):
                self.module_index.static_methods.add(next_callable)
            if self._has_decorator(node, CLASSMETHOD_DECORATOR_NAMES):
                self.module_index.class_methods.add(next_callable)

        self.current_callable = next_callable
        self.generic_visit(node)
        self.current_callable = prev_callable

    def _has_decorator(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        decorator_names: Set[str],
    ) -> bool:
        for decorator in node.decorator_list:
            decorator_expr = decorator.func if isinstance(decorator, ast.Call) else decorator
            decorator_name = attribute_to_name(decorator_expr)
            if decorator_name in decorator_names:
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable_def(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable_def(node, node.name)


class CallCollector(ast.NodeVisitor):
    def __init__(
        self,
        module: str,
        file: Path,
        module_index: ModuleIndex,
        callable_ids: Set[str],
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        include_external: bool,
        package_prefix: Optional[str] | Sequence[str],
        return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
        param_summaries: Optional[Dict[str, FunctionParamSummary]] = None,
        class_attr_types: Optional[Dict[Tuple[str, str], Set[str]]] = None,
    ):
        self.module = module
        self.file = file
        self.module_index = module_index
        self.callable_ids = callable_ids
        self.module_map = module_map
        self.known_classes = known_classes
        self.include_external = include_external
        self.package_prefix = package_prefix
        self.return_summaries = (
            return_summaries if return_summaries is not None else {}
        )
        self.param_summaries = param_summaries if param_summaries is not None else {}
        self.class_attr_types = (
            class_attr_types if class_attr_types is not None else {}
        )
        self.class_bases = {
            class_id: bases
            for module_info in module_map.values()
            for class_id, bases in module_info.class_bases.items()
        }
        self.module_callable = module_callable_id(module)
        self.property_ids = {
            property_id
            for module_info in module_map.values()
            for property_id in module_info.properties
        }
        self.static_method_ids = {
            method_id
            for module_info in module_map.values()
            for method_id in module_info.static_methods
        }
        self.class_method_ids = {
            method_id
            for module_info in module_map.values()
            for method_id in module_info.class_methods
        }

        self.current_callable: Optional[str] = None
        self.current_class: Optional[str] = None
        self.callable_stack: List[str] = []
        self.var_types_stack: List[Dict[str, Set[str]]] = []
        self.container_types_stack: List[Dict[str, Set[str]]] = []
        self.attr_types_stack: List[Dict[str, Set[str]]] = []
        self.edges: List[Edge] = []

    def _is_internal_callee(self, callee: str) -> bool:
        if not self.package_prefix:
            return True
        if isinstance(self.package_prefix, str):
            prefixes = (self.package_prefix,)
        else:
            prefixes = tuple(self.package_prefix)
        return any(callee == prefix or callee.startswith(prefix + ".") for prefix in prefixes)

    def push_scope(self) -> None:
        self.var_types_stack.append({})
        self.container_types_stack.append({})
        self.attr_types_stack.append({})

    def pop_scope(self) -> None:
        self.var_types_stack.pop()
        self.container_types_stack.pop()
        self.attr_types_stack.pop()

    def _copy_type_state(
        self,
    ) -> Tuple[
        List[Dict[str, Set[str]]],
        List[Dict[str, Set[str]]],
        List[Dict[str, Set[str]]],
    ]:
        return (
            [copy_type_map(scope) for scope in self.var_types_stack],
            [copy_type_map(scope) for scope in self.container_types_stack],
            [copy_type_map(scope) for scope in self.attr_types_stack],
        )

    def _restore_type_state(
        self,
        state: Tuple[
            List[Dict[str, Set[str]]],
            List[Dict[str, Set[str]]],
            List[Dict[str, Set[str]]],
        ],
    ) -> None:
        self.var_types_stack = [copy_type_map(scope) for scope in state[0]]
        self.container_types_stack = [copy_type_map(scope) for scope in state[1]]
        self.attr_types_stack = [copy_type_map(scope) for scope in state[2]]

    def _merge_type_maps(
        self, maps: Iterable[Dict[str, Set[str]]]
    ) -> Dict[str, Set[str]]:
        merged: Dict[str, Set[str]] = {}
        for type_map in maps:
            for key, values in type_map.items():
                add_types(merged, key, values)
        return merged

    def _merge_type_stack_states(
        self, states: Iterable[List[Dict[str, Set[str]]]]
    ) -> List[Dict[str, Set[str]]]:
        state_list = list(states)
        if not state_list:
            return []
        depth = len(state_list[0])
        return [
            self._merge_type_maps(state[index] for state in state_list)
            for index in range(depth)
        ]

    def _merge_type_states(
        self,
        states: Iterable[
            Tuple[
                List[Dict[str, Set[str]]],
                List[Dict[str, Set[str]]],
                List[Dict[str, Set[str]]],
            ]
        ],
    ) -> Tuple[
        List[Dict[str, Set[str]]],
        List[Dict[str, Set[str]]],
        List[Dict[str, Set[str]]],
    ]:
        state_list = list(states)
        return (
            self._merge_type_stack_states(state[0] for state in state_list),
            self._merge_type_stack_states(state[1] for state in state_list),
            self._merge_type_stack_states(state[2] for state in state_list),
        )

    def set_var_type(self, var: str, typ: str) -> None:
        self.set_var_types(var, {typ})

    def set_var_types(self, var: str, types: Set[str]) -> None:
        if self.var_types_stack:
            self.var_types_stack[-1][var] = set(types)

    def get_var_types(self, var: str) -> Set[str]:
        for scope in reversed(self.var_types_stack):
            if var in scope:
                return set(scope[var])
        return set()

    def set_container_types(self, var: str, types: Set[str]) -> None:
        if self.container_types_stack:
            self.container_types_stack[-1][var] = set(types)

    def add_container_types(self, var: str, types: Set[str]) -> None:
        if not self.container_types_stack or not types:
            return
        existing = self.container_types_stack[-1].setdefault(var, set())
        existing.update(types)

    def get_container_types(self, var: str) -> Set[str]:
        for scope in reversed(self.container_types_stack):
            if var in scope:
                return set(scope[var])
        return set()

    def get_var_type(self, var: str) -> Optional[str]:
        types = sorted(self.get_var_types(var))
        if types:
            return types[0]
        return None

    def set_attr_expr_types(self, attr_expr: str, types: Set[str]) -> None:
        if self.attr_types_stack and types:
            add_types(self.attr_types_stack[-1], attr_expr, types)

    def get_attr_expr_types(self, attr_expr: str) -> Set[str]:
        for scope in reversed(self.attr_types_stack):
            if attr_expr in scope:
                return set(scope[attr_expr])
        return set()

    def get_expr_types(self, expr: ast.AST) -> Set[str]:
        if isinstance(expr, ast.Name):
            return self.get_var_types(expr.id)
        if isinstance(expr, ast.Attribute):
            expr_name = attribute_to_name(expr)
            if expr_name:
                local_types = self.get_attr_expr_types(expr_name)
                if local_types:
                    return local_types

            base_types: Set[str] = set()
            if (
                isinstance(expr.value, ast.Name)
                and expr.value.id in {"self", "cls"}
                and self.current_class
            ):
                class_id = self.current_class_id()
                if class_id:
                    base_types.add(class_id)
            base_types.update(self.get_expr_types(expr.value))

            attr_types: Set[str] = set()
            for base_type in base_types:
                attr_types.update(
                    self.class_attr_types.get((base_type, expr.attr), set())
                )
            attr_types.update(self._infer_property_return_class_types(expr))
            return attr_types
        return set()

    def _receiver_types_for_expr(self, expr: ast.AST) -> Set[str]:
        receiver_types: Set[str] = set()
        if (
            isinstance(expr, ast.Name)
            and expr.id in {"self", "cls"}
            and self.current_class
        ):
            class_id = self.current_class_id()
            if class_id:
                receiver_types.add(class_id)
        receiver_types.update(self.get_expr_types(expr))
        return receiver_types

    def _resolve_property_getter_targets(self, node: ast.Attribute) -> List[str]:
        targets: List[str] = []
        for receiver_type in sorted(self._receiver_types_for_expr(node.value)):
            for target in self._resolve_method_targets(receiver_type, node.attr):
                if target in self.property_ids:
                    targets.append(target)
        return self._unique(targets)

    def _infer_property_return_class_types(self, node: ast.Attribute) -> Set[str]:
        class_types: Set[str] = set()
        for getter in self._resolve_property_getter_targets(node):
            summary = self.return_summaries.get(getter)
            if summary:
                class_types.update(summary.class_types)
        return class_types

    def _infer_property_return_element_types(self, node: ast.Attribute) -> Set[str]:
        element_types: Set[str] = set()
        for getter in self._resolve_property_getter_targets(node):
            summary = self.return_summaries.get(getter)
            if summary:
                element_types.update(summary.element_types)
        return element_types

    def set_attribute_types(self, target: ast.Attribute, types: Set[str]) -> None:
        if not types:
            return

        attr_name = attribute_to_name(target)
        if attr_name:
            self.set_attr_expr_types(attr_name, types)

        base_types: Set[str] = set()
        if (
            isinstance(target.value, ast.Name)
            and target.value.id in {"self", "cls"}
            and self.current_class
        ):
            class_id = self.current_class_id()
            if class_id:
                base_types.add(class_id)
        base_types.update(self.get_expr_types(target.value))

        for base_type in base_types:
            self.class_attr_types.setdefault((base_type, target.attr), set()).update(types)

    def current_class_id(self) -> Optional[str]:
        if self.current_class:
            return f"{self.module}.{self.current_class}"
        return None

    def _is_known_class(self, class_id: str) -> bool:
        return (
            class_id in self.known_classes
            or class_id in self.known_classes.values()
        )

    def _known_class_id(self, class_id: str) -> str:
        return self.known_classes.get(class_id, class_id)

    def _unique(self, values: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _resolve_class_reference_name(self, name: str) -> List[str]:
        matches: Set[str] = set()

        if "." not in name:
            local_class = self.module_index.classes.get(name)
            if local_class:
                matches.add(local_class)

            imported = self.module_index.imports.get(name)
            if imported and self._is_known_class(imported):
                matches.add(self._known_class_id(imported))

            for imported in self._resolve_star_import_targets(name):
                if self._is_known_class(imported):
                    matches.add(self._known_class_id(imported))

            local_name = f"{self.module}.{name}"
            if self._is_known_class(local_name):
                matches.add(self._known_class_id(local_name))

            return sorted(matches)

        base, _rest = name.split(".", 1)
        imported_base = self.module_index.imports.get(base)
        if imported_base:
            candidate = name.replace(base, imported_base, 1)
            if self._is_known_class(candidate):
                matches.add(self._known_class_id(candidate))

        if self._is_known_class(name):
            matches.add(self._known_class_id(name))

        return sorted(matches)

    def _resolve_method_targets(
        self, class_id: str, method_name: str, seen: Optional[Set[str]] = None
    ) -> List[str]:
        if seen is None:
            seen = set()
        if class_id in seen:
            return []
        seen.add(class_id)

        direct = f"{class_id}.{method_name}"
        if direct in self.callable_ids:
            return [direct]

        targets: List[str] = []
        for base_id in self.class_bases.get(class_id, []):
            if not self._is_known_class(base_id):
                continue
            targets.extend(
                self._resolve_method_targets(
                    self._known_class_id(base_id), method_name, seen=seen.copy()
                )
            )
        return self._unique(targets)

    def _resolve_constructor_targets(self, class_id: str) -> List[str]:
        return self._resolve_method_targets(class_id, "__init__")

    def _resolve_super_method_targets(self, method_name: str) -> List[str]:
        class_id = self.current_class_id()
        if not class_id:
            return []

        targets: List[str] = []
        for base_id in self.class_bases.get(class_id, []):
            if not self._is_known_class(base_id):
                continue
            targets.extend(
                self._resolve_method_targets(
                    self._known_class_id(base_id), method_name
                )
            )
        return self._unique(targets)

    def _add_edge(
        self, callee: str, relation: str, is_resolved: bool, lineno: int
    ) -> None:
        if self.current_callable is None:
            return
        if not is_resolved and not self.include_external:
            return
        if is_resolved and not self._is_internal_callee(callee):
            return
        self.edges.append(
            Edge(
                caller=self.current_callable,
                callee=callee,
                file=str(self.file),
                lineno=lineno,
                resolved=is_resolved,
                relation=relation,
            )
        )

    def _unique_callee_results(
        self, values: Iterable[Tuple[str, str, bool]]
    ) -> List[Tuple[str, str, bool]]:
        seen: Set[Tuple[str, str, bool]] = set()
        ordered: List[Tuple[str, str, bool]] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _resolve_method_callees_for_expr(
        self, receiver: ast.AST, method_name: str, relation: str
    ) -> List[Tuple[str, str, bool]]:
        targets: List[Tuple[str, str, bool]] = []
        for receiver_type in sorted(self._receiver_types_for_expr(receiver)):
            resolved_targets = self._resolve_method_targets(receiver_type, method_name)
            if resolved_targets:
                targets.extend(
                    (target, relation, True) for target in resolved_targets
                )
            else:
                targets.append((f"{receiver_type}.{method_name}", relation, False))
        return self._unique_callee_results(targets)

    def _callable_belongs_to_known_class(self, callable_id: str) -> bool:
        return any(
            callable_id.startswith(f"{class_id}.")
            for class_id in self.known_classes.values()
        )

    def _implicit_receiver_arg_offset(self, callee: str, relation: str) -> int:
        if callee in self.static_method_ids:
            return 0
        if relation == "constructor":
            return 1
        if relation == "class_method":
            return 1 if callee in self.class_method_ids else 0
        if relation in {"self_method", "super_method", "inferred_type"}:
            if self._callable_belongs_to_known_class(callee):
                return 1
        return 0

    def _add_dunder_edges(
        self, receiver: ast.AST, method_name: str, relation: str, lineno: int
    ) -> None:
        for callee, edge_relation, is_resolved in self._resolve_method_callees_for_expr(
            receiver, method_name, relation
        ):
            self._add_edge(callee, edge_relation, is_resolved, lineno)

    def _add_membership_edges(self, container: ast.AST, lineno: int) -> None:
        targets: List[Tuple[str, str, bool]] = []
        for receiver_type in sorted(self._receiver_types_for_expr(container)):
            contains_targets = self._resolve_method_targets(receiver_type, "__contains__")
            if contains_targets:
                targets.extend(
                    (target, "dunder_contains", True)
                    for target in contains_targets
                )
                continue

            iter_targets = self._resolve_method_targets(receiver_type, "__iter__")
            if iter_targets:
                targets.extend(
                    (target, "dunder_iter", True) for target in iter_targets
                )
            else:
                targets.append(
                    (f"{receiver_type}.__contains__", "dunder_contains", False)
                )

        for callee, relation, is_resolved in self._unique_callee_results(targets):
            self._add_edge(callee, relation, is_resolved, lineno)

    def _visit_call_children(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)
        else:
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def _call_is_empty_container(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"list", "set", "tuple"}
            and not node.args
        )

    def _is_container_literal(self, node: ast.AST) -> bool:
        return isinstance(
            node, (ast.List, ast.Set, ast.Tuple, ast.ListComp, ast.SetComp)
        ) or self._call_is_empty_container(node)

    def _assign_target_types(
        self,
        target: ast.AST,
        class_types: Set[str],
        container_types: Set[str],
        value: ast.AST,
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                self._assign_target_types(
                    item,
                    self._infer_sequence_slot_class_types(value, index),
                    self._infer_sequence_slot_container_types(value, index),
                    value,
                )
            return

        if isinstance(target, ast.Name) and self.var_types_stack:
            if class_types:
                self.set_var_types(target.id, class_types)
            if container_types or self._is_container_literal(value):
                self.set_container_types(target.id, container_types)
            return

        if isinstance(target, ast.Attribute):
            self.set_attribute_types(target, class_types)

    def visit_Module(self, node: ast.Module) -> None:
        prev_callable = self.current_callable
        self.current_callable = self.module_callable
        self.callable_stack.append(self.module_callable)
        self.push_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable

    def _add_import_edge(self, module_name: str, lineno: int) -> None:
        imported_module_callable = module_callable_id(module_name)
        if imported_module_callable in self.callable_ids:
            self._add_edge(imported_module_callable, "import", True, lineno)
        else:
            self._add_edge(module_name, "import", False, lineno)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_import_edge(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported_module = resolve_import_from_module(
            self.module,
            node.module,
            node.level,
            current_is_package=is_package_file(self.file),
        )
        if imported_module is None:
            return
        if node.module:
            self._add_import_edge(imported_module, node.lineno)
            return
        for alias in node.names:
            target = resolve_import_from_target(
                self.module,
                node.module,
                node.level,
                alias.name,
                current_is_package=is_package_file(self.file),
            )
            if target:
                self._add_import_edge(target, node.lineno)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev

    def _enter_callable(self, node: ast.AST, name: str) -> None:
        prev_callable = self.current_callable
        if prev_callable is not None and prev_callable != self.module_callable:
            self.current_callable = f"{prev_callable}.<locals>.{name}"
        elif self.current_class:
            self.current_callable = f"{self.module}.{self.current_class}.{name}"
        else:
            self.current_callable = f"{self.module}.{name}"

        self.callable_stack.append(self.current_callable)
        self.push_scope()

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if self.current_class and node.args.args:
                first_arg = node.args.args[0].arg
                if first_arg in {"self", "cls"}:
                    cls_id = f"{self.module}.{self.current_class}"
                    self.set_var_type(first_arg, cls_id)
            param_summary = self.param_summaries.get(self.current_callable)
            if param_summary:
                positional_args = [*node.args.posonlyargs, *node.args.args]
                for index, arg in enumerate(positional_args):
                    param_types = set(param_summary.positional_types.get(index, set()))
                    param_types.update(param_summary.named_types.get(arg.arg, set()))
                    if param_types:
                        self.set_var_types(arg.arg, param_types)
                for index, arg in enumerate(
                    node.args.kwonlyargs, start=len(positional_args)
                ):
                    param_types = set(param_summary.positional_types.get(index, set()))
                    param_types.update(param_summary.named_types.get(arg.arg, set()))
                    if param_types:
                        self.set_var_types(arg.arg, param_types)

        self.generic_visit(node)
        self.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable

    def _resolve_enclosing_local_callable(self, name: str) -> Optional[str]:
        for scope_callable in reversed(self.callable_stack):
            candidate = f"{scope_callable}.<locals>.{name}"
            if candidate in self.callable_ids:
                return candidate
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_callable(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_callable(node, node.name)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        base_state = self._copy_type_state()

        self._restore_type_state(base_state)
        for stmt in node.body:
            self.visit(stmt)
        body_state = self._copy_type_state()

        self._restore_type_state(base_state)
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
        orelse_state = self._copy_type_state()

        self._restore_type_state(self._merge_type_states([body_state, orelse_state]))

    def visit_Assign(self, node: ast.Assign) -> None:
        class_types = self._infer_class_types_from_value(node.value)
        container_types = self._infer_container_element_types(node.value)
        for target in node.targets:
            self._assign_target_types(target, class_types, container_types, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            class_types = self._infer_class_types_from_value(node.value)
            container_types = self._infer_container_element_types(node.value)
            self._assign_target_types(node.target, class_types, container_types, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            self.add_container_types(
                node.target.id, self._infer_container_element_types(node.value)
            )
        method_name = AUGASSIGN_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.target, method_name, "dunder_operator", node.lineno
            )
        self.visit(node.target)
        self.visit(node.value)

    def _visit_for_like(self, node: ast.For | ast.AsyncFor) -> None:
        self._add_dunder_edges(node.iter, "__iter__", "dunder_iter", node.lineno)
        self.visit(node.iter)
        if isinstance(node.target, ast.Name):
            element_types = self._infer_container_element_types(node.iter)
            if element_types:
                self.set_var_types(node.target.id, element_types)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for_like(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for_like(node)

    def _record_container_mutation(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        if not isinstance(node.func.value, ast.Name):
            return

        container_name = node.func.value.id
        if node.func.attr == "append" and node.args:
            self.add_container_types(
                container_name, self._infer_class_types_from_value(node.args[0])
            )
        elif node.func.attr == "extend" and node.args:
            self.add_container_types(
                container_name, self._infer_container_element_types(node.args[0])
            )

    def _infer_class_types_from_value(self, value: ast.AST) -> Set[str]:
        if isinstance(value, ast.Call):
            return self._infer_class_types_from_call(value)
        if isinstance(value, (ast.Name, ast.Attribute)):
            return self.get_expr_types(value)
        return set()

    def _infer_class_types_from_call(self, value: ast.Call) -> Set[str]:
        if self._is_copy_call(value) and value.args:
            return self._infer_class_types_from_value(value.args[0])

        fn_name = attribute_to_name(value.func)
        if fn_name is None:
            return set()

        class_matches = self._resolve_class_reference_name(fn_name)
        if class_matches:
            return set(class_matches)

        class_types: Set[str] = set()
        for callee, _relation, is_resolved in self._resolve_callees(value.func):
            if not is_resolved:
                continue
            summary = self.return_summaries.get(callee)
            if summary:
                class_types.update(summary.class_types)
        return class_types

    def _infer_container_element_types(self, value: ast.AST) -> Set[str]:
        if isinstance(value, ast.Name):
            return self.get_container_types(value.id)
        if isinstance(value, ast.Attribute):
            return self._infer_property_return_element_types(value)
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            element_types: Set[str] = set()
            for item in value.elts:
                element_types.update(self._infer_class_types_from_value(item))
            return element_types
        if isinstance(value, (ast.ListComp, ast.SetComp)):
            return self._infer_class_types_from_value(value.elt)
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            return self._infer_container_element_types(
                value.left
            ) | self._infer_container_element_types(value.right)
        if isinstance(value, ast.Call):
            if self._is_copy_call(value) and value.args:
                return self._infer_container_element_types(value.args[0])
            if (
                isinstance(value.func, ast.Name)
                and value.func.id in {"list", "set", "tuple"}
            ):
                if value.args:
                    return self._infer_container_element_types(value.args[0])
                return set()

            element_types: Set[str] = set()
            for callee, _relation, is_resolved in self._resolve_callees(value.func):
                if not is_resolved:
                    continue
                summary = self.return_summaries.get(callee)
                if summary:
                    element_types.update(summary.element_types)
            return element_types
        return set()

    def _infer_sequence_slot_class_types(self, value: ast.AST, index: int) -> Set[str]:
        if isinstance(value, (ast.Tuple, ast.List)) and index < len(value.elts):
            return self._infer_class_types_from_value(value.elts[index])
        if isinstance(value, ast.Call):
            slot_types: Set[str] = set()
            for callee, _relation, is_resolved in self._resolve_callees(value.func):
                if not is_resolved:
                    continue
                summary = self.return_summaries.get(callee)
                if summary:
                    slot_types.update(summary.slot_types.get(index, set()))
            return slot_types
        return set()

    def _infer_sequence_slot_container_types(
        self, value: ast.AST, index: int
    ) -> Set[str]:
        if isinstance(value, (ast.Tuple, ast.List)) and index < len(value.elts):
            return self._infer_container_element_types(value.elts[index])
        if isinstance(value, ast.Call):
            slot_types: Set[str] = set()
            for callee, _relation, is_resolved in self._resolve_callees(value.func):
                if not is_resolved:
                    continue
                summary = self.return_summaries.get(callee)
                if summary:
                    slot_types.update(summary.slot_element_types.get(index, set()))
            return slot_types
        return set()

    def _is_copy_call(self, node: ast.Call) -> bool:
        fn_name = attribute_to_name(node.func)
        return fn_name in {"copy", "deepcopy", "copy.copy", "copy.deepcopy"}

    def _infer_class_from_call(self, value: ast.AST) -> Optional[str]:
        if not isinstance(value, ast.Call):
            return None
        inferred = sorted(self._infer_class_types_from_call(value))
        if inferred:
            return inferred[0]
        return None

    def visit_Call(self, node: ast.Call) -> None:
        self._record_container_mutation(node)

        dynamic_resolved = self._resolve_dynamic_getattr_callees(node.func)
        resolved_callees = dynamic_resolved or self._resolve_callees(node.func)
        for callee, relation, is_resolved in resolved_callees:
            self._add_edge(callee, relation, is_resolved, node.lineno)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            for callee in self._resolve_property_getter_targets(node):
                self._add_edge(callee, "property_getter", True, node.lineno)
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Store):
            self._add_dunder_edges(
                node.value, "__setitem__", "dunder_setitem", node.lineno
            )
        elif isinstance(node.ctx, ast.Del):
            self._add_dunder_edges(
                node.value, "__delitem__", "dunder_delitem", node.lineno
            )
        else:
            self._add_dunder_edges(
                node.value, "__getitem__", "dunder_getitem", node.lineno
            )
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        method_name = BINOP_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.left, method_name, "dunder_operator", node.lineno
            )
        reverse_method_name = REVERSE_BINOP_DUNDER_METHODS.get(type(node.op))
        if reverse_method_name:
            self._add_dunder_edges(
                node.right, reverse_method_name, "dunder_operator", node.lineno
            )
        self.visit(node.left)
        self.visit(node.right)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        self.visit(left)
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                self._add_membership_edges(comparator, node.lineno)
            else:
                method_name = COMPARE_DUNDER_METHODS.get(type(op))
                if method_name:
                    self._add_dunder_edges(
                        left, method_name, "dunder_operator", node.lineno
                    )
            self.visit(comparator)
            left = comparator

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        method_name = UNARY_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.operand, method_name, "dunder_operator", node.lineno
            )
        self.visit(node.operand)

    def _resolve_dynamic_getattr_callees(
        self, func: ast.AST
    ) -> List[Tuple[str, str, bool]]:
        """Resolve calls like getattr(module_alias, dynamic_name)(...) conservatively."""
        if not isinstance(func, ast.Call):
            return []
        if not isinstance(func.func, ast.Name) or func.func.id != "getattr":
            return []
        if len(func.args) < 2:
            return []

        target_obj, attr_arg = func.args[0], func.args[1]
        if not isinstance(target_obj, ast.Name):
            return []

        imported_base = self.module_index.imports.get(target_obj.id)
        if not imported_base:
            return []

        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
            target = f"{imported_base}.{attr_arg.value}"
            return [(target, "dynamic_getattr", target in self.callable_ids)]

        prefix = f"{imported_base}."
        matches = sorted(
            callable_id
            for callable_id in self.callable_ids
            if callable_id.startswith(prefix)
        )
        return [(callable_id, "dynamic_getattr", True) for callable_id in matches]

    def _resolve_name_from_module_exports(
        self, module_name: str, name: str, seen: Optional[Set[str]] = None
    ) -> List[str]:
        if seen is None:
            seen = set()
        if module_name in seen:
            return []
        seen.add(module_name)

        matches: Set[str] = set()
        module_index = self.module_map.get(module_name)
        local_name = f"{module_name}.{name}"

        if local_name in self.callable_ids:
            matches.add(local_name)
        if module_index and name in module_index.classes:
            matches.add(module_index.classes[name])
        elif local_name in self.known_classes.values() or local_name in self.known_classes:
            matches.add(self.known_classes.get(local_name, local_name))

        if not module_index:
            return sorted(matches)

        imported = module_index.imports.get(name)
        if imported:
            matches.add(self.known_classes.get(imported, imported))

        for imported_module in module_index.star_imports:
            matches.update(
                self._resolve_name_from_module_exports(
                    imported_module, name, seen=seen.copy()
                )
            )

        return sorted(matches)

    def _resolve_star_import_targets(self, name: str) -> List[str]:
        matches: Set[str] = set()
        for imported_module in self.module_index.star_imports:
            matches.update(self._resolve_name_from_module_exports(imported_module, name))
        return sorted(matches)

    def _resolve_callees(self, func: ast.AST) -> List[Tuple[str, str, bool]]:
        if isinstance(func, ast.Attribute):
            full = attribute_to_name(func)
            if not full:
                return []

            if self._is_super_call(func.value):
                targets = self._resolve_super_method_targets(func.attr)
                if targets:
                    return [(target, "super_method", True) for target in targets]

            if (
                isinstance(func.value, ast.Name)
                and func.value.id in {"self", "cls"}
                and self.current_class
            ):
                class_id = self.current_class_id()
                if class_id:
                    targets = self._resolve_method_targets(class_id, func.attr)
                    if targets:
                        return [(target, "self_method", True) for target in targets]
                    return [(f"{class_id}.{func.attr}", "self_method", False)]

            if isinstance(func.value, ast.Name):
                var_types = self.get_var_types(func.value.id)
                if var_types:
                    targets: List[str] = []
                    for var_type in sorted(var_types):
                        targets.extend(self._resolve_method_targets(var_type, func.attr))
                    if targets:
                        return [
                            (target, "inferred_type", True)
                            for target in self._unique(targets)
                        ]
                    first_type = sorted(var_types)[0]
                    return [(f"{first_type}.{func.attr}", "inferred_type", False)]

            receiver_types = self.get_expr_types(func.value)
            if receiver_types:
                targets = []
                for receiver_type in sorted(receiver_types):
                    targets.extend(
                        self._resolve_method_targets(receiver_type, func.attr)
                    )
                if targets:
                    return [
                        (target, "inferred_type", True)
                        for target in self._unique(targets)
                    ]
                first_type = sorted(receiver_types)[0]
                return [(f"{first_type}.{func.attr}", "inferred_type", False)]

            receiver_name = attribute_to_name(func.value)
            if receiver_name:
                class_ids = self._resolve_class_reference_name(receiver_name)
                if class_ids:
                    targets = []
                    for class_id in class_ids:
                        targets.extend(self._resolve_method_targets(class_id, func.attr))
                    if targets:
                        return [
                            (target, "class_method", True)
                            for target in self._unique(targets)
                        ]

        resolved = self._resolve_callee(func)
        if resolved:
            return [resolved]
        return []

    def _is_super_call(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "super"
        )

    def _resolve_callee(self, func: ast.AST) -> Optional[Tuple[str, str, bool]]:
        if isinstance(func, ast.Name):
            name = func.id
            enclosing_local = self._resolve_enclosing_local_callable(name)
            if enclosing_local:
                return (enclosing_local, "direct", True)

            local = f"{self.module}.{name}"
            if local in self.callable_ids:
                return (local, "direct", True)

            class_matches = self._resolve_class_reference_name(name)
            if class_matches:
                constructor_targets: List[str] = []
                for class_id in class_matches:
                    constructor_targets.extend(
                        self._resolve_constructor_targets(class_id)
                    )
                if len(constructor_targets) == 1:
                    return (constructor_targets[0], "constructor", True)
                if len(constructor_targets) > 1:
                    return (self._unique(constructor_targets)[0], "constructor", False)

            imported = self.module_index.imports.get(name)
            if imported:
                is_known = imported in self.callable_ids
                return (imported, "imported", is_known)

            star_import_targets = self._resolve_star_import_targets(name)
            star_import_matches = [
                target for target in star_import_targets if target in self.callable_ids
            ]
            if len(star_import_matches) == 1:
                return (star_import_matches[0], "imported", True)
            if len(star_import_matches) > 1:
                return (star_import_matches[0], "imported", False)

            star_import_class_matches = []
            for candidate_class in star_import_targets:
                if self._is_known_class(candidate_class):
                    class_id = self._known_class_id(candidate_class)
                    star_import_class_matches.extend(
                        self._resolve_constructor_targets(class_id)
                    )
            if len(star_import_class_matches) == 1:
                return (star_import_class_matches[0], "constructor", True)
            if len(star_import_class_matches) > 1:
                return (
                    self._unique(star_import_class_matches)[0],
                    "constructor",
                    False,
                )

            return (name, "direct", False)

        if isinstance(func, ast.Attribute):
            full = attribute_to_name(func)
            if not full:
                return None

            # variable type inferred method call: obj.method()
            if isinstance(func.value, ast.Name):
                var_type = self.get_var_type(func.value.id)
                if var_type:
                    inferred_targets = self._resolve_method_targets(var_type, func.attr)
                    if inferred_targets:
                        return (inferred_targets[0], "inferred_type", True)
                    return (f"{var_type}.{func.attr}", "inferred_type", False)

                # imported module alias call: alias.fn()
                imported_base = self.module_index.imports.get(func.value.id)
                if imported_base:
                    target = f"{imported_base}.{func.attr}"
                    if self._is_known_class(target):
                        class_id = self._known_class_id(target)
                        init_targets = self._resolve_constructor_targets(class_id)
                        if init_targets:
                            return (init_targets[0], "constructor", True)
                    is_known = target in self.callable_ids
                    return (target, "imported", is_known)

            # nested imported alias: alias.sub.fn()
            if "." in full:
                base, _attr = full.split(".", 1)
                imported_base = self.module_index.imports.get(base)
                if imported_base:
                    target = full.replace(base, imported_base, 1)
                    if self._is_known_class(target):
                        class_id = self._known_class_id(target)
                        init_targets = self._resolve_constructor_targets(class_id)
                        if init_targets:
                            return (init_targets[0], "constructor", True)
                    is_known = target in self.callable_ids
                    return (target, "imported", is_known)

            # maybe fully qualified local reference
            is_known = full in self.callable_ids
            return (full, "attribute", is_known)

        return None


class ReturnSummaryCollector(CallCollector):
    def __init__(
        self,
        module: str,
        file: Path,
        module_index: ModuleIndex,
        callable_ids: Set[str],
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
    ):
        super().__init__(
            module=module,
            file=file,
            module_index=module_index,
            callable_ids=callable_ids,
            module_map=module_map,
            known_classes=known_classes,
            include_external=False,
            package_prefix=None,
            return_summaries=return_summaries,
        )
        self.collected_summaries: Dict[str, FunctionReturnSummary] = {}

    def visit_Call(self, node: ast.Call) -> None:
        self._record_container_mutation(node)
        self._visit_call_children(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self.current_callable and node.value is not None:
            summary = self.collected_summaries.setdefault(
                self.current_callable, FunctionReturnSummary()
            )
            summary.class_types.update(self._infer_class_types_from_value(node.value))
            summary.element_types.update(self._infer_container_element_types(node.value))
            if isinstance(node.value, (ast.Tuple, ast.List)):
                for index, item in enumerate(node.value.elts):
                    add_indexed_types(
                        summary.slot_types,
                        index,
                        self._infer_class_types_from_value(item),
                    )
                    add_indexed_types(
                        summary.slot_element_types,
                        index,
                        self._infer_container_element_types(item),
                    )
        self.generic_visit(node)


class TypeSummaryCollector(CallCollector):
    def __init__(
        self,
        module: str,
        file: Path,
        module_index: ModuleIndex,
        callable_ids: Set[str],
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        return_summaries: Dict[str, FunctionReturnSummary],
        param_summaries: Dict[str, FunctionParamSummary],
        class_attr_types: Dict[Tuple[str, str], Set[str]],
    ):
        super().__init__(
            module=module,
            file=file,
            module_index=module_index,
            callable_ids=callable_ids,
            module_map=module_map,
            known_classes=known_classes,
            include_external=False,
            package_prefix=None,
            return_summaries=return_summaries,
            param_summaries=param_summaries,
            class_attr_types=class_attr_types,
        )

    def visit_Call(self, node: ast.Call) -> None:
        self._record_container_mutation(node)

        dynamic_resolved = self._resolve_dynamic_getattr_callees(node.func)
        resolved_callees = dynamic_resolved or self._resolve_callees(node.func)
        for callee, relation, is_resolved in resolved_callees:
            if not is_resolved:
                continue
            summary = self.param_summaries.setdefault(callee, FunctionParamSummary())
            positional_offset = self._implicit_receiver_arg_offset(callee, relation)
            for index, arg in enumerate(node.args):
                if isinstance(arg, ast.Starred):
                    continue
                add_indexed_types(
                    summary.positional_types,
                    index + positional_offset,
                    self._infer_class_types_from_value(arg),
                )
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                add_types(
                    summary.named_types,
                    keyword.arg,
                    self._infer_class_types_from_value(keyword.value),
                )

        self.generic_visit(node)


def attach_parents(node: ast.AST) -> None:
    for child in ast.iter_child_nodes(node):
        setattr(child, "parent", node)
        attach_parents(child)


def iter_analysis_files_for_source_roots(
    source_roots: Sequence[SourceRootConfig],
    *,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
) -> List[AnalysisFile]:
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


def build_indices_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
) -> Tuple[Dict[str, CallableDef], Dict[str, ModuleIndex], Dict[str, str]]:
    callable_map: Dict[str, CallableDef] = {}
    module_map: Dict[str, ModuleIndex] = {}
    known_classes: Dict[str, str] = {}

    for analysis_file in analysis_files:
        py_file = analysis_file.path
        module = analysis_file.module
        tree = parse_python_file(py_file)
        attach_parents(tree)

        defs = DefinitionCollector(module=module, file=py_file)
        defs.visit(tree)

        module_map[module] = defs.module_index
        for c in defs.callables:
            callable_map[c.id] = c
        for _class_name, class_id in defs.module_index.classes.items():
            known_classes[class_id] = class_id

    return callable_map, module_map, known_classes


def build_indices(
    src_root: Path,
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
) -> Tuple[Dict[str, CallableDef], Dict[str, ModuleIndex], Dict[str, str]]:
    return build_indices_from_analysis_files(
        list(
            iter_analysis_files(
                src_root,
                module_prefix=module_prefix,
                entrypoints=entrypoints,
                project_root=project_root,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        )
    )


def build_return_summaries_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    max_iterations: int = 3,
) -> Dict[str, FunctionReturnSummary]:
    summaries: Dict[str, FunctionReturnSummary] = {}
    callable_ids = set(callable_map.keys())

    for _ in range(max_iterations):
        collected: Dict[str, FunctionReturnSummary] = {}

        for analysis_file in analysis_files:
            py_file = analysis_file.path
            module = analysis_file.module
            tree = parse_python_file(py_file)
            collector = ReturnSummaryCollector(
                module=module,
                file=py_file,
                module_index=module_map[module],
                callable_ids=callable_ids,
                module_map=module_map,
                known_classes=known_classes,
                return_summaries=summaries,
            )
            collector.visit(tree)
            collected.update(collector.collected_summaries)

        if collected == summaries:
            break
        summaries = collected

    return summaries


def build_return_summaries(
    src_root: Path,
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
    max_iterations: int = 3,
) -> Dict[str, FunctionReturnSummary]:
    return build_return_summaries_from_analysis_files(
        list(
            iter_analysis_files(
                src_root,
                module_prefix=module_prefix,
                entrypoints=entrypoints,
                project_root=project_root,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        ),
        callable_map=callable_map,
        module_map=module_map,
        known_classes=known_classes,
        max_iterations=max_iterations,
    )


def build_type_summaries_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    return_summaries: Dict[str, FunctionReturnSummary],
    max_iterations: int = 5,
) -> Tuple[Dict[str, FunctionParamSummary], Dict[Tuple[str, str], Set[str]]]:
    param_summaries: Dict[str, FunctionParamSummary] = {}
    class_attr_types: Dict[Tuple[str, str], Set[str]] = {}
    callable_ids = set(callable_map.keys())

    for _ in range(max_iterations):
        next_param_summaries = copy_param_summaries(param_summaries)
        next_class_attr_types = copy_class_attr_types(class_attr_types)

        for analysis_file in analysis_files:
            py_file = analysis_file.path
            module = analysis_file.module
            tree = parse_python_file(py_file)
            collector = TypeSummaryCollector(
                module=module,
                file=py_file,
                module_index=module_map[module],
                callable_ids=callable_ids,
                module_map=module_map,
                known_classes=known_classes,
                return_summaries=return_summaries,
                param_summaries=next_param_summaries,
                class_attr_types=next_class_attr_types,
            )
            collector.visit(tree)

        if (
            next_param_summaries == param_summaries
            and next_class_attr_types == class_attr_types
        ):
            break
        param_summaries = next_param_summaries
        class_attr_types = next_class_attr_types

    return param_summaries, class_attr_types


def build_type_summaries(
    src_root: Path,
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    return_summaries: Dict[str, FunctionReturnSummary],
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
    max_iterations: int = 5,
) -> Tuple[Dict[str, FunctionParamSummary], Dict[Tuple[str, str], Set[str]]]:
    return build_type_summaries_from_analysis_files(
        list(
            iter_analysis_files(
                src_root,
                module_prefix=module_prefix,
                entrypoints=entrypoints,
                project_root=project_root,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        ),
        callable_map=callable_map,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
        max_iterations=max_iterations,
    )


def collect_edges_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    include_external: bool,
    package_prefix: Optional[str] | Sequence[str],
    return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
    param_summaries: Optional[Dict[str, FunctionParamSummary]] = None,
    class_attr_types: Optional[Dict[Tuple[str, str], Set[str]]] = None,
) -> List[Edge]:
    edges: List[Edge] = []
    callable_ids = set(callable_map.keys())

    for analysis_file in analysis_files:
        py_file = analysis_file.path
        module = analysis_file.module
        tree = parse_python_file(py_file)
        collector = CallCollector(
            module=module,
            file=py_file,
            module_index=module_map[module],
            callable_ids=callable_ids,
            module_map=module_map,
            known_classes=known_classes,
            include_external=include_external,
            package_prefix=package_prefix,
            return_summaries=return_summaries,
            param_summaries=param_summaries,
            class_attr_types=class_attr_types,
        )
        collector.visit(tree)
        edges.extend(collector.edges)

    return edges


def collect_edges(
    src_root: Path,
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    include_external: bool,
    package_prefix: Optional[str] | Sequence[str],
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
    return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
    param_summaries: Optional[Dict[str, FunctionParamSummary]] = None,
    class_attr_types: Optional[Dict[Tuple[str, str], Set[str]]] = None,
) -> List[Edge]:
    return collect_edges_from_analysis_files(
        list(
            iter_analysis_files(
                src_root,
                module_prefix=module_prefix,
                entrypoints=entrypoints,
                project_root=project_root,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        ),
        callable_map=callable_map,
        module_map=module_map,
        known_classes=known_classes,
        include_external=include_external,
        package_prefix=package_prefix,
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
    )


def merge_module_index(target: ModuleIndex, source: ModuleIndex) -> None:
    target.imports.update(source.imports)
    target.classes.update(source.classes)
    target.class_bases.update(source.class_bases)
    target.properties.update(source.properties)
    target.static_methods.update(source.static_methods)
    target.class_methods.update(source.class_methods)
    for module_name in source.star_imports:
        if module_name not in target.star_imports:
            target.star_imports.append(module_name)


class SourceCallResolver:
    """Resolve calls from in-memory source snippets using call-graph semantics.

    This is intended for notebook cells and other non-file entrypoints. It keeps
    module-scope import and simple variable-type state across calls to
    ``resolve_source`` so sequential notebook cells can resolve aliases and
    instance method calls in the same way as a Python module.
    """

    def __init__(
        self,
        module: str,
        file: Path,
        callable_map: Dict[str, CallableDef],
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        include_external: bool,
        package_prefix: Optional[str],
        return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
        param_summaries: Optional[Dict[str, FunctionParamSummary]] = None,
        class_attr_types: Optional[Dict[Tuple[str, str], Set[str]]] = None,
    ):
        self.module = module
        self.file = file
        self.module_index = ModuleIndex(module=module, path=file)
        self.module_map = dict(module_map)
        self.module_map[module] = self.module_index
        self.collector = CallCollector(
            module=module,
            file=file,
            module_index=self.module_index,
            callable_ids=set(callable_map.keys()),
            module_map=self.module_map,
            known_classes=known_classes,
            include_external=include_external,
            package_prefix=package_prefix,
            return_summaries=return_summaries,
            param_summaries=param_summaries,
            class_attr_types=class_attr_types,
        )
        self.collector.current_callable = self.collector.module_callable
        self.collector.callable_stack.append(self.collector.module_callable)
        self.collector.push_scope()

    def resolve_source(self, source: str, filename: str | None = None) -> List[Edge]:
        filename = filename or str(self.file)
        tree = parse_python_source(source, filename=filename)
        attach_parents(tree)

        defs = DefinitionCollector(module=self.module, file=Path(filename))
        defs.visit(tree)
        merge_module_index(self.module_index, defs.module_index)

        self.collector.file = Path(filename)
        self.collector.module_index = self.module_index
        before = len(self.collector.edges)
        for statement in tree.body:
            self.collector.visit(statement)
        return self.collector.edges[before:]


def build_call_graph_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    *,
    include_external: bool = False,
    package_prefix: Optional[str] | Sequence[str] = None,
) -> Tuple[Dict[str, CallableDef], List[Edge]]:
    nodes, module_map, known_classes = build_indices_from_analysis_files(analysis_files)
    return_summaries = build_return_summaries_from_analysis_files(
        analysis_files,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
    )
    param_summaries, class_attr_types = build_type_summaries_from_analysis_files(
        analysis_files,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
    )
    edges = collect_edges_from_analysis_files(
        analysis_files,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        include_external=include_external,
        package_prefix=package_prefix,
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
    )
    return nodes, edges


def run_from_extraction_config(config: ExtractionConfig, *, outdir: Optional[Path] = None) -> Tuple[int, int, int]:
    analysis_files = iter_analysis_files_for_source_roots(
        config.source_roots,
        entrypoints=config.entrypoints,
        project_root=config.project_root,
        include_globs=config.include_globs,
        exclude_globs=config.exclude_globs,
    )
    nodes, edges = build_call_graph_from_analysis_files(
        analysis_files,
        include_external=config.call_graph.include_external,
        package_prefix=config.package_prefixes,
    )
    output_dir = (outdir or config.call_graph.outdir).resolve()
    write_outputs(outdir=output_dir, nodes=nodes, edges=edges)
    resolved_edges = sum(1 for e in edges if e.resolved)
    return len(nodes), len(edges), resolved_edges


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an AST-based call graph")
    parser.add_argument("--config", default=None, type=Path, help="Optional extraction JSON/JSONC config")
    parser.add_argument("--root", default=None, help="Python source root (e.g., src/shop)")
    parser.add_argument("--package", default=None, help="Package prefix for filtering internal calls")
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include unresolved/external callee nodes in edge list",
    )
    parser.add_argument(
        "--module-prefix",
        default=None,
        help="Prefix to prepend to discovered module names (e.g., shop when root is src/shop)",
    )
    parser.add_argument(
        "--entrypoint",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional Python file to include as an explicit entrypoint. "
            "May be passed more than once; module names are project-relative."
        ),
    )
    parser.add_argument("--outdir", default=None, help="Output directory")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.config:
        config = load_extraction_config(args.config)
        node_count, edge_count, resolved_edges = run_from_extraction_config(
            config,
            outdir=Path(args.outdir).resolve() if args.outdir else None,
        )
        output_dir = Path(args.outdir).resolve() if args.outdir else config.call_graph.outdir
        print(f"Call graph generated in {output_dir}")
        print(f"Nodes: {node_count}")
        print(f"Edges: {edge_count} (resolved: {resolved_edges}, unresolved: {edge_count - resolved_edges})")
        return

    if not args.root:
        raise SystemExit("--root is required unless --config is provided")
    root = Path(args.root).resolve()
    outdir = Path(args.outdir or "artifacts/call_graph").resolve()
    entrypoints = tuple(Path(path).resolve() for path in args.entrypoint)

    nodes, module_map, known_classes = build_indices(
        root,
        module_prefix=args.module_prefix,
        entrypoints=entrypoints,
    )
    return_summaries = build_return_summaries(
        src_root=root,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        module_prefix=args.module_prefix,
        entrypoints=entrypoints,
    )
    param_summaries, class_attr_types = build_type_summaries(
        src_root=root,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
        module_prefix=args.module_prefix,
        entrypoints=entrypoints,
    )
    edges = collect_edges(
        src_root=root,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        include_external=args.include_external,
        package_prefix=args.package,
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
        module_prefix=args.module_prefix,
        entrypoints=entrypoints,
    )

    write_outputs(outdir=outdir, nodes=nodes, edges=edges)

    resolved_edges = sum(1 for e in edges if e.resolved)
    print(f"Call graph generated in {outdir}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)} (resolved: {resolved_edges}, unresolved: {len(edges) - resolved_edges})")


if __name__ == "__main__":
    main()
