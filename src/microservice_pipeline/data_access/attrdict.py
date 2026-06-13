"""Detection for mapping classes that expose keys as attributes."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Sequence, Set

from microservice_pipeline.import_resolution import (
    is_package_file,
    resolve_import_from_module,
    resolve_import_from_target,
)
from microservice_pipeline.call_graph.generate_call_graph_ast import parse_python_file
from microservice_pipeline.data_access.rules import _attribute_path, _slice_value


def _module_imports(tree: ast.Module, module: str, file: Path) -> Dict[str, str]:
    imports: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            imported_module = resolve_import_from_module(
                module,
                node.module,
                node.level,
                current_is_package=is_package_file(file),
            )
            if imported_module is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                target = resolve_import_from_target(
                    module,
                    node.module,
                    node.level,
                    alias.name,
                    current_is_package=is_package_file(file),
                )
                if target is not None:
                    imports[alias.asname or alias.name] = target
    return imports


def _resolve_class_reference(node: ast.AST, module: str, module_imports: Dict[str, str]) -> str:
    name = _attribute_path(node) or ""
    if not name:
        return ""
    if "." not in name:
        return module_imports.get(name) or f"{module}.{name}"
    base, _sep, _rest = name.partition(".")
    imported_base = module_imports.get(base)
    if imported_base:
        return f"{imported_base}.{_rest}"
    return name


def _getattr_delegates_to_mapping(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.name != "__getattr__":
        return False
    arg_names = [arg.arg for arg in node.args.args]
    if len(arg_names) < 2:
        return False
    key_name = arg_names[1]
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        if not isinstance(_slice_value(child), ast.Name) or _slice_value(child).id != key_name:
            continue
        if isinstance(child.value, ast.Name) and child.value.id == "self":
            return True
        if (
            isinstance(child.value, ast.Attribute)
            and isinstance(child.value.value, ast.Name)
            and child.value.value.id == "self"
        ):
            return True
    return False


def collect_attrdict_classes(tree: ast.Module, module: str, file: Path) -> Set[str]:
    module_imports = _module_imports(tree, module, file)
    direct_attr_classes: Set[str] = set()
    class_bases: Dict[str, List[str]] = {}
    attrdict_base_names = {"Attr", "MutableAttr", "AttrDict", "AttrMap", "AttrDefault"}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_id = f"{module}.{node.name}"
        class_bases[class_id] = [
            base_id
            for base in node.bases
            if (base_id := _resolve_class_reference(base, module, module_imports))
        ]
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and _getattr_delegates_to_mapping(stmt):
                direct_attr_classes.add(class_id)

    attrdict_classes = set(direct_attr_classes)
    changed = True
    while changed:
        changed = False
        for class_id, bases in class_bases.items():
            if class_id in attrdict_classes:
                continue
            if any(base in attrdict_classes or base.rsplit(".", 1)[-1] in attrdict_base_names for base in bases):
                attrdict_classes.add(class_id)
                changed = True
    return attrdict_classes


def collect_attrdict_classes_from_analysis_files(analysis_files: Sequence[object]) -> Set[str]:
    attrdict_classes: Set[str] = set()
    for analysis_file in analysis_files:
        path = analysis_file.path
        tree = parse_python_file(path)
        attrdict_classes.update(collect_attrdict_classes(tree, analysis_file.module, path))
    return attrdict_classes
