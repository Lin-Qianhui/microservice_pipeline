"""Rule tables and small AST helpers for data-access extraction."""

from __future__ import annotations

import ast
from typing import List, Set, Tuple

from microservice_pipeline.data_access.pyright_type_probe import (
    FAMILY_DATAFRAME,
    FAMILY_DICT,
    FAMILY_FIELD,
    FAMILY_FILE,
    FAMILY_LIST,
    FAMILY_OBJECT,
    FAMILY_SET,
    FAMILY_UNKNOWN,
    FAMILY_XARRAY,
)


CONTAINER_TYPES = {
    FAMILY_DATAFRAME,
    FAMILY_DICT,
    FAMILY_LIST,
    FAMILY_SET,
    FAMILY_FILE,
    FAMILY_FIELD,
    FAMILY_XARRAY,
}
COORDINATOR_ATTR_THRESHOLD = 4
COORDINATOR_METHOD_THRESHOLD = 3
COORDINATOR_CONTAINER_THRESHOLD = 2
MUTATING_METHODS = {
    "append",
    "extend",
    "insert",
    "update",
    "setdefault",
    "pop",
    "remove",
    "clear",
    "add",
    "discard",
    "sort",
}
PANDAS_INPLACE_METHODS = {
    "drop",
    "fillna",
    "replace",
    "rename",
    "reset_index",
    "set_index",
    "sort_values",
}
FILE_READ_FUNCS = {"read_csv", "read_json", "read_excel", "read_table", "read_parquet"}
POOCH_READ_FUNCS = {"retrieve"}
XARRAY_OPEN_FUNCS = {"open_dataset", "open_dataarray"}
FILE_WRITE_METHODS = {
    "to_csv",
    "to_json",
    "to_excel",
    "to_parquet",
    "to_pickle",
}
PANDAS_INDEXER_ATTRS = {"loc", "iloc", "at", "iat"}
XARRAY_LABEL_METHODS = {"sel", "isel"}
XARRAY_INDEXER_ATTRS = {"loc", "iloc"}
MAX_RETURN_SUMMARY_PASSES = 8
CONTAINER_FIELD_KIND = "container_field"


def _field_inferred_type(kind: str) -> str:
    if kind == "df_col":
        return FAMILY_DATAFRAME
    if kind == "dict_key":
        return FAMILY_DICT
    return FAMILY_UNKNOWN


def _is_containerish_family(family: str) -> bool:
    return family in {
        FAMILY_DATAFRAME,
        FAMILY_DICT,
        FAMILY_FIELD,
        FAMILY_LIST,
        FAMILY_SET,
        FAMILY_XARRAY,
        FAMILY_OBJECT,
    }


def _literal_container_family(node: ast.AST) -> str:
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return FAMILY_DICT
    if isinstance(node, (ast.List, ast.ListComp)):
        return FAMILY_LIST
    if isinstance(node, (ast.Set, ast.SetComp)):
        return FAMILY_SET
    if isinstance(node, ast.Call):
        func_name = _name_from_expr(node.func)
        if func_name in {"dict", "list", "set"}:
            return {"dict": FAMILY_DICT, "list": FAMILY_LIST, "set": FAMILY_SET}[func_name]
    return FAMILY_UNKNOWN


def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _name_from_expr(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive for unusual nodes
        return node.__class__.__name__


def _keyword_truthy(call: ast.Call, keyword_name: str) -> bool:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    return False


def _call_name_matches(call_name: str, names: Set[str]) -> bool:
    return any(call_name == name or call_name.endswith(f".{name}") for name in names)


def _literal_strings(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: List[str] = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.append(element.value)
        return values
    return []


def _slice_value(node: ast.AST) -> ast.AST:
    return node.slice if isinstance(node, ast.Subscript) else node


def _indexer_fields(slice_node: ast.AST) -> Tuple[List[str], str]:
    if isinstance(slice_node, ast.Tuple) and slice_node.elts:
        candidates = list(slice_node.elts)
        for candidate in reversed(candidates):
            fields = _literal_strings(candidate)
            if fields:
                return fields, "high"
        return [], "low"

    fields = _literal_strings(slice_node)
    if fields:
        return fields, "high"
    return [], "low"


def _dict_literal_keys(node: ast.AST) -> List[str]:
    if not isinstance(node, ast.Dict):
        return []
    keys: List[str] = []
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.append(key.value)
    return keys


def _xarray_indexer_fields(slice_node: ast.AST) -> Tuple[List[str], str]:
    if isinstance(slice_node, ast.Dict):
        fields = _dict_literal_keys(slice_node)
        return fields, "medium" if fields else "low"
    if isinstance(slice_node, ast.Tuple):
        fields: List[str] = []
        for element in slice_node.elts:
            fields.extend(_dict_literal_keys(element))
        if fields:
            return fields, "medium"
    fields = _literal_strings(slice_node)
    return fields, "medium" if fields else "low"


def _xarray_call_fields(node: ast.Call) -> Tuple[List[str], str]:
    fields: List[str] = []
    for keyword in node.keywords:
        if keyword.arg is None:
            fields.extend(_dict_literal_keys(keyword.value))
        elif keyword.arg not in {"indexers", "indexers_kwargs"}:
            fields.append(keyword.arg)
        else:
            fields.extend(_dict_literal_keys(keyword.value))
    for arg in node.args:
        fields.extend(_dict_literal_keys(arg))
    return list(dict.fromkeys(fields)), "medium" if fields else "low"


def _slice_selects_multiple_fields(slice_node: ast.AST) -> bool:
    if isinstance(slice_node, (ast.List, ast.Set)):
        return True
    if isinstance(slice_node, ast.Tuple):
        if any(isinstance(element, (ast.List, ast.Set)) for element in slice_node.elts):
            return True
        if len(slice_node.elts) > 1 and any(
            not isinstance(element, (ast.Slice, ast.Constant)) for element in slice_node.elts
        ):
            return True
    return False


def _top_level_attr_name(attr_path: str) -> str:
    return attr_path.split(".", 1)[0].strip()


def _class_attr_family_key(owner: str, attr_name: str) -> str:
    return f"class_attr:{owner}:{attr_name}"


def _attr_expr_family_key(callable_id: str, attr_path: str) -> str:
    return f"attr_expr:{callable_id}:{attr_path}"
