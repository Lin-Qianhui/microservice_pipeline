"""Subprocess shared-state lineage helpers for data-access extraction."""

from __future__ import annotations

import ast
from typing import Set

from microservice_pipeline.data_access.pyright_type_probe import FAMILY_OBJECT
from microservice_pipeline.data_access.rules import _attribute_path
from microservice_pipeline.subprocess_coupling import (
    add_subprocess_child_expr,
    add_subprocess_name,
    is_add_subprocess_call,
)


def _class_state_display(owner: str) -> str:
    return f"{owner.rsplit('.', 1)[-1]} state"


def _class_attr_state_display(owner: str, attr_name: str) -> str:
    class_name = owner.rsplit(".", 1)[-1]
    return f"{class_name}.{attr_name} state"


class SubprocessStateLineageMixin:
    def _state_owner_types_from_expr(self, node: ast.AST) -> Set[str]:
        if isinstance(node, ast.Attribute) and node.attr == "state":
            return self._class_types_from_expr(node.value)
        if isinstance(node, ast.Name) and self.scope:
            return set(self.scope.local_shared_state_owner_types.get(node.id, set()))
        if isinstance(node, ast.Attribute) and self.scope:
            path = _attribute_path(node)
            if path:
                return set(self.scope.attr_shared_state_owner_types.get(path, set()))
        return set()

    def _shared_state_owner_types_from_value(self, value: object) -> Set[str]:
        if isinstance(value, ast.Call):
            owners: Set[str] = set()
            for keyword in value.keywords:
                if keyword.arg == "state":
                    owners.update(self._state_owner_types_from_expr(keyword.value))
            return owners
        if isinstance(value, ast.Name) and self.scope:
            return set(self.scope.local_shared_state_owner_types.get(value.id, set()))
        if isinstance(value, ast.Attribute) and value.attr == "state":
            return self._state_owner_types_from_expr(value)
        if isinstance(value, ast.Attribute) and self.scope:
            path = _attribute_path(value)
            if path:
                return set(self.scope.attr_shared_state_owner_types.get(path, set()))
        return set()

    def _element_shared_state_owner_types_from_value(self, value: object) -> Set[str]:
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            owners: Set[str] = set()
            for element in value.elts:
                owners.update(self._shared_state_owner_types_from_value(element))
            return owners
        if isinstance(value, ast.Name) and self.scope:
            return set(self.scope.local_element_shared_state_owner_types.get(value.id, set()))
        return set()

    def _class_state_object_id_for_owner(self, owner: str, node: ast.AST) -> str:
        split_class_state = owner in self.split_class_owners
        object_id = f"class_attr_state:{owner}:state" if split_class_state else f"class_state:{owner}"
        self._register_object(
            object_id=object_id,
            kind="class_attr_state" if split_class_state else "class_state",
            display_name=_class_attr_state_display(owner, "state") if split_class_state else _class_state_display(owner),
            scope="class",
            owner=owner,
            field="state" if split_class_state else "",
            node=node,
            inferred_type=FAMILY_OBJECT,
            confidence="medium",
            access_path="self.state" if split_class_state else f"{owner.rsplit('.', 1)[-1]} state",
        )
        return object_id

    def _record_subprocess_state_lineage(self, node: ast.Call) -> None:
        if not is_add_subprocess_call(node) or not isinstance(node.func, ast.Attribute):
            return
        child_expr = add_subprocess_child_expr(node)
        if child_expr is None:
            return
        parent_types = sorted(self._class_types_from_expr(node.func.value))
        child_types = sorted(self._class_types_from_expr(child_expr))
        if len(parent_types) != 1 or len(child_types) != 1:
            return
        if parent_types[0] not in self._shared_state_owner_types_from_value(child_expr):
            return
        self._record_lineage(
            self._class_state_object_id_for_owner(child_types[0], node),
            self._class_state_object_id_for_owner(parent_types[0], node),
            "state_assign",
            node,
            slot=add_subprocess_name(node),
        )
