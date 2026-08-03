"""Shared-state lineage across a registration call.

When a parent registers a child and hands it the parent's own state --
``self.add_subprocess('albedo', StepFunctionAlbedo(state=self.state))`` -- the two
objects' state is one object at runtime. That is a data dependency no assignment
in either class expresses, so nothing else in the data-access pass finds it.

Two independent facts have to hold, and they are checked separately:

*This call registers the child.* Derived, not recognised by name. The call graph's
``registration`` module works out which callables retain a parameter on ``self``
and later invoke it; this module resolves a call site onto those rules. Reusing
them is what makes the analysis work on any framework rather than on the one
whose method name used to be hardcoded here.

*The child received the parent's state.* Checked by
``_shared_state_owner_types_from_value``. This is the part that makes the lineage
mean something -- registration alone says the objects are coupled, not that they
share state.
"""

from __future__ import annotations

import ast
from typing import Optional, Set, Tuple

from microservice_pipeline.call_graph.registration import (
    RegistrationRule,
    registration_child_exprs,
    registration_slot,
)
from microservice_pipeline.data_access.pyright_type_probe import FAMILY_OBJECT
from microservice_pipeline.data_access.rules import _attribute_path


def _class_state_display(owner: str) -> str:
    return f"{owner.rsplit('.', 1)[-1]} state"


def _class_attr_state_display(owner: str, attr_name: str) -> str:
    class_name = owner.rsplit(".", 1)[-1]
    return f"{class_name}.{attr_name} state"


class RegisteredStateLineageMixin:
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

    def _registration_rule_for_call(
        self, node: ast.Call
    ) -> Tuple[Optional[RegistrationRule], str]:
        """Match a call against the derived registration rules.

        Resolution goes through ``ProjectIndex`` rather than the collector's own
        ``known_classes``, which is a flat set with no inheritance. climlab calls
        ``self.add_subprocess(...)`` on a ``StepFunctionAlbedo``, but the method
        that retains the child is defined three classes up on ``Process``; only
        an ancestor walk connects the two.
        """
        if not self.registration_rules or self.project_index is None:
            return None, ""

        # ``Pipeline(steps=[...])`` registers into the instance being built, so
        # the rule sits on the constructor and there is no receiver to resolve.
        if not isinstance(node.func, ast.Attribute):
            for class_id in sorted(self._class_types_from_expr(node)):
                for target in self.project_index.resolve_constructor_targets(class_id):
                    rule = self.registration_rules.get(target)
                    if rule is not None:
                        return rule, target
            return None, ""

        for receiver_type in sorted(self._class_types_from_expr(node.func.value)):
            for target in self.project_index.resolve_method_targets(
                receiver_type, node.func.attr
            ):
                rule = self.registration_rules.get(target)
                if rule is not None:
                    return rule, target
        return None, ""

    def _registration_parent_types(self, node: ast.Call) -> Set[str]:
        if isinstance(node.func, ast.Attribute):
            return self._class_types_from_expr(node.func.value)
        return self._class_types_from_expr(node)

    def _record_registered_state_lineage(self, node: ast.Call) -> None:
        rule, target = self._registration_rule_for_call(node)
        if rule is None:
            return

        # A bound method's parameter 1 is the call site's argument 0. A
        # staticmethod has no receiver to skip, and a constructor's ``self`` is
        # supplied by the runtime rather than written at the call site.
        offset = 0 if target in self.project_index.static_method_ids else 1

        parent_types = sorted(self._registration_parent_types(node))
        if len(parent_types) != 1:
            return
        slot = registration_slot(node, rule, offset)

        for child_expr in registration_child_exprs(node, rule, offset):
            child_types = sorted(self._class_types_from_expr(child_expr))
            if len(child_types) != 1:
                continue
            # Registration on its own is coupling, not shared state. Only a child
            # that was handed the parent's state shares an object with it.
            if parent_types[0] not in self._shared_state_owner_types_from_value(child_expr):
                continue
            self._record_lineage(
                self._class_state_object_id_for_owner(child_types[0], node),
                self._class_state_object_id_for_owner(parent_types[0], node),
                "state_assign",
                node,
                slot=slot,
            )