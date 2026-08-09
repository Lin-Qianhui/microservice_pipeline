"""The class types of a name or attribute expression.

Reading and writing the type facts for the two expression forms the analyzer
can say anything about: a bare name, and an attribute chain rooted at one. This
is where ``self.grid`` becomes a set of class ids, and where a property getter
is followed through to the types its ``return`` produces -- ``obj.field`` is a
call when ``field`` is a property, and pretending otherwise loses the edge.

Requires from siblings: ``_resolve_method_targets``, ``_class_and_ancestors``,
``_infer_class_types_from_value``, ``_lambda_id``.
"""

from __future__ import annotations

import ast
from typing import List, Set
from ..ast_utils import attribute_to_name
from .constants import SELF_NAMES

from ..project_index import unique
from .state import CollectorState


class ExpressionsMixin(CollectorState):
    """The class types of a name or attribute expression."""

    def get_expr_types(self, expr: ast.AST) -> Set[str]:
        """Infer possible class IDs for a simple name or attribute expression."""
        if isinstance(expr, ast.Name):
            return self.types.get_var_types(expr.id)
        if isinstance(expr, ast.Attribute):
            expr_name = attribute_to_name(expr)
            if expr_name:
                local_types = self.types.get_attr_expr_types(expr_name)
                if local_types:
                    return local_types

            base_types: Set[str] = set()
            if (
                isinstance(expr.value, ast.Name)
                and expr.value.id in SELF_NAMES
                and self.current_class
            ):
                class_id = self.current_class_id()
                if class_id:
                    base_types.add(class_id)
            base_types.update(self.get_expr_types(expr.value))

            attr_types: Set[str] = set()
            for base_type in base_types:
                for owner in self.project_index.class_and_ancestors(base_type):
                    attr_types.update(
                        self.class_attr_types.object_types.get((owner, expr.attr), set())
                    )
            attr_types.update(self._infer_property_return_class_types(expr))
            return attr_types
        return set()

    def _class_attr_element_types(self, expr: ast.AST) -> Set[str]:
        """Element types recorded for ``self.<attr>`` and friends across methods.

        This is what makes a registry attribute usable: ``add_child`` stores into
        ``self.children`` in one method and another method iterates it, so the
        fact has to survive the boundary between them.
        """
        if not isinstance(expr, ast.Attribute):
            return set()

        base_types: Set[str] = set()
        if (
            isinstance(expr.value, ast.Name)
            and expr.value.id in SELF_NAMES
            and self.current_class
        ):
            class_id = self.current_class_id()
            if class_id:
                base_types.add(class_id)
        base_types.update(self.get_expr_types(expr.value))

        element_types: Set[str] = set()
        for base_type in base_types:
            for owner in self.project_index.class_and_ancestors(base_type):
                element_types.update(
                    self.class_attr_types.element_types.get((owner, expr.attr), set())
                )
        return element_types

    def _receiver_types_for_expr(self, expr: ast.AST) -> Set[str]:
        receiver_types: Set[str] = set()
        if (
            isinstance(expr, ast.Name)
            and expr.id in SELF_NAMES
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
            for target in self.project_index.resolve_method_targets(receiver_type, node.attr):
                if target in self.project_index.property_ids:
                    targets.append(target)
        return unique(targets)

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

    def _attribute_owner_types(self, target: ast.Attribute) -> Set[str]:
        base_types: Set[str] = set()
        if (
            isinstance(target.value, ast.Name)
            and target.value.id in SELF_NAMES
            and self.current_class
        ):
            class_id = self.current_class_id()
            if class_id:
                base_types.add(class_id)
        base_types.update(self.get_expr_types(target.value))
        return base_types

    def set_attribute_types(self, target: ast.Attribute, types: Set[str]) -> None:
        if not types:
            return

        attr_name = attribute_to_name(target)
        if attr_name:
            self.types.set_attr_expr_types(attr_name, types)

        for base_type in self._attribute_owner_types(target):
            self.class_attr_types.add_object_types((base_type, target.attr), types)

    def set_attribute_element_types(self, target: ast.Attribute, types: Set[str]) -> None:
        """Record what an attribute *contains*, from a whole-value assignment."""
        if not types:
            return
        for base_type in self._attribute_owner_types(target):
            self.class_attr_types.add_element_types((base_type, target.attr), types)

    def record_attribute_container_store(self, container: ast.AST, values: Set[str]) -> None:
        """Note types added into a container attribute, however they got there.

        Covers both ``self.x[k] = v`` and ``self.x.append(v)`` styles. Only the
        element facts are touched: storing into a container says nothing about
        what the attribute itself is, and claiming otherwise would make the
        container look like the thing it holds.
        """
        if not values or not isinstance(container, ast.Attribute):
            return
        for base_type in self._attribute_owner_types(container):
            self.class_attr_types.add_element_types((base_type, container.attr), values)
