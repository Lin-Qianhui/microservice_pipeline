"""Statements that move types around: assignment, control flow, mutation.

Assignment is where an inferred type gets bound to a name, so most of this file
is about deciding what a target should hold. Control flow is where two bindings
have to be reconciled: ``visit_If`` walks each branch against a copy of the type
state and merges the results, because a name assigned in only one branch is
possibly-either afterwards, not definitely-the-last-one.

Container mutation is tracked separately from assignment because
``self.children.append(child)`` changes what a container holds without ever
rebinding a name -- and that call is the registration idiom.

Requires from siblings: ``_infer_class_types_from_value``,
``_infer_container_element_types``, ``_infer_callable_ids_from_value``,
``_value_origins``, ``_element_origins``, ``set_attribute_types``,
``_note_attribute_store``, ``_note_key_param``, ``_copy_type_state``.
"""

from __future__ import annotations

import ast
from typing import Optional, Set
from ..dunders import AUGASSIGN_DUNDER_METHODS
from ..type_env import Origin
from .constants import ELEMENT_SUFFIX, SELF_NAMES
from .shapes import container_mutation_key, is_container_literal, is_dict_items_call

from .state import CollectorState


class StatementsMixin(CollectorState):
    """Statements that move types around: assignment, control flow, mutation."""

    def _assign_target_types(
        self,
        target: ast.AST,
        class_types: Set[str],
        container_types: Set[str],
        value: ast.AST,
        slot: Optional[int] = None,
    ) -> None:
        """Attach inferred value types to a name, attribute, or destructuring target.

        ``slot`` records which tuple position of ``value`` this target took, so
        provenance for ``a, b = make_pair()`` stays position-specific.
        """
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                self._assign_target_types(
                    item,
                    self._infer_sequence_slot_class_types(value, index),
                    self._infer_sequence_slot_container_types(value, index),
                    value,
                    slot=index,
                )
            return

        if isinstance(target, ast.Name) and self.types.var_types_stack:
            if class_types:
                self.types.set_var_types(target.id, class_types)
            if container_types or is_container_literal(value):
                self.types.set_container_types(target.id, container_types)
            # Unconditional: the point of provenance is the case where nothing
            # is known about ``value`` yet, so an empty inference above is
            # exactly when this matters most.
            self.types.set_var_source(target.id, value, slot)
            # A destructured position holds one element of the value, not the
            # value; ``for name, proc in registry.items()`` is the shape that
            # matters and it is handled where the loop is visited.
            if slot is None:
                self.types.set_var_origins(target.id, self._value_origins(value))
            return

        if isinstance(target, ast.Attribute):
            self.set_attribute_types(target, class_types)
            self.set_attribute_element_types(target, container_types)
            # ``self.parent = parent`` retains a single value rather than
            # collecting one, so it is recorded as a non-container escape and
            # will not on its own make the callable a registrar.
            self._note_attribute_store(
                target, self._value_origins(value), is_container=False
            )
            self._note_attribute_store(
                target, self._element_origins(value), is_container=True
            )
            return

        # ``self.registry[key] = value``: the assignment target is a subscript
        # of an attribute, so the value becomes an element of that attribute.
        if isinstance(target, ast.Subscript):
            self.record_attribute_container_store(target.value, class_types)
            self._note_attribute_store(
                target.value,
                self._value_origins(value),
                is_container=True,
                key=target.slice,
            )

    def visit_If(self, node: ast.If) -> None:
        """Analyze both branches independently and keep the union of their facts.

        Static analysis does not know which branch will run. Mutating one shared
        map in source order would make the second branch incorrectly depend on
        the first, so both start from the same snapshot.
        """
        self.visit(node.test)
        base_state = self.types.copy_state()

        self.types.restore_state(base_state)
        for stmt in node.body:
            self.visit(stmt)
        body_state = self.types.copy_state()

        self.types.restore_state(base_state)
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
        orelse_state = self.types.copy_state()

        self.types.restore_state(self.types.merge_states([body_state, orelse_state]))

    def visit_Assign(self, node: ast.Assign) -> None:
        class_types = self._infer_class_types_from_value(node.value)
        container_types = self._infer_container_element_types(node.value)
        for target in node.targets:
            self._assign_target_types(target, class_types, container_types, node.value)
            self._assign_target_callables(target, node.value)
        self.generic_visit(node)

    def _assign_target_callables(self, target: ast.AST, value: ast.AST) -> None:
        """Record callables an assignment binds, for locals and ``self.<attr>``.

        Runs beside the class-type assignment rather than inside it: the two
        dimensions are independent, and ``handler = self.process`` produces a
        callable fact and no type fact at all.
        """
        callables = self._infer_callable_ids_from_value(value)
        elements = self._infer_container_callable_ids(value)
        if not callables and not elements:
            return

        if isinstance(target, ast.Name):
            self.types.add_var_callables(target.id, callables)
            self.types.add_var_callables(f"{target.id}{ELEMENT_SUFFIX}", elements)
            return

        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in SELF_NAMES
        ):
            class_id = self.current_class_id()
            if class_id:
                key = (class_id, target.attr)
                self.class_attr_types.add_callable_types(key, callables)
                self.class_attr_types.add_element_callable_types(key, elements)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # The annotation is a fact in its own right, so ``self.grid: Grid`` is
        # useful even with no assigned value -- which is exactly the form a
        # class-level attribute declaration takes.
        annotated_objects, annotated_elements = self._annotation_types(node.annotation)
        class_types = set(annotated_objects)
        container_types = set(annotated_elements)
        if node.value is not None:
            class_types |= self._infer_class_types_from_value(node.value)
            container_types |= self._infer_container_element_types(node.value)

        if class_types or container_types:
            # ``engine: Engine`` written directly in a class body declares an
            # instance attribute, not a local. Its target is a bare Name, so
            # without this it would be filed as a variable in the class scope and
            # never seen by ``self.engine`` lookups in the methods below it.
            if (
                isinstance(node.target, ast.Name)
                and isinstance(getattr(node, "parent", None), ast.ClassDef)
                and self.current_class
            ):
                class_id = self.current_class_id()
                if class_id:
                    key = (class_id, node.target.id)
                    self.class_attr_types.add_object_types(key, class_types)
                    self.class_attr_types.add_element_types(key, container_types)
            else:
                self._assign_target_types(
                    node.target, class_types, container_types, node.value or node
                )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            self.types.add_container_types(
                node.target.id, self._infer_container_element_types(node.value)
            )
        method_name = AUGASSIGN_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.target, method_name, "dunder_operator", node.lineno, node.col_offset
            )
        self.visit(node.target)
        self.visit(node.value)

    def _visit_for_like(self, node: ast.For | ast.AsyncFor) -> None:
        """Model iteration and propagate known collection element types."""
        self._add_dunder_edges(node.iter, "__iter__", "dunder_iter", node.lineno, node.col_offset)
        self.visit(node.iter)
        element_origins = self._element_origins(node.iter)
        if isinstance(node.target, ast.Name):
            element_types = self._infer_container_element_types(node.iter)
            if element_types:
                self.types.set_var_types(node.target.id, element_types)
            self.types.set_var_origins(node.target.id, element_origins)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            # ``for name, proc in registry.items()``: each loop variable takes
            # one slot of the element, so bind them position by position.
            for index, item in enumerate(node.target.elts):
                if not isinstance(item, ast.Name):
                    continue
                slot_types = self._infer_sequence_slot_class_types(node.iter, index)
                if slot_types:
                    self.types.set_var_types(item.id, slot_types)
                # Destructuring a mapping's items gives keys at slot 0 and the
                # stored values at slot 1. Only the values are the registry's
                # elements; the keys are the labels they were filed under.
                self.types.set_var_origins(
                    item.id,
                    element_origins
                    if index == 1 and is_dict_items_call(node.iter)
                    else set(),
                )
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for_like(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for_like(node)

    def _container_mutation_element_types(self, node: ast.Call) -> Set[str]:
        """Element types a mutating container call introduces, if any.

        ``append``/``add`` contribute the argument's own type; ``extend``/
        ``update`` contribute the argument's *elements*, since they merge one
        collection into another. ``update`` is what makes dict-style registries
        work -- ``self.children.update({name: child})`` is the idiom frameworks
        use to register something under a key.
        """
        if not node.args:
            return set()
        if node.func.attr in {"append", "add"}:  # type: ignore[union-attr]
            return self._infer_class_types_from_value(node.args[0])
        if node.func.attr in {"extend", "update"}:  # type: ignore[union-attr]
            return self._infer_container_element_types(node.args[0])
        return set()

    def _container_mutation_value_origins(self, node: ast.Call) -> Set[Origin]:
        """Origins a mutating container call stores, mirroring the type rule.

        The same append/extend split applies: ``append`` stores the argument
        itself, ``update`` stores the argument's contents. Keeping the two rules
        in step is what makes ``self.children.update({name: child})`` register
        ``child`` rather than the dict wrapped around it.
        """
        if not node.args or not isinstance(node.func, ast.Attribute):
            return set()
        if node.func.attr in {"append", "add"}:
            return self._value_origins(node.args[0])
        if node.func.attr in {"extend", "update"}:
            return self._element_origins(node.args[0])
        return set()

    def _record_container_mutation(self, node: ast.Call) -> None:
        """Learn element types introduced by mutating container calls.

        Handles both a local (``items.append(x)``) and an attribute container
        (``self.items.append(x)``). The attribute case is the one that crosses
        method boundaries, so it is what lets a registry populated in one method
        be resolved where another method iterates it.
        """
        if not isinstance(node.func, ast.Attribute):
            return

        container = node.func.value
        # ``self.buckets[key].append(x)`` -- a container nested one level inside
        # an attribute. The nesting level is deliberately flattened away: what
        # matters for resolving a later ``for item in self.buckets[k]`` is which
        # classes ended up somewhere under ``self.buckets``, not the shape of the
        # boxes they sit in. climlab's process tree needs exactly this, since it
        # bins subprocesses into ``self.process_types[time_type]`` lists.
        if isinstance(container, ast.Subscript):
            container = container.value

        # Recorded before the type facts and independently of them. Which value
        # was stored is knowable on the first pass; what type it has may take
        # several, and gating the escape on a type would lose the fact entirely
        # for a parameter whose callers are not yet resolved.
        if isinstance(container, ast.Attribute):
            self._note_attribute_store(
                container,
                self._container_mutation_value_origins(node),
                is_container=True,
                key=container_mutation_key(node),
            )

        element_types = self._container_mutation_element_types(node)
        if not element_types:
            return

        if isinstance(container, ast.Name):
            self.types.add_container_types(container.id, element_types)
        elif isinstance(container, ast.Attribute):
            self.record_attribute_container_store(container, element_types)
