"""Re-project framework registration into hook-to-hook edges.

A registration is a call like ``parent.add_subprocess(child)``: it does not
itself invoke anything, but it wires the child in so that a later call to the
parent's hook reaches the child's. The rules that say which calls count are
derived in ``registration`` from escape and invoke summaries, never hard-coded,
so a new framework is a declared rule rather than a code change.

Requires from siblings: ``_add_edge``, ``get_expr_types``,
``_class_attr_element_types``, ``_resolve_method_targets``,
``_class_and_ancestors``, ``_resolve_subclass_override_targets``.
"""

from __future__ import annotations

import ast
from typing import Set
from ..registration import registration_child_exprs
from .constants import SELF_NAMES

from .state import CollectorState


class RegistrationEdgesMixin(CollectorState):
    """Re-project framework registration into hook-to-hook edges."""

    def _registration_parent_types(
        self, node: ast.Call, callee: str, relation: str
    ) -> Set[str]:
        """Which object this call registers a child *into*.

        Usually the receiver: ``parent.attach(child)``. For a constructor it is
        the class being built, since ``Pipeline(steps=[...])`` registers into the
        instance the call is producing and there is no receiver expression to
        read it from.
        """
        if relation == "constructor":
            owner = callee.rsplit(".", 1)[0]
            return {owner} if self.project_index.is_known_class(owner) else set()

        if not isinstance(node.func, ast.Attribute):
            return set()
        receiver = node.func.value
        if (
            isinstance(receiver, ast.Name)
            and receiver.id in SELF_NAMES
            and self.current_class
        ):
            class_id = self.current_class_id()
            return {class_id} if class_id else set()
        return self._receiver_types_for_expr(receiver)

    def _registration_child_types(self, child_expr: ast.AST) -> Set[str]:
        if isinstance(child_expr, ast.Call):
            return self._infer_class_types_from_call(child_expr)
        if isinstance(child_expr, (ast.Name, ast.Attribute)):
            return self.get_expr_types(child_expr)
        return set()

    def _registration_hook(self, parent_type: str, child_type: str, method: str) -> str:
        """Pick the method to link, re-projecting through a template method.

        The registry is usually invoked through a method the framework's base
        class defines and nobody overrides -- climlab calls ``proc.compute()``,
        which resolves to ``TimeDependentProcess.compute`` for every process in
        the tree. An edge between two copies of that node says nothing, and hub
        policy discards it downstream anyway.

        So when the invoked method resolves to the same definition for parent and
        child, follow that definition's delegation one hop and look for the hook
        it hands off to that subclasses actually override -- the Template Method
        shape. ``compute`` delegates to ``_compute_type``, ``_compute`` and
        ``_build_process_type_list``; only ``_compute`` is overridden anywhere, so
        only ``_compute`` carries information about which process this is.

        Returns the method name to link, or ``""`` when re-projection is needed
        but does not resolve to exactly one hook.
        """
        parent_targets = self.project_index.resolve_method_targets(parent_type, method)
        child_targets = self.project_index.resolve_method_targets(child_type, method)
        if len(parent_targets) != 1 or len(child_targets) != 1:
            return ""
        if parent_targets[0] != child_targets[0]:
            # Parent and child give the invoked method different bodies, so it
            # already distinguishes them and needs no re-projection.
            return method

        shared = parent_targets[0]
        hooks = [
            hook
            for hook in sorted(self.registry_facts.self_delegations.get(shared, ()))
            if self._is_overridden_hook(shared, hook)
        ]
        # More than one candidate means the base delegates to several genuine
        # hooks and nothing here says which one carries the coupling. Guessing
        # would fabricate edges, so decline.
        return hooks[0] if len(hooks) == 1 else ""

    def _is_overridden_hook(self, base_callable: str, method: str) -> bool:
        owner = base_callable.rsplit(".", 1)[0]
        return bool(self.project_index.resolve_subclass_override_targets(owner, method))

    def _add_registration_edges(self, node: ast.Call) -> None:
        """Emit parent/child coupling for a call that registers one into the other.

        Registering a child establishes execution coupling that is not an
        ordinary source-level call from one hook to the other, so no amount of
        call resolution finds it. What makes this call a registration is derived
        rather than recognised by name: some parameter of the callee is known to
        escape into an attribute of ``self``, and elements of that attribute are
        known to be invoked. See ``registration``.

        The edge is emitted only when both sides resolve unambiguously. That
        precision is the whole value of doing this at the call site -- the
        attribute itself has lost which parent each child belongs to.
        """
        if not self.registration_rules:
            return

        for callee, relation, is_resolved in self._resolve_callees(node.func):
            if not is_resolved:
                continue
            rule = self.registration_rules.get(callee)
            if rule is None:
                continue

            offset = self._implicit_receiver_arg_offset(callee, relation)
            parent_types = sorted(
                self._registration_parent_types(node, callee, relation)
            )
            if len(parent_types) != 1:
                continue

            for child_expr in registration_child_exprs(node, rule, offset):
                child_types = sorted(self._registration_child_types(child_expr))
                if len(child_types) != 1:
                    continue

                hook = self._registration_hook(
                    parent_types[0], child_types[0], rule.invoked_method
                )
                if not hook:
                    continue

                parent_hook = self.project_index.resolve_method_targets(parent_types[0], hook)
                child_hook = self.project_index.resolve_method_targets(child_types[0], hook)
                if len(parent_hook) != 1 or len(child_hook) != 1:
                    continue
                if not self._is_internal_callee(parent_hook[0]):
                    continue

                # Emitted against the parent's hook rather than the enclosing
                # callable: the coupling belongs to the two objects, not to
                # whichever function happened to wire them together.
                caller = self.current_callable
                self.current_callable = parent_hook[0]
                try:
                    # No ``col_offset`` on purpose. Rebinding the caller above
                    # leaves the position pointing at the registration call,
                    # which sits in a different callable, so ``(caller, line,
                    # col)`` would not name a site inside ``parent_hook`` at all.
                    # NO_SOURCE_SITE keeps it out of any site-level comparison
                    # rather than matching it against an unrelated call.
                    self._add_edge(
                        child_hook[0], "registered_invoke", True, node.lineno
                    )
                finally:
                    self.current_callable = caller
