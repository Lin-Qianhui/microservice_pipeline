"""Which *callables* an expression may hold.

The type lattice is a set of class ids, so a value that *is* code is
inexpressible in it -- ``f = solve; f(x)`` loses the edge unless callable ids
are tracked as their own dimension. This file is that dimension: the callable
a name is bound to, the callable inside a ``functools.partial``, the callables
in a dispatch table, and the identity of a lambda.

That generalization is what turns higher-order flow (``solve_ivp(rhs, ...)``,
``Thread(target=f)``, ``Pool.map(f, xs)``) from a special case into the ordinary
path.

Requires from siblings: ``_resolve_callees``, ``_lambda_id``,
``_class_attr_element_types``, ``_attribute_owner_types``, ``get_var_sources``.
"""

from __future__ import annotations

import ast
from typing import Dict, Optional, Set
from ..ast_utils import attribute_to_name, unwrap_passthrough
from ..models import ClassAttr
from .constants import ELEMENT_SUFFIX, PARTIAL_NAMES, SELF_NAMES

from .state import CollectorState


class CallableValueMixin(CollectorState):
    """Which *callables* an expression may hold."""

    def _infer_callable_ids_from_value(
        self, value: ast.AST, *, resolve_names: bool = True
    ) -> Set[str]:
        """Callables an expression may evaluate to, mirroring class inference.

        The counterpart of ``_infer_class_types_from_value`` for values that
        *are* code. Written as a separate walk rather than an extra branch there
        because the two answers must never mix: a class id and a callable id are
        both dotted strings, and every consumer of a type set assumes the former.

        ``resolve_names`` distinguishes the two contexts this is asked from. As a
        *value* -- ``handler = helper``, or ``f(helper)`` -- a bare name should
        resolve to the function it refers to. As a *callee* it must not: ordinary
        name resolution already handles ``helper()`` and gives it the more
        specific relation, so resolving it here too would relabel every plain
        call as ``inferred_callable``.
        """
        value = unwrap_passthrough(value)

        if isinstance(value, ast.Name):
            known = self.types.get_var_callables(value.id)
            if known:
                return set(known)
            if not resolve_names:
                return set()
            resolved = self._resolve_callee_definition(value)
            return {resolved} if resolved else set()

        if isinstance(value, ast.Attribute):
            callables = self._class_attr_callable_types(value)
            if callables:
                return callables
            if not resolve_names:
                return set()
            resolved = self._resolve_callee_definition(value)
            return {resolved} if resolved else set()

        if isinstance(value, ast.Lambda):
            lambda_id = self._lambda_id(value)
            return {lambda_id} if lambda_id in self.callable_ids else set()

        if isinstance(value, ast.Call):
            return self._infer_callable_ids_from_call(value)

        # Same recursion as the class walk: these evaluate to one of their
        # sub-expressions, and the answer is knowable for each.
        if isinstance(value, ast.IfExp):
            return self._infer_callable_ids_from_value(
                value.body
            ) | self._infer_callable_ids_from_value(value.orelse)
        if isinstance(value, ast.BoolOp):
            callables: Set[str] = set()
            for operand in value.values:
                callables.update(self._infer_callable_ids_from_value(operand))
            return callables

        # ``SOLVERS[name]()`` -- a dispatch table, which is the dominant
        # config-driven dispatch idiom in scientific code.
        if isinstance(value, ast.Subscript) and not isinstance(value.slice, ast.Slice):
            return self._infer_container_callable_ids(value.value)

        return set()

    def _infer_callable_ids_from_call(self, value: ast.Call) -> Set[str]:
        """Callables produced *by* a call: ``partial(f, x)`` and factories."""
        fn_name = attribute_to_name(value.func)
        if fn_name in PARTIAL_NAMES and value.args:
            # ``partial(f, ...)`` is ``f`` with arguments pre-bound; calling the
            # result calls ``f``.
            return self._infer_callable_ids_from_value(value.args[0])

        callables: Set[str] = set()
        for callee, _relation, is_resolved in self._resolve_callees(value.func):
            if not is_resolved:
                continue
            summary = self.return_summaries.get(callee)
            if summary:
                callables.update(summary.callable_ids)
        return callables

    def _infer_container_callable_ids(self, value: ast.AST) -> Set[str]:
        """Callables held *inside* a collection, for dispatch tables."""
        value = unwrap_passthrough(value)
        if isinstance(value, ast.Name):
            return set(self.types.get_var_callables(f"{value.id}{ELEMENT_SUFFIX}"))
        if isinstance(value, ast.Attribute):
            return self._class_attr_element_callable_types(value)
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            callables: Set[str] = set()
            for item in value.elts:
                callables.update(self._infer_callable_ids_from_value(item))
            return callables
        if isinstance(value, ast.Dict):
            callables = set()
            for item in value.values:
                callables.update(self._infer_callable_ids_from_value(item))
            return callables
        return set()

    def _resolve_callee_definition(self, value: ast.AST) -> Optional[str]:
        """Resolve a name/attribute to a project callable, or ``None``.

        A *reference* to a function, not a call of it. Only fully resolved
        project callables count -- an unresolved guess as a value would be a
        fabricated callable id rather than a fabricated callee, which is worse.
        """
        resolved = self._resolve_callee(value)
        if resolved and resolved[2] and resolved[0] in self.callable_ids:
            return resolved[0]
        return None

    def _class_attr_callable_types(self, value: ast.Attribute) -> Set[str]:
        """Callables stored on ``self.<attr>``, looked up across the hierarchy."""
        return self._attr_callables(value, self.class_attr_types.callable_types)

    def _class_attr_element_callable_types(self, value: ast.AST) -> Set[str]:
        if not isinstance(value, ast.Attribute):
            return set()
        return self._attr_callables(value, self.class_attr_types.element_callable_types)

    def _attr_callables(
        self, value: ast.Attribute, table: Dict[ClassAttr, Set[str]]
    ) -> Set[str]:
        if not (isinstance(value.value, ast.Name) and value.value.id in SELF_NAMES):
            return set()
        class_id = self.current_class_id()
        if not class_id:
            return set()
        found: Set[str] = set()
        for ancestor in self.project_index.class_and_ancestors(class_id):
            found.update(table.get((ancestor, value.attr), set()))
        return found
