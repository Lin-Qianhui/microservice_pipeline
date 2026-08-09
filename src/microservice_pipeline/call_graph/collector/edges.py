"""Emitting edges, including the calls Python makes on your behalf.

``_add_edge`` is the single place an edge is appended. Every policy decision
about what survives into the graph -- package-prefix filtering, whether an
unresolved target is kept as a description, what confidence a fan-out earns --
is applied there and nowhere else, so the answer to \"why is this edge missing\"
has one address.

The visitors here cover explicit calls and the implicit ones: ``a + b`` reaches
``a.__add__``, ``x in c`` reaches ``c.__contains__``, ``obj[k]`` reaches
``__getitem__``, and an attribute access reaches a property getter. Those edges
are real calls and are invisible in the source text.

Requires from siblings: ``_resolve_callees``, ``_resolve_dynamic_getattr_callees``,
``_receiver_types_for_expr``, ``_resolve_property_getter_targets``,
``_record_container_mutation``, ``_note_receiver_invocation``,
``_add_registration_edges``, ``_visit_call_children``.
"""

from __future__ import annotations

import ast
from typing import List, Tuple
from ..dunders import (
    BINOP_DUNDER_METHODS,
    COMPARE_DUNDER_METHODS,
    REVERSE_BINOP_DUNDER_METHODS,
    UNARY_DUNDER_METHODS,
)
from ..models import (
    CONFIDENCE_HIGH,
    CONFIDENCE_UNKNOWN,
    NO_SOURCE_SITE,
    Edge,
    confidence_for_fanout,
)
from .health import note_call_health
from .shapes import unique_callee_results

from .state import CollectorState


class EdgesMixin(CollectorState):
    """Emitting edges, including the calls Python makes on your behalf."""

    def _add_edge(
        self,
        callee: str,
        relation: str,
        is_resolved: bool,
        lineno: int,
        col_offset: int = NO_SOURCE_SITE,
        confidence: str = CONFIDENCE_HIGH,
    ) -> None:
        """Append an edge after applying unresolved/external filtering policy.

        ``col_offset`` completes the source *site*, which a line alone does not
        identify -- on climlab 30% of call expressions share a line with another
        call. It stays ``NO_SOURCE_SITE`` for edges whose position is not a call
        the interpreter makes at that spot, so they can be excluded from any
        site-level comparison instead of being matched against the wrong call.
        """
        if self.current_callable is None:
            self.health.dropped_no_caller += 1
            return
        # Tallied *before* the filter that drops it. Counting only what survives
        # is what made "unresolved: 0" a tautology rather than a measurement:
        # with include_external off, an unresolved edge cannot reach the list, so
        # the reported unresolved count was structurally always zero.
        if not is_resolved and not self.include_external:
            self.health.dropped_unresolved[relation] += 1
            return
        if is_resolved and not self._is_internal_callee(callee):
            self.health.dropped_external[relation] += 1
            return
        self.health.emitted[relation] += 1
        self.edges.append(
            Edge(
                caller=self.current_callable,
                callee=callee,
                file=str(self.file),
                lineno=lineno,
                col_offset=col_offset,
                resolved=is_resolved,
                relation=relation,
                confidence=confidence if is_resolved else CONFIDENCE_UNKNOWN,
            )
        )

    def _resolve_method_callees_for_expr(
        self, receiver: ast.AST, method_name: str, relation: str
    ) -> List[Tuple[str, str, bool]]:
        targets: List[Tuple[str, str, bool]] = []
        for receiver_type in sorted(self._receiver_types_for_expr(receiver)):
            resolved_targets = self.project_index.resolve_method_targets(receiver_type, method_name)
            if resolved_targets:
                targets.extend(
                    (target, relation, True) for target in resolved_targets
                )
            else:
                targets.append((f"{receiver_type}.{method_name}", relation, False))
        return unique_callee_results(targets)

    def _implicit_receiver_arg_offset(self, callee: str, relation: str) -> int:
        """Account for implicit ``self``/``cls`` in positional parameter facts."""
        if callee in self.project_index.static_method_ids:
            return 0
        if relation == "constructor":
            return 1
        if relation == "class_method":
            return 1 if callee in self.project_index.class_method_ids else 0
        # ``virtual_override`` belongs here for the same reason ``self_method``
        # does: both arise from ``self.hook(...)`` and both name a bound method of
        # a known class, so argument 0 is the first *real* parameter. Omitting it
        # does not drop the fact, it files it one slot early -- argument 0's type
        # lands in the ``self`` slot of ``positional_types`` and is read back by
        # ``_seed_param_types``. Worse for the registry passes, which restrict
        # themselves to ``self``/``cls`` calls and so see little else.
        if relation in {
            "self_method",
            "super_method",
            "inferred_type",
            "virtual_override",
        }:
            if self.project_index.callable_belongs_to_known_class(callee):
                return 1
        return 0

    def _add_dunder_edges(
        self,
        receiver: ast.AST,
        method_name: str,
        relation: str,
        lineno: int,
        col_offset: int = NO_SOURCE_SITE,
        confidence: str = CONFIDENCE_HIGH,
    ) -> None:
        results = self._resolve_method_callees_for_expr(receiver, method_name, relation)
        confidence = confidence_for_fanout(sum(1 for result in results if result[2]))
        for callee, edge_relation, is_resolved in results:
            self._add_edge(
                callee, edge_relation, is_resolved, lineno, col_offset, confidence
            )

    def _add_membership_edges(
        self, container: ast.AST, lineno: int, col_offset: int = NO_SOURCE_SITE
    ) -> None:
        """Model ``x in container`` as ``__contains__`` or iteration fallback."""
        targets: List[Tuple[str, str, bool]] = []
        for receiver_type in sorted(self._receiver_types_for_expr(container)):
            contains_targets = self.project_index.resolve_method_targets(receiver_type, "__contains__")
            if contains_targets:
                targets.extend(
                    (target, "dunder_contains", True)
                    for target in contains_targets
                )
                continue

            iter_targets = self.project_index.resolve_method_targets(receiver_type, "__iter__")
            if iter_targets:
                targets.extend(
                    (target, "dunder_iter", True) for target in iter_targets
                )
            else:
                targets.append(
                    (f"{receiver_type}.__contains__", "dunder_contains", False)
                )

        results = unique_callee_results(targets)
        confidence = confidence_for_fanout(sum(1 for result in results if result[2]))
        for callee, relation, is_resolved in results:
            self._add_edge(
                callee, relation, is_resolved, lineno, col_offset, confidence
            )

    # Subclass API. Nothing in this package calls it; the sole caller is
    # ``summary_collectors.ReturnSummaryCollector.visit_Call``, which replaces
    # ``visit_Call`` wholesale and still needs the child-visiting half of it.
    # It looks like dead code and is not -- deleting it fails at runtime, not
    # at import.
    def _visit_call_children(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)
        else:
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Call(self, node: ast.Call) -> None:
        """Resolve an explicit call, record its edge, then visit child expressions."""
        self._record_container_mutation(node)
        self._note_receiver_invocation(node.func)

        dynamic_resolved = self._resolve_dynamic_getattr_callees(node.func)
        resolved_callees = dynamic_resolved or self._resolve_callees(node.func)
        note_call_health(self.health, self.current_callable, node, resolved_callees)
        confidence = confidence_for_fanout(
            sum(1 for result in resolved_callees if result[2])
        )
        for callee, relation, is_resolved in resolved_callees:
            self._add_edge(
                callee,
                relation,
                is_resolved,
                node.lineno,
                node.col_offset,
                confidence,
            )
        self._add_registration_edges(node)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Turn a loaded property access into its implicit getter call."""
        if isinstance(node.ctx, ast.Load):
            targets = self._resolve_property_getter_targets(node)
            confidence = confidence_for_fanout(len(targets))
            for callee in targets:
                self._add_edge(
                    callee,
                    "property_getter",
                    True,
                    node.lineno,
                    node.col_offset,
                    confidence,
                )
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Model ``obj[key]`` load/store/delete through the relevant dunder method."""
        if isinstance(node.ctx, ast.Store):
            self._add_dunder_edges(
                node.value, "__setitem__", "dunder_setitem", node.lineno, node.col_offset
            )
        elif isinstance(node.ctx, ast.Del):
            self._add_dunder_edges(
                node.value, "__delitem__", "dunder_delitem", node.lineno, node.col_offset
            )
        else:
            self._add_dunder_edges(
                node.value, "__getitem__", "dunder_getitem", node.lineno, node.col_offset
            )
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        method_name = BINOP_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.left, method_name, "dunder_operator", node.lineno, node.col_offset
            )
        reverse_method_name = REVERSE_BINOP_DUNDER_METHODS.get(type(node.op))
        if reverse_method_name:
            self._add_dunder_edges(
                node.right, reverse_method_name, "dunder_operator", node.lineno, node.col_offset
            )
        self.visit(node.left)
        self.visit(node.right)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        self.visit(left)
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                self._add_membership_edges(comparator, node.lineno, node.col_offset)
            else:
                method_name = COMPARE_DUNDER_METHODS.get(type(op))
                if method_name:
                    self._add_dunder_edges(
                        left, method_name, "dunder_operator", node.lineno, node.col_offset
                    )
            self.visit(comparator)
            left = comparator

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        method_name = UNARY_DUNDER_METHODS.get(type(node.op))
        if method_name:
            self._add_dunder_edges(
                node.operand, method_name, "dunder_operator", node.lineno, node.col_offset
            )
        self.visit(node.operand)
