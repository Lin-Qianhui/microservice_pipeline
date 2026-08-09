"""The collector's shared state, and the facades onto it.

Every mixin in this package operates on one instance of ``CollectorState``.
It owns the constructor, all mutable state, the class-level configuration
flags, the scoping of deferred return facts, and the delegating facades onto
:class:`~..type_env.TypeEnv` and :class:`~..project_index.ProjectIndex`.

Nothing here resolves or infers anything. It is the substrate the rest of the
package writes to, and the single place to look up what an attribute means.

Held by reference, not copied: ``return_summaries``, ``param_summaries``,
``class_attr_types``, ``escape_summaries``, ``registry_facts``,
``registration_rules``, ``module_map``, ``known_classes``, ``callable_ids``,
and ``project_index``. That is deliberate -- the pass driver in ``passes``
decides when an accumulated fact becomes visible to the next pass, and it does
so by mutating the object every collector already holds.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Set, Tuple

from ..discovery import module_callable_id
from ..models import (
    CallGraphHealth,
    ClassAttrTypes,
    Edge,
    FunctionEscapeSummary,
    FunctionParamSummary,
    FunctionReturnSummary,
    ModuleIndex,
    RegistryFacts,
)
from ..project_index import ProjectIndex
from ..return_links import ReturnLink, ReturnLinkTable
from ..type_env import TypeEnv
from ..registration import RegistrationRule


class CollectorState(ast.NodeVisitor):
    """Everything a collector holds, and the facades onto the objects it shares.

    The visitor performs deliberately small-scale type inference, whose scoped
    state lives in ``self.types`` (:class:`~..type_env.TypeEnv`) and whose
    whole-project facts live in ``self.project_index``
    (:class:`~..project_index.ProjectIndex`).

    Resolution prefers known project definitions, but can retain an unresolved
    descriptive target when ``include_external`` is enabled.
    """

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
        class_attr_types: Optional[ClassAttrTypes] = None,
        return_links: Optional[ReturnLinkTable] = None,
        project_index: Optional[ProjectIndex] = None,
        escape_summaries: Optional[Dict[str, FunctionEscapeSummary]] = None,
        registry_facts: Optional[RegistryFacts] = None,
        registration_rules: Optional[Dict[str, RegistrationRule]] = None,
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
            class_attr_types if class_attr_types is not None else ClassAttrTypes()
        )
        # Registry evidence, held by reference like the summaries above so the
        # pass driver decides when a fact becomes visible. Only
        # ``TypeSummaryCollector`` writes to these; every collector reads them,
        # since re-projection needs the delegation facts during the edge pass.
        self.escape_summaries = (
            escape_summaries if escape_summaries is not None else {}
        )
        self.registry_facts = (
            registry_facts if registry_facts is not None else RegistryFacts()
        )
        # Populated only for the final edge pass, once escape and invoke facts
        # have been joined. Empty everywhere else, which is what keeps the
        # summary passes from emitting edges.
        self.registration_rules = registration_rules or {}
        # Whole-project class facts. Shared across every file and every pass of a
        # run when the caller supplies one; built here only for callers that have
        # no run to share (tests, one-off use).
        self.project_index = (
            project_index
            if project_index is not None
            else ProjectIndex(module_map, known_classes, callable_ids)
        )
        self.module_callable = module_callable_id(module)

        # Deferred "we return whatever that callee returns" facts. ``None``
        # switches recording off, which is what every collector except
        # ``ReturnSummaryCollector`` wants.
        self.return_links = return_links
        self._link_target: List[Tuple[str, Optional[int]]] = []

        self.current_callable: Optional[str] = None
        self.current_class: Optional[str] = None
        # Lexical *function* nesting, which is what decides a nested callable's
        # ID. Tracked apart from ``current_callable`` because a class body is a
        # callable that edges are attributed to but that never prefixes the name
        # of a method defined inside it -- ``C.method``, never ``C.<locals>.method``.
        self.enclosing_function: Optional[str] = None
        self.callable_stack: List[str] = []
        self.types = TypeEnv()
        self.edges: List[Edge] = []
        self.health = CallGraphHealth()

    @contextmanager
    def _recording_return_links(
        self, target_field: str, slot: Optional[int] = None
    ) -> Generator[None, None, None]:
        """Declare which part of the current callable's return summary is being filled.

        Type inference recurses through arbitrary expressions, and by the time it
        reaches a call it no longer knows what it is computing a value *for*.
        This marks that at the top, so ``_note_return_dependency`` further down
        can pair "which field we are filling" with "which field we read".
        """
        self._link_target.append((target_field, slot))
        try:
            yield
        finally:
            self._link_target.pop()

    @contextmanager
    def _resolving_receiver(self) -> Generator[None, None, None]:
        """Infer a receiver's type without charging it to the enclosing return.

        ``return build().submit()`` returns *submit's* value, not ``build``'s.
        Working out that the receiver is an ``Order`` runs the same inference
        that records return links, so without this the enclosing
        ``_recording_return_links`` block would still be in force and we would
        record "our return type includes build's" -- which is false.

        The link for ``submit`` itself is recorded by the caller of this block,
        outside the suspension, so only the receiver hop is dropped.
        """
        saved = self._link_target
        self._link_target = []
        try:
            yield
        finally:
            self._link_target = saved

    def _note_return_dependency(
        self, callee: str, source_field: str, source_slot: Optional[int] = None
    ) -> None:
        """Remember that our return type depends on ``callee``'s, known or not.

        Recorded even when the callee already has a summary: that summary can
        still grow in a later round, and the link is what carries the growth
        onwards.
        """
        if self.return_links is None or not self._link_target:
            return
        if not self.current_callable:
            return
        target_field, target_slot = self._link_target[-1]
        self.return_links.record(
            ReturnLink(
                caller=self.current_callable,
                callee=callee,
                source_field=source_field,
                target_field=target_field,
                source_slot=source_slot,
                target_slot=target_slot,
            )
        )

    def _is_internal_callee(self, callee: str) -> bool:
        """Apply optional package-prefix filtering to a resolved target."""
        if not self.package_prefix:
            return True
        if isinstance(self.package_prefix, str):
            prefixes = (self.package_prefix,)
        else:
            prefixes = tuple(self.package_prefix)
        return any(callee == prefix or callee.startswith(prefix + ".") for prefix in prefixes)

    # -- value origins ---------------------------------------------------------
    #
    # Tracking which *value* a name currently holds, as opposed to which type.
    # A registration is an identity fact -- "the thing you passed is the thing
    # that got stored" -- and types cannot express it: two unrelated parameters
    # of the same class are indistinguishable by type and completely different
    # by identity. Only ``TypeSummaryCollector`` turns the resulting facts into
    # summaries; the others compute origins and discard them, which costs a few
    # dictionary lookups and keeps one implementation of the store shapes.

    records_registry_facts = False

    def current_class_id(self) -> Optional[str]:
        if self.current_class:
            return f"{self.module}.{self.current_class}"
        return None

    # This ``ProjectIndex`` adapter is not pure forwarding: it binds this
    # collector's module context into a whole-project lookup. The other adapter,
    # ``_resolve_star_import_targets``, lives in ``ResolutionMixin`` beside its
    # only consumers and similarly supplies ``self.module_index``. Every other
    # facade was deleted in favour of calling ``self.types`` and
    # ``self.project_index`` directly, so the rule now reads: a method on the
    # collector always does something. Keeping this adapter avoids repeating the
    # module-context threading at each call site.
    def _resolve_class_reference_name(self, name: str) -> List[str]:
        return self.project_index.resolve_class_reference_name(
            self.module, self.module_index, name
        )
