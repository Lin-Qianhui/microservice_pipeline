"""Where ``CallCollector`` is assembled.

``CallCollector`` is the heart of the analyzer. It walks a module tracking what
type each name currently holds, and turns every call -- and every implicit call
Python performs on your behalf, such as ``a + b`` reaching ``a.__add__`` -- into
a graph edge. Two subclasses in ``summary_collectors`` reuse that machinery
without emitting edges:

``ReturnSummaryCollector``
    Records what each callable gives back.
``TypeSummaryCollector``
    Records the types flowing into parameters and instance attributes.

All three read accumulated type facts through the ``return_summaries``,
``param_summaries``, and ``class_attr_types`` dictionaries handed to them. Those
are held by reference, not copied, so the pass driver in ``passes`` controls
when facts become visible.

The class body lives in one module per concern, listed below in the order they
are mixed in. That order is a reading gradient -- most-derived semantics first,
the shared state last -- so this list doubles as a table of contents.

Three invariants make the split safe, and are pinned by ``test_collector_*`` in
``tests/test_call_graph_ast.py``:

1. **Every mixin's base list is exactly ``(CollectorState,)``.** A mixin must
   never inherit another mixin. That keeps the MRO a single diamond C3 can
   linearise trivially.
2. **``CollectorState`` is always last** in the bases below. Putting it first
   makes C3 fail outright.
3. **No method name is defined in two mixins.** Because nothing is overridden,
   C3 never has to arbitrate, which is what makes the order above semantically
   inert rather than load-bearing.

The mixins are slices of one object, not collaborators: they all read and write
the same ``self``. That is deliberate. ``resolution`` and ``inference`` are
mutually recursive -- resolving ``build().submit()`` needs the inferred type of
``build()``, and inferring that type needs to resolve ``build`` -- so splitting
them into separate *objects* would produce an import cycle rather than a
separation of concerns. See ``modularisation_plan.md`` for what that means for
what this split does and does not buy.

Subclass API. ``summary_collectors`` reaches past the public surface into these
private members. Treat them as a contract; none of them is dead code even when
nothing in this package calls it:

    _element_origins            _recording_return_links
    _implicit_receiver_arg_offset   _resolve_callees
    _infer_callable_ids_from_value  _resolve_dynamic_getattr_callees
    _infer_class_types_from_value   _value_origins
    _infer_container_element_types  _visit_call_children
    _note_receiver_invocation       _record_container_mutation

plus the attributes ``current_callable``, ``escape_summaries``,
``param_summaries``, ``registry_facts``, ``return_links``, ``types``, and the
class-level flag ``records_registry_facts``.
"""


from __future__ import annotations

from .callables import CallableValueMixin
from .edges import EdgesMixin
from .expressions import ExpressionsMixin
from .inference import InferenceMixin
from .origins import OriginsMixin
from .registration_edges import RegistrationEdgesMixin
from .resolution import ResolutionMixin
from .scopes import ScopesMixin
from .state import CollectorState
from .statements import StatementsMixin


class CallCollector(
    ResolutionMixin,
    InferenceMixin,
    CallableValueMixin,
    RegistrationEdgesMixin,
    EdgesMixin,
    StatementsMixin,
    ScopesMixin,
    OriginsMixin,
    ExpressionsMixin,
    CollectorState,
):
    """Resolve calls and implicit Python operations into graph edges.

    The visitor also performs deliberately small-scale type inference, whose
    scoped state lives in ``self.types`` (:class:`~..type_env.TypeEnv`) and
    whose whole-project facts live in ``self.project_index``
    (:class:`~..project_index.ProjectIndex`).

    Resolution prefers known project definitions, but can retain an unresolved
    descriptive target when ``include_external`` is enabled.
    """
