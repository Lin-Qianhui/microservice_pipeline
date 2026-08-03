"""Data structures shared by every call-graph analysis pass.

This module is the bottom of the ``call_graph`` package: it holds the graph
node/edge records, the per-module index, the inferred-type summaries, and the
small helpers that union type facts together. It imports nothing else from the
package, so every other module here is free to depend on it.

The type-fact helpers all *merge* rather than replace. Inference runs repeatedly
over the same code and each visit can only ever learn more, so a later pass must
never be able to erase what an earlier one established.
"""


from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Imports and top-level statements execute in module scope, so each module gets
# a synthetic callable node that can act as their caller.
MODULE_CALLABLE_QUALNAME = "<module>"


@dataclass
class CallableDef:
    """A graph node representing executable code discovered in one source file.

    ``id`` is the globally unique dotted identifier used by edges. ``qualname``
    is its module-relative form, while ``kind`` distinguishes module scope,
    functions, and methods. ``class_name`` is populated only for class members.
    """
    id: str
    module: str
    qualname: str
    file: str
    lineno: int
    kind: str  # module | function | method
    class_name: Optional[str] = None


@dataclass
class ModuleIndex:
    """Names and class metadata learned during the definition-indexing pass.

    This is the analyzer's lightweight substitute for importing a module and
    inspecting its runtime namespace.
    """
    module: str
    path: Path
    imports: Dict[str, str] = field(default_factory=dict)  # alias -> fully qualified target
    classes: Dict[str, str] = field(default_factory=dict)  # class name -> callable id namespace
    class_bases: Dict[str, List[str]] = field(default_factory=dict)  # class id -> base class ids
    # Classes whose body runs a call and therefore exists as a callable node.
    # Only these may be used as an edge's caller; see definitions.class_body_evaluates_calls.
    class_bodies: Set[str] = field(default_factory=set)
    properties: Set[str] = field(default_factory=set)  # property getter callable ids
    static_methods: Set[str] = field(default_factory=set)
    class_methods: Set[str] = field(default_factory=set)
    star_imports: List[str] = field(default_factory=list)  # modules imported via from x import *


# ``col_offset`` for an edge that has no single call expression behind it, so it
# can never be matched against an observed call site. ``registered_invoke`` is
# the case that matters: it is attributed to the parent's hook but positioned at
# the registration call, which lives in a different callable entirely.
NO_SOURCE_SITE = -1

# How sure the analyzer is that an edge describes a call that really happens,
# as distinct from ``relation``, which says how the edge was *derived*. The two
# were conflated before: a receiver with one possible type and a receiver with
# five both emitted ``inferred_type``, and the structural graph could not tell a
# certainty from one guess among several.
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_UNKNOWN = "unknown"


def confidence_for_fanout(fanout: int) -> str:
    """Grade a call site by how many targets it resolved to.

    The thresholds are taken from measured confirmation rates on climlab rather
    than chosen for tidiness. Fan-out is *not* monotone with wrongness at the low
    end -- sites with two targets confirmed as often as or better than sites with
    one (``constructor`` 3/3 vs 62/63, ``direct`` 6/6 vs 60/64, ``imported`` 9/9
    vs 58/73) -- so a rule that demoted everything above one target would punish
    the wrong bucket. Degradation begins at three (``self_method`` 6/14,
    ``virtual_override`` 36/60), which is where the grading begins too.

    Re-derive these on a second codebase before treating them as settled.
    """
    if fanout <= 0:
        return CONFIDENCE_UNKNOWN
    if fanout <= 2:
        return CONFIDENCE_HIGH
    if fanout <= 5:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


@dataclass
class Edge:
    """A directed relationship from one callable to another.

    ``resolved`` means ``callee`` was matched to an indexed project callable;
    ``relation`` records the rule that produced the match. ``file``, ``lineno``
    and ``col_offset`` locate the source expression responsible for the edge --
    all three, because a line is not a site: nearly a third of real call
    expressions share a line with another call, so ``(line, col)`` is what
    identifies the call the interpreter actually made.
    """
    caller: str
    callee: str
    file: str
    lineno: int
    resolved: bool
    # direct | self_method | super_method | imported | inferred_type
    # attribute | dynamic_getattr | constructor | import | class_method
    # property_getter | dunder_getitem | dunder_setitem | dunder_delitem
    # dunder_contains | dunder_iter | dunder_operator
    # virtual_override | registered_invoke
    relation: str
    col_offset: int = NO_SOURCE_SITE
    # How many of the site's alternatives this edge is one of. ``relation`` says
    # how the edge was derived, which is a different question: two edges can share
    # a relation while one is the only possibility and the other is one guess in
    # five. See ``confidence_for_fanout``.
    confidence: str = CONFIDENCE_HIGH


@dataclass
class CallGraphHealth:
    """What the resolver could *not* do, counted where the loss happens.

    The artifact counts only edges that survived, which says nothing about what
    was dropped on the way. Every field here is tallied at the point of loss so
    the numbers cannot be tautological:

    ``dropped_unresolved``
        A call whose callee was named but never matched to a project callable.
        Mostly third-party (numpy, scipy) and expected; the entries that matter
        are the intra-project relations such as ``self_method``, which mean the
        analyzer failed on code it can see.
    ``dropped_external``
        Resolved, but outside the configured package prefix. Working as
        intended, separated from the above so the two are not confused.
    ``unresolvable_calls``
        A call expression the resolver returned *nothing* for -- not an
        unresolved name, but no candidate at all. This is the shape a value the
        abstract domain cannot represent produces: calling something held in a
        variable, a parameter, or a container. Bucketed by the syntactic form of
        the callee so the dominant shape is visible rather than inferred.
    ``site_fanout``
        How many targets each call site resolved to, as ``{k: number of sites}``.
        Counted at the emission point on purpose: derived downstream from
        ``edges.csv`` it would instead measure co-located independent calls,
        since nearly a third of call expressions share a line with another.
    """

    emitted: Counter = field(default_factory=Counter)
    dropped_unresolved: Counter = field(default_factory=Counter)
    dropped_external: Counter = field(default_factory=Counter)
    dropped_no_caller: int = 0
    unresolvable_calls: Counter = field(default_factory=Counter)
    site_fanout: Counter = field(default_factory=Counter)

    def merge(self, other: "CallGraphHealth") -> None:
        self.emitted.update(other.emitted)
        self.dropped_unresolved.update(other.dropped_unresolved)
        self.dropped_external.update(other.dropped_external)
        self.dropped_no_caller += other.dropped_no_caller
        self.unresolvable_calls.update(other.unresolvable_calls)
        self.site_fanout.update(other.site_fanout)

    def as_dict(self) -> Dict[str, object]:
        total_sites = sum(self.site_fanout.values())
        weighted = sum(k * n for k, n in self.site_fanout.items())
        return {
            "emitted": dict(sorted(self.emitted.items())),
            "emitted_total": sum(self.emitted.values()),
            "dropped_unresolved": dict(sorted(self.dropped_unresolved.items())),
            "dropped_unresolved_total": sum(self.dropped_unresolved.values()),
            "dropped_external": dict(sorted(self.dropped_external.items())),
            "dropped_external_total": sum(self.dropped_external.values()),
            "dropped_no_caller": self.dropped_no_caller,
            "unresolvable_calls": dict(sorted(self.unresolvable_calls.items())),
            "unresolvable_calls_total": sum(self.unresolvable_calls.values()),
            "site_fanout": {str(k): self.site_fanout[k] for k in sorted(self.site_fanout)},
            "resolved_sites": total_sites,
            "mean_fanout": (weighted / total_sites) if total_sites else 0.0,
        }


def resolvable_callable_ids(callable_map: Dict[str, CallableDef]) -> Set[str]:
    """The callables a call site is allowed to resolve *to*.

    Class bodies are nodes, because code runs there and edges are attributed to
    them, but they are never call targets: Python offers no way to invoke a class
    body, and ``SlabOcean(...)`` means ``SlabOcean.__init__``. Leaving them in the
    resolution universe makes every construction of such a class resolve to the
    body instead of the constructor, which silently deletes the real edge.
    """
    return {
        callable_id
        for callable_id, definition in callable_map.items()
        if definition.kind != "class_body"
    }


@dataclass(frozen=True)
class AnalysisFile:
    """A physical Python file paired with its import-style module name."""
    path: Path
    module: str


@dataclass
class FunctionReturnSummary:
    """Types inferred from all return statements in one callable.

    Besides direct returned object types, the analyzer tracks collection element
    types and individual tuple/list slots. Slot facts make destructuring such as
    ``order, items = build()`` useful to later inference.
    """
    class_types: Set[str] = field(default_factory=set)
    element_types: Set[str] = field(default_factory=set)
    slot_types: Dict[int, Set[str]] = field(default_factory=dict)
    slot_element_types: Dict[int, Set[str]] = field(default_factory=dict)
    # Callables handed back rather than objects: a factory returning a function,
    # a decorator returning its wrapper. Separate from ``class_types`` because
    # these are code, not instances, and nothing may call
    # ``resolve_method_targets`` on them.
    callable_ids: Set[str] = field(default_factory=set)
    slot_callable_ids: Dict[int, Set[str]] = field(default_factory=dict)


@dataclass
class FunctionParamSummary:
    """Types observed at a callable's positional and named parameters.

    Element types are tracked beside object types for the same reason
    ``FunctionReturnSummary`` does it: a parameter that receives a list of
    ``Order`` is a different fact from one that receives an ``Order``, and code
    inside the callee iterates the former. Without this, any collection handed
    across a call boundary loses its contents.
    """
    positional_types: Dict[int, Set[str]] = field(default_factory=dict)
    named_types: Dict[str, Set[str]] = field(default_factory=dict)
    positional_element_types: Dict[int, Set[str]] = field(default_factory=dict)
    named_element_types: Dict[str, Set[str]] = field(default_factory=dict)
    # Callables passed at a parameter. This is what carries a function across a
    # call boundary: ``crp.apply(msg, cryptops.encrypt)`` records ``encrypt``
    # here, and the ``func(...)`` inside ``apply`` can then resolve.
    positional_callables: Dict[int, Set[str]] = field(default_factory=dict)
    named_callables: Dict[str, Set[str]] = field(default_factory=dict)


# Which parameter of a callable a fact is about, as ``(position, name)``. Both
# halves are kept in one key because a call site may pass the same parameter
# either way, and a fact recorded inside the callee cannot know which form its
# callers will use. ``position`` is ``-1`` for keyword-only parameters, which
# have no position to match against.
ParamKey = Tuple[int, str]

KEYWORD_ONLY_POSITION = -1


def param_key(index: int, name: str) -> ParamKey:
    return (index, name)


def keyword_only_param_key(name: str) -> ParamKey:
    return (KEYWORD_ONLY_POSITION, name)


# The attribute a value escaped into, and whether it landed inside a container
# there. ``self.children.append(x)`` is a container escape; ``self.parent = x``
# is not. The distinction is kept because the two are read back differently --
# a container's contents are element types, a scalar's are object types.
AttrSlot = Tuple[str, bool]  # (attribute name, is_container)

# A ``(class id, attribute name)`` pair. The unit that registry facts are keyed
# by, since a registry is always an attribute of some class.
ClassAttr = Tuple[str, str]


@dataclass
class FunctionEscapeSummary:
    """Which of a callable's parameters end up stored on ``self``.

    This is an *identity* fact, not a type fact: it says that whatever value was
    passed at this parameter is reachable from ``self.<attr>`` afterwards. That
    is what marks a call site as a registration -- ``parent.register(name, child)``
    couples parent to child precisely because ``child`` is retained.

    Deliberately separate from ``FunctionParamSummary``. That records what types
    flow *in*; this records where the value flows *to*, and the two are joined at
    different points.

    ``element_escapes`` is the same fact one level down: ``self.steps.extend(steps)``
    retains the *contents* of its parameter, not the parameter. The distinction
    has to survive to the call site, because it decides whether the child being
    registered is the argument or each of the argument's elements --
    ``Pipeline(steps=[Scaler(), Model()])`` registers two children, not a list.

    ``key_params`` records which parameter supplied the *key* a child was filed
    under -- the ``name`` in ``self.children.update({name: child})``. Registries
    are usually keyed by a caller-supplied label, and that label is the natural
    name for the relationship at the call site.
    """
    escapes: Dict[ParamKey, Set[AttrSlot]] = field(default_factory=dict)
    element_escapes: Dict[ParamKey, Set[AttrSlot]] = field(default_factory=dict)
    key_params: Set[ParamKey] = field(default_factory=set)

    def add(self, key: ParamKey, slots: Set[AttrSlot]) -> None:
        if slots:
            self.escapes.setdefault(key, set()).update(slots)

    def add_element(self, key: ParamKey, slots: Set[AttrSlot]) -> None:
        if slots:
            self.element_escapes.setdefault(key, set()).update(slots)

    def add_key_param(self, key: ParamKey) -> None:
        self.key_params.add(key)


@dataclass
class RegistryFacts:
    """Project-wide facts about attributes used as registries.

    ``element_flow`` records that elements of one class attribute are stored into
    another -- ``for name, proc in self.subprocess.items(): self.by_type[k].append(proc)``.
    Without it the registry an object is *registered into* and the one it is
    *invoked from* look unrelated, which is the shape climlab actually has.

    ``invocations`` records that elements of an attribute receive a method, and
    which callable does the invoking. It is the evidence that a stored value is
    executed rather than merely held, which is what separates a registry from an
    ordinary back-reference such as ``self.config``.

    ``self_delegations`` records the methods each callable invokes on ``self``.
    It exists so the template-method re-projection can ask what a base method
    hands off to without re-walking any AST; see
    ``CallCollector._registration_hook``.

    ``returned_attrs`` and ``param_attrs`` carry that same provenance across a
    call boundary. ``element_flow`` only follows a value that moves from one
    attribute straight into another, which loses the registry the moment its
    contents leave through a ``return`` or an argument::

        def get_children(self):     return [*self._children, ...]
        def draw(self, renderer):   _draw_all(self.get_children())
        def _draw_all(artists):     for a in artists: a.draw(...)

    Nothing here is stored in a second attribute, so without these two the
    ``.draw()`` cannot be attributed back to ``self._children`` and the registry
    goes unrecognised. The type summaries already carry what such a call returns;
    these carry *where it came from*.
    """
    element_flow: Dict[ClassAttr, Set[ClassAttr]] = field(default_factory=dict)
    invocations: Dict[ClassAttr, Dict[str, Set[str]]] = field(default_factory=dict)
    self_delegations: Dict[str, Set[str]] = field(default_factory=dict)
    returned_attrs: Dict[str, Set[ClassAttr]] = field(default_factory=dict)
    param_attrs: Dict[Tuple[str, ParamKey], Set[ClassAttr]] = field(default_factory=dict)

    def add_element_flow(self, source: ClassAttr, target: ClassAttr) -> None:
        if source != target:
            self.element_flow.setdefault(source, set()).add(target)

    def add_invocation(self, key: ClassAttr, method: str, caller: str) -> None:
        if method and caller:
            self.invocations.setdefault(key, {}).setdefault(method, set()).add(caller)

    def add_self_delegation(self, caller: str, method: str) -> None:
        if caller and method:
            self.self_delegations.setdefault(caller, set()).add(method)

    def add_returned_attrs(self, callable_id: str, attrs: Set[ClassAttr]) -> None:
        if callable_id and attrs:
            self.returned_attrs.setdefault(callable_id, set()).update(attrs)

    def add_param_attrs(
        self, callable_id: str, param: ParamKey, attrs: Set[ClassAttr]
    ) -> None:
        if callable_id and attrs:
            self.param_attrs.setdefault((callable_id, param), set()).update(attrs)

    def attrs_for_param(
        self, callable_id: str, index: int, name: str
    ) -> Set[ClassAttr]:
        """Look a parameter up by whichever half the recorder knew.

        A call site knows the position it passed but not the callee's parameter
        name; the callee knows its own names but not how callers spelled them.
        So each records the half it has -- ``(position, "")`` or
        ``(KEYWORD_ONLY_POSITION, name)`` -- and this joins them back up.
        """
        return self.param_attrs.get((callable_id, (index, "")), set()) | (
            self.param_attrs.get((callable_id, (KEYWORD_ONLY_POSITION, name)), set())
        )

    def reachable_attrs(self, start: ClassAttr) -> Set[ClassAttr]:
        """Every attribute an element of ``start`` can reach, including itself.

        Transitive because a value can be relayed through more than one
        attribute before it is invoked, and each relay is recorded as a single
        hop. The visited set makes a cyclic relay terminate.
        """
        seen: Set[ClassAttr] = set()
        queue = [start]
        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self.element_flow.get(current, ()))
        return seen


@dataclass
class ClassAttrTypes:
    """Types inferred for ``self.<attr>`` on each class, by attribute.

    Both maps are keyed by ``(class id, attribute name)``. ``object_types`` is
    what the attribute holds; ``element_types`` is what it *contains*, which is
    how a registry attribute such as ``self.subprocess`` carries the classes
    stored into it across method boundaries.

    Grouped into one object rather than two parallel arguments because every
    pass threads them together, and splitting them would double the plumbing on
    signatures that are already wide.
    """
    object_types: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    element_types: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    # ``self.handler = fn`` and ``self.hooks.append(fn)``. Kept out of
    # ``object_types``/``element_types`` because ``data_access`` mints a data
    # node from every entry there, and a callable id would become a phantom
    # data object in the structural graph.
    callable_types: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    element_callable_types: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)

    def add_callable_types(self, key: Tuple[str, str], callables: Set[str]) -> None:
        if callables:
            self.callable_types.setdefault(key, set()).update(callables)

    def add_element_callable_types(
        self, key: Tuple[str, str], callables: Set[str]
    ) -> None:
        if callables:
            self.element_callable_types.setdefault(key, set()).update(callables)

    def add_object_types(self, key: Tuple[str, str], types: Set[str]) -> None:
        if types:
            self.object_types.setdefault(key, set()).update(types)

    def add_element_types(self, key: Tuple[str, str], types: Set[str]) -> None:
        if types:
            self.element_types.setdefault(key, set()).update(types)


def add_types(target: Dict[str, Set[str]], key: str, types: Set[str]) -> None:
    """Union non-empty type facts into a string-keyed inference map."""
    if not types:
        return
    target.setdefault(key, set()).update(types)


def add_indexed_types(target: Dict[int, Set[str]], key: int, types: Set[str]) -> None:
    """Union non-empty type facts into an integer-keyed inference map."""
    if not types:
        return
    target.setdefault(key, set()).update(types)


def copy_type_map(source: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    return {key: set(values) for key, values in source.items()}


def copy_indexed_type_map(source: Dict[int, Set[str]]) -> Dict[int, Set[str]]:
    return {key: set(values) for key, values in source.items()}


def copy_return_summaries(
    summaries: Dict[str, FunctionReturnSummary],
) -> Dict[str, FunctionReturnSummary]:
    return {
        key: FunctionReturnSummary(
            class_types=set(value.class_types),
            element_types=set(value.element_types),
            slot_types=copy_indexed_type_map(value.slot_types),
            slot_element_types=copy_indexed_type_map(value.slot_element_types),
        )
        for key, value in summaries.items()
    }


def copy_param_summaries(
    summaries: Dict[str, FunctionParamSummary],
) -> Dict[str, FunctionParamSummary]:
    return {
        key: FunctionParamSummary(
            positional_types=copy_indexed_type_map(value.positional_types),
            named_types=copy_type_map(value.named_types),
            positional_element_types=copy_indexed_type_map(value.positional_element_types),
            named_element_types=copy_type_map(value.named_element_types),
            # Every field must be copied. One left out is *shared* between the
            # pending and previous summaries, so the equality test that decides
            # convergence is trivially satisfied on that field and the fixpoint
            # exits after one round -- silently, because nothing warns.
            positional_callables=copy_indexed_type_map(value.positional_callables),
            named_callables=copy_type_map(value.named_callables),
        )
        for key, value in summaries.items()
    }


def copy_escape_summaries(
    summaries: Dict[str, FunctionEscapeSummary],
) -> Dict[str, FunctionEscapeSummary]:
    return {
        key: FunctionEscapeSummary(
            escapes={param: set(slots) for param, slots in value.escapes.items()},
            element_escapes={
                param: set(slots) for param, slots in value.element_escapes.items()
            },
            key_params=set(value.key_params),
        )
        for key, value in summaries.items()
    }


def copy_registry_facts(facts: RegistryFacts) -> RegistryFacts:
    return RegistryFacts(
        element_flow={key: set(values) for key, values in facts.element_flow.items()},
        invocations={
            key: {method: set(callers) for method, callers in methods.items()}
            for key, methods in facts.invocations.items()
        },
        self_delegations={
            key: set(values) for key, values in facts.self_delegations.items()
        },
        returned_attrs={
            key: set(values) for key, values in facts.returned_attrs.items()
        },
        param_attrs={key: set(values) for key, values in facts.param_attrs.items()},
    )


def copy_class_attr_types(class_attr_types: ClassAttrTypes) -> ClassAttrTypes:
    # Every field, for the reason given in ``copy_param_summaries``: one omitted
    # here is shared with the previous round and its convergence test becomes a
    # no-op, ending the fixpoint early without a warning.
    return ClassAttrTypes(
        object_types={key: set(values) for key, values in class_attr_types.object_types.items()},
        element_types={key: set(values) for key, values in class_attr_types.element_types.items()},
        callable_types={
            key: set(values) for key, values in class_attr_types.callable_types.items()
        },
        element_callable_types={
            key: set(values)
            for key, values in class_attr_types.element_callable_types.items()
        },
    )
