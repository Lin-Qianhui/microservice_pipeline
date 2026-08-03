"""Deriving which calls register one object into another.

Frameworks couple objects by handing one to the other and invoking it later:
``parent.add_subprocess("albedo", child)``, ``module.add_module(name, layer)``,
``pipeline.register(step)``. The coupling is real but there is no call from the
parent's code to the child's, so no amount of call resolution finds it. Naming
the method was the previous answer, and it worked on exactly one framework.

This module derives the same fact from two summaries the collectors gather:

**escape**
    Some parameter of the registering method ends up stored on ``self``. That is
    what makes the call a registration rather than a plain use -- the value is
    retained. It is an identity fact, not a type fact, which is why it needs the
    origin tracking in ``type_env`` rather than the type environment proper.

**invoke**
    Elements of that attribute later receive a method. This is the gate. Without
    it, ``def __init__(self, config): self.config = config`` looks exactly like a
    registration, and ``self.parent = parent`` looks like one pointing the wrong
    way. Requiring that the stored values are *executed* is what separates a
    registry from ordinary retained state.

Joining them yields a rule per registering callable. The rule is applied back at
the call site, in ``CallCollector._add_registration_edges``, because that is the
only place both the parent and the child type are known exactly; the attribute
they meet in has already forgotten which child belongs to which parent.

Two things make the join less direct than it sounds:

*The attribute registered into is not always the attribute invoked.* climlab
stores into ``self.subprocess`` and invokes out of ``self.process_types``, having
copied between them in ``_build_process_type_list``. ``RegistryFacts.element_flow``
records those copies and the join closes over them transitively.

*The attribute is rarely owned by one class.* ``Process.add_subprocess`` writes
``self.subprocess``, but ``TimeDependentProcess`` is what reads it. Facts are
recorded against whichever class the code was written in, so the join has to
treat a base and its subclasses as naming the same attribute.
"""


from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .ast_utils import unwrap_passthrough
from .models import (
    AttrSlot,
    ClassAttr,
    FunctionEscapeSummary,
    ParamKey,
    RegistryFacts,
)
from .project_index import ProjectIndex


@dataclass(frozen=True)
class RegistrationRule:
    """One callable whose parameter registers a child into its receiver.

    ``invoked_method`` is what elements of the target registry are called with.
    It is not necessarily the method finally linked -- see
    ``CallCollector._registration_hook`` for the template-method re-projection
    that turns a shared base method into the hook concrete classes override.

    ``child_is_element`` distinguishes ``register(child)`` from
    ``Pipeline(steps=[a, b])``: in the second the argument is the container and
    the children are inside it, so the call site has to look one level in.

    ``slot_param`` is the parameter the caller names the relationship with, when
    the registry is keyed. It carries no weight in the call graph, but the
    data-access lineage labels its edges with it.
    """
    registrar: str
    child_param: ParamKey
    invoked_method: str
    child_is_element: bool = False
    slot_param: Optional[ParamKey] = None


# -- applying a rule to a call site -------------------------------------------
#
# Free functions rather than collector methods because two stages need identical
# behaviour from them: the call graph turns a registration into an edge, and the
# data-access pass turns the same call into shared-state lineage. A second
# implementation of the argument mapping is a second thing to get wrong.


def registration_argument(
    node: ast.Call, param: ParamKey, offset: int
) -> Optional[ast.AST]:
    """Find the argument a call passes at one of the callee's parameters.

    ``offset`` accounts for the implicit ``self``: the callee's parameter 2 is
    the call site's argument 1 when the receiver is bound. The keyword form is
    checked too, since ``register(name="x", proc=child)`` states the same thing
    as ``register("x", child)``.
    """
    position, name = param
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value

    index = position - offset
    if position >= 0 and 0 <= index < len(node.args):
        argument = node.args[index]
        # ``register(*children)`` says nothing about which child landed at this
        # position, and a starred argument shifts every position after it, so
        # anything at or beyond one is untrustworthy.
        if not any(
            isinstance(earlier, ast.Starred) for earlier in node.args[: index + 1]
        ):
            return argument
    return None


def registration_child_exprs(
    node: ast.Call, rule: RegistrationRule, offset: int
) -> List[ast.AST]:
    """The child expressions a call registers, one or many.

    A rule whose parameter is filled wholesale registers each element of the
    argument, so ``Pipeline([Scaler(), Model()])`` couples two children. Only a
    literal collection can be taken apart here; anything else keeps its contents
    hidden and yields nothing rather than a guess.
    """
    argument = registration_argument(node, rule.child_param, offset)
    if argument is None:
        return []
    if not rule.child_is_element:
        return [argument]

    argument = unwrap_passthrough(argument)
    if isinstance(argument, ast.Dict):
        return list(argument.values)
    if isinstance(argument, (ast.List, ast.Set, ast.Tuple)):
        return [
            element
            for element in argument.elts
            if not isinstance(element, ast.Starred)
        ]
    return []


def registration_slot(node: ast.Call, rule: RegistrationRule, offset: int) -> str:
    """The literal label this call files its child under, when it passes one."""
    if rule.slot_param is None:
        return ""
    argument = registration_argument(node, rule.slot_param, offset)
    argument = unwrap_passthrough(argument) if argument is not None else None
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return ""


def _attr_aliases(key: ClassAttr, project_index: ProjectIndex) -> Set[ClassAttr]:
    """The same attribute as named from anywhere in its class hierarchy.

    ``(Process, "subprocess")`` and ``(TimeDependentProcess, "subprocess")`` are
    one attribute; which name a fact carries depends only on which class the
    method that touched it was written in. Both directions are needed: a base
    class writes attributes its subclasses read, and subclasses write attributes
    the base reads back through a template method.
    """
    class_id, attr = key
    related = {class_id}
    related.update(project_index.class_and_ancestors(class_id))
    for candidate in project_index.subclasses.get(class_id, ()):
        related.add(candidate)
        related.update(project_index.class_and_ancestors(candidate))
    return {(related_class, attr) for related_class in related}


def _reachable_attrs(
    start: ClassAttr, facts: RegistryFacts, project_index: ProjectIndex
) -> Set[ClassAttr]:
    """Every attribute a value stored into ``start`` can end up in.

    A hop is either an explicit copy between attributes or a restatement of the
    same attribute from elsewhere in the hierarchy. Both are followed to a fixed
    point, since a value can be relayed more than once before it is invoked.
    """
    seen: Set[ClassAttr] = set()
    queue: List[ClassAttr] = [start]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        for alias in _attr_aliases(current, project_index):
            if alias not in seen:
                queue.append(alias)
            queue.extend(facts.element_flow.get(alias, ()))
    return seen


def build_registration_rules(
    escape_summaries: Dict[str, FunctionEscapeSummary],
    facts: RegistryFacts,
    project_index: ProjectIndex,
) -> Dict[str, RegistrationRule]:
    """Join escape and invoke facts into one rule per registering callable.

    Only container escapes qualify. A registry holds many children; a scalar
    ``self.parent = parent`` holds one and is a back-reference, which points the
    opposite way from the coupling we want to record.

    A callable with more than one qualifying parameter, or whose registry is
    invoked with more than one method, is skipped rather than guessed at. Both
    shapes are rare, and a wrong pairing here fabricates coupling that clustering
    would then treat as real.
    """
    rules: Dict[str, RegistrationRule] = {}

    for registrar, summary in escape_summaries.items():
        candidates: List[RegistrationRule] = []

        owners = _owner_classes(registrar, project_index)
        for escapes, is_element in (
            (summary.escapes, False),
            (summary.element_escapes, True),
        ):
            for param, slots in escapes.items():
                methods = _invoked_methods(slots, owners, facts, project_index)
                if len(methods) == 1:
                    candidates.append(
                        RegistrationRule(
                            registrar=registrar,
                            child_param=param,
                            invoked_method=next(iter(methods)),
                            child_is_element=is_element,
                            slot_param=_slot_param(summary, param),
                        )
                    )

        if len(candidates) == 1:
            rules[registrar] = candidates[0]

    return rules


def _slot_param(summary: FunctionEscapeSummary, child_param: ParamKey) -> Optional[ParamKey]:
    """The parameter naming the child, if exactly one parameter did the naming.

    A registrar that keys by something other than a parameter -- a counter, an
    attribute of the child -- has no caller-supplied label, and one that uses
    several has no unambiguous one. Both give up rather than pick.
    """
    candidates = summary.key_params - {child_param}
    return next(iter(candidates)) if len(candidates) == 1 else None


def _invoked_methods(
    slots: Set[AttrSlot],
    owners: Set[str],
    facts: RegistryFacts,
    project_index: ProjectIndex,
) -> Set[str]:
    """Methods that elements of the attributes in ``slots`` are called with."""
    methods: Set[str] = set()
    for attr, is_container in slots:
        if not is_container:
            continue
        for owner in owners:
            for reachable in _reachable_attrs((owner, attr), facts, project_index):
                methods.update(facts.invocations.get(reachable, {}))
    return methods


def _owner_classes(callable_id: str, project_index: ProjectIndex) -> Set[str]:
    """The class a method belongs to, if it belongs to one.

    Escapes are recorded against ``self``, so the attribute's owner is the class
    enclosing the registering method. A module-level function has none and
    cannot register anything into ``self``.
    """
    owner = callable_id.rsplit(".", 1)[0]
    return {owner} if project_index.is_known_class(owner) else set()