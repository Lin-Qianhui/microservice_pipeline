"""Data-access schema objects and confidence helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
CONFIDENCE_WEIGHT = {"low": 0.25, "medium": 0.6, "high": 1.0}


# A lineage edge makes one of two very different claims, and until Step 4a the
# vocabulary could only spell the first:
#
#   identity   -- these two names hold *the same object*. Merging them is right.
#   derivation -- this value was *made from* that one. They are two objects, and
#                 merging them is a false merge.
#
# Step 1b measured what happens when the second is recorded as the first: 31 of
# 78 scored ``local_assign`` claims on climlab are contradicted by the running
# program, against 6 of 106 for ``arg_to_param``. ``x.copy()`` and "one return
# path speaks for the whole function" were the two causes.
#
# ``derived_from`` keeps the flow visible without claiming identity. Consumers
# opt in by name rather than by default, which is why every set below is an
# allowlist: ``_apply_lineage_aliases`` (which turns lineage into ``alias_of``),
# ``identity_comparison`` (which scores identity claims), and the two clustering
# must-link sets all ignore a relation they were not told about.
RELATION_ARG_TO_PARAM = "arg_to_param"
RELATION_LOCAL_ASSIGN = "local_assign"
RELATION_STATE_ASSIGN = "state_assign"
RELATION_RETURN_VALUE = "return_value"
RELATION_RETURN_SLOT = "return_slot"
RELATION_TUPLE_UNPACK = "tuple_unpack"
RELATION_DERIVED_FROM = "derived_from"

IDENTITY_RELATIONS = frozenset(
    {
        RELATION_ARG_TO_PARAM,
        RELATION_LOCAL_ASSIGN,
        RELATION_STATE_ASSIGN,
        RELATION_RETURN_VALUE,
        RELATION_RETURN_SLOT,
        RELATION_TUPLE_UNPACK,
    }
)


@dataclass
class DataObject:
    id: str
    kind: str
    display_name: str
    scope: str
    owner: str
    container: str
    field: str
    file: str
    lineno: int
    inferred_type: str
    confidence: str
    alias_of: str = ""
    access_path: str = ""
    structural_role: str = "primary"


@dataclass
class AccessEdge:
    callable: str
    object_id: str
    access: str
    operation: str
    file: str
    lineno: int
    confidence: str
    evidence: str


@dataclass
class LineageEdge:
    src_object_id: str
    dst_object_id: str
    relation: str
    file: str
    lineno: int
    caller: str = ""
    callee: str = ""
    slot: str = ""


@dataclass
class LocalBinding:
    object_id: str
    inferred_type: str = ""
    confidence: str = "medium"
    alias_of: str = ""
    display_name: str = ""
    access_path: str = ""
    exposed: bool = True
    node: Optional[ast.AST] = None
    class_types: Set[str] = field(default_factory=set)


@dataclass
class Scope:
    callable_id: str
    params: Set[str]
    locals: Dict[str, LocalBinding] = field(default_factory=dict)
    attr_bindings: Dict[str, LocalBinding] = field(default_factory=dict)
    local_class_types: Dict[str, Set[str]] = field(default_factory=dict)
    local_element_class_types: Dict[str, Set[str]] = field(default_factory=dict)
    local_shared_state_owner_types: Dict[str, Set[str]] = field(default_factory=dict)
    local_element_shared_state_owner_types: Dict[str, Set[str]] = field(default_factory=dict)
    attr_class_types: Dict[str, Set[str]] = field(default_factory=dict)
    attr_shared_state_owner_types: Dict[str, Set[str]] = field(default_factory=dict)
    shadowed: Set[str] = field(default_factory=set)


@dataclass
class ExprRef:
    object_id: str
    inferred_type: str
    confidence: str
    display_name: str
    access_path: str = ""
    coarse_object_id: str = ""
    # ``True`` when ``object_id`` names the object this expression was *made
    # from* rather than the object it *is* -- ``x.copy()``, or a call whose
    # callee only returns its argument on some paths. The family and the field
    # structure still come from that object, which is why the ref is kept
    # rather than dropped; only the identity claim is withheld.
    derived: bool = False


@dataclass
class ValueOrigin:
    """Where an assigned value came from, and whether it *is* that thing.

    Replaces the ``(inferred_type, alias_of, confidence)`` triple the collector
    used to pass around, which had no way to say "made from" -- so every source
    it could name became an alias. ``identity`` carries that distinction:
    ``source_id`` is still recorded either way, as an alias when ``identity`` is
    true and as a ``derived_from`` lineage edge when it is false.
    """

    inferred_type: str
    source_id: str
    confidence: str
    identity: bool = True

    @property
    def alias_of(self) -> str:
        return self.source_id if self.identity else ""


def confidence_max(a: str, b: str) -> str:
    return a if CONFIDENCE_RANK.get(a, 0) >= CONFIDENCE_RANK.get(b, 0) else b


def confidence_weight(confidence: str) -> float:
    return CONFIDENCE_WEIGHT.get(confidence, CONFIDENCE_WEIGHT["low"])
