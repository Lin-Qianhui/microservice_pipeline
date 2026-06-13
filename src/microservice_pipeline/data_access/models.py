"""Data-access schema objects and confidence helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
CONFIDENCE_WEIGHT = {"low": 0.25, "medium": 0.6, "high": 1.0}


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


def confidence_max(a: str, b: str) -> str:
    return a if CONFIDENCE_RANK.get(a, 0) >= CONFIDENCE_RANK.get(b, 0) else b


def confidence_weight(confidence: str) -> float:
    return CONFIDENCE_WEIGHT.get(confidence, CONFIDENCE_WEIGHT["low"])
