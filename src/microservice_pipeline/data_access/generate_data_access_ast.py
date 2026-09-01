#!/usr/bin/env python3
"""Generate a static data-access view for Python callables.

Outputs:
- data_objects.csv: service-relevant data objects discovered in scanned files
- access_edges.csv: callable -> data object access edges
- callable_data_access.csv: denormalized callable/object/access rows
- data_access.json: combined callables, objects, and access edges payload
- data_access_report.md: readable Markdown summary

The extractor is static and conservative. It is designed as a companion view to
the call graph, using the same callable IDs where possible.
"""

from __future__ import annotations

import argparse
import ast
import sys
import warnings
from dataclasses import astuple, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from microservice_pipeline.data_access.attrdict import (
        collect_attrdict_classes,
        collect_attrdict_classes_from_analysis_files,
    )
    from microservice_pipeline.config import (
        DataAccessExtractionConfig,
        ExtractionConfig,
        PyrightConfig,
        load_extraction_config,
    )
    from microservice_pipeline.import_resolution import (
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )
    from microservice_pipeline.jsonc_config import load_jsonc
    from microservice_pipeline.data_access.models import (
        CONFIDENCE_RANK,
        IDENTITY_RELATIONS,
        RELATION_ARG_TO_PARAM,
        RELATION_DERIVED_FROM,
        RELATION_LOCAL_ASSIGN,
        RELATION_RETURN_SLOT,
        RELATION_RETURN_VALUE,
        RELATION_STATE_ASSIGN,
        RELATION_TUPLE_UNPACK,
        AccessEdge,
        DataObject,
        ExprRef,
        LineageEdge,
        LocalBinding,
        Scope,
        ValueOrigin,
        confidence_max,
        confidence_weight,
    )
    from microservice_pipeline.data_access.outputs import _edges_payload, _objects_payload, write_outputs
    from microservice_pipeline.data_access.rules import (
        CONTAINER_FIELD_KIND,
        CONTAINER_TYPES,
        COORDINATOR_ATTR_THRESHOLD,
        COORDINATOR_CONTAINER_THRESHOLD,
        COORDINATOR_METHOD_THRESHOLD,
        FILE_READ_FUNCS,
        FILE_WRITE_METHODS,
        MAX_RETURN_SUMMARY_PASSES,
        MUTATING_METHODS,
        PANDAS_INDEXER_ATTRS,
        PANDAS_INPLACE_METHODS,
        POOCH_READ_FUNCS,
        XARRAY_INDEXER_ATTRS,
        XARRAY_LABEL_METHODS,
        XARRAY_OPEN_FUNCS,
        _attr_expr_family_key,
        _attribute_path,
        _call_name_matches,
        _class_attr_family_key,
        _field_inferred_type,
        _indexer_fields,
        _is_containerish_family,
        _keyword_truthy,
        _literal_container_family,
        _name_from_expr,
        _slice_selects_multiple_fields,
        _slice_value,
        _top_level_attr_name,
        _xarray_call_fields,
        _xarray_indexer_fields,
    )
    from microservice_pipeline.data_access.registration_lineage import (
        RegisteredStateLineageMixin,
        _class_attr_state_display,
        _class_state_display,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from microservice_pipeline.data_access.attrdict import (  # type: ignore
        collect_attrdict_classes,
        collect_attrdict_classes_from_analysis_files,
    )
    from microservice_pipeline.config import (  # type: ignore
        DataAccessExtractionConfig,
        ExtractionConfig,
        PyrightConfig,
        load_extraction_config,
    )
    from import_resolution import (  # type: ignore
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )
    from jsonc_config import load_jsonc  # type: ignore
    from microservice_pipeline.data_access.models import (  # type: ignore
        CONFIDENCE_RANK,
        IDENTITY_RELATIONS,
        RELATION_ARG_TO_PARAM,
        RELATION_DERIVED_FROM,
        RELATION_LOCAL_ASSIGN,
        RELATION_RETURN_SLOT,
        RELATION_RETURN_VALUE,
        RELATION_STATE_ASSIGN,
        RELATION_TUPLE_UNPACK,
        AccessEdge,
        DataObject,
        ExprRef,
        LineageEdge,
        LocalBinding,
        Scope,
        ValueOrigin,
        confidence_max,
        confidence_weight,
    )
    from microservice_pipeline.data_access.outputs import _edges_payload, _objects_payload, write_outputs  # type: ignore
    from microservice_pipeline.data_access.rules import (  # type: ignore
        CONTAINER_FIELD_KIND,
        CONTAINER_TYPES,
        COORDINATOR_ATTR_THRESHOLD,
        COORDINATOR_CONTAINER_THRESHOLD,
        COORDINATOR_METHOD_THRESHOLD,
        FILE_READ_FUNCS,
        FILE_WRITE_METHODS,
        MAX_RETURN_SUMMARY_PASSES,
        MUTATING_METHODS,
        PANDAS_INDEXER_ATTRS,
        PANDAS_INPLACE_METHODS,
        POOCH_READ_FUNCS,
        XARRAY_INDEXER_ATTRS,
        XARRAY_LABEL_METHODS,
        XARRAY_OPEN_FUNCS,
        _attr_expr_family_key,
        _attribute_path,
        _call_name_matches,
        _class_attr_family_key,
        _field_inferred_type,
        _indexer_fields,
        _is_containerish_family,
        _keyword_truthy,
        _literal_container_family,
        _name_from_expr,
        _slice_selects_multiple_fields,
        _slice_value,
        _top_level_attr_name,
        _xarray_call_fields,
        _xarray_indexer_fields,
    )
    from microservice_pipeline.data_access.registration_lineage import (
        RegisteredStateLineageMixin,
        _class_attr_state_display,
        _class_state_display,
    )  # type: ignore

try:
    from microservice_pipeline.call_graph.generate_call_graph_ast import (
        CallGraphAnalysis,
        ParsedFileCache,
        ProjectIndex,
        RegistrationRule,
        analyze_analysis_files,
        AnalysisFile,
        CallableDef,
        attach_parents,
        build_indices_from_analysis_files,
        build_indices,
        iter_analysis_files_for_source_roots,
        iter_analysis_files,
        iter_python_files,
        module_callable_id,
        parse_python_file,
        parse_python_source,
        path_to_module,
    )
    from microservice_pipeline.call_graph.ast_utils import unwrap_passthrough
    from microservice_pipeline.data_access.pyright_type_probe import (
        FAMILY_DATAFRAME,
        FAMILY_DICT,
        FAMILY_FIELD,
        FAMILY_FILE,
        FAMILY_LIST,
        FAMILY_OBJECT,
        FAMILY_PATH,
        FAMILY_SET,
        FAMILY_UNKNOWN,
        FAMILY_XARRAY,
        PyrightProbeReport,
        PyrightProbeTarget,
        discover_project_root,
        probe_pyright_targets,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from microservice_pipeline.call_graph.generate_call_graph_ast import (  # type: ignore
        CallGraphAnalysis,
        ParsedFileCache,
        ProjectIndex,
        RegistrationRule,
        analyze_analysis_files,
        AnalysisFile,
        CallableDef,
        attach_parents,
        build_indices_from_analysis_files,
        build_indices,
        iter_analysis_files_for_source_roots,
        iter_analysis_files,
        iter_python_files,
        module_callable_id,
        parse_python_file,
        parse_python_source,
        path_to_module,
    )
    from microservice_pipeline.call_graph.ast_utils import unwrap_passthrough  # type: ignore
    from microservice_pipeline.data_access.pyright_type_probe import (  # type: ignore
        FAMILY_DATAFRAME,
        FAMILY_DICT,
        FAMILY_FIELD,
        FAMILY_FILE,
        FAMILY_LIST,
        FAMILY_OBJECT,
        FAMILY_PATH,
        FAMILY_SET,
        FAMILY_UNKNOWN,
        FAMILY_XARRAY,
        PyrightProbeReport,
        PyrightProbeTarget,
        discover_project_root,
        probe_pyright_targets,
    )

# Statements after which nothing in the same block runs, so a Pyright probe
# placed below one sits in unreachable code. Pyright does not report a type for
# it at all, and says nothing about why.
TERMINAL_STATEMENTS = (ast.Return, ast.Raise, ast.Break, ast.Continue)

SHARED_FIELD_CONTAINERS = {
    "df_col": {},
    "dict_key": {},
}


def _is_state_object_id(object_id: str) -> bool:
    return (
        object_id.startswith("class_state:")
        or object_id.startswith("class_attr_state:")
        or object_id.startswith("object_state:")
    )


def _is_class_state_object_id(object_id: str) -> bool:
    return object_id.startswith("class_state:") or object_id.startswith("class_attr_state:")


def _confidence_max(a: str, b: str) -> str:
    return confidence_max(a, b)


def _confidence_weight(confidence: str) -> float:
    return confidence_weight(confidence)


def _is_unknown_family(family: str) -> bool:
    return family in {"", FAMILY_UNKNOWN}


def _prefer_inferred_type(current: str, incoming: str) -> str:
    """The more specific of two families, tie broken on the value itself.

    Keeping ``current`` when both are known made the answer "whichever was
    registered first", which across files is "whichever file was processed
    first". Two different known families for one object is a disagreement worth
    resolving the same way every run rather than by arrival.
    """
    if _is_unknown_family(current):
        return incoming or current
    if _is_unknown_family(incoming):
        return current
    return min(current, incoming)


def _object_identity_rank(obj: DataObject) -> Tuple[int, str, str, str, str, str, str]:
    """Order over the fields that say *what an object is*.

    ``_record_access`` mints a placeholder with ``kind == "unknown"`` whenever an
    edge names an ID no pass has materialised yet, so a real registration must
    always beat one. Beyond that the tie is broken on content, so two files that
    disagree resolve identically whichever order they arrive in.
    """
    return (
        0 if obj.kind == "unknown" else 1,
        obj.kind,
        obj.display_name,
        obj.scope,
        obj.owner,
        obj.container,
        obj.field,
    )


def _prefer_earliest_site(existing: DataObject, incoming: DataObject) -> Optional[Tuple[str, int]]:
    """The earliest ``(file, lineno)`` either record carries, or ``None``.

    Previously "the first non-zero one wins", which across files is again a
    function of processing order. An object reachable from several modules now
    reports the same site every run.
    """
    sites = [
        (obj.file, obj.lineno)
        for obj in (existing, incoming)
        if obj.lineno
    ]
    return min(sites) if sites else None


def _prefer_populated(current: str, incoming: str) -> str:
    """A non-empty value beats an empty one; two non-empty ones tie on content."""
    if not current:
        return incoming
    if not incoming:
        return current
    return min(current, incoming)


def merge_data_object(existing: DataObject, incoming: DataObject) -> bool:
    """Fold ``incoming`` into ``existing``. Returns True on a real-kind conflict.

    One rule, used both inside a file and across files. The cross-file loop used
    to keep only ``confidence`` and discard every later refinement -- a known
    ``inferred_type`` arriving to replace ``unknown``, a real kind arriving to
    replace a placeholder, an ``access_path`` arriving where there was none --
    so an object reachable from two modules kept whatever the first file said.
    First-file-wins is only deterministic if file order is.

    Every rule below is commutative: ``merge(a, b)`` and ``merge(b, a)`` leave
    the same values. That is what makes the artifacts a function of the input
    rather than of the order the input was read in.
    """
    conflict = (
        existing.kind != "unknown"
        and incoming.kind != "unknown"
        and existing.kind != incoming.kind
    )

    # The identity fields move as a block. Splitting them field by field could
    # pair one record's ``kind`` with another's ``owner``, describing an object
    # neither pass ever saw.
    if _object_identity_rank(incoming) > _object_identity_rank(existing):
        existing.kind = incoming.kind
        existing.display_name = incoming.display_name
        existing.scope = incoming.scope
        existing.owner = incoming.owner
        existing.container = incoming.container
        existing.field = incoming.field

    existing.confidence = _confidence_max(existing.confidence, incoming.confidence)
    existing.inferred_type = _prefer_inferred_type(existing.inferred_type, incoming.inferred_type)

    site = _prefer_earliest_site(existing, incoming)
    if site is not None:
        existing.file, existing.lineno = site

    existing.alias_of = _prefer_populated(existing.alias_of, incoming.alias_of)
    existing.access_path = _prefer_populated(existing.access_path, incoming.access_path)
    if existing.structural_role == "primary":
        existing.structural_role = incoming.structural_role
    elif incoming.structural_role != "primary":
        existing.structural_role = min(existing.structural_role, incoming.structural_role)
    return conflict


def _specific_family(ref: ExprRef) -> bool:
    return not _is_unknown_family(ref.inferred_type)


def _return_summary_score(ref: ExprRef) -> Tuple[int, int, int, int]:
    return (
        1 if _specific_family(ref) else 0,
        CONFIDENCE_RANK.get(ref.confidence, 0),
        1 if ref.object_id else 0,
        1 if ref.display_name else 0,
    )


def _return_summary_rank(ref: ExprRef) -> Tuple[Tuple[int, int, int, int], Tuple[object, ...]]:
    """Total order over candidate return summaries.

    ``_return_summary_score`` is four coarse flags, so two genuinely different
    candidates tie constantly. A tie used to be resolved by keeping whichever
    arrived first, which is whichever file the loop reached first -- so the
    ``object_id``, ``display_name`` and ``access_path`` that the whole fixpoint
    then propagates were a function of file order. The content itself breaks the
    tie instead; which candidate wins is arbitrary, but it is the same one every
    run.
    """
    return _return_summary_score(ref), astuple(ref)


def _merge_return_summary(existing: Optional[ExprRef], candidate: ExprRef) -> ExprRef:
    if existing is None:
        return candidate
    if _return_summary_rank(candidate) > _return_summary_rank(existing):
        return candidate
    return existing


def _expr_ref_snapshot(ref: Optional[ExprRef]) -> Tuple[object, ...]:
    """Every field of an ``ExprRef``, for comparing one fixpoint pass to the next.

    ``astuple`` rather than a hand-written tuple: the two snapshot helpers used
    to list fields by hand and listed *different* ones, so a pass that changed
    only a tuple slot's ``access_path`` produced an identical snapshot and the
    loop declared convergence and stopped. ``access_path`` is what
    ``_field_object_id`` builds field IDs from, so that is not a cosmetic loss.
    Deriving the snapshot from the dataclass also means a field added to
    ``ExprRef`` later is compared without anyone remembering to add it here.
    """
    return () if ref is None else astuple(ref)


def _return_summary_snapshot(
    return_summaries: Dict[str, ExprRef],
) -> Dict[str, Tuple[object, ...]]:
    return {
        callable_id: _expr_ref_snapshot(ref)
        for callable_id, ref in sorted(return_summaries.items())
    }


def _merge_return_tuple_summary(
    existing: Optional[Tuple[Optional[ExprRef], ...]],
    candidate: Tuple[Optional[ExprRef], ...],
) -> Tuple[Optional[ExprRef], ...]:
    if existing is None:
        return candidate
    size = max(len(existing), len(candidate))
    merged: List[Optional[ExprRef]] = []
    for idx in range(size):
        current = existing[idx] if idx < len(existing) else None
        incoming = candidate[idx] if idx < len(candidate) else None
        if current is None:
            merged.append(incoming)
        elif incoming is None:
            merged.append(current)
        else:
            merged.append(_merge_return_summary(current, incoming))
    return tuple(merged)


def _return_tuple_summary_snapshot(
    return_tuple_summaries: Dict[str, Tuple[Optional[ExprRef], ...]],
) -> Dict[str, Tuple[Tuple[object, ...], ...]]:
    return {
        callable_id: tuple(_expr_ref_snapshot(ref) for ref in refs)
        for callable_id, refs in sorted(return_tuple_summaries.items())
    }


# A callable's own statements stop at any nested scope: a ``return`` inside a
# nested function belongs to that function, not to this one.
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _own_statements(body: Sequence[ast.stmt]) -> List[ast.AST]:
    found: List[ast.AST] = []
    stack: List[ast.AST] = list(reversed(list(body)))
    while stack:
        node = stack.pop()
        if isinstance(node, _NESTED_SCOPES):
            continue
        found.append(node)
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return found


def _always_returns(body: Sequence[ast.stmt]) -> bool:
    """Whether control can never fall off the end of ``body``.

    A function that falls off the end returns ``None``, which is a *different*
    value from the one its single ``return`` names -- so ``if c: return x`` with
    nothing after it does not return ``x`` on every path, and enumerating its
    ``Return`` nodes alone cannot tell you that.

    Deliberately conservative: any shape not listed answers ``False``, which
    withholds an alias rather than inventing one. ``while True:`` loops that only
    exit through a ``return`` are the notable omission.
    """
    for stmt in body:
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return True
        if isinstance(stmt, ast.If):
            if stmt.orelse and _always_returns(stmt.body) and _always_returns(stmt.orelse):
                return True
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            if _always_returns(stmt.body):
                return True
        elif isinstance(stmt, ast.Try):
            if stmt.finalbody and _always_returns(stmt.finalbody):
                return True
            # No handlers is ``try/finally``, where ``all`` over nothing is the
            # right answer: only the body has to settle it.
            if _always_returns(list(stmt.body) + list(stmt.orelse)) and all(
                _always_returns(handler.body) for handler in stmt.handlers
            ):
                return True
        elif isinstance(stmt, ast.Match):
            has_wildcard = any(
                isinstance(case.pattern, ast.MatchAs)
                and case.pattern.pattern is None
                and case.guard is None
                for case in stmt.cases
            )
            if has_wildcard and all(_always_returns(case.body) for case in stmt.cases):
                return True
    return False


def _returns_one_source(node: ast.AST) -> bool:
    """Whether every path out of this callable hands back the same expression.

    This is the gate on treating a call's result as *being* one of the objects
    inside the callee. Step 1b measured what happens without it:
    ``_climlab_to_cam3`` starts ``if np.isscalar(field): return field`` and
    builds a new array on every other path, and that one branch made all
    eighteen of its call sites claim they held ``field`` itself. Roughly 41 of
    the 51 contradictions on climlab were this one shape.

    Answered syntactically, on the text of the returned expressions, for three
    reasons: it is settled on the first pass, so it needs no place in the
    fixpoint's convergence test (the section 1.5 trap); it cannot oscillate; and
    when it is wrong it is wrong in the safe direction -- ``return d`` and
    ``return self.data`` naming one object read as two, which withholds a true
    alias rather than asserting a false one.
    """
    body = getattr(node, "body", None)
    if not isinstance(body, list):
        return False
    own = _own_statements(body)
    if any(isinstance(stmt, (ast.Yield, ast.YieldFrom)) for stmt in own):
        # A generator call returns a generator, never the value in its
        # ``return``, so no expression inside it describes what the caller gets.
        return False
    returns = [stmt for stmt in own if isinstance(stmt, ast.Return)]
    if not returns:
        return False
    texts: Set[str] = set()
    for stmt in returns:
        if stmt.value is None:
            return False  # a bare ``return`` yields None -- a different value
        texts.add(_unparse(unwrap_passthrough(stmt.value)))
        if len(texts) > 1:
            return False
    return _always_returns(body)


def _is_copy_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy"
    )


def _as_derived(ref: Optional[ExprRef]) -> Optional[ExprRef]:
    """The same reference, marked as "made from" rather than "is".

    ``replace`` rather than mutation: the resolvers hand back refs that other
    call sites also hold, and flipping the flag in place would spread the
    derivation to expressions that really are the object.
    """
    if ref is None or ref.derived:
        return ref
    return replace(ref, derived=True)


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive for unusual nodes
        return node.__class__.__name__


def _stringify_literal(value: object) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _safe_id_part(value: str) -> str:
    return value.replace("\n", " ").strip()


def _safe_path_id_part(value: str) -> str:
    cleaned = _safe_id_part(value).replace("[]", ".__item__")
    cleaned = cleaned.replace("[", ".").replace("]", "")
    cleaned = cleaned.replace("'", "").replace('"', "")
    return cleaned.replace(".__item__", "[]")


def _path_with_attr(base_path: str, attr: str) -> str:
    return f"{base_path}.{attr}" if base_path else attr


def _path_with_field(base_path: str, field: str) -> str:
    return f"{base_path}[{field!r}]" if base_path else f"[{field!r}]"


def _root_relative_access_path(root_object_id: str, access_path: str) -> str:
    root_name = ""
    if root_object_id.startswith("param:"):
        _callable_id, _, root_name = root_object_id.removeprefix("param:").rpartition(":")
    elif root_object_id.startswith("local_exposed:"):
        _callable_id, _, root_name = root_object_id.removeprefix("local_exposed:").rpartition(":")
    if root_name and access_path.startswith(root_name + "."):
        return access_path[len(root_name) + 1 :]
    if root_name and access_path.startswith(root_name + "["):
        return access_path[len(root_name) :]
    return access_path


def _shared_field_container(display_name: str, object_kind: str) -> Optional[str]:
    leaf_name = display_name.rsplit(".", 1)[-1].strip()
    return SHARED_FIELD_CONTAINERS.get(object_kind, {}).get(leaf_name.lower())


def configure_shared_field_containers(config: Optional[Dict[str, object]] = None) -> None:
    global SHARED_FIELD_CONTAINERS
    merged: Dict[str, Dict[str, str]] = {
        "df_col": {},
        "dict_key": {},
    }
    if config:
        for kind, raw_mapping in config.items():
            if kind not in {"df_col", "dict_key"}:
                continue
            merged.setdefault(kind, {})
            if isinstance(raw_mapping, dict):
                for source, canonical in raw_mapping.items():
                    merged[kind][str(source).lower()] = str(canonical)
            elif isinstance(raw_mapping, list):
                for source in raw_mapping:
                    merged[kind][str(source).lower()] = str(source)
    SHARED_FIELD_CONTAINERS = merged


def load_shared_field_containers(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None:
        return None
    return load_jsonc(path)


def _alias_root(
    object_id: str,
    objects: Dict[str, DataObject],
    _seen: Optional[Set[str]] = None,
) -> str:
    seen = _seen if _seen is not None else set()
    if object_id in seen:
        return object_id
    seen.add(object_id)
    obj = objects.get(object_id)
    if obj is None or not obj.alias_of:
        return object_id
    return _alias_root(obj.alias_of, objects, seen)


def _apply_confirmed_param_aliases(
    objects: Dict[str, DataObject],
    param_bindings: Dict[str, Set[str]],
) -> None:
    for param_id, source_ids in param_bindings.items():
        param_obj = objects.get(param_id)
        if param_obj is None:
            continue
        roots = {_alias_root(source_id, objects) for source_id in source_ids if source_id}
        roots.discard(param_id)
        if len(roots) == 1:
            param_obj.alias_of = next(iter(roots))


def _apply_confirmed_param_access_paths(
    objects: Dict[str, DataObject],
    param_access_paths: Dict[str, Set[str]],
) -> None:
    for param_id, access_paths in param_access_paths.items():
        param_obj = objects.get(param_id)
        if param_obj is None:
            continue
        paths = {path for path in access_paths if path}
        if len(paths) != 1:
            continue
        access_path = next(iter(paths))
        root_name = param_obj.field or param_obj.display_name
        if not param_obj.access_path or param_obj.access_path == root_name:
            param_obj.access_path = access_path


def _return_slot_id(callable_id: str, slot: Optional[int] = None) -> str:
    if slot is None:
        return f"return:{callable_id}"
    return f"return:{callable_id}:{slot}"


def _lineage_roots_with_cycle_flag(
    object_id: str,
    incoming: Dict[str, Set[str]],
    objects: Dict[str, DataObject],
    cache: Dict[str, Set[str]],
    seen: Set[str],
) -> Tuple[Set[str], bool]:
    """Reaching roots for ``object_id``, and whether a live cycle truncated them.

    The cycle guard returns an empty set for a node already on the stack, which
    is correct for *that* traversal and only that traversal. Caching a value
    computed above such a node published the truncated answer to every later
    query, so the result depended on which node happened to be asked first --
    and the asking order was ``objects.items()``, i.e. the order files were
    processed. ``_apply_lineage_aliases`` turns a one-root answer into an alias
    and erases the alias on a many-root answer, so this silently flipped
    ``alias_of`` between runs on any project with a lineage cycle.

    The flag rides back up so an ancestor knows its own answer is provisional
    too. On an acyclic graph it is never set and every node still caches, so
    there is no cost to the common case.
    """
    if object_id in cache:
        return cache[object_id], False
    if object_id in seen:
        return set(), True
    seen.add(object_id)

    roots: Set[str] = set()
    truncated = False
    for parent_id in sorted(incoming.get(object_id, set())):
        if parent_id == object_id:
            continue
        parent_roots, parent_truncated = _lineage_roots_with_cycle_flag(
            parent_id, incoming, objects, cache, seen
        )
        truncated = truncated or parent_truncated
        if parent_roots:
            roots.update(parent_roots)
        elif parent_id in objects:
            roots.add(parent_id)

    seen.remove(object_id)
    if not truncated:
        cache[object_id] = roots
    return roots, truncated


def _lineage_roots(
    object_id: str,
    incoming: Dict[str, Set[str]],
    objects: Dict[str, DataObject],
    cache: Dict[str, Set[str]],
    seen: Optional[Set[str]] = None,
) -> Set[str]:
    roots, _truncated = _lineage_roots_with_cycle_flag(
        object_id, incoming, objects, cache, seen if seen is not None else set()
    )
    return roots


def _incoming_lineage(
    objects: Dict[str, DataObject],
    lineage_edges: Sequence[LineageEdge],
    *,
    identity_only: bool,
) -> Dict[str, Set[str]]:
    incoming: Dict[str, Set[str]] = {}
    for edge in lineage_edges:
        if not edge.src_object_id or not edge.dst_object_id:
            continue
        if identity_only and edge.relation not in IDENTITY_RELATIONS:
            continue
        incoming.setdefault(edge.dst_object_id, set()).add(edge.src_object_id)
    for object_id, obj in objects.items():
        if obj.alias_of:
            incoming.setdefault(object_id, set()).add(obj.alias_of)
    return incoming


def _apply_lineage_aliases(
    objects: Dict[str, DataObject],
    lineage_edges: Sequence[LineageEdge],
) -> None:
    """Turn the lineage graph into ``alias_of``, using both readings of it.

    ``alias_of`` means "is the same object as", so only ``IDENTITY_RELATIONS``
    may create one -- a ``derived_from`` edge says the value was *made from* its
    source, and merging the two is the false merge Step 1b measured.

    But the identity-only graph is not enough on its own. Dropping the derived
    edges also drops *ambiguity* the solver currently detects: a local reachable
    from two different values has no alias today, and would acquire one if the
    edge that made it ambiguous were filtered out. That would trade one kind of
    false merge for another. So both graphs are consulted and they have to
    agree: an alias is recorded only when the identity graph and the full graph
    reach the same single root. The rule can therefore only ever *withhold* an
    alias relative to the previous behaviour, never invent one -- which is what
    makes the direction of every measurement in Step 4a unambiguous.
    """
    incoming_all = _incoming_lineage(objects, lineage_edges, identity_only=False)
    incoming_identity = _incoming_lineage(objects, lineage_edges, identity_only=True)

    cache_all: Dict[str, Set[str]] = {}
    cache_identity: Dict[str, Set[str]] = {}
    for object_id, obj in sorted(objects.items()):
        if obj.kind not in {"local_exposed", "param", "object_state", "class_attr_state"}:
            continue
        roots = {
            root_id
            for root_id in _lineage_roots(object_id, incoming_all, objects, cache_all)
            if root_id and root_id != object_id and root_id in objects
        }
        identity_roots = {
            root_id
            for root_id in _lineage_roots(object_id, incoming_identity, objects, cache_identity)
            if root_id and root_id != object_id and root_id in objects
        }
        if len(roots) == 1 and roots == identity_roots:
            obj.alias_of = next(iter(roots))
        elif obj.alias_of and (len(roots) > 1 or roots != identity_roots):
            # Either the full lineage graph has several possible roots -- a
            # direct return summary may have picked one "best" return object
            # while reaching definitions are ambiguous -- or the only root is
            # reached through a ``derived_from`` edge, which is not an identity.
            obj.alias_of = ""


def infer_split_class_owners(
    tree: ast.Module,
    module: str,
    pyright_families: Optional[Dict[str, str]] = None,
) -> Set[str]:
    """Classes whose state is modelled as one object per attribute, not one per class.

    ``ast.walk`` rather than ``tree.body``: ``visit_ClassDef`` sets
    ``current_class`` for a class defined anywhere -- inside a function, inside
    another class -- and ``_current_owner`` then names it ``{module}.{name}``
    with no qualification. Scanning only top-level classes left every nested
    class permanently unsplittable while the collector was perfectly willing to
    ask about it, so the two disagreed.

    Only direct ``FunctionDef`` children of each class body are inspected, so a
    nested class's methods are attributed to the nested class and not also to
    the class enclosing it.
    """
    pyright_families = pyright_families if pyright_families is not None else {}
    split_owners: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        attrs: Set[str] = set()
        touched_methods: Set[str] = set()
        containerish_attrs: Set[str] = set()
        for stmt in node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            method_attrs: Set[str] = set()
            for child in ast.walk(stmt):
                literal_family = ""
                literal_targets: Sequence[ast.AST] = ()
                if isinstance(child, ast.Assign):
                    literal_family = _literal_container_family(child.value)
                    literal_targets = child.targets
                elif isinstance(child, ast.AnnAssign) and child.value is not None:
                    literal_family = _literal_container_family(child.value)
                    literal_targets = (child.target,)
                if _is_containerish_family(literal_family):
                    for target in literal_targets:
                        target_path = _attribute_path(target)
                        if not target_path or not target_path.startswith(("self.", "cls.")):
                            continue
                        target_attr = _top_level_attr_name(target_path.split(".", 1)[1])
                        if target_attr:
                            containerish_attrs.add(target_attr)

                if not isinstance(child, ast.Attribute):
                    continue
                path = _attribute_path(child)
                if not path or not path.startswith(("self.", "cls.")):
                    continue
                attr_name = _top_level_attr_name(path.split(".", 1)[1])
                if not attr_name:
                    continue
                method_attrs.add(attr_name)
                attrs.add(attr_name)
                if _is_containerish_family(
                    pyright_families.get(_class_attr_family_key(f"{module}.{node.name}", attr_name), "")
                ):
                    containerish_attrs.add(attr_name)
            if method_attrs:
                touched_methods.add(stmt.name)
        if (
            len(attrs) >= COORDINATOR_ATTR_THRESHOLD
            and len(touched_methods) >= COORDINATOR_METHOD_THRESHOLD
            and len(containerish_attrs) >= COORDINATOR_CONTAINER_THRESHOLD
        ):
            split_owners.add(f"{module}.{node.name}")
    return split_owners


def infer_split_class_owners_for_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    pyright_families: Optional[Dict[str, str]] = None,
    cache: Optional[ParsedFileCache] = None,
) -> Set[str]:
    """Split-class owners for the whole project, computed once before any pass.

    Whether a class's state is one object or one object per attribute decides
    that class's *object IDs*, and it has to be one answer for the run.
    Computing it per file made it a different answer depending on which file was
    being walked: ``registration_lineage._class_state_object_id_for_owner`` asks
    about an owner resolved from a class reference, which is very often defined
    in another module, so the same class was ``class_attr_state:X:state`` while
    its own file was walked and ``class_state:X`` while any other file was. The
    lineage edges recorded from the second case pointed at an ID the object
    table never registered.

    A union is the right join: a class qualifies on the evidence in its own
    definition, and only the file that defines it can supply that evidence.
    """
    resolved_cache = cache if cache is not None else ParsedFileCache()
    split_owners: Set[str] = set()
    for analysis_file in analysis_files:
        split_owners.update(
            infer_split_class_owners(
                resolved_cache.get(analysis_file.path),
                analysis_file.module,
                pyright_families,
            )
        )
    return split_owners


def _target_name_nodes(target: ast.AST) -> List[ast.Name]:
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[ast.Name] = []
        for element in target.elts:
            names.extend(_target_name_nodes(element))
        return names
    return []


def collect_pyright_probe_targets(tree: ast.Module, module: str, file: Path) -> List[PyrightProbeTarget]:
    attach_parents(tree)
    targets: Dict[str, PyrightProbeTarget] = {}

    def record_target(target: PyrightProbeTarget) -> None:
        current = targets.get(target.target_id)
        if current is None or (target.lineno and (current.lineno == 0 or target.lineno < current.lineno)):
            targets[target.target_id] = target

    def probe_insert_position(node: ast.AST) -> Tuple[str, int, int]:
        """Where to put the probe for ``node``: ``(mode, lineno, column)``.

        Normally after the statement the expression sits in. But a probe placed
        after a ``return``, ``raise``, ``break`` or ``continue`` is in code that
        never runs, and pyright simply does not answer for it -- so the family
        was silently lost. Those go in front.
        """
        current: Optional[ast.AST] = node
        while current is not None and not isinstance(current, ast.stmt):
            current = getattr(current, "parent", None)
        if current is None:
            return "after_line", getattr(node, "end_lineno", getattr(node, "lineno", 0)), 0
        if isinstance(current, TERMINAL_STATEMENTS):
            return (
                "before_line",
                getattr(current, "lineno", getattr(node, "lineno", 0)),
                getattr(current, "col_offset", 0),
            )
        return (
            "after_line",
            getattr(current, "end_lineno", getattr(current, "lineno", getattr(node, "lineno", 0))),
            0,
        )

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.current_class: Optional[str] = None
            self.current_callable: Optional[str] = None

        def _callable_id(self, name: str) -> str:
            if self.current_callable is not None:
                return f"{self.current_callable}.<locals>.{name}"
            if self.current_class:
                return f"{module}.{self.current_class}.{name}"
            return f"{module}.{name}"

        def _record_name_target(self, node: ast.Name) -> None:
            if node.id in {"self", "cls"}:
                return
            mode, insert_lineno, insert_col = probe_insert_position(node)
            if self.current_callable:
                target_id = f"local_exposed:{self.current_callable}:{node.id}"
                record_target(
                    PyrightProbeTarget(
                        target_id=target_id,
                        expression=node.id,
                        file=file,
                        module=module,
                        mode=mode,
                        callable_id=self.current_callable,
                        lineno=getattr(node, "lineno", 0),
                        insert_lineno=insert_lineno,
                        insert_col=insert_col,
                    )
                )
            else:
                target_id = f"module_global:{module}.{node.id}"
                record_target(
                    PyrightProbeTarget(
                        target_id=target_id,
                        expression=node.id,
                        file=file,
                        module=module,
                        mode=mode,
                        lineno=getattr(node, "lineno", 0),
                        insert_lineno=insert_lineno,
                        insert_col=insert_col,
                    )
                )

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            prev_class = self.current_class
            self.current_class = node.name
            for stmt in node.body:
                self.visit(stmt)
            self.current_class = prev_class

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_callable(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_callable(node, node.name)

        def _visit_callable(self, node: ast.AST, name: str) -> None:
            prev_callable = self.current_callable
            callable_id = self._callable_id(name)
            self.current_callable = callable_id
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
                if node.args.vararg:
                    all_args.append(node.args.vararg)
                if node.args.kwarg:
                    all_args.append(node.args.kwarg)
                for arg in all_args:
                    if arg.arg in {"self", "cls"}:
                        continue
                    record_target(
                        PyrightProbeTarget(
                            target_id=f"param:{callable_id}:{arg.arg}",
                            expression=arg.arg,
                            file=file,
                            module=module,
                            mode="callable_entry",
                            callable_id=callable_id,
                            lineno=getattr(node, "lineno", 0),
                        )
                    )
            for stmt in getattr(node, "body", []):
                self.visit(stmt)
            self.current_callable = prev_callable

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                for name_node in _target_name_nodes(target):
                    self._record_name_target(name_node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            for name_node in _target_name_nodes(node.target):
                self._record_name_target(name_node)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            for name_node in _target_name_nodes(node.target):
                self._record_name_target(name_node)
            self.generic_visit(node)

        def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
            for item in node.items:
                if item.optional_vars is not None:
                    for name_node in _target_name_nodes(item.optional_vars):
                        self._record_name_target(name_node)
            self.generic_visit(node)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            self.visit_With(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            path = _attribute_path(node)
            if path and self.current_callable:
                mode, insert_lineno, insert_col = probe_insert_position(node)
                if path.startswith(("self.", "cls.")) and self.current_class:
                    owner = f"{module}.{self.current_class}"
                    top_attr = _top_level_attr_name(path.split(".", 1)[1])
                    qualifier = "self" if path.startswith("self.") else "cls"
                    record_target(
                        PyrightProbeTarget(
                            target_id=_class_attr_family_key(owner, top_attr),
                            expression=f"{qualifier}.{top_attr}",
                            file=file,
                            module=module,
                            mode=mode,
                            callable_id=self.current_callable,
                            lineno=getattr(node, "lineno", 0),
                            insert_lineno=insert_lineno,
                            insert_col=insert_col,
                        )
                    )
                elif not path.startswith(("self.", "cls.")):
                    record_target(
                        PyrightProbeTarget(
                            target_id=_attr_expr_family_key(self.current_callable, path),
                            expression=path,
                            file=file,
                            module=module,
                            mode=mode,
                            callable_id=self.current_callable,
                            lineno=getattr(node, "lineno", 0),
                            insert_lineno=insert_lineno,
                            insert_col=insert_col,
                        )
                    )
            self.generic_visit(node)

    Visitor().visit(tree)
    return list(sorted(targets.values(), key=lambda target: (str(target.file), target.lineno, target.target_id)))


def _collect_project_pyright_families(
    src_root: Path,
    module_prefix: Optional[str],
    pyright_bin: str,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
) -> Dict[str, str]:
    resolved_project_root = project_root.resolve() if project_root is not None else discover_project_root(src_root)
    analysis_files = list(
        iter_analysis_files(
            src_root,
            module_prefix=module_prefix,
            entrypoints=entrypoints,
            project_root=resolved_project_root,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        )
    )
    return collect_pyright_families_from_analysis_files(resolved_project_root, analysis_files, pyright_bin)


def _import_root_for_analysis_file(analysis_file: AnalysisFile) -> Optional[Path]:
    """The directory this file's top-level package must be imported from.

    ``climlab/domain/xarray.py`` with module ``climlab.domain.xarray`` is
    importable from the directory holding ``climlab/``, which is what pyright
    needs on its search path. Derived from the module name rather than assumed,
    because a source root may sit inside a package already (path ``src/shop``
    with prefix ``shop``).
    """
    depth = len(analysis_file.module.split(".")) if analysis_file.module else 1
    if analysis_file.path.name == "__init__.py":
        depth += 1
    parents = analysis_file.path.resolve().parents
    if depth - 1 >= len(parents):
        return None
    return parents[depth - 1]


def collect_pyright_families_from_analysis_files(
    project_root: Path,
    analysis_files: Sequence[AnalysisFile],
    pyright_bin: str,
    *,
    cache: Optional[ParsedFileCache] = None,
) -> Dict[str, str]:
    resolved_cache = cache if cache is not None else ParsedFileCache()
    targets: List[PyrightProbeTarget] = []
    support_files: List[Path] = []
    import_roots: Set[Path] = set()
    module_imports: Dict[str, Dict[str, str]] = {}
    for analysis_file in analysis_files:
        py_file = analysis_file.path
        module = analysis_file.module
        tree = resolved_cache.get(py_file)
        targets.extend(collect_pyright_probe_targets(tree, module, py_file))
        # Pyright names a class by the bare name the source wrote, so the probe
        # answer ``Dataset`` is only xarray's in a module that imported it from
        # there. This is the map that says which module did.
        imports, _star = collect_module_imports(
            tree, module, current_is_package=is_package_file(py_file)
        )
        module_imports[module] = imports
        # Every analyzed file, not only the probed ones. A module with no probe
        # target of its own still has to exist for the imports through it to
        # resolve, and package __init__ files -- pure re-exports, so never a
        # probe target -- are exactly the ones everything imports through.
        support_files.append(py_file)
        import_root = _import_root_for_analysis_file(analysis_file)
        if import_root is not None:
            import_roots.add(import_root)

    resolved_root = project_root.resolve()
    extra_paths: List[Path] = []
    for import_root in sorted(import_roots):
        try:
            extra_paths.append(import_root.relative_to(resolved_root))
        except ValueError:
            continue

    report = probe_pyright_targets(
        project_root,
        targets,
        pyright_bin,
        support_files=support_files,
        extra_paths=extra_paths,
        module_imports=module_imports,
    )
    _report_pyright_resolution(report)
    return report.families


def _report_pyright_resolution(report: PyrightProbeReport) -> None:
    """Print what the probe achieved, and refuse a run where it achieved nothing.

    No threshold is chosen here. Nothing resolving at all, while probes were
    emitted, is unambiguous and needs no number; anything short of that is
    printed rather than judged, because what counts as "enough" is a property of
    the project being analyzed, not of this code.
    """
    if not report.probes_emitted:
        return
    print(
        f"Pyright probes: {report.probes_emitted} emitted, "
        f"{report.probes_answered} answered, "
        f"{report.probes_resolved} resolved to a type "
        f"({report.answers_unknown} came back Unknown)",
        file=sys.stderr,
    )
    if report.files_outside_project_root:
        print(
            f"  {report.files_outside_project_root} file(s) skipped: outside the project root",
            file=sys.stderr,
        )
    if report.probes_resolved == 0:
        raise RuntimeError(
            f"Pyright resolved none of {report.probes_emitted} probes. "
            "Every container family would be 'unknown', which is indistinguishable "
            "from a correctly-typed unknown in the artifacts."
        )


def _mode_to_access(mode: str) -> str:
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        return "write"
    return "read"


def collect_module_globals(tree: ast.Module) -> Set[str]:
    globals_: Set[str] = set()

    def add_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            globals_.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                add_target(element)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                add_target(target)
        elif isinstance(node, ast.AnnAssign):
            add_target(node.target)
        elif isinstance(node, ast.AugAssign):
            add_target(node.target)
    return globals_


def collect_module_imports(
    tree: ast.Module,
    module: str = "",
    *,
    current_is_package: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    imports: Dict[str, str] = {}
    star_imports: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            imported_module = resolve_import_from_module(
                module,
                node.module,
                node.level,
                current_is_package=current_is_package,
            )
            if imported_module is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    if imported_module:
                        star_imports.append(imported_module)
                else:
                    target = resolve_import_from_target(
                        module,
                        node.module,
                        node.level,
                        alias.name,
                        current_is_package=current_is_package,
                    )
                    if target is not None:
                        imports[alias.asname or alias.name] = target
    return imports, star_imports


def known_class_ids_from_callable_map(callable_map: Dict[str, CallableDef]) -> Set[str]:
    """Every ``module.Class`` the callable map mentions."""
    return {
        f"{callable_def.module}.{callable_def.class_name}"
        for callable_def in callable_map.values()
        if getattr(callable_def, "class_name", None)
    }


class DataAccessCollector(RegisteredStateLineageMixin, ast.NodeVisitor):
    def __init__(
        self,
        module: str,
        file: Path,
        callable_map: Dict[str, CallableDef],
        module_globals: Set[str],
        module_imports: Optional[Dict[str, str]] = None,
        star_imports: Optional[List[str]] = None,
        return_summaries: Optional[Dict[str, ExprRef]] = None,
        return_tuple_summaries: Optional[Dict[str, Tuple[Optional[ExprRef], ...]]] = None,
        callable_params: Optional[Dict[str, Tuple[str, ...]]] = None,
        callable_return_identity: Optional[Dict[str, bool]] = None,
        param_bindings: Optional[Dict[str, Set[str]]] = None,
        param_access_paths: Optional[Dict[str, Set[str]]] = None,
        lineage_edges: Optional[List[LineageEdge]] = None,
        split_class_owners: Optional[Set[str]] = None,
        pyright_families: Optional[Dict[str, str]] = None,
        attrdict_classes: Optional[Set[str]] = None,
        registration_rules: Optional[Dict[str, RegistrationRule]] = None,
        project_index: Optional[ProjectIndex] = None,
        known_classes: Optional[Set[str]] = None,
    ):
        self.module = module
        self.file = file
        self.callable_map = callable_map
        self.module_globals = module_globals
        self.module_imports = module_imports if module_imports is not None else {}
        self.star_imports = star_imports if star_imports is not None else []
        self.return_summaries = return_summaries if return_summaries is not None else {}
        self.return_tuple_summaries = (
            return_tuple_summaries if return_tuple_summaries is not None else {}
        )
        self.callable_params = callable_params if callable_params is not None else {}
        # Per callable: does every path out of it hand back the same expression?
        # Filled in ``_enter_callable`` from the syntax alone, so it is settled
        # on the first pass and can ride the same plumbing as
        # ``callable_params`` instead of needing its own fixpoint entry.
        self.callable_return_identity = (
            callable_return_identity if callable_return_identity is not None else {}
        )
        self.param_bindings = param_bindings if param_bindings is not None else {}
        self.param_access_paths = param_access_paths if param_access_paths is not None else {}
        self.lineage_edges = lineage_edges if lineage_edges is not None else []
        self.split_class_owners = split_class_owners if split_class_owners is not None else set()
        self.pyright_families = pyright_families if pyright_families is not None else {}
        self.attrdict_classes = attrdict_classes if attrdict_classes is not None else set()
        # Registration facts derived by the call-graph passes. Both default to
        # empty so a caller that only wants data access -- the direct
        # ``collect_data_access`` entry point, and every test using it -- pays
        # nothing for them; registration lineage is then simply not recorded.
        self.registration_rules = registration_rules or {}
        self.project_index = project_index
        self.attrdict_access_paths: Set[str] = set()
        # ``callable_map`` does not change during a run, so this set is the same
        # every time it is built -- and a collector is constructed once per file
        # per fixpoint pass. Callers that have more than one file compute it once
        # and hand it in; the fallback keeps single-file callers self-contained.
        self.known_classes = (
            known_classes
            if known_classes is not None
            else known_class_ids_from_callable_map(callable_map)
        )
        self.objects: Dict[str, DataObject] = {}
        self.edges: List[AccessEdge] = []
        self.created_object_ids: Set[str] = set()
        # IDs that two passes described with two different real kinds. Not an
        # artifact column -- a diagnostic, in the spirit of the ``unknown``-kind
        # counter: every entry is two passes disagreeing about what one object is.
        self.kind_conflicts: Set[str] = set()

        self.module_callable = module_callable_id(module)
        self.current_class: Optional[str] = None
        self.current_callable: Optional[str] = None
        self.callable_stack: List[str] = []
        self.scopes: List[Scope] = []

    @property
    def scope(self) -> Optional[Scope]:
        return self.scopes[-1] if self.scopes else None

    def _current_owner(self) -> str:
        if self.current_class:
            return f"{self.module}.{self.current_class}"
        return self.module

    def _known_class_ids(self) -> Set[str]:
        return set(self.known_classes) | set(self.attrdict_classes)

    def _resolve_class_reference_name(self, name: str) -> Set[str]:
        """Every project class a class-reference expression could name.

        Two resolvers unioned rather than one replacing the other.
        ``ProjectIndex`` is strictly better on the four axes that matter --
        it canonicalizes re-export aliases, follows star imports, indexes
        classes that define no methods, and sees imports nested inside
        functions and ``try`` blocks -- but it is built from the call graph's
        class index, which has no notion of an attrdict class. The local
        resolver is the only thing that resolves those, so dropping it would
        silently lose resolutions, and the registration-lineage gates are
        ``len(...) != 1``: a loss produces no edge and no warning, which is
        indistinguishable from the defect this fixes.
        """
        matches = self._local_class_reference_matches(name)
        if name and self.project_index is not None:
            module_index = self.project_index.module_map.get(self.module)
            if module_index is not None:
                matches.update(
                    self.project_index.resolve_class_reference_name(
                        self.module, module_index, name
                    )
                )
        return matches

    def _local_class_reference_matches(self, name: str) -> Set[str]:
        if not name:
            return set()
        known_classes = self._known_class_ids()
        matches: Set[str] = set()
        if "." not in name:
            local = f"{self.module}.{name}"
            if local in known_classes:
                matches.add(local)
            imported = self.module_imports.get(name)
            if imported and imported in known_classes:
                matches.add(imported)
            return matches

        base, _sep, _rest = name.partition(".")
        imported_base = self.module_imports.get(base)
        if imported_base:
            candidate = f"{imported_base}.{_rest}"
            if candidate in known_classes:
                matches.add(candidate)
        if name in known_classes:
            matches.add(name)
        return matches

    def _class_types_from_call(self, value: ast.Call) -> Set[str]:
        call_name = _attribute_path(value.func) or ""
        return self._resolve_class_reference_name(call_name)

    def _class_types_from_value(self, value: object) -> Set[str]:
        if isinstance(value, ast.Call):
            return self._class_types_from_call(value)
        if isinstance(value, ast.Name) and self.scope:
            return set(self.scope.local_class_types.get(value.id, set()))
        if isinstance(value, ast.Attribute) and self.scope:
            path = _attribute_path(value)
            if path:
                return set(self.scope.attr_class_types.get(path, set()))
        return set()

    def _element_class_types_from_value(self, value: object) -> Set[str]:
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            element_types: Set[str] = set()
            for element in value.elts:
                element_types.update(self._class_types_from_value(element))
            return element_types
        if isinstance(value, ast.Name) and self.scope:
            return set(self.scope.local_element_class_types.get(value.id, set()))
        return set()

    def _class_types_from_expr(self, node: ast.AST) -> Set[str]:
        if (
            isinstance(node, ast.Name)
            and node.id in {"self", "cls"}
            and self.current_class
        ):
            return {f"{self.module}.{self.current_class}"}
        return self._class_types_from_value(node)

    def _value_is_attrdict_constructor(self, value: object) -> bool:
        return isinstance(value, ast.Call) and bool(
            self._class_types_from_call(value) & self.attrdict_classes
        )

    def _mark_attrdict_access_path(self, path: str) -> None:
        if path:
            self.attrdict_access_paths.add(path)

    def _family_for_object_id(self, object_id: str) -> str:
        family = self.pyright_families.get(object_id, "")
        return family if family else FAMILY_UNKNOWN

    def _family_for_attr_expr(self, attr_path: str) -> str:
        if not self.current_callable:
            return FAMILY_UNKNOWN
        family = self.pyright_families.get(_attr_expr_family_key(self.current_callable, attr_path), "")
        return family if family else FAMILY_UNKNOWN

    def _family_for_class_attr(self, owner: str, attr_name: str) -> str:
        family = self.pyright_families.get(_class_attr_family_key(owner, attr_name), "")
        return family if family else FAMILY_UNKNOWN

    def _register_object(
        self,
        object_id: str,
        kind: str,
        display_name: str,
        scope: str,
        owner: str = "",
        container: str = "",
        field: str = "",
        node: Optional[ast.AST] = None,
        inferred_type: str = "",
        confidence: str = "high",
        alias_of: str = "",
        access_path: str = "",
        structural_role: str = "primary",
    ) -> None:
        lineno = getattr(node, "lineno", 0) if node is not None else 0
        candidate = DataObject(
            id=object_id,
            kind=kind,
            display_name=display_name,
            scope=scope,
            owner=owner,
            container=container,
            field=field,
            file=str(self.file),
            lineno=lineno,
            inferred_type=inferred_type,
            confidence=confidence,
            alias_of=alias_of,
            access_path=access_path,
            structural_role=structural_role,
        )
        existing = self.objects.get(object_id)
        if existing is None:
            self.objects[object_id] = candidate
            return
        if merge_data_object(existing, candidate):
            self.kind_conflicts.add(object_id)

    def _record_access(
        self,
        object_id: str,
        access: str,
        operation: str,
        node: ast.AST,
        confidence: str = "high",
        evidence: Optional[str] = None,
    ) -> None:
        if self.current_callable is None:
            return
        if object_id not in self.objects:
            self._register_object(
                object_id=object_id,
                kind="unknown",
                display_name=object_id,
                scope="unknown",
                node=node,
                confidence=confidence,
            )
        self.edges.append(
            AccessEdge(
                callable=self.current_callable,
                object_id=object_id,
                access=access,
                operation=operation,
                file=str(self.file),
                lineno=getattr(node, "lineno", 0),
                confidence=confidence,
                evidence=evidence if evidence is not None else _unparse(node),
            )
        )

    def _store_access_for_object(self, object_id: str, requested_access: str) -> str:
        if requested_access != "create":
            return requested_access
        if object_id in self.created_object_ids:
            return "write"
        self.created_object_ids.add(object_id)
        return "create"

    def _record_lineage(
        self,
        src_object_id: str,
        dst_object_id: str,
        relation: str,
        node: ast.AST,
        callee: str = "",
        slot: str = "",
    ) -> None:
        if not src_object_id or not dst_object_id:
            return
        self.lineage_edges.append(
            LineageEdge(
                src_object_id=src_object_id,
                dst_object_id=dst_object_id,
                relation=relation,
                file=str(self.file),
                lineno=getattr(node, "lineno", 0),
                caller=self.current_callable or "",
                callee=callee,
                slot=slot,
            )
        )

    def _class_attr_ref(self, attr_path: str, node: ast.AST, confidence: str = "high") -> ExprRef:
        owner = self._current_owner()
        top_attr = _top_level_attr_name(attr_path)
        split_class_state = owner in self.split_class_owners and bool(top_attr)
        object_id = (
            f"class_attr_state:{owner}:{top_attr}"
            if split_class_state
            else f"class_state:{owner}"
        )
        kind = "class_attr_state" if split_class_state else "class_state"
        display_name = (
            _class_attr_state_display(owner, top_attr)
            if split_class_state
            else _class_state_display(owner)
        )
        inferred = FAMILY_UNKNOWN
        if self.scope and attr_path in self.scope.attr_bindings:
            binding = self.scope.attr_bindings[attr_path]
            inferred = binding.inferred_type or self._family_for_object_id(binding.alias_of or binding.object_id)
        if _is_unknown_family(inferred):
            inferred = self._family_for_class_attr(owner, top_attr or attr_path)
        self._register_object(
            object_id=object_id,
            kind=kind,
            display_name=display_name,
            scope="class",
            owner=owner,
            field=top_attr if split_class_state else "",
            node=node,
            inferred_type=FAMILY_OBJECT,
            confidence=confidence,
            access_path=f"self.{top_attr or attr_path}",
        )
        access_path = f"self.{top_attr or attr_path}"
        if inferred == FAMILY_DICT:
            self._mark_attrdict_access_path(access_path)
        return ExprRef(object_id, inferred, confidence, access_path, access_path)

    def _param_ref(self, name: str, node: ast.AST) -> ExprRef:
        callable_id = self.current_callable or ""
        object_id = f"param:{callable_id}:{name}"
        inferred_type = self._family_for_object_id(object_id)
        self._register_object(
            object_id=object_id,
            kind="param",
            display_name=name,
            scope="callable",
            owner=callable_id,
            field=name,
            node=node,
            inferred_type=inferred_type,
            confidence="high",
            access_path=name,
        )
        return ExprRef(object_id, inferred_type, "high", name, name)

    def _global_ref(self, name: str, node: ast.AST) -> ExprRef:
        object_id = f"module_global:{self.module}.{name}"
        inferred_type = self._family_for_object_id(object_id)
        self._register_object(
            object_id=object_id,
            kind="module_global",
            display_name=name,
            scope="module",
            owner=self.module,
            field=name,
            node=node,
            inferred_type=inferred_type,
            confidence="high",
            access_path=name,
        )
        return ExprRef(object_id, inferred_type, "high", name, name)

    def _local_ref(
        self,
        name: str,
        node: ast.AST,
        inferred_type: str = "",
        confidence: str = "medium",
        alias_of: str = "",
        alias_display_name: str = "",
        expose: bool = True,
        class_types: Optional[Set[str]] = None,
    ) -> ExprRef:
        callable_id = self.current_callable or ""
        object_id = f"local_exposed:{callable_id}:{name}"
        inferred = inferred_type or self._family_for_object_id(object_id)
        # We always track the local in scope, but only materialize a
        # local_exposed object when it escapes or aliases something meaningful.
        if expose:
            self._register_object(
                object_id=object_id,
                kind="local_exposed",
                display_name=name,
                scope="callable",
                owner=callable_id,
                field=name,
                node=node,
                inferred_type=inferred,
                confidence=confidence,
                alias_of=alias_of,
                access_path=alias_display_name or name,
            )
        if self.scope is not None:
            self.scope.locals[name] = LocalBinding(
                object_id=object_id,
                inferred_type=inferred,
                confidence=confidence,
                alias_of=alias_of,
                display_name=alias_display_name,
                access_path=alias_display_name or name,
                exposed=expose,
                node=node,
                class_types=set(class_types or set()),
            )
        return ExprRef(object_id, inferred, confidence, name, name)

    def _ensure_local_exposed(self, name: str, node: ast.AST) -> Optional[LocalBinding]:
        # Promote a previously internal local to a first-class object once the
        # container itself becomes architecturally relevant (returned, passed,
        # mutated, used as the receiver of a method call, etc.).
        if self.scope is None or name not in self.scope.locals:
            return None
        binding = self.scope.locals[name]
        if binding.exposed:
            return binding
        callable_id = self.current_callable or ""
        if not binding.object_id:
            binding.object_id = f"local_exposed:{callable_id}:{name}"
        self._register_object(
            object_id=binding.object_id,
            kind="local_exposed",
            display_name=name,
            scope="callable",
            owner=callable_id,
            field=name,
            node=binding.node or node,
            inferred_type=binding.inferred_type,
            confidence=binding.confidence,
            alias_of=binding.alias_of,
            access_path=binding.access_path or name,
        )
        self._record_access(
            binding.object_id,
            "create",
            "assign",
            binding.node or node,
            binding.confidence,
            evidence=name,
        )
        binding.exposed = True
        return binding

    def _file_ref(self, file_expr: ast.AST, node: ast.AST, confidence: str = "high") -> ExprRef:
        if isinstance(file_expr, ast.Constant) and isinstance(file_expr.value, str):
            display = file_expr.value
            object_part = file_expr.value
        else:
            display = _unparse(file_expr)
            object_part = display
            confidence = "medium" if confidence == "high" else confidence
        object_id = f"file:{_safe_id_part(object_part)}"
        self._register_object(
            object_id=object_id,
            kind="file",
            display_name=display,
            scope="external",
            owner="filesystem",
            node=node,
            inferred_type=FAMILY_FILE,
            confidence=confidence,
            access_path=display,
        )
        return ExprRef(object_id, FAMILY_FILE, confidence, display, display)

    def _field_ref(
        self,
        base: ExprRef,
        field: str,
        node: ast.AST,
        kind: Optional[str] = None,
        confidence: str = "high",
    ) -> ExprRef:
        object_kind = kind or self._field_kind(base)
        object_id = self._field_object_id(object_kind, base, field)
        base_path = base.access_path or base.display_name
        shared_container = _shared_field_container(base_path, object_kind)
        display_base = shared_container or base_path
        display = _path_with_field(display_base, field)
        access_path = _path_with_field(base_path, field)
        inferred = _field_inferred_type(object_kind)
        owner = f"shared_container:{shared_container}" if shared_container else base.object_id
        structural_role = "precise" if base.object_id.startswith("object_state:") else "primary"
        self._register_object(
            object_id=object_id,
            kind=object_kind,
            display_name=display,
            scope="field",
            owner=owner,
            container=owner,
            field=field,
            node=node,
            inferred_type=inferred,
            confidence=confidence,
            access_path=access_path,
            structural_role=structural_role,
        )
        return ExprRef(object_id, inferred, confidence, display, access_path)

    def _field_object_id(self, object_kind: str, base: ExprRef, field: str) -> str:
        container_path = _safe_id_part(base.access_path or base.display_name)
        container_path_id = _safe_path_id_part(container_path)
        field_part = _safe_id_part(field)
        base_object = self.objects.get(base.object_id)
        shared_container = _shared_field_container(container_path, object_kind)
        if shared_container:
            return f"{object_kind}:{shared_container}:{field_part}"

        if _is_class_state_object_id(base.object_id):
            return f"{object_kind}:{base.object_id}:{container_path_id}:{field_part}"

        if base.object_id.startswith("object_state:"):
            if base_object and base_object.container.startswith("param:"):
                callable_id, _param = self._split_param_id(base_object.container)
                return f"{object_kind}:{callable_id}:{container_path_id}:{field_part}"
            if base_object and base_object.container.startswith("local_exposed:"):
                callable_id, _local = self._split_local_id(base_object.container)
                return f"{object_kind}:{callable_id}:{container_path_id}:{field_part}"
            return f"{object_kind}:{base.object_id}:{container_path_id}:{field_part}"

        if base.object_id.startswith("param:"):
            callable_id, _param = self._split_param_id(base.object_id)
            return f"{object_kind}:{callable_id}:{container_path_id}:{field_part}"

        if base.object_id.startswith("local_exposed:"):
            callable_id, _local = self._split_local_id(base.object_id)
            return f"{object_kind}:{callable_id}:{container_path_id}:{field_part}"

        return f"{object_kind}:{base.object_id}:{field_part}"

    def _split_param_id(self, object_id: str) -> Tuple[str, str]:
        rest = object_id.removeprefix("param:")
        callable_id, _, name = rest.rpartition(":")
        return callable_id, name

    def _split_local_id(self, object_id: str) -> Tuple[str, str]:
        rest = object_id.removeprefix("local_exposed:")
        callable_id, _, name = rest.rpartition(":")
        return callable_id, name

    def _object_attr_ref(self, base: ExprRef, attr: str, node: ast.AST, attr_path: str = "") -> ExprRef:
        if base.object_id.startswith("dict_key:"):
            base_path = base.access_path or base.display_name
            display = _path_with_attr(base.display_name, attr)
            access_path = _path_with_attr(base_path, attr)
            inferred = self._family_for_attr_expr(attr_path) if attr_path else FAMILY_UNKNOWN
            confidence = "medium" if base.confidence == "medium" else base.confidence
            return ExprRef(base.object_id, inferred, confidence, display, access_path)

        base_path = base.access_path or base.display_name
        if base_path in self.attrdict_access_paths or base.inferred_type == FAMILY_DICT:
            return self._field_ref(base, attr, node, kind="dict_key", confidence="medium")
        access_path = _path_with_attr(base_path, attr)
        display = access_path
        inferred = self._family_for_attr_expr(attr_path) if attr_path else FAMILY_UNKNOWN
        confidence = "medium" if base.confidence == "medium" else base.confidence
        base_object = self.objects.get(base.object_id)
        if base.object_id.startswith("object_state:") and base_object:
            root_object_id = base_object.container or base_object.owner or base.object_id
        else:
            root_object_id = base.object_id
        object_path_id = _safe_path_id_part(
            _root_relative_access_path(root_object_id, access_path)
        )
        object_id = f"object_state:{root_object_id}:{object_path_id}"
        coarse_object_id = ""
        if not _is_state_object_id(base.object_id):
            coarse_object_id = f"object_state:{base.object_id}"
            self._register_object(
                object_id=coarse_object_id,
                kind="object_state",
                display_name=f"{base.display_name} state",
                scope="object",
                owner=base.object_id,
                container=base.object_id,
                node=node,
                inferred_type=FAMILY_OBJECT,
                confidence=confidence,
                access_path=base_path,
                structural_role="coarse",
            )
        self._register_object(
            object_id=object_id,
            kind="object_state",
            display_name=display,
            scope="object",
            owner=root_object_id,
            container=root_object_id,
            field=attr,
            node=node,
            inferred_type=FAMILY_OBJECT,
            confidence=confidence,
            access_path=access_path,
            structural_role="precise",
        )
        return ExprRef(object_id, inferred, confidence, display, access_path, coarse_object_id)

    def _field_kind(self, base: ExprRef) -> str:
        if base.inferred_type == FAMILY_DATAFRAME:
            return "df_col"
        if base.inferred_type == FAMILY_DICT:
            return "dict_key"
        if base.inferred_type == FAMILY_XARRAY:
            return "dict_key"
        return CONTAINER_FIELD_KIND

    def _infer_type_from_value(self, value: object) -> ValueOrigin:
        """The family of an assigned value, and the object it came from.

        ``ValueOrigin.identity`` says whether the value *is* that object or was
        merely *made from* it. Only the first may become ``alias_of``; the
        second becomes a ``derived_from`` lineage edge, which keeps the flow
        without merging the two nodes.
        """
        if isinstance(value, ExprRef):
            return ValueOrigin(
                value.inferred_type, value.object_id, value.confidence, not value.derived
            )
        if isinstance(value, ast.Dict):
            return ValueOrigin(FAMILY_DICT, "", "high")
        if isinstance(value, ast.List):
            return ValueOrigin(FAMILY_LIST, "", "high")
        if isinstance(value, ast.Set):
            return ValueOrigin(FAMILY_SET, "", "high")
        if isinstance(value, ast.Tuple):
            return ValueOrigin(FAMILY_UNKNOWN, "", "medium")
        if isinstance(value, ast.Subscript):
            base = self._resolve_expr(value.value)
            if (
                base
                and self._field_kind(base) == "df_col"
                and _slice_selects_multiple_fields(_slice_value(value))
            ):
                return ValueOrigin(FAMILY_DATAFRAME, base.object_id, "medium", not base.derived)
        if isinstance(value, ast.Call):
            call_name = _attribute_path(value.func) or ""
            returned = self._return_ref_from_call(value)
            if returned:
                return ValueOrigin(
                    returned.inferred_type,
                    returned.object_id,
                    returned.confidence,
                    not returned.derived,
                )
            if self._value_is_attrdict_constructor(value):
                return ValueOrigin(FAMILY_DICT, "", "medium")
            if call_name in {"dict", "builtins.dict"} or call_name.endswith(".dict"):
                return ValueOrigin(FAMILY_DICT, "", "high")
            if call_name in {"list", "builtins.list"} or call_name.endswith(".list"):
                return ValueOrigin(FAMILY_LIST, "", "high")
            if call_name in {"set", "builtins.set"} or call_name.endswith(".set"):
                return ValueOrigin(FAMILY_SET, "", "high")
            if call_name.endswith("DataFrame") or call_name.endswith("read_csv"):
                return ValueOrigin(FAMILY_DATAFRAME, "", "high")
            if call_name.endswith(tuple(FILE_READ_FUNCS)):
                return ValueOrigin(FAMILY_DATAFRAME, "", "high")
            if _call_name_matches(call_name, XARRAY_OPEN_FUNCS):
                return ValueOrigin(FAMILY_XARRAY, "", "high")
            if _call_name_matches(call_name, POOCH_READ_FUNCS):
                return ValueOrigin(FAMILY_PATH, "", "medium")
            if _is_copy_call(value):
                # A copy is the one call in the language whose whole purpose is
                # to *not* be the same object, so the base is where the value
                # came from and its family, but never what it is. Step 1b caught
                # this directly: ``dic = self.state.copy()`` was recorded as
                # ``dic`` being ``self.state``.
                base = self._resolve_expr(value.func.value)
                if base:
                    return ValueOrigin(
                        base.inferred_type or FAMILY_UNKNOWN, base.object_id, "medium", False
                    )
            if isinstance(value.func, ast.Attribute) and value.func.attr in {
                "loc",
                "iloc",
                "pivot_table",
            }:
                return ValueOrigin(FAMILY_DATAFRAME, "", "medium")
        if isinstance(value, ast.Name) and self.scope and value.id in self.scope.locals:
            binding = self.scope.locals[value.id]
            return ValueOrigin(
                binding.inferred_type,
                binding.alias_of or binding.object_id,
                binding.confidence,
            )
        if isinstance(value, (ast.Attribute, ast.Name)):
            ref = self._resolve_expr(value)
            if ref:
                return ValueOrigin(
                    ref.inferred_type, ref.object_id, ref.confidence, not ref.derived
                )
        return ValueOrigin(FAMILY_UNKNOWN, "", "low")

    def _alias_display_from_value(self, value: object, alias_of: str) -> str:
        if not alias_of:
            return ""
        if isinstance(value, ExprRef):
            return value.display_name
        if isinstance(value, ast.Subscript):
            base = self._resolve_expr(value.value)
            return base.display_name if base else ""
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            returned = self._return_ref_from_call(value)
            if returned:
                return returned.display_name
            base = self._resolve_expr(value.func.value)
            return base.display_name if base else ""
        if isinstance(value, ast.Call):
            returned = self._return_ref_from_call(value)
            if returned:
                return returned.display_name
        if isinstance(value, ast.Name) and self.scope and value.id in self.scope.locals:
            return self.scope.locals[value.id].display_name
        if isinstance(value, (ast.Attribute, ast.Name)):
            ref = self._resolve_expr(value)
            return ref.display_name if ref else ""
        return ""

    def _resolve_unpack_value_refs(
        self,
        value: ast.AST,
    ) -> Optional[Tuple[Optional[ExprRef], ...]]:
        if isinstance(value, (ast.Tuple, ast.List)):
            return tuple(self._resolve_expr(element) for element in value.elts)
        if isinstance(value, ast.Call):
            return self._return_refs_from_call(value)
        return None

    def _callable_id_for_node(self, name: str) -> str:
        prev_callable = self.current_callable
        if prev_callable is not None and prev_callable != self.module_callable:
            return f"{prev_callable}.<locals>.{name}"
        if self.current_class:
            return f"{self.module}.{self.current_class}.{name}"
        return f"{self.module}.{name}"

    def _canonical_callable_id(self, candidate: str) -> Optional[str]:
        """The defining path of a re-export alias, when that path is a known callable.

        The attribute path a call is spelled with is only the *spelling*:
        ``import parcels._sgrid as sgrid`` followed by ``sgrid.get_n_faces()``
        names ``parcels._sgrid.get_n_faces``, but the function is defined in
        ``parcels._sgrid.core`` and merely re-exported by the package
        ``__init__``. Only the defining path is a key in ``callable_map``, so
        without this the call resolves to nothing at all -- and unlike the call
        graph, where that costs an edge, here it costs the object too.

        Mirrors ``call_graph.collector.resolution._resolve_module_alias_target``.
        Returns ``None`` when there is nothing to add, so callers append rather
        than substitute and today's candidate order is preserved.
        """
        if self.project_index is None:
            return None
        canonical = self.project_index.canonical_callable_id(candidate)
        if canonical != candidate and canonical in self.callable_map:
            return canonical
        return None

    def _append_candidate(self, candidates: List[str], candidate: str) -> None:
        """Record a candidate, followed by its canonical form when one exists.

        Appended rather than substituted: ``_return_ref_from_call`` takes the
        first candidate that hits a map, so keeping the raw spelling first means
        every resolution that works today still wins, and the canonical entry
        only fires where nothing resolved before.
        """
        candidates.append(candidate)
        canonical = self._canonical_callable_id(candidate)
        if canonical is not None:
            candidates.append(canonical)

    def _is_known_callable_id(self, candidate: str) -> bool:
        """Is this ID a callable any pass knows about?

        Four dict lookups, where this used to be four whole-corpus set
        constructions unioned together -- rebuilt for every ``ast.Call`` node
        visited, and profiled at 26% of the stage's self time on climlab.

        A cached union is not an option: ``return_summaries`` and
        ``return_tuple_summaries`` are written by the collector *during* its own
        traversal, so a snapshot goes stale mid-walk. Testing membership against
        the live dicts is equivalent by definition and cannot go stale, which is
        why this and not the frozen index the review proposed.
        """
        return (
            candidate in self.callable_map
            or candidate in self.callable_params
            or candidate in self.return_summaries
            or candidate in self.return_tuple_summaries
        )

    def _all_known_callable_ids(self) -> Set[str]:
        """Every known callable ID, for the one caller that needs a prefix scan."""
        return (
            set(self.callable_map)
            | set(self.callable_params)
            | set(self.return_summaries)
            | set(self.return_tuple_summaries)
        )

    def _candidate_callable_ids_for_call(self, node: ast.Call) -> List[str]:
        dynamic_candidates = self._dynamic_getattr_callable_ids(node.func)
        if dynamic_candidates:
            return dynamic_candidates

        call_name = _attribute_path(node.func) or ""
        if not call_name:
            return []
        candidates: List[str] = []
        if call_name in self.callable_map:
            candidates.append(call_name)
        if call_name in self.module_imports:
            self._append_candidate(candidates, self.module_imports[call_name])
        if "." in call_name:
            prefix, _, suffix = call_name.partition(".")
            if prefix in self.module_imports:
                self._append_candidate(candidates, f"{self.module_imports[prefix]}.{suffix}")
        if "." not in call_name:
            for imported_module in self.star_imports:
                candidates.append(f"{imported_module}.{call_name}")
            candidates.append(f"{self.module}.{call_name}")
            if self.current_class:
                candidates.append(f"{self.module}.{self.current_class}.{call_name}")
        elif call_name.startswith(("self.", "cls.")) and self.current_class:
            method = call_name.split(".", 1)[1]
            candidates.append(f"{self.module}.{self.current_class}.{method}")
        else:
            self._append_candidate(candidates, call_name)
            candidates.append(f"{self.module}.{call_name}")
        expanded: List[str] = []
        for candidate in candidates:
            expanded.append(candidate)
            init_candidate = f"{candidate}.__init__"
            if self._is_known_callable_id(init_candidate):
                expanded.append(init_candidate)
        return list(dict.fromkeys(expanded))

    def _dynamic_getattr_callable_ids(self, func: ast.AST) -> List[str]:
        """Resolve calls like getattr(module_alias, dynamic_name)(...) conservatively."""
        if not isinstance(func, ast.Call):
            return []
        if not isinstance(func.func, ast.Name) or func.func.id != "getattr":
            return []
        if len(func.args) < 2:
            return []

        target_obj, attr_arg = func.args[0], func.args[1]
        if not isinstance(target_obj, ast.Name):
            return []

        imported_base = self.module_imports.get(target_obj.id)
        if not imported_base:
            return []

        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
            target = f"{imported_base}.{attr_arg.value}"
            if self._is_known_callable_id(target):
                return [target]
            # Same alias canonicalization as the direct-call path: the attribute
            # a package re-exports is not the path the callable is defined at.
            canonical = self._canonical_callable_id(target)
            return [canonical] if canonical is not None else []

        # The only branch that genuinely needs the whole set rather than a
        # membership test, and it is reached only by a getattr with a computed
        # attribute name. Built here rather than above so the common path never
        # pays for it.
        prefix = f"{imported_base}."
        return sorted(
            callable_id
            for callable_id in self._all_known_callable_ids()
            if callable_id.startswith(prefix)
        )

    def _bind_call_args_to_params(
        self,
        node: ast.Call,
        param_names: Sequence[str],
    ) -> List[Tuple[str, ast.AST]]:
        bindings: List[Tuple[str, ast.AST]] = []
        used_params: Set[str] = set()
        for param_name, arg in zip(param_names, node.args):
            bindings.append((param_name, arg))
            used_params.add(param_name)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.value is None:
                continue
            if keyword.arg not in param_names or keyword.arg in used_params:
                continue
            bindings.append((keyword.arg, keyword.value))
            used_params.add(keyword.arg)
        return bindings

    def _record_confirmed_param_lineage(self, node: ast.Call) -> None:
        candidates = [
            callable_id
            for callable_id in self._candidate_callable_ids_for_call(node)
            if callable_id in self.callable_params
        ]
        is_dynamic_getattr = bool(self._dynamic_getattr_callable_ids(node.func))
        if not candidates or (len(candidates) != 1 and not is_dynamic_getattr):
            return

        for callee_id in candidates:
            param_names = self.callable_params.get(callee_id, ())
            for param_name, arg in self._bind_call_args_to_params(node, param_names):
                arg_ref = self._resolve_expr_for_lineage(arg)
                sources: List[Tuple[str, bool]] = []
                if arg_ref is not None and arg_ref.object_id:
                    sources.append((arg_ref.object_id, not arg_ref.derived))
                sources.extend(self._lineage_source_ids_from_value(arg))
                if not sources:
                    continue
                param_id = f"param:{callee_id}:{param_name}"
                if arg_ref is not None and arg_ref.access_path:
                    self.param_access_paths.setdefault(param_id, set()).add(arg_ref.access_path)
                for source_id, identity in dict.fromkeys(sources):
                    if identity:
                        # ``param_bindings`` feeds ``_apply_confirmed_param_aliases``,
                        # which mints ``alias_of`` from it directly. A derived
                        # argument left in here would put back the same false
                        # merge from the other side.
                        self.param_bindings.setdefault(param_id, set()).add(source_id)
                    self._record_lineage(
                        source_id,
                        param_id,
                        RELATION_ARG_TO_PARAM if identity else RELATION_DERIVED_FROM,
                        node,
                        callee=callee_id,
                        slot=param_name,
                    )

    def _return_refs_from_call(self, node: ast.Call) -> Optional[Tuple[Optional[ExprRef], ...]]:
        for callable_id in self._candidate_callable_ids_for_call(node):
            refs = self.return_tuple_summaries.get(callable_id)
            if refs:
                if self.callable_return_identity.get(callable_id, False):
                    return refs
                return tuple(_as_derived(ref) for ref in refs)
        return None

    def _call_returns_one_source(self, node: ast.AST) -> bool:
        """Whether this call site's callee returns the same thing on every path.

        ``False`` when the callee is unknown as well as when it is known to have
        several return paths: an unresolved call is not evidence of identity.
        """
        if not isinstance(node, ast.Call):
            return False
        for callable_id in self._candidate_callable_ids_for_call(node):
            if callable_id in self.callable_return_identity:
                return self.callable_return_identity[callable_id]
        return False

    def _unique_returning_callable_for_call(
        self,
        node: ast.Call,
        slot: Optional[int] = None,
    ) -> str:
        candidates: List[str] = []
        for callable_id in self._candidate_callable_ids_for_call(node):
            if slot is None:
                if callable_id in self.return_summaries:
                    candidates.append(callable_id)
            else:
                refs = self.return_tuple_summaries.get(callable_id)
                if refs is not None and slot < len(refs):
                    candidates.append(callable_id)
        candidates = list(dict.fromkeys(candidates))
        return candidates[0] if len(candidates) == 1 else ""

    def _return_slot_lineage_source_from_call(
        self,
        node: ast.AST,
        slot: Optional[int] = None,
    ) -> str:
        if not isinstance(node, ast.Call):
            return ""
        callable_id = self._unique_returning_callable_for_call(node, slot=slot)
        if not callable_id:
            return ""
        return _return_slot_id(callable_id, slot)

    def _return_ref_from_call(self, node: ast.Call) -> Optional[ExprRef]:
        # Lightweight interprocedural step: if we already know what a callee
        # returns, treat that as the value flowing into this call site.
        for callable_id in self._candidate_callable_ids_for_call(node):
            if callable_id in self.return_summaries:
                summary = self.return_summaries[callable_id]
                confidence = "medium" if summary.confidence == "high" else summary.confidence
                return ExprRef(
                    summary.object_id,
                    summary.inferred_type,
                    confidence,
                    summary.display_name,
                    summary.access_path,
                    # The summary keeps one "best" return per callable, so on a
                    # callable with several return paths it describes one of
                    # them. That is fine for the family and the field structure
                    # and wrong for identity, which is what ``derived`` records.
                    derived=not self.callable_return_identity.get(callable_id, False),
                )
        return None

    def _raw_bound_object_id_for_lineage(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name) and self.scope and node.id in self.scope.locals:
            binding = self.scope.locals[node.id]
            if binding.alias_of and binding.object_id:
                return binding.object_id
        if isinstance(node, ast.Attribute):
            path = _attribute_path(node)
            if path and self.scope and path in self.scope.attr_bindings:
                binding = self.scope.attr_bindings[path]
                if (
                    binding.alias_of
                    and binding.object_id
                    and not binding.object_id.startswith("class_state:")
                ):
                    return binding.object_id
        return ""

    def _lineage_source_ids_from_value(
        self,
        value: object,
        slot: Optional[int] = None,
    ) -> List[Tuple[str, bool]]:
        """Extra lineage sources for a value, each with whether it is an identity.

        The ``return:`` marker is an identity source only when the callee
        returns one and the same thing on every path -- otherwise the value here
        was merely *made by* that call. See ``_returns_one_source``.
        """
        if not isinstance(value, ast.AST):
            return []
        sources: List[Tuple[str, bool]] = []
        raw_bound_id = self._raw_bound_object_id_for_lineage(value)
        if raw_bound_id:
            sources.append((raw_bound_id, True))
        return_slot_id = self._return_slot_lineage_source_from_call(value, slot=slot)
        if return_slot_id:
            sources.append((return_slot_id, self._call_returns_one_source(value)))
        return list(dict.fromkeys(sources))

    def _resolve_expr_for_lineage(self, node: ast.AST) -> Optional[ExprRef]:
        if isinstance(node, ast.Attribute):
            path = _attribute_path(node)
            if path and self.scope and path in self.scope.attr_bindings:
                binding = self.scope.attr_bindings[path]
                return ExprRef(
                    binding.alias_of or binding.object_id,
                    binding.inferred_type,
                    binding.confidence,
                    binding.display_name or path.rsplit(".", 1)[-1],
                    binding.access_path or binding.display_name or path,
                )
        return self._resolve_expr(node)

    def _enter_callable(self, node: ast.AST, name: str) -> None:
        callable_id = self._callable_id_for_node(name)
        prev_callable = self.current_callable
        self.current_callable = callable_id
        self.callable_stack.append(callable_id)

        params: Tuple[str, ...] = ()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
            if node.args.vararg:
                all_args.append(node.args.vararg)
            if node.args.kwarg:
                all_args.append(node.args.kwarg)
            params = tuple(arg.arg for arg in all_args if arg.arg not in {"self", "cls"})
            self.callable_params[callable_id] = params
            self.callable_return_identity[callable_id] = _returns_one_source(node)

            # Defaults can read module globals or outer-scope data.
            for default in list(node.args.defaults) + list(node.args.kw_defaults):
                if default is not None:
                    self.visit(default)

        self.scopes.append(Scope(callable_id=callable_id, params=set(params)))
        for stmt in getattr(node, "body", []):
            self.visit(stmt)
        self.scopes.pop()

        self.callable_stack.pop()
        self.current_callable = prev_callable

    def visit_Module(self, node: ast.Module) -> None:
        prev_callable = self.current_callable
        self.current_callable = self.module_callable
        self.callable_stack.append(self.module_callable)
        self.scopes.append(Scope(callable_id=self.module_callable, params=set()))
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()
        self.callable_stack.pop()
        self.current_callable = prev_callable

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev = self.current_class
        self.current_class = node.name
        for stmt in node.body:
            self.visit(stmt)
        self.current_class = prev

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_callable(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_callable(node, node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Lambdas do not have call-graph nodes in the existing extractor.
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            # "Direct" returned names mark containers that escape this callable.
            for name in self._direct_returned_names(node.value):
                self._expose_local_by_name(name, node, operation="return")
            returned = self._resolve_expr(node.value)
            if returned and self.current_callable:
                self._record_lineage(
                    returned.object_id,
                    _return_slot_id(self.current_callable),
                    RELATION_RETURN_VALUE if not returned.derived else RELATION_DERIVED_FROM,
                    node,
                )
                # Functions can have multiple return paths; keep the most
                # informative summary instead of letting the last one win.
                self.return_summaries[self.current_callable] = _merge_return_summary(
                    self.return_summaries.get(self.current_callable),
                    ExprRef(
                        returned.object_id,
                        returned.inferred_type,
                        returned.confidence,
                        returned.display_name,
                        returned.access_path,
                    ),
                )
            tuple_summary = self._resolve_unpack_value_refs(node.value)
            if tuple_summary and self.current_callable:
                for idx, tuple_ref in enumerate(tuple_summary):
                    if tuple_ref is None or not tuple_ref.object_id:
                        continue
                    self._record_lineage(
                        tuple_ref.object_id,
                        _return_slot_id(self.current_callable, idx),
                        RELATION_RETURN_SLOT if not tuple_ref.derived else RELATION_DERIVED_FROM,
                        node,
                        slot=str(idx),
                    )
                self.return_tuple_summaries[self.current_callable] = _merge_return_tuple_summary(
                    self.return_tuple_summaries.get(self.current_callable),
                    tuple_summary,
                )
            self.visit(node.value)

    def _direct_returned_names(self, node: ast.AST) -> Set[str]:
        # This is intentionally narrower than ast.walk(): we only want names
        # that indicate a container escapes, not every scalar dependency that
        # happens to appear inside the returned expression.
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, ast.Attribute):
            return self._direct_returned_names(node.value)
        if isinstance(node, ast.Subscript):
            return self._direct_returned_names(node.value)
        if isinstance(node, ast.Call):
            return set()
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            names: Set[str] = set()
            for element in node.elts:
                names.update(self._direct_returned_names(element))
            return names
        if isinstance(node, ast.Dict):
            names: Set[str] = set()
            for value in node.values:
                if value is not None:
                    names.update(self._direct_returned_names(value))
            return names
        return set()

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        iter_ref = self._resolve_expr(node.iter)
        element_class_types = self._element_class_types_from_value(node.iter)
        element_state_owners = self._element_shared_state_owner_types_from_value(node.iter)
        self.visit(node.iter)
        if iter_ref:
            iter_binding: Optional[LocalBinding] = None
            if (
                isinstance(node.iter, ast.Name)
                and self.scope is not None
                and node.iter.id in self.scope.locals
            ):
                iter_binding = self.scope.locals[node.iter.id]
            for name in self._target_names(node.target):
                if self.scope is not None:
                    alias_of = ""
                    object_id = iter_ref.object_id
                    exposed = True
                    iter_path = iter_ref.access_path or iter_ref.display_name
                    element_path = f"{iter_path}[]" if iter_path else name
                    if iter_binding and not iter_binding.exposed and not iter_binding.alias_of:
                        object_id = ""
                        exposed = False
                    else:
                        alias_of = iter_ref.object_id
                    self.scope.locals[name] = LocalBinding(
                        object_id=object_id,
                        inferred_type=FAMILY_OBJECT,
                        confidence="medium",
                        alias_of=alias_of,
                        display_name=element_path,
                        access_path=element_path,
                        exposed=exposed,
                        class_types=set(element_class_types),
                    )
                    if element_class_types:
                        self.scope.local_class_types[name] = set(element_class_types)
                    else:
                        self.scope.local_class_types.pop(name, None)
                    if element_state_owners:
                        self.scope.local_shared_state_owner_types[name] = set(element_state_owners)
                    else:
                        self.scope.local_shared_state_owner_types.pop(name, None)
        elif element_class_types and self.scope is not None:
            for name in self._target_names(node.target):
                self.scope.local_class_types[name] = set(element_class_types)
                if element_state_owners:
                    self.scope.local_shared_state_owner_types[name] = set(element_state_owners)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            context = item.context_expr
            if isinstance(context, ast.Call) and self._is_open_call(context):
                file_ref, access = self._file_from_open_call(context)
                if file_ref:
                    edge_access = "create" if access == "write" else "read"
                    self._record_access(
                        file_ref.object_id,
                        edge_access,
                        "open",
                        context,
                        file_ref.confidence,
                    )
                    if isinstance(item.optional_vars, ast.Name) and self.scope:
                        self.scope.locals[item.optional_vars.id] = LocalBinding(
                            object_id=file_ref.object_id,
                            inferred_type=FAMILY_FILE,
                            confidence=file_ref.confidence,
                            alias_of=file_ref.object_id,
                            display_name=file_ref.display_name,
                            access_path=file_ref.access_path or file_ref.display_name,
                        )
            else:
                self.visit(context)
                if isinstance(item.optional_vars, ast.Name) and self.scope:
                    class_types = self._class_types_from_value(context)
                    state_owners = self._shared_state_owner_types_from_value(context)
                    if class_types:
                        self.scope.local_class_types[item.optional_vars.id] = set(class_types)
                    else:
                        self.scope.local_class_types.pop(item.optional_vars.id, None)
                    if state_owners:
                        self.scope.local_shared_state_owner_types[item.optional_vars.id] = set(state_owners)
                    else:
                        self.scope.local_shared_state_owner_types.pop(item.optional_vars.id, None)
            if item.optional_vars is not None and not isinstance(item.optional_vars, ast.Name):
                self._handle_store_target(item.optional_vars, item.context_expr, "create")
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._handle_store_target(target, node.value, "create")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._handle_store_target(node.target, node.value, "create")
        else:
            self._handle_store_target(node.target, node.annotation, "write")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._handle_store_target(node.target, node.value, "read_write")

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._handle_store_target(target, target, "write", operation="delete")

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load) or self.current_callable is None:
            return
        if self.scope and node.id in self.scope.locals and not self.scope.locals[node.id].exposed:
            return
        ref = self._resolve_name(node.id, node)
        if ref:
            self._record_access(ref.object_id, "read", "load", node, ref.confidence)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.ctx, ast.Load) or self.current_callable is None:
            return
        if node.attr in PANDAS_INDEXER_ATTRS | {"index", "columns"}:
            self.visit(node.value)
            return
        if isinstance(getattr(node, "parent", None), ast.Attribute):
            parent = getattr(node, "parent")
            if getattr(parent, "value", None) is node:
                return
        ref = self._resolve_attribute(node)
        if ref:
            self._record_access(ref.object_id, "read", "attribute_load", node, ref.confidence)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Load) and self.current_callable is not None:
            refs = self._resolve_subscript_fields(node)
            for ref in refs:
                self._record_access(ref.object_id, "read", "subscript_load", node, ref.confidence)
            self.visit(node.value)
            self._visit_dynamic_slice(node.slice)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._handle_file_call(node)
        labeled_access_recorded = self._handle_xarray_labeled_call(node)
        self._handle_mutating_call(node)
        self._handle_method_receiver_read(node, suppress_receiver_read=labeled_access_recorded)
        self._record_confirmed_param_lineage(node)
        self._record_registered_state_lineage(node)
        known_callees = [
            callable_id
            for callable_id in self._candidate_callable_ids_for_call(node)
            if callable_id in self.callable_map
            or callable_id in self.callable_params
            or callable_id in self.return_summaries
        ]
        for arg in node.args:
            if known_callees:
                for name in self._direct_returned_names(arg):
                    self._expose_local_by_name(name, arg, operation="passed_arg")
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.value is not None:
                if known_callees:
                    for name in self._direct_returned_names(keyword.value):
                        self._expose_local_by_name(name, keyword.value, operation="passed_kwarg")
                self.visit(keyword.value)
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)
        else:
            self.visit(node.func)

    def _visit_comprehension(self, node: ast.AST, result_nodes: Sequence[ast.AST]) -> None:
        if self.scope is None:
            self.generic_visit(node)
            return
        # Comprehension targets shadow outer names; model that explicitly so we
        # do not accidentally treat loop variables as data dependencies.
        original_shadowed = set(self.scope.shadowed)
        try:
            generators = getattr(node, "generators", [])
            for generator in generators:
                self.visit(generator.iter)
                self.scope.shadowed.update(self._target_names(generator.target))
                for condition in generator.ifs:
                    self.visit(condition)
            for result_node in result_nodes:
                self.visit(result_node)
        finally:
            self.scope.shadowed = original_shadowed

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, [node.key, node.value])

    def _handle_method_receiver_read(self, node: ast.Call, *, suppress_receiver_read: bool = False) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        method = node.func.attr
        if suppress_receiver_read and method in XARRAY_LABEL_METHODS:
            return
        if method in MUTATING_METHODS:
            return
        if method in PANDAS_INPLACE_METHODS and _keyword_truthy(node, "inplace"):
            return
        if method in FILE_WRITE_METHODS:
            return
        receiver = self._resolve_receiver_expr(node.func.value)
        if receiver:
            self._record_access(
                receiver.object_id,
                "read",
                f"method:{method}:receiver",
                node,
                receiver.confidence,
            )

    def _handle_xarray_labeled_call(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr not in XARRAY_LABEL_METHODS:
            return False
        receiver = self._resolve_expr(node.func.value)
        if not receiver or receiver.inferred_type != FAMILY_XARRAY:
            return False
        fields, confidence = _xarray_call_fields(node)
        recorded = False
        for field in fields:
            ref = self._field_ref(receiver, field, node, kind="dict_key", confidence=confidence)
            self._record_access(
                ref.object_id,
                "read",
                f"method:{node.func.attr}:labeled_access",
                node,
                ref.confidence,
            )
            recorded = True
        return recorded

    def _resolve_expr(self, node: ast.AST) -> Optional[ExprRef]:
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id, node)
        if isinstance(node, ast.Attribute):
            return self._resolve_attribute(node)
        if isinstance(node, ast.Subscript):
            refs = self._resolve_subscript_fields(node)
            if refs:
                return refs[0]
            if isinstance(node.value, ast.Attribute) and node.value.attr in PANDAS_INDEXER_ATTRS:
                return self._resolve_expr(node.value.value)
            return self._resolve_expr(node.value)
        if isinstance(node, ast.Call):
            if self._is_open_call(node):
                file_ref, _access = self._file_from_open_call(node)
                return file_ref
            returned = self._return_ref_from_call(node)
            if returned:
                return returned
            if _is_copy_call(node):
                # Still the base object: a shallow copy shares the *values* the
                # field resolution below is after, and the family is unchanged.
                # But the container itself is a new object, so anything asking
                # about identity has to be told.
                return _as_derived(self._resolve_expr(node.func.value))
        return None

    def _resolve_receiver_expr(self, node: ast.AST) -> Optional[ExprRef]:
        if self._expr_uses_hidden_local(node):
            return None
        return self._resolve_expr(node)

    def _expr_uses_hidden_local(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            if self.scope and node.id in self.scope.locals:
                binding = self.scope.locals[node.id]
                return not binding.exposed and not binding.alias_of
            return False
        if isinstance(node, ast.Attribute):
            return self._expr_uses_hidden_local(node.value)
        if isinstance(node, ast.Subscript):
            return self._expr_uses_hidden_local(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return self._expr_uses_hidden_local(node.func.value)
        return False

    def _target_causes_escape(self, target: ast.AST) -> bool:
        if isinstance(target, ast.Attribute):
            path = _attribute_path(target)
            if path and path.startswith(("self.", "cls.")):
                return True
            return bool(self._resolve_receiver_expr(target.value))
        if isinstance(target, ast.Subscript):
            base = self._resolve_receiver_expr(target.value)
            return base is not None
        return False

    def _expose_escaping_locals_in_value(self, value: object, node: ast.AST, operation: str) -> None:
        if not isinstance(value, ast.AST):
            return
        for name in self._direct_returned_names(value):
            self._expose_local_by_name(name, node, operation=operation)

    def _resolve_name(self, name: str, node: ast.AST) -> Optional[ExprRef]:
        if self.scope and name in self.scope.shadowed:
            return None
        if self.scope and name in self.scope.locals:
            binding = self.scope.locals[name]
            object_id = binding.alias_of or binding.object_id
            if not object_id:
                return None
            return ExprRef(
                object_id,
                binding.inferred_type,
                binding.confidence,
                binding.display_name or name,
                binding.access_path or binding.display_name or name,
            )
        if self.scope and name in self.scope.params:
            return self._param_ref(name, node)
        if name in self.module_globals:
            return self._global_ref(name, node)
        return None

    def _resolve_attribute(self, node: ast.Attribute) -> Optional[ExprRef]:
        path = _attribute_path(node)
        if not path:
            return None
        if path.startswith("self.") or path.startswith("cls."):
            attr_path = path.split(".", 1)[1]
            if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
                return self._class_attr_ref(attr_path, node)
            base = self._resolve_expr(node.value)
            if base:
                if (
                    base.object_id.startswith("class_state:")
                    and (base.access_path or base.display_name) not in self.attrdict_access_paths
                ):
                    return base
                return self._object_attr_ref(base, node.attr, node, attr_path=path)
            return None
        if isinstance(node.value, ast.Name):
            base = self._resolve_name(node.value.id, node.value)
            if base:
                return self._object_attr_ref(base, node.attr, node, attr_path=path)
        if isinstance(node.value, (ast.Attribute, ast.Subscript, ast.Call)):
            base = self._resolve_expr(node.value)
            if base:
                return self._object_attr_ref(base, node.attr, node, attr_path=path)
        return None

    def _resolve_subscript_fields(self, node: ast.Subscript) -> List[ExprRef]:
        if isinstance(node.value, ast.Attribute) and node.value.attr in PANDAS_INDEXER_ATTRS:
            base = self._resolve_expr(node.value.value)
            if not base:
                return []
            if base.inferred_type == FAMILY_XARRAY and node.value.attr in XARRAY_INDEXER_ATTRS:
                fields, confidence = _xarray_indexer_fields(_slice_value(node))
                return [
                    self._field_ref(base, field, node, kind="dict_key", confidence=confidence)
                    for field in fields
                ]
            fields, confidence = _indexer_fields(_slice_value(node))
            confidence = "low" if node.value.attr in {"iloc", "iat"} and confidence != "high" else confidence
            return [
                self._field_ref(base, field, node, kind=self._field_kind(base), confidence=confidence)
                for field in fields
            ]

        base = self._resolve_expr(node.value)
        if not base:
            return []
        fields, confidence = _indexer_fields(_slice_value(node))
        return [
            self._field_ref(base, field, node, confidence=confidence)
            for field in fields
        ]

    def _visit_dynamic_slice(self, slice_node: ast.AST) -> None:
        if isinstance(slice_node, ast.Constant):
            return
        if isinstance(slice_node, (ast.List, ast.Tuple, ast.Set)):
            for element in slice_node.elts:
                if not isinstance(element, ast.Constant):
                    self.visit(element)
            return
        self.visit(slice_node)

    def _handle_store_target(
        self,
        target: ast.AST,
        value: object,
        access: str,
        operation: str = "assign",
    ) -> None:
        class_types = self._class_types_from_value(value)
        element_class_types = self._element_class_types_from_value(value)
        state_owners = self._shared_state_owner_types_from_value(value)
        element_state_owners = self._element_shared_state_owner_types_from_value(value)
        if isinstance(target, ast.Name):
            if self.scope is not None:
                if class_types:
                    self.scope.local_class_types[target.id] = set(class_types)
                else:
                    self.scope.local_class_types.pop(target.id, None)
                if element_class_types:
                    self.scope.local_element_class_types[target.id] = set(element_class_types)
                else:
                    self.scope.local_element_class_types.pop(target.id, None)
                if state_owners:
                    self.scope.local_shared_state_owner_types[target.id] = set(state_owners)
                else:
                    self.scope.local_shared_state_owner_types.pop(target.id, None)
                if element_state_owners:
                    self.scope.local_element_shared_state_owner_types[target.id] = set(element_state_owners)
                else:
                    self.scope.local_element_shared_state_owner_types.pop(target.id, None)
            origin = self._infer_type_from_value(value)
            inferred_type, alias_of, confidence = (
                origin.inferred_type,
                origin.alias_of,
                origin.confidence,
            )
            if inferred_type in CONTAINER_TYPES or origin.source_id:
                # Deliberately ``source_id`` rather than ``alias_of``: a derived
                # value is still a real object worth materialising, with real
                # accesses on it. Gating exposure on the alias would delete the
                # object along with the false identity claim, and the Step 1a
                # recall figure would fall while Step 1b's precision rose.
                expose = bool(origin.source_id)
                ref = self._local_ref(
                    target.id,
                    target,
                    inferred_type=inferred_type,
                    confidence=confidence if confidence != "low" else "medium",
                    alias_of=alias_of,
                    alias_display_name=self._alias_display_from_value(value, alias_of),
                    expose=expose,
                    class_types=class_types,
                )
                if self._value_is_attrdict_constructor(value):
                    self._mark_attrdict_access_path(ref.access_path)
                if expose:
                    relation = (
                        RELATION_LOCAL_ASSIGN if origin.identity else RELATION_DERIVED_FROM
                    )
                    if origin.source_id:
                        self._record_lineage(origin.source_id, ref.object_id, relation, target)
                    for source_id, identity in self._lineage_source_ids_from_value(value):
                        if source_id != origin.source_id:
                            self._record_lineage(
                                source_id,
                                ref.object_id,
                                RELATION_LOCAL_ASSIGN if identity else RELATION_DERIVED_FROM,
                                target,
                            )
                    self._record_access(
                        ref.object_id,
                        self._store_access_for_object(ref.object_id, access),
                        operation,
                        target,
                        ref.confidence,
                    )
            elif self.scope and target.id in self.scope.locals:
                del self.scope.locals[target.id]
            return

        if isinstance(target, ast.Attribute):
            attr_path = _attribute_path(target)
            if self.scope is not None and attr_path:
                if class_types:
                    self.scope.attr_class_types[attr_path] = set(class_types)
                else:
                    self.scope.attr_class_types.pop(attr_path, None)
                if state_owners:
                    self.scope.attr_shared_state_owner_types[attr_path] = set(state_owners)
                else:
                    self.scope.attr_shared_state_owner_types.pop(attr_path, None)
            if self._target_causes_escape(target):
                self._expose_escaping_locals_in_value(value, target, operation="escape_assign")
            ref = self._resolve_attribute(target)
            if ref:
                origin = self._infer_type_from_value(value)
                inferred_type, alias_of, confidence = (
                    origin.inferred_type,
                    origin.alias_of,
                    origin.confidence,
                )
                if (
                    attr_path
                    and self.scope
                    and attr_path.startswith(("self.", "cls."))
                    and (inferred_type in CONTAINER_TYPES or origin.source_id)
                ):
                    self.scope.attr_bindings[attr_path] = LocalBinding(
                        object_id=ref.object_id,
                        inferred_type=inferred_type,
                        confidence=confidence if confidence != "low" else "medium",
                        alias_of=alias_of,
                        display_name=self._alias_display_from_value(value, alias_of) or target.attr,
                        access_path=self._alias_display_from_value(value, alias_of) or attr_path,
                        class_types=set(class_types),
                    )
                if self._value_is_attrdict_constructor(value):
                    self._mark_attrdict_access_path(ref.access_path or attr_path or "")
                if (
                    inferred_type
                    and ref.object_id in self.objects
                    and not _is_state_object_id(ref.object_id)
                ):
                    self.objects[ref.object_id].inferred_type = inferred_type
                if origin.source_id and ref.object_id != origin.source_id:
                    self._record_lineage(
                        origin.source_id,
                        ref.object_id,
                        RELATION_STATE_ASSIGN if origin.identity else RELATION_DERIVED_FROM,
                        target,
                    )
                for source_id, identity in self._lineage_source_ids_from_value(value):
                    if source_id != origin.source_id and source_id != ref.object_id:
                        self._record_lineage(
                            source_id,
                            ref.object_id,
                            RELATION_STATE_ASSIGN if identity else RELATION_DERIVED_FROM,
                            target,
                        )
                self._record_access(
                    ref.object_id,
                    self._store_access_for_object(ref.object_id, access),
                    operation,
                    target,
                    _confidence_max(ref.confidence, confidence),
                )
            return

        if isinstance(target, ast.Subscript):
            if self._target_causes_escape(target):
                self._expose_escaping_locals_in_value(value, target, operation="escape_assign")
            refs = self._resolve_subscript_fields(target)
            edge_access = "read_write" if access == "read_write" else "write"
            for ref in refs:
                self._record_access(ref.object_id, edge_access, operation, target, ref.confidence)
            self.visit(target.value)
            self._visit_dynamic_slice(target.slice)
            return

        if isinstance(target, (ast.Tuple, ast.List)):
            unpacked_refs = self._resolve_unpack_value_refs(value) if isinstance(value, ast.AST) else None
            if unpacked_refs and len(unpacked_refs) == len(target.elts):
                for idx, (element, element_ref) in enumerate(zip(target.elts, unpacked_refs)):
                    target_object_id = self._target_object_id_for_lineage(element)
                    if element_ref is not None and target_object_id:
                        self._record_lineage(
                            element_ref.object_id,
                            target_object_id,
                            RELATION_TUPLE_UNPACK if not element_ref.derived else RELATION_DERIVED_FROM,
                            element,
                            slot=str(idx),
                        )
                    if target_object_id:
                        for source_id, identity in self._lineage_source_ids_from_value(value, slot=idx):
                            if element_ref is None or source_id != element_ref.object_id:
                                self._record_lineage(
                                    source_id,
                                    target_object_id,
                                    RELATION_TUPLE_UNPACK if identity else RELATION_DERIVED_FROM,
                                    element,
                                    slot=str(idx),
                                )
                    self._handle_store_target(element, element_ref if element_ref is not None else value, access, operation)
                return
            for element in target.elts:
                self._handle_store_target(element, value, access, operation)

    def _handle_mutating_call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        method = node.func.attr
        receiver = self._resolve_receiver_expr(node.func.value)
        if not receiver:
            return
        if method in MUTATING_METHODS:
            self._record_access(
                receiver.object_id,
                "read_write",
                f"method:{method}",
                node,
                receiver.confidence,
            )
        elif method in PANDAS_INPLACE_METHODS and _keyword_truthy(node, "inplace"):
            self._record_access(
                receiver.object_id,
                "read_write",
                f"method:{method}:inplace",
                node,
                receiver.confidence,
            )
        elif method in FILE_WRITE_METHODS:
            if node.args:
                file_ref = self._file_ref(node.args[0], node)
                self._record_access(file_ref.object_id, "write", f"method:{method}", node, file_ref.confidence)
            self._record_access(
                receiver.object_id,
                "read",
                f"method:{method}:receiver",
                node,
                receiver.confidence,
            )

    def _handle_file_call(self, node: ast.Call) -> None:
        call_name = _attribute_path(node.func) or ""
        if self._is_open_call(node):
            file_ref, access = self._file_from_open_call(node)
            if file_ref:
                edge_access = "create" if access == "write" else "read"
                self._record_access(file_ref.object_id, edge_access, "open", node, file_ref.confidence)
            return

        if call_name.endswith(tuple(FILE_READ_FUNCS)) and node.args:
            file_ref = self._file_ref(node.args[0], node)
            self._record_access(file_ref.object_id, "read", call_name, node, file_ref.confidence)
            return

        if _call_name_matches(call_name, XARRAY_OPEN_FUNCS):
            file_expr = self._file_arg_from_call(node, ("filename_or_obj", "path", "path_or_file"))
            if file_expr is not None:
                file_ref = self._file_ref(file_expr, node)
                self._record_access(file_ref.object_id, "read", call_name, node, file_ref.confidence)
            return

        if _call_name_matches(call_name, POOCH_READ_FUNCS):
            file_expr = self._file_arg_from_call(node, ("url", "fname", "path"))
            if file_expr is not None:
                file_ref = self._file_ref(file_expr, node)
                self._record_access(file_ref.object_id, "read", call_name, node, file_ref.confidence)
            return

        if call_name.endswith("json.load") and node.args:
            file_ref = self._resolve_expr(node.args[0])
            if file_ref and file_ref.inferred_type == "file":
                self._record_access(file_ref.object_id, "read", "json.load", node, file_ref.confidence)
            return

        if call_name.endswith("json.dump") and len(node.args) >= 2:
            file_ref = self._resolve_expr(node.args[1])
            if file_ref and file_ref.inferred_type == "file":
                self._record_access(file_ref.object_id, "write", "json.dump", node, file_ref.confidence)

    def _file_arg_from_call(self, node: ast.Call, keyword_names: Sequence[str]) -> Optional[ast.AST]:
        if node.args:
            return node.args[0]
        for keyword in node.keywords:
            if keyword.arg in keyword_names:
                return keyword.value
        return None

    def _is_open_call(self, node: ast.Call) -> bool:
        call_name = _attribute_path(node.func) or ""
        return call_name in {"open", "builtins.open"} or call_name.endswith(".open")

    def _file_from_open_call(self, node: ast.Call) -> Tuple[Optional[ExprRef], str]:
        if not node.args:
            return None, "read"
        mode = "r"
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        for keyword in node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = str(keyword.value.value)
        return self._file_ref(node.args[0], node), _mode_to_access(mode)

    def _target_names(self, target: ast.AST) -> List[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            names: List[str] = []
            for element in target.elts:
                names.extend(self._target_names(element))
            return names
        return []

    def _target_object_id_for_lineage(self, target: ast.AST) -> str:
        if isinstance(target, ast.Name):
            callable_id = self.current_callable or ""
            return f"local_exposed:{callable_id}:{target.id}"
        if isinstance(target, ast.Attribute):
            ref = self._resolve_attribute(target)
            return ref.object_id if ref else ""
        return ""

    def _names_in_expr(self, node: ast.AST) -> Set[str]:
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names

    def _expose_local_by_name(self, name: str, node: ast.AST, operation: str) -> None:
        if self.scope is None or name not in self.scope.locals:
            return
        self._ensure_local_exposed(name, node)
        binding = self.scope.locals[name]
        if binding.object_id.startswith("local_exposed:"):
            self._record_access(
                binding.object_id,
                "read",
                operation,
                node,
                binding.confidence,
            )


def collect_data_access_from_tree(
    tree: ast.Module,
    module: str,
    file: Path,
    callable_map: Optional[Dict[str, CallableDef]] = None,
    return_summaries: Optional[Dict[str, ExprRef]] = None,
    return_tuple_summaries: Optional[Dict[str, Tuple[Optional[ExprRef], ...]]] = None,
    callable_params: Optional[Dict[str, Tuple[str, ...]]] = None,
    callable_return_identity: Optional[Dict[str, bool]] = None,
    param_bindings: Optional[Dict[str, Set[str]]] = None,
    param_access_paths: Optional[Dict[str, Set[str]]] = None,
    lineage_edges: Optional[List[LineageEdge]] = None,
    pyright_families: Optional[Dict[str, str]] = None,
    attrdict_classes: Optional[Set[str]] = None,
    registration_rules: Optional[Dict[str, RegistrationRule]] = None,
    project_index: Optional[ProjectIndex] = None,
    split_class_owners: Optional[Set[str]] = None,
    known_classes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, DataObject], List[AccessEdge], List[LineageEdge]]:
    attach_parents(tree)
    if callable_map is None:
        callable_map = {}
    module_imports, star_imports = collect_module_imports(
        tree,
        module,
        current_is_package=is_package_file(file),
    )
    # ``None`` means "this caller has only one file", which is the single-source
    # path: there, per-file and project-wide are the same set.
    resolved_split_class_owners = (
        split_class_owners
        if split_class_owners is not None
        else infer_split_class_owners(tree, module, pyright_families)
    )
    resolved_attrdict_classes = (
        attrdict_classes
        if attrdict_classes is not None
        else collect_attrdict_classes(tree, module, file)
    )
    collector = DataAccessCollector(
        module=module,
        file=file,
        callable_map=callable_map,
        module_globals=collect_module_globals(tree),
        module_imports=module_imports,
        star_imports=star_imports,
        return_summaries=return_summaries,
        return_tuple_summaries=return_tuple_summaries,
        callable_params=callable_params,
        callable_return_identity=callable_return_identity,
        param_bindings=param_bindings,
        param_access_paths=param_access_paths,
        lineage_edges=lineage_edges,
        split_class_owners=resolved_split_class_owners,
        pyright_families=pyright_families,
        attrdict_classes=resolved_attrdict_classes,
        registration_rules=registration_rules,
        project_index=project_index,
        known_classes=known_classes,
    )
    collector.visit(tree)
    return collector.objects, collector.edges, collector.lineage_edges


def collect_data_access_from_source(
    source: str,
    module: str = "sample",
    filename: str = "sample.py",
    shared_containers_config: Optional[Dict[str, object]] = None,
    pyright_families: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, DataObject], List[AccessEdge]]:
    configure_shared_field_containers(shared_containers_config)
    pyright_families = pyright_families if pyright_families is not None else {}
    return_summaries: Dict[str, ExprRef] = {}
    return_tuple_summaries: Dict[str, Tuple[Optional[ExprRef], ...]] = {}
    callable_params: Dict[str, Tuple[str, ...]] = {}
    callable_return_identity: Dict[str, bool] = {}
    file = Path(filename)
    attrdict_classes = collect_attrdict_classes(parse_python_source(source, filename=filename), module, file)
    # Iterate to a small fixpoint so transitive return aliases like A -> B -> C
    # can stabilize without relying on source-order.
    for _ in range(MAX_RETURN_SUMMARY_PASSES):
        before = _return_summary_snapshot(return_summaries)
        before_tuple = _return_tuple_summary_snapshot(return_tuple_summaries)
        tree = parse_python_source(source, filename=filename)
        collect_data_access_from_tree(
            tree,
            module=module,
            file=file,
            return_summaries=return_summaries,
            return_tuple_summaries=return_tuple_summaries,
            callable_params=callable_params,
            callable_return_identity=callable_return_identity,
            pyright_families=pyright_families,
            attrdict_classes=attrdict_classes,
        )
        if (
            _return_summary_snapshot(return_summaries) == before
            and _return_tuple_summary_snapshot(return_tuple_summaries) == before_tuple
        ):
            break
    else:
        warnings.warn(
            f"data-access return summaries did not converge within "
            f"MAX_RETURN_SUMMARY_PASSES={MAX_RETURN_SUMMARY_PASSES}; "
            f"some objects and access edges may be missing",
            stacklevel=2,
        )

    tree = parse_python_source(source, filename=filename)
    param_bindings: Dict[str, Set[str]] = {}
    param_access_paths: Dict[str, Set[str]] = {}
    lineage_edges: List[LineageEdge] = []
    module_lineage_edges: List[LineageEdge] = []
    objects, edges, module_lineage_edges = collect_data_access_from_tree(
        tree,
        module=module,
        file=file,
        return_summaries=return_summaries,
        return_tuple_summaries=return_tuple_summaries,
        callable_params=callable_params,
        callable_return_identity=callable_return_identity,
        param_bindings=param_bindings,
        param_access_paths=param_access_paths,
        lineage_edges=module_lineage_edges,
        pyright_families=pyright_families,
        attrdict_classes=attrdict_classes,
    )
    lineage_edges.extend(module_lineage_edges)
    _apply_confirmed_param_aliases(objects, param_bindings)
    _apply_confirmed_param_access_paths(objects, param_access_paths)
    _apply_lineage_aliases(objects, lineage_edges)
    return objects, edges


def collect_data_access_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    callable_map: Dict[str, CallableDef],
    *,
    project_root: Optional[Path] = None,
    pyright_bin: str = "pyright",
    pyright_families: Optional[Dict[str, str]] = None,
    registration_rules: Optional[Dict[str, RegistrationRule]] = None,
    project_index: Optional[ProjectIndex] = None,
    cache: Optional[ParsedFileCache] = None,
) -> Tuple[Dict[str, DataObject], List[AccessEdge], List[LineageEdge]]:
    objects: Dict[str, DataObject] = {}
    edges: List[AccessEdge] = []
    lineage_edges: List[LineageEdge] = []
    analysis_files = list(analysis_files)
    if not analysis_files:
        return objects, edges, lineage_edges
    # One tree per file for the whole stage. The caller normally hands in the
    # cache the call-graph passes already populated -- by the time
    # ``analyze_analysis_files`` returns it holds every file, parsed once, with
    # ``attach_parents`` applied -- but a local one still collapses the four to
    # ten parses per file this stage used to perform down to one.
    resolved_cache = cache if cache is not None else ParsedFileCache()
    resolved_pyright_families = (
        pyright_families
        if pyright_families is not None
        else collect_pyright_families_from_analysis_files(
            project_root or discover_project_root(analysis_files[0].path),
            analysis_files,
            pyright_bin,
            cache=resolved_cache,
        )
    )
    return_summaries: Dict[str, ExprRef] = {}
    return_tuple_summaries: Dict[str, Tuple[Optional[ExprRef], ...]] = {}
    callable_params: Dict[str, Tuple[str, ...]] = {}
    callable_return_identity: Dict[str, bool] = {}
    attrdict_classes = collect_attrdict_classes_from_analysis_files(
        analysis_files, cache=resolved_cache
    )
    # Two whole-project facts settled before any pass runs. Both used to be
    # recomputed inside every collector, which meant once per file per pass --
    # and in the split-owner case the per-file answers disagreed with each
    # other, so a class's object IDs depended on which file was being walked.
    split_class_owners = infer_split_class_owners_for_analysis_files(
        analysis_files, resolved_pyright_families, cache=resolved_cache
    )
    known_classes = known_class_ids_from_callable_map(callable_map)

    # Iterate to a small fixpoint so return summaries can propagate through
    # shallow call chains even when files are not processed in dependency order.
    #
    # ``registration_rules`` and ``project_index`` are threaded here as well as
    # into the final pass below. They used to be omitted, which meant the
    # fixpoint converged on a *weaker* analysis than the one that actually
    # produces the artifacts: with the project index in hand the final pass
    # resolves re-exported callables the fixpoint could not, discovers new return
    # summaries while it runs, and hands them only to the files it happens to
    # reach afterwards. That made two objects on climlab exist or not exist
    # depending on file order.
    for _ in range(MAX_RETURN_SUMMARY_PASSES):
        before = _return_summary_snapshot(return_summaries)
        before_tuple = _return_tuple_summary_snapshot(return_tuple_summaries)
        for analysis_file in analysis_files:
            py_file = analysis_file.path
            module = analysis_file.module
            tree = resolved_cache.get(py_file)
            collect_data_access_from_tree(
                tree=tree,
                module=module,
                file=py_file,
                callable_map=callable_map,
                return_summaries=return_summaries,
                return_tuple_summaries=return_tuple_summaries,
                callable_params=callable_params,
                callable_return_identity=callable_return_identity,
                pyright_families=resolved_pyright_families,
                attrdict_classes=attrdict_classes,
                registration_rules=registration_rules,
                project_index=project_index,
                split_class_owners=split_class_owners,
                known_classes=known_classes,
            )
        if (
            _return_summary_snapshot(return_summaries) == before
            and _return_tuple_summary_snapshot(return_tuple_summaries) == before_tuple
        ):
            break
    else:
        # The cap stopped the loop rather than convergence, so some return
        # summaries may still be one hop short. Silence here is what let an
        # incomplete result look like a settled one -- same reasoning, and same
        # wording, as ``call_graph.passes``.
        warnings.warn(
            f"data-access return summaries did not converge within "
            f"MAX_RETURN_SUMMARY_PASSES={MAX_RETURN_SUMMARY_PASSES}; "
            f"some objects and access edges may be missing",
            stacklevel=2,
        )

    param_bindings: Dict[str, Set[str]] = {}
    param_access_paths: Dict[str, Set[str]] = {}
    kind_conflicts: Set[str] = set()
    for analysis_file in analysis_files:
        py_file = analysis_file.path
        module = analysis_file.module
        tree = resolved_cache.get(py_file)
        module_lineage_edges: List[LineageEdge] = []
        module_objects, module_edges, module_lineage_edges = collect_data_access_from_tree(
            tree=tree,
            module=module,
            file=py_file,
            callable_map=callable_map,
            return_summaries=return_summaries,
            return_tuple_summaries=return_tuple_summaries,
            callable_params=callable_params,
            callable_return_identity=callable_return_identity,
            param_bindings=param_bindings,
            param_access_paths=param_access_paths,
            lineage_edges=module_lineage_edges,
            pyright_families=resolved_pyright_families,
            attrdict_classes=attrdict_classes,
            registration_rules=registration_rules,
            project_index=project_index,
            split_class_owners=split_class_owners,
            known_classes=known_classes,
        )
        for object_id, data_object in module_objects.items():
            if object_id not in objects:
                objects[object_id] = data_object
            elif merge_data_object(objects[object_id], data_object):
                kind_conflicts.add(object_id)
        edges.extend(module_edges)
        lineage_edges.extend(module_lineage_edges)

    if kind_conflicts:
        # Two files described one ID with two different real kinds. The merge
        # resolves it the same way every run, but the disagreement itself is a
        # defect signal -- §4.6's argument that a placeholder count is better
        # read as a defect counter than as a legitimate object kind.
        warnings.warn(
            f"{len(kind_conflicts)} data object(s) were described with conflicting "
            f"kinds by different files, e.g. {sorted(kind_conflicts)[0]}",
            stacklevel=2,
        )

    _apply_confirmed_param_aliases(objects, param_bindings)
    _apply_confirmed_param_access_paths(objects, param_access_paths)
    _apply_lineage_aliases(objects, lineage_edges)
    return objects, edges, lineage_edges


def collect_data_access(
    src_root: Path,
    callable_map: Dict[str, CallableDef],
    module_prefix: Optional[str] = None,
    pyright_bin: str = "pyright",
    pyright_families: Optional[Dict[str, str]] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
    registration_rules: Optional[Dict[str, RegistrationRule]] = None,
    project_index: Optional[ProjectIndex] = None,
    cache: Optional[ParsedFileCache] = None,
) -> Tuple[Dict[str, DataObject], List[AccessEdge], List[LineageEdge]]:
    analysis_files = list(
        iter_analysis_files(
            src_root,
            module_prefix=module_prefix,
            entrypoints=entrypoints,
            project_root=project_root,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        )
    )
    return collect_data_access_from_analysis_files(
        analysis_files,
        callable_map,
        project_root=project_root or discover_project_root(src_root),
        pyright_bin=pyright_bin,
        pyright_families=pyright_families,
        registration_rules=registration_rules,
        project_index=project_index,
        cache=cache,
    )


def _pyright_families_for_config(
    config: ExtractionConfig,
    analysis_files: Sequence[AnalysisFile],
    *,
    cache: Optional[ParsedFileCache] = None,
) -> Optional[Dict[str, str]]:
    if not config.data_access.pyright.enabled:
        return {}
    if config.data_access.pyright.fail_policy == "error":
        return None
    try:
        return collect_pyright_families_from_analysis_files(
            config.project_root,
            analysis_files,
            config.data_access.pyright.bin,
            cache=cache,
        )
    except RuntimeError as exc:
        print(f"Pyright failed; continuing with unknown families: {exc}", file=sys.stderr)
        return {}


def run_from_extraction_config(
    config: ExtractionConfig,
    *,
    outdir: Optional[Path] = None,
    analysis_files: Optional[Sequence[AnalysisFile]] = None,
) -> Tuple[int, int, int]:
    """Run the whole data-access stage for a config.

    ``analysis_files`` overrides file discovery. Only the determinism check uses
    it, to run this exact function twice with the file list in two different
    orders. It is a parameter rather than something that check reimplements
    because a paraphrase of this function would drift from it, and a determinism
    check that has quietly stopped testing the real pipeline reports success
    either way -- which is the failure mode the check exists to rule out.
    """
    shared_config = (
        load_shared_field_containers(config.data_access.shared_containers_config)
        if config.data_access.shared_containers_config
        else None
    )
    configure_shared_field_containers(shared_config)
    if analysis_files is None:
        analysis_files = iter_analysis_files_for_source_roots(
            config.source_roots,
            entrypoints=config.entrypoints,
            project_root=config.project_root,
            include_globs=config.include_globs,
            exclude_globs=config.exclude_globs,
        )
    analysis_files = list(analysis_files)
    # The call-graph passes are re-run here rather than read from that stage's
    # artifacts: registration lineage would otherwise be silently wrong whenever
    # edges.csv was stale, and neither stage would be runnable on its own.
    analysis = analyze_analysis_files(
        analysis_files, summary_packages=config.call_graph.summary_packages
    )
    callable_map = analysis.project_nodes()
    # ``analysis.cache`` already holds every file, parsed once, with parents
    # attached. Re-parsing here is not a missed optimization but a discarded
    # value: the cache is the first field of ``CallGraphAnalysis`` precisely
    # because this stage is its second consumer.
    pyright_families = _pyright_families_for_config(
        config, analysis_files, cache=analysis.cache
    )
    objects, edges, lineage_edges = collect_data_access_from_analysis_files(
        analysis_files,
        callable_map,
        project_root=config.project_root,
        pyright_bin=config.data_access.pyright.bin,
        pyright_families=pyright_families,
        registration_rules=analysis.registration_rules,
        project_index=analysis.project_index,
        cache=analysis.cache,
    )
    output_dir = (outdir or config.data_access.outdir).resolve()
    write_outputs(output_dir, callable_map, objects, edges, lineage_edges)
    return len(callable_map), len(objects), len(edges)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an AST-based data-access view")
    parser.add_argument("--config", default=None, type=Path, help="Optional extraction JSON/JSONC config")
    parser.add_argument("--root", default=None, help="Python source root (e.g., src/shop)")
    parser.add_argument(
        "--module-prefix",
        default=None,
        help="Prefix to prepend to discovered module names (e.g., shop when root is src/shop)",
    )
    parser.add_argument(
        "--entrypoint",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional Python file to include as an explicit entrypoint. "
            "May be passed more than once; module names are project-relative."
        ),
    )
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument(
        "--pyright-bin",
        default="pyright",
        help="Pyright binary or command name. Required for authoritative family inference.",
    )
    parser.add_argument(
        "--shared-containers-config",
        default=None,
        help=(
            "Optional JSON or JSONC file defining shared field containers explicitly. "
            "See the packaged shared_containers.jsonc starter template."
        ),
    )
    parser.add_argument(
        "--no-pyright",
        action="store_true",
        help="Skip Pyright probing and keep unavailable container families unknown.",
    )
    parser.add_argument(
        "--pyright-fail-policy",
        choices=["fallback_unknown", "error"],
        default=None,
        help="Pyright failure behavior when --config is used.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.config:
        config = load_extraction_config(args.config)
        if args.no_pyright or args.pyright_fail_policy or args.shared_containers_config:
            config = ExtractionConfig(
                project_root=config.project_root,
                source_roots=config.source_roots,
                package_prefixes=config.package_prefixes,
                entrypoints=config.entrypoints,
                include_globs=config.include_globs,
                exclude_globs=config.exclude_globs,
                call_graph=config.call_graph,
                data_access=DataAccessExtractionConfig(
                    outdir=config.data_access.outdir,
                    pyright=PyrightConfig(
                        enabled=False if args.no_pyright else config.data_access.pyright.enabled,
                        bin=config.data_access.pyright.bin,
                        fail_policy=args.pyright_fail_policy or config.data_access.pyright.fail_policy,
                    ),
                    shared_containers_config=(
                        Path(args.shared_containers_config).resolve()
                        if args.shared_containers_config
                        else config.data_access.shared_containers_config
                    ),
                    inferred_shared_containers_config=config.data_access.inferred_shared_containers_config,
                    inferred_shared_containers_report=config.data_access.inferred_shared_containers_report,
                ),
            )
        callable_count, object_count, edge_count = run_from_extraction_config(
            config,
            outdir=Path(args.outdir).resolve() if args.outdir else None,
        )
        output_dir = Path(args.outdir).resolve() if args.outdir else config.data_access.outdir
        print(f"Data access view generated in {output_dir}")
        print(f"Callables: {callable_count}")
        print(f"Data objects: {object_count}")
        print(f"Access edges: {edge_count}")
        return

    if not args.root:
        raise SystemExit("--root is required unless --config is provided")
    root = Path(args.root).resolve()
    outdir = Path(args.outdir or "artifacts/data_access").resolve()
    entrypoints = tuple(Path(path).resolve() for path in args.entrypoint)
    shared_config = (
        load_shared_field_containers(Path(args.shared_containers_config).resolve())
        if args.shared_containers_config
        else None
    )
    configure_shared_field_containers(shared_config)

    # Same passes as the config path, so ``--root`` is not quietly a
    # lesser analysis that drops registration lineage.
    analysis = analyze_analysis_files(
        list(
            iter_analysis_files(
                root,
                module_prefix=args.module_prefix,
                entrypoints=entrypoints,
            )
        )
    )
    callable_map = analysis.project_nodes()
    objects, edges, lineage_edges = collect_data_access(
        src_root=root,
        callable_map=callable_map,
        module_prefix=args.module_prefix,
        pyright_bin=args.pyright_bin,
        pyright_families={} if args.no_pyright else None,
        entrypoints=entrypoints,
        registration_rules=analysis.registration_rules,
        project_index=analysis.project_index,
        cache=analysis.cache,
    )
    write_outputs(outdir, callable_map, objects, edges, lineage_edges)

    print(f"Data access view generated in {outdir}")
    print(f"Callables: {len(callable_map)}")
    print(f"Data objects: {len(objects)}")
    print(f"Access edges: {len(edges)}")


if __name__ == "__main__":
    main()
