#!/usr/bin/env python3
"""Generate an AST-based call graph for a Python project.

An abstract syntax tree (AST) is Python's structured representation of source
code. For example, ``service.run()`` becomes a call node whose function is an
attribute node. Walking that tree lets this package discover likely call
relationships without importing or executing the project.

High-level pipeline
-------------------

The analyzer makes several passes over the same source files:

1. **Index definitions.** ``DefinitionCollector`` records modules, functions,
   methods, classes, inheritance, decorators, and imports.
2. **Summarize returns.** ``ReturnSummaryCollector`` learns simple facts such as
   "this function returns an ``Order``" or "this function returns a collection
   of ``Order`` objects."
3. **Propagate types.** ``TypeSummaryCollector`` carries those facts through
   call arguments and instance attributes. It repeats until no new facts are
   found (or a small safety limit is reached).
4. **Collect edges.** ``CallCollector`` revisits calls and Python operations,
   resolving as many caller-to-callee relationships as the accumulated facts
   allow.
5. **Write artifacts.** ``call_graph_outputs.write_outputs`` emits CSV and JSON.

Multiple passes matter because useful information is often learned later than
it is used. If ``make_order()`` returns ``Order()``, then a later
``order = make_order(); order.submit()`` can be resolved only after the return
summary for ``make_order`` is available.

Where the code lives
--------------------

This module is the entry point and the stable import surface; the analysis
itself is split across the package, each module depending only on those above
it:

``models``
    Graph records, the module index, and inferred-type summaries.
``ast_utils``
    Parsing and small AST helpers.
``discovery``
    Choosing which files to analyze and naming their modules.
``definitions``
    ``DefinitionCollector`` and the definition-indexing pass.
``return_links``
    Deferred "this callable returns whatever that one returns" facts.
``dunders``
    Operator-to-dunder-method tables. Pure data.
``project_index``
    ``ProjectIndex``: the class hierarchy and the resolution that depends only
    on whole-project definition facts. Built once and shared by every pass.
``type_env``
    ``TypeEnv``: the scoped variable/container/attribute type state a single
    collector accumulates as it walks one file.
``collectors``
    ``CallCollector`` -- the AST visitor together with the type inference and
    callee resolution it drives. These stay in one module because they are
    mutually recursive: resolving ``build().submit()`` needs the inferred type
    of ``build()``, and inferring that type needs to resolve ``build``.
``summary_collectors``
    ``ReturnSummaryCollector`` and ``TypeSummaryCollector``, the two edge-free
    passes built on ``CallCollector``.
``passes``
    Drivers that run the collectors over a file set, plus ``SourceCallResolver``.

Outputs:
- nodes.csv: callable definitions discovered in scanned files
- edges.csv: caller -> callee edges with resolution metadata
- call_graph.json: combined nodes/edges payload

This extractor is static and conservative:
- It resolves direct function calls, class/static-like calls, and self/cls method calls.
- It attempts simple type inference for ``obj = ClassName(...)`` assignments.
- It does not execute code and cannot fully resolve dynamic dispatch.

Important terminology
---------------------

``callable ID``
    A project-wide name such as ``shop.orders.Order.submit``. Module-level
    execution is represented by the synthetic suffix ``<module>``.
``resolved edge``
    The callee matches a callable definition discovered in the analyzed files.
``unresolved edge``
    The source names a possible callee, but the analyzer cannot prove which
    local definition it refers to. These can optionally be kept to show calls
    into libraries or dynamic code.
``relation``
    A short explanation of how an edge was inferred, such as ``direct``,
    ``constructor``, ``property_getter``, or ``dunder_operator``.
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Re-exported below so that ``from ...generate_call_graph_ast import X`` keeps
# working for the nine production modules and three test modules that import
# from here. New code should prefer importing from the specific module instead.
from .ast_utils import (
    ParsedFileCache,
    attach_parents,
    attribute_to_name,
    parse_python_file,
    parse_python_source,
)
from .definitions import (
    DefinitionCollector,
    build_indices,
    build_indices_from_analysis_files,
    merge_module_index,
)
from .discovery import (
    infer_project_root,
    iter_analysis_files,
    iter_analysis_files_for_source_roots,
    iter_python_files,
    iter_summary_only_files,
    module_callable_id,
    module_for_analysis_file,
    path_to_module,
)
from .collectors import CallCollector
from .summary_collectors import ReturnSummaryCollector, TypeSummaryCollector
from .models import (
    MODULE_CALLABLE_QUALNAME,
    AnalysisFile,
    CallGraphHealth,
    CallableDef,
    Edge,
    FunctionEscapeSummary,
    FunctionParamSummary,
    FunctionReturnSummary,
    ModuleIndex,
    RegistryFacts,
)
from .project_index import ProjectIndex
from .registration import (
    RegistrationRule,
    build_registration_rules,
    registration_child_exprs,
    registration_slot,
)
from .passes import (
    SourceCallResolver,
    TypeSummaries,
    build_return_summaries,
    build_return_summaries_from_analysis_files,
    build_type_summaries,
    build_type_summaries_from_analysis_files,
    collect_edges,
    collect_edges_from_analysis_files,
)

try:
    from microservice_pipeline.call_graph.call_graph_outputs import write_outputs
    from microservice_pipeline.config import ExtractionConfig, load_extraction_config
except ImportError:  # pragma: no cover - supports direct script execution
    from call_graph_outputs import write_outputs  # type: ignore
    from config import ExtractionConfig, load_extraction_config  # type: ignore


__all__ = [
    # Models
    "AnalysisFile",
    "CallableDef",
    "Edge",
    "FunctionEscapeSummary",
    "FunctionParamSummary",
    "FunctionReturnSummary",
    "ModuleIndex",
    "RegistryFacts",
    "MODULE_CALLABLE_QUALNAME",
    # AST helpers
    "ParsedFileCache",
    "attach_parents",
    "attribute_to_name",
    "parse_python_file",
    "parse_python_source",
    # Discovery
    "infer_project_root",
    "iter_analysis_files",
    "iter_analysis_files_for_source_roots",
    "iter_python_files",
    "iter_summary_only_files",
    "module_callable_id",
    "module_for_analysis_file",
    "path_to_module",
    # Collectors
    "CallCollector",
    "DefinitionCollector",
    "ReturnSummaryCollector",
    "SourceCallResolver",
    "TypeSummaryCollector",
    # Whole-project facts
    "ProjectIndex",
    # Registry coupling
    "RegistrationRule",
    "build_registration_rules",
    "registration_child_exprs",
    "registration_slot",
    # Passes
    "TypeSummaries",
    "build_indices",
    "build_indices_from_analysis_files",
    "build_return_summaries",
    "build_return_summaries_from_analysis_files",
    "build_type_summaries",
    "build_type_summaries_from_analysis_files",
    "collect_edges",
    "collect_edges_from_analysis_files",
    "merge_module_index",
    # Orchestration and CLI
    "CallGraphAnalysis",
    "analyze_analysis_files",
    "build_call_graph_from_analysis_files",
    "main",
    "parse_args",
    "run_from_extraction_config",
]


@dataclass(frozen=True)
class CallGraphAnalysis:
    """Everything the analysis passes learn, before any graph is emitted.

    Split out from graph construction because the facts have a second consumer:
    the data-access stage derives its registration lineage from
    ``registration_rules``, and needs ``project_index`` to resolve a call onto
    them. Recomputing here rather than reading an artifact keeps the two stages
    independently runnable and makes a stale hand-off impossible.
    """
    cache: ParsedFileCache
    nodes: Dict[str, CallableDef]
    module_map: Dict[str, ModuleIndex]
    known_classes: Dict[str, str]
    project_index: ProjectIndex
    return_summaries: Dict[str, FunctionReturnSummary]
    summaries: TypeSummaries
    registration_rules: Dict[str, RegistrationRule]
    summary_only_modules: Set[str]

    def project_nodes(self) -> Dict[str, CallableDef]:
        """``nodes`` without the summary-only packages.

        Those definitions are indexed so calls into a framework can resolve, but
        they belong to someone else's codebase. Anything emitted -- graph nodes,
        the data-access callable list -- must use this rather than ``nodes``.
        """
        if not self.summary_only_modules:
            return self.nodes
        return {
            node_id: node
            for node_id, node in self.nodes.items()
            if node.module not in self.summary_only_modules
        }


def analyze_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    *,
    summary_packages: Sequence[str] = (),
) -> CallGraphAnalysis:
    """Run every fact-gathering pass over an already selected set of files.

    One ``ParsedFileCache`` is shared by every pass, so each file is read and
    parsed exactly once instead of once per pass per loop iteration.

    One ``ProjectIndex`` is shared the same way. It holds the whole-project class
    hierarchy and callable facts, which the definition pass has already settled
    by this point and which no later pass changes -- so building it once here
    replaces one rebuild per collector, of which there is one per file per
    fixpoint iteration.

    ``summary_packages`` names installed third-party packages to read for
    registry evidence only. Their definitions join the index so a call into them
    can resolve, but callers must exclude them from anything they emit; the
    module names are returned for exactly that.
    """
    cache = ParsedFileCache()
    summary_only_files = list(iter_summary_only_files(summary_packages))
    every_file = [*analysis_files, *summary_only_files]

    nodes, module_map, known_classes = build_indices_from_analysis_files(
        every_file,
        cache=cache,
    )
    project_index = ProjectIndex(module_map, known_classes, set(nodes.keys()))
    return_summaries = build_return_summaries_from_analysis_files(
        analysis_files,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        cache=cache,
        project_index=project_index,
    )
    summaries = build_type_summaries_from_analysis_files(
        analysis_files,
        callable_map=nodes,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
        cache=cache,
        project_index=project_index,
        summary_only_files=summary_only_files,
    )
    return CallGraphAnalysis(
        cache=cache,
        nodes=nodes,
        module_map=module_map,
        known_classes=known_classes,
        project_index=project_index,
        return_summaries=return_summaries,
        summaries=summaries,
        registration_rules=build_registration_rules(
            summaries.escapes, summaries.registry, project_index
        ),
        summary_only_modules={analysis.module for analysis in summary_only_files},
    )


def build_call_graph_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    *,
    include_external: bool = False,
    package_prefix: Optional[str] | Sequence[str] = None,
    summary_packages: Sequence[str] = (),
    analysis: Optional[CallGraphAnalysis] = None,
    health: Optional[CallGraphHealth] = None,
) -> Tuple[Dict[str, CallableDef], List[Edge]]:
    """Turn the analysis facts into graph nodes and edges.

    ``analysis`` lets a caller that already ran ``analyze_analysis_files`` reuse
    it rather than pay for the passes twice.
    """
    analysis = analysis or analyze_analysis_files(
        analysis_files, summary_packages=summary_packages
    )
    nodes = analysis.nodes
    summaries = analysis.summaries

    edges = collect_edges_from_analysis_files(
        analysis_files,
        callable_map=nodes,
        module_map=analysis.module_map,
        known_classes=analysis.known_classes,
        include_external=include_external,
        package_prefix=package_prefix,
        return_summaries=analysis.return_summaries,
        param_summaries=summaries.params,
        class_attr_types=summaries.class_attrs,
        cache=analysis.cache,
        project_index=analysis.project_index,
        registry_facts=summaries.registry,
        registration_rules=analysis.registration_rules,
        health=health,
    )
    # Definitions from the summary-only packages were indexed so calls into them
    # could resolve; they are not part of this project and must not appear in the
    # graph. Edges are filtered as well as nodes: indexing a framework makes
    # calls into it resolvable that used to be external, and an edge pointing at
    # a node that does not exist is worse than the unresolved call it replaced.
    if analysis.summary_only_modules:
        nodes = analysis.project_nodes()
        edges = [
            edge
            for edge in edges
            if edge.caller in nodes and (edge.callee in nodes or not edge.resolved)
        ]
    return nodes, edges


# Relations that can only ever name a callable inside the analyzed project. An
# unresolved edge carrying one of these is a resolver failure on visible code,
# unlike an unresolved ``imported`` or ``direct``, which is usually just numpy.
INTRA_PROJECT_RELATIONS = frozenset(
    {"self_method", "super_method", "virtual_override", "inferred_type"}
)


def summarize_health(health: CallGraphHealth, *, top_n: int = 4) -> List[str]:
    """Console lines describing what the resolver could not do.

    Replaces ``Edges: N (resolved: N, unresolved: 0)``. That line looked like a
    perfect score and was a tautology: with ``include_external`` off, an
    unresolved edge is dropped before it can be counted, so the second number was
    structurally always zero. These counts are taken where the loss happens.

    Intra-project drops are called out separately from third-party ones because
    only the first kind is a defect -- an unresolved ``numpy`` call is the
    configuration working, an unresolved ``self_method`` is the analyzer failing
    on code it can see.
    """
    lines: List[str] = []
    unresolved = health.dropped_unresolved
    if unresolved:
        total = sum(unresolved.values())
        top = ", ".join(
            f"{relation} {count}" for relation, count in unresolved.most_common(top_n)
        )
        lines.append(f"Unresolved calls dropped: {total} ({top})")
        internal = {
            relation: count
            for relation, count in unresolved.items()
            if relation in INTRA_PROJECT_RELATIONS
        }
        if internal:
            detail = ", ".join(
                f"{relation} {count}" for relation, count in sorted(internal.items())
            )
            lines.append(
                f"  of which intra-project (a resolver gap, not third-party): {detail}"
            )
    if health.dropped_external:
        lines.append(
            f"External calls dropped by package prefix: "
            f"{sum(health.dropped_external.values())}"
        )
    if health.unresolvable_calls:
        shapes = ", ".join(
            f"{shape} {count}"
            for shape, count in health.unresolvable_calls.most_common(top_n)
        )
        lines.append(
            f"Calls with no candidate at all: "
            f"{sum(health.unresolvable_calls.values())} ({shapes})"
        )
    sites = sum(health.site_fanout.values())
    if sites:
        ambiguous = sum(count for k, count in health.site_fanout.items() if k > 1)
        mean = sum(k * n for k, n in health.site_fanout.items()) / sites
        lines.append(
            f"Resolved call sites: {sites} "
            f"({ambiguous} with more than one target, mean fan-out {mean:.2f})"
        )
    return lines


def run_from_extraction_config(
    config: ExtractionConfig, *, outdir: Optional[Path] = None
) -> Tuple[int, int, CallGraphHealth]:
    """Build/write a configured graph and return node/edge counts plus health."""
    analysis_files = iter_analysis_files_for_source_roots(
        config.source_roots,
        entrypoints=config.entrypoints,
        project_root=config.project_root,
        include_globs=config.include_globs,
        exclude_globs=config.exclude_globs,
    )
    health = CallGraphHealth()
    nodes, edges = build_call_graph_from_analysis_files(
        analysis_files,
        include_external=config.call_graph.include_external,
        package_prefix=config.package_prefixes,
        summary_packages=config.call_graph.summary_packages,
        health=health,
    )
    output_dir = (outdir or config.call_graph.outdir).resolve()
    write_outputs(outdir=output_dir, nodes=nodes, edges=edges, health=health)
    return len(nodes), len(edges), health


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Define the command-line interface for config-driven or direct scans."""
    parser = argparse.ArgumentParser(description="Generate an AST-based call graph")
    parser.add_argument("--config", default=None, type=Path, help="Optional extraction JSON/JSONC config")
    parser.add_argument("--root", default=None, help="Python source root (e.g., src/shop)")
    parser.add_argument("--package", default=None, help="Package prefix for filtering internal calls")
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include unresolved/external callee nodes in edge list",
    )
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: select inputs, run all passes, write files, print totals."""
    args = parse_args(argv)
    if args.config:
        config = load_extraction_config(args.config)
        node_count, edge_count, health = run_from_extraction_config(
            config,
            outdir=Path(args.outdir).resolve() if args.outdir else None,
        )
        output_dir = Path(args.outdir).resolve() if args.outdir else config.call_graph.outdir
        print(f"Call graph generated in {output_dir}")
        print(f"Nodes: {node_count}")
        print(f"Edges: {edge_count}")
        for line in summarize_health(health):
            print(line)
        return

    if not args.root:
        raise SystemExit("--root is required unless --config is provided")
    root = Path(args.root).resolve()
    outdir = Path(args.outdir or "artifacts/call_graph").resolve()
    entrypoints = tuple(Path(path).resolve() for path in args.entrypoint)

    # Enumerate the file set once. Calling the four ``src_root`` wrappers
    # separately would re-walk the filesystem for each pass and give them no way
    # to share a parse cache.
    analysis_files = list(
        iter_analysis_files(
            root,
            module_prefix=args.module_prefix,
            entrypoints=entrypoints,
        )
    )
    health = CallGraphHealth()
    nodes, edges = build_call_graph_from_analysis_files(
        analysis_files,
        include_external=args.include_external,
        package_prefix=args.package,
        health=health,
    )

    write_outputs(outdir=outdir, nodes=nodes, edges=edges, health=health)

    print(f"Call graph generated in {outdir}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    for line in summarize_health(health):
        print(line)


if __name__ == "__main__":
    main()
