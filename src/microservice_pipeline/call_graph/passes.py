"""Pass drivers: run the collectors over a file set and accumulate their facts.

Each ``build_*``/``collect_*`` function here owns one stage of the pipeline. The
``*_from_analysis_files`` form takes an already-selected file list and is what
the orchestrator calls; the thin ``src_root`` wrapper beside it enumerates files
itself and exists for callers that only have a directory.

The two summary stages are fixed-point loops: they re-run until a pass produces
nothing new, because a fact learned about one callable can unlock a fact about
another. ``SourceCallResolver`` sits at the end -- it applies the same machinery
to in-memory snippets such as notebook cells, which have no file on disk.
"""


from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .ast_utils import ParsedFileCache, attach_parents, parse_python_source
from .collector import CallCollector
from .summary_collectors import ReturnSummaryCollector, TypeSummaryCollector
from .definitions import DefinitionCollector, merge_module_index
from .discovery import iter_analysis_files
from .models import (
    AnalysisFile,
    CallGraphHealth,
    ClassAttrTypes,
    CallableDef,
    Edge,
    FunctionEscapeSummary,
    FunctionParamSummary,
    FunctionReturnSummary,
    ModuleIndex,
    RegistryFacts,
    copy_class_attr_types,
    copy_escape_summaries,
    copy_param_summaries,
    copy_registry_facts,
    resolvable_callable_ids,
)
from .project_index import ProjectIndex
from .registration import RegistrationRule
from .return_links import ReturnLinkTable, resolve_return_links


@dataclass(frozen=True)
class TypeSummaries:
    """Everything the type-propagation fixed point learns.

    Grouped rather than returned as a four-tuple because the registry facts
    joined out of ``escapes`` and ``registry`` travel with the type facts through
    every caller, and widening the tuple would have meant editing each one.
    """
    params: Dict[str, FunctionParamSummary] = field(default_factory=dict)
    class_attrs: ClassAttrTypes = field(default_factory=ClassAttrTypes)
    escapes: Dict[str, FunctionEscapeSummary] = field(default_factory=dict)
    registry: RegistryFacts = field(default_factory=RegistryFacts)


def build_return_summaries_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    max_iterations: int = 3,
    *,
    cache: Optional[ParsedFileCache] = None,
    project_index: Optional[ProjectIndex] = None,
) -> Dict[str, FunctionReturnSummary]:
    """Infer return types repeatedly so facts can cross call chains.

    Each pass reads the previous generation of summaries and writes a fresh one,
    so a pass on its own advances a return chain by exactly one hop. That is what
    ``resolve_return_links`` is for: the collectors record every "we return
    whatever that callee returns" dependency they meet, and chasing those links
    afterwards carries facts along a chain of any depth without re-reading a
    single file.

    The link chase runs inside the loop, before the convergence test, so the next
    pass starts from the richer types and can uncover dependencies the previous
    one could not see. Links accumulate across passes and are never cleared --
    which callee a call resolves to depends on inferred types, so a later pass
    can find links an earlier one missed.
    """
    cache = cache if cache is not None else ParsedFileCache()
    summaries: Dict[str, FunctionReturnSummary] = {}
    links = ReturnLinkTable()
    callable_ids = resolvable_callable_ids(callable_map)
    project_index = project_index or ProjectIndex(
        module_map, known_classes, callable_ids
    )

    for _ in range(max_iterations):
        collected: Dict[str, FunctionReturnSummary] = {}

        for analysis_file in analysis_files:
            py_file = analysis_file.path
            module = analysis_file.module
            tree = cache.get(py_file)
            collector = ReturnSummaryCollector(
                module=module,
                file=py_file,
                module_index=module_map[module],
                callable_ids=callable_ids,
                module_map=module_map,
                known_classes=known_classes,
                return_summaries=summaries,
                return_links=links,
                project_index=project_index,
            )
            collector.visit(tree)
            collected.update(collector.collected_summaries)

        resolve_return_links(collected, links)

        if collected == summaries:
            break
        summaries = collected
    else:
        # The cap stopped the loop rather than convergence, so some facts may be
        # missing. Silence here is what made an incomplete graph look complete.
        warnings.warn(
            f"return summaries did not converge within max_iterations="
            f"{max_iterations}; some call edges may be unresolved",
            stacklevel=2,
        )

    return summaries


def build_return_summaries(
    src_root: Path,
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
    max_iterations: int = 3,
) -> Dict[str, FunctionReturnSummary]:
    return build_return_summaries_from_analysis_files(
        list(
            iter_analysis_files(
                src_root,
                module_prefix=module_prefix,
                entrypoints=entrypoints,
                project_root=project_root,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        ),
        callable_map=callable_map,
        module_map=module_map,
        known_classes=known_classes,
        max_iterations=max_iterations,
    )


def build_type_summaries_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    return_summaries: Dict[str, FunctionReturnSummary],
    max_iterations: int = 5,
    *,
    cache: Optional[ParsedFileCache] = None,
    project_index: Optional[ProjectIndex] = None,
    summary_only_files: Sequence[AnalysisFile] = (),
) -> TypeSummaries:
    """Propagate parameter and class-attribute types until facts stabilize.

    ``summary_only_files`` are read for their facts and nothing else. A framework
    installed as a third-party package defines the method that registers a child
    -- ``nn.Module.add_module`` and friends -- so without reading it the escape
    that makes a call site a registration is invisible, and generalising past
    codebases that define their own wiring method would be impossible. Those
    files never reach edge collection, so no external callable can become a node.
    """
    cache = cache if cache is not None else ParsedFileCache()
    summaries = TypeSummaries()
    callable_ids = resolvable_callable_ids(callable_map)
    project_index = project_index or ProjectIndex(
        module_map, known_classes, callable_ids
    )
    every_file = [*analysis_files, *summary_only_files]

    for _ in range(max_iterations):
        pending = TypeSummaries(
            params=copy_param_summaries(summaries.params),
            class_attrs=copy_class_attr_types(summaries.class_attrs),
            escapes=copy_escape_summaries(summaries.escapes),
            registry=copy_registry_facts(summaries.registry),
        )

        for analysis_file in every_file:
            py_file = analysis_file.path
            module = analysis_file.module
            tree = cache.get(py_file)
            collector = TypeSummaryCollector(
                module=module,
                file=py_file,
                module_index=module_map[module],
                callable_ids=callable_ids,
                module_map=module_map,
                known_classes=known_classes,
                return_summaries=return_summaries,
                param_summaries=pending.params,
                class_attr_types=pending.class_attrs,
                project_index=project_index,
                escape_summaries=pending.escapes,
                registry_facts=pending.registry,
            )
            collector.visit(tree)

        if pending == summaries:
            break
        summaries = pending
    else:
        # As above: report a truncated result rather than passing it off as a
        # settled one.
        warnings.warn(
            f"type summaries did not converge within max_iterations="
            f"{max_iterations}; some call edges may be unresolved",
            stacklevel=2,
        )

    return summaries


def build_type_summaries(
    src_root: Path,
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    return_summaries: Dict[str, FunctionReturnSummary],
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
    max_iterations: int = 5,
) -> TypeSummaries:
    return build_type_summaries_from_analysis_files(
        list(
            iter_analysis_files(
                src_root,
                module_prefix=module_prefix,
                entrypoints=entrypoints,
                project_root=project_root,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        ),
        callable_map=callable_map,
        module_map=module_map,
        known_classes=known_classes,
        return_summaries=return_summaries,
        max_iterations=max_iterations,
    )


def collect_edges_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    include_external: bool,
    package_prefix: Optional[str] | Sequence[str],
    return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
    param_summaries: Optional[Dict[str, FunctionParamSummary]] = None,
    class_attr_types: Optional[ClassAttrTypes] = None,
    *,
    cache: Optional[ParsedFileCache] = None,
    project_index: Optional[ProjectIndex] = None,
    registry_facts: Optional[RegistryFacts] = None,
    registration_rules: Optional[Dict[str, RegistrationRule]] = None,
    health: Optional[CallGraphHealth] = None,
) -> List[Edge]:
    """Run the final, fully informed edge-collection pass over every file.

    ``health`` is an optional accumulator rather than a second return value so
    the many existing callers keep working unchanged; pass one to learn what the
    pass could not resolve as well as what it could.
    """
    cache = cache if cache is not None else ParsedFileCache()
    edges: List[Edge] = []
    callable_ids = resolvable_callable_ids(callable_map)
    project_index = project_index or ProjectIndex(
        module_map, known_classes, callable_ids
    )

    for analysis_file in analysis_files:
        py_file = analysis_file.path
        module = analysis_file.module
        tree = cache.get(py_file)
        collector = CallCollector(
            module=module,
            file=py_file,
            module_index=module_map[module],
            callable_ids=callable_ids,
            module_map=module_map,
            known_classes=known_classes,
            include_external=include_external,
            package_prefix=package_prefix,
            return_summaries=return_summaries,
            param_summaries=param_summaries,
            class_attr_types=class_attr_types,
            project_index=project_index,
            registry_facts=registry_facts,
            registration_rules=registration_rules,
        )
        collector.visit(tree)
        edges.extend(collector.edges)
        if health is not None:
            health.merge(collector.health)

    return edges


def collect_edges(
    src_root: Path,
    callable_map: Dict[str, CallableDef],
    module_map: Dict[str, ModuleIndex],
    known_classes: Dict[str, str],
    include_external: bool,
    package_prefix: Optional[str] | Sequence[str],
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
    return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
    param_summaries: Optional[Dict[str, FunctionParamSummary]] = None,
    class_attr_types: Optional[ClassAttrTypes] = None,
    registry_facts: Optional[RegistryFacts] = None,
    registration_rules: Optional[Dict[str, RegistrationRule]] = None,
) -> List[Edge]:
    return collect_edges_from_analysis_files(
        list(
            iter_analysis_files(
                src_root,
                module_prefix=module_prefix,
                entrypoints=entrypoints,
                project_root=project_root,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        ),
        callable_map=callable_map,
        module_map=module_map,
        known_classes=known_classes,
        include_external=include_external,
        package_prefix=package_prefix,
        return_summaries=return_summaries,
        param_summaries=param_summaries,
        class_attr_types=class_attr_types,
        registry_facts=registry_facts,
        registration_rules=registration_rules,
    )


class SourceCallResolver:
    """Resolve calls from in-memory source snippets using call-graph semantics.

    This is intended for notebook cells and other non-file entrypoints. It keeps
    module-scope import and simple variable-type state across calls to
    ``resolve_source`` so sequential notebook cells can resolve aliases and
    instance method calls in the same way as a Python module.
    """

    def __init__(
        self,
        module: str,
        file: Path,
        callable_map: Dict[str, CallableDef],
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        include_external: bool,
        package_prefix: Optional[str],
        return_summaries: Optional[Dict[str, FunctionReturnSummary]] = None,
        param_summaries: Optional[Dict[str, FunctionParamSummary]] = None,
        class_attr_types: Optional[ClassAttrTypes] = None,
    ):
        self.module = module
        self.file = file
        self.module_index = ModuleIndex(module=module, path=file)
        self.module_map = dict(module_map)
        self.module_map[module] = self.module_index
        self.collector = CallCollector(
            module=module,
            file=file,
            module_index=self.module_index,
            callable_ids=resolvable_callable_ids(callable_map),
            module_map=self.module_map,
            known_classes=known_classes,
            include_external=include_external,
            package_prefix=package_prefix,
            return_summaries=return_summaries,
            param_summaries=param_summaries,
            class_attr_types=class_attr_types,
        )
        self.collector.current_callable = self.collector.module_callable
        self.collector.callable_stack.append(self.collector.module_callable)
        self.collector.types.push_scope()

    def resolve_source(self, source: str, filename: str | None = None) -> List[Edge]:
        """Analyze one snippet and return only edges newly produced by it.

        Imports and simple variable types remain in the collector, which makes a
        sequence of notebook cells behave like successive statements in a file.
        """
        filename = filename or str(self.file)
        tree = parse_python_source(source, filename=filename)
        attach_parents(tree)

        defs = DefinitionCollector(module=self.module, file=Path(filename))
        defs.visit(tree)
        merge_module_index(self.module_index, defs.module_index)

        self.collector.file = Path(filename)
        self.collector.module_index = self.module_index
        before = len(self.collector.edges)
        for statement in tree.body:
            self.collector.visit(statement)
        return self.collector.edges[before:]
