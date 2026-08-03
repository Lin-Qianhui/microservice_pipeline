"""The definition-indexing pass: what exists, and where.

``DefinitionCollector`` runs before any call resolution. It answers the
questions every later pass depends on -- which callables exist, which of them
are methods, what each class inherits from, which names a module imported and
what they point at. Nothing here tries to resolve a call; it only builds the
index that makes resolution possible.

Decorators are matched by their source-level dotted name rather than by
importing the decorator object, because importing would execute analyzed code.
"""


from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .ast_utils import ParsedFileCache, attribute_to_name
from .discovery import iter_analysis_files, module_callable_id
from .models import MODULE_CALLABLE_QUALNAME, AnalysisFile, CallableDef, ModuleIndex

try:
    from microservice_pipeline.import_resolution import (
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from import_resolution import (  # type: ignore
        is_package_file,
        resolve_import_from_module,
        resolve_import_from_target,
    )


# Decorators are compared by their source-level dotted names. This intentionally
# avoids importing decorator objects, which would execute analyzed code.
PROPERTY_DECORATOR_NAMES = {
    "property",
    "cached_property",
    "functools.cached_property",
}

STATICMETHOD_DECORATOR_NAMES = {"staticmethod"}

CLASSMETHOD_DECORATOR_NAMES = {"classmethod"}


def class_body_expressions(node: ast.ClassDef) -> List[ast.AST]:
    """Expressions the ``class`` statement itself evaluates, not the methods.

    A class body is a code object that runs once, when the ``class`` statement
    executes. Everything it evaluates therefore belongs to the class, not to the
    method it is written next to -- most importantly a method's decorators and
    its *default arguments*, which is why ``def __init__(self, axes=make_axis())``
    calls ``make_axis`` at class-creation time and never again.

    Method bodies are deliberately excluded: those run later, on invocation, in
    their own scope.
    """
    expressions: List[ast.AST] = [*node.decorator_list, *node.bases]
    expressions += [keyword.value for keyword in node.keywords]
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            expressions += signature_expressions(statement)
        else:
            expressions.append(statement)
    return expressions


def signature_expressions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> List[ast.AST]:
    """The parts of a ``def`` evaluated where the ``def`` appears, not inside it.

    Decorators, default arguments, and annotations are all evaluated in the
    enclosing scope at definition time. Attributing them to the function being
    defined credits the callee with work its caller did, and disagrees with the
    interpreter about *when* the code runs.
    """
    expressions: List[ast.AST] = [*node.decorator_list, *node.args.defaults]
    expressions += [default for default in node.args.kw_defaults if default is not None]
    if node.returns is not None:
        expressions.append(node.returns)
    all_args = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        node.args.vararg,
        node.args.kwarg,
    ]
    expressions += [
        arg.annotation for arg in all_args if arg is not None and arg.annotation is not None
    ]
    return expressions


def class_body_evaluates_calls(node: ast.ClassDef) -> bool:
    """Whether a class body runs any call, i.e. whether it can own an edge.

    Gating the node on this keeps the graph free of isolated class nodes: on
    climlab only 8 of 77 classes qualify, so the node set grows by 8 rather than
    77 and nothing degree-based downstream shifts under a crowd of leaves.
    """
    return any(
        isinstance(inner, ast.Call)
        for expression in class_body_expressions(node)
        for inner in ast.walk(expression)
    )


class DefinitionCollector(ast.NodeVisitor):
    """First-pass AST visitor that indexes definitions and module-level names.

    No call edges are produced here. The collector establishes the universe of
    known callables/classes that later passes are allowed to resolve against.
    ``current_class`` and ``current_callable`` provide enough lexical context to
    construct IDs for methods and nested functions.
    """

    def __init__(self, module: str, file: Path):
        self.module = module
        self.file = file
        self.callables: List[CallableDef] = [
            CallableDef(
                id=module_callable_id(module),
                module=module,
                qualname=MODULE_CALLABLE_QUALNAME,
                file=str(file),
                lineno=1,
                kind="module",
                class_name="",
            )
        ]
        self.module_index = ModuleIndex(module=module, path=file)
        self.current_class: Optional[str] = None
        self.current_callable: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        """Record the local spelling introduced by each ``import`` statement."""
        for alias in node.names:
            asname = alias.asname or alias.name.split(".")[0]
            self.module_index.imports[asname] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record absolute targets for ``from`` imports, including relatives."""
        imported_module = resolve_import_from_module(
            self.module,
            node.module,
            node.level,
            current_is_package=is_package_file(self.file),
        )
        if imported_module is None:
            return
        for alias in node.names:
            if alias.name == "*":
                if imported_module:
                    self.module_index.star_imports.append(imported_module)
                continue
            asname = alias.asname or alias.name
            target = resolve_import_from_target(
                self.module,
                node.module,
                node.level,
                alias.name,
                current_is_package=is_package_file(self.file),
            )
            if target is not None:
                self.module_index.imports[asname] = target

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Index a class and the best statically resolvable IDs of its bases."""
        class_id = f"{self.module}.{node.name}"
        self.module_index.classes[node.name] = class_id
        bases = [
            base_id
            for base in node.bases
            if (base_id := self._resolve_class_reference(base)) is not None
        ]
        self.module_index.class_bases[class_id] = bases

        if class_body_evaluates_calls(node):
            self.module_index.class_bodies.add(class_id)
            self.callables.append(
                CallableDef(
                    id=class_id,
                    module=self.module,
                    qualname=node.name,
                    file=str(self.file),
                    lineno=node.lineno,
                    kind="class_body",
                    class_name=node.name,
                )
            )

        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def _resolve_class_reference(self, node: ast.AST) -> Optional[str]:
        name = attribute_to_name(node)
        if not name:
            return None

        if "." not in name:
            local_class = self.module_index.classes.get(name)
            if local_class:
                return local_class
            imported = self.module_index.imports.get(name)
            if imported:
                return imported
            return f"{self.module}.{name}"

        base, rest = name.split(".", 1)
        imported_base = self.module_index.imports.get(base)
        if imported_base:
            return f"{imported_base}.{rest}"
        return name

    def _add_callable(
        self, name: str, lineno: int, kind: str, class_name: Optional[str] = None
    ) -> str:
        """Create a stable callable ID appropriate to the current lexical scope."""
        if self.current_callable is not None:
            call_id = f"{self.current_callable}.<locals>.{name}"
        elif kind == "method" and class_name:
            call_id = f"{self.module}.{class_name}.{name}"
        else:
            call_id = f"{self.module}.{name}"

        qualname = call_id[len(self.module) + 1 :]
        self.callables.append(
            CallableDef(
                id=call_id,
                module=self.module,
                qualname=qualname,
                file=str(self.file),
                lineno=lineno,
                kind=kind,
                class_name=class_name or "",
            )
        )
        return call_id

    def _visit_callable_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> None:
        """Index a sync/async function and visit definitions nested inside it."""
        parent = getattr(node, "parent", None)
        prev_callable = self.current_callable

        if self.current_callable is not None:
            next_callable = self._add_callable(
                name=name,
                lineno=node.lineno,
                kind="function",
                class_name=self.current_class,
            )
        elif isinstance(parent, ast.ClassDef):
            next_callable = self._add_callable(
                name=name,
                lineno=node.lineno,
                kind="method",
                class_name=self.current_class,
            )
        else:
            next_callable = self._add_callable(
                name=name,
                lineno=node.lineno,
                kind="function",
                class_name=None,
            )

        if isinstance(parent, ast.ClassDef):
            if self._has_decorator(node, PROPERTY_DECORATOR_NAMES):
                self.module_index.properties.add(next_callable)
            if self._has_decorator(node, STATICMETHOD_DECORATOR_NAMES):
                self.module_index.static_methods.add(next_callable)
            if self._has_decorator(node, CLASSMETHOD_DECORATOR_NAMES):
                self.module_index.class_methods.add(next_callable)

        self.current_callable = next_callable
        self.generic_visit(node)
        self.current_callable = prev_callable

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Index a lambda under the name CPython gives its code object.

        ``<enclosing>.<locals>.<lambda>`` is exactly ``co_qualname``, so the
        runtime tracer and the static pass agree on the ID by construction.
        CPython does not disambiguate sibling lambdas in one scope, and neither
        does this: a disambiguator would buy precision at the price of that
        agreement, and agreement is what the whole comparison rests on.
        """
        if self.current_callable is not None:
            lambda_id = f"{self.current_callable}.<locals>.<lambda>"
        elif self.current_class:
            lambda_id = f"{self.module}.{self.current_class}.<locals>.<lambda>"
        else:
            lambda_id = f"{self.module}.<lambda>"

        if not any(existing.id == lambda_id for existing in self.callables):
            self.callables.append(
                CallableDef(
                    id=lambda_id,
                    module=self.module,
                    qualname=lambda_id[len(self.module) + 1 :],
                    file=str(self.file),
                    lineno=node.lineno,
                    kind="function",
                    class_name=self.current_class or "",
                )
            )

        prev_callable = self.current_callable
        self.current_callable = lambda_id
        self.generic_visit(node)
        self.current_callable = prev_callable

    def _has_decorator(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        decorator_names: Set[str],
    ) -> bool:
        for decorator in node.decorator_list:
            decorator_expr = decorator.func if isinstance(decorator, ast.Call) else decorator
            decorator_name = attribute_to_name(decorator_expr)
            if decorator_name in decorator_names:
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable_def(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable_def(node, node.name)


def build_indices_from_analysis_files(
    analysis_files: Sequence[AnalysisFile],
    *,
    cache: Optional[ParsedFileCache] = None,
) -> Tuple[Dict[str, CallableDef], Dict[str, ModuleIndex], Dict[str, str]]:
    """Run the definition pass and build the shared symbol indexes.

    Pass the ``cache`` shared by the rest of the run to avoid re-parsing these
    files in the later passes; omitting it parses them here and throws the
    trees away, which is fine for a caller that only wants the indexes.
    """
    cache = cache if cache is not None else ParsedFileCache()
    callable_map: Dict[str, CallableDef] = {}
    module_map: Dict[str, ModuleIndex] = {}
    known_classes: Dict[str, str] = {}

    for analysis_file in analysis_files:
        py_file = analysis_file.path
        module = analysis_file.module
        # The cache attaches parent links on first parse, so no attach_parents here.
        tree = cache.get(py_file)

        defs = DefinitionCollector(module=module, file=py_file)
        defs.visit(tree)

        module_map[module] = defs.module_index
        for c in defs.callables:
            callable_map[c.id] = c
        for _class_name, class_id in defs.module_index.classes.items():
            known_classes[class_id] = class_id

    add_reexport_class_aliases(known_classes, module_map)

    return callable_map, module_map, known_classes


def add_reexport_class_aliases(
    known_classes: Dict[str, str],
    module_map: Dict[str, ModuleIndex],
    max_iterations: int = 10,
) -> int:
    """Teach ``known_classes`` the names a class is *re-exported* under.

    A class reference is recorded as the symbol path the referring module used,
    not the path where the class is defined. So ``from climlab.process import
    TimeDependentProcess`` makes a subclass record its base as
    ``climlab.process.TimeDependentProcess``, while the class itself lives at
    ``climlab.process.time_dependent_process.TimeDependentProcess``. Those are
    different strings, and only the second one is a known class.

    That difference is expensive: method resolution walks base classes and stops
    at the first one it cannot recognize, so a class reached through a package
    re-export loses *every* inherited method call. On climlab that was 55 of 110
    missing edges.

    ``known_classes`` is the intended place to fix it -- it is consulted as an
    alias-to-canonical map and was simply never given any aliases. Filling it in
    here means no resolver code has to change.

    Iterates because re-exports chain: a top-level ``__init__.py`` re-exports
    from a subpackage ``__init__.py`` which re-exports from the defining module.
    Returns the number of aliases added, which is useful for diagnostics.
    """
    added_total = 0
    for _ in range(max_iterations):
        added = 0
        for module, module_index in module_map.items():
            for name, target in module_index.imports.items():
                alias = f"{module}.{name}"
                # A real class definition of this name always wins; an alias may
                # never shadow one, and an alias already recorded is left alone
                # so the loop stays monotone and terminates.
                if alias in known_classes:
                    continue
                canonical = known_classes.get(target)
                if canonical is None:
                    continue
                known_classes[alias] = canonical
                added += 1
        added_total += added
        if not added:
            break
    return added_total


def build_callable_aliases(
    callable_ids: Set[str],
    module_map: Dict[str, ModuleIndex],
    max_iterations: int = 10,
) -> Dict[str, str]:
    """Map the names a *callable* is re-exported under to where it is defined.

    The exact counterpart of ``add_reexport_class_aliases``, which solved this
    for classes only. ``from climlab.domain import column_state`` records the
    target as ``climlab.domain.column_state``, but the function is defined at
    ``climlab.domain.initial.column_state`` and only the second is a known
    callable -- so the call resolves to nothing and the edge is dropped.

    Iterates for the same reason: re-exports chain through nested ``__init__``
    files. A real definition always wins over an alias, which keeps the map
    monotone and the loop terminating.
    """
    aliases: Dict[str, str] = {}
    for _ in range(max_iterations):
        added = 0
        for module, module_index in module_map.items():
            for name, target in module_index.imports.items():
                alias = f"{module}.{name}"
                if alias in callable_ids or alias in aliases:
                    continue
                canonical = target if target in callable_ids else aliases.get(target)
                if canonical is None:
                    continue
                aliases[alias] = canonical
                added += 1
        if not added:
            break
    return aliases


def build_indices(
    src_root: Path,
    module_prefix: Optional[str] = None,
    entrypoints: Sequence[Path] = (),
    project_root: Optional[Path] = None,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
) -> Tuple[Dict[str, CallableDef], Dict[str, ModuleIndex], Dict[str, str]]:
    return build_indices_from_analysis_files(
        list(
            iter_analysis_files(
                src_root,
                module_prefix=module_prefix,
                entrypoints=entrypoints,
                project_root=project_root,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        )
    )


def merge_module_index(target: ModuleIndex, source: ModuleIndex) -> None:
    """Accumulate definitions/imports from another snippet of the same module."""
    target.imports.update(source.imports)
    target.classes.update(source.classes)
    target.class_bases.update(source.class_bases)
    target.properties.update(source.properties)
    target.static_methods.update(source.static_methods)
    target.class_methods.update(source.class_methods)
    for module_name in source.star_imports:
        if module_name not in target.star_imports:
            target.star_imports.append(module_name)
