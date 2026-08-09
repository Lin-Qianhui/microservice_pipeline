"""Entering and leaving the scopes a callable id is built from.

Modules, classes, functions and lambdas: what each one does to the scope stack,
what name the thing being entered gets, and what is known about its parameters
on the way in. Imports live here too, since an import is what binds a name at
module scope.

Parameter seeding is the cheap half of inference: ``def solve(mesh: Mesh)``
states outright what a call site might never reveal, so annotations are read
here and turned into the same type facts a call site would have produced.

Requires from siblings: ``_add_edge``, ``_add_import_edge`` callers,
``get_expr_types``, ``_resolve_class_reference_name``, ``_annotation_types``
consumers, ``_infer_callable_ids_from_value``.
"""

from __future__ import annotations

import ast
from typing import Optional, Set, Tuple
from ..ast_utils import attribute_to_name
from ..definitions import class_body_expressions, signature_expressions
from ..discovery import module_callable_id
from ..type_env import attr_container_origin, param_origin
from ..models import (
    NO_SOURCE_SITE,
    FunctionParamSummary,
    keyword_only_param_key,
    param_key,
)
from .constants import (
    CONTAINER_ANNOTATION_NAMES,
    SELF_NAMES,
    TRANSPARENT_ANNOTATION_NAMES,
)
from .shapes import annotation_head

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
# The fallback above is the package-wide idiom for letting a module run as a
# loose script. A file inside this sub-package cannot meaningfully do that, so
# it is dead insurance here -- kept because removing it is a behaviour change,
# and because the audit belongs to the whole package, not to this refactor.

from .state import CollectorState


class ScopesMixin(CollectorState):
    """Entering and leaving the scopes a callable id is built from."""

    def visit_Module(self, node: ast.Module) -> None:
        """Treat top-level statements as the body of the synthetic module node."""
        prev_callable = self.current_callable
        self.current_callable = self.module_callable
        self.callable_stack.append(self.module_callable)
        self.types.push_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.types.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable

    def _add_import_edge(
        self, module_name: str, lineno: int, col_offset: int = NO_SOURCE_SITE
    ) -> None:
        imported_module_callable = module_callable_id(module_name)
        if imported_module_callable in self.callable_ids:
            self._add_edge(
                imported_module_callable, "import", True, lineno, col_offset
            )
        else:
            self._add_edge(module_name, "import", False, lineno, col_offset)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_import_edge(alias.name, node.lineno, node.col_offset)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported_module = resolve_import_from_module(
            self.module,
            node.module,
            node.level,
            current_is_package=is_package_file(self.file),
        )
        if imported_module is None:
            return
        if node.module:
            self._add_import_edge(imported_module, node.lineno, node.col_offset)
            return
        for alias in node.names:
            target = resolve_import_from_target(
                self.module,
                node.module,
                node.level,
                alias.name,
                current_is_package=is_package_file(self.file),
            )
            if target:
                self._add_import_edge(target, node.lineno, node.col_offset)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class body, attributing what it evaluates to the class itself.

        The class body is a code object that runs once, when the ``class``
        statement executes, so a computed default argument belongs to the class
        and not to the method whose signature it is written in. The interpreter
        agrees: it reports the caller of ``make_slabatm_axis()`` as
        ``SlabAtmosphere``, not ``SlabAtmosphere.__init__``.
        """
        prev_class = self.current_class
        prev_callable = self.current_callable
        self.current_class = node.name

        # ``class_bodies``, not ``callable_ids`` -- a class body is a valid edge
        # caller but never a callee, so it is deliberately absent from the
        # resolution universe. See models.resolvable_callable_ids.
        class_id = f"{self.module}.{node.name}"
        if class_id in self.module_index.class_bodies:
            self.current_callable = class_id

        # Everything the class statement evaluates, attributed to the class...
        for expression in class_body_expressions(node):
            self.visit(expression)

        # ...and then each method, entered in its own scope with the signature
        # already accounted for above.
        self.current_callable = prev_callable
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._enter_callable(statement, statement.name)

        self.current_class = prev_class
        self.current_callable = prev_callable

    def _enter_callable(self, node: ast.AST, name: str) -> None:
        """Enter a function scope, seed parameter types, visit it, and restore state."""
        prev_callable = self.current_callable
        prev_enclosing = self.enclosing_function
        if self.enclosing_function is not None:
            self.current_callable = f"{self.enclosing_function}.<locals>.{name}"
        elif self.current_class:
            self.current_callable = f"{self.module}.{self.current_class}.{name}"
        else:
            self.current_callable = f"{self.module}.{name}"

        self.enclosing_function = self.current_callable
        self.callable_stack.append(self.current_callable)
        self.types.push_scope()

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if self.current_class and node.args.args:
                first_arg = node.args.args[0].arg
                if first_arg in SELF_NAMES:
                    cls_id = f"{self.module}.{self.current_class}"
                    self.types.set_var_type(first_arg, cls_id)
            # An empty summary still matters: annotations are read here too, and
            # they are available whether or not any call site was observed.
            param_summary = self.param_summaries.get(self.current_callable) or FunctionParamSummary()
            positional_args = [*node.args.posonlyargs, *node.args.args]
            for index, arg in enumerate(positional_args):
                self._seed_param_types(param_summary, index, arg)
                self._seed_param_origin(index, arg, is_receiver=index == 0)
            for index, arg in enumerate(
                node.args.kwonlyargs, start=len(positional_args)
            ):
                self._seed_param_types(param_summary, index, arg)
                self._seed_param_origin(index, arg, keyword_only=True)

        # Only the body. Decorators, defaults and annotations were visited by the
        # enclosing scope, which is where the interpreter evaluates them.
        for statement in getattr(node, "body", []):
            self.visit(statement)
        self.types.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable
        self.enclosing_function = prev_enclosing

    def _annotation_types(self, node: Optional[ast.AST]) -> Tuple[Set[str], Set[str]]:
        """Resolve an annotation to ``(object types, element types)``.

        Annotations are free, reliable inference that the collectors previously
        ignored entirely. ``def solve(mesh: Mesh)`` states outright what a call
        site might never reveal.

        A container annotation contributes only element types: ``items: List[Order]``
        means ``items`` *holds* orders, so treating ``Order`` as the type of
        ``items`` itself would resolve ``items.submit()`` to a method that
        instances of ``list`` do not have. Mapping annotations follow the same
        rule as dict literals -- the values are the elements.
        """
        if node is None:
            return set(), set()

        # ``"Mesh"`` as a forward reference, and the string halves of
        # ``from __future__ import annotations`` output.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return set(self._resolve_class_reference_name(node.value)), set()

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left_objects, left_elements = self._annotation_types(node.left)
            right_objects, right_elements = self._annotation_types(node.right)
            return left_objects | right_objects, left_elements | right_elements

        if isinstance(node, ast.Subscript):
            head = annotation_head(node.value)
            inner = node.slice
            parts = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]

            if head in CONTAINER_ANNOTATION_NAMES:
                # ``Dict[K, V]`` -- the values are the elements; a one-argument
                # container annotates its elements directly.
                targets = parts[1:] if head.lower() in {"dict", "mapping", "mutablemapping", "defaultdict", "ordereddict"} and len(parts) > 1 else parts
                element_types: Set[str] = set()
                for part in targets:
                    part_objects, part_elements = self._annotation_types(part)
                    element_types |= part_objects | part_elements
                return set(), element_types

            if head in TRANSPARENT_ANNOTATION_NAMES:
                objects: Set[str] = set()
                elements: Set[str] = set()
                for part in parts:
                    part_objects, part_elements = self._annotation_types(part)
                    objects |= part_objects
                    elements |= part_elements
                return objects, elements
            return set(), set()

        name = attribute_to_name(node)
        if not name or name == "None":
            return set(), set()
        return set(self._resolve_class_reference_name(name)), set()

    def _seed_param_types(
        self, param_summary: FunctionParamSummary, index: int, arg: ast.arg
    ) -> None:
        """Bind one parameter to the types observed at its call sites.

        Object and element facts are seeded into different maps: a parameter that
        received a list of ``Order`` must answer ``Order`` when iterated, not
        when called on directly.
        """
        annotated_objects, annotated_elements = self._annotation_types(arg.annotation)

        param_types = set(param_summary.positional_types.get(index, set()))
        param_types.update(param_summary.named_types.get(arg.arg, set()))
        param_types |= annotated_objects
        if param_types:
            self.types.set_var_types(arg.arg, param_types)

        element_types = set(param_summary.positional_element_types.get(index, set()))
        element_types.update(param_summary.named_element_types.get(arg.arg, set()))
        element_types |= annotated_elements
        if element_types:
            self.types.set_container_types(arg.arg, element_types)

        # The callable dimension of the same fact. This is what makes
        # ``def apply(self, msg, func): return func(...)`` resolvable: the
        # callable arrived at a call site and has to be carried into the callee.
        param_callables = set(param_summary.positional_callables.get(index, set()))
        param_callables.update(param_summary.named_callables.get(arg.arg, set()))
        if param_callables:
            self.types.set_var_callables(arg.arg, param_callables)

    def _seed_param_origin(
        self,
        index: int,
        arg: ast.arg,
        *,
        is_receiver: bool = False,
        keyword_only: bool = False,
    ) -> None:
        """Mark a parameter name as holding the value its caller passed.

        The receiver is skipped. ``self`` is not something a caller registers --
        it is the thing being registered *into*, and treating it as a parameter
        would read every ``self.x = ...`` in a method as an escape of ``self``.
        """
        if is_receiver and self.current_class and arg.arg in SELF_NAMES:
            return
        key = (
            keyword_only_param_key(arg.arg)
            if keyword_only
            else param_key(index, arg.arg)
        )
        origins = {param_origin(key)}
        # A parameter that callers fill with the contents of a registry keeps
        # that provenance inside the callee, so ``for a in artists: a.draw()``
        # in a free function still counts as invoking the registry.
        if self.current_callable is not None:
            origins.update(
                attr_container_origin(attr)
                for attr in self.registry_facts.attrs_for_param(
                    self.current_callable, index, arg.arg
                )
            )
        self.types.set_var_origins(arg.arg, origins)

    def _resolve_enclosing_local_callable(self, name: str) -> Optional[str]:
        for scope_callable in reversed(self.callable_stack):
            candidate = f"{scope_callable}.<locals>.{name}"
            if candidate in self.callable_ids:
                return candidate
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_signature(node)
        self._enter_callable(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_signature(node)
        self._enter_callable(node, node.name)

    def _visit_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Visit a ``def``'s decorators, defaults and annotations where they run.

        ``visit_ClassDef`` handles methods itself, so this only fires for module
        level and nested functions, whose enclosing scope is already current.
        """
        for expression in signature_expressions(node):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Enter a lambda body as its own callable scope.

        Without this a call inside a lambda body is attributed to the enclosing
        function, while the tracer reports it under ``<locals>.<lambda>`` -- the
        two graphs disagree about who made the call.
        """
        lambda_id = self._lambda_id(node)
        if lambda_id not in self.callable_ids:
            self.generic_visit(node)
            return

        prev_callable = self.current_callable
        prev_enclosing = self.enclosing_function
        self.current_callable = lambda_id
        self.enclosing_function = lambda_id
        self.callable_stack.append(lambda_id)
        self.types.push_scope()
        self.visit(node.body)
        self.types.pop_scope()
        self.callable_stack.pop()
        self.current_callable = prev_callable
        self.enclosing_function = prev_enclosing

    def _lambda_id(self, node: ast.Lambda) -> str:
        """CPython's own name for a lambda's code object.

        Spelled ``<enclosing>.<locals>.<lambda>`` to match ``co_qualname``, so
        the runtime tracer's IDs keep agreeing with the static ones by
        construction. CPython does not disambiguate sibling lambdas in one
        scope and neither does this: inventing a disambiguator would buy
        precision at the cost of that agreement.
        """
        if self.enclosing_function:
            return f"{self.enclosing_function}.<locals>.<lambda>"
        if self.current_class:
            return f"{self.module}.{self.current_class}.<locals>.<lambda>"
        return f"{self.module}.<lambda>"
