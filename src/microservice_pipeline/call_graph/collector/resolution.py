"""Turning a callee expression into candidate targets.

Every call site arrives here as an expression, and leaves as a list of
``(callee_id, relation, resolved)`` triples. ``_resolve_callees`` may return
several -- a method name on a receiver whose type is uncertain is genuinely
several candidates, and collapsing that to one would invent precision the
analysis does not have. ``resolved=False`` marks a target named but not matched
to a definition, which ``edges`` decides whether to keep.

Resolution prefers known project definitions over descriptive external targets,
and it is where ``super()``, dynamic ``getattr``, star imports, module aliases
and constructor calls each get their special handling.

This is the other half of the cycle with ``inference``; see that module's
docstring.

Requires from siblings: ``_infer_class_types_from_value``, ``get_expr_types``,
``_infer_callable_ids_from_value``, ``_receiver_types_for_expr``,
``_resolve_method_targets``, ``_resolve_class_reference_name``,
``_resolve_constructor_targets``, ``_is_known_class``, ``_known_class_id``.
"""

from __future__ import annotations

import ast
from typing import List, Optional, Set, Tuple
from ..ast_utils import attribute_to_name
from .constants import SELF_NAMES
from .shapes import is_super_call

from ..project_index import unique
from .state import CollectorState


class ResolutionMixin(CollectorState):
    """Turning a callee expression into candidate targets."""

    def _resolve_dynamic_getattr_callees(
        self, func: ast.AST
    ) -> List[Tuple[str, str, bool]]:
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

        imported_base = self.module_index.imports.get(target_obj.id)
        if not imported_base:
            return []

        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
            target = f"{imported_base}.{attr_arg.value}"
            return [(target, "dynamic_getattr", target in self.callable_ids)]

        prefix = f"{imported_base}."
        matches = sorted(
            callable_id
            for callable_id in self.callable_ids
            if callable_id.startswith(prefix)
        )
        return [(callable_id, "dynamic_getattr", True) for callable_id in matches]

    def _resolve_star_import_targets(self, name: str) -> List[str]:
        return self.project_index.resolve_star_import_targets(self.module_index, name)

    def _resolve_callees(self, func: ast.AST) -> List[Tuple[str, str, bool]]:
        """Resolve a call target, allowing multiple method candidates.

        Attribute calls are checked from most informative to least informative:
        ``super()``, ``self``/``cls``, inferred receiver type, explicit class,
        then general name/import resolution.
        """
        if isinstance(func, ast.Attribute):
            full = attribute_to_name(func)
            if not full:
                return []

            if is_super_call(func.value):
                super_class_id = self._super_class_id(func.value)
                targets = self.project_index.resolve_super_method_targets(
                    super_class_id, func.attr
                )
                cooperative = self.project_index.resolve_cooperative_super_targets(
                    super_class_id, func.attr
                )
                if targets or cooperative:
                    return [
                        *((target, "super_method", True) for target in targets),
                        *(
                            (target, "cooperative_super", True)
                            for target in cooperative
                        ),
                    ]

            if (
                isinstance(func.value, ast.Name)
                and func.value.id in SELF_NAMES
                and self.current_class
            ):
                class_id = self.current_class_id()
                if class_id:
                    targets = self.project_index.resolve_method_targets(class_id, func.attr)
                    # ``self`` is not necessarily an instance of the class the
                    # code is written in, so a subclass override is an equally
                    # real target -- and usually the one that runs. This holds
                    # whether or not the base supplies its own definition, so the
                    # override edges are added alongside rather than as a
                    # fallback: on climlab the base defines the hook in most
                    # cases (``_Insolation._compute_fixed``,
                    # ``GreyGas._compute_emission``), which a fallback-only rule
                    # would never see past.
                    overrides = [
                        target
                        for target in self.project_index.resolve_subclass_override_targets(class_id, func.attr)
                        if target not in targets
                    ]
                    if targets or overrides:
                        return [
                            *((target, "self_method", True) for target in targets),
                            *((target, "virtual_override", True) for target in overrides),
                        ]
                    # No method of that name anywhere in the hierarchy. Before
                    # giving up, ask whether the attribute holds a *callable*:
                    # ``self.handler = on_event`` then ``self.handler()`` is a
                    # stored callback, not a method, and the class-id domain had
                    # no way to represent it.
                    stored = sorted(
                        target
                        for target in self._class_attr_callable_types(func)
                        if target in self.callable_ids
                    )
                    if stored:
                        return [
                            (target, "inferred_callable", True) for target in stored
                        ]
                    return [(f"{class_id}.{func.attr}", "self_method", False)]

            if isinstance(func.value, ast.Name):
                var_types = self.types.get_var_types(func.value.id)
                if var_types:
                    targets: List[str] = []
                    for var_type in sorted(var_types):
                        targets.extend(self.project_index.resolve_method_targets(var_type, func.attr))
                    if targets:
                        return [
                            (target, "inferred_type", True)
                            for target in unique(targets)
                        ]
                    first_type = sorted(var_types)[0]
                    return [(f"{first_type}.{func.attr}", "inferred_type", False)]

            # Not just names and attributes: a receiver can be any expression
            # whose type we can infer, including another call. ``get_expr_types``
            # alone left ``build().submit()`` unresolved.
            with self._resolving_receiver():
                receiver_types = self._infer_class_types_from_value(func.value)
            if receiver_types:
                targets = []
                for receiver_type in sorted(receiver_types):
                    targets.extend(
                        self.project_index.resolve_method_targets(receiver_type, func.attr)
                    )
                if targets:
                    return [
                        (target, "inferred_type", True)
                        for target in unique(targets)
                    ]
                first_type = sorted(receiver_types)[0]
                return [(f"{first_type}.{func.attr}", "inferred_type", False)]

            receiver_name = attribute_to_name(func.value)
            if receiver_name:
                class_ids = self._resolve_class_reference_name(receiver_name)
                if class_ids:
                    targets = []
                    for class_id in class_ids:
                        targets.extend(self.project_index.resolve_method_targets(class_id, func.attr))
                    if targets:
                        return [
                            (target, "class_method", True)
                            for target in unique(targets)
                        ]

        # ``cls(...)`` inside a classmethod constructs an instance of the class
        # the method belongs to. Resolved here rather than left to the bare-name
        # fallback, which cannot see that ``cls`` is bound at all and reported it
        # as a call to an unresolvable value.
        if isinstance(func, ast.Name) and func.id == "cls" and self.current_class:
            class_id = self.current_class_id()
            if class_id:
                targets = self.project_index.resolve_constructor_targets(class_id)
                if targets:
                    return [(target, "constructor", True) for target in targets]

        # The callable-value rung. Placed after every class-based rule, so a
        # receiver whose type is known still resolves as a method call, and
        # before the single-target fallback, which can only guess a bare name.
        callable_targets = sorted(
            target
            for target in self._infer_callable_ids_from_value(func, resolve_names=False)
            if target in self.callable_ids
        )
        if callable_targets:
            return [(target, "inferred_callable", True) for target in callable_targets]

        # ``model(x)`` where ``model`` is an instance of a class defining
        # ``__call__``. Cheap, and independent of everything above: the receiver
        # has an ordinary class type, it is the *call* that is implicit.
        dunder_call_targets: List[str] = []
        for receiver_type in sorted(self._callee_expression_types(func)):
            dunder_call_targets.extend(
                self.project_index.resolve_method_targets(receiver_type, "__call__")
            )
        if dunder_call_targets:
            return [
                (target, "dunder_call", True)
                for target in unique(dunder_call_targets)
            ]

        resolved = self._resolve_callee(func)
        if resolved:
            return [resolved]
        return []

    def _callee_expression_types(self, func: ast.AST) -> Set[str]:
        """Class types of the thing being called, for the ``__call__`` rung.

        ``cls`` is excluded. It is seeded with its class id like ``self``, but it
        denotes the *class object*, so ``cls(...)`` constructs an instance and
        does not invoke ``__call__`` on one. Treating the two alike turns every
        alternative constructor in a class that also defines ``__call__`` into an
        edge to the wrong method.
        """
        if isinstance(func, ast.Name):
            if func.id == "cls":
                return set()
            return self.types.get_var_types(func.id)
        if isinstance(func, ast.Attribute):
            with self._resolving_receiver():
                return self._infer_class_types_from_value(func)
        return set()

    def _super_class_id(self, node: ast.AST) -> Optional[str]:
        """Which class a ``super()`` call starts its MRO search after.

        Usually the enclosing class, but ``super(Other, self)`` names it
        explicitly and means something different. That distinction was harmless
        while resolution unioned over bases; under C3 the starting point decides
        the answer, so the argument has to be read.
        """
        if isinstance(node, ast.Call) and node.args:
            named = self._resolve_class_reference_name(attribute_to_name(node.args[0]) or "")
            if named:
                return named[0]
        return self.current_class_id()

    def _resolve_callee(self, func: ast.AST) -> Optional[Tuple[str, str, bool]]:
        """Resolve a simple name or dotted attribute to one best target.

        The returned tuple is ``(callee_id, relation, resolved)``. Keeping a
        useful unresolved ID lets callers optionally expose external/dynamic
        dependencies instead of silently discarding them.
        """
        if isinstance(func, ast.Name):
            name = func.id
            enclosing_local = self._resolve_enclosing_local_callable(name)
            if enclosing_local:
                return (enclosing_local, "direct", True)

            local = f"{self.module}.{name}"
            if local in self.callable_ids:
                return (local, "direct", True)

            class_matches = self._resolve_class_reference_name(name)
            if class_matches:
                constructor_targets: List[str] = []
                for class_id in class_matches:
                    constructor_targets.extend(
                        self.project_index.resolve_constructor_targets(class_id)
                    )
                if len(constructor_targets) == 1:
                    return (constructor_targets[0], "constructor", True)
                if len(constructor_targets) > 1:
                    return (unique(constructor_targets)[0], "constructor", False)

            imported = self.module_index.imports.get(name)
            if imported:
                # Through the alias map first: a callable imported from a package
                # is recorded under the package path, not the defining module.
                canonical = self.project_index.canonical_callable_id(imported)
                is_known = canonical in self.callable_ids
                return (canonical if is_known else imported, "imported", is_known)

            star_import_targets = self._resolve_star_import_targets(name)
            star_import_matches = [
                target for target in star_import_targets if target in self.callable_ids
            ]
            if len(star_import_matches) == 1:
                return (star_import_matches[0], "imported", True)
            if len(star_import_matches) > 1:
                return (star_import_matches[0], "imported", False)

            star_import_class_matches = []
            for candidate_class in star_import_targets:
                if self.project_index.is_known_class(candidate_class):
                    class_id = self.project_index.known_class_id(candidate_class)
                    star_import_class_matches.extend(
                        self.project_index.resolve_constructor_targets(class_id)
                    )
            if len(star_import_class_matches) == 1:
                return (star_import_class_matches[0], "constructor", True)
            if len(star_import_class_matches) > 1:
                return (
                    unique(star_import_class_matches)[0],
                    "constructor",
                    False,
                )

            if self.types.is_bound_local(name):
                # The name resolved; what it holds is the problem. ``handler(x)``
                # where ``handler`` is a parameter or a local is a call to a
                # value, and the abstract domain is a set of *class* ids with no
                # way to say "this variable holds that function". Distinguished
                # from an ordinary unresolved name so the health report can
                # count the gap instead of burying it among third-party calls.
                return (name, "callable_value", False)
            return (name, "direct", False)

        if isinstance(func, ast.Attribute):
            full = attribute_to_name(func)
            if not full:
                return None

            # variable type inferred method call: obj.method()
            if isinstance(func.value, ast.Name):
                var_type = self.types.get_var_type(func.value.id)
                if var_type:
                    inferred_targets = self.project_index.resolve_method_targets(var_type, func.attr)
                    if inferred_targets:
                        return (inferred_targets[0], "inferred_type", True)
                    return (f"{var_type}.{func.attr}", "inferred_type", False)

                # imported module alias call: alias.fn()
                imported_base = self.module_index.imports.get(func.value.id)
                if imported_base:
                    target = f"{imported_base}.{func.attr}"
                    if self.project_index.is_known_class(target):
                        class_id = self.project_index.known_class_id(target)
                        init_targets = self.project_index.resolve_constructor_targets(class_id)
                        if init_targets:
                            return (init_targets[0], "constructor", True)
                    is_known = target in self.callable_ids
                    return (target, "imported", is_known)

            # nested imported alias: alias.sub.fn()
            if "." in full:
                base, attr = full.split(".", 1)
                imported_base = self.module_index.imports.get(base)
                if imported_base:
                    target = f"{imported_base}.{attr}"
                    if self.project_index.is_known_class(target):
                        class_id = self.project_index.known_class_id(target)
                        init_targets = self.project_index.resolve_constructor_targets(class_id)
                        if init_targets:
                            return (init_targets[0], "constructor", True)
                    is_known = target in self.callable_ids
                    return (target, "imported", is_known)

            # maybe fully qualified local reference
            is_known = full in self.callable_ids
            return (full, "attribute", is_known)

        return None
