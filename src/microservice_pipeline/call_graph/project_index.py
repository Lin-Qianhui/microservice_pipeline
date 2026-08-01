"""Static, whole-project facts and the resolution that depends on nothing else.

Everything here is derivable from the definition pass alone: which classes exist,
what they inherit from, which callables are properties or static methods, and
which name in which module refers to which definition. None of it depends on the
type inference happening inside a collector, so it is computed once for a run and
shared by every pass.

That sharing is the point. ``CallCollector`` used to rebuild these indexes in its
own ``__init__``, and the pass drivers construct one collector per file per
fixpoint iteration -- roughly nine times the file count per run, each rescanning
every module in the project. The result was identical every time.

The methods that need per-file context (``module``, ``module_index``) take it as
an argument rather than holding it, which is what keeps one instance valid for
every file in the run.
"""


from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .models import ModuleIndex


def unique(values: Iterable[str]) -> List[str]:
    """Deduplicate while preserving first-seen order."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


class ProjectIndex:
    """Whole-project class and callable facts, built once per analysis run."""

    def __init__(
        self,
        module_map: Dict[str, ModuleIndex],
        known_classes: Dict[str, str],
        callable_ids: Set[str],
    ):
        self.module_map = module_map
        self.known_classes = known_classes
        self.callable_ids = callable_ids

        self.class_bases = {
            class_id: bases
            for module_info in module_map.values()
            for class_id, bases in module_info.class_bases.items()
        }
        # Reverse of ``class_bases``, for resolving a ``self.hook()`` call in a
        # base class down to the subclass overrides that really run. Base IDs are
        # canonicalized on the way in because a base recorded through a package
        # re-export is an alias of the defining path.
        self.subclasses: Dict[str, Set[str]] = {}
        for class_id, bases in self.class_bases.items():
            for base_id in bases:
                canonical_base = known_classes.get(base_id, base_id)
                self.subclasses.setdefault(canonical_base, set()).add(class_id)

        self.property_ids = {
            property_id
            for module_info in module_map.values()
            for property_id in module_info.properties
        }
        self.static_method_ids = {
            method_id
            for module_info in module_map.values()
            for method_id in module_info.static_methods
        }
        self.class_method_ids = {
            method_id
            for module_info in module_map.values()
            for method_id in module_info.class_methods
        }
        # ``_callable_belongs_to_known_class`` ran this comprehension on every
        # call; the answer only depends on the class set, so build it once.
        self._known_class_prefixes = tuple(
            f"{class_id}." for class_id in known_classes.values()
        )

    def is_known_class(self, class_id: str) -> bool:
        return (
            class_id in self.known_classes
            or class_id in self.known_classes.values()
        )

    def known_class_id(self, class_id: str) -> str:
        return self.known_classes.get(class_id, class_id)

    def callable_belongs_to_known_class(self, callable_id: str) -> bool:
        return callable_id.startswith(self._known_class_prefixes)

    def resolve_class_reference_name(
        self, module: str, module_index: ModuleIndex, name: str
    ) -> List[str]:
        matches: Set[str] = set()

        if "." not in name:
            local_class = module_index.classes.get(name)
            if local_class:
                matches.add(local_class)

            imported = module_index.imports.get(name)
            if imported and self.is_known_class(imported):
                matches.add(self.known_class_id(imported))

            for imported in self.resolve_star_import_targets(module_index, name):
                if self.is_known_class(imported):
                    matches.add(self.known_class_id(imported))

            local_name = f"{module}.{name}"
            if self.is_known_class(local_name):
                matches.add(self.known_class_id(local_name))

            return sorted(matches)

        base, rest = name.split(".", 1)
        imported_base = module_index.imports.get(base)
        if imported_base:
            candidate = f"{imported_base}.{rest}"
            if self.is_known_class(candidate):
                matches.add(self.known_class_id(candidate))

        if self.is_known_class(name):
            matches.add(self.known_class_id(name))

        return sorted(matches)

    def resolve_method_targets(
        self, class_id: str, method_name: str, seen: Optional[Set[str]] = None
    ) -> List[str]:
        """Find a method on a class or, recursively, its indexed base classes."""
        if seen is None:
            seen = set()
        if class_id in seen:
            return []
        seen.add(class_id)

        direct = f"{class_id}.{method_name}"
        if direct in self.callable_ids:
            return [direct]

        targets: List[str] = []
        for base_id in self.class_bases.get(class_id, []):
            if not self.is_known_class(base_id):
                continue
            targets.extend(
                self.resolve_method_targets(
                    self.known_class_id(base_id), method_name, seen=seen.copy()
                )
            )
        return unique(targets)

    def class_and_ancestors(self, class_id: str) -> List[str]:
        """``class_id`` followed by every base class reachable from it.

        Attribute facts are recorded against whichever class the assigning method
        belongs to, which is often a base. ``self.subprocess`` is populated by
        ``Process.add_subprocess`` but read by ``TimeDependentProcess``, so a
        lookup that checks only the reading class finds nothing.
        """
        chain: List[str] = []
        queue: List[str] = [class_id]
        seen: Set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            chain.append(current)
            for base_id in self.class_bases.get(current, []):
                canonical = self.known_class_id(base_id)
                if canonical not in seen:
                    queue.append(canonical)
        return chain

    def resolve_subclass_override_targets(
        self, class_id: str, method_name: str
    ) -> List[str]:
        """Find overrides of ``method_name`` below ``class_id`` in the hierarchy.

        This is the Template Method pattern, which scientific frameworks lean on
        heavily. ``_SurfaceFlux._compute_heating_rates`` calls
        ``self._compute_flux()``, but ``_SurfaceFlux`` never defines
        ``_compute_flux`` -- its two subclasses do, and one of those is what
        actually runs. Resolving ``self`` only to the enclosing class, as normal
        method resolution does, finds nothing at all here. On climlab this shape
        was 49 of the 55 edges still missing after alias canonicalization.

        Every override found is a genuine runtime possibility, so all of them are
        returned. That deliberately over-approximates -- a base with ten
        subclasses yields ten edges -- which is why callers label these
        ``virtual_override`` and the structural graph weights them below literal
        calls rather than treating them as ordinary resolved edges.
        """
        targets: List[str] = []
        queue: List[str] = [class_id]
        seen: Set[str] = {class_id}
        while queue:
            current = queue.pop()
            for subclass_id in sorted(self.subclasses.get(current, ())):
                if subclass_id in seen:
                    continue
                seen.add(subclass_id)
                queue.append(subclass_id)
                # A subclass that overrides the method is one possible target;
                # keep descending regardless, because a deeper subclass can
                # override it again and instances of that class dispatch there.
                candidate = f"{subclass_id}.{method_name}"
                if candidate in self.callable_ids:
                    targets.append(candidate)
        return unique(targets)

    def resolve_constructor_targets(self, class_id: str) -> List[str]:
        return self.resolve_method_targets(class_id, "__init__")

    def resolve_super_method_targets(
        self, class_id: Optional[str], method_name: str
    ) -> List[str]:
        """Targets for ``super().<method_name>()`` inside ``class_id``."""
        if not class_id:
            return []

        targets: List[str] = []
        for base_id in self.class_bases.get(class_id, []):
            if not self.is_known_class(base_id):
                continue
            targets.extend(
                self.resolve_method_targets(self.known_class_id(base_id), method_name)
            )
        return unique(targets)

    def resolve_name_from_module_exports(
        self, module_name: str, name: str, seen: Optional[Set[str]] = None
    ) -> List[str]:
        if seen is None:
            seen = set()
        if module_name in seen:
            return []
        seen.add(module_name)

        matches: Set[str] = set()
        module_index = self.module_map.get(module_name)
        local_name = f"{module_name}.{name}"

        if local_name in self.callable_ids:
            matches.add(local_name)
        if module_index and name in module_index.classes:
            matches.add(module_index.classes[name])
        elif local_name in self.known_classes.values() or local_name in self.known_classes:
            matches.add(self.known_classes.get(local_name, local_name))

        if not module_index:
            return sorted(matches)

        imported = module_index.imports.get(name)
        if imported:
            matches.add(self.known_classes.get(imported, imported))

        for imported_module in module_index.star_imports:
            matches.update(
                self.resolve_name_from_module_exports(
                    imported_module, name, seen=seen.copy()
                )
            )

        return sorted(matches)

    def resolve_star_import_targets(
        self, module_index: ModuleIndex, name: str
    ) -> List[str]:
        matches: Set[str] = set()
        for imported_module in module_index.star_imports:
            matches.update(self.resolve_name_from_module_exports(imported_module, name))
        return sorted(matches)