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

from .definitions import build_callable_aliases
from .models import ModuleIndex


def _c3_merge(sequences: List[List[str]]) -> Optional[List[str]]:
    """The C3 merge, or ``None`` when no consistent order exists.

    Repeatedly takes the head of some sequence that appears in no other
    sequence's tail. Returning ``None`` rather than raising keeps a malformed
    hierarchy from aborting the analysis run.
    """
    sequences = [sequence for sequence in sequences if sequence]
    result: List[str] = []
    while sequences:
        for sequence in sequences:
            head = sequence[0]
            if any(head in other[1:] for other in sequences):
                continue
            result.append(head)
            for other in sequences:
                if other and other[0] == head:
                    del other[0]
            sequences = [other for other in sequences if other]
            break
        else:
            return None
    return result


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
        # The callable counterpart of ``known_classes``' alias entries.
        self.callable_aliases = build_callable_aliases(callable_ids, module_map)
        # Linearization is recursive and called from every ``super()`` site, so
        # it is memoized on the index, which is built once per run.
        self._mro_cache: Dict[str, List[str]] = {}

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

    def canonical_callable_id(self, callable_id: str) -> str:
        """Resolve a re-export alias to the path the callable is defined at."""
        return self.callable_aliases.get(callable_id, callable_id)

    def linearize(self, class_id: str) -> List[str]:
        """The C3 method resolution order for ``class_id``.

        This is the order Python itself resolves attributes in, which matters
        because unioning over base classes and letting C3 *choose* are different
        answers whenever a class has more than one base. ``super()`` inside
        ``CAM3(_Radiation_SW, _Radiation_LW)`` reaches ``_Radiation_SW`` and
        nothing else; a union reports both and one of the two is invented.

        Bases outside the project are kept as opaque leaves rather than dropped.
        Dropping one can only change the answer if it defines the method, which
        is unknowable either way, but keeping it preserves the relative order of
        everything around it and makes the function total.

        Never raises. A hierarchy this walks may be inconsistent or cyclic even
        though Python would reject it at runtime, because ``known_class_id``
        aliasing can collapse two classes onto one ID. An exception escaping into
        the edge pass would abort a whole run over one malformed base list, so a
        failure falls back to the breadth-first ancestor order instead.
        """
        cached = self._mro_cache.get(class_id)
        if cached is not None:
            return cached

        computed = self._c3_linearize(class_id, set())
        if computed is None:
            computed = self.class_and_ancestors(class_id)
        self._mro_cache[class_id] = computed
        return computed

    def _c3_linearize(self, class_id: str, active: Set[str]) -> Optional[List[str]]:
        """C3 merge over ``class_bases``, or ``None`` if the hierarchy is broken."""
        if class_id in active:
            return None  # cycle
        cached = self._mro_cache.get(class_id)
        if cached is not None:
            return cached

        bases = [
            self.known_class_id(base_id)
            for base_id in self.class_bases.get(class_id, [])
        ]
        if not bases:
            return [class_id]

        active = active | {class_id}
        sequences: List[List[str]] = []
        for base_id in bases:
            base_mro = self._c3_linearize(base_id, active)
            if base_mro is None:
                return None
            sequences.append(list(base_mro))
        sequences.append(list(bases))

        merged = _c3_merge(sequences)
        if merged is None:
            return None
        return [class_id, *merged]

    def next_in_mro(
        self, class_id: str, after: str, method_name: str
    ) -> Optional[str]:
        """The definition ``super().<method>()`` reaches, for an instance of ``class_id``.

        ``after`` is the class the ``super()`` call is *written in*, which is not
        necessarily ``class_id``: that is the whole point of cooperative multiple
        inheritance. The search starts one past ``after`` in ``class_id``'s MRO.
        """
        mro = self.linearize(class_id)
        try:
            start = mro.index(after) + 1
        except ValueError:
            return None
        for candidate in mro[start:]:
            definition = f"{candidate}.{method_name}"
            if definition in self.callable_ids:
                return definition
        return None

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
        """The single target of ``super().<method_name>()`` inside ``class_id``.

        Python picks exactly one, so this returns at most one. The previous
        implementation walked the lexical bases and unioned whatever they
        defined, which on any class with two bases reported both and invented
        one of them: ``super()`` in ``CAM3(_Radiation_SW, _Radiation_LW)`` emitted
        edges to *both* ``__init__`` methods when C3 says only ``_Radiation_SW``
        runs. The runtime trace confirms the single target.
        """
        if not class_id:
            return []
        target = self.next_in_mro(class_id, class_id, method_name)
        return [target] if target else []

    def resolve_cooperative_super_targets(
        self, class_id: Optional[str], method_name: str
    ) -> List[str]:
        """Extra ``super()`` targets reachable only through a subclass's MRO.

        ``self`` inside ``C`` need not be a ``C``. When some subclass mixes ``C``
        with a sibling, C3 threads ``super()`` inside ``C`` *sideways* into that
        sibling rather than up into ``C``'s own base -- which is the whole
        mechanism cooperative multiple inheritance runs on.

        climlab has exactly this shape: ``super().__init__()`` inside
        ``_Radiation_SW`` reaches ``_Radiation_LW.__init__`` whenever ``self`` is
        an ``RRTMG`` or ``CAM3``, and no walk of ``_Radiation_SW``'s own bases can
        ever see it. The runtime dispatches it; the static graph missed it.

        Which subclass is instantiated is genuinely unknown here, so every
        distinct answer is returned. That over-approximates, which is why these
        are emitted under their own relation rather than folded into
        ``super_method`` -- unlabelled over-approximation is what silently fuses
        services.
        """
        if not class_id:
            return []

        primary = self.next_in_mro(class_id, class_id, method_name)
        targets: List[str] = []
        for subclass_id in sorted(self._all_subclasses(class_id)):
            target = self.next_in_mro(subclass_id, class_id, method_name)
            if target and target != primary:
                targets.append(target)
        return unique(targets)

    def _all_subclasses(self, class_id: str) -> Set[str]:
        """Every class below ``class_id``, transitively."""
        found: Set[str] = set()
        queue = [class_id]
        while queue:
            for subclass_id in self.subclasses.get(queue.pop(), ()):
                if subclass_id not in found:
                    found.add(subclass_id)
                    queue.append(subclass_id)
        return found

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