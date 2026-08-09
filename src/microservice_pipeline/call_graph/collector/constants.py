"""Source-level name tables the collector matches against.

Every table here is matched by *name as written in the source*, never by import
and inspection, because importing to check would execute the analyzed project.
That is the same trade the decorator names in ``definitions`` make: it misses
aliases, and it never runs arbitrary code.
"""

from __future__ import annotations


# ``functools.partial(f, ...)`` evaluates to ``f`` with arguments pre-bound, so
# calling the result calls ``f``. Matched by source-level name, like the
# decorator names in ``definitions``, because importing to check would execute
# the analyzed project.
PARTIAL_NAMES = frozenset({"partial", "functools.partial"})

# Element callables are keyed under the variable's name plus this suffix in the
# same scope map. A separate stack for one level of nesting would double the
# scope machinery for a dimension that is only ever read here, and the suffix
# cannot collide with a Python identifier.
ELEMENT_SUFFIX = "[]"

# The conventional names for a method's implicit receiver. Matched by name
# because that is all the AST offers -- a positional first parameter is only
# ``self`` by convention, and code that renames it is code we deliberately do
# not follow.
SELF_NAMES = frozenset({"self", "cls"})

# Typing constructs that wrap the type actually of interest. ``Optional[X]``
# and ``Union[X, None]`` still denote an ``X``; the container forms denote a
# collection *of* ``X``, which is a different fact and handled separately.
TRANSPARENT_ANNOTATION_NAMES = frozenset(
    {"Optional", "Union", "Final", "Annotated", "ClassVar", "typing"}
)
CONTAINER_ANNOTATION_NAMES = frozenset(
    {
        "List", "list", "Set", "set", "FrozenSet", "frozenset",
        "Sequence", "Iterable", "Iterator", "Collection", "MutableSequence",
        "Tuple", "tuple", "Dict", "dict", "Mapping", "MutableMapping",
        "DefaultDict", "defaultdict", "OrderedDict", "Deque", "deque",
    }
)
