"""Deferred return-type facts: "this callable returns whatever that one does".

Why this exists
---------------

When the analyzer reads ``def a(): return b()`` it resolves ``b`` immediately,
asks what ``b`` returns, and -- if nothing is known about ``b`` yet -- drops the
answer and moves on. The callee's *name* was in hand and got discarded, so the
only way to recover the fact was to re-read the whole file on a later pass.

That made each pass propagate exactly one hop of a chain. With a cap of three
passes, ``a -> b -> c -> d -> Order()`` never reached ``a``, and every
``x = a(); x.submit()`` silently became an unresolved edge.

Instead of discarding the name, the collectors now record a :class:`ReturnLink`.
After the passes finish, :func:`resolve_return_links` walks those links and
copies facts along them until nothing changes. That step touches no files and no
ASTs -- it is dictionary work measured in microseconds -- so it can afford to run
to a true fixed point and resolve chains of any depth.

Shape shifts
------------

A link is not always a straight copy of the same field. ``return b()`` feeds b's
class types into the caller's class types, but ``return [b()]`` feeds b's *class*
types into the caller's *element* types. So a link names both a source field and
a target field. The collectors compose these automatically: the target comes from
which part of the summary is being filled in, the source from which part of the
callee's summary was consulted.
"""


from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from .models import FunctionReturnSummary, add_indexed_types


# ``FunctionReturnSummary`` fields holding a flat ``Set[str]`` ...
FLAT_RETURN_FIELDS = frozenset({"class_types", "element_types", "callable_ids"})

# ... versus those holding ``Dict[int, Set[str]]`` keyed by tuple/list position.
SLOT_RETURN_FIELDS = frozenset({"slot_types", "slot_element_types", "slot_callable_ids"})


@dataclass(frozen=True)
class ReturnLink:
    """One deferred fact: ``caller``'s return types include ``callee``'s.

    ``source_field`` and ``target_field`` name fields on
    ``FunctionReturnSummary``. They differ whenever the value changes shape on
    the way through -- see the module docstring.

    The slot fields select one index of a slot-keyed field. ``source_slot=None``
    against a slot-keyed ``source_field`` means "every index", which is how a
    whole tuple shape is carried across ``return make_pair()``.
    """

    caller: str
    callee: str
    source_field: str
    target_field: str
    source_slot: Optional[int] = None
    target_slot: Optional[int] = None


class ReturnLinkTable:
    """The links found so far, de-duplicated and in discovery order.

    Order is preserved only so that repeated runs behave identically; the
    fixed-point below reaches the same answer whatever order it visits links in.
    """

    def __init__(self) -> None:
        self._links: Dict[ReturnLink, None] = {}

    def record(self, link: ReturnLink) -> None:
        self._links.setdefault(link, None)

    def __iter__(self) -> Iterator[ReturnLink]:
        return iter(self._links)

    def __len__(self) -> int:
        return len(self._links)

    def __bool__(self) -> bool:
        return bool(self._links)


def _source_types(summary: FunctionReturnSummary, link: ReturnLink):
    """Read the piece of ``summary`` that ``link`` draws from.

    Returns a ``Set[str]`` normally, or a ``Dict[int, Set[str]]`` when the link
    carries a whole slot-keyed field across.
    """
    value = getattr(summary, link.source_field)
    if link.source_field in SLOT_RETURN_FIELDS and link.source_slot is not None:
        return value.get(link.source_slot, set())
    return value


def _merge_into(
    summaries: Dict[str, FunctionReturnSummary], link: ReturnLink
) -> bool:
    """Copy one hop along ``link``; report whether that taught us anything new.

    Returns ``False`` without touching ``summaries`` when there is nothing to
    copy, so a caller with no facts of its own never gains an empty entry.
    """
    source_summary = summaries.get(link.callee)
    if source_summary is None:
        return False

    source = _source_types(source_summary, link)
    if not source:
        return False

    # Whole slot-keyed field: only meaningful onto another slot-keyed field.
    if isinstance(source, dict):
        if link.target_field not in SLOT_RETURN_FIELDS or link.target_slot is not None:
            return False
        target = summaries.setdefault(link.caller, FunctionReturnSummary())
        destination = getattr(target, link.target_field)
        changed = False
        for index, values in source.items():
            before = len(destination.get(index, ()))
            add_indexed_types(destination, index, values)
            if len(destination.get(index, ())) != before:
                changed = True
        return changed

    if link.target_field in SLOT_RETURN_FIELDS:
        if link.target_slot is None:
            return False
        target = summaries.setdefault(link.caller, FunctionReturnSummary())
        destination = getattr(target, link.target_field)
        before = len(destination.get(link.target_slot, ()))
        add_indexed_types(destination, link.target_slot, source)
        return len(destination.get(link.target_slot, ())) != before

    target = summaries.setdefault(link.caller, FunctionReturnSummary())
    destination = getattr(target, link.target_field)
    before = len(destination)
    destination.update(source)
    return len(destination) != before


def resolve_return_links(
    summaries: Dict[str, FunctionReturnSummary],
    links: ReturnLinkTable,
    *,
    max_rounds: int = 100,
) -> Dict[str, FunctionReturnSummary]:
    """Chase recorded links until no new facts appear. Mutates and returns ``summaries``.

    Each round copies facts one hop along every link, so a chain of any depth
    resolves given enough rounds. This terminates on its own, including for
    mutually recursive functions: type facts only ever grow, and they are drawn
    from the finite set of class IDs found in the project, so some round must
    eventually add nothing.

    ``max_rounds`` is therefore a bug backstop rather than a real limit, and
    hitting it is worth a warning -- unlike the pass caps this replaces, it
    should never be what stops the loop.
    """
    for _ in range(max_rounds):
        changed = False
        for link in links:
            if _merge_into(summaries, link):
                changed = True
        if not changed:
            return summaries

    warnings.warn(
        f"resolve_return_links did not settle within {max_rounds} rounds; "
        f"return types may be incomplete ({len(links)} links, "
        f"{len(summaries)} summaries)",
        stacklevel=2,
    )
    return summaries
