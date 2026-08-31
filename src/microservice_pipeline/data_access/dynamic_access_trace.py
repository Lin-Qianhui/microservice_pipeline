"""Record the data accesses a program *actually* performs, as ground truth.

``generate_data_access_ast.py`` infers accesses by reading source code. This
module gets them the other way round: it runs the analyzed project under
``sys.monitoring`` (PEP 669) and records which attribute, subscript and name
access instructions the interpreter really executed. Comparing the two is what
turns every finding in ``code_review.md`` from a statement about *mechanism*
into a statement about *impact* -- see that document's section 6.

It is the data-access sibling of ``call_graph/dynamic_trace.py`` and reuses its
drivers wholesale. Three things make the result comparable to the static
artifacts:

* Callable IDs come from ``CodeIdResolver``, the same object the call tracer
  uses, so ``climlab.model.ebm.EBM._compute`` means one string everywhere.
* Only code inside the analyzed source roots is monitored at all.
* Drivers run **in-process**, because ``sys.monitoring`` cannot see a subprocess.

**The unit of observation** is ``(callable, tier, name, access)``:

    o.spacing        ->  (caller, "attr", "spacing", "read")
    d['lit'] = 1     ->  (caller, "key",  "lit",     "write")
    d[k]             ->  (caller, "key",  "<computed>", "read")
    param            ->  (caller, "name", "param",   "read")

No source position appears in the key. That is deliberate: unlike a callee, the
complete set of access instructions in a code object is knowable from its
bytecode, so a static claim naming an attribute that appears nowhere in the
bytecode is *wrong* regardless of how little the drivers covered. Position would
add per-occurrence precision at the cost of that coverage-independence, and
``AccessEdge`` carries no ``col_offset`` to match on anyway.

**This module observes; it does not judge.** Every access instruction is decoded
and recorded, tagged with a ``role`` saying what kind it is -- a method load
rather than a data attribute, a literal key rather than a computed one, a
parameter rather than an ordinary local. Deciding which roles may enter a recall
denominator is policy, and policy lives in ``access_comparison`` next to the
static definitions it has to stay symmetric with. Keeping the split here would
put the rule that mirrors ``collect_module_globals`` in a module that cannot see
it, which is how the three copies of the callable-ID convention in section 2.5
of the review came about.

The trace is a *lower bound*: it sees what the drivers exercised and nothing
else. An access it never observed is not thereby a false one.

**The identity channel** (``--identity``, off by default) answers a different
question, the second half of section 6: *are two static object IDs the same
runtime object?* That is what ``alias_of`` and the lineage graph claim, and
nothing has ever checked it.

Section 7 describes this as "``id()`` at the observed access". The operand stack
is not reachable from Python, so that is not literally possible -- but
``sys._getframe`` inside the callback gives the executing frame, and the frame's
locals *are* reachable. So identity is recorded for:

* **name-tier loads** -- ``frame.f_locals[name]``, falling back to
  ``f_globals``. Covers ``param`` and ``local_exposed``, the two kinds
  ``_apply_lineage_aliases`` most often aliases.
* **attribute accesses with a name receiver** -- ``self.data``, ``model.R``. The
  receiver is read from the frame and the attribute out of its ``__dict__``.
  Never ``getattr``: that would run properties and ``__getattr__``, i.e. the
  instrument would change the program it is measuring.

Three consequences, all of which the comparison counts rather than hides:

* Values that are only *written* are invisible, because ``INSTRUCTION`` fires
  **before** the instruction runs -- at ``STORE_FAST b``, ``b`` is not bound yet.
  A written name is picked up at its next load.
* An attribute reached through anything but a plain name (``a.b.c``, ``xs[0].y``)
  has an intermediate receiver that lives only on the stack.
* ``id()`` is *not* an identity: CPython reuses addresses. ``ObjectTokens`` keeps
  one strong reference per recorded object so an address cannot be recycled while
  its token is live. Weak references cannot do this job -- ``list`` and ``dict``
  are not weak-referenceable, and those are the containers this exists to judge.

The identity channel also records **string values** at name-tier sites, which is
a separate question ``id()`` cannot answer at all: section 1.7's ``file:path``
nodes are keyed on source text, and whether two callables really share a file is
a matter of the path *value*, not of object identity.
"""

from __future__ import annotations

import dis
import opcode as _opcode
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from microservice_pipeline.call_graph.dynamic_trace import CodeIdResolver, module_map_from_analysis_files
except ImportError:  # pragma: no cover - supports direct script execution
    from call_graph.dynamic_trace import CodeIdResolver, module_map_from_analysis_files  # type: ignore


TOOL_ID = sys.monitoring.PROFILER_ID
TOOL_NAME = "microservice-pipeline-access-trace"

TIER_ATTR = "attr"
TIER_KEY = "key"
TIER_NAME = "name"

ACCESS_READ = "read"
ACCESS_WRITE = "write"

# A subscript whose key is not a compile-time constant. The static side names a
# key by its literal and cannot name this one, so the two are scored in separate
# tables rather than folded together -- code_review.md section 6, "key identity".
COMPUTED_KEY = "<computed>"

# What kind of access this is, within its tier. The comparison uses these to
# decide comparability; nothing here decides it.
ROLE_ATTRIBUTE = "attribute"  # x.data      -- a data attribute
ROLE_METHOD = "method"        # x.method()  -- a method load; the static side
                              #                models the *receiver* instead
ROLE_SUPER = "super"          # super().x   -- the receiver is a dispatch proxy,
                              #                never a data object
ROLE_LITERAL = "literal"      # d['k']
ROLE_COMPUTED = "computed"    # d[k]
ROLE_PARAM = "param"          # a declared parameter
ROLE_LOCAL = "local"          # an ordinary local
ROLE_GLOBAL = "global"        # a module-level name
ROLE_DEREF = "deref"          # a closure cell


def _opcodes(*names: str) -> Set[int]:
    """Resolve opcode names to numbers, silently skipping ones this build lacks.

    The opcode set is not stable across CPython releases and this package is not
    pinned to one. ``BINARY_SUBSCR`` -- which ``code_review.md`` section 6 names
    -- was folded into ``BINARY_OP`` in 3.14 and no longer exists; conversely the
    ``LOAD_FAST_BORROW`` family did not exist before it. Resolving by name and
    dropping the absent ones means a version change degrades the *coverage* of
    the instrument rather than raising ``KeyError`` at import.
    """
    return {_opcode.opmap[name] for name in names if name in _opcode.opmap}


ATTR_READ_OPS = _opcodes("LOAD_ATTR")
# ``super().x``. Separated because the receiver is a ``super`` proxy rather than
# an object: ``_resolve_attribute`` returns nothing for a call receiver, so the
# extractor records no edge for any of these -- verified, zero static rows on
# climlab mention ``super(``. Note also that the method bit is only set for the
# zero-argument form, so ``super(C, self).__init__(**kwargs)`` looks like a plain
# attribute read and would otherwise be scored as one.
SUPER_ATTR_OPS = _opcodes("LOAD_SUPER_ATTR")
ATTR_WRITE_OPS = _opcodes("STORE_ATTR", "DELETE_ATTR")

KEY_READ_OPS = _opcodes("BINARY_SUBSCR")
KEY_WRITE_OPS = _opcodes("STORE_SUBSCR", "DELETE_SUBSCR")
# ``x[a:b]`` when the compiler emits a dedicated slice opcode rather than
# BUILD_SLICE + subscript. A slice is never a key either side can name, so these
# are always computed.
SLICE_READ_OPS = _opcodes("BINARY_SLICE")
SLICE_WRITE_OPS = _opcodes("STORE_SLICE", "DELETE_SLICE")

# 3.14 merged BINARY_SUBSCR into BINARY_OP under the NB_SUBSCR oparg. ``dis``
# exposes the oparg table, so the number is read from the interpreter rather than
# hardcoded to the 26 observed here.
BINARY_OP = _opcode.opmap.get("BINARY_OP")
NB_SUBSCR = next(
    (index for index, spec in enumerate(getattr(_opcode, "_nb_ops", ())) if spec[0] == "NB_SUBSCR"),
    None,
)

FAST_READ_OPS = _opcodes(
    "LOAD_FAST",
    "LOAD_FAST_BORROW",
    "LOAD_FAST_CHECK",
    "LOAD_FAST_AND_CLEAR",
)
# These load two locals at once; ``dis`` reports ``argval`` as a pair.
FAST_READ_PAIR_OPS = _opcodes("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW")
FAST_WRITE_OPS = _opcodes("STORE_FAST", "DELETE_FAST", "STORE_FAST_MAYBE_NULL")
# Two stores at once. Tuple unpacking compiles almost entirely to these, so
# omitting them makes every destructured assignment invisible -- which is how the
# first version of this module falsely accused 34 unpack targets in
# ``RRTMG_LW._compute_heating_rates`` of not existing.
FAST_WRITE_PAIR_OPS = _opcodes("STORE_FAST_STORE_FAST")
# Stores one local and loads another, so it contributes one of each.
FAST_STORE_LOAD_OPS = _opcodes("STORE_FAST_LOAD_FAST")

GLOBAL_READ_OPS = _opcodes("LOAD_GLOBAL")
GLOBAL_WRITE_OPS = _opcodes("STORE_GLOBAL", "DELETE_GLOBAL")

# Module and class bodies address their namespace by name, not by slot, so a
# module-level global is ``LOAD_NAME``/``STORE_NAME`` and never ``LOAD_GLOBAL``.
# On climlab that is 2,284 instructions the first version of this module did not
# decode at all.
NAME_READ_OPS = _opcodes("LOAD_NAME")
NAME_WRITE_OPS = _opcodes("STORE_NAME", "DELETE_NAME")

DEREF_READ_OPS = _opcodes("LOAD_DEREF", "LOAD_FROM_DICT_OR_DEREF")
DEREF_WRITE_OPS = _opcodes("STORE_DEREF", "DELETE_DEREF")

LOAD_CONST_OPS = _opcodes("LOAD_CONST", "LOAD_SMALL_INT")

# Executions recorded per identity site before the offset is retired.
#
# **This number is calibrated, not chosen.** On climlab, sweeping it changes the
# answer until it does not: 8 samples reported 152 contradicted alias claims at
# 52.4% precision, 64 reported 56 at 83.0%, and 512 and 4096 both report 51 at
# 84.5%. So two thirds of what a small budget calls a defect is the budget
# running out before the two sites happened to hold the same object -- the same
# false-accusation failure Step 1a hit, in a different disguise.
#
# 512 is where the claim direction stops moving on this project, at a cost of
# roughly six seconds. ``identity_offsets_at_cap`` reports how many sites still
# used their whole budget, so a project where this default is too small says so
# rather than quietly reporting inflated contradictions.
DEFAULT_IDENTITY_SAMPLES = 512


# One observation, fully qualified: which callable, which tier, which name, read
# or write, and what kind of access it is.
AccessKey = Tuple[str, str, str, str, str]

# One identity observation: which callable, which tier, which source-like path,
# and which runtime object was found there. The token is an ``ObjectTokens``
# token, not an ``id()`` -- see that class for why the difference matters.
IdentityKey = Tuple[str, str, str, int]

# One string value seen at a name-tier site, for the section 1.7 question.
ValueKey = Tuple[str, str, str]

# How the identity channel names the site an observation came from. ``attr``
# paths are always ``receiver.attribute`` with a plain-name receiver.
IdentitySite = Tuple[str, str, str]


@dataclass(frozen=True)
class DecodedAccess:
    """One access instruction found in a code object, executed or not.

    ``receiver`` is set only for attribute accesses whose receiver is loaded by
    the immediately preceding instruction as a plain name -- ``self.data``,
    ``model.R``. It is what lets the identity channel find the object without
    reading the operand stack, and it is ignored by the access channel, whose
    ``key`` is unchanged.
    """

    offset: int
    tier: str
    name: str
    access: str
    role: str
    lineno: int
    col_offset: int
    receiver: str = ""

    def key(self, callable_id: str) -> AccessKey:
        return (callable_id, self.tier, self.name, self.access, self.role)

    def identity_site(self, callable_id: str) -> Optional[IdentitySite]:
        """Where the identity channel would look, or ``None`` if it cannot.

        Writes are excluded here rather than in the tracer: the value a store
        is about to write is not bound yet when the event fires, so a write site
        has nothing to read. It is picked up at the next load of the same name.
        """
        if self.access != ACCESS_READ:
            return None
        if self.tier == TIER_NAME and self.role in {ROLE_PARAM, ROLE_LOCAL, ROLE_GLOBAL, ROLE_DEREF}:
            return (callable_id, TIER_NAME, self.name)
        if self.tier == TIER_ATTR and self.role == ROLE_ATTRIBUTE and self.receiver:
            return (callable_id, TIER_ATTR, f"{self.receiver}.{self.name}")
        return None


# Types whose identity is an artifact of the interpreter rather than a fact about
# the program. ``a is b`` for two equal small integers, interned strings or empty
# tuples says nothing about data flowing from one place to another, and counting
# it would manufacture aliases between every callable in the project. Excluded
# before a token is minted, and counted.
VALUE_TYPES = (int, float, complex, bool, str, bytes, tuple, frozenset, type(None))

# Longest string value kept for the section 1.7 comparison. Paths are short; this
# is a guard against a stray multi-megabyte string, not a meaningful limit.
MAX_VALUE_TEXT = 200


class ObjectTokens:
    """Stable identities for observed objects, safe against address reuse.

    ``id()`` is an address, and CPython reuses addresses. Two unrelated objects
    living at the same address at different times would look like one object,
    which is exactly the false-alias this instrument exists to detect -- so using
    ``id()`` raw would make the measurement report its own bug as a finding.

    The fix is to keep the object alive: one strong reference per distinct
    recorded object means its address cannot be recycled while its token is in
    use, so within a run an ``id()`` *is* an identity. The tidier alternative --
    a weak reference with a callback that retires the id -- does not work here,
    because ``list`` and ``dict`` are not weak-referenceable and they are the
    container types the alias graph is mostly about.

    Retention is bounded, and the bound is reported rather than silently
    applied: a run that hits the cap says so, and its recall figure is marked
    incomplete instead of being quietly computed on a truncated set.
    """

    def __init__(self, cap: int = 500_000):
        self.cap = cap
        self.cap_hit = False
        self.excluded_value_type = 0
        self.excluded_cap = 0
        self._tokens: Dict[int, int] = {}
        # The strong references. Index in this list *is* the token, which makes
        # tokens dense, ordered by first sight, and cheap to write to CSV.
        self._retained: List[Any] = []

    def token(self, value: Any) -> Optional[int]:
        if isinstance(value, VALUE_TYPES):
            self.excluded_value_type += 1
            return None
        existing = self._tokens.get(id(value))
        if existing is not None:
            return existing
        if len(self._retained) >= self.cap:
            self.cap_hit = True
            self.excluded_cap += 1
            return None
        token = len(self._retained)
        self._retained.append(value)
        self._tokens[id(value)] = token
        return token

    @property
    def retained(self) -> int:
        return len(self._retained)


@dataclass
class AccessTraceResult:
    # Accesses the interpreter actually executed.
    observed: Set[AccessKey] = field(default_factory=set)
    # Callables the tracer entered. Scopes the weak direction, exactly as
    # ``TraceResult.executed_callables`` does for the call graph: only a callable
    # that ran can say anything about a static edge that did not appear.
    executed_callables: Set[str] = field(default_factory=set)
    # A source position per observed access, for human lookup in the report. Not
    # part of any key.
    sites: Dict[AccessKey, Tuple[int, int]] = field(default_factory=dict)
    code_objects_armed: int = 0
    driver_errors: List[str] = field(default_factory=list)

    # -- identity channel, empty unless the tracer ran with identity=True ----
    identities: Set[IdentityKey] = field(default_factory=set)
    values: Set[ValueKey] = field(default_factory=set)
    # Identity sites that ran but yielded nothing, by reason. Every one of these
    # is a place the comparison must call "unobserved" rather than "contradicted".
    identity_exclusions: Dict[str, int] = field(default_factory=dict)
    # Sites the tracer reached at all, so the comparison can tell "ran and held
    # nothing recordable" from "never ran".
    identity_sites_seen: Set[IdentitySite] = field(default_factory=set)
    # Offsets that used up their whole sample budget. These are the sites whose
    # observation is known to be partial, and the count is what says whether the
    # sample cap is still shaping the answer -- see ``DEFAULT_IDENTITY_SAMPLES``.
    identity_offsets_at_cap: int = 0
    identity_offsets_sampled: int = 0

    def merge(self, other: "AccessTraceResult") -> None:
        self.observed |= other.observed
        self.executed_callables |= other.executed_callables
        for key, position in other.sites.items():
            self.sites.setdefault(key, position)
        self.code_objects_armed += other.code_objects_armed
        self.driver_errors.extend(other.driver_errors)
        self.identities |= other.identities
        self.values |= other.values
        self.identity_sites_seen |= other.identity_sites_seen
        self.identity_offsets_at_cap += other.identity_offsets_at_cap
        self.identity_offsets_sampled += other.identity_offsets_sampled
        for reason, count in other.identity_exclusions.items():
            self.identity_exclusions[reason] = self.identity_exclusions.get(reason, 0) + count


def _const_key_names(instruction: Any) -> List[str]:
    """The literal key(s) a ``LOAD_CONST`` contributes to a following subscript.

    ``df.loc[:, 'col']`` compiles the whole slice to a single constant tuple, so
    a tuple contributes each of its string elements. Anything that is not a
    string -- an integer index, a bare slice -- is not a key the static side
    names either, and is reported as computed rather than invented.
    """
    value = instruction.argval
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        names = [item for item in value if isinstance(item, str)]
        if names:
            return names
    return []


def _is_subscript_read(instruction: Any) -> bool:
    if instruction.opcode in KEY_READ_OPS:
        return True
    return (
        BINARY_OP is not None
        and NB_SUBSCR is not None
        and instruction.opcode == BINARY_OP
        and instruction.arg == NB_SUBSCR
    )


def _name_role(code: Any, instruction: Any, name: str) -> str:
    if instruction.opcode in GLOBAL_READ_OPS | GLOBAL_WRITE_OPS | NAME_READ_OPS | NAME_WRITE_OPS:
        # ``LOAD_NAME`` is a namespace lookup: the module globals in a module
        # body, the class namespace in a class body. The comparison narrows this
        # with the extractor's own ``collect_module_globals``, which is what
        # separates the two cases.
        return ROLE_GLOBAL
    if instruction.opcode in DEREF_READ_OPS | DEREF_WRITE_OPS:
        return ROLE_DEREF
    return ROLE_PARAM if name in _parameter_names(code) else ROLE_LOCAL


_PARAMETER_CACHE: Dict[Any, frozenset] = {}


def _parameter_names(code: Any) -> frozenset:
    """The declared parameters of a code object, ``*args``/``**kwargs`` included.

    ``co_varnames`` lists parameters first, in this order, which is the documented
    layout and the only way to tell a parameter from an ordinary local at the
    bytecode level -- both are ``LOAD_FAST``.
    """
    cached = _PARAMETER_CACHE.get(code)
    if cached is not None:
        return cached

    count = code.co_argcount + code.co_kwonlyargcount
    flags = code.co_flags
    if flags & 0x04:  # CO_VARARGS
        count += 1
    if flags & 0x08:  # CO_VARKEYWORDS
        count += 1
    names = frozenset(code.co_varnames[:count])
    _PARAMETER_CACHE[code] = names
    return names


def decode_access_instructions(code: Any) -> List[DecodedAccess]:
    """Every attribute/subscript/name access instruction in one code object.

    Pure: it compiles nothing and executes nothing. The tracer uses it to know
    which offsets are worth recording, and ``access_comparison`` uses it to know
    which accesses a code object is *capable* of -- the second use is what makes
    a falsified static edge provable without any runtime coverage at all.

    Nested code objects (lambdas, generator expressions, methods) are *not*
    descended into: they are separate callables with their own IDs, and
    ``iter_code_objects`` walks them separately.
    """
    decoded: List[DecodedAccess] = []
    try:
        instructions = list(dis.get_instructions(code))
    except Exception:  # pragma: no cover - a malformed code object must not kill a run
        return decoded

    previous: Optional[Any] = None
    for instruction in instructions:
        position = instruction.positions
        lineno = position.lineno if position and position.lineno is not None else 0
        col_offset = position.col_offset if position and position.col_offset is not None else -1

        def emit(tier: str, name: str, access: str, role: str, receiver: str = "") -> None:
            decoded.append(
                DecodedAccess(
                    offset=instruction.offset,
                    tier=tier,
                    name=name,
                    access=access,
                    role=role,
                    lineno=lineno,
                    col_offset=col_offset,
                    receiver=receiver,
                )
            )

        receiver_name = _receiver_name(previous)
        op = instruction.opcode
        if op in ATTR_READ_OPS:
            # From 3.12, ``LOAD_ATTR`` carries the old ``LOAD_METHOD`` in the low
            # oparg bit: set means ``x.f()``, clear means ``x.f``. The static side
            # never models the method name as a data access -- it records a read
            # of the *receiver* -- so the two must not be compared. Verified:
            # ``x.method(1)`` -> arg 1, ``x.attr`` -> arg 2. ``LOAD_SUPER_ATTR``
            # uses the same bit: ``super().run(1)`` -> arg 9, ``super().value`` -> arg 4.
            role = ROLE_METHOD if (instruction.arg or 0) & 1 else ROLE_ATTRIBUTE
            emit(TIER_ATTR, str(instruction.argval), ACCESS_READ, role, receiver_name)
        elif op in SUPER_ATTR_OPS:
            emit(TIER_ATTR, str(instruction.argval), ACCESS_READ, ROLE_SUPER)
        elif op in ATTR_WRITE_OPS:
            # ``self.x = v`` compiles to LOAD v, LOAD self, STORE_ATTR, so the
            # receiver is still the preceding instruction even though the value
            # was pushed first.
            emit(TIER_ATTR, str(instruction.argval), ACCESS_WRITE, ROLE_ATTRIBUTE, receiver_name)
        elif op in SLICE_READ_OPS:
            emit(TIER_KEY, COMPUTED_KEY, ACCESS_READ, ROLE_COMPUTED)
        elif op in SLICE_WRITE_OPS:
            emit(TIER_KEY, COMPUTED_KEY, ACCESS_WRITE, ROLE_COMPUTED)
        elif _is_subscript_read(instruction) or op in KEY_WRITE_OPS:
            access = ACCESS_READ if _is_subscript_read(instruction) else ACCESS_WRITE
            # The key is the last thing pushed before the subscript in every
            # form, so a literal key is always the immediately preceding
            # instruction. Verified: ``LOAD_CONST 'lit'`` at 42 -> ``BINARY_OP
            # 26`` at 44, and ``LOAD_CONST 'gone'`` at 68 -> ``DELETE_SUBSCR``
            # at 70.
            literals = (
                _const_key_names(previous)
                if previous is not None and previous.opcode in LOAD_CONST_OPS
                else []
            )
            if literals:
                for name in literals:
                    emit(TIER_KEY, name, access, ROLE_LITERAL)
            else:
                emit(TIER_KEY, COMPUTED_KEY, access, ROLE_COMPUTED)
        elif op in FAST_READ_OPS | GLOBAL_READ_OPS | DEREF_READ_OPS | NAME_READ_OPS:
            name = str(instruction.argval)
            emit(TIER_NAME, name, ACCESS_READ, _name_role(code, instruction, name))
        elif op in FAST_WRITE_OPS | GLOBAL_WRITE_OPS | DEREF_WRITE_OPS | NAME_WRITE_OPS:
            name = str(instruction.argval)
            emit(TIER_NAME, name, ACCESS_WRITE, _name_role(code, instruction, name))
        elif op in FAST_READ_PAIR_OPS:
            for name in _pair_names(instruction):
                emit(TIER_NAME, name, ACCESS_READ, _name_role(code, instruction, name))
        elif op in FAST_WRITE_PAIR_OPS:
            for name in _pair_names(instruction):
                emit(TIER_NAME, name, ACCESS_WRITE, _name_role(code, instruction, name))
        elif op in FAST_STORE_LOAD_OPS:
            names = _pair_names(instruction)
            if names:
                emit(TIER_NAME, names[0], ACCESS_WRITE, _name_role(code, instruction, names[0]))
            if len(names) > 1:
                emit(TIER_NAME, names[1], ACCESS_READ, _name_role(code, instruction, names[1]))

        previous = instruction

    return decoded


def _pair_names(instruction: Any) -> List[str]:
    value = instruction.argval
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _receiver_name(previous: Optional[Any]) -> str:
    """The plain name an attribute access is reached through, if there is one.

    ``self.data`` loads ``self`` immediately before ``LOAD_ATTR data``, so the
    receiver is nameable and the identity channel can find the object in the
    frame. ``a.b.c`` and ``xs[0].y`` are not: their receiver is an intermediate
    value that exists only on the operand stack. Returning ``""`` for those is
    what makes them excluded-and-counted rather than silently mis-attributed.
    """
    if previous is None:
        return ""
    if previous.opcode in FAST_READ_OPS | DEREF_READ_OPS | GLOBAL_READ_OPS | NAME_READ_OPS:
        return str(previous.argval)
    if previous.opcode in FAST_READ_PAIR_OPS:
        names = _pair_names(previous)
        # The pair pushes both, so the receiver is the one on top.
        return names[-1] if names else ""
    return ""


def iter_code_objects(code: Any) -> Iterable[Any]:
    """``code`` and every code object nested inside it, depth first.

    Lambdas and generator expressions are separate code objects whose
    ``co_qualname`` is ``<enclosing>.<locals>.<lambda>`` -- the same key
    ``call_graph.definitions.visit_Lambda`` uses. Comprehensions are *not*: PEP
    709 inlines them into the enclosing function from 3.12 on, so their accesses
    correctly belong to the enclosing callable.
    """
    yield code
    for const in code.co_consts:
        if hasattr(const, "co_qualname"):
            yield from iter_code_objects(const)


class AccessTracer:
    """Collect executed data accesses for code inside a known set of modules.

    **Why ``PY_START`` arms and ``INSTRUCTION`` records.** ``INSTRUCTION`` is a
    per-instruction event. Enabled globally it fires inside numpy, pytest and the
    whole standard library, which on a numerical model does not finish. So the
    tracer subscribes globally only to ``PY_START``, and the moment a code object
    from an analyzed module starts executing it switches ``INSTRUCTION`` on for
    *that code object alone* via ``set_local_events``.

    **Why every callback returns ``DISABLE``.** The question this instrument asks
    is "did this access site ever execute", never "how often". Retiring each
    offset on its first hit makes the cost proportional to the number of distinct
    instructions executed rather than to the number of executions, and -- unlike
    the call tracer's adaptive rule -- it loses nothing, because a bytecode offset
    has one fixed attribute name. There is no polymorphism to hide.

    A module already imported before the trace starts is missed, since its body
    never fires ``PY_START`` again. That is the same limitation the call tracer
    has.

    **Why identity mode does not retire on first hit.** An access site has one
    fixed attribute name, so seeing it once is seeing everything it can say. An
    *identity* site does not: the same ``LOAD_FAST x`` inside a loop holds a
    different object on every pass, and an alias that only shows up on the third
    iteration is still a real alias. So identity mode counts hits per offset and
    retires the offset at ``identity_samples``, which bounds the cost without
    reducing the question to the first execution.
    """

    def __init__(
        self,
        module_by_file: Mapping[str, str],
        *,
        driver: str = "",
        identity: bool = False,
        identity_samples: int = DEFAULT_IDENTITY_SAMPLES,
        tokens: Optional[ObjectTokens] = None,
    ):
        self.driver = driver
        self.identity = identity
        self.identity_samples = max(1, identity_samples)
        self._ids = CodeIdResolver(module_by_file)
        # code object -> offset -> the accesses decoded at that offset.
        self._armed: Dict[Any, Dict[int, List[DecodedAccess]]] = {}
        self._callable_ids: Dict[Any, str] = {}
        self._observed: Set[AccessKey] = set()
        self._executed: Set[str] = set()
        self._sites: Dict[AccessKey, Tuple[int, int]] = {}
        # Shared across drivers by ``trace_all`` so a token means the same object
        # in every one of them; a per-driver table would make the same object
        # look like two and turn real aliases into contradictions.
        self._tokens = tokens if tokens is not None else ObjectTokens()
        self._identities: Set[IdentityKey] = set()
        self._values: Set[ValueKey] = set()
        self._identity_sites_seen: Set[IdentitySite] = set()
        self._identity_exclusions: Dict[str, int] = {}
        self._hits: Dict[Any, Dict[int, int]] = {}
        self._offsets_at_cap = 0
        self._offsets_sampled = 0

    # -- callbacks -------------------------------------------------------

    def _exclude(self, reason: str) -> None:
        self._identity_exclusions[reason] = self._identity_exclusions.get(reason, 0) + 1

    def _on_start(self, code: Any, offset: int) -> Any:
        if code in self._armed:
            return sys.monitoring.DISABLE
        callable_id = self._ids.code_id(code)
        if callable_id is None:
            return sys.monitoring.DISABLE

        by_offset: Dict[int, List[DecodedAccess]] = {}
        for access in decode_access_instructions(code):
            by_offset.setdefault(access.offset, []).append(access)

        self._armed[code] = by_offset
        self._callable_ids[code] = callable_id
        self._executed.add(callable_id)
        if by_offset:
            # A code object with no access instructions has nothing to watch, and
            # instrumenting it would cost a callback per executed instruction for
            # no observations.
            sys.monitoring.set_local_events(TOOL_ID, code, sys.monitoring.events.INSTRUCTION)
        return sys.monitoring.DISABLE

    def _on_instruction(self, code: Any, offset: int) -> Any:
        by_offset = self._armed.get(code)
        if not by_offset:
            return sys.monitoring.DISABLE
        accesses = by_offset.get(offset)
        if not accesses:
            return sys.monitoring.DISABLE

        callable_id = self._callable_ids[code]
        for access in accesses:
            key = access.key(callable_id)
            self._observed.add(key)
            self._sites.setdefault(key, (access.lineno, access.col_offset))

        if not self.identity:
            return sys.monitoring.DISABLE

        frame = self._frame_for(code)
        if frame is None:
            self._exclude("frame not found")
        else:
            for access in accesses:
                site = access.identity_site(callable_id)
                if site is not None:
                    self._record_identity(site, access, frame)

        hits = self._hits.setdefault(code, {})
        count = hits.get(offset, 0) + 1
        hits[offset] = count
        if count == 1:
            self._offsets_sampled += 1
        if count >= self.identity_samples:
            # This offset's observation is now known to be partial. Counting them
            # is what tells a later reader whether the sample budget was still
            # shaping the answer -- the calibration note on
            # ``DEFAULT_IDENTITY_SAMPLES`` is why that matters.
            self._offsets_at_cap += 1
            return sys.monitoring.DISABLE
        return None

    # -- identity --------------------------------------------------------

    @staticmethod
    def _frame_for(code: Any) -> Any:
        """The frame currently executing ``code``.

        ``sys._getframe(1)`` is the answer in practice, but the callback's depth
        is an implementation detail of the monitoring machinery rather than a
        documented guarantee, so this walks a few frames to find the matching
        code object and gives up rather than guessing.
        """
        try:
            frame = sys._getframe(1)
        except ValueError:  # pragma: no cover - no caller frame
            return None
        for _ in range(4):
            if frame is None:
                return None
            if frame.f_code is code:
                return frame
            frame = frame.f_back
        return None

    _MISSING = object()

    def _lookup_name(self, frame: Any, name: str) -> Any:
        """The value bound to ``name`` in ``frame``, or ``_MISSING``.

        Locals first, then module globals: ``LOAD_NAME`` in a class body reads a
        namespace that is neither, and ``LOAD_GLOBAL`` reads the second. Builtins
        are deliberately not consulted -- ``len`` and ``range`` are not data, and
        the access channel already excludes them from scoring.
        """
        try:
            local_names = frame.f_locals
            if name in local_names:
                return local_names[name]
            globals_names = frame.f_globals
            if name in globals_names:
                return globals_names[name]
        except Exception:  # pragma: no cover - a hostile namespace must not kill a run
            return self._MISSING
        return self._MISSING

    def _record_identity(self, site: IdentitySite, access: DecodedAccess, frame: Any) -> None:
        self._identity_sites_seen.add(site)
        _callable_id, tier, _path = site

        if tier == TIER_NAME:
            value = self._lookup_name(frame, access.name)
            if value is self._MISSING:
                # Read before assignment on this path, or a name the frame does
                # not carry. Not an error, and not evidence of anything.
                self._exclude("name unbound")
                return
        else:
            receiver = self._lookup_name(frame, access.receiver)
            if receiver is self._MISSING:
                self._exclude("receiver unbound")
                return
            try:
                # ``__dict__`` rather than ``getattr``: a property or a
                # ``__getattr__`` would otherwise be executed by the act of
                # measuring, which changes the program under observation.
                namespace = object.__getattribute__(receiver, "__dict__")
            except Exception:
                namespace = None
            if not isinstance(namespace, dict) or access.name not in namespace:
                # __slots__, a C extension type, a descriptor, or simply an
                # attribute this instance does not carry.
                self._exclude("attribute not in instance dict")
                return
            value = namespace[access.name]

        if isinstance(value, str):
            # The section 1.7 channel. Strings get no token -- interning makes
            # their identity meaningless -- but their *value* is exactly what
            # decides whether two callables share a file.
            self._values.add((site[0], site[2], value[:MAX_VALUE_TEXT]))

        token = self._tokens.token(value)
        if token is None:
            self._exclude("value type excluded" if isinstance(value, VALUE_TYPES) else "retention cap")
            return
        self._identities.add((site[0], site[1], site[2], token))

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "AccessTracer":
        events = sys.monitoring.events
        sys.monitoring.use_tool_id(TOOL_ID, TOOL_NAME)
        sys.monitoring.register_callback(TOOL_ID, events.PY_START, self._on_start)
        sys.monitoring.register_callback(TOOL_ID, events.INSTRUCTION, self._on_instruction)
        # Offsets retired by a previous tracer stay retired unless asked to
        # resume, so each driver starts from a clean slate.
        sys.monitoring.restart_events()
        sys.monitoring.set_events(TOOL_ID, events.PY_START)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        events = sys.monitoring.events
        sys.monitoring.set_events(TOOL_ID, 0)
        for code in self._armed:
            try:
                sys.monitoring.set_local_events(TOOL_ID, code, 0)
            except Exception:  # pragma: no cover - a freed code object cannot be reset
                pass
        sys.monitoring.register_callback(TOOL_ID, events.PY_START, None)
        sys.monitoring.register_callback(TOOL_ID, events.INSTRUCTION, None)
        sys.monitoring.free_tool_id(TOOL_ID)

    def result(self) -> AccessTraceResult:
        return AccessTraceResult(
            observed=set(self._observed),
            executed_callables=set(self._executed),
            sites=dict(self._sites),
            code_objects_armed=len(self._armed),
            identities=set(self._identities),
            values=set(self._values),
            identity_sites_seen=set(self._identity_sites_seen),
            identity_exclusions=dict(self._identity_exclusions),
            identity_offsets_at_cap=self._offsets_at_cap,
            identity_offsets_sampled=self._offsets_sampled,
        )


def trace_driver(
    driver_name: str,
    invoke: Callable[[], Optional[str]],
    module_by_file: Mapping[str, str],
    *,
    identity: bool = False,
    identity_samples: int = DEFAULT_IDENTITY_SAMPLES,
    tokens: Optional[ObjectTokens] = None,
) -> AccessTraceResult:
    """Run one driver under the tracer and return what it observed."""
    tracer = AccessTracer(
        module_by_file,
        driver=driver_name,
        identity=identity,
        identity_samples=identity_samples,
        tokens=tokens,
    )
    error: Optional[str] = None
    with tracer:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            error = invoke()

    result = tracer.result()
    if error:
        result.driver_errors.append(f"{driver_name}: {error}")
    return result


def trace_all(
    module_by_file: Mapping[str, str],
    *,
    pytest_args: Sequence[str] = (),
    notebooks: Sequence[Path] = (),
    scripts: Sequence[Path] = (),
    identity: bool = False,
    identity_samples: int = DEFAULT_IDENTITY_SAMPLES,
    tokens: Optional[ObjectTokens] = None,
) -> Tuple[AccessTraceResult, ObjectTokens]:
    """Run every configured driver in turn and union what they observed.

    Drivers are independent: one that fails still contributes what it reached
    before failing, and the others run regardless. The token table is shared
    across all of them, so an object passed from a notebook into library code is
    one object rather than two.
    """
    try:
        from microservice_pipeline.call_graph.dynamic_trace import run_notebook, run_pytest, run_script
    except ImportError:  # pragma: no cover
        from call_graph.dynamic_trace import run_notebook, run_pytest, run_script  # type: ignore

    combined = AccessTraceResult()
    tokens = tokens if tokens is not None else ObjectTokens()
    options = {"identity": identity, "identity_samples": identity_samples, "tokens": tokens}

    if pytest_args:
        combined.merge(trace_driver("pytest", lambda: run_pytest(pytest_args), module_by_file, **options))

    for notebook in notebooks:
        combined.merge(
            trace_driver(
                f"notebook:{notebook.name}",
                lambda path=notebook: run_notebook(path),
                module_by_file,
                **options,
            )
        )

    for script in scripts:
        combined.merge(
            trace_driver(
                f"script:{script.name}",
                lambda path=script: run_script(path),
                module_by_file,
                **options,
            )
        )

    return combined, tokens


# -- artifacts -----------------------------------------------------------

ACCESS_FIELDS = ["callable", "tier", "name", "access", "role", "lineno", "col_offset"]
IDENTITY_FIELDS = ["callable", "tier", "path", "token"]
IDENTITY_VALUE_FIELDS = ["callable", "path", "value"]


def load_observed_accesses(path: Path) -> Set[AccessKey]:
    """Read ``dynamic_access.csv`` back into the set the comparison scores."""
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["callable"], row["tier"], row["name"], row["access"], row["role"])
            for row in csv.DictReader(handle)
        }


def load_observed_identities(path: Path) -> Set[IdentityKey]:
    """Read ``dynamic_identity.csv`` back into the set the comparison scores."""
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["callable"], row["tier"], row["path"], int(row["token"]))
            for row in csv.DictReader(handle)
        }


def load_observed_values(path: Path) -> Set[ValueKey]:
    """Read ``dynamic_identity_values.csv`` back, for the section 1.7 question."""
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["callable"], row["path"], row["value"]) for row in csv.DictReader(handle)}


def write_trace_outputs(outdir: Path, result: AccessTraceResult) -> None:
    try:
        from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json
    except ImportError:  # pragma: no cover
        from artifact_io import ensure_dir, write_csv_rows, write_json  # type: ignore

    ensure_dir(outdir)
    write_csv_rows(
        outdir / "dynamic_access.csv",
        ACCESS_FIELDS,
        [
            {
                "callable": key[0],
                "tier": key[1],
                "name": key[2],
                "access": key[3],
                "role": key[4],
                "lineno": result.sites.get(key, (0, -1))[0],
                "col_offset": result.sites.get(key, (0, -1))[1],
            }
            for key in sorted(result.observed)
        ],
    )
    # ``executed_callables`` is not derivable from the rows above -- a callable
    # that ran and touched nothing appears in neither -- and the comparison needs
    # it to scope the weak direction.
    write_json(
        outdir / "dynamic_access.json",
        {
            "executed_callables": sorted(result.executed_callables),
            "code_objects_armed": result.code_objects_armed,
            "observed_total": len(result.observed),
            "driver_errors": result.driver_errors,
        },
    )


def write_identity_outputs(outdir: Path, result: AccessTraceResult, tokens: ObjectTokens) -> None:
    """Write the identity channel's artifacts, beside the access channel's.

    Deliberately separate files rather than extra columns: an identity run
    samples each site up to ``identity_samples`` times and an access run retires
    it on first hit, so the two runs are not interchangeable and merging them
    into one artifact would let a reader compare numbers that were not measured
    the same way. Keeping them apart also means an identity run cannot overwrite
    the recorded Step 1a baseline.
    """
    try:
        from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json
    except ImportError:  # pragma: no cover
        from artifact_io import ensure_dir, write_csv_rows, write_json  # type: ignore

    ensure_dir(outdir)
    write_csv_rows(
        outdir / "dynamic_identity.csv",
        IDENTITY_FIELDS,
        [
            {"callable": key[0], "tier": key[1], "path": key[2], "token": key[3]}
            for key in sorted(result.identities)
        ],
    )
    write_csv_rows(
        outdir / "dynamic_identity_values.csv",
        IDENTITY_VALUE_FIELDS,
        [
            {"callable": key[0], "path": key[1], "value": key[2]}
            for key in sorted(result.values)
        ],
    )
    write_json(
        outdir / "dynamic_identity.json",
        {
            "executed_callables": sorted(result.executed_callables),
            # A site that ran but yielded nothing recordable is not the same as a
            # site that never ran, and only the comparison's "unobserved" bucket
            # can tell them apart if it is told which is which.
            "identity_sites_seen": sorted(list(site) for site in result.identity_sites_seen),
            "identity_total": len(result.identities),
            "value_total": len(result.values),
            "objects_retained": tokens.retained,
            "retention_cap": tokens.cap,
            # The number that makes this measurement able to fail. If it is true,
            # every recall figure below is computed on a truncated set and must
            # be reported as incomplete rather than as a result.
            "retention_cap_hit": tokens.cap_hit,
            "identity_offsets_sampled": result.identity_offsets_sampled,
            "identity_offsets_at_cap": result.identity_offsets_at_cap,
            "exclusions": dict(sorted(result.identity_exclusions.items())),
            "driver_errors": result.driver_errors,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    try:
        from microservice_pipeline.call_graph.discovery import (
            iter_analysis_files,
            iter_analysis_files_for_source_roots,
        )
        from microservice_pipeline.call_graph.dynamic_trace import _expand_notebooks
        from microservice_pipeline.config import load_extraction_config
    except ImportError:  # pragma: no cover
        from call_graph.discovery import iter_analysis_files, iter_analysis_files_for_source_roots  # type: ignore
        from call_graph.dynamic_trace import _expand_notebooks  # type: ignore
        from config import load_extraction_config  # type: ignore

    parser = argparse.ArgumentParser(
        description="Record the data accesses a project performs by executing it under sys.monitoring",
    )
    parser.add_argument("--config", default=None, type=Path, help="Extraction JSON/JSONC config")
    parser.add_argument("--root", default=None, type=Path, help="Python source root (when no config is given)")
    parser.add_argument("--module-prefix", default=None, help="Prefix for discovered module names")
    parser.add_argument("--outdir", default=None, type=Path, help="Output directory")
    parser.add_argument("--pytest-arg", action="append", default=[], help="Argument forwarded to in-process pytest. Repeatable.")
    parser.add_argument("--notebook", action="append", default=[], type=Path, help="Notebook to execute in-process. Repeatable.")
    parser.add_argument("--script", action="append", default=[], type=Path, help="Script to execute in-process. Repeatable.")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Also record which runtime object each site held, for scoring alias_of and lineage",
    )
    parser.add_argument(
        "--identity-samples",
        type=int,
        default=DEFAULT_IDENTITY_SAMPLES,
        help=(
            "Executions recorded per site in identity mode "
            f"(default {DEFAULT_IDENTITY_SAMPLES}). Calibrated, not chosen -- see DEFAULT_IDENTITY_SAMPLES."
        ),
    )
    parser.add_argument(
        "--identity-cap",
        type=int,
        default=500_000,
        help="Maximum distinct objects retained to keep ids from being recycled (default 500000)",
    )
    args = parser.parse_args(argv)

    if sys.version_info < (3, 12):
        raise SystemExit("trace-data-access requires Python 3.12+ (sys.monitoring / PEP 669)")

    pytest_args: List[str] = list(args.pytest_arg)
    notebooks: List[Path] = [path.resolve() for path in args.notebook]
    scripts: List[Path] = [path.resolve() for path in args.script]

    if args.config:
        config = load_extraction_config(args.config)
        analysis_files = iter_analysis_files_for_source_roots(
            config.source_roots,
            entrypoints=config.entrypoints,
            project_root=config.project_root,
            include_globs=config.include_globs,
            exclude_globs=config.exclude_globs,
        )
        # Deliberately the *call graph's* trace block. The drivers that exercise a
        # project are a property of the project, not of which analysis is being
        # scored, and a second copy would drift.
        trace_config = config.call_graph.trace
        pytest_args = pytest_args or list(trace_config.pytest_args)
        notebooks = notebooks or _expand_notebooks(config.project_root, trace_config.notebook_globs)
        scripts = scripts or [path.resolve() for path in trace_config.scripts]
        outdir = (args.outdir or config.data_access.outdir).resolve()
    else:
        if not args.root:
            raise SystemExit("--root is required unless --config is provided")
        analysis_files = list(iter_analysis_files(args.root.resolve(), module_prefix=args.module_prefix))
        outdir = (args.outdir or Path("artifacts/data_access")).resolve()

    if not (pytest_args or notebooks or scripts):
        raise SystemExit(
            "No drivers configured. Pass --pytest-arg/--notebook/--script, "
            "or set call_graph.trace in the extraction config."
        )

    module_by_file = module_map_from_analysis_files(analysis_files)
    result, tokens = trace_all(
        module_by_file,
        pytest_args=pytest_args,
        notebooks=notebooks,
        scripts=scripts,
        identity=args.identity,
        identity_samples=args.identity_samples,
        tokens=ObjectTokens(cap=args.identity_cap),
    )
    if args.identity:
        # An identity run samples sites repeatedly, so its access observations
        # are not comparable with a plain run's. Writing only the identity
        # artifacts keeps the Step 1a baseline untouched.
        write_identity_outputs(outdir, result, tokens)
    else:
        write_trace_outputs(outdir, result)

    by_role: Dict[Tuple[str, str], int] = {}
    for _callable_id, tier, _name, _access, role in result.observed:
        by_role[(tier, role)] = by_role.get((tier, role), 0) + 1

    print(f"Runtime data-access trace written to {outdir}")
    print(f"Modules in scope: {len(set(module_by_file.values()))}")
    print(f"Code objects armed: {result.code_objects_armed}")
    print(f"Callables entered: {len(result.executed_callables)}")
    print(f"Accesses observed: {len(result.observed)}")
    for tier, role in sorted(by_role):
        print(f"  {tier}/{role}: {by_role[(tier, role)]}")
    if args.identity:
        print(f"Identity observations: {len(result.identities)}")
        print(f"Identity sites reached: {len(result.identity_sites_seen)}")
        print(f"String values recorded: {len(result.values)}")
        print(f"Objects retained: {tokens.retained} of a {tokens.cap} cap")
        print(
            f"Sites that used their whole sample budget: "
            f"{result.identity_offsets_at_cap} of {result.identity_offsets_sampled}"
        )
        if tokens.cap_hit:
            print("  WARNING: retention cap hit -- identity results are incomplete")
        for reason, count in sorted(result.identity_exclusions.items()):
            print(f"  excluded, {reason}: {count}")
    for message in result.driver_errors:
        print(f"  driver issue: {message}")


if __name__ == "__main__":
    main()
