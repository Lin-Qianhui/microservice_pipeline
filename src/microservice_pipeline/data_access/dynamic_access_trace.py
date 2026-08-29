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


# One observation, fully qualified: which callable, which tier, which name, read
# or write, and what kind of access it is.
AccessKey = Tuple[str, str, str, str, str]


@dataclass(frozen=True)
class DecodedAccess:
    """One access instruction found in a code object, executed or not."""

    offset: int
    tier: str
    name: str
    access: str
    role: str
    lineno: int
    col_offset: int

    def key(self, callable_id: str) -> AccessKey:
        return (callable_id, self.tier, self.name, self.access, self.role)


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

    def merge(self, other: "AccessTraceResult") -> None:
        self.observed |= other.observed
        self.executed_callables |= other.executed_callables
        for key, position in other.sites.items():
            self.sites.setdefault(key, position)
        self.code_objects_armed += other.code_objects_armed
        self.driver_errors.extend(other.driver_errors)


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

        def emit(tier: str, name: str, access: str, role: str) -> None:
            decoded.append(
                DecodedAccess(
                    offset=instruction.offset,
                    tier=tier,
                    name=name,
                    access=access,
                    role=role,
                    lineno=lineno,
                    col_offset=col_offset,
                )
            )

        op = instruction.opcode
        if op in ATTR_READ_OPS:
            # From 3.12, ``LOAD_ATTR`` carries the old ``LOAD_METHOD`` in the low
            # oparg bit: set means ``x.f()``, clear means ``x.f``. The static side
            # never models the method name as a data access -- it records a read
            # of the *receiver* -- so the two must not be compared. Verified:
            # ``x.method(1)`` -> arg 1, ``x.attr`` -> arg 2. ``LOAD_SUPER_ATTR``
            # uses the same bit: ``super().run(1)`` -> arg 9, ``super().value`` -> arg 4.
            role = ROLE_METHOD if (instruction.arg or 0) & 1 else ROLE_ATTRIBUTE
            emit(TIER_ATTR, str(instruction.argval), ACCESS_READ, role)
        elif op in SUPER_ATTR_OPS:
            emit(TIER_ATTR, str(instruction.argval), ACCESS_READ, ROLE_SUPER)
        elif op in ATTR_WRITE_OPS:
            emit(TIER_ATTR, str(instruction.argval), ACCESS_WRITE, ROLE_ATTRIBUTE)
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
    """

    def __init__(self, module_by_file: Mapping[str, str], *, driver: str = ""):
        self.driver = driver
        self._ids = CodeIdResolver(module_by_file)
        # code object -> offset -> the accesses decoded at that offset.
        self._armed: Dict[Any, Dict[int, List[DecodedAccess]]] = {}
        self._callable_ids: Dict[Any, str] = {}
        self._observed: Set[AccessKey] = set()
        self._executed: Set[str] = set()
        self._sites: Dict[AccessKey, Tuple[int, int]] = {}

    # -- callbacks -------------------------------------------------------

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
        if by_offset:
            accesses = by_offset.get(offset)
            if accesses:
                callable_id = self._callable_ids[code]
                for access in accesses:
                    key = access.key(callable_id)
                    self._observed.add(key)
                    self._sites.setdefault(key, (access.lineno, access.col_offset))
        return sys.monitoring.DISABLE

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
        )


def trace_driver(
    driver_name: str,
    invoke: Callable[[], Optional[str]],
    module_by_file: Mapping[str, str],
) -> AccessTraceResult:
    """Run one driver under the tracer and return what it observed."""
    tracer = AccessTracer(module_by_file, driver=driver_name)
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
) -> AccessTraceResult:
    """Run every configured driver in turn and union what they observed.

    Drivers are independent: one that fails still contributes what it reached
    before failing, and the others run regardless.
    """
    try:
        from microservice_pipeline.call_graph.dynamic_trace import run_notebook, run_pytest, run_script
    except ImportError:  # pragma: no cover
        from call_graph.dynamic_trace import run_notebook, run_pytest, run_script  # type: ignore

    combined = AccessTraceResult()

    if pytest_args:
        combined.merge(trace_driver("pytest", lambda: run_pytest(pytest_args), module_by_file))

    for notebook in notebooks:
        combined.merge(
            trace_driver(
                f"notebook:{notebook.name}",
                lambda path=notebook: run_notebook(path),
                module_by_file,
            )
        )

    for script in scripts:
        combined.merge(
            trace_driver(
                f"script:{script.name}",
                lambda path=script: run_script(path),
                module_by_file,
            )
        )

    return combined


# -- artifacts -----------------------------------------------------------

ACCESS_FIELDS = ["callable", "tier", "name", "access", "role", "lineno", "col_offset"]


def load_observed_accesses(path: Path) -> Set[AccessKey]:
    """Read ``dynamic_access.csv`` back into the set the comparison scores."""
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["callable"], row["tier"], row["name"], row["access"], row["role"])
            for row in csv.DictReader(handle)
        }


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
    result = trace_all(module_by_file, pytest_args=pytest_args, notebooks=notebooks, scripts=scripts)
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
    for message in result.driver_errors:
        print(f"  driver issue: {message}")


if __name__ == "__main__":
    main()
