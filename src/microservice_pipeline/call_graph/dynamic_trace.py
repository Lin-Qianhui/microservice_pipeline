"""Record the call graph a program *actually* executes, as ground truth.

The rest of this package infers edges by reading source code. This module gets
them the other way round: it runs the analyzed project under ``sys.monitoring``
(PEP 669, Python 3.12+) and records every caller/callee pair the interpreter
really dispatches. Comparing the two tells us which static gaps cost real edges,
instead of guessing from a list of known limitations.

Three things make the result comparable to the static graph:

* Callable IDs are built from the very same file-to-module mapping the static
  passes use, so ``climlab.model.ebm.EBM._compute`` means the same string on
  both sides. The mapping is *handed in* rather than recomputed -- see
  ``module_map_from_analysis_files``.
* Only code inside the analyzed source roots is recorded. Library and builtin
  frames are counted for the summary but are not edges.
* Drivers run **in-process**. A subprocess is invisible to ``sys.monitoring``,
  which is why neither pytest nor notebooks are shelled out.

The trace is a *lower bound*: it sees what the drivers exercised and nothing
else. That asymmetry is the whole reason ``graph_comparison`` treats recall and
precision differently.
"""

from __future__ import annotations

import os
import runpy
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from microservice_pipeline.call_graph.models import MODULE_CALLABLE_QUALNAME, AnalysisFile
except ImportError:  # pragma: no cover - supports direct script execution
    from models import MODULE_CALLABLE_QUALNAME, AnalysisFile  # type: ignore


# ``PROFILER_ID`` is the slot CPython reserves for exactly this kind of tool.
# Using it keeps us from colliding with a debugger or coverage tool that may be
# attached to the same interpreter.
TOOL_ID = sys.monitoring.PROFILER_ID
TOOL_NAME = "microservice-pipeline-trace"

# How many consecutive hits at one call site may pass without revealing a new
# callee before that site stops being monitored. See ``CallTracer`` for why this
# is not simply "disable on first hit".
DEFAULT_DISABLE_AFTER = 20

CALLEE_KIND_FUNCTION = "function"
CALLEE_KIND_METHOD = "method"
CALLEE_KIND_CONSTRUCTOR = "constructor"


@dataclass(frozen=True)
class DynamicEdge:
    """One caller/callee pair observed at runtime.

    ``hits`` counts *observed* dispatches, not total ones: monitoring at a call
    site is switched off once it stops producing new information, so treat the
    number as "at least this many" and never as a profile.
    """
    caller: str
    callee: str
    callee_kind: str
    driver: str
    hits: int


@dataclass
class TraceResult:
    edges: List[DynamicEdge] = field(default_factory=list)
    # Callables the tracer actually entered. ``graph_comparison`` needs this to
    # scope precision to code that ran, which is the only region where a static
    # edge's absence from the trace means anything at all.
    executed_callables: Set[str] = field(default_factory=set)
    # ``(caller id, line, column)`` -> the callees observed dispatching there.
    # This is what makes a static edge falsifiable: if the site ran and never
    # went to the claimed callee, the claim is wrong, not merely unexercised.
    observed_sites: Dict[Tuple[str, int, int], Set[str]] = field(default_factory=dict)
    # Sites whose monitoring was switched off early. Their observed callee set is
    # a lower bound even by this module's standards, so they are excluded from
    # falsification rather than trusted.
    disabled_positions: Set[Tuple[str, int, int]] = field(default_factory=set)
    external_calls: int = 0
    disabled_sites: int = 0
    driver_errors: List[str] = field(default_factory=list)

    def merge(self, other: "TraceResult") -> None:
        self.edges.extend(other.edges)
        self.executed_callables |= other.executed_callables
        for site, callees in other.observed_sites.items():
            self.observed_sites.setdefault(site, set()).update(callees)
        self.disabled_positions |= other.disabled_positions
        self.external_calls += other.external_calls
        self.disabled_sites += other.disabled_sites
        self.driver_errors.extend(other.driver_errors)


@dataclass
class _Site:
    """Per-call-site bookkeeping that drives the adaptive disable decision."""
    callees: Set[str] = field(default_factory=set)
    stable_hits: int = 0
    disabled: bool = False


def module_map_from_analysis_files(analysis_files: Sequence[AnalysisFile]) -> Dict[str, str]:
    """Map resolved source path -> module name, reusing the static file list.

    Deriving module names again here would be a second implementation of
    ``path_to_module`` that could drift from the first. Taking the already
    computed ``AnalysisFile`` records instead makes ID agreement with the static
    graph true by construction rather than by coincidence.
    """
    return {str(analysis_file.path.resolve()): analysis_file.module for analysis_file in analysis_files}


class CodeIdResolver:
    """Turn a runtime code object into the callable ID the static passes emit.

    Shared rather than copied. Every tracer in this pipeline compares its
    observations against artifacts keyed by these strings, so a second
    formulation of the rule would not fail loudly -- it would simply match
    nothing, and a recall of zero looks exactly like a recall of zero. That is
    the failure mode ``data_access/code_review.md`` 2.5 records against the three
    existing copies of the *static* callable-ID convention; there is no reason to
    open a fourth front on the runtime side.
    """

    def __init__(self, module_by_file: Mapping[str, str]):
        self.module_by_file = module_by_file
        # ``co_filename`` is a raw string that may be relative or contain
        # symlinks, so it needs resolving before lookup. Resolution is a syscall
        # and this runs on every unseen call site, hence the cache.
        self._file_cache: Dict[str, Optional[str]] = {}

    def module_for_filename(self, filename: str) -> Optional[str]:
        cached = self._file_cache.get(filename, "\0")
        if cached != "\0":
            return cached  # type: ignore[return-value]

        module = self.module_by_file.get(filename)
        if module is None:
            try:
                module = self.module_by_file.get(str(Path(filename).resolve()))
            except (OSError, ValueError):
                module = None
        self._file_cache[filename] = module
        return module

    def code_id(self, code: Any) -> Optional[str]:
        """Build ``module.qualname`` for a code object, or ``None`` if external."""
        module = self.module_for_filename(code.co_filename)
        if module is None:
            return None
        return callable_id_for_qualname(module, code.co_qualname)


def callable_id_for_qualname(module: str, qualname: str) -> str:
    """Join a module and a ``co_qualname`` the way the static passes do.

    Kept as a free function because the static bytecode index in
    ``data_access.access_comparison`` needs the same rule for code objects it
    compiled itself, where there is no tracer and no file-to-module cache.
    """
    if qualname == MODULE_CALLABLE_QUALNAME:
        # Matches ``discovery.module_callable_id``: module scope is a node.
        return f"{module}.{MODULE_CALLABLE_QUALNAME}" if module else MODULE_CALLABLE_QUALNAME
    return f"{module}.{qualname}"


class CallTracer:
    """Collect runtime call edges for code inside a known set of modules.

    **Why the disable rule is adaptive.** Returning ``sys.monitoring.DISABLE``
    from the callback stops the event firing at that one bytecode offset
    forever, which is what makes tracing a long-running numerical model
    affordable: cost becomes proportional to distinct call sites rather than to
    executions. But disabling on the first hit would quietly destroy the very
    measurement this module exists for. A polymorphic site such as
    ``proc.compute()`` inside ``climlab``'s ``_compute_type`` dispatches to a
    different subclass on each loop iteration; switching it off after the first
    would report one subclass and hide the rest -- precisely the container-
    mediated dispatch edges the static analyzer is suspected of missing.

    So each site keeps the set of callees seen there and is only retired once
    ``disable_after`` consecutive hits have produced nothing new.
    """

    def __init__(
        self,
        module_by_file: Mapping[str, str],
        *,
        driver: str = "",
        disable_after: int = DEFAULT_DISABLE_AFTER,
    ):
        self.module_by_file = module_by_file
        self.driver = driver
        self.disable_after = disable_after

        self._edges: Dict[Tuple[str, str, str], int] = {}
        self._executed: Set[str] = set()
        self._sites: Dict[Tuple[Any, int], _Site] = {}
        # (caller id, line, column) -> the callees dispatched from that site.
        self._observed_sites: Dict[Tuple[str, int, int], Set[str]] = {}
        # Sites retired by the adaptive disable rule. Their callee set is
        # incomplete by construction, so nothing may be concluded from a callee's
        # absence there.
        self._disabled_positions: Set[Tuple[str, int, int]] = set()
        self._positions: Dict[Any, List[Tuple[Any, Any, Any, Any]]] = {}
        self._external_calls = 0
        self._ids = CodeIdResolver(module_by_file)

    # -- ID normalization ------------------------------------------------

    def _module_for_filename(self, filename: str) -> Optional[str]:
        return self._ids.module_for_filename(filename)

    def _code_id(self, code: Any) -> Optional[str]:
        """Build ``module.qualname`` for a code object, or ``None`` if external."""
        return self._ids.code_id(code)

    def _callee_id(self, callable_: Any) -> Optional[Tuple[str, str]]:
        """Resolve a called object to ``(id, kind)``, or ``None`` if external.

        Classes are reported as their ``__init__``, which is what the static
        ``constructor`` relation emits. Because attribute lookup on the class
        already walks the MRO, an inherited ``__init__`` resolves to the class
        that actually defines it -- the same answer
        ``_resolve_constructor_targets`` produces statically.
        """
        try:
            kind = CALLEE_KIND_FUNCTION

            if isinstance(callable_, type):
                init = getattr(callable_, "__init__", None)
                code = getattr(init, "__code__", None)
                if code is None:
                    return None  # object.__init__ and other C slots
                callee_id = self._code_id(code)
                return (callee_id, CALLEE_KIND_CONSTRUCTOR) if callee_id else None

            # Bound/unbound methods, decorated wrappers, and partials all hide
            # the real function one level down.
            seen = 0
            while seen < 8:
                seen += 1
                if hasattr(callable_, "__func__"):
                    callable_ = callable_.__func__
                    kind = CALLEE_KIND_METHOD
                elif hasattr(callable_, "__wrapped__"):
                    callable_ = callable_.__wrapped__
                elif hasattr(callable_, "func") and not hasattr(callable_, "__code__"):
                    callable_ = callable_.func  # functools.partial
                else:
                    break

            code = getattr(callable_, "__code__", None)
            if code is None:
                return None  # builtin or C extension

            callee_id = self._code_id(code)
            if callee_id is None:
                return None
            if kind == CALLEE_KIND_FUNCTION and "." in code.co_qualname:
                kind = CALLEE_KIND_METHOD
            return (callee_id, kind)
        except Exception:
            # A callback that raises would be silently unregistered by
            # sys.monitoring, losing the rest of the trace. Never let that
            # happen because some object had an exotic __getattr__.
            return None

    # -- the callback ----------------------------------------------------

    def _position(self, code: Any, offset: int) -> Optional[Tuple[int, int]]:
        """The ``(line, column)`` of the call instruction at ``offset``.

        ``co_positions`` yields one entry per code unit, so the instruction at a
        byte offset is at index ``offset // 2``. The list is cached per code
        object because building it walks the whole position table, and a hot call
        site is visited many times.

        Column matters as much as line: on a real codebase roughly a third of
        call expressions share a line with another call, so a line alone names
        the wrong site often enough to invent failures.
        """
        positions = self._positions.get(code)
        if positions is None:
            try:
                positions = list(code.co_positions())
            except Exception:
                positions = []
            self._positions[code] = positions
        index = offset // 2
        if index >= len(positions):
            return None
        line, _end_line, column, _end_column = positions[index]
        if line is None or column is None:
            return None
        return (line, column)

    def _on_call(self, code: Any, offset: int, callable_: Any, arg0: Any) -> Any:
        callee = self._callee_id(callable_)
        if callee is None:
            self._external_calls += 1
            return None
        callee_id, callee_kind = callee

        # Being dispatched proves this callable ran, whoever called it. That
        # matters for ``executed_callables``, which scopes precision: a project
        # function reached only from a test or a notebook cell still has its own
        # static out-edges become checkable.
        self._executed.add(callee_id)

        caller_id = self._code_id(code)
        if caller_id is None:
            # The call came from outside the analyzed project -- a test body, a
            # notebook cell, or library code invoking a callback. The callee is
            # ours but the caller is not, so there is no *project* edge here and
            # the static graph was never expected to contain one. Recording it
            # would manufacture a gap that does not exist.
            return None

        self._executed.add(caller_id)

        key = (caller_id, callee_id, callee_kind)
        self._edges[key] = self._edges.get(key, 0) + 1

        # Where in the source the call was made, so a static edge can be checked
        # against the *site* it claims rather than only against the pair of
        # callables. That is what separates "this branch never ran" from "this
        # branch ran and never went there".
        position = self._position(code, offset)
        if position is not None:
            self._observed_sites.setdefault((caller_id, *position), set()).add(callee_id)

        site_key = (code, offset)
        site = self._sites.get(site_key)
        if site is None:
            site = _Site()
            self._sites[site_key] = site

        if callee_id in site.callees:
            site.stable_hits += 1
        else:
            site.callees.add(callee_id)
            site.stable_hits = 0  # a new callee restarts the patience counter

        if site.stable_hits >= self.disable_after:
            site.disabled = True
            if position is not None:
                # Recorded so the comparison cannot mistake "we stopped looking"
                # for "it never happened". Monitoring at this offset is now off
                # for good, so a callee that would have appeared on a later
                # iteration never will -- which would make an edge to it look
                # falsified when it is merely unobserved.
                self._disabled_positions.add((caller_id, *position))
            return sys.monitoring.DISABLE
        return None

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "CallTracer":
        sys.monitoring.use_tool_id(TOOL_ID, TOOL_NAME)
        sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.CALL, self._on_call)
        # Locations retired by a previous tracer stay retired unless asked to
        # resume, so each driver starts from a clean slate.
        sys.monitoring.restart_events()
        sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.CALL)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        sys.monitoring.set_events(TOOL_ID, 0)
        sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.CALL, None)
        sys.monitoring.free_tool_id(TOOL_ID)

    def result(self) -> TraceResult:
        return TraceResult(
            edges=[
                DynamicEdge(
                    caller=caller,
                    callee=callee,
                    callee_kind=kind,
                    driver=self.driver,
                    hits=hits,
                )
                for (caller, callee, kind), hits in self._edges.items()
            ],
            executed_callables=set(self._executed),
            observed_sites={
                site: set(callees) for site, callees in self._observed_sites.items()
            },
            disabled_positions=set(self._disabled_positions),
            external_calls=self._external_calls,
            disabled_sites=sum(1 for site in self._sites.values() if site.disabled),
        )


# -- drivers -------------------------------------------------------------
#
# Every driver must execute the target in *this* interpreter. sys.monitoring is
# per-process, so anything that forks a kernel or spawns a subprocess (nbclient's
# default executor, ``subprocess.run(["pytest", ...])``) would produce an empty
# trace while appearing to work.


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    """Run with ``path`` as cwd; notebooks routinely open relative data files."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _headless_matplotlib() -> Iterator[None]:
    """Force a non-interactive backend so plotting cells cannot block."""
    previous = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib  # noqa: F401

        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous


def run_pytest(args: Sequence[str]) -> Optional[str]:
    """Run pytest in-process. Returns an error description, or ``None``."""
    try:
        import pytest
    except ImportError:
        return "pytest is not installed; install the 'trace' extra"

    try:
        exit_code = pytest.main(list(args))
    except Exception as exc:  # a plugin blowing up should not kill the trace
        return f"pytest raised {type(exc).__name__}: {exc}"

    # Test failures are expected and still yield useful edges (climlab's
    # Fortran-dependent tests fail on import of the extension), so a non-zero
    # exit is reported but not treated as fatal.
    if exit_code not in (0, 1):
        return f"pytest exited with code {exit_code}"
    return None


def _notebook_code_cells(path: Path) -> List[str]:
    import json

    try:
        from microservice_pipeline.notebook_tasks.analyze_notebook_tasks import (
            _sanitize_code_cell,
            _source,
        )
    except ImportError:  # pragma: no cover
        from notebook_tasks.analyze_notebook_tasks import _sanitize_code_cell, _source  # type: ignore

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        _sanitize_code_cell(_source(cell))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def run_notebook(path: Path) -> Optional[str]:
    """Execute a notebook's code cells in-process, tolerating cell failures.

    Cells are compiled and executed one at a time in a shared namespace, which
    mirrors how a kernel runs them. A cell that raises does not end the run: a
    notebook that dies two thirds of the way through still contributes every
    edge it reached, and partial coverage is strictly better than none.
    """
    namespace: Dict[str, Any] = {"__name__": "__main__", "__file__": str(path)}
    failures = 0
    cells = _notebook_code_cells(path)

    with _pushd(path.parent), _headless_matplotlib():
        for index, source in enumerate(cells):
            if not source.strip():
                continue
            try:
                compiled = compile(source, f"{path.name}[cell {index}]", "exec")
                exec(compiled, namespace)  # noqa: S102 - executing the analyzed project is the point
            except Exception:
                failures += 1
            except SystemExit:
                break

    if failures:
        return f"{path.name}: {failures}/{len(cells)} cells failed"
    return None


def run_script(path: Path) -> Optional[str]:
    try:
        with _pushd(path.parent), _headless_matplotlib():
            runpy.run_path(str(path), run_name="__main__")
    except Exception as exc:
        return f"{path.name} raised {type(exc).__name__}: {exc}"
    return None


def trace_driver(
    driver_name: str,
    invoke: Callable[[], Optional[str]],
    module_by_file: Mapping[str, str],
    *,
    disable_after: int = DEFAULT_DISABLE_AFTER,
) -> TraceResult:
    """Run one driver under the tracer and return what it observed."""
    tracer = CallTracer(module_by_file, driver=driver_name, disable_after=disable_after)
    error: Optional[str] = None
    with tracer:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            error = invoke()

    result = tracer.result()
    if error:
        result.driver_errors.append(f"{driver_name}: {error}")
    return result


# -- orchestration and CLI -----------------------------------------------

EDGE_FIELDS = ["caller", "callee", "callee_kind", "driver", "hits"]

# Columns of ``dynamic_sites.csv``. ``lineno``/``col_offset`` are the position of
# the call instruction in the *caller*, matching ``Edge.lineno``/``col_offset``
# on the static side so the two can be joined on the site rather than the pair.
SITE_FIELDS = ["caller", "lineno", "col_offset", "callee"]


def trace_all(
    module_by_file: Mapping[str, str],
    *,
    pytest_args: Sequence[str] = (),
    notebooks: Sequence[Path] = (),
    scripts: Sequence[Path] = (),
    disable_after: int = DEFAULT_DISABLE_AFTER,
) -> TraceResult:
    """Run every configured driver in turn and union what they observed.

    Drivers are independent: one that fails still contributes the edges it
    reached before failing, and the others run regardless.
    """
    combined = TraceResult()

    if pytest_args:
        combined.merge(
            trace_driver("pytest", lambda: run_pytest(pytest_args), module_by_file, disable_after=disable_after)
        )

    for notebook in notebooks:
        combined.merge(
            trace_driver(
                f"notebook:{notebook.name}",
                lambda path=notebook: run_notebook(path),
                module_by_file,
                disable_after=disable_after,
            )
        )

    for script in scripts:
        combined.merge(
            trace_driver(
                f"script:{script.name}",
                lambda path=script: run_script(path),
                module_by_file,
                disable_after=disable_after,
            )
        )

    return combined


def write_trace_outputs(outdir: Path, result: TraceResult) -> None:
    try:
        from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json
    except ImportError:  # pragma: no cover
        from artifact_io import ensure_dir, write_csv_rows, write_json  # type: ignore

    ensure_dir(outdir)
    write_csv_rows(
        outdir / "dynamic_edges.csv",
        EDGE_FIELDS,
        [
            {
                "caller": edge.caller,
                "callee": edge.callee,
                "callee_kind": edge.callee_kind,
                "driver": edge.driver,
                "hits": edge.hits,
            }
            for edge in sorted(result.edges, key=lambda item: (item.caller, item.callee, item.driver))
        ],
    )
    # One row per (site, callee) actually dispatched. Kept out of
    # ``dynamic_edges.csv`` because that file is keyed by callable pair: the same
    # pair reached from two different sites is one edge but two sites, and it is
    # the site that decides whether a static claim was falsified.
    write_csv_rows(
        outdir / "dynamic_sites.csv",
        SITE_FIELDS,
        [
            {"caller": caller, "lineno": line, "col_offset": col, "callee": callee}
            for (caller, line, col), callees in sorted(result.observed_sites.items())
            for callee in sorted(callees)
        ],
    )
    # ``executed_callables`` is not derivable from the edge list -- a callable
    # that ran but called nothing internal appears nowhere in it -- and the
    # comparison step needs it to scope precision, so it is persisted here.
    write_json(
        outdir / "dynamic_trace.json",
        {
            "executed_callables": sorted(result.executed_callables),
            "disabled_positions": [
                list(position) for position in sorted(result.disabled_positions)
            ],
            "external_calls": result.external_calls,
            "disabled_sites": result.disabled_sites,
            "driver_errors": result.driver_errors,
        },
    )


def _expand_notebooks(project_root: Path, globs: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    seen: Set[Path] = set()
    for pattern in globs:
        for path in sorted(project_root.glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen and resolved.suffix == ".ipynb":
                seen.add(resolved)
                found.append(resolved)
    return found


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    try:
        from microservice_pipeline.call_graph.discovery import (
            iter_analysis_files,
            iter_analysis_files_for_source_roots,
        )
        from microservice_pipeline.config import load_extraction_config
    except ImportError:  # pragma: no cover
        from call_graph.discovery import iter_analysis_files, iter_analysis_files_for_source_roots  # type: ignore
        from config import load_extraction_config  # type: ignore

    parser = argparse.ArgumentParser(
        description="Record a runtime call graph by executing the project under sys.monitoring",
    )
    parser.add_argument("--config", default=None, type=Path, help="Extraction JSON/JSONC config")
    parser.add_argument("--root", default=None, type=Path, help="Python source root (when no config is given)")
    parser.add_argument("--module-prefix", default=None, help="Prefix for discovered module names")
    parser.add_argument("--outdir", default=None, type=Path, help="Output directory")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Argument forwarded to an in-process pytest run. Repeatable.",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        type=Path,
        help="Notebook to execute in-process. Repeatable.",
    )
    parser.add_argument(
        "--script",
        action="append",
        default=[],
        type=Path,
        help="Python script to execute in-process. Repeatable.",
    )
    parser.add_argument(
        "--disable-after",
        default=None,
        type=int,
        help="Consecutive hits without a new callee before a call site stops being monitored",
    )
    args = parser.parse_args(argv)

    if sys.version_info < (3, 12):
        raise SystemExit("trace-runtime requires Python 3.12+ (sys.monitoring / PEP 669)")

    pytest_args: List[str] = list(args.pytest_arg)
    notebooks: List[Path] = [path.resolve() for path in args.notebook]
    scripts: List[Path] = [path.resolve() for path in args.script]
    disable_after = args.disable_after or DEFAULT_DISABLE_AFTER

    if args.config:
        config = load_extraction_config(args.config)
        analysis_files = iter_analysis_files_for_source_roots(
            config.source_roots,
            entrypoints=config.entrypoints,
            project_root=config.project_root,
            include_globs=config.include_globs,
            exclude_globs=config.exclude_globs,
        )
        trace_config = config.call_graph.trace
        pytest_args = pytest_args or list(trace_config.pytest_args)
        notebooks = notebooks or _expand_notebooks(config.project_root, trace_config.notebook_globs)
        scripts = scripts or [path.resolve() for path in trace_config.scripts]
        if args.disable_after is None:
            disable_after = trace_config.disable_after
        outdir = (args.outdir or config.call_graph.outdir).resolve()
    else:
        if not args.root:
            raise SystemExit("--root is required unless --config is provided")
        analysis_files = list(iter_analysis_files(args.root.resolve(), module_prefix=args.module_prefix))
        outdir = (args.outdir or Path("artifacts/call_graph")).resolve()

    if not (pytest_args or notebooks or scripts):
        raise SystemExit(
            "No drivers configured. Pass --pytest-arg/--notebook/--script, "
            "or set call_graph.trace in the extraction config."
        )

    module_by_file = module_map_from_analysis_files(analysis_files)
    result = trace_all(
        module_by_file,
        pytest_args=pytest_args,
        notebooks=notebooks,
        scripts=scripts,
        disable_after=disable_after,
    )
    write_trace_outputs(outdir, result)

    distinct = {(edge.caller, edge.callee) for edge in result.edges}
    print(f"Runtime trace written to {outdir}")
    print(f"Modules in scope: {len(set(module_by_file.values()))}")
    print(f"Distinct runtime edges: {len(distinct)} (rows: {len(result.edges)})")
    print(f"Callables entered: {len(result.executed_callables)}")
    print(f"Calls outside source roots: {result.external_calls}")
    print(f"Call sites retired early: {result.disabled_sites}")
    for message in result.driver_errors:
        print(f"  driver issue: {message}")


if __name__ == "__main__":
    main()
