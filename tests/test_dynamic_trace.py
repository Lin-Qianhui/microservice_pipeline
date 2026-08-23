"""Tests for the runtime ground-truth tracer and the static/dynamic comparison.

The tracer's whole value depends on two properties that fail silently if broken:
its callable IDs must be byte-identical to the static graph's, and its
cost-saving disable rule must not hide polymorphic dispatch. Both are asserted
directly here rather than inferred from aggregate numbers.
"""

import importlib
import json
import sys

import pytest

from microservice_pipeline.call_graph.discovery import iter_analysis_files
from microservice_pipeline.call_graph.dynamic_trace import (
    CallTracer,
    module_map_from_analysis_files,
    run_notebook,
    trace_driver,
)
from microservice_pipeline.call_graph.generate_call_graph_ast import (
    build_call_graph_from_analysis_files,
)
from microservice_pipeline.call_graph.graph_comparison import StaticEdge, compare


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="sys.monitoring requires Python 3.12+"
)


# A registry framework whose wiring method is deliberately *not* named
# ``add_subprocess``, so the hardcoded climlab rule cannot fire. The static
# analyzer is expected to miss the parent->child compute edges here; the tracer
# is expected to find them. That contrast is the harness working as intended.
REGISTRY_SOURCE = '''
class ChildA:
    def run(self):
        return 1


class ChildB:
    def run(self):
        return 2


class ChildC:
    def run(self):
        return 3


class Parent:
    def __init__(self):
        self.children = {}

    def register(self, name, child):
        self.children[name] = child

    def run(self):
        total = 0
        for child in self.children.values():
            total += child.run()
        return total


def build():
    parent = Parent()
    parent.register("a", ChildA())
    parent.register("b", ChildB())
    parent.register("c", ChildC())
    for _ in range(200):
        parent.run()
    return parent
'''


def _write_registry_module(tmp_path, name="registry_sample"):
    (tmp_path / f"{name}.py").write_text(REGISTRY_SOURCE, encoding="utf-8")
    return name


def _trace_module(tmp_path, module_name, *, disable_after=20, entry="build"):
    analysis_files = list(iter_analysis_files(tmp_path))
    module_by_file = module_map_from_analysis_files(analysis_files)

    sys.path.insert(0, str(tmp_path))
    try:
        def drive():
            module = importlib.import_module(module_name)
            getattr(module, entry)()
            return None

        result = trace_driver("test", drive, module_by_file, disable_after=disable_after)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(module_name, None)
    return result


def test_dynamic_ids_match_static_node_ids(tmp_path):
    """Every traced endpoint must be a node the static pass also produced."""
    module_name = _write_registry_module(tmp_path)
    result = _trace_module(tmp_path, module_name)

    analysis_files = list(iter_analysis_files(tmp_path))
    nodes, _edges = build_call_graph_from_analysis_files(analysis_files)

    traced_ids = {edge.caller for edge in result.edges} | {edge.callee for edge in result.edges}
    assert traced_ids, "tracer produced no edges"
    assert traced_ids <= set(nodes), f"IDs absent from the static graph: {sorted(traced_ids - set(nodes))}"


def test_polymorphic_site_survives_adaptive_disable(tmp_path):
    """One call site, three registered classes -- all three must be recorded.

    ``disable_after=2`` against 200 iterations means a disable-on-first-hit rule
    would report exactly one callee. Anything less than three here would make
    the tracer blind to precisely the dispatch it exists to measure.
    """
    module_name = _write_registry_module(tmp_path)
    result = _trace_module(tmp_path, module_name, disable_after=2)

    children = {
        edge.callee
        for edge in result.edges
        if edge.caller == f"{module_name}.Parent.run"
    }
    assert children == {
        f"{module_name}.ChildA.run",
        f"{module_name}.ChildB.run",
        f"{module_name}.ChildC.run",
    }


def test_adaptive_disable_bounds_cost(tmp_path):
    """Hit counts must stay near the disable threshold, not track iterations."""
    module_name = _write_registry_module(tmp_path)
    result = _trace_module(tmp_path, module_name, disable_after=3)

    parent_run = max(
        edge.hits for edge in result.edges if edge.callee == f"{module_name}.Parent.run"
    )
    assert parent_run < 200, "call site was never retired; tracing cost is unbounded"
    assert result.disabled_sites > 0


def test_constructor_calls_normalize_to_dunder_init(tmp_path):
    """Runtime sees a class object; the static graph names ``Class.__init__``."""
    module_name = _write_registry_module(tmp_path)
    result = _trace_module(tmp_path, module_name)

    constructors = {
        (edge.callee, edge.callee_kind)
        for edge in result.edges
        if edge.callee_kind == "constructor"
    }
    assert (f"{module_name}.Parent.__init__", "constructor") in constructors


def test_generic_registry_dispatch_is_fully_resolved_statically(tmp_path):
    """Registry dispatch must resolve with no framework knowledge at all.

    The fixture's wiring method is called ``register``, not ``add_subprocess``,
    so no framework-specific rule can fire. The container dimension on class
    attributes is what carries the registered types from ``register`` to the
    loop in ``run``. This test used to assert the opposite -- that the static
    graph *missed* these edges -- and inverting it is the point of the work.
    """
    module_name = _write_registry_module(tmp_path)
    result = _trace_module(tmp_path, module_name)

    analysis_files = list(iter_analysis_files(tmp_path))
    _nodes, static_edges = build_call_graph_from_analysis_files(analysis_files)

    report = compare(
        [
            StaticEdge(caller=e.caller, callee=e.callee, relation=e.relation, resolved=e.resolved)
            for e in static_edges
        ],
        [(edge.caller, edge.callee) for edge in result.edges],
        result.executed_callables,
    )

    assert report.missing_from_static == set()
    assert report.recall == 1.0


def test_external_calls_are_counted_but_not_recorded_as_edges(tmp_path):
    (tmp_path / "uses_stdlib.py").write_text(
        """
import json


def go():
    return json.dumps({"a": 1})
""",
        encoding="utf-8",
    )
    result = _trace_module(tmp_path, "uses_stdlib", entry="go")

    assert result.external_calls > 0
    assert all("json" not in edge.callee for edge in result.edges)


def test_import_and_unresolved_static_edges_are_excluded_from_scoring():
    """Runtime cannot express these, so counting them would invent fake gaps."""
    static_edges = [
        StaticEdge(caller="pkg.a.<module>", callee="pkg.b.<module>", relation="import", resolved=True),
        StaticEdge(caller="pkg.a.go", callee="somelib.thing", relation="direct", resolved=False),
        StaticEdge(caller="pkg.a.go", callee="pkg.b.work", relation="direct", resolved=True),
    ]
    report = compare(static_edges, [("pkg.a.go", "pkg.b.work")], {"pkg.a.go"})

    assert report.comparable_static_count == 1
    assert report.recall == 1.0
    assert report.unconfirmed_static == set()


def test_precision_is_scoped_to_callers_that_actually_ran():
    """A static edge from code that never ran is unexercised, not wrong."""
    static_edges = [
        StaticEdge(caller="pkg.ran", callee="pkg.missed", relation="direct", resolved=True),
        StaticEdge(caller="pkg.never_ran", callee="pkg.other", relation="direct", resolved=True),
    ]
    report = compare(static_edges, [], {"pkg.ran"})

    assert report.unconfirmed_static == {("pkg.ran", "pkg.missed")}


def _write_notebook(tmp_path):
    (tmp_path / "lib.py").write_text(
        """
def inner_early():
    return "early"


def inner_late():
    return "late"


def touched():
    return inner_early()


def after_failure():
    return inner_late()
""",
        encoding="utf-8",
    )
    notebook = tmp_path / "demo.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": "import lib\nlib.touched()\n"},
                    {"cell_type": "code", "source": "raise ValueError('boom')\n"},
                    {"cell_type": "markdown", "source": "# heading\n"},
                    {"cell_type": "code", "source": "lib.after_failure()\n"},
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    return notebook


def _trace_notebook(tmp_path, notebook):
    analysis_files = list(iter_analysis_files(tmp_path))
    module_by_file = module_map_from_analysis_files(analysis_files)

    sys.path.insert(0, str(tmp_path))
    try:
        return trace_driver("notebook", lambda: run_notebook(notebook), module_by_file)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("lib", None)


def test_notebook_driver_continues_past_a_failing_cell(tmp_path):
    """A notebook that dies halfway must still yield the edges it reached."""
    notebook = _write_notebook(tmp_path)
    result = _trace_notebook(tmp_path, notebook)

    edges = {(edge.caller, edge.callee) for edge in result.edges}
    assert ("lib.touched", "lib.inner_early") in edges
    assert ("lib.after_failure", "lib.inner_late") in edges, "execution stopped at the failing cell"
    assert result.driver_errors, "cell failure should be reported"


def test_calls_made_directly_from_a_cell_are_not_project_edges(tmp_path):
    """A notebook cell is a driver, not project code the static graph analyzed.

    The cell calls ``lib.touched()`` directly. Recording that as an edge would
    invent a gap, because no static pass ever looked at the notebook. The
    function is still marked as executed so precision scoping can use it.
    """
    notebook = _write_notebook(tmp_path)
    result = _trace_notebook(tmp_path, notebook)

    assert all(edge.caller.startswith("lib.") for edge in result.edges)
    assert "lib.touched" in result.executed_callables


def test_tracer_releases_the_tool_id_on_exit(tmp_path):
    """A leaked tool ID would make the next trace in the same process fail."""
    tracer = CallTracer({}, driver="probe")
    with tracer:
        pass
    with CallTracer({}, driver="probe-again"):
        pass


def test_a_site_that_ran_and_dispatched_elsewhere_falsifies_the_edge():
    """The one direction in which a trace can call a static edge wrong.

    An unconfirmed edge is usually an untaken branch, which is why this module
    refuses to call one false. But if the trace watched the exact site and saw it
    dispatch somewhere else, "untaken branch" is ruled out.
    """
    static = [
        StaticEdge("pkg.run", "pkg.taken", "self_method", True, lineno=10, col_offset=4),
        StaticEdge("pkg.run", "pkg.invented", "self_method", True, lineno=10, col_offset=4),
    ]
    report = compare(
        static,
        [("pkg.run", "pkg.taken")],
        {"pkg.run"},
        observed_sites={("pkg.run", 10, 4): {"pkg.taken"}},
    )

    assert report.falsified_static == {("pkg.run", "pkg.invented")}
    assert report.falsified_by_relation["self_method"] == 1


def test_a_retired_site_can_never_falsify_anything():
    """The disable rule truncates a site's callee set, so absence proves nothing.

    Without this the cost-saving heuristic silently becomes a false accusation:
    the tracer stops watching after ``disable_after`` uneventful hits, and every
    callee that would have arrived later looks invented. On climlab this is the
    difference between 29 falsified edges and 7.
    """
    static = [
        StaticEdge("pkg.run", "pkg.rare", "virtual_override", True, lineno=10, col_offset=4),
    ]
    report = compare(
        static,
        [],
        {"pkg.run"},
        observed_sites={("pkg.run", 10, 4): {"pkg.common"}},
        disabled_positions={("pkg.run", 10, 4)},
    )

    assert report.falsified_static == set()
    assert report.matchable_site_count == 0


def test_an_edge_confirmed_from_another_site_is_not_falsified():
    """The same callee is often reached from several places."""
    static = [
        StaticEdge("pkg.run", "pkg.target", "self_method", True, lineno=10, col_offset=4),
    ]
    report = compare(
        static,
        [("pkg.run", "pkg.target")],
        {"pkg.run"},
        observed_sites={("pkg.run", 10, 4): {"pkg.other"}},
    )

    assert report.falsified_static == set()


def test_an_edge_with_no_site_is_never_falsified():
    """``registered_invoke`` is positioned in a different callable than its caller."""
    static = [
        StaticEdge("pkg.parent_hook", "pkg.child_hook", "registered_invoke", True),
    ]
    report = compare(
        static,
        [],
        {"pkg.parent_hook"},
        observed_sites={("pkg.parent_hook", 10, 4): {"pkg.other"}},
    )

    assert report.falsified_static == set()
    assert report.sited_static_count == 0


def test_call_sites_are_recorded_with_line_and_column(tmp_path):
    """A line alone is not a site: distinct calls routinely share one."""
    (tmp_path / "sited.py").write_text(
        """
def left():
    pass


def right():
    pass


def run():
    return (left(), right())
""",
        encoding="utf-8",
    )

    result = _trace_module(tmp_path, "sited", entry="run")

    sites = {
        site: callees
        for site, callees in result.observed_sites.items()
        if site[0] == "sited.run"
    }
    # Two calls on one line must be two sites, distinguished by column.
    assert len(sites) == 2
    assert {next(iter(c)) for c in sites.values()} == {"sited.left", "sited.right"}
    assert len({site[1] for site in sites}) == 1  # same line
    assert len({site[2] for site in sites}) == 2  # different columns
