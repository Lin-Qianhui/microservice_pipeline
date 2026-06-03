"""Top-level command dispatcher for the packaged pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence


def _commands() -> dict[str, Callable[[Sequence[str] | None], None]]:
    from microservice_pipeline.call_graph.generate_call_graph_ast import main as call_graph_main
    from microservice_pipeline.cli.init_config import main as init_config_main
    from microservice_pipeline.cluster_call_graph import main as call_cluster_main
    from microservice_pipeline.cluster_structural_graph import main as structural_cluster_main
    from microservice_pipeline.data_access.generate_data_access_ast import main as data_access_main
    from microservice_pipeline.data_access.infer_shared_containers import main as infer_shared_containers_main
    from microservice_pipeline.evaluation.evaluate_microservice_clustering import main as evaluate_main
    from microservice_pipeline.notebook_tasks.analyze_notebook_tasks import main as notebook_tasks_main
    from microservice_pipeline.structural_dependency_graph.generate_structural_dependency_graph import (
        main as structural_graph_main,
    )

    return {
        "call-graph": call_graph_main,
        "call-cluster": call_cluster_main,
        "data-access": data_access_main,
        "evaluate": evaluate_main,
        "infer-shared-containers": infer_shared_containers_main,
        "init-config": init_config_main,
        "notebook-tasks": notebook_tasks_main,
        "structural-graph": structural_graph_main,
        "structural-cluster": structural_cluster_main,
    }


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = _commands()
    if argv and argv[0] in commands:
        commands[argv[0]](argv[1:])
        return

    parser = argparse.ArgumentParser(prog="microservice-pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(commands),
        help="Pipeline command to run",
    )
    args, remainder = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return
    commands[args.command](remainder)


if __name__ == "__main__":
    main()
