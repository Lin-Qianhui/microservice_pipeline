"""Access packaged config templates and built-in structural weight profiles."""

from __future__ import annotations

from importlib.resources import files


CONFIG_TEMPLATE_NAMES = (
    "extraction.jsonc",
    "structural_graph.jsonc",
    "structural_clustering.jsonc",
    "evaluation.jsonc",
    "notebook_task_analysis.jsonc",
    "shared_containers.jsonc",
    "manual_mapping.csv",
)

BUILTIN_WEIGHT_PROFILE_NAMES = (
    "default",
    "ownership_biased",
)


def config_template_resource(name: str):
    if name not in CONFIG_TEMPLATE_NAMES:
        raise ValueError(f"Unknown config template: {name}")
    return files("microservice_pipeline").joinpath("configs", "templates", name)


def builtin_weight_profile_resource(name: str):
    if name not in BUILTIN_WEIGHT_PROFILE_NAMES:
        available = ", ".join(f"builtin:{profile}" for profile in BUILTIN_WEIGHT_PROFILE_NAMES)
        raise ValueError(f"Unknown built-in weight profile: builtin:{name}. Available profiles: {available}")
    return files("microservice_pipeline").joinpath("configs", "weight_profiles", f"{name}.json")
