"""
YAML/JSON-driven pipeline runner for eDisGo.

Two entry points share the same core:

    from edisgo.run import run_edisgo
    edisgo = run_edisgo("presets/uc2_flex_opf.yaml")

    # or, on an existing EDisGo instance:
    edisgo = EDisGo(ding0_grid="30879")
    edisgo.run_pipeline("my_run.yaml")

Pipelines are lists of named tasks from :mod:`edisgo.run.tasks`. Each step
is either a string (``worst_case_ts``) or a single-key mapping with
parameters (``import_electromobility: {charging_strategy: dumb}``). Tasks
can be grouped into ordered ``stages`` that can save artifacts and reload
them with ``load_from``, enabling two-phase workflows (base reinforce +
per-scenario reinforce).
"""

from edisgo.run.context import RunContext
from edisgo.run.registry import known_tasks, register_task
from edisgo.run.runner import _run_pipeline_on, run_edisgo

__all__ = [
    "RunContext",
    "_run_pipeline_on",
    "known_tasks",
    "register_task",
    "run_edisgo",
]
