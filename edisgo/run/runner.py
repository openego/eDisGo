"""
Pipeline execution engine for the eDisGo runner.

This module ties the other three pieces — :mod:`edisgo.run.config`
(loader), :mod:`edisgo.run.validator` (static checks), and
:mod:`edisgo.run.registry` (task lookup) — together into a linear
stage-by-stage executor.

The execution model:

1. Load and validate the config.
2. Build a :class:`~edisgo.run.context.RunContext`.
3. For each stage, if the stage declares ``load_from: X``, reload
   the EDisGo object from stage ``X``'s save-artifact (topology +
   results only; time series are dropped to let the new stage set
   fresh ones).
4. For each step in the stage's pipeline, look up the task function
   in the registry and call it with the current EDisGo object and
   the context. A task may return a new EDisGo object (``setup_grid``,
   ``load_from_base``) which then replaces the current one.
5. Repeat for all stages, finally return the EDisGo object.

Two entry points are exposed:

* :func:`run_edisgo` — starts from no EDisGo object; the first task
  must create one (usually ``setup_grid``).
* :func:`_run_pipeline_on` — starts from an existing EDisGo instance;
  used by :meth:`edisgo.EDisGo.run_pipeline`.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from edisgo.run import tasks as _tasks  # noqa: F401 — triggers registration
from edisgo.run.config import load_config
from edisgo.run.context import RunContext
from edisgo.run.registry import get_task
from edisgo.run.validator import _split_step, validate

logger = logging.getLogger("edisgo.run.runner")


def run_edisgo(config, overlying_grid_data=None) -> Any:
    """
    Run an eDisGo pipeline from a YAML/JSON config or dict.

    This is the standalone entry point. The pipeline's first task is
    typically ``setup_grid`` or ``load_from_base`` to bootstrap the
    :class:`~edisgo.EDisGo` instance. If you already have one,
    prefer :meth:`edisgo.EDisGo.run_pipeline` instead.

    Parameters
    ----------
    config : str, pathlib.Path, or dict
        Path to a YAML/JSON pipeline config, or an in-memory dict of
        the same shape.

    Returns
    -------
    :class:`~edisgo.EDisGo`
        The EDisGo instance after the last stage has run. For
        multi-stage configs this is the object produced by the final
        stage.

    """
    return _run_pipeline_on(None, config, overlying_grid_data=overlying_grid_data)


def _run_pipeline_on(edisgo, config, overlying_grid_data=None):
    """
    Internal runner shared by :func:`run_edisgo` and the EDisGo method.

    Parameters
    ----------
    edisgo : edisgo.EDisGo or None
        Existing EDisGo instance to operate on, or ``None`` to have
        the first task create one.
    config : str, pathlib.Path, or dict
        Config to execute. Passed through to
        :func:`edisgo.run.config.load_config`.

    Returns
    -------
    edisgo.EDisGo
        The final EDisGo instance.

    Raises
    ------
    RuntimeError
        If a stage declares ``load_from: X`` but ``X`` produced no
        artifact (typically because validate() was skipped).

    """
    cfg = load_config(config)
    validate(cfg)
    ctx = _build_context(cfg)
    ctx.overlying_grid_data = overlying_grid_data

    for stage in cfg["stages"]:
        ctx.current_stage = stage["name"]
        ctx.logger.info(f"=== stage '{stage['name']}' ===")

        load_from = stage.get("load_from")
        if load_from is not None:
            artifact = ctx.stage_artifacts.get(load_from)
            if artifact is None:
                raise RuntimeError(
                    f"Stage '{stage['name']}' wants to load from "
                    f"'{load_from}' but no artifact is registered."
                )
            edisgo = _load_artifact(str(artifact))

        params = stage.get("params", {}) or {}
        for step in stage["pipeline"]:
            name, step_params = _split_step(step)
            step_params = _resolve_templating(step_params, params)
            ctx.logger.info(f"  -> task '{name}'")
            task_fn = get_task(name)
            result = task_fn(edisgo, ctx, **step_params)
            if result is not None:
                edisgo = result

    return edisgo


def _build_context(cfg: dict) -> RunContext:
    """
    Build a :class:`~edisgo.run.context.RunContext` from a config.

    Wires ``scenario`` and ``results.directory`` into the context and
    stores the full config under :attr:`RunContext.raw_config` so
    tasks can read supplementary sections.

    Parameters
    ----------
    cfg : dict
        Normalized config.

    Returns
    -------
    RunContext
        Initialized context with no engine, no artifacts, empty flags.

    """
    results_cfg = cfg.get("results") or {}
    results_dir = results_cfg.get("directory")
    return RunContext(
        scenario=cfg.get("scenario"),
        results_dir=Path(results_dir) if results_dir else None,
        raw_config=cfg,
    )


def _load_artifact(path: str):
    """
    Reload an EDisGo instance from a save-artifact for a ``load_from``.

    Loads topology + results only; time series and flex data are
    dropped so the consuming stage can set them fresh. Equipment
    changes are reset so the next stage's reinforce accounts only
    for its own scenario.

    Parameters
    ----------
    path : str
        Path to a directory or ``.zip`` produced by the ``save``
        task.

    Returns
    -------
    edisgo.EDisGo
        The restored EDisGo instance.

    """
    import pandas as pd

    from edisgo.edisgo import import_edisgo_from_files

    from_zip = path.endswith(".zip")
    edisgo = import_edisgo_from_files(
        edisgo_path=path,
        import_topology=True,
        import_timeseries=False,
        import_results=True,
        import_electromobility=False,
        import_heat_pump=False,
        import_dsm=False,
        import_overlying_grid=False,
        from_zip_archive=from_zip,
    )
    edisgo.legacy_grids = False
    edisgo.results.equipment_changes = pd.DataFrame()
    return edisgo


def _resolve_templating(step_params: dict, stage_params: dict) -> dict:
    """
    Substitute ``{{params.x}}`` placeholders in step parameters.

    Stage-level ``params:`` allows a preset to expose a few knobs that
    individual step parameters can reference. Only simple
    ``{{params.KEY}}`` expansions inside string values are supported
    (no filters, no conditionals, no nested expressions) — deliberately
    kept trivial to avoid a Jinja dependency.

    Parameters
    ----------
    step_params : dict
        Keyword arguments for a single step.
    stage_params : dict
        Stage-level ``params:`` dict.

    Returns
    -------
    dict
        ``step_params`` with template strings resolved.

    """
    if not stage_params or not step_params:
        return step_params
    out = {}
    for k, v in step_params.items():
        if isinstance(v, str) and "{{" in v:
            out[k] = _render_template(v, stage_params)
        else:
            out[k] = v
    return out


def _render_template(s: str, stage_params: dict) -> str:
    """
    Expand ``{{params.KEY}}`` references in a single string.

    Parameters
    ----------
    s : str
        Source string.
    stage_params : dict
        Mapping of stage-level parameters.

    Returns
    -------
    str
        Rendered string. Unknown keys are left in place (the original
        placeholder remains) so downstream errors point at the
        typo-ed key rather than silently turning into an empty
        string.

    """
    import re

    def repl(match):
        expr = match.group(1).strip()
        if expr.startswith("params."):
            key = expr.split(".", 1)[1]
            return str(stage_params.get(key, match.group(0)))
        return match.group(0)

    return re.sub(r"\{\{\s*([^}]+)\s*\}\}", repl, s)
