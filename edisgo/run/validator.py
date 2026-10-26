# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Static validator for pipeline configs.

The validator enforces structural and ordering rules that the runner
would otherwise hit at execution time — often after 20 minutes of work.
Running these checks up-front turns "cryptic AttributeError after half
the pipeline" into a clear ``ValueError`` at startup.

Checked rules (driven by the ``requires``/``provides``/``ts_altering``
metadata each task declares on registration):

* every step maps to a known, registered task name and passes only
  parameters the task accepts;
* ``reactive_power`` comes after every time-series task, never
  before — ``set_time_series_reactive_power_control`` overwrites
  reactive power on the currently set active-power time series;
* ``analyze`` and ``reinforce`` require a time-series task earlier in
  the pipeline;
* ``optimize`` requires both a time-series task and at least one flex
  import earlier in the pipeline — OPF without flexibility is
  meaningless;
* flex imports (``import_flex``, ``import_heat_pumps``, …) require a
  loaded grid, i.e. an earlier ``setup_grid`` / ``load_from_base``;
* the parameters of the time-step selection tasks are valid, and a
  ``timestep_selection`` block names a task of the pipeline; the task it
  skips is ignored by the rules above;
* a ``spatial_reduction`` block is a mapping and, if enabled, has a
  ``spatial_reduce`` step.
"""

from __future__ import annotations

import inspect

from typing import Any

from edisgo.run.registry import get_task, get_task_meta, known_tasks

# Human-readable message per required capability. The wording keeps the
# substrings the validator tests assert on ("loaded grid", "time series",
# "flex asset").
_REQUIREMENT_MESSAGES = {
    "grid": "requires a loaded grid (setup_grid or load_from_base) before it",
    "timeseries": (
        "requires time series to be set (e.g. worst_case_ts or oedb_ts) before it"
    ),
    "flex": "requires at least one flex asset to be imported",
}
# Order in which a missing capability is reported when several are missing.
_REQUIREMENT_PRIORITY = ("grid", "timeseries", "flex")


def validate(cfg: dict) -> None:
    """
    Validate a pipeline config against the ordering rules.

    This function does not return a value. On success it simply
    returns; on any rule violation it raises :class:`ValueError` with
    a message identifying the offending task.

    Parameters
    ----------
    cfg : dict
        Config as returned by :func:`edisgo.run.config.load_config`.
        Must have a ``pipeline`` list at the top level.

    Raises
    ------
    ValueError
        If the pipeline is empty, contains an unknown task name, or
        violates an ordering rule (reactive before TS, reinforce
        without TS, optimize without flex, flex import without grid,
        …), passes unknown task parameters, or if the time-step
        selection or ``spatial_reduction`` settings are invalid.

    """
    pipeline = cfg.get("pipeline") or []
    if not pipeline:
        raise ValueError("Config has no pipeline to run.")

    steps = [_split_step(step) for step in pipeline]
    known = set(known_tasks())
    for task_name, params in steps:
        if task_name not in known:
            raise ValueError(f"Unknown task '{task_name}'. Known: {sorted(known)}")
        _check_param_names(task_name, params)

    skipped = _check_timestep_selection(cfg, steps)
    satisfied: set[str] = set()
    reactive_set = False

    for task_name, _params in steps:
        if task_name in skipped:
            continue
        meta = get_task_meta(task_name)

        # reactive_power must be the last time-series-altering step.
        if meta.ts_altering and reactive_set:
            raise ValueError(
                f"Time-series task '{task_name}' comes after "
                f"'reactive_power' — reactive_power must be the last "
                f"time-series-altering step."
            )

        # Check declared requirements against what earlier tasks provide.
        missing = meta.requires - satisfied
        if missing:
            cap = next(
                (c for c in _REQUIREMENT_PRIORITY if c in missing),
                sorted(missing)[0],
            )
            detail = _REQUIREMENT_MESSAGES.get(
                cap, f"requires '{cap}' to be established before it"
            )
            raise ValueError(f"Task '{task_name}' {detail}.")

        satisfied |= meta.provides
        if task_name == "reactive_power":
            reactive_set = True

    spatial_reduction = cfg.get("spatial_reduction") or {}
    if not isinstance(spatial_reduction, dict):
        raise ValueError(
            f"spatial_reduction must be a mapping, got {spatial_reduction!r}."
        )
    if spatial_reduction.get("enabled") and "spatial_reduce" not in dict(steps):
        raise ValueError(
            "spatial_reduction is enabled, but the pipeline has no "
            "'spatial_reduce' step. Add one or set 'enabled: false'."
        )


def _check_param_names(task_name: str, params: dict) -> None:
    """
    Reject step params the task does not accept. Tasks taking ``**kwargs``
    check their params themselves.
    """
    signature = inspect.signature(get_task(task_name)).parameters
    if any(p.kind is p.VAR_KEYWORD for p in signature.values()):
        return
    accepted = set(signature) - {"edisgo", "ctx"}
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(
            f"Unknown parameters {sorted(unknown)} for task '{task_name}'. "
            f"Accepted: {sorted(accepted)}."
        )


def _check_timestep_selection(cfg: dict, steps: list) -> set:
    """
    Check the parameters of the time-step selection steps and the
    ``timestep_selection`` block (see :mod:`edisgo.run.tasks.timeseries`).

    Returns the selection tasks the block skips.

    """
    from edisgo.run.tasks.timeseries import (
        SELECTION_TASKS,
        check_selection_params,
        check_timestep_selection,
    )

    selection_steps = [
        (name, params) for name, params in steps if name in SELECTION_TASKS
    ]
    selection = cfg.get("timestep_selection")
    if selection is None:
        for name, params in selection_steps:
            check_selection_params(name, params)
        return set()

    check_timestep_selection(selection)
    for name, params in selection_steps:
        if params:
            raise ValueError(
                f"Step '{name}' has inline params, but a timestep_selection "
                f"block is set. Move them into timestep_selection.{name}, or "
                f"set timestep_selection to null to drop the block."
            )
    task = selection["task"]
    if task == "none":
        return set(SELECTION_TASKS)
    if task not in dict(selection_steps):
        raise ValueError(
            f"timestep_selection 'task' is {task!r}, but the pipeline has no "
            f"'{task}' step."
        )
    check_selection_params(task, selection.get(task) or {})
    return set(SELECTION_TASKS) - {task}


def _split_step(step: Any) -> tuple[str, dict]:
    """
    Normalize a pipeline step into ``(task_name, params)``.

    Steps are allowed in two forms in YAML/JSON:

    * bare string — ``worst_case_ts`` → ``("worst_case_ts", {})``
    * single-key mapping —
      ``import_electromobility: {charging_strategy: dumb}``
      → ``("import_electromobility", {"charging_strategy": "dumb"})``

    ``None`` as the parameter value is treated as an empty dict so
    that YAML's ``task:`` (with nothing after the colon) works.

    Parameters
    ----------
    step : str or dict
        Raw step as it appears in the pipeline list.

    Returns
    -------
    tuple of (str, dict)
        The task name and its keyword arguments.

    Raises
    ------
    ValueError
        If ``step`` is not a string or a single-key mapping, or if
        the parameter value is not a mapping.

    """
    if isinstance(step, str):
        return step, {}
    if isinstance(step, dict):
        if len(step) != 1:
            raise ValueError(
                f"Task step must be a string or single-key mapping, got: {step}"
            )
        ((name, params),) = step.items()
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(
                f"Parameters for task '{name}' must be a mapping, "
                f"got: {type(params).__name__}"
            )
        return name, params
    raise ValueError(f"Task step must be string or mapping, got: {step!r}")
