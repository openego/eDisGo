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

* every step maps to a known, registered task name;
* ``reactive_power`` comes after every time-series task, never
  before — ``set_time_series_reactive_power_control`` overwrites
  reactive power on the currently set active-power time series;
* ``analyze`` and ``reinforce`` require a time-series task earlier in
  the pipeline;
* ``optimize`` requires both a time-series task and at least one flex
  import earlier in the pipeline — OPF without flexibility is
  meaningless;
* flex imports (``import_flex``, ``import_heat_pumps``, …) require a
  loaded grid, i.e. an earlier ``setup_grid`` / ``load_from_base``.
"""

from __future__ import annotations

from typing import Any

from edisgo.run.registry import get_task_meta, known_tasks

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
        …).

    """
    pipeline = cfg.get("pipeline") or []
    if not pipeline:
        raise ValueError("Config has no pipeline to run.")

    known = set(known_tasks())
    satisfied: set[str] = set()
    reactive_set = False

    for step in pipeline:
        task_name, _params = _split_step(step)
        if task_name not in known:
            raise ValueError(f"Unknown task '{task_name}'. Known: {sorted(known)}")

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
