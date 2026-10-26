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
Pipeline execution engine for the eDisGo runner.

This module ties the other three pieces — :mod:`edisgo.run.config`
(loader), :mod:`edisgo.run.validator` (static checks), and
:mod:`edisgo.run.registry` (task lookup) — together into a linear
executor.

The execution model:

1. Load and validate the config.
2. Build a :class:`~edisgo.run.context.RunContext`.
3. For each step in the pipeline, look up the task function in the
   registry and call it with the current EDisGo object and the
   context. A task may return a new EDisGo object (``setup_grid``,
   ``load_from_base``) which then replaces the current one.
4. Return the final EDisGo object.

:func:`run_edisgo` is the single entry point. It starts from no
EDisGo object; the first task must create one (usually
``setup_grid`` or ``load_from_base``).
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

    The pipeline's first task is typically ``setup_grid`` or
    ``load_from_base`` to bootstrap the :class:`~edisgo.EDisGo`
    instance.

    Parameters
    ----------
    config : str, pathlib.Path, or dict
        Path to a YAML/JSON pipeline config, or an in-memory dict of
        the same shape.
    overlying_grid_data : dict, optional
        Overlying-grid data (e.g. eTraGo results) consumed by the
        ``import_overlying_grid_data`` task.

    Returns
    -------
    :class:`~edisgo.EDisGo`
        The EDisGo instance after the last task has run.

    """
    cfg = load_config(config)
    validate(cfg)
    ctx = _build_context(cfg)
    ctx.overlying_grid_data = overlying_grid_data

    edisgo = None
    for step in cfg["pipeline"]:
        name, step_params = _split_step(step)
        ctx.logger.info(f"-> task '{name}'")
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
        Loaded config.

    Returns
    -------
    RunContext
        Initialized context with no engine, empty flags.

    """
    results_cfg = cfg.get("results") or {}
    results_dir = results_cfg.get("directory")
    return RunContext(
        scenario=cfg.get("scenario"),
        results_dir=Path(results_dir) if results_dir else None,
        raw_config=cfg,
    )
