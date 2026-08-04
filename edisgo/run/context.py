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
Runtime context passed to every task during pipeline execution.

The context is a small mutable object that threads shared state between
tasks without polluting the :class:`~edisgo.EDisGo` instance itself.
Typical uses:

* ``scenario`` — the active eGon scenario name (``eGon2035``,
  ``eGon100RE``, …) so tasks don't have to re-read it from the config.
* ``engine`` — a SQLAlchemy engine, lazily created on first DB access
  via :meth:`RunContext.ensure_engine`. Tasks that don't touch the
  database never pay connection cost.
* ``results_dir`` — base directory for the ``save`` task.
* ``flags`` — free-form boolean/state flags tasks set to coordinate
  with each other (``has_heat_pumps``, ``timeseries_set``, …).

Tasks should treat ``flags`` as advisory — they MAY short-circuit based
on a flag but MUST NOT assume a flag is present.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """
    Mutable per-run state shared across all tasks of a pipeline.

    Attributes
    ----------
    scenario : str or None
        Active scenario name from the top-level ``scenario:`` key.
    engine : sqlalchemy.engine.Engine or None
        Database engine for DB-backed imports. Created lazily;
        see :meth:`ensure_engine`.
    results_dir : pathlib.Path or None
        Base directory for outputs. Resolved from
        ``results.directory`` in the config.
    logger : logging.Logger
        Logger instance used by tasks and the runner. Defaults to
        the ``edisgo.run`` logger.
    flags : dict
        Free-form state flags that tasks use to communicate. Common
        keys: ``grid_loaded``, ``timeseries_set``,
        ``reactive_power_set``, ``has_heat_pumps``, ``has_dsm``,
        ``has_home_batteries``, ``has_electromobility``,
        ``base_reinforced``, ``last_saved``.
    raw_config : dict
        The fully resolved pipeline config (after ``extends``). Tasks
        can read supplementary sections like ``database`` or
        ``flexibilities`` from here.
    overlying_grid_data : dict or None
        Overlying-grid data (e.g. eTraGo results) injected via the
        ``overlying_grid_data=`` argument of :func:`edisgo.run.run_edisgo`.
        Consumed by the ``import_overlying_grid_data`` task when
        ``overlying_grid.source == "etrago"``.
    full_grid_stash : edisgo.EDisGo or None
        The pre-reduction :class:`~edisgo.EDisGo` instance, deepcopied and
        stashed by the ``spatial_reduce`` task before it spatially reduces
        the working object. Consumed (and cleared back to ``None``) by
        ``spatial_restore``. ``None`` when spatial reduction is not in use.

    """

    scenario: str | None = None
    engine: Any = None
    results_dir: Path | None = None
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("edisgo.run")
    )
    flags: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)
    overlying_grid_data: Any = None
    full_grid_stash: Any = None

    def ensure_engine(self):
        """
        Return a database engine, creating it on first call.

        The data source is resolved from the ``database`` section of
        :attr:`raw_config` by :func:`edisgo.io.db.engine_from_settings`:

        * ``source: "egon-data"`` — the egon-data database described by
          the configuration file (``config_path`` if given, otherwise
          the default location).
        * ``source: "oep"`` — the remote Open Energy Platform.
        * no ``database`` section / no ``source`` — auto-detect: the
          egon-data database if a configuration file is found at the
          default location, otherwise the OEP. The choice is logged.

        The engine is cached on the context so subsequent calls reuse
        the same connection.

        Returns
        -------
        sqlalchemy.engine.Engine
            The active database engine.

        """
        if self.engine is None:
            from edisgo.io.db import engine_from_settings

            self.engine = engine_from_settings(self.raw_config.get("database") or {})
        return self.engine
