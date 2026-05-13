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
* ``results_dir`` — base directory for stage artifacts and ``save``.
* ``flags`` — free-form boolean/state flags tasks set to coordinate
  with each other (``has_heat_pumps``, ``timeseries_set``, …).
* ``stage_artifacts`` — map ``stage_name -> path`` of zip/dir artifacts
  emitted by ``save``, consumed by later stages via ``load_from``.

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
        Database engine for oedb-backed imports. Created lazily;
        see :meth:`ensure_engine`.
    results_dir : pathlib.Path or None
        Base directory for stage outputs. Resolved from
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
    stage_artifacts : dict
        Map ``stage_name -> Path`` of save-artifacts. Populated by the
        ``save`` task when running inside a named stage, consumed by
        subsequent stages that set ``load_from:``.
    current_stage : str or None
        Name of the stage currently executing. Set by the runner.
    raw_config : dict
        The fully resolved pipeline config (after ``extends``,
        ``external_config``, and eGo-legacy adaptation). Tasks can
        read supplementary keys like ``database.*`` from here.

    """

    scenario: str | None = None
    engine: Any = None
    results_dir: Path | None = None
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("edisgo.run")
    )
    flags: dict[str, Any] = field(default_factory=dict)
    stage_artifacts: dict[str, Path] = field(default_factory=dict)
    current_stage: str | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)

    def ensure_engine(self):
        """
        Return a database engine, creating it on first call.

        Reads the ``database`` section of :attr:`raw_config` and calls
        :func:`edisgo.io.db.engine`. Caches the engine on the context
        so subsequent calls reuse the same connection.

        Returns
        -------
        sqlalchemy.engine.Engine
            The active database engine.

        Raises
        ------
        RuntimeError
            If the config has no ``database`` section — indicates the
            pipeline wants to reach the database without configuring
            it.

        """
        if self.engine is not None:
            return self.engine
        db_cfg = self.raw_config.get("database")
        if not db_cfg:
            raise RuntimeError(
                "Task needs a database engine but no 'database' section "
                "is configured."
            )
        from edisgo.io.db import engine as egon_engine

        ssh_cfg = db_cfg.get("ssh") or {}
        self.engine = egon_engine(
            path=db_cfg.get("credentials_path"),
            ssh=bool(ssh_cfg.get("enabled", False)),
        )
        return self.engine
