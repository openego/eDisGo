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
    stage_artifacts: dict[str, Path] = field(default_factory=dict)
    current_stage: str | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)
    overlying_grid_data: Any = None
    full_grid_stash: Any = None

    def ensure_engine(self):
        """
        Return a database engine, creating it on first call.

        The data source is chosen from the ``database`` section of
        :attr:`raw_config`:

        * ``source: "local"`` — egon-data database via SSH tunnel, using
          ``config_path`` if given, otherwise the default location
          (``~/.ssh/egon-data.configuration.yaml``).
        * ``source: "oep"`` or no ``database`` section — remote Open Energy
          Platform (previous default behaviour).

        A legacy explicit direct-local database (``host`` given with SSH
        disabled) is still honoured for backward compatibility. The engine is
        cached on the context so subsequent calls reuse the same connection.

        Returns
        -------
        sqlalchemy.engine.Engine
            The active database engine.

        """
        if self.engine is not None:
            return self.engine
        db_cfg = self.raw_config.get("database") or {}
        source = str(db_cfg.get("source") or "").lower()

        # Legacy explicit direct local database: SSH disabled and explicit
        # connection parameters given (host/port/user/password as passed by
        # eGo). Connect straight to that postgres via psycopg2.
        ssh_cfg = db_cfg.get("ssh") or {}
        ssh_enabled = bool(ssh_cfg.get("enabled", False))
        host = db_cfg.get("host")
        if source not in ("local", "oep") and host and not ssh_enabled:
            from sqlalchemy import create_engine

            user = db_cfg.get("user")
            password = db_cfg.get("password")
            port = db_cfg.get("port")
            name = db_cfg.get("database_name") or db_cfg.get("database")
            self.logger.info(
                f"ensure_engine: using local database "
                f"{user}@{host}:{port}/{name} (no OEP, no SSH tunnel)."
            )
            self.engine = create_engine(
                f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}",
                connect_args={"connect_timeout": 10},
                # The engine is cached and reused across long-running tasks
                # (e.g. electromobility can idle the connection for many
                # minutes). pool_pre_ping detects connections the server/SSH
                # tunnel dropped while idle and transparently reconnects,
                # avoiding "server closed the connection unexpectedly".
                pool_pre_ping=True,
            )
            return self.engine

        # Source-driven engine: source="local" -> egon-data via SSH tunnel
        # (config_path or ~/.ssh default), source="oep"/absent -> OEP.
        from edisgo.io.db import engine_from_settings

        # A legacy ssh.enabled flag maps to source "local".
        if not source and ssh_enabled:
            db_cfg = {**db_cfg, "source": "local"}
        self.engine = engine_from_settings(db_cfg)
        return self.engine
