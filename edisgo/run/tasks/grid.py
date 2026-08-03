"""
Grid loading tasks — bring an EDisGo instance into existence.

Two ways to start a pipeline:

* :func:`task_setup_grid` (``setup_grid``) — read a ding0 topology
  from disk. This is the typical first step of every pipeline.
* :func:`task_load_from_base` (``load_from_base``) — reload a
  previously saved EDisGo instance. Used to split a computation into
  a slow "base" phase and one or more fast "scenario" phases that
  reuse the base-reinforced grid.
"""

from __future__ import annotations

import pandas as pd

from edisgo.run.registry import register_task


@register_task("setup_grid", provides={"grid"})
def task_setup_grid(
    edisgo,
    ctx,
    *,
    timeindex=None,
    ding0_path=None,
    legacy_ding0_grids=None,
    import_generators=False,
    generator_scenario=None,
):
    """
    Load a ding0 grid into an EDisGo instance.

    Creates the EDisGo object from the ding0 CSV directory — this is
    the typical first task of every pipeline.

    Parameters
    ----------
    edisgo : edisgo.EDisGo or None
        Current EDisGo instance, or ``None`` to create a fresh one.
    ctx : RunContext
        Run context. ``ctx.raw_config['grid']`` is consulted when
        parameters are not passed explicitly.
    ding0_path : str, optional
        Path to the ding0 grid directory. Falls back to
        ``ctx.raw_config['grid']['ding0_path']``.
    legacy_ding0_grids : bool, optional
        Whether to treat the ding0 directory as the legacy format.
        Falls back to ``ctx.raw_config['grid']['legacy_ding0_grids']``
        and ultimately to ``False``.
    import_generators : bool, optional
        If ``True``, call :meth:`EDisGo.import_generators` after
        loading the grid.
    generator_scenario : str, optional
        Generator scenario name passed to
        :meth:`EDisGo.import_generators` (only if
        ``import_generators=True``).

    Returns
    -------
    edisgo.EDisGo
        The EDisGo instance with the ding0 topology loaded.

    Raises
    ------
    ValueError
        If no ``ding0_path`` is given either as a task parameter or
        under ``config.grid.ding0_path``.

    """
    from edisgo import EDisGo

    grid_cfg = ctx.raw_config.get("grid", {})
    ding0_path = ding0_path or grid_cfg.get("ding0_path")
    if ding0_path is None:
        raise ValueError(
            "Task 'setup_grid' requires 'ding0_path' either as task "
            "parameter or under config.grid.ding0_path."
        )
    if legacy_ding0_grids is None:
        legacy_ding0_grids = grid_cfg.get("legacy_ding0_grids", False)

    if edisgo is None:
        edisgo = EDisGo(
            ding0_grid=str(ding0_path),
            legacy_ding0_grids=legacy_ding0_grids,
        )
    else:
        edisgo.import_ding0_grid(
            path=str(ding0_path), legacy_ding0_grids=legacy_ding0_grids
        )

    if import_generators:
        edisgo.import_generators(
            generator_scenario=generator_scenario,
            engine=ctx.ensure_engine(),
        )

    if timeindex is not None:
        ti_df = pd.date_range(
            start=timeindex["start"],
            periods=timeindex["periods"],
            freq=timeindex.get("freq", "h"),
        )
        edisgo.set_timeindex(ti_df)

    ctx.flags["grid_loaded"] = True
    return edisgo


def load_saved_edisgo(
    path,
    *,
    reset_equipment_changes=True,
    import_timeseries=False,
    import_results=False,
    import_electromobility=False,
    import_heat_pump=False,
    import_dsm=False,
    import_overlying_grid=False,
):
    """
    Reload a previously saved EDisGo object from a directory or ``.zip``.

    Backing implementation of the ``load_from_base`` task. Topology is
    always imported; time series and flex data default to off (the
    consuming pipeline sets them fresh). ``legacy_grids`` is cleared and,
    by default, ``results.equipment_changes`` is reset so a subsequent
    reinforce reflects only the current scenario.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory or ``.zip`` produced by the ``save`` task.
    reset_equipment_changes : bool, optional
        If ``True`` (default), clear ``results.equipment_changes``.
    import_timeseries, import_results, import_electromobility, \
    import_heat_pump, import_dsm, import_overlying_grid : bool, optional
        Which saved sub-datasets to import (all off by default except as
        overridden by the caller).

    Returns
    -------
    edisgo.EDisGo
        The restored EDisGo instance.

    """
    import os

    import pandas as pd

    from edisgo.edisgo import import_edisgo_from_files

    path = str(path)
    from_zip = path.endswith(".zip") or not os.path.isdir(path)
    edisgo = import_edisgo_from_files(
        edisgo_path=path,
        import_topology=True,
        import_timeseries=import_timeseries,
        import_results=import_results,
        import_electromobility=import_electromobility,
        import_heat_pump=import_heat_pump,
        import_dsm=import_dsm,
        import_overlying_grid=import_overlying_grid,
        from_zip_archive=from_zip,
    )
    edisgo.legacy_grids = False
    if reset_equipment_changes:
        edisgo.results.equipment_changes = pd.DataFrame()
    return edisgo


@register_task("load_from_base", provides={"grid"})
def task_load_from_base(
    edisgo,
    ctx,
    *,
    path=None,
    reset_equipment_changes=True,
    import_timeseries=False,
    import_results=False,
    import_electromobility=False,
    import_heat_pump=False,
    import_dsm=False,
    import_overlying_grid=False,
):
    """
    Reload an EDisGo instance from a previously saved directory/zip.

    Entry point for multi-phase workflows split over separate runner
    calls: a first run produces a (base-reinforced) grid and saves it,
    a later run starts from ``load_from_base`` to pick up that grid
    and apply scenario-specific modifications. The cost of the
    scenario then shows up cleanly in ``equipment_changes`` because we
    reset it on load.

    Parameters
    ----------
    edisgo : edisgo.EDisGo or None
        Unused — the task always replaces whatever was there.
    ctx : RunContext
        Run context (logger only).
    path : str
        Directory or ``.zip`` produced by :func:`task_save`.
    reset_equipment_changes : bool, optional
        If ``True`` (default), clear
        :attr:`Results.equipment_changes` so only the scenario's
        reinforce is tracked.
    import_timeseries : bool, optional
        Whether to import the saved time series. Default: ``False``
        so the next stage sets its own.
    import_results : bool, optional
        Whether to import saved results. Default: ``False``.
    import_electromobility : bool, optional
        Whether to import saved electromobility data.
    import_heat_pump : bool, optional
        Whether to import saved heat-pump data.
    import_dsm : bool, optional
        Whether to import saved DSM data.
    import_overlying_grid : bool, optional
        Whether to import saved overlying-grid data (eTraGo
        specifications).

    Returns
    -------
    edisgo.EDisGo
        The restored EDisGo instance.

    """
    if path is None:
        grid_cfg = ctx.raw_config.get("grid", {}) or {}
        path = grid_cfg.get("ding0_path")
    if path is None:
        raise ValueError(
            "Task 'load_from_base' requires 'path' either as task "
            "parameter or under config.grid.ding0_path."
        )
    edisgo = load_saved_edisgo(
        path,
        reset_equipment_changes=reset_equipment_changes,
        import_timeseries=import_timeseries,
        import_results=import_results,
        import_electromobility=import_electromobility,
        import_heat_pump=import_heat_pump,
        import_dsm=import_dsm,
        import_overlying_grid=import_overlying_grid,
    )
    ctx.flags["grid_loaded"] = True
    return edisgo
