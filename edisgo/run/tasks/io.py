"""
Input/output tasks — persisting results and ingesting external data.

* :func:`task_save` (``save``) — persist topology, time series, and
  results to disk (directory or zip).
* :func:`task_import_overlying_grid_data`
  (``import_overlying_grid_data``) — set overlying-grid requirements
  (e.g. eTraGo results) on the EDisGo instance, from the
  ``overlying_grid_data=`` kwarg of :func:`edisgo.run.run_edisgo` or
  from CSVs.
"""

from __future__ import annotations

import os

from edisgo.run.registry import register_task


@register_task("save")
def task_save(
    edisgo,
    ctx,
    *,
    directory=None,
    save_topology=True,
    save_timeseries=True,
    save_results=True,
    save_electromobility=None,
    save_opf_results=False,
    save_heatpump=None,
    save_overlying_grid=False,
    save_dsm=None,
    archive=False,
    archive_type="zip",
    reduce_memory=False,
    parameters=None,
):
    """
    Save the current EDisGo state to disk.

    If ``directory`` is not given, the output is written to
    ``ctx.results_dir``. When ``archive=True`` the result is a single
    zip; the written path (including ``.zip``) is recorded in
    ``ctx.flags['last_saved']``.

    Flags drive smart defaults for the optional ``save_*`` switches:
    if flex data is absent (per ``ctx.flags``), saving it is skipped.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to persist.
    ctx : RunContext
        Run context. Uses ``ctx.results_dir`` and reads
        ``has_heat_pumps`` / ``has_dsm`` / ``has_electromobility``
        flags.
    directory : str, optional
        Absolute target directory. If omitted, ``ctx.results_dir`` is
        used.
    save_topology : bool, optional
        Write the topology CSVs. Default ``True``.
    save_timeseries : bool, optional
        Write time-series CSVs. Default ``True``.
    save_results : bool, optional
        Write the results CSVs (equipment changes, expansion costs,
        etc.). Default ``True``.
    save_electromobility : bool or None, optional
        If ``None``, auto-enabled iff
        ``ctx.flags['has_electromobility']`` is truthy.
    save_opf_results : bool, optional
        Write OPF results if present.
    save_heatpump : bool or None, optional
        If ``None``, auto-enabled iff ``ctx.flags['has_heat_pumps']``
        is truthy.
    save_overlying_grid : bool, optional
        Write overlying-grid (eTraGo) specs if present.
    save_dsm : bool or None, optional
        If ``None``, auto-enabled iff ``ctx.flags['has_dsm']`` is
        truthy.
    archive : bool, optional
        Pack the directory into a single ``.zip`` archive.
    archive_type : str, optional
        Archive format (currently only ``"zip"``).
    reduce_memory : bool, optional
        Downcast float time-series to ``float32`` to save disk.
    parameters : dict, optional
        Fine-grained selection of which results fields to write,
        e.g. ``{"grid_expansion_results": ["equipment_changes"]}``.

    Returns
    -------
    edisgo.EDisGo
        The unchanged EDisGo instance.

    Raises
    ------
    ValueError
        If no ``directory`` is given and ``ctx.results_dir`` is also
        unset.

    """
    if directory is None:
        if ctx.results_dir is None:
            raise ValueError(
                "Task 'save' needs a 'directory' parameter or config.results.directory."
            )
        directory = str(ctx.results_dir)

    if save_heatpump is None:
        save_heatpump = ctx.flags.get("has_heat_pumps", False)
    if save_dsm is None:
        save_dsm = ctx.flags.get("has_dsm", False)
    if save_electromobility is None:
        save_electromobility = ctx.flags.get("has_electromobility", False)

    kwargs = dict(
        directory=directory,
        save_topology=save_topology,
        save_timeseries=save_timeseries,
        save_results=save_results,
        save_electromobility=save_electromobility,
        save_opf_results=save_opf_results,
        save_heatpump=save_heatpump,
        save_overlying_grid=save_overlying_grid,
        save_dsm=save_dsm,
    )
    if archive:
        kwargs["archive"] = True
        kwargs["archive_type"] = archive_type
    if reduce_memory:
        kwargs["reduce_memory"] = True
    if parameters is not None:
        kwargs["parameters"] = parameters

    edisgo.save(**kwargs)

    ctx.flags["last_saved"] = directory + (".zip" if archive else "")
    return edisgo


@register_task("import_overlying_grid_data")
def task_import_overlying_grid_data(edisgo, ctx, *, overlying_grid_path=None):
    """
    Import overlying grid data into the EDisGo instance.

    Behavior controlled by ``ctx.raw_config['overlying_grid']``:

    * ``enabled`` (bool) — master switch. Falsy → task no-ops.
    * ``source`` (str) — ``"etrago"`` or ``"csv"``.

    ``source: etrago`` consumes ``ctx.overlying_grid_data`` (a dict of
    DataFrames as returned by ``get_etrago_results_per_bus``), injected
    via the ``overlying_grid_data=`` kwarg of
    :func:`edisgo.run.run_edisgo`. Sets overlying-grid attributes and
    dispatchable/fluctuating generator time series from it.

    ``source: csv`` loads overlying-grid attributes from CSVs in
    ``overlying_grid.path`` (full directory path for ONE grid — same
    leaf-dir convention as ``grid.ding0_path``; callers handling many
    grids must compose the per-grid subdirectory themselves).
    ``dispatchable_generators_active_power.csv`` and
    ``renewables_potential.csv``, if present in that dir, are applied
    as generator time series.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Reads ``raw_config['overlying_grid']`` and
        ``overlying_grid_data`` attribute.
    overlying_grid_path : str, optional
        CSV directory override (takes precedence over
        ``overlying_grid.path`` from the config) when ``source='csv'``.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    import pandas as pd

    og_cfg = ctx.raw_config.get("overlying_grid") or {}
    if not og_cfg.get("enabled"):
        return edisgo

    source = og_cfg.get("source")
    overlying_grid_data = ctx.overlying_grid_data
    edisgo_ti = edisgo.timeseries.timeindex

    soc_attrs = {
        "storage_units_soc",
        "thermal_storage_units_decentral_soc",
        "thermal_storage_units_central_soc",
    }

    from edisgo.tools.tools import align_series_to_timeindex

    def _to_edisgo_timeindex(ts, extra_step=False):
        # bind the stage's edisgo time index to the shared aligner
        return align_series_to_timeindex(ts, edisgo_ti, extra_step=extra_step)

    if source not in ("etrago", "csv"):
        ctx.logger.warning(
            f"task 'import_overlying_grid_data': unknown source={source!r} "
            "(expected 'etrago' or 'csv') — skipping."
        )
        return edisgo

    # --- 1) load the overlying-grid attributes for the chosen source ---
    if source == "etrago":
        if overlying_grid_data is None:
            ctx.logger.warning(
                "task 'import_overlying_grid_data': source='etrago' but no "
                "overlying_grid_data passed to run_edisgo — skipping."
            )
            return edisgo
        for attr in edisgo.overlying_grid._attributes:
            if attr in overlying_grid_data:
                setattr(edisgo.overlying_grid, attr, overlying_grid_data[attr])
    else:  # source == "csv"
        overlying_grid_path = overlying_grid_path or og_cfg.get("path")
        if overlying_grid_path is None:
            ctx.logger.warning(
                "task 'import_overlying_grid_data': source='csv' but no "
                "overlying_grid.path configured — skipping."
            )
            return edisgo
        edisgo.overlying_grid.from_csv(overlying_grid_path)

    # --- 2) reindex the overlying-grid attributes onto the edisgo timeindex
    # (data may use a different year; SOC series carry one extra end step) ---
    for attr in edisgo.overlying_grid._attributes:
        ts = getattr(edisgo.overlying_grid, attr)
        if ts is None or ts.empty:
            continue
        setattr(
            edisgo.overlying_grid,
            attr,
            _to_edisgo_timeindex(ts, extra_step=attr in soc_attrs),
        )

    # --- 3) set dispatchable/fluctuating generator time series ---
    if source == "etrago":
        disp_ts = overlying_grid_data.get("dispatchable_generators_active_power")
        pot_ts = overlying_grid_data.get("renewables_potential")
        if disp_ts is not None and not disp_ts.empty:
            edisgo.set_time_series_active_power_predefined(
                dispatchable_generators_ts=disp_ts,
            )
        if pot_ts is not None and not pot_ts.empty:
            edisgo.set_time_series_active_power_predefined(
                fluctuating_generators_ts=_to_edisgo_timeindex(pot_ts),
            )
    else:  # source == "csv": load the two generator-TS CSVs from the dir

        def _load_generator_ts(filename):
            path = os.path.join(overlying_grid_path, filename)
            if not os.path.isfile(path):
                return None
            ts = pd.read_csv(path, index_col=0, parse_dates=True)
            return _to_edisgo_timeindex(ts)

        disp_ts = _load_generator_ts("dispatchable_generators_active_power.csv")
        pot_ts = _load_generator_ts("renewables_potential.csv")
        if disp_ts is not None or pot_ts is not None:
            edisgo.set_time_series_active_power_predefined(
                dispatchable_generators_ts=disp_ts,
                fluctuating_generators_ts=pot_ts,
            )

    return edisgo
