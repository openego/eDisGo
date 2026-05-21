"""
Input/output tasks — persisting results and ingesting external files.

* :func:`task_save` (``save``) — persist topology, time series, and
  results to disk (directory or zip). Also publishes the artifact
  path into ``ctx.stage_artifacts`` so a later stage can
  ``load_from:``.
* :func:`task_load_charging_from_files`
  (``load_charging_from_files``) — R4MU-specific placeholder for
  integrating scenario charging stations from a directory of CSV /
  GeoPackage files; implementation is deferred until needed.
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

    If ``directory`` is not given, the artifact is written under
    ``ctx.results_dir / <stage_name>`` so every stage gets its own
    subdirectory. When ``archive=True`` the result is a single zip;
    the artifact path (including ``.zip``) is recorded in
    ``ctx.stage_artifacts[<stage_name>]`` so a downstream stage can
    declare ``load_from: <stage_name>``.

    Flags drive smart defaults for the optional ``save_*`` switches:
    if flex data is absent (per ``ctx.flags``), saving it is skipped.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to persist.
    ctx : RunContext
        Run context. Uses ``ctx.results_dir``, ``ctx.current_stage``,
        and reads ``has_heat_pumps`` / ``has_dsm`` /
        ``has_electromobility`` flags.
    directory : str, optional
        Absolute target directory. If omitted, derived from
        ``ctx.results_dir / ctx.current_stage``.
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
        stage = ctx.current_stage or "main"
        directory = os.path.join(str(ctx.results_dir), stage)

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

    saved_path = directory + (".zip" if archive else "")
    if ctx.current_stage:
        ctx.stage_artifacts[ctx.current_stage] = saved_path
    ctx.flags["last_saved"] = saved_path
    return edisgo


@register_task("load_charging_from_files")
def task_load_charging_from_files(
    edisgo, ctx, *, charging_dir, use_case_to_sector=None, mv_threshold_kw=100.0
):
    """
    Integrate scenario charging stations from files (R4MU workflow).

    PLACEHOLDER — the full implementation lives in eGo's
    ``_run_edisgo_task_load_charging_from_files`` and needs to be
    ported when R4MU is prioritised. The eGo version reads a
    GeoPackage / CSV of charging locations, filters by the MV grid
    district geometry, and integrates them into the topology via
    :func:`find_nearest_bus` / ``integrate_component_based_on_geolocation``
    with a use-case-to-sector mapping and an MV/LV connection
    threshold.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context.
    charging_dir : str
        Directory containing the charging-station source files.
    use_case_to_sector : dict, optional
        Maps raw use-case labels (``"home_detached"`` etc.) to
        eDisGo sector names (``"home"``, ``"work"``, …).
    mv_threshold_kw : float, optional
        Capacity threshold above which stations connect to an MV
        bus; below connect to LV.

    Raises
    ------
    NotImplementedError
        Always — port the eGo implementation before using.

    """
    raise NotImplementedError(
        "Task 'load_charging_from_files' is a placeholder port from "
        "eGo R4MU. Port the logic from eGo's "
        "_run_edisgo_task_load_charging_from_files when R4MU is "
        "needed."
    )


@register_task("import_overlying_grid_data")
def task_import_overlying_grid_data(edisgo, ctx, *, overlying_grid_path=None):
    """
    Import overlying grid data into the EDisGo instance.

    When ``overlying_grid_data`` is a dict of DataFrames (as returned by
    ``get_etrago_results_per_bus``), the overlying-grid attributes and
    dispatchable/fluctuating generator time series are set from it.

    When ``overlying_grid_path`` is a directory path, the overlying-grid
    attributes are loaded from CSV files in that directory, and
    ``dispatchable_generators_active_power.csv`` /
    ``renewables_potential.csv`` are applied as generator time series
    if present.

    Falls back to ``ctx.raw_config['eDisGo']['overlying_grid_source']``
    as the directory path when neither argument is given.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context.
    overlying_grid_path : str, optional
        Directory containing overlying-grid CSV files.
    overlying_grid_data : dict, optional
        Dict of DataFrames as returned by ``get_etrago_results_per_bus``.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    import pandas as pd

    overlying_grid_data = getattr(ctx, "overlying_grid_data", None)

    if overlying_grid_data is not None:
        # eTraGo results dict — set standard overlying-grid attributes
        for attr in edisgo.overlying_grid._attributes:
            if attr in overlying_grid_data:
                setattr(edisgo.overlying_grid, attr, overlying_grid_data[attr])
        # set generator time series
        edisgo.set_time_series_active_power_predefined(
            dispatchable_generators_ts=overlying_grid_data.get(
                "dispatchable_generators_active_power"
            ),
            fluctuating_generators_ts=overlying_grid_data.get("renewables_potential"),
        )
        return edisgo

    # resolve path: explicit arg → runner config overlying_grid.path → skip
    if overlying_grid_path is None:
        overlying_grid_path = (ctx.raw_config.get("overlying_grid") or {}).get("path")

    if overlying_grid_path is None:
        ctx.logger.warning(
            "task 'import_overlying_grid_data': no overlying_grid_data or "
            "overlying_grid_path provided — skipping."
        )
        return edisgo

    # load overlying-grid attributes from CSV directory
    edisgo.overlying_grid.from_csv(overlying_grid_path)

    # reindex overlying-grid attributes to match edisgo timeindex
    # CSVs may use a different year — shift year then reindex
    edisgo_ti = edisgo.timeseries.timeindex
    if not edisgo_ti.empty:
        # SOC needs one extra step at the end (end-of-period state)
        ti_freq = edisgo_ti.freq or (edisgo_ti[1] - edisgo_ti[0])
        edisgo_ti_plus1 = edisgo_ti.union([edisgo_ti[-1] + ti_freq])
        soc_attrs = {
            "storage_units_soc",
            "thermal_storage_units_decentral_soc",
            "thermal_storage_units_central_soc",
        }
        for attr in edisgo.overlying_grid._attributes:
            ts = getattr(edisgo.overlying_grid, attr)
            if ts.empty:
                continue
            csv_year = ts.index[0].year
            edisgo_year = edisgo_ti[0].year
            if csv_year != edisgo_year:
                ts.index = ts.index + pd.DateOffset(years=edisgo_year - csv_year)
            target_ti = edisgo_ti_plus1 if attr in soc_attrs else edisgo_ti
            setattr(edisgo.overlying_grid, attr, ts.reindex(target_ti))

    # load dispatchable generator and renewables time series from the same dir
    disp_path = os.path.join(
        overlying_grid_path, "dispatchable_generators_active_power.csv"
    )
    if os.path.isfile(disp_path):
        disp_ts = pd.read_csv(disp_path, index_col=0, parse_dates=True)
        if not edisgo_ti.empty:
            csv_year = disp_ts.index[0].year
            edisgo_year = edisgo_ti[0].year
            if csv_year != edisgo_year:
                disp_ts.index = disp_ts.index + pd.DateOffset(
                    years=edisgo_year - csv_year
                )
            disp_ts = disp_ts.reindex(edisgo_ti)
    else:
        disp_ts = None

    pot_path = os.path.join(overlying_grid_path, "renewables_potential.csv")
    if os.path.isfile(pot_path):
        pot_ts = pd.read_csv(pot_path, index_col=0, parse_dates=True)
        if not edisgo_ti.empty:
            csv_year = pot_ts.index[0].year
            edisgo_year = edisgo_ti[0].year
            if csv_year != edisgo_year:
                pot_ts.index = pot_ts.index + pd.DateOffset(
                    years=edisgo_year - csv_year
                )
            pot_ts = pot_ts.reindex(edisgo_ti)
    else:
        pot_ts = None

    if disp_ts is not None or pot_ts is not None:
        edisgo.set_time_series_active_power_predefined(
            dispatchable_generators_ts=disp_ts,
            fluctuating_generators_ts=pot_ts,
        )

    return edisgo
