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


def normalise_district_heating_labels(overlying_grid, logger):
    """
    Rewrite the district-heating column labels as the string of an integer.

    Both district-heating-indexed frames are addressed downstream by the district
    heating ID in that form ("130", never "130.0"): ``_build_heat_storage`` looks
    up ``thermal_storage_units_central_soc`` as
    ``loads_df.district_heating_id.astype(int).astype(str)``, and
    ``aggregate_district_heating_components`` matches ``feedin_district_heating``
    with ``str(int(district))``. eTraGo and CSV data arrive with float or integer
    labels, which silently miss (feed-in) or raise a KeyError (SoC).

    Shared by the two paths that can put overlying-grid data on an EDisGo object:
    ``import_overlying_grid_data`` and ``load_from_base``.

    Parameters
    ----------
    overlying_grid : edisgo.network.overlying_grid.OverlyingGrid
        Component whose frames are normalised in place.
    logger : logging.Logger
        Logger to report unreadable or colliding labels on.

    """
    for attr in ("feedin_district_heating", "thermal_storage_units_central_soc"):
        df = getattr(overlying_grid, attr)
        if df is None or df.empty:
            continue
        # Per column, so that one unreadable label does not leave every other
        # label of the frame unnormalised. The realistic producer of a single bad
        # label is a geo-join in eGo that yields NaN for a heat bus without a
        # matching district heating area.
        renamed, unreadable = [], []
        for col in df.columns:
            try:
                renamed.append(str(int(float(col))))
            except (TypeError, ValueError, OverflowError):
                # OverflowError is what inf raises, and it is not a subclass of
                # the other two -- without it an inf label kills the whole run.
                renamed.append(col)
                unreadable.append(col)
        if unreadable:
            logger.warning(
                f"Could not read {len(unreadable)} column label(s) of '{attr}' as "
                f"district heating IDs ({unreadable}) — leaving those unchanged. "
                f"Downstream lookups expect the ID as the string of an integer, so "
                f"this data will not be found."
            )
        if len(set(map(str, renamed))) != len(renamed):
            # Normalising can collapse distinct labels onto one another, e.g.
            # 130.0 next to "130", or the "130.1" pandas produces for a duplicate
            # CSV header. A duplicate label makes the downstream lookups return a
            # DataFrame where a Series is expected, which fails far from here.
            logger.warning(
                f"Normalising the column labels of '{attr}' ({list(df.columns)}) "
                f"would produce duplicates ({renamed}) — leaving them unchanged."
            )
            continue
        if renamed != list(df.columns):
            df = df.copy()
            df.columns = renamed
            setattr(overlying_grid, attr, df)


@register_task("import_overlying_grid_data", provides={"overlying_grid"})
def task_import_overlying_grid_data(edisgo, ctx, *, overlying_grid_path=None):
    """
    Import overlying grid data into the EDisGo instance.

    Behavior controlled by ``ctx.raw_config['overlying_grid']``:

    * ``enabled`` (bool) — master switch. Falsy → task no-ops.
    * ``source`` (str) — ``"etrago"`` or ``"csv"``.

    ``source: etrago`` consumes ``ctx.overlying_grid_data`` (a dict of
    DataFrames as returned by ``get_etrago_results_per_bus``), injected
    via the ``overlying_grid_data=`` kwarg of
    :func:`edisgo.run.run_edisgo`, via
    :meth:`~.network.overlying_grid.OverlyingGrid.from_etrago`. Sets
    overlying-grid attributes and dispatchable/fluctuating generator
    time series from it.

    ``source: csv`` loads overlying-grid attributes from CSVs in
    ``overlying_grid.path`` (full directory path for ONE grid — same
    leaf-dir convention as ``grid.ding0_path``; callers handling many
    grids must compose the per-grid subdirectory themselves) via
    :meth:`~.network.overlying_grid.OverlyingGrid.from_csv`.
    ``dispatchable_generators_active_power.csv`` and
    ``renewables_potential.csv``, if present in that dir, are applied
    as generator time series.

    Both ``OverlyingGrid`` methods above align the data they load onto
    ``edisgo.timeseries.timeindex`` themselves (see
    :func:`~.tools.tools.align_to_edisgo_timeindex`), warning on a year
    shift — this task contains no calendar-year logic of its own.

    District-heating column labels are normalised afterwards (see
    :func:`normalise_district_heating_labels`), so downstream consumers
    (e.g. the ``aggregate_district_heating`` task, which ``requires`` the
    ``overlying_grid`` this task ``provides``) can address areas by the
    string of an integer regardless of whether the source used float or
    integer labels.

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
    og_cfg = ctx.raw_config.get("overlying_grid") or {}
    if not og_cfg.get("enabled"):
        return edisgo

    source = og_cfg.get("source")
    overlying_grid_data = ctx.overlying_grid_data

    if source not in ("etrago", "csv"):
        ctx.logger.warning(
            f"task 'import_overlying_grid_data': unknown source={source!r} "
            "(expected 'etrago' or 'csv') — skipping."
        )
        return edisgo

    # load (and, internally, year-align) the overlying-grid attributes for
    # the chosen source
    if source == "etrago":
        if overlying_grid_data is None:
            ctx.logger.warning(
                "task 'import_overlying_grid_data': source='etrago' but no "
                "overlying_grid_data passed to run_edisgo — skipping."
            )
            return edisgo
        edisgo.overlying_grid.from_etrago(edisgo, overlying_grid_data)
    else:  # source == "csv"
        overlying_grid_path = overlying_grid_path or og_cfg.get("path")
        if overlying_grid_path is None:
            ctx.logger.warning(
                "task 'import_overlying_grid_data': source='csv' but no "
                "overlying_grid.path configured — skipping."
            )
            return edisgo
        edisgo.overlying_grid.from_csv(overlying_grid_path, edisgo_obj=edisgo)

    # normalise the district-heating column labels (eTraGo/CSV data may carry
    # float or integer labels; downstream consumers expect the string of an
    # integer)
    normalise_district_heating_labels(edisgo.overlying_grid, ctx.logger)

    # set dispatchable/fluctuating generator time series - both attributes
    # were already loaded and year-aligned onto edisgo.timeseries.timeindex
    # above, for either source
    disp_ts = edisgo.overlying_grid.dispatchable_generators_active_power
    pot_ts = edisgo.overlying_grid.renewables_potential
    if disp_ts is not None and not disp_ts.empty:
        edisgo.set_time_series_active_power_predefined(
            dispatchable_generators_ts=disp_ts,
        )
    if pot_ts is not None and not pot_ts.empty:
        edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts=pot_ts,
        )

    return edisgo
