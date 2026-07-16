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
Time-series tasks — set active/reactive power profiles on EDisGo.

Time series drive every downstream step: ``analyze``, ``reinforce``
and ``optimize`` all operate on the time index and power time series
attached to the EDisGo object. The order inside a stage matters:

1. Set the time index and active-power profiles with one of
   :func:`task_worst_case_ts`, :func:`task_oedb_ts`,
   :func:`task_manual_ts`, possibly :func:`task_set_timeindex`.
2. Optionally reduce the time index to a selected subset with
   :func:`task_select_timesteps` (manual before the imports, auto after
   ``import_overlying_grid_data``).
3. Finally call :func:`task_reactive_power` to fix reactive power
   control — this MUST come last because it overwrites whatever
   reactive power was set by the earlier steps.
"""

from __future__ import annotations

import pandas as pd

from edisgo.run.registry import register_task


@register_task("worst_case_ts", provides={"timeseries"}, ts_altering=True)
def task_worst_case_ts(
    edisgo,
    ctx,
    *,
    cases=None,
    generators_names=None,
    loads_names=None,
    storage_units_names=None,
):
    """
    Set synthetic worst-case active-power time series.

    Produces two snapshots (load case and feed-in case) that
    represent the network's extremes. Useful for a coarse first
    reinforce that does not require real load/generation data.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Sets ``ctx.flags['timeseries_set'] = True``.
    cases : list of str, optional
        Subset of ``{"load_case", "feed-in_case"}``. Default is both.
    generators_names : list of str, optional
        Restrict to these generator names; default is all.
    loads_names : list of str, optional
        Restrict to these load names; default is all.
    storage_units_names : list of str, optional
        Restrict to these storage units; default is all.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    edisgo.set_time_series_worst_case_analysis(
        cases=cases,
        generators_names=generators_names,
        loads_names=loads_names,
        storage_units_names=storage_units_names,
    )
    ctx.flags["timeseries_set"] = True
    return edisgo


@register_task("set_timeindex", provides={"timeseries"}, ts_altering=True)
def task_set_timeindex(edisgo, ctx, *, start, periods=None, end=None, freq="h"):
    """
    Set the time index on the EDisGo object.

    Useful as a stand-alone step when you want a specific hourly
    range without immediately attaching time-series data (the
    ``oedb_ts`` task already accepts a ``timeindex`` argument and
    does this internally).

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context.
    start : str or pandas.Timestamp
        First timestamp of the range.
    periods : int, optional
        Number of periods; mutually exclusive with ``end``.
    end : str or pandas.Timestamp, optional
        Last timestamp; mutually exclusive with ``periods``.
    freq : str, optional
        pandas frequency string, default hourly (``"h"``).

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    Raises
    ------
    ValueError
        If neither ``periods`` nor ``end`` is provided.

    """
    from edisgo.tools.tools import reduce_timeseries_data_to_given_timeindex

    if end is not None:
        timeindex = pd.date_range(start=start, end=end, freq=freq)
    else:
        if periods is None:
            raise ValueError("set_timeindex needs either 'periods' or 'end'.")
        timeindex = pd.date_range(start=start, periods=periods, freq=freq)
    if edisgo.timeseries.timeindex.empty:
        edisgo.set_timeindex(timeindex)
    else:
        reduce_timeseries_data_to_given_timeindex(edisgo, timeindex)
    return edisgo


@register_task("oedb_ts", provides={"timeseries"}, ts_altering=True)
def task_oedb_ts(
    edisgo,
    ctx,
    *,
    timeindex=None,
    dispatchable=None,
    fluctuating="oedb",
    conventional_loads="oedb",
    charging_points_ts=None,
):
    """
    Set active-power time series from egon_data (OEP) plus overrides.

    This is the "real data" path: wind and solar profiles come from
    ``egon_era5_renewable_feedin``, conventional loads come from the
    egon demand tables. Dispatchable generators (conventional,
    etc.) are set via a per-technology-type profile since egon_data
    does not dispatch them. Storage units default to zero if not
    already set.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Uses ``ctx.scenario`` and
        ``ctx.ensure_engine()`` when any source is ``"oedb"``. Sets
        ``ctx.flags['timeseries_set'] = True``.
    timeindex : dict, optional
        ``{"start": ..., "periods": N, "freq": "h"}``. If present, a
        matching :class:`~pandas.DatetimeIndex` is set before
        importing data.
    dispatchable : dict, optional
        Per-technology scaling factors, e.g. ``{"other": 0.7}`` →
        constant profile of 0.7 p.u. for all non-fluctuating
        generators of type "other".
    fluctuating : str or pandas.DataFrame, optional
        How to populate wind/solar. ``"oedb"`` pulls egon_data,
        ``"default"`` uses bundled standard profiles, or a DataFrame
        with columns "solar" / "wind" is passed through.
    conventional_loads : str, optional
        Source for conventional loads (not heat pumps / charging
        points). ``"oedb"`` or ``"demandlib"``.
    charging_points_ts : pandas.DataFrame, optional
        Explicit active-power profile for charging points; default
        ``None`` leaves them untouched so
        :func:`task_apply_charging_strategy` can set them.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    if timeindex is not None:
        ti_df = pd.date_range(
            start=timeindex["start"],
            periods=timeindex["periods"],
            freq=timeindex.get("freq", "h"),
        )
        edisgo.set_timeindex(ti_df)
    elif edisgo.timeseries.timeindex.empty:
        # No explicit timeindex and none set yet (e.g. no manual time series
        # earlier): fall back to a full year derived from the scenario, the
        # same default the flex imports use.
        from edisgo.tools.tools import get_year_based_on_scenario

        year = get_year_based_on_scenario(ctx.scenario)
        if year is None:
            raise ValueError(
                f"Cannot derive a default time index: invalid scenario "
                f"{ctx.scenario!r}. Provide a 'timeindex' or a valid scenario "
                f"('eGon2035', 'eGon100RE')."
            )
        edisgo.set_timeindex(pd.date_range(f"1/1/{year}", periods=8760, freq="h"))

    dispatchable_df = None
    if dispatchable is not None:
        ti = edisgo.timeseries.timeindex
        dispatchable_df = pd.DataFrame(dispatchable, index=ti)

    conv_loads_names = None
    if conventional_loads == "oedb":
        conv_loads_names = edisgo.topology.loads_df.loc[
            ~edisgo.topology.loads_df.type.isin(["heat_pump", "charging_point"])
        ].index.tolist()

    edisgo.set_time_series_active_power_predefined(
        fluctuating_generators_ts=fluctuating,
        conventional_loads_ts=conventional_loads,
        conventional_loads_names=conv_loads_names,
        dispatchable_generators_ts=dispatchable_df,
        charging_points_ts=charging_points_ts,
        scenario=ctx.scenario,
        engine=ctx.ensure_engine()
        if fluctuating == "oedb" or conventional_loads == "oedb"
        else None,
    )

    su_names = edisgo.topology.storage_units_df.index
    if len(su_names) > 0 and edisgo.timeseries.storage_units_active_power.empty:
        edisgo.timeseries.storage_units_active_power = pd.DataFrame(
            0.0,
            index=edisgo.timeseries.timeindex,
            columns=su_names,
        )
    ctx.flags["timeseries_set"] = True
    return edisgo


@register_task("manual_ts", provides={"timeseries"}, ts_altering=True)
def task_manual_ts(
    edisgo,
    ctx,
    *,
    generators_active_power=None,
    generators_reactive_power=None,
    loads_active_power=None,
    loads_reactive_power=None,
    storage_units_active_power=None,
    storage_units_reactive_power=None,
):
    """
    Set active/reactive power time series from explicit DataFrames.

    Used when the caller already has the raw profiles (e.g. from a
    coupled run) and wants to inject them directly. Any argument left
    at ``None`` is not touched.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Sets ``ctx.flags['timeseries_set'] = True``.
    generators_active_power : dict or pandas.DataFrame, optional
        Generator active-power profile(s). Converted via
        :class:`pandas.DataFrame`.
    generators_reactive_power : dict or pandas.DataFrame, optional
        Generator reactive-power profile(s).
    loads_active_power : dict or pandas.DataFrame, optional
        Load active-power profile(s).
    loads_reactive_power : dict or pandas.DataFrame, optional
        Load reactive-power profile(s).
    storage_units_active_power : dict or pandas.DataFrame, optional
        Storage-unit active-power profile(s).
    storage_units_reactive_power : dict or pandas.DataFrame, optional
        Storage-unit reactive-power profile(s).

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """

    def _as_df(obj):
        return pd.DataFrame(obj) if obj is not None else None

    edisgo.set_time_series_manual(
        generators_p=_as_df(generators_active_power),
        generators_q=_as_df(generators_reactive_power),
        loads_p=_as_df(loads_active_power),
        loads_q=_as_df(loads_reactive_power),
        storage_units_p=_as_df(storage_units_active_power),
        storage_units_q=_as_df(storage_units_reactive_power),
    )
    ctx.flags["timeseries_set"] = True
    return edisgo


def _set_default_full_year_timeindex(edisgo, ctx):
    """
    Set a full-year hourly time index derived from the scenario.

    Used as a fallback so time-index-dependent imports (notably the EV
    flexibility bands built in ``import_electromobility``) run on an hourly,
    full-year index rather than their raw source resolution. The year is only a
    label — the DB imports fetch scenario-correct data regardless — and the index
    can be overridden later (e.g. by ``oedb_ts`` or the auto ``select_timesteps``
    step).
    """
    from edisgo.tools.tools import get_year_based_on_scenario

    year = get_year_based_on_scenario(ctx.scenario) or 2011
    edisgo.set_timeindex(pd.date_range(f"1/1/{year}", periods=8760, freq="h"))
    ctx.logger.info(
        f"select_timesteps: no time index set; using default full year "
        f"{year} (8760 h) so imports build hourly full-year data."
    )


@register_task("select_timesteps", provides={"timeseries"}, ts_altering=True)
def task_select_timesteps(edisgo, ctx, **overrides):
    """
    Select the time steps the grid is analyzed/optimized for.

    Reduces the time index to a configurable subset. Configuration is
    read from the top-level ``timeseries_selection:`` config block (so
    eGo can inject it the same way it injects ``overlying_grid``);
    inline step params override individual keys of that block. Two
    modes:

    ``manual``
        Reduce to an explicit set of time steps. Positioned *before*
        the data-import tasks so ``import_heat_pumps`` / ``import_dsm``
        download only the requested steps. The selected index is
        stashed in ``ctx.flags['selected_timeindex']`` for those
        imports to pick up.

    ``auto``
        Determine the two most critical time intervals and reduce to
        them. Must be positioned *after* ``import_overlying_grid_data``
        (needs all active-power time series) and *before*
        ``reactive_power``. Two ``method`` options:

        * ``power_flow`` (default) — score intervals via a power flow
          (:func:`~.tools.temporal_complexity_reduction.get_most_critical_time_intervals`).
          A reactive-power series is set internally to run the scoring
          power flow, but ``ctx.flags['reactive_power_set']`` is left
          unset so the pipeline's own ``reactive_power`` step still runs
          on the reduced index.
        * ``residual_load`` — no power flow. The overlying-grid dispatch
          is distributed onto the components and the residual load is
          ranked over the whole year; intervals are centered on the
          highest (load case) and lowest (feed-in case) residual-load
          steps. Requires overlying-grid data to be present.

        Both methods delegate to
        :func:`~.tools.temporal_complexity_reduction.get_most_critical_time_intervals`
        (via its ``by`` parameter) and reduce to a non-overlapping pair
        chosen by
        :func:`~.tools.temporal_complexity_reduction.select_two_intervals`.

    The auto mode normally yields two disconnected intervals (one for
    overloading, one for voltage issues). These are kept separate in the
    resulting time index (there is a gap between them). If they overlap,
    a non-overlapping pair is chosen if possible, otherwise they are
    concatenated into one interval. The intervals themselves are not
    stored — a later ``optimize`` step can detect the gap in the time
    index and run separate optimizations per interval.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Reads ``ctx.raw_config['timeseries_selection']``.
        Sets ``ctx.flags['selected_timeindex']`` (manual) and
        ``ctx.flags['timesteps_selected'] = True``.
    **overrides
        Inline step params overriding keys of the
        ``timeseries_selection`` block. Recognized keys:
        ``position`` (``"pre_import"`` | ``"post_grid"``, optional) — the
        step only acts when the configured ``mode`` matches this
        position (``pre_import`` ↔ ``manual``, ``post_grid`` ↔ ``auto``),
        otherwise it is a no-op; this lets one pipeline carry both a
        pre-import and a post-grid ``select_timesteps`` step and support
        either mode via config. When omitted, the step always acts.
        ``mode`` (``"manual"`` | ``"auto"``);
        for manual: ``timestamps`` (list) or ``start`` /
        ``periods`` / ``end`` / ``freq``;
        for auto: ``method`` (``"power_flow"`` (default) |
        ``"residual_load"``), ``time_steps_per_time_interval``;
        for ``method="power_flow"`` additionally ``percentage``,
        ``time_step_day_start`` (default 4), ``save_steps`` (default
        True; CSV written to ``ctx.results_dir``),
        ``use_troubleshooting_mode``, ``overloading_factor``,
        ``voltage_deviation_factor``.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    Raises
    ------
    ValueError
        If ``mode`` is missing/unknown, if manual mode has neither
        ``timestamps`` nor a range, or if auto mode runs before active
        power time series are set.
    """
    from edisgo.tools.temporal_complexity_reduction import (
        get_most_critical_time_intervals,
        select_two_intervals,
    )
    from edisgo.tools.tools import reduce_timeseries_data_to_given_timeindex

    cfg = {**ctx.raw_config.get("timeseries_selection", {}), **overrides}
    mode = cfg.get("mode")
    if mode not in ("manual", "auto"):
        raise ValueError(
            f"select_timesteps needs mode 'manual' or 'auto', got {mode!r}."
        )

    # A pipeline may include two select_timesteps steps — one before the
    # imports (``position: pre_import``, where manual selection belongs) and
    # one after import_overlying_grid_data (``position: post_grid``, where auto
    # selection belongs) — so the same preset supports both modes. Each step
    # only acts when the configured mode matches its position; otherwise it is
    # a no-op. When ``position`` is omitted (single-step usage) the step always
    # acts.
    position = overrides.get("position")
    expected_mode = {"pre_import": "manual", "post_grid": "auto"}
    if position is not None:
        if position not in expected_mode:
            raise ValueError(
                f"select_timesteps 'position' must be 'pre_import' or "
                f"'post_grid', got {position!r}."
            )
        if mode != expected_mode[position]:
            # This positioned step is not the active selector. If it is the
            # pre-import step and no time index has been set yet (i.e. manual
            # selection is not driving the index), establish a full-year default
            # so the following imports build their time-index-dependent data
            # (e.g. EV flexibility bands) on an hourly full-year index. Only a
            # label — DB imports fetch scenario-correct data regardless — and it
            # is overridden later by oedb_ts / the auto select_timesteps step.
            if position == "pre_import" and edisgo.timeseries.timeindex.empty:
                _set_default_full_year_timeindex(edisgo, ctx)
            ctx.logger.debug(
                f"select_timesteps at position {position!r} is a no-op for "
                f"mode {mode!r}."
            )
            return edisgo

    if mode == "manual":
        timestamps = cfg.get("timestamps")
        if timestamps is not None:
            timeindex = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
        elif cfg.get("end") is not None:
            timeindex = pd.date_range(
                start=cfg["start"], end=cfg["end"], freq=cfg.get("freq", "h")
            )
        elif cfg.get("periods") is not None:
            timeindex = pd.date_range(
                start=cfg["start"],
                periods=cfg["periods"],
                freq=cfg.get("freq", "h"),
            )
        else:
            raise ValueError(
                "select_timesteps manual mode needs 'timestamps' or a "
                "'start' plus 'periods'/'end' range."
            )
        timeindex = timeindex.sort_values().unique()
        if not edisgo.timeseries.timeindex.empty:
            # A time index is already set (manual selection reducing an existing
            # full time series): align the user-supplied timestamps to that
            # index's year so date-based slicing matches even if the user wrote
            # them in a different (e.g. scenario) year than the internally used
            # reference year.
            year_diff = edisgo.timeseries.timeindex[0].year - timeindex[0].year
            if year_diff != 0:
                timeindex = timeindex + pd.DateOffset(years=year_diff)
        ctx.flags["selected_timeindex"] = timeindex
        if edisgo.timeseries.timeindex.empty:
            # positioned before imports: just set the index so HP/DSM
            # imports restrict their downloads to it
            edisgo.set_timeindex(timeindex)
        else:
            reduce_timeseries_data_to_given_timeindex(edisgo, timeindex)
        ctx.logger.info(
            f"select_timesteps (manual): selected {len(timeindex)} time steps."
        )
        ctx.flags["timesteps_selected"] = True
        return edisgo

    # auto mode
    # if not ctx.flags.get("timeseries_set"):
    #     raise ValueError(
    #         "select_timesteps mode 'auto' needs active-power time series to "
    #         "be set first (e.g. run oedb_ts before it)."
    #     )

    method = cfg.get("method", "power_flow")
    if method not in ("power_flow", "residual_load"):
        raise ValueError(
            f"select_timesteps auto 'method' must be 'power_flow' or "
            f"'residual_load', got {method!r}."
        )
    tsp = cfg.get("time_steps_per_time_interval", 168)

    if method == "residual_load":
        # residual-load selection requires overlying-grid data (the dispatch
        # distributed onto the components); guard here (mode selection) before
        # delegating the computation to the tools function.
        og = edisgo.overlying_grid
        if all(
            s.empty
            for s in (
                og.electromobility_active_power,
                og.storage_units_active_power,
                og.heat_pump_central_active_power,
                og.heat_pump_decentral_active_power,
                og.dsm_active_power,
                og.renewables_curtailment,
            )
        ):
            raise ValueError(
                "select_timesteps method 'residual_load' needs overlying-grid "
                "data to be present (run import_overlying_grid_data before it)."
            )
        col_a, col_b = "time_steps_load_case", "time_steps_feedin_case"
    else:  # power_flow
        # throwaway reactive power so the scoring power flow yields meaningful
        # voltages; do NOT mark reactive_power_set — the pipeline's own
        # reactive_power step runs afterwards on the reduced index.
        edisgo.set_time_series_reactive_power_control(control="fixed_cosphi")
        col_a, col_b = "time_steps_overloading", "time_steps_voltage_issues"

    intervals_df = get_most_critical_time_intervals(
        edisgo,
        by=method,
        percentage=cfg.get("percentage", 1.0),
        time_steps_per_time_interval=tsp,
        time_step_day_start=cfg.get("time_step_day_start", 4),
        save_steps=cfg.get("save_steps", True),
        path=str(ctx.results_dir) if ctx.results_dir is not None else "",
        use_troubleshooting_mode=cfg.get("use_troubleshooting_mode", True),
        overloading_factor=cfg.get("overloading_factor", 0.95),
        voltage_deviation_factor=cfg.get("voltage_deviation_factor", 0.95),
    )
    intervals = select_two_intervals(
        list(intervals_df.get(col_a, [])),
        list(intervals_df.get(col_b, [])),
    )

    if not intervals:
        raise ValueError(
            "select_timesteps mode 'auto' found no critical time intervals; "
            "cannot reduce the time index."
        )

    timeindex = intervals[0]
    for interval in intervals[1:]:
        timeindex = timeindex.union(interval)
    timeindex = timeindex.sort_values()

    reduce_timeseries_data_to_given_timeindex(edisgo, timeindex)
    ctx.logger.info(
        f"select_timesteps (auto): selected {len(intervals)} interval(s), "
        f"{len(timeindex)} time steps total."
    )
    ctx.flags["timesteps_selected"] = True
    return edisgo


@register_task("reactive_power")
def task_reactive_power(
    edisgo,
    ctx,
    *,
    control="fixed_cosphi",
    generators_parametrisation="default",
    loads_parametrisation="default",
    storage_units_parametrisation="default",
):
    """
    Apply reactive-power control on top of the active-power time series.

    This MUST be the last time-series-altering step before
    ``analyze`` / ``reinforce`` / ``optimize``. The validator
    enforces this ordering rule statically.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Sets ``ctx.flags['reactive_power_set'] = True``.
    control : str, optional
        Reactive-power control strategy; typically ``"fixed_cosphi"``.
    generators_parametrisation : str or dict, optional
        Per-generator parametrisation, ``"default"`` uses the config.
    loads_parametrisation : str or dict, optional
        Per-load parametrisation.
    storage_units_parametrisation : str or dict, optional
        Per-storage-unit parametrisation.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    edisgo.set_time_series_reactive_power_control(
        control=control,
        generators_parametrisation=generators_parametrisation,
        loads_parametrisation=loads_parametrisation,
        storage_units_parametrisation=storage_units_parametrisation,
    )
    ctx.flags["reactive_power_set"] = True
    return edisgo
