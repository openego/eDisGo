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
attached to the EDisGo object. The order inside a pipeline matters:

1. Optionally fix the time index early with :func:`task_set_timeindex`
   (manual selection — data imports then only fetch the selected
   steps).
2. Set the active-power profiles with one of
   :func:`task_worst_case_ts`, :func:`task_oedb_ts`,
   :func:`task_manual_ts`.
3. Optionally reduce the time index to the most critical intervals
   with :func:`task_select_critical_timesteps` (automatic selection —
   needs all active-power time series, so it runs late).
4. Finally call :func:`task_reactive_power` to fix reactive power
   control — this MUST come last because it overwrites whatever
   reactive power was set by the earlier steps.

A pipeline may contain both time-step selection tasks (1 and 3). The
optional top-level ``timestep_selection`` config block then picks the one
that runs; the other is skipped (it stays in the pipeline and does
nothing)::

    timestep_selection:
      task: select_critical_timesteps   # or set_timeindex, or none
      select_critical_timesteps: {method: power_flow}
      set_timeindex: {start: "2035-01-15 00:00", periods: 24}

Each task takes its parameters from the sub-block named after it, so a
config that ``extends`` a preset can switch the task or change single
parameters without mixing the parameters of the two tasks. ``task: none``
runs neither and keeps the time index as loaded. While the block is set,
the steps of both tasks take no inline params (parameters written on the
pipeline step itself, e.g. ``- set_timeindex: {periods: 24}``). Without the
block (or with ``timestep_selection: null``) both tasks run with their
inline params.

``set_timeindex`` selects either a list of ``timestamps``, one range
(``start`` plus ``periods`` or ``end``) or several separate ``ranges``,
e.g. two weeks::

    timestep_selection:
      task: set_timeindex
      set_timeindex:
        ranges:
          - {start: "2035-01-09 00:00", periods: 168}
          - {start: "2035-07-03 00:00", end: "2035-07-09 23:00"}

In a config that ``extends`` a preset, these parameters replace the
preset's parameters they cannot be combined with. The example above, on
top of ``spatial_reduction_opf``, selects the two weeks instead of the
preset's ``start`` and ``periods``; setting only ``end`` replaces the
preset's ``periods`` and keeps its ``start``.
"""

from __future__ import annotations

import copy
import functools
import inspect

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


SELECTION_TASKS = ("set_timeindex", "select_critical_timesteps")

# Parameters of set_timeindex that each parameter cannot be combined with
_EXCLUDED_TIMEINDEX_PARAMS = {
    "timestamps": {"ranges", "start", "periods", "end"},
    "ranges": {"timestamps", "start", "periods", "end"},
    "start": {"timestamps", "ranges"},
    "periods": {"timestamps", "ranges", "end"},
    "end": {"timestamps", "ranges", "periods"},
}


def drop_replaced_timeindex_params(parent, child):
    """
    Drop the ``set_timeindex`` parameters of ``parent`` that ``child`` replaces.

    Called before ``child`` is merged into the config ``parent`` it
    extends: a parameter in ``child``'s ``timestep_selection.set_timeindex``
    removes the parameters of ``parent`` it cannot be combined with, e.g.
    ``ranges`` removes ``start`` and ``periods``.

    Returns
    -------
    dict
        ``parent``, or a copy of it without the replaced parameters.

    """
    try:
        parent_params = parent["timestep_selection"]["set_timeindex"]
        child_params = child["timestep_selection"]["set_timeindex"]
        replaced = set().union(
            *(
                _EXCLUDED_TIMEINDEX_PARAMS.get(key, set())
                for key, value in child_params.items()
                if value is not None
            )
        ) & set(parent_params)
    except (KeyError, TypeError, AttributeError):
        return parent
    if not replaced:
        return parent
    parent = copy.deepcopy(parent)
    for key in replaced:
        del parent["timestep_selection"]["set_timeindex"][key]
    return parent


def check_timestep_selection(selection):
    """
    Check a ``timestep_selection`` config block (see the module docstring).

    Called by the validator, so a wrong block is reported before the
    pipeline runs.

    Raises
    ------
    ValueError
        If the block is not a mapping, has unknown keys, names no valid
        ``task``, or a sub-block holds parameters its task does not accept.

    """
    if not isinstance(selection, dict):
        raise ValueError(
            f"timestep_selection must be a mapping with a 'task', got "
            f"{selection!r}."
        )
    unknown = set(selection) - {"task", *SELECTION_TASKS}
    if unknown:
        raise ValueError(
            f"Unknown timestep_selection keys {sorted(unknown)}. Task "
            f"parameters go under timestep_selection.<task>, e.g. "
            f"timestep_selection.select_critical_timesteps.method."
        )
    if selection.get("task") not in (*SELECTION_TASKS, "none"):
        raise ValueError(
            f"timestep_selection 'task' must be 'set_timeindex', "
            f"'select_critical_timesteps' or 'none', got "
            f"{selection.get('task')!r}."
        )
    for task in SELECTION_TASKS:
        params = selection.get(task)
        if params is None:
            continue
        if not isinstance(params, dict):
            raise ValueError(
                f"timestep_selection.{task} must be a mapping of task "
                f"parameters, got {params!r}."
            )
        _check_param_names(task, params)


def check_selection_params(task, params):
    """
    Check the parameters a time-step selection task will run with.

    Raises a ValueError for parameters ``task`` does not accept and, for
    ``set_timeindex``, for a missing or conflicting time-step range.

    """
    _check_param_names(task, params)
    if task == "set_timeindex":
        _timeindex_from_params(**params)


def _check_param_names(task, params):
    implementation = {
        "set_timeindex": _set_timeindex,
        "select_critical_timesteps": _select_critical_timesteps,
    }[task]
    accepted = set(inspect.signature(implementation).parameters) - {"edisgo", "ctx"}
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(
            f"Unknown parameters {sorted(unknown)} for {task}. Accepted: "
            f"{sorted(accepted)}."
        )


def _selection_params(ctx, task, params):
    """
    Return the parameters to run time-step selection ``task`` with, or
    None if the ``timestep_selection`` block skips it.

    Without the block the inline step params are used. The validator has
    already checked the block and that the steps have no inline params
    when it is set.

    """
    selection = ctx.raw_config.get("timestep_selection")
    if selection is None:
        return params
    if selection["task"] != task:
        ctx.logger.info(
            f"Skipping {task} (timestep_selection 'task' is {selection['task']!r})."
        )
        return None
    return selection.get(task) or {}


@register_task("set_timeindex", provides={"timeseries"}, ts_altering=True)
def task_set_timeindex(edisgo, ctx, **params):
    """
    Manually fix the time index (and reduce existing data to it).

    This is the *manual* time-step selection, positioned early in the
    pipeline: data imports that follow (``import_flex``, ``oedb_ts``)
    restrict their downloads to the selected steps. The automatic
    counterpart is :func:`task_select_critical_timesteps`, which runs
    late.

    If time-series data is already attached, it is reduced to the
    given index; user timestamps written in a different year (e.g. the
    scenario year) are shifted to the year of the existing index so
    date-based slicing matches.

    Takes its parameters from ``timestep_selection.set_timeindex`` if that
    config block is set, otherwise from the inline step params (see the
    module docstring).

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Reads ``ctx.raw_config['timestep_selection']``.
        Stashes the selected index in ``ctx.flags['selected_timeindex']``
        so subsequent data imports can restrict their downloads to it.
    timestamps : list of str or pandas.Timestamp, optional
        Explicit time steps to select; mutually exclusive with ``ranges``
        and the ``start``/``periods``/``end`` range.
    ranges : list of dict, optional
        Several ranges to select, e.g. two separate weeks. Each range is a
        mapping with exactly a ``start`` plus either ``periods`` or
        ``end``; ``freq`` is set once for all ranges. The ranges are
        joined into one sorted time index without duplicates; the gaps
        between them are kept. Mutually exclusive with ``timestamps`` and
        the ``start``/``periods``/``end`` range.
    start : str or pandas.Timestamp, optional
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
        If neither ``timestamps``, ``ranges`` nor a ``start`` plus
        ``periods``/``end`` range is provided, if more than one of them is
        given, if ``periods`` is combined with ``end``, if ``ranges`` is not
        a non-empty list or a range has other keys than a ``start`` plus
        either ``periods`` or ``end``, if a range selects no time steps, or
        if the selected time steps lie in more than one year.

    """
    params = _selection_params(ctx, "set_timeindex", params)
    if params is None:
        return edisgo
    return _set_timeindex(edisgo, ctx, **params)


def _set_timeindex(
    edisgo,
    ctx,
    *,
    timestamps=None,
    ranges=None,
    start=None,
    periods=None,
    end=None,
    freq="h",
):
    """Implementation of :func:`task_set_timeindex`."""
    from edisgo.tools.tools import reduce_timeseries_data_to_given_timeindex

    timeindex = _timeindex_from_params(timestamps, ranges, start, periods, end, freq)

    if edisgo.timeseries.timeindex.empty:
        edisgo.set_timeindex(timeindex)
    else:
        # A time index is already set (manual selection reducing existing
        # data): align the user-supplied timestamps to that index's year so
        # date-based slicing matches even if the user wrote them in a
        # different (e.g. scenario) year than the internally used reference
        # year.
        year_diff = edisgo.timeseries.timeindex[0].year - timeindex[0].year
        if year_diff != 0:
            timeindex = timeindex + pd.DateOffset(years=year_diff)
        reduce_timeseries_data_to_given_timeindex(edisgo, timeindex)

    ctx.flags["selected_timeindex"] = timeindex
    ctx.logger.info(f"set_timeindex: selected {len(timeindex)} time steps.")
    return edisgo


def _timeindex_from_params(
    timestamps=None, ranges=None, start=None, periods=None, end=None, freq="h"
):
    """Build the time index of :func:`task_set_timeindex` from its params."""
    given = [
        name
        for name, value in (
            ("timestamps", timestamps),
            ("ranges", ranges),
            ("start", start),
            ("periods", periods),
            ("end", end),
        )
        if value is not None
    ]
    exclusive = {"timestamps", "ranges"} & set(given)
    if (exclusive and len(given) > 1) or {"periods", "end"} <= set(given):
        raise ValueError(
            f"set_timeindex takes either 'timestamps', 'ranges' or 'start' plus "
            f"'periods' or 'end', got {given}."
        )
    if timestamps is not None:
        timeindex = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
        timeindex = timeindex.sort_values().unique()
    else:
        if ranges is None:
            if start is None or (periods is None and end is None):
                raise ValueError(
                    "set_timeindex needs 'timestamps', 'ranges' or a 'start' "
                    "plus 'periods' or 'end'."
                )
            ranges = [{"start": start, "periods": periods, "end": end}]
        elif not isinstance(ranges, (list, tuple)) or not ranges:
            raise ValueError(
                f"set_timeindex 'ranges' must be a non-empty list of ranges, "
                f"got {ranges!r}."
            )
        timeindex = functools.reduce(
            pd.Index.union, [_date_range(time_range, freq) for time_range in ranges]
        )
    years = sorted(timeindex.year.unique().tolist())
    if len(years) > 1:
        raise ValueError(
            f"The time steps selected by set_timeindex lie in the years {years}. "
            f"They must lie in one year, as the time series are taken from one "
            f"year."
        )
    return timeindex


def _date_range(time_range, freq):
    """Build the time steps of one range of :func:`task_set_timeindex`."""
    keys = (
        {key for key, value in time_range.items() if value is not None}
        if isinstance(time_range, dict)
        else None
    )
    if keys not in ({"start", "periods"}, {"start", "end"}):
        raise ValueError(
            f"Each range of set_timeindex takes exactly a 'start' plus either "
            f"'periods' or 'end' ('freq' is set once for all ranges), got "
            f"{time_range!r}."
        )
    timeindex = pd.date_range(**time_range, freq=freq)
    if timeindex.empty:
        raise ValueError(
            f"The set_timeindex range {time_range!r} selects no time steps."
        )
    return timeindex


@register_task(
    "select_critical_timesteps",
    requires={"timeseries"},
    provides={"timeseries"},
    ts_altering=True,
)
def task_select_critical_timesteps(edisgo, ctx, **params):
    """
    Automatically reduce the time index to the most critical intervals.

    This is the *automatic* time-step selection, positioned late in the
    pipeline: it needs all active-power time series (including
    overlying-grid dispatch, if any) to be set, and must run before
    ``reactive_power``. The manual counterpart is
    :func:`task_set_timeindex`, which runs early.

    Takes its parameters from
    ``timestep_selection.select_critical_timesteps`` if that config block is
    set, otherwise from the inline step params (see the module docstring).

    Two ``method`` options:

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

    The result is normally two disconnected intervals (one for
    overloading / the load case, one for voltage issues / the feed-in
    case). These are kept separate in the resulting time index (there
    is a gap between them). If they overlap, a non-overlapping pair is
    chosen if possible, otherwise they are concatenated into one
    interval. A later ``optimize`` step detects gaps in the time index
    and runs separate optimizations per interval.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Reads ``ctx.raw_config['timestep_selection']``.
        Sets ``ctx.flags['timesteps_selected'] = True``.
    method : str, optional
        ``"power_flow"`` (default) or ``"residual_load"``.
    time_steps_per_time_interval : int, optional
        Interval length in time steps (default 168 = one week; must be
        a multiple of 24).
    time_step_day_start : int, optional
        Hour of day the intervals start/end on (default 4).
    percentage : float, optional
        Share of most critical intervals to consider
        (``power_flow`` method only).
    save_steps : bool, optional
        Write the selected intervals as CSV to ``ctx.results_dir``
        (``power_flow`` method only). Default ``True``.
    use_troubleshooting_mode : bool, optional
        Handle power-flow non-convergence during scoring
        (``power_flow`` method only).
    overloading_factor : float, optional
        Scoring threshold factor (``power_flow`` method only).
    voltage_deviation_factor : float, optional
        Scoring threshold factor (``power_flow`` method only).

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    Raises
    ------
    ValueError
        If ``method`` is unknown, if no active-power time series are
        set, if ``residual_load`` is requested without overlying-grid
        data, or if no critical intervals are found.

    """
    params = _selection_params(ctx, "select_critical_timesteps", params)
    if params is None:
        return edisgo
    return _select_critical_timesteps(edisgo, ctx, **params)


def _select_critical_timesteps(
    edisgo,
    ctx,
    *,
    method="power_flow",
    time_steps_per_time_interval=168,
    time_step_day_start=4,
    percentage=1.0,
    save_steps=True,
    use_troubleshooting_mode=True,
    overloading_factor=0.95,
    voltage_deviation_factor=0.95,
):
    """Implementation of :func:`task_select_critical_timesteps`."""
    from edisgo.tools.temporal_complexity_reduction import (
        get_most_critical_time_intervals,
        select_two_intervals,
    )
    from edisgo.tools.tools import reduce_timeseries_data_to_given_timeindex

    if not ctx.flags.get("timeseries_set"):
        raise ValueError(
            "select_critical_timesteps needs active-power time series to "
            "be set first (e.g. run oedb_ts before it)."
        )
    if method not in ("power_flow", "residual_load"):
        raise ValueError(
            f"select_critical_timesteps 'method' must be 'power_flow' or "
            f"'residual_load', got {method!r}."
        )

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
                "select_critical_timesteps method 'residual_load' needs "
                "overlying-grid data to be present (run "
                "import_overlying_grid_data before it), or use method "
                "'power_flow'."
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
        percentage=percentage,
        time_steps_per_time_interval=time_steps_per_time_interval,
        time_step_day_start=time_step_day_start,
        save_steps=save_steps,
        path=str(ctx.results_dir) if ctx.results_dir is not None else "",
        use_troubleshooting_mode=use_troubleshooting_mode,
        overloading_factor=overloading_factor,
        voltage_deviation_factor=voltage_deviation_factor,
    )
    intervals = select_two_intervals(
        list(intervals_df.get(col_a, [])),
        list(intervals_df.get(col_b, [])),
    )

    if not intervals:
        raise ValueError(
            "select_critical_timesteps found no critical time intervals; "
            "cannot reduce the time index."
        )

    timeindex = intervals[0]
    for interval in intervals[1:]:
        timeindex = timeindex.union(interval)
    timeindex = timeindex.sort_values()

    reduce_timeseries_data_to_given_timeindex(edisgo, timeindex)
    ctx.logger.info(
        f"select_critical_timesteps: selected {len(intervals)} interval(s), "
        f"{len(timeindex)} time steps total."
    )
    ctx.flags["timesteps_selected"] = True
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
        # No explicit timeindex and none set yet (e.g. no set_timeindex step
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
