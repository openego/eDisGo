"""
Time-series tasks — set active/reactive power profiles on EDisGo.

Time series drive every downstream step: ``analyze``, ``reinforce``
and ``optimize`` all operate on the time index and power time series
attached to the EDisGo object. The order inside a stage matters:

1. Set the time index and active-power profiles with one of
   :func:`task_worst_case_ts`, :func:`task_oedb_ts`,
   :func:`task_manual_ts`, possibly :func:`task_set_timeindex`.
2. Finally call :func:`task_reactive_power` to fix reactive power
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
