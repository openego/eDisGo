"""
Power-flow, reinforcement, and optimization tasks.

The three analysis layers:

* :func:`task_analyze` (``analyze``) — non-linear AC load flow over
  the active time series; does not modify the topology.
* :func:`task_reinforce` (``reinforce``) — iterative reinforcement
  that adds/upgrades equipment until all technical constraints are
  met. Populates ``results.equipment_changes``.
* :func:`task_optimize` (``optimize``) — powermodels OPF over
  flexibilities (heat pumps, EV, DSM, storage) to minimize
  reinforcement need.

In addition:

* :func:`task_check_integrity` (``check_integrity``) — a cheap
  sanity check before the expensive steps.
* :func:`task_base_reinforce` (``base_reinforce``) — two-phase helper:
  worst-case TS → reinforce → reset ``equipment_changes``. Used to
  produce a "base" grid whose subsequent reinforce costs reflect
  only a scenario overlay.
"""

from __future__ import annotations

import pandas as pd

from edisgo.run.registry import register_task


@register_task("check_integrity")
def task_check_integrity(edisgo, ctx):
    """
    Run EDisGo's integrity checks on the topology and time series.

    Catches bus mismatches, missing time series for components, and
    similar structural problems. Raises if something is off — do not
    swallow it silently.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to check.
    ctx : RunContext
        Run context (unused).

    Returns
    -------
    edisgo.EDisGo
        The unchanged EDisGo instance.

    """
    edisgo.check_integrity()
    return edisgo


@register_task("analyze")
def task_analyze(
    edisgo,
    ctx,
    *,
    mode=None,
    timesteps=None,
    raise_not_converged=False,
    troubleshooting_mode=None,
):
    """
    Run AC power flow over the active time series.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to analyze.
    ctx : RunContext
        Run context. Stores the number of non-converged time steps
        under ``ctx.flags['not_converged_steps']`` and warns if any.
    mode : str, optional
        ``None`` (default) runs the full grid; ``"mv"`` runs only the
        medium-voltage level; ``"lv"`` runs only LV.
    timesteps : pandas.DatetimeIndex, optional
        Restrict the analysis to these time steps.
    raise_not_converged : bool, optional
        If ``True``, raise on non-convergence. Default ``False`` so
        the pipeline can continue and ``reinforce`` can attempt to
        resolve the issue.
    troubleshooting_mode : str, optional
        Extra diagnostic mode passed through to
        :meth:`EDisGo.analyze`.

    Returns
    -------
    edisgo.EDisGo
        The analyzed EDisGo instance.

    """
    result = edisgo.analyze(
        mode=mode,
        timesteps=timesteps,
        raise_not_converged=raise_not_converged,
        troubleshooting_mode=troubleshooting_mode,
    )
    if isinstance(result, tuple) and len(result) == 2:
        converged, not_converged = result
        ctx.flags["not_converged_steps"] = len(not_converged)
        if len(not_converged) > 0:
            ctx.logger.warning(
                f"Power flow did not converge for {len(not_converged)} time steps."
            )
    return edisgo


@register_task("reinforce")
def task_reinforce(
    edisgo,
    ctx,
    *,
    timesteps_pfa=None,
    reduced_analysis=False,
    copy_grid=False,
    max_while_iterations=20,
    split_voltage_band=True,
    mode=None,
    without_generator_import=False,
    n_minus_one=False,
    catch_convergence_problems=False,
):
    """
    Run iterative grid reinforcement.

    Adds/upgrades lines and transformers until voltage and loading
    constraints are met for all time steps. Results accumulate in
    :attr:`EDisGo.results.equipment_changes` and
    :attr:`~EDisGo.results.grid_expansion_costs`.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to reinforce.
    ctx : RunContext
        Run context (unused beyond logging).
    timesteps_pfa : pandas.DatetimeIndex, optional
        Restrict the reinforcement's analysis to these time steps.
    reduced_analysis : bool, optional
        If ``True``, use a cheaper convergence check during
        reinforcement.
    copy_grid : bool, optional
        If ``True``, operate on a copy and return it as a new
        instance (default ``False``).
    max_while_iterations : int, optional
        Cap on the outer iteration loop.
    split_voltage_band : bool, optional
        Split the allowed voltage deviation between MV and LV
        (typical MV/LV coupling rule).
    mode : str, optional
        ``None``, ``"mv"``, ``"lv"``, or ``"mvlv"``. Restricts
        reinforcement to a voltage level.
    without_generator_import : bool, optional
        Skip the implicit generator import step.
    n_minus_one : bool, optional
        Enable (N-1) contingency reinforcement. Expensive.
    catch_convergence_problems : bool, optional
        Wrap in the catch-convergence helper for troublesome grids.

    Returns
    -------
    edisgo.EDisGo
        The reinforced EDisGo instance.

    """
    edisgo.reinforce(
        timesteps_pfa=timesteps_pfa,
        reduced_analysis=reduced_analysis,
        copy_grid=copy_grid,
        max_while_iterations=max_while_iterations,
        split_voltage_band=split_voltage_band,
        mode=mode,
        without_generator_import=without_generator_import,
        n_minus_one=n_minus_one,
        catch_convergence_problems=catch_convergence_problems,
    )
    return edisgo


@register_task("base_reinforce")
def task_base_reinforce(
    edisgo, ctx, *, cases=None, reset_equipment_changes=True, save_artifact=True
):
    """
    Produce a base-reinforced grid and reset the cost accumulator.

    This is the composite step ported from eGo's two-phase reinforce
    workflow:

    1. Set synthetic worst-case time series (``feed-in_case`` +
       ``load_case``).
    2. Run :meth:`EDisGo.reinforce` to bring the grid to a neutral
       baseline.
    3. Optionally save the resulting grid so downstream stages can
       ``load_from: ...``.
    4. Clear :attr:`Results.equipment_changes` so the next reinforce
       captures only scenario-specific deltas.
    5. Restore the prior time index so the next TS-setting task
       starts from a clean state.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to base-reinforce.
    ctx : RunContext
        Run context. ``ctx.results_dir`` is the artifact destination.
        Sets ``ctx.flags['base_reinforced'] = True`` and
        ``ctx.stage_artifacts['__base_reinforce__']`` on save.
    cases : list of str, optional
        Which worst cases to set (subset of
        ``{"load_case", "feed-in_case"}``). Default is both.
    reset_equipment_changes : bool, optional
        Clear the equipment-changes DataFrame after reinforcement.
    save_artifact : bool, optional
        Write a ``grid_data_base_reinforcement.zip`` next to the
        other results.

    Returns
    -------
    edisgo.EDisGo
        The base-reinforced EDisGo instance.

    """
    import os

    prev_timeindex = edisgo.timeseries.timeindex

    edisgo.set_time_series_worst_case_analysis(cases=cases)
    edisgo.reinforce()

    if save_artifact and ctx.results_dir is not None:
        artifact_dir = os.path.join(
            str(ctx.results_dir), "grid_data_base_reinforcement"
        )
        edisgo.save(
            directory=artifact_dir,
            save_topology=True,
            save_timeseries=False,
            save_results=True,
            archive=True,
            archive_type="zip",
            parameters={"grid_expansion_results": ["equipment_changes"]},
        )
        ctx.stage_artifacts["__base_reinforce__"] = artifact_dir + ".zip"

    if reset_equipment_changes:
        edisgo.results.equipment_changes = pd.DataFrame()

    if len(prev_timeindex) > 0:
        edisgo.set_timeindex(prev_timeindex)

    ctx.flags["base_reinforced"] = True
    return edisgo


@register_task("optimize")
def task_optimize(
    edisgo,
    ctx,
    *,
    flexible=None,
    flexible_cps=None,
    flexible_hps=None,
    flexible_loads=None,
    flexible_storage_units=None,
    opf_version=2,
    method="soc",
    warm_start=False,
    s_base=1,
):
    """
    Run a powermodels optimal-power-flow (OPF) over flexibilities.

    If ``flexible`` is given (high-level shortcut), it expands to the
    lower-level ``flexible_*`` lists automatically:

    * ``"heat_pumps"`` → all loads of type ``heat_pump``
    * ``"charging_points"`` → all loads of type ``charging_point``
    * ``"storage"`` → all storage-unit indices
    * ``"loads"`` → all DSM-ready load indices

    Explicit ``flexible_*`` kwargs override the shortcut.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to optimize.
    ctx : RunContext
        Run context (unused).
    flexible : list of str, optional
        High-level selector, subset of ``{"heat_pumps",
        "charging_points", "storage"}``. If ``None``, nothing is
        auto-populated.
    flexible_cps : list of str, optional
        Explicit list of flexible charging-point names.
    flexible_hps : list of str, optional
        Explicit list of flexible heat-pump load names.
    flexible_loads : list of str, optional
        Explicit list of flexible DSM load names.
    flexible_storage_units : list of str, optional
        Explicit list of flexible storage-unit names.
    opf_version : int, optional
        Powermodels OPF formulation version (1 or 2, default 2).
    method : str, optional
        OPF relaxation method, e.g. ``"soc"`` (second-order cone).
    warm_start : bool, optional
        Reuse a previous solution as the starting point.
    s_base : float, optional
        Per-unit base power for normalization.

    Returns
    -------
    edisgo.EDisGo
        The optimized EDisGo instance.

    """
    flexible = flexible or []

    if flexible_hps is None and "heat_pumps" in flexible:
        flexible_hps = edisgo.topology.loads_df.loc[
            edisgo.topology.loads_df.type == "heat_pump"
        ].index.tolist()
    if flexible_cps is None and "charging_points" in flexible:
        flexible_cps = edisgo.topology.loads_df.loc[
            edisgo.topology.loads_df.type == "charging_point"
        ].index.tolist()
    if flexible_storage_units is None and "storage" in flexible:
        flexible_storage_units = edisgo.topology.storage_units_df.index.tolist()
    if flexible_loads is None and "dsm" in flexible:
        flexible_loads = edisgo.dsm.p_min.columns.values

    if flexible_cps is None:
        flexible_cps = []
    if flexible_hps is None:
        flexible_hps = []
    if flexible_loads is None:
        flexible_loads = []
    if flexible_storage_units is None:
        flexible_storage_units = []

    edisgo.pm_optimize(
        flexible_cps=flexible_cps,
        flexible_hps=flexible_hps,
        flexible_loads=flexible_loads,
        flexible_storage_units=flexible_storage_units,
        opf_version=opf_version,
        method=method,
        warm_start=warm_start,
        s_base=s_base,
    )
    return edisgo


@register_task("optimize_temporal_complexity_reduction")
def optimize_temporal_complexity_reduction(
    edisgo,
    ctx,
    *,
    flexible=None,
    flexible_cps=None,
    flexible_hps=None,
    flexible_loads=None,
    flexible_storage_units=None,
    opf_version=2,
    method="soc",
    warm_start=False,
    s_base=1,
):
    """
    """
    from edisgo.tools.temporal_complexity_reduction import (
        get_most_critical_time_intervals,
    )
    timeindex = pd.Index([])
    time_intervals = get_most_critical_time_intervals(
        edisgo,
        percentage=1.0,
        time_steps_per_time_interval=168,
        time_step_day_start=4,
        save_steps=True,
        #path=results_dir,
        use_troubleshooting_mode=True,
        overloading_factor=0.95,
        voltage_deviation_factor=0.95,
    )
    
    # select time intervals
    if not time_intervals.loc[:, "time_steps_overloading"].dropna().empty:
        tmp = time_intervals.loc[:, "time_steps_overloading"].dropna()
        time_interval_1 = tmp.iloc[0]
        time_interval_1_ind = tmp.index[0]
    else:
        time_interval_1 = pd.Index([])
        time_interval_1_ind = None
    if not time_intervals.loc[:, "time_steps_voltage_issues"].dropna().empty:
        tmp = time_intervals.loc[:, "time_steps_voltage_issues"].dropna()
        time_interval_2 = tmp.iloc[0]
        time_interval_2_ind = tmp.index[0]
    else:
        time_interval_2 = pd.Index([])
        time_interval_2_ind = None

    # check if time intervals overlap
    overlap = [_ for _ in time_interval_1 if _ in time_interval_2]
    if len(overlap) > 0:
        print(
            "Selected time intervals overlap. Trying to find another "
            "time interval in voltage_issues intervals."
        )
        # check if time interval without overlap can be found
        for ti in time_intervals.loc[:, "time_steps_voltage_issues"].dropna().index:
            overlap = [
                _
                for _ in time_interval_1
                if _ in time_intervals.at[ti, "time_steps_voltage_issues"]
            ]
            if len(overlap) == 0:
                time_interval_2 = time_intervals.at[ti, "time_steps_voltage_issues"]
                time_interval_2_ind = ti
                break
    overlap = [_ for _ in time_interval_1 if _ in time_interval_2]
    if len(overlap) > 0:
        print(
            "Selected time intervals overlap. Trying to find another "
            "time interval in overloading intervals."
        )
        # check if time interval without overlap can be found
        for ti in time_intervals.loc[:, "time_steps_overloading"].dropna().index:
            overlap = [
                _
                for _ in time_interval_2
                if _ in time_intervals.at[ti, "time_steps_overloading"]
            ]
            if len(overlap) == 0:
                time_interval_1 = time_intervals.at[ti, "time_steps_overloading"]
                time_interval_1_ind = ti
                break

    overlap = [_ for _ in time_interval_1 if _ in time_interval_2]
    if len(overlap) > 0:
        print(
            "Overlap of selected time intervals cannot be avoided. "
            "Time intervals are therefore concatenated."
        )
        time_interval_1 = (
            time_interval_1.append(time_interval_2).unique().sort_values()
        )
        time_interval_2 = None

    # save to csv
    percentage = pd.Series()
    percentage["time_interval_1"] = (
        None
        if time_interval_1_ind is None
        else time_intervals.at[
            time_interval_1_ind, "percentage_max_overloaded_components"
        ]
    )
    percentage["time_interval_2"] = (
        None
        if time_interval_2_ind is None
        else time_intervals.at[
            time_interval_2_ind, "percentage_buses_max_voltage_deviation"
        ]
    )

    scenario = ctx.scenario
    reduction_factor = 0.3 # aus eGon paper
    timeindex = pd.Index([])
    from copy import deepcopy
    from edisgo.tools.tools import reduce_timeseries_data_to_given_timeindex
    for time_steps in [time_interval_1, time_interval_2]:
        timeindex = timeindex.append(pd.Index(time_steps))
        # copy edisgo object
        edisgo_copy = deepcopy(edisgo)
        # temporal complexity reduction
        reduce_timeseries_data_to_given_timeindex(edisgo_copy, time_steps)
        edisgo_copy.timeseries.timeindex.freq = "H"
        # spatial complexity reduction
        edisgo_copy.spatial_complexity_reduction(
            mode="kmeansdijkstra",
            cluster_area="feeder",
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=False,
        )

        # OPF
        # flexibilities in full flex: DSM, decentral and central PtH units,
        # curtailment, EVs, storage units
        # flexibilities in low flex: curtailment, storage units
        psa_net = edisgo_copy.to_pypsa()
        if scenario in ["eGon2035", "eGon100RE"]:
            flexible_loads = edisgo_copy.dsm.p_max.columns
            # flexible_hps = (
            #     edisgo_copy.heat_pump.thermal_storage_units_df.index.values
            # )
            flexible_cps = psa_net.loads.loc[
                psa_net.loads.index.str.contains("home")
                | (psa_net.loads.index.str.contains("work"))
            ].index.values
        else:
            flexible_loads = []
            # flexible_hps = []
            flexible_cps = []
        flexible_hps = edisgo_copy.heat_pump.heat_demand_df.columns.values
        flexible_storage_units = (
            edisgo_copy.topology.storage_units_df.index.values
        )

        edisgo_copy.pm_optimize(
            flexible_cps=flexible_cps,
            flexible_hps=flexible_hps,
            flexible_loads=flexible_loads,
            flexible_storage_units=flexible_storage_units,
            s_base=1,
            opf_version=4,
            silence_moi=False,
            method="soc",
        )
        
        # save OPF results
        # zip_name = f"opf_results_{ti}"
        # if scenario in ["eGon2035_lowflex", "eGon100RE_lowflex"]:
        #     zip_name += "_lowflex"
        # edisgo_copy.save(
        #     directory=os.path.join(results_dir, zip_name),
        #     save_topology=True,
        #     save_timeseries=False,
        #     save_results=False,
        #     save_opf_results=True,
        #     reduce_memory=True,
        #     archive=True,
        #     archive_type="zip",
        # )

        # write flexibility dispatch results to spatially unreduced edisgo
        # object
        edisgo.timeseries._loads_active_power.loc[
            time_steps, :
        ] = edisgo_copy.timeseries.loads_active_power
        edisgo.timeseries._loads_reactive_power.loc[
            time_steps, :
        ] = edisgo_copy.timeseries.loads_reactive_power
        edisgo.timeseries._generators_active_power.loc[
            time_steps, :
        ] = edisgo_copy.timeseries.generators_active_power
        edisgo.timeseries._generators_reactive_power.loc[
            time_steps, :
        ] = edisgo_copy.timeseries.generators_reactive_power

        try:
            edisgo.timeseries._storage_units_active_power
        except AttributeError:
            edisgo.timeseries.storage_units_active_power = pd.DataFrame(
                index=edisgo.timeseries.timeindex
            )
        edisgo.timeseries._storage_units_active_power.loc[
            time_steps,
            edisgo_copy.timeseries.storage_units_active_power.columns,
        ] = edisgo_copy.timeseries.storage_units_active_power
        try:
            edisgo.timeseries._storage_units_reactive_power
        except AttributeError:
            edisgo.timeseries.storage_units_reactive_power = pd.DataFrame(
                index=edisgo.timeseries.timeindex
            )
        edisgo.timeseries._storage_units_reactive_power.loc[
            time_steps,
            edisgo_copy.timeseries.storage_units_reactive_power.columns,
        ] = edisgo_copy.timeseries.storage_units_reactive_power

        # write OPF results back
        edisgo.opf_results.overlying_grid = pd.concat(
            [
                edisgo.opf_results.overlying_grid,
                edisgo_copy.opf_results.overlying_grid,
            ]
        )
        edisgo.opf_results.battery_storage_t.p = pd.concat(
            [
                edisgo.opf_results.battery_storage_t.p,
                edisgo_copy.opf_results.battery_storage_t.p,
            ]
        )
        edisgo.opf_results.battery_storage_t.e = pd.concat(
            [
                edisgo.opf_results.battery_storage_t.e,
                edisgo_copy.opf_results.battery_storage_t.e,
            ]
        )

    return edisgo