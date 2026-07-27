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


@register_task("analyze", requires={"timeseries"})
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


@register_task("reinforce", requires={"timeseries"})
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


@register_task("base_reinforce", requires={"grid"})
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


@register_task(
    "optimize", requires={"timeseries", "flex"}, provides={"optimized_dispatch"}
)
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

    This task only performs the ``flexible`` shortcut expansion (mode selection)
    and calls :meth:`EDisGo.pm_optimize`. Handling of a non-contiguous (reduced)
    time index — running a separate OPF per contiguous interval and merging the
    results — lives in :func:`~.opf.powermodels_opf.pm_optimize`.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to optimize.
    ctx : RunContext
        Run context. Used for logging and, for multi-interval runs, nothing
        else is required from it. The resolved ``flexible_*`` name lists are
        written to ``ctx.flags['flexible_cps']`` / ``ctx.flags['flexible_hps']``
        / ``ctx.flags['flexible_loads']`` / ``ctx.flags['flexible_storage_units']``
        so a later ``spatial_restore`` step knows which components' dispatch
        needs mapping back onto the full grid.
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

    ctx.flags["flexible_cps"] = flexible_cps
    ctx.flags["flexible_hps"] = flexible_hps
    ctx.flags["flexible_loads"] = flexible_loads
    ctx.flags["flexible_storage_units"] = flexible_storage_units

    # pm_optimize handles a non-contiguous (reduced) time index internally:
    # it runs one OPF per contiguous interval and merges the results.
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
