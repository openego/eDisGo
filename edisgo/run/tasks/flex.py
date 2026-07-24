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
Flex-asset import and operation-strategy tasks.

These tasks either pull flex assets (heat pumps, home batteries, DSM,
electromobility, generators) from egon_data / OEP into the topology,
or apply an operating strategy on assets already present. They must
run AFTER the grid is loaded (``setup_grid`` or ``load_from_base``)
and typically BEFORE the time-series step, so the time series can
cover the new assets.

The usual entry point is the collective :func:`task_import_flex`
(``import_flex``), which reads the top-level ``flexibilities:`` list
from the config and imports each named carrier. The per-carrier tasks
(``import_heat_pumps``, …) remain registered as building blocks for
custom pipelines.
"""

from __future__ import annotations

from edisgo.run.registry import register_task

#: Carriers accepted in the top-level ``flexibilities:`` list, mapped to
#: their import-task implementation (bound below, after the tasks are
#: defined).
FLEX_CARRIERS = ("heat_pumps", "home_batteries", "dsm", "electromobility")


@register_task("import_flex", requires={"grid"}, provides={"flex"})
def task_import_flex(
    edisgo,
    ctx,
    *,
    carriers=None,
    heat_pumps=None,
    home_batteries=None,
    dsm=None,
    electromobility=None,
):
    """
    Import all flexibilities selected in the config's ``flexibilities:`` list.

    Reads the top-level ``flexibilities:`` section (a list of carrier
    names) and runs the matching per-carrier import task for each
    entry, in the order of :data:`FLEX_CARRIERS`. The same list is the
    default for the ``optimize`` task's ``flexible`` selection, so one
    config key controls both which assets are imported and which are
    optimized.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Reads ``ctx.raw_config['flexibilities']`` when
        ``carriers`` is not given.
    carriers : list of str, optional
        Explicit carrier selection overriding the config's
        ``flexibilities:`` list. Valid entries: ``"heat_pumps"``,
        ``"home_batteries"``, ``"dsm"``, ``"electromobility"``.
    heat_pumps : dict, optional
        Extra keyword arguments for :func:`task_import_heat_pumps`
        (e.g. ``{"import_types": [...]}``).
    home_batteries : dict, optional
        Extra keyword arguments for :func:`task_import_home_batteries`.
    dsm : dict, optional
        Extra keyword arguments for :func:`task_import_dsm`.
    electromobility : dict, optional
        Extra keyword arguments for
        :func:`task_import_electromobility` (e.g.
        ``{"charging_strategy": "dumb"}``).

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    Raises
    ------
    ValueError
        If no carriers are selected (neither ``carriers`` nor a
        ``flexibilities:`` config section) or an unknown carrier name
        is listed.

    """
    if carriers is None:
        carriers = ctx.raw_config.get("flexibilities")
    if not carriers:
        raise ValueError(
            "Task 'import_flex' has nothing to import: provide a top-level "
            "'flexibilities' list in the config (or a 'carriers' task "
            f"parameter). Valid carriers: {list(FLEX_CARRIERS)}."
        )
    unknown = [c for c in carriers if c not in FLEX_CARRIERS]
    if unknown:
        raise ValueError(
            f"Unknown flexibilities {unknown}. Valid carriers: {list(FLEX_CARRIERS)}."
        )

    carrier_kwargs = {
        "heat_pumps": heat_pumps or {},
        "home_batteries": home_batteries or {},
        "dsm": dsm or {},
        "electromobility": electromobility or {},
    }
    for carrier in FLEX_CARRIERS:
        if carrier not in carriers:
            continue
        result = _CARRIER_TASKS[carrier](edisgo, ctx, **carrier_kwargs[carrier])
        if result is not None:
            edisgo = result
    return edisgo


@register_task("import_heat_pumps", requires={"grid"}, provides={"flex"})
def task_import_heat_pumps(edisgo, ctx, *, import_types=None, timeindex=None):
    """
    Import heat pumps from egon_data into the topology.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Uses ``ctx.scenario`` and
        ``ctx.ensure_engine()``. Sets
        ``ctx.flags['has_heat_pumps']`` to the observed count.
    import_types : list of str, optional
        Subset of ``["individual_heat_pumps", "central_heat_pumps"]``;
        default imports both.
    timeindex : pandas.DatetimeIndex, optional
        Restrict COP / heat-demand time series to this index. If None,
        falls back to ``ctx.flags['selected_timeindex']`` (set by a
        preceding ``set_timeindex`` step) so the download is
        restricted to the selected steps.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    if timeindex is None:
        timeindex = ctx.flags.get("selected_timeindex")
    edisgo.import_heat_pumps(
        scenario=ctx.scenario,
        engine=ctx.ensure_engine(),
        timeindex=timeindex,
        import_types=import_types,
    )
    ctx.flags["has_heat_pumps"] = (
        len(edisgo.topology.loads_df.loc[edisgo.topology.loads_df.type == "heat_pump"])
        > 0
    )
    return edisgo


@register_task("import_home_batteries", requires={"grid"}, provides={"flex"})
def task_import_home_batteries(edisgo, ctx):
    """
    Import home batteries from egon_data into the topology.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Uses ``ctx.scenario`` and
        ``ctx.ensure_engine()``. Sets
        ``ctx.flags['has_home_batteries']``.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    edisgo.import_home_batteries(scenario=ctx.scenario, engine=ctx.ensure_engine())
    ctx.flags["has_home_batteries"] = not edisgo.topology.storage_units_df.empty
    return edisgo


@register_task("import_dsm", requires={"grid"}, provides={"flex"})
def task_import_dsm(edisgo, ctx, *, timeindex=None):
    """
    Import demand-side-management potential from egon_data.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Uses ``ctx.scenario`` and
        ``ctx.ensure_engine()``. Sets ``ctx.flags['has_dsm']``.
    timeindex : pandas.DatetimeIndex, optional
        Restrict DSM availability time series to this index. If None,
        falls back to ``ctx.flags['selected_timeindex']`` (set by a
        preceding ``set_timeindex`` step) so the download is
        restricted to the selected steps.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    if timeindex is None:
        timeindex = ctx.flags.get("selected_timeindex")
    edisgo.import_dsm(
        scenario=ctx.scenario,
        engine=ctx.ensure_engine(),
        timeindex=timeindex,
    )
    ctx.flags["has_dsm"] = edisgo.dsm.p_max is not None and not edisgo.dsm.p_max.empty
    return edisgo


@register_task("import_electromobility", requires={"grid"}, provides={"flex"})
def task_import_electromobility(
    edisgo,
    ctx,
    *,
    data_source="oedb",
    charging_strategy="dumb",
    flexibility_bands_ucs=None,
    import_electromobility_data_kwds=None,
    allocate_charging_demand_kwds=None,
):
    """
    Import electromobility data (charging processes + parks).

    Optionally applies a charging strategy directly after import to
    turn the raw charging processes into active-power time series on
    the charging points.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. Uses ``ctx.scenario`` and
        ``ctx.ensure_engine()`` (for ``data_source='oedb'``). Sets
        ``ctx.flags['has_electromobility'] = True``.
    data_source : str, optional
        ``"oedb"`` (egon_data) or ``"directory"`` (requires
        ``import_electromobility_data_kwds={"charging_processes_dir":
        ..., "potential_charging_points_dir": ...}``).
    charging_strategy : str or None, optional
        Charging strategy applied right after import. ``"dumb"``
        (uncontrolled, default), ``"reduced"``, ``"residual"``, or
        ``None`` to skip.
    flexibility_bands_ucs : str or list of str, optional
        Charging-point use case(s) to compute flexibility bands for
        via :meth:`Electromobility.get_flexibility_bands` after import
        and charging-strategy application. Valid entries:
        ``"home"``, ``"work"``, ``"public"``, ``"hpc"``. Pass a single
        string for one use case or a list for multiple. ``None``
        (default) skips flexibility-band computation — build them later
        with the standalone :func:`task_build_flexibility_bands` once the
        analysis time index is fixed, so the bands are resampled to it
        (mirrors heat-pump handling, where the HP time series are not set
        inside ``import_heat_pumps``).
    import_electromobility_data_kwds : dict, optional
        Extra kwargs passed through to the underlying importer.
    allocate_charging_demand_kwds : dict, optional
        Extra kwargs for charging-demand allocation.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    edisgo.import_electromobility(
        data_source=data_source,
        scenario=ctx.scenario,
        engine=ctx.ensure_engine(),
        import_electromobility_data_kwds=import_electromobility_data_kwds,
        allocate_charging_demand_kwds=allocate_charging_demand_kwds,
    )
    if charging_strategy:
        edisgo.apply_charging_strategy(strategy=charging_strategy)
    if flexibility_bands_ucs is not None:
        edisgo.electromobility.get_flexibility_bands(
            edisgo,
            use_case=flexibility_bands_ucs,
        )
    ctx.flags["has_electromobility"] = True
    return edisgo


@register_task("build_flexibility_bands", requires={"flex"})
def task_build_flexibility_bands(edisgo, ctx, *, use_case=None):
    """
    Build EV charging flexibility bands from imported electromobility data.

    Standalone variant of the band computation that
    :func:`task_import_electromobility` can do inline. Running it as a
    separate step lets it execute *after* the analysis time index is fixed
    (e.g. after ``oedb_ts`` / timestep selection), so
    :meth:`Electromobility.get_flexibility_bands` resamples the bands to the
    edisgo time-series frequency instead of leaving them at the raw SimBEV
    resolution. This mirrors how the heat-pump time series are set outside
    ``import_heat_pumps``, and is more efficient than building bands over a
    non-final index.

    Skipped with an info-log if no electromobility data is present
    (``ctx.flags['has_electromobility']`` is falsy), so presets can
    include this step regardless of the ``flexibilities:`` selection.

    ``get_flexibility_bands`` only resamples to the active *frequency* - the
    bands still span whatever date range the underlying SimBEV charging-
    process data covers, which is not necessarily the same range as
    ``edisgo.timeseries.timeindex`` (e.g. a manually-selected window). This
    task additionally trims the bands down to that exact index, so
    ``electromobility.flexibility_bands`` always matches
    ``edisgo.timeseries.timeindex`` after this step runs.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context.
    use_case : str or list of str, optional
        Charging-point use case(s) to compute bands for. Valid entries:
        ``"home"``, ``"work"``, ``"public"``, ``"hpc"``. Defaults to all
        four.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.
    """
    from edisgo.tools.tools import reduce_timeseries_data_to_given_timeindex

    if not ctx.flags.get("has_electromobility"):
        ctx.logger.info(
            "Skipping 'build_flexibility_bands': no electromobility data present."
        )
        return edisgo
    if use_case is None:
        use_case = ["home", "work", "public", "hpc"]
    edisgo.electromobility.get_flexibility_bands(edisgo, use_case=use_case)
    reduce_timeseries_data_to_given_timeindex(
        edisgo,
        edisgo.timeseries.timeindex,
        timeseries=False,
        electromobility=True,
        heat_pump=False,
        dsm=False,
        overlying_grid=False,
    )
    return edisgo


@register_task("apply_charging_strategy")
def task_apply_charging_strategy(
    edisgo, ctx, *, strategy="dumb", charging_park_ids=None
):
    """
    Apply a charging strategy to the already-imported EV fleet.

    Standalone variant of the step that ``import_electromobility``
    does inline. Useful when you want to import once and then try
    multiple strategies in different runs. Skipped with an info-log
    if no electromobility data is present
    (``ctx.flags['has_electromobility']`` is falsy), so presets can
    include this step regardless of the ``flexibilities:`` selection.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context.
    strategy : str, optional
        Strategy name (``"dumb"`` / ``"reduced"`` / ``"residual"``).
    charging_park_ids : list of int, optional
        Restrict the strategy to these charging-park IDs.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    if not ctx.flags.get("has_electromobility"):
        ctx.logger.info(
            "Skipping 'apply_charging_strategy': no electromobility data present."
        )
        return edisgo
    edisgo.apply_charging_strategy(
        strategy=strategy, charging_park_ids=charging_park_ids
    )
    return edisgo


@register_task("apply_heat_pump_strategy")
def task_apply_heat_pump_strategy(
    edisgo, ctx, *, strategy="uncontrolled", heat_pump_names=None
):
    """
    Apply a heat-pump operating strategy.

    Skipped with an info-log if no heat pumps are present
    (``ctx.flags['has_heat_pumps']`` is falsy), so pipelines can
    safely include this step without a conditional guard.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context.
    strategy : str, optional
        Operating strategy (``"uncontrolled"``, ``"flexible"``, …).
    heat_pump_names : list of str, optional
        Restrict to specific heat-pump load names; default is all.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    if not ctx.flags.get("has_heat_pumps"):
        ctx.logger.info("Skipping 'apply_heat_pump_strategy': no heat pumps present.")
        return edisgo
    edisgo.apply_heat_pump_operating_strategy(
        strategy=strategy, heat_pump_names=heat_pump_names
    )
    return edisgo


@register_task("import_generators")
def task_import_generators(edisgo, ctx, *, generator_scenario=None):
    """
    Import future generators for the active scenario.

    Thin wrapper around :meth:`EDisGo.import_generators`. Mostly
    useful when you want to split grid loading and generator import
    into two separate pipeline steps.

    Parameters
    ----------
    edisgo : edisgo.EDisGo
        EDisGo instance to modify in place.
    ctx : RunContext
        Run context. ``ctx.scenario`` is used if
        ``generator_scenario`` is not given.
    generator_scenario : str, optional
        Scenario name, e.g. ``"nep2035"`` or ``"ego100"``. Defaults
        to ``ctx.scenario``.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    edisgo.import_generators(
        generator_scenario=generator_scenario or ctx.scenario,
        engine=ctx.ensure_engine(),
    )
    return edisgo


# Bound here (after the task definitions) so task_import_flex can dispatch
# per carrier without repeating the mapping in every call.
_CARRIER_TASKS = {
    "heat_pumps": task_import_heat_pumps,
    "home_batteries": task_import_home_batteries,
    "dsm": task_import_dsm,
    "electromobility": task_import_electromobility,
}
