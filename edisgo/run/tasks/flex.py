"""
Flex-asset import and operation-strategy tasks.

These tasks either pull flex assets (heat pumps, home batteries, DSM,
electromobility, generators) from egon_data / OEP into the topology,
or apply an operating strategy on assets already present. They must
run AFTER the grid is loaded (``setup_grid`` or ``load_from_base``)
and typically BEFORE the time-series step, so the time series can
cover the new assets.
"""
from __future__ import annotations

from edisgo.run.registry import register_task


@register_task("import_heat_pumps")
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
        Restrict COP / heat-demand time series to this index.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    edisgo.import_heat_pumps(
        scenario=ctx.scenario,
        engine=ctx.ensure_engine(),
        timeindex=timeindex,
        import_types=import_types,
    )
    ctx.flags["has_heat_pumps"] = len(
        edisgo.topology.loads_df.loc[
            edisgo.topology.loads_df.type == "heat_pump"
        ]
    ) > 0
    return edisgo


@register_task("import_home_batteries")
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
    edisgo.import_home_batteries(
        scenario=ctx.scenario, engine=ctx.ensure_engine()
    )
    ctx.flags["has_home_batteries"] = (
        not edisgo.topology.storage_units_df.empty
    )
    return edisgo


@register_task("import_dsm")
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
        Restrict DSM availability time series to this index.

    Returns
    -------
    edisgo.EDisGo
        The modified EDisGo instance.

    """
    edisgo.import_dsm(
        scenario=ctx.scenario,
        engine=ctx.ensure_engine(),
        timeindex=timeindex,
    )
    ctx.flags["has_dsm"] = (
        edisgo.dsm.p_max is not None and not edisgo.dsm.p_max.empty
    )
    return edisgo


@register_task("import_electromobility")
def task_import_electromobility(edisgo, ctx, *, data_source="oedb",
                                charging_strategy="dumb",
                                flexibility_bands_ucs = None,
                                import_electromobility_data_kwds=None,
                                allocate_charging_demand_kwds=None):
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
        (default) skips flexibility-band computation.
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


@register_task("apply_charging_strategy")
def task_apply_charging_strategy(edisgo, ctx, *, strategy="dumb",
                                 charging_park_ids=None):
    """
    Apply a charging strategy to the already-imported EV fleet.

    Standalone variant of the step that ``import_electromobility``
    does inline. Useful when you want to import once and then try
    multiple strategies in different runs.

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
    edisgo.apply_charging_strategy(
        strategy=strategy, charging_park_ids=charging_park_ids
    )
    return edisgo


@register_task("apply_heat_pump_strategy")
def task_apply_heat_pump_strategy(edisgo, ctx, *, strategy="uncontrolled",
                                  heat_pump_names=None):
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
        ctx.logger.info(
            "Skipping 'apply_heat_pump_strategy': no heat pumps "
            "present."
        )
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
        generator_scenario=generator_scenario or ctx.scenario
    )
    return edisgo
