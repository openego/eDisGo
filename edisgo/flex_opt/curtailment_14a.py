"""
This module implements curtailment according to §14a EnWG (Netzausbaugebiet).

§14a EnWG allows network operators to temporarily curtail controllable consumption
devices (especially heat pumps and charging points) to a MINIMUM power of 4.2 kW.

IMPORTANT: This means devices can operate ABOVE 4.2 kW normally, and the operator
can reduce them DOWN TO (but not below) 4.2 kW.

The functions in this module are primarily for worst-case analysis and testing.
For realistic §14a implementation in grid optimization, use the OPF with the
curtailment_14a parameter in edisgo.optimize() or edisgo.reinforce().

"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def apply_curtailment_14a(
    edisgo_obj: EDisGo,
    components: list[str] | None = None,
    max_power_kw: float = 4.2,
    components_type: str | None = None,
) -> dict[str, pd.Series]:
    """
    Apply §14a EnWG curtailment to heat pumps and/or charging points.

    WARNING: This function applies PERMANENT curtailment to time series data,
    limiting all power values to a maximum of 4.2 kW. This is a WORST-CASE
    scenario and NOT how §14a is typically used in practice!

    For realistic §14a implementation, use the OPF with curtailment_14a parameter:
    >>> edisgo.optimize(curtailment_14a={'apply_curtailment': True})

    §14a EnWG allows network operators to curtail devices DOWN TO a minimum of
    4.2 kW (not TO a maximum of 4.2 kW). Devices normally operate at their
    requested power, and curtailment to 4.2 kW is only applied when needed.

    This function simulates a permanent worst-case where ALL power values are
    limited to 4.2 kW, which is useful for comparison but not realistic.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo object containing the network and time series data.
    components : list of str or None
        List of component names to apply curtailment to. If None, all
        heat pumps and charging points in the network are curtailed.
        Default: None.
    max_power_kw : float
        Maximum allowed power in kW after curtailment. According to §14a EnWG,
        this is typically 4.2 kW. Default: 4.2.
    components_type : str or None
        Type of components to curtail. Can be 'heat_pump', 'charging_point',
        or None. If None, both heat pumps and charging points are considered.
        Only used if `components` is None. Default: None.

    Returns
    -------
    dict of str to :pandas:`pandas.Series<Series>`
        Dictionary with component names as keys and the curtailed energy
        (in MWh) as values. This shows how much energy was curtailed for
        each component.

    Notes
    -----
    The curtailment is applied by limiting ALL active power time series values
    to the specified maximum power. The reactive power is adjusted
    proportionally based on the power factor.

    This function PERMANENTLY modifies time series data. For a non-destructive
    analysis, use check_curtailment_14a_effect() instead.

    For realistic §14a optimization (where curtailment only happens when needed
    to avoid grid issues), use the OPF:
    >>> edisgo.optimize(curtailment_14a={'apply_curtailment': True})

    """
    max_power_mw = max_power_kw / 1000.0  # Convert kW to MW

    # Get components to curtail
    if components is None:
        components = _get_curtailable_components(edisgo_obj, components_type)
    else:
        # Validate that provided components exist
        all_loads = edisgo_obj.topology.loads_df.index.tolist()
        invalid_components = [c for c in components if c not in all_loads]
        if invalid_components:
            raise ValueError(
                f"The following components do not exist in the network: "
                f"{invalid_components}"
            )

    if not components:
        logger.warning(
            "No components found for §14a curtailment. No curtailment applied."
        )
        return {}

    logger.info(
        f"Applying §14a EnWG curtailment to {len(components)} components "
        f"with max power {max_power_kw} kW."
    )

    curtailed_energy = {}

    # Get time series data
    ts_active = edisgo_obj.timeseries.loads_active_power
    ts_reactive = edisgo_obj.timeseries.loads_reactive_power

    for component in components:
        if component not in ts_active.columns:
            logger.warning(
                f"Component {component} has no active power time series. Skipping."
            )
            continue

        # Get original time series
        original_active = ts_active[component].copy()

        # Apply curtailment
        curtailed_active = original_active.clip(upper=max_power_mw)

        # Calculate curtailed energy (difference between original and curtailed)
        time_delta = (
            edisgo_obj.timeseries.timeindex[1] - edisgo_obj.timeseries.timeindex[0]
        ).total_seconds() / 3600.0  # in hours
        energy_curtailed = ((original_active - curtailed_active) * time_delta).sum()
        curtailed_energy[component] = energy_curtailed

        # Update active power time series
        ts_active[component] = curtailed_active

        # Adjust reactive power proportionally if it exists
        if component in ts_reactive.columns:
            original_reactive = ts_reactive[component].copy()
            # Calculate ratio of curtailed to original power (avoid division by zero)
            ratio = pd.Series(1.0, index=original_active.index)
            non_zero_mask = original_active != 0
            ratio[non_zero_mask] = (
                curtailed_active[non_zero_mask] / original_active[non_zero_mask]
            )
            # Apply same ratio to reactive power
            ts_reactive[component] = original_reactive * ratio

    logger.info(
        f"§14a curtailment applied. Total curtailed energy: "
        f"{sum(curtailed_energy.values()):.2f} MWh across {len(curtailed_energy)} "
        f"components."
    )

    return curtailed_energy


def _get_curtailable_components(
    edisgo_obj: EDisGo, components_type: str | None = None
) -> list[str]:
    """
    Get list of components that can be curtailed according to §14a EnWG.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo object containing the network topology.
    components_type : str or None
        Type of components to get. Can be 'heat_pump', 'charging_point',
        or None (both types). Default: None.

    Returns
    -------
    list of str
        List of component names that can be curtailed.

    """
    loads_df = edisgo_obj.topology.loads_df

    if components_type == "heat_pump":
        curtailable = loads_df[loads_df.type == "heat_pump"].index.tolist()
    elif components_type == "charging_point":
        curtailable = loads_df[loads_df.type == "charging_point"].index.tolist()
    elif components_type is None or components_type == "both":
        # Get both heat pumps and charging points
        curtailable = loads_df[
            loads_df.type.isin(["heat_pump", "charging_point"])
        ].index.tolist()
    else:
        raise ValueError(
            f"Invalid components_type '{components_type}'. Must be "
            f"'heat_pump', 'charging_point', 'both', or None."
        )

    return curtailable


def identify_components_for_curtailment(
    edisgo_obj: EDisGo,
    critical_components: list[str] | None = None,
    curtailment_priority: str = "p_set",
) -> list[str]:
    """
    Identify which components should be curtailed to avoid grid reinforcement.

    This function can be used as part of the grid reinforcement optimization
    to identify which components should be curtailed according to §14a EnWG
    instead of performing grid expansion.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo object containing the network and analysis results.
    critical_components : list of str or None
        List of components that are causing grid issues (overloading or
        voltage violations). If None, all curtailable components are considered.
        Default: None.
    curtailment_priority : str
        Defines how to prioritize components for curtailment. Options:
        'p_set' (largest nominal power first), 'random', 'grid_level'
        (start with LV, then MV). Default: 'p_set'.

    Returns
    -------
    list of str
        Sorted list of component names to curtail, ordered by priority.

    """
    # Get all curtailable components
    all_curtailable = _get_curtailable_components(edisgo_obj, components_type=None)

    if critical_components is not None:
        # Filter to only include critical components that are curtailable
        components = [c for c in critical_components if c in all_curtailable]
    else:
        components = all_curtailable

    if not components:
        return []

    # Apply prioritization
    loads_df = edisgo_obj.topology.loads_df.loc[components]

    if curtailment_priority == "p_set":
        # Prioritize components with highest nominal power
        sorted_components = loads_df.sort_values(
            "p_set", ascending=False
        ).index.tolist()
    elif curtailment_priority == "random":
        # Random order
        sorted_components = loads_df.sample(frac=1).index.tolist()
    elif curtailment_priority == "grid_level":
        # Prioritize based on grid level (LV first, then MV)
        buses_df = edisgo_obj.topology.buses_df
        loads_with_grid = loads_df.copy()
        loads_with_grid["v_nom"] = loads_with_grid["bus"].map(buses_df["v_nom"])
        # Sort by voltage level (ascending = LV first)
        sorted_components = loads_with_grid.sort_values("v_nom").index.tolist()
    else:
        raise ValueError(
            f"Invalid curtailment_priority '{curtailment_priority}'. "
            f"Must be 'p_set', 'random', or 'grid_level'."
        )

    return sorted_components


def check_curtailment_effect(
    edisgo_obj: EDisGo,
    components: list[str],
    max_power_kw: float = 4.2,
) -> dict[str, float]:
    """
    Check the effect of curtailing components without actually applying it.

    This function analyzes what the effect of curtailing specified components
    would be, without modifying the time series data.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo object containing the network and time series data.
    components : list of str
        List of component names to check curtailment effect for.
    max_power_kw : float
        Maximum allowed power in kW after curtailment. Default: 4.2.

    Returns
    -------
    dict of str to float
        Dictionary with the following keys:
        - 'total_curtailed_energy_mwh': Total energy that would be curtailed
        - 'max_simultaneous_curtailment_mw': Maximum simultaneous curtailed power
        - 'avg_curtailed_power_mw': Average curtailed power across all time steps
        - 'hours_with_curtailment': Number of hours with active curtailment

    """
    max_power_mw = max_power_kw / 1000.0

    ts_active = edisgo_obj.timeseries.loads_active_power

    # Calculate curtailment for each component
    total_curtailment = pd.Series(0.0, index=ts_active.index)

    for component in components:
        if component not in ts_active.columns:
            continue

        original = ts_active[component]
        curtailed = original.clip(upper=max_power_mw)
        curtailment = original - curtailed
        total_curtailment += curtailment

    # Calculate statistics
    time_delta = (
        edisgo_obj.timeseries.timeindex[1] - edisgo_obj.timeseries.timeindex[0]
    ).total_seconds() / 3600.0

    total_curtailed_energy = (total_curtailment * time_delta).sum()
    max_simultaneous_curtailment = total_curtailment.max()
    avg_curtailed_power = total_curtailment.mean()
    hours_with_curtailment = (total_curtailment > 0).sum() * time_delta

    return {
        "total_curtailed_energy_mwh": total_curtailed_energy,
        "max_simultaneous_curtailment_mw": max_simultaneous_curtailment,
        "avg_curtailed_power_mw": avg_curtailed_power,
        "hours_with_curtailment": hours_with_curtailment,
    }


def apply_curtailment_during_reinforcement(
    edisgo_obj: EDisGo,
    max_power_kw: float | None = None,
    components_type: str | None = None,
    curtailment_priority: str | None = None,
) -> dict[str, float]:
    """
    Apply §14a EnWG curtailment as part of grid reinforcement optimization.

    This function integrates §14a curtailment into the grid reinforcement workflow.
    It reads configuration from the config file and applies curtailment to
    controllable consumers before grid expansion is considered.

    This function is intended to be called from within the grid reinforcement
    process (e.g., from :func:`~.flex_opt.reinforce_grid.reinforce_grid`).

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo object containing the network and time series data.
    max_power_kw : float or None
        Maximum allowed power in kW after curtailment. If None, the value from
        the config is used. Default: None.
    components_type : str or None
        Type of components to curtail ('heat_pump', 'charging_point', 'both').
        If None, the value from the config is used. Default: None.
    curtailment_priority : str or None
        Curtailment priority strategy. If None, the value from the config is used.
        Default: None.

    Returns
    -------
    dict of str to float
        Dictionary with component names as keys and the curtailed energy
        (in MWh) as values.

    Notes
    -----
    This function should only be called if curtailment is enabled in the config
    (check `config['curtailment_14a_enwg']['enable_curtailment']`).

    """
    # Check if curtailment is enabled
    config = edisgo_obj.config
    if not config["curtailment_14a_enwg"]["enable_curtailment"]:
        logger.debug("§14a curtailment is disabled in config.")
        return {}

    # Read config values if not provided
    if max_power_kw is None:
        max_power_kw = float(config["curtailment_14a_enwg"]["max_power_kw"])

    if components_type is None:
        components_type = config["curtailment_14a_enwg"]["components_type"]
        if components_type == "both":
            components_type = None

    if curtailment_priority is None:
        curtailment_priority = config["curtailment_14a_enwg"]["curtailment_priority"]

    # Identify components to curtail
    components = identify_components_for_curtailment(
        edisgo_obj,
        critical_components=None,  # Consider all components
        curtailment_priority=curtailment_priority,
    )

    if not components:
        logger.debug("No components available for §14a curtailment.")
        return {}

    # Apply curtailment
    logger.debug(
        f"Applying §14a EnWG curtailment during grid reinforcement to "
        f"{len(components)} components."
    )

    curtailed_energy = apply_curtailment_14a(
        edisgo_obj,
        components=components,
        max_power_kw=max_power_kw,
        components_type=components_type,
    )

    return curtailed_energy
