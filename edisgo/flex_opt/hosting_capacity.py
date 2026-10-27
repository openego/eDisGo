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
Hosting capacity of single buses.

The hosting capacity of a bus is the largest additional generation or load that
can be connected to it without violating the technical constraints checked in
grid reinforcement (line and station loading, allowed voltage band). It answers
the question a grid operator asks for every connection request: how much fits at
this bus before the grid has to be reinforced?
"""

from __future__ import annotations

import copy
import logging

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from edisgo.flex_opt import check_tech_constraints, q_control
from edisgo.io import pypsa_io
from edisgo.network.results import Results
from edisgo.network.timeseries import TimeSeries

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)

TEST_COMPONENT = "hosting_capacity_test_component"

COMPONENT_TYPES = ["generator", "conventional_load", "charging_point", "heat_pump"]

# numerical tolerance when comparing loading and voltage deviation against the
# reference state, so that power flow noise is not reported as a violation
_CHECK_TOLERANCE = 1e-6


def hosting_capacity(
    edisgo_obj: EDisGo,
    buses: str | list[str] | pd.Index,
    component_type: str = "generator",
    timesteps: pd.Timestamp | pd.DatetimeIndex | None = None,
    profile: pd.Series | None = None,
    power_factor: float | None = None,
    reactive_power_mode: str | None = None,
    split_voltage_band: bool = True,
    p_start: float = 0.1,
    p_max: float = 100.0,
    atol: float = 0.001,
    rtol: float = 0.01,
) -> pd.DataFrame:
    """
    Determines the hosting capacity of the given buses.

    For every bus, a test component (a generator or a load) is placed at the bus
    and its active power is increased until one of the technical constraints
    checked in grid reinforcement is violated: a line or station is overloaded
    (see :func:`~.flex_opt.check_tech_constraints.components_relative_load`) or a
    bus voltage leaves the allowed voltage band (see
    :func:`~.flex_opt.check_tech_constraints.voltage_deviation_from_allowed_voltage_limits`).
    The power is first doubled starting at `p_start` until a violation occurs and
    then bisected until the given tolerance is reached. Each step is checked with
    a non-linear power flow of the whole grid.

    Constraints are checked relative to the grid as it is: a component that is
    already overloaded, or a bus whose voltage is already outside the allowed
    band, only counts as violated if the test component makes it worse. The
    hosting capacity is therefore the additional power that does not create new
    problems or worsen existing ones. Run :attr:`~.edisgo.EDisGo.reinforce` first
    to get the hosting capacity of a grid without existing problems.

    The given :class:`~.EDisGo` object is not changed. The buses are analysed
    independently of each other, so the hosting capacities of several buses
    cannot be added up.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    buses : str or list(str) or :pandas:`pandas.Index<Index>`
        Names of the buses to determine the hosting capacity for, as in index of
        :attr:`~.network.topology.Topology.buses_df`.
    component_type : str
        Type of the test component. Possible options are 'generator' (hosting
        capacity for feed-in) and 'conventional_load', 'charging_point' and
        'heat_pump' (hosting capacity for load). The type only determines the
        sign of the active power and the default reactive power behaviour.
        Default: 'generator'.
    timesteps : :pandas:`pandas.Timestamp<Timestamp>` or \
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Time steps to check. Per default, all time steps in
        :attr:`~.network.timeseries.TimeSeries.timeindex` are used. For worst-case
        time series, these are the four worst cases. For long time series,
        consider passing only the critical time steps, e.g. from
        :func:`~.tools.temporal_complexity_reduction.get_most_critical_time_steps`.
    profile : :pandas:`pandas.Series<Series>` or None
        Normalised active power profile of the test component, indexed by time
        step. The active power of the test component in a time step is the tested
        power times the profile value. Per default, the test component is operated
        at full power in all time steps. Default: None.
    power_factor : float or None
        Power factor of the test component. Per default, the power factor of the
        given `component_type` and the bus' voltage level is taken from the
        config section `reactive_power_factor`. Default: None.
    reactive_power_mode : str or None
        Reactive power mode of the test component, 'inductive' or 'capacitive'.
        Per default, it is taken from the config section `reactive_power_mode`.
        Default: None.
    split_voltage_band : bool
        Whether the allowed voltage band is split between the voltage levels, see
        :func:`~.flex_opt.check_tech_constraints.allowed_voltage_limits`.
        Default: True.
    p_start : float
        First tested active power in MW. Default: 0.1.
    p_max : float
        Largest tested active power in MW. If no constraint is violated at
        `p_max`, the hosting capacity is reported as `p_max`. Default: 100.
    atol : float
        Absolute tolerance of the hosting capacity in MW. Default: 0.001.
    rtol : float
        Relative tolerance of the hosting capacity. The search stops as soon as
        the hosting capacity is known up to the larger of `atol` and `rtol` times
        the power at which a violation occurred. Default: 0.01.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with the buses as index and the following columns:

        hosting_capacity : float
            Largest tested active power in MW at which no constraint is violated.
        limiting_constraint : str or None
            Constraint that is violated first: 'overloading', 'overvoltage',
            'undervoltage' or 'non_convergence' if the power flow does not
            converge. None if no constraint is violated up to `p_max`.
        limiting_component : str or None
            Name of the line or station (overloading) or bus (voltage) where the
            constraint is violated first.
        limiting_timestep : :pandas:`pandas.Timestamp<Timestamp>` or None
            Time step in which the constraint is violated first.
        voltage_level : str
            Voltage level of the bus, 'mv' or 'lv'.

    """
    if component_type not in COMPONENT_TYPES:
        raise ValueError(
            f"Given component_type '{component_type}' is not valid. Valid options "
            f"are {COMPONENT_TYPES}."
        )
    if not 0 < p_start <= p_max:
        raise ValueError("p_start and p_max must fulfil 0 < p_start <= p_max.")

    buses = pd.Index([buses] if isinstance(buses, str) else buses)
    missing = buses.difference(edisgo_obj.topology.buses_df.index)
    if len(missing) > 0:
        raise ValueError(f"The following buses are not in the grid: {list(missing)}.")

    if timesteps is None:
        timesteps = edisgo_obj.timeseries.timeindex
    elif isinstance(timesteps, pd.Timestamp):
        timesteps = pd.DatetimeIndex([timesteps])
    # copy, because PyPSA renames the index of the snapshots it is given, which
    # would also rename the time index of `edisgo_obj`
    timesteps = pd.DatetimeIndex(timesteps).copy()
    if len(timesteps) == 0:
        raise ValueError("No time steps to check. Set time series or pass `timesteps`.")

    if profile is None:
        profile = pd.Series(1.0, index=timesteps)
    else:
        profile = profile.reindex(timesteps)
        if profile.isna().any():
            raise ValueError("`profile` must contain a value for every time step.")

    checker = _HostingCapacityCheck(
        edisgo_obj, component_type, timesteps, profile, split_voltage_band
    )

    results = []
    for bus in buses:
        voltage_level = (
            "mv" if bus in edisgo_obj.topology.mv_grid.buses_df.index else "lv"
        )
        q_per_p = _reactive_power_per_active_power(
            edisgo_obj,
            component_type,
            voltage_level,
            power_factor,
            reactive_power_mode,
        )
        checker.move_to(bus, q_per_p)
        capacity, violation = _search_hosting_capacity(
            checker.check, p_start, p_max, atol, rtol
        )
        results.append(
            {
                "bus": bus,
                "hosting_capacity": capacity,
                "limiting_constraint": violation[0] if violation else None,
                "limiting_component": violation[1] if violation else None,
                "limiting_timestep": violation[2] if violation else None,
                "voltage_level": voltage_level,
            }
        )
        logger.debug(f"Hosting capacity at bus {bus}: {capacity:.3f} MW.")

    return pd.DataFrame(results).set_index("bus")


def _search_hosting_capacity(check, p_start, p_max, atol, rtol):
    """
    Searches the largest power at which `check` reports no violation.

    The power is doubled starting at `p_start` until `check` reports a violation
    or `p_max` is reached, then the interval between the last power without and
    the first power with violation is bisected.

    Parameters
    ----------
    check : callable
        Takes the active power in MW and returns None if no constraint is
        violated, else a tuple (constraint, component, time step).
    p_start, p_max, atol, rtol : float
        See :func:`hosting_capacity`.

    Returns
    -------
    tuple(float, tuple or None)
        Hosting capacity in MW and the violation at the smallest tested power
        above it (None if no violation occurred up to `p_max`).

    """
    lower = 0.0
    upper, violation = None, None
    p = p_start
    while upper is None:
        result = check(p)
        if result is None:
            lower = p
            if p >= p_max:
                return p_max, None
            p = min(2 * p, p_max)
        else:
            upper, violation = p, result

    while upper - lower > max(atol, rtol * upper):
        p = (lower + upper) / 2
        result = check(p)
        if result is None:
            lower = p
        else:
            upper, violation = p, result
    return lower, violation


def _reactive_power_per_active_power(
    edisgo_obj, component_type, voltage_level, power_factor, reactive_power_mode
):
    """
    Returns the reactive power of the test component per MW active power.

    The sign follows the eDisGo convention for the given component type, see
    :func:`~.flex_opt.q_control.fixed_cosphi`.

    """
    comp_df = pd.DataFrame({"voltage_level": [voltage_level]}, index=[TEST_COMPONENT])
    if power_factor is None:
        power_factor = q_control._fixed_cosphi_default_power_factor(
            comp_df, component_type, edisgo_obj.config
        ).iloc[0]
    if reactive_power_mode is None:
        q_sign = q_control._fixed_cosphi_default_reactive_power_sign(
            comp_df, component_type, edisgo_obj.config
        ).iloc[0]
    elif component_type == "generator":
        q_sign = q_control.get_q_sign_generator(reactive_power_mode)
    else:
        q_sign = q_control.get_q_sign_load(reactive_power_mode)
    return q_control.fixed_cosphi(1.0, q_sign, power_factor)


class _HostingCapacityCheck:
    """
    Power flow model with one movable test component.

    The PyPSA network is built once. For every bus, only the bus of the test
    component and its time series are changed, so the network does not have to
    be rebuilt for every tested power.

    Power flow results are written to a lightweight copy of the
    :class:`~.EDisGo` object (own results and time series, shared topology and
    config), so that the checks from
    :mod:`~.flex_opt.check_tech_constraints` can be reused and the original object
    stays unchanged. The test component is part of the copy's time series, so
    that the load and feed-in case used for the allowed line and station loading
    accounts for it.

    """

    def __init__(
        self, edisgo_obj, component_type, timesteps, profile, split_voltage_band
    ):
        self.timesteps = timesteps
        self.profile = profile
        self.q_per_p = 0.0
        self.split_voltage_band = split_voltage_band
        self.is_generator = component_type == "generator"

        self.edisgo = copy.copy(edisgo_obj)
        self.edisgo.results = Results(self.edisgo)
        self.edisgo.timeseries = _time_series_for_timesteps(
            edisgo_obj.timeseries, timesteps
        )

        self.pypsa = self.edisgo.to_pypsa(timesteps=timesteps)
        pypsa_comp = "Generator" if self.is_generator else "Load"
        self.pypsa_comp_t = (
            self.pypsa.generators_t if self.is_generator else self.pypsa.loads_t
        )
        self.pypsa.add(
            pypsa_comp,
            TEST_COMPONENT,
            bus=self.pypsa.buses.index[0],
            **({"control": "PQ"} if self.is_generator else {}),
        )
        self.ts_attr = "generators" if self.is_generator else "loads"

        # reference state without the test component
        self._set_power(0.0)
        if self._power_flow() is not None:
            raise ValueError(
                "Power flow analysis does not converge without the test "
                "component. Hosting capacity cannot be determined."
            )
        self.reference_loading = self._relative_loading().max()
        self.reference_voltage = self._voltage_deviation().abs().max()
        self.allowed_loading = self.reference_loading.clip(lower=1.0)

        n_overloaded = (self.reference_loading > 1.0 + _CHECK_TOLERANCE).sum()
        n_voltage_issues = (self.reference_voltage > _CHECK_TOLERANCE).sum()
        if n_overloaded + n_voltage_issues > 0:
            logger.warning(
                f"The grid already has {n_overloaded} overloaded lines or stations "
                f"and {n_voltage_issues} buses with voltage issues. Hosting "
                "capacities are determined such that these do not get worse, which "
                "can make them very small. Consider running EDisGo.reinforce first."
            )

    def move_to(self, bus, q_per_p):
        comp_df = self.pypsa.generators if self.is_generator else self.pypsa.loads
        comp_df.at[TEST_COMPONENT, "bus"] = bus
        self.q_per_p = q_per_p

    def check(self, p):
        """
        Returns the first violation at active power `p` or None.

        See :func:`_search_hosting_capacity` for the format.

        """
        self._set_power(p)
        not_converged = self._power_flow()
        if not_converged is not None:
            return "non_convergence", None, not_converged

        loading = self._relative_loading()
        exceedance = loading - self.allowed_loading
        if (exceedance.max() > _CHECK_TOLERANCE).any():
            component, timestep = exceedance.stack().idxmax()[::-1]
            return "overloading", component, timestep

        v_dev = self._voltage_deviation()
        exceedance = v_dev.abs() - self.reference_voltage
        if (exceedance.max() > _CHECK_TOLERANCE).any():
            timestep, bus = exceedance.stack().idxmax()
            constraint = (
                "overvoltage" if v_dev.at[timestep, bus] > 0 else ("undervoltage")
            )
            return constraint, bus, timestep
        return None

    def _set_power(self, p):
        active_power = p * self.profile
        reactive_power = active_power * self.q_per_p
        self.pypsa_comp_t.p_set[TEST_COMPONENT] = active_power
        self.pypsa_comp_t.q_set[TEST_COMPONENT] = reactive_power
        getattr(self.edisgo.timeseries, f"{self.ts_attr}_active_power")[
            TEST_COMPONENT
        ] = active_power

    def _power_flow(self):
        """
        Runs the power flow and returns the first not converged time step or None.

        """
        # the network topology does not change when the test component is moved,
        # so the preparation of the power flow is only needed in the first run;
        # the last power flow result is used as starting point
        first_run = self.edisgo.results.v_res.empty
        pf_results = self.pypsa.pf(
            self.timesteps, skip_pre=not first_run, use_seed=not first_run
        )
        converged = pf_results["converged"].all(axis=1)
        if not converged.all():
            return converged.index[~converged][0]
        pypsa_io.process_pfa_results(self.edisgo, self.pypsa, self.timesteps)
        return None

    def _relative_loading(self):
        loading = check_tech_constraints.components_relative_load(self.edisgo)
        return loading.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _voltage_deviation(self):
        return check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(
            self.edisgo, split_voltage_band=self.split_voltage_band
        )


def _time_series_for_timesteps(timeseries, timesteps):
    """
    Returns a new :class:`~.network.timeseries.TimeSeries` reduced to `timesteps`.

    The original time series stay unchanged.

    """
    reduced = TimeSeries(timeindex=timesteps)
    for attr in reduced._attributes:
        setattr(reduced, attr, getattr(timeseries, attr).loc[timesteps].copy())
    return reduced
