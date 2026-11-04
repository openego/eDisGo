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

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def _reference_operation(
    df: pd.DataFrame,
    soe_init: float,
    soe_max: float,
    storage_p_nom: float,
    freq: float,
    efficiency_store: float,
    efficiency_dispatch: float,
) -> pd.DataFrame:
    """
    Reference operation of storage system where it is directly charged when PV feed-in
    is higher than electricity demand of the building.

    Battery model handles generation positive, demand negative.

    Parameters
    -----------
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with time index and the buildings residual electricity demand
        (PV generation minus electricity demand) in column "feedin_minus_demand".
    soe_init : float
        Initial state of energy of storage device in MWh.
    soe_max : float
        Maximum energy level of storage device in MWh.
    storage_p_nom : float
        Nominal charging power of storage device in MW.
    freq : float
        Frequency of provided time series. Set to one, in case of hourly time series or
        0.5 in case of half-hourly time series.
    efficiency_store : float
        Efficiency of storage system in case of charging.
    efficiency_dispatch : float
        Efficiency of storage system in case of discharging.

    Returns
    ---------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe provided through parameter `df` extended by columns "storage_power",
        holding the charging (negative values) and discharging (positive values) power
        of the storage unit in MW, and "storage_soe" holding the storage unit's state of
        energy in MWh.

    """
    # Pull the single column needed for the recursion into a raw numpy array once,
    # so the per-timestep loop does not box each row into a pandas Series (iterrows).
    feedin_minus_demand = df["feedin_minus_demand"].to_numpy()
    n = feedin_minus_demand.shape[0]
    # Preallocate result arrays; results are written back to df once at the end.
    storage_power_arr = np.empty(n, dtype=float)
    storage_soe_arr = np.empty(n, dtype=float)
    storage_soe = soe_init

    # Sequential state-of-energy recursion with clamping. The math and the order of
    # operations within each iteration are kept exactly as in the original
    # iterrows-based implementation - only the iteration mechanism changed.
    for idx in range(n):
        fmd = feedin_minus_demand[idx]
        # If the house were to feed electricity into the grid, charge the storage first.
        # No electricity exchange with grid as long as charger power is not exceeded.
        if (fmd > 0.0) & (storage_soe < soe_max):
            # Check if energy produced exceeds charger power
            if fmd < storage_p_nom:
                storage_power = -fmd
            # If it does, feed the rest to the grid
            else:
                storage_power = -storage_p_nom
            storage_soe = storage_soe + (-storage_power * efficiency_store * freq)
            # If the storage is overcharged, feed the 'rest' to the grid
            if storage_soe > soe_max:
                storage_power = storage_power + (storage_soe - soe_max) / (
                    efficiency_store * freq
                )
                storage_soe = soe_max

        # If the house needs electricity from the grid, discharge the storage first.
        # In this case feedin_minus_demand is negative!
        # No electricity exchange with grid as long as demand does not exceed charging
        # power
        elif (fmd < 0.0) & (storage_soe > 0.0):
            # Check if energy demand exceeds charger power
            if fmd / efficiency_dispatch < (storage_p_nom * -1):
                storage_soe = storage_soe - (storage_p_nom * freq)
                storage_power = storage_p_nom * efficiency_dispatch
            else:
                storage_soe = storage_soe + (fmd / efficiency_dispatch * freq)
                storage_power = -fmd
            # If the storage is undercharged, take the 'rest' from the grid
            if storage_soe < 0.0:
                # since storage_soe is negative in this case it can be taken as
                # demand
                storage_power = storage_power + storage_soe * efficiency_dispatch / freq
                storage_soe = 0.0

        # If the storage is full or empty, the demand is not affected
        else:
            storage_power = 0.0
        storage_power_arr[idx] = storage_power
        storage_soe_arr[idx] = storage_soe

    df["storage_power"] = storage_power_arr
    df["storage_soe"] = storage_soe_arr

    return df.round(6)


def _check_inputs(edisgo_obj: EDisGo, storage_units: pd.DataFrame):
    """
    Raises a ValueError if data needed for the reference storage operation is missing.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
    storage_units : :pandas:`pandas.DataFrame<DataFrame>`
        Storage units to apply the reference operation to, with the same columns as
        :attr:`~.network.topology.Topology.storage_units_df`.

    """
    topology = edisgo_obj.topology
    timeseries = edisgo_obj.timeseries

    if len(timeseries.timeindex) == 0:
        raise ValueError(
            "No time index is set. Set time series of generators and loads first, "
            "e.g. using EDisGo.set_time_series_active_power_predefined()."
        )
    if "building_id" not in storage_units.columns:
        raise ValueError(
            "Topology.storage_units_df has no column 'building_id'. Import home "
            "batteries using EDisGo.import_home_batteries()."
        )
    if "building_id" not in topology.generators_df.columns:
        raise ValueError(
            "Topology.generators_df has no column 'building_id'. Import generators "
            "from the eGon database using EDisGo.import_generators()."
        )
    if "building_id" not in topology.loads_df.columns:
        raise ValueError(
            "Topology.loads_df has no column 'building_id'. Use a ding0 grid created "
            "from eGon data."
        )

    building_ids = storage_units.building_id.dropna().unique()
    gens = topology.generators_df[topology.generators_df.building_id.isin(building_ids)]
    loads = topology.loads_df[topology.loads_df.building_id.isin(building_ids)]
    # storage operation is only calculated for buildings with PV plant and load,
    # otherwise it is set to zero and no time series are needed
    buildings = set(gens.building_id) & set(loads.building_id)
    for comp_type, comps, ts_df in (
        ("generators", gens, timeseries.generators_active_power),
        ("loads", loads, timeseries.loads_active_power),
    ):
        missing = comps.index[
            comps.building_id.isin(buildings) & ~comps.index.isin(ts_df.columns)
        ]
        if len(missing) > 0:
            raise ValueError(
                f"No active power time series set for {comp_type} "
                f"{', '.join(missing)}. Set them first, e.g. using "
                f"EDisGo.set_time_series_active_power_predefined()."
            )


def apply_reference_operation(
    edisgo_obj: EDisGo,
    storage_units_names: list[str] | None = None,
    soe_init: float = 0.0,
    freq: float | None = None,
    efficiency_store: float = 0.95,
    efficiency_dispatch: float = 0.95,
) -> pd.DataFrame:
    """
    Applies reference storage operation to specified home storage units.

    In the reference storage operation, the home storage system is directly charged when
    PV feed-in is higher than electricity demand of the building until the storage
    is fully charged. The storage is directly discharged, in case electricity demand
    of the building is higher than the PV feed-in, until it is fully discharged.
    The battery model handles generation positive and demand negative.

    To determine the PV feed-in and electricity demand of the building that the home
    storage is located in (including demand from heat pumps
    and electric vehicles), this function matches the storage units to PV plants and
    building electricity demand using the building ID.
    In case there is no electricity load or no PV system, the storage operation is set
    to zero.

    The resulting storage units' active power time series are written to
    :attr:`~.network.timeseries.TimeSeries.storage_units_active_power`.
    Further, reactive power time series are set up using function
    :attr:`~.edisgo.EDisGo.set_time_series_reactive_power_control` with default values.
    The state of energy time series that are calculated within this function are not
    written anywhere, but are returned by this function.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object to obtain storage units and PV feed-in and electricity demand
        in same building from.
    storage_units_names : list(str) or None
        Names of storage units as in
        :attr:`~.network.topology.Topology.storage_units_df` to set time for. If None,
        time series are set for all storage units in
        :attr:`~.network.topology.Topology.storage_units_df` that have a building ID.
    soe_init : float
        Initial state of energy of storage device in MWh. Default: 0 MWh.
    freq : float or None
        Frequency of provided time series. Set to one, in case of hourly time series or
        0.5 in case of half-hourly time series. If None, it is derived from
        :attr:`~.network.timeseries.TimeSeries.timeindex`. Default: None.
    efficiency_store : float
        Efficiency of storage system in case of charging. Only used for storage units
        where `efficiency_store` is not set in
        :attr:`~.network.topology.Topology.storage_units_df`, e.g. home batteries
        imported from the eGon database. Default: 0.95.
    efficiency_dispatch : float
        Efficiency of storage system in case of discharging. Only used for storage
        units where `efficiency_dispatch` is not set in
        :attr:`~.network.topology.Topology.storage_units_df`. Default: 0.95.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with time index and state of energy in MWh of each storage in columns.
        Column names correspond to storage name as in
        :attr:`~.network.topology.Topology.storage_units_df`.

    Notes
    ------
    This function requires that the storage parameters `building_id` and
    `max_hours` are set in :attr:`~.network.topology.Topology.storage_units_df` for
    all storage units specified in parameter `storage_units_names`. Missing
    `efficiency_store` and `efficiency_dispatch` values are filled with the values
    given in the parameters of the same name. The filled values are only used in this
    function and are not written to
    :attr:`~.network.topology.Topology.storage_units_df`.

    """
    if storage_units_names is None:
        # storage units without building, e.g. large-scale storage, would otherwise
        # get their time series overwritten with zeros
        storage_units_df = edisgo_obj.topology.storage_units_df
        storage_units_names = (
            storage_units_df.index[storage_units_df.building_id.notna()]
            if "building_id" in storage_units_df.columns
            else storage_units_df.index
        )

    storage_units = edisgo_obj.topology.storage_units_df.loc[storage_units_names].copy()
    if storage_units.empty:
        return pd.DataFrame(index=edisgo_obj.timeseries.timeindex)
    # home batteries from the eGon database come without efficiencies
    for col, default in (
        ("efficiency_store", efficiency_store),
        ("efficiency_dispatch", efficiency_dispatch),
    ):
        storage_units[col] = storage_units[col].fillna(default).astype(float)
    _check_inputs(edisgo_obj, storage_units)

    soe_index = edisgo_obj.timeseries.timeindex
    if freq is None:
        freq = (
            (soe_index[1] - soe_index[0]) / pd.Timedelta("1h")
            if len(soe_index) > 1
            else 1
        )
    # Collect per-storage state-of-energy series and concatenate once after the loop
    # to avoid the quadratic cost of repeated pd.concat inside the loop.
    soe_series = []

    for stor_name, stor_data in storage_units.iterrows():
        # get corresponding PV systems and electric loads
        building_id = stor_data["building_id"]
        pv_gens = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.building_id == building_id
        ].index
        loads = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.building_id == building_id
        ].index
        if len(loads) == 0 or len(pv_gens) == 0:
            if len(loads) == 0:
                logger.warning(
                    f"Storage unit {stor_name} in building {building_id} has no load. "
                    f"Storage operation is therefore set to zero."
                )
            if len(pv_gens) == 0:
                logger.warning(
                    f"Storage unit {stor_name} in building {building_id} has no PV "
                    f"system. Storage operation is therefore set to zero."
                )
            edisgo_obj.set_time_series_manual(
                storage_units_p=pd.DataFrame(
                    columns=[stor_name],
                    index=soe_index,
                    data=0.0,
                )
            )
        else:
            # check storage values
            if pd.isna(stor_data.max_hours):
                raise ValueError(
                    f"Parameter max_hours for storage unit {stor_name} is not a "
                    f"number. It needs to be set in Topology.storage_units_df."
                )
            pv_feedin = edisgo_obj.timeseries.generators_active_power[pv_gens].sum(
                axis=1
            )
            house_demand = edisgo_obj.timeseries.loads_active_power[loads].sum(axis=1)
            # apply operation strategy
            storage_ts = _reference_operation(
                df=pd.DataFrame(
                    columns=["feedin_minus_demand"], data=pv_feedin - house_demand
                ),
                soe_init=soe_init,
                soe_max=stor_data.p_nom * stor_data.max_hours,
                storage_p_nom=stor_data.p_nom,
                freq=freq,
                efficiency_store=stor_data.efficiency_store,
                efficiency_dispatch=stor_data.efficiency_dispatch,
            )
            # add storage time series to storage_units_active_power dataframe
            edisgo_obj.set_time_series_manual(
                storage_units_p=pd.DataFrame(
                    columns=[stor_name],
                    index=storage_ts.index,
                    data=storage_ts.storage_power.values,
                )
            )
            soe_series.append(storage_ts.storage_soe.rename(stor_name))

    # Single concat preserves storage_units order (= collection order) and index.
    if soe_series:
        soe_df = pd.concat(soe_series, axis=1)
    else:
        soe_df = pd.DataFrame(index=soe_index)

    edisgo_obj.set_time_series_reactive_power_control(
        generators_parametrisation=None,
        loads_parametrisation=None,
        storage_units_parametrisation=pd.DataFrame(
            {
                "components": [storage_units_names],
                "mode": ["default"],
                "power_factor": ["default"],
            },
            index=[1],
        ),
    )

    return soe_df
