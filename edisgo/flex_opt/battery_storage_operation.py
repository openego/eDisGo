import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)


def _reference_operation(
    df,
    soe_init,
    soe_max,
    storage_p_nom,
    freq,
    efficiency_store,
    efficiency_dispatch,
):
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
    lst_storage_power = []
    lst_storage_soe = []
    storage_soe = soe_init

    for i, d in df.iterrows():
        # If the house were to feed electricity into the grid, charge the storage first.
        # No electricity exchange with grid as long as charger power is not exceeded.
        if (d.feedin_minus_demand > 0.0) & (storage_soe < soe_max):
            # Check if energy produced exceeds charger power
            if d.feedin_minus_demand < storage_p_nom:
                storage_power = -d.feedin_minus_demand
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
        # In this case d.feedin_minus_demand is negative!
        # No electricity exchange with grid as long as demand does not exceed charging
        # power
        elif (d.feedin_minus_demand < 0.0) & (storage_soe > 0.0):
            # Check if energy demand exceeds charger power
            if d.feedin_minus_demand / efficiency_dispatch < (storage_p_nom * -1):
                storage_soe = storage_soe - (storage_p_nom * freq)
                storage_power = storage_p_nom * efficiency_dispatch
            else:
                storage_soe = storage_soe + (
                    d.feedin_minus_demand / efficiency_dispatch * freq
                )
                storage_power = -d.feedin_minus_demand
            # If the storage is undercharged, take the 'rest' from the grid
            if storage_soe < 0.0:
                # since storage_soe is negative in this case it can be taken as
                # demand
                storage_power = (
                    storage_power + storage_soe * efficiency_dispatch / freq
                )
                storage_soe = 0.0

        # If the storage is full or empty, the demand is not affected
        else:
            storage_power = 0.0
        lst_storage_power.append(storage_power)
        lst_storage_soe.append(storage_soe)

    df["storage_power"] = lst_storage_power
    df["storage_soe"] = lst_storage_soe

    return df.round(6)


def apply_reference_operation(edisgo_obj, storage_units_names=None, soe_init=0.0, freq=1):
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
    :attr:`~.network.timeseries.TimeSeries.loads_active_power`.
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
        :attr:`~.network.topology.Topology.storage_units_df`.
    soe_init : float
        Initial state of energy of storage device in MWh. Default: 0 MWh.
    freq : float
        Frequency of provided time series. Set to one, in case of hourly time series or
        0.5 in case of half-hourly time series. Default: 1.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with time index and state of energy in MWh of each storage in columns.
        Column names correspond to storage name as in
        :attr:`~.network.topology.Topology.storage_units_df`.

    Notes
    ------
    This function requires that the storage parameters `building_id`,
    `efficiency_store`, `efficiency_dispatch` and `max_hours` are set in
    :attr:`~.network.topology.Topology.storage_units_df` for all storage units
    specified in parameter `storage_units_names`.

    """
    if storage_units_names is None:
        storage_units_names = edisgo_obj.topology.storage_units_df.index

    storage_units = edisgo_obj.topology.storage_units_df.loc[storage_units_names]
    soe_df = pd.DataFrame(index=edisgo_obj.timeseries.timeindex)

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
                    index=soe_df.index,
                    data=0.0,
                )
            )
        else:
            # check storage values
            if math.isnan(stor_data.max_hours) is True:
                raise ValueError(
                    f"Parameter max_hours for storage unit {stor_name} is not a "
                    f"number. It needs to be set in Topology.storage_units_df."
                )
            if math.isnan(stor_data.efficiency_store) is True:
                raise ValueError(
                    f"Parameter efficiency_store for storage unit {stor_name} is not a "
                    f"number. It needs to be set in Topology.storage_units_df."
                )
            if math.isnan(stor_data.efficiency_dispatch) is True:
                raise ValueError(
                    f"Parameter efficiency_dispatch for storage unit {stor_name} is "
                    f"not a number. It needs to be set in Topology.storage_units_df."
                )
            pv_feedin = edisgo_obj.timeseries.generators_active_power[pv_gens].sum(axis=1)
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
            soe_df = pd.concat([soe_df, storage_ts.storage_soe.to_frame(stor_name)], axis=1)

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
