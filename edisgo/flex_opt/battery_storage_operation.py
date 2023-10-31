import logging

import pandas as pd

logger = logging.getLogger(__name__)


def reference_operation(
    df,
    soe_init,
    soe_max,
    storage_p_nom,
    freq,
    efficiency_charge=0.9,
    efficiency_discharge=0.9,
):
    """
    Reference operation of storage system where it directly charges when PV feed-in is
    higher than electricity demand of the building.

    Battery model handles generation positive, demand negative

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
    efficiency_charge : float
        Efficiency of storage system in case of charging.
    efficiency_discharge : float
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
        # If the house would feed electricity into the grid, charge the storage first.
        # No electricity exchange with grid as long as charger power is not exceeded.
        if (d.feedin_minus_demand > 0.0) & (storage_soe < soe_max):
            # Check if energy produced exceeds charger power
            if d.feedin_minus_demand < storage_p_nom:
                storage_power = -d.feedin_minus_demand
            # If it does, feed the rest to the grid
            else:
                storage_power = -storage_p_nom
            storage_soe = storage_soe + (-storage_power * efficiency_charge * freq)
            # If the storage is overcharged, feed the 'rest' to the grid
            if storage_soe > soe_max:
                storage_power = storage_power + (storage_soe - soe_max) / (
                    efficiency_charge * freq
                )
                storage_soe = soe_max

        # If the house needs electricity from the grid, discharge the storage first.
        # In this case d.feedin_minus_demand is negative!
        # No electricity exchange with grid as long as demand does not exceed charging
        # power
        elif (d.feedin_minus_demand < 0.0) & (storage_soe > 0.0):
            # Check if energy demand exceeds charger power
            if d.feedin_minus_demand / efficiency_discharge < (storage_p_nom * -1):
                storage_soe = storage_soe - (storage_p_nom * freq)
                storage_power = storage_p_nom * efficiency_discharge
            else:
                storage_soe = storage_soe + (
                    d.feedin_minus_demand / efficiency_discharge * freq
                )
                storage_power = -d.feedin_minus_demand
            # If the storage is undercharged, take the 'rest' from the grid
            if storage_soe < 0.0:
                # since storage_soe is negative in this case it can be taken as
                # demand
                storage_power = (
                    storage_power + storage_soe * efficiency_discharge / freq
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


def create_storage_data(edisgo_obj, soe_init=0.0, freq=1):
    """
    Matches storage units to PV plants and building electricity demand using the
    building ID and applies reference storage operation.
    The storage units active power time series are written to
    timeseries.loads_active_power.
    Reactive power is as well set with default values.
    State of energy time series is returned.

    In case there is no electricity load, the storage operation is set to zero.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object to obtain storage units and PV feed-in and electricity demand
        in same building from.
    soe_init : float
        Initial state of energy of storage device in MWh. Default: 0 MWh.
    freq : float
        Frequency of provided time series. Set to one, in case of hourly time series or
        0.5 in case of half-hourly time series. Default: 1.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with time index and state of energy in MWh of each storage in columns.
        Column names correspond to storage name as in topology.storage_units_df.

    """
    # ToDo add automatic determination of freq
    # ToDo allow setting efficiency through storage_units_df
    # ToDo allow specifying storage units for which to apply reference strategy
    storage_units = edisgo_obj.topology.storage_units_df
    soc_df = pd.DataFrame(index=edisgo_obj.timeseries.timeindex)
    # one storage per roof mounted solar generator
    for idx, row in storage_units.iterrows():
        building_id = row["building_id"]
        pv_gen = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.building_id == building_id
        ].index[0]
        pv_feedin = edisgo_obj.timeseries.generators_active_power[pv_gen]
        loads = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.building_id == building_id
        ].index
        if len(loads) == 0:
            logger.info(
                f"Storage unit {idx} in building {building_id} has not load. "
                f"Storage operation is therefore set to zero."
            )
            edisgo_obj.set_time_series_manual(
                storage_units_p=pd.DataFrame(
                    columns=[idx],
                    index=soc_df.index,
                    data=0.0,
                )
            )
        else:
            house_demand = edisgo_obj.timeseries.loads_active_power[loads].sum(axis=1)
            storage_ts = reference_operation(
                df=pd.DataFrame(
                    columns=["feedin_minus_demand"], data=pv_feedin - house_demand
                ),
                soe_init=soe_init,
                soe_max=row.p_nom * row.max_hours,
                storage_p_nom=row.p_nom,
                freq=freq,
            )
            # import matplotlib
            # from matplotlib import pyplot as plt
            # matplotlib.use('TkAgg', force=True)
            # storage_ts.plot()
            # plt.show()
            # Add storage time series to storage_units_active_power dataframe
            edisgo_obj.set_time_series_manual(
                storage_units_p=pd.DataFrame(
                    columns=[idx],
                    index=storage_ts.index,
                    data=storage_ts.storage_power.values,
                )
            )
            soc_df = pd.concat([soc_df, storage_ts.storage_soe], axis=1)

    soc_df.columns = edisgo_obj.topology.storage_units_df.index
    edisgo_obj.set_time_series_reactive_power_control()
    return soc_df
