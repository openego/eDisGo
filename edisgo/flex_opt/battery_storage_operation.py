from copy import deepcopy

import pandas as pd


def battery_storage_reference_operation(
    df,
    init_storage_charge,
    storage_max,
    charger_power,
    time_base,
    efficiency_charge=0.9,
    efficiency_discharge=0.9,
):
    """
    Reference operation of storage system where it directly charges
    Todo: Find original source

    Parameters
    -----------
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Timeseries of house demand - PV generation
    init_storage_charge : float
        Initial state of energy of storage device
    storage_max : float
        Maximum energy level of storage device
    charger_power : float
        Nominal charging power of storage device
    time_base : float
        Timestep of inserted timeseries
    efficiency_charge: float
        Efficiency of storage system in case of charging
    efficiency_discharge: float
        Efficiency of storage system in case of discharging

    Returns
    ---------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with storage operation timeseries

    """
    # Battery model handles generation positive, demand negative
    lst_storage_power = []
    lst_storage_charge = []
    storage_charge = init_storage_charge

    for i, d in df.iterrows():
        # If the house would feed electricity into the grid, charge the storage first.
        # No electricity exchange with grid as long as charger power is not exceeded
        if (d.house_demand > 0) & (storage_charge < storage_max):
            # Check if energy produced exceeds charger power
            if d.house_demand < charger_power:
                storage_charge = storage_charge + (
                    d.house_demand * efficiency_charge * time_base
                )
                storage_power = -d.house_demand
            # If it does, feed the rest to the grid
            else:
                storage_charge = storage_charge + (
                    charger_power * efficiency_charge * time_base
                )
                storage_power = -charger_power

            # If the storage would be overcharged, feed the 'rest' to the grid
            if storage_charge > storage_max:
                storage_power = storage_power + (storage_charge - storage_max) / (
                    efficiency_charge * time_base
                )
                storage_charge = storage_max

        # If the house needs electricity from the grid, discharge the storage first.
        # In this case d.house_demand is negative!
        # No electricity exchange with grid as long as demand does not exceed charger
        # power
        elif (d.house_demand < 0) & (storage_charge > 0):
            # Check if energy demand exceeds charger power
            if d.house_demand / efficiency_discharge < (charger_power * -1):
                storage_charge = storage_charge - (charger_power * time_base)
                storage_power = charger_power * efficiency_discharge

            else:
                storage_charge = storage_charge + (
                    d.house_demand / efficiency_discharge * time_base
                )
                storage_power = -d.house_demand

            # If the storage would be undercharged, take the 'rest' from the grid
            if storage_charge < 0:
                # since storage_charge is negative in this case it can be taken as
                # demand
                storage_power = (
                    storage_power + storage_charge * efficiency_discharge / time_base
                )
                storage_charge = 0

        # If the storage is full or empty, the demand is not affected
        # elif(storage_charge == 0) | (storage_charge == storage_max):
        else:
            storage_power = 0
        lst_storage_power.append(storage_power)
        lst_storage_charge.append(storage_charge)
    df["storage_power"] = lst_storage_power
    df["storage_charge"] = lst_storage_charge

    return df.round(6)


def create_storage_data(edisgo_obj):
    storage_units = edisgo_obj.topology.storage_units_df
    soc_df = pd.DataFrame(index=edisgo_obj.timeseries.timeindex)
    # one storage per roof mounted solar generator
    for row in storage_units.iterrows():
        building_id = row[1]["building_id"]
        pv_gen = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.building_id == building_id
        ].index[0]
        pv_feedin = edisgo_obj.timeseries.generators_active_power[pv_gen]
        loads = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.building_id == building_id
        ].index
        if len(loads) == 0:
            pass
        else:
            house_demand = deepcopy(
                edisgo_obj.timeseries.loads_active_power[loads].sum(axis=1)
            )
            storage_ts = battery_storage_reference_operation(
                pd.DataFrame(columns=["house_demand"], data=pv_feedin - house_demand),
                0,
                row[1].p_nom,
                row[1].p_nom,
                1,
            )
            # Add storage ts to storage_units_active_power dataframe
            edisgo_obj.set_time_series_manual(
                storage_units_p=pd.DataFrame(
                    columns=[row[0]],
                    index=storage_ts.index,
                    data=storage_ts.storage_power.values,
                )
            )

            soc_df = pd.concat([soc_df, storage_ts.storage_charge], axis=1)

    soc_df.columns = edisgo_obj.topology.storage_units_df.index
    edisgo_obj.overlying_grid.storage_units_soc = soc_df
    edisgo_obj.set_time_series_reactive_power_control()
