import logging
import os
import random

from copy import deepcopy

import numpy as np
import pandas as pd

from edisgo.tools.tools import battery_storage_reference_operation

logger = logging.getLogger(__name__)


def create_hp_data(edisgo_obj, directory=None, save_edisgo=False):
    edisgo_obj.apply_heat_pump_operating_strategy()
    edisgo_obj.set_time_series_reactive_power_control()
    if save_edisgo:
        edisgo_obj.save(
            os.path.join(f"{directory}"),
            save_results=False,
            save_timeseries=True,
            save_electromobility=True,
            save_heatpump=True,
            save_dsm=True,
        )


def create_storage_data(edisgo_obj, directory=None, save_edisgo=False):
    storage_units = edisgo_obj.topology.generators_df.loc[
        edisgo_obj.topology.generators_df.index.str.contains("solar_roof")
    ]
    SOC_df = pd.DataFrame(index=edisgo_obj.timeseries.timeindex)
    # one storage per roof mounted solar generator
    for row in storage_units.iterrows():
        pv_feedin = edisgo_obj.timeseries.generators_active_power[row[0]]
        loads = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.bus == row[1].bus
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
            # Add storage instance
            edisgo_obj.add_component(
                comp_type="storage_unit",
                bus=row[1].bus,
                p_nom=row[1].p_nom,
                max_hours=1,
                ts_active_power=storage_ts.storage_power,
            )
            SOC_df = pd.concat([SOC_df, storage_ts.storage_charge], axis=1)

    SOC_df.columns = edisgo_obj.topology.storage_units_df.index
    edisgo_obj.timeseries.storage_units_state_of_charge = SOC_df
    edisgo_obj.set_time_series_reactive_power_control()
    if save_edisgo:
        edisgo_obj.save(
            os.path.join(f"{directory}"),
            save_results=False,
            save_timeseries=True,
            save_electromobility=True,
            save_heatpump=True,
            save_dsm=True,
        )


def create_storage_data_NG(edisgo_obj, directory=None, save_edisgo=False):
    storage_units = edisgo_obj.topology.storage_units_df
    SOC_df = pd.DataFrame(index=edisgo_obj.timeseries.timeindex)
    # one storage per roof mounted solar generator
    for row in storage_units.iterrows():
        building_id = row[1]["building_id"]
        pv_gen = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.building_id == building_id
        ].index[0]
        # lieber über building id?
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

            SOC_df = pd.concat([SOC_df, storage_ts.storage_charge], axis=1)

    SOC_df.columns = edisgo_obj.topology.storage_units_df.index
    edisgo_obj.timeseries.storage_units_state_of_charge = SOC_df
    edisgo_obj.set_time_series_reactive_power_control()
    if save_edisgo:
        edisgo_obj.save(
            os.path.join(f"{directory}"),
            save_results=False,
            save_timeseries=True,
            save_electromobility=True,
            save_heatpump=True,
            save_dsm=True,
        )


def create_dsm_data(
    edisgo_obj,
    dsm_data,
    timeindex,
    reinforce_grid=False,
    directory=None,
    save_edisgo=False,
):
    lv_loads = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.voltage_level == "lv"
    ]
    # amount of dsm loads
    amount_retail_dsm_loads = int(
        np.ceil(0.35 * len(lv_loads.loc[lv_loads.sector == "retail"]))
    )
    amount_industrial_dsm_loads = int(
        np.ceil(0.35 * len(lv_loads.loc[lv_loads.sector == "industrial"]))
    )
    amount_mv_dsm_loads = int(
        np.ceil(0.25 * (amount_retail_dsm_loads + amount_industrial_dsm_loads))
    )
    # name of lv loads to be replaced by dsm loads
    retail_dsm_loads = random.choices(
        lv_loads.loc[lv_loads.sector == "retail"].index.values,
        k=amount_retail_dsm_loads,
    )
    industrial_dsm_loads = random.choices(
        lv_loads.loc[lv_loads.sector == "industrial"].index.values,
        k=amount_industrial_dsm_loads,
    )

    dsm_data["name"] = [str("dsm_load_" + str(i)) for i in range(len(dsm_data))]
    # calculate peak load
    data = [dsm_data.p_set[i][0 : len(timeindex)] for i in range(len(dsm_data))]
    dsm_data["peak_load"] = [max(p) for p in data]
    lv_data = dsm_data.loc[dsm_data.peak_load < 0.3]
    mv_data = dsm_data.loc[(dsm_data.peak_load >= 0.3) & (dsm_data.peak_load < 4.5)]
    # name of dsm loads
    lv_dsm_loads_retail = lv_data.name.iloc[0:amount_retail_dsm_loads].values
    lv_dsm_loads_industrial = lv_data.name.iloc[
        amount_retail_dsm_loads : amount_retail_dsm_loads + amount_industrial_dsm_loads
    ].values

    if amount_mv_dsm_loads > len(mv_data):
        n = int(np.ceil(amount_mv_dsm_loads / len(mv_data)))
        mv_data = pd.concat([mv_data] * n)
        mv_name = ["mv_dsm_load_" + str(i) for i in range(len(mv_data))]
        mv_data.name = mv_name
    mv_dsm_loads = mv_data.name.iloc[0:amount_mv_dsm_loads].values

    loads = np.concatenate([lv_dsm_loads_retail, lv_dsm_loads_industrial, mv_dsm_loads])
    for parameter, parameter_str in [
        (dsm_data.p_set, "p_set"),
        (dsm_data.e_min, "e_min"),
        (dsm_data.p_min, "p_min"),
        (dsm_data.p_max, "p_max"),
        (dsm_data.e_max, "e_max"),
    ]:
        df = _create_dsm_parameter_data(
            dsm_data, parameter, parameter_str, loads, timeindex
        )
        if parameter_str == "e_min":
            edisgo_obj.dsm.e_min = df
        elif parameter_str == "e_max":
            edisgo_obj.dsm.e_max = df
        elif parameter_str == "p_min":
            edisgo_obj.dsm.p_min = df
        elif parameter_str == "p_max":
            edisgo_obj.dsm.p_max = df
        elif parameter_str == "p_set":
            # drop lv retail and industrial loads that will be replaced by dsm loads
            edisgo_obj.timeseries._loads_active_power = (
                edisgo_obj.timeseries.loads_active_power.drop(
                    np.concatenate([retail_dsm_loads, industrial_dsm_loads]), axis=1
                )
            )
            edisgo_obj.timeseries._loads_reactive_power = (
                edisgo_obj.timeseries.loads_reactive_power.drop(
                    np.concatenate([retail_dsm_loads, industrial_dsm_loads]), axis=1
                )
            )

            # add lv and mv dsm loads
            edisgo_obj.timeseries._loads_active_power = pd.concat(
                [
                    edisgo_obj.timeseries.loads_active_power,
                    pd.DataFrame(
                        df.values,
                        index=edisgo_obj.timeseries.loads_active_power.index,
                        columns=loads,
                    ),
                ],
                axis=1,
            )

    # check if p_max is higher than p_set in first timestep
    dsm_check = (
        edisgo_obj.timeseries._loads_active_power.loc[
            :, edisgo_obj.timeseries._loads_active_power.columns.str.contains("dsm")
        ]
        < edisgo_obj.dsm.p_max
    )
    for load in dsm_check.columns.values:
        if dsm_check[load][0]:
            logger.warning(
                "Max shiftable power (p_max) of dsm load "
                + str(load)
                + " is higher than total power (p_set) of dsm load at first "
                "timestep."
            )

    # add dsm loads to topology_df
    for vl in ["mv", "lv"]:
        if vl == "mv":
            dsm_df = pd.DataFrame(
                index=mv_dsm_loads, columns=edisgo_obj.topology.loads_df.columns
            )
            dsm_df.bus = random.choices(
                edisgo_obj.topology.buses_df.loc[
                    edisgo_obj.topology.buses_df.v_nom > 0.4
                ].index.values,
                k=amount_mv_dsm_loads,
            )
            dsm_df.p_set = (
                edisgo_obj.timeseries.loads_active_power[mv_dsm_loads].max().values
            )
            dsm_df.sector = "industrial"
        elif vl == "lv":
            dsm_df = pd.DataFrame(
                index=np.concatenate([lv_dsm_loads_retail, lv_dsm_loads_industrial]),
                columns=edisgo_obj.topology.loads_df.columns,
            )
            dsm_df.bus = edisgo_obj.topology.loads_df.loc[
                np.concatenate([retail_dsm_loads, industrial_dsm_loads])
            ].bus.values
            dsm_df.p_set = edisgo_obj.topology.loads_df.loc[
                np.concatenate([retail_dsm_loads, industrial_dsm_loads])
            ].p_set.values
            dsm_df.sector = edisgo_obj.topology.loads_df.loc[
                np.concatenate([retail_dsm_loads, industrial_dsm_loads])
            ].sector.values
        dsm_df.type = "conventional_load"
        dsm_df.voltage_level = vl
        edisgo_obj.topology.loads_df = pd.concat([edisgo_obj.topology.loads_df, dsm_df])

    # drop lv loads from topology_df that will be replaced by dsm loads
    edisgo_obj.topology.loads_df = edisgo_obj.topology.loads_df.drop(
        np.concatenate([retail_dsm_loads, industrial_dsm_loads])
    )

    # set reactive power timeseries
    edisgo_obj.set_time_series_reactive_power_control()
    if reinforce_grid:
        # Reinforce grid to accommodate for additional mv dsm loads
        edisgo_obj_copy = deepcopy(edisgo_obj)
        simultaneties = [
            "mv_feed-in_case_cp_home",
            "mv_feed-in_case_cp_work",
            "mv_feed-in_case_cp_public",
            "mv_feed-in_case_cp_hpc",
            "lv_feed-in_case_cp_home",
            "lv_feed-in_case_cp_work",
            "lv_feed-in_case_cp_public",
            "lv_feed-in_case_cp_hpc",
            "mv_load_case_cp_home",
            "mv_load_case_cp_work",
            "mv_load_case_cp_public",
            "mv_load_case_cp_hpc",
            "lv_load_case_cp_home",
            "lv_load_case_cp_work",
            "lv_load_case_cp_public",
            "lv_load_case_cp_hpc",
            "mv_feed-in_case_hp",
            "lv_feed-in_case_hp",
            "mv_load_case_hp",
            "lv_load_case_hp",
        ]

        for factor in simultaneties:
            edisgo_obj_copy.config._data["worst_case_scale_factor"][factor] = 0

        edisgo_obj_copy.set_time_series_worst_case_analysis()
        edisgo_obj_copy.reinforce()

        edisgo_obj.topology.lines_df = edisgo_obj_copy.topology.lines_df
        edisgo_obj.topology.transformers_df = edisgo_obj_copy.topology.transformers_df
        edisgo_obj.topology.transformers_hvmv_df = (
            edisgo_obj_copy.topology.transformers_hvmv_df
        )

    if save_edisgo:
        edisgo_obj.save(
            os.path.join(f"{directory}"),
            save_results=False,
            save_timeseries=True,
            save_electromobility=True,
            save_heatpump=True,
            save_dsm=True,
        )


def _create_dsm_parameter_data(dsm_data, parameter, parameter_str, loads, timeindex):
    data = [parameter[i][0 : len(timeindex)] for i in range(len(dsm_data))]
    # Check if min/max values are below/above 0
    if parameter_str == "e_min":
        if (pd.DataFrame(data) > 0).any().any():
            logger.warning("e_min is bigger than 0 for some DSM loads.")
    elif parameter_str == "e_max":
        if (pd.DataFrame(data) < 0).any().any():
            logger.warning("e_max is smaller than 0 for some DSM loads.")
    elif parameter_str == "p_min":
        if (pd.DataFrame(data) > 0).any().any():
            logger.warning("p_min is bigger than 0 for some DSM loads.")
    elif parameter_str == "p_max":
        if (pd.DataFrame(data) < 0).any().any():
            logger.warning("p_max is smaller than 0 for some DSM loads.")
    elif parameter_str == "p_set":
        if (pd.DataFrame(data) < 0).any().any():
            logger.warning("p_set is smaller than 0 for some DSM loads.")
    return pd.DataFrame(
        columns=timeindex, index=dsm_data["name"], data=data
    ).transpose()[loads]
