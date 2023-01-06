import logging
import os

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
    storages = edisgo_obj.topology.generators_df.loc[
        edisgo_obj.topology.generators_df.index.str.contains("solar_roof")
    ]
    # one storage per roof mounted solar generator
    for row in storages.iterrows():
        pv_feedin = edisgo_obj.timeseries.generators_active_power[row[0]]
        loads = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.bus == row[1].bus
        ].index
        if len(loads) == 0:
            pass
        else:
            house_demand = edisgo_obj.timeseries.loads_active_power[loads].sum(axis=1)
            storage_ts = battery_storage_reference_operation(
                pd.DataFrame(house_demand - pv_feedin), 0, row[1].p_nom, row[1].p_nom, 1
            )
            # Add storage instance
            edisgo_obj.add_component(
                comp_type="storage_unit",
                bus=row[1].bus,
                p_nom=row[1].p_nom,
                max_hours=1,
                ts_active_power=storage_ts.storage_power,
            )

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


def create_dsm_data(edisgo_obj, dsm_data, timeindex, directory=None, save_edisgo=False):
    amount_dsm_loads = 5
    dsm_data["name"] = [str("dsm_load_" + str(i)) for i in range(len(dsm_data))]
    # ToDo: richtige DSM loads anhängen
    data = [
        np.fromstring(dsm_data.p_max_pu[i].strip("[]"), sep=",")[0 : len(timeindex)]
        * dsm_data.p_nom.iloc[i]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    dsm_loads = df2.columns[0:amount_dsm_loads]
    edisgo_obj.dsm.p_max = df2[dsm_loads]

    data = [
        np.fromstring(dsm_data.e_max_pu[i].strip("[]"), sep=",")[0 : len(timeindex)]
        * dsm_data.e_nom.iloc[i]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    edisgo_obj.dsm.e_max = df2[dsm_loads]

    data = [
        np.fromstring(dsm_data.p_min_pu[i].strip("[]"), sep=",")[0 : len(timeindex)]
        * dsm_data.p_nom.iloc[i]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    edisgo_obj.dsm.p_min = df2[dsm_loads]

    data = [
        np.fromstring(dsm_data.e_min_pu[i].strip("[]"), sep=",")[0 : len(timeindex)]
        * dsm_data.e_nom.iloc[i]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    edisgo_obj.dsm.e_min = df2[dsm_loads]

    data = [
        np.fromstring(dsm_data.p_set[i].strip("[]"), sep=",")[0 : len(timeindex)]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    edisgo_obj.timeseries._loads_active_power = pd.concat(
        [
            edisgo_obj.timeseries.loads_active_power,
            pd.DataFrame(
                df2[dsm_loads].values,
                index=edisgo_obj.timeseries.loads_active_power.index,
                columns=dsm_loads,
            ),
        ],
        axis=1,
    )

    dsm_df = pd.DataFrame(index=dsm_loads, columns=edisgo_obj.topology.loads_df.columns)
    dsm_df.bus = edisgo_obj.topology.loads_df.bus[0 : len(dsm_loads)].values
    dsm_df.type = "conventional_load"
    dsm_df.sector = "industrial"
    dsm_df.p_set = dsm_data.p_nom[0:amount_dsm_loads].values
    dsm_df.voltage_level = edisgo_obj.topology.loads_df.voltage_level[
        0 : len(dsm_loads)
    ].values
    edisgo_obj.topology.loads_df = pd.concat([edisgo_obj.topology.loads_df, dsm_df])

    edisgo_obj.set_time_series_reactive_power_control()
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
        edisgo_obj.config._data["worst_case_scale_factor"][factor] = 0

    edisgo_obj.set_time_series_worst_case_analysis()
    edisgo_obj.reinforce()

    edisgo_obj_copy.topology.lines_df = edisgo_obj.topology.lines_df
    edisgo_obj_copy.topology.transformers_df = edisgo_obj.topology.transformers_df
    edisgo_obj_copy.topology.transformers_hvmv_df = (
        edisgo_obj.topology.transformers_hvmv_df
    )

    if save_edisgo:
        edisgo_obj_copy.save(
            os.path.join(f"{directory}"),
            save_results=False,
            save_timeseries=True,
            save_electromobility=True,
            save_heatpump=True,
            save_dsm=True,
        )
