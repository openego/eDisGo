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
    lv_loads = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.voltage_level == "lv"
    ]
    # amount of dsm loads
    amount_retail_dsm_loads = int(
        np.ceil(0.25 * len(lv_loads.loc[lv_loads.sector == "retail"]))
    )
    amount_industrial_dsm_loads = int(
        np.ceil(0.25 * len(lv_loads.loc[lv_loads.sector == "industrial"]))
    )
    amount_mv_dsm_loads = int(
        np.ceil(0.1 * (amount_retail_dsm_loads + amount_industrial_dsm_loads))
    )
    # name of lv loads to be replaced by dsm loads
    retail_dsm_loads = lv_loads.loc[lv_loads.sector == "retail"][
        0:amount_retail_dsm_loads
    ].index.values
    industrial_dsm_loads = lv_loads.loc[lv_loads.sector == "industrial"][
        0:amount_industrial_dsm_loads
    ].index.values

    dsm_data["name"] = [str("dsm_load_" + str(i)) for i in range(len(dsm_data))]
    # p_max
    data = [
        np.fromstring(dsm_data.p_max_pu[i].strip("[]"), sep=",")[0 : len(timeindex)]
        * dsm_data.p_nom.iloc[i]
        for i in range(len(dsm_data))
    ]
    dsm_data["peak_load"] = [max(p) for p in data]
    lv_data = dsm_data.loc[dsm_data.peak_load < 0.3]
    mv_data = dsm_data.loc[dsm_data.peak_load >= 0.3]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    # name of dsm loads
    lv_dsm_loads_retail = lv_data.name[0:amount_retail_dsm_loads].values
    lv_dsm_loads_industrial = lv_data.name[
        amount_retail_dsm_loads : amount_retail_dsm_loads + amount_industrial_dsm_loads
    ].values
    mv_dsm_loads = mv_data.name[0:amount_mv_dsm_loads].values
    if amount_mv_dsm_loads > len(mv_data):
        mv_dsm_loads2 = lv_data.name[len(mv_data) - amount_mv_dsm_loads - 1 : -1].values
        mv_dsm_loads = np.concatenate([mv_dsm_loads, mv_dsm_loads2])
    edisgo_obj.dsm.p_max = df2[
        np.concatenate([lv_dsm_loads_retail, lv_dsm_loads_industrial, mv_dsm_loads])
    ]
    # e_max
    data = [
        np.fromstring(dsm_data.e_max_pu[i].strip("[]"), sep=",")[0 : len(timeindex)]
        * dsm_data.e_nom.iloc[i]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    edisgo_obj.dsm.e_max = df2[
        np.concatenate([lv_dsm_loads_retail, lv_dsm_loads_industrial, mv_dsm_loads])
    ]
    # p_min
    data = [
        np.fromstring(dsm_data.p_min_pu[i].strip("[]"), sep=",")[0 : len(timeindex)]
        * dsm_data.p_nom.iloc[i]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    edisgo_obj.dsm.p_min = df2[
        np.concatenate([lv_dsm_loads_retail, lv_dsm_loads_industrial, mv_dsm_loads])
    ]
    # e_min
    data = [
        np.fromstring(dsm_data.e_min_pu[i].strip("[]"), sep=",")[0 : len(timeindex)]
        * dsm_data.e_nom.iloc[i]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
    edisgo_obj.dsm.e_min = df2[
        np.concatenate([lv_dsm_loads_retail, lv_dsm_loads_industrial, mv_dsm_loads])
    ]
    # timeseries p
    data = [
        np.fromstring(dsm_data.p_set[i].strip("[]"), sep=",")[0 : len(timeindex)]
        for i in range(len(dsm_data))
    ]
    df2 = pd.DataFrame(columns=timeindex, index=dsm_data["name"], data=data).transpose()
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
                df2[
                    np.concatenate(
                        [lv_dsm_loads_retail, lv_dsm_loads_industrial, mv_dsm_loads]
                    )
                ].values,
                index=edisgo_obj.timeseries.loads_active_power.index,
                columns=np.concatenate(
                    [lv_dsm_loads_retail, lv_dsm_loads_industrial, mv_dsm_loads]
                ),
            ),
        ],
        axis=1,
    )

    for vl in ["mv", "lv"]:
        if vl == "mv":
            dsm_df = pd.DataFrame(
                index=mv_dsm_loads, columns=edisgo_obj.topology.loads_df.columns
            )
            dsm_df.bus = edisgo_obj.topology.buses_df.loc[
                edisgo_obj.topology.buses_df.v_nom > 0.4
            ].index.values[0:amount_mv_dsm_loads]
            dsm_df.p_set = dsm_data.loc[dsm_data.name.isin(mv_dsm_loads)].p_nom.values
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
    # drop lv loads that have been replaced by dsm loads
    edisgo_obj.topology.loads_df = edisgo_obj.topology.loads_df.drop(
        np.concatenate([retail_dsm_loads, industrial_dsm_loads])
    )
    # set reactive power timeseries
    edisgo_obj.set_time_series_reactive_power_control()
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