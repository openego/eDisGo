import multiprocessing as mp
import os

import pandas as pd
from shapely import wkt

from edisgo.edisgo import EDisGo, import_edisgo_from_files
from edisgo.flex_opt.charging_strategies import charging_strategy
from edisgo.network.timeseries import get_component_timeseries

data_dir = r"C:\Users\aheider\Documents\Grids"
server_dir = r"H:\Grids"
ding0_dir = r"H:\ding0_grids"
strategy = "reduced"


def adapt_rules_based_charging_strategies(grid_id):
    edisgo_obj = import_edisgo_from_files(
        data_dir + r"\{}\dumb".format(grid_id),
        import_timeseries=True,
        import_electromobility=True,
    )
    charging_demand_pre = (
        edisgo_obj.timeseries.charging_points_active_power.sum().sum() / 4 * 0.9
    )
    charging_strategy(edisgo_obj, strategy=strategy)
    # Sanity check
    charging_demand = (
        edisgo_obj.timeseries.charging_points_active_power.sum().sum() / 4 * 0.9
    )
    charging_demand_simbev = (
        edisgo_obj.electromobility.charging_processes_df.loc[
            (edisgo_obj.electromobility.charging_processes_df.park_start < (8760 * 4))
        ].chargingdemand.sum()
        / 1000
    )
    print(
        "Total charging demand of grid {} with charging strategy {}: {}, "
        "simbev demand: {}, original: {}".format(
            grid_id,
            strategy,
            charging_demand,
            charging_demand_simbev,
            charging_demand_pre,
        )
    )
    # Save resulting timeseries
    if strategy == "dumb":
        save_dir = data_dir + r"\{}\dumb\timeseries".format(grid_id)
    else:
        save_dir = data_dir + r"\{}\{}".format(grid_id, strategy)
    edisgo_obj.timeseries.charging_points_active_power.to_csv(
        save_dir + "/charging_points_active_power.csv"
    )


def save_adapted_extreme_weeks_to_server(grid_id, strategy):
    if strategy == "dumb":
        save_dir = data_dir + r"\{}\dumb\timeseries".format(grid_id)
        server_save_dir = server_dir + r"\{}\dumb\timeseries".format(grid_id)
    else:
        save_dir = data_dir + r"\{}\{}".format(grid_id, strategy)
        server_save_dir = server_dir + r"\{}\{}".format(grid_id, strategy)
    ts_charging = pd.read_csv(
        save_dir + "/charging_points_active_power.csv", index_col=0, parse_dates=True
    )
    idx_extreme_weeks = pd.read_csv(
        os.path.join(server_dir, str(grid_id), "timeindex_extreme_weeks.csv"),
        index_col=0,
        parse_dates=True,
    )
    ts_charging.loc[idx_extreme_weeks.index].to_csv(
        server_save_dir + "/charging_points_active_power.csv"
    )


def adapt_feeder(grid_id):
    server_save_dir = server_dir + r"\{}\dumb\timeseries".format(grid_id)
    ts_charging = pd.read_csv(
        server_save_dir + "/charging_points_active_power.csv",
        index_col=0,
        parse_dates=True,
    )
    feeders = os.listdir(server_dir + r"\{}\feeder".format(grid_id))
    for feeder in feeders:
        cps = pd.read_csv(
            server_dir
            + r"\{}\feeder\{}\topology\charging_points.csv".format(grid_id, feeder),
            index_col=0,
        )
        ts_charging_feeder = ts_charging[cps.index]
        ts_charging_feeder.to_csv(
            server_dir
            + r"\{}\feeder\{}\timeseries\charging_points_active_power.csv".format(
                grid_id, feeder
            )
        )


def adapt_loads_to_remove_hps(grid_id):
    edisgo_dir = data_dir + r"\{}\dumb".format(grid_id)
    edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True)
    # drop all components except for loads, replace load with original
    edisgo.topology._charging_points_df = edisgo.topology._charging_points_df.drop(
        edisgo.topology._charging_points_df.index
    )
    edisgo_ding0 = EDisGo(
        ding0_grid=ding0_dir + r"\{}".format(grid_id), import_timeseries=False
    )
    edisgo.topology._loads_df = edisgo_ding0.topology.loads_df
    edisgo.topology._generators_df = edisgo.topology.generators_df.loc[
        edisgo.topology.generators_df.index.isin(
            edisgo_ding0.topology.generators_df.index)
        & edisgo.topology.generators_df.type.isin(["wind", "solar"])]
    edisgo.topology._generators_df.p_nom = edisgo_ding0.topology.generators_df.loc[
        edisgo.topology._generators_df.index, "p_nom"
    ]
    get_component_timeseries(
        edisgo, timeseries_load="demandlib", timeseries_generation_fluctuating="oedb"
    )
    # reinforce to have stable grid
    #edisgo.reinforce()
    save_dir = r"H:\no_HP\{}".format(grid_id)
    edisgo.save(save_dir)


def add_generators_and_cps_no_hp(grid_id):
    edisgo_dir = r"H:\no_HP\{}".format(grid_id)
    edisgo = import_edisgo_from_files(edisgo_dir, import_timeseries=True)
    edisgo_orig_dir_topology = r"H:\Grids\{}\dumb".format(grid_id)
    edisgo_orig_topology = import_edisgo_from_files(edisgo_orig_dir_topology,
                                                    import_electromobility=True)
    edisgo.electromobility = edisgo_orig_topology.electromobility
    mapping = \
        pd.read_csv(server_dir + r"\{}\dumb\topology_orig\mapping.csv".format(grid_id),
                    index_col=0)
    edisgo_orig_dir_timeseries = r"C:\Users\aheider\Documents\Grids\{}\dumb".format(grid_id)
    edisgo_orig_timeseries = import_edisgo_from_files(edisgo_orig_dir_timeseries,
                                                      import_timeseries=True)
    for attr in ["generators_df", "charging_points_df"]:
        setattr(edisgo.topology, attr, getattr(edisgo_orig_topology.topology, attr))
    for attr in ["generators_active_power", "generators_reactive_power",
                 "charging_points_active_power", "charging_points_reactive_power"]:
        setattr(edisgo.timeseries, attr,
                getattr(edisgo_orig_timeseries.timeseries, attr))
    problematic_cps = edisgo.topology.charging_points_df.loc[
        ~edisgo.topology.charging_points_df.bus.isin(edisgo.topology.buses_df.index)]
    edisgo.timeseries.charging_points_active_power = \
        edisgo.timeseries.charging_points_active_power.rename(
            columns=mapping['0'].to_dict())
    edisgo.timeseries.charging_points_reactive_power = \
        edisgo.timeseries.charging_points_reactive_power.rename(
            columns=mapping['0'].to_dict())
    mapping_dict = reintegrate_charging(problematic_cps, edisgo,
                                        edisgo.timeseries.charging_points_active_power)
    pd.Series(mapping_dict).to_csv(
        edisgo_dir + r"\mapping.csv")
    edisgo.analyze(timesteps=edisgo.timeseries.timeindex[0])
    edisgo.save(edisgo_dir, archive=True, save_results=False)


def check_ev_timeseries_for_maximum_powers(grid_id):
    timeseries_dir = server_dir + r"\{}\dumb\timeseries".format(grid_id)
    ts_charging = pd.read_csv(timeseries_dir + "/charging_points_active_power.csv",
                              index_col=0, parse_dates=True)
    topology_dir = server_dir + r"\{}\dumb\topology".format(grid_id)
    cps = pd.read_csv(topology_dir + "/charging_points.csv", index_col=0)
    problematic_cps = cps.loc[cps.p_nom < ts_charging[cps.index].max()]
    print("The following charging points exceed their limits: {}".format(
        problematic_cps.index))
    return problematic_cps


def adapt_problematic_charging(grid_id):
    """
    Method to move charging points to the grid level they belong.
    Todo: apply to original grids
    """
    print("Adapting charging for grid {}".format(grid_id))
    edisgo = import_edisgo_from_files(server_dir + r"\{}\dumb".format(grid_id),
                                      import_timeseries=True, import_electromobility=True)
    # Save current topology
    edisgo.topology.to_csv(server_dir + r"\{}\dumb\topology_orig".format(grid_id))
    cps = edisgo.topology.charging_points_df
    ts_charging = edisgo.timeseries.charging_points_active_power
    problematic_cps = cps.loc[cps.p_nom < ts_charging[cps.index].max()]
    mapping_dict = reintegrate_charging(problematic_cps, edisgo, ts_charging)
    # save results
    pd.Series(mapping_dict).to_csv(server_dir + r"\{}\dumb\topology_orig\mapping.csv".format(grid_id))
    # check integrity of pypsa network
    pypsa_network = edisgo.to_pypsa()
    # rename elements in electromobility object
    edisgo.electromobility.integrated_charging_parks_df.replace(mapping_dict,inplace=True)
    # Todo: Handle potential charging parks
    edisgo.save(server_dir + r"\{}\dumb".format(grid_id))
    print("All charging points adapted.")


def reintegrate_charging(cps, edisgo, ts_charging, mode_p_nom="ts"):
    # remove cp from topology, timeseries and emobility
    comp_type = "ChargingPoint"
    changed_cps = {}
    mapping_dict = {}
    for cp_name in cps.index:
        changed_cps[cp_name] = {"topology": cps.loc[cp_name],
                                "active_power": ts_charging[cp_name],
                                "reactive_power": edisgo.timeseries.charging_points_reactive_power[cp_name]}
        edisgo.remove_component(comp_type, cp_name)
    # edisgo.analyze()
    for cp_name, cp in changed_cps.items():
        if mode_p_nom == "ts":
            p_nom = cp["active_power"].max()
        else:
            p_nom = cp["topology"].p_nom
        new_name = edisgo.integrate_component(comp_type=comp_type,
                                              geolocation=wkt.loads(cp["topology"].geom),
                                              use_case=cp["topology"].use_case,
                                              add_ts=True,
                                              ts_active_power=cp["active_power"],
                                              ts_reactive_power=cp["reactive_power"],
                                              p_nom=p_nom)
        changed_cps[cp_name]["edisgo_id"] = new_name
        mapping_dict[cp_name] = new_name
    return mapping_dict


if __name__ == "__main__":
    # adapt_rules_based_charging_strategies(176)
    # save_adapted_extreme_weeks_to_server(176)
    grid_ids = [176]
    # pool = mp.Pool(6)
    # pool = mp.Pool(min(len(grid_ids), int(mp.cpu_count() / 2)))
    # pool.map_async(add_generators_and_cps_no_hp, grid_ids).get()
    # pool.close()
    for grid_id in grid_ids:
        add_generators_and_cps_no_hp(grid_id)
    # adapt_feeder(grid_id)
    # for strategy in ["dumb", "reduced", "residual"]:
    #     save_adapted_extreme_weeks_to_server(grid_id, strategy)
    print("SUCCESS")
