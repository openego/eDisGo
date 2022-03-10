import os
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.charging_strategies import charging_strategy
import multiprocessing as mp
import pandas as pd

data_dir = r"C:\Users\aheider\Documents\Grids"
server_dir = r"H:\Grids"
strategy = "reduced"


def adapt_rules_based_charging_strategies(grid_id):
    edisgo_obj = import_edisgo_from_files(data_dir + r"\{}\dumb".format(grid_id),
                                          import_timeseries=True,
                                          import_electromobility=True)
    charging_demand_pre = edisgo_obj.timeseries.charging_points_active_power.sum().sum() / 4 * 0.9
    charging_strategy(edisgo_obj, strategy=strategy)
    # Sanity check
    charging_demand = edisgo_obj.timeseries.charging_points_active_power.sum().sum()/4*0.9
    charging_demand_simbev = edisgo_obj.electromobility.charging_processes_df.loc[
        (edisgo_obj.electromobility.charging_processes_df.park_start < (8760 * 4))].chargingdemand.sum()/1000
    print("Total charging demand of grid {} with charging strategy {}: {}, "
          "simbev demand: {}, original: {}".format(grid_id, strategy, charging_demand,
                                                   charging_demand_simbev, charging_demand_pre))
    # Save resulting timeseries
    if strategy == "dumb":
        save_dir = data_dir + r"\{}\dumb\timeseries".format(grid_id)
    else:
        save_dir = data_dir + r"\{}\{}".format(grid_id, strategy)
    edisgo_obj.timeseries.charging_points_active_power.to_csv(
        save_dir + "/charging_points_active_power.csv")


def save_adapted_extreme_weeks_to_server(grid_id, strategy):
    if strategy == "dumb":
        save_dir = data_dir + r"\{}\dumb\timeseries".format(grid_id)
        server_save_dir = server_dir + r"\{}\dumb\timeseries".format(grid_id)
    else:
        save_dir = data_dir + r"\{}\{}".format(grid_id, strategy)
        server_save_dir = server_dir + r"\{}\{}".format(grid_id, strategy)
    ts_charging = pd.read_csv(save_dir + "/charging_points_active_power.csv", index_col=0,
                              parse_dates=True)
    idx_extreme_weeks = pd.read_csv(os.path.join(server_dir, str(grid_id), "timeindex_extreme_weeks.csv"), index_col=0,
                     parse_dates=True)
    ts_charging.loc[idx_extreme_weeks.index].to_csv(server_save_dir + "/charging_points_active_power.csv")


def adapt_feeder(grid_id):
    server_save_dir = server_dir + r"\{}\dumb\timeseries".format(grid_id)
    ts_charging = pd.read_csv(server_save_dir + "/charging_points_active_power.csv", index_col=0,
                              parse_dates=True)
    feeders = os.listdir(server_dir + r"\{}\feeder".format(grid_id))
    for feeder in feeders:
        cps = pd.read_csv(server_dir + r"\{}\feeder\{}\topology\charging_points.csv".format(
            grid_id, feeder), index_col=0)
        ts_charging_feeder = ts_charging[cps.index]
        ts_charging_feeder.to_csv(server_dir + r"\{}\feeder\{}\timeseries\charging_points_active_power.csv".format(
            grid_id, feeder))


if __name__ == '__main__':
    # adapt_rules_based_charging_strategies(176)
    # save_adapted_extreme_weeks_to_server(176)
    grid_ids = [176, 177, 1056, 1811, 1690, 2534]
    # pool = mp.Pool(min(len(grid_ids), int(mp.cpu_count() / 2)))
    # pool.map_async(adapt_rules_based_charging_strategies, grid_ids).get()
    # pool.close()
    for grid_id in grid_ids:
        adapt_feeder(grid_id)
        # for strategy in ["dumb", "reduced", "residual"]:
        #     save_adapted_extreme_weeks_to_server(grid_id, strategy)
    print("SUCCESS")