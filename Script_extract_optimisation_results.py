import os
import multiprocessing as mp
import pandas as pd

from edisgo.opf.lopf import combine_results_for_grid

grid_base_dir = r"H:\Grids"
res_dir = "results/minimize_loading_SEST"
cpu_count = 6


def extract_optimised_charging_timeseries(grid_id):
    print("Extracting optimised charging for {}".format(grid_id))
    edisgo_dir = grid_base_dir+r'\{}\feeder'.format(grid_id)

    feeders=[]
    for feeder in os.listdir(edisgo_dir):
        feeders.append(feeder)
    edisgo_dir = grid_base_dir + r'\{}\reduced'.format(grid_id)
    edisgo_dir_new = grid_base_dir + r'\{}\optimised'.format(grid_id)
    reference_charging_ts = pd.read_csv(edisgo_dir+r"\charging_points_active_power.csv",
                                        index_col=0, parse_dates=True)
    x_charge_ev_grid = combine_results_for_grid(feeders, grid_id, res_dir, 'x_charge_ev')
    reference_charging_ts.update(x_charge_ev_grid)
    os.makedirs(edisgo_dir_new, exist_ok=True)
    reference_charging_ts.loc[reference_charging_ts.index, reference_charging_ts.columns].to_csv(
        edisgo_dir_new+r"\charging_points_active_power.csv")


if __name__ == '__main__':
    # extract_optimised_charging_timeseries(176)
    grid_ids = [176, 177, 1056, 1690, 1811, 2534]
    pool = mp.Pool(cpu_count)
    pool.map_async(extract_optimised_charging_timeseries, grid_ids).get()
    pool.close()
    print("SUCCESS")