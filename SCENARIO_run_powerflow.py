import os
import logging
import multiprocessing
import traceback
import pandas as pd
import numpy as np

import results_helper_functions
from edisgo.edisgo import import_edisgo_from_files
from edisgo.io import pypsa_io

# suppress infos from pypsa
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# ###### USER SETTINGS #######
num_threads = 6#int(multiprocessing.cpu_count()/2)

scenarios = ["optimised","no_ev", "dumb", "reduced", "residual"]#

results_base_path = r"H:\Grids"

mv_grid_ids = [176, 177, 1056, 1690, 1811, 2534]#
feeders = [6]
single_feeder = False
number_time_intervals = 4
pypsa_mode = None


# ###### END USER SETTINGS #######


def combine_results_for_grid(feeders, grid_id, res_dir, res_name):
    res_grid = pd.DataFrame()
    for feeder_id in feeders:
        res_feeder = pd.DataFrame()
        for i in range(14):
            try:
                res_feeder_tmp = pd.read_csv(res_dir + '/{}/{}/{}_{}_{}_{}.csv'.format(
                    grid_id, feeder_id, res_name, grid_id, feeder_id, i),
                                                     index_col=0, parse_dates=True)
                res_feeder = pd.concat([res_feeder, res_feeder_tmp], sort=False)
            except:
                print('Results for feeder {} in grid {} could not be loaded.'.format(feeder_id, grid_id))
        try:
            res_grid = pd.concat([res_grid, res_feeder], axis=1, sort=False)
        except:
            print('Feeder {} not added'.format(feeder_id))
    res_grid = res_grid.loc[~res_grid.index.duplicated(keep='last')]
    return res_grid


def run_power_flow(mv_grid_id):
    for scenario in scenarios:

        # try:
        elia_logger = logging.getLogger('elia_project: {}, {}'.format(
            mv_grid_id, scenario))
        elia_logger.setLevel(logging.INFO)

        grid_dir = os.path.join(
            results_base_path, str(mv_grid_id), scenario)
        # if os.path.exists(os.path.join(grid_dir, 'time_series_sums.csv')):
        #     print('{} {} already solved.'.format(mv_grid_id, scenario))
        #     continue

        # reimport edisgo object
        if not single_feeder:
            grid_dir = os.path.join(
                results_base_path, str(mv_grid_id), scenario)
            edisgo_orig_dir = results_base_path+r'\{}\dumb'.format(mv_grid_id)
        else:
            edisgo_orig_dir = results_base_path+r'\{}\feeder\{}'.format(mv_grid_id,
                                                                                                             feeders[0])
        # if os.path.exists(os.path.join(grid_dir, 'time_series_sums.csv')):
        #     print('{} {} already solved.'.format(mv_grid_id, scenario))
        #     continue

        # reimport edisgo object

        res_dir = r'results\{}'.format(scenario)
        edisgo = import_edisgo_object_with_adapted_charging_timeseries(edisgo_orig_dir, grid_dir,
                                                                       mv_grid_id, scenario, res_dir)

        # run powerflow and get grid issues
        elia_logger.info(
            'Running power flow analysis for whole year.')
        split_length = int(np.ceil(
            len(edisgo.timeseries.timeindex) / number_time_intervals))
        time_intervals = [edisgo.timeseries.timeindex[
                          i * split_length:(i + 1) * split_length]
                          for i in range(0, number_time_intervals, 1)]

        timesteps_not_converged = pd.DatetimeIndex([])
        relative_load = pd.DataFrame()
        voltage_deviation = pd.DataFrame()
        for time_interval in time_intervals:
            pypsa_network = edisgo.to_pypsa(
                mode=pypsa_mode,
                timesteps=time_interval)
            pf_results = pypsa_network.pf(
                time_interval,
                use_seed=False)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]].index
            pypsa_io.process_pfa_results(
                edisgo, pypsa_network, timesteps_converged)

            relative_load_in_time_interval = \
                results_helper_functions.relative_load(edisgo)
            relative_load = pd.concat(
                [relative_load, relative_load_in_time_interval],
                sort=True)
            voltage_deviation_in_time_interval = \
                results_helper_functions.voltage_diff(edisgo)
            voltage_deviation = pd.concat(
                [voltage_deviation, voltage_deviation_in_time_interval],
                sort=True)

            timesteps_not_converged_in_time_interval = \
                pf_results["converged"][
                    ~pf_results["converged"]["0"]].index
            if len(timesteps_not_converged_in_time_interval) > 0:
                timesteps_not_converged = timesteps_not_converged.append(
                    timesteps_not_converged_in_time_interval
                )

        elia_logger.info('Power flow analysis finished.')

        # save results
        relative_load = relative_load.apply(
            lambda _: _.astype("float32")
        )
        try:
            relative_load.to_csv(
                os.path.join(res_dir, "relative_load.csv"))
        except:
            relative_load.to_csv(
                os.path.join(grid_dir, "relative_load.csv"))

        voltage_deviation = voltage_deviation.apply(
            lambda _: _.astype("float32")
        )
        try:
            voltage_deviation.to_csv(
                os.path.join(res_dir, "voltage_deviation.csv"))
        except:
            voltage_deviation.to_csv(
                os.path.join(grid_dir, "voltage_deviation.csv"))

        # save residual load
        ts_df = pd.DataFrame()
        ts_df["gens"] = edisgo.timeseries.generators_active_power.sum(
            axis=1)
        ts_df["loads"] = edisgo.timeseries.loads_active_power.sum(axis=1)
        ts_df[
            "charging_points"] = edisgo.timeseries.charging_points_active_power.sum(
            axis=1)
        ts_df["residual_load"] = edisgo.timeseries.residual_load
        try:
            ts_df.to_csv(os.path.join(res_dir, "time_series_sums.csv"))
        except:
            ts_df.to_csv(os.path.join(grid_dir, "time_series_sums.csv"))

        if len(timesteps_not_converged) > 0:
            elia_logger.info(
                'Not all time steps converged.')
            try:
                pd.Series(timesteps_not_converged).to_csv(
                    os.path.join(res_dir, "timesteps_not_converged.csv")
                )
            except:
                pd.Series(timesteps_not_converged).to_csv(
                    os.path.join(grid_dir, "timesteps_not_converged.csv")
                )
        else:
            elia_logger.info(
                'All time steps converged.')

            # edisgo.results.pfa_v_mag_pu_seed = \
            #     edisgo.results.pfa_v_mag_pu_seed.apply(
            #         lambda _: _.astype("float32")
            #     )
            # edisgo.results.pfa_v_ang_seed = edisgo.results.pfa_v_ang_seed.apply(
            #     lambda _: _.astype("float32")
            # )
            # edisgo.save(grid_dir,
            #             save_timeseries=False,
            #             save_topology=False,
            #             parameters={"powerflow_results": [
            #                 "pfa_v_mag_pu_seed", "pfa_v_ang_seed"]},
            #             save_seed=True)
        # except:
        #     traceback.print_exc()


def import_edisgo_object_with_adapted_charging_timeseries(edisgo_orig_dir, grid_dir, mv_grid_id, scenario,
                                                          res_dir=None):
    edisgo = import_edisgo_from_files(edisgo_orig_dir, import_timeseries=True)
    if scenario != 'dumb':
        if scenario == 'no_ev':
            charging_ts = pd.DataFrame(index=edisgo.timeseries.charging_points_active_power.index,
                                       columns=edisgo.timeseries.charging_points_active_power.columns,
                                       data=0.0)
        elif scenario in ["reduced", "residual", "optimised"]:
            charging_ts = pd.read_csv(grid_dir + r'\charging_points_active_power.csv', index_col=0,
                                      parse_dates=True)
        else:
            edisgo_dir = results_base_path + r'\{}\feeder'.format(mv_grid_id)
            if len(feeders) == 0:
                for feeder in os.listdir(edisgo_dir):
                    feeders.append(feeder)
            x_charge_ev_grid = combine_results_for_grid(feeders, mv_grid_id, res_dir, 'x_charge_ev')
            edisgo.timeseries._charging_points_active_power.update(
                x_charge_ev_grid.loc[edisgo.timeseries.charging_points_active_power.index])
            charging_ts = edisgo.timeseries.charging_points_active_power
        edisgo.timeseries.charging_points_active_power = charging_ts
    return edisgo


if __name__ == "__main__":
    if num_threads == 1:
        for mv_grid_id in mv_grid_ids:
            run_power_flow(mv_grid_id)
    else:
        with multiprocessing.Pool(num_threads) as pool:
            pool.map(run_power_flow, mv_grid_ids)
