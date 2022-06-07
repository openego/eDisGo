import os
from SCENARIO_run_powerflow import import_edisgo_object_with_adapted_charging_timeseries

import pandas as pd
from itertools import product
import multiprocessing as mp
import traceback

import results_helper_functions
from SCENARIO_run_curtailment import solve_convergence_issues_reinforce, assign_feeder, get_path_length_to_station


results_base_path = r"H:\Grids"


def _overwrite_edisgo_timeseries(edisgo, curt_gens, curt_reactive_gens,
                                 curt_loads, curt_reactive_loads):
    """
    Overwrites generator and load time series in edisgo after reinforcement.

    ts equals the curtailed energy that has to be added back to edisgo ts

    """

    # overwrite time series in edisgo
    time_steps = curt_gens.index

    # generators: overwrite time series for all except slack
    gens = curt_gens.columns
    edisgo.timeseries._generators_active_power.loc[
        time_steps, gens] = edisgo.timeseries._generators_active_power.loc[
        time_steps, gens] + curt_gens.loc[time_steps, gens]
    edisgo.timeseries._generators_reactive_power.loc[
        time_steps, gens] = edisgo.timeseries._generators_reactive_power.loc[
        time_steps, gens] + curt_reactive_gens.loc[time_steps, gens]

    # loads: distinguish between charging points and conventional loads
    loads = [_ for _ in curt_loads.columns if "ChargingPoint" not in _]
    edisgo.timeseries._loads_active_power.loc[
        time_steps, loads] = edisgo.timeseries._loads_active_power.loc[
        time_steps, loads] + curt_loads.loc[time_steps, loads]
    edisgo.timeseries._loads_reactive_power.loc[
        time_steps, loads] = edisgo.timeseries._loads_reactive_power.loc[
        time_steps, loads] + curt_reactive_loads.loc[time_steps, loads]

    if not edisgo.topology.charging_points_df.empty:
        charging_points = [_ for _ in curt_loads.columns
                           if "ChargingPoint" in _]
        edisgo.timeseries._charging_points_active_power.loc[
            time_steps, charging_points] = edisgo.timeseries._charging_points_active_power.loc[
            time_steps, charging_points] + curt_loads.loc[time_steps, charging_points]
        edisgo.timeseries._charging_points_reactive_power.loc[
            time_steps, charging_points] = edisgo.timeseries._charging_points_reactive_power.loc[
            time_steps, charging_points] + curt_reactive_loads.loc[time_steps, charging_points]

import logging
logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)


def run_reinforcement(variation):
    try:
        grid_id = variation[0]
        strategy = variation[1]
        # reimport edisgo object
        edisgo_dir = os.path.join(
            results_base_path, str(grid_id), strategy)
        edisgo_orig_dir = results_base_path + r'\{}\dumb'.format(grid_id)
        edisgo = import_edisgo_object_with_adapted_charging_timeseries(
            edisgo_orig_dir, edisgo_dir, grid_id, strategy)

        # assign feeders and path length to station
        assign_feeder(edisgo, mode="mv_feeder")
        assign_feeder(edisgo, mode="lv_feeder")
        get_path_length_to_station(edisgo)

        # get time steps with convergence problems
        src = os.path.join(edisgo_dir, "timesteps_not_converged.csv")
        if os.path.isfile(src):
            timesteps_not_converged = pd.read_csv(
                src, index_col=1, parse_dates=True).index
        else:
            timesteps_not_converged = []
        # import voltage deviation and relative load
        path = os.path.join(edisgo_dir, 'voltage_deviation.csv')
        voltage_dev = pd.read_csv(path, index_col=0, parse_dates=True)
        path = os.path.join(edisgo_dir, 'relative_load.csv')
        rel_load = pd.read_csv(path, index_col=0, parse_dates=True)

        # get time steps with issues and reduce voltage_dev and rel_load
        # to those time steps
        voltage_issues = voltage_dev[
            voltage_dev != 0.].dropna(how="all").dropna(
            axis=1, how="all")
        overloading_issues = rel_load[rel_load > 1.].dropna(
            how="all").dropna(axis=1, how="all")

        os.makedirs(edisgo_dir+r'\results_before_reinforcement', exist_ok=True)
        overloading_issues.to_csv(edisgo_dir+r'\results_before_reinforcement\overloading.csv')
        voltage_issues.to_csv(edisgo_dir+r'\results_before_reinforcement\voltage_issues.csv')


        curtailment = pd.DataFrame(
            data=0.,
            columns=["feed-in", "load"],
            index=["lv_problems", "mv_problems",
                   "convergence_issues_mv", "convergence_issues_lv"])

        # handle not converged time steps
        i = 0
        while len(timesteps_not_converged) > 0:
            print('Starting curtailment due to convergence problems. Iteration {}. '
                  'Timesteps not converged: {}.'.format(i, timesteps_not_converged))
            if i==0:
                pypsa_network = edisgo.to_pypsa()
                curtailed_feedin, curtailed_feedin_reactive, \
                curtailed_load, curtailed_load_reactive, curtailment, voltage_dev, rel_load = \
                    solve_convergence_issues_reinforce(
                        edisgo, pypsa_network, timesteps_not_converged,
                        curtailment, voltage_dev, rel_load)
            else:
                curtailed_feedin_2, curtailed_feedin_reactive_2, \
                curtailed_load_2, curtailed_load_reactive_2, curtailment, voltage_dev, rel_load = \
                    solve_convergence_issues_reinforce(
                        edisgo, pypsa_network, timesteps_not_converged,
                        curtailment, voltage_dev, rel_load)
                curtailed_load = curtailed_load + curtailed_load_2
                curtailed_load_reactive = curtailed_load_reactive + curtailed_load_reactive_2
                curtailed_feedin = curtailed_feedin + curtailed_feedin_2
                curtailed_feedin_reactive = curtailed_feedin_reactive + curtailed_feedin_reactive_2
            pypsa_network =edisgo.to_pypsa(mode=None)

            # run power flow analysis
            pypsa_network.lpf(edisgo.timeseries.timeindex)
            pf_results = pypsa_network.pf(edisgo.timeseries.timeindex, use_seed=True)
            timesteps_not_converged = edisgo.timeseries.timeindex[~pf_results["converged"]["0"]].tolist()
            i += 1


        print('Starting reinforcement {} {}'.format(grid_id, strategy))
        edisgo.reinforce(combined_analysis=True)

        if len(timesteps_not_converged) > 0:
            _overwrite_edisgo_timeseries(edisgo, curt_gens=curtailed_feedin,
                                         curt_loads=curtailed_load,
                                         curt_reactive_loads=curtailed_load_reactive,
                                         curt_reactive_gens=curtailed_feedin_reactive)
            edisgo.reinforce(combined_analysis=True)

        edisgo.results.to_csv(edisgo_dir+r'\results_after_reinforcement',
                            parameters={'grid_expansion_results':None})
        rel_load = results_helper_functions.relative_load(edisgo)
        voltage_dev = results_helper_functions.voltage_diff(edisgo)
        # get time steps with issues and reduce voltage_dev and rel_load
        # to those time steps
        voltage_issues = voltage_dev[
            voltage_dev != 0.].dropna(how="all").dropna(
            axis=1, how="all")
        overloading_issues = rel_load[rel_load > 1.].dropna(
            how="all").dropna(axis=1, how="all")

        if not overloading_issues.empty:
            print('Not all overloading issues solves for {}'.format(grid_id))
            overloading_issues.to_csv(edisgo_dir + r'\results_after_reinforcement\overloading.csv')
        if not voltage_issues.empty:
            print('Not all voltage issues solves for {}'.format(grid_id))
            voltage_issues.to_csv(edisgo_dir+r'\results_after_reinforcement\voltage_issues.csv')

        edisgo.topology.to_csv(edisgo_dir+r'\topology_after_reinforcement')
        print('Finished reinforcement evaluation of {} {}'.format(grid_id, strategy))
    except:
        print('Something went wrong with grid {} {}'.format(variation[0], variation[1]))
        traceback.print_exc()


if __name__ == '__main__':
    pool = mp.Pool(5) # mp.Pool(int(mp.cpu_count()/2))
    grid_ids = [1690] # 176, 177, 1056, 1690, 1811, 2534
    strategies = ['no_ev', 'dumb', 'reduced', 'residual', 'optimised']
    variations = list(product(grid_ids, strategies))
    results = pool.map(run_reinforcement, variations)
    pool.close()
    pool.join()
    print('SUCCESS')

# grid_ids = [176]
# strategies = ['reduced']
# variations = list(product(grid_ids, strategies))
# for variation in variations:
#     run_reinforcement(variation)