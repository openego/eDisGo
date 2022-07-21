import multiprocessing as mp
import os
import datetime
from time import perf_counter


from edisgo.edisgo import import_edisgo_from_files
from edisgo.opf.lopf import setup_model, optimize, prepare_time_invariant_parameters, \
    update_model, import_flexibility_bands, BANDS
from edisgo.tools.tools import convert_impedances_to_mv
import pandas as pd
import numpy as np

optimize_storage = False
optimize_ev = False
optimize_hp = True


def run_optimized_charging_feeder_parallel(grid_feeder_tuple, run='_hp_test',
                                           load_results=False, iteration=0):
    objective = 'minimize_loading'
    timesteps_per_iteration = 24*4
    iterations_per_era = 7
    overlap_iterations = 24
    solver = 'gurobi'
    kwargs = {}#{'v_min':0.91, 'v_max':1.09, 'thermal_limit':0.9}
    config_dict={'objective':objective, 'solver': solver,
                 'timesteps_per_iteration': timesteps_per_iteration,
                 'iterations_per_era': iterations_per_era,
                 'overlap_iterations': overlap_iterations}
    config_dict.update(kwargs)
    config = pd.Series(config_dict)

    grid_id = grid_feeder_tuple[0]
    feeder_id = grid_feeder_tuple[1]
    root_dir = r'H:\Grids'
    grid_dir = root_dir + r'\{}'.format(grid_id)
    feeder_dir = root_dir + r'\{}\feeder\{}'.format(grid_id, feeder_id)
    result_dir = 'results/{}/{}/{}'.format(objective+run, grid_id, feeder_id)

    os.makedirs(result_dir, exist_ok=True)

    if (len(os.listdir(result_dir)) > 239) and load_results:
        print('Feeder {} of grid {} already solved.'.format(feeder_id, grid_id))
        return
    elif (len(os.listdir(result_dir))>1) and load_results:
        iterations_finished = int((len(os.listdir(result_dir))-1)/17)
        charging_start, energy_level_start, start_iter = load_values_from_previous_failed_run(feeder_id, grid_id,
                                                                                              iteration,
                                                                                              iterations_per_era,
                                                                                              overlap_iterations,
                                                                                              result_dir)

    else:
        charging_start = None
        energy_level_start = None
        start_iter = 0
    config.to_csv(result_dir+'/config.csv')

    # try:
    edisgo_orig = import_edisgo_from_files(feeder_dir, import_timeseries=True)

    print('eDisGo object imported.')

    edisgo_obj = convert_impedances_to_mv(edisgo_orig)

    downstream_nodes_matrix = pd.read_csv(os.path.join(
        feeder_dir, 'downstream_node_matrix_{}_{}.csv'.format(grid_id, feeder_id)),
        index_col=0)

    print('Converted impedances to mv.')

    downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
    downstream_nodes_matrix = downstream_nodes_matrix.loc[
        edisgo_obj.topology.buses_df.index,
        edisgo_obj.topology.buses_df.index]
    print('Downstream node matrix imported.')

    # add Hps
    timeindex = pd.date_range("2011-01-01", periods=8760, freq="h")
    cop = pd.read_csv("examples/minimum_working/COP_2011.csv").set_index(timeindex).resample("15min").ffill()
    heat_demand = (
        pd.read_csv("examples/minimum_working/hp_heat_2011.csv", index_col=0)
            .set_index(timeindex)
    ).resample("15min").ffill()

    residential_loads = edisgo_obj.topology.loads_df.loc[
        edisgo_obj.topology.loads_df.sector == "residential"
    ]
    hp_names = []
    buses = []
    heat_demands = pd.DataFrame()
    for load, values in residential_loads.iterrows():
        hp_names.append(f"HP_{load}")
        buses.append(values.bus)
        heat_demands[f"HP_{load}"] = heat_demand["0"]

    heat_pump_df = pd.DataFrame(
        index=hp_names,
        columns=["bus", "p_nom", "capacity_tes"],
        data={"bus": buses, "p_nom": 0.003, "capacity_tes": 0.05},
    )
    edisgo_obj.topology.heat_pumps_df = heat_pump_df

    # Create dict with time invariant parameters
    parameters = prepare_time_invariant_parameters(edisgo_obj, downstream_nodes_matrix, pu=False,
                                                   optimize_storage=False,
                                                   optimize_ev_charging=False,
                                                   optimize_hp=True)
    print('Time-invariant parameters extracted.')

    energy_level = {}
    charging_hp = {}
    charging_tes = {}

    for iteration in range(start_iter, int(len(
               edisgo_obj.timeseries.timeindex) / timesteps_per_iteration)):  # edisgo_obj.timeseries.timeindex.week.unique()

        print('Starting optimisation for iteration {}.'.format(iteration))
        if iteration % iterations_per_era != iterations_per_era - 1:
            timesteps = edisgo_obj.timeseries.timeindex[
                        iteration * timesteps_per_iteration:(iteration + 1) *
                                                            timesteps_per_iteration + overlap_iterations]
            energy_level_end = None
        else:
            timesteps = edisgo_obj.timeseries.timeindex[
                        iteration * timesteps_per_iteration:(iteration + 1) * timesteps_per_iteration]
            energy_level_end = True
        try:
            model = update_model(model, timesteps, parameters, optimize_storage=optimize_storage,
                                 optimize_ev=optimize_ev, optimize_hp=optimize_hp,
                                 cop=cop, heat_demand=heat_demands,
                                 charging_start_hp=charging_start, energy_level_start_tes=energy_level_start,
                                 energy_level_end_tes=energy_level_end,  **kwargs)
        except NameError:
            model = setup_model(parameters, timesteps, objective=objective,
                                optimize_storage=optimize_storage, optimize_ev_charging=optimize_ev,
                                cop=cop, heat_demand=heat_demands,
                                charging_start_hp=charging_start, energy_level_start_tes=energy_level_start,
                                energy_level_end_tes=energy_level_end, **kwargs)

        print('Set up model for week {}.'.format(iteration))

        result_dict = optimize(model, solver)
        charging_hp[iteration] = result_dict['charging_hp_el']
        charging_tes[iteration] = result_dict['charging_tes']
        energy_level[iteration] = result_dict['energy_tes']

        if iteration % iterations_per_era != iterations_per_era - 1:
            charging_start = {"hp": charging_hp[iteration].iloc[-overlap_iterations],
                              "tes": charging_tes[iteration].iloc[-overlap_iterations]}
            energy_level_start = energy_level[iteration].iloc[-overlap_iterations]
        else:
            charging_start = None
            energy_level_start = None

        print('Finished optimisation for week {}.'.format(iteration))
        for res_name, res in result_dict.items():
            try:
                res = res.loc[edisgo_obj.timeseries.timeindex]
            except:
                pass
            if 'slack' in res_name:
                res = res[res>1e-6]
                res = res.dropna(how='all')
                res = res.dropna(how='all')
            if not res.empty:
                res.astype(np.float16).to_csv(result_dir + '/{}_{}_{}_{}.csv'.format(
                    res_name, grid_id, feeder_id, iteration))
        print('Saved results for week {}.'.format(iteration))

    # except Exception as e:
    #     print('Something went wrong with feeder {} of grid {}'.format(feeder_id, grid_id))
    #     print(e)
    #     if 'iteration' in locals():
    #         if iteration >= 1:
    #             charging_start = charging_hp[iteration-1].iloc[-overlap_iterations]
    #             charging_start.to_csv(result_dir +'/charging_start_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))
    #             energy_level_start = energy_level[iteration-1].iloc[-overlap_iterations]
    #             energy_level_start.to_csv(result_dir +'/energy_level_start_{}_{}_{}.csv'.format(grid_id, feeder_id, iteration))


if __name__ == '__main__':

    # t1 = perf_counter()
    # grid_feeder_tuple = (176, 1)
    # run_optimized_charging_feeder_parallel(grid_feeder_tuple, load_results=False)
    # print('It took {} seconds to run the full optimisation.'.format(perf_counter()-t1))
    # print('SUCCESS')
    t1 = perf_counter()

    # grid_ids = [2534]
    root_dir = r'H:\Grids'
    grid_id_feeder_tuples = [(2534,0)]#[(2534,0), (2534,1), (2534,6)](176, 6)
    # run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    # for grid_id in grid_ids:
    #     edisgo_dir = root_dir + r'\{}\feeder'.format(grid_id)
    #     for feeder in os.listdir(edisgo_dir):
    #         grid_id_feeder_tuples.append((grid_id, feeder))
    for grid_id_feeder_tuple in grid_id_feeder_tuples:
        run_optimized_charging_feeder_parallel(grid_id_feeder_tuple)
    # pool = mp.Pool(min(len(grid_id_feeder_tuples), int(mp.cpu_count()/2)))  # int(mp.cpu_count()/2)

    # results = [pool.apply_async(func=run_optimized_charging_feeder_parallel,
    #                             args=(grid_feeder_tuple, run_id))
    #            for grid_feeder_tuple in grid_id_feeder_tuples]
    # results = pool.map_async(run_optimized_charging_feeder_parallel, grid_id_feeder_tuples).get()
    # pool.close()
    print('It took {} seconds to run the full optimisation.'.format(perf_counter() - t1))
    print('SUCCESS')