import datetime
import multiprocessing as mp
import os
from pathlib import Path

from time import perf_counter

import numpy as np
import pandas as pd

from edisgo.edisgo import import_edisgo_from_files
from edisgo.opf.lopf import (
    BANDS,
    import_flexibility_bands,
    optimize,
    prepare_time_invariant_parameters,
    setup_model,
    update_model,
)
from edisgo.tools.tools import convert_impedances_to_mv

optimize_storage = False
optimize_ev = True


def run_optimized_charging_feeder_parallel(
    grid_feeder_tuple, run="anya_emob", load_results=False, iteration=0
):
    objective = "minimize_loading"
    timesteps_per_iteration = 24
    iterations_per_era = 7
    overlap_iterations = 6
    solver = "gurobi"
    kwargs = {}  # {'v_min':0.91, 'v_max':1.09, 'thermal_limit':0.9}
    config_dict = {
        "objective": objective,
        "solver": solver,
        "timesteps_per_iteration": timesteps_per_iteration,
        "iterations_per_era": iterations_per_era,
        "overlap_iterations": overlap_iterations,
    }
    config_dict.update(kwargs)
    config = pd.Series(config_dict)

    grid_id = grid_feeder_tuple[0]
    feeder_id = grid_feeder_tuple[1]
    # root_dir = r"H:\Grids"
    root_dir = "/home/local/RL-INSTITUT/julian.endres/Projekte/eDisGo-lobaflex/data/load_n_gen_n_emob_n_hp_grids_extracted"
    root_dir = Path(root_dir)
    grid_dir = root_dir / str(grid_id)
    feeder_dir = root_dir / str(grid_id) / "feeder" / f"{feeder_id:02}"
    results_dir = "/home/local/RL-INSTITUT/julian.endres/Projekte/eDisGo-lobaflex/results/"
    results_dir = Path(results_dir)
    # result_dir = "results/{}/{}/{}".format(objective + run, grid_id, feeder_id)
    result_dir = results_dir / run / str(grid_id) / f"{feeder_id:02}"
    os.makedirs(result_dir, exist_ok=True)

    if (len(os.listdir(result_dir)) > 239) and load_results:
        print("Feeder {} of grid {} already solved.".format(feeder_id, grid_id))
        return
    elif (len(os.listdir(result_dir)) > 1) and load_results:
        iterations_finished = int((len(os.listdir(result_dir)) - 1) / 17)
        (
            charging_start,
            energy_level_start,
            start_iter,
        ) = load_values_from_previous_failed_run(
            feeder_id,
            grid_id,
            iteration,
            iterations_per_era,
            overlap_iterations,
            result_dir,
        )

    else:
        charging_start = None
        energy_level_start = None
        start_iter = 0
    config.to_csv(result_dir / "config.csv")

    # try:
    if True:
        edisgo_orig = import_edisgo_from_files(feeder_dir,
                                               import_timeseries=True,
                                               import_topology=True,
                                               import_electromobility=True)

        print("eDisGo object imported.")

        edisgo_obj = convert_impedances_to_mv(edisgo_orig)

        downstream_nodes_matrix = pd.read_csv(
            os.path.join(
                feeder_dir,
                f"downstream_node_matrix_{grid_id}_{feeder_id:02}.csv"
            ),
            index_col=0,
        )

        print("Converted impedances to mv.")

        downstream_nodes_matrix = downstream_nodes_matrix.astype(np.uint8)
        downstream_nodes_matrix = downstream_nodes_matrix.loc[
            edisgo_obj.topology.buses_df.index, edisgo_obj.topology.buses_df.index
        ]
        print("Downstream node matrix imported.")

        # flexibility_bands = import_flexibility_bands(feeder_dir, ["home", "work"])
        flexibility_bands = edisgo_obj.electromobility.flexibility_bands
        print("Flexibility bands imported.")


        # extract data for feeder
        for band in BANDS:
            flexibility_bands[band] = flexibility_bands[band][
                edisgo_obj.topology.charging_points_df.index[
                    edisgo_obj.topology.charging_points_df.index.isin(
                        flexibility_bands[band].columns
                    )
                ]
            ]

        # Create dict with time invariant parameters
        parameters = prepare_time_invariant_parameters(
            edisgo_obj,
            downstream_nodes_matrix,
            pu=False,
            optimize_storage=False,
            optimize_ev_charging=True,
            optimize_hp=False,
            ev_flex_bands=flexibility_bands,
        )
        print("Time-invariant parameters extracted.")

        energy_level = {}
        charging_ev = {}

        for iteration in range(
            start_iter,
            # int(len(edisgo_obj.timeseries.timeindex) / timesteps_per_iteration),
            7
        ):  # edisgo_obj.timeseries.timeindex.week.unique()

            print("Starting optimisation for iteration {}.".format(iteration))
            if iteration % iterations_per_era != iterations_per_era - 1:
                timesteps = edisgo_obj.timeseries.timeindex[
                    iteration
                    * timesteps_per_iteration : (iteration + 1)
                    * timesteps_per_iteration
                    + overlap_iterations
                ]
                energy_level_end = None
            else:
                timesteps = edisgo_obj.timeseries.timeindex[
                    iteration
                    * timesteps_per_iteration : (iteration + 1)
                    * timesteps_per_iteration
                ]
                energy_level_end = True
            # Check if problem will be feasible
            if charging_start is not None:
                low_power_cp = []
                violation_lower_bound_cp = []
                violation_upper_bound_cp = []
                for cp_tmp in energy_level_start.index:
                    if (
                        energy_level_start[cp_tmp]
                        > parameters["ev_flex_bands"]["upper_energy"].loc[
                            timesteps[0], cp_tmp
                        ]
                    ):
                        if (
                            energy_level_start[cp_tmp]
                            - parameters["ev_flex_bands"]["upper_energy"].loc[
                                timesteps[0], cp_tmp
                            ]
                            > 1e-4
                        ):
                            raise ValueError(
                                "Optimisation should not return values higher than "
                                "upper bound. "
                                "Problem for {}. Initial energy level is {}, but "
                                "upper bound {}.".format(
                                    cp_tmp,
                                    energy_level_start[cp_tmp],
                                    parameters["ev_flex_bands"]["upper_energy"].loc[
                                        timesteps[0], cp_tmp
                                    ],
                                )
                            )
                        else:
                            energy_level_start[cp_tmp] = (
                                parameters["ev_flex_bands"]["upper_energy"].loc[
                                    timesteps[0], cp_tmp
                                ]
                                - 1e-6
                            )
                            violation_upper_bound_cp.append(cp_tmp)
                    if (
                        energy_level_start[cp_tmp]
                        < parameters["ev_flex_bands"]["lower_energy"].loc[
                            timesteps[0], cp_tmp
                        ]
                    ):

                        if (
                            -energy_level_start[cp_tmp]
                            + parameters["ev_flex_bands"]["lower_energy"].loc[
                                timesteps[0], cp_tmp
                            ]
                            > 1e-4
                        ):
                            raise ValueError(
                                "Optimisation should not return values lower than "
                                "lower bound. "
                                "Problem for {}. Initial energy level is {}, but lower "
                                "bound {}.".format(
                                    cp_tmp,
                                    energy_level_start[cp_tmp],
                                    parameters["ev_flex_bands"]["lower_energy"].loc[
                                        timesteps[0], cp_tmp
                                    ],
                                )
                            )
                        else:
                            energy_level_start[cp_tmp] = (
                                parameters["ev_flex_bands"]["lower_energy"].loc[
                                    timesteps[0], cp_tmp
                                ]
                                + 1e-6
                            )
                            violation_lower_bound_cp.append(cp_tmp)
                    if charging_start[cp_tmp] < 1e-5:
                        low_power_cp.append(cp_tmp)
                        charging_start[cp_tmp] = 0
                print("Very small charging power: {}, set to 0.".format(low_power_cp))
                print(
                    "Charging points {} violates lower bound.".format(
                        violation_lower_bound_cp
                    )
                )
                print(
                    "Charging points {} violates upper bound.".format(
                        violation_upper_bound_cp
                    )
                )
            try:
                model = update_model(
                    model,
                    timesteps,
                    parameters,
                    optimize_storage=optimize_storage,
                    optimize_ev=optimize_ev,
                    charging_start_ev=charging_start,
                    energy_level_start_ev=energy_level_start,
                    energy_level_end_ev=energy_level_end,
                    **kwargs
                )
            except NameError:
                model = setup_model(
                    parameters,
                    timesteps,
                    objective=objective,
                    optimize_storage=False,
                    optimize_ev_charging=True,
                    charging_start_ev=charging_start,
                    energy_level_start_ev=energy_level_start,
                    energy_level_end_ev=energy_level_end,
                    overlap_interations=overlap_iterations,
                    **kwargs
                )

            print("Set up model for week {}.".format(iteration))

            result_dict = optimize(model, solver)
            charging_ev[iteration] = result_dict["x_charge_ev"]
            energy_level[iteration] = result_dict["energy_level_cp"]

            if iteration % iterations_per_era != iterations_per_era - 1:
                charging_start = charging_ev[iteration].iloc[-overlap_iterations]
                energy_level_start = energy_level[iteration].iloc[-overlap_iterations]
            else:
                charging_start = None
                energy_level_start = None

            print("Finished optimisation for week {}.".format(iteration))
            for res_name, res in result_dict.items():
                try:
                    res = res.loc[edisgo_obj.timeseries.timeindex]
                except Exception:
                    pass
                if "slack" in res_name:
                    res = res[res > 1e-6]
                    res = res.dropna(how="all")
                    res = res.dropna(how="all")
                if not res.empty:
                    res.astype(np.float16).to_csv(
                        result_dir
                        / "{}_{}_{}_{}.csv".format(
                            res_name, grid_id, feeder_id, iteration
                        )
                    )
            print("Saved results for week {}.".format(iteration))
    else:
    # except Exception as e:
        print(
            "Something went wrong with feeder {} of grid {}".format(feeder_id, grid_id)
        )
        print(e)
        if "iteration" in locals():
            if iteration >= 1:
                charging_start = charging_ev[iteration - 1].iloc[-overlap_iterations]
                charging_start.to_csv(
                    result_dir
                    / "charging_start_{}_{}_{}.csv".format(
                        grid_id, feeder_id, iteration
                    )
                )
                energy_level_start = energy_level[iteration - 1].iloc[
                    -overlap_iterations
                ]
                energy_level_start.to_csv(
                    result_dir
                    / "energy_level_start_{}_{}_{}.csv".format(
                        grid_id, feeder_id, iteration
                    )
                )


def load_values_from_previous_failed_run(
    feeder_id, grid_id, iteration, iterations_per_era, overlap_interations, result_dir
):
    print("Importing values from previous run")
    starts = os.listdir(result_dir)
    relevant_starts = [
        start
        for start in starts
        if ("charging_start_{}_{}_".format(grid_id, feeder_id) in start)
        or ("energy_level_start_{}_{}_".format(grid_id, feeder_id) in start)
    ]
    if (len(relevant_starts) > 0) and (
        int(relevant_starts[0].split(".")[0].split("_")[-1]) == iteration
    ):
        iteration = int(relevant_starts[0].split(".")[0].split("_")[-1])
        charging_start = pd.read_csv(
            os.path.join(
                result_dir,
                "charging_start_{}_{}_{}.csv".format(grid_id, feeder_id, iteration),
            ),
            header=None,
            index_col=0,
        )[1]
        energy_level_start = pd.read_csv(
            os.path.join(
                result_dir,
                "energy_level_start_{}_{}_{}.csv".format(grid_id, feeder_id, iteration),
            ),
            header=None,
            index_col=0,
        )[1]
        # if new era starts, set start values to None
        if iteration % iterations_per_era == iterations_per_era - 1:
            charging_start = None
            energy_level_start = None
        start_iter = iteration
    else:
        if iteration is None:
            iteration = int((len(os.listdir(result_dir)) - 1) / 17)
        charging_ev_tmp = pd.read_csv(
            os.path.join(
                result_dir,
                "x_charge_ev_{}_{}_{}.csv".format(grid_id, feeder_id, iteration - 1),
            ),
            index_col=0,
            parse_dates=[0],
        )

        energy_level_tmp = pd.read_csv(
            os.path.join(
                result_dir,
                "energy_band_cp_{}_{}_{}.csv".format(grid_id, feeder_id, iteration - 1),
            ),
            index_col=0,
            parse_dates=[0],
        )
        charging_start = charging_ev_tmp.iloc[-overlap_interations]
        energy_level_start = energy_level_tmp.iloc[-overlap_interations]
        start_iter = iteration
    return charging_start, energy_level_start, start_iter


if __name__ == "__main__":

    # t1 = perf_counter()
    # grid_feeder_tuple = (176, 1)
    # run_optimized_charging_feeder_parallel(grid_feeder_tuple, load_results=False)
    # print('It took {} seconds to run the full optimisation.'.format(
    #     perf_counter()-t1))
    # print('SUCCESS')
    # t1 = perf_counter()

    # grid_ids = [1056]
    # # root_dir = r"H:\Grids"
    # root_dir = "/home/local/RL-INSTITUT/julian.endres/Projekte/eDisGo-lobaflex/data/load_n_gen_n_emob_n_hp_grids_extracted/1056"
    # grid_id_feeder_tuples = []  # [(2534,0), (2534,1), (2534,6)](176, 6)
    # run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    # for grid_id in grid_ids:
    #     edisgo_dir = root_dir + r"\{}\feeder".format(grid_id)
    #     for feeder in os.listdir(edisgo_dir):
    #         grid_id_feeder_tuples.append((grid_id, feeder))
    #
    # pool = mp.Pool(
    #     min(len(grid_id_feeder_tuples), int(mp.cpu_count() / 2))
    # )  # int(mp.cpu_count()/2)
    #
    # # results = [pool.apply_async(func=run_optimized_charging_feeder_parallel,
    # #                             args=(grid_feeder_tuple, run_id))
    # #            for grid_feeder_tuple in grid_id_feeder_tuples]
    # results = pool.map_async(
    #     run_optimized_charging_feeder_parallel, grid_id_feeder_tuples
    # ).get()
    # pool.close()
    # print(
    #     "It took {} seconds to run the full optimisation.".format(perf_counter() - t1)
    # )
    # print("SUCCESS")

    run_optimized_charging_feeder_parallel(grid_feeder_tuple=(1056,1))
