import logging
import multiprocessing
import os

from itertools import product

import pandas as pd

import results_helper_functions

from edisgo.io import pypsa_io
from edisgo.tools.tools import assign_feeder, get_path_length_to_station
from SCENARIO_run_powerflow import import_edisgo_object_with_adapted_charging_timeseries

logger = logging.getLogger("pypsa")
logger.setLevel(logging.ERROR)

# possible options for scenario are "optimised", "no_ev", "dumb", "reduced","residual"
scenarios = ["optimised", "no_ev", "dumb", "reduced", "residual"]

results_base_path = r"H:\Grids"


mv_grid_ids = [176, 177, 1056, 1690, 1811, 2534]  # 176, 177, 1056, 1690, 1811, 2534
variations = list(product(mv_grid_ids, scenarios))

num_threads = int(multiprocessing.cpu_count() / 2)
curtailment_step = 0.01
max_iterations = 1000
use_seed = False


def _overwrite_edisgo_timeseries(edisgo, pypsa_network):
    """
    Overwrites generator and load time series in edisgo after curtailment.

    pypsa_network contains the curtailed time series that are written to
    edisgo object.

    """

    # overwrite time series in edisgo
    time_steps = pypsa_network.generators_t.p_set.index

    # generators: overwrite time series for all except slack
    gens = pypsa_network.generators[pypsa_network.generators.control != "Slack"].index
    edisgo.timeseries._generators_active_power.loc[
        time_steps, gens
    ] = pypsa_network.generators_t.p_set.loc[time_steps, gens]
    edisgo.timeseries._generators_reactive_power.loc[
        time_steps, gens
    ] = pypsa_network.generators_t.q_set.loc[time_steps, gens]

    # loads: distinguish between charging points and conventional loads
    loads = [_ for _ in pypsa_network.loads.index if "ChargingPoint" not in _]
    edisgo.timeseries._loads_active_power.loc[
        time_steps, loads
    ] = pypsa_network.loads_t.p_set.loc[time_steps, loads]
    edisgo.timeseries._loads_reactive_power.loc[
        time_steps, loads
    ] = pypsa_network.loads_t.q_set.loc[time_steps, loads]

    if not edisgo.topology.charging_points_df.empty:
        charging_points = [_ for _ in pypsa_network.loads.index if "ChargingPoint" in _]
        edisgo.timeseries._charging_points_active_power.loc[
            time_steps, charging_points
        ] = pypsa_network.loads_t.p_set.loc[time_steps, charging_points]
        edisgo.timeseries._charging_points_reactive_power.loc[
            time_steps, charging_points
        ] = pypsa_network.loads_t.q_set.loc[time_steps, charging_points]


def _save_results_when_curtailment_failed(edisgo_obj, results_dir, mode):
    edisgo_obj.save(
        os.path.join(results_dir, "edisgo_curtailment_{}".format(mode)),
        parameters="powerflow_results",
    )

    rel_load = results_helper_functions.relative_load(edisgo_obj)
    rel_load.to_csv(
        os.path.join(
            results_dir, "edisgo_curtailment_{}".format(mode), "relative_load.csv"
        )
    )
    voltage_dev = results_helper_functions.voltage_diff(edisgo_obj)
    voltage_dev.to_csv(
        os.path.join(
            results_dir, "edisgo_curtailment_{}".format(mode), "voltage_deviation.csv"
        )
    )


def _curtail(pypsa_network, gens, loads, time_steps, elia_logger):
    # get time series for loads and generators
    gens_ts = pypsa_network.generators_t.p_set.loc[time_steps, gens]
    loads_ts = pypsa_network.loads_t.p_set.loc[time_steps, loads]

    # evaluate whether it is a load or feed-in case
    # calculate residual load
    residual_load = gens_ts.sum(axis=1) - loads_ts.sum(axis=1)
    ts_res_load_zero = residual_load[residual_load == 0.0].index
    if len(ts_res_load_zero) > 0:
        elia_logger.info(
            "Number of time steps with residual load of zero: {}".format(
                len(ts_res_load_zero)
            )
        )

    # get time steps where to curtail generators and where to
    # curtail loads
    ts_curtail_gens = residual_load[residual_load > 0.0].index
    ts_curtail_loads = residual_load[residual_load < 0.0].index

    # curtail loads or generators by specified curtailment factor
    # active power
    pypsa_network.generators_t.p_set.loc[ts_curtail_gens, gens] = (
        gens_ts.loc[ts_curtail_gens, :]
        - curtailment_step * gens_ts.loc[ts_curtail_gens, :]
    )
    pypsa_network.loads_t.p_set.loc[ts_curtail_loads, loads] = (
        loads_ts.loc[ts_curtail_loads, :]
        - curtailment_step * loads_ts.loc[ts_curtail_loads, :]
    )
    # reactive power
    tmp = pypsa_network.generators_t.q_set.loc[ts_curtail_gens, gens]
    pypsa_network.generators_t.q_set.loc[ts_curtail_gens, gens] = (
        tmp - curtailment_step * tmp
    )
    tmp = pypsa_network.loads_t.q_set.loc[ts_curtail_loads, loads]
    pypsa_network.loads_t.q_set.loc[ts_curtail_loads, loads] = (
        tmp - curtailment_step * tmp
    )

    return pypsa_network


def _calculate_curtailed_energy(pypsa_network_orig, pypsa_network):
    gens = pypsa_network_orig.generators[
        pypsa_network_orig.generators.control != "Slack"
    ].index
    curtailed_feedin_ts = (
        pypsa_network_orig.generators_t.p_set.loc[:, gens]
        - pypsa_network.generators_t.p_set.loc[:, gens]
    )
    curtailed_load_ts = pypsa_network_orig.loads_t.p_set - pypsa_network.loads_t.p_set
    return curtailed_feedin_ts, curtailed_load_ts


def _calculate_curtailed_energy_incl_reactive(pypsa_network_orig, pypsa_network):
    gens = pypsa_network_orig.generators[
        pypsa_network_orig.generators.control != "Slack"
    ].index
    curtailed_feedin_ts = (
        pypsa_network_orig.generators_t.p_set.loc[:, gens]
        - pypsa_network.generators_t.p_set.loc[:, gens]
    )
    curtailed_feedin_reactive_ts = (
        pypsa_network_orig.generators_t.q_set.loc[:, gens]
        - pypsa_network.generators_t.q_set.loc[:, gens]
    )
    curtailed_load_ts = pypsa_network_orig.loads_t.p_set - pypsa_network.loads_t.p_set
    curtailed_load_reactive_ts = (
        pypsa_network_orig.loads_t.q_set - pypsa_network.loads_t.q_set
    )
    return (
        curtailed_feedin_ts,
        curtailed_feedin_reactive_ts,
        curtailed_load_ts,
        curtailed_load_reactive_ts,
    )


def _update_df(df, df_update):
    # update columns and indexes already existing in df
    df.update(df_update)
    # in case of time steps with convergence issues, these might not yet
    # be in the index of df and therefore need to be appended
    ts_new = [_ for _ in df_update.index if _ not in df.index]
    if len(ts_new) > 0:
        print("ts_new")
        print(ts_new)
        df = pd.concat([df, df_update.loc[ts_new, :]], sort=False)
    return df


def solve_convergence_issues(
    edisgo, pypsa_network, timesteps, curtailment, voltage_dev, rel_load
):
    # for convergence issues in MV/LV grid, find feeder with convergence issues
    # and curtail only in that respective feeder

    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to convergence issues.")
    elia_logger.debug(
        "Number of time steps with convergence issues: {}".format(len(timesteps))
    )

    timesteps_not_converged = timesteps

    # save original pypsa network to determine curtailed energy
    pypsa_network_orig = pypsa_network.copy()

    for timestep in timesteps:

        elia_logger.debug("Time step with convergence issue: {}".format(timestep))

        iteration_count = 0
        converged = False
        while converged is False and iteration_count < max_iterations:

            elia_logger.debug("Iteration round: {}".format(iteration_count))

            # find feeder to curtail

            # perform linear power flow to check where voltage angle is largest
            # this is used as an indicator as to where the convergence problem
            # originates from
            pypsa_network.lpf(timestep)

            angle_diff = pd.Series(
                pypsa_network.buses_t.v_ang.loc[
                    timestep, pypsa_network.lines.bus0
                ].values
                - pypsa_network.buses_t.v_ang.loc[
                    timestep, pypsa_network.lines.bus1
                ].values,
                index=pypsa_network.lines.index,
            )
            angle_diff.sort_values(ascending=False, inplace=True)

            feeder = edisgo.topology.lines_df.at[angle_diff.index[0], "mv_feeder"]

            # get all buses in feeder
            buses = edisgo.topology.buses_df[
                edisgo.topology.buses_df.mv_feeder == feeder
            ].index

            gens_feeder = edisgo.topology.generators_df[
                edisgo.topology.generators_df.bus.isin(buses)
            ].index
            loads_feeder = edisgo.topology.loads_df[
                edisgo.topology.loads_df.bus.isin(buses)
            ].index
            loads_feeder = loads_feeder.append(
                edisgo.topology.charging_points_df[
                    edisgo.topology.charging_points_df.bus.isin(buses)
                ].index
            )

            pypsa_network = _curtail(
                pypsa_network, gens_feeder, loads_feeder, timestep, elia_logger
            )

            pypsa_network.lpf(timestep)
            pf_results = pypsa_network.pf(timestep, use_seed=True)

            # check if power flow converged
            tmp = pf_results["converged"][~pf_results["converged"]["0"]].index

            if len(tmp) == 0:
                converged = True

            iteration_count += 1

    # update voltage dev and relative load
    pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_not_converged)
    voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
    voltage_dev = _update_df(voltage_dev, voltage_dev_new)
    rel_load_new = results_helper_functions.relative_load(edisgo)
    rel_load = _update_df(rel_load, rel_load_new)

    # overwrite time series in edisgo
    _overwrite_edisgo_timeseries(edisgo, pypsa_network)

    # calculate curtailment
    curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
        pypsa_network_orig, pypsa_network
    )
    curtailment.at["convergence_issues_mv", "feed-in"] += curtailed_feedin.sum().sum()
    curtailment.at["convergence_issues_mv", "load"] += curtailed_load.sum().sum()

    return curtailment, voltage_dev, rel_load


def solve_convergence_issues_reinforce(
    edisgo, pypsa_network, timesteps, curtailment, voltage_dev, rel_load
):
    # for convergence issues in MV/LV grid, find feeder with convergence issues
    # and curtail only in that respective feeder

    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to convergence issues.")
    elia_logger.debug(
        "Number of time steps with convergence issues: {}".format(len(timesteps))
    )

    timesteps_not_converged = timesteps

    # save original pypsa network to determine curtailed energy
    pypsa_network_orig = pypsa_network.copy()

    for timestep in timesteps:

        elia_logger.debug("Time step with convergence issue: {}".format(timestep))

        iteration_count = 0
        converged = False
        while converged is False and iteration_count < max_iterations:

            elia_logger.debug("Iteration round: {}".format(iteration_count))

            # find feeder to curtail

            # perform linear power flow to check where voltage angle is largest
            # this is used as an indicator as to where the convergence problem
            # originates from
            pypsa_network.lpf(timestep)

            angle_diff = pd.Series(
                pypsa_network.buses_t.v_ang.loc[
                    timestep, pypsa_network.lines.bus0
                ].values
                - pypsa_network.buses_t.v_ang.loc[
                    timestep, pypsa_network.lines.bus1
                ].values,
                index=pypsa_network.lines.index,
            )
            angle_diff.sort_values(ascending=False, inplace=True)

            feeder = edisgo.topology.lines_df.at[angle_diff.index[0], "mv_feeder"]

            # get all buses in feeder
            buses = edisgo.topology.buses_df[
                edisgo.topology.buses_df.mv_feeder == feeder
            ].index

            gens_feeder = edisgo.topology.generators_df[
                edisgo.topology.generators_df.bus.isin(buses)
            ].index
            loads_feeder = edisgo.topology.loads_df[
                edisgo.topology.loads_df.bus.isin(buses)
            ].index
            loads_feeder = loads_feeder.append(
                edisgo.topology.charging_points_df[
                    edisgo.topology.charging_points_df.bus.isin(buses)
                ].index
            )

            pypsa_network = _curtail(
                pypsa_network, gens_feeder, loads_feeder, [timestep], elia_logger
            )

            pypsa_network.lpf(timestep)
            pf_results = pypsa_network.pf(timestep, use_seed=True)

            # check if power flow converged
            tmp = pf_results["converged"][~pf_results["converged"]["0"]].index

            if len(tmp) == 0:
                converged = True

            iteration_count += 1

    # update voltage dev and relative load
    pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_not_converged)
    voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
    voltage_dev = _update_df(voltage_dev, voltage_dev_new)
    rel_load_new = results_helper_functions.relative_load(edisgo)
    rel_load = _update_df(rel_load, rel_load_new)

    # overwrite time series in edisgo
    _overwrite_edisgo_timeseries(edisgo, pypsa_network)

    # calculate curtailment
    (
        curtailed_feedin,
        curtailed_feedin_reactive,
        curtailed_load,
        curtailed_load_reactive,
    ) = _calculate_curtailed_energy_incl_reactive(pypsa_network_orig, pypsa_network)
    curtailment.at["convergence_issues_mv", "feed-in"] += curtailed_feedin.sum().sum()
    curtailment.at["convergence_issues_mv", "load"] += curtailed_load.sum().sum()

    return (
        curtailed_feedin,
        curtailed_feedin_reactive,
        curtailed_load,
        curtailed_load_reactive,
        curtailment,
        voltage_dev,
        rel_load,
    )


def solve_convergence_issues_lv_grid(
    edisgo, pypsa_network, timesteps, curtailment, voltage_dev, rel_load
):
    # for convergence issues in LV grid, all loads/generators in grid are
    # curtailed

    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to convergence issues.")

    timesteps_not_converged = timesteps

    # save original pypsa network to determine curtailed energy
    pypsa_network_orig = pypsa_network.copy()

    # curtail time steps with convergence issues
    iteration_count = 0
    while len(timesteps) > 0 and iteration_count < max_iterations:
        elia_logger.debug(
            "Number of time steps with convergence issues: {}".format(len(timesteps))
        )

        pypsa_network = _curtail(
            pypsa_network,
            pypsa_network.generators.index,
            pypsa_network.loads.index,
            timesteps,
            elia_logger,
        )

        # perform linear power flow first to attain seed
        pypsa_network.lpf(timesteps)
        pf_results = pypsa_network.pf(timesteps, use_seed=True)

        # get time steps where power flow still does not converge
        timesteps = pf_results["converged"][~pf_results["converged"]["0"]].index

        iteration_count += 1

    # update voltage dev and relative load
    pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_not_converged)
    voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
    voltage_dev = _update_df(voltage_dev, voltage_dev_new)
    rel_load_new = results_helper_functions.relative_load(edisgo)
    rel_load = _update_df(rel_load, rel_load_new)

    # overwrite time series in edisgo
    _overwrite_edisgo_timeseries(edisgo, pypsa_network)

    # calculate curtailment
    curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
        pypsa_network_orig, pypsa_network
    )
    curtailment.at["convergence_issues_lv", "feed-in"] += curtailed_feedin.sum().sum()
    curtailment.at["convergence_issues_lv", "load"] += curtailed_load.sum().sum()

    return curtailment, voltage_dev, rel_load


def curtailment_lv_voltage(
    edisgo, curtailment, voltage_dev, rel_load, grid_results_dir
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to voltage issues in the LV.")

    # get voltage issues in LV
    lv_buses = edisgo.topology.buses_df.lv_feeder.dropna().index
    voltage_dev_lv = voltage_dev.loc[:, lv_buses]
    voltage_issues = (
        voltage_dev_lv[voltage_dev_lv != 0.0]
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    buses_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(timesteps=time_steps_issues, use_seed=use_seed)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with voltage issues
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "lv_feeder"].unique()

            elia_logger.debug(
                "Number of LV feeders with voltage issues: {}".format(len(feeders))
            )
            elia_logger.debug(
                "Number of time steps with voltage issues in LV: {}".format(
                    len(time_steps_issues)
                )
            )

            for feeder in feeders:
                # get all buses in feeder
                buses = edisgo.topology.buses_df[
                    edisgo.topology.buses_df.lv_feeder == feeder
                ].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses)
                ].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses)
                ].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(buses)
                    ].index
                )

                # get time steps with voltage issues in feeder
                ts_issues = (
                    voltage_issues.loc[:, buses[buses.isin(voltage_issues.columns)]]
                    .dropna(how="all")
                    .index
                )

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues, elia_logger
                )

            # run power flow analysis on all time steps with voltage issues
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues_lv_grid(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # get voltage issues in LV
            voltage_dev_lv = voltage_dev.loc[:, lv_buses]
            voltage_issues = (
                voltage_dev_lv[voltage_dev_lv != 0.0]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            buses_issues = voltage_issues.columns
            time_steps_issues = voltage_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "lv_voltage"
            )

            raise ValueError(
                "Curtailment not sufficient to solve LV voltage " "issues."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No LV voltage issues to solve.")
    return curtailment, voltage_dev, rel_load


def curtailment_lv_voltage_single_lv_grid(
    edisgo,
    pypsa_network,
    curtailment,
    voltage_dev,
    rel_load,
    grid_results_dir,
    timesteps_not_converged,
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))

    # get voltage issues in LV grid
    lv_buses = pypsa_network.buses.lv_feeder.dropna().index
    voltage_dev_grid = voltage_dev.loc[:, lv_buses]
    voltage_issues = (
        voltage_dev_grid[voltage_dev_grid != 0.0]
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    buses_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index.append(timesteps_not_converged).unique()

    if len(time_steps_issues) > 0:

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with voltage issues
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "lv_feeder"].unique()

            elia_logger.debug(
                "Number of LV feeders with voltage issues: {}".format(len(feeders))
            )
            elia_logger.debug(
                "Number of time steps with voltage issues "
                "in LV grid: {}".format(len(time_steps_issues))
            )

            for feeder in feeders:
                # get all buses in feeder
                buses = pypsa_network.buses[
                    pypsa_network.buses.lv_feeder == feeder
                ].index

                gens_feeder = pypsa_network.generators[
                    pypsa_network.generators.bus.isin(buses)
                ].index
                loads_feeder = pypsa_network.loads[
                    pypsa_network.loads.bus.isin(buses)
                ].index

                # get time steps with voltage issues in feeder
                ts_issues = (
                    voltage_issues.loc[:, buses[buses.isin(voltage_issues.columns)]]
                    .dropna(how="all")
                    .index
                )

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues, elia_logger
                )

            # run power flow analysis on all time steps with voltage issues
            pf_results = pypsa_network.pf(time_steps_issues)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues_lv_grid(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # get voltage issues in LV
            voltage_dev_lv = voltage_dev.loc[:, lv_buses]
            voltage_issues = (
                voltage_dev_lv[voltage_dev_lv != 0.0]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            buses_issues = voltage_issues.columns
            time_steps_issues = voltage_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "lv_voltage_single_lv_grid"
            )

            raise ValueError(
                "Curtailment not sufficient to solve LV voltage "
                "issues for single LV grid."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No LV voltage issues to solve.")
    return curtailment, voltage_dev, rel_load


def curtailment_mvlv_stations_voltage(
    edisgo, curtailment, voltage_dev, rel_load, grid_results_dir
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to voltage issues at MV/LV " "stations.")

    # get stations with voltage issues
    mvlv_stations = edisgo.topology.transformers_df.bus1.unique()
    voltage_dev_mvlv_stations = voltage_dev.loc[:, mvlv_stations]
    voltage_issues = (
        voltage_dev_mvlv_stations[voltage_dev_mvlv_stations != 0.0]
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    stations_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        # create pypsa network with aggregated loads and generators at
        # station's secondary side
        # ToDo Aggregating the LV leads to slightly different voltage results
        #  wherefore checking voltage after running power flow with
        #  non-aggregated LV might show some remaining voltage issues. The
        #  following might therefore need to be changed.
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues, use_seed=use_seed
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(stations_issues) > 0 and iteration_count < max_iterations:

            elia_logger.debug(
                "Number of MV/LV stations with voltage issues: {}".format(
                    len(stations_issues)
                )
            )
            elia_logger.debug(
                "Number of time steps with voltage issues at "
                "MV/LV stations: {}".format(len(time_steps_issues))
            )

            # for each station calculate curtailment
            for station in stations_issues:
                # get loads and gens in grid
                gens_grid = pypsa_network.generators[
                    pypsa_network.generators.bus == station
                ].index
                loads_grid = pypsa_network.loads[
                    pypsa_network.loads.bus == station
                ].index

                # get time steps with issues at that station
                ts_issues = voltage_issues.loc[:, station].dropna(how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_grid, loads_grid, ts_issues, elia_logger
                )

            # run power flow analysis on limited number of time steps
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # get stations with voltage issues
            voltage_dev_mvlv_stations = voltage_dev.loc[:, mvlv_stations]
            voltage_issues = (
                voltage_dev_mvlv_stations[voltage_dev_mvlv_stations != 0.0]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            stations_issues = voltage_issues.columns
            time_steps_issues = voltage_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(stations_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "mvlv_stations_voltage"
            )

            raise ValueError(
                "Curtailment not sufficient to solve voltage "
                "issues at MV/LV stations."
            )

        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV/LV stations with voltage issues.")
    return curtailment, voltage_dev, rel_load


def curtailment_mv_voltage(
    edisgo, curtailment, voltage_dev, rel_load, grid_results_dir
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to voltage issues in the MV.")

    # get voltage issues in MV
    mv_buses = edisgo.topology.mv_grid.buses_df.index
    voltage_dev_mv = voltage_dev.loc[:, mv_buses]
    voltage_issues = (
        voltage_dev_mv[voltage_dev_mv != 0.0]
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    buses_issues = voltage_issues.columns
    time_steps_issues = voltage_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues, use_seed=use_seed
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with voltage issues
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "mv_feeder"].unique()

            elia_logger.debug(
                "Number of MV feeders with voltage issues: {}".format(len(feeders))
            )
            elia_logger.debug(
                "Number of time steps with voltage issues in MV: {}".format(
                    len(time_steps_issues)
                )
            )

            for feeder in feeders:
                # get all buses in feeder
                buses = edisgo.topology.buses_df[
                    edisgo.topology.buses_df.mv_feeder == feeder
                ].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses)
                ].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses)
                ].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(buses)
                    ].index
                )

                # get time steps with voltage issues in feeder
                ts_issues = (
                    voltage_issues.loc[:, buses[buses.isin(voltage_issues.columns)]]
                    .dropna(how="all")
                    .index
                )

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues, elia_logger
                )

            # run power flow analysis on all time steps with MV issues
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # get voltage issues in MV
            voltage_dev_mv = voltage_dev.loc[:, mv_buses]
            voltage_issues = (
                voltage_dev_mv[voltage_dev_mv != 0.0]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            buses_issues = voltage_issues.columns
            time_steps_issues = voltage_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "mv_voltage"
            )

            raise ValueError(
                "Curtailment not sufficient to solve MV voltage " "issues."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV voltage issues to solve.")
    return curtailment, voltage_dev, rel_load


def curtailment_mv_voltage_10_percent(
    edisgo, curtailment, voltage_dev, rel_load, grid_results_dir
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to 10 percent issues.")

    # get 10% voltage issues
    v_mag_pu_pfa = edisgo.results.v_res
    voltage_issues = (
        v_mag_pu_pfa[(v_mag_pu_pfa > 1.1) | (v_mag_pu_pfa < 0.9)]
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    time_steps_issues = voltage_issues.index
    buses_issues = voltage_issues.columns

    if len(time_steps_issues) > 0:
        # ToDo maybe set mode
        pypsa_network = edisgo.to_pypsa(timesteps=time_steps_issues, use_seed=use_seed)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with voltage issues
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "mv_feeder"].unique()

            elia_logger.debug(
                "Number of MV feeders with voltage issues: {}".format(len(feeders))
            )
            elia_logger.debug(
                "Number of time steps with voltage issues in MV: {}".format(
                    len(time_steps_issues)
                )
            )

            for feeder in feeders:
                # get all buses in feeder
                buses = edisgo.topology.buses_df[
                    edisgo.topology.buses_df.mv_feeder == feeder
                ].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses)
                ].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses)
                ].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(buses)
                    ].index
                )

                # get time steps with voltage issues in feeder
                ts_issues = (
                    voltage_issues.loc[:, buses[buses.isin(voltage_issues.columns)]]
                    .dropna(how="all")
                    .index
                )

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues, elia_logger
                )

            # run power flow analysis on all time steps with MV issues
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # get voltage issues in MV
            v_mag_pu_pfa = edisgo.results.v_res
            voltage_issues = (
                v_mag_pu_pfa[(v_mag_pu_pfa > 1.1) | (v_mag_pu_pfa < 0.9)]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            time_steps_issues = voltage_issues.index
            buses_issues = voltage_issues.columns

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "mv_voltage_10_percent"
            )

            raise ValueError(
                "Curtailment not sufficient to solve MV voltage " "issues."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV voltage issues to solve.")
    return curtailment, voltage_dev, rel_load


def curtailment_lv_lines_overloading(
    edisgo, curtailment, voltage_dev, rel_load, grid_results_dir
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to overloading issues in the LV.")

    # get overloading issues in LV
    lv_lines = edisgo.topology.lines_df.lv_feeder.dropna().index
    rel_load_lv = rel_load.loc[:, lv_lines]
    overloading_issues = (
        rel_load_lv[rel_load_lv > 1.0].dropna(how="all").dropna(axis=1, how="all")
    )
    lines_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(timesteps=time_steps_issues, use_seed=use_seed)

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        buses_df = edisgo.topology.buses_df

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with overloading issues
            # get all buses with issues
            buses_issues = (
                edisgo.topology.lines_df.loc[lines_issues, ["bus0", "bus1"]]
                .stack()
                .unique()
            )
            buses_df_issues = buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "lv_feeder"].dropna().unique()

            elia_logger.debug(
                "Number of LV feeders with overloading issues: {}".format(len(feeders))
            )
            elia_logger.debug(
                "Number of time steps with overloading issues "
                "in LV: {}".format(len(time_steps_issues))
            )

            for feeder in feeders:
                # get bus with issues in feeder farthest away from station
                # in order to start curtailment there
                buses_in_feeder = buses_df_issues[buses_df_issues.lv_feeder == feeder]
                b = (
                    buses_in_feeder.loc[:, "path_length_to_station"]
                    .sort_values(ascending=False)
                    .index[0]
                )

                # get all generators and loads downstream
                buses_downstream = buses_df[
                    (buses_df.lv_feeder == feeder)
                    & (
                        buses_df.path_length_to_station
                        >= buses_in_feeder.at[b, "path_length_to_station"]
                    )
                ].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses_downstream)
                ].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses_downstream)
                ].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(buses_downstream)
                    ].index
                )

                # get time steps with overloading issues at that line
                connected_lines = edisgo.topology.get_connected_lines_from_bus(b).index
                rel_load_connected_lines = rel_load.loc[:, connected_lines]
                ts_issues = (
                    rel_load_connected_lines[rel_load_connected_lines > 1.0]
                    .dropna(how="all")
                    .dropna(axis=1, how="all")
                    .index
                )

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues, elia_logger
                )

            # run power flow analysis on all time steps with MV issues
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues_lv_grid(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # recheck for overloading issues in LV
            rel_load_lv = rel_load.loc[:, lv_lines]
            overloading_issues = (
                rel_load_lv[rel_load_lv > 1].dropna(how="all").dropna(axis=1, how="all")
            )
            lines_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "lv_overloading"
            )

            raise ValueError(
                "Curtailment not sufficient to solve overloading " "issues in LV."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No LV overloading issues to solve.")
    return curtailment, voltage_dev, rel_load


def curtailment_lv_lines_overloading_single_lv_grid(
    edisgo,
    pypsa_network,
    curtailment,
    voltage_dev,
    rel_load,
    grid_results_dir,
    timesteps_not_converged,
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))

    # get overloading issues in LV
    rel_load_grid = rel_load.loc[:, pypsa_network.lines.index]
    overloading_issues = (
        rel_load_grid[rel_load_grid > 1.0].dropna(how="all").dropna(axis=1, how="all")
    )
    lines_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index.append(
        timesteps_not_converged
    ).unique()

    if len(time_steps_issues) > 0:

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        buses_df = pypsa_network.buses

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with overloading issues
            # get all buses with issues
            buses_issues = (
                pypsa_network.lines.loc[lines_issues, ["bus0", "bus1"]].stack().unique()
            )
            buses_df_issues = buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "lv_feeder"].dropna().unique()

            elia_logger.debug(
                "Number of LV feeders with overloading issues: {}".format(len(feeders))
            )
            elia_logger.debug(
                "Number of time steps with overloading issues "
                "in LV grid: {}".format(len(time_steps_issues))
            )

            for feeder in feeders:
                # get bus with issues in feeder farthest away from station
                # in order to start curtailment there
                buses_in_feeder = buses_df_issues[buses_df_issues.lv_feeder == feeder]
                b = (
                    buses_in_feeder.loc[:, "path_length_to_station"]
                    .sort_values(ascending=False)
                    .index[0]
                )

                # get all generators and loads downstream
                buses_downstream = buses_df[
                    (buses_df.lv_feeder == feeder)
                    & (
                        buses_df.path_length_to_station
                        >= buses_in_feeder.at[b, "path_length_to_station"]
                    )
                ].index

                gens_feeder = pypsa_network.generators[
                    pypsa_network.generators.bus.isin(buses_downstream)
                ].index
                loads_feeder = pypsa_network.loads[
                    pypsa_network.loads.bus.isin(buses_downstream)
                ].index

                # get time steps with overloading issues at that line
                connected_lines = edisgo.topology.get_connected_lines_from_bus(b).index
                rel_load_connected_lines = rel_load.loc[:, connected_lines]
                ts_issues = (
                    rel_load_connected_lines[rel_load_connected_lines > 1]
                    .dropna(how="all")
                    .dropna(axis=1, how="all")
                    .index
                )

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues, elia_logger
                )

            # run power flow analysis on all time steps with overloading issues
            # in LV grid
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues_lv_grid(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # recheck for overloading issues in LV
            rel_load_grid = rel_load.loc[:, pypsa_network.lines.index]
            overloading_issues = (
                rel_load_grid[rel_load_grid > 1.0]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            lines_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "lv_overloading_single_lv_grid"
            )

            raise ValueError(
                "Curtailment not sufficient to solve overloading "
                "issues in single LV grid."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No LV overloading issues to solve.")
    return curtailment, voltage_dev, rel_load


def curtailment_mvlv_stations_overloading(
    edisgo, curtailment, voltage_dev, rel_load, grid_results_dir
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info(
        "Start curtailment due to overloading issues at MV/LV " "stations."
    )

    # get overloading issues at MV/LV stations
    mvlv_stations = [_ for _ in rel_load.columns if "mvlv_station" in _]
    rel_load_mvlv_stations = rel_load.loc[:, mvlv_stations]
    overloading_issues = (
        rel_load_mvlv_stations[rel_load_mvlv_stations > 0.98]
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    stations_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index

    stations_secondary_sides = {
        _: "BusBar_mvgd_{}_lvgd_{}_LV".format(edisgo.topology.id, _.split("_")[-1])
        for _ in mvlv_stations
    }

    if len(time_steps_issues) > 0:
        # create pypsa network with aggregated loads and generators at
        # station's secondary side
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues, use_seed=use_seed
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(stations_issues) > 0 and iteration_count < max_iterations:

            elia_logger.debug(
                "Number of MV/LV stations with overloading issues: {}".format(
                    len(stations_issues)
                )
            )
            elia_logger.debug(
                "Number of time steps with overloading issues at "
                "MV/LV stations: {}".format(len(time_steps_issues))
            )

            # for each station calculate curtailment
            for station in stations_issues:
                # get loads and gens in grid
                gens_grid = pypsa_network.generators[
                    pypsa_network.generators.bus == stations_secondary_sides[station]
                ].index
                loads_grid = pypsa_network.loads[
                    pypsa_network.loads.bus == stations_secondary_sides[station]
                ].index

                # get time steps with issues at that station
                ts_issues = overloading_issues.loc[:, station].dropna(how="all").index

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_grid, loads_grid, ts_issues, elia_logger
                )

            # run power flow analysis on limited number of time steps
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # recheck for overloading and voltage issues at stations
            rel_load_mvlv_stations = rel_load.loc[:, mvlv_stations]
            overloading_issues = (
                rel_load_mvlv_stations[rel_load_mvlv_stations > 0.98]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            stations_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "mvlv_stations_overloading"
            )

            raise ValueError(
                "Curtailment not sufficient to solve overloading "
                "issues at MV/LV stations."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["lv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["lv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV/LV stations with overloading issues.")
    return curtailment, voltage_dev, rel_load


def curtailment_mv_lines_overloading(
    edisgo, curtailment, voltage_dev, rel_load, grid_results_dir
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info("Start curtailment due to overloading issues in the MV.")

    mv_lines = edisgo.topology.mv_grid.lines_df.index
    rel_load_mv = rel_load.loc[:, mv_lines]
    overloading_issues = (
        rel_load_mv[rel_load_mv > 0.98]
        .dropna(how="all")  # Todo: Why 0.98?
        .dropna(axis=1, how="all")
    )
    lines_issues = overloading_issues.columns
    time_steps_issues = overloading_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues, use_seed=use_seed
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        buses_df = edisgo.topology.buses_df

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            # get feeders with overloading issues
            # get all buses with issues
            buses_issues = (
                edisgo.topology.lines_df.loc[lines_issues, ["bus0", "bus1"]]
                .stack()
                .unique()
            )
            buses_df_issues = edisgo.topology.buses_df.loc[buses_issues, :]
            feeders = buses_df_issues.loc[:, "mv_feeder"].dropna().unique()

            elia_logger.debug(
                "Number of MV feeders with overloading issues: {}".format(len(feeders))
            )
            elia_logger.debug(
                "Number of time steps with overloading issues "
                "in LV: {}".format(len(time_steps_issues))
            )

            for feeder in feeders:
                # get bus with issues in feeder farthest away from station
                # in order to start curtailment there
                buses_in_feeder = buses_df_issues[buses_df_issues.mv_feeder == feeder]
                b = (
                    buses_in_feeder.loc[:, "path_length_to_station"]
                    .sort_values(ascending=False)
                    .index[0]
                )

                # get all generators and loads downstream
                buses_downstream = buses_df[
                    (buses_df.mv_feeder == feeder)
                    & (
                        buses_df.path_length_to_station
                        >= buses_in_feeder.at[b, "path_length_to_station"]
                    )
                ].index

                gens_feeder = edisgo.topology.generators_df[
                    edisgo.topology.generators_df.bus.isin(buses_downstream)
                ].index
                loads_feeder = edisgo.topology.loads_df[
                    edisgo.topology.loads_df.bus.isin(buses_downstream)
                ].index
                loads_feeder = loads_feeder.append(
                    edisgo.topology.charging_points_df[
                        edisgo.topology.charging_points_df.bus.isin(buses_downstream)
                    ].index
                )

                # get time steps with overloading issues at that line
                connected_lines = edisgo.topology.get_connected_lines_from_bus(b).index
                rel_load_connected_lines = rel_load.loc[:, connected_lines]
                ts_issues = (
                    rel_load_connected_lines[rel_load_connected_lines > 0.98]
                    .dropna(how="all")
                    .dropna(axis=1, how="all")
                    .index
                )

                # reduce active and reactive power of loads or generators
                # (depending on whether it is a load or feed-in case)
                pypsa_network = _curtail(
                    pypsa_network, gens_feeder, loads_feeder, ts_issues, elia_logger
                )

            # run power flow analysis on all time steps with MV issues
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # recheck for overloading issues in MV
            rel_load_mv = rel_load.loc[:, mv_lines]
            overloading_issues = (
                rel_load_mv[rel_load_mv > 0.98]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            lines_issues = overloading_issues.columns
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "mv_overloading"
            )

            raise ValueError(
                "Curtailment not sufficient to solve grid " "issues in MV."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No MV overloading issues to solve.")
    return curtailment, voltage_dev, rel_load


def curtailment_hvmv_station_overloading(
    edisgo, curtailment, voltage_dev, rel_load, grid_results_dir
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.info(
        "Start curtailment due to overloading issues of the " "HV/MV station."
    )

    hvmv_station = "hvmv_station_{}".format(edisgo.topology.mv_grid)
    rel_load_hvmv_station = rel_load.loc[:, hvmv_station]
    overloading_issues = rel_load_hvmv_station[rel_load_hvmv_station > 1.0].dropna(
        how="all"
    )
    time_steps_issues = overloading_issues.index

    if len(time_steps_issues) > 0:
        pypsa_network = edisgo.to_pypsa(
            mode="mvlv", timesteps=time_steps_issues, use_seed=use_seed
        )

        # save original pypsa network to determine curtailed energy
        pypsa_network_orig = pypsa_network.copy()

        iteration_count = 0
        while len(time_steps_issues) > 0 and iteration_count < max_iterations:

            gens = edisgo.topology.generators_df.index
            loads = edisgo.topology.loads_df.index

            # reduce active and reactive power of loads or generators
            # (depending on whether it is a load or feed-in case)
            pypsa_network = _curtail(
                pypsa_network, gens, loads, time_steps_issues, elia_logger
            )

            # run power flow analysis on all time steps with overloading issues
            pf_results = pypsa_network.pf(time_steps_issues, use_seed=use_seed)

            timesteps_converged = pf_results["converged"][
                pf_results["converged"]["0"]
            ].index
            timesteps_not_converged = pf_results["converged"][
                ~pf_results["converged"]["0"]
            ].index

            # handle converged time steps
            pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
            voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
            voltage_dev = _update_df(voltage_dev, voltage_dev_new)
            rel_load_new = results_helper_functions.relative_load(edisgo)
            rel_load = _update_df(rel_load, rel_load_new)

            # handle not converged time steps
            if len(timesteps_not_converged) > 0:
                curtailment, voltage_dev, rel_load = solve_convergence_issues(
                    edisgo,
                    pypsa_network,
                    timesteps_not_converged,
                    curtailment,
                    voltage_dev,
                    rel_load,
                )

            curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
                pypsa_network_orig, pypsa_network
            )
            elia_logger.debug(
                "Curtailed energy (feed-in/load): {}, {}".format(
                    curtailed_feedin.sum().sum(), curtailed_load.sum().sum()
                )
            )

            # recheck for overloading issues
            rel_load_hvmv_station = rel_load.loc[:, hvmv_station]
            overloading_issues = rel_load_hvmv_station[
                rel_load_hvmv_station > 1.0
            ].dropna(how="all")
            time_steps_issues = overloading_issues.index

            iteration_count += 1

        # overwrite time series in edisgo
        _overwrite_edisgo_timeseries(edisgo, pypsa_network)

        if len(time_steps_issues) > 0:
            edisgo.analyze()
            _save_results_when_curtailment_failed(
                edisgo, grid_results_dir, "hvmv_station_overloading"
            )

            raise ValueError(
                "Curtailment not sufficient to solve grid " "issues at HV/MV station."
            )

        # calculate curtailment
        curtailed_feedin, curtailed_load = _calculate_curtailed_energy(
            pypsa_network_orig, pypsa_network
        )
        curtailment.at["mv_problems", "feed-in"] += curtailed_feedin.sum().sum()
        curtailment.at["mv_problems", "load"] += curtailed_load.sum().sum()

    else:
        elia_logger.debug("No overloading issues at HV/MV station to solve.")
    return curtailment, voltage_dev, rel_load


def curtail_lv_grids(
    edisgo,
    curtailment,
    timesteps_not_converged,
    grid_results_dir,
    voltage_dev,
    rel_load,
):
    elia_logger = logging.getLogger("elia_project: {}".format(edisgo.topology.id))
    elia_logger.debug(
        "Number of time steps with convergence issues: {}".format(
            len(timesteps_not_converged)
        )
    )
    elia_logger.info("Start curtailment of LV grids.")

    # get time steps with overloading and/or voltage issues in LV
    # get time steps with voltage issues
    lv_buses = edisgo.topology.buses_df.lv_feeder.dropna().index
    voltage_dev_lv = voltage_dev.loc[:, lv_buses]
    voltage_issues = (
        voltage_dev_lv[voltage_dev_lv != 0.0]
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    # get time steps with overloading issues
    lv_lines = edisgo.topology.lines_df.lv_feeder.dropna().index
    rel_load_lv = rel_load.loc[:, lv_lines]
    overloading_issues = (
        rel_load_lv[rel_load_lv > 1.0].dropna(how="all").dropna(axis=1, how="all")
    )
    # time steps with issues in the LV
    time_steps_issues_lv = voltage_issues.index.append(
        overloading_issues.index
    ).unique()

    if len(time_steps_issues_lv) == 0:
        elia_logger.debug("No issues in LV grids to solve.")
        return curtailment, voltage_dev, rel_load

    if len(timesteps_not_converged) > 0:
        # get all LV grids
        lv_grids = list(edisgo.topology._grids.keys())
        lv_grids = [lv_grid for lv_grid in lv_grids if "LVGrid" in lv_grid]
        lv_grids = [_.split("_")[-1] for _ in lv_grids]
    else:
        # get LV grids with voltage issues
        lv_grids_voltage_issues = edisgo.topology.buses_df.loc[
            voltage_issues.columns, "lv_grid_id"
        ]
        # get LV grids with overloading issues
        buses_issues = (
            edisgo.topology.lines_df.loc[overloading_issues.columns, ["bus0", "bus1"]]
            .stack()
            .unique()
        )
        lv_grids_overloading_issues = edisgo.topology.buses_df.loc[
            buses_issues, "lv_grid_id"
        ]
        # get set of LV grids with issues
        lv_grids = lv_grids_voltage_issues.append(lv_grids_overloading_issues).unique()

    elia_logger.info("Number of LV grids to curtail: {}".format(len(lv_grids)))

    for lv_grid_id in lv_grids:
        lv_grid_id = int(lv_grid_id)
        elia_logger.debug("Curtailment of LV grid: {}".format(lv_grid_id))

        lv_grid = edisgo.topology._grids["LVGrid_{}".format(lv_grid_id)]
        pypsa_lv = edisgo.to_pypsa(
            mode="lv",
            lv_grid_name=repr(lv_grid),
        )
        # append information on feeder and path length to station
        pypsa_lv.buses = pypsa_lv.buses.join(
            edisgo.topology.buses_df.loc[:, ["lv_feeder", "path_length_to_station"]]
        )
        pypsa_lv.lines = pypsa_lv.lines.join(
            edisgo.topology.lines_df.loc[:, ["lv_feeder"]]
        )

        curtailment, voltage_dev, rel_load = curtailment_lv_voltage_single_lv_grid(
            edisgo,
            pypsa_lv,
            curtailment,
            voltage_dev,
            rel_load,
            grid_results_dir,
            timesteps_not_converged,
        )

        pypsa_lv = edisgo.to_pypsa(
            mode="lv",
            lv_grid_name=repr(lv_grid),
        )
        # append information on feeder and path length to station
        pypsa_lv.buses = pypsa_lv.buses.join(
            edisgo.topology.buses_df.loc[:, ["lv_feeder", "path_length_to_station"]]
        )
        pypsa_lv.lines = pypsa_lv.lines.join(
            edisgo.topology.lines_df.loc[:, ["lv_feeder"]]
        )

        (
            curtailment,
            voltage_dev,
            rel_load,
        ) = curtailment_lv_lines_overloading_single_lv_grid(
            edisgo,
            pypsa_lv,
            curtailment,
            voltage_dev,
            rel_load,
            grid_results_dir,
            timesteps_not_converged,
        )

    elia_logger.info("Grid issues in LV grids solved.")

    # run powerflow for MV grid (including MV/LV stations) to recalculate
    # relative line loading and voltage deviation (only for time steps
    # with curtailment in the LV, including time steps with convergence issues)
    time_steps = time_steps_issues_lv.append(timesteps_not_converged)
    pypsa_network = edisgo.to_pypsa(mode="mvlv", timesteps=time_steps)
    pf_results = pypsa_network.pf(time_steps)
    timesteps_converged = pf_results["converged"][pf_results["converged"]["0"]].index
    timesteps_not_converged = pf_results["converged"][
        ~pf_results["converged"]["0"]
    ].index

    # handle converged time steps
    pypsa_io.process_pfa_results(edisgo, pypsa_network, timesteps_converged)
    voltage_dev_new = results_helper_functions.voltage_diff(edisgo)
    voltage_dev = _update_df(voltage_dev, voltage_dev_new)
    rel_load_new = results_helper_functions.relative_load(edisgo)
    rel_load = _update_df(rel_load, rel_load_new)

    # handle not converged time steps
    if len(timesteps_not_converged) > 0:
        curtailment, voltage_dev, rel_load = solve_convergence_issues(
            edisgo,
            pypsa_network,
            timesteps_not_converged,
            curtailment,
            voltage_dev,
            rel_load,
        )

    return curtailment, voltage_dev, rel_load


def calculate_curtailment(variation, mode="mv_lv"):
    # in case of mode "mv_lv", the low voltage level is considered as well
    # in case of mode "mv" the low voltage level is aggregated at the MV/LV
    # stations' secondary side
    mv_grid_id = variation[0]
    scenario = variation[1]
    # try:
    elia_logger = logging.getLogger("elia_project: {}, {}".format(mv_grid_id, scenario))
    elia_logger.setLevel(logging.DEBUG)

    grid_dir = os.path.join(results_base_path, str(mv_grid_id), scenario)

    if os.path.exists(os.path.join(grid_dir, "curtailment_01percent.csv")):
        print("{} {} already solved.".format(mv_grid_id, scenario))
        return

    # reimport edisgo object
    grid_dir = os.path.join(results_base_path, str(mv_grid_id), scenario)
    edisgo_orig_dir = results_base_path + r"\{}\dumb".format(mv_grid_id)
    edisgo = import_edisgo_object_with_adapted_charging_timeseries(
        edisgo_orig_dir, grid_dir, mv_grid_id, scenario
    )

    # save original time series before curtailment
    feedin_ts = edisgo.timeseries.generators_active_power.copy()
    load_ts = edisgo.timeseries.loads_active_power.copy()
    charging_ts = edisgo.timeseries.charging_points_active_power.copy()

    # assign feeders and path length to station
    assign_feeder(edisgo, mode="mv_feeder")
    assign_feeder(edisgo, mode="lv_feeder")
    get_path_length_to_station(edisgo)

    curtailment = pd.DataFrame(
        data=0.0,
        columns=["feed-in", "load"],
        index=[
            "lv_problems",
            "mv_problems",
            "convergence_issues_mv",
            "convergence_issues_lv",
        ],
    )

    # get time steps with convergence problems
    src = os.path.join(grid_dir, "timesteps_not_converged.csv")
    if os.path.isfile(src):
        timesteps_not_converged = pd.read_csv(src, index_col=1, parse_dates=True).index
    else:
        timesteps_not_converged = []

    # import voltage deviation and relative load
    path = os.path.join(grid_dir, "voltage_deviation.csv")
    voltage_dev = pd.read_csv(path, index_col=0, parse_dates=True)
    path = os.path.join(grid_dir, "relative_load.csv")
    rel_load = pd.read_csv(path, index_col=0, parse_dates=True)

    # get time steps with issues and reduce voltage_dev and rel_load
    # to those time steps
    voltage_issues = (
        voltage_dev[voltage_dev != 0.0].dropna(how="all").dropna(axis=1, how="all")
    )
    overloading_issues = (
        rel_load[rel_load > 1.0].dropna(how="all").dropna(axis=1, how="all")
    )
    time_steps_issues = voltage_issues.index.append(overloading_issues.index).unique()
    voltage_dev = voltage_dev.loc[time_steps_issues, :]
    rel_load = rel_load.loc[time_steps_issues, :]

    i = 0
    remaining_issues = True

    pypsa_mode = {"mv": "mvlv", "mv_lv": None}

    while remaining_issues and i < max_iterations:

        elia_logger.info(
            "Number of time steps with grid issues in round "
            "{}: {}.".format(i, len(time_steps_issues))
        )

        if mode != "mv":
            # curtail LV grids
            curtailment, voltage_dev, rel_load = curtail_lv_grids(
                edisgo,
                curtailment,
                timesteps_not_converged,
                grid_dir,
                voltage_dev,
                rel_load,
            )

        # curtailment MV-LV stations
        curtailment, voltage_dev, rel_load = curtailment_mvlv_stations_overloading(
            edisgo, curtailment, voltage_dev, rel_load, grid_dir
        )
        curtailment, voltage_dev, rel_load = curtailment_mvlv_stations_voltage(
            edisgo, curtailment, voltage_dev, rel_load, grid_dir
        )

        # curtailment MV lines
        # check 10% criterion
        curtailment, voltage_dev, rel_load = curtailment_mv_voltage_10_percent(
            edisgo, curtailment, voltage_dev, rel_load, grid_dir
        )
        curtailment, voltage_dev, rel_load = curtailment_mv_lines_overloading(
            edisgo, curtailment, voltage_dev, rel_load, grid_dir
        )
        curtailment, voltage_dev, rel_load = curtailment_mv_voltage(
            edisgo, curtailment, voltage_dev, rel_load, grid_dir
        )

        # curtailment HV-MV station
        curtailment, voltage_dev, rel_load = curtailment_hvmv_station_overloading(
            edisgo, curtailment, voltage_dev, rel_load, grid_dir
        )

        # save curtailment results
        # curtailment.to_csv(
        #     os.path.join(grid_dir, "curtailment_{}.csv".format(i)))

        # curtailed_feedin = feedin_ts - \
        #                    edisgo.timeseries.generators_active_power
        # curtailed_load = pd.concat(
        #     [(load_ts - edisgo.timeseries.loads_active_power),
        #      (charging_ts -
        #       edisgo.timeseries.charging_points_active_power)],
        #     axis=1)
        # curtailed_feedin.to_csv(
        #     os.path.join(
        #         grid_dir, "curtailment_ts_per_gen_{}.csv".format(i))
        # )
        # curtailed_load.to_csv(
        #     os.path.join(
        #         grid_dir, "curtailment_ts_per_load_{}.csv".format(i))
        # )

        # rerun power flow to check for remaining issues
        elia_logger.info("Check for remaining issues after round {}.".format(i))
        remaining_issues = False
        edisgo.analyze(timesteps=time_steps_issues, mode=pypsa_mode[mode])
        voltage_dev = results_helper_functions.voltage_diff(edisgo)
        rel_load = results_helper_functions.relative_load(edisgo)

        # check if everything was solved
        tol = 1e-4
        voltage_issues = (
            voltage_dev[abs(voltage_dev) > tol]
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )
        v_mag_pu_pfa = edisgo.results.v_res
        voltage_issues_10_percent = (
            v_mag_pu_pfa[(v_mag_pu_pfa > 1.1) | (v_mag_pu_pfa < 0.9)]
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )
        if not voltage_issues.empty or not voltage_issues_10_percent.empty:
            remaining_issues = True
            elia_logger.info("Not all voltage issues solved.")
        else:
            elia_logger.info("Success. All voltage issues solved.")
        tol = 1e-4
        overloading_issues = (
            rel_load[rel_load > 1.0 + tol].dropna(how="all").dropna(axis=1, how="all")
        )
        if not overloading_issues.empty:
            remaining_issues = True
            elia_logger.info("Not all overloading issues solved.")
        else:
            elia_logger.info("Success. All overloading issues solved.")

        # get time steps with issues and reduce voltage_dev and rel_load
        # to those time steps
        time_steps_issues = (
            voltage_issues.index.append(overloading_issues.index)
            .append(voltage_issues_10_percent.index)
            .unique()
        )
        pd.Series(time_steps_issues).to_csv(
            os.path.join(grid_dir, "timesteps_issues_{}.csv".format(i))
        )
        voltage_dev = voltage_dev.loc[time_steps_issues, :]
        rel_load = rel_load.loc[time_steps_issues, :]

        i += 1
        timesteps_not_converged = []

    if remaining_issues:
        elia_logger.info("Not all issues solved.")
        if not voltage_issues.empty:
            voltage_issues.to_csv(
                os.path.join(grid_dir, "remaining_voltage_issues.csv")
            )
        if not voltage_issues_10_percent.empty:
            voltage_issues.to_csv(
                os.path.join(grid_dir, "remaining_voltage_issues_10_percent.csv")
            )
        if not overloading_issues.empty:
            overloading_issues.to_csv(
                os.path.join(grid_dir, "remaining_overloading_issues.csv")
            )
    else:
        elia_logger.info("Success. All issues solved.")

    # save curtailment sums
    curtailment.to_csv(os.path.join(grid_dir, "curtailment_01percent.csv"))

    # save time series
    curtailed_feedin = feedin_ts - edisgo.timeseries.generators_active_power
    curtailed_load = pd.concat(
        [
            (load_ts - edisgo.timeseries.loads_active_power),
            (charging_ts - edisgo.timeseries.charging_points_active_power),
        ],
        axis=1,
    )
    curtailed_feedin.to_csv(
        os.path.join(grid_dir, "curtailment_ts_per_gen_01percent.csv")
    )
    curtailed_load.to_csv(
        os.path.join(grid_dir, "curtailment_ts_per_load_01percent.csv")
    )
    curtailed_feedin.sum(axis=1).to_csv(
        os.path.join(grid_dir, "curtailment_ts_feedin_01percent.csv")
    )
    curtailed_load.sum(axis=1).to_csv(
        os.path.join(grid_dir, "curtailment_ts_demand_01percent.csv")
    )

    # except Exception as e:
    #     print("Error in MV grid {} {}.".format(mv_grid_id, scenario))
    #     traceback.print_exc()


if __name__ == "__main__":
    if num_threads == 1:
        for variation in variations:
            print("Starting curtailment for {} {}".format(variation[0], variation[1]))
            calculate_curtailment(variation)
    else:
        with multiprocessing.Pool(num_threads) as pool:
            pool.map(calculate_curtailment, variations)
            pool.close()
            pool.join()
    print("SUCCESS")
