import os
import pandas as pd
import numpy as np
from edisgo import EDisGo
from edisgo.tools.tools import select_worstcase_snapshots
from edisgo.tools.preprocess_pypsa_opf_structure import preprocess_pypsa_opf_structure, aggregate_fluct_generators
from edisgo.tools.powermodels_io import to_powermodels
import json
from edisgo.opf.util.scenario_settings import opf_settings
import subprocess
from timeit import default_timer as timer
from edisgo.opf.results.opf_result_class import OPFResults


def run_mp_opf(edisgo_network,timesteps=None,**kwargs):
    """
    :param edisgo_network:
    :param timesteps: `pandas.DatetimeIndex<datetimeindex>` or `pandas.Timestamp<timestamp>`
    :param **kwargs:
        "scenario" : "nep"
        # objective function
        "objective": "nep",
        # chosen relaxation
        "relaxation": "none",
        # upper bound on network expansion
        "max_exp": 10,
        # number of time steps considered in optimization
        "time_horizon": 2,
        # length of time step in hours
        "time_elapsed": 1.0,
        # storage units are considered
        "storage_units": False,
        # positioning of storage units, if empty list, all buses are potential positions of storage units and
        # capacity is optimized
        "storage_buses": [],
        # total storage capacity in the network
        "total_storage_capacity": 0.0,
        # Requirements for curtailment in every time step is considered
        "curtailment_requirement": False,
        # List of total curtailment for each time step, len(list)== "time_horizon"
        "curtailment_requirement_series": [],
        # An overall allowance of curtailment is considered
        "curtailment_allowance": False,
        # Maximal allowed curtailment over entire time horizon,
        # DEFAULT: "3percent"=> 3% of total RES generation in time horizon may be curtailed, else: Float
        "curtailment_total": "3percent",
    :return:
    """
    opf_dir = os.path.dirname(os.path.abspath(__file__))
    julia_env_dir = os.path.join(opf_dir, "edisgoOPF")

    scenario_data_dir = os.path.join(opf_dir, "edisgo_scenario_data")
    print(julia_env_dir)
    print(scenario_data_dir)
    # solution_dir = os.path.join(opf_dir, "opf_solutions/")

                    # set path to edisgoOPF folder for scenario data and julia module relative to this file
                    # abspath = os.path.dirname(os.path.abspath(__file__))
                    # opf_dir = os.path.join(abspath, "edisgoOPF/")
                    # scenario_data_dir = os.path.join(opf_dir, "edisgo_scenario_data")
                    # set julia env path
                    # julia_env_dir = os.path.join(opf_dir, "edisgoOPF/")

    if timesteps is None:
        # TODO worst case snapshot analysis
        print("TODO implement worst case snapshots")
        return
    # convert edisgo network to pypsa network for timesteps on MV-level
    # aggregate all loads and generators in LV-grids
    # TODO check aggregation
    print("convert to pypsa_mv")
    pypsa_mv = edisgo_network.to_pypsa(mode="mv",
                                       # aggregate_loads="all",
                                       # aggregate_generators="all",
                                       timesteps=timesteps)
    timehorizon = len(pypsa_mv.snapshots)
    # set name of pypsa network
    pypsa_mv.name = "ding0_{}_t_{}".format(edisgo_network.topology.id,timehorizon)

    # preprocess pypsa structure
    print("preprocsee pypsa structure for opf")
    preprocess_pypsa_opf_structure(edisgo_network, pypsa_mv, hvmv_trafo=False)
    aggregate_fluct_generators(pypsa_mv)
    # convert pypsa structure to network dictionary and create dictionaries for time series of loads and generators
    pm, load_data, gen_data = to_powermodels(pypsa_mv)

    # read settings from kwargs
    settings = opf_settings()
    settings["time_horizon"] = timehorizon
    for args in kwargs.items():
        if hasattr(settings,args[0]):
            settings[args[0]] = args[1]

    # dump json files for static network information, timeseries of loads and generators, and opf settings
    with open("{}/{}_static.json".format(scenario_data_dir, pm["name"]), 'w') as outfile:
        json.dump(pm, outfile)
    with open("{}/{}_loads.json".format(scenario_data_dir, pm["name"]), 'w') as outfile:
        json.dump(load_data, outfile)
    with open("{}/{}_gens.json".format(scenario_data_dir, pm["name"]), 'w') as outfile:
        json.dump(gen_data, outfile)
    with open("{}/{}_opf_setting.json".format(scenario_data_dir, pm["name"]), 'w') as outfile:
        json.dump(settings, outfile)
    print("start julia process")
    start = timer()
    subprocess.run(['julia', '--project={}'.format(julia_env_dir),
                    '{}/optimization_evaluation.jl'.format(opf_dir), opf_dir, pm["name"]])
    end = timer()
    run_time = end-start
    print("julia terminated after {} s".format(run_time))

    solution_dir = os.path.join(opf_dir, "opf_solutions")
    solution_file = "{}_{}_{}_opf_sol.json".format(pm["name"],settings["scenario"],settings["relaxation"])

    opf_results = OPFResults()
    opf_results.set_solution(solution_name="{}/{}".format(solution_dir,solution_file),
                             pypsa_net=pypsa_mv)

    return opf_results
