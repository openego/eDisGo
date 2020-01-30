import os
import json
import subprocess
import logging
from timeit import default_timer as timer

from edisgo.tools.preprocess_pypsa_opf_structure import preprocess_pypsa_opf_structure, aggregate_fluct_generators
from edisgo.tools.powermodels_io import to_powermodels
from edisgo.opf.util.scenario_settings import opf_settings

logger = logging.getLogger(__name__)

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
    logger.debug(julia_env_dir)
    logger.debug(scenario_data_dir)
    # solution_dir = os.path.join(opf_dir, "opf_solutions/")

                    # set path to edisgoOPF folder for scenario data and julia module relative to this file
                    # abspath = os.path.dirname(os.path.abspath(__file__))
                    # opf_dir = os.path.join(abspath, "edisgoOPF/")
                    # scenario_data_dir = os.path.join(opf_dir, "edisgo_scenario_data")
                    # set julia env path
                    # julia_env_dir = os.path.join(opf_dir, "edisgoOPF/")

    if timesteps is None:
        # TODO worst case snapshot analysis
        logger.error("TODO implement worst case snapshots")
        raise ValueError("Need to specify timesteps for multiperiod opf")


    # only mv mode possible
    mode="mv"
    # read settings from kwargs
    settings = opf_settings()
    settings["time_horizon"] = len(timesteps)
    for args in kwargs.items():
        if args[0] in settings:
            # if hasattr(settings,args[0]):
            settings[args[0]] = args[1]

    # convert edisgo network to pypsa network for timesteps on MV-level
    # aggregate all loads and generators in LV-grids
    # TODO check aggregation
    logger.debug("converting to pypsa_mv")
    pypsa_mv = edisgo_network.to_pypsa(mode=mode,
                                       # aggregate_loads="all",
                                       # aggregate_generators="all",
                                       timesteps=timesteps)
    timehorizon = len(pypsa_mv.snapshots)
    # set name of pypsa network
    pypsa_mv.name = "ding0_{}_t_{}".format(edisgo_network.topology.id,timehorizon)

    # preprocess pypsa structure
    logger.debug("preprocessing pypsa structure for opf")
    preprocess_pypsa_opf_structure(edisgo_network, pypsa_mv, hvmv_trafo=False)
    aggregate_fluct_generators(pypsa_mv)
    # convert pypsa structure to network dictionary and create dictionaries for time series of loads and generators
    pm, load_data, gen_data = to_powermodels(pypsa_mv)


    # dump json files for static network information, timeseries of loads and generators, and opf settings
    with open(os.path.join(scenario_data_dir, "{}_static.json".format(pm["name"])), 'w') as outfile:
        json.dump(pm, outfile)
    with open(os.path.join(scenario_data_dir, "{}_loads.json".format(pm["name"])), 'w') as outfile:
        json.dump(load_data, outfile)
    with open(os.path.join(scenario_data_dir, "{}_gens.json".format(pm["name"])), 'w') as outfile:
        json.dump(gen_data, outfile)
    with open(os.path.join(scenario_data_dir, "{}_opf_setting.json".format(pm["name"])), 'w') as outfile:
        json.dump(settings, outfile)
    logger.info("starting julia process")
    start = timer()
    julia_process = subprocess.run(['julia', '--project={}'.format(julia_env_dir),
                    os.path.join(opf_dir, 'optimization_evaluation.jl'),
                    opf_dir, pm["name"]])
    end = timer()
    run_time = end-start
    logger.info("julia terminated after {} s".format(run_time))

    if julia_process.returncode != 0:
        raise RuntimeError("Julia Subprocess failed")

    solution_dir = os.path.join(opf_dir, "opf_solutions")
    solution_file = "{}_{}_{}_opf_sol.json".format(pm["name"],settings["scenario"],settings["relaxation"])

    # opf_results = OPFResults()
    edisgo_network.opf_results.set_solution(
        solution_name=os.path.join(solution_dir,solution_file),
        pypsa_net=pypsa_mv)

    return edisgo_network.opf_results.status #opf_results
