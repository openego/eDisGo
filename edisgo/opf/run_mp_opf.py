import os
import json
import subprocess
import logging
import numpy as np
from timeit import default_timer as timer

from edisgo.tools.preprocess_pypsa_opf_structure import (
    preprocess_pypsa_opf_structure,
    aggregate_fluct_generators,
)
from edisgo.tools.powermodels_io import (
    to_powermodels,
    convert_storage_series,
    add_storage_from_edisgo,
)

from edisgo.opf.util.scenario_settings import opf_settings

logger = logging.getLogger(__name__)


def convert(o):
    """
    Helper function for json dump, as int64 cannot be dumped.

    """
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def bus_names_to_ints(pypsa_network, bus_names):
    """
    This remaps a list of eDisGo bus names from Strings to Integers.

    Integer indices are needed for the optimization.
    The result uses one-based indexing, as it gets passed on to Julia
    directly.

    Parameters
    -----------
    pypsa_network :
    bus_names : list(str)
        List of bus names to be remapped to indices.

    Returns
    --------
    list(int)
        List of one-based bus indices.

    """

    # map bus name to its integer index
    bus_indices = []
    for name in bus_names:
        bus_indices.append(pypsa_network.buses.index.get_loc(name))

    # Increment each Python index by one, as Julia uses one-based indexing
    bus_indices = [i + 1 for i in bus_indices]

    return bus_indices


def run_mp_opf(edisgo_network, timesteps=None, storage_series=[], **kwargs):
    """

    Parameters
    ----------
    edisgo_network :
    timesteps: `pandas.DatetimeIndex<DatetimeIndex>` or `pandas.Timestamp<Timestamp>`
    **kwargs :
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
        "storage_series": [],
        # Time series for storage operation required by upper grid layer
        "curtailment_requirement": False,
        # List of total curtailment for each time step, len(list)== "time_horizon"
        "curtailment_requirement_series": [],
        # An overall allowance of curtailment is considered
        "curtailment_allowance": False,
        # Maximal allowed curtailment over entire time horizon,
        # DEFAULT: "3percent"=> 3% of total RES generation in time horizon may be curtailed, else: Float
        "curtailment_total": "3percent",
        "results_path": "opf_solutions"
        # path to where OPF results are stored

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
    mode = "mv"
    # read settings from kwargs
    settings = opf_settings()
    settings["time_horizon"] = len(timesteps)

    # convert edisgo network to pypsa network for timesteps on MV-level
    # aggregate all loads and generators in LV-grids
    # TODO check aggregation
    logger.debug("converting to pypsa_mv")
    pypsa_mv = edisgo_network.to_pypsa(
        mode=mode,
        # aggregate_loads="all",
        # aggregate_generators="all",
        timesteps=timesteps,
    )
    # adapt allowed s_nom for load case
    if kwargs.get("load_case", False):
        pypsa_mv.lines.loc[:, "s_nom"] = pypsa_mv.lines.loc[:, "s_nom"] * 0.5
    timehorizon = len(pypsa_mv.snapshots)
    # set name of pypsa network
    pypsa_mv.name = "ding0_{}_t_{}".format(
        edisgo_network.topology.id, timehorizon
    )

    # Remap storage bus names to Integers, if any
    if "storage_buses" in kwargs:
        bus_names = kwargs["storage_buses"]
        bus_indices = bus_names_to_ints(pypsa_mv, bus_names)
        kwargs["storage_buses"] = bus_indices

    for args in kwargs.items():
        if args[0] in settings:
            # if hasattr(settings,args[0]):
            settings[args[0]] = args[1]

    # preprocess pypsa structure
    logger.debug("preprocessing pypsa structure for opf")
    preprocess_pypsa_opf_structure(edisgo_network, pypsa_mv, hvmv_trafo=False)
    aggregate_fluct_generators(pypsa_mv)
    # convert pypsa structure to network dictionary and create dictionaries for time series of loads and generators
    pm, load_data, gen_data = to_powermodels(pypsa_mv)
    storage_data = convert_storage_series(storage_series)

    # Export eDisGo storage only for operation only as they would interfere with positioning
    if settings["storage_operation_only"]:
        add_storage_from_edisgo(edisgo_network, pypsa_mv, pm)

    # dump json files for static network information, timeseries of loads and generators, and opf settings
    with open(
        os.path.join(scenario_data_dir, "{}_static.json".format(pm["name"])),
        "w",
    ) as outfile:
        json.dump(pm, outfile, default=convert)
    with open(
        os.path.join(scenario_data_dir, "{}_loads.json".format(pm["name"])),
        "w",
    ) as outfile:
        json.dump(load_data, outfile, default=convert)
    with open(
        os.path.join(scenario_data_dir, "{}_gens.json".format(pm["name"])), "w"
    ) as outfile:
        json.dump(gen_data, outfile, default=convert)
    with open(
        os.path.join(scenario_data_dir, "{}_storage.json".format(pm["name"])),
        "w",
    ) as outfile:
        json.dump(storage_data, outfile, default=convert)
    with open(
        os.path.join(
            scenario_data_dir, "{}_opf_setting.json".format(pm["name"])
        ),
        "w",
    ) as outfile:
        json.dump(settings, outfile, default=convert)

    logger.info("starting julia process")
    start = timer()
    solution_dir = kwargs.get(
        "results_path", os.path.join(opf_dir, "opf_solutions")
    )
    julia_process = subprocess.run(
        [
            "julia",
            "--project={}".format(julia_env_dir),
            os.path.join(opf_dir, "optimization_evaluation.jl"),
            opf_dir,
            pm["name"],
            solution_dir,
        ]
    )
    end = timer()
    run_time = end - start
    logger.info("julia terminated after {} s".format(run_time))

    if julia_process.returncode != 0:
        raise RuntimeError("Julia subprocess failed.")

    solution_file = "{}_{}_{}_opf_sol.json".format(
        pm["name"], settings["scenario"], settings["relaxation"]
    )

    # opf_results = OPFResults()
    edisgo_network.opf_results.set_solution(
        solution_name=os.path.join(solution_dir, solution_file),
        pypsa_net=pypsa_mv,
    )

    if edisgo_network.opf_results.status != "Optimal":
        raise RuntimeError("Optimal solution not found.")

    return edisgo_network.opf_results.status
