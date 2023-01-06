# Methods to perform linearised DistFlow
import itertools
import logging

from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
import pyomo.environ as pm

from pyomo.opt import SolverStatus, TerminationCondition

from edisgo.tools.tools import (
    calculate_impedance_for_parallel_components,
    get_nodal_residual_load,
)

logger = logging.getLogger(__name__)

BANDS = ["upper_power", "upper_energy", "lower_energy"]


def import_flexibility_bands(dir, use_cases):
    flexibility_bands = {}

    for band in BANDS:
        band_df = pd.DataFrame()
        for use_case in use_cases:
            flexibility_bands_tmp = pd.read_csv(
                dir + "/{}_{}.csv".format(band, use_case),
                index_col=0,
                parse_dates=True,
                dtype=np.float32,
            )
            band_df = pd.concat([band_df, flexibility_bands_tmp], axis=1)
        if band_df.columns.duplicated().any():
            raise ValueError(
                "Charging points with the same name in flexibility bands. "
                "Please check"
            )
        flexibility_bands[band] = band_df
        # remove numeric problems
        if "upper" in band:
            flexibility_bands[band] = flexibility_bands[band] + 1e-6
        elif "lower" in band:
            flexibility_bands[band] = flexibility_bands[band] - 1e-6
    return flexibility_bands


def prepare_time_invariant_parameters(
    edisgo_obj,
    downstream_nodes_matrix,
    **kwargs,
):
    """
    Prepare parameters that do not change within the iterations of the
    rolling horizon approach. These include topological parameters and
    timeseries of the inflexible units which are not influenced by the
    optimisation.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object
    downstream_nodes_matrix : pd.DataFrame
        Matrix describing the mutual dependencies of the nodes
    kwargs : (default: False)
        per_unit : bool
        optimize_bess : bool
        optimize_emob : bool
        optimize_hp : bool
        flexible_loads : pd.DataFrame
            DataFrame containing ids of all flexible loads

    Returns
    -------
    dict
        Containing all necessary information for the optimization and
        especially the time invariant parameters

    """
    t1 = perf_counter()
    fixed_parameters = {}
    # set grid and edisgo objects as well as slack
    # TODO is this necesary?
    (
        fixed_parameters["edisgo_object"],
        fixed_parameters["grid_object"],
        fixed_parameters["slack"],
    ) = setup_grid_object(edisgo_obj)

    fixed_parameters["downstream_nodes_matrix"] = downstream_nodes_matrix
    fixed_parameters["optimize_emob"] = kwargs.get("optimize_emob", False)
    fixed_parameters["optimize_bess"] = kwargs.get("optimize_bess", False)
    fixed_parameters["optimize_hp"] = kwargs.get("optimize_hp", False)
    fixed_parameters["per_unit"] = kwargs.get("per_unit", False)
    fixed_parameters["flexible_loads"] = kwargs.get("flexible_loads", pd.DataFrame())

    if fixed_parameters["optimize_bess"]:
        if not fixed_parameters["flexible_loads"].empty:
            fixed_parameters["optimized_storage_units"] = (
                fixed_parameters["flexible_loads"]
                .loc[fixed_parameters["flexible_loads"]["type"] == "storage"]
                .index
            )
            fixed_parameters["inflexible_storage_units"] = fixed_parameters[
                "grid_object"
            ].storage_units_df.index.drop(fixed_parameters["optimized_storage_units"])
        else:
            fixed_parameters["optimized_storage_units"] = kwargs.get(
                "flexible_storage_units",
                fixed_parameters["grid_object"].storage_units_df.index,
            )
        fixed_parameters["inflexible_storage_units"] = fixed_parameters[
            "grid_object"
        ].storage_units_df.index.drop(fixed_parameters["optimized_storage_units"])

        # If no storage available set optimization to false
        if fixed_parameters["optimized_storage_units"].empty:
            fixed_parameters["optimize_bess"] = False
    else:
        # Add empty list to later define inflexible loads
        fixed_parameters["optimized_storage_units"] = []

    if fixed_parameters["optimize_emob"]:
        # fixed_parameters["ev_flex_bands"] = kwargs.get("ev_flex_bands")
        fixed_parameters["ev_flex_bands"] = edisgo_obj.electromobility.flexibility_bands

        if any([i.empty for i in fixed_parameters["ev_flex_bands"].values()]):
            # TODO check if necessary to pass empty list
            fixed_parameters["optimized_charging_points"] = []
            fixed_parameters["optimize_emob"] = False
            logger.info("Emob optimization is set to False as flex bands empty.")
        else:
            if not fixed_parameters["flexible_loads"].empty:
                fixed_parameters["optimized_charging_points"] = (
                    fixed_parameters["flexible_loads"]
                    .loc[fixed_parameters["flexible_loads"]["type"] == "charging_point"]
                    .index
                )
            else:
                fixed_parameters["optimized_charging_points"] = fixed_parameters[
                    "ev_flex_bands"
                ]["upper_power"].columns

                fixed_parameters["inflexible_charging_points"] = fixed_parameters[
                    "grid_object"
                ].charging_points_df.index.drop(
                    fixed_parameters["optimized_charging_points"]
                )
    else:
        # Add empty list to later define inflexible loads
        fixed_parameters["optimized_charging_points"] = []

    if fixed_parameters["optimize_hp"]:

        if not fixed_parameters["flexible_loads"].empty:
            fixed_parameters["optimized_heat_pumps"] = (
                fixed_parameters["flexible_loads"]
                .loc[fixed_parameters["flexible_loads"]["type"] == "heat_pump"]
                .index
            )
        else:
            fixed_parameters["optimized_heat_pumps"] = kwargs.get(
                "optimized_heat_pumps", edisgo_obj.heat_pump.heat_demand_df.index
            )
        fixed_parameters["heat_pumps"] = fixed_parameters["grid_object"].loads_df.loc[
            fixed_parameters["grid_object"].loads_df.type == "heat_pump"
        ]

        fixed_parameters["cop"] = edisgo_obj.heat_pump.cop_df
        fixed_parameters["heat_demand"] = edisgo_obj.heat_pump.heat_demand_df
        fixed_parameters["tes"] = edisgo_obj.heat_pump.thermal_storage_units_df

        if fixed_parameters["optimized_heat_pumps"].empty:
            fixed_parameters["optimize_hp"] = False
            logger.info("HP optimization is set to False as optimized hps empty.")
    else:
        # Add empty list to later define inflexible loads
        fixed_parameters["optimized_heat_pumps"] = []

    # save non flexible loads
    if not fixed_parameters["flexible_loads"].empty:
        fixed_parameters["inflexible_loads"] = fixed_parameters[
            "grid_object"
        ].loads_df.index.drop(fixed_parameters["flexible_loads"].index)
    else:
        fixed_parameters["inflexible_loads"] = (
            fixed_parameters["grid_object"]
            .loads_df.index.drop(fixed_parameters["optimized_charging_points"])
            .drop(fixed_parameters["optimized_heat_pumps"])
        )

    # extract residual load of non optimised components
    fixed_parameters[
        "res_load_inflexible_units"
    ] = get_residual_load_of_not_optimized_components(
        grid=fixed_parameters["grid_object"],
        edisgo_obj=fixed_parameters["edisgo_object"],
        # TODO add relevant storages/generators
        relevant_storage_units=fixed_parameters.get("inflexible_storage_units", None),
        # relevant_generators=None,
        relevant_loads=fixed_parameters["inflexible_loads"],
    )
    # get nodal active and reactive powers of non optimised components
    # Todo: add handling of storage and hp once become relevant
    (
        nodal_active_power,
        nodal_reactive_power,
        nodal_active_load,
        nodal_reactive_load,
        nodal_active_generation,
        nodal_reactive_generation,
        nodal_active_storage,
        nodal_reactive_storage,
    ) = get_nodal_residual_load(
        grid=fixed_parameters["grid_object"],
        edisgo=fixed_parameters["edisgo_object"],
        considered_storage=fixed_parameters.get(
            "inflexible_storage_units",
            fixed_parameters["grid_object"].storage_units_df.index,
        ),
        considered_loads=fixed_parameters["inflexible_loads"],
        # considered_generators=
    )
    fixed_parameters["nodal_active_power"] = nodal_active_power.T
    fixed_parameters["nodal_reactive_power"] = nodal_reactive_power.T
    fixed_parameters["nodal_active_load"] = nodal_active_load.T
    fixed_parameters["nodal_reactive_load"] = nodal_reactive_load.T
    fixed_parameters["nodal_active_feedin"] = nodal_active_generation.T
    fixed_parameters["nodal_reactive_feedin"] = nodal_reactive_generation.T
    fixed_parameters["tan_phi_load"] = (
        nodal_reactive_load.divide(nodal_active_load)
    ).fillna(0)
    fixed_parameters["tan_phi_feedin"] = (
        nodal_reactive_generation.divide(nodal_active_generation)
    ).fillna(0)
    # get underlying branch elements and power factors
    # handle pu conversion
    if fixed_parameters["per_unit"]:
        logger.info(
            "Optimisation in pu-system. Make sure the inserted energy "
            "bands are also converted to the same pu-system."
        )
        fixed_parameters["v_nom"] = 1.0
        s_base = kwargs.get("s_base", 1)
        fixed_parameters["grid_object"].convert_to_pu_system(
            s_base, timeseries_inplace=True
        )
        fixed_parameters["pars"] = {
            "r": "r_pu",
            "x": "x_pu",
            "s_nom": "s_nom_pu",
            "p_nom": "p_nom_pu",
            "peak_load": "peak_load_pu",
            "capacity": "capacity_pu",
        }
    else:
        fixed_parameters["v_nom"] = fixed_parameters["grid_object"].buses_df.v_nom.iloc[
            0
        ]
        fixed_parameters["pars"] = {
            "r": "r",
            "x": "x",
            "s_nom": "s_nom",
            "p_nom": "p_nom",
            "peak_load": "peak_load",
            "capacity": "capacity",
        }
        fixed_parameters["grid_object"].transformers_df["r"] = (
            fixed_parameters["grid_object"].transformers_df["r_pu"]
            * np.square(fixed_parameters["v_nom"])
            / fixed_parameters["grid_object"].transformers_df.s_nom
        )
        fixed_parameters["grid_object"].transformers_df["x"] = (
            fixed_parameters["grid_object"].transformers_df["x_pu"]
            * np.square(fixed_parameters["v_nom"])
            / fixed_parameters["grid_object"].transformers_df.s_nom
        )
    fixed_parameters["branches"] = concat_parallel_branch_elements(
        fixed_parameters["grid_object"]
    )
    (
        fixed_parameters["underlying_branch_elements"],
        fixed_parameters["power_factors"],
    ) = get_underlying_elements(
        fixed_parameters
    )  # Todo: time invariant
    logger.info(
        f"It took {perf_counter() - t1} seconds to extract timeinvariant "
        f"parameters."
    )

    return fixed_parameters


def setup_model(
    fixed_parameters,
    timesteps,
    objective="curtailment",
    **kwargs,
):
    """
    Method to set up pyomo model for optimisation of storage procurement
    and/or ev charging and/or heat pumps including thermal storage with linear
    approximation of power flow from eDisGo-object.

    Parameters
    ----------
    fixed_parameters : dict
        All fixed parameters which are needed to setup the model
    timesteps :
        Number of timesteps for which the model is set up
    objective :
        Optimization objectives. Possible keys: "curtailment", "peak_load",
        "residual_load", "minimize_energy_level", "maximize_energy_level",
        "minimize_loading",
    kwargs :
        name :
            Name of the model (default: optimization objective)
        overlap_iterations :
            Number of timesteps which the iteration windows overlap
            (default: None)
        charging_efficiency: (default: 0.9)
            Depends on Simbev data

        energy_level_starts : dict('ev':pd.Series, 'tes':pd.Series)
            initial energy level of component (first timestep)
        charging_starts : dict('ev':pd.Series, 'tes':pd.Series, 'hp':pd.Series)
            initial charging value of component (first timestep)

        energy_level_end_ev : (default: None)
        energy_level_end_hp : (default: None)
        energy_level_beginning_ev : (default: None)
            wurde eingeführt um Differenz zu referenzladen zu beheben
            letzter Iterationsschritt auf mean energyband if True
        energy_level_beginning_hp : (default: None)
            wurde eingeführt um Differenz zu referenzladen zu beheben

        fix_relative_soc : (default: 0.5)
            Wärmespeicher, bei Bander vermutlich mean
        delta_min : (default: 0.9)
        delta_max : (default: 0.1)
        v_min : (default: 0.9)
        v_max : (default: 1.1)
            Spannungsgrenzwerte
        thermal_limit : (default: 1.0)
            für Lines und Transformer
        v_slack : (default: fixed_parameters["v_nom"]
            Spannungsnennwert am Umspannwerk
        load_factor_rings : (default: 1.0)
            n-1 kriterium bei 0.5
        print_model : (default: False)


    Returns
    -------

    """

    t1 = perf_counter()
    model = pm.ConcreteModel()
    model.name = kwargs.get("name", objective)
    # check if correct value of objective is inserted
    if objective not in [
        "curtailment",
        "peak_load",
        "minimize_energy_level",
        "residual_load",
        "maximize_energy_level",
        "minimize_loading",
    ]:
        raise ValueError("The objective you inserted is not implemented yet.")

    # DEFINE SETS AND FIX PARAMETERS
    logger.info("Setup model: Defining sets and parameters.")
    model.time_set = pm.RangeSet(0, len(timesteps) - 1)
    model.time_zero = [model.time_set.at(1)]
    overlap_interations = kwargs.get("overlap_interations", None)
    if overlap_interations is not None:
        model.time_end = [model.time_set.at(-overlap_interations)]
    else:
        model.time_final = [model.time_set.at(-1)]
        model.time_end = [model.time_set.at(-1)]
    model.time_non_zero = model.time_set - [model.time_set.at(1)]
    model.times_fixed_soc = pm.Set(
        initialize=[model.time_set.at(1), model.time_set.at(-1)]
    )
    model.timeindex = pm.Param(
        model.time_set,
        initialize={i: timesteps[i] for i in model.time_set},
        within=pm.Any,
        mutable=True,
    )
    model.time_increment = pd.infer_freq(timesteps)
    if not any(char.isdigit() for char in model.time_increment):
        model.time_increment = "1" + model.time_increment

    if fixed_parameters["optimize_bess"]:
        model.storage_set = pm.Set(
            initialize=fixed_parameters["grid_object"].storage_units_df.index
        )
        model.optimized_storage_set = pm.Set(
            initialize=fixed_parameters["optimized_storage_units"]
        )
        model.fixed_storage_set = model.storage_set - model.optimized_storage_set
        model.fix_relative_soc = kwargs.get("fix_relative_soc", 0.5)

    res_load = {
        i: fixed_parameters["res_load_inflexible_units"][model.timeindex[i]]
        for i in model.time_set
    }
    model.residual_load = pm.Param(model.time_set, initialize=res_load, mutable=True)

    if objective == "peak_load":
        model.delta_min = kwargs.get("delta_min", 0.9)
        model.delta_max = kwargs.get("delta_max", 0.1)
        model.min_load_factor = pm.Var()
        model.max_load_factor = pm.Var()
    elif objective == "minimize_energy_level" or objective == "maximize_energy_level":
        model.grid_power_flexible = pm.Var(model.time_set)

    # DEFINE VARIABLES
    if fixed_parameters["optimize_bess"]:
        model.soc = pm.Var(
            model.optimized_storage_set,
            model.time_set,
            bounds=lambda m, b, t: (
                0,
                m.grid.storage_units_df.loc[b, model.pars["capacity"]],
            ),
        )
        model.charging = pm.Var(
            model.optimized_storage_set,
            model.time_set,
            bounds=lambda m, b, t: (
                -m.grid.storage_units_df.loc[b, model.pars["p_nom"]],
                m.grid.storage_units_df.loc[b, model.pars["p_nom"]],
            ),
        )

    if fixed_parameters["optimize_emob"]:
        # TODO check for EV in flex loads
        logger.info("Setup model: Adding EV model.")
        model = add_ev_model_bands(
            model=model,
            fixed_parameters=fixed_parameters,
            charging_efficiency=fixed_parameters[
                "edisgo_object"
            ].electromobility.eta_charging_points,
            # energy_level_start=kwargs.get("energy_level_start_ev", None),
            energy_level_starts=kwargs.get("energy_level_start", {"ev": None}),
            energy_level_end=kwargs.get("energy_level_end_ev", None),
            energy_level_beginning=kwargs.get("energy_level_beginning_ev", None),
            # charging_start=kwargs.get("charging_start_ev", None),
            charging_starts=kwargs.get("charging_starts", {"ev": None}),
        )

    if fixed_parameters["optimize_hp"]:
        logger.info("Setup model: Adding HP model.")
        model = add_heat_pump_model(
            model=model,
            fixed_parameters=fixed_parameters,
            # energy_level_start=kwargs.get("energy_level_start_hp", None),
            energy_level_starts=kwargs.get("energy_level_start", {"tes": None}),
            energy_level_end=kwargs.get("energy_level_end_hp", None),
            energy_level_beginning=kwargs.get("energy_level_beginning_hp", None),
            charging_starts=kwargs.get("charging_starts", {"hp": None, "tes": None}),
        )

    if not objective == "minimize_energy_level" or objective == "maximize_energy_level":
        logger.info("Setup model: Adding grid model.")
        model = add_grid_model_lopf(
            model=model,
            fixed_parameters=fixed_parameters,
            timesteps=timesteps,
            edisgo_obj=fixed_parameters["edisgo_object"],
            grid_object=fixed_parameters["grid_object"],
            slack=fixed_parameters["slack"],
            v_min=kwargs.get("v_min", 0.9),
            v_max=kwargs.get("v_max", 1.1),
            thermal_limits=kwargs.get("thermal_limit", 1.0),
            v_slack=kwargs.get("v_slack", fixed_parameters["v_nom"]),
            load_factor_rings=kwargs.get("load_factor_rings", 1.0),
            # 0.5 Todo: change to edisgo.config["grid_expansion_load_factors"]
            #  ["mv_load_case_line"]?
        )

    # DEFINE CONSTRAINTS

    if fixed_parameters["optimize_bess"]:
        model.BatteryCharging = pm.Constraint(
            model.storage_set, model.time_non_zero, rule=soc
        )
        model.FixedSOC = pm.Constraint(
            model.storage_set, model.times_fixed_soc, rule=fix_soc
        )

    if objective == "minimize_energy_level" or objective == "maximize_energy_level":
        model.AggrGrid = pm.Constraint(model.time_set, rule=aggregated_power)

    # DEFINE OBJECTIVE
    logger.info("Setup model: Setting objective.")
    if objective == "peak_load":
        model.LoadFactorMin = pm.Constraint(model.time_set, rule=load_factor_min)
        model.LoadFactorMax = pm.Constraint(model.time_set, rule=load_factor_max)
        model.objective = pm.Objective(
            rule=minimize_max_residual_load,
            sense=pm.minimize,
            doc="Define objective function",
        )
    elif objective == "curtailment":
        model.objective = pm.Objective(
            rule=minimize_curtailment,
            sense=pm.minimize,
            doc="Define objective function",
        )
    elif objective == "minimize_energy_level":
        model.objective = pm.Objective(
            rule=minimize_energy_level,
            sense=pm.minimize,
            doc="Define objective function",
        )
    elif objective == "maximize_energy_level":
        model.objective = pm.Objective(
            rule=maximize_energy_level,
            sense=pm.minimize,
            doc="Define objective function",
        )
    elif objective == "residual_load":
        model.grid_residual_load = pm.Var(model.time_set)
        model.GridResidualLoad = pm.Constraint(model.time_set, rule=grid_residual_load)
        model.objective = pm.Objective(
            rule=minimize_residual_load,
            sense=pm.minimize,
            doc="Define objective function",
        )
    elif objective == "minimize_loading":
        model.objective = pm.Objective(
            rule=minimize_loading, sense=pm.minimize, doc="Define objective function"
        )
    else:
        raise Exception("Unknown objective.")

    if kwargs.get("print_model", False):
        model.pprint()
    logger.info("Successfully set up optimisation model.")
    logger.info(f"It took {perf_counter() - t1} seconds to set up model.")
    return model


def add_grid_model_lopf(
    model,
    fixed_parameters,
    timesteps,
    edisgo_obj,
    grid_object,
    slack,
    v_min,
    v_max,
    thermal_limits,
    v_slack,
    load_factor_rings,
):
    """
    Method to add sets variables and constraints for including a representation of the
    grid with a linearised power flow under omission of losses. Only applicable to
    radial networks.
    # Todo: add docstrings
    """

    def init_active_nodal_power(model, bus, time):
        return (
            fixed_parameters["nodal_active_power"].T.loc[model.timeindex[time]].loc[bus]
        )

    def init_reactive_nodal_power(model, bus, time):
        return (
            fixed_parameters["nodal_reactive_power"]
            .T.loc[model.timeindex[time]]
            .loc[bus]
        )

    def init_active_nodal_load(model, bus, time):
        return (
            fixed_parameters["nodal_active_load"].T.loc[model.timeindex[time]].loc[bus]
        )

    def init_reactive_nodal_load(model, bus, time):
        return (
            fixed_parameters["nodal_reactive_load"]
            .T.loc[model.timeindex[time]]
            .loc[bus]
        )

    def init_active_nodal_feedin(model, bus, time):
        return (
            fixed_parameters["nodal_active_feedin"]
            .T.loc[model.timeindex[time]]
            .loc[bus]
        )

    def init_reactive_nodal_feedin(model, bus, time):
        return (
            fixed_parameters["nodal_reactive_feedin"]
            .T.loc[model.timeindex[time]]
            .loc[bus]
        )

    def init_power_factors(model, branch, time):
        return fixed_parameters["power_factors"].loc[branch, model.timeindex[time]]

    # check if multiple voltage levels are present
    if len(grid_object.buses_df.v_nom.unique()) > 1:
        logger.info(
            "More than one voltage level included. Please make sure to "
            "adapt all impedance values to one reference system."
        )
    # Sets and parameters
    model.bus_set = pm.Set(initialize=grid_object.buses_df.index)
    model.slack_bus = pm.Set(initialize=slack)
    model.v_min = v_min
    model.v_max = v_max

    model.v_nom = fixed_parameters["v_nom"]
    model.thermal_limit = thermal_limits
    model.pars = fixed_parameters["pars"]
    model.grid = grid_object
    model.downstream_nodes_matrix = fixed_parameters["downstream_nodes_matrix"]
    model.nodal_active_power = pm.Param(
        model.bus_set, model.time_set, initialize=init_active_nodal_power, mutable=True
    )
    model.nodal_reactive_power = pm.Param(
        model.bus_set,
        model.time_set,
        initialize=init_reactive_nodal_power,
        mutable=True,
    )
    model.nodal_active_load = pm.Param(
        model.bus_set, model.time_set, initialize=init_active_nodal_load, mutable=True
    )
    model.nodal_reactive_load = pm.Param(
        model.bus_set, model.time_set, initialize=init_reactive_nodal_load, mutable=True
    )
    model.nodal_active_feedin = pm.Param(
        model.bus_set, model.time_set, initialize=init_active_nodal_feedin, mutable=True
    )
    model.nodal_reactive_feedin = pm.Param(
        model.bus_set,
        model.time_set,
        initialize=init_reactive_nodal_feedin,
        mutable=True,
    )
    model.tan_phi_load = fixed_parameters["tan_phi_load"]
    model.tan_phi_feedin = fixed_parameters["tan_phi_feedin"]
    model.v_slack = v_slack
    model.branches = fixed_parameters["branches"]
    model.branch_set = pm.Set(initialize=model.branches.index)
    model.underlying_branch_elements = fixed_parameters["underlying_branch_elements"]
    model.power_factors = pm.Param(
        model.branch_set, model.time_set, initialize=init_power_factors, mutable=True
    )
    # add n-1 security # Todo: make optional?
    # adapt i_lines_allowed for radial feeders
    buses_in_cycles = list(
        set(itertools.chain.from_iterable(edisgo_obj.topology.rings))
    )

    # Find lines in cycles
    lines_in_cycles = list(
        grid_object.lines_df.loc[
            grid_object.lines_df[["bus0", "bus1"]].isin(buses_in_cycles).all(axis=1)
        ].index.values
    )

    model.branches_load_factors = pd.DataFrame(
        index=model.time_set, columns=model.branch_set
    )
    model.branches_load_factors.loc[:, :] = 1
    tmp_residual_load = edisgo_obj.timeseries.residual_load.loc[timesteps]
    indices = pd.DataFrame(index=timesteps, columns=["index"])
    indices["index"] = [i for i in range(len(timesteps))]
    model.branches_load_factors.loc[
        indices.loc[tmp_residual_load.loc[timesteps] < 0].values.T[0], lines_in_cycles
    ] = load_factor_rings  # Todo: distinction of mv and lv?
    # Note: So far LV does not contain rings
    # Variables
    model.p_cum = pm.Var(model.branch_set, model.time_set)
    model.slack_p_cum_pos = pm.Var(model.branch_set, model.time_set, bounds=(0, None))
    model.slack_p_cum_neg = pm.Var(model.branch_set, model.time_set, bounds=(0, None))
    model.q_cum = pm.Var(model.branch_set, model.time_set)
    model.v = pm.Var(model.bus_set, model.time_set)
    model.slack_v_pos = pm.Var(model.bus_set, model.time_set, bounds=(0, None))
    model.slack_v_neg = pm.Var(model.bus_set, model.time_set, bounds=(0, None))
    model.curtailment_load = pm.Var(
        model.bus_set,
        model.time_set,
        bounds=lambda m, b, t: (0, m.nodal_active_load[b, t]),
    )
    model.curtailment_feedin = pm.Var(
        model.bus_set,
        model.time_set,
        bounds=lambda m, b, t: (0, m.nodal_active_feedin[b, t]),
    )
    # add curtailment of flexible units if present
    if hasattr(model, "flexible_charging_points_set"):
        model.curtailment_ev = pm.Var(model.bus_set, model.time_set, bounds=(0, None))

        model.UpperCurtEV = pm.Constraint(
            model.bus_set, model.time_set, rule=upper_bound_curtailment_ev
        )
    if hasattr(model, "flexible_heat_pumps_set"):
        model.curtailment_hp = pm.Var(model.bus_set, model.time_set, bounds=(0, None))

        model.UpperCurtHP = pm.Constraint(
            model.bus_set, model.time_set, rule=upper_bound_curtailment_hp
        )
    # Constraints
    model.ActivePower = pm.Constraint(
        model.branch_set, model.time_set, rule=active_power
    )
    model.UpperActive = pm.Constraint(
        model.branch_set, model.time_set, rule=upper_active_power
    )
    model.LowerActive = pm.Constraint(
        model.branch_set, model.time_set, rule=lower_active_power
    )
    # model.ReactivePower = pm.Constraint(model.branch_set, model.time_set,
    #                                     rule=reactive_power)
    model.SlackVoltage = pm.Constraint(
        model.slack_bus, model.time_set, rule=slack_voltage
    )
    model.VoltageDrop = pm.Constraint(
        model.branch_set, model.time_set, rule=voltage_drop
    )
    model.UpperVoltage = pm.Constraint(
        model.bus_set, model.time_set, rule=upper_voltage
    )
    model.LowerVoltage = pm.Constraint(
        model.bus_set, model.time_set, rule=lower_voltage
    )
    # model.UpperCurtLoad = pm.Constraint(model.bus_set, model.time_set,
    #                                     rule=upper_bound_curtailment_load)
    return model


def add_ev_model_bands(
    model,
    fixed_parameters,
    charging_efficiency,
    energy_level_starts={"ev": None},
    energy_level_end=None,
    energy_level_beginning=None,
    charging_starts={"ev": None},
):
    """
    Method to add sets, variables and constraints for including EV flexibility in terms
    of energy bands.

    Parameters
    ----------
    model :
    fixed_parameters :
    energy_level_starts : dict('ev':pd.Series)
    energy_level_end :
    energy_level_beginning :
    charging_starts : dict('ev':pd.Series)

    Returns
    -------
    """
    # Sets and parameters
    model.flexible_charging_points_set = pm.Set(
        initialize=fixed_parameters["optimized_charging_points"]
    )
    model.upper_ev_power = fixed_parameters["ev_flex_bands"]["upper_power"]
    model.upper_ev_energy = fixed_parameters["ev_flex_bands"]["upper_energy"]
    model.lower_ev_energy = fixed_parameters["ev_flex_bands"]["lower_energy"]
    model.charging_efficiency = charging_efficiency
    model.lower_bound_ev = pm.Param(
        model.flexible_charging_points_set,
        model.time_set,
        initialize=set_lower_band_ev,
        mutable=True,
    )
    model.upper_bound_ev = pm.Param(
        model.flexible_charging_points_set,
        model.time_set,
        initialize=set_upper_band_ev,
        mutable=True,
    )
    model.power_bound_ev = pm.Param(
        model.flexible_charging_points_set,
        model.time_set,
        initialize=set_power_band_ev,
        mutable=True,
    )
    # Variables
    model.charging_ev = pm.Var(
        model.flexible_charging_points_set,
        model.time_set,
        bounds=lambda m, b, t: (0, m.power_bound_ev[b, t]),
    )

    model.energy_level_ev = pm.Var(
        model.flexible_charging_points_set,
        model.time_set,
        bounds=lambda m, b, t: (m.lower_bound_ev[b, t], m.upper_bound_ev[b, t]),
    )
    # Constraints
    model.EVCharging = pm.Constraint(
        model.flexible_charging_points_set, model.time_non_zero, rule=charging_ev
    )
    model = add_rolling_horizon(
        comp_type="ev",
        charging_starts=charging_starts,
        energy_level_beginning=energy_level_beginning,
        energy_level_end=energy_level_end,
        energy_level_starts=energy_level_starts,
        model=model,
    )

    return model


def add_rolling_horizon(
    comp_type,
    charging_starts,
    energy_level_beginning,
    energy_level_end,
    energy_level_starts,
    model,
):
    """

    Parameters
    ----------
    comp_type : "ev" or "hp"
    charging_starts : dict('ev':pd.Series, 'tes':pd.Series, 'hp':pd.Series)
    energy_level_beginning :
    energy_level_end :
    energy_level_starts : dict('ev':pd.Series, 'tes':pd.Series)
    model :

    Returns
    -------

    """
    charging_attrs, energy_attrs, flex_set = get_attrs_rolling_horizon(comp_type, model)
    # set initial energy level
    for energy_attr in energy_attrs[comp_type.lower()]:
        setattr(
            model,
            f"energy_level_start_{energy_attr}",
            pm.Param(
                flex_set,
                initialize=energy_level_starts[energy_attr],
                mutable=True,
                within=pm.Any,
            ),
        )
        setattr(
            model,
            f"slack_initial_energy_pos_{energy_attr}",
            pm.Var(flex_set, bounds=(0, None)),
        )
        setattr(
            model,
            f"slack_initial_energy_neg_{energy_attr}",
            pm.Var(flex_set, bounds=(0, None)),
        )
        setattr(
            model,
            f"InitialEnergyLevel{energy_attr.upper()}",
            pm.Constraint(
                flex_set,
                model.time_zero,
                rule=globals()[f"initial_energy_level_{energy_attr}"],
            ),
        )
        setattr(
            model,
            f"InitialEnergyLevelStart{energy_attr.upper()}",
            pm.Constraint(
                flex_set,
                model.time_zero,
                rule=globals()[f"fixed_energy_level_{energy_attr}"],
            ),
        )
        if energy_level_starts[energy_attr] is None:
            getattr(model, f"InitialEnergyLevel{energy_attr.upper()}").deactivate()
        else:
            getattr(model, f"InitialEnergyLevelStart{energy_attr.upper()}").deactivate()
        # set final energy level and if necessary charging power
        setattr(
            model,
            f"energy_level_end_{energy_attr}",
            pm.Param(
                flex_set,
                initialize=energy_level_end,
                mutable=True,
                within=pm.Any,
            ),
        )
        setattr(
            model,
            f"FinalEnergyLevelFix{energy_attr.upper()}",
            pm.Constraint(
                flex_set,
                model.time_end,
                rule=globals()[f"fixed_energy_level_{energy_attr}"],
            ),
        )
        if energy_level_beginning is None:
            setattr(
                model,
                f"energy_level_beginning_{energy_attr}",
                pm.Param(flex_set, initialize=0, mutable=True),
            )
        else:
            setattr(
                model,
                f"energy_level_beginning_{energy_attr}",
                pm.Param(flex_set, initialize=energy_level_beginning, mutable=True),
            )
        setattr(
            model,
            f"FinalEnergyLevelEnd{energy_attr.upper()}",
            pm.Constraint(
                flex_set,
                model.time_end,
                rule=globals()[f"final_energy_level_{energy_attr}"],
            ),
        )
        if energy_level_end is None:
            getattr(model, f"FinalEnergyLevelFix{energy_attr.upper()}").deactivate()
            getattr(model, f"FinalEnergyLevelEnd{energy_attr.upper()}").deactivate()
        else:
            if comp_type(energy_level_end) != bool:
                getattr(model, f"FinalEnergyLevelFix{energy_attr.upper()}").deactivate()
            elif comp_type(energy_level_end) == bool:
                getattr(model, f"FinalEnergyLevelEnd{energy_attr.upper()}").deactivate()

    # set initial charging power
    for charging_attr in charging_attrs[comp_type.lower()]:
        setattr(
            model,
            f"charging_initial_{charging_attr}",
            pm.Param(
                flex_set,
                initialize=charging_starts[charging_attr],
                mutable=True,
                within=pm.Any,
            ),
        )
        setattr(
            model,
            f"slack_initial_charging_pos_{charging_attr}",
            pm.Var(flex_set, bounds=(0, None)),
        )
        setattr(
            model,
            f"slack_initial_charging_neg_{charging_attr}",
            pm.Var(flex_set, bounds=(0, None)),
        )
        setattr(
            model,
            f"InitialChargingPower{charging_attr.upper()}",
            pm.Constraint(
                flex_set,
                model.time_zero,
                rule=globals()[f"initial_charging_power_{charging_attr}"],
            ),
        )
        if charging_starts[charging_attr] is None:
            getattr(model, f"InitialChargingPower{charging_attr.upper()}").deactivate()

        # uncommented as subjected to old emob formulation
        # setattr(
        #     model,
        #     f"FinalChargingPower{charging_attr.upper()}",
        #     pm.Constraint(
        #         flex_set,
        #         model.time_end,
        #         rule=globals()[f"final_charging_power_{charging_attr}"],
        #     ),
        # )
        # if energy_level_end is None:
        #     getattr(model, f"FinalChargingPower{charging_attr.upper()}").deactivate()
    return model


def get_attrs_rolling_horizon(comp_type, model):
    sets = {"ev": "charging_points", "hp": "heat_pumps"}
    energy_attrs = {"ev": ["ev"], "hp": ["tes"]}
    charging_attrs = {"ev": ["ev"], "hp": ["hp", "tes"]}
    flex_set = getattr(model, f"flexible_{sets[comp_type.lower()]}_set")
    return charging_attrs, energy_attrs, flex_set


def add_heat_pump_model(
    model,
    fixed_parameters,
    energy_level_starts={"tes": None},
    energy_level_end=None,
    energy_level_beginning=None,
    charging_starts={"hp": None, "tes": None},
):
    """

    Parameters
    ----------
    model :
    fixed_parameters :
    energy_level_starts : dict('ev':pd.Series, 'tes':pd.Series)
    energy_level_end :
    energy_level_beginning :
    charging_starts : dict('ev':pd.Series, 'tes':pd.Series, 'hp':pd.Series)

    Returns
    -------

    """

    def energy_balance_hp_tes(model, hp, time):
        return (
            model.charging_hp[hp, time] * model.cop_hp[hp, time]
            == model.heat_demand_hp[hp, time] + model.charging_tes[hp, time]
        )

    def charging_tes(model, hp, time):
        return model.energy_level_tes[hp, time] == model.energy_level_tes[
            hp, time - 1
        ] + model.charging_tes[hp, time] * (
            pd.to_timedelta(model.time_increment) / pd.to_timedelta("1h")
        )

    # add set of hps
    model.flexible_heat_pumps_set = pm.Set(
        initialize=fixed_parameters["optimized_heat_pumps"]
    )
    # save fix parameters
    model.heat_pumps = fixed_parameters["heat_pumps"]
    model.tes = fixed_parameters["tes"]
    model.cop = fixed_parameters["cop"]
    model.cop_hp = pm.Param(
        model.flexible_heat_pumps_set,
        model.time_set,
        initialize=set_cop_hp,
        mutable=True,
        within=pm.Any,
    )
    model.heat_demand = fixed_parameters["heat_demand"]
    model.heat_demand_hp = pm.Param(
        model.flexible_heat_pumps_set,
        model.time_set,
        initialize=set_heat_demand,
        mutable=True,
        within=pm.Any,
    )
    # set up variables
    model.energy_level_tes = pm.Var(
        model.flexible_heat_pumps_set,
        model.time_set,
        bounds=lambda m, hp, t: (0, m.tes.loc[hp, "capacity"]),
    )
    model.charging_tes = pm.Var(model.flexible_heat_pumps_set, model.time_set)
    model.charging_hp = pm.Var(
        model.flexible_heat_pumps_set,
        model.time_set,
        bounds=lambda m, hp, t: (0, m.heat_pumps.loc[hp, "p_set"]),
    )
    # add constraints
    model.EnergyBalanceHPTES = pm.Constraint(
        model.flexible_heat_pumps_set, model.time_set, rule=energy_balance_hp_tes
    )
    model.ChargingTES = pm.Constraint(
        model.flexible_heat_pumps_set, model.time_non_zero, rule=charging_tes
    )
    model = add_rolling_horizon(
        comp_type="hp",
        charging_starts=charging_starts,
        energy_level_beginning=energy_level_beginning,
        energy_level_end=energy_level_end,
        energy_level_starts=energy_level_starts,
        model=model,
    )
    return model


def update_model(
    model,
    timesteps,
    fixed_parameters,
    **kwargs,
):
    """
    Method to update model parameter where necessary if rolling horizon
    optimization is chosen.

    Parameters
    ----------
    model :
    timesteps :
    fixed_parameters :
    kwargs :
        energy_level_beginning : default None
        energy_level_end_tes : default None
        energy_level_end_ev : default None

        charging_start : dict('ev':pd.Series, 'tes':pd.Series, 'hp':pd.Series)
            starting value after 1st Iteration, dynamic
        energy_level_starts : dict('ev':pd.Series, 'tes':pd.Series)
            starting value after 1st Iteration, dynamic

    Returns
    -------

    """
    logger.info("Updating model")
    t1 = perf_counter()
    # TODO Warum iteration über jeden Zeitschritt?
    for i in model.time_set:
        overlap = i - len(timesteps) + 1
        if overlap > 0:
            timeindex = timesteps[-1] + pd.to_timedelta(model.time_increment) * overlap
            indexer = timesteps[-1]
        else:
            timeindex = timesteps[i]
            indexer = timesteps[i]
        model.timeindex[i].set_value(timeindex)
        model.residual_load[i].set_value(
            fixed_parameters["res_load_inflexible_units"][indexer]
        )
        for bus in model.bus_set:
            model.nodal_active_power[bus, i].set_value(
                fixed_parameters["nodal_active_power"].loc[bus, indexer]
            )
            model.nodal_reactive_power[bus, i].set_value(
                fixed_parameters["nodal_reactive_power"].loc[bus, indexer]
            )
            model.nodal_active_load[bus, i].set_value(
                fixed_parameters["nodal_active_load"].loc[bus, indexer]
            )
            model.nodal_reactive_load[bus, i].set_value(
                fixed_parameters["nodal_reactive_load"].loc[bus, indexer]
            )
            model.nodal_active_feedin[bus, i].set_value(
                fixed_parameters["nodal_active_feedin"].loc[bus, indexer]
            )
            model.nodal_reactive_feedin[bus, i].set_value(
                fixed_parameters["nodal_reactive_feedin"].loc[bus, indexer]
            )

        for branch in model.branch_set:
            model.power_factors[branch, i].set_value(
                fixed_parameters["power_factors"].loc[branch, indexer]
            )

    if fixed_parameters["optimize_emob"]:
        for t in model.time_set:
            overlap = t - len(timesteps) + 1
            if overlap > 0:
                indexer = len(timesteps) - 1
            else:
                indexer = t
            for cp in model.flexible_charging_points_set:
                model.power_bound_ev[cp, t].set_value(
                    set_power_band_ev(model, cp, indexer)
                )
                model.lower_bound_ev[cp, t].set_value(
                    set_lower_band_ev(model, cp, indexer)
                )
                model.upper_bound_ev[cp, t].set_value(
                    set_upper_band_ev(model, cp, indexer)
                )
        model = update_rolling_horizon("ev", model, **kwargs)

    if fixed_parameters["optimize_hp"]:
        for t in model.time_set:
            overlap = t - len(timesteps) + 1
            if overlap > 0:
                indexer = len(timesteps) - 1
            else:
                indexer = t

            for hp in model.flexible_heat_pumps_set:
                model.heat_demand_hp[hp, t].set_value(
                    set_heat_demand(model, hp, indexer)
                )
                model.cop_hp[hp, t].set_value(set_cop_hp(model, hp, indexer))
        model = update_rolling_horizon("hp", model, **kwargs)

    if fixed_parameters["optimize_bess"]:
        raise NotImplementedError
    logger.info(f"It took {perf_counter() - t1} seconds to update the model.")
    return model


def update_rolling_horizon(comp_type, model, **kwargs):
    """

    Parameters
    ----------
    comp_type :
    model :
    kwargs :
        energy_level_beginning : default None
        energy_level_end_tes : default None
        energy_level_end_ev : default None

        charging_start : dict('ev':pd.Series, 'tes':pd.Series, 'hp':pd.Series)
        energy_level_start : dict('ev':pd.Series, 'tes':pd.Series)
    Returns
    -------

    """
    charging_attrs, energy_attrs, flex_set = get_attrs_rolling_horizon(comp_type, model)

    # set energy level start
    for energy_attr in energy_attrs[comp_type.lower()]:

        energy_level_starts = kwargs.get(
            "energy_level_starts", {"ev": None, "tes": None}
        )
        # if run is new start of era deactivate initial energy level,
        # otherwise activate initial energy and charging
        if energy_level_starts[energy_attr] is None:
            getattr(model, f"InitialEnergyLevel{energy_attr.upper()}").deactivate()
            getattr(model, f"InitialEnergyLevelStart{energy_attr.upper()}").activate()
        else:
            for comp in flex_set:
                getattr(model, f"energy_level_start_{energy_attr}")[comp].set_value(
                    energy_level_starts[energy_attr][comp]
                )
            getattr(model, f"InitialEnergyLevel{energy_attr.upper()}").activate()
            getattr(model, f"InitialEnergyLevelStart{energy_attr.upper()}").deactivate()

        # set energy level beginning if necessary
        # this is implemented to compensate difference to reference charging
        # scenario
        energy_level_beginning = kwargs.get("energy_level_beginning", None)
        if energy_level_beginning is not None:
            for comp in flex_set:
                getattr(model, f"energy_level_beginning_{energy_attr}")[comp].set_value(
                    energy_level_beginning[comp]
                )

        # set final energy level and if necessary charging power
        energy_level_end = kwargs.get(f"energy_level_end_{energy_attr}", None)
        if energy_level_end is None:
            getattr(model, f"FinalEnergyLevelFix{energy_attr.upper()}").deactivate()
            getattr(model, f"FinalEnergyLevelEnd{energy_attr.upper()}").deactivate()
        elif type(energy_level_end) == bool:
            getattr(model, f"FinalEnergyLevelFix{energy_attr.upper()}").activate()
            getattr(model, f"FinalEnergyLevelEnd{energy_attr.upper()}").deactivate()
        else:
            for comp in flex_set:
                getattr(model, f"energy_level_end_{energy_attr}")[comp].set_value(
                    energy_level_end[comp]
                )
            getattr(model, f"FinalEnergyLevelEnd{energy_attr.upper()}").activate()
            getattr(model, f"FinalEnergyLevelFix{energy_attr.upper()}").deactivate()

    # set initial charging value
    for charging_attr in charging_attrs[comp_type.lower()]:
        charging_starts = kwargs.get(
            "charging_starts",
            {
                "ev": None,
                "tes": None,
                "hp": None,
            },
        )

        if charging_starts[charging_attr] is not None:
            for comp in flex_set:
                getattr(model, f"charging_initial_{charging_attr}")[comp].set_value(
                    charging_starts[charging_attr][comp]
                )
            getattr(model, f"InitialChargingPower{charging_attr.upper()}").activate()
        # remove as not needed anymore with new formulation
        # if energy_level_end is None:
        #     getattr(model, f"FinalChargingPower{charging_attr.upper()}").deactivate()
        # else:
        #     getattr(model, f"FinalChargingPower{charging_attr.upper()}").activate()
    return model


def optimize(model, solver, load_solutions=True, mode=None, logfile=None,
                                                                    **kwargs):
    """
    Method to run the optimization and extract the results.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
    solver : str
        Solver type, e.g. 'glpk', 'gurobi', 'ipopt'
    load_solutions :
    mode : str
        directory to which results are saved, default None will no saving of
        the results
    logfile : str
        dir/name of logfile for solver

    Returns
    -------

    """

    filename = kwargs.get("lp_filename", False)
    if filename:
        logger.info(f"Save lp file to: {filename}")
        model.write(filename=str(filename), io_options={"symbolic_solver_labels": True})

    logger.info("Starting optimisation")

    t1 = perf_counter()
    opt = pm.SolverFactory(solver)
    opt.options["threads"] = 16

    # Optimize
    results = opt.solve(model,
                        tee=True,
                        load_solutions=load_solutions,
                        logfile=logfile)

    if (results.solver.status == SolverStatus.ok) and (
        results.solver.termination_condition == TerminationCondition.optimal
    ):
        logger.info("Model Solved to Optimality")
        # Extract results
        time_dict = {t: model.timeindex[t].value for t in model.time_set}
        result_dict = {}
        if hasattr(model, "optimized_storage_set"):
            result_dict["x_charge"] = (
                pd.Series(model.charging.extract_values())
                .unstack()
                .rename(columns=time_dict)
                .T
            )
            result_dict["soc"] = (
                pd.Series(model.soc.extract_values())
                .unstack()
                .rename(columns=time_dict)
                .T
            )
        if hasattr(model, "flexible_charging_points_set"):
            result_dict["x_charge_ev"] = (
                pd.Series(model.charging_ev.extract_values())
                .unstack()
                .rename(columns=time_dict)
                .T
            )
            result_dict["energy_level_cp"] = (
                pd.Series(model.energy_level_ev.extract_values())
                .unstack()
                .rename(columns=time_dict)
                .T
            )
            result_dict["slack_charging"] = pd.Series(
                model.slack_initial_charging_pos_ev.extract_values()
            ) + pd.Series(model.slack_initial_charging_neg_ev.extract_values())
            result_dict["slack_energy"] = pd.Series(
                model.slack_initial_energy_pos_ev.extract_values()
            ) + pd.Series(model.slack_initial_energy_neg_ev.extract_values())
            result_dict["curtailment_ev"] = (
                pd.Series(model.curtailment_ev.extract_values())
                .unstack()
                .rename(columns=time_dict)
                .T
            )
        if hasattr(model, "flexible_heat_pumps_set"):
            result_dict["charging_hp_el"] = (
                pd.Series(model.charging_hp.extract_values())
                .unstack()
                .rename(columns=time_dict)
                .T
            )
            result_dict["charging_tes"] = (
                pd.Series(model.charging_tes.extract_values())
                .unstack()
                .rename(columns=time_dict)
                .T
            )
            result_dict["energy_tes"] = (
                pd.Series(model.energy_level_tes.extract_values())
                .unstack()
                .rename(columns=time_dict)
                .T
            )
        result_dict["curtailment_load"] = (
            pd.Series(model.curtailment_load.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T
        )
        result_dict["curtailment_feedin"] = (
            pd.Series(model.curtailment_feedin.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T
        )

        result_dict["p_line"] = (
            pd.Series(model.p_cum.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T
        )
        result_dict["q_line"] = (
            pd.Series(model.q_cum.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T
        )
        result_dict["v_bus"] = (
            pd.Series(model.v.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T.apply(np.sqrt)
        )
        result_dict["slack_v_pos"] = (
            pd.Series(model.slack_v_pos.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T
        )
        result_dict["slack_v_neg"] = (
            pd.Series(model.slack_v_neg.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T
        )
        result_dict["slack_p_cum_pos"] = (
            pd.Series(model.slack_p_cum_pos.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T
        )
        result_dict["slack_p_cum_neg"] = (
            pd.Series(model.slack_p_cum_pos.extract_values())
            .unstack()
            .rename(columns=time_dict)
            .T
        )
        if mode == "energy_band":
            result_dict["p_aggr"] = pd.Series(
                model.grid_power_flexible.extract_values()
            ).rename(time_dict)
        # Todo: check if this works
        index = result_dict["curtailment_load"].index[
            result_dict["curtailment_load"].index.isin(model.tan_phi_load.index)
        ]
        result_dict["curtailment_reactive_load"] = (
            result_dict["curtailment_load"]
            .multiply(
                model.tan_phi_load.loc[index, result_dict["curtailment_load"].columns]
            )
            .dropna(how="all")
        )
        result_dict["curtailment_reactive_feedin"] = (
            result_dict["curtailment_feedin"]
            .multiply(
                model.tan_phi_feedin.loc[
                    index, result_dict["curtailment_feedin"].columns
                ]
            )
            .dropna(how="all")
        )

        logger.info(f"It took {perf_counter() - t1} seconds to optimize model.")
        return result_dict
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        logger.info("Model is infeasible")
        return
        # Do something when model in infeasible
    else:
        logger.info(f"Solver Status: {results.solver.status}")
        return


def setup_grid_object(object):
    """
    Set up the grid and edisgo object.


    Parameters
    ----------
    object :

    Returns
    -------

    """
    if hasattr(object, "topology"):  # EDisGo object
        grid_object = deepcopy(object.topology)
        edisgo_object = deepcopy(object)
        slack = grid_object.mv_grid.station.index
    else:  # Grid object
        grid_object = deepcopy(object)
        edisgo_object = deepcopy(object.edisgo_obj)
        # slack = [
        #     grid_object.transformers_df.bus1.iloc[0]
        # ]  # Todo: careful with MV grid, does not work with that right?
        slack = grid_object.station.index
    return edisgo_object, grid_object, slack


def concat_parallel_branch_elements(grid_object):
    """
    Method to merge parallel lines and transformers into one element,
    respectively.

    Parameters
    ----------
    grid_object

    Returns
    -------

    """
    lines = fuse_parallel_branches(grid_object.lines_df)
    trafos = grid_object.transformers_df.loc[
        grid_object.transformers_df.bus0.isin(grid_object.buses_df.index)
    ].loc[grid_object.transformers_df.bus1.isin(grid_object.buses_df.index)]
    transformers = fuse_parallel_branches(trafos)
    return pd.concat([lines, transformers], sort=False)


def fuse_parallel_branches(branches):
    branches_tmp = branches[["bus0", "bus1"]]
    parallel_branches = pd.DataFrame(columns=branches.columns)
    if branches_tmp.duplicated().any():
        duplicated_branches = branches_tmp.loc[branches_tmp.duplicated(keep=False)]
        duplicated_branches["visited"] = False
        branches_tmp = branches_tmp.drop(duplicated_branches.index)
        for name, buses in duplicated_branches.iterrows():
            if duplicated_branches.loc[name, "visited"]:
                continue
            else:
                parallel_branches_tmp = duplicated_branches.loc[
                    (duplicated_branches == buses).all(axis=1)
                ]
                duplicated_branches.loc[parallel_branches_tmp.index, "visited"] = True
                name_par = "_".join(str.split(name, "_")[:-1])
                parallel_branches.loc[name_par] = branches.loc[name]
                parallel_branches.loc[
                    name_par, ["r", "x", "s_nom"]
                ] = calculate_impedance_for_parallel_components(
                    branches.loc[parallel_branches_tmp.index, ["r", "x", "s_nom"]],
                    pu=False,
                )
    fused_branches = pd.concat(
        [branches.loc[branches_tmp.index], parallel_branches], sort=False
    )
    return fused_branches


def get_underlying_elements(fixed_parameters):
    """

    Parameters
    ----------
    parameters :

    Returns
    -------

    """

    def _get_underlying_elements(
        downstream_elements, power_factors, fixed_parameters, branch
    ):
        bus0 = fixed_parameters["branches"].loc[branch, "bus0"]
        bus1 = fixed_parameters["branches"].loc[branch, "bus1"]
        s_nom = fixed_parameters["branches"].loc[
            branch, fixed_parameters["pars"]["s_nom"]
        ]
        relevant_buses_bus0 = (
            fixed_parameters["downstream_nodes_matrix"]
            .loc[bus0][fixed_parameters["downstream_nodes_matrix"].loc[bus0] == 1]
            .index.values
        )
        relevant_buses_bus1 = (
            fixed_parameters["downstream_nodes_matrix"]
            .loc[bus1][fixed_parameters["downstream_nodes_matrix"].loc[bus1] == 1]
            .index.values
        )
        relevant_buses = list(
            set(relevant_buses_bus0).intersection(relevant_buses_bus1)
        )
        downstream_elements.loc[branch, "buses"] = relevant_buses
        if (
            fixed_parameters["nodal_reactive_power"]
            .loc[relevant_buses]
            .sum()
            .divide(s_nom)
            .apply(abs)
            > 1
        ).any():
            logger.info(
                f"Careful: Reactive power already exceeding line "
                f"capacity for branch {branch}."
            )
        power_factors.loc[branch] = (
            1
            - fixed_parameters["nodal_reactive_power"]
            .loc[relevant_buses]
            .sum()
            .divide(s_nom)
            .apply(np.square)
        ).apply(np.sqrt)
        if fixed_parameters["optimize_bess"]:
            downstream_elements.loc[branch, "flexible_storage"] = (
                fixed_parameters["grid_object"]
                .storage_units_df.loc[
                    fixed_parameters["grid_object"].storage_units_df.index.isin(
                        fixed_parameters["optimized_storage_units"]
                    )
                    & fixed_parameters["grid_object"].storage_units_df.bus.isin(
                        relevant_buses
                    )
                ]
                .index.values
            )
        else:
            downstream_elements.loc[branch, "flexible_storage"] = []
        if fixed_parameters["optimize_emob"]:
            downstream_elements.loc[branch, "flexible_ev"] = (
                fixed_parameters["grid_object"]
                .charging_points_df.loc[
                    fixed_parameters["grid_object"].charging_points_df.index.isin(
                        fixed_parameters["optimized_charging_points"]
                    )
                    & fixed_parameters["grid_object"].charging_points_df.bus.isin(
                        relevant_buses
                    )
                ]
                .index.values
            )
        else:
            downstream_elements.loc[branch, "flexible_ev"] = []
        if fixed_parameters["optimize_hp"]:
            hps = fixed_parameters["grid_object"].loads_df.loc[
                fixed_parameters["grid_object"].loads_df.type == "heat_pump"
            ]
            downstream_elements.loc[branch, "flexible_hp"] = hps.loc[
                hps.index.isin(fixed_parameters["optimized_heat_pumps"])
                & hps.bus.isin(relevant_buses)
            ].index.values
        else:
            downstream_elements.loc[branch, "flexible_hp"] = []
        return downstream_elements, power_factors

    downstream_elements = pd.DataFrame(
        index=fixed_parameters["branches"].index,
        columns=["buses", "flexible_storage", "flexible_ev", "flexible_hp"],
    )
    power_factors = pd.DataFrame(
        index=fixed_parameters["branches"].index,
        columns=fixed_parameters["nodal_active_power"].columns,
    )
    for branch in downstream_elements.index:
        downstream_elements, power_factors = _get_underlying_elements(
            downstream_elements, power_factors, fixed_parameters, branch
        )
    if power_factors.isna().any().any():
        logger.info(
            f"WARNING: Branch {branch} is overloaded with reactive "
            f"power. Still needs handling."
        )
        power_factors = power_factors.fillna(
            0.01
        )  # Todo: ask Gaby and Birgit about this
    return downstream_elements, power_factors


def get_residual_load_of_not_optimized_components(
    grid,
    edisgo_obj,
    relevant_storage_units=None,
    relevant_generators=None,
    relevant_loads=None,
):
    """
    Method to get residual load of fixed components.

    Parameters
    ----------
    grid
    edisgo_obj
    relevant_storage_units
    relevant_generators
    relevant_loads

    Returns
    -------

    """
    if relevant_loads is None:
        relevant_loads = grid.loads_df.index
    if relevant_generators is None:
        relevant_generators = grid.generators_df.index
    if relevant_storage_units is None:
        relevant_storage_units = grid.storage_units_df.index

    return (
        edisgo_obj.timeseries.generators_active_power[relevant_generators].sum(axis=1)
        + edisgo_obj.timeseries.storage_units_active_power[relevant_storage_units].sum(
            axis=1
        )
        - edisgo_obj.timeseries.loads_active_power[relevant_loads].sum(axis=1)
    ).loc[edisgo_obj.timeseries.timeindex]


def set_lower_band_ev(model, cp, time):
    return model.lower_ev_energy.loc[model.timeindex[time], cp]


def set_upper_band_ev(model, cp, time):
    return model.upper_ev_energy.loc[model.timeindex[time], cp]


def set_power_band_ev(model, cp, time):
    return model.upper_ev_power.loc[model.timeindex[time], cp]


def set_cop_hp(model, hp, time):
    return model.cop.loc[model.timeindex[time], hp]


def set_heat_demand(model, hp, time):
    return model.heat_demand.loc[model.timeindex[time], hp]


def active_power(model, branch, time):
    """
    Constraint for active power at node

    Parameters
    ----------
    model :
    branch :
    time :

    Returns
    -------

    """
    relevant_buses = model.underlying_branch_elements.loc[branch, "buses"]
    relevant_storage_units = model.underlying_branch_elements.loc[
        branch, "flexible_storage"
    ]
    relevant_charging_points = model.underlying_branch_elements.loc[
        branch, "flexible_ev"
    ]
    relevant_heat_pumps = model.underlying_branch_elements.loc[branch, "flexible_hp"]
    load_flow_on_line = sum(
        model.nodal_active_power[bus, time] for bus in relevant_buses
    )
    if hasattr(model, "flexible_charging_points_set"):
        ev_curtailment = sum(model.curtailment_ev[bus, time] for bus in relevant_buses)
    else:
        ev_curtailment = 0
    if hasattr(model, "flexible_heat_pumps_set"):
        hp_curtailment = sum(model.curtailment_hp[bus, time] for bus in relevant_buses)
    else:
        hp_curtailment = 0
    return (
        model.p_cum[branch, time]
        == load_flow_on_line
        + sum(model.charging[storage, time] for storage in relevant_storage_units)
        - sum(model.charging_ev[cp, time] for cp in relevant_charging_points)
        - sum(model.charging_hp[cp, time] for cp in relevant_heat_pumps)
        + sum(
            model.curtailment_load[bus, time] - model.curtailment_feedin[bus, time]
            for bus in relevant_buses
        )
        + ev_curtailment
        + hp_curtailment
    )


def upper_active_power(model, branch, time):
    """
    Upper bound of active branch power

    Parameters
    ----------
    model :
    branch :
    time :

    Returns
    -------

    """
    return (
        model.p_cum[branch, time]
        <= model.thermal_limit
        * model.power_factors[branch, time]
        * model.branches.loc[branch, model.pars["s_nom"]]
        + model.slack_p_cum_pos[branch, time]
    )


def lower_active_power(model, branch, time):
    """
    Lower bound of active branch power

    Parameters
    ----------
    model :
    branch :
    time :

    Returns
    -------

    """
    return (
        model.p_cum[branch, time]
        >= -model.thermal_limit
        * model.power_factors[branch, time]
        * model.branches.loc[branch, model.pars["s_nom"]]
        - model.slack_p_cum_neg[branch, time]
    )


def slack_voltage(model, bus, time):
    """
    Constraint that fixes voltage to nominal voltage

    Parameters
    ----------
    model :
    bus :
    time :

    Returns
    -------

    """
    timeindex = model.timeindex[time]
    if isinstance(model.v_slack, pd.Series):
        return model.v[bus, time] == np.square(model.v_slack[timeindex] * model.v_nom)
    else:
        return model.v[bus, time] == np.square(model.v_slack)


def voltage_drop(model, branch, time):
    """
    Constraint that describes the voltage drop over one line

    Parameters
    ----------
    model :
    branch :
    time :

    Returns
    -------

    """
    bus0 = model.branches.loc[branch, "bus0"]
    bus1 = model.branches.loc[branch, "bus1"]
    if model.downstream_nodes_matrix.loc[bus0, bus1] == 1:
        upstream_bus = bus0
        downstream_bus = bus1
    elif model.downstream_nodes_matrix.loc[bus1, bus0] == 1:
        upstream_bus = bus1
        downstream_bus = bus0
    else:
        raise Exception(
            "Something went wrong. Bus0 and bus1 of line {} are "
            "not connected in downstream_nodes_matrix.".format(branch)
        )
    q_cum = get_q_line(model, branch, time)
    return model.v[downstream_bus, time] == model.v[upstream_bus, time] + 2 * (
        model.p_cum[branch, time] * model.branches.loc[branch, model.pars["r"]]
        + q_cum * model.branches.loc[branch, model.pars["x"]]
    )


def get_q_line(model, branch, time, get_results=False):
    """
    Method to extract reactive power flow on line.

    Parameters
    ----------
    model :
    branch :
    time :
    get_results :

    Returns
    -------

    """
    timeindex = model.timeindex[time]
    relevant_buses = model.underlying_branch_elements.loc[branch, "buses"]
    load_flow_on_line = sum(
        model.nodal_reactive_power[bus, time] for bus in relevant_buses
    )
    if get_results:
        return load_flow_on_line + sum(
            model.curtailment_load[bus, time].value
            * model.tan_phi_load.loc[timeindex, bus]
            - model.curtailment_feedin[  # Todo: Find out if this should be pos or neg
                bus, time
            ].value
            * model.tan_phi_feedin.loc[timeindex, bus]
            for bus in relevant_buses
        )
    else:
        return load_flow_on_line + sum(
            model.curtailment_load[bus, time] * model.tan_phi_load.loc[timeindex, bus]
            - model.curtailment_feedin[  # Todo: Find out if this should be pos or neg,
                # analogously to q_cum
                bus,
                time,
            ]
            * model.tan_phi_feedin.loc[timeindex, bus]
            for bus in relevant_buses
        )


def upper_voltage(model, bus, time):
    """
    Upper bound on voltage at buses

    Parameters
    ----------
    model :
    bus :
    time :

    Returns
    -------
    """

    try:
        v_max = model.v_min[bus]
    except KeyError as e:
        # logging.debug(f"No Series passed for v_max: {e}")
        v_max = model.v_max

    return (
        model.v[bus, time]
        <= np.square(v_max * model.v_nom) + model.slack_v_pos[bus, time]
    )


def lower_voltage(model, bus, time):
    """
    Lower bound on voltage at buses.

    Parameters
    ----------
    model :
    bus :
    time :

    Returns
    -------

    """
    try:
        v_min = model.v_min[bus]
    except KeyError as e:
        # logging.debug(f"No Series passed for v_min: {e}")
        v_min = model.v_min

    return (
        model.v[bus, time]
        >= np.square(v_min * model.v_nom) - model.slack_v_neg[bus, time]
    )


def soc(model, storage, time):
    """
    Constraint for battery charging #Todo: Check if time-1 or time for charging

    Parameters
    ----------
    model :
    storage :
    time :

    Returns
    -------

    """
    return model.soc[storage, time] == model.soc[
        storage, time - 1
    ] - model.grid.storage_units_df.loc[storage, "efficiency_store"] * model.charging[
        storage, time - 1
    ] * (
        pd.to_timedelta(model.time_increment) / pd.to_timedelta("1h")
    )


def fix_soc(model, bus, time):
    """
    Constraint with which state of charge at beginning and end of charging
    period is fixed at certain value

    Parameters
    ----------
    model :
    bus :
    time :

    Returns
    -------

    """
    return (
        model.soc[bus, time]
        == model.fix_relative_soc
        * model.grid.storage_units_df.loc[bus, model.pars["capacity"]]
    )


def charging_ev(model, charging_point, time):
    """
    Constraint for charging of EV that has to ly between the lower and upper
    energy band. #Todo: Check if time-1 or time for charging

    Parameters
    ----------
    model :
    charging_point :
    time :

    Returns
    -------

    """
    return model.energy_level_ev[charging_point, time] == model.energy_level_ev[
        charging_point, time - 1
    ] + model.charging_efficiency * model.charging_ev[charging_point, time] * (
        pd.to_timedelta(model.time_increment) / pd.to_timedelta("1h")
    )


def upper_bound_curtailment_ev(model, bus, time):
    """
    Upper bound for the curtailment of flexible EVs.

    Parameters
    ----------
    model
    bus
    time

    Returns
    -------
    Constraint for optimisation
    """
    relevant_charging_points = model.grid.charging_points_df.loc[
        model.grid.charging_points_df.index.isin(model.flexible_charging_points_set)
        & model.grid.charging_points_df.bus.isin([bus])
    ].index.values
    if len(relevant_charging_points) < 1:
        return model.curtailment_ev[bus, time] <= 0
    else:
        return model.curtailment_ev[bus, time] <= sum(
            model.charging_ev[cp, time] for cp in relevant_charging_points
        )


def upper_bound_curtailment_hp(model, bus, time):
    """
    Upper bound for the curtailment of flexible EVs.

    Parameters
    ----------
    model
    bus
    time

    Returns
    -------
    Constraint for optimisation
    """
    relevant_heat_pumps = model.grid.loads_df.loc[
        model.grid.loads_df.index.isin(model.flexible_heat_pumps_set)
        & model.grid.loads_df.bus.isin([bus])
    ].index.values
    if len(relevant_heat_pumps) < 1:
        return model.curtailment_hp[bus, time] <= 0
    else:
        return model.curtailment_hp[bus, time] <= sum(
            model.charging_hp[hp, time] for hp in relevant_heat_pumps
        )


def initial_energy_level(model, comp_type, comp, time):
    """
    Constraint for initial value of energy

    Parameters
    ----------
    model :
    comp_type :
    comp :
    time :

    Returns
    -------

    """
    return (
        getattr(model, f"energy_level_{comp_type.lower()}")[comp, time]
        == getattr(model, f"energy_level_start_{comp_type.lower()}")[comp]
        + getattr(model, f"slack_initial_energy_pos_{comp_type.lower()}")[comp]
        - getattr(model, f"slack_initial_energy_neg_{comp_type.lower()}")[comp]
    )


def initial_energy_level_ev(model, charging_point, time):
    """
    Constraint for initial value of energy

    Parameters
    ----------
    model :
    charging_point :
    time :

    Returns
    -------

    """
    return initial_energy_level(model, "ev", charging_point, time)


def initial_energy_level_tes(model, heat_pump, time):
    """
    Constraint for initial value of energy

    Parameters
    ----------
    model :
    heat_pump :
    time :

    Returns
    -------

    """
    return initial_energy_level(model, "tes", heat_pump, time)


def fixed_energy_level_ev(model, charging_point, time):
    """
    Constraint for initial value of energy

    Parameters
    ----------
    model :
    charging_point :
    time :

    Returns
    -------

    """
    initial_lower_band = model.lower_bound_ev[charging_point, time]
    initial_upper_band = model.upper_bound_ev[charging_point, time]
    return (
        model.energy_level_ev[charging_point, time]
        == (initial_lower_band + initial_upper_band) / 2
    )


def fixed_energy_level_tes(model, hp, time):
    """

    Parameters
    ----------
    model :
    hp :
    time :

    Returns
    -------

    """
    return (
        model.energy_level_tes[hp, time]
        == model.tes.loc[hp, "capacity"] * model.tes.loc[hp, "state_of_charge_initial"]
    )


def final_energy_level(model, comp_type, comp, time):
    """
    Constraint for final value of energy in last iteration

    Parameters
    ----------
    model :
    comp_type :
    comp :
    time :

    Returns
    -------

    """
    return (
        getattr(model, f"energy_level_{comp_type.lower()}")[comp, time]
        == getattr(model, f"energy_level_beginning_{comp_type.lower()}")[comp]
        + getattr(model, f"energy_level_end_{comp_type.lower()}")[comp]
    )


def final_energy_level_ev(model, charging_point, time):
    """
    Constraint for final value of energy in last iteration

    Parameters
    ----------
    model :
    charging_point :
    time :

    Returns
    -------

    """
    return final_energy_level(model, "ev", charging_point, time)


def final_energy_level_tes(model, heat_pump, time):
    """
    Constraint for final value of energy in last iteration

    Parameters
    ----------
    model :
    heat_pump :
    time :

    Returns
    -------

    """
    return final_energy_level(model, "tes", heat_pump, time)


def final_charging_power(model, comp_type, comp, time):
    """
    Constraint for final value of charging power, setting it to 0

    Parameters
    ----------
    model :
    comp_type :
    comp :
    time :

    Returns
    -------

    """
    return getattr(model, f"charging_{comp_type.lower()}")[comp, time] == 0


def final_charging_power_ev(model, charging_point, time):
    """
    Constraint for final value of charging power, setting it to 0

    Parameters
    ----------
    model :
    charging_point :
    time :

    Returns
    -------

    """
    return final_charging_power(model, "ev", charging_point, time)


def final_charging_power_hp(model, heat_pump, time):
    """
    Constraint for final value of charging power, setting it to 0

    Parameters
    ----------
    model :
    heat_pump :
    time :

    Returns
    -------

    """
    return final_charging_power(model, "hp", heat_pump, time)


def final_charging_power_tes(model, heat_pump, time):
    """
    Constraint for final value of charging power, setting it to 0

    Parameters
    ----------
    model :
    heat_pump :
    time :

    Returns
    -------

    """
    return final_charging_power(model, "tes", heat_pump, time)


def initial_charging_power(model, comp_type, comp, time):
    """
    Constraint for initial value of charging power

    Parameters
    ----------
    model :
    comp_type :
    comp :
    time :

    Returns
    -------

    """
    return (
        getattr(model, f"charging_{comp_type.lower()}")[comp, time]
        == getattr(model, f"charging_initial_{comp_type.lower()}")[comp]
        + getattr(model, f"slack_initial_charging_pos_{comp_type.lower()}")[comp]
        - getattr(model, f"slack_initial_charging_neg_{comp_type.lower()}")[comp]
    )


def initial_charging_power_ev(model, charging_point, time):
    """
    Constraint for initial value of charging power

    Parameters
    ----------
    model :
    charging_point :
    time :

    Returns
    -------

    """
    return initial_charging_power(model, "ev", charging_point, time)


def initial_charging_power_hp(model, heat_pump, time):
    """
    Constraint for initial value of charging power

    Parameters
    ----------
    model :
    heat_pump :
    time :

    Returns
    -------

    """
    return initial_charging_power(model, "hp", heat_pump, time)


def initial_charging_power_tes(model, heat_pump, time):
    """
    Constraint for initial value of charging power

    Parameters
    ----------
    model :
    heat_pump :
    time :

    Returns
    -------

    """
    return initial_charging_power(model, "tes", heat_pump, time)


def aggregated_power(model, time):
    """
    Constraint aggregating the power of bess and emob charging?

    Parameters
    ----------
    model :
    time :

    Returns
    -------

    """
    if hasattr(model, "optimized_storage_set"):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, "flexible_charging_points_set"):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.grid_power_flexible[time] == -sum(
        model.charging[storage, time] for storage in relevant_storage_units
    ) + sum(model.charging_ev[cp, time] for cp in relevant_charging_points)


def load_factor_min(model, time):
    """
    Constraint that describes the minimum load factor.

    Parameters
    ----------
    model :
    time :

    Returns
    -------

    """
    if hasattr(model, "optimized_storage_set"):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, "flexible_charging_points_set"):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.min_load_factor <= model.residual_load[time] + sum(
        model.charging[storage, time] for storage in relevant_storage_units
    ) - sum(model.charging_ev[cp, time] for cp in relevant_charging_points) + sum(
        model.curtailment_load[bus, time] - model.curtailment_feedin[bus, time]
        for bus in model.bus_set
    )


def load_factor_max(model, time):
    """
    Constraint that describes the maximum load factor.

    Parameters
    ----------
    model :
    time :

    Returns
    -------

    """
    if hasattr(model, "optimized_storage_set"):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, "flexible_charging_points_set"):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    return model.max_load_factor >= model.residual_load[time] + sum(
        model.charging[storage, time] for storage in relevant_storage_units
    ) - sum(model.charging_ev[cp, time] for cp in relevant_charging_points) + sum(
        model.curtailment_load[bus, time] - model.curtailment_feedin[bus, time]
        for bus in model.bus_set
    )


def minimize_max_residual_load(model):
    """
    Objective minimizing extreme load factors

    Parameters
    ----------
    model :

    Returns
    -------

    """
    slack_charging, slack_energy = extract_slack_charging(model)
    ev_curtailment, hp_curtailment = extract_curtailment_of_flexible_components(model)
    return (
        -model.delta_min * model.min_load_factor
        + model.delta_max * model.max_load_factor
        + sum(
            model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time]
            for bus in model.bus_set
            for time in model.time_set
        )
        + 0.5 * ev_curtailment
        + hp_curtailment
        + 1000 * (slack_charging + slack_energy)
    )


def minimize_curtailment(model):
    """
    Objective minimizing required curtailment.

    !!! CAREFUL: Solution ambiguous.

    Parameters
    ----------
    model :

    Returns
    -------

    """
    slack_charging, slack_energy = extract_slack_charging(model)
    ev_curtailment, hp_curtailment = extract_curtailment_of_flexible_components(model)
    if hasattr(model, "flexible_charging_points_set"):
        return (
            sum(
                model.curtailment_load[bus, time]
                + model.curtailment_feedin[bus, time]
                + 0.5 * model.curtailment_ev[bus, time]
                for bus in model.bus_set
                for time in model.time_set
            )
            + 0.5 * ev_curtailment
            + hp_curtailment
            + 1000 * (slack_charging + slack_energy)
        )


def minimize_energy_level(model):
    """
    Objective minimizing energy level of grid while also minimizing necessary
    curtailment

    Parameters
    ----------
    model :

    Returns
    -------

    """
    slack_charging, slack_energy = extract_slack_charging(model)
    ev_curtailment, hp_curtailment = extract_curtailment_of_flexible_components(model)
    return (
        (
            sum(
                model.curtailment_load[bus, time]
                + model.curtailment_feedin[bus, time]
                + 0.5 * model.curtailment_ev[bus, time]
                for bus in model.bus_set
                for time in model.time_set
            )
            + 0.5 * ev_curtailment
            + hp_curtailment
        )
        * 1e6
        + sum(model.grid_power_flexible[time] for time in model.time_set)
        + 1000 * (slack_charging + slack_energy)
    )


def maximize_energy_level(model):
    """
    Objective maximizing energy level of grid while also minimizing necessary
    curtailment

    Parameters
    ----------
    model :

    Returns
    -------

    """
    slack_charging, slack_energy = extract_slack_charging(model)
    ev_curtailment, hp_curtailment = extract_curtailment_of_flexible_components(model)
    return (
        (
            sum(
                model.curtailment_load[bus, time]
                + model.curtailment_feedin[bus, time]
                + 0.5 * model.curtailment_ev[bus, time]
                for bus in model.bus_set
                for time in model.time_set
            )
            + 0.5 * ev_curtailment
            + hp_curtailment
        )
        * 1e6
        - sum(model.grid_power_flexible[time] for time in model.time_set)
        + 1000 * (slack_charging + slack_energy)
    )


def grid_residual_load(model, time):
    """

    Parameters
    ----------
    model :
    time :

    Returns
    -------

    """
    if hasattr(model, "optimized_storage_set"):
        relevant_storage_units = model.optimized_storage_set
    else:
        relevant_storage_units = []
    if hasattr(model, "flexible_charging_points_set"):
        relevant_charging_points = model.flexible_charging_points_set
    else:
        relevant_charging_points = []
    if hasattr(model, "flexible_heat_pumps_set"):
        relevant_heat_pumps = model.flexible_heat_pumps_set
    else:
        relevant_heat_pumps = []
    return model.grid_residual_load[time] == model.residual_load[time] + sum(
        model.charging[storage, time] for storage in relevant_storage_units
    ) - sum(model.charging_ev[cp, time] for cp in relevant_charging_points) - sum(
        model.charging_hp[hp, time] for hp in relevant_heat_pumps
    )  # + \
    # sum(model.curtailment_load[bus, time] -
    #     model.curtailment_feedin[bus, time] for bus in model.bus_set)


def minimize_residual_load(model):
    """
    Objective minimizing curtailment and squared residual load

    Parameters
    ----------
    model :

    Returns
    -------

    """
    slack_charging, slack_energy = extract_slack_charging(model)
    ev_curtailment, hp_curtailment = extract_curtailment_of_flexible_components(model)
    return (
        1e-5 * sum(model.grid_residual_load[time] ** 2 for time in model.time_set)
        + sum(
            1e-2
            * (model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time])
            + 1000 * (model.slack_v_pos[bus, time] + model.slack_v_neg[bus, time])
            for bus in model.bus_set
            for time in model.time_set
        )
        + 1e-2 * (0.5 * ev_curtailment + hp_curtailment)
        + 1000 * (slack_charging + slack_energy)
        + 1000
        * sum(
            model.slack_p_cum_pos[branch, time] + model.slack_p_cum_neg[branch, time]
            for branch in model.branch_set
            for time in model.time_set
        )
    )


def minimize_loading(model):
    """
    Objective minimizing curtailment and squared term of component loading

    Parameters
    ----------
    model :

    Returns
    -------

    """
    slack_charging, slack_energy = extract_slack_charging(model)
    ev_curtailment, hp_curtailment = extract_curtailment_of_flexible_components(model)
    return (
        1e-5
        * sum(
            (
                model.p_cum[branch, time]
                / (
                    model.power_factors[branch, time]
                    * model.branches.loc[branch, model.pars["s_nom"]]
                )
            )
            ** 2
            for branch in model.branch_set
            for time in model.time_set
        )
        + sum(
            1e-2
            * (model.curtailment_load[bus, time] + model.curtailment_feedin[bus, time])
            + 1000 * (model.slack_v_pos[bus, time] + model.slack_v_neg[bus, time])
            for bus in model.bus_set
            for time in model.time_set
        )
        + 1e-2 * (0.5 * ev_curtailment + hp_curtailment)
        + 1000 * (slack_charging + slack_energy)
        + 1000
        * sum(
            model.slack_p_cum_pos[branch, time] + model.slack_p_cum_neg[branch, time]
            for branch in model.branch_set
            for time in model.time_set
        )
    )


def extract_curtailment_of_flexible_components(model):
    """

    Parameters
    ----------
    model :

    Returns
    -------

    """
    if hasattr(model, "flexible_charging_points_set"):
        ev_curtailment = sum(
            model.curtailment_ev[bus, time]
            for bus in model.bus_set
            for time in model.time_set
        )
    else:
        ev_curtailment = 0
    if hasattr(model, "flexible_heat_pumps_set"):
        hp_curtailment = sum(
            model.curtailment_hp[bus, time]
            for bus in model.bus_set
            for time in model.time_set
        )
    else:
        hp_curtailment = 0
    return ev_curtailment, hp_curtailment


def extract_slack_charging(model):
    """

    Parameters
    ----------
    model :

    Returns
    -------

    """
    if hasattr(model, "slack_initial_charging_pos"):
        slack_charging = sum(
            model.slack_initial_charging_pos_ev[cp]
            + model.slack_initial_charging_neg_ev[cp]
            for cp in model.flexible_charging_points_set
        )
    else:
        slack_charging = 0
    if hasattr(model, "slack_initial_energy_pos"):
        slack_energy = sum(
            model.slack_initial_energy_pos_ev[cp]
            + model.slack_initial_energy_neg_ev[cp]
            for cp in model.flexible_charging_points_set
        )
    else:
        slack_energy = 0
    return slack_charging, slack_energy


def combine_results_for_grid(feeders, grid_id, res_dir, res_name):
    """

    Parameters
    ----------
    feeders :
    grid_id :
    res_dir :
    res_name :

    Returns
    -------

    """
    res_grid = pd.DataFrame()
    for feeder_id in feeders:
        res_feeder = pd.DataFrame()
        for i in range(14):
            try:
                res_feeder_tmp = pd.read_csv(
                    res_dir
                    + "/{}/{}/{}_{}_{}_{}.csv".format(
                        grid_id, feeder_id, res_name, grid_id, feeder_id, i
                    ),
                    index_col=0,
                    parse_dates=True,
                )
                res_feeder = pd.concat([res_feeder, res_feeder_tmp], sort=False)
            except ImportError:
                logger.info(
                    f"Results for feeder {feeder_id} in grid "
                    f"{grid_id} could not be loaded."
                )
        try:
            res_grid = pd.concat([res_grid, res_feeder], axis=1, sort=False)
        except ValueError:
            logger.info(f"Feeder {feeder_id} not added")
    res_grid = res_grid.loc[~res_grid.index.duplicated(keep="last")]
    return res_grid
