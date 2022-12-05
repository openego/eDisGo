"""
This module provides tools to convert eDisGo representation of the network
topology and timeseries to PowerModels network data format and to retrieve results from
PowerModels OPF in PowerModels network data format to eDisGo representation.
Call :func:`to_powermodels` to retrieve the PowerModels network container and
:func:`from_powermodels` to write OPF results to edisgo object.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import pypsa

from edisgo.flex_opt.costs import line_expansion_costs
from edisgo.tools.tools import calculate_impedance_for_parallel_components

logger = logging.getLogger(__name__)


class Etrago:  # ToDo: delete as soon as etrago class is implemented
    def __init__(self):
        self.renewables_curtailment = pd.Series(dtype="float64")
        self.storage_units_active_power = pd.Series(dtype="float64")
        self.dsm_active_power = pd.Series(dtype="float64")
        self.electromobility_active_power = pd.Series(dtype="float64")
        self.heat_pump_rural_active_power = pd.Series(dtype="float64")
        self.heat_central_active_power = pd.Series(dtype="float64")


def to_powermodels(
    edisgo_object,
    flexible_cps=None,
    flexible_hps=None,
    flexible_loads=None,
    opt_version=1,
    opt_flex=None,
):
    """
    Converts eDisGo representation of the network topology and timeseries to
    PowerModels network data format.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_loads: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    opt_version: Int
        Version of optimization models to choose from. For more information see MA.
        Must be one of [1, 2].
    opt_flex: list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]

    Returns
    -------
    pm: dict
        Dictionary that contains all network data in PowerModels network data
        format.
    """
    tol = 1e-4
    if opt_flex is None:
        opt_flex = ["curt"]  # ToDo: adden falls nicht drin
    if flexible_cps is None:
        flexible_cps = []
    if flexible_hps is None:
        flexible_hps = []
    if flexible_loads is None:
        flexible_loads = []
    # Check if names of flexible loads for optimization are supplied
    for (flex, loads, text) in [
        ("cp", flexible_cps, "flexible charging parks"),
        ("hp", flexible_hps, "flexible heatpumps"),
        ("dsm", flexible_loads, "flexible loads"),
        ("storage", edisgo_object.topology.storage_units_df, "storage units"),
    ]:
        if (flex in opt_flex) & (len(loads) == 0):
            logger.warning(
                " No {} found in network, {} will not be optimized.".format(text, text)
            )
            opt_flex.remove(flex)
        elif (flex not in opt_flex) & (len(loads) != 0):
            logger.warning(
                " {} found in network, {} will be optimized.".format(text, text)
            )
            opt_flex.append(flex)
    hv_flex_dict = dict()
    # Sorts buses such that bus0 is always the upstream bus
    edisgo_object.topology.sort_buses()
    # Calculate line costs
    costs = line_expansion_costs(edisgo_object).drop(columns="voltage_level")
    # convert eDisGo object to pypsa network structure
    psa_net = edisgo_object.to_pypsa()
    # add line costs to psa_net
    psa_net.lines = psa_net.lines.merge(costs, left_index=True, right_index=True)
    psa_net.lines.capital_cost = (
        psa_net.lines.costs_earthworks + psa_net.lines.costs_cable
    )
    # aggregate parallel transformers
    aggregate_parallel_transformers(psa_net)
    psa_net.transformers.capital_cost = edisgo_object.config._data[
        "costs_transformers"
    ]["lv"]
    # calculate per unit values
    pypsa.pf.calculate_dependent_values(psa_net)
    # build PowerModels structure
    pm = _init_pm()
    timesteps = len(psa_net.snapshots)  # number of considered timesteps
    pm["name"] = "ding0_{}_t_{}".format(edisgo_object.topology.id, timesteps)
    pm["time_elapsed"] = int(
        (psa_net.snapshots[1] - psa_net.snapshots[0]).seconds / 3600
    )  # length of timesteps in hours
    pm["baseMVA"] = 1
    pm["source_version"] = 2
    pm["flexibilities"] = opt_flex
    _build_bus(psa_net, pm)
    _build_gen(edisgo_object, psa_net, pm, tol)
    _build_branch(psa_net, pm, tol)
    if len(edisgo_object.topology.storage_units_df) > 0:
        _build_battery_storage(edisgo_object, psa_net, pm)
    if len(psa_net.loads) > 0:
        _build_load(psa_net, pm, flexible_cps, flexible_hps, flexible_loads, tol)
    else:
        logger.warning("No loads found in network.")
    if len(flexible_cps) > 0:
        _build_electromobility(edisgo_object, psa_net, pm, flexible_cps, tol)
    if len(flexible_hps) > 0:
        _build_heatpump(psa_net, pm, edisgo_object, flexible_hps, tol)
    if "hp" in opt_flex:
        _build_heat_storage(psa_net, pm, edisgo_object, flexible_hps)
    if len(flexible_loads) > 0:
        _build_dsm(edisgo_object, psa_net, pm, flexible_loads, tol)
    if (opt_version == 1) | (opt_version == 2):
        edisgo_object.etrago.renewables_curtailment[
            edisgo_object.etrago.renewables_curtailment < tol
        ] = 0
        edisgo_object.etrago.storage_units_active_power[
            edisgo_object.etrago.storage_units_active_power < tol
        ] = 0
        edisgo_object.etrago.electromobility_active_power[
            edisgo_object.etrago.electromobility_active_power < tol
        ] = 0
        edisgo_object.etrago.heat_pump_rural_active_power[
            edisgo_object.etrago.heat_pump_rural_active_power < tol
        ] = 0
        edisgo_object.etrago.heat_central_active_power[
            edisgo_object.etrago.heat_central_active_power < tol
        ] = 0
        edisgo_object.etrago.dsm_active_power[
            edisgo_object.etrago.dsm_active_power < tol
        ] = 0
        hv_flex_dict = {
            "curt": edisgo_object.etrago.renewables_curtailment,
            "storage": edisgo_object.etrago.storage_units_active_power,
            "cp": edisgo_object.etrago.electromobility_active_power,
            "hp": (
                edisgo_object.etrago.heat_pump_rural_active_power
                + edisgo_object.etrago.heat_central_active_power
            ),
            "dsm": edisgo_object.etrago.dsm_active_power,
        }
        try:
            _build_HV_requirements(
                psa_net, pm, opt_flex, flexible_cps, flexible_hps, hv_flex_dict
            )
        except IndexError:
            logger.warning(
                "Etrago component of eDisGo object has no entries."
                " Changing optimization version to '3' (without high voltage"
                " requirements)."
            )
            opt_version = 3

    pm["opt_version"] = opt_version

    _build_timeseries(
        psa_net,
        pm,
        edisgo_object,
        flexible_cps,
        flexible_hps,
        flexible_loads,
        opt_flex,
        hv_flex_dict,
    )
    return pm


def from_powermodels(
    edisgo_object,
    pm_results,
    save_heat_storage=False,
    save_gen_slack=False,
    save_hv_slack=False,
    path="",
):
    """
    Converts results from optimization in PowerModels network data format to eDisGo data
    format and updates timeseries values of flexibilities on eDisGo object.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    pm_results: dict or str
        Dictionary or path to json file that contains all optimization results in
        PowerModels network data format.
    save_heat_storage: bool
        Indicates whether to save results of heat storage variables from the
        optimization to csv file in the current working directory. Set parameter
        "path" to change the directory the file is saved to.
        directory.
            Default: False
    save_gen_slack: bool
        Indicates whether to save results of slack generator variables from the
        optimization to csv file in the current working directory. Set parameter
        "path" to change the directory the file is saved to.
        Default: False
    save_hv_slack: bool
        Indicates whether to save results of slack variables for high voltage
        requirements (sum, minimal and maximal and mean deviation) from the optimization
        to csv file in the current working directory. Set parameter "path" to change the
        directory the file is saved to.
        Default: False
    path : str
        Directory the csv file is saved to. Per default it takes the current
        working directory.
    """

    abs_path = os.path.abspath(path)

    if type(pm_results) == str:
        with open(pm_results) as f:
            pm = json.loads(json.load(f))
    elif type(pm_results) == dict:
        pm = pm_results
    else:
        raise ValueError(
            "Parameter 'pm_results' must be either dictionary or path " "to json file."
        )

    flex_dicts = {
        "curt": ["gen_nd", "pgc"],
        "hp": ["heatpumps", "php"],
        "cp": ["electromobility", "pcp"],
        "storage": ["storage", "ps"],
        "dsm": ["dsm", "pdsm"],
    }

    timesteps = pm["nw"].keys()

    for flexibility in pm_results["nw"]["1"]["flexibilities"]:
        flex, variable = flex_dicts[flexibility]
        names = [
            pm["nw"]["1"][flex][flex_comp]["name"]
            for flex_comp in list(pm["nw"]["1"][flex].keys())
        ]
        data = [
            [
                pm["nw"][t][flex][flex_comp][variable]
                for flex_comp in list(pm["nw"]["1"][flex].keys())
            ]
            for t in timesteps
        ]
        results = pd.DataFrame(index=timesteps, columns=names, data=data)
        if flex in ["gen_nd"]:
            if variable != "qgs":
                edisgo_object.timeseries._generators_active_power.loc[:, names] = (
                    edisgo_object.timeseries.generators_active_power.loc[
                        :, names
                    ].values
                    - results[names].values
                )
            else:
                edisgo_object.timeseries._generators_reactive_power.loc[:, names] = (
                    edisgo_object.timeseries.generators_reactive_power.loc[
                        :, names
                    ].values
                    - results[names].values
                )
        elif flex in ["dsm", "heatpumps", "electromobility"]:
            edisgo_object.timeseries._loads_active_power.loc[:, names] = results[
                names
            ].values
        elif flex == "storage":
            edisgo_object.timeseries._storage_units_active_power.loc[
                :, names
            ] = results[names].values

    edisgo_object.set_time_series_reactive_power_control()

    # Check values of slack variables for HV requirement constraint
    names = [
        pm["nw"]["1"]["HV_requirements"][flex]["flexibility"]
        for flex in list(pm["nw"]["1"]["HV_requirements"].keys())
    ]
    data = [
        [
            pm["nw"][t]["HV_requirements"][flex]["phvs"]
            for flex in list(pm["nw"]["1"]["HV_requirements"].keys())
        ]
        for t in timesteps
    ]
    df = pd.DataFrame(
        index=edisgo_object.timeseries.timeindex,
        columns=names,
        data=data,
    )
    df2 = pd.DataFrame(
        columns=[
            "Highest negative error",
            "Highest positive error",
            "Mean absolute error",
            "Sum absolute error",
        ],
        index=names,
    )
    # highest negative error
    df2["Highest negative error"] = df.min()
    # highest positive error
    df2["Highest positive error"] = df.max()
    # mean absolute error
    df2["Mean absolute error"] = df.abs().sum() / len(df)
    # sum of absolut error -> an edisgo übergeben
    df2["Sum absolute error"] = df.abs().sum()
    if (df2["Highest positive error"].values > 0.00001).any():  # ToDo: value of error
        logger.warning("Highest absolute error of HV slack variables exceed 0.00001")
    # ToDo: write sum absolute error to edisgo object
    if save_hv_slack:
        df2.to_csv(os.path.join(abs_path, "hv_requirements_slack.csv"))
    if save_gen_slack:
        df = pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=["pg", "qg"]
        )
        for gen in list(pm["nw"]["1"]["gen_slack"].keys()):
            df["pg"] = [pm["nw"][t]["gen_slack"][gen]["pgs"] for t in timesteps]
            df["qg"] = [pm["nw"][t]["gen_slack"][gen]["qgs"] for t in timesteps]
        df.to_csv(os.path.join(abs_path, "gen_slack.csv"))
    if save_heat_storage:
        for variable in ["phs", "hse"]:
            names = [
                pm["nw"]["1"]["heat_storage"][hs]["name"]
                for hs in list(pm["nw"]["1"]["heat_storage"].keys())
            ]
            data = [
                [
                    pm["nw"][t]["heat_storage"][hs][variable]
                    for hs in list(pm["nw"]["1"]["heat_storage"].keys())
                ]
                for t in timesteps
            ]
            name = {"phs": "p", "hse": "e"}
            pd.DataFrame(
                index=edisgo_object.timeseries.timeindex, columns=names, data=data
            ).to_csv(
                os.path.join(abs_path, str("heat_storage_" + name[variable] + ".csv"))
            )
        names = [
            pm["nw"]["1"]["heatpumps"][hp]["name"]
            for hp in list(pm["nw"]["1"]["heatpumps"].keys())
        ]
        data = [
            [
                pm["nw"][t]["heatpumps"][hp]["phps"]
                for hp in list(pm["nw"]["1"]["heatpumps"].keys())
            ]
            for t in timesteps
        ]
        pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=names, data=data
        ).to_csv(os.path.join(abs_path, "heat_pump_p_slack.csv"))


def _init_pm():
    # init empty PowerModels dictionary
    pm = {
        "gen": dict(),
        "gen_nd": dict(),
        "gen_slack": dict(),
        "branch": dict(),
        "bus": dict(),
        "dcline": dict(),
        "load": dict(),
        "storage": dict(),
        "switch": dict(),
        "electromobility": dict(),
        "heatpumps": dict(),
        "heat_storage": dict(),
        "dsm": dict(),
        "HV_requirements": dict(),
        "baseMVA": 1,
        "source_version": 2,
        "shunt": dict(),
        "sourcetype": "eDisGo",
        "per_unit": True,
        "name": "name",
        "time_series": {
            "gen": dict(),
            "load": dict(),
            "storage": dict(),
            "electromobility": dict(),
            "heatpumps": dict(),
            "dsm": dict(),
            "HV_requirements": dict(),
            "num_steps": int,
        },
    }
    return pm


def _build_bus(psa_net, pm):
    """
    Builds bus dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    """
    bus_types = ["PQ", "PV", "Slack", "None"]
    bus_types_int = np.array(
        [bus_types.index(b_type) + 1 for b_type in psa_net.buses["control"].values],
        dtype=int,
    )
    grid_level = {20: "mv", 10: "mv", 0.4: "lv"}
    v_max = [min(val, 1.1) for val in psa_net.buses["v_mag_pu_max"].values]
    v_min = [max(val, 0.9) for val in psa_net.buses["v_mag_pu_min"].values]
    for bus_i in np.arange(len(psa_net.buses.index)):
        pm["bus"][str(bus_i + 1)] = {
            "index": bus_i + 1,
            "bus_i": bus_i + 1,
            "zone": 1,
            "bus_type": bus_types_int[bus_i],
            "vmax": v_max[bus_i],
            "vmin": v_min[bus_i],
            "va": 0,
            "vm": 1,
            "base_kv": psa_net.buses.v_nom[bus_i],
            "grid_level": grid_level[psa_net.buses.v_nom[bus_i]],
        }


def _build_gen(edisgo_obj, psa_net, pm, tol):
    """
    Builds dispatchable and non-dispatchable generator dictionaries in PowerModels
    network data format and adds both to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    """
    # Divide in slack, dispatchable and non-dispatchable generator sets
    gen_slack = psa_net.generators.loc[psa_net.generators.index == "Generator_slack"]
    gen_nondisp = psa_net.generators.loc[
        (psa_net.generators.index.str.contains("solar"))
        | (psa_net.generators.index.str.contains("wind"))
    ]
    gen_disp = psa_net.generators.drop(
        np.concatenate((gen_nondisp.index, gen_slack.index))
    )
    # determine slack buses through slack generators
    slack_gens_bus = gen_slack.bus.values
    for bus in slack_gens_bus:
        pm["bus"][str(_mapping(psa_net, bus))]["bus_type"] = 3

    for gen_i in np.arange(len(gen_disp.index)):
        idx_bus = _mapping(psa_net, gen_disp.bus[gen_i])
        # retrieve power factor and sign from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "gen")
        q = [
            sign * np.tan(np.arccos(pf)) * gen_disp.p_nom[gen_i],
            sign * np.tan(np.arccos(pf)) * gen_disp.p_nom_min[gen_i],
        ]
        psa_net.generators_t.p_set[gen_disp.index[gen_i]].loc[
            psa_net.generators_t.p_set[gen_disp.index[gen_i]] < tol
        ] = 0
        psa_net.generators_t.q_set[gen_disp.index[gen_i]].loc[
            psa_net.generators_t.q_set[gen_disp.index[gen_i]] > -tol
        ] = 0
        pm["gen"][str(gen_i + 1)] = {
            "pg": psa_net.generators_t.p_set[gen_disp.index[gen_i]][0],
            "qg": psa_net.generators_t.q_set[gen_disp.index[gen_i]][0],
            "pmax": gen_disp.p_nom[gen_i],
            "pmin": gen_disp.p_nom_min[gen_i],
            "qmax": max(q),
            "qmin": min(q),
            "vg": 1,
            "mbase": gen_disp.p_nom[gen_i],
            "gen_bus": idx_bus,
            "gen_status": 1,
            "index": gen_i + 1,
        }

    for gen_i in np.arange(len(gen_nondisp.index)):
        idx_bus = _mapping(psa_net, gen_nondisp.bus[gen_i])
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "gen")
        q = [
            sign * np.tan(np.arccos(pf)) * gen_nondisp.p_nom[gen_i],
            sign * np.tan(np.arccos(pf)) * gen_nondisp.p_nom_min[gen_i],
        ]
        psa_net.generators_t.p_set[gen_nondisp.index[gen_i]].loc[
            psa_net.generators_t.p_set[gen_nondisp.index[gen_i]] < tol
        ] = 0
        psa_net.generators_t.q_set[gen_nondisp.index[gen_i]].loc[
            psa_net.generators_t.q_set[gen_nondisp.index[gen_i]] > -tol
        ] = 0
        pm["gen_nd"][str(gen_i + 1)] = {
            "pg": psa_net.generators_t.p_set[gen_nondisp.index[gen_i]][0],
            "qg": psa_net.generators_t.q_set[gen_nondisp.index[gen_i]][0],
            "pmax": gen_nondisp.p_nom[gen_i],
            "pmin": gen_nondisp.p_nom_min[gen_i],
            "qmax": max(q),
            "qmin": min(q),
            "P": 0,
            "Q": 0,
            "vg": 1,
            "pf": pf,
            "sign": sign,
            "mbase": gen_nondisp.p_nom[gen_i],
            "gen_bus": idx_bus,
            "gen_status": 1,
            "name": gen_nondisp.index[gen_i],
            "index": gen_i + 1,
        }

    for gen_i in np.arange(len(gen_slack.index)):
        idx_bus = _mapping(psa_net, gen_slack.bus[gen_i])
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "gen")
        q = [
            sign * np.tan(np.arccos(pf)) * gen_slack.p_nom[gen_i],
            sign * np.tan(np.arccos(pf)) * gen_slack.p_nom_min[gen_i],
        ]
        pm["gen_slack"][str(gen_i + 1)] = {
            "pg": psa_net.generators_t.p_set[gen_slack.index[gen_i]][0],
            "qg": psa_net.generators_t.q_set[gen_slack.index[gen_i]][0],
            "pmax": gen_slack.p_nom[gen_i],
            "pmin": gen_slack.p_nom_min[gen_i],
            "qmax": max(q),
            "qmin": min(q),
            "P": 0,
            "Q": 0,
            "vg": 1,
            "mbase": gen_slack.p_nom[gen_i],
            "gen_bus": idx_bus,
            "name": gen_slack.index[gen_i],
            "gen_status": 1,
            "index": gen_i + 1,
        }


def _build_branch(psa_net, pm, tol):
    """
    Builds branch dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    """
    branches = pd.concat([psa_net.lines, psa_net.transformers])
    transformer = ~branches.tap_ratio.isna()
    tap = branches.tap_ratio.fillna(1)
    shift = branches.phase_shift.fillna(0)
    for branch_i in np.arange(len(branches.index)):
        idx_f_bus = _mapping(psa_net, branches.bus0[branch_i])
        idx_t_bus = _mapping(psa_net, branches.bus1[branch_i])
        if branches.r_pu[branch_i] < tol:
            branches.r_pu.loc[branch_i] = 0
        if branches.x_pu[branch_i] < tol:
            branches.x_pu.loc[branch_i] = 0
        if branches.length.fillna(1)[branch_i] < tol:
            branches.length.fillna(1).loc[branch_i] = 0
        if branches.capital_cost[branch_i] < tol:
            branches.capital_cost.loc[branch_i] = 0
        pm["branch"][str(branch_i + 1)] = {
            "name": branches.index[branch_i],
            "br_r": branches.r_pu[branch_i],
            "br_x": branches.x_pu[branch_i],
            "f_bus": idx_f_bus,
            "t_bus": idx_t_bus,
            "g_to": branches.g_pu[branch_i] / 2,
            "g_fr": branches.g_pu[branch_i] / 2,
            "b_to": branches.b_pu[branch_i] / 2,
            "b_fr": branches.b_pu[branch_i] / 2,
            "shift": shift[branch_i],
            "br_status": 1.0,
            "rate_a": branches.s_nom[branch_i].real,
            "rate_b": 250,
            "rate_c": 250,
            "angmin": -np.pi / 6,
            "angmax": np.pi / 6,
            "transformer": bool(transformer[branch_i]),
            "tap": tap[branch_i],
            "length": branches.length.fillna(1)[branch_i],
            "cost": branches.capital_cost[branch_i],
            "index": branch_i + 1,
        }


def _build_load(psa_net, pm, flexible_cps, flexible_hps, flexible_loads, tol):
    """
    Builds load dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_loads: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    """
    flex_loads = np.concatenate((flexible_hps, flexible_cps, flexible_loads))
    if len(flex_loads) == 0:
        loads_df = psa_net.loads
    else:
        loads_df = psa_net.loads.drop(flex_loads)
    for load_i in np.arange(len(loads_df.index)):
        idx_bus = _mapping(psa_net, loads_df.bus[load_i])
        p_d = psa_net.loads_t.p_set[loads_df.index[load_i]]
        q_d = psa_net.loads_t.q_set[loads_df.index[load_i]]
        p_d.loc[p_d < tol] = 0
        q_d.loc[q_d < tol] = 0
        pm["load"][str(load_i + 1)] = {
            "pd": p_d[0],
            "qd": q_d[0],
            "load_bus": idx_bus,
            "status": True,
            "index": load_i + 1,
        }


def _build_battery_storage(edisgo_obj, psa_net, pm):
    """
    Builds (battery) storage  dictionary in PowerModels network data format and adds
    it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    """
    for stor_i in np.arange(len(psa_net.storage_units.index)):
        idx_bus = _mapping(psa_net, psa_net.storage_units.bus[stor_i])
        # retrieve power factor from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "storage")
        e_max = (
            psa_net.storage_units.p_nom[stor_i]
            * psa_net.storage_units.max_hours[stor_i]
        )
        pm["storage"][str(stor_i + 1)] = {
            "r": 0,
            "x": 0,
            "p_loss": 0,
            "q_loss": 0,
            "pf": pf,
            "sign": sign,
            "ps": psa_net.storage_units_t.p_set[psa_net.storage_units.index[stor_i]][0],
            "qs": psa_net.storage_units_t.q_set[psa_net.storage_units.index[stor_i]][0],
            "pmax": psa_net.storage_units.p_nom[stor_i],
            "pmin": -psa_net.storage_units.p_nom[stor_i],
            "qmax": np.tan(np.arccos(pf)) * psa_net.storage_units.p_nom[stor_i],
            "qmin": -np.tan(np.arccos(pf)) * psa_net.storage_units.p_nom[stor_i],
            "energy": psa_net.storage_units.state_of_charge_initial[stor_i] * e_max,
            "energy_rating": e_max,
            "thermal_rating": 1,  # TODO unbegrenzt
            "charge_rating": psa_net.storage_units.p_nom[stor_i],
            "discharge_rating": psa_net.storage_units.p_nom[stor_i],
            "charge_efficiency": 1,
            "discharge_efficiency": 1,
            "storage_bus": idx_bus,
            "name": psa_net.storage_units.index[stor_i],
            "status": True,
            "index": stor_i + 1,
        }


def _build_electromobility(edisgo_obj, psa_net, pm, flexible_cps, tol):
    """
    Builds electromobility dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    """
    emob_df = psa_net.loads.loc[flexible_cps]
    flex_bands_df = edisgo_obj.electromobility.flexibility_bands
    for cp_i in np.arange(len(emob_df.index)):
        idx_bus = _mapping(psa_net, emob_df.bus[cp_i])
        # retrieve power factor and sign from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "cp")
        q = (
            sign
            * np.tan(np.arccos(pf))
            * flex_bands_df["upper_power"][emob_df.index[cp_i]][0]
        )
        p_max = flex_bands_df["upper_power"][emob_df.index[cp_i]]
        e_min = flex_bands_df["lower_energy"][emob_df.index[cp_i]]
        e_max = flex_bands_df["upper_energy"][emob_df.index[cp_i]]
        p_max.loc[p_max < tol] = 0
        e_min.loc[e_min < tol] = 0
        e_max.loc[e_max < tol] = 0
        pm["electromobility"][str(cp_i + 1)] = {
            "pd": 0,
            "qd": 0,
            "pf": pf,
            "sign": sign,
            "p_min": 0,
            "p_max": p_max[0],
            "q_min": min(q, 0),
            "q_max": max(q, 0),
            "e_min": e_min[0],
            "e_max": e_max[0],
            "cp_bus": idx_bus,
            "name": emob_df.index[cp_i],
            "index": cp_i + 1,
        }


def _build_heatpump(psa_net, pm, edisgo_obj, flexible_hps, tol):
    """
    Builds heat pump dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    edisgo_obj : :class:`~.EDisGo`
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.

    """
    heat_df = psa_net.loads.loc[flexible_hps]  # electric load
    heat_df2 = edisgo_obj.heat_pump.heat_demand_df[flexible_hps]  # thermal load
    for hp_i in np.arange(len(heat_df.index)):
        idx_bus = _mapping(psa_net, heat_df.bus[hp_i])
        # retrieve power factor and sign from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "hp")
        q = sign * np.tan(np.arccos(pf)) * heat_df.p_set[hp_i]
        p_d = heat_df2[heat_df.index[hp_i]]
        p_d.loc[p_d < tol] = 0
        pm["heatpumps"][str(hp_i + 1)] = {
            "pd": p_d[0],  # heat demand
            "pf": pf,
            "sign": sign,
            "p_min": 0,
            "p_max": heat_df.p_set[hp_i],
            "q_min": min(q, 0),
            "q_max": max(q, 0),
            "cop": edisgo_obj.heat_pump.cop_df[heat_df.index[hp_i]][0],
            "hp_bus": idx_bus,
            "name": heat_df.index[hp_i],
            "index": hp_i + 1,
        }


def _build_heat_storage(psa_net, pm, edisgo_obj, flexible_hps):
    """
    Builds heat storage dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    edisgo_obj : :class:`~.EDisGo`

    """

    heat_storage_df = edisgo_obj.heat_pump.thermal_storage_units_df.loc[flexible_hps]
    for stor_i in np.arange(len(heat_storage_df.index)):
        idx_bus = _mapping(psa_net, psa_net.loads.bus[stor_i])
        pm["heat_storage"][str(stor_i + 1)] = {
            "ps": 0,
            "p_loss": 0,
            "energy": (
                heat_storage_df.state_of_charge_initial[stor_i]
                * heat_storage_df.capacity[stor_i]
            ),
            "capacity": heat_storage_df.capacity[stor_i],
            "charge_efficiency": heat_storage_df.efficiency[stor_i],
            "discharge_efficiency": heat_storage_df.efficiency[stor_i],
            "storage_bus": idx_bus,
            "name": heat_storage_df.index[stor_i],
            "status": True,
            "index": stor_i + 1,
        }


def _build_dsm(edisgo_obj, psa_net, pm, flexible_loads, tol):
    """
    Builds dsm 'storage' dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    flexible_loads : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    """
    dsm_df = psa_net.loads.loc[flexible_loads]
    for dsm_i in np.arange(len(dsm_df.index)):
        idx_bus = _mapping(psa_net, dsm_df.bus[dsm_i])
        # retrieve power factor and sign from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "load")
        p_max = edisgo_obj.dsm.p_max[dsm_df.index[dsm_i]]
        p_min = edisgo_obj.dsm.p_min[dsm_df.index[dsm_i]]
        e_min = edisgo_obj.dsm.e_min[dsm_df.index[dsm_i]]
        e_max = edisgo_obj.dsm.e_max[dsm_df.index[dsm_i]]
        p_max.loc[p_max < tol] = 0
        p_min.loc[e_min < tol] = 0
        e_min.loc[e_min < tol] = 0
        e_max.loc[e_max < tol] = 0
        q = [
            sign * np.tan(np.arccos(pf)) * p_max[0],
            sign * np.tan(np.arccos(pf)) * p_min[0],
        ]
        pm["dsm"][str(dsm_i + 1)] = {
            "pd": 0,
            "qd": 0,
            "pf": pf,
            "sign": sign,
            "energy": 0,  # TODO: am Anfang immer 0?
            "p_min": p_min[0],
            "p_max": p_max[0],
            "q_max": max(q),
            "q_min": min(q),
            "e_min": e_min[0],
            "e_max": e_max[0],
            "charge_efficiency": 1,
            "discharge_efficiency": 1,
            "dsm_bus": idx_bus,
            "name": dsm_df.index[dsm_i],
            "index": dsm_i + 1,
        }


def _build_HV_requirements(
    psa_net, pm, opt_flex, flexible_cps, flexible_hps, hv_flex_dict
):
    """
    Builds dictionary for HV requirement data in PowerModels network data format and
    adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    opt_flex : list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from etrago component of edisgo object.
    """

    inflexible_cps = [
        cp
        for cp in psa_net.loads.loc[
            psa_net.loads.index.str.contains("Charging")
        ].index.values
        if cp not in flexible_cps
    ]
    inflexible_hps = [
        hp
        for hp in psa_net.loads.loc[
            psa_net.loads.index.str.contains("Heat")
        ].index.values
        if hp not in flexible_hps
    ]
    if len(inflexible_cps) > 0:
        hv_flex_dict["cp"] = hv_flex_dict["cp"] - psa_net.loads_t.p_set.loc[
            :, inflexible_cps
        ].sum(axis=1)
    if len(inflexible_hps) > 0:
        hv_flex_dict["hp"] = hv_flex_dict["hp"] - psa_net.loads_t.p_set.loc[
            :, inflexible_hps
        ].sum(axis=1)
    for i in np.arange(len(opt_flex)):
        pm["HV_requirements"][str(i + 1)] = {
            "P": hv_flex_dict[opt_flex[i]][0],
            "flexibility": opt_flex[i],
        }


def _build_timeseries(
    psa_net,
    pm,
    edisgo_obj,
    flexible_cps,
    flexible_hps,
    flexible_loads,
    opt_flex,
    hv_flex_dict,
):
    """
    Builds timeseries dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'. PowerModels' timeseries dictionary contains one
    timeseries dictionary each for: gen, load, (battery) storage,
    electromobility, heat pumps and dsm storage.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    edisgo_obj : :class:`~.EDisGo`
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_loads : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    opt_flex: list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from etrago component of edisgo object.
    """
    for kind in [
        "gen",
        "gen_nd",
        "gen_slack",
        "load",
        "storage",
        "electromobility",
        "heatpumps",
        "dsm",
        "HV_requirements",
    ]:
        _build_component_timeseries(
            psa_net,
            pm,
            kind,
            edisgo_obj,
            flexible_cps,
            flexible_hps,
            flexible_loads,
            opt_flex,
            hv_flex_dict,
        )
    pm["time_series"]["num_steps"] = len(psa_net.snapshots)


def _build_component_timeseries(
    psa_net,
    pm,
    kind,
    edisgo_obj=None,
    flexible_cps=None,
    flexible_hps=None,
    flexible_loads=None,
    opt_flex=None,
    hv_flex_dict=None,
):
    """
    Builds timeseries dictionary for given kind and adds it to 'time_series'
    dictionary in PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    kind: str
        Must be one of ["gen", "gen_nd", "gen_slack", "load", "storage",
        "electromobility", "heatpumps", "heat_storage", "dsm", "HV_requirements"]
    edisgo_obj : :class:`~.EDisGo`
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_loads : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    opt_flex: list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from etrago component of edisgo object.
    """
    pm_comp = dict()
    if kind == "gen":
        p_set = psa_net.generators_t.p_set.loc[
            :,
            ~(psa_net.generators_t.p_set.columns.str.contains("solar"))
            & ~(psa_net.generators_t.p_set.columns.str.contains("wind"))
            & ~(psa_net.generators_t.p_set.columns.str.contains("slack")),
        ]
        q_set = psa_net.generators_t.q_set[p_set.columns]
    elif kind == "gen_nd":
        p_set = psa_net.generators_t.p_set.loc[
            :,
            psa_net.generators_t.p_set.columns.str.contains("solar")
            | psa_net.generators_t.p_set.columns.str.contains("wind"),
        ]
        q_set = psa_net.generators_t.q_set[p_set.columns]
    elif kind == "gen_slack":
        p_set = psa_net.generators_t.p_set.loc[
            :,
            psa_net.generators_t.p_set.columns.str.contains("slack"),
        ]
        q_set = psa_net.generators_t.q_set[p_set.columns]
    elif kind == "load":
        flex_loads = np.concatenate((flexible_hps, flexible_cps, flexible_loads))
        if len(flex_loads) == 0:
            p_set = psa_net.loads_t.p_set
            q_set = psa_net.loads_t.q_set
        else:
            p_set = psa_net.loads_t.p_set.drop(columns=flex_loads)
            q_set = psa_net.loads_t.q_set.drop(columns=flex_loads)
    elif kind == "storage":
        p_set = psa_net.storage_units_t.p_set
        q_set = psa_net.storage_units_t.q_set
    elif kind == "electromobility":
        if len(flexible_cps) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = edisgo_obj.electromobility.flexibility_bands["upper_power"][
                flexible_cps
            ]
            e_min = edisgo_obj.electromobility.flexibility_bands["lower_energy"][
                flexible_cps
            ]
            e_max = edisgo_obj.electromobility.flexibility_bands["upper_energy"][
                flexible_cps
            ]
    elif kind == "heatpumps":
        if len(flexible_hps) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = edisgo_obj.heat_pump.heat_demand_df[flexible_hps]
            cop = edisgo_obj.heat_pump.cop_df[flexible_hps]
    elif kind == "dsm":
        if len(flexible_loads) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = edisgo_obj.dsm.p_max
            p_min = edisgo_obj.dsm.p_min
            e_min = edisgo_obj.dsm.e_min
            e_max = edisgo_obj.dsm.e_max
    elif kind == "HV_requirements":
        p_set = pd.DataFrame()
    for comp in p_set.columns:
        if kind == "gen":
            comp_i = _mapping(psa_net, comp, kind)
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }
        elif kind == "gen_nd":
            comp_i = _mapping(psa_net, comp, kind)
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }
        elif kind == "gen_slack":
            comp_i = _mapping(psa_net, comp, kind)
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }
        elif kind == "load":
            comp_i = _mapping(
                psa_net, comp, kind, flexible_cps, flexible_hps, flexible_loads
            )
            p_d = p_set[comp].values
            q_d = q_set[comp].values
            pm_comp[str(comp_i)] = {
                "pd": p_d.tolist(),
                "qd": q_d.tolist(),
            }
        elif kind == "storage":
            comp_i = _mapping(psa_net, comp, kind)
            pm_comp[str(comp_i)] = {
                "ps": p_set[comp].values.tolist(),
                "qs": q_set[comp].values.tolist(),
            }
        elif kind == "electromobility":
            comp_i = _mapping(psa_net, comp, kind, flexible_cps=flexible_cps)
            if len(flexible_cps) > 0:
                pm_comp[str(comp_i)] = {
                    "p_max": p_set[comp].values.tolist(),
                    "e_min": e_min[comp].values.tolist(),
                    "e_max": e_max[comp].values.tolist(),
                }
        elif kind == "heatpumps":
            comp_i = _mapping(psa_net, comp, kind, flexible_hps=flexible_hps)
            pm_comp[str(comp_i)] = {
                "pd": p_set[comp].values.tolist(),
                "cop": cop[comp].values.tolist(),
            }
        elif kind == "dsm":
            comp_i = _mapping(psa_net, comp, kind, flexible_loads=flexible_loads)
            pm_comp[str(comp_i)] = {
                "p_max": p_set[comp].values.tolist(),
                "p_min": p_min[comp].values.tolist(),
                "e_min": e_min[comp].values.tolist(),
                "e_max": e_max[comp].values.tolist(),
            }
    if (kind == "HV_requirements") & (
        (pm["opt_version"] == 1) | (pm["opt_version"] == 2)
    ):
        for i in np.arange(len(opt_flex)):
            pm_comp[(str(i + 1))] = {
                "P": hv_flex_dict[opt_flex[i]].tolist(),
            }

    pm["time_series"][kind] = pm_comp


def _mapping(
    psa_net, name, kind="bus", flexible_cps=None, flexible_hps=None, flexible_loads=None
):
    if kind == "bus":
        df = psa_net.buses
    elif kind == "gen":
        df = psa_net.generators.loc[
            ~(psa_net.generators.index.str.contains("solar"))
            & ~(psa_net.generators.index.str.contains("wind"))
            & ~(psa_net.generators.index.str.contains("slack"))
        ]
    elif kind == "gen_nd":
        df = psa_net.generators.loc[
            (psa_net.generators.index.str.contains("solar"))
            | (psa_net.generators.index.str.contains("wind"))
        ]
    elif kind == "gen_slack":
        df = psa_net.generators.loc[(psa_net.generators.index.str.contains("slack"))]
    elif kind == "storage":
        df = psa_net.storage_units
    elif kind == "load":
        flex_loads = np.concatenate((flexible_hps, flexible_cps, flexible_loads))
        if len(flex_loads) == 0:
            df = psa_net.loads
        else:
            df = psa_net.loads.drop(flex_loads)
    elif kind == "electromobility":
        df = psa_net.loads.loc[flexible_cps]
    elif (kind == "heatpumps") | (kind == "heat_storage"):
        df = psa_net.loads.loc[flexible_hps]
    elif kind == "dsm":
        df = psa_net.loads.loc[flexible_loads]
    else:
        logging.warning("Mapping for '{}' not implemented.".format(kind))
    idx = df.reset_index()[df.index == name].index[0] + 1
    return idx


def aggregate_parallel_transformers(psa_net):
    """
    Calculates impedance for parallel transformers and aggregates them. Replaces
    psa_net.transformers dataframe by aggregated transformer dataframe.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.

    """

    # TODO: what about b, g?
    psa_trafos = psa_net.transformers
    trafo_df = (
        psa_trafos.groupby(by=[psa_trafos.bus0, psa_trafos.bus1])[["r", "x", "s_nom"]]
        .apply(calculate_impedance_for_parallel_components)
        .reset_index()
    )
    psa_trafos.index = [index[:-2] for index in psa_trafos.index]
    psa_trafos = psa_trafos[~psa_trafos.index.duplicated(keep="first")]
    psa_trafos = (
        psa_trafos.reset_index()
        .drop(columns=["r", "x", "s_nom"])
        .merge(trafo_df, how="left")
        .set_index("index")
    )
    psa_net.transformers = psa_trafos


def _get_pf(edisgo_obj, pm, idx_bus, kind):
    """
    Retrieves and returns power factor from edisgo config files.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    pm : dict
        (PowerModels) dictionary.
    idx_bus: int
        Bus index from PowerModels bus dictionary.
    kind: str
        Must be one of ["gen", "load", "storage", "hp", "cp"]

    Returns
    -------
    pf: float
    sign: int
    """

    grid_level = pm["bus"][str(idx_bus)]["grid_level"]
    pf = edisgo_obj.config._data["reactive_power_factor"][
        "{}_{}".format(grid_level, kind)
    ]
    sign = edisgo_obj.config._data["reactive_power_mode"][
        "{}_{}".format(grid_level, kind)
    ]
    if kind in ["gen", "storage"]:
        if sign == "inductive":
            sign = -1
        else:
            sign = 1
    elif kind in ["load", "hp", "cp"]:
        if sign == "inductive":
            sign = 1
        else:
            sign = -1
    return pf, sign
