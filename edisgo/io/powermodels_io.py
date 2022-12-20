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
        self.opf_results = pd.DataFrame(dtype="float64")
        self.geothermal_energy_feedin_district_heating = pd.DataFrame(dtype="float64")
        self.solarthermal_energy_feedin_district_heating = pd.DataFrame(dtype="float64")


def to_powermodels(
    edisgo_object,
    s_base=1,
    flexible_cps=None,
    flexible_hps=None,
    flexible_loads=None,
    opt_version=4,
    opt_flex=None,
):
    """
    Converts eDisGo representation of the network topology and timeseries to
    PowerModels network data format.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    s_base : int
        Base value of apparent power for per unit system.
        Default: 1 MVA
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_loads: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    opt_version: Int
        Version of optimization models to choose from. Must be one of [1, 2, 3, 4].
        For more information see :func:`edisgo.opf.powermodels_opf.pm_optimize`.
        Default: 4
    opt_flex: list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["storage", "cp", "hp", "dsm"]

    Returns
    -------
    pm: dict
        Dictionary that contains all network data in PowerModels network data
        format.
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from etrago component of edisgo object.
    """
    # tol = 1e-4
    if opt_flex is None:
        opt_flex = ["curt"]
    if "curt" not in opt_flex:
        opt_flex.append("curt")
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
    pm["baseMVA"] = s_base
    pm["source_version"] = 2
    pm["flexibilities"] = opt_flex
    _build_bus(psa_net, pm)
    _build_gen(edisgo_object, psa_net, pm, s_base)
    _build_branch(psa_net, pm, s_base)
    if len(edisgo_object.topology.storage_units_df) > 0:
        _build_battery_storage(edisgo_object, psa_net, pm, s_base)
    if len(flexible_cps) > 0:
        flexible_cps = _build_electromobility(
            edisgo_object,
            psa_net,
            pm,
            s_base,
            flexible_cps,
        )
    if len(flexible_hps) > 0:
        _build_heatpump(psa_net, pm, edisgo_object, s_base, flexible_hps)
    if "hp" in opt_flex:
        _build_heat_storage(psa_net, pm, edisgo_object, s_base, flexible_hps)
    if len(flexible_loads) > 0:
        flexible_loads = _build_dsm(edisgo_object, psa_net, pm, s_base, flexible_loads)
    if len(psa_net.loads) > 0:
        _build_load(edisgo_object, psa_net, pm, s_base, flexible_cps, flexible_hps)
    else:
        logger.warning("No loads found in network.")
    if (opt_version == 1) | (opt_version == 2):
        # edisgo_object.etrago.renewables_curtailment[
        #     edisgo_object.etrago.renewables_curtailment < tol
        # ] = 0
        # edisgo_object.etrago.storage_units_active_power[
        #     edisgo_object.etrago.storage_units_active_power < tol
        # ] = 0
        # edisgo_object.etrago.electromobility_active_power[
        #     edisgo_object.etrago.electromobility_active_power < tol
        # ] = 0
        # edisgo_object.etrago.heat_pump_rural_active_power[
        #     edisgo_object.etrago.heat_pump_rural_active_power < tol
        # ] = 0
        # edisgo_object.etrago.heat_central_active_power[
        #     edisgo_object.etrago.heat_central_active_power < tol
        # ] = 0
        # edisgo_object.etrago.dsm_active_power[
        #     edisgo_object.etrago.dsm_active_power < tol
        # ] = 0
        hv_flex_dict = {
            "curt": edisgo_object.etrago.renewables_curtailment / s_base,
            "storage": edisgo_object.etrago.storage_units_active_power / s_base,
            "cp": edisgo_object.etrago.electromobility_active_power / s_base,
            "hp": (
                edisgo_object.etrago.heat_pump_rural_active_power
                + edisgo_object.etrago.heat_central_active_power
            )
            / s_base,
            "dsm": edisgo_object.etrago.dsm_active_power / s_base,
        }
        try:
            _build_HV_requirements(
                psa_net, pm, s_base, opt_flex, flexible_cps, flexible_hps, hv_flex_dict
            )
        except IndexError:
            logger.warning(
                "Etrago component of eDisGo object has no entries."
                " Changing optimization version to '4' (without high voltage"
                " requirements)."
            )
            opt_version = 4

    pm["opt_version"] = opt_version

    _build_timeseries(
        psa_net,
        pm,
        edisgo_object,
        s_base,
        flexible_cps,
        flexible_hps,
        flexible_loads,
        opt_flex,
        hv_flex_dict,
    )
    return pm, hv_flex_dict


def from_powermodels(
    edisgo_object,
    pm_results,
    hv_flex_dict,
    s_base=1,
    save_heat_storage=False,
    save_slack_gen=False,
    save_slacks=False,
    path="",
):
    """
    Converts results from optimization in PowerModels network data format to eDisGo data
    format and updates timeseries values of flexibilities on eDisGo object.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    s_base : int
        Base value of apparent power for per unit system.
        Default: 1 MVA
    pm_results: dict or str
        Dictionary or path to json file that contains all optimization results in
        PowerModels network data format.
    save_heat_storage: bool
        Indicates whether to save results of heat storage variables from the
        optimization to csv file in the current working directory. Set parameter
        "path" to change the directory the file is saved to.
        directory.
            Default: False
    save_slack_gen: bool
        Indicates whether to save results of slack generator variables from the
        optimization to csv file in the current working directory. Set parameter
        "path" to change the directory the file is saved to.
        Default: False
    save_slacks: bool
        Indicates whether to save results of slack variables from the OPF run to csv
        files in the current working directory. Set parameter "path" to change the
        directory the file is saved to. Depending on the chosen opt_version, different
        slacks are created and saved:
        1 : high voltage requirement slacks
        2 : high voltage requirements slacks and grid related slacks (load shedding,
            dispatchable and non-dispatchable generator curtailment, heat pump slack)
        3 : -
        4 : grid related slacks cf. version 2
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

    # write active power OPF results to edisgo object
    for flexibility in pm_results["nw"]["1"]["flexibilities"]:
        flex, variable = flex_dicts[flexibility]
        names = [
            pm["nw"]["1"][flex][flex_comp]["name"]
            for flex_comp in list(pm["nw"]["1"][flex].keys())
        ]
        data = [
            [
                pm["nw"][t][flex][flex_comp][variable] * s_base
                for flex_comp in list(pm["nw"]["1"][flex].keys())
            ]
            for t in timesteps
        ]
        results = pd.DataFrame(index=timesteps, columns=names, data=data)
        if flex == "gen_nd":
            edisgo_object.timeseries._generators_active_power.loc[:, names] = (
                edisgo_object.timeseries.generators_active_power.loc[:, names].values
                - results[names].values
            )
        elif flex in ["heatpumps", "electromobility"]:
            edisgo_object.timeseries._loads_active_power.loc[:, names] = results[
                names
            ].values
        elif flex == "dsm":
            edisgo_object.timeseries._loads_active_power.loc[:, names] = (
                edisgo_object.timeseries._loads_active_power.loc[:, names].values
                + results[names].values
            )
        elif flex == "storage":
            edisgo_object.timeseries._storage_units_active_power.loc[
                :, names
            ] = results[names].values

    # calculate corresponding reactive power values
    edisgo_object.set_time_series_reactive_power_control()

    # Check values of slack variables for HV requirement constraint
    if pm["nw"]["1"]["opt_version"] in [1, 2]:
        names = [
            pm["nw"]["1"]["HV_requirements"][flex]["flexibility"]
            for flex in list(pm["nw"]["1"]["HV_requirements"].keys())
        ]
        data = [
            [
                pm["nw"][t]["HV_requirements"][flex]["phvs"] * s_base
                for flex in list(pm["nw"]["1"]["HV_requirements"].keys())
            ]
            for t in timesteps
        ]
        df = pd.DataFrame(
            index=edisgo_object.timeseries.timeindex,
            columns=names,
            data=data,
        )
        # save HV slack results to csv
        if save_slacks:
            df.to_csv(os.path.join(abs_path, "hv_requirements_slack.csv"))

        # calculate relative error
        for flex in names:
            df[flex] = abs(df[flex].values - hv_flex_dict[flex]) / hv_flex_dict[flex]

        df2 = pd.DataFrame(
            columns=[
                "Highest relative error",
                "Mean relative error",
                "Sum relative error",
            ],
            index=names,
        )
        df2["Highest relative error"] = df.max()
        df2["Mean relative error"] = df.sum() / len(df)
        df2["Sum relative error"] = df.sum()
        edisgo_object.etrago.opf_results = df2  # write results to edisgo object
        for flex in names:
            if (df2["Highest relative error"][flex] > 0.05).any():
                logger.warning(
                    "Highest relative error of {} variable exceeds 5%.".format(flex)
                )

    if save_slack_gen:  # save slack generator variable to csv file
        df = pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=["pg", "qg"]
        )
        for gen in list(pm["nw"]["1"]["gen_slack"].keys()):
            df["pg"] = [
                pm["nw"][t]["gen_slack"][gen]["pgs"] * s_base for t in timesteps
            ]
            df["qg"] = [
                pm["nw"][t]["gen_slack"][gen]["qgs"] * s_base for t in timesteps
            ]
        df.to_csv(os.path.join(abs_path, "slack_gen.csv"))

    if save_heat_storage:  # save heat storage variables to csv file
        for variable in ["phs", "hse"]:
            names = [
                pm["nw"]["1"]["heat_storage"][hs]["name"]
                for hs in list(pm["nw"]["1"]["heat_storage"].keys())
            ]
            data = [
                [
                    pm["nw"][t]["heat_storage"][hs][variable] * s_base
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

    if (pm["nw"]["1"]["opt_version"] in [2, 4]) & save_slacks:
        # save heatpump slacks to csv file
        names = [
            pm["nw"]["1"]["heatpumps"][hp]["name"]
            for hp in list(pm["nw"]["1"]["heatpumps"].keys())
        ]
        data = [
            [
                pm["nw"][t]["heatpumps"][hp]["phps"] * s_base
                for hp in list(pm["nw"]["1"]["heatpumps"].keys())
            ]
            for t in timesteps
        ]
        pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=names, data=data
        ).to_csv(os.path.join(abs_path, "heat_pump_p_slack.csv"))

        # save dispatchable generator slacks (slack curtailment)
        names = [
            pm["nw"]["1"]["gen"][gen]["name"]
            for gen in list(pm["nw"]["1"]["gen"].keys())
        ]
        data = [
            [
                pm["nw"][t]["gen"][gen]["pgens"] * s_base
                for gen in list(pm["nw"]["1"]["gen"].keys())
            ]
            for t in timesteps
        ]
        pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=names, data=data
        ).to_csv(os.path.join(abs_path, "disp_generator_slack.csv"))

        # save non-dispatchable generator slacks (curtailment) to csv file
        names = [
            pm["nw"]["1"]["gen_nd"][gen]["name"]
            for gen in list(pm["nw"]["1"]["gen_nd"].keys())
        ]
        data = [
            [
                pm["nw"][t]["gen_nd"][gen]["pgc"] * s_base
                for gen in list(pm["nw"]["1"]["gen_nd"].keys())
            ]
            for t in timesteps
        ]
        pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=names, data=data
        ).to_csv(os.path.join(abs_path, "nondisp_generator_curtailment.csv"))

        # save load slacks (load shedding)
        names = [
            pm["nw"]["1"]["load"][load]["name"]
            for load in list(pm["nw"]["1"]["load"].keys())
        ]
        data = [
            [
                pm["nw"][t]["load"][load]["pds"] * s_base
                for load in list(pm["nw"]["1"]["load"].keys())
            ]
            for t in timesteps
        ]
        pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=names, data=data
        ).to_csv(os.path.join(abs_path, "load_shedding.csv"))

        # save cp load slacks (cp load shedding)
        names = [
            pm["nw"]["1"]["electromobility"][cp]["name"]
            for cp in list(pm["nw"]["1"]["electromobility"].keys())
        ]
        data = [
            [
                pm["nw"][t]["electromobility"][cp]["pcps"] * s_base
                for cp in list(pm["nw"]["1"]["electromobility"].keys())
            ]
            for t in timesteps
        ]
        pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=names, data=data
        ).to_csv(os.path.join(abs_path, "cp_load_shedding.csv"))


def _init_pm():
    """
    Initializes empty PowerModels dictionary.

    Returns
    -------
    pm: dict
        Dictionary that contains all network data in PowerModels network data
        format.
    """
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
            "name": psa_net.buses.index[bus_i],
            "base_kv": psa_net.buses.v_nom[bus_i],
            "grid_level": grid_level[psa_net.buses.v_nom[bus_i]],
        }


def _build_gen(edisgo_obj, psa_net, pm, s_base):
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
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
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
        # psa_net.generators_t.p_set[gen_disp.index[gen_i]].loc[
        #     psa_net.generators_t.p_set[gen_disp.index[gen_i]] < tol
        # ] = 0
        # psa_net.generators_t.q_set[gen_disp.index[gen_i]].loc[
        #     psa_net.generators_t.q_set[gen_disp.index[gen_i]] > -tol
        # ] = 0
        pm["gen"][str(gen_i + 1)] = {
            "pg": psa_net.generators_t.p_set[gen_disp.index[gen_i]][0] / s_base,
            "qg": psa_net.generators_t.q_set[gen_disp.index[gen_i]][0] / s_base,
            "pmax": gen_disp.p_nom[gen_i] / s_base,
            "pmin": gen_disp.p_nom_min[gen_i] / s_base,
            "qmax": max(q) / s_base,
            "qmin": min(q) / s_base,
            "vg": 1,
            "mbase": gen_disp.p_nom[gen_i] / s_base,
            "gen_bus": idx_bus,
            "name": gen_disp.index[gen_i],
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
        # psa_net.generators_t.p_set[gen_nondisp.index[gen_i]].loc[
        #     psa_net.generators_t.p_set[gen_nondisp.index[gen_i]] < tol
        # ] = 0
        # psa_net.generators_t.q_set[gen_nondisp.index[gen_i]].loc[
        #     psa_net.generators_t.q_set[gen_nondisp.index[gen_i]] > -tol
        # ] = 0
        pm["gen_nd"][str(gen_i + 1)] = {
            "pg": psa_net.generators_t.p_set[gen_nondisp.index[gen_i]][0] / s_base,
            "qg": psa_net.generators_t.q_set[gen_nondisp.index[gen_i]][0] / s_base,
            "pmax": gen_nondisp.p_nom[gen_i] / s_base,
            "pmin": gen_nondisp.p_nom_min[gen_i] / s_base,
            "qmax": max(q) / s_base,
            "qmin": min(q) / s_base,
            "P": 0,
            "Q": 0,
            "vg": 1,
            "pf": pf,
            "sign": sign,
            "mbase": gen_nondisp.p_nom[gen_i] / s_base,
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
            "pg": psa_net.generators_t.p_set[gen_slack.index[gen_i]][0] / s_base,
            "qg": psa_net.generators_t.q_set[gen_slack.index[gen_i]][0] / s_base,
            "pmax": gen_slack.p_nom[gen_i] / s_base,
            "pmin": gen_slack.p_nom_min[gen_i] / s_base,
            "qmax": max(q) / s_base,
            "qmin": min(q) / s_base,
            "P": 0,
            "Q": 0,
            "vg": 1,
            "mbase": gen_slack.p_nom[gen_i] / s_base,
            "gen_bus": idx_bus,
            "name": gen_slack.index[gen_i],
            "gen_status": 1,
            "index": gen_i + 1,
        }


def _build_branch(psa_net, pm, s_base):
    """
    Builds branch dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    """
    branches = pd.concat([psa_net.lines, psa_net.transformers])
    transformer = ~branches.tap_ratio.isna()
    tap = branches.tap_ratio.fillna(1)
    shift = branches.phase_shift.fillna(0)
    max_r = np.round(
        max(branches.r_pu.loc[branches.r_pu < branches.r_pu.quantile(0.998)]), 4
    )
    min_r = np.round(
        min(branches.r_pu.loc[branches.r_pu > branches.r_pu.quantile(0.002)]), 6
    )
    max_x = np.round(
        max(branches.x_pu.loc[branches.x_pu < branches.x_pu.quantile(0.998)]), 4
    )
    min_x = np.round(
        min(branches.x_pu.loc[branches.x_pu > branches.x_pu.quantile(0.002)]), 6
    )
    for branch_i in np.arange(len(branches.index)):
        idx_f_bus = _mapping(psa_net, branches.bus0[branch_i])
        idx_t_bus = _mapping(psa_net, branches.bus1[branch_i])
        if branches.r_pu[branch_i] > np.round(branches.r_pu.quantile(0.998), 4):
            logger.warning(
                "Resistance of branch {} is higher than {} p.u. Resistance "
                "will be set to {} for optimization process.".format(
                    branches.index[branch_i],
                    np.round(branches.r_pu.quantile(0.998), 4),
                    max_r,
                )
            )
            r = min(branches.r_pu[branch_i], max_r)
        elif branches.r_pu[branch_i] < np.round(branches.r_pu.quantile(0.002), 6):
            logger.warning(
                "Resistance of branch {} is smaller than {} p.u. Resistance "
                "will be set to {} for optimization process.".format(
                    branches.index[branch_i],
                    np.round(branches.r_pu.quantile(0.002), 6),
                    min_r,
                )
            )
            r = max(branches.r_pu[branch_i], min_r)
        else:
            r = branches.r_pu[branch_i]
        if branches.x_pu[branch_i] > np.round(branches.x_pu.quantile(0.998), 4):
            logger.warning(
                "Reactance of branch {} is higher than {} p.u. Reactance "
                "will be set to {} for optimization process.".format(
                    branches.index[branch_i],
                    np.round(branches.x_pu.quantile(0.998), 4),
                    max_x,
                )
            )
            x = min(branches.x_pu[branch_i], max_x)
        elif branches.x_pu[branch_i] < np.round(branches.x_pu.quantile(0.002), 6):
            logger.warning(
                "Reactance of branch {} is smaller than {} p.u. Reactance "
                "will be set to {} for optimization process.".format(
                    branches.index[branch_i],
                    np.round(branches.x_pu.quantile(0.002), 6),
                    min_x,
                )
            )
            x = max(branches.x_pu[branch_i], min_x)
        else:
            x = branches.x_pu[branch_i]
        # x = branches.x_pu[branch_i]
        # r = branches.r_pu[branch_i]
        # if branches.r_pu[branch_i] < tol:
        #     branches.r_pu.loc[branch_i] = 0
        # if branches.x_pu[branch_i] < tol:
        #     branches.x_pu.loc[branch_i] = 0
        # if branches.length.fillna(1)[branch_i] < tol:
        #     branches.length.fillna(1).loc[branch_i] = 0
        # if branches.capital_cost[branch_i] < tol:
        #     branches.capital_cost.loc[branch_i] = 0
        pm["branch"][str(branch_i + 1)] = {
            "name": branches.index[branch_i],
            "br_r": r * s_base,
            "br_x": x * s_base,
            "f_bus": idx_f_bus,
            "t_bus": idx_t_bus,
            "g_to": branches.g_pu[branch_i] / 2 * s_base,  # ToDo: check if * or /
            "g_fr": branches.g_pu[branch_i] / 2 * s_base,
            "b_to": branches.b_pu[branch_i] / 2 * s_base,
            "b_fr": branches.b_pu[branch_i] / 2 * s_base,
            "shift": shift[branch_i],
            "br_status": 1.0,
            "rate_a": branches.s_nom[branch_i].real / s_base,
            "rate_b": 250 / s_base,
            "rate_c": 250 / s_base,
            "angmin": -np.pi / 6,
            "angmax": np.pi / 6,
            "transformer": bool(transformer[branch_i]),
            "tap": tap[branch_i],
            "length": branches.length.fillna(1)[branch_i],
            "cost": branches.capital_cost[branch_i],
            "index": branch_i + 1,
        }


def _build_load(edisgo_obj, psa_net, pm, s_base, flexible_cps, flexible_hps):
    """
    Builds load dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    """
    flex_loads = np.concatenate((flexible_hps, flexible_cps))
    if len(flex_loads) == 0:
        loads_df = psa_net.loads
    else:
        loads_df = psa_net.loads.drop(flex_loads)
    for load_i in np.arange(len(loads_df.index)):
        idx_bus = _mapping(psa_net, loads_df.bus[load_i])
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "load")
        p_d = psa_net.loads_t.p_set[loads_df.index[load_i]]
        q_d = psa_net.loads_t.q_set[loads_df.index[load_i]]
        # p_d.loc[p_d < tol] = 0
        # q_d.loc[q_d < tol] = 0
        pm["load"][str(load_i + 1)] = {
            "pd": p_d[0] / s_base,
            "qd": q_d[0] / s_base,
            "load_bus": idx_bus,
            "status": True,
            "pf": pf,
            "sign": sign,
            "name": loads_df.index[load_i],
            "index": load_i + 1,
        }


def _build_battery_storage(edisgo_obj, psa_net, pm, s_base):
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
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
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
            "ps": psa_net.storage_units_t.p_set[psa_net.storage_units.index[stor_i]][0]
            / s_base,
            "qs": psa_net.storage_units_t.q_set[psa_net.storage_units.index[stor_i]][0]
            / s_base,
            "pmax": psa_net.storage_units.p_nom[stor_i] / s_base,
            "pmin": -psa_net.storage_units.p_nom[stor_i] / s_base,
            "qmax": np.tan(np.arccos(pf))
            * psa_net.storage_units.p_nom[stor_i]
            / s_base,
            "qmin": -np.tan(np.arccos(pf))
            * psa_net.storage_units.p_nom[stor_i]
            / s_base,
            "energy": psa_net.storage_units.state_of_charge_initial[stor_i]
            * e_max
            / s_base,
            "energy_rating": e_max / s_base,
            "thermal_rating": 1,  # TODO unbegrenzt
            "charge_rating": psa_net.storage_units.p_nom[stor_i] / s_base,
            "discharge_rating": psa_net.storage_units.p_nom[stor_i] / s_base,
            "charge_efficiency": 0.9,  # ToDo
            "discharge_efficiency": 0.9,  # ToDo
            "storage_bus": idx_bus,
            "name": psa_net.storage_units.index[stor_i],
            "status": True,
            "index": stor_i + 1,
        }


def _build_electromobility(edisgo_obj, psa_net, pm, s_base, flexible_cps):
    """
    Builds electromobility dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.

    Returns
     ----------
     flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Updated array containing all charging points that allow for flexible charging.
    """

    flex_bands_df = edisgo_obj.electromobility.flexibility_bands
    if (
        len(
            flexible_cps[
                (flex_bands_df["lower_energy"] > flex_bands_df["upper_energy"]).any()
            ]
        )
        > 0
    ):
        logger.warning(
            "Upper energy level is smaller than lower energy level for "
            "charging parks {}! Charging Parks will be changed into inflexible "
            "loads.".format(
                flexible_cps[
                    (
                        flex_bands_df["lower_energy"] > flex_bands_df["upper_energy"]
                    ).any()
                ]
            )
        )
        flexible_cps = flexible_cps[
            ~(flex_bands_df["lower_energy"] > flex_bands_df["upper_energy"]).any()
        ]
    emob_df = psa_net.loads.loc[flexible_cps]
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
        # p_max.loc[p_max < tol] = 0
        # e_min.loc[e_min < tol] = 0
        # e_max.loc[e_max < tol] = 0
        pm["electromobility"][str(cp_i + 1)] = {
            "pd": 0,
            "qd": 0,
            "pf": pf,
            "sign": sign,
            "p_min": 0,
            "p_max": p_max[0] / s_base,
            "q_min": min(q, 0) / s_base,
            "q_max": max(q, 0) / s_base,
            "e_min": e_min[0] / s_base,
            "e_max": e_max[0] / s_base,
            "cp_bus": idx_bus,
            "name": emob_df.index[cp_i],
            "index": cp_i + 1,
        }
    return flexible_cps


def _build_heatpump(psa_net, pm, edisgo_obj, s_base, flexible_hps):
    """
    Builds heat pump dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    edisgo_obj : :class:`~.EDisGo`
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    flexible_hps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.

    """
    heat_df = psa_net.loads.loc[flexible_hps]  # electric load
    heat_df2 = edisgo_obj.heat_pump.heat_demand_df[flexible_hps]  # thermal load
    # solarthermal_feedin = edisgo_obj.etrago.geothermal_energy_feedin_district_heating
    # geothermal_feedin = edisgo_obj.etrago.solarthermal_energy_feedin_district_heating
    for hp_i in np.arange(len(heat_df.index)):
        idx_bus = _mapping(psa_net, heat_df.bus[hp_i])
        # retrieve power factor and sign from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "hp")
        q = sign * np.tan(np.arccos(pf)) * heat_df.p_set[hp_i]
        p_d = heat_df2[heat_df.index[hp_i]]
        # p_d[p_d < tol] = 0
        # TODO: hier den Demand um das solar/geothermal feedin verringern
        # TODO: check einbauen ob Einspeisung höher als Verbrauch. Dann Verbrauch auf 0
        # setzen
        if (
            max(p_d)
            > heat_df.p_set[hp_i] * edisgo_obj.heat_pump.cop_df[heat_df.index[hp_i]][0]
        ):
            logger.warning(
                "Heat demand at bus {} is higher than maximum heatpump power"
                " of heatpump {}. Demand can not be covered if no sufficient"
                " heat storage capaciies are available. This will cause "
                "problems in the optimization process if OPF version is"
                "set to 3".format(pm["bus"][str(idx_bus)]["name"], heat_df.index[hp_i])
            )
        pm["heatpumps"][str(hp_i + 1)] = {
            "pd": p_d[0] / s_base,  # heat demand
            "pf": pf,
            "sign": sign,
            "p_min": 0,
            "p_max": heat_df.p_set[hp_i] / s_base,
            "q_min": min(q, 0) / s_base,
            "q_max": max(q, 0) / s_base,
            "cop": edisgo_obj.heat_pump.cop_df[heat_df.index[hp_i]][0],
            "hp_bus": idx_bus,
            "name": heat_df.index[hp_i],
            "index": hp_i + 1,
        }


def _build_heat_storage(psa_net, pm, edisgo_obj, s_base, flexible_hps):
    """
    Builds heat storage dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    edisgo_obj : :class:`~.EDisGo`
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    flexible_hps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    """

    heat_storage_df = edisgo_obj.heat_pump.thermal_storage_units_df.loc[flexible_hps]
    for stor_i in np.arange(len(heat_storage_df.index)):
        idx_bus = _mapping(psa_net, psa_net.loads.bus[stor_i])
        pm["heat_storage"][str(stor_i + 1)] = {
            "ps": 0,
            "p_loss": 0.05,  # ToDo
            "energy": (
                heat_storage_df.state_of_charge_initial[stor_i]
                * heat_storage_df.capacity[stor_i]
                / s_base
            ),
            "capacity": heat_storage_df.capacity[stor_i] / s_base,
            "charge_efficiency": heat_storage_df.efficiency[stor_i],
            "discharge_efficiency": heat_storage_df.efficiency[stor_i],
            "storage_bus": idx_bus,
            "name": heat_storage_df.index[stor_i],
            "status": True,
            "index": stor_i + 1,
        }


def _build_dsm(edisgo_obj, psa_net, pm, s_base, flexible_loads):
    """
    Builds dsm 'storage' dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    flexible_loads : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    Returns
     ----------
     flexible_loads : :numpy:`numpy.ndarray<ndarray>` or list
        Updated array containing all flexible loads that allow for application of demand
        side management strategy.
    """

    if len(flexible_loads[(edisgo_obj.dsm.p_min > edisgo_obj.dsm.p_max).any()]) > 0:
        logger.warning(
            "Upper power level is smaller than lower power level for "
            "DSM loads {}! DSM loads will be changed into inflexible loads.".format(
                flexible_loads[(edisgo_obj.dsm.p_min > edisgo_obj.dsm.p_max).any()]
            )
        )
        flexible_loads = flexible_loads[
            ~(edisgo_obj.dsm.p_min > edisgo_obj.dsm.p_max).any()
        ]
    if len(flexible_loads[(edisgo_obj.dsm.e_min > edisgo_obj.dsm.e_max).any()]) > 0:
        logger.warning(
            "Upper energy level is smaller than lower energy level for "
            "DSM loads {}! DSM loads will be changed into inflexible loads.".format(
                flexible_loads[(edisgo_obj.dsm.e_min > edisgo_obj.dsm.e_max).any()]
            )
        )
        flexible_loads = flexible_loads[
            ~(edisgo_obj.dsm.e_min > edisgo_obj.dsm.e_max).any()
        ]
    dsm_df = psa_net.loads.loc[flexible_loads]
    for dsm_i in np.arange(len(dsm_df.index)):
        idx_bus = _mapping(psa_net, dsm_df.bus[dsm_i])
        # retrieve power factor and sign from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "load")
        p_max = edisgo_obj.dsm.p_max[dsm_df.index[dsm_i]]
        p_min = edisgo_obj.dsm.p_min[dsm_df.index[dsm_i]]
        e_min = edisgo_obj.dsm.e_min[dsm_df.index[dsm_i]]
        e_max = edisgo_obj.dsm.e_max[dsm_df.index[dsm_i]]
        # p_max.loc[p_max < tol] = 0
        # p_min.loc[e_min < tol] = 0
        # e_min.loc[e_min < tol] = 0
        # e_max.loc[e_max < tol] = 0

        q = [
            sign * np.tan(np.arccos(pf)) * p_max[0],
            sign * np.tan(np.arccos(pf)) * p_min[0],
        ]
        pm["dsm"][str(dsm_i + 1)] = {
            "pd": 0,
            "qd": 0,
            "pf": pf,
            "sign": sign,
            "energy": 0 / s_base,  # TODO: am Anfang immer 0?
            "p_min": p_min[0] / s_base,
            "p_max": p_max[0] / s_base,
            "q_max": max(q) / s_base,
            "q_min": min(q) / s_base,
            "e_min": e_min[0] / s_base,
            "e_max": e_max[0] / s_base,
            "charge_efficiency": 1,
            "discharge_efficiency": 1,
            "dsm_bus": idx_bus,
            "name": dsm_df.index[dsm_i],
            "index": dsm_i + 1,
        }
    return flexible_loads


def _build_HV_requirements(
    psa_net, pm, s_base, opt_flex, flexible_cps, flexible_hps, hv_flex_dict
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
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    opt_flex : list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]. For more information see
        :func:`edisgo.opf.powermodels_opf.pm_optimize`.
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
        hv_flex_dict["cp"] = (
            hv_flex_dict["cp"]
            - psa_net.loads_t.p_set.loc[:, inflexible_cps].sum(axis=1) / s_base
        )
    if len(inflexible_hps) > 0:
        hv_flex_dict["hp"] = (
            hv_flex_dict["hp"]
            - psa_net.loads_t.p_set.loc[:, inflexible_hps].sum(axis=1) / s_base
        )
    for i in np.arange(len(opt_flex)):
        pm["HV_requirements"][str(i + 1)] = {
            "P": hv_flex_dict[opt_flex[i]][0],
            "flexibility": opt_flex[i],
        }


def _build_timeseries(
    psa_net,
    pm,
    edisgo_obj,
    s_base,
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
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
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
            s_base,
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
    s_base,
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
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
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
        p_set = (
            psa_net.generators_t.p_set.loc[
                :,
                ~(psa_net.generators_t.p_set.columns.str.contains("solar"))
                & ~(psa_net.generators_t.p_set.columns.str.contains("wind"))
                & ~(psa_net.generators_t.p_set.columns.str.contains("slack")),
            ]
            / s_base
        )
        q_set = psa_net.generators_t.q_set[p_set.columns] / s_base
    elif kind == "gen_nd":
        p_set = (
            psa_net.generators_t.p_set.loc[
                :,
                psa_net.generators_t.p_set.columns.str.contains("solar")
                | psa_net.generators_t.p_set.columns.str.contains("wind"),
            ]
            / s_base
        )
        q_set = psa_net.generators_t.q_set[p_set.columns] / s_base
    elif kind == "gen_slack":
        p_set = (
            psa_net.generators_t.p_set.loc[
                :,
                psa_net.generators_t.p_set.columns.str.contains("slack"),
            ]
            / s_base
        )
        q_set = psa_net.generators_t.q_set[p_set.columns] / s_base
    elif kind == "load":
        flex_loads = np.concatenate((flexible_hps, flexible_cps))
        if len(flex_loads) == 0:
            p_set = psa_net.loads_t.p_set / s_base
            q_set = psa_net.loads_t.q_set / s_base
        else:
            p_set = psa_net.loads_t.p_set.drop(columns=flex_loads) / s_base
            q_set = psa_net.loads_t.q_set.drop(columns=flex_loads) / s_base
    elif kind == "storage":
        p_set = psa_net.storage_units_t.p_set / s_base
        q_set = psa_net.storage_units_t.q_set / s_base
    elif kind == "electromobility":
        if len(flexible_cps) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = (
                edisgo_obj.electromobility.flexibility_bands["upper_power"][
                    flexible_cps
                ]
                / s_base
            )
            e_min = (
                edisgo_obj.electromobility.flexibility_bands["lower_energy"][
                    flexible_cps
                ]
                / s_base
            )
            e_max = (
                edisgo_obj.electromobility.flexibility_bands["upper_energy"][
                    flexible_cps
                ]
                / s_base
            )
    elif kind == "heatpumps":
        if len(flexible_hps) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = edisgo_obj.heat_pump.heat_demand_df[flexible_hps] / s_base
            cop = edisgo_obj.heat_pump.cop_df[flexible_hps]
    elif kind == "dsm":
        if len(flexible_loads) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = edisgo_obj.dsm.p_max / s_base
            p_min = edisgo_obj.dsm.p_min / s_base
            e_min = edisgo_obj.dsm.e_min / s_base
            e_max = edisgo_obj.dsm.e_max / s_base
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
        flex_loads = np.concatenate((flexible_hps, flexible_cps))
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
