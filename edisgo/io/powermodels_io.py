"""
This module provides tools to convert eDisGo representation of the network
topology and timeseries to PowerModels network data format and to retrieve results from
PowerModels OPF in PowerModels network data format to eDisGo representation.
Call :func:`to_powermodels` to retrieve the PowerModels network container and
:func:`from_powermodels` to write OPF results to edisgo object.
"""

import json
import logging

import numpy as np
import pandas as pd
import pypsa

from edisgo.flex_opt.costs import line_expansion_costs
from edisgo.tools.tools import calculate_impedance_for_parallel_components

logger = logging.getLogger(__name__)


def to_powermodels(
    edisgo_object,
    s_base=1,
    flexible_cps=None,
    flexible_hps=None,
    flexible_loads=None,
    flexible_storages=None,
    opf_version=1,
):
    """
    Convert eDisGo representation of the network topology and timeseries to
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
    flexible_storages: :numpy:`numpy.ndarray<ndarray>` or list or None
        Array containing all flexible storages. Non-flexible storages operate to
        optimize self consumption.
        Default: None
    opf_version: Int
        Version of optimization models to choose from. Must be one of [1, 2, 3, 4].
        For more information see :func:`edisgo.opf.powermodels_opf.pm_optimize`.
        Default: 1
    Returns
    -------
    pm: dict
        Dictionary that contains all network data in PowerModels network data
        format.
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from overlying_grid component of edisgo object.
    """
    if opf_version in [2, 3, 4]:
        opf_flex = ["curt"]
    else:
        opf_flex = []
    if flexible_cps is None:
        flexible_cps = []
    if flexible_hps is None:
        flexible_hps = []
    if flexible_loads is None:
        flexible_loads = []
    if flexible_storages is None:
        flexible_storages = []
    # Append names of flexibilities for OPF
    for (flex, loads, text) in [
        ("cp", flexible_cps, "Flexible charging parks"),
        ("hp", flexible_hps, "Flexible heatpumps"),
        ("dsm", flexible_loads, "Flexible loads"),
        (
            "storage",
            flexible_storages,
            "Storage units",
        ),  # edisgo_object.topology.storage_units_df
    ]:
        if (flex not in opf_flex) & (len(loads) != 0):
            logger.info("{} will be optimized.".format(text))
            opf_flex.append(flex)
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
    pm["flexibilities"] = opf_flex
    _build_bus(psa_net, pm, flexible_storages)
    _build_gen(edisgo_object, psa_net, pm, flexible_storages, s_base)
    _build_branch(edisgo_object, psa_net, pm, flexible_storages, s_base)
    if len(flexible_storages) > 0:
        _build_battery_storage(edisgo_object, psa_net, pm, flexible_storages, s_base)
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
        _build_heat_storage(psa_net, pm, edisgo_object, s_base, flexible_hps)
    if len(flexible_loads) > 0:
        flexible_loads = _build_dsm(edisgo_object, psa_net, pm, s_base, flexible_loads)
    if len(psa_net.loads) > 0:
        _build_load(
            edisgo_object,
            psa_net,
            pm,
            s_base,
            flexible_cps,
            flexible_hps,
            flexible_storages,
        )
    else:
        logger.warning("No loads found in network.")
    if (opf_version == 3) | (opf_version == 4):
        hv_flex_dict = {
            "curt": edisgo_object.overlying_grid.renewables_curtailment / s_base,
            "storage": edisgo_object.overlying_grid.storage_units_active_power / s_base,
            "cp": edisgo_object.overlying_grid.electromobility_active_power / s_base,
            "hp": (
                edisgo_object.overlying_grid.heat_pump_rural_active_power
                + edisgo_object.overlying_grid.heat_central_active_power
            )
            / s_base,
            "dsm": edisgo_object.overlying_grid.dsm_active_power / s_base,
        }
        try:
            _build_hv_requirements(
                psa_net,
                pm,
                s_base,
                opf_flex,
                flexible_cps,
                flexible_hps,
                flexible_storages,
                hv_flex_dict,
            )
        except IndexError:
            logger.warning(
                "Overlying grid component of eDisGo object has no entries."
                " Changing optimization version to '4' (without high voltage"
                " requirements)."
            )
            opf_version = 1

    pm["opf_version"] = opf_version

    _build_timeseries(
        psa_net,
        pm,
        edisgo_object,
        s_base,
        flexible_cps,
        flexible_hps,
        flexible_loads,
        flexible_storages,
        opf_flex,
        hv_flex_dict,
    )
    return pm, hv_flex_dict


def from_powermodels(
    edisgo_object,
    pm_results,
    hv_flex_dict,
    s_base=1,
    save_heat_storage=True,
    save_slack_gen=True,
    save_slacks=True,
):
    """
    Convert results from optimization in PowerModels network data format to eDisGo data
    format and updates timeseries values of flexibilities on eDisGo object.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    pm_results: dict or str
        Dictionary or path to json file that contains all optimization results in
        PowerModels network data format.
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from overlying grid component of edisgo object.
    s_base : int
        Base value of apparent power for per unit system.
        Default: 1 MVA
    save_heat_storage: bool
        Indicates whether to save results of heat storage variables from the
        optimization to eDisGo object.
            Default: True
    save_slack_gen: bool
        Indicates whether to save results of slack generator variables from the
        optimization to eDisGo object.
        Default: True
    save_slacks: bool
        Indicates whether to save results of slack variables from the OPF run to eDisGo
        object. Depending on the chosen opf_version, different slacks are created and
        saved:
        1 : -
        2 : grid related slacks (load shedding, dispatchable and non-dispatchable
            generator curtailment, heat pump slack)
        3 : high voltage requirement slacks
        4 : high voltage requirements slacks and grid related slacks cf. version 2

        Default: False
    """

    if type(pm_results) == str:
        with open(pm_results) as f:
            pm = json.loads(json.load(f))
    elif type(pm_results) == dict:
        pm = pm_results
    else:
        raise ValueError(
            "Parameter 'pm_results' must be either dictionary or path to json file."
        )
    edisgo_object.opf_results.solution_time = pm["solve_time"]
    edisgo_object.opf_results.status = pm["status"]
    edisgo_object.opf_results.solver = pm["solver"]
    flex_dicts = {
        "curt": ["gen_nd", "pgc"],
        "hp": ["heatpumps", "php"],
        "cp": ["electromobility", "pcp"],
        "storage": ["storage", "pf"],
        "dsm": ["dsm", "pdsm"],
    }

    timesteps = pd.Series([int(k) for k in pm["nw"].keys()]).sort_values().values

    # write active power OPF results to edisgo object
    for flexibility in pm_results["nw"]["1"]["flexibilities"]:
        flex, variable = flex_dicts[flexibility]
        names = [
            pm["nw"]["1"][flex][flex_comp]["name"]
            for flex_comp in list(pm["nw"]["1"][flex].keys())
        ]
        # replace storage power values by branch power values of virtual branch to
        # account for losses and save efficiency values to edisgo object
        if flex == "storage":
            branches = [
                pm["nw"]["1"][flex][flex_comp]["virtual_branch"]
                for flex_comp in list(pm["nw"]["1"][flex].keys())
            ]
            data = [
                [
                    -pm["nw"][str(t)]["branch"][branch][variable] * s_base
                    for branch in branches
                ]
                for t in timesteps
            ]
            data2 = [
                [
                    pm["nw"][str(t)][flex][flex_comp]["ps"] * s_base
                    for flex_comp in list(pm["nw"]["1"][flex].keys())
                ]
                for t in timesteps
            ]
            efficiency = pd.DataFrame(
                index=edisgo_object.timeseries.timeindex,
                columns=names,
                data=abs(pd.DataFrame(data) / pd.DataFrame(data2)).values,
            )
            edisgo_object.opf_results.storage_efficiency = efficiency
        else:
            data = [
                [
                    pm["nw"][str(t)][flex][flex_comp][variable] * s_base
                    for flex_comp in list(pm["nw"]["1"][flex].keys())
                ]
                for t in timesteps
            ]
        results = pd.DataFrame(index=timesteps, columns=names, data=data)
        if flex == "gen_nd":
            # edisgo_object.timeseries._generators_active_power.loc[:, names] = (
            #     edisgo_object.timeseries.generators_active_power.loc[:, names].values
            #     - results[names].values
            # )
            pass
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
    if pm["nw"]["1"]["opf_version"] in [3, 4]:
        df = _result_df(
            pm,
            "HV_requirements",
            "phvs",
            timesteps,
            edisgo_object.timeseries.timeindex,
            s_base,
        )
        # save HV slack results to edisgo object
        if save_slacks:
            edisgo_object.opf_results.hv_requirement_slacks_t = df

        # calculate relative error
        for flex in df.columns:
            df[flex] = abs(df[flex].values - hv_flex_dict[flex]) / hv_flex_dict[flex]
        # write results to edisgo object
        edisgo_object.overlying_grid.opf_results = pd.DataFrame(
            columns=[
                "Highest relative error",
                "Mean relative error",
                "Sum relative error",
            ],
            index=df.columns,
            data=np.asarray(
                [df.max().values, (df.sum() / len(df)).values, df.sum().values]
            ).transpose(),
        )

        for flex in df.columns:
            if (
                edisgo_object.overlying_grid.opf_results["Highest relative error"][flex]
                > 0.05
            ).any():
                logger.warning(
                    "Highest relative error of {} variable exceeds 5%.".format(flex)
                )

    if save_slack_gen:  # save slack generator variable to edisgo object
        df = pd.DataFrame(
            index=edisgo_object.timeseries.timeindex, columns=["pg", "qg"]
        )
        for gen in list(pm["nw"]["1"]["gen_slack"].keys()):
            df["pg"] = [
                pm["nw"][str(t)]["gen_slack"][gen]["pgs"] * s_base for t in timesteps
            ]
            df["qg"] = [
                pm["nw"][str(t)]["gen_slack"][gen]["qgs"] * s_base for t in timesteps
            ]
        edisgo_object.opf_results.slack_generator_t = df

    if save_heat_storage:  # save heat storage variables to edisgo object
        df = _result_df(
            pm,
            "heat_storage",
            "phs",
            timesteps,
            edisgo_object.timeseries.timeindex,
            s_base,
        )
        edisgo_object.opf_results.heat_storage_t.p = df
        df = _result_df(
            pm,
            "heat_storage",
            "hse",
            timesteps,
            edisgo_object.timeseries.timeindex,
            s_base,
        )
        edisgo_object.opf_results.heat_storage_t.e = df

    if (pm["nw"]["1"]["opf_version"] in [2, 4]) & save_slacks:
        slacks = [
            ("gen", "pgens"),
            ("gen_nd", "pgc"),
            ("load", "pds"),
            ("electromobility", "pcps"),
        ]
        for comp, var in slacks:
            # save slacks to edisgo object
            df = _result_df(
                pm, comp, var, timesteps, edisgo_object.timeseries.timeindex, s_base
            )
            if comp == "gen":
                edisgo_object.opf_results.grid_slacks_t.gen_d_crt = df
            elif comp == "gen_nd":
                edisgo_object.opf_results.grid_slacks_t.gen_nd_crt = df
            elif comp == "load":
                edisgo_object.opf_results.grid_slacks_t.load_shedding = df
            elif comp == "electromobility":
                edisgo_object.opf_results.grid_slacks_t.cp_load_shedding = df

    # save line flows and currents to edisgo object
    for variable in ["pf", "qf", "ccm"]:
        df = _result_df(
            pm,
            "branch",
            variable,
            timesteps,
            edisgo_object.timeseries.timeindex,
            s_base,
        )
        if variable == "pf":
            edisgo_object.opf_results.lines_t.p = df
        elif variable == "qf":
            edisgo_object.opf_results.lines_t.q = df
        elif variable == "ccm":
            edisgo_object.opf_results.lines_t.ccm = df
    # Save bus voltages to edisgo object
    # df = _result_df(
    #     pm,
    #     "bus",
    #     "w",
    #     timesteps,
    #     edisgo_object.timeseries.timeindex,
    #     1,
    # )
    # edisgo_object.opf_results.buses_t.w = df


def _init_pm():
    """
    Initialize empty PowerModels dictionary.

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


def _build_bus(psa_net, pm, flexible_storages):
    """
    Build bus dictionary in PowerModels network data format and add it to
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
            "storage": False,
            "name": psa_net.buses.index[bus_i],
            "base_kv": psa_net.buses.v_nom[bus_i],
            "grid_level": grid_level[psa_net.buses.v_nom[bus_i]],
        }

    for stor_i in np.arange(len(flexible_storages)):
        idx_bus = _mapping(
            psa_net, psa_net.storage_units.bus.loc[flexible_storages[stor_i]]
        )
        pm["bus"][str(stor_i + len(psa_net.buses.index) + 1)] = {
            "index": stor_i + len(psa_net.buses.index) + 1,
            "bus_i": stor_i + len(psa_net.buses.index) + 1,
            "zone": 1,
            "bus_type": bus_types_int[idx_bus - 1],
            "vmax": v_max[idx_bus - 1],
            "vmin": v_min[idx_bus - 1],
            "va": 0,
            "vm": 1,
            "storage": True,
            "name": psa_net.buses.index[idx_bus - 1] + "_bss",
            "base_kv": psa_net.buses.v_nom[idx_bus - 1],
            "grid_level": grid_level[psa_net.buses.v_nom[idx_bus - 1]],
        }


def _build_gen(edisgo_obj, psa_net, pm, flexible_storages, s_base):
    """
    Build slack, dispatchable and non-dispatchable generator dictionaries in PowerModels
    network data format and add them to PowerModels dictionary 'pm'.

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

    for gen, text in [
        (gen_disp, "gen"),
        (gen_nondisp, "gen_nd"),
        (gen_slack, "gen_slack"),
    ]:
        for gen_i in np.arange(len(gen.index)):
            idx_bus = _mapping(psa_net, gen.bus[gen_i])
            pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "gen")
            q = [
                sign * np.tan(np.arccos(pf)) * gen.p_nom[gen_i],
                sign * np.tan(np.arccos(pf)) * gen.p_nom_min[gen_i],
            ]
            pm[text][str(gen_i + 1)] = {
                "pg": psa_net.generators_t.p_set[gen.index[gen_i]][0] / s_base,
                "qg": psa_net.generators_t.q_set[gen.index[gen_i]][0] / s_base,
                "pmax": gen.p_nom[gen_i] / s_base,
                "pmin": gen.p_nom_min[gen_i] / s_base,
                "qmax": max(q) / s_base,
                "qmin": min(q) / s_base,
                "P": 0,
                "Q": 0,
                "vg": 1,
                "pf": pf,
                "sign": sign,
                "mbase": gen.p_nom[gen_i] / s_base,
                "gen_bus": idx_bus,
                "gen_status": 1,
                "name": gen.index[gen_i],
                "index": gen_i + 1,
            }

    inflexible_storages = psa_net.storage_units.index[
        [storage not in flexible_storages for storage in psa_net.storage_units.index]
    ]
    if len(inflexible_storages) > 0:
        for stor_i in np.arange(len(inflexible_storages)):
            idx_bus = _mapping(
                psa_net, psa_net.storage_units.bus.loc[inflexible_storages[stor_i]]
            )
            pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "storage")
            p_g = max(
                [psa_net.storage_units_t.p_set[inflexible_storages[stor_i]][0], 0]
            )
            q_g = min(
                [psa_net.storage_units_t.q_set[inflexible_storages[stor_i]][0], 0]
            )
            pm["gen"][str(stor_i + len(gen_disp.index) + 1)] = {
                "pg": p_g / s_base,
                "qg": q_g / s_base,
                "pmax": psa_net.storage_units.p_nom.loc[inflexible_storages[stor_i]]
                / s_base,
                "pmin": -psa_net.storage_units.p_nom.loc[inflexible_storages[stor_i]]
                / s_base,
                "qmax": np.tan(np.arccos(pf))
                * psa_net.storage_units.p_nom.loc[inflexible_storages[stor_i]]
                / s_base,
                "qmin": -np.tan(np.arccos(pf))
                * psa_net.storage_units.p_nom.loc[inflexible_storages[stor_i]]
                / s_base,
                "P": 0,
                "Q": 0,
                "vg": 1,
                "pf": pf,
                "sign": sign,
                "mbase": psa_net.storage_units.p_nom.loc[inflexible_storages[stor_i]]
                / s_base,
                "gen_bus": idx_bus,
                "gen_status": 1,
                "name": inflexible_storages[stor_i],
                "index": stor_i + len(gen_disp.index) + 1,
            }


def _build_branch(edisgo_obj, psa_net, pm, flexible_storages, s_base):
    """
    Build branch dictionary in PowerModels network data format and add it to
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
        Default: 1 MVA
    """
    branches = pd.concat([psa_net.lines, psa_net.transformers])
    transformer = ~branches.tap_ratio.isna()
    tap = branches.tap_ratio.fillna(1)
    shift = branches.phase_shift.fillna(0)
    length = branches.length.fillna(1)
    for par, val, decimal, decade, quant, text, unit in [
        ("r_pu", branches.r_pu, 5, 1e5, 0.005, "resistance", "p.u."),
        ("x_pu", branches.x_pu, 5, 1e5, 0.005, "reactance", "p.u."),
        ("length", length, 3, 1e3, 0.002, "branch length", "km"),
    ]:
        max_value = np.round(max(val.loc[val < val.quantile(1 - quant)]), decimal)
        min_value = np.round(min(val.loc[val > val.quantile(quant)]), decimal)
        if val.max() / val.min() > decade:
            # only modify r, x and l values if min/max value differences are too big
            branches[par] = val.clip(lower=min_value, upper=max_value)
            logger.warning(
                "Range between min and max {} values is too high. Highest and "
                "lowest {}% of {} values will be set to {} {} and"
                " {} {}, respectively.".format(
                    text, 100 * quant, text, max_value, unit, min_value, unit
                )
            )

    for branch_i in np.arange(len(branches.index)):
        idx_f_bus = _mapping(psa_net, branches.bus0[branch_i])
        idx_t_bus = _mapping(psa_net, branches.bus1[branch_i])
        cost_factor = {"mv": 1, "lv": 100}
        pm["branch"][str(branch_i + 1)] = {
            "name": branches.index[branch_i],
            "br_r": branches.r_pu[branch_i] * s_base,
            "r": branches.r[branch_i],
            "br_x": branches.x_pu[branch_i] * s_base,
            "f_bus": idx_f_bus,
            "t_bus": idx_t_bus,
            "g_to": branches.g_pu[branch_i] / 2 * s_base,
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
            "storage": False,
            "tap": tap[branch_i],
            "length": branches.length.fillna(1)[branch_i],
            "cost": branches.capital_cost[branch_i],
            "cost_factor": cost_factor[pm["bus"][str(idx_f_bus)]["grid_level"]],
            "cost_inverse": 1
            / (branches.length.fillna(1)[branch_i] * branches.capital_cost[branch_i]),
            "storage_pf": 0,
            "index": branch_i + 1,
        }

    for stor_i in np.arange(len(flexible_storages)):
        idx_bus = _mapping(
            psa_net, psa_net.storage_units.bus.loc[flexible_storages[stor_i]]
        )
        # retrieve power factor from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "storage")

        pm["branch"][str(stor_i + len(branches.index) + 1)] = {
            "name": "bss_branch_" + str(stor_i + 1),
            "br_r": 0.017 * s_base / (psa_net.buses.v_nom[idx_bus - 1] ** 2),
            "r": 0.017,
            "br_x": 0,
            "f_bus": idx_bus,
            "t_bus": stor_i + len(psa_net.buses.index) + 1,
            "g_to": 0,
            "g_fr": 0,
            "b_to": 0,
            "b_fr": 0,
            "shift": 0,
            "br_status": 1.0,
            "rate_a": psa_net.storage_units.p_nom.loc[flexible_storages[stor_i]]
            / s_base,
            "rate_b": 250 / s_base,
            "rate_c": 250 / s_base,
            "angmin": -np.pi / 6,
            "angmax": np.pi / 6,
            "transformer": False,
            "storage": True,
            "tap": 1,
            "length": 1,
            "cost": 0,
            "cost_factor": 0,
            "cost_inverse": 0,
            "storage_pf": np.tan(np.arccos(pf)) * sign,
            "index": stor_i + len(branches.index) + 1,
        }


def _build_load(
    edisgo_obj, psa_net, pm, s_base, flexible_cps, flexible_hps, flexible_storages
):
    """
    Build load dictionary in PowerModels network data format and add it to
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
    flexible_storages: :numpy:`numpy.ndarray<ndarray>` or list or None
        Array containing all flexible storages. Non-flexible storages operate to
        optimize self consumption.
        Default: None
    """
    flex_loads = np.concatenate((flexible_hps, flexible_cps))
    inflexible_storages = psa_net.storage_units.index[
        [storage not in flexible_storages for storage in psa_net.storage_units.index]
    ]
    if len(flex_loads) == 0:
        loads_df = psa_net.loads
    else:
        loads_df = psa_net.loads.drop(flex_loads)
    for load_i in np.arange(len(loads_df.index)):
        idx_bus = _mapping(psa_net, loads_df.bus[load_i])
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "load")
        p_d = psa_net.loads_t.p_set[loads_df.index[load_i]]
        q_d = psa_net.loads_t.q_set[loads_df.index[load_i]]
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
    if len(inflexible_storages) > 0:
        for stor_i in np.arange(len(inflexible_storages)):
            idx_bus = _mapping(
                psa_net, psa_net.storage_units.bus.loc[inflexible_storages[stor_i]]
            )
            pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "storage")
            p_d = -min(
                [psa_net.storage_units_t.p_set[inflexible_storages[stor_i]][0], 0]
            )
            q_d = -max(
                [psa_net.storage_units_t.q_set[inflexible_storages[stor_i]][0], 0]
            )
            pm["load"][str(stor_i + len(loads_df.index) + 1)] = {
                "pd": p_d / s_base,
                "qd": q_d / s_base,
                "load_bus": idx_bus,
                "status": True,
                "pf": pf,
                "sign": sign,
                "name": inflexible_storages[stor_i],
                "index": stor_i + len(loads_df.index) + 1,
            }


def _build_battery_storage(edisgo_obj, psa_net, pm, flexible_storages, s_base):
    """
    Build battery storage dictionary in PowerModels network data format and add
    it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    flexible_storages: :numpy:`numpy.ndarray<ndarray>` or list or None
        Array containing all flexible storages. Non-flexible storages operate to
        optimize self consumption.
        Default: None
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    """
    branches = pd.concat([psa_net.lines, psa_net.transformers])
    for stor_i in np.arange(len(flexible_storages)):
        idx_bus = _mapping(
            psa_net, psa_net.storage_units.bus.loc[flexible_storages[stor_i]]
        )
        # retrieve power factor from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "storage")
        e_max = (
            psa_net.storage_units.p_nom.loc[flexible_storages[stor_i]]
            * psa_net.storage_units.max_hours.loc[flexible_storages[stor_i]]
        )
        pm["storage"][str(stor_i + 1)] = {
            "r": 0,
            "x": 0,
            "p_loss": 0,
            "q_loss": 0,
            "pf": pf,
            "sign": sign,
            "virtual_branch": str(stor_i + len(branches.index) + 1),
            "ps": psa_net.storage_units.p_set[flexible_storages[stor_i]] / s_base,
            "qs": psa_net.storage_units.q_set[flexible_storages[stor_i]] / s_base,
            "pmax": psa_net.storage_units.p_nom.loc[flexible_storages[stor_i]] / s_base,
            "pmin": -psa_net.storage_units.p_nom.loc[flexible_storages[stor_i]]
            / s_base,
            "qmax": np.tan(np.arccos(pf))
            * psa_net.storage_units.p_nom.loc[flexible_storages[stor_i]]
            / s_base,
            "qmin": -np.tan(np.arccos(pf))
            * psa_net.storage_units.p_nom.loc[flexible_storages[stor_i]]
            / s_base,
            "energy": psa_net.storage_units.state_of_charge_initial.loc[  # 0, für OG
                flexible_storages[stor_i]
            ]
            * e_max
            / s_base,
            "soc_initial": (  # 0, für OG
                edisgo_obj.timeseries.storage_units_state_of_charge[
                    flexible_storages[stor_i]
                ].iloc[0]
                + edisgo_obj.timeseries.storage_units_active_power[
                    flexible_storages[stor_i]
                ].iloc[0]
                * 0.9
            ),
            "soc_end": edisgo_obj.timeseries.storage_units_state_of_charge[  # 0, für OG
                flexible_storages[stor_i]
            ].iloc[-1],
            "energy_rating": e_max / s_base,
            "thermal_rating": 100,
            "charge_rating": psa_net.storage_units.p_nom.loc[flexible_storages[stor_i]]
            / s_base,
            "discharge_rating": psa_net.storage_units.p_nom.loc[
                flexible_storages[stor_i]
            ]
            / s_base,
            "charge_efficiency": 0.9,
            "discharge_efficiency": 0.9,
            "storage_bus": stor_i + len(psa_net.buses.index) + 1,
            "name": flexible_storages[stor_i],
            "status": True,
            "index": stor_i + 1,
        }


def _build_electromobility(edisgo_obj, psa_net, pm, s_base, flexible_cps):
    """
    Build electromobility dictionary and add it to PowerModels dictionary 'pm'.

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
        eta = edisgo_obj.electromobility.simbev_config_df.eta_cp.values[0]
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "cp")
        q = (
            sign
            * np.tan(np.arccos(pf))
            * flex_bands_df["upper_power"][emob_df.index[cp_i]][0]
        )
        p_max = flex_bands_df["upper_power"][emob_df.index[cp_i]]
        e_min = flex_bands_df["lower_energy"][emob_df.index[cp_i]]
        e_max = flex_bands_df["upper_energy"][emob_df.index[cp_i]]
        try:
            soc_initial = edisgo_obj.electromobility.initial_soc_df[emob_df.index[cp_i]]
        except AttributeError:
            soc_initial = 1 / 2 * (e_min[0] + e_max[0])
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
            "energy": soc_initial,
            "eta": eta,
            "cp_bus": idx_bus,
            "name": emob_df.index[cp_i],
            "index": cp_i + 1,
        }
    return flexible_cps


def _build_heatpump(psa_net, pm, edisgo_obj, s_base, flexible_hps):
    """
    Build heat pump dictionary and add it to PowerModels dictionary 'pm'.

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
    # solarthermal_feedin = (
    # edisgo_obj.overlying_grid.geothermal_energy_feedin_district_heating)
    # geothermal_feedin = (
    # edisgo_obj.overlying_grid.solarthermal_energy_feedin_district_heating)
    for hp_i in np.arange(len(heat_df.index)):
        idx_bus = _mapping(psa_net, heat_df.bus[hp_i])
        # retrieve power factor and sign from config
        pf, sign = _get_pf(edisgo_obj, pm, idx_bus, "hp")
        q = sign * np.tan(np.arccos(pf)) * heat_df.p_set[hp_i]
        p_d = heat_df2[heat_df.index[hp_i]]
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
                " heat storage capacities are available.".format(
                    pm["bus"][str(idx_bus)]["name"], heat_df.index[hp_i]
                )
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
    Build heat storage dictionary and add it to PowerModels dictionary 'pm'.

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
    heat_storage_df["state_of_charge_initial"] = 0  # ToDo: Zeile löschen
    for stor_i in np.arange(len(heat_storage_df.index)):
        idx_bus = _mapping(psa_net, psa_net.loads.bus[stor_i])
        pm["heat_storage"][str(stor_i + 1)] = {
            "ps": 0,
            "p_loss": 0.04,  # 4% of SOC per day
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
    Build dsm 'storage' dictionary and add it to PowerModels dictionary 'pm'.

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
        q = [
            sign * np.tan(np.arccos(pf)) * p_max[0],
            sign * np.tan(np.arccos(pf)) * p_min[0],
        ]
        pm["dsm"][str(dsm_i + 1)] = {
            "pd": 0,
            "qd": 0,
            "pf": pf,
            "sign": sign,
            "energy": 0 / s_base,
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


def _build_hv_requirements(
    psa_net,
    pm,
    s_base,
    opf_flex,
    flexible_cps,
    flexible_hps,
    flexible_storages,
    hv_flex_dict,
):
    """
    Build dictionary for HV requirement data in PowerModels network data format and
    add it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    s_base : int
        Base value of apparent power for per unit system.
        Default: 100 MVA
    opf_flex : list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]. For more information see
        :func:`edisgo.opf.powermodels_opf.pm_optimize`.
    flexible_cps : :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>` or list
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_storages: :numpy:`numpy.ndarray<ndarray>` or list or None
        Array containing all flexible storages. Non-flexible storages operate to
        optimize self consumption.
        Default: None
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from overlying grid component of edisgo object.
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
    inflexible_storages = psa_net.storage_units.index[
        [storage not in flexible_storages for storage in psa_net.storage_units.index]
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
    if len(inflexible_storages) > 0:
        hv_flex_dict["storage"] = (
            hv_flex_dict["storage"]
            - psa_net.storage_units_t.p_set.loc[:, inflexible_storages].sum(axis=1)
            / s_base
        )
    for i in np.arange(len(opf_flex)):
        pm["HV_requirements"][str(i + 1)] = {
            "P": hv_flex_dict[opf_flex[i]][0],
            "name": opf_flex[i],
        }


def _build_timeseries(
    psa_net,
    pm,
    edisgo_obj,
    s_base,
    flexible_cps,
    flexible_hps,
    flexible_loads,
    flexible_storages,
    opf_flex,
    hv_flex_dict,
):
    """
    Build timeseries dictionary in PowerModels network data format and add it to
    PowerModels dictionary 'pm'. PowerModels' timeseries dictionary contains one
    timeseries dictionary each for: gen, load, battery storage,
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
    flexible_storages: :numpy:`numpy.ndarray<ndarray>` or list or None
        Array containing all flexible storages. Non-flexible storages operate to
        optimize self consumption.
        Default: None
    opf_flex: list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from overlying_grid component of edisgo object.
    """
    for kind in [
        "gen",
        "gen_nd",
        "gen_slack",
        "load",
        # "storage",
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
            flexible_storages,
            opf_flex,
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
    flexible_storages=None,
    opf_flex=None,
    hv_flex_dict=None,
):
    """
    Build timeseries dictionary for given kind and add it to 'time_series'
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
    flexible_storages: :numpy:`numpy.ndarray<ndarray>` or list or None
        Array containing all flexible storages. Non-flexible storages operate to
        optimize self consumption.
        Default: None
    opf_flex: list
        List of flexibilities that should be considered in the optimization. Must be any
        subset of ["curt", "storage", "cp", "hp", "dsm"]
    hv_flex_dict: dict
        Dictionary containing time series of HV requirement for each flexibility
        retrieved from overlying grid component of edisgo object.
    """
    pm_comp = dict()
    if kind == "gen":
        p_set2 = (
            psa_net.generators_t.p_set.loc[
                :,
                ~(psa_net.generators_t.p_set.columns.str.contains("solar"))
                & ~(psa_net.generators_t.p_set.columns.str.contains("wind"))
                & ~(psa_net.generators_t.p_set.columns.str.contains("slack")),
            ]
            / s_base
        )
        q_set2 = psa_net.generators_t.q_set[p_set2.columns] / s_base
        inflexible_storages = psa_net.storage_units.index[
            [
                storage not in flexible_storages
                for storage in psa_net.storage_units.index
            ]
        ]
        p_set = (
            pd.concat(
                [
                    p_set2,
                    psa_net.storage_units_t.p_set[inflexible_storages].clip(lower=0),
                ],
                axis=1,
            )
            / s_base
        )
        q_set = (
            pd.concat(
                [
                    q_set2,
                    psa_net.storage_units_t.q_set[inflexible_storages].clip(upper=0),
                ],
                axis=1,
            )
            / s_base
        )

        for comp in p_set.columns:
            comp_i = _mapping(psa_net, comp, kind, flexible_storages=flexible_storages)
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }

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
        for comp in p_set.columns:
            comp_i = _mapping(psa_net, comp, kind)
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }
    elif kind == "gen_slack":
        p_set = (
            psa_net.generators_t.p_set.loc[
                :,
                psa_net.generators_t.p_set.columns.str.contains("slack"),
            ]
            / s_base
        )
        q_set = psa_net.generators_t.q_set[p_set.columns] / s_base
        for comp in p_set.columns:
            comp_i = _mapping(psa_net, comp, kind)
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }
    elif kind == "load":
        flex_loads = np.concatenate((flexible_hps, flexible_cps))
        inflexible_storages = psa_net.storage_units.index[
            [
                storage not in flexible_storages
                for storage in psa_net.storage_units.index
            ]
        ]
        if len(flex_loads) == 0:
            p_set = (
                pd.concat(
                    [
                        psa_net.loads_t.p_set,
                        -1
                        * psa_net.storage_units_t.p_set[inflexible_storages].clip(
                            upper=0
                        ),
                    ],
                    axis=1,
                )
                / s_base
            )
            q_set = (
                pd.concat(
                    [
                        psa_net.loads_t.q_set,
                        -1
                        * psa_net.storage_units_t.q_set[inflexible_storages].clip(
                            lower=0
                        ),
                    ],
                    axis=1,
                )
                / s_base
            )
        else:
            p_set = (
                pd.concat(
                    [
                        psa_net.loads_t.p_set.drop(columns=flex_loads),
                        -1
                        * psa_net.storage_units_t.p_set[inflexible_storages].clip(
                            upper=0
                        ),
                    ],
                    axis=1,
                )
                / s_base
            )
            q_set = (
                pd.concat(
                    [
                        psa_net.loads_t.q_set.drop(columns=flex_loads),
                        -1
                        * psa_net.storage_units_t.q_set[inflexible_storages].clip(
                            lower=0
                        ),
                    ],
                    axis=1,
                )
                / s_base
            )
        for comp in p_set.columns:
            comp_i = _mapping(
                psa_net,
                comp,
                kind,
                flexible_cps=flexible_cps,
                flexible_hps=flexible_hps,
                flexible_loads=flexible_loads,
                flexible_storages=flexible_storages,
            )
            p_d = p_set[comp].values
            q_d = q_set[comp].values
            pm_comp[str(comp_i)] = {
                "pd": p_d.tolist(),
                "qd": q_d.tolist(),
            }
    elif kind == "electromobility":
        if len(flexible_cps) > 0:
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
            for comp in p_set.columns:
                comp_i = _mapping(psa_net, comp, kind, flexible_cps=flexible_cps)
                pm_comp[str(comp_i)] = {
                    "p_max": p_set[comp].values.tolist(),
                    "e_min": e_min[comp].values.tolist(),
                    "e_max": e_max[comp].values.tolist(),
                }
    elif kind == "heatpumps":
        if len(flexible_hps) > 0:
            p_set = edisgo_obj.heat_pump.heat_demand_df[flexible_hps] / s_base
            cop = edisgo_obj.heat_pump.cop_df[flexible_hps]
            for comp in p_set.columns:
                comp_i = _mapping(psa_net, comp, kind, flexible_hps=flexible_hps)
                pm_comp[str(comp_i)] = {
                    "pd": p_set[comp].values.tolist(),
                    "cop": cop[comp].values.tolist(),
                }
    elif kind == "dsm":
        if len(flexible_loads) > 0:
            p_set = edisgo_obj.dsm.p_max / s_base
            p_min = edisgo_obj.dsm.p_min / s_base
            e_min = edisgo_obj.dsm.e_min / s_base
            e_max = edisgo_obj.dsm.e_max / s_base
            for comp in p_set.columns:
                comp_i = _mapping(psa_net, comp, kind, flexible_loads=flexible_loads)
                pm_comp[str(comp_i)] = {
                    "p_max": p_set[comp].values.tolist(),
                    "p_min": p_min[comp].values.tolist(),
                    "e_min": e_min[comp].values.tolist(),
                    "e_max": e_max[comp].values.tolist(),
                }

    if (kind == "HV_requirements") & (
        (pm["opf_version"] == 3) | (pm["opf_version"] == 4)
    ):
        for i in np.arange(len(opf_flex)):
            pm_comp[(str(i + 1))] = {
                "P": hv_flex_dict[opf_flex[i]].tolist(),
            }

    pm["time_series"][kind] = pm_comp


def _mapping(
    psa_net,
    name,
    kind="bus",
    flexible_cps=None,
    flexible_hps=None,
    flexible_loads=None,
    flexible_storages=None,
):

    if kind == "bus":
        df = psa_net.buses
    elif kind == "gen":
        df2 = psa_net.generators.loc[
            ~(psa_net.generators.index.str.contains("solar"))
            & ~(psa_net.generators.index.str.contains("wind"))
            & ~(psa_net.generators.index.str.contains("slack"))
        ]
        inflexible_storages = psa_net.storage_units.index[
            [
                storage not in flexible_storages
                for storage in psa_net.storage_units.index
            ]
        ]
        df = pd.concat([df2, psa_net.storage_units.loc[inflexible_storages]])
    elif kind == "gen_nd":
        df = psa_net.generators.loc[
            (psa_net.generators.index.str.contains("solar"))
            | (psa_net.generators.index.str.contains("wind"))
        ]
    elif kind == "gen_slack":
        df = psa_net.generators.loc[(psa_net.generators.index.str.contains("slack"))]
    elif kind == "storage":
        df = psa_net.storage_units.loc[flexible_storages]
    elif kind == "load":
        flex_loads = np.concatenate((flexible_hps, flexible_cps))
        inflexible_storages = psa_net.storage_units.index[
            [
                storage not in flexible_storages
                for storage in psa_net.storage_units.index
            ]
        ]
        if len(flex_loads) == 0:
            df = pd.concat(
                [psa_net.loads, psa_net.storage_units.loc[inflexible_storages]]
            )
        else:
            df = pd.concat(
                [
                    psa_net.loads.drop(flex_loads),
                    psa_net.storage_units.loc[inflexible_storages],
                ]
            )
    elif kind == "electromobility":
        df = psa_net.loads.loc[flexible_cps]
    elif (kind == "heatpumps") | (kind == "heat_storage"):
        df = psa_net.loads.loc[flexible_hps]
    elif kind == "dsm":
        df = psa_net.loads.loc[flexible_loads]
    else:
        df = pd.DataFrame()
        logging.warning("Mapping for '{}' not implemented.".format(kind))
    idx = df.reset_index()[df.index == name].index[0] + 1
    return idx


def aggregate_parallel_transformers(psa_net):
    """
    Calculate impedance for parallel transformers and aggregate them. Replace
    psa_net.transformers dataframe by aggregated transformer dataframe.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.

    """
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
    Retrieve and return power factor from edisgo config files.

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


def _result_df(pm, component, variable, timesteps, index, s_base):
    cols = [
        pm["nw"]["1"][component][n]["name"]
        for n in list(pm["nw"]["1"][component].keys())
    ]
    data = [
        [
            pm["nw"][str(t)][component][n][variable] * s_base
            for n in list(pm["nw"]["1"][component].keys())
        ]
        for t in timesteps
    ]
    return pd.DataFrame(index=index, columns=cols, data=data)
