"""
This module provides tools to convert eDisGo representation of the network
topology and timeseries to PowerModels network data format. Call :func:`to_powermodels`
to retrieve the PowerModels network container.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from edisgo.tools.tools import calculate_impedance_for_parallel_components


def to_powermodels(edisgo_object, flexible_cps, flexible_hps, opt_version):
    """
    Converts eDisGo representation of the network topology and timeseries to
    PowerModels network data format.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    flexible_cps : :numpy:`numpy.ndarray<ndarray>`
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>`
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    opt_version: Int
        Version of optimization models to choose from. For more information see MA.
        Must be one of [1, 2].

    Returns
    -------
    pm: dict
        Dictionary that contains all network data in PowerModels network data
        format.
    """

    # convert eDisGo object to pypsa network structure
    psa_net = edisgo_object.to_pypsa(use_seed=True)
    # aggregate parallel transformers
    aggregate_parallel_transformers(psa_net)
    # calculate per unit values
    pypsa.pf.calculate_dependent_values(psa_net)
    # build PowerModels structure
    pm = _init_pm()
    timesteps = len(psa_net.snapshots)  # number of considered timesteps
    flexible_loads = (
        edisgo_object.dsm.e_max.columns
    )  # TODO: soll das auch an to_powermodels übergeben werden?
    pm["name"] = "ding0_{}_t_{}".format(edisgo_object.topology.id, timesteps)
    pm["time_elapsed"] = int(
        (psa_net.snapshots[1] - psa_net.snapshots[0]).seconds / 3600
    )  # length of timesteps in hours
    pm["baseMVA"] = 1
    pm["source_version"] = 2
    pm["opt_version"] = opt_version
    _build_bus(psa_net, pm)
    _build_gen(edisgo_object, psa_net, pm)
    _build_branch(psa_net, pm)
    _build_battery_storage(edisgo_object, psa_net, pm)
    _build_load(psa_net, pm, flexible_cps, flexible_hps, flexible_loads)
    _build_electromobility(edisgo_object, psa_net, pm, flexible_cps)
    _build_heatpump(psa_net, pm, edisgo_object, flexible_hps)
    _build_heat_storage(psa_net, pm, edisgo_object)
    _build_dsm(edisgo_object, psa_net, pm, flexible_loads)
    _build_HV_requirements(pm)
    _build_timeseries(
        psa_net, pm, edisgo_object, flexible_cps, flexible_hps, flexible_loads
    )
    pm["PF"] = {  # TODO: PF != 1 immer für beide voltage levels oder keins?
        "cp": edisgo_object.config._data["reactive_power_factor"]["mv_cp"],
        "hp": edisgo_object.config._data["reactive_power_factor"]["mv_hp"],
    }
    return pm


def _init_pm():
    # init empty PowerModels dictionary
    pm = {
        "gen": dict(),
        "gen_nd": dict(),
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
    grid_level = {20: "mv", 0.4: "lv"}
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


def _build_gen(edisgo_obj, psa_net, pm):
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
    # Divide in dispatchable and non-dispatchable generator sets
    gen_disp = psa_net.generators.loc[
        ~(psa_net.generators.index.str.contains("solar"))
        & ~(psa_net.generators.index.str.contains("wind"))
    ]
    gen_nondisp = psa_net.generators.loc[
        (psa_net.generators.index.str.contains("solar"))
        | (psa_net.generators.index.str.contains("wind"))
    ]

    # determine slack bus through slack generator
    slack_gen = psa_net.generators.bus[
        psa_net.generators.index == "Generator_slack"
    ].values[0]
    pm["bus"][str(_mapping(psa_net, slack_gen))]["bus_type"] = 3

    for gen_i in np.arange(len(gen_disp.index)):
        idx_bus = _mapping(psa_net, gen_disp.bus[gen_i])
        # retrieve power factor and sign from config
        pf_sign = _calculate_q(edisgo_obj, pm, idx_bus, "gen")
        q = [pf_sign * gen_disp.p_nom[gen_i], pf_sign * gen_disp.p_nom_min[gen_i]]
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
        pf_sign = _calculate_q(edisgo_obj, pm, idx_bus, "gen")
        q = [pf_sign * gen_nondisp.p_nom[gen_i], pf_sign * gen_nondisp.p_nom_min[gen_i]]
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
            "mbase": gen_nondisp.p_nom[gen_i],
            "gen_bus": idx_bus,
            "gen_status": 1,
            "index": gen_i + 1,
        }


def _build_branch(psa_net, pm):
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
            "index": branch_i + 1,
        }


def _build_load(psa_net, pm, flexible_cps, flexible_hps, flexible_loads):
    """
    Builds load dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    flexible_cps : :numpy:`numpy.ndarray<ndarray>`
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>`
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_loads : :numpy:`numpy.ndarray<ndarray>`
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    """
    loads_df = psa_net.loads.drop(
        np.concatenate((flexible_hps, flexible_cps, flexible_loads))
    )
    for load_i in np.arange(len(loads_df.index)):
        idx_bus = _mapping(psa_net, loads_df.bus[load_i])
        pm["load"][str(load_i + 1)] = {
            "pd": psa_net.loads_t.p_set[loads_df.index[load_i]][0],
            "qd": psa_net.loads_t.q_set[loads_df.index[load_i]][0],
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
        # retrieve power factor and sign from config
        pf_sign = _calculate_q(edisgo_obj, pm, idx_bus, "storage")
        q = [
            -pf_sign * psa_net.storage_units.p_nom[stor_i],
            pf_sign * psa_net.storage_units.p_nom[stor_i],
        ]
        pm["storage"][str(stor_i + 1)] = {
            "r": 0,
            "x": 0,
            "p_loss": 0,
            "q_loss": 0,
            "ps": psa_net.storage_units_t.p_set[psa_net.storage_units.index[stor_i]][0],
            "qs": psa_net.storage_units_t.q_set[psa_net.storage_units.index[stor_i]][0],
            "pmax": psa_net.storage_units.p_nom[stor_i],
            "pmin": -psa_net.storage_units.p_nom[stor_i],
            "qmax": max(q),
            "qmin": min(q),
            "energy": psa_net.storage_units.state_of_charge_initial[stor_i],
            "energy_rating": psa_net.storage_units.capacity[stor_i],
            "thermal_rating": 1,  # TODO unbegrenzt
            "charge_rating": psa_net.storage_units.p_nom[stor_i],
            "discharge_rating": psa_net.storage_units.p_nom[stor_i],
            "charge_efficiency": 1,
            "discharge_efficiency": 1,
            "storage_bus": idx_bus,
            "status": True,
            "index": stor_i + 1,
        }


def _build_electromobility(edisgo_obj, psa_net, pm, flexible_cps):
    """
    Builds electromobility dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    flexible_cps : :numpy:`numpy.ndarray<ndarray>`
        Array containing all charging points that allow for flexible charging.
    """
    if len(flexible_cps) == 0:
        print("There are no flexible charging points in network.")
    else:
        emob_df = psa_net.loads.loc[flexible_cps]
        flex_bands_df = edisgo_obj.electromobility.flexibility_bands
        for cp_i in np.arange(len(emob_df.index)):
            idx_bus = _mapping(psa_net, emob_df.bus[cp_i])
            # retrieve power factor and sign from config
            pf_sign = _calculate_q(edisgo_obj, pm, idx_bus, "cp")
            q = pf_sign * psa_net.storage_units.p_nom[cp_i]
            pm["electromobility"][str(cp_i + 1)] = {
                "pd": 0,
                "qd": 0,
                "p_min": 0,
                "p_max": flex_bands_df["upper_power"][emob_df.index[cp_i]][0],
                "q_min": min(q, 0),
                "q_max": max(q, 0),
                "e_min": flex_bands_df["lower_energy"][emob_df.index[cp_i]][0],
                "e_max": flex_bands_df["upper_energy"][emob_df.index[cp_i]][0],
                "cp_bus": idx_bus,
                "index": cp_i + 1,
            }


def _build_heatpump(psa_net, pm, edisgo_obj, flexible_hps):
    """
    Builds heat pump dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    edisgo_obj : :class:`~.EDisGo`
    flexible_hps: :numpy:`numpy.ndarray<ndarray>`
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.

    """
    if len(flexible_hps) == 0:
        print("There are no flexible heatpumps in network.")
    else:
        heat_df = psa_net.loads.loc[flexible_hps]
        for hp_i in np.arange(len(heat_df.index)):
            idx_bus = _mapping(psa_net, heat_df.bus[hp_i])
            # retrieve power factor and sign from config
            pf_sign = _calculate_q(edisgo_obj, pm, idx_bus, "hp")
            q = pf_sign * heat_df.p_set[hp_i]
            pm["heatpumps"][str(hp_i + 1)] = {
                "pd": psa_net.loads_t.p_set[heat_df.index[hp_i]][0],  # heat demand
                "p_min": 0,
                "p_max": heat_df.p_set[hp_i],
                "q_min": min(q, 0),
                "q_max": max(q, 0),
                "cop": edisgo_obj.heat_pump.cop_df[heat_df.index[hp_i]][0],
                "hp_bus": idx_bus,
                "index": hp_i + 1,
            }


def _build_heat_storage(psa_net, pm, edisgo_obj):
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

    heat_storage_df = edisgo_obj.heat_pump.thermal_storage_units_df
    for stor_i in np.arange(len(heat_storage_df.index)):
        idx_bus = _mapping(psa_net, psa_net.loads.bus[stor_i])
        pm["heat_storage"][str(stor_i + 1)] = {
            "ps": 0,
            "p_loss": 0,
            "energy": heat_storage_df.state_of_charge_initial[stor_i],
            "capacity": heat_storage_df.capacity[stor_i],
            "charge_efficiency": heat_storage_df.efficiency[stor_i],
            "discharge_efficiency": heat_storage_df.efficiency[stor_i],
            "storage_bus": idx_bus,
            "status": True,
            "index": stor_i + 1,
        }


def _build_dsm(edisgo_obj, psa_net, pm, flexible_loads):
    """
    Builds dsm 'storage' dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    flexible_loads : :numpy:`numpy.ndarray<ndarray>`
        Array containing all flexible loads that allow for application of demand side
        management strategy.
    """
    if len(flexible_loads) == 0:
        print("There are no flexible loads (DSM) in network.")
    else:
        dsm_df = psa_net.loads.loc[flexible_loads]
        for dsm_i in np.arange(len(dsm_df.index)):
            idx_bus = _mapping(psa_net, dsm_df.bus[dsm_i])
            # retrieve power factor and sign from config
            pf_sign = _calculate_q(edisgo_obj, pm, idx_bus, "load")
            q = [
                pf_sign * edisgo_obj.dsm.p_max[dsm_df.index[dsm_i]][0],
                pf_sign * edisgo_obj.dsm.p_min[dsm_df.index[dsm_i]][0],
            ]
            pm["dsm"][str(dsm_i + 1)] = {
                "pd": 0,
                "qd": 0,
                "energy": 0,  # TODO: am Anfang immer 0?
                "p_min": edisgo_obj.dsm.p_min[dsm_df.index[dsm_i]][0],
                "p_max": edisgo_obj.dsm.p_max[dsm_df.index[dsm_i]][0],
                "q_max": max(q),
                "q_min": min(q),
                "e_min": edisgo_obj.dsm.e_min[dsm_df.index[dsm_i]][0],
                "e_max": edisgo_obj.dsm.e_max[dsm_df.index[dsm_i]][0],
                "charge_efficiency": 1,
                "discharge_efficiency": 1,
                "dsm_bus": idx_bus,
                "index": dsm_i + 1,
            }


def _build_HV_requirements(pm):
    """
    Builds dictionary for HV requirement data in PowerModels network data format and
    adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    pm : dict
        (PowerModels) dictionary.
    """
    pm["HV_requirements"] = {
        "P_curt": 0,
        "Q_curt": 0,
        "P_cp": 0,
        "P_hp": 0,
        "Q_hp": 0,
        "P_dsm": 0,
        "Q_dsm": 0,
    }


def _build_timeseries(
    psa_net, pm, edisgo_obj, flexible_cps, flexible_hps, flexible_loads
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
    flexible_cps : :numpy:`numpy.ndarray<ndarray>`
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>`
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.
    flexible_loads : :numpy:`numpy.ndarray<ndarray>`
        Array containing all flexible loads that allow for application of demand side
        management strategy.

    """
    for kind in [
        "gen",
        "gen_nd",
        "load",
        "storage",
        "electromobility",
        "heatpumps",
        "dsm",
        "HV_requirements",
    ]:
        _build_component_timeseries(
            psa_net, pm, kind, edisgo_obj, flexible_cps, flexible_hps, flexible_loads
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
        Must be one of ["gen", "gen_nd", "load", "storage", "electromobility",
        "heatpumps", "heat_storage", "dsm", "HV_requirements"]
    edisgo_obj : :class:`~.EDisGo`
    flexible_cps : :numpy:`numpy.ndarray<ndarray>`
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>`
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.

    """
    pm_comp = dict()
    if kind == "gen":
        p_set = psa_net.generators_t.p_set.loc[
            :,
            ~(psa_net.generators_t.p_set.columns.str.contains("solar"))
            & ~(psa_net.generators_t.p_set.columns.str.contains("wind")),
        ]
        q_set = psa_net.generators_t.q_set.loc[
            :,
            ~(psa_net.generators_t.q_set.columns.str.contains("solar"))
            & ~(psa_net.generators_t.q_set.columns.str.contains("wind")),
        ]
    elif kind == "gen_nd":
        p_set = psa_net.generators_t.p_set.loc[
            :,
            psa_net.generators_t.p_set.columns.str.contains("solar")
            | psa_net.generators_t.p_set.columns.str.contains("wind"),
        ]
        q_set = psa_net.generators_t.q_set.loc[
            :,
            psa_net.generators_t.q_set.columns.str.contains("solar")
            | psa_net.generators_t.q_set.columns.str.contains("wind"),
        ]
    elif kind == "load":
        p_set = psa_net.loads_t.p_set.drop(
            columns=np.concatenate((flexible_hps, flexible_cps, flexible_loads))
        )
        q_set = psa_net.loads_t.q_set.drop(
            columns=np.concatenate((flexible_hps, flexible_cps, flexible_loads))
        )
    elif kind == "storage":
        p_set = psa_net.storage_units_t.p_set
        q_set = psa_net.storage_units_t.q_set
    elif kind == "electromobility":
        if len(flexible_cps) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = psa_net.loads_t.p_set.loc[:, flexible_cps]
            p_max = edisgo_obj.electromobility.flexibility_bands["upper_power"]
            e_min = edisgo_obj.electromobility.flexibility_bands["lower_energy"]
            e_max = edisgo_obj.electromobility.flexibility_bands["upper_energy"]
    elif kind == "heatpumps":
        if len(flexible_hps) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = psa_net.loads_t.p_set.loc[:, flexible_hps]
            cop = edisgo_obj.heat_pump.cop_df
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
        if kind == "gen_nd":
            comp_i = _mapping(psa_net, comp, kind)
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }
        elif kind == "load":
            comp_i = _mapping(
                psa_net, comp, kind, flexible_cps, flexible_hps, flexible_loads
            )
            pm_comp[str(comp_i)] = {
                "pd": p_set[comp].values.tolist(),
                "qd": q_set[comp].values.tolist(),
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
                    "p_max": p_max[comp].values.tolist(),
                    "e_min": e_min[comp].values.tolist(),
                    "e_max": e_max[comp].values.tolist(),
                }
        elif kind == "heatpumps":
            comp_i = _mapping(psa_net, comp, kind, flexible_hps=flexible_hps)
            pm_comp[str(comp_i)] = {
                "pd": p_set[comp].values.tolist(),
                "cop": cop[comp].values.tolist(),  # ändert der sich über die Zeit?
            }
        elif kind == "dsm":
            comp_i = _mapping(psa_net, comp, kind, flexible_loads=flexible_loads)
            pm_comp[str(comp_i)] = {
                "p_max": p_set[comp].values.tolist(),
                "p_min": p_min[comp].values.tolist(),
                "e_min": e_min[comp].values.tolist(),
                "e_max": e_max[comp].values.tolist(),
            }
    if kind == "HV_requirements":  # TODO: add correct time series from edisgo.etrago
        timesteps = len(psa_net.snapshots)
        pm_comp = {
            "P_curt": np.ones(timesteps).tolist(),
            "Q_curt": np.ones(timesteps).tolist(),
            "P_cp": np.ones(timesteps).tolist(),
            "P_hp": np.ones(timesteps).tolist(),
            "Q_hp": np.ones(timesteps).tolist(),
            "P_dsm": np.ones(timesteps).tolist(),
            "Q_dsm": np.ones(timesteps).tolist(),
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
        ]
    elif kind == "gen_nd":
        df = psa_net.generators.loc[
            (psa_net.generators.index.str.contains("solar"))
            | (psa_net.generators.index.str.contains("wind"))
        ]
    elif kind == "storage":
        df = psa_net.storage_units
    elif kind == "load":
        df = psa_net.loads.drop(
            np.concatenate((flexible_hps, flexible_cps, flexible_loads))
        )
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
        psa_trafos.groupby(by=[psa_trafos.bus0, psa_trafos.bus1])["r", "x", "s_nom"]
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


def _calculate_q(edisgo_obj, pm, idx_bus, kind):
    """
    Retrieves power factor and sign from edisgo config files to calculate reactive
    power.

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
    pf_sign: float
        Product of sign and tan(arccos(PF)).
    """

    grid_level = pm["bus"][str(idx_bus)]["grid_level"]
    pf = edisgo_obj.config._data["reactive_power_factor"][
        "{}_{}".format(grid_level, kind)
    ]
    mode = edisgo_obj.config._data["reactive_power_mode"][
        "{}_{}".format(grid_level, kind)
    ]
    if kind in ["gen", "storage"]:
        if mode == "inductive":
            pf_sign = -1 * np.tan(np.arccos(pf))
        else:
            pf_sign = np.tan(np.arccos(pf))
    elif kind in ["load", "cp", "hp"]:
        if mode == "inductive":
            pf_sign = np.tan(np.arccos(pf))
        else:
            pf_sign = -1 * np.tan(np.arccos(pf))
    return pf_sign
