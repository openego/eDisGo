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


def to_powermodels(edisgo_object, flexible_cps, flexible_hps):
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

    Returns
    -------
    dict
        Dictionary that contains all network data in PowerModels network data
        format.
    """

    # convert eDisGo object to pypsa network structure
    psa_net = edisgo_object.to_pypsa(use_seed=True)
    # aggregate parallel transformers
    psa_net.transformers = aggregate_parallel_transformers(psa_net.transformers)
    # calculate per unit values
    pypsa.pf.calculate_dependent_values(psa_net)
    # build PowerModels structure
    pm = _init_pm()
    timesteps = len(psa_net.snapshots)  # length of considered timesteps
    pm["name"] = "ding0_{}_t_{}".format(edisgo_object.topology.id, timesteps)
    pm["time_elapsed"] = int(
        (psa_net.snapshots[1] - psa_net.snapshots[0]).seconds / 3600
    )  # length of timesteps in hours
    pm["baseMVA"] = 1  # TODO
    pm["source_version"] = 2  # TODO
    _build_bus(psa_net, pm)
    _build_gen(psa_net, pm)
    _build_branch(psa_net, pm)
    _build_battery_storage(psa_net, pm)
    _build_load(psa_net, pm, flexible_cps, flexible_hps)
    _build_electromobility(psa_net, pm, flexible_cps)
    _build_heatpump(psa_net, pm, edisgo_object, flexible_hps)
    # _build_heat_storage(psa_net, pm, edisgo_object)
    # _build_dsm(psa_net, pm)
    _build_timeseries(psa_net, pm, edisgo_object, flexible_cps, flexible_hps)
    return pm


def _init_pm():
    # init empty PowerModels dictionary
    pm = {
        "gen": dict(),
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
    v_max = [min(val, 1.05) for val in psa_net.buses["v_mag_pu_max"].values]
    v_min = [max(val, 0.985) for val in psa_net.buses["v_mag_pu_min"].values]
    v_set = psa_net.buses["v_mag_pu_set"].values
    for bus_i in np.arange(len(psa_net.buses.index)):
        pm["bus"][str(bus_i + 1)] = {
            "index": bus_i + 1,
            "bus_i": bus_i + 1,
            "zone": 1,
            "bus_type": bus_types_int[bus_i],
            "vmax": v_max[bus_i],
            "vmin": v_min[bus_i],
            "va": 0,
            "vm": v_set[bus_i],
            "base_kv": psa_net.buses["v_nom"].values[bus_i],
        }


def _build_gen(psa_net, pm):
    """
    Builds generator dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    """
    pg = psa_net.generators.p_set
    qg = psa_net.generators.q_set
    p_max = psa_net.generators.p_max_pu
    p_min = psa_net.generators.p_min_pu
    p_nom = psa_net.generators.p_nom
    # determine slack bus through slack generator
    slack_gen = psa_net.generators.bus[
        psa_net.generators.index == "Generator_slack"
    ].values[0]
    pm["bus"][str(_mapping(psa_net, slack_gen))]["bus_type"] = 3
    for gen_i in np.arange(len(psa_net.generators.index)):
        idx_bus = _mapping(psa_net, psa_net.generators.bus[gen_i])
        pm["gen"][str(gen_i + 1)] = {
            "pg": pg[gen_i],
            "qg": qg[gen_i],  # TODO: über PF
            "pmax": p_max[gen_i],
            "pmin": p_min[gen_i],
            "qmax": 1,  # TODO: aus Zeitreihe
            "qmin": 0,
            "vg": 1,
            "mbase": p_nom[gen_i],
            "gen_bus": idx_bus,
            "gen_status": 1,
            "index": gen_i + 1,
            "model": 2,  # wird eigentlich nicht benötigt
            "ncost": 3,  # wird eigentlich nicht benötigt
            "cost": [120, 20, 0],  # wird eigentlich nicht benötigt
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
    name = branches.index
    r = branches.r_pu
    x = branches.x_pu
    b = branches.b_pu
    g = branches.g_pu
    s_nom = branches.s_nom
    transformer = ~branches.tap_ratio.isna()
    tap = branches.tap_ratio.fillna(1)
    shift = branches.phase_shift.fillna(0)

    for branch_i in np.arange(len(branches.index)):
        idx_f_bus = _mapping(psa_net, branches.bus0[branch_i])
        idx_t_bus = _mapping(psa_net, branches.bus1[branch_i])
        pm["branch"][str(branch_i + 1)] = {
            "name": name[branch_i],
            "br_r": r[branch_i],
            "br_x": x[branch_i],
            "f_bus": idx_f_bus,
            "t_bus": idx_t_bus,
            "g_to": g[branch_i] / 2,
            "g_fr": g[branch_i] / 2,
            "b_to": b[branch_i] / 2,  # Beide positiv?
            # https://github.com/lanl-ansi/PowerModels.jl/blob/de7da4d11d04ce48b34d7b5f601f32f49361626b/src/io/matpower.jl#L459
            "b_fr": b[branch_i] / 2,
            "shift": shift[branch_i],
            "br_status": 1.0,
            "rate_a": s_nom[branch_i].real,
            "rate_b": 250,
            "rate_c": 250,
            "angmin": -np.pi / 6,
            "angmax": np.pi / 6,
            "transformer": bool(transformer[branch_i]),
            "tap": tap[branch_i],
            "index": branch_i + 1,
        }


def _build_load(psa_net, pm, flexible_cps, flexible_hps):
    """
    Builds load dictionary in PowerModels network data format and adds it to
    PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    """
    loads_df = psa_net.loads.drop(np.concatenate((flexible_hps, flexible_cps)))
    pd = loads_df.p_set
    qd = loads_df.q_set
    for load_i in np.arange(len(loads_df.index)):
        idx_bus = _mapping(psa_net, loads_df.bus[load_i])
        pm["load"][str(load_i + 1)] = {
            "pd": pd[load_i],
            "qd": qd[load_i],
            "load_bus": idx_bus,
            "status": True,
            "index": load_i + 1,
        }


def _build_battery_storage(psa_net, pm):
    """
    Builds (battery) storage  dictionary in PowerModels network data format and adds
    it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    """
    ps = psa_net.storage_units.p_set
    qs = psa_net.storage_units.q_set
    soc = psa_net.storage_units.state_of_charge_initial
    p_max = psa_net.storage_units.p_max_pu
    p_min = psa_net.storage_units.p_min_pu
    for stor_i in np.arange(len(psa_net.storage_units.index)):
        idx_bus = _mapping(psa_net, psa_net.storage_units.bus[stor_i])
        pm["storage"][str(stor_i + 1)] = {
            "x": 0,  # TODO
            "r": 0,  # TODO
            "ps": ps[stor_i],
            "qs": qs[stor_i],
            "pmax": p_max[stor_i],
            "pmin": p_min[stor_i],
            "p_loss": 0,  # TODO
            "qmax": 1,  # TODO: über PF?
            "qmin": 0,  # TODO: über PF?
            "q_loss": 0,  # TODO
            "energy": soc[stor_i],  # TODO: initial energy?
            "energy_rating": 0,  # TODO
            "thermal_rating": 0,  # TODO
            "charge_rating": 0,  # TODO
            "discharge_rating": 0,  # TODO
            "charge_efficiency": 1,  # TODO
            "discharge_efficiency": 1,  # TODO
            "storage_bus": idx_bus,
            "status": True,
            "index": stor_i + 1,
        }


def _build_electromobility(psa_net, pm, flexible_cps):
    """
    Builds electromobility dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
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
        pd = emob_df.p_set
        qd = emob_df.q_set
        for cp_i in np.arange(len(emob_df.index)):
            idx_bus = _mapping(psa_net, emob_df.bus[cp_i])
            pm["electromobility"][str(cp_i + 1)] = {
                "pd": pd[cp_i],
                "qd": qd[cp_i],
                "p_max": 1,
                "e_min": 0,
                "e_max": 1,
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
        pd = heat_df.p_set  # heat demand
        qd = heat_df.q_set
        cop = edisgo_obj.heat_pump.cop_df  # Wird das auch nach pypsa übersetzt?
        # p_max = (
        #     edisgo_obj.heat_pump.p_max
        # )  # TODO: liegt noch nicht auf dem eDisGo object
        for hp_i in np.arange(len(heat_df.index)):
            idx_bus = _mapping(psa_net, heat_df.bus[hp_i])
            pm["heatpumps"][str(hp_i + 1)] = {
                "pd": pd[hp_i],
                "qd": qd[hp_i],
                # "p_max": p_max[heat_df.index[hp_i]][0],
                "cop": cop[heat_df.index[hp_i]][0],
                "hp_bus": idx_bus,
                "index": hp_i + 1,
            }


def _build_heat_storage(psa_net, pm, edisgo_obj):  # TODO
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
    # ps = heat_storage_df.p_set # TODO
    soc = heat_storage_df.state_of_charge_initial
    efficiency = heat_storage_df.efficiency
    capacity = heat_storage_df.capacity
    # p_max = heat_storage_df.p_max_pu  # TODO
    for stor_i in np.arange(len(heat_storage_df.index)):
        idx_bus = _mapping(
            psa_net, heat_storage_df.bus[stor_i]
        )  # TODO 'bus' column doesn't exist yet
        pm["heat_storage"][str(stor_i + 1)] = {
            # "ps": ps[stor_i],
            # "pmax": p_max[stor_i],
            # "p_loss": 0,
            "energy": soc[stor_i],
            "capacity": capacity[stor_i],
            "charge_efficiency": efficiency[stor_i],
            "discharge_efficiency": efficiency[stor_i],
            "storage_bus": idx_bus,
            "status": True,
            "index": stor_i + 1,
        }


def _build_dsm(psa_net, pm, edisgo_obj):  # TODO
    """
    Builds dsm 'storage' dictionary and adds it to PowerModels dictionary 'pm'.

    Parameters
    ----------
    psa_net : :pypsa:`PyPSA.Network<network>`
        :pypsa:`PyPSA.Network<network>` representation of network.
    pm : dict
        (PowerModels) dictionary.
    edisgo_obj : :class:`~.EDisGo`

    """
    dsm_df = psa_net.loads.loc[psa_net.loads.index.str.startswith("dsm_load")]
    pd = dsm_df.p_set
    qd = dsm_df.q_set
    for dsm_i in np.arange(len(dsm_df.index)):
        idx_bus = _mapping(psa_net, dsm_df.bus[dsm_i])
        pm["electromobility"][str(dsm_i + 1)] = {
            "pd": pd[dsm_i],
            "qd": qd[dsm_i],
            "p_min": 0,
            "p_max": 1,
            "e_min": 0,
            "e_max": 1,
            "dsm_bus": idx_bus,
            "index": dsm_i + 1,
        }


def _build_timeseries(psa_net, pm, edisgo_obj, flexible_cps, flexible_hps):
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

    """
    for kind in ["gen", "load", "storage", "electromobility"]:  # , "heatpumps", "dsm"]
        _build_component_timeseries(
            psa_net, pm, kind, edisgo_obj, flexible_cps, flexible_hps
        )
    pm["time_series"]["num_steps"] = len(psa_net.snapshots)


def _build_component_timeseries(
    psa_net, pm, kind, edisgo_obj=None, flexible_cps=None, flexible_hps=None
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
        Must be one of ["gen", "load", "storage", "electromobility", "heatpumps",
        "dsm"]
    edisgo_obj : :class:`~.EDisGo`
    flexible_cps : :numpy:`numpy.ndarray<ndarray>`
        Array containing all charging points that allow for flexible charging.
    flexible_hps: :numpy:`numpy.ndarray<ndarray>`
        Array containing all heat pumps that allow for flexible operation due to an
        attached heat storage.

    """
    pm_comp = dict()
    if kind == "gen":
        p_set = psa_net.generators_t.p_set
        q_set = psa_net.generators_t.q_set
    elif kind == "load":
        p_set = psa_net.loads_t.p_set.drop(
            columns=np.concatenate((flexible_hps, flexible_cps))
        )
        q_set = psa_net.loads_t.q_set.drop(
            columns=np.concatenate((flexible_hps, flexible_cps))
        )
    elif kind == "storage":
        p_set = psa_net.storage_units_t.p_set
        q_set = psa_net.storage_units_t.q_set
    elif kind == "electromobility":
        if len(flexible_cps) == 0:
            p_set = pd.DataFrame()
        else:
            p_set = psa_net.loads_t.p_set.loc[:, flexible_cps]
            flex_bands = edisgo_obj.electromobility.get_flexibility_bands(
                edisgo_obj, ["home", "work"]
            )
            p_max = flex_bands["upper_power"]
            e_min = flex_bands["lower_energy"]
            e_max = flex_bands["upper_energy"]
            # TODO: Flexbänder aus eDisGo object auslesen
            # p_max = pd.read_csv(
            #     "/home/local/RL-INSTITUT/maike.held/Documents/PythonProjects/eDisGo_orig/eDisGo/examples/data_opf/2534/flex_bands_upper_power.csv"
            # )
            # e_min = pd.read_csv(
            #     "/home/local/RL-INSTITUT/maike.held/Documents/PythonProjects/eDisGo_orig/eDisGo/examples/data_opf/2534/flex_bands_lower_energy.csv"
            # )
            # e_max = pd.read_csv(
            #     "/home/local/RL-INSTITUT/maike.held/Documents/PythonProjects/eDisGo_orig/eDisGo/examples/data_opf/2534/flex_bands_upper_energy.csv"
            # )
    elif kind == "heatpumps":  # TODO
        if len(flexible_hps) == 0:
            p_set = pd.DataFrame()
        else:  # TODO add heat storages (== flexible hps)
            p_set = psa_net.loads_t.p_set.loc[:, flexible_hps]
            # cop
            # wärmespeicherzeitreihe?
    elif kind == "dsm":  # TODO
        print("To Do")

    for comp in p_set.columns:
        if kind == "gen":
            comp_i = _mapping(psa_net, comp, kind)
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }
        elif kind == "load":
            comp_i = _mapping(psa_net, comp, kind, flexible_cps, flexible_hps)
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
        elif kind == "heatpumps":  # TODO
            comp_i = _mapping(psa_net, comp, kind, flexible_hps=flexible_hps)
            print("To Do")
            # p_d (heat demand)
            # cop
            # p wärmespeicher
        elif kind == "dsm":  # TODO
            print("To Do")
            # e_min, e_max, p_min, p_max
        # TODO: more time dependable values?
    pm["time_series"][kind] = pm_comp


def _mapping(psa_net, name, kind="bus", flexible_cps=None, flexible_hps=None):
    # TODO: add heat storages and dsm
    if kind == "bus":
        df = psa_net.buses
    elif kind == "gen":
        df = psa_net.generators
    elif kind == "storage":
        df = psa_net.storage_units
    elif kind == "load":
        df = psa_net.loads.drop(np.concatenate((flexible_hps, flexible_cps)))
    elif kind == "electromobility":
        df = psa_net.loads.loc[flexible_cps]
    elif kind == "hetapumps":
        df = psa_net.loads.loc[flexible_hps]
    else:
        logging.warning("Mapping for '{}' not implemented.".format(kind))
    idx = df.reset_index()[df.index == name].index[0] + 1
    return idx


def aggregate_parallel_transformers(psa_trafos):
    # TODO: what about b, g?
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
    return psa_trafos
