"""
This module provides tools to convert eDisGo representation of the network
topology and timeseries to PowerModels network data format . Call :func:`to_powermodels`
to retrieve the PowerModels network container.
"""

# import math

import numpy as np

# import pandas as pd
import pypsa

from edisgo.io.pypsa_io import to_pypsa


def to_powermodels(edisgo_object):
    # convert eDisGo object to pypsa network structure
    psa_net = to_pypsa(edisgo_object)
    # calculate per unit values
    pypsa.pf.calculate_dependent_values(psa_net)

    # build PowerModels structure
    pm = _init_pm()
    timesteps = len(psa_net.snapshots)  # length of considered timesteps
    pm["name"] = "ding0_{}_t_{}".format(edisgo_object.topology.id, timesteps)
    pm["time_elapsed"] = 0  # TODO
    pm["baseMVA"] = 1  # TODO
    pm["source_version"] = 2  # TODO: kann man auf die Version von eDisGo zugreifen?
    _build_bus(psa_net, pm)
    _build_gen(psa_net, pm)
    _build_branch(psa_net, pm)
    _build_storage(psa_net, pm)
    _build_switch(psa_net, pm)  # betrachten wir switches?
    _build_shunt(psa_net, pm)
    _build_load(psa_net, pm)
    # Hier können jetzt noch die Dicts für die Flexibilitäten dem pm hinzugefügt werden
    _build_timeseries(psa_net, pm)
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
        "baseMVA": 1,
        "source_version": 2,
        "shunt": dict(),
        "sourcetype": "eDisGo",
        "per_unit": True,
        "name": "name",
        "time_series": dict(),
    }
    return pm


def _build_bus(psa_net, pm):
    bus_types = ["PQ", "PV", "Slack", "None"]
    bus_types_int = np.array(
        [bus_types.index(b_type) + 1 for b_type in psa_net.buses["control"].values],
        dtype=int,
    )
    v_max = [min(val, 1.05) for val in psa_net.buses["v_mag_pu_max"].values]
    v_min = [max(val, 0.985) for val in psa_net.buses["v_mag_pu_min"].values]
    for bus_i in np.arange(len(psa_net.buses.index)):
        pm["bus"][str(bus_i + 1)] = {
            "index": bus_i + 1,
            "bus_i": bus_i + 1,
            "zone": 1,  # TODO
            "bus_type": bus_types_int[bus_i],
            "vmax": v_max[bus_i],
            "vmin": v_min[bus_i],
            "va": 0,  # TODO
            "vm": 1,  # TODO
            "base_kv": psa_net.buses["v_nom"].values[bus_i],
        }


def _build_gen(psa_net, pm):
    pg = psa_net.generators.p_set  # Das hier sind die vorgegebenen Einspeisezeitreihen?
    qg = psa_net.generators.q_set
    p_max = psa_net.generators.p_max_pu
    p_min = psa_net.generators.p_min_pu
    p_nom = psa_net.generators.p_nom
    # vg =
    for gen_i in np.arange(len(psa_net.generators.index)):
        idx_bus = _mapping(psa_net, psa_net.generators.bus[gen_i])
        pm["gen"][str(gen_i + 1)] = {
            "pg": pg[gen_i],
            "qg": qg[gen_i],
            "pmax": p_max[gen_i],
            "pmin": p_min[gen_i],
            "qmax": 0,  # pmax[gen_i],#TODO *tan(phi)
            "qmin": 0,  # pmax[gen_i],#TODO *tan(phi)
            "vg": 0,  # TODO
            "mbase": p_nom[gen_i],  # s_nom?
            "gen_bus": idx_bus,
            "gen_status": 1,
            "index": gen_i + 1,
        }


def _build_branch(psa_net, pm):
    r = psa_net.lines.r_pu
    x = psa_net.lines.x_pu
    b = psa_net.lines.b_pu
    g = psa_net.lines.g_pu
    s_nom = psa_net.lines.s_nom

    for branch_i in np.arange(len(psa_net.lines.index)):
        idx_f_bus = _mapping(psa_net, psa_net.lines.bus0[branch_i])
        idx_t_bus = _mapping(psa_net, psa_net.lines.bus1[branch_i])
        pm["branch"][str(branch_i + 1)] = {
            "br_r": r[branch_i],
            "br_x": x[branch_i],
            "f_bus": idx_f_bus,
            "t_bus": idx_t_bus,
            "g_to": g[branch_i] / 2,  # TODO: Malte fragen
            "g_fr": g[branch_i] / 2,
            "b_to": b[branch_i] / 2,  # Beide positiv?
            # https://github.com/lanl-ansi/PowerModels.jl/blob/de7da4d11d04ce48b34d7b5f601f32f49361626b/src/io/matpower.jl#L459
            "b_fr": b[branch_i] / 2,
            "shift": 0.0,  # Default 0.0 if no transformer is attached
            "br_status": 1.0,  # TODO
            "rate_a": s_nom[branch_i].real,  # TODO: Berechnungsvorschrift Jaap?
            "rate_b": 250,  # TODO
            "rate_c": 250,  # TODO
            "angmin": -np.pi / 6,  # TODO: Deg oder Rad?
            "angmax": np.pi / 6,
            "transformer": False,  # TODO: add transformer: tap + shift
            "tap": 1.0,  # Default 1.0 if no transformer is attached
            "index": branch_i + 1,
        }


def _build_switch(psa_net, pm):
    return


def _build_load(psa_net, pm):
    pd = psa_net.loads.p_set  # Das hier sind die vorgegebenen load Zeitreihen?
    qd = psa_net.loads.q_set
    for load_i in np.arange(len(psa_net.loads.index)):
        idx_bus = _mapping(psa_net, psa_net.loads.bus[load_i])
        pm["load"][str(load_i + 1)] = {
            "pd": pd[load_i],
            "qd": qd[load_i],
            "load_bus": idx_bus,
            "status": True,
            "index": load_i + 1,
        }


def _build_shunt(psa_net, pm):
    bs = psa_net.shunt_impedances.b
    gs = psa_net.shunt_impedances.g
    for shunt_i in np.arange(len(psa_net.shunt_impedances.index)):
        idx_bus = _mapping(psa_net, psa_net.shunt_impedances.bus[shunt_i])
        pm["shunt"][str(shunt_i + 1)] = {
            "gs": gs[shunt_i],
            "bs": bs[shunt_i],
            "shunt_bus": idx_bus,
            "status": True,
            "index": shunt_i + 1,
        }


def _build_storage(psa_net, pm):
    ps = psa_net.storage_units.p_set
    qs = psa_net.storage_units.q_set
    soc = psa_net.storage_units.state_of_charge_set
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
            "qmax": 0,  # pmax[stor_i],#TODO *tan(phi)
            "qmin": 0,  # pmax[stor_i],#TODO *tan(phi)
            "q_loss": 0,  # TODO
            "energy": soc[stor_i],  # TODO: Ist das richtig?
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


# TODO: _build_flexibility(psa_net, pm)-function for every flexibility


def _build_timeseries(psa_net, pm):  # TODO add flexibilities
    pm["time_series"] = {
        "gen": _build_component_timeseries(psa_net, "gen"),
        "load": _build_component_timeseries(psa_net, "load"),
        "storage": _build_component_timeseries(psa_net, "storage"),
        "num_steps": len(psa_net.snapshots),
    }
    return


def _build_component_timeseries(psa_net, kind):  # TODO add flexibilities
    if kind == "gen":
        pm_comp = dict()
        p_set = psa_net.generators_t.p_set
        q_set = psa_net.generators_t.q_set
    elif kind == "load":
        pm_comp = dict()
        p_set = psa_net.loads_t.p_set
        q_set = psa_net.loads_t.q_set
    elif kind == "storage":
        pm_comp = dict()
        p_set = psa_net.storage_units_t.p_set
        q_set = psa_net.storage_units_t.q_set
    #   else: #TODO NotImplementedError/Warning
    for comp in p_set.columns:
        comp_i = _mapping(psa_net, comp, kind)
        if kind == "gen":
            pm_comp[str(comp_i)] = {
                "pg": p_set[comp].values.tolist(),
                "qg": q_set[comp].values.tolist(),
            }
        elif kind == "load":
            pm_comp[str(comp_i)] = {
                "pd": p_set[comp].values.tolist(),
                "qd": q_set[comp].values.tolist(),
            }
        elif kind == "storage":
            pm_comp[str(comp_i)] = {
                "ps": p_set[comp].values.tolist(),
                "qs": q_set[comp].values.tolist(),
            }
        # TODO: more time dependable values?
    return pm_comp


def _mapping(psa_net, name, kind="bus"):
    if kind == "bus":
        df = psa_net.buses
    elif kind == "gen":
        df = psa_net.generators
    elif kind == "storage":
        df = psa_net.storage_units
    elif kind == "load":
        df = psa_net.loads
    #    else: #TODO
    #        logging.warning("No mapping for "{}" implemented yet.".format(kind))
    idx = df.reset_index()[df.index == name].index[0] + 1
    return idx
