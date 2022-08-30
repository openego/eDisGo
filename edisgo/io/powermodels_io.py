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

    # build static PowerModels structure
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

    # Build timeseries objects in PM format for function make_multinetwork
    load_dict = _build_load_dict(psa_net)
    gen_dict = _build_gen_dict(psa_net)
    # also for flexiblities

    return pm, load_dict, gen_dict


def _init_pm():
    # init empty powermodels dictionary
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
        idx_bus = _map_bus(psa_net, psa_net.generators.bus[gen_i])
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
    s_nom = psa_net.lines.s_nom

    for branch_i in np.arange(len(psa_net.lines.index)):
        idx_f_bus = _map_bus(psa_net, psa_net.lines.bus0[branch_i])
        idx_t_bus = _map_bus(psa_net, psa_net.lines.bus1[branch_i])
        pm["branch"][str(branch_i + 1)] = {
            "br_r": r[branch_i],
            "br_x": x[branch_i],
            "f_bus": idx_f_bus,
            "t_bus": idx_t_bus,
            "g_to": b[branch_i].imag / 2,  # TODO: Berechnung überprüfen
            "g_fr": -b[branch_i].imag / 2,
            "b_to": b[branch_i].real / 2,
            "b_fr": -b[branch_i].real / 2,
            "shift": 0.0,  # TODO
            "br_status": 1.0,  # TODO
            "rate_a": s_nom[branch_i].real,  # TODO: Berechnungsvorschrift Jaap?
            "rate_b": 250,  # TODO
            "rate_c": 250,  # TODO
            "angmin": -np.pi / 6,  # TODO: Deg oder Rad?
            "angmax": np.pi / 6,
            "transformer": 0,  # TODO: Was genau für Daten sollen hier rein?
            "tap": 0.0,  # TODO
            "index": branch_i + 1,
        }


def _build_switch(psa_net, pm):
    return


def _build_load(psa_net, pm):
    pd = psa_net.loads.p_set  # Das hier sind die vorgegebenen load Zeitreihen?
    qd = psa_net.loads.q_set
    for load_i in np.arange(len(psa_net.loads.index)):
        idx_bus = _map_bus(psa_net, psa_net.loads.bus[load_i])
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
        idx_bus = _map_bus(psa_net, psa_net.shunt_impedances.bus[shunt_i])
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
        idx_bus = _map_bus(psa_net, psa_net.storage_units.bus[stor_i])
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


def _build_load_dict(psa_net):  # TODO
    load_dict = dict()
    return load_dict


def _build_gen_dict(psa_net):  # TODO
    gen_dict = dict()
    return gen_dict


def _map_bus(psa_net, bus_name):
    bus_idx = psa_net.buses.reset_index()[psa_net.buses.index == bus_name].index[0] + 1
    return bus_idx
