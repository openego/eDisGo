import numpy as np
import pandas as pd
import math
from pypower.idx_brch import *
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_cost import *
import pypsa


def to_powermodels(pypsa_net):
    """
    Convert pypsa network to network dictionary format, using the pypower
    structure as an intermediate steps

    powermodels network dictionary:
    https://lanl-ansi.github.io/PowerModels.jl/stable/network-data/

    pypower caseformat:
    https://github.com/rwl/PYPOWER/blob/master/pypower/caseformat.py

    :param pypsa_net:
    :return:
    """

    # calculate per unit values
    pypsa.pf.calculate_dependent_values(pypsa_net)

    # convert pypsa network to pypower datastructure
    ppc, loads_t, gens_t = pypsa2ppc(pypsa_net)

    # conver pypower datastructure to powermodels network dictionary
    pm = ppc2pm(ppc, pypsa_net)

    return pm, loads_t, gens_t


def convert_storage_series(timeseries):
    if len(timeseries) == 0:
        return {}
    else:
        storage = {"time_horizon": len(timeseries), "storage_data": {}}
        for i, v in enumerate(timeseries.values):
            storage["storage_data"][i + 1] = {"p_req": v}
        return storage


# FIXME: Static storage data is exported from the eDisGo network rather than the PyPSA network as the capacity of network doesn't seem to be available there. For consistency with the rest of the conversion, it should be converted from PyPSA as well.
# TODO: This will (probably) not work if there are multiple storage units connected to the same bus.
def add_storage_from_edisgo(edisgo_obj, psa_net, pm_dict):
    """
    Read static storage data (position and capacity) from eDisGo and export to Powermodels dict
    """
    # Drop values that are not available
    storage = pd.DataFrame(
        edisgo_obj.topology.storage_units_df[["bus", "p_nom"]]
    )

    # Rename storage parameters to PowerModels naming convention
    storage.columns = ["storage_bus", "energy_rating"]

    # Fill in missing values so that PowerModels doesn't complain
    # TODO: Get (some of) these from eDisGo/PyPSA as well
    storage["energy"] = 0.0

    storage["charge_rating"] = 1.0
    storage["discharge_rating"] = 1.0
    storage["charge_efficiency"] = 0.9487
    storage["discharge_efficiency"] = 0.9487

    storage["ps"] = 0.0
    storage["qs"] = 0.0

    storage["qmin"] = 0.0
    storage["qmax"] = 0.0

    storage["r"] = 0.0
    storage["x"] = 0.0

    storage["p_loss"] = 0.0
    storage["q_loss"] = 0.0
    storage["standby_loss"] = 0.0

    storage["status"] = 1

    # Get Bus indices from PyPSA net
    storage["storage_bus"] = [
        psa_net.buses.index.get_loc(bus) + 1 for bus in storage["storage_bus"]
    ]
    storage.index = [i + 1 for i in range(len(storage))]

    # Add dedicated 'index' column because PowerModels likes it
    storage["index"] = storage.index.to_list()

    # Add to PowerModels statics Dict
    pm_dict["storage"] = storage.to_dict(orient="index")


def pypsa2ppc(psa_net):
    """Converter from pypsa data structure to pypower data structure

        adapted from pandapower's pd2ppc converter

        https://github.com/e2nIEE/pandapower/blob/911f300a96ee0ac062d82f7684083168ff052586/pandapower/pd2ppc.py

    """

    # build static pypower structure
    ppc = _init_ppc()
    ppc["name"] = psa_net.name

    _build_bus(psa_net, ppc)
    _build_gen(psa_net, ppc)

    _build_branch(psa_net, ppc)
    _build_transformers(psa_net, ppc)

    _build_load(psa_net, ppc)

    # TODO STORAGE UNITS
    _build_storage_units(psa_net, ppc)

    # built dictionaries if timeseries is used
    time_horizon = len(psa_net.loads_t["p_set"])
    try:
        load_dict = _build_load_dict(psa_net, ppc)
        print(
            "Dictionary for load timeseries of timehorizon {} created".format(
                time_horizon
            )
        )
    except IndexError as e:
        print(
            "No load timeseries. Create empty dicts " "for timeseries of load"
        )
        load_dict = dict()
    try:
        gen_dict = _build_generator_dict(psa_net, ppc)
        print(
            "Dictionary for generator timeseries of timehorizon {} created".format(
                time_horizon
            )
        )
    except IndexError as e:
        print(
            "no generator timeseries Create empty dicts "
            "for timeseries of load and generation "
        )
        gen_dict = dict()

    return ppc, load_dict, gen_dict


def ppc2pm(ppc, psa_net):  # pragma: no cover
    """
    converter from pypower datastructure to powermodels dictionary,

    adapted from pandapower to powermodels converter:
    https://github.com/e2nIEE/pandapower/blob/develop/pandapower/converter/powermodels/to_pm.py

    :param ppc:
    :return:
    """

    pm = {
        "gen": dict(),
        "branch": dict(),
        "bus": dict(),
        "dcline": dict(),
        "load": dict(),
        "storage": dict(),
        "baseMVA": ppc["baseMVA"],
        "source_version": "2.0.0",
        "shunt": dict(),
        "sourcetype": "matpower",
        "per_unit": True,
        "name": ppc["name"],
    }
    load_idx = 1
    shunt_idx = 1
    for row in ppc["bus"]:
        bus = dict()
        idx = int(row[BUS_I]) + 1
        bus["index"] = idx
        bus["bus_i"] = idx
        bus["zone"] = int(row[ZONE])
        bus["bus_type"] = int(row[BUS_TYPE])
        bus["vmax"] = row[VMAX]
        bus["vmin"] = row[VMIN]
        bus["va"] = row[VA]
        bus["vm"] = row[VM]
        bus["base_kv"] = row[BASE_KV]
        pd = row[PD]
        qd = row[QD]
        if pd != 0 or qd != 0:
            pm["load"][str(load_idx)] = {
                "pd": pd,
                "qd": qd,
                "load_bus": idx,
                "status": True,
                "index": load_idx,
            }
            load_idx += 1
        bs = row[BS]
        gs = row[GS]
        if pd != 0 or qd != 0:
            pm["shunt"][str(shunt_idx)] = {
                "gs": gs,
                "bs": bs,
                "shunt_bus": idx,
                "status": True,
                "index": shunt_idx,
            }
            shunt_idx += 1
        pm["bus"][str(idx)] = bus

    n_lines = len(ppc["branch"])
    for idx, row in enumerate(ppc["branch"], start=1):
        branch = dict()
        branch["index"] = idx
        branch["transformer"] = idx > n_lines
        branch["br_r"] = row[BR_R].real
        branch["br_x"] = row[BR_X].real
        branch["g_fr"] = -row[BR_B].imag / 2.0
        branch["g_to"] = -row[BR_B].imag / 2.0
        branch["b_fr"] = row[BR_B].real / 2.0
        branch["b_to"] = row[BR_B].real / 2.0
        branch["rate_a"] = (
            row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
        )
        branch["rate_b"] = row[RATE_B].real
        branch["rate_c"] = row[RATE_C].real
        branch["f_bus"] = int(row[F_BUS].real) + 1
        branch["t_bus"] = int(row[T_BUS].real) + 1
        branch["br_status"] = int(row[BR_STATUS].real)
        branch["angmin"] = row[ANGMIN].real
        branch["angmax"] = row[ANGMAX].real
        branch["tap"] = row[TAP].real
        branch["shift"] = math.radians(row[SHIFT].real)
        pm["branch"][str(idx)] = branch

    for idx, row in enumerate(ppc["gen"], start=1):
        gen = dict()
        gen["pg"] = row[PG]
        gen["qg"] = row[QG]
        gen["gen_bus"] = int(row[GEN_BUS]) + 1
        gen["vg"] = row[VG]
        gen["qmax"] = row[QMAX]
        gen["gen_status"] = int(row[GEN_STATUS])
        gen["qmin"] = row[QMIN]
        gen["pmin"] = row[PMIN]
        gen["pmax"] = row[PMAX]
        gen["index"] = idx
        pm["gen"][str(idx)] = gen

    # TODO add attribute "fluctuating" to generators from psa_net, maybe move to ppc first
    # is_fluctuating = [int("fluctuating" in index.lower()) for index in psa_net.generators.index]
    # for idx, row in enumerate(is_fluctuating, start=1):
    #     pm["gen"][str(idx)]["fluctuating"] = row

    for idx, row in enumerate(psa_net.generators["control"], start=1):
        pm["gen"][str(idx)]["gen_slack"] = (row == "Slack") * 1

    for idx, row in enumerate(psa_net.generators["fluctuating"], start=1):
        # convert boolean to 0 and 1, check if row is nan, e.g. slack bus
        pm["gen"][str(idx)]["fluctuating"] = (
            not (math.isnan(row)) and row
        ) * 1

    if len(ppc["gencost"]) > len(ppc["gen"]):
        ppc["gencost"] = ppc["gencost"][: ppc["gen"].shape[0], :]
    for idx, row in enumerate(ppc["gencost"], start=1):
        gen = pm["gen"][str(idx)]
        gen["model"] = int(row[MODEL])
        if gen["model"] == 1:
            gen["ncost"] = int(row[NCOST])
            gen["cost"] = row[COST : COST + gen["ncost"] * 2].tolist()
        elif gen["model"] == 2:
            gen["ncost"] = int(row[NCOST])
            gen["cost"] = [0] * 3
            costs = row[COST:]
            if len(costs) > 3:
                print(costs)
                raise ValueError("Maximum quadratic cost function allowed")
            gen["cost"][-len(costs) :] = costs

    if len(ppc["branchcost"]) > len(ppc["branch"]):
        ppc["branchcost"] = ppc["branchcost"][: ppc["branch"].shape[0], :]
    for idx, row in enumerate(ppc["branchcost"], start=1):
        ncost = int(row[0])
        branch = pm["branch"][str(idx)]
        branch["ncost"] = ncost
        branch["cost"] = [0] * ncost
        costs = row[1:]
        branch["cost"][-len(costs) :] = costs
    # TODO STORAGE UNITS!

    return pm


def _init_ppc():
    # init empty ppc
    ppc = {
        "name": "name",
        "baseMVA": 1,
        "version": 2,
        "bus": np.array([], dtype=float),
        "branch": np.array([], dtype=np.complex128),
        "gen": np.array([], dtype=float),
        "gencost": np.array([], dtype=float),
        "branchcost": np.array([], dtype=float),
        "internal": {
            "Ybus": np.array([], dtype=np.complex128),
            "Yf": np.array([], dtype=np.complex128),
            "Yt": np.array([], dtype=np.complex128),
            "branch_is": np.array([], dtype=bool),
            "gen_is": np.array([], dtype=bool),
            "DLF": np.array([], dtype=np.complex128),
            "buses_ord_bfs_nets": np.array([], dtype=float),
        },
    }
    return ppc


def _build_bus(psa_net, ppc):
    n_bus = len(psa_net.buses.index)
    print("build {} buses".format(n_bus))
    col_names = (
        "index",
        "type",
        "Pd",
        "Qd",
        "Gs",
        "Bs",
        "area",
        "v_mag_pu_set",
        "v_ang_set",
        "v_nom",
        "zone",
        "v_mag_pu_max",
        "v_mag_pu_min".split(", "),
    )
    bus_cols = len(col_names)
    ppc["bus"] = np.zeros(shape=(n_bus, bus_cols), dtype=float)
    ppc["bus"][:, :bus_cols] = np.array(
        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.05, 0.95]
    )
    ppc["bus"][:, BUS_I] = np.arange(n_bus)
    bus_types = ["PQ", "PV", "Slack", "None"]
    bus_types_int = np.array(
        [
            bus_types.index(b_type) + 1
            for b_type in psa_net.buses["control"].values
        ],
        dtype=int,
    )
    ppc["bus"][:, BUS_TYPE] = bus_types_int
    ppc["bus"][:, BASE_KV] = psa_net.buses["v_nom"].values
    # for edisgo scenario voltage bounds defined for load and feedin case with 0.985<= v <= 1.05
    # bounds have to be at least in that range, only accept stronger bounds if given
    ppc["bus"][:, VMAX] = [
        min(val, 1.05) for val in psa_net.buses["v_mag_pu_max"].values
    ]
    ppc["bus"][:, VMIN] = [
        max(val, 0.985) for val in psa_net.buses["v_mag_pu_min"].values
    ]
    return


def _build_gen(psa_net, ppc):
    n_gen = psa_net.generators.shape[0]
    gen_cols = 21
    # "bus, p_set, q_set, q_max, q_min, v_set_pu, mva_base, status, p_nom, p_min, Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = np.zeros(shape=(n_gen, gen_cols), dtype=float)
    # get bus indices for generators
    bus_indices = np.array(
        [
            psa_net.buses.index.get_loc(bus_name)
            for bus_name in psa_net.generators["bus"]
        ]
    )
    print(
        "build {} generators, distributed on {} buses".format(
            n_gen, len(np.unique(bus_indices))
        )
    )
    ppc["gen"][:, GEN_BUS] = bus_indices
    # adjust bus types
    bus_types = ["PQ", "PV", "Slack", "None"]
    gen_types = np.array(
        [
            bus_types.index(gen_type) + 1
            for gen_type in psa_net.generators["control"].values
        ],
        dtype=int,
    )
    # ppc["bus"][bus_indices,BUS_TYPE] = gen_types
    # set setpoint of pg and qg
    ppc["gen"][:, PG] = psa_net.generators["p_set"].values
    ppc["gen"][:, QG] = psa_net.generators["q_set"].values

    ppc["gen"][:, MBASE] = 1.0
    ppc["gen"][:, GEN_STATUS] = 1.0

    ppc["gen"][:, PMAX] = psa_net.generators["p_nom"].values
    ppc["gen"][:, PMIN] = 0
    # TODO SET QMAX AND QMIN! e.g.: cos(phi) value from config
    # ppc["gen"][:,QMAX] = 0
    # ppc["gen"][:, QMIN] = 0

    # build field for generator costs
    # 2	startup	shutdown	n	c(n-1)	...	c0
    # for quadratic cost function n=3--> c2,c1,c0 lead to 7 columns
    cost_cols = 7
    ppc["gencost"] = np.zeros(shape=(n_gen, cost_cols), dtype=float)
    # polynomial cost function
    ppc["gencost"][:, MODEL] = POLYNOMIAL
    ppc["gencost"][:, STARTUP] = psa_net.generators["start_up_cost"].values
    ppc["gencost"][:, SHUTDOWN] = psa_net.generators["shut_down_cost"].values
    # quadratic cost function has 3 cost coefficients
    ppc["gencost"][:, NCOST] = 3
    ppc["gencost"][:, COST] = 0.0
    ppc["gencost"][:, COST + 1] = psa_net.generators["marginal_cost"].values
    ppc["gencost"][:, COST + 2] = 0.0
    return


def _build_branch(psa_net, ppc):
    n_branch = len(psa_net.lines.index)
    print("build {} lines".format(n_branch))
    col_names = "fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax".split(
        ", "
    )
    branch_cols = len(col_names)
    ppc["branch"] = np.zeros(shape=(n_branch, branch_cols), dtype=float)
    from_bus = np.array(
        [
            psa_net.buses.index.get_loc(bus_name)
            for bus_name in psa_net.lines["bus0"]
        ]
    )
    to_bus = np.array(
        [
            psa_net.buses.index.get_loc(bus_name)
            for bus_name in psa_net.lines["bus1"]
        ]
    )
    ppc["branch"][:, F_BUS] = from_bus
    ppc["branch"][:, T_BUS] = to_bus

    ppc["branch"][:, BR_R] = psa_net.lines["r_pu"].values
    ppc["branch"][:, BR_X] = psa_net.lines["x_pu"].values
    ppc["branch"][:, BR_B] = psa_net.lines["b_pu"].values
    ppc["branch"][:, RATE_A] = psa_net.lines["s_nom"].values
    ppc["branch"][:, RATE_B] = 250  # Default values
    ppc["branch"][:, RATE_C] = 250  # Default values
    ppc["branch"][:, TAP] = 0.0
    ppc["branch"][:, SHIFT] = 0.0
    ppc["branch"][:, BR_STATUS] = 1.0
    ppc["branch"][:, ANGMIN] = -360
    ppc["branch"][:, ANGMAX] = 360
    # TODO BRANCHCOSTS!
    # check which branch costs are given in psa_net,
    ncost = sum(
        [
            (colName in psa_net.lines.columns) * 1
            for colName in ["costs_earthworks", "costs_cable"]
        ]
    )
    if ncost == 0:
        print("no branch costs are given in pypsa network")
    elif ncost == 1:
        if not "costs_cable" in psa_net.lines.columns:
            print(
                "costs for cables not in pypsa network, not possible to define cost function for network expansion"
            )
        else:
            ppc["branchcost"] = np.zeros(shape=(n_branch, 2), dtype=float)
            ppc["branchcost"][:, 0] = ncost
            ppc["branchcost"][:, 1] = psa_net.lines["costs_cable"].values
    elif ncost == 2:
        ppc["branchcost"] = np.zeros(shape=(n_branch, 3), dtype=float)
        ppc["branchcost"][:, 0] = ncost
        ppc["branchcost"][:, 1] = psa_net.lines["costs_cable"].values
        ppc["branchcost"][:, 2] = psa_net.lines["costs_earthworks"].values

    return


def _build_transformers(psa_net, ppc):
    n_transformers = len(psa_net.transformers.index)
    print("appending {} transformers".format(n_transformers))
    col_names = "fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax".split(
        ", "
    )

    transformers = np.zeros(
        shape=(n_transformers, len(col_names)), dtype=float
    )
    from_bus = np.array(
        [
            psa_net.buses.index.get_loc(bus_name)
            for bus_name in psa_net.transformers["bus0"]
        ]
    )
    to_bus = np.array(
        [
            psa_net.buses.index.get_loc(bus_name)
            for bus_name in psa_net.transformers["bus1"]
        ]
    )
    transformers[:, F_BUS] = from_bus
    transformers[:, T_BUS] = to_bus

    transformers[:, BR_R] = psa_net.transformers["r_pu"].values
    transformers[:, BR_X] = psa_net.transformers["x_pu"].values
    transformers[:, BR_B] = psa_net.transformers["b_pu"].values
    transformers[:, RATE_A] = psa_net.transformers["s_nom"].values
    transformers[:, RATE_B] = 250  # Default values
    transformers[:, RATE_C] = 250  # Default values
    transformers[:, TAP] = psa_net.transformers["tap_ratio"].values
    transformers[:, SHIFT] = psa_net.transformers["phase_shift"].values
    transformers[:, BR_STATUS] = 1.0
    transformers[:, ANGMIN] = -360
    transformers[:, ANGMAX] = 360

    ppc["branch"] = np.append(ppc["branch"], transformers, axis=0)
    # add trafo costs to branch cost with same shape
    if len(ppc["branchcost"]) > 0:
        print("append transformer costs")
        ncost = ppc["branchcost"].shape[1] - 1

        trafo_costs = np.zeros(shape=(n_transformers, ncost + 1), dtype=float)

        if hasattr(psa_net.transformers, "trafo_costs"):
            trafo_costs[:, 0] = ncost
            trafo_costs[:, 1] = psa_net.transformers["trafo_costs"].values
        print(trafo_costs)
        ppc["branchcost"] = np.append(ppc["branchcost"], trafo_costs, axis=0)
    return


def _build_load(psa_net, ppc):
    n_load = psa_net.loads.shape[0]
    load_buses = np.array(
        [
            psa_net.buses.index.get_loc(bus_name)
            for bus_name in psa_net.loads["bus"]
        ]
    )
    print(
        "build {} loads, distributed on {} buses".format(
            n_load, len(np.unique(load_buses))
        )
    )

    ## USE LOAD DATA FROM psa_net.loads as static network data
    # set bool if loads contains a timeseries
    # istime = len(psa_net.loads_t["p_set"].values[0]) != 0
    # istime = False
    # print("network has timeseries for load: {}".format(istime))

    for (load_idx, bus_idx) in enumerate(load_buses):
        # if istime:
        #     # if timeseries take maximal value of load_bus for static information of the network
        #     p_d = max(psa_net.loads_t["p_set"].values[:,load_idx])
        #     q_d = max(psa_net.loads_t["q_set"].values[:,load_idx])
        # else:
        p_d = psa_net.loads["p_set"].values[load_idx]
        q_d = psa_net.loads["q_set"].values[load_idx]
        # increase demand at bus_idx by p_d and q_d from load_idx, as multiple loads can be attached to single bus
        ppc["bus"][bus_idx, PD] += p_d
        ppc["bus"][bus_idx, QD] += q_d

    return


def _build_storage_units(psa_net, ppc):
    print("storage units are not implemented yet")


def _build_load_dict(psa_net, ppc):
    """
    build load dict containing timeseries from psa_net.loads_t
    :param psa_net: pypsa network
    :param ppc:
    :return: load_dict: Dict()
    """
    load_dict = {"load_data": dict()}
    load_buses = np.array(
        [
            psa_net.buses.index.get_loc(bus_name)
            for bus_name in psa_net.loads["bus"]
        ]
    )
    time_horizon = len(psa_net.loads_t["p_set"])

    load_dict["time_horizon"] = time_horizon
    for t in range(time_horizon):
        load_dict["load_data"][str(t + 1)] = dict()
        for (load_idx, bus_idx) in enumerate(load_buses):
            # p_d = psa_net.loads_t["p_set"].values[t,load_idx]
            # qd = psa_net.loads_t["q_set"].values[t,load_idx]
            p_d = psa_net.loads_t["p_set"][psa_net.loads.index[load_idx]][t]
            qd = psa_net.loads_t["q_set"][psa_net.loads.index[load_idx]][t]
            load_dict["load_data"][str(t + 1)][str(load_idx + 1)] = {
                "pd": p_d,
                "qd": qd,
                "load_bus": int(bus_idx + 1),
                "status": True,
                "index": int(load_idx + 1),
            }

    return load_dict


def _build_generator_dict(psa_net, ppc):
    generator_dict = {"gen_data": dict()}
    time_horizon = len(psa_net.generators_t["p_set"])
    generator_dict["time_horizon"] = time_horizon
    # buses_with_gens = [psa_net.generators.loc[busname]["bus"] for busname in psa_net.generators_t["p_set"].columns]
    # gen_buses = np.array([psa_net.buses.index.get_loc(bus_name) for bus_name in buses_with_gens])
    gen_buses = [
        psa_net.buses.index.get_loc(bus_name)
        for bus_name in psa_net.generators["bus"]
    ]
    for t in range(time_horizon):
        generator_dict["gen_data"][str(t + 1)] = dict()
        for (gen_idx, bus_idx) in enumerate(gen_buses):
            # pg = psa_net.generators_t["p_set"].values[t, gen_idx]
            # qg = psa_net.generators_t["q_set"].values[t, gen_idx]
            pg = psa_net.generators_t["p_set"][
                psa_net.generators.index[gen_idx]
            ][t]
            qg = psa_net.generators_t["q_set"][
                psa_net.generators.index[gen_idx]
            ][t]
            # if no value is set, set pg and qg to large value, e.g. representing slack
            # TODO verify or find another solution not using "large" value
            if np.isnan(pg):
                pg = 99999
            if np.isnan(qg):
                qg = 99999
            generator_dict["gen_data"][str(t + 1)][str(gen_idx + 1)] = {
                "pg": pg,
                "qg": qg,
                "gen_bus": int(bus_idx) + 1,
                "status": True,
                "index": int(gen_idx + 1),
            }

    return generator_dict
