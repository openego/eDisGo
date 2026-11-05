# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Import of `SimBench <https://simbench.de>`_ grids.

SimBench publishes its benchmark grids as a folder of semicolon separated csv
files per grid, e.g. ``1-MVLV-rural-all-2-sw``. This module reads such a folder
into an :class:`~.EDisGo` object. It covers grids that contain an MV grid
(SimBench codes ``MV`` and ``MVLV``); LV grids then have their own LV grid ID.

The csv folder of a single grid can be downloaded from simbench.de, or written
with the `simbench <https://github.com/e2nIEE/simbench>`_ python package from its
complete data set.

"""

import logging
import os

import networkx as nx
import numpy as np
import pandas as pd

from edisgo.network.grids import MVGrid
from edisgo.tools.tools import shift_to_reference_year

if "READTHEDOCS" not in os.environ:
    from shapely.geometry import MultiPoint

logger = logging.getLogger(__name__)

# SimBench voltage levels: 1 EHV, 2 EHV/HV, 3 HV, 4 HV/MV, 5 MV, 6 MV/LV, 7 LV
MV_LEVEL = 5
LV_LEVEL = 7

GENERATOR_TYPES = {
    "pv": "solar",
    "wind": "wind",
    "biomass": "biomass",
    "hydro": "run_of_river",
}

LOAD_SECTORS = {"H": "residential", "G": "cts", "L": "agricultural"}


def _read(path, table):
    """Read one SimBench csv table; missing tables are returned empty."""
    file = os.path.join(path, f"{table}.csv")
    if not os.path.isfile(file):
        return pd.DataFrame()
    return pd.read_csv(file, sep=";", low_memory=False)


def _subnet_id(subnet):
    """Integer grid ID from a SimBench subnet name, e.g. 'LV1.108' -> 1108."""
    level, number = subnet.split(".")
    return int(level.lstrip("MLV")) * 1000 + int(number)


def _fuse_closed_switches(nodes, switches):
    """
    Map every node to the node it is fused with via closed switches.

    Of the nodes connected by closed switches, the first node that is not an
    auxiliary node is kept, so that the remaining buses carry the names of the
    SimBench busbars and nodes.

    """
    closed = switches[switches.cond == 1]
    graph = nx.Graph()
    graph.add_nodes_from(nodes.index)
    graph.add_edges_from(zip(closed.nodeA, closed.nodeB))
    mapping = {}
    for component in nx.connected_components(graph):
        component = sorted(component)
        non_aux = [n for n in component if nodes.at[n, "type"] != "auxiliary"]
        keep = non_aux[0] if non_aux else component[0]
        mapping.update({n: keep for n in component})
    return mapping


def _close_switches_to_islands(nodes, switches, lines, trafos):
    """
    Close open switches that are the only connection of a grid part.

    SimBench may operate the busbars of an HV/MV substation separately, each fed
    by its own HV/MV transformer. eDisGo models one MV station, so such grid parts
    would be cut off from it. Open switches between buses that are not connected
    otherwise are therefore closed. Ring splitting points stay open.

    """
    switches = switches.copy()
    bus = _fuse_closed_switches(nodes, switches)
    mvlv = trafos[trafos.voltLvl == MV_LEVEL + 1]
    graph = nx.Graph()
    graph.add_nodes_from(set(bus.values()))
    graph.add_edges_from(zip(lines.nodeA.map(bus), lines.nodeB.map(bus)))
    graph.add_edges_from(zip(mvlv.nodeHV.map(bus), mvlv.nodeLV.map(bus)))
    for sw in switches[switches.cond == 0].itertuples():
        a, b = bus[sw.nodeA], bus[sw.nodeB]
        if not nx.has_path(graph, a, b):
            switches.at[sw.Index, "cond"] = 1
            graph.add_edge(a, b)
    return switches


def _load_type_and_sector(profile):
    """eDisGo load type and sector from the name of a SimBench load profile."""
    if profile.startswith(("Air_", "Soil_")):
        return "heat_pump", "individual_heating"
    if profile.startswith("HLS_"):
        return "charging_point", "home"
    return "conventional_load", LOAD_SECTORS.get(profile[0], "other")


def _generator_type(simbench_type):
    """eDisGo generator type from the SimBench RES type, e.g. 'Wind_MV'."""
    for key, value in GENERATOR_TYPES.items():
        if simbench_type.lower().startswith(key):
            return value
    return "others"


def _transformer_impedance(trafos, trafo_types):
    """Add s_nom, r_pu and x_pu (on the transformer's own rating) to `trafos`."""
    types = trafo_types.set_index("id").loc[trafos.type]
    s_nom = types.sR.values
    r_pu = types.pCu.values / (s_nom * 1e3)
    z_pu = types.vmImp.values / 100
    return trafos.assign(
        s_nom=s_nom, r_pu=r_pu, x_pu=np.sqrt(z_pu**2 - r_pu**2), type_info=trafos.type
    )


def import_simbench_grid(path, edisgo_obj):
    """
    Import the network topology of a SimBench grid.

    The grid must contain an MV grid, i.e. its SimBench code must be of the form
    ``1-MV-<type>--<scenario>-sw`` or ``1-MVLV-<type>-<lv grid>-<scenario>-sw``.
    It is mapped to eDisGo as follows:

    * Buses connected via closed switches are fused into one bus. Open switches,
      where SimBench splits MV rings, become eDisGo switch disconnectors. Open
      switches that are the only connection of a grid part to the MV station,
      e.g. between separately operated busbars, are closed, as eDisGo models one
      MV station.
    * The HV side of the HV/MV substation is left out; its transformers become
      :attr:`~.network.topology.Topology.transformers_hvmv_df`.
    * Each SimBench LV grid gets an LV grid ID built from its subnet name,
      e.g. 'LV1.108' becomes 1108. The MV grid ID is built the same way.
    * Heat pump loads (profiles 'Air_*' and 'Soil_*') get type 'heat_pump',
      home charging loads (profiles 'HLS_*') get type 'charging_point'. All other
      loads, including LV grids that SimBench represents as one load at an MV
      bus, are conventional loads.
    * Storage units, e.g. home batteries of PV systems, become storage units.

    Conventional power plants, shunts, three-winding transformers and
    measurements of SimBench are not imported.

    Parameters
    ----------
    path : str
        Path to the folder with the csv files of one SimBench grid.
    edisgo_obj : :class:`~.EDisGo`

    """
    nodes = _read(path, "Node").set_index("id")
    coordinates = _read(path, "Coordinates").set_index("id")
    switches = _read(path, "Switch")
    trafos = _read(path, "Transformer")
    trafo_types = _read(path, "TransformerType")

    for table in ["PowerPlant", "Shunt", "Transformer3W"]:
        if not _read(path, table).empty:
            logger.warning(f"SimBench table {table} is not imported to eDisGo.")

    hvmv_trafos = trafos[trafos.voltLvl == MV_LEVEL - 1]
    if hvmv_trafos.empty:
        raise ValueError(
            "The SimBench grid has no HV/MV transformer. Only SimBench grids "
            "containing an MV grid can be imported."
        )

    lines = _read(path, "Line")
    switches = _close_switches_to_islands(nodes, switches, lines, trafos)
    bus = _fuse_closed_switches(nodes, switches)

    # buses: all fused MV and LV nodes
    buses = nodes.loc[sorted(set(bus.values()))]
    buses = buses[buses.voltLvl >= MV_LEVEL]
    mv_subnet = buses[buses.voltLvl == MV_LEVEL].subnet.str.split("_").str[0]
    mv_grid_id = _subnet_id(mv_subnet.mode()[0])
    lv_subnet = buses.subnet.str.split("_").str[0]
    buses_df = pd.DataFrame(
        {
            "v_nom": buses.vmR,
            "x": coordinates.loc[buses.coordID, "x"].values,
            "y": coordinates.loc[buses.coordID, "y"].values,
            "mv_grid_id": mv_grid_id,
            "lv_grid_id": lv_subnet.map(_subnet_id).where(buses.voltLvl == LV_LEVEL),
            "in_building": False,
        },
        index=buses.index.rename("name"),
    )

    # lines; SimBench gives r and x in Ohm/km, b in uS/km and iMax in A
    line_types = _read(path, "LineType").set_index("id").loc[lines.type]
    length = lines.length.values
    bus0 = lines.nodeA.map(bus).values
    lines_df = pd.DataFrame(
        {
            "bus0": bus0,
            "bus1": lines.nodeB.map(bus).values,
            "length": length,
            "x": line_types.x.values * length,
            "r": line_types.r.values * length,
            "b": line_types.b.values * 1e-6 * length,
            "s_nom": np.sqrt(3)
            * buses_df.loc[bus0, "v_nom"].values
            * line_types.iMax.values
            / 1e3,
            "num_parallel": 1,
            "type_info": lines.type.values,
            "kind": np.where(line_types.type.values == "ohl", "line", "cable"),
        },
        index=pd.Index(lines.id.values, name="name"),
    )

    # transformers; bus0 is the upper voltage side
    trafo_columns = ["bus0", "bus1", "x_pu", "r_pu", "s_nom", "type_info"]
    trafos = _transformer_impedance(trafos, trafo_types).assign(
        bus0=trafos.nodeHV.map(bus), bus1=trafos.nodeLV.map(bus)
    )
    trafos = trafos.set_index(pd.Index(trafos.id.values, name="name"))
    transformers_hvmv_df = trafos.loc[hvmv_trafos.id.values, trafo_columns]
    transformers_df = trafos.loc[trafos.voltLvl == MV_LEVEL + 1, trafo_columns]

    # open switches split rings; the branch is the line at the auxiliary node
    open_switches = switches[switches.cond == 0]
    switch_rows = {}
    for sw in open_switches.itertuples():
        a, b = bus[sw.nodeA], bus[sw.nodeB]
        for bus_open, bus_closed in [(a, b), (b, a)]:
            branch = lines_df.index[
                (lines_df.bus0 == bus_open) | (lines_df.bus1 == bus_open)
            ]
            if nodes.at[bus_open, "type"] == "auxiliary" and len(branch) == 1:
                switch_rows[sw.id] = [bus_open, bus_closed, branch[0]]
                break
        else:
            logger.info(
                f"Open SimBench switch {sw.id} is not imported, as it is not "
                f"located at the end of a line. Its buses stay disconnected."
            )
    switches_df = pd.DataFrame.from_dict(
        switch_rows,
        orient="index",
        columns=["bus_open", "bus_closed", "branch"],
    ).rename_axis("name")
    switches_df["type_info"] = "Switch Disconnector"

    # loads
    loads = _read(path, "Load")
    type_sector = loads.profile.map(_load_type_and_sector)
    loads_df = pd.DataFrame(
        {
            "bus": loads.node.map(bus).values,
            "p_set": loads.pLoad.values,
            "building_id": np.nan,
            "type": type_sector.str[0].values,
            "annual_consumption": np.nan,
            "sector": type_sector.str[1].values,
            "number_households": np.nan,
        },
        index=pd.Index(loads.id.values, name="name"),
    )

    # generators
    res = _read(path, "RES")
    generators_df = pd.DataFrame(
        {
            "bus": res.node.map(bus).values,
            "p_nom": res.pRES.values,
            "type": res.type.map(_generator_type).values,
            "control": "PQ",
            "weather_cell_id": np.nan,
            "subtype": res.type.values,
            "source_id": np.nan,
        },
        index=pd.Index(res.id.values, name="name"),
    )

    # storage units; SimBench gives one efficiency for charging and discharging
    storage = _read(path, "Storage")
    if storage.empty:
        storage = pd.DataFrame(columns=["id", "node", "sR", "eStore", "etaStore"])
    storage_units_df = pd.DataFrame(
        {
            "bus": storage.node.map(bus).values,
            "control": "PQ",
            "p_nom": storage.sR.values,
            "max_hours": (storage.eStore / storage.sR).values,
            "efficiency_store": storage.etaStore.values,
            "efficiency_dispatch": storage.etaStore.values,
        },
        index=pd.Index(storage.id.values, name="name"),
    )

    topology = edisgo_obj.topology
    topology.buses_df = buses_df
    topology.lines_df = lines_df
    topology.transformers_df = transformers_df
    topology.transformers_hvmv_df = transformers_hvmv_df
    topology.switches_df = switches_df
    topology.loads_df = loads_df
    topology.generators_df = generators_df
    topology.storage_units_df = storage_units_df
    topology.grid_district = {
        "population": np.nan,
        "geom": MultiPoint(buses_df[["x", "y"]].values).convex_hull,
        "srid": 4326,
    }
    topology.mv_grid = MVGrid(id=mv_grid_id, edisgo_obj=edisgo_obj)
    topology.check_integrity()


def simbench_time_series(path, edisgo_obj, timeindex=None):
    """
    Active and reactive power time series of all components from SimBench profiles.

    SimBench profiles are relative values that are scaled with the rated power of
    each component, as in the `simbench` python package: loads with `pLoad` and
    `qLoad`, generators with `pRES` and `qRES`, storage units with `pStor` and
    `qStor`. SimBench uses the load sign convention for storage units, eDisGo the
    generator sign convention, so the sign of storage time series is flipped.

    SimBench profiles are 15-minute values for 2016. Their time labels are in
    local time, so that the hour switched to daylight saving time is missing and
    the hour switched back occurs twice. The rows are consecutive 15-minute steps;
    they are therefore indexed as such, starting at the first label, and then
    relabelled to the reference year (see
    :func:`~.tools.tools.shift_to_reference_year`), which drops 29 February.

    Parameters
    ----------
    path : str
        Path to the folder with the csv files of one SimBench grid.
    edisgo_obj : :class:`~.EDisGo`
        eDisGo object the SimBench grid in `path` was imported to with
        :func:`import_simbench_grid`.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Time steps in the reference year to return time series for. If None, all
        time steps of the year are returned. Default: None.

    Returns
    -------
    dict(str, :pandas:`pandas.DataFrame<DataFrame>`)
        Time series in MW and MVA with keys as the parameters of
        :attr:`~.EDisGo.set_time_series_manual`, e.g. 'loads_p'.

    """

    def scaled(table, p_col, q_col, suffix=("", ""), sign=1):
        elements = _read(path, table)
        if elements.empty:
            return {}
        profiles = _read(path, f"{table}Profile")
        start = pd.to_datetime(profiles.pop("time")[0], format="%d.%m.%Y %H:%M")
        profiles.index = pd.date_range(start, periods=len(profiles), freq="15min")
        profiles = shift_to_reference_year(edisgo_obj, profiles, f"{table}Profile")
        if timeindex is not None:
            profiles = profiles.loc[timeindex]
        name = {"Load": "loads", "RES": "generators", "Storage": "storage_units"}
        return {
            f"{name[table]}_{pq}": sign
            * profiles[elements.profile + sfx].set_axis(elements.id, axis=1)
            * elements[col].values
            for pq, col, sfx in zip(["p", "q"], [p_col, q_col], suffix)
        }

    return {
        **scaled("Load", "pLoad", "qLoad", suffix=("_pload", "_qload")),
        **scaled("RES", "pRES", "qRES"),
        **scaled("Storage", "pStor", "qStor", sign=-1),
    }
