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

from __future__ import annotations

import copy
import logging
import math

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from pandas import DataFrame
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.network import timeseries
from edisgo.network.grids import Grid
from edisgo.tools.pseudo_coordinates import make_pseudo_coordinates

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from edisgo import EDisGo


# Transform coordinates between the different coordinate systems
coor_transform = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
coor_transform_back = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def _transform_coordinates(
    df: DataFrame, transformer: Transformer, x_col: str = "x", y_col: str = "y"
) -> DataFrame:
    """
    Return a copy of `df` with the coordinates in `x_col` and `y_col` transformed.

    The input dataframe is not changed.

    """
    df = df.copy()
    x, y = transformer.transform(df[x_col].to_numpy(), df[y_col].to_numpy())
    df[x_col] = x
    df[y_col] = y
    return df


def _bus_weights(
    edisgo_obj: EDisGo,
    bus_to_new_bus: pd.Series,
    buses: list | pd.Index,
    with_transformers: bool,
) -> pd.Series:
    """
    Weight of each bus in `buses` for the k-means clustering.

    The weight is 1 + 1000 * (p_gen + p_load + s_tran) with p_gen and p_load the
    nominal generation and peak load in MW at all buses that `bus_to_new_bus` maps
    to the bus, and s_tran the nominal apparent power in MVA of the transformers
    whose primary side is the bus (only if `with_transformers` is True).

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    bus_to_new_bus : :pandas:`pandas.Series<Series>`
        Maps the buses of the grid (index) to the bus they are aggregated to.
    buses : list(str) or :pandas:`pandas.Index<Index>`
        Buses to get the weight for.
    with_transformers : bool
        If True, the transformer capacity is part of the weight (used for the MV
        grid).

    Returns
    -------
    :pandas:`pandas.Series<Series>`
        Weight of each bus in `buses`.

    """
    topology = edisgo_obj.topology
    generators_df = topology.generators_df
    loads_df = topology.loads_df
    p_gen = (
        generators_df.p_nom.groupby(generators_df.bus.map(bus_to_new_bus))
        .sum()
        .reindex(buses, fill_value=0.0)
    )
    p_load = (
        loads_df.p_set.groupby(loads_df.bus.map(bus_to_new_bus))
        .sum()
        .reindex(buses, fill_value=0.0)
    )
    if with_transformers:
        s_tran = (
            topology.transformers_df.groupby("bus0")
            .s_nom.sum()
            .reindex(buses, fill_value=0.0)
        )
    else:
        s_tran = 0
    return 1 + 1000 * (p_gen + p_load + s_tran)


def _assign_to_medoids(
    graph: nx.Graph, buses_df: DataFrame, kmeans: KMeans, n_clusters: int
) -> tuple[np.ndarray, list]:
    """
    Assign every bus to the closest medoid of the k-means clusters.

    The medoid of a cluster is the bus closest to the cluster center. Every bus is
    then assigned to the medoid with the shortest path in the grid graph (Dijkstra,
    weighted by line length). On equal distance, the medoid of the cluster with the
    lower number is chosen.

    Parameters
    ----------
    graph : :networkx:`networkx.Graph<>`
        Graph containing all buses in `buses_df`, with the line length as edge
        attribute "length".
    buses_df : :pandas:`pandas.DataFrame<DataFrame>`
        Buses to assign, with the coordinates the clustering was fitted on in
        columns "x" and "y".
    kmeans : ``sklearn.cluster.KMeans``
        Fitted k-means clustering.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    tuple(:numpy:`numpy.ndarray<ndarray>`, list(str))
        Number of the assigned medoid for each bus in `buses_df` and the bus names
        of the medoids, ordered by cluster number.

    Raises
    ------
    ValueError
        If a bus in `buses_df` cannot be reached from any medoid in `graph`.

    """
    buses = buses_df.index
    dist_to_cluster_center = kmeans.transform(buses_df.loc[:, ["x", "y"]]).min(axis=1)
    medoids = []
    for n in range(n_clusters):
        members = np.flatnonzero(kmeans.labels_ == n)
        medoids.append(buses[members[np.argmin(dist_to_cluster_center[members])]])

    position = {bus: i for i, bus in enumerate(buses)}
    distances = np.full((len(buses), n_clusters), np.inf)
    for n, medoid in enumerate(medoids):
        path_lengths = nx.single_source_dijkstra_path_length(
            graph, medoid, weight="length"
        )
        for bus, length in path_lengths.items():
            i = position.get(bus)
            if i is not None:
                distances[i, n] = length
    unreachable = np.isinf(distances).all(axis=1)
    if unreachable.any():
        raise ValueError(
            f"Bus(es) {buses[unreachable].tolist()} cannot be reached from any "
            f"cluster medoid in the grid graph. Check that they are connected by "
            f"lines."
        )
    return distances.argmin(axis=1), medoids


def _make_grid_list(edisgo_obj: EDisGo, grid: object = None) -> list:
    """
    Get a list of all grids in the EDisGo object or a list with the specified grid.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object to get the grids from.
    grid : str
        Name of the grid. If None, all grids of the EDisGo objects are used.
        Default: None.

    Returns
    -------
    list(:class:`~.network.grids.Grid`)
        List of the :class:`~.network.grids.Grid` objects.

    """
    if edisgo_obj is None and grid is None:
        raise ValueError("Pass an EDisGo object and an grid")
    elif grid is not None:
        grid_list = edisgo_obj.topology.grids
        grid_name_list = list(map(str, grid_list))
        grid_list = [grid_list[grid_name_list.index(str(grid))]]
    else:
        grid_list = edisgo_obj.topology.grids

    return grid_list


def find_buses_of_interest(edisgo_root: EDisGo) -> set:
    """
    Return buses with load and voltage issues, determined doing a worst-case powerflow
    analysis.

    Parameters
    ----------
    edisgo_root : :class:`~.EDisGo`
        The investigated EDisGo object.

    Returns
    -------
    set(str)
        Set with the names of the buses with load and voltage issues.

    """
    logger.debug("Find buses of interest.")

    edisgo_obj = copy.deepcopy(edisgo_root)
    edisgo_obj.timeseries = timeseries.TimeSeries()
    edisgo_obj.timeseries.set_worst_case(edisgo_obj, ["feed-in_case", "load_case"])
    edisgo_obj.analyze()

    buses_of_interest = set()
    mv_lines = checks.mv_line_max_relative_overload(edisgo_obj)
    lv_lines = checks.lv_line_max_relative_overload(edisgo_obj)
    lines = mv_lines.index.tolist()
    lines = lines + lv_lines.index.tolist()
    for line in lines:
        buses_of_interest.add(edisgo_obj.topology.lines_df.loc[line, "bus0"])
        buses_of_interest.add(edisgo_obj.topology.lines_df.loc[line, "bus1"])

    mv_buses = checks.voltage_issues(edisgo_obj, voltage_level="mv")
    buses_of_interest.update(mv_buses.index.tolist())

    lv_buses = checks.voltage_issues(edisgo_obj, voltage_level="lv")
    buses_of_interest.update(lv_buses.index.tolist())

    return buses_of_interest


def rename_virtual_buses(
    partial_busmap_df: DataFrame, transformer_node: str
) -> DataFrame:
    """
    Rename virtual buses so that no virtual transformer bus is created.

    Parameters
    ----------
    partial_busmap_df : :pandas:`pandas.DataFrame<DataFrame>`
        Busmap to work on.
    transformer_node : str
        Transformer node name.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Busmap with applied changes.

    """
    nodes = partial_busmap_df.index.to_list()
    pairs = []
    for node in nodes:
        if node.split("_")[0] == "virtual":
            if (
                partial_busmap_df.loc[node, "new_bus"]
                != partial_busmap_df.loc[node.removeprefix("virtual_"), "new_bus"]
            ):
                pairs.append([node, node.removeprefix("virtual_")])

    logger.debug(f"Pairs: {pairs}")
    logger.debug(f"Length pairs: {len(pairs)}")
    if len(pairs) > 0:
        logger.debug("Rename virtual buses")
        for feeder_virtual, feeder_non_virtual in pairs:
            old_name_of_virtual_node = partial_busmap_df.loc[feeder_virtual, "new_bus"]
            nodes_in_the_same_cluster_as_virtual_node = partial_busmap_df.loc[
                partial_busmap_df.loc[:, "new_bus"].isin([old_name_of_virtual_node])
            ].index.tolist()

            for nodes_to_add_a_virtual in nodes_in_the_same_cluster_as_virtual_node:
                # Stop making a virtual transformer bus
                # Stop renaming the transformer bus
                if (
                    partial_busmap_df.loc[feeder_non_virtual, "new_bus"]
                    != transformer_node
                ) and (
                    partial_busmap_df.loc[nodes_to_add_a_virtual, "new_bus"]
                    != transformer_node
                ):
                    partial_busmap_df.loc[nodes_to_add_a_virtual, "new_bus"] = (
                        "virtual_"
                        + partial_busmap_df.loc[feeder_non_virtual, "new_bus"]
                    )
    return partial_busmap_df


def remove_short_end_lines(edisgo_obj: EDisGo):
    """
    Method to remove end lines under 1 meter to reduce size of edisgo object.

    Short lines inside at the end are removed in this function, including the end node.
    Components that were originally connected to the end node are reconnected to the
    upstream node.

    This function does currently not remove short lines that are no end lines.

    Parameters
    ----------
    edisgo : :class:`~.EDisGo`

    """

    def apply_busmap_on_buses_df(series):
        if series.name in busmap:
            series.loc["new_bus"] = busmap[series.name]
        else:
            series.loc["new_bus"] = series.name

        return series

    def apply_busmap_on_lines_df(series):
        if series.bus0 in busmap:
            series.loc["bus0"] = busmap[series.bus0]
        if series.bus1 in busmap:
            series.loc["bus1"] = busmap[series.bus1]

        return series

    def apply_busmap(series):
        if series.bus in busmap:
            series.loc["bus"] = busmap[series.bus]

        return series

    logger.debug("Removing 1 m end lines.")

    G = edisgo_obj.to_graph()
    lines_df = edisgo_obj.topology.lines_df.copy()
    busmap = {}
    unused_lines = []
    for index, row in lines_df.iterrows():
        if row.length <= 0.001:
            # find lines that have at one bus only one neighbor
            # and at the other more than one
            number_of_neighbors_bus0 = G.degree(row.bus0)
            number_of_neighbors_bus1 = G.degree(row.bus1)
            if (
                (number_of_neighbors_bus0 != number_of_neighbors_bus1)
                and (row.bus0.split("_")[0] != "virtual")
                and (row.bus1.split("_")[0] != "virtual")
            ):
                if (number_of_neighbors_bus0 > number_of_neighbors_bus1) and (
                    number_of_neighbors_bus1 == 1
                ):
                    unused_lines.append(index)
                    busmap[row.bus1] = row.bus0
                elif (
                    number_of_neighbors_bus1 > number_of_neighbors_bus0
                ) and number_of_neighbors_bus0 == 1:
                    unused_lines.append(index)
                    busmap[row.bus0] = row.bus1

    logger.info(f"Drop {len(unused_lines)} of {lines_df.shape[0]} 1 m lines.")
    # Apply the busmap on the components
    lines_df = lines_df.drop(unused_lines)
    lines_df = lines_df.apply(apply_busmap_on_lines_df, axis="columns")

    buses_df = edisgo_obj.topology.buses_df
    buses_df = buses_df.apply(apply_busmap_on_buses_df, axis="columns")
    buses_df = buses_df.groupby(
        by=["new_bus"], dropna=False, as_index=False, sort=False
    ).first()
    buses_df = buses_df.set_index("new_bus")

    loads_df = edisgo_obj.topology.loads_df
    loads_df = loads_df.apply(apply_busmap, axis="columns")

    generators_df = edisgo_obj.topology.generators_df
    generators_df = generators_df.apply(apply_busmap, axis="columns")

    storage_units_df = edisgo_obj.topology.storage_units_df
    storage_units_df = storage_units_df.apply(apply_busmap, axis="columns")

    edisgo_obj.topology.lines_df = lines_df
    edisgo_obj.topology.buses_df = buses_df
    edisgo_obj.topology.loads_df = loads_df
    edisgo_obj.topology.generators_df = generators_df
    edisgo_obj.topology.storage_units_df = storage_units_df


def remove_lines_under_one_meter(edisgo_obj: EDisGo) -> EDisGo:
    """
    Remove the lines under one meter. Sometimes these line are causing convergence
    problems of the power flow calculation or making problems with the clustering
    methods.

    Function might be a bit overengineered, so that the station bus is never dropped.
    """
    # ToDo this function does currently not work correctly as it may lead to
    #  isolated nodes and possibly broken switches, plus it should be merged with
    #  function remove_one_meter_lines
    # def apply_busmap_on_buses_df(series):
    #     if series.name in busmap:
    #         series.loc["new_bus"] = busmap[series.name]
    #     else:
    #         series.loc["new_bus"] = series.name
    #     return series
    #
    # def apply_busmap_on_lines_df(series):
    #     if series.bus0 in busmap:
    #         series.loc["bus0"] = busmap[series.bus0]
    #     if series.bus1 in busmap:
    #         series.loc["bus1"] = busmap[series.bus1]
    #
    #     return series
    #
    # def apply_busmap(series):
    #     if series.bus in busmap:
    #         series.loc["bus"] = busmap[series.bus]
    #
    #     return series
    #
    # busmap = {}
    # unused_lines = []
    #
    # grid_list = [edisgo_obj.topology.mv_grid]
    # grid_list = grid_list + list(edisgo_obj.topology.mv_grid.lv_grids)
    #
    # for grid in grid_list:
    #     G = grid.graph
    #
    #     transformer_node = grid.transformers_df.bus1.values[0]
    #
    #     lines_df = grid.lines_df.copy()
    #
    #     for index, row in lines_df.iterrows():
    #         if row.length < 0.001:
    #
    #             distance_bus_0, path = nx.single_source_dijkstra(
    #                 G, source=transformer_node, target=row.bus0, weight="length"
    #             )
    #             distance_bus_1, path = nx.single_source_dijkstra(
    #                 G, source=transformer_node, target=row.bus1, weight="length"
    #             )
    #
    #             logger.debug(
    #                 'Line "{}" is {:.5f}m long and will be removed.'.format(
    #                     index, row.length * 1000
    #                 )
    #             )
    #             logger.debug(
    #                 "Bus0: {} - Distance0: {}".format(row.bus0, distance_bus_0)
    #             )
    #             logger.debug(
    #                 "Bus1: {} - Distance1: {}".format(row.bus1, distance_bus_1)
    #             )
    #             # map bus farther away to bus closer to the station
    #             # ToDo check if either node is already in the busmap
    #             # ToDo make sure no virtual bus is dropped
    #             if distance_bus_0 < distance_bus_1:
    #                 busmap[row.bus1] = row.bus0
    #                 if distance_bus_0 < 0.001:
    #                     busmap[row.bus0] = transformer_node
    #                     busmap[row.bus1] = transformer_node
    #             elif distance_bus_0 > distance_bus_1:
    #                 busmap[row.bus0] = row.bus1
    #                 if distance_bus_1 < 0.001:
    #                     busmap[row.bus0] = transformer_node
    #                     busmap[row.bus1] = transformer_node
    #             else:
    #                 raise ValueError("ERROR")
    #
    #             unused_lines.append(index)
    #
    # logger.debug("Busmap: {}".format(busmap))
    # # Apply the busmap on the components
    # transformers_df = edisgo_obj.topology.transformers_df.copy()
    # transformers_df = transformers_df.apply(apply_busmap_on_lines_df, axis="columns")
    # edisgo_obj.topology.transformers_df = transformers_df
    #
    # lines_df = edisgo_obj.topology.lines_df.copy()
    # lines_df = lines_df.drop(unused_lines)
    # lines_df = lines_df.apply(apply_busmap_on_lines_df, axis="columns")
    # edisgo_obj.topology.lines_df = lines_df
    #
    # buses_df = edisgo_obj.topology.buses_df.copy()
    # buses_df.index.name = "bus"
    # buses_df = buses_df.apply(apply_busmap_on_buses_df, axis="columns")
    # buses_df = buses_df.groupby(
    #     by=["new_bus"], dropna=False, as_index=False, sort=False
    # ).first()
    # buses_df = buses_df.set_index("new_bus")
    # edisgo_obj.topology.buses_df = buses_df
    #
    # loads_df = edisgo_obj.topology.loads_df.copy()
    # loads_df = loads_df.apply(apply_busmap, axis="columns")
    # edisgo_obj.topology.loads_df = loads_df
    #
    # generators_df = edisgo_obj.topology.generators_df.copy()
    # generators_df = generators_df.apply(apply_busmap, axis="columns")
    # edisgo_obj.topology.generators_df = generators_df
    #
    # storage_units_df = edisgo_obj.topology.storage_units_df.copy()
    # storage_units_df = storage_units_df.apply(apply_busmap, axis="columns")
    # edisgo_obj.topology.storage_units_df = storage_units_df
    #
    # return edisgo_obj
    raise NotImplementedError


def make_busmap_grid(
    edisgo_obj: EDisGo,
    grid: None | str = None,
    mode: str = "kmeansdijkstra",
    reduction_factor: float = 0.25,
    preserve_trafo_bus_coordinates: bool = True,
) -> DataFrame:
    """
    Making busmap for the cluster area 'grid'.

    Every grid is clustered individually.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object for which the busmap is created.
    grid : str or None
        If None, busmap is created for all grids, else only for the selected grid.
        Default: None.
    mode : str
        "kmeans" or "kmeansdijkstra" as clustering method. See parameter `mode` in
        function :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
        Default: "kmeansdijkstra".
    reduction_factor : float
        Factor to reduce number of nodes by. Must be between 0 and 1. Default: 0.25.
    preserve_trafo_bus_coordinates : bool
        If True, transformers have the same coordinates after the clustering, else
        the transformer coordinates are changed by the clustering. Default: True.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Busmap which maps the old bus names to the new bus names with new coordinates.
        See return value in function :func:`~make_busmap` for more information.

    References
    ----------
    In parts based on `PyPSA spatial complexity reduction <https://docs.pypsa.org
    /v0.35.1/examples/spatial-clustering.html>`_.

    """

    logger.debug("Start making busmap for grids.")

    grid_list = _make_grid_list(edisgo_obj, grid=grid)
    mvgd_id = edisgo_obj.topology.mv_grid.id

    # Collect per-grid busmaps in a list and concat once after the loop instead
    # of growing the DataFrame on every iteration.
    busmap_df_list: list[pd.DataFrame] = []
    # Cluster every grid
    for grid in grid_list:
        v_grid = grid.nominal_voltage
        logger.debug(f"Make busmap for grid: {grid}, v_nom={v_grid}")

        grid_type = str(grid).split("_")[0]
        buses_df = _transform_coordinates(grid.buses_df, coor_transform)
        graph = grid.graph
        transformer_bus = grid.transformers_df.bus1.iloc[0]

        buses_df["weight"] = _bus_weights(
            edisgo_obj,
            pd.Series(buses_df.index, index=buses_df.index),
            buses_df.index,
            with_transformers=grid_type == "MVGrid",
        )
        # Calculate number of clusters
        number_of_distinct_nodes = buses_df.groupby(by=["x", "y"]).first().shape[0]
        logger.debug("Number_of_distinct_nodes = " + str(number_of_distinct_nodes))
        n_clusters = math.ceil(number_of_distinct_nodes * reduction_factor)
        logger.debug(f"n_clusters = {n_clusters}")
        # Cluster with kmeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

        kmeans.fit(buses_df.loc[:, ["x", "y"]], sample_weight=buses_df.loc[:, "weight"])

        if mode == "kmeans":
            cluster = kmeans.labels_
            new_coordinates = kmeans.cluster_centers_[cluster]
        elif mode == "kmeansdijkstra":
            # Use dijkstra to select clusters
            cluster, medoids = _assign_to_medoids(graph, buses_df, kmeans, n_clusters)
            new_coordinates = buses_df.loc[medoids, ["x", "y"]].to_numpy()[cluster]

        if grid_type == "LVGrid":
            prefix = f"Bus_mvgd_{mvgd_id}_lvgd_{grid.id}_"
        else:
            prefix = f"Bus_mvgd_{mvgd_id}_"
        partial_busmap_df = pd.DataFrame(index=buses_df.index)
        partial_busmap_df["new_bus"] = [prefix + str(int(n)) for n in cluster]
        partial_busmap_df[["new_x", "new_y"]] = new_coordinates

        partial_busmap_df.loc[
            partial_busmap_df.new_bus.isin(
                [partial_busmap_df.loc[transformer_bus, "new_bus"]]
            ),
            "new_bus",
        ] = transformer_bus
        # Write trafo coordinates back, this helps to get better results
        if preserve_trafo_bus_coordinates:
            partial_busmap_df.loc[
                partial_busmap_df.new_bus.isin(
                    [partial_busmap_df.loc[transformer_bus, "new_bus"]]
                ),
                "new_x",
            ] = buses_df.loc[transformer_bus, "x"]
            partial_busmap_df.loc[
                partial_busmap_df.new_bus.isin(
                    [partial_busmap_df.loc[transformer_bus, "new_bus"]]
                ),
                "new_y",
            ] = buses_df.loc[transformer_bus, "y"]

        partial_busmap_df.index.name = "old_bus"

        if grid_type == "MVGrid":
            partial_busmap_df = rename_virtual_buses(partial_busmap_df, transformer_bus)

        partial_busmap_df = _transform_coordinates(
            partial_busmap_df, coor_transform_back, "new_x", "new_y"
        )

        busmap_df_list.append(partial_busmap_df)

    busmap_df = pd.concat(busmap_df_list) if busmap_df_list else pd.DataFrame()

    return busmap_df


def make_busmap_feeders(
    edisgo_obj: EDisGo = None,
    grid: None | Grid = None,
    mode: str = "kmeansdijkstra",
    reduction_factor: float = 0.25,
    reduction_factor_not_focused: bool | float = False,
) -> DataFrame:
    """
    Making busmap for the cluster area 'feeder'.

    Every feeder is clustered individually.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object for which the busmap is created.
    grid : str or None
        If None, busmap is created for all grids, else only for the selected grid.
        Default: None.
    mode : str
        "kmeans" or "kmeansdijkstra" as clustering method. See parameter `mode` in
        function :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
        Default: "kmeansdijkstra".
    reduction_factor : float
        Factor to reduce number of nodes by. Must be between 0 and 1. Default: 0.25.
    reduction_factor_not_focused : bool or float
        If False, the focus method is not used. If between 0 and 1, this sets the
        reduction factor for buses not of interest. See parameter
        `reduction_factor_not_focused` in function :func:`~make_busmap`
        for more information. Default: False.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Busmap which maps the old bus names to the new bus names with new coordinates.
        See return value in function :func:`~make_busmap` for more information.

    References
    ----------
    In parts based on `PyPSA spatial complexity reduction <https://docs.pypsa.org
    /v0.35.1/examples/spatial-clustering.html>`_.

    """

    def make_name(number_of_feeder_node):
        if number_of_feeder_node == 0:
            name = transformer_node
        elif mvgd_id == grid_id:
            name = (
                "Bus_mvgd_"
                + str(mvgd_id)
                + "_F"
                + str(number_of_feeder)
                + "_B"
                + str(number_of_feeder_node)
            )
        else:
            name = (
                "Bus_mvgd_"
                + str(mvgd_id)
                + "_lvgd_"
                + str(grid_id)
                + "_F"
                + str(number_of_feeder)
                + "_B"
                + str(number_of_feeder_node)
            )

        return name

    logger.debug("Start making busmap for feeders.")

    # Clustering is done in EPSG:3035 (metres). The coordinates are transformed in a
    # copy, so that the coordinates of the EDisGo object stay unchanged.
    buses_df = _transform_coordinates(edisgo_obj.topology.buses_df, coor_transform)

    grid_list = _make_grid_list(edisgo_obj, grid=grid)
    # Collect per-grid busmaps in a list and concat once after the loop instead
    # of growing the DataFrame on every iteration.
    busmap_df_list: list[pd.DataFrame] = []
    mvgd_id = edisgo_obj.topology.mv_grid.id

    if reduction_factor_not_focused is False:
        focus_mode = False
    else:
        focus_mode = True

    if focus_mode:
        buses_of_interest = find_buses_of_interest(edisgo_obj)

    for grid in grid_list:
        grid_id = grid.id
        v_grid = grid.nominal_voltage
        logger.debug(f"Make busmap for grid: {grid}, v_nom={v_grid}")

        grid_buses_df = buses_df.loc[grid.buses_df.index]
        graph_root = grid.graph
        transformer_node = grid.transformers_df.bus1.values[0]
        transformer_coordinates = grid_buses_df.loc[
            transformer_node, ["x", "y"]
        ].tolist()
        logger.debug(f"Transformer node: {transformer_node}")

        neighbors = list(nx.neighbors(graph_root, transformer_node))
        neighbors.sort()
        logger.debug(f"Transformer has {len(neighbors)} neighbors: {neighbors}")

        graph_without_transformer = graph_root.copy()
        graph_without_transformer.remove_node(transformer_node)

        partial_busmap_df = pd.DataFrame(index=grid_buses_df.index)
        partial_busmap_df.index.name = "old_bus"
        partial_busmap_df["new_bus"] = partial_busmap_df.index
        partial_busmap_df[["new_x", "new_y"]] = grid_buses_df.loc[
            partial_busmap_df.index, ["x", "y"]
        ].values

        weights = _bus_weights(
            edisgo_obj,
            partial_busmap_df["new_bus"],
            partial_busmap_df.index,
            with_transformers=str(grid).split("_")[0] == "MVGrid",
        )

        # New bus and coordinates of the buses of all feeders, written to the
        # busmap at once after the loop.
        feeder_buses = []
        feeder_new_buses = []
        feeder_new_coordinates = []

        number_of_feeder = 0
        feeder_graphs = list(nx.connected_components(graph_without_transformer))
        feeder_graphs.sort()
        for feeder_nodes in feeder_graphs:
            feeder_nodes = list(feeder_nodes)
            feeder_nodes.sort()

            feeder_buses_df = grid_buses_df.loc[feeder_nodes, :]

            if focus_mode and buses_of_interest.isdisjoint(feeder_nodes):
                selected_reduction_factor = reduction_factor_not_focused
            else:
                selected_reduction_factor = reduction_factor

            number_of_distinct_nodes = (
                feeder_buses_df.groupby(by=["x", "y"]).first().shape[0]
            )
            logger.debug("Number_of_distinct_nodes = " + str(number_of_distinct_nodes))
            n_clusters = math.ceil(selected_reduction_factor * number_of_distinct_nodes)
            logger.debug(f"n_clusters = {n_clusters}")

            # Aggregate to transformer bus if there are no clusters
            if n_clusters == 0:
                feeder_new_buses += [transformer_node] * len(feeder_nodes)
                feeder_new_coordinates.append(
                    np.tile(transformer_coordinates, (len(feeder_nodes), 1))
                )
            else:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                kmeans.fit(
                    feeder_buses_df.loc[:, ["x", "y"]],
                    sample_weight=weights.loc[feeder_nodes],
                )

                if mode == "kmeans":
                    cluster = kmeans.labels_
                    new_coordinates = kmeans.cluster_centers_[cluster]
                elif mode == "kmeansdijkstra":
                    # A feeder is a connected component of the grid without the
                    # transformer bus. A shortest path between two of its buses
                    # never enters another feeder, as it would have to pass the
                    # transformer bus twice. It may pass the transformer bus once,
                    # e.g. in a ring closed at the station, so the subgraph of the
                    # feeder and the transformer bus is enough.
                    cluster, medoids = _assign_to_medoids(
                        graph_root.subgraph(feeder_nodes + [transformer_node]).copy(),
                        feeder_buses_df,
                        kmeans,
                        n_clusters,
                    )
                    new_coordinates = feeder_buses_df.loc[
                        medoids, ["x", "y"]
                    ].to_numpy()[cluster]

                names = [make_name(n + 1) for n in range(n_clusters)]
                feeder_new_buses += [names[n] for n in cluster]
                feeder_new_coordinates.append(new_coordinates)
            feeder_buses += feeder_nodes
            number_of_feeder = number_of_feeder + 1

        if feeder_buses:
            partial_busmap_df.loc[feeder_buses, "new_bus"] = feeder_new_buses
            partial_busmap_df.loc[feeder_buses, ["new_x", "new_y"]] = np.concatenate(
                feeder_new_coordinates
            )

        if str(grid).split("_")[0] == "MVGrid":
            partial_busmap_df = rename_virtual_buses(
                partial_busmap_df, transformer_node
            )

        busmap_df_list.append(partial_busmap_df)

    busmap_df = pd.concat(busmap_df_list) if busmap_df_list else pd.DataFrame()
    busmap_df = _transform_coordinates(busmap_df, coor_transform_back, "new_x", "new_y")
    busmap_df.sort_index(inplace=True)
    return busmap_df


def make_busmap_main_feeders(
    edisgo_obj: EDisGo = None,
    grid: None | Grid = None,
    mode: str = "kmeansdijkstra",
    reduction_factor: float = 0.25,
    reduction_factor_not_focused: bool | float = False,
) -> DataFrame:
    """
    Making busmap for the cluster area 'main_feeder'.

    Every main feeder is clustered individually. The main feeder is selected as the
    longest path in the feeder. All nodes are aggregated to this main feeder and
    then the feeder is clustered.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object for which the busmap is created.
    grid : str or None
        If None, busmap is created for all grids, else only for the selected grid.
        Default: None.
    mode : str
        "kmeans", "kmeansdijkstra", "aggregate_to_main_feeder" or
        "equidistant_nodes" as clustering method. See parameter `mode` in
        function :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
        Default: "kmeansdijkstra".
    reduction_factor : float
        Factor to reduce number of nodes by. Must be between 0 and 1. Default: 0.25.
    reduction_factor_not_focused : bool or float
        If False, the focus method is not used. If between 0 and 1, this sets the
        reduction factor for buses not of interest. See parameter
        `reduction_factor_not_focused` in function :func:`~make_busmap`
        for more information. Default: False.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Busmap which maps the old bus names to the new bus names with new coordinates.
        See return value in function :func:`~make_busmap` for more information.

    References
    ----------
    In parts based on `PyPSA spatial complexity reduction <https://docs.pypsa.org
    /v0.35.1/examples/spatial-clustering.html>`_.

    """

    def make_name(number_of_feeder_node):
        if number_of_feeder_node == 0:
            name = transformer_node
        elif mvgd_id == grid_id:
            name = (
                "Bus_mvgd_"
                + str(mvgd_id)
                + "_F"
                + str(number_of_feeder)
                + "_B"
                + str(number_of_feeder_node)
            )
        else:
            name = (
                "Bus_mvgd_"
                + str(mvgd_id)
                + "_lvgd_"
                + str(grid_id)
                + "_F"
                + str(number_of_feeder)
                + "_B"
                + str(number_of_feeder_node)
            )

        return name

    def next_main_node(node_to_delete, graph_root, main_feeder_nodes):
        for node, predecessor in nx.bfs_predecessors(graph_root, source=node_to_delete):
            if node in main_feeder_nodes:
                return node

    logger.debug("Start making busmap for main feeders")

    # Clustering is done in EPSG:3035 (metres). The coordinates are transformed in a
    # copy, so that the coordinates of the EDisGo object stay unchanged.
    if mode != "aggregate_to_main_feeder":
        buses_df = _transform_coordinates(edisgo_obj.topology.buses_df, coor_transform)
    else:
        buses_df = edisgo_obj.topology.buses_df

    grid_list = _make_grid_list(edisgo_obj, grid=grid)
    # Collect per-grid busmaps in a list and concat once after the loop instead
    # of growing the DataFrame on every iteration.
    busmap_df_list: list[pd.DataFrame] = []
    mvgd_id = edisgo_obj.topology.mv_grid.id

    if reduction_factor_not_focused is False:
        focus_mode = False
    else:
        focus_mode = True

    if focus_mode:
        buses_of_interest = find_buses_of_interest(edisgo_obj)

    for grid in grid_list:
        grid_id = grid.id
        v_grid = grid.nominal_voltage
        logger.debug(f"Make busmap for grid: {grid}, v_nom={v_grid}")

        grid_buses_df = buses_df.loc[grid.buses_df.index]
        graph_root = grid.graph
        transformer_node = grid.transformers_df.bus1.values[0]
        transformer_coordinates = grid_buses_df.loc[
            transformer_node, ["x", "y"]
        ].tolist()
        logger.debug(f"Transformer node: {transformer_node}")

        neighbors = list(nx.neighbors(graph_root, transformer_node))
        neighbors.sort()
        logger.debug(f"Transformer has {len(neighbors)} neighbors: {neighbors}")

        graph_without_transformer = graph_root.copy()
        graph_without_transformer.remove_node(transformer_node)

        end_nodes = []
        for node in neighbors:
            path = nx.single_source_dijkstra_path_length(
                graph_without_transformer, node, weight="length"
            )
            end_node = max(path, key=path.get)
            end_nodes.append(end_node)

        main_feeders_df = pd.DataFrame(
            columns=["distance", "number_of_nodes_in_path", "path", "end_node"],
            dtype=object,
        )

        i = 0
        main_feeder_nodes = [transformer_node]
        for end_node in end_nodes:
            distance, path = nx.single_source_dijkstra(
                graph_root, source=transformer_node, target=end_node, weight="length"
            )

            main_feeder_nodes = main_feeder_nodes + path

            # Advanced method
            if mode != "aggregate_to_main_feeder":
                main_feeders_df.loc[i, "distance"] = distance
                main_feeders_df.loc[i, "end_node"] = end_node
                # transformer node is not included
                main_feeders_df.loc[i, "number_of_nodes_in_path"] = len(path) - 1
                main_feeders_df.loc[i, "path"] = path
                i = i + 1

        # delete duplicates
        main_feeder_nodes = list(dict.fromkeys(main_feeder_nodes))
        main_feeder_node_set = set(main_feeder_nodes)
        not_main_nodes = [
            node for node in graph_root.nodes() if node not in main_feeder_node_set
        ]
        partial_busmap_df = pd.DataFrame(index=grid_buses_df.index)
        partial_busmap_df.index.name = "old_bus"
        # Initially every bus is mapped to itself with its own coordinates.
        partial_busmap_df["new_bus"] = partial_busmap_df.index
        partial_busmap_df[["new_x", "new_y"]] = grid_buses_df.loc[
            partial_busmap_df.index, ["x", "y"]
        ].values

        graph_cleaned = graph_root.copy()
        graph_cleaned.remove_nodes_from(not_main_nodes)

        if not_main_nodes:
            next_main_nodes = [
                next_main_node(node, graph_root, main_feeder_node_set)
                for node in not_main_nodes
            ]
            partial_busmap_df.loc[not_main_nodes, ["new_x", "new_y"]] = (
                partial_busmap_df.loc[next_main_nodes, ["new_x", "new_y"]].values
            )
            partial_busmap_df.loc[not_main_nodes, "new_bus"] = next_main_nodes

        # Advanced method
        if mode == "equidistant_nodes":

            def short_coordinates(root_node, end_node, branch_length, node_number):
                # Calculate coordinates for the feeder nodes
                angle = math.degrees(
                    math.atan2(end_node[1] - root_node[1], end_node[0] - root_node[0])
                )

                branch_length = 1000 * branch_length / 1.3

                x_new = root_node[0] + branch_length * node_number * math.cos(
                    math.radians(angle)
                )
                y_new = root_node[1] + branch_length * node_number * math.sin(
                    math.radians(angle)
                )

                return x_new, y_new

            distances_from_transformer = nx.single_source_dijkstra_path_length(
                graph_cleaned, transformer_node, weight="length"
            )
            i = 0
            for end_node in end_nodes:
                number_of_feeder = end_nodes.index(end_node)
                feeder = main_feeders_df.loc[number_of_feeder, :]

                # Calculate nodes per feeder
                if focus_mode and buses_of_interest.isdisjoint(
                    partial_busmap_df.index[
                        partial_busmap_df.new_bus.isin(feeder.path[1:])
                    ]
                ):
                    selected_reduction_factor = reduction_factor_not_focused
                else:
                    selected_reduction_factor = reduction_factor
                # Nodes per feeder should be minimum 1
                if selected_reduction_factor < 1:
                    nodes_per_feeder = math.ceil(
                        selected_reduction_factor * feeder.number_of_nodes_in_path
                    )
                else:
                    nodes_per_feeder = int(selected_reduction_factor)
                # Nodes per feeder should not be bigger than nodes in path
                if nodes_per_feeder > feeder.number_of_nodes_in_path:
                    nodes_per_feeder = feeder.number_of_nodes_in_path

                branch_length = feeder.distance / nodes_per_feeder

                # Calculate the assignment of the feeders
                new_feeder = np.zeros(nodes_per_feeder + 1)
                for n in range(0, new_feeder.shape[0]):
                    new_feeder[n] = n * branch_length
                old_feeder = np.zeros(feeder.number_of_nodes_in_path + 1, dtype=int)
                node_number = 0
                for node in feeder.path:
                    old_feeder[node_number] = np.abs(
                        new_feeder - distances_from_transformer[node]
                    ).argmin()
                    node_number += 1

                # Make busmap for feeders
                end_coordinates = grid_buses_df.loc[
                    end_nodes[number_of_feeder], ["x", "y"]
                ].tolist()
                for node_number in range(0, old_feeder.shape[0]):
                    old_bus = feeder.path[node_number]
                    partial_busmap_df.loc[old_bus, "new_bus"] = make_name(
                        old_feeder[node_number]
                    )
                    coor = short_coordinates(
                        transformer_coordinates,
                        end_coordinates,
                        branch_length,
                        old_feeder[node_number],
                    )

                    partial_busmap_df.loc[old_bus, "new_x"] = coor[0]
                    partial_busmap_df.loc[old_bus, "new_y"] = coor[1]
                i += 1

        elif (mode == "kmeans") or (mode == "kmeansdijkstra"):
            i = 0
            for end_node in end_nodes:
                number_of_feeder = end_nodes.index(end_node)
                feeder = main_feeders_df.loc[number_of_feeder, :]
                feeder.loc["path"].remove(transformer_node)
                feeder_buses_df = grid_buses_df.loc[feeder.path, :]
                weights = _bus_weights(
                    edisgo_obj,
                    partial_busmap_df["new_bus"],
                    feeder_buses_df.index,
                    with_transformers=str(grid).split("_")[0] == "MVGrid",
                )

                if focus_mode and buses_of_interest.isdisjoint(
                    partial_busmap_df.index[
                        partial_busmap_df.new_bus.isin(feeder_buses_df.index)
                    ]
                ):
                    selected_reduction_factor = reduction_factor_not_focused
                else:
                    selected_reduction_factor = reduction_factor

                number_of_distinct_nodes = (
                    feeder_buses_df.groupby(by=["x", "y"]).first().shape[0]
                )
                logger.debug(
                    "Number_of_distinct_nodes = " + str(number_of_distinct_nodes)
                )
                n_clusters = math.ceil(
                    selected_reduction_factor * number_of_distinct_nodes
                )
                logger.debug(f"n_clusters = {n_clusters}")

                # Aggregate to transformer bus if there are no clusters
                feeder_nodes = feeder_buses_df.index
                if n_clusters == 0:
                    partial_busmap_df.loc[feeder_nodes, "new_bus"] = transformer_node
                    partial_busmap_df.loc[feeder_nodes, "new_x"] = (
                        transformer_coordinates[0]
                    )
                    partial_busmap_df.loc[feeder_nodes, "new_y"] = (
                        transformer_coordinates[1]
                    )
                else:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                    kmeans.fit(
                        feeder_buses_df.loc[:, ["x", "y"]],
                        sample_weight=weights,
                    )

                    if mode == "kmeans":
                        cluster = kmeans.labels_
                        new_coordinates = kmeans.cluster_centers_[cluster]
                    elif mode == "kmeansdijkstra":
                        cluster, medoids = _assign_to_medoids(
                            graph_root, feeder_buses_df, kmeans, n_clusters
                        )
                        new_coordinates = feeder_buses_df.loc[
                            medoids, ["x", "y"]
                        ].to_numpy()[cluster]

                    names = [make_name(n + 1) for n in range(n_clusters)]
                    partial_busmap_df.loc[feeder_nodes, "new_bus"] = [
                        names[n] for n in cluster
                    ]
                    partial_busmap_df.loc[feeder_nodes, ["new_x", "new_y"]] = (
                        new_coordinates
                    )

        if mode != "aggregate_to_main_feeder":
            # Backmap
            # The other nodes point to their main feeder node, whose row is final.
            if not_main_nodes:
                partial_busmap_df.loc[not_main_nodes] = partial_busmap_df.loc[
                    partial_busmap_df.loc[not_main_nodes, "new_bus"]
                ].values

        if str(grid).split("_")[0] == "MVGrid":
            partial_busmap_df = rename_virtual_buses(
                partial_busmap_df, transformer_node
            )

        busmap_df_list.append(partial_busmap_df)

    busmap_df = pd.concat(busmap_df_list) if busmap_df_list else pd.DataFrame()
    if mode != "aggregate_to_main_feeder":
        busmap_df = _transform_coordinates(
            busmap_df, coor_transform_back, "new_x", "new_y"
        )

    return busmap_df


def make_busmap(
    edisgo_obj: EDisGo,
    mode: str = "kmeansdijkstra",
    cluster_area: str = "feeder",
    reduction_factor: float = 0.25,
    reduction_factor_not_focused: bool | float = False,
    grid: None | Grid = None,
) -> DataFrame:
    """
    Determines which busses are clustered.

    The information on which original busses are clustered to which new busses is
    given in the so-called busmap dataframe. The busmap can be used with the function
    :func:`~apply_busmap` to perform a spatial complexity reduction.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object for which the busmap is created.
    mode : str
        Clustering method to use.
        See parameter `mode` in function :attr:`~.EDisGo.spatial_complexity_reduction`
        for more information.
    cluster_area : str
        The cluster area is the area the different clustering methods are applied to.
        Possible options are 'grid', 'feeder' or 'main_feeder'. Default: "feeder".
    reduction_factor : float
        Factor to reduce number of nodes by. Must be between 0 and 1. Default: 0.25.
    reduction_factor_not_focused : bool or float
        If False, uses the same reduction factor for all cluster areas. If between 0
        and 1, this sets the reduction factor for buses not of interest (these are buses
        without voltage or overloading issues, that are determined through a worst case
        power flow analysis). When selecting 0, the nodes of the clustering area are
        aggregated to the transformer bus. This parameter is only used when parameter
        `cluster_area` is set to 'feeder' or 'main_feeder'.
        Default: False.
    grid : str or None
        If None, busmap is created for all grids, else only for the selected grid.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Busmap which maps the old bus names to the new bus names with new coordinates.
        Columns are "new_bus" with new bus name, "new_x" with new x-coordinate and
        "new_y" with new y-coordinate. Index of the dataframe holds bus names of
        original buses as in buses_df.

    References
    ----------
    In parts based on `PyPSA spatial complexity reduction <https://docs.pypsa.org/
    v0.35.1/examples/spatial-clustering.html>`_.

    """

    # Check for false input.
    if mode == "aggregate_to_main_feeder":
        pass  # Aggregate to the main feeder
    elif not 0 < reduction_factor < 1.0:
        raise ValueError("Reduction factor must be between 0 and 1.")

    modes = [
        "aggregate_to_main_feeder",
        "kmeans",
        "kmeansdijkstra",
        "equidistant_nodes",
    ]
    if mode not in modes:
        raise ValueError(f"Invalid input for parameter 'mode'. Must be one of {modes}.")
    if (reduction_factor_not_focused is not False) and not (
        0 <= reduction_factor_not_focused < 1.0
    ):
        raise ValueError(
            "Invalid input for parameter 'reduction_factor_not_focused'. Should be"
            "'False' or between 0 and 1."
        )

    if cluster_area == "grid":
        busmap_df = make_busmap_grid(
            edisgo_obj,
            mode=mode,
            grid=grid,
            reduction_factor=reduction_factor,
        )
    elif cluster_area == "feeder":
        busmap_df = make_busmap_feeders(
            edisgo_obj,
            grid=grid,
            mode=mode,
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=reduction_factor_not_focused,
        )
    elif cluster_area == "main_feeder":
        busmap_df = make_busmap_main_feeders(
            edisgo_obj,
            grid=grid,
            mode=mode,
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=reduction_factor_not_focused,
        )
    else:
        raise ValueError(
            "Invalid input for parameter 'cluster_area'. Must be one of 'grid', "
            "'feeder' or 'main_feeder'."
        )

    return busmap_df


def apply_busmap(
    edisgo_obj: EDisGo,
    busmap_df: DataFrame,
    line_naming_convention: str = "standard_lines",
    aggregation_mode: bool = False,
    load_aggregation_mode: str = "sector",
    generator_aggregation_mode: str = "type",
) -> DataFrame:
    """
    Function to reduce the EDisGo object with a previously generated busmap.

    Warning: After reduction, 'in_building' of all buses is set to False.
    Also, the method only works if all buses have x and y coordinates. If this is not
    the case, you can use the function
    :func:`~.tools.pseudo_coordinates.make_pseudo_coordinates` to set coordinates for
    all buses.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object to reduce.
    busmap_df : :pandas:`pandas.DataFrame<DataFrame>`
        Busmap holding the information which nodes are merged together.
    line_naming_convention : str
        Determines how to set "type_info" and "kind" in case two or more lines are
        aggregated. Possible options are "standard_lines" or "combined_name".
        If "standard_lines" is selected, the values of the standard line of the
        respective voltage level are used to set "type_info" and "kind".
        If "combined_name" is selected, "type_info" and "kind" contain the
        concatenated values of the merged lines. The electrical parameters do not
        depend on this setting: r and x are computed from the per-km values of the
        line types of the aggregated lines, treated as parallel systems, and the
        susceptance b is the new length times the sum of b per km of the aggregated
        lines.
        Default: "standard_lines".
    aggregation_mode : bool
        Specifies, whether to aggregate loads and generators at the same bus or not.
        If True, loads and generators at the same bus are aggregated
        according to their selected modes (see parameters `load_aggregation_mode` and
        `generator_aggregation_mode`). Default: False.
    load_aggregation_mode : str
        Specifies, how to aggregate loads at the same bus, in case parameter
        `aggregation_mode` is set to True. Possible options are "bus" or "sector".
        If "bus" is chosen, loads are aggregated per bus. When "sector" is chosen,
        loads are aggregated by bus, type and sector. Default: "sector".
    generator_aggregation_mode : str
        Specifies, how to aggregate generators at the same bus, in case parameter
        `aggregation_mode` is set to True. Possible options are "bus" or "type".
        If "bus" is chosen, generators are aggregated per bus. When "type" is chosen,
        generators are aggregated by bus and type.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Linemap which maps the old line names (in the index of the dataframe) to the
        new line names (in column "new_line_name").

    References
    ----------
    In parts based on `PyPSA spatial complexity reduction <https://docs.pypsa.org
    /v0.35.1/examples/spatial-clustering.html>`_.

    """

    def get_ordered_lines_df(lines_df):
        """Order lines so that a grouping is possible."""
        order = lines_df.bus0 < lines_df.bus1
        lines_df_p = lines_df[order]
        lines_df_n = lines_df[~order].rename(columns={"bus0": "bus1", "bus1": "bus0"})
        lines_df = pd.concat([lines_df_p, lines_df_n], sort=True)
        return lines_df

    def aggregate_lines(lines_df):
        """
        Aggregate all lines between the same pair of buses to one line.

        The length is the distance between the new buses times the branch detour
        factor, at least 1 m. x and r follow from the per-km values of the aggregated
        line types, which are treated as parallel systems. b is the length times the
        sum of b per km of the aggregated lines, so that the susceptances of parallel
        lines add up; lines without length or b do not contribute. s_nom and
        num_parallel are summed.
        """
        missing_type = lines_df.type_info.isna()
        if missing_type.any():
            raise ValueError(
                f"Line(s) {lines_df.index[missing_type].tolist()} have no "
                f"type_info. The spatial complexity reduction needs the line type "
                f"to compute r and x of the aggregated lines."
            )
        detour_factor = edisgo_obj.config["grid_connection"]["branch_detour_factor"]
        bus_x, bus_y = coor_transform.transform(
            buses_df.x.to_numpy(), buses_df.y.to_numpy()
        )
        coordinates = dict(zip(buses_df.index, zip(bus_x, bus_y)))
        v_nom_of_bus = buses_df.v_nom.to_dict()
        is_mv_bus = buses_df.lv_grid_id.isna().to_dict()

        per_km_values = {}

        def per_km(line_type, v_nom, column):
            key = (line_type, v_nom, column)
            if key not in per_km_values:
                try:
                    value = line_data_df.loc[line_data_df.U_n.isin([v_nom])].loc[
                        line_type, column
                    ]
                except KeyError:
                    value = line_data_df.loc[line_type, column]
                    logger.error(
                        f"Line type {line_type} not in voltage level {v_nom} kV."
                    )
                per_km_values[key] = value
            return per_km_values[key]

        names = lines_df.index.to_numpy()
        s_nom = lines_df["s_nom"].to_numpy(dtype=float)
        type_infos = lines_df["type_info"].to_numpy()
        kinds = lines_df["kind"].to_numpy()
        num_parallel = lines_df["num_parallel"].to_numpy()
        parallel_systems = lines_df["num_parallel"].astype(float).to_numpy()
        old_length = lines_df["length"].to_numpy(dtype=float)
        if "b" in lines_df.columns:
            old_b = lines_df["b"].to_numpy(dtype=float)
        else:
            old_b = np.zeros(len(lines_df))
        # Susceptance per km of each line. Lines without length or susceptance
        # do not contribute.
        with np.errstate(divide="ignore", invalid="ignore"):
            b_per_km = np.where(old_length > 0, old_b / old_length, 0.0)
        b_per_km = np.nan_to_num(b_per_km, nan=0.0, posinf=0.0, neginf=0.0)

        groups = lines_df.groupby(by=["bus0", "bus1"]).indices
        rows = []
        for bus0, bus1 in sorted(groups):
            idx = groups[(bus0, bus1)]
            v_nom = v_nom_of_bus[bus0]

            length = (
                math.dist(coordinates[bus0], coordinates[bus1]) / 1000 * detour_factor
            )
            if length == 0:
                length = 0.001
                logger.warning(
                    f"Length of line between {bus0} and {bus1} cannot be 0 m and is "
                    f"therefore set to 1 m."
                )
            if length < 0.001:
                logger.debug(
                    f"Length of line between {bus0} and {bus1} is smaller than 1 m. "
                    f"To avoid stability issues in the power flow analysis it is set "
                    f"to 1 m."
                )

            # Get type of the line to get the according standard line for the
            # voltage level
            if is_mv_bus[bus0]:
                type_line = f"mv_line_{int(v_nom)}kv"
            else:
                type_line = "lv_line"

            if len(idx) > 1:
                if line_naming_convention == "combined_name":
                    type_info = "Merged: "
                    for value in type_infos[idx]:
                        type_info = type_info + str(value) + " "
                    kind = "Combined: "
                    for value in kinds[idx]:
                        kind = kind + str(value) + " "
                elif line_naming_convention == "standard_lines":
                    type_info = edisgo_obj.config["grid_expansion_standard_equipment"][
                        type_line
                    ]
                    kind = kinds[idx[0]]
            else:
                type_info = type_infos[idx[0]]
                kind = kinds[idx[0]]

            # The per-km values describe one system of the given type, so a line
            # standing for ``num_parallel`` systems contributes that many of them
            # to the parallel sum.
            x_sum = 0
            for line_type, n_parallel in zip(type_infos[idx], parallel_systems[idx]):
                x_sum = x_sum + n_parallel / per_km(line_type, v_nom, "L_per_km")
            x_sum = 1 / x_sum
            x = length * 2 * math.pi * 50 * x_sum / 1000

            r_sum = 0
            for line_type, n_parallel in zip(type_infos[idx], parallel_systems[idx]):
                r_sum = r_sum + n_parallel / per_km(line_type, v_nom, "R_per_km")
            r_sum = 1 / r_sum
            r = length * r_sum

            rows.append(
                {
                    "length": length,
                    "bus0": bus0,
                    "bus1": bus1,
                    "x": x,
                    "r": r,
                    # The susceptances of parallel lines add up.
                    "b": length * b_per_km[idx].sum(),
                    # Summed one after the other in a fixed order, so that s_nom
                    # does not depend on how numpy groups the additions.
                    "s_nom": sum(v for v in s_nom[idx].tolist() if not math.isnan(v)),
                    "num_parallel": int(np.nansum(num_parallel[idx])),
                    "type_info": type_info,
                    "kind": kind,
                    "old_line_name": names[idx].tolist(),
                }
            )
        columns = list(lines_df.columns)
        columns += [c for c in ["b", "old_line_name"] if c not in columns]
        aggregated_lines_df = pd.DataFrame(rows, columns=columns)
        aggregated_lines_df.index = (
            "Line_" + aggregated_lines_df["bus0"] + "_to_" + aggregated_lines_df["bus1"]
        ).to_numpy()
        return aggregated_lines_df

    def aggregate_loads_df(df):
        series = pd.Series(index=df.columns, dtype="object")
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_set"] = df.loc[:, "p_set"].sum()
        series.loc["annual_consumption"] = df.loc[:, "annual_consumption"].sum()
        if load_aggregation_mode == "sector":
            series.loc["type"] = df.loc[:, "type"].values[0]
            series.loc["sector"] = df.loc[:, "sector"].values[0]
        elif load_aggregation_mode == "bus":
            series.loc["type"] = "aggregated"
            series.loc["sector"] = "aggregated"
        series.loc["old_name"] = df.index.tolist()
        return series

    def aggregate_generators_df(df):
        series = pd.Series(index=df.columns, dtype="object")
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_nom"] = df.loc[:, "p_nom"].sum()
        series.loc["control"] = df.loc[:, "control"].values[0]
        series.loc["subtype"] = "aggregated"
        series.loc["old_name"] = df.index.tolist()
        if generator_aggregation_mode == "bus":
            series.loc["type"] = "aggregated"
            series.loc["weather_cell_id"] = "aggregated"
        elif generator_aggregation_mode == "type":
            series.loc["type"] = df.loc[:, "type"].values[0]
            series.loc["weather_cell_id"] = df.loc[:, "weather_cell_id"].values[0]
        return series

    def extract_weather_cell_id(series):
        if pd.isna(series):
            series = "NaN"
        else:
            series = str(int(series))
        return series

    def aggregate_timeseries(
        df: DataFrame, edisgo_obj: EDisGo, timeseries_to_aggregate: list
    ):
        # comp = component
        # aggregate load timeseries
        name_map_df = df.loc[:, "old_name"].to_dict()
        name_map = {}
        for i in range(0, len(name_map_df.keys())):
            for j in range(0, len(list(name_map_df.values())[i])):
                name_map[list(name_map_df.values())[i][j]] = list(name_map_df.keys())[i]

        rename_index = []
        for timeseries_name in timeseries_to_aggregate:
            timeseries = getattr(edisgo_obj.timeseries, timeseries_name).T

            if len(rename_index) == 0:
                new_index = []
                for i in range(0, timeseries.shape[0]):
                    new_load_name = name_map[timeseries.index[i]]
                    new_index.append(new_load_name)

                old_index = timeseries.index.tolist()
                rename_index = dict(zip(old_index, new_index))

            timeseries = timeseries.rename(index=rename_index)
            timeseries = timeseries.groupby(level=0).sum().T

            setattr(edisgo_obj.timeseries, timeseries_name, timeseries)

    # Copy dataframes from edisgo object
    buses_df = edisgo_obj.topology.buses_df.copy()
    lines_df = edisgo_obj.topology.lines_df.copy()
    loads_df = edisgo_obj.topology.loads_df.copy()
    generators_df = edisgo_obj.topology.generators_df.copy()
    storage_units_df = edisgo_obj.topology.storage_units_df.copy()
    transformers_df = edisgo_obj.topology.transformers_df.copy()
    switches_df = edisgo_obj.topology.switches_df.copy()

    slack_bus = edisgo_obj.topology.transformers_hvmv_df.bus1.iloc[0]

    new_bus = busmap_df["new_bus"]

    def map_buses(buses: pd.Series) -> np.ndarray:
        return new_bus.loc[buses.to_numpy()].to_numpy()

    # Manipulate buses_df
    buses_df["bus"] = map_buses(buses_df.index.to_series())
    buses_df["x"] = busmap_df.loc[buses_df.index, "new_x"].to_numpy()
    buses_df["y"] = busmap_df.loc[buses_df.index, "new_y"].to_numpy()
    buses_df = buses_df.groupby(by=["bus"], dropna=False, as_index=False).first()
    buses_df["in_building"] = False
    buses_df = buses_df.set_index("bus")

    # Manipulate lines_df
    if not lines_df.empty:
        # Get one dataframe with all data of the line types
        line_data_df = pd.concat(
            [
                edisgo_obj.topology.equipment_data["mv_overhead_lines"],
                edisgo_obj.topology.equipment_data["mv_cables"],
                edisgo_obj.topology.equipment_data["lv_cables"],
            ]
        )

        lines_df["bus0"] = map_buses(lines_df.bus0)
        lines_df["bus1"] = map_buses(lines_df.bus1)
        # Drop lines which connect the same bus, are connected to the virtual
        # slack bus or short the circuit breaker.
        bus0 = lines_df.bus0.str.removeprefix("virtual_")
        bus1 = lines_df.bus1.str.removeprefix("virtual_")
        is_virtual_0 = lines_df.bus0.str.startswith("virtual_")
        is_virtual_1 = lines_df.bus1.str.startswith("virtual_")
        drop = (
            (lines_df.bus0 == lines_df.bus1)
            | (is_virtual_0 & (bus0 == slack_bus))
            | (is_virtual_1 & (bus1 == slack_bus))
            | (bus0 == bus1)
        )
        logger.debug(
            f"Drop {drop.sum()} lines that connect buses of the same new bus, the "
            f"virtual slack bus or both sides of a circuit breaker."
        )
        lines_df = lines_df.loc[~drop]
        lines_df = get_ordered_lines_df(lines_df)
        lines_df = aggregate_lines(lines_df)

    # Manipulate loads_df
    if not loads_df.empty:
        loads_df["bus"] = map_buses(loads_df.bus)

        if aggregation_mode:
            if load_aggregation_mode == "sector":
                loads_df = loads_df.groupby(by=["bus", "type", "sector"]).apply(
                    aggregate_loads_df
                )
                loads_df.index = (
                    "Load_"
                    + loads_df.loc[:, "bus"]
                    + "_"
                    + loads_df.loc[:, "type"]
                    + "_"
                    + loads_df.loc[:, "sector"]
                )
            elif load_aggregation_mode == "bus":
                loads_df = loads_df.groupby(by=["bus"]).apply(aggregate_loads_df)
                loads_df.index = "Load_" + loads_df.loc[:, "bus"]

            loads_df.index.name = "name"

            aggregate_timeseries(
                loads_df, edisgo_obj, ["loads_active_power", "loads_reactive_power"]
            )

    # Manipulate generators_df
    if not generators_df.empty:
        generators_df = generators_df.loc[
            generators_df.loc[:, "bus"].isin(busmap_df.index), :
        ]
        generators_df = generators_df.copy()
        generators_df["bus"] = map_buses(generators_df.bus)

        if aggregation_mode:
            if generator_aggregation_mode == "bus":
                generators_df = generators_df.groupby("bus").apply(
                    aggregate_generators_df
                )
                generators_df.index = "Generator_" + generators_df.loc[:, "bus"]
            elif generator_aggregation_mode == "type":
                generators_df = generators_df.groupby(
                    by=["bus", "type", "weather_cell_id"], dropna=False
                ).apply(aggregate_generators_df)
                generators_df.index = (
                    "Generator_"
                    + generators_df.loc[:, "bus"].values
                    + "_"
                    + generators_df.loc[:, "type"].values
                    + "_weather_cell_id_"
                    + generators_df.loc[:, "weather_cell_id"]
                    .apply(extract_weather_cell_id)
                    .values
                )

            aggregate_timeseries(
                generators_df,
                edisgo_obj,
                ["generators_active_power", "generators_reactive_power"],
            )

    # Manipulate storage_units_df
    if not storage_units_df.empty:
        storage_units_df["bus"] = map_buses(storage_units_df.bus)

    # Manipulate transformers_df
    transformers_df["bus0"] = map_buses(transformers_df.bus0)
    transformers_df["bus1"] = map_buses(transformers_df.bus1)

    # Manipulate switches_df
    if not switches_df.empty:
        # drop switches unused switches
        switches_to_drop = []
        for index, new_bus in zip(busmap_df.index, busmap_df.new_bus):
            if (index.split("_")[0] == "virtual") and (
                new_bus.split("_")[0] != "virtual"
            ):
                switches_to_drop.append(
                    switches_df.loc[switches_df.bus_open == index].index[0]
                )

        if len(switches_to_drop) > 0:
            logger.warning(f"Drop unused switches: {switches_to_drop}")
            switches_df = switches_df.drop(switches_to_drop)

        # apply busmap
        for index, bus_open, bus_closed in zip(
            switches_df.index, switches_df.bus_open, switches_df.bus_closed
        ):
            switches_df.loc[index, "bus_closed"] = busmap_df.loc[bus_closed, "new_bus"]
            switches_df.loc[index, "bus_open"] = busmap_df.loc[bus_open, "new_bus"]

        # update the branches in switches_df
        for index, bus_open, bus_closed in zip(
            switches_df.index, switches_df.bus_open, switches_df.bus_closed
        ):
            if lines_df.bus0.isin([bus_open]).any():
                switches_df.loc[index, "branch"] = lines_df.loc[
                    lines_df.bus0.isin([bus_open])
                ].index[0]
            if lines_df.bus1.isin([bus_open]).any():
                switches_df.loc[index, "branch"] = lines_df.loc[
                    lines_df.bus1.isin([bus_open])
                ].index[0]

        # remove duplicate switches_df
        switches_df = switches_df.groupby(
            by=["bus_closed", "bus_open", "branch"], as_index=False
        ).first()
        for index in switches_df.index.to_list():
            switches_df.loc[index, "name"] = "circuit_breaker_" + str(index)
        if not switches_df.empty:
            switches_df = switches_df.set_index("name")

    # Write dataframes back to the edisgo object.
    edisgo_obj.topology.buses_df = buses_df
    edisgo_obj.topology.lines_df = lines_df
    edisgo_obj.topology.loads_df = loads_df
    edisgo_obj.topology.generators_df = generators_df
    edisgo_obj.topology.storage_units_df = storage_units_df
    edisgo_obj.topology.transformers_df = transformers_df
    edisgo_obj.topology.switches_df = switches_df

    # Make linemap_df
    if lines_df.empty:
        linemap_df = pd.DataFrame()
    else:
        old_line_names = lines_df.old_line_name.explode()
        linemap_df = pd.DataFrame(
            {"new_line_name": old_line_names.index.to_numpy()},
            index=old_line_names.to_numpy(),
        )

    return linemap_df


def spatial_complexity_reduction(
    edisgo_obj: EDisGo,
    mode: str = "kmeansdijkstra",
    cluster_area: str = "feeder",
    reduction_factor: float = 0.25,
    reduction_factor_not_focused: bool | float = False,
    apply_pseudo_coordinates: bool = True,
    threads: int = 1,
    **kwargs,
) -> tuple[DataFrame, DataFrame]:
    """
    Reduces the number of busses and lines by applying a spatial clustering.

    See function :attr:`~.EDisGo.spatial_complexity_reduction` for more information.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object to apply spatial complexity reduction to.
    mode : str
        Clustering method to use.
        See parameter `mode` in function :attr:`~.EDisGo.spatial_complexity_reduction`
        for more information.
    cluster_area : str
        The cluster area is the area the different clustering methods are applied to.
        See parameter `cluster_area` in function
        :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
    reduction_factor : float
        Factor to reduce number of nodes by. Must be between 0 and 1. Default: 0.25.
    reduction_factor_not_focused : bool or float
        If False, uses the same reduction factor for all cluster areas. If between 0
        and 1, this sets the reduction factor for buses not of interest. See parameter
        `reduction_factor_not_focused` in function
        :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
    apply_pseudo_coordinates : bool
        If True pseudo coordinates are applied. The spatial complexity reduction method
        is only tested with pseudo coordinates. Default: True.
    threads : int
        Maximum number of threads the numerical libraries (BLAS and OpenMP, e.g. used
        by the k-means clustering of scikit-learn) may use. 0 lets them use all
        cores. More threads hardly speed up the reduction, as most of its run time
        is spent in single-threaded code. Default: 1.

    Other Parameters
    -----------------
    line_naming_convention : str
        Determines how to set "type_info" and "kind" in case two or more lines are
        aggregated. See parameter `line_naming_convention` in function
        :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
    aggregation_mode : bool
        Specifies, whether to aggregate loads and generators at the same bus or not.
        See parameter `aggregation_mode` in function
        :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
    load_aggregation_mode : str
        Specifies, how to aggregate loads at the same bus, in case parameter
        `aggregation_mode` is set to True. See parameter `load_aggregation_mode` in
        function :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
    generator_aggregation_mode : str
        Specifies, how to aggregate generators at the same bus, in case parameter
        `aggregation_mode` is set to True. See parameter `generator_aggregation_mode` in
        function :attr:`~.EDisGo.spatial_complexity_reduction` for more information.
    mv_pseudo_coordinates : bool, optional
        If True pseudo coordinates are also generated for MV grid.
        Default: False.

    Returns
    -------
    tuple(:pandas:`pandas.DataFrame<DataFrame>`, :pandas:`pandas.DataFrame<DataFrame>`)
        Returns busmap and linemap dataframes.
        The busmap maps the original busses to the new busses with new coordinates.
        Columns are "new_bus" with new bus name, "new_x" with new x-coordinate and
        "new_y" with new y-coordinate. Index of the dataframe holds bus names of
        original buses as in buses_df.
        The linemap maps the original line names (in the index of the dataframe) to the
        new line names (in column "new_line_name").

    """

    # Without a limit, scikit-learn (OpenMP) and numpy/scipy (BLAS) use all cores
    # of the machine. threadpool_limits(None) leaves the thread pools unchanged.
    with threadpool_limits(limits=threads or None):
        if apply_pseudo_coordinates:
            make_pseudo_coordinates(
                edisgo_obj, mv_coordinates=kwargs.pop("mv_pseudo_coordinates", False)
            )

        busmap_df = make_busmap(
            edisgo_obj,
            mode=mode,
            cluster_area=cluster_area,
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=reduction_factor_not_focused,
        )
        linemap_df = apply_busmap(edisgo_obj, busmap_df, **kwargs)

    return busmap_df, linemap_df


def apply_reduced_results_to_full_grid(
    full_grid: EDisGo,
    reduced_grid: EDisGo,
    *,
    flexible_cps: list | None = None,
    flexible_hps: list | None = None,
    flexible_loads: list | None = None,
    flexible_storage_units: list | None = None,
) -> EDisGo:
    """
    Write the OPF's active-power results from a spatially-reduced grid back
    onto the full grid.

    Counterpart to :func:`spatial_complexity_reduction`: where that function
    shrinks a grid for a faster OPF, this function maps the OPF's active-power
    results back onto the pre-reduction grid so reinforcement can run on the
    full topology. It writes back the given flexible charging points, heat
    pumps, DSM loads and storage units, and always all generators: with
    ``opf_version`` 3 and 4 the OPF curtails solar and wind generators (see
    :func:`~.io.powermodels_io.from_powermodels`). Generators the OPF did not
    change get their own series back, because a merged generator's series is
    the sum of its members' series. Inflexible loads are untouched, as the
    OPF does not change them.

    ``full_grid`` and ``reduced_grid`` are matched by name for storage units
    (never aggregated by :func:`spatial_complexity_reduction`, so their names
    are unchanged) and, for the other flexibility types and the generators,
    by the ``old_name`` column that
    :func:`spatial_complexity_reduction` writes onto
    ``reduced_grid.topology.loads_df`` / ``generators_df`` when
    ``aggregation_mode=True``. A member listed in ``old_name`` is a component
    whose active-power series was merged into one representative row; when
    ``aggregation_mode=False`` (or a given member was not merged),
    ``old_name`` is absent and the member's own name is used directly — i.e.
    a plain by-name write-back.

    For a merged representative, the representative's optimized series is
    disaggregated onto its ``old_name`` members **per time step**, weighted by
    a pre-OPF value of each member (a known input, never the optimized
    result):

    * charging points — ``upper_power(t)`` from
      ``electromobility.flexibility_bands`` (a charging point with no
      connected vehicle has ``upper_power(t) == 0``, so it receives none of
      the representative's dispatch that time step);
    * heat pumps — ``min(heat_demand(t) / cop(t), p_set)``, i.e. the
      electrical-equivalent heat demand capped at the heat pump's own rated
      power, mirroring how a charging point's ``upper_power(t)`` is already a
      capped bound rather than raw uncapped demand;
    * DSM loads — ``p_max(t)`` from :attr:`~.network.dsm.DSM.p_max`;
    * generators — the member's uncurtailed feed-in ``p(t)``, which
      ``full_grid`` still holds, so all members of a representative are
      curtailed by the same fraction.

    Weights always sum back to the representative's value exactly at every
    time step; a time step where every member's weight is 0 falls back to an
    equal split.

    Limitations of the generator write-back:

    * The split ignores where the members sit within their cluster, so it is
      not necessarily the curtailment an OPF on the full grid would choose.
    * With ``generator_aggregation_mode="bus"`` merged generators get type
      ``"aggregated"``. The OPF only curtails generators of type ``"solar"``
      and ``"wind"``, so the feed-in of these generators comes back
      unchanged.
    * Solver tolerance can leave a curtailed generator slightly below zero
      (about -1e-6 MW).

    Reactive power is not read from ``reduced_grid``. After writing active
    power, this function calls
    :meth:`~.EDisGo.set_time_series_reactive_power_control` on ``full_grid``
    with its defaults, mirroring how the OPF itself derives reactive power
    for the components it just optimized (see
    :func:`~.io.powermodels_io.from_powermodels`) — reactive power is always
    a function of whatever active power is currently set, regardless of
    whether that active power came from a default, worst case, or the OPF.

    Parameters
    ----------
    full_grid : :class:`~.EDisGo`
        The pre-reduction EDisGo instance to write dispatch onto, modified in
        place. Must contain every component named in ``flexible_cps`` /
        ``flexible_hps`` / ``flexible_loads`` / ``flexible_storage_units``,
        every generator of ``reduced_grid`` and (for merged components) every
        name listed in ``reduced_grid``'s ``old_name`` columns. It must still
        hold the uncurtailed generator feed-in, i.e. be a copy taken before
        the OPF.
    reduced_grid : :class:`~.EDisGo`
        The spatially-reduced EDisGo instance the OPF ran on. Supplies the
        optimized active-power series and, if aggregated, the ``old_name``
        provenance.
    flexible_cps : list of str, optional
        Names of flexible charging points in ``reduced_grid`` to map back.
    flexible_hps : list of str, optional
        Names of flexible heat-pump loads in ``reduced_grid`` to map back.
    flexible_loads : list of str, optional
        Names of flexible DSM loads in ``reduced_grid`` to map back.
    flexible_storage_units : list of str, optional
        Names of flexible storage units in ``reduced_grid`` to map back.

    Returns
    -------
    :class:`~.EDisGo`
        ``full_grid``, with active power written for the given flexible
        components and all generators, and reactive power recomputed.

    """
    # NOTE: "x or []" is unsafe here - callers may pass a numpy array (e.g.
    # task_optimize derives flexible_loads as
    # edisgo.dsm.p_min.columns.values), and "array or []" raises
    # ValueError ("truth value of an array... is ambiguous") for any array
    # with more than one element. "is None" is the correct emptiness check
    # for an optional list-like argument.
    flexible_cps = list(flexible_cps) if flexible_cps is not None else []
    flexible_hps = list(flexible_hps) if flexible_hps is not None else []
    flexible_loads = list(flexible_loads) if flexible_loads is not None else []
    flexible_storage_units = (
        list(flexible_storage_units) if flexible_storage_units is not None else []
    )

    def _require_full_timeindex(envelope: DataFrame, envelope_name: str) -> None:
        """Raise a clear error if ``envelope`` doesn't cover the full grid's
        active time index, instead of a bare ``KeyError`` deep inside a
        ``.loc`` lookup.

        This can only happen if ``full_grid``'s flexibility-band/DSM/heat-pump
        attributes were never trimmed to the same time index as
        ``full_grid.timeseries.timeindex`` - i.e. if the pre-reduction stash
        was taken before the run's time index was finalized.
        """
        ti = full_grid.timeseries.timeindex
        missing = ti.difference(envelope.index)
        if len(missing) > 0:
            raise ValueError(
                f"apply_reduced_results_to_full_grid: full_grid's "
                f"{envelope_name} does not cover {len(missing)} of "
                f"full_grid.timeseries.timeindex's time steps (e.g. "
                f"{missing[0]!r}). This usually means the full-grid stash "
                f"was taken before the time index was finalized - run "
                f"the time-step selection (set_timeindex or "
                f"select_critical_timesteps) before spatial_reduce."
            )

    def _old_name_map(components_df: DataFrame, names: list) -> dict:
        """Map each representative name in ``names`` to its member names.

        A name absent from ``old_name`` (not merged, or
        ``aggregation_mode=False``) maps to itself.
        """
        name_map = {}
        for name in names:
            old_name = (
                components_df.at[name, "old_name"]
                if "old_name" in components_df
                else None
            )
            name_map[name] = old_name if isinstance(old_name, list) else [name]
        return name_map

    def _write_by_name(active_power: DataFrame, names: list, target: DataFrame) -> None:
        ti = full_grid.timeseries.timeindex
        target.loc[ti, names] = active_power.loc[ti, names].values

    def _disaggregate(
        active_power: DataFrame,
        name_map: dict,
        weights_source: DataFrame,
        target: DataFrame,
    ) -> None:
        """Split each representative's series onto its members per time step.

        ``weights_source`` holds each member's pre-OPF weight (columns =
        member names, index = time index): a flexibility band, the capped
        heat demand or the uncurtailed feed-in. A member missing from it gets
        weight 0. A time step where all members' weights are 0 is split
        equally.
        """
        ti = full_grid.timeseries.timeindex
        for representative, members in name_map.items():
            if len(members) == 1 and members[0] == representative:
                target.loc[ti, representative] = active_power.loc[ti, representative]
                continue
            weights = pd.DataFrame(index=ti, columns=members, dtype=float)
            for member in members:
                weights[member] = (
                    weights_source.loc[ti, member]
                    if member in weights_source.columns
                    else 0.0
                )
            weight_sum = weights.sum(axis="columns")
            no_weight = weight_sum == 0
            shares = weights.div(weight_sum.replace(0, np.nan), axis="index")
            shares.loc[no_weight, :] = 1.0 / len(members)
            representative_power = active_power.loc[ti, representative]
            for member in members:
                target.loc[ti, member] = shares[member] * representative_power

    reduced_loads_df = reduced_grid.topology.loads_df
    full_loads_df = full_grid.topology.loads_df

    # Always routed through _disaggregate (never the by-name fast path): under
    # aggregation_mode=True, spatial_complexity_reduction renames EVERY group's
    # representative row, including singleton groups (a bus with exactly one
    # flexible load of a given type/sector) - so a representative's own name
    # can differ from its single old_name member's name. _disaggregate already
    # handles that case correctly (a singleton's one weight, whether zero or
    # not, always resolves its share to the representative's full value), so
    # there is no correct case left for a by-name fast path to shortcut.
    if flexible_cps:
        name_map = _old_name_map(reduced_loads_df, flexible_cps)
        envelope = reduced_grid.electromobility.flexibility_bands["upper_power"]
        _require_full_timeindex(envelope, "electromobility.flexibility_bands")
        _disaggregate(
            reduced_grid.timeseries.loads_active_power,
            name_map,
            envelope,
            full_grid.timeseries._loads_active_power,
        )

    if flexible_hps:
        name_map = _old_name_map(reduced_loads_df, flexible_hps)
        members_flat = [m for members in name_map.values() for m in members]
        heat_demand = full_grid.heat_pump.heat_demand_df[members_flat]
        cop = full_grid.heat_pump.cop_df[members_flat]
        p_set = full_loads_df.p_set[members_flat]
        envelope = (heat_demand / cop).clip(upper=p_set, axis="columns")
        _require_full_timeindex(envelope, "heat_pump.heat_demand_df/cop_df")
        _disaggregate(
            reduced_grid.timeseries.loads_active_power,
            name_map,
            envelope,
            full_grid.timeseries._loads_active_power,
        )

    if flexible_loads:
        name_map = _old_name_map(reduced_loads_df, flexible_loads)
        _require_full_timeindex(full_grid.dsm.p_max, "dsm.p_max")
        _disaggregate(
            reduced_grid.timeseries.loads_active_power,
            name_map,
            full_grid.dsm.p_max,
            full_grid.timeseries._loads_active_power,
        )

    if flexible_storage_units:
        # Storage units are never aggregated by spatial_complexity_reduction
        # (only bus-relabeled), so this is always a plain by-name write-back.
        _write_by_name(
            reduced_grid.timeseries.storage_units_active_power,
            flexible_storage_units,
            full_grid.timeseries._storage_units_active_power,
        )

    gens_df = reduced_grid.topology.generators_df
    if not gens_df.empty:
        name_map = _old_name_map(gens_df, gens_df.index.tolist())
        members_flat = [m for members in name_map.values() for m in members]
        # weights: the uncurtailed feed-in that full_grid still holds (the
        # list selection returns a copy, so the write-back does not alter it)
        _disaggregate(
            reduced_grid.timeseries.generators_active_power,
            name_map,
            full_grid.timeseries.generators_active_power[members_flat],
            full_grid.timeseries._generators_active_power,
        )

    full_grid.set_time_series_reactive_power_control()

    return full_grid


def compare_voltage(
    edisgo_unreduced: EDisGo,
    edisgo_reduced: EDisGo,
    busmap_df: DataFrame,
    timestep: str | pd.Timestamp,
) -> tuple[DataFrame, float]:
    """
    Compares the voltages per node between the unreduced and the reduced EDisGo object.

    The voltage difference for each node in p.u. as well as the root-mean-square error
    is returned. For the mapping of nodes in the unreduced and reduced network the
    busmap is used.
    The calculation is performed for one timestep or the minimum or maximum values of
    the node voltages.

    Parameters
    ----------
    edisgo_unreduced : :class:`~.EDisGo`
        Unreduced EDisGo object.
    edisgo_reduced : :class:`~.EDisGo`
        Reduced EDisGo object.
    busmap_df : :pandas:`pandas.DataFrame<DataFrame>`
        Busmap for the mapping of nodes.
    timestep : str or :pandas:`pandas.Timestamp<Timestamp>`
        Timestep for which to compare the bus voltage. Can either be a certain time
        step or 'min' or 'max'.

    Returns
    -------
    (:pandas:`pandas.DataFrame<DataFrame>`, rms)
        Returns a tuple with the first entry being a DataFrame containing the node
        voltages as well as voltage differences and the second entry being the
        root-mean-square error.
        Columns of the DataFrame are "v_unreduced" with voltage in p.u. in unreduced
        EDisGo object, "v_reduced" with voltage in p.u. in reduced EDisGo object, and
        "v_diff" with voltage difference in p.u. between voltages in unreduced and
        reduced EDisGo object. Index of the DataFrame contains the bus names of buses in
        the unreduced EDisGo object.

    """
    if timestep == "min":
        logger.debug("Voltage mapping for the minium values.")
        v_root = edisgo_unreduced.results.v_res.min()
        v_reduced = edisgo_reduced.results.v_res.min()
    elif timestep == "max":
        logger.debug("Voltage mapping for the maximum values.")
        v_root = edisgo_unreduced.results.v_res.max()
        v_reduced = edisgo_reduced.results.v_res.max()
    else:
        logger.debug(f"Voltage mapping for timestep {timestep}.")
        v_root = edisgo_unreduced.results.v_res.loc[timestep]
        v_reduced = edisgo_reduced.results.v_res.loc[timestep]

    v_root.name = "v_unreduced"
    v_root = v_root.loc[busmap_df.index]

    voltages_df = v_root.to_frame()

    for index, row in voltages_df.iterrows():
        try:
            voltages_df.loc[index, "v_reduced"] = v_reduced.loc[
                busmap_df.loc[index, "new_bus"]
            ]
        except KeyError:
            voltages_df.loc[index, "v_reduced"] = v_reduced.loc[
                busmap_df.loc[index, "new_bus"].removeprefix("virtual_")
            ]
    voltages_df.loc[:, "v_diff"] = (
        voltages_df.loc[:, "v_unreduced"] - voltages_df.loc[:, "v_reduced"]
    )
    rms = np.sqrt(
        mean_squared_error(
            voltages_df.loc[:, "v_unreduced"], voltages_df.loc[:, "v_reduced"]
        )
    )
    logger.debug(
        "Root-mean-square error between voltages in unreduced and reduced "
        f"EDisGo object is: rms = {rms:.2%}"
    )
    return voltages_df, rms


def compare_apparent_power(
    edisgo_unreduced: EDisGo,
    edisgo_reduced: EDisGo,
    linemap_df: DataFrame,
    timestep: str,
) -> tuple[DataFrame, float]:
    """
    Compares the apparent power over each line between the unreduced and the reduced
    EDisGo object.

    The difference of apparent power over each line in MVA as well as the
    root-mean-square error is returned. For the mapping of lines in the unreduced and
    reduced network the linemap is used.
    The calculation is performed for one timestep or the minimum or maximum values of
    the node voltages.

    Parameters
    ----------
    edisgo_unreduced : :class:`~.EDisGo`
        Unreduced EDisGo object.
    edisgo_reduced : :class:`~.EDisGo`
        Reduced EDisGo object.
    linemap_df : :pandas:`pandas.DataFrame<DataFrame>`
        Linemap for the mapping.
    timestep : str or :pandas:`pandas.Timestamp<Timestamp>`
        Timestep for which to compare the apparent power. Can either be a certain time
        step or 'min' or 'max'.

    Returns
    -------
    (:pandas:`pandas.DataFrame<DataFrame>`, rms)
        Returns a tuple with the first entry being a DataFrame containing the apparent
        power as well as difference of apparent power for each line and the second entry
        being the root-mean-square error.
        Columns of the DataFrame are "s_unreduced" with apparent power in MVA in
        unreduced EDisGo object, "s_reduced" with apparent power in MVA in reduced
        EDisGo object, and "s_diff" with difference in apparent power in MVA between
        apparent power over line in unreduced and reduced EDisGo object. Index of the
        DataFrame contains the line names of lines in the unreduced EDisGo object.

    """
    if timestep == "min":
        logger.debug("Apparent power mapping for the minium values.")
        s_root = edisgo_unreduced.results.s_res.min()
        s_reduced = edisgo_reduced.results.s_res.min()
    elif timestep == "max":
        logger.debug("Apparent power mapping for the maximum values.")
        s_root = edisgo_unreduced.results.s_res.max()
        s_reduced = edisgo_reduced.results.s_res.max()
    else:
        logger.debug("fApparent power mapping for timestep {timestep}.")
        s_root = edisgo_unreduced.results.s_res.loc[timestep]
        s_reduced = edisgo_reduced.results.s_res.loc[timestep]

    s_root.name = "s_unreduced"
    s_root = s_root.loc[linemap_df.index]

    s_df = s_root.to_frame()

    for index, row in s_df.iterrows():
        s_df.loc[index, "s_reduced"] = s_reduced.loc[linemap_df.loc[index][0]]
    s_df.loc[:, "s_diff"] = s_df.loc[:, "s_unreduced"] - s_df.loc[:, "s_reduced"]
    rms = np.sqrt(
        mean_squared_error(s_df.loc[:, "s_unreduced"], s_df.loc[:, "s_reduced"])
    )
    logger.debug(
        "Root-mean-square error between apparent power in unreduced and reduced "
        f"EDisGo object is: rms = {rms:.2}"
    )
    return s_df, rms
