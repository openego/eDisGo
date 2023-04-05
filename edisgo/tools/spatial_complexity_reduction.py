import copy
import logging
import math

from hashlib import md5
from time import time
from typing import Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from pandas import DataFrame
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.network import timeseries
from edisgo.network.grids import Grid
from edisgo.tools.pseudo_coordinates import make_pseudo_coordinates

logger = logging.getLogger(__name__)


# region Used functions
def hash_df(df: DataFrame) -> str:
    """Calculate hash from busmap, good to check if dataframe has the same content."""
    s = df.to_json()
    return md5(s.encode()).hexdigest()


# Transform coordinates between the different coordinates systems
# ToDo: The spatial complexity reduction forces the usage of EPSG:4326 and than
# transforms to other coordinate systems if needed. Maybe write it more dynamically.
coor_transform = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
coor_transform_back = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def make_grid_list(edisgo_obj: EDisGo, grid: object = None) -> list:
    """
    Get a list of all grids in the EDisGo object or a list with the specified grid.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object to get the grids from.
    grid : :obj:`str`
        Name of the grid, if None than all grids of the EDisGo objects are used.

    Returns
    -------
    :obj:`list`
        List of the Grid objects
    """
    if edisgo_obj is None and grid is None:
        raise ValueError("Pass an EDisGo object and an grid")
    elif grid is not None:
        grid_name_list = [str(edisgo_obj.topology.mv_grid)]
        grid_name_list = grid_name_list + list(
            map(str, edisgo_obj.topology.mv_grid.lv_grids)
        )
        grid_list = [edisgo_obj.topology.mv_grid]
        grid_list = grid_list + list(edisgo_obj.topology.mv_grid.lv_grids)
        grid_list = [grid_list[grid_name_list.index(str(grid))]]
    else:
        grid_list = [edisgo_obj.topology.mv_grid]
        grid_list = grid_list + list(edisgo_obj.topology.mv_grid.lv_grids)

    return grid_list


def find_buses_of_interest(edisgo_root: EDisGo) -> set:
    """
    Find the buses of interest of the EDisGo object. With a worst-case powerflow the
    buses with problems are found. Check for line load and voltage deviation.

    Parameters
    ----------
    edisgo_root : :class:`~.EDisGo`
        The investigated EDisGo object.

    Returns
    -------
    :obj:`set`
        Set with the names of the buses of interest.
    """
    start_time = time()
    logger.info("Start - Find buses of interest")

    edisgo_obj = copy.deepcopy(edisgo_root)
    edisgo_obj.timeseries = timeseries.TimeSeries()
    edisgo_obj.timeseries.set_worst_case(edisgo_obj, ["feed-in_case", "load_case"])
    edisgo_obj.analyze()

    buses_of_interest = set()
    mv_lines = checks.mv_line_load(edisgo_obj)
    lv_lines = checks.lv_line_load(edisgo_obj)
    lines = mv_lines.index.tolist()
    lines = lines + lv_lines.index.tolist()
    for line in lines:
        buses_of_interest.add(edisgo_obj.topology.lines_df.loc[line, "bus0"])
        buses_of_interest.add(edisgo_obj.topology.lines_df.loc[line, "bus1"])

    mv_buses = checks.mv_voltage_deviation(edisgo_obj, voltage_levels="mv")
    for value in mv_buses.values():
        buses_of_interest.update(value.index.tolist())

    lv_buses = checks.lv_voltage_deviation(edisgo_obj, voltage_levels="lv")
    for value in lv_buses.values():
        buses_of_interest.update(value.index.tolist())

    logger.info("Finished in {}s".format(time() - start_time))
    return buses_of_interest


def rename_virtual_buses(
    partial_busmap_df: DataFrame, transformer_node: str
) -> DataFrame:
    """
    Rename virtual buses so that no virtual transformer bus is created.

    Parameters
    ----------
    partial_busmap_df: :obj:`pd.DataFrame`
        Busmap to work on.
    transformer_node: :obj:`str`
        Transformer node name.
    Returns
    -------
    :obj:`pd.DataFrame`
        Busmap with applied changes.
    """
    nodes = partial_busmap_df.index.to_list()
    pairs = []
    for node in nodes:
        if node.split("_")[0] == "virtual":
            if (
                partial_busmap_df.loc[node, "new_bus"]
                != partial_busmap_df.loc[node.lstrip("virtual_"), "new_bus"]
            ):
                pairs.append([node, node.lstrip("virtual_")])

    logger.debug("Pairs: {}".format(pairs))
    logger.debug("Length pairs: {}".format(len(pairs)))
    if len(pairs) > 0:
        logger.info("Rename virtual buses")
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


# endregion

# region Preprocessing
def remove_one_meter_lines(edisgo_root: EDisGo) -> EDisGo:
    """
    Remove one meter lines between the feeder and the buildings. This function is
    a relict from the legacy ding0 grids.

    Only one meter lines are dropped which have on one side only one neighbour.
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

    start_time = time()
    logger.info("Start - Removing 1m lines")

    edisgo_obj = copy.deepcopy(edisgo_root)
    G = edisgo_obj.to_graph()
    lines_df = edisgo_obj.topology.lines_df.copy()
    busmap = {}
    unused_lines = []
    for index, row in lines_df.iterrows():
        if row.length < 0.001:
            logger.info(
                'Line "{}" is {:.3f}m long and will not be removed.'.format(
                    index, row.length * 1000
                )
            )
        if row.length == 0.001:
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
            else:
                logger.info(
                    'Line "{}" is {:.3f}m long and will not be removed.'.format(
                        index, row.length * 1000
                    )
                )
    logger.info(
        "Drop {} of {} short lines ({:.0f}%)".format(
            len(unused_lines),
            lines_df.shape[0],
            (len(unused_lines) / lines_df.shape[0] * 100),
        )
    )
    # Apply the busmap on the components
    lines_df = lines_df.drop(unused_lines)
    lines_df = lines_df.apply(apply_busmap_on_lines_df, axis="columns")

    buses_df = edisgo_obj.topology.buses_df.copy()
    buses_df = buses_df.apply(apply_busmap_on_buses_df, axis="columns")
    buses_df = buses_df.groupby(
        by=["new_bus"], dropna=False, as_index=False, sort=False
    ).first()
    buses_df = buses_df.set_index("new_bus")

    loads_df = edisgo_obj.topology.loads_df.copy()
    loads_df = loads_df.apply(apply_busmap, axis="columns")

    generators_df = edisgo_obj.topology.generators_df.copy()
    generators_df = generators_df.apply(apply_busmap, axis="columns")

    storage_units_df = edisgo_obj.topology.storage_units_df.copy()
    storage_units_df = storage_units_df.apply(apply_busmap, axis="columns")

    edisgo_obj.topology.lines_df = lines_df
    edisgo_obj.topology.buses_df = buses_df
    edisgo_obj.topology.loads_df = loads_df
    edisgo_obj.topology.generators_df = generators_df
    edisgo_obj.topology.storage_units_df = storage_units_df

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj


def remove_lines_under_one_meter(edisgo_root: EDisGo) -> EDisGo:
    """
    Remove the lines under one meter. Sometimes these line are causing convergence
    problems of the power flow calculation or making problems with the clustering
    methods.

    Function might be a bit overengineered, so that the station bus is never dropped.
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

    start_time = time()
    logger.info("Start - Removing lines under 1m")

    edisgo_obj = copy.deepcopy(edisgo_root)

    busmap = {}
    unused_lines = []

    grid_list = [edisgo_obj.topology.mv_grid]
    grid_list = grid_list + list(edisgo_obj.topology.mv_grid.lv_grids)

    for grid in grid_list:
        G = grid.graph

        transformer_node = grid.transformers_df.bus1.values[0]

        lines_df = grid.lines_df.copy()

        for index, row in lines_df.iterrows():
            if row.length < 0.001:

                distance_bus_0, path = nx.single_source_dijkstra(
                    G, source=transformer_node, target=row.bus0, weight="length"
                )
                distance_bus_1, path = nx.single_source_dijkstra(
                    G, source=transformer_node, target=row.bus1, weight="length"
                )

                logger.debug(
                    'Line "{}" is {:.5f}m long and will be removed.'.format(
                        index, row.length * 1000
                    )
                )
                logger.debug(
                    "Bus0: {} - Distance0: {}".format(row.bus0, distance_bus_0)
                )
                logger.debug(
                    "Bus1: {} - Distance1: {}".format(row.bus1, distance_bus_1)
                )

                if distance_bus_0 < distance_bus_1:
                    busmap[row.bus1] = row.bus0
                    if distance_bus_0 < 0.001:
                        busmap[row.bus0] = transformer_node
                        busmap[row.bus1] = transformer_node
                elif distance_bus_0 > distance_bus_1:
                    busmap[row.bus0] = row.bus1
                    if distance_bus_1 < 0.001:
                        busmap[row.bus0] = transformer_node
                        busmap[row.bus1] = transformer_node
                else:
                    raise ValueError("ERROR")

                unused_lines.append(index)

    logger.debug("Busmap: {}".format(busmap))
    # Apply the busmap on the components
    transformers_df = edisgo_obj.topology.transformers_df.copy()
    transformers_df = transformers_df.apply(apply_busmap_on_lines_df, axis="columns")
    edisgo_obj.topology.transformers_df = transformers_df

    lines_df = edisgo_obj.topology.lines_df.copy()
    lines_df = lines_df.drop(unused_lines)
    lines_df = lines_df.apply(apply_busmap_on_lines_df, axis="columns")
    edisgo_obj.topology.lines_df = lines_df

    buses_df = edisgo_obj.topology.buses_df.copy()
    buses_df.index.name = "bus"
    buses_df = buses_df.apply(apply_busmap_on_buses_df, axis="columns")
    buses_df = buses_df.groupby(
        by=["new_bus"], dropna=False, as_index=False, sort=False
    ).first()
    buses_df = buses_df.set_index("new_bus")
    edisgo_obj.topology.buses_df = buses_df

    loads_df = edisgo_obj.topology.loads_df.copy()
    loads_df = loads_df.apply(apply_busmap, axis="columns")
    edisgo_obj.topology.loads_df = loads_df

    generators_df = edisgo_obj.topology.generators_df.copy()
    generators_df = generators_df.apply(apply_busmap, axis="columns")
    edisgo_obj.topology.generators_df = generators_df

    storage_units_df = edisgo_obj.topology.storage_units_df.copy()
    storage_units_df = storage_units_df.apply(apply_busmap, axis="columns")
    edisgo_obj.topology.storage_units_df = storage_units_df

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj


# endregion

# region Complexity reduction
def make_busmap_from_clustering(
    edisgo_root: EDisGo,
    grid: Union[None, str] = None,
    mode: str = None,
    reduction_factor: float = 0.25,
    preserve_trafo_bus_coordinates: bool = True,
) -> DataFrame:
    """
    See for more information the function
    :func:`make_busmap<edisgo.tools.make_busmap>`.

    Making busmap for the cluster area 'grid'. Every grid is clustered individually.

    Parameters
    ----------
    edisgo_root: :obj:`EDisGo`
        EDisGo object for which the busmap ist created.
    grid: :obj:`str` or None
        If None, busmap is created for all grids, else only for the selected Grid.
    mode: :obj:`str`
        "kmeans" or "kmeansdijkstra" as clustering method.
    reduction_factor: :obj:`float`
        Must bigger than 0 smaller than 1. Nodes are reduced with this factor.
    preserve_trafo_bus_coordinates: :obj:`bool`
        If true transformators have the same coordinates after the clustering, else
        the transformer coordinates are get by the clustering.

    Returns
    -------
    :obj:`pd.DataFrame`
        Busmap which maps the old bus names to the new bus names with new coordinates.

    References
    ----------
    In parts based on PyPSA spatial complexity reduction
    `https://pypsa.readthedocs.io/en/latest/examples/spatial-clustering.html`
    """

    def calculate_weighting(series):
        p_gen = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.bus == series.name, "p_nom"
        ].sum()
        p_load = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.bus == series.name, "p_set"
        ].sum()
        if str(grid).split("_")[0] == "MVGrid":
            s_tran = edisgo_obj.topology.transformers_df.loc[
                edisgo_obj.topology.transformers_df.bus0 == series.name, "s_nom"
            ].sum()
        else:
            s_tran = 0
        series.loc["weight"] = 1 + 1000 * (p_gen + p_load + s_tran)
        return series

    def transform_coordinates(series):
        x = series.x
        y = series.y
        x, y = coor_transform.transform(x, y)
        series["x"] = x
        series["y"] = y
        return series

    def transform_coordinates_back(series):
        x = series.new_x
        y = series.new_y
        x, y = coor_transform_back.transform(x, y)
        series["new_x"] = x
        series["new_y"] = y
        return series

    def rename_new_buses(series):
        if str(grid).split("_")[0] == "LVGrid":
            series["new_bus"] = (
                "Bus_mvgd_"
                + str(edisgo_obj.topology.mv_grid.id)
                + "_lvgd_"
                + str(grid.id)
                + "_"
                + str(int(series["new_bus"]))
            )

        elif str(grid).split("_")[0] == "MVGrid":
            series["new_bus"] = (
                "Bus_mvgd_"
                + str(edisgo_obj.topology.mv_grid.id)
                + "_"
                + str(int(series["new_bus"]))
            )
        elif grid is None:
            logger.error("Grid is None")
        return series

    start_time = time()
    logger.info("Start - Make busmap from clustering, mode = {}".format(mode))

    edisgo_obj = copy.deepcopy(edisgo_root)
    grid_list = make_grid_list(edisgo_obj, grid=grid)

    busmap_df = pd.DataFrame()
    # Cluster every grid
    for grid in grid_list:
        v_grid = grid.nominal_voltage
        logger.debug("Make busmap for grid: {}, v_nom={}".format(grid, v_grid))

        buses_df = grid.buses_df
        graph = grid.graph
        transformer_bus = grid.transformers_df.bus1[0]

        buses_df = buses_df.apply(transform_coordinates, axis="columns")
        buses_df = buses_df.apply(calculate_weighting, axis="columns")
        # Calculate number of clusters
        number_of_distinct_nodes = buses_df.groupby(by=["x", "y"]).first().shape[0]
        logger.debug("Number_of_distinct_nodes = " + str(number_of_distinct_nodes))
        n_clusters = math.ceil(number_of_distinct_nodes * reduction_factor)
        logger.debug("n_clusters = {}".format(n_clusters))
        # Cluster with kmeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

        kmeans.fit(buses_df.loc[:, ["x", "y"]], sample_weight=buses_df.loc[:, "weight"])

        partial_busmap_df = pd.DataFrame(index=buses_df.index)

        if mode == "kmeans":
            partial_busmap_df.loc[:, "new_bus"] = kmeans.labels_
            for index, new_bus in zip(
                partial_busmap_df.index, partial_busmap_df.new_bus
            ):
                partial_busmap_df.loc[
                    index, ["new_x", "new_y"]
                ] = kmeans.cluster_centers_[new_bus]

        elif mode == "kmeansdijkstra":
            # Use dijkstra to select clusters
            dist_to_cluster_center = pd.DataFrame(
                data=kmeans.transform(buses_df.loc[:, ["x", "y"]]), index=buses_df.index
            ).min(axis="columns")

            buses_df.loc[:, "cluster_number"] = kmeans.labels_
            medoid_bus_name = {}

            for n in range(0, n_clusters):
                medoid_bus_name[
                    dist_to_cluster_center.loc[
                        buses_df.loc[:, "cluster_number"].isin([n])
                    ].idxmin()
                ] = int(n)

            dijkstra_distances_df = pd.DataFrame(
                index=buses_df.index, columns=medoid_bus_name
            )

            for bus in medoid_bus_name:
                path_series = pd.Series(
                    nx.single_source_dijkstra_path_length(graph, bus, weight="length")
                )
                dijkstra_distances_df.loc[:, bus] = path_series

            buses_df.loc[:, "medoid"] = dijkstra_distances_df.idxmin(axis=1)
            partial_busmap_df = pd.DataFrame(index=buses_df.index)

            for index in buses_df.index:
                partial_busmap_df.loc[index, "new_bus"] = int(
                    medoid_bus_name[buses_df.loc[index, "medoid"]]
                )
                partial_busmap_df.loc[index, ["new_x", "new_y"]] = buses_df.loc[
                    buses_df.loc[index, "medoid"], ["x", "y"]
                ].values

        partial_busmap_df = partial_busmap_df.apply(rename_new_buses, axis="columns")

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

        if str(grid).split("_")[0] == "MVGrid":
            partial_busmap_df = rename_virtual_buses(partial_busmap_df, transformer_bus)

        partial_busmap_df = partial_busmap_df.apply(
            transform_coordinates_back, axis="columns"
        )

        busmap_df = pd.concat([busmap_df, partial_busmap_df])

    logger.info("Finished in {}s".format(time() - start_time))
    return busmap_df


def make_busmap_from_feeders(
    edisgo_root: EDisGo = None,
    grid: Union[None, Grid] = None,
    mode: str = None,
    reduction_factor: float = 0.25,
    reduction_factor_not_focused: Union[bool, float] = False,
) -> DataFrame:
    """
    See for more information the function :func:`make_busmap<edisgo.tools.make_busmap>`.

    Making busmap for the cluster area 'feeder'. Every feeder is clustered individually.

    Parameters
    ----------
    edisgo_root: :obj:`EDisGo`
        EDisGo object for which the busmap ist created.
    grid: :obj:`str` or None
        If None, busmap is created for all grids, else only for the selected Grid.
    mode: :obj:`str`
        "kmeans" or "kmeansdijkstra" as clustering method.
    reduction_factor: :obj:`float`
        Must bigger than 0 smaller than 1. Nodes are reduced with this factor.
    reduction_factor_not_focused: :obj:`bool` or :obj:`floar`
        If false the focus method is not used. If 0 or smaller than 1, this sets the
        reduction factor for buses not of interest. When selecting 0 the nodes of the
        clustering area are aggregated to the transformer bus.

    Returns
    -------
    :obj:`pd.DataFrame`
        Busmap which maps the old bus names to the new bus names with new coordinates.

    References
    ----------
    In parts based on PyPSA spatial complexity reduction
    `https://pypsa.readthedocs.io/en/latest/examples/spatial-clustering.html`
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

    def calculate_weighting(series):
        buses = partial_busmap_df.loc[
            partial_busmap_df.new_bus.isin([series.name])
        ].index.tolist()
        p_gen = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.bus.isin(buses), "p_nom"
        ].sum()
        p_load = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.bus.isin(buses), "p_set"
        ].sum()
        if str(grid).split("_")[0] == "MVGrid":
            s_tran = edisgo_obj.topology.transformers_df.loc[
                edisgo_obj.topology.transformers_df.bus0 == series.name, "s_nom"
            ].sum()
        else:
            s_tran = 0
        series.loc["weight"] = 1 + 1000 * (p_gen + p_load + s_tran)
        return series

    def transform_coordinates(series):
        x = series.x
        y = series.y
        x, y = coor_transform.transform(x, y)
        series["x"] = x
        series["y"] = y
        return series

    def transform_coordinates_back(ser):
        x = ser.new_x
        y = ser.new_y
        x, y = coor_transform_back.transform(x, y)
        ser["new_x"] = x
        ser["new_y"] = y
        return ser

    edisgo_obj = copy.deepcopy(edisgo_root)
    start_time = time()
    logger.info("Start - Make busmap from feeders, mode = {}".format(mode))

    edisgo_obj.topology.buses_df = edisgo_obj.topology.buses_df.apply(
        transform_coordinates, axis="columns"
    )

    grid_list = make_grid_list(edisgo_obj, grid=grid)
    busmap_df = pd.DataFrame()
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
        logger.info("Make busmap for grid: {}, v_nom={}".format(grid, v_grid))

        graph_root = grid.graph
        transformer_node = grid.transformers_df.bus1.values[0]
        transformer_coordinates = grid.buses_df.loc[
            transformer_node, ["x", "y"]
        ].tolist()
        logger.debug("Transformer node: {}".format(transformer_node))

        neighbors = list(nx.neighbors(graph_root, transformer_node))
        neighbors.sort()
        logger.debug(
            "Transformer neighbors has {} neighbors: {}".format(
                len(neighbors), neighbors
            )
        )

        graph_without_transformer = copy.deepcopy(graph_root)
        graph_without_transformer.remove_node(transformer_node)

        partial_busmap_df = pd.DataFrame(index=grid.buses_df.index)
        partial_busmap_df.index.name = "old_bus"
        for index in partial_busmap_df.index.tolist():
            partial_busmap_df.loc[index, "new_bus"] = index
            coordinates = grid.buses_df.loc[index, ["x", "y"]].values
            partial_busmap_df.loc[index, ["new_x", "new_y"]] = coordinates

        number_of_feeder = 0
        feeder_graphs = list(nx.connected_components(graph_without_transformer))
        feeder_graphs.sort()
        for feeder_nodes in feeder_graphs:
            feeder_nodes = list(feeder_nodes)
            feeder_nodes.sort()

            feeder_buses_df = grid.buses_df.loc[feeder_nodes, :]
            # return feeder_buses_df
            feeder_buses_df = feeder_buses_df.apply(calculate_weighting, axis="columns")

            if focus_mode:
                selected_reduction_factor = reduction_factor_not_focused
                for bus in feeder_nodes:
                    if bus in buses_of_interest:
                        selected_reduction_factor = reduction_factor
            else:
                selected_reduction_factor = reduction_factor

            number_of_distinct_nodes = (
                feeder_buses_df.groupby(by=["x", "y"]).first().shape[0]
            )
            logger.debug("Number_of_distinct_nodes = " + str(number_of_distinct_nodes))
            n_clusters = math.ceil(selected_reduction_factor * number_of_distinct_nodes)
            logger.debug("n_clusters = {}".format(n_clusters))

            # Aggregate to transformer bus if there are no clusters
            if n_clusters == 0:
                for index in feeder_buses_df.index.tolist():
                    partial_busmap_df.loc[index, "new_bus"] = transformer_node
                    partial_busmap_df.loc[
                        index, ["new_x", "new_y"]
                    ] = transformer_coordinates
            else:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                kmeans.fit(
                    feeder_buses_df.loc[:, ["x", "y"]],
                    sample_weight=feeder_buses_df.loc[:, "weight"],
                )

                if mode == "kmeans":
                    n = 0
                    for index in feeder_buses_df.index.tolist():
                        partial_busmap_df.loc[index, "new_bus"] = make_name(
                            kmeans.labels_[n] + 1
                        )
                        partial_busmap_df.loc[
                            index, ["new_x", "new_y"]
                        ] = kmeans.cluster_centers_[kmeans.labels_[n]]
                        n = n + 1
                elif mode == "kmeansdijkstra":
                    dist_to_cluster_center = pd.DataFrame(
                        data=kmeans.transform(feeder_buses_df.loc[:, ["x", "y"]]),
                        index=feeder_buses_df.index,
                    ).min(axis="columns")
                    feeder_buses_df.loc[:, "cluster_number"] = kmeans.labels_
                    medoid_bus_name = {}

                    for n in range(0, n_clusters):
                        medoid_bus_name[
                            dist_to_cluster_center.loc[
                                feeder_buses_df.loc[:, "cluster_number"].isin([n])
                            ].idxmin()
                        ] = int(n)

                    dijkstra_distances_df = pd.DataFrame(
                        index=feeder_buses_df.index, columns=medoid_bus_name
                    )

                    for bus in medoid_bus_name:
                        path_series = pd.Series(
                            nx.single_source_dijkstra_path_length(
                                graph_root, bus, cutoff=None, weight="length"
                            )
                        )
                        dijkstra_distances_df.loc[:, bus] = path_series

                    feeder_buses_df.loc[:, "medoid"] = dijkstra_distances_df.idxmin(
                        axis=1
                    )

                    for index in feeder_buses_df.index.tolist():
                        partial_busmap_df.loc[index, "new_bus"] = make_name(
                            medoid_bus_name[feeder_buses_df.loc[index, "medoid"]] + 1
                        )
                        partial_busmap_df.loc[
                            index, ["new_x", "new_y"]
                        ] = feeder_buses_df.loc[
                            feeder_buses_df.loc[index, "medoid"], ["x", "y"]
                        ].values
            number_of_feeder = number_of_feeder + 1

        if str(grid).split("_")[0] == "MVGrid":
            partial_busmap_df = rename_virtual_buses(
                partial_busmap_df, transformer_node
            )

        busmap_df = pd.concat([busmap_df, partial_busmap_df])

    busmap_df = busmap_df.apply(transform_coordinates_back, axis="columns")
    busmap_df.sort_index(inplace=True)
    logger.info("Finished in {}s".format(time() - start_time))
    return busmap_df


def make_busmap_from_main_feeders(
    edisgo_root: EDisGo = None,
    grid: Union[None, Grid] = None,
    mode: str = None,
    reduction_factor: float = 0.25,
    reduction_factor_not_focused: Union[bool, float] = False,
) -> DataFrame:
    """
    See for more information the function :func:`make_busmap<edisgo.tools.make_busmap>`.

    Making busmap for the cluster area 'main_feeder'. Every main feeder is clustered
    individually. The main feeder is selected as the longest way in the feeder. All
    nodes are aggregated to this main feeder and than the feeder is clustered.

    Parameters
    ----------
    edisgo_root: :obj:`EDisGo`
        EDisGo object for which the busmap ist created.
    grid: :obj:`str` or None
        If None, busmap is created for all grids, else only for the selected Grid.
    mode: :obj:`str`
        "kmeans", "kmeansdijkstra", "aggregate_to_main_feeder" or
        "equidistant_nodes" as clustering method.
    reduction_factor: :obj:`float`
        Must bigger than 0 smaller than 1. Nodes are reduced with this factor.
    reduction_factor_not_focused: :obj:`bool` or :obj:`floar`
        If false the focus method is not used. If 0 or smaller than 1, this sets the
        reduction factor for buses not of interest. When selecting 0 the nodes of the
        clustering area are aggregated to the transformer bus.

    Returns
    -------
    :obj:`pd.DataFrame`
        Busmap which maps the old bus names to the new bus names with new coordinates.

    References
    ----------
    In parts based on PyPSA spatial complexity reduction
    `https://pypsa.readthedocs.io/en/latest/examples/spatial-clustering.html`
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

    def calculate_weighting(series):
        buses = partial_busmap_df.loc[
            partial_busmap_df.new_bus.isin([series.name])
        ].index.tolist()
        p_gen = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.bus.isin(buses), "p_nom"
        ].sum()
        p_load = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.bus.isin(buses), "p_set"
        ].sum()
        if str(grid).split("_")[0] == "MVGrid":
            s_tran = edisgo_obj.topology.transformers_df.loc[
                edisgo_obj.topology.transformers_df.bus0 == series.name, "s_nom"
            ].sum()
        else:
            s_tran = 0
        series.loc["weight"] = 1 + 1000 * (p_gen + p_load + s_tran)
        return series

    def transform_coordinates(series):
        x = series.x
        y = series.y
        x, y = coor_transform.transform(x, y)
        series["x"] = x
        series["y"] = y
        return series

    def transform_coordinates_back(ser):
        x = ser.new_x
        y = ser.new_y
        x, y = coor_transform_back.transform(x, y)
        ser["new_x"] = x
        ser["new_y"] = y
        return ser

    def next_main_node(node_to_delete, graph_root, main_feeder_nodes):
        for node, predecessor in nx.bfs_predecessors(graph_root, source=node_to_delete):
            if node in main_feeder_nodes:
                return node

    edisgo_obj = copy.deepcopy(edisgo_root)
    start_time = time()
    logger.info("Start - Make busmap from main feeders, mode = {}".format(mode))

    if mode != "aggregate_to_main_feeder":
        edisgo_obj.topology.buses_df = edisgo_obj.topology.buses_df.apply(
            transform_coordinates, axis="columns"
        )

    grid_list = make_grid_list(edisgo_obj, grid=grid)
    busmap_df = pd.DataFrame()
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
        logger.debug("Make busmap for grid: {}, v_nom={}".format(grid, v_grid))

        graph_root = grid.graph
        transformer_node = grid.transformers_df.bus1.values[0]
        transformer_coordinates = grid.buses_df.loc[
            transformer_node, ["x", "y"]
        ].tolist()
        logger.debug("Transformer node: {}".format(transformer_node))

        neighbors = list(nx.neighbors(graph_root, transformer_node))
        neighbors.sort()
        logger.debug(
            "Transformer neighbors has {} neighbors: {}".format(
                len(neighbors), neighbors
            )
        )

        graph_without_transformer = copy.deepcopy(graph_root)
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
        nodes = list(graph_root.nodes())
        not_main_nodes = []
        for node in nodes:
            if node not in main_feeder_nodes:
                not_main_nodes.append(node)
        partial_busmap_df = pd.DataFrame(index=grid.buses_df.index)
        partial_busmap_df.index.name = "old_bus"
        for index in partial_busmap_df.index.tolist():
            partial_busmap_df.loc[index, "new_bus"] = index
            coordinates = grid.buses_df.loc[index, ["x", "y"]].values
            partial_busmap_df.loc[index, ["new_x", "new_y"]] = coordinates

        graph_cleaned = copy.deepcopy(graph_root)
        graph_cleaned.remove_nodes_from(not_main_nodes)

        for node_to_delete in not_main_nodes:
            node = next_main_node(node_to_delete, graph_root, main_feeder_nodes)
            partial_busmap_df.loc[node_to_delete, "new_bus"] = node
            coordinates = partial_busmap_df.loc[node, ["new_x", "new_y"]].values
            partial_busmap_df.loc[node_to_delete, ["new_x", "new_y"]] = coordinates

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

            i = 0
            for end_node in end_nodes:
                number_of_feeder = end_nodes.index(end_node)
                feeder = main_feeders_df.loc[number_of_feeder, :]

                # Calculate nodes per feeder
                if focus_mode:
                    selected_reduction_factor = reduction_factor_not_focused
                    for node in feeder.path[1:]:
                        buses = partial_busmap_df.loc[
                            partial_busmap_df.new_bus.isin([node])
                        ].index.tolist()
                        for bus in buses:
                            if bus in buses_of_interest:
                                selected_reduction_factor = reduction_factor
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
                    distance_from_transformer = nx.shortest_path_length(
                        graph_cleaned,
                        source=transformer_node,
                        target=node,
                        weight="length",
                    )
                    old_feeder[node_number] = np.abs(
                        new_feeder - distance_from_transformer
                    ).argmin()
                    node_number += 1

                # Make busmap for feeders
                end_coordinates = grid.buses_df.loc[
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
                feeder_buses_df = grid.buses_df.loc[feeder.path, :]
                feeder_buses_df = feeder_buses_df.apply(
                    calculate_weighting, axis="columns"
                )

                if focus_mode:
                    selected_reduction_factor = reduction_factor_not_focused
                    for node in feeder_buses_df.index.to_list():
                        buses = partial_busmap_df.loc[
                            partial_busmap_df.new_bus.isin([node])
                        ].index.tolist()
                        for bus in buses:
                            if bus in buses_of_interest:
                                selected_reduction_factor = reduction_factor
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
                logger.debug("n_clusters = {}".format(n_clusters))

                # Aggregate to transformer bus if there are no clusters
                if n_clusters == 0:
                    for index in feeder_buses_df.index.tolist():
                        partial_busmap_df.loc[index, "new_bus"] = transformer_node
                        partial_busmap_df.loc[
                            index, ["new_x", "new_y"]
                        ] = transformer_coordinates
                else:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                    kmeans.fit(
                        feeder_buses_df.loc[:, ["x", "y"]],
                        sample_weight=feeder_buses_df.loc[:, "weight"],
                    )

                    if mode == "kmeans":
                        n = 0
                        for index in feeder_buses_df.index.tolist():
                            partial_busmap_df.loc[index, "new_bus"] = make_name(
                                kmeans.labels_[n] + 1
                            )
                            partial_busmap_df.loc[
                                index, ["new_x", "new_y"]
                            ] = kmeans.cluster_centers_[kmeans.labels_[n]]
                            n = n + 1
                    elif mode == "kmeansdijkstra":
                        dist_to_cluster_center = pd.DataFrame(
                            data=kmeans.transform(feeder_buses_df.loc[:, ["x", "y"]]),
                            index=feeder_buses_df.index,
                        ).min(axis="columns")
                        feeder_buses_df.loc[:, "cluster_number"] = kmeans.labels_
                        medoid_bus_name = {}

                        for n in range(0, n_clusters):
                            medoid_bus_name[
                                dist_to_cluster_center.loc[
                                    feeder_buses_df.loc[:, "cluster_number"].isin([n])
                                ].idxmin()
                            ] = int(n)

                        dijkstra_distances_df = pd.DataFrame(
                            index=feeder_buses_df.index, columns=medoid_bus_name
                        )

                        for bus in medoid_bus_name:
                            path_series = pd.Series(
                                nx.single_source_dijkstra_path_length(
                                    graph_root, bus, cutoff=None, weight="length"
                                )
                            )
                            dijkstra_distances_df[bus] = path_series

                        feeder_buses_df.loc[:, "medoid"] = dijkstra_distances_df.idxmin(
                            axis=1
                        )

                        for index in feeder_buses_df.index.tolist():
                            partial_busmap_df.loc[index, "new_bus"] = make_name(
                                medoid_bus_name[feeder_buses_df.loc[index, "medoid"]]
                                + 1
                            )
                            partial_busmap_df.loc[
                                index, ["new_x", "new_y"]
                            ] = feeder_buses_df.loc[
                                feeder_buses_df.loc[index, "medoid"], ["x", "y"]
                            ].values

        if mode != "aggregate_to_main_feeder":
            # Backmap
            for node in not_main_nodes:
                partial_busmap_df.loc[node] = partial_busmap_df.loc[
                    partial_busmap_df.loc[node, "new_bus"]
                ]

        if str(grid).split("_")[0] == "MVGrid":
            partial_busmap_df = rename_virtual_buses(
                partial_busmap_df, transformer_node
            )

        busmap_df = pd.concat([busmap_df, partial_busmap_df])

    if mode != "aggregate_to_main_feeder":
        busmap_df = busmap_df.apply(transform_coordinates_back, axis="columns")

    logger.info("Finished in {}s".format(time() - start_time))
    return busmap_df


def make_busmap(
    edisgo_root: EDisGo,
    mode: str = None,
    cluster_area: str = None,
    reduction_factor: float = 0.25,
    reduction_factor_not_focused: Union[bool, float] = False,
    grid: Union[None, Grid] = None,
) -> DataFrame:
    """
    Wrapper around the different make_busmap methods for the different cluster areas.
    From the EDisGo object a busmap is generated. The bus names are mapped to new bus
    names with new coordinates. The busmap can be used with the function
    :func:`reduce_edisgo <edisgo.tools.spatial_complexity_reduction.reduce_edisgo>` to
    perform a spatial complexity reduction.

    Parameters
    ----------
    edisgo_root: :obj:`EDisGo`
        EDisGo object for which the busmap ist created.
    mode: :obj:`str`
        "kmeans", "kmeansdijkstra", "aggregate_to_main_feeder" or
        "equidistant_nodes" as clustering method. "aggregate_to_main_feeder" or
        "equidistant_nodes" only work with the cluster area "main_feeder".

        - "kmeans":
            Perform the k-means algorithm on the cluster area and map then the buses to
            the cluster centers.
        - "kmeansdijkstra":
            Perform the k-mean algorithm but then map the nodes to the cluster centers
            through the distance shortest distance in the graph. The distances are
            calculated by the dijkstra algorithm.
        - "aggregate_to_main_feeder":
            Aggregate the nodes in the feeder to the longest path in the feeder, here
            named main feeder.
        - "equidistant_nodes":
            Use the method aggregate to main feeder and then reduce the nodes again.
            Through a reduction of the nodes by the reduction factor and then
            distributing the remaining nodes on the graph.

    cluster_area: :obj:`str`
        Select: 'grid', 'feeder' or 'main_feeder' as the cluster area. The cluster area
        is the area on which the different clustering methods are used on.
    reduction_factor: :obj:`float`
        Must bigger than 0 smaller than 1. Nodes are reduced with this factor.
    reduction_factor_not_focused: :obj:`bool` or :obj:`float`
        If false the focus method is not used. If 0 or smaller than 1, this sets the
        reduction factor for buses not of interest. When selecting 0 the nodes of the
        clustering area are aggregated to the transformer bus.
    grid: :obj:`str` or None
        If None, busmap is created for all grids, else only for the selected Grid.

    Returns
    -------
    :obj:`pd.DataFrame`
        Busmap which maps the old bus names to the new bus names with new coordinates.

    References
    ----------
    In parts based on PyPSA spatial complexity reduction
    `https://pypsa.readthedocs.io/en/latest/examples/spatial-clustering.html`
    """
    # Check for false input.
    if mode == "aggregate_to_main_feeder":
        pass  # Aggregate to the main feeder
    elif not 0 < reduction_factor < 1.0:
        raise ValueError("Reduction factor must bigger than 0 and smaller than 1")

    if mode not in [
        "aggregate_to_main_feeder",
        "kmeans",
        "kmeansdijkstra",
        "equidistant_nodes",
    ]:
        raise ValueError(f"Selected false {mode=}")
    if (reduction_factor_not_focused is not False) and not (
        0 <= reduction_factor_not_focused < 1.0
    ):
        raise ValueError(
            f"{reduction_factor_not_focused=}, should be 'False' "
            f"or 0 or bigger than 0 but smaller than 1"
        )

    if cluster_area == "grid":
        busmap_df = make_busmap_from_clustering(
            edisgo_root,
            mode=mode,
            grid=grid,
            reduction_factor=reduction_factor,
        )
    elif cluster_area == "feeder":
        busmap_df = make_busmap_from_feeders(
            edisgo_root,
            grid=grid,
            mode=mode,
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=reduction_factor_not_focused,
        )
    elif cluster_area == "main_feeder":
        busmap_df = make_busmap_from_main_feeders(
            edisgo_root,
            grid=grid,
            mode=mode,
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=reduction_factor_not_focused,
        )
    else:
        raise ValueError(f"Selected false {cluster_area=}!")

    return busmap_df


def make_remaining_busmap(busmap_df: DataFrame, edisgo_root: EDisGo) -> DataFrame:
    """
    Make the remaining busmap out of an existing busmap and the EDisGo object.
    If only the busmap for one grid is generated than with this function the remaining
    busmap can be generated. Cause the
    :func:`reduce_edisgo <edisgo.tools.spatial_complexity_reduction.reduce_edisgo>`
    needs a busmap with all nodes of a bus.

    Parameters
    ----------
    busmap_df: :obj:`pd.DataFrame`
        Busmap with missing nodes.
    edisgo_root: :obj:`EDisGo`
        EDisGo object for which the busmap is generated.

    Returns
    -------
    :obj:`DataFrame`
        Busmap for the full edisgo object.
    """

    start_time = time()
    logger.info("Start - Make remaining busmap")

    remaining_busmap_df = edisgo_root.topology.buses_df.loc[
        ~edisgo_root.topology.buses_df.index.isin(busmap_df.index)
    ].copy()
    remaining_busmap_df.loc[
        remaining_busmap_df.index, "new_bus"
    ] = remaining_busmap_df.index
    remaining_busmap_df.loc[remaining_busmap_df.index, "new_x"] = remaining_busmap_df.x
    remaining_busmap_df.loc[remaining_busmap_df.index, "new_y"] = remaining_busmap_df.y
    remaining_busmap_df = remaining_busmap_df.drop(
        labels=["v_nom", "mv_grid_id", "lv_grid_id", "in_building", "x", "y"], axis=1
    )
    remaining_busmap_df.index.name = "old_bus"
    busmap_df = pd.concat([busmap_df, remaining_busmap_df])

    logger.info("Finished in {}s".format(time() - start_time))
    return busmap_df


def reduce_edisgo(
    edisgo_root: EDisGo,
    busmap_df: DataFrame,
    line_naming_convention: str = "standard_lines",
    aggregation_mode: bool = True,
    load_aggregation_mode: str = "sector",
    generator_aggregation_mode: str = "type",
) -> Tuple[EDisGo, DataFrame]:
    """
    Function to reduce the EDisGo object with a previously generated busmap.

    Warning: After reducing all attributes 'in_building' of the buses is set to False.
    Also, the method only works for lines which length can be calculated with a detour
    factor, which has to be set in the config. Else the line length is calculated
    false which leads to false results. If the grid doesn't have coordinates or
    matches the requirements use the :func:`make_pseudo_coordinates
    <edisgo.tools.pseudo_coordinates.make_pseudo_coordinates>`.

    Parameters
    ----------
    edisgo_root: :obj:`EDisGo`
        EDisGo object to reduce.
    busmap_df: :obj:`DataFrame`
        Busmap, holds the information which nodes are merged together.
    aggregation_mode: :obj:`str`
        If True loads and generators and their timeseries are aggregated
        according to their selected modes.
    line_naming_convention: :obj:`str`
        Select "standard_lines" or "combined_name". When the lines are aggregated it
        can happen that two or more lines are merged. This leads to the problem of
        new line types. If "standard_lines" is selected. In the case of a line merge
        as line_tipe and kind the values of the standard type of the voltage level is
        selected. If "combined_name" is selected the new type and kind is contains the
        concated names of the merged lines.

    load_aggregation_mode: :obj:`str`
        Choose: "bus" or "sector" if bus is chosen all loads in the loads_df are
        aggregated per bus. When sector is chosen its aggregated by bus and sector.
    generator_aggregation_mode: :obj:`str`
        Choose: "bus" or "type" if bus is chosen all generators in the generators_df
        are aggregated per bus. When type is chosen its aggregated by bus and type.

    Returns
    -------
    :obj:`EDisGo`
        Reduced EDisGo object.

    References
    ----------
    In parts based on PyPSA spatial complexity reduction
    `https://pypsa.readthedocs.io/en/latest/examples/spatial-clustering.html`
    """
    # region Define local functions
    def apply_busmap_on_busmap_df(series):
        series.loc["bus"] = busmap_df.loc[series.name, "new_bus"]
        series.loc["x"] = busmap_df.loc[series.name, "new_x"]
        series.loc["y"] = busmap_df.loc[series.name, "new_y"]
        return series

    def apply_busmap_on_lines_df(series):
        series.loc["bus0"] = busmap_df.loc[series.bus0, "new_bus"]
        series.loc["bus1"] = busmap_df.loc[series.bus1, "new_bus"]
        return series

    def remove_lines_with_the_same_bus(series):
        if series.bus0 == series.bus1:
            return  # Drop lines which connect to the same bus.
        elif (
            series.bus0.split("_")[0] == "virtual"
            and series.bus0.lstrip("virtual_") == slack_bus
        ) or (
            series.bus1.split("_")[0] == "virtual"
            and series.bus1.lstrip("virtual_") == slack_bus
        ):
            logger.debug(
                f"Drop line because connected to virtual_slack bus {series.name}"
            )
            return
        elif series.bus0.lstrip("virtual_") == series.bus1.lstrip("virtual_"):
            logger.debug(
                f"Drop line because it shorts the circuit breaker {series.name}"
            )
            return
        else:
            return series

    def get_ordered_lines_df(lines_df):
        """Order lines so that a grouping is possible."""
        order = lines_df.bus0 < lines_df.bus1
        lines_df_p = lines_df[order]
        lines_df_n = lines_df[~order].rename(columns={"bus0": "bus1", "bus1": "bus0"})
        lines_df = pd.concat([lines_df_p, lines_df_n], sort=True)
        return lines_df

    def aggregate_lines_df(df):
        series = pd.Series(index=lines_df.columns, dtype="object")

        bus0 = df.loc[:, "bus0"].values[0]
        bus1 = df.loc[:, "bus1"].values[0]
        v_nom = buses_df.loc[bus0, "v_nom"]

        coor_bus0 = buses_df.loc[bus0, ["x", "y"]].values.tolist()
        coor_bus1 = buses_df.loc[bus1, ["x", "y"]].values.tolist()

        coor_bus0 = coor_transform.transform(coor_bus0[0], coor_bus0[1])
        coor_bus1 = coor_transform.transform(coor_bus1[0], coor_bus1[1])

        length = (
            math.dist(coor_bus0, coor_bus1)
            / 1000
            * edisgo_obj.config["grid_connection"]["branch_detour_factor"]
        )

        if length == 0:
            length = 0.001
            logger.warning(
                f"Length of line between {bus0} and {bus1} can't be 0, set to 1m."
            )
        if length < 0.001:
            logger.warning(f"Length of line between {bus0} and {bus1} smaller than 1m.")

        # Get type of the line to get the according standard line for the voltage_level
        if np.isnan(buses_df.loc[df.bus0, "lv_grid_id"])[0]:
            type_line = f"mv_line_{int(v_nom)}kv"
        else:
            type_line = "lv_line"

        if len(df["type_info"].values) > 1:
            if line_naming_convention == "combined_name":
                type_info = "Merged: "
                for x in df["type_info"].values:
                    type_info = type_info + str(x) + " "
            elif line_naming_convention == "standard_lines":
                type_info = edisgo_obj.config["grid_expansion_standard_equipment"][
                    type_line
                ]
        else:
            type_info = df["type_info"].values[0]

        if len(df["kind"].values) > 1:
            if line_naming_convention == "combined_name":
                kind = "Combined: "
                for x in df["kind"].values:
                    kind = kind + str(x) + " "
            elif line_naming_convention == "standard_lines":
                kind = df["kind"].values[0]
        else:
            kind = df["kind"].values[0]

        x_sum = 0
        for line_type in df["type_info"].values:
            try:
                x_line = line_data_df.loc[line_data_df.U_n.isin([v_nom])].loc[
                    line_type, "L_per_km"
                ]
            except KeyError:
                x_line = line_data_df.loc[line_type, "L_per_km"]
                logger.error(
                    f"Line type doesn't matches voltage level"
                    f"{line_type} not in voltage level {v_nom}."
                )
            x_sum = x_sum + 1 / x_line
        x_sum = 1 / x_sum
        x = length * 2 * math.pi * 50 * x_sum / 1000

        r_sum = 0
        for line_type in df["type_info"].values:
            try:
                r_line = line_data_df.loc[line_data_df.U_n.isin([v_nom])].loc[
                    line_type, "R_per_km"
                ]
            except KeyError:
                r_line = line_data_df.loc[line_type, "R_per_km"]
                logger.error(
                    f"Line type doesn't matches voltage level"
                    f"{line_type} not in voltage level {v_nom}."
                )

            r_sum = r_sum + 1 / r_line
        r_sum = 1 / r_sum
        r = length * r_sum

        series.loc["length"] = length
        series.loc["bus0"] = bus0
        series.loc["bus1"] = bus1
        series.loc["x"] = x
        series.loc["r"] = r
        series.loc["b"] = 0.0
        series.loc["s_nom"] = df["s_nom"].sum()
        series.loc["num_parallel"] = int(df.loc[:, "num_parallel"].sum())
        series.loc["type_info"] = type_info
        series.loc["kind"] = kind
        series.loc["old_line_name"] = df.index.to_list()
        return series

    def apply_busmap_on_components(series):
        series.loc["bus"] = busmap_df.loc[series.loc["bus"], "new_bus"]
        return series

    def aggregate_loads_df(df):
        series = pd.Series(index=df.columns, dtype="object")  # l.values[0],
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_set"] = df.loc[:, "p_set"].sum()
        series.loc["annual_consumption"] = df.loc[:, "annual_consumption"].sum()
        if load_aggregation_mode == "sector":
            series.loc["sector"] = df.loc[:, "sector"].values[0]
        elif load_aggregation_mode == "bus":
            series.loc["sector"] = "aggregated"
        series.loc["old_name"] = df.index.tolist()
        return series

    def aggregate_generators_df(df):
        series = pd.Series(index=df.columns, dtype="object")
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_nom"] = df.loc[:, "p_nom"].sum()
        series.loc["control"] = df.loc[:, "control"].values[0]
        series.loc["subtype"] = df.loc[:, "subtype"].values[0]
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
    ) -> None:
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

        return edisgo_obj

    def apply_busmap_on_transformers_df(series):
        series.loc["bus0"] = busmap_df.loc[series.loc["bus0"], "new_bus"]
        series.loc["bus1"] = busmap_df.loc[series.loc["bus1"], "new_bus"]
        return series

    # endregion

    # region Reduce EDisGO object
    start_time = time()
    logger.info("Start - Reducing edisgo object")

    edisgo_obj = copy.deepcopy(edisgo_root)  # Make deepcopy to preserve root object

    logger.info("Copy dataframes from edisgo object")
    buses_df = edisgo_obj.topology.buses_df.copy()
    lines_df = edisgo_obj.topology.lines_df.copy()
    loads_df = edisgo_obj.topology.loads_df.copy()
    generators_df = edisgo_obj.topology.generators_df.copy()
    storage_units_df = edisgo_obj.topology.storage_units_df.copy()
    switches_df = edisgo_obj.topology.switches_df.copy()

    slack_bus = edisgo_obj.topology.transformers_hvmv_df.bus1[0]

    logger.info("Manipulate buses_df")
    buses_df = buses_df.apply(apply_busmap_on_busmap_df, axis="columns")
    buses_df = buses_df.groupby(by=["bus"], dropna=False, as_index=False).first()

    buses_df.loc[:, "in_building"] = False
    buses_df = buses_df.set_index("bus")

    logger.info("Manipulate lines_df")
    if not lines_df.empty:
        # Get one dataframe with all data of the line types
        line_data_df = pd.concat(
            [
                edisgo_obj.topology.equipment_data["mv_overhead_lines"],
                edisgo_obj.topology.equipment_data["mv_cables"],
                edisgo_obj.topology.equipment_data["lv_cables"],
            ]
        )

        lines_df = lines_df.apply(apply_busmap_on_lines_df, axis=1)
        lines_df = lines_df.apply(
            remove_lines_with_the_same_bus, axis="columns", result_type="broadcast"
        ).dropna()
        lines_df = get_ordered_lines_df(lines_df)
        lines_df = lines_df.groupby(by=["bus0", "bus1"]).apply(aggregate_lines_df)
        lines_df.index = (
            "Line_" + lines_df.loc[:, "bus0"] + "_to_" + lines_df.loc[:, "bus1"]
        )

    logger.info("Manipulate loads_df")
    if not loads_df.empty:
        loads_df = loads_df.apply(apply_busmap_on_components, axis="columns")

        if aggregation_mode:
            if load_aggregation_mode == "sector":
                loads_df = loads_df.groupby(by=["bus", "sector"]).apply(
                    aggregate_loads_df
                )
                loads_df.index = (
                    "Load_" + loads_df.loc[:, "bus"] + "_" + loads_df.loc[:, "sector"]
                )
            elif load_aggregation_mode == "bus":
                loads_df = loads_df.groupby(by=["bus"]).apply(aggregate_loads_df)
                loads_df.index = "Load_" + loads_df.loc[:, "bus"]

            loads_df.index.name = "name"

            aggregate_timeseries(
                loads_df, edisgo_obj, ["loads_active_power", "loads_reactive_power"]
            )

    logger.info("Manipulate generators_df")
    if not generators_df.empty:
        generators_df = generators_df.loc[
            generators_df.loc[:, "bus"].isin(busmap_df.index), :
        ]
        generators_df = generators_df.apply(apply_busmap_on_components, axis="columns")

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

    logger.info("Manipulate storage_units_df")

    if not storage_units_df.empty:
        storage_units_df = storage_units_df.apply(
            apply_busmap_on_components, axis="columns"
        )

    logger.info("Manipulate transformers_df")
    transformers_df = edisgo_obj.topology.transformers_df
    transformers_df = transformers_df.apply(
        apply_busmap_on_transformers_df, axis="columns"
    )

    logger.info("Manipulate switches_df")
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
            logger.warning("Drop unused switches: {}".format(switches_to_drop))
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

    logger.info("Make linemap_df")
    linemap_df = pd.DataFrame()
    for new_line_name, old_line_names in zip(lines_df.index, lines_df.old_line_name):
        for old_line_name in old_line_names:
            linemap_df.loc[old_line_name, "new_line_name"] = new_line_name

    logger.info("Finished in {}s".format(time() - start_time))
    # endregion

    return edisgo_obj, linemap_df


def spatial_complexity_reduction(
    edisgo_root: EDisGo,
    mode: str = "kmeansdijkstra",
    cluster_area: str = "feeder",
    reduction_factor: float = 0.5,
    reduction_factor_not_focused: Union[float, bool] = 0.2,
    apply_pseudo_coordinates: bool = True,
) -> Tuple[EDisGo, DataFrame, DataFrame]:
    """
    Wrapper around the functions :func:`make_busmap<edisgo.tools.make_busmap>` and
    :func:`reduce_edisgo <edisgo.tools.spatial_complexity_reduction.reduce_edisgo>`
    look there for more information.

    Parameters
    ----------
    apply_pseudo_coordinates : bool
        If True Pseudo Coordinates are applied. The spatial complexity reduction method
        is only tested for pseudo coordinates.
    """
    edisgo_obj = copy.deepcopy(edisgo_root)
    if apply_pseudo_coordinates:
        edisgo_obj = make_pseudo_coordinates(edisgo_obj)

    busmap_df = make_busmap(
        edisgo_obj,
        mode=mode,
        cluster_area=cluster_area,
        reduction_factor=reduction_factor,
        reduction_factor_not_focused=reduction_factor_not_focused,
    )
    edisgo_reduced, linemap_df = reduce_edisgo(
        edisgo_obj, busmap_df, aggregation_mode=False
    )

    return edisgo_reduced, busmap_df, linemap_df


# endregion

# region Postprocessing/Evaluation
def save_results_reduced_to_min_max(
    edisgo_root: EDisGo, edisgo_object_name: str
) -> EDisGo:
    """
    Save only the biggest and smallest value of the results DataFrames (v_res, i_res,
    pfa_p, pfa_q), because they are most important for evaluating the violation of
    operating limits.

    Parameters
    ----------
    edisgo_root: :obj:`EDisGo`
        EDisGo object to save.
    edisgo_object_name: :obj:`str`
        The path with name to save the object.
    """
    edisgo_obj = copy.deepcopy(edisgo_root)

    def min_max(df):
        min_df = df.min()
        min_df.name = "min"
        max_df = df.max()
        max_df.name = "max"

        df = pd.concat([min_df, max_df], axis=1)
        df = df.T
        return df

    start_time = time()
    logger.info("Start - Reduce results to min and max")

    edisgo_obj.results.v_res = min_max(edisgo_obj.results.v_res)
    edisgo_obj.results.i_res = min_max(edisgo_obj.results.i_res)
    edisgo_obj.results.pfa_p = min_max(edisgo_obj.results.pfa_p)
    edisgo_obj.results.pfa_q = min_max(edisgo_obj.results.pfa_q)

    edisgo_obj.save(
        edisgo_object_name, save_results=True, save_topology=True, save_timeseries=False
    )

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj


# Analyze results
def length_analysis(edisgo_obj: EDisGo) -> Tuple[float, float, float]:
    """Calculates the line length in the grid_districts and voltage levels.

    Parameters
    ----------
    edisgo_obj: :obj:`EDisGo`
        EDisGo object to analyze.

    Returns
    -------
    length_total: :obj:`float`
        Total line length of the grid district.
    length_mv: :obj:`float`
        Total line length of the MV grid.
    length_lv: :obj:`float`
        Total line length of all LV grids.
    """
    start_time = time()
    logger.info("Start - Length analysis")

    length_total = edisgo_obj.topology.lines_df.length.sum()
    mv_grid = edisgo_obj.topology.mv_grid
    length_mv = mv_grid.lines_df.length.sum()
    length_lv = length_total - length_mv

    logger.info("Total length of lines: {:.2f}km".format(length_total))
    logger.info("Total length of mv lines: {:.2f}km".format(length_mv))
    logger.info("Total length of lv lines: {:.2f}km".format(length_lv))

    logger.info("Finished in {}s".format(time() - start_time))
    return length_total, length_mv, length_lv


def voltage_mapping(
    edisgo_root: EDisGo, edisgo_reduced: EDisGo, busmap_df: DataFrame, timestep: str
) -> Tuple[DataFrame, float]:
    """
    Maps the results of the node voltages between the root object and the reduced
    object. For the mapping the busmap is used. It is calculated the voltage
    difference and the root-mean-square error.

    The calculation is performed for one timestep or the min or max values of the
    node voltages.

    Parameters
    ----------
    edisgo_root: :obj:`EDisGo`
        Unreduced EDisGo object.
    edisgo_reduced: :obj:`EDisGo`
        Reduced EDisGo object.
    busmap_df: :obj:`pd.DataFrame`
        Busmap for the mapping.
    timestep: :obj:`str`
        Select a timestep or 'min' or 'max'.

    Returns
    -------
    voltages_df: :obj:`pd.DataFrame`
        DataFrame with the voltages of the nodes and the differences.
    rms: :obj:`str`
        Calculated root-mean-square error.

    """
    start_time = time()
    logger.info("Start - Voltage mapping")

    if timestep == "min":
        logger.info("Voltage mapping for the minium values")
        v_root = edisgo_root.results.v_res.min()
        v_reduced = edisgo_reduced.results.v_res.min()
    elif timestep == "max":
        logger.info("Voltage mapping for the maximum values")
        v_root = edisgo_root.results.v_res.max()
        v_reduced = edisgo_reduced.results.v_res.max()
    else:
        logger.info("Voltage mapping for timestep {}".format(timestep))
        v_root = edisgo_root.results.v_res.loc[timestep]
        v_reduced = edisgo_reduced.results.v_res.loc[timestep]

    v_root.name = "v_root"
    v_root = v_root.loc[busmap_df.index]

    voltages_df = v_root.to_frame()

    for index, row in voltages_df.iterrows():
        try:
            voltages_df.loc[index, "v_reduced"] = v_reduced.loc[
                busmap_df.loc[index, "new_bus"]
            ]
            voltages_df.loc[index, "new_bus_name"] = busmap_df.loc[index, "new_bus"]
        except KeyError:
            voltages_df.loc[index, "v_reduced"] = v_reduced.loc[
                busmap_df.loc[index, "new_bus"].lstrip("virtual_")
            ]
            voltages_df.loc[index, "new_bus_name"] = busmap_df.loc[
                index, "new_bus"
            ].lstrip("virtual_")
    voltages_df.loc[:, "v_diff"] = (
        voltages_df.loc[:, "v_root"] - voltages_df.loc[:, "v_reduced"]
    )
    rms = np.sqrt(
        mean_squared_error(
            voltages_df.loc[:, "v_root"], voltages_df.loc[:, "v_reduced"]
        )
    )
    logger.info(
        "Root mean square value between edisgo_root "
        "voltages and edisgo_reduced: v_rms = {:.2%}".format(rms)
    )

    logger.info("Finished in {}s".format(time() - start_time))
    return voltages_df, rms


def line_apparent_power_mapping(
    edisgo_root: EDisGo, edisgo_reduced: EDisGo, linemap_df: DataFrame, timestep: str
) -> Tuple[DataFrame, float]:
    """
    Maps the results of the line loads between the root object and the reduced
    object. For the mapping the linemap is used. It is calculated the apparent power
    difference and the root-mean-square error.

    The calculation is performed for one timestep or the min or max values of the
    line loads (apparent power).

    Parameters
    ----------
    edisgo_root: :obj:`EDisGo`
        Unreduced EDisGo object.
    edisgo_reduced: :obj:`EDisGo`
        Reduced EDisGo object.
    linemap_df: :obj:`pd.DataFrame`
        Linemap for the mapping.
    timestep: :obj:`str`
        Select a timestep or 'min' or 'max'.

    Returns
    -------
    s_df: :obj:`pd.DataFrame`
        DataFrame with the apparent power of the lines and the differences.
    rms: :obj:`str`
        Calculated root-mean-square error.

    """
    start_time = time()
    logger.info("Start - Line apparent power mapping")

    if timestep == "min":
        logger.info("Apparent power mapping for the minium values")
        s_root = edisgo_root.results.s_res.min()
        s_reduced = edisgo_reduced.results.s_res.min()
    elif timestep == "max":
        logger.info("Apparent power mapping for the maximum values")
        s_root = edisgo_root.results.s_res.max()
        s_reduced = edisgo_reduced.results.s_res.max()
    else:
        logger.info("Apparent power mapping for timestep {}".format(timestep))
        s_root = edisgo_root.results.s_res.loc[timestep]
        s_reduced = edisgo_reduced.results.s_res.loc[timestep]

    s_root.name = "s_root"
    s_root = s_root.loc[linemap_df.index]

    s_df = s_root.to_frame()

    for index, row in s_df.iterrows():
        s_df.loc[index, "s_reduced"] = s_reduced.loc[linemap_df.loc[index][0]]
        s_df.loc[index, "new_bus_name"] = linemap_df.loc[index][0]
    s_df.loc[:, "s_diff"] = s_df.loc[:, "s_root"] - s_df.loc[:, "s_reduced"]
    rms = np.sqrt(mean_squared_error(s_df.loc[:, "s_root"], s_df.loc[:, "s_reduced"]))
    logger.info(
        "Root mean square value between edisgo_root "
        "s_res and edisgo_reduced: s_rms = {:.2}".format(rms)
    )

    logger.info("Finished in {}s".format(time() - start_time))
    return s_df, rms


# endregion
