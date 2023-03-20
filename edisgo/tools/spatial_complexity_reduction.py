import copy
import logging
import math

from hashlib import md5
from time import time

import networkx as nx
import numpy as np
import pandas as pd

from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.network import timeseries


def hash_df(df):
    s = df.to_json()
    return md5(s.encode()).hexdigest()


# Preprocessing
def remove_one_meter_lines(edisgo_root):
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

    logger = logging.getLogger("edisgo.cr_remove_one_meter_lines")
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

    charging_points_df = edisgo_obj.topology.charging_points_df.copy()
    charging_points_df = charging_points_df.apply(apply_busmap, axis="columns")

    edisgo_obj.topology.lines_df = lines_df
    edisgo_obj.topology.buses_df = buses_df
    edisgo_obj.topology.loads_df = loads_df
    edisgo_obj.topology.generators_df = generators_df
    edisgo_obj.topology.charging_points_df = charging_points_df

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj


def remove_lines_under_one_meter(edisgo_root):
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

    logger = logging.getLogger("edisgo.cr_remove_lines_under_one_meter")
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

    # charging_points_df = edisgo_obj.topology.charging_points_df.copy()
    # charging_points_df = charging_points_df.apply(apply_busmap, axis="columns")
    # edisgo_obj.topology.charging_points_df = charging_points_df

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj


def aggregate_to_bus(edisgo_root, aggregate_charging_points_mode=True):
    def aggregate_loads(df):
        series = pd.Series(index=df.columns, dtype="object")
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_set"] = df.loc[:, "p_set"].sum()
        series.loc["annual_consumption"] = df.loc[:, "annual_consumption"].sum()
        if load_aggregation_mode == "sector":
            series.loc["sector"] = df.loc[:, "sector"].values[0]
        # elif load_aggregation_mode == 'bus':
        #     series.loc['sector'] = 'aggregated'
        series.loc["old_load_name"] = df.index.tolist()
        return series

    def aggregate_generators(df):
        series = pd.Series(index=df.columns, dtype="object")
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_nom"] = df.loc[:, "p_nom"].sum()
        series.loc["control"] = df.loc[:, "control"].values[0]
        series.loc["subtype"] = df.loc[:, "subtype"].values[0]
        series.loc["old_generator_name"] = df.index.tolist()
        series.loc["type"] = df.loc[:, "type"].values[0]
        series.loc["weather_cell_id"] = df.loc[:, "weather_cell_id"].values[0]
        return series

    def extract_weather_cell_id(series):
        if pd.isna(series):
            series = "NaN"
        else:
            series = str(int(series))
        return series

    def aggregate_charging_points(df):
        series = pd.Series(dtype="object")
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_nom"] = df.loc[:, "p_nom"].sum()
        series.loc["use_case"] = df.loc[:, "use_case"].values[0]
        series.loc["old_charging_point_name"] = df.index.tolist()
        return series

    logger = logging.getLogger("edisgo.cr_aggregate_to_bus")
    start_time = time()
    logger.info("Start - Aggregate to bus")

    edisgo_obj = copy.deepcopy(edisgo_root)

    loads_df = edisgo_obj.topology.loads_df.copy()
    generators_df = edisgo_obj.topology.generators_df.copy()
    charging_points_df = edisgo_obj.topology.charging_points_df.copy()

    if not loads_df.empty:
        load_aggregation_mode = "bus"

        logger.info("Aggregate loads_df")

        if load_aggregation_mode == "sector":
            loads_df = loads_df.groupby(by=["bus", "sector"]).apply(aggregate_loads)
            loads_df.index = (
                "Load_" + loads_df.loc[:, "bus"] + "_" + loads_df.loc[:, "sector"]
            )
        elif load_aggregation_mode == "bus":
            loads_df = loads_df.groupby("bus").apply(aggregate_loads)
            loads_df.index = "Load_" + loads_df.loc[:, "bus"]

        loads_df.index.name = "name"
        edisgo_obj.topology.loads_df = loads_df

        # aggregate load timeseries
        logger.info("Aggregate loads timeseries")

        load_name_map_df = edisgo_obj.topology.loads_df.loc[
            :, "old_load_name"
        ].to_dict()
        load_name_map = {}
        for i in range(0, len(load_name_map_df.keys())):
            for j in range(0, len(list(load_name_map_df.values())[i])):
                load_name_map[list(load_name_map_df.values())[i][j]] = list(
                    load_name_map_df.keys()
                )[i]

        timeseries_loads_p_df = edisgo_obj.timeseries.loads_active_power.T.copy()
        timeseries_loads_q_df = edisgo_obj.timeseries.loads_reactive_power.T.copy()

        new_index = []
        for i in range(0, timeseries_loads_p_df.shape[0]):
            new_load_name = load_name_map[timeseries_loads_p_df.index[i]]
            new_index.append(new_load_name)

        old_index = timeseries_loads_p_df.index.tolist()
        rename_index = dict(zip(old_index, new_index))

        timeseries_loads_p_df = timeseries_loads_p_df.rename(index=rename_index)
        timeseries_loads_q_df = timeseries_loads_q_df.rename(index=rename_index)
        edisgo_obj.timeseries.loads_active_power = (
            timeseries_loads_p_df.groupby(level=0).sum().T
        )
        edisgo_obj.timeseries.loads_reactive_power = (
            timeseries_loads_q_df.groupby(level=0).sum().T
        )

    if not generators_df.empty:
        logger.info("Aggregate generators_df")

        generators_df = generators_df.groupby(
            by=["bus", "type", "weather_cell_id"], dropna=False
        ).apply(aggregate_generators)
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

        edisgo_obj.topology.generators_df = generators_df

        logger.info("Aggregate generator timeseries")
        timeseries_generators_p_df = (
            edisgo_obj.timeseries.generators_active_power.T.copy()
        )
        timeseries_generators_q_df = (
            edisgo_obj.timeseries.generators_reactive_power.T.copy()
        )

        generator_name_map_df = edisgo_obj.topology.generators_df.loc[
            :, "old_generator_name"
        ].to_dict()

        generator_name_map = {}
        for i in range(0, len(generator_name_map_df.keys())):
            for j in range(0, len(list(generator_name_map_df.values())[i])):
                generator_name_map[list(generator_name_map_df.values())[i][j]] = list(
                    generator_name_map_df.keys()
                )[i]

        new_index = []
        for i in range(0, timeseries_generators_p_df.shape[0]):
            new_generator_name = generator_name_map[timeseries_generators_p_df.index[i]]
            new_index.append(new_generator_name)

        old_index = timeseries_generators_p_df.index.tolist()
        rename_index = dict(zip(old_index, new_index))

        timeseries_generators_p_df = timeseries_generators_p_df.rename(
            index=rename_index
        )
        timeseries_generators_q_df = timeseries_generators_q_df.rename(
            index=rename_index
        )

        edisgo_obj.timeseries.generators_active_power = (
            timeseries_generators_p_df.groupby(level=0).sum().T
        )
        edisgo_obj.timeseries.generators_reactive_power = (
            timeseries_generators_q_df.groupby(level=0).sum().T
        )

        edisgo_obj.topology.generators_df = generators_df

    if not charging_points_df.empty and aggregate_charging_points_mode:
        logger.info("Aggregate charging_points_df")

        charging_points_df = charging_points_df.groupby(
            by=["bus", "use_case"], dropna=False
        ).apply(aggregate_charging_points)

        edisgo_obj.topology.charging_points_df = charging_points_df

        charging_points_df.index = (
            "ChargingPoint_"
            + charging_points_df.loc[:, "bus"].values
            + "_"
            + charging_points_df.loc[:, "use_case"].values
        )

        logger.info("Aggregate charging points timeseries")
        timeseries_charging_points_p_df = (
            edisgo_obj.timeseries.charging_points_active_power.T.copy()
        )
        timeseries_charging_points_q_df = (
            edisgo_obj.timeseries.charging_points_reactive_power.T.copy()
        )

        charging_point_name_map_df = charging_points_df.loc[
            :, "old_charging_point_name"
        ].to_dict()

        charging_point_name_map = {}
        for i in range(0, len(charging_point_name_map_df.keys())):
            for j in range(0, len(list(charging_point_name_map_df.values())[i])):
                charging_point_name_map[
                    list(charging_point_name_map_df.values())[i][j]
                ] = list(charging_point_name_map_df.keys())[i]

        new_index = []
        for index, row in timeseries_charging_points_p_df.iterrows():
            new_index.append(charging_point_name_map[index])
        old_index = timeseries_charging_points_p_df.index.tolist()
        rename_index = dict(zip(old_index, new_index))

        timeseries_charging_points_p_df = timeseries_charging_points_p_df.rename(
            index=rename_index
        )
        timeseries_charging_points_q_df = timeseries_charging_points_q_df.rename(
            index=rename_index
        )

        timeseries_charging_points_p_df = (
            timeseries_charging_points_p_df.groupby(level=0).sum().T
        )
        timeseries_charging_points_q_df = (
            timeseries_charging_points_q_df.groupby(level=0).sum().T
        )

        edisgo_obj.timeseries.charging_points_active_power = (
            timeseries_charging_points_p_df
        )
        edisgo_obj.timeseries.charging_points_reactive_power = (
            timeseries_charging_points_q_df
        )

        edisgo_obj.topology.charging_points_df = charging_points_df

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj


# Complexity reduction
def make_busmap_from_clustering(
    edisgo_root=None,
    grid=None,
    mode=None,
    reduction_factor=None,
    preserve_trafo_bus_coordinates=True,
    n_init=10,
    random_state=42,
):
    def calculate_weighting(series):
        p_gen = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.bus == series.name, "p_nom"
        ].sum()
        p_load = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.bus == series.name, "p_set"
        ].sum()
        p_charge = edisgo_obj.topology.charging_points_df.loc[
            edisgo_obj.topology.charging_points_df.bus == series.name, "p_set"
        ].sum()
        if str(grid).split("_")[0] == "MVGrid":
            s_tran = edisgo_obj.topology.transformers_df.loc[
                edisgo_obj.topology.transformers_df.bus0 == series.name, "s_nom"
            ].sum()
        else:
            s_tran = 0
        series.loc["weight"] = 1 + 1000 * (p_gen + p_load + s_tran + p_charge)
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

    logger = logging.getLogger("edisgo.cr_make_busmap")
    start_time = time()
    logger.info("Start - Make busmap from clustering, mode = {}".format(mode))

    edisgo_obj = copy.deepcopy(edisgo_root)
    grid_list = make_grid_list(edisgo_obj, grid)

    busmap_df = pd.DataFrame()

    for grid in grid_list:
        v_grid = grid.nominal_voltage
        logger.debug("Make busmap for grid: {}, v_nom={}".format(grid, v_grid))

        buses_df = grid.buses_df
        graph = grid.graph
        transformer_bus = grid.transformers_df.bus1[0]

        buses_df = buses_df.apply(transform_coordinates, axis="columns")
        buses_df = buses_df.apply(calculate_weighting, axis="columns")

        number_of_distinct_nodes = buses_df.groupby(by=["x", "y"]).first().shape[0]
        logger.debug("Number_of_distinct_nodes = " + str(number_of_distinct_nodes))
        n_clusters = math.ceil(number_of_distinct_nodes * reduction_factor)
        logger.debug("n_clusters = {}".format(n_clusters))

        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)

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
            partial_busmap_df = rename_virtual_buses(
                logger, partial_busmap_df, transformer_bus
            )

        partial_busmap_df = partial_busmap_df.apply(
            transform_coordinates_back, axis="columns"
        )

        busmap_df = pd.concat([busmap_df, partial_busmap_df])

    logger.info("Finished in {}s".format(time() - start_time))
    return busmap_df


def make_busmap_from_feeders(
    edisgo_root=None,
    grid=None,
    mode=None,
    reduction_factor=None,
    reduction_factor_not_focused=0,
):
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
        p_charge = edisgo_obj.topology.charging_points_df.loc[
            edisgo_obj.topology.charging_points_df.bus.isin(buses), "p_set"
        ].sum()
        if str(grid).split("_")[0] == "MVGrid":
            s_tran = edisgo_obj.topology.transformers_df.loc[
                edisgo_obj.topology.transformers_df.bus0 == series.name, "s_nom"
            ].sum()
        else:
            s_tran = 0
        series.loc["weight"] = 1 + 1000 * (p_gen + p_load + s_tran + p_charge)
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
    logger = logging.getLogger("edisgo.cr_make_busmap")
    start_time = time()
    logger.info("Start - Make busmap from feeders, mode = {}".format(mode))

    edisgo_obj.topology.buses_df = edisgo_obj.topology.buses_df.apply(
        transform_coordinates, axis="columns"
    )

    grid_list = make_grid_list(edisgo_obj, grid)
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
                logger, partial_busmap_df, transformer_node
            )

        busmap_df = pd.concat([busmap_df, partial_busmap_df])

    busmap_df = busmap_df.apply(transform_coordinates_back, axis="columns")
    busmap_df.sort_index(inplace=True)
    logger.info("Finished in {}s".format(time() - start_time))
    return busmap_df


def make_busmap_from_main_feeders(
    edisgo_root=None,
    grid=None,
    mode=None,
    reduction_factor=None,
    reduction_factor_not_focused=0,
):
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
        p_charge = edisgo_obj.topology.charging_points_df.loc[
            edisgo_obj.topology.charging_points_df.bus.isin(buses), "p_set"
        ].sum()
        if str(grid).split("_")[0] == "MVGrid":
            s_tran = edisgo_obj.topology.transformers_df.loc[
                edisgo_obj.topology.transformers_df.bus0 == series.name, "s_nom"
            ].sum()
        else:
            s_tran = 0
        series.loc["weight"] = 1 + 1000 * (p_gen + p_load + s_tran + p_charge)
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
    logger = logging.getLogger("edisgo.cr_make_busmap")
    start_time = time()
    logger.info("Start - Make busmap from main feeders, mode = {}".format(mode))

    if mode != "aggregate_to_longest_feeder":
        edisgo_obj.topology.buses_df = edisgo_obj.topology.buses_df.apply(
            transform_coordinates, axis="columns"
        )

    grid_list = make_grid_list(edisgo_obj, grid)
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
            if mode != "aggregate_to_longest_feeder":
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

        if mode != "aggregate_to_longest_feeder":
            # Backmap
            for node in not_main_nodes:
                partial_busmap_df.loc[node] = partial_busmap_df.loc[
                    partial_busmap_df.loc[node, "new_bus"]
                ]

        if str(grid).split("_")[0] == "MVGrid":
            partial_busmap_df = rename_virtual_buses(
                logger, partial_busmap_df, transformer_node
            )

        busmap_df = pd.concat([busmap_df, partial_busmap_df])

    if mode != "aggregate_to_longest_feeder":
        busmap_df = busmap_df.apply(transform_coordinates_back, axis="columns")

    logger.info("Finished in {}s".format(time() - start_time))
    return busmap_df


def make_busmap(
    edisgo_root,
    mode=None,
    cluster_area=None,
    reduction_factor=None,
    reduction_factor_not_focused=None,
    grid=None,
):
    # Check for false input.
    if not 0 < reduction_factor < 1.0:
        raise ValueError("Reduction factor must bigger than 0 and smaller than 1.")
    if mode not in [
        "aggregate_to_main_feeder",
        "kmeans",
        "kmeansdijkstra",
        "equidistant_nodes",
    ]:
        raise ValueError(f"Selected false {mode=}.")
    if (reduction_factor_not_focused is not False) and not (
        0 <= reduction_factor_not_focused < 1.0
    ):
        raise ValueError(
            f"{reduction_factor_not_focused=}, should be 'False' "
            f"or 0 or bigger than 0 but smaller than 1."
        )

    if cluster_area == "grid":
        busmap_df = make_busmap_from_clustering(
            edisgo_root=edisgo_root,
            mode=mode,
            grid=grid,
            reduction_factor=reduction_factor,
        )
    elif cluster_area == "feeder":
        busmap_df = make_busmap_from_feeders(
            edisgo_root=edisgo_root,
            grid=grid,
            mode=mode,
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=reduction_factor_not_focused,
        )
    elif cluster_area == "main_feeder":
        busmap_df = make_busmap_from_main_feeders(
            edisgo_root=edisgo_root,
            grid=grid,
            mode=mode,
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=reduction_factor_not_focused,
        )
    else:
        raise ValueError(f"Selected false {cluster_area=}!")

    return busmap_df


def make_remaining_busmap(busmap_df, edisgo_root):
    logger = logging.getLogger("edisgo.cr_make_remaining_busmap")
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


def reduce_edisgo(edisgo_root, busmap_df, aggregation_mode=True):
    logger = logging.getLogger("edisgo.cr_reduce_edisgo")
    start_time = time()
    logger.info("Start - Reducing edisgo object")

    edisgo_obj = copy.deepcopy(edisgo_root)

    def apply_busmap_df(series):
        series.loc["bus"] = busmap_df.loc[series.name, "new_bus"]
        series.loc["x"] = busmap_df.loc[series.name, "new_x"]
        series.loc["y"] = busmap_df.loc[series.name, "new_y"]
        return series

    # preserve data
    logger.info("Preserve data")
    buses_df = edisgo_obj.topology.buses_df
    lines_df = edisgo_obj.topology.lines_df
    loads_changed_df = edisgo_obj.topology.loads_df
    generators_changed_df = edisgo_obj.topology.generators_df
    # charging_points_df = edisgo_obj.topology.charging_points_df
    slack_bus = edisgo_obj.topology.transformers_hvmv_df.bus1[0]

    # manipulate buses_df
    logger.info("Manipulate buses_df")
    buses_df = buses_df.apply(apply_busmap_df, axis="columns")
    buses_df = buses_df.groupby(by=["bus"], dropna=False, as_index=False).first()

    buses_df.loc[:, "in_building"] = False
    buses_df = buses_df.set_index("bus")
    edisgo_obj.topology.buses_df = buses_df

    # manipulate lines_df
    def apply_busmap_df(series):
        series.loc["bus0"] = busmap_df.loc[series.bus0, "new_bus"]
        series.loc["bus1"] = busmap_df.loc[series.bus1, "new_bus"]
        return series

    def remove_lines_with_the_same_bus(series):
        if series.bus0 == series.bus1:
            return
        elif (
            series.bus0.split("_")[0] == "virtual"
            and series.bus0.lstrip("virtual_") == slack_bus
        ) or (
            series.bus1.split("_")[0] == "virtual"
            and series.bus1.lstrip("virtual_") == slack_bus
        ):
            logger.debug(
                "Drop line because it is connected to "
                "the virtual_slack bus \n{}".format(series.name)
            )

            return
        elif series.bus0.lstrip("virtual_") == series.bus1.lstrip("virtual_"):
            logger.debug(
                "Drop line because it shorts the circuit breaker \n{}".format(
                    series.name
                )
            )

            return
        else:
            return series

    def aggregate_lines_df(df):
        series = pd.Series(index=edisgo_obj.topology.lines_df.columns, dtype="object")

        bus0 = df.loc[:, "bus0"].values[0]
        bus1 = df.loc[:, "bus1"].values[0]
        v_nom = buses_df.loc[bus0, "v_nom"]

        coordinates_bus0 = edisgo_obj.topology.buses_df.loc[
            bus0, ["x", "y"]
        ].values.tolist()
        coordinates_bus1 = edisgo_obj.topology.buses_df.loc[
            bus1, ["x", "y"]
        ].values.tolist()

        coordinates_bus0 = coor_transform.transform(
            coordinates_bus0[0], coordinates_bus0[1]
        )
        coordinates_bus1 = coor_transform.transform(
            coordinates_bus1[0], coordinates_bus1[1]
        )

        length = (
            math.dist(coordinates_bus0, coordinates_bus1)
            / 1000
            * edisgo_obj.config["grid_connection"]["branch_detour_factor"]
        )

        if length == 0:
            length = 0.001
            logger.warning(
                "Length of line between "
                + str(bus0)
                + " and "
                + str(bus1)
                + " can't be 0, set to 1m"
            )
        if length < 0.001:
            logger.warning(
                "WARNING: Length of line between "
                + str(bus0)
                + " and "
                + str(bus1)
                + " smaller than 1m"
            )

        if np.isnan(edisgo_obj.topology.buses_df.loc[df.bus0, "lv_grid_id"])[0]:
            # voltage_level = 'MV'
            # type_cable = 'mv_cables'
            type_line = "mv_line"
        else:
            # voltage_level = 'LV'
            # type_cable = 'lv_cables'
            type_line = "lv_line"

        if len(df["type_info"].values) > 1:
            # type_info = 'Combined: '
            # for x in l['type_info'].values:
            #    type_info = type_info + str(x) + ' '
            type_info = edisgo_obj.config["grid_expansion_standard_equipment"][
                type_line
            ]
        else:
            type_info = df["type_info"].values[0]

        if len(df["kind"].values) > 1:
            # kind = 'Combined: '
            # for x in l['kind'].values:
            #    kind = kind + str(x) + ' '
            kind = df["kind"].values[0]
        else:
            kind = df["kind"].values[0]

        x_sum = 0
        for line_type in df["type_info"].values:
            x_sum = (
                x_sum
                + 1
                / line_data_df.loc[line_data_df.U_n.isin([v_nom])].loc[
                    line_type, "L_per_km"
                ]
            )
        x_sum = 1 / x_sum
        x = length * 2 * math.pi * 50 * x_sum / 1000

        r_sum = 0
        for line_type in df["type_info"].values:
            r_sum = (
                r_sum
                + 1
                / line_data_df.loc[line_data_df.U_n.isin([v_nom])].loc[
                    line_type, "R_per_km"
                ]
            )
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

    if not lines_df.empty:
        logger.info("Manipulate lines_df")
        line_data_df = pd.concat(
            [
                edisgo_obj.topology.equipment_data["mv_overhead_lines"],
                edisgo_obj.topology.equipment_data["mv_cables"],
                edisgo_obj.topology.equipment_data["lv_cables"],
            ]
        )

        lines_df = lines_df.apply(apply_busmap_df, axis=1)

        lines_df = lines_df.apply(
            remove_lines_with_the_same_bus, axis=1, result_type="broadcast"
        ).dropna()

        order = lines_df.bus0 < lines_df.bus1
        lines_df_p = lines_df[order]
        lines_df_n = lines_df[~order].rename(columns={"bus0": "bus1", "bus1": "bus0"})
        lines_df = pd.concat([lines_df_p, lines_df_n], sort=True)

        lines_df = lines_df.groupby(by=["bus0", "bus1"]).apply(aggregate_lines_df)

        lines_df.index = (
            "Line_" + lines_df.loc[:, "bus0"] + "_to_" + lines_df.loc[:, "bus1"]
        )

        edisgo_obj.topology.lines_df = lines_df

    load_aggregation_mode = "bus"

    # aggregate loads
    def apply_busmap(series):
        series.loc["bus"] = busmap_df.loc[series.loc["bus"], "new_bus"]
        return series

    def aggregate_loads(df):
        series = pd.Series(index=df.columns, dtype="object")  # l.values[0],
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_set"] = df.loc[:, "p_set"].sum()
        series.loc["annual_consumption"] = df.loc[:, "annual_consumption"].sum()
        if load_aggregation_mode == "sector":
            series.loc["sector"] = df.loc[:, "sector"].values[0]
        # elif load_aggregation_mode == 'bus':
        #     series.loc['sector'] = 'aggregated'
        series.loc["old_load_name"] = df.index.tolist()
        return series

    if not loads_changed_df.empty:
        logger.info("Manipulate loads")
        loads_changed_df = loads_changed_df.apply(apply_busmap, axis="columns")
        if aggregation_mode:
            if load_aggregation_mode == "sector":
                loads_changed_df = loads_changed_df.groupby(by=["bus", "sector"]).apply(
                    aggregate_loads
                )
                loads_changed_df.index = (
                    "Load_"
                    + loads_changed_df.loc[:, "bus"]
                    + "_"
                    + loads_changed_df.loc[:, "sector"]
                )
            elif load_aggregation_mode == "bus":
                loads_changed_df = loads_changed_df.groupby("bus").apply(
                    aggregate_loads
                )
                loads_changed_df.index = "Load_" + loads_changed_df.loc[:, "bus"]

        edisgo_obj.topology.loads_df = loads_changed_df
        if aggregation_mode:
            edisgo_obj.topology.loads_df.index.name = "name"

            # aggregate load timeseries
            load_name_map_df = edisgo_obj.topology.loads_df.loc[
                :, "old_load_name"
            ].to_dict()
            load_name_map = {}
            for i in range(0, len(load_name_map_df.keys())):
                for j in range(0, len(list(load_name_map_df.values())[i])):
                    load_name_map[list(load_name_map_df.values())[i][j]] = list(
                        load_name_map_df.keys()
                    )[i]
            # return load_name_map

            timeseries_loads_p_df = edisgo_obj.timeseries.loads_active_power.T
            timeseries_loads_q_df = edisgo_obj.timeseries.loads_reactive_power.T

            new_index = []
            for i in range(0, timeseries_loads_p_df.shape[0]):
                new_load_name = load_name_map[timeseries_loads_p_df.index[i]]
                new_index.append(new_load_name)

            old_index = timeseries_loads_p_df.index.tolist()
            rename_index = dict(zip(old_index, new_index))

            timeseries_loads_p_df = timeseries_loads_p_df.rename(index=rename_index)
            timeseries_loads_q_df = timeseries_loads_q_df.rename(index=rename_index)
            edisgo_obj.timeseries.loads_active_power = (
                timeseries_loads_p_df.groupby(level=0).sum().T
            )
            edisgo_obj.timeseries.loads_reactive_power = (
                timeseries_loads_q_df.groupby(level=0).sum().T
            )

    # aggregate generators
    generator_aggregation_mode = "type"

    def apply_busmap(series):
        series.loc["bus"] = busmap_df.loc[series.loc["bus"], "new_bus"]
        return series

    def aggregate_generators_df(df):
        series = pd.Series(index=df.columns, dtype="object")
        series.loc["bus"] = df.loc[:, "bus"].values[0]
        series.loc["p_nom"] = df.loc[:, "p_nom"].sum()
        series.loc["control"] = df.loc[:, "control"].values[0]
        series.loc["subtype"] = df.loc[:, "subtype"].values[0]
        series.loc["old_generator_name"] = df.index.tolist()
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

    if not generators_changed_df.empty:
        logger.info("Manipulate generators")
        generators_changed_df = generators_changed_df.loc[
            generators_changed_df.loc[:, "bus"].isin(busmap_df.index), :
        ]
        generators_changed_df = generators_changed_df.apply(
            apply_busmap, axis="columns"
        )

        if aggregation_mode:
            if generator_aggregation_mode == "bus":
                generators_changed_df = generators_changed_df.groupby("bus").apply(
                    aggregate_generators_df
                )
                generators_changed_df.index = (
                    "Generator_" + generators_changed_df.loc[:, "bus"]
                )
            elif generator_aggregation_mode == "type":
                generators_changed_df = generators_changed_df.groupby(
                    by=["bus", "type", "weather_cell_id"], dropna=False
                ).apply(aggregate_generators_df)
                generators_changed_df.index = (
                    "Generator_"
                    + generators_changed_df.loc[:, "bus"].values
                    + "_"
                    + generators_changed_df.loc[:, "type"].values
                    + "_weather_cell_id_"
                    + generators_changed_df.loc[:, "weather_cell_id"]
                    .apply(extract_weather_cell_id)
                    .values
                )

        edisgo_obj.topology.generators_df = generators_changed_df
        if aggregation_mode:
            timeseries_generators_p_df = edisgo_obj.timeseries.generators_active_power.T
            timeseries_generators_q_df = (
                edisgo_obj.timeseries.generators_reactive_power.T
            )

            generator_name_map_df = edisgo_obj.topology.generators_df.loc[
                :, "old_generator_name"
            ].to_dict()
            # return generator_name_map_df
            generator_name_map = {}
            for i in range(0, len(generator_name_map_df.keys())):
                for j in range(0, len(list(generator_name_map_df.values())[i])):
                    generator_name_map[
                        list(generator_name_map_df.values())[i][j]
                    ] = list(generator_name_map_df.keys())[i]

            # return generator_name_map
            new_index = []
            for i in range(0, timeseries_generators_p_df.shape[0]):
                new_generator_name = generator_name_map[
                    timeseries_generators_p_df.index[i]
                ]
                new_index.append(new_generator_name)

            old_index = timeseries_generators_p_df.index.tolist()
            rename_index = dict(zip(old_index, new_index))

            timeseries_generators_p_df = timeseries_generators_p_df.rename(
                index=rename_index
            )
            timeseries_generators_q_df = timeseries_generators_q_df.rename(
                index=rename_index
            )

            edisgo_obj.timeseries.generators_active_power = (
                timeseries_generators_p_df.groupby(level=0).sum().T
            )

            edisgo_obj.timeseries.generators_reactive_power = (
                timeseries_generators_q_df.groupby(level=0).sum().T
            )

    # if not charging_points_df.empty:
    #     logger.info("Manipulate charging points")
    #
    #     def aggregate_charging_points_df(df):
    #         series = pd.Series(dtype="object")
    #         series.loc["bus"] = df.loc[:, "bus"].values[0]
    #         series.loc["p_set"] = df.loc[:, "p_set"].sum()
    #         series.loc["use_case"] = df.loc[:, "use_case"].values[0]
    #         series.loc["old_charging_point_name"] = df.index.tolist()
    #         return series
    #
    #     charging_points_df = edisgo_obj.topology.charging_points_df
    #     charging_points_df = charging_points_df.apply(apply_busmap, axis="columns")
    #
    #     if aggregate_charging_points_mode:
    #         charging_points_df = charging_points_df.groupby(
    #             by=["bus", "use_case"], dropna=False
    #         ).apply(aggregate_charging_points_df)
    #
    #     edisgo_obj.topology.charging_points_df = charging_points_df
    #
    #     if aggregate_charging_points_mode:
    #         charging_points_df.index = (
    #             "ChargingPoint_"
    #             + charging_points_df.loc[:, "bus"].values
    #             + "_"
    #             + charging_points_df.loc[:, "use_case"].values
    #         )
    #
    #         timeseries_charging_points_p_df = (
    #             edisgo_obj.timeseries.charging_points_active_power.T
    #         )
    #         timeseries_charging_points_q_df = (
    #             edisgo_obj.timeseries.charging_points_reactive_power.T
    #         )
    #
    #         charging_point_name_map_df = charging_points_df.loc[
    #             :, "old_charging_point_name"
    #         ].to_dict()
    #
    #         charging_point_name_map = {}
    #         for i in range(0, len(charging_point_name_map_df.keys())):
    #             for j in range(0, len(list(charging_point_name_map_df.values())[i])):
    #                 charging_point_name_map[
    #                     list(charging_point_name_map_df.values())[i][j]
    #                 ] = list(charging_point_name_map_df.keys())[i]
    #
    #         new_index = []
    #         for index in timeseries_charging_points_p_df.index.tolist():
    #             new_index.append(charging_point_name_map[index])
    #
    #         old_index = timeseries_charging_points_p_df.index.tolist()
    #         rename_index = dict(zip(old_index, new_index))
    #
    #         timeseries_charging_points_p_df = timeseries_charging_points_p_df.rename(
    #             index=rename_index
    #         )
    #         timeseries_charging_points_q_df = timeseries_charging_points_q_df.rename(
    #             index=rename_index
    #         )
    #
    #         timeseries_charging_points_p_df = (
    #             timeseries_charging_points_p_df.groupby(level=0).sum().T
    #         )
    #         timeseries_charging_points_q_df = (
    #             timeseries_charging_points_q_df.groupby(level=0).sum().T
    #         )
    #
    #         edisgo_obj.timeseries.charging_points_active_power = (
    #             timeseries_charging_points_p_df
    #         )
    #         edisgo_obj.timeseries.charging_points_reactive_power = (
    #             timeseries_charging_points_q_df
    #         )

    # apply busmap on transformers_df
    logger.info("Manipulate transformers_df")

    def apply_busmap(series):
        series.loc["bus0"] = busmap_df.loc[series.loc["bus0"], "new_bus"]
        series.loc["bus1"] = busmap_df.loc[series.loc["bus1"], "new_bus"]
        return series

    transformers_df = edisgo_obj.topology.transformers_df
    transformers_df = transformers_df.apply(apply_busmap, axis="columns")
    edisgo_obj.topology.transformers_df = transformers_df

    # manipulate switches_df
    logger.info("Manipulate switches_df")
    switches_df = edisgo_obj.topology.switches_df

    # drop switches unused switches
    switches_to_drop = []
    for index, new_bus in zip(busmap_df.index, busmap_df.new_bus):
        if (index.split("_")[0] == "virtual") and (new_bus.split("_")[0] != "virtual"):
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

    # drop switches which are not connected to any lines
    # switches_to_drop = []
    # for index, row in switches_df.iterrows():
    #     if ~(lines_df.bus0.isin([row.bus_open]).any()
    #          or lines_df.bus1.isin([row.bus_open]).any()):
    #         switches_to_drop.append(index)
    #
    # if len(switches_to_drop) > 0:
    #      logger.info('Drop switches which are '
    #                  'not connected to any lines: {}'.format(switches_to_drop))
    #     switches_df = switches_df.drop(switches_to_drop)

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

    # write switches_df into object
    edisgo_obj.topology.switches_df = switches_df

    # drop isolated nodes
    # G = edisgo_obj.topology.mv_grid.graph
    # isolated_nodes = list(nx.algorithms.isolate.isolates(G))
    # if len(isolated_nodes) > 0:
    #     logger.warning(
    #         "The following isolated nodes are droped: {}".format(isolated_nodes)
    #     )
    #     edisgo_obj.topology.buses_df = (
    #         edisgo_obj.topology.buses_df.drop(isolated_nodes)
    #     )

    # make line_map_df
    logger.info("Make line_map_df")
    linemap_df = pd.DataFrame()
    for new_line_name, old_line_names in zip(lines_df.index, lines_df.old_line_name):
        for old_line_name in old_line_names:
            linemap_df.loc[old_line_name, "new_line_name"] = new_line_name

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj, linemap_df


# Postprocessing
def save_results_reduced_to_min_max(edisgo_root, edisgo_object_name):
    edisgo_obj = copy.deepcopy(edisgo_root)

    def min_max(df):
        min_df = df.min()
        min_df.name = "min"
        max_df = df.max()
        max_df.name = "max"

        df = pd.concat([min_df, max_df], axis=1)
        df = df.T
        return df

    logger = logging.getLogger("edisgo.cr_reduce_results_to_min_max")
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
def length_analysis(edisgo_obj):
    logger = logging.getLogger("edisgo.cr_length_analysis")
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


def voltage_mapping(edisgo_root, edisgo_reduced, busmap_df, timestep):
    logger = logging.getLogger("edisgo.cr_voltage_mapping")
    start_time = time()
    logger.info("Start - Voltage mapping")

    if timestep == "min":
        logger.info("Voltage mapping for the minium values.")
        v_root = edisgo_root.results.v_res.min()
        v_reduced = edisgo_reduced.results.v_res.min()
    elif timestep == "max":
        logger.info("Voltage mapping for the maximum values.")
        v_root = edisgo_root.results.v_res.max()
        v_reduced = edisgo_reduced.results.v_res.max()
    else:
        logger.info("Voltage mapping for timestep {}.".format(timestep))
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


def line_apparent_power_mapping(edisgo_root, edisgo_reduced, linemap_df, timestep):
    logger = logging.getLogger("edisgo.cr_line_apparent_power_mapping")
    start_time = time()
    logger.info("Start - Line apparent power mapping")

    if timestep == "min":
        logger.info("Apparent power mapping for the minium values.")
        s_root = edisgo_root.results.s_res.min()
        s_reduced = edisgo_reduced.results.s_res.min()
    elif timestep == "max":
        logger.info("Apparent power mapping for the maximum values.")
        s_root = edisgo_root.results.s_res.max()
        s_reduced = edisgo_reduced.results.s_res.max()
    else:
        logger.info("Apparent power mapping for timestep {}.".format(timestep))
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


# Functions for other functions
coor_transform = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
coor_transform_back = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def make_grid_list(edisgo_obj, grid):
    if edisgo_obj is None and grid is None:
        raise ValueError("Pass an Grid")
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


def find_buses_of_interest(edisgo_root):
    logger = logging.getLogger("edisgo.cr_find_buses_of_interest")
    start_time = time()
    logger.info("Start - Find buses of interest")

    edisgo_wc = copy.deepcopy(edisgo_root)
    edisgo_wc.timeseries = timeseries.TimeSeries()
    edisgo_wc.timeseries.set_worst_case(edisgo_wc, ["feed-in_case", "load_case"])
    edisgo_wc.analyze()

    buses_of_interest = set()
    mv_lines = checks.mv_line_load(edisgo_wc)
    lv_lines = checks.lv_line_load(edisgo_wc)
    lines = mv_lines.index.tolist()
    lines = lines + lv_lines.index.tolist()
    for line in lines:
        buses_of_interest.add(edisgo_wc.topology.lines_df.loc[line, "bus0"])
        buses_of_interest.add(edisgo_wc.topology.lines_df.loc[line, "bus1"])

    mv_buses = checks.mv_voltage_deviation(edisgo_wc, voltage_levels="mv")
    for value in mv_buses.values():
        buses_of_interest.update(value.index.tolist())

    lv_buses = checks.lv_voltage_deviation(edisgo_wc, voltage_levels="lv")
    for value in lv_buses.values():
        buses_of_interest.update(value.index.tolist())

    logger.info("Finished in {}s".format(time() - start_time))
    # Sort for deterministic reasons
    buses_of_interest = list(buses_of_interest)
    buses_of_interest.sort()
    return buses_of_interest


def rename_virtual_buses(logger, partial_busmap_df, transformer_node):
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


def spatial_complexity_reduction(
    edisgo_root,
    mode=None,
    cluster_area=None,
    reduction_factor=None,
    reduction_factor_not_focused=None,
):
    edisgo_obj = copy.deepcopy(edisgo_root)
    # edisgo_obj.results.equipment_changes = pd.DataFrame()

    busmap_df = make_busmap(
        mode=mode,
        cluster_area=cluster_area,
        reduction_factor=reduction_factor,
        reduction_factor_not_focused=reduction_factor_not_focused,
    )
    edisgo_reduced, linemap_df = reduce_edisgo(
        edisgo_obj, busmap_df, aggregation_mode=False
    )

    return edisgo_reduced, busmap_df, linemap_df
