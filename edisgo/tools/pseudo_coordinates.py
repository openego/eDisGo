from __future__ import annotations

import copy
import logging
import math

from time import time
from typing import TYPE_CHECKING

import networkx as nx

from pyproj import Transformer

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)

# Transform coordinates to equidistant and back
coor_transform = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
coor_transform_back = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


# Pseudo coordinates
def make_coordinates(graph_root, branch_detour_factor=1.3):
    # EDisGo().config["grid_connection"]["branch_detour_factor"]):
    def coordinate_source(pos_start, length, node_numerator, node_total_numerator):
        length = length / branch_detour_factor
        angle = node_numerator * 360 / node_total_numerator
        x0, y0 = pos_start
        x1 = x0 + 1000 * length * math.cos(math.radians(angle))
        y1 = y0 + 1000 * length * math.sin(math.radians(angle))
        pos_end = (x1, y1)
        origin_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        return pos_end, origin_angle

    def coordinate_branch(
        pos_start, angle_offset, length, node_numerator, node_total_numerator
    ):
        length = length / branch_detour_factor
        angle = node_numerator * 180 / (node_total_numerator + 1) + angle_offset - 90
        x0, y0 = pos_start
        x1 = x0 + 1000 * length * math.cos(math.radians(angle))
        y1 = y0 + 1000 * length * math.sin(math.radians(angle))
        origin_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        pos_end = (x1, y1)
        return pos_end, origin_angle

    def coordinate_longest_path(pos_start, angle_offset, length):
        length = length / branch_detour_factor
        angle = angle_offset
        x0, y0 = pos_start
        x1 = x0 + 1000 * length * math.cos(math.radians(angle))
        y1 = y0 + 1000 * length * math.sin(math.radians(angle))
        origin_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        pos_end = (x1, y1)
        return pos_end, origin_angle

    def coordinate_longest_path_neighbor(pos_start, angle_offset, length, direction):
        length = length / branch_detour_factor
        if direction:
            angle_random_offset = 90
        else:
            angle_random_offset = -90
        angle = angle_offset + angle_random_offset
        x0, y0 = pos_start
        x1 = x0 + 1000 * length * math.cos(math.radians(angle))
        y1 = y0 + 1000 * length * math.sin(math.radians(angle))
        origin_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        pos_end = (x1, y1)

        return pos_end, origin_angle

    start_node = list(nx.nodes(graph_root))[0]
    graph_root.nodes[start_node]["pos"] = (0, 0)
    graph_copy = graph_root.copy()

    long_paths = []
    next_nodes = []

    for i in range(0, len(list(nx.neighbors(graph_root, start_node)))):
        path_length_to_transformer = []
        for node in graph_copy.nodes():
            try:
                paths = list(nx.shortest_simple_paths(graph_copy, start_node, node))
            except nx.NetworkXNoPath:
                paths = [[]]
            path_length_to_transformer.append(len(paths[0]))
        index = path_length_to_transformer.index(max(path_length_to_transformer))
        path_to_max_distance_node = list(
            nx.shortest_simple_paths(
                graph_copy, start_node, list(nx.nodes(graph_copy))[index]
            )
        )[0]
        path_to_max_distance_node.remove(start_node)
        graph_copy.remove_nodes_from(path_to_max_distance_node)
        for node in path_to_max_distance_node:
            long_paths.append(node)

    path_to_max_distance_node = long_paths
    n = 0

    for node in list(nx.neighbors(graph_root, start_node)):
        n = n + 1
        pos, origin_angle = coordinate_source(
            graph_root.nodes[start_node]["pos"],
            graph_root.edges[start_node, node]["length"],
            n,
            len(list(nx.neighbors(graph_root, start_node))),
        )
        graph_root.nodes[node]["pos"] = pos
        graph_root.nodes[node]["origin_angle"] = origin_angle
        next_nodes.append(node)

    graph_copy = graph_root.copy()
    graph_copy.remove_node(start_node)
    while graph_copy.number_of_nodes() > 0:
        next_node = next_nodes[0]
        n = 0
        for node in list(nx.neighbors(graph_copy, next_node)):
            n = n + 1
            if node in path_to_max_distance_node:
                pos, origin_angle = coordinate_longest_path(
                    graph_root.nodes[next_node]["pos"],
                    graph_root.nodes[next_node]["origin_angle"],
                    graph_root.edges[next_node, node]["length"],
                )
            elif next_node in path_to_max_distance_node:
                direction = math.fmod(
                    len(
                        list(
                            nx.shortest_simple_paths(graph_root, start_node, next_node)
                        )[0]
                    ),
                    2,
                )
                pos, origin_angle = coordinate_longest_path_neighbor(
                    graph_root.nodes[next_node]["pos"],
                    graph_root.nodes[next_node]["origin_angle"],
                    graph_root.edges[next_node, node]["length"],
                    direction,
                )
            else:
                pos, origin_angle = coordinate_branch(
                    graph_root.nodes[next_node]["pos"],
                    graph_root.nodes[next_node]["origin_angle"],
                    graph_root.edges[next_node, node]["length"],
                    n,
                    len(list(nx.neighbors(graph_copy, next_node))),
                )

            graph_root.nodes[node]["pos"] = pos
            graph_root.nodes[node]["origin_angle"] = origin_angle
            next_nodes.append(node)

        graph_copy.remove_node(next_node)
        next_nodes.remove(next_node)

    return graph_root


def make_pseudo_coordinates(
    edisgo_root: EDisGo, mv_coordinates: bool = False
) -> EDisGo:
    """
    Generates pseudo coordinates for grids.

    Parameters
    ----------
    edisgo_root : :class:`~.EDisGo`
        eDisGo Object
    mv_coordinates : bool, optional
        If True pseudo coordinates are also generated for mv_grid.
        Default: False
    Returns
    -------
    edisgo_object : :class:`~.EDisGo`
        eDisGo object with coordinates for all nodes

    """
    start_time = time()
    logger.info(
        "Start - Making pseudo coordinates for grid: {}".format(
            str(edisgo_root.topology.mv_grid)
        )
    )

    edisgo_obj = copy.deepcopy(edisgo_root)

    grids = list(edisgo_obj.topology.mv_grid.lv_grids)
    if mv_coordinates:
        grids = [edisgo_obj.topology.mv_grid] + grids

    for grid in grids:
        logger.debug("Make pseudo coordinates for: {}".format(grid))
        G = grid.graph
        x0, y0 = G.nodes[list(nx.nodes(G))[0]]["pos"]
        G = make_coordinates(G)
        x0, y0 = coor_transform.transform(x0, y0)
        for node in G.nodes():
            x, y = G.nodes[node]["pos"]
            x, y = coor_transform_back.transform(x + x0, y + y0)
            edisgo_obj.topology.buses_df.loc[node, "x"] = x
            edisgo_obj.topology.buses_df.loc[node, "y"] = y

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj


def make_pseudo_coordinates_graph(G):
    """
    Generates pseudo coordinates for one graph.

    Parameters
    ----------
    edisgo_root : :class:`~.EDisGo`
        eDisGo Object
    mv_coordinates : bool, optional
        If True pseudo coordinates are also generated for mv_grid.
        Default: False
    Returns
    -------
    edisgo_object : :class:`~.EDisGo`
        eDisGo object with coordinates for all nodes

    """
    start_time = time()
    logger.info("Start - Making pseudo coordinates for graph")

    x0, y0 = G.nodes[list(nx.nodes(G))[0]]["pos"]
    G = make_coordinates(G)
    x0, y0 = coor_transform.transform(x0, y0)
    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        G.nodes[node]["pos"] = coor_transform_back.transform(x + x0, y + y0)

    logger.info("Finished in {}s".format(time() - start_time))
    return G
