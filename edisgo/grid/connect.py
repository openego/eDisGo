from ..grid.components import Line, MVStation, LVStation, MVDisconnectingPoint, Generator, Load, BranchTee
from ..grid.tools import select_cable
from ..tools.geo import calc_geo_dist_vincenty, \
                        calc_geo_lines_in_buffer, \
                        proj2equidistant, \
                        proj2conformal

import networkx as nx
import random

import os
if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString
    from shapely.ops import transform

import logging
logger = logging.getLogger('edisgo')


def connect_generators(network):
    """Connect generators to existing grids.

    This function searches for unconnected generators in MV and LV grids and connects them.

    Steps:
        * Step 1: Connect MV generators in MV grid
        * Step 2: Connect LV generators in all LV grids

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    """

    # get params from config
    buffer_radius = int(network.config['connect']['conn_buffer_radius'])
    buffer_radius_inc = int(network.config['connect']['conn_buffer_radius_inc'])
    pfac_mv_gen = network.config['scenario']['pfac_mv_gen']
    pfac_lv_gen = network.config['scenario']['pfac_lv_gen']

    # =====================
    # Step 1: MV generators
    # =====================
    for geno in sorted(network.mv_grid.graph.nodes_by_attribute('generator'), key=lambda x: repr(x)):
        if nx.is_isolate(network.mv_grid.graph, geno):

            # ===== voltage level 4: generator has to be connected to MV station =====
            if geno.v_level == 4:

                line_length = calc_geo_dist_vincenty(network=network,
                                                     node_source=geno,
                                                     node_target=network.mv_grid.station)

                line_type, line_count = select_cable(network=network,
                                                       level='mv',
                                                       apparent_power=geno.nominal_capacity / pfac_mv_gen)

                line = Line(id=random.randint(10**8, 10**9),
                            type=line_type,
                            kind='cable',
                            quantity=line_count,
                            length=line_length / 1e3,
                            grid=network.mv_grid)

                network.mv_grid.graph.add_edge(network.mv_grid.station,
                                               geno,
                                               line=line,
                                               type='line')

            # ===== voltage level 5: generator has to be connected to MV grid (next-neighbor) =====
            elif geno.v_level == 5:

                # get branches within a the predefined radius `generator_buffer_radius`
                branches = calc_geo_lines_in_buffer(node=geno,
                                                    mv_grid=network.mv_grid,
                                                    radius=buffer_radius,
                                                    radius_inc=buffer_radius_inc)

                # calc distance between generator and grid's lines -> find nearest line
                conn_objects_min_stack = _find_nearest_conn_objects(network=network,
                                                                    node=geno,
                                                                    branches=branches)

                # connect!
                # go through the stack (from nearest to most far connection target object)
                generator_connected = False
                for dist_min_obj in conn_objects_min_stack:
                    # Note 1: conn_dist_ring_mod=0 to avoid re-routing of existent lines
                    # Note 2: In connect_node(), the default cable/line type of grid is used. This is reasonable since
                    #         the max. allowed power of the smallest possible cable/line type (3.64 MVA for overhead
                    #         line of type 48-AL1/8-ST1A) exceeds the max. allowed power of a generator (4.5 MVA (dena))
                    #         (if connected separately!)
                    target_obj_result = _connect_mv_node(network=network,
                                                         node=geno,
                                                         target_obj=dist_min_obj)

                    if target_obj_result is not None:
                        generator_connected = True
                        break

                if not generator_connected:
                    logger.debug(
                        'Generator {0} could not be connected, try to '
                        'increase the parameter `generator_buffer_radius` in '
                        'config file `config_calc.cfg` to gain more possible '
                        'connection points.'.format(geno))


def _find_nearest_conn_objects(network, node, branches):
    """Searches all branches for the nearest possible connection object per branch

    It picks out 1 object out of 3 possible objects: 2 branch-adjacent stations
    and 1 potentially created branch tee on the line (using perpendicular projection).
    The resulting stack (list) is sorted ascending by distance from node.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    node : :class:`~.grid.components.Component`
        Node to connect (e.g. :class:`~.grid.components.Generator`)
    branches :
        List of branches (NetworkX branch objects)

    Returns
    -------
    :obj:`list` of :obj:`dict`
        List of connection objects (each object is represented by dict with eDisGo object,
        shapely object and distance to node.
    """

    # threshold which is used to determine if 2 objects are on the same position (see below for details on usage)
    conn_diff_tolerance = network.config['connect']['conn_diff_tolerance']

    conn_objects_min_stack = []

    node_shp = transform(proj2equidistant(), node.geom)

    for branch in branches:
        stations = branch['adj_nodes']

        # create shapely objects for 2 stations and line between them, transform to equidistant CRS
        station1_shp = transform(proj2equidistant(), stations[0].geom)
        station2_shp = transform(proj2equidistant(), stations[1].geom)
        line_shp = LineString([station1_shp, station2_shp])

        # create dict with DING0 objects (line & 2 adjacent stations), shapely objects and distances
        conn_objects = {'s1': {'obj': stations[0],
                               'shp': station1_shp,
                               'dist': node_shp.distance(station1_shp) * 0.999},
                        's2': {'obj': stations[1],
                               'shp': station2_shp,
                               'dist': node_shp.distance(station2_shp) * 0.999},
                        'b': {'obj': branch,
                              'shp': line_shp,
                              'dist': node_shp.distance(line_shp)}}

        # Remove branch from the dict of possible conn. objects if it is too close to a node.
        # Without this solution, the target object is not unique for different runs (and so
        # were the topology)
        if (
                abs(conn_objects['s1']['dist'] - conn_objects['b']['dist']) < conn_diff_tolerance
             or abs(conn_objects['s2']['dist'] - conn_objects['b']['dist']) < conn_diff_tolerance
           ):
            del conn_objects['b']

        # remove MV station as possible connection point
        if isinstance(conn_objects['s1']['obj'], MVStation):
            del conn_objects['s1']
        elif isinstance(conn_objects['s2']['obj'], MVStation):
            del conn_objects['s2']

        # find nearest connection point on given triple dict (2 branch-adjacent stations + cable dist. on line)
        conn_objects_min = min(conn_objects.values(), key=lambda v: v['dist'])

        conn_objects_min_stack.append(conn_objects_min)

    # sort all objects by distance from node
    conn_objects_min_stack = [_ for _ in sorted(conn_objects_min_stack, key=lambda x: x['dist'])]

    return conn_objects_min_stack


def _connect_mv_node(network, node, target_obj):
    """Connects MV node to target object in MV grid

    If the target object is a node, a new line is created to it.
    If the target object is a line, the node is connected to a newly created branch tee
    (using perpendicular projection) on this line.
    New lines are created using standard equipment.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    node : :class:`~.grid.components.Component`
        Node to connect (e.g. :class:`~.grid.components.Generator`)
        Node must be a member of MV grid's graph (network.mv_grid.graph)
    target_obj : :class:`~.grid.components.Component`
        Object that node shall be connected to

    Returns
    -------
    :class:`~.grid.components.Component` or None
        Node that node was connected to
    """

    std_line_type = network.config['grid_expansion']['std_mv_line']
    std_line_kind = 'cable'

    target_obj_result = None

    node_shp = transform(proj2equidistant(), node.geom)

    # MV line is nearest connection point
    if isinstance(target_obj['shp'], LineString):

        adj_node1 = target_obj['obj']['adj_nodes'][0]
        adj_node2 = target_obj['obj']['adj_nodes'][1]

        # find nearest point on MV line
        conn_point_shp = target_obj['shp'].interpolate(target_obj['shp'].project(node_shp))
        conn_point_shp = transform(proj2conformal(), conn_point_shp)

        line = network.mv_grid.graph.edge[adj_node1][adj_node2]

        # target MV line does currently not connect a load area of type aggregated
        if not line['type'] == 'line_aggr':

            # create branch tee and add it to grid
            branch_tee = BranchTee(geom=conn_point_shp,
                                   grid=network.mv_grid)
            network.mv_grid.graph.add_node(branch_tee,
                                           type='branch_tee')

            # split old branch into 2 segments
            # (delete old branch and create 2 new ones along cable_dist)
            # ==========================================================

            # backup kind and type of branch
            line_kind = line['line'].kind
            line_type = line['line'].type

            network.mv_grid.graph.remove_edge(adj_node1, adj_node2)

            line_length = calc_geo_dist_vincenty(network=network,
                                                 node_source=adj_node1,
                                                 node_target=branch_tee)
            line = Line(id=random.randint(10 ** 8, 10 ** 9),
                        length=line_length / 1e3,
                        quantity=1,
                        kind=line_kind,
                        type=line_type)
            network.mv_grid.graph.add_edge(adj_node1,
                                           branch_tee,
                                           line=line,
                                           type='line')

            line_length = calc_geo_dist_vincenty(network=network,
                                                 node_source=adj_node2,
                                                 node_target=branch_tee)
            line = Line(id=random.randint(10 ** 8, 10 ** 9),
                        length=line_length / 1e3,
                        quantity=1,
                        kind=line_kind,
                        type=line_type)
            network.mv_grid.graph.add_edge(adj_node2,
                                           branch_tee,
                                           line=line,
                                           type='line')

            # add new branch for new node (station to branch tee)
            # ===================================================
            line_length = calc_geo_dist_vincenty(network=network,
                                                 node_source=node,
                                                 node_target=branch_tee)
            line = Line(id=random.randint(10 ** 8, 10 ** 9),
                        length=line_length / 1e3,
                        quantity=1,
                        kind=std_line_kind,
                        type=std_line_type)
            network.mv_grid.graph.add_edge(node,
                                           branch_tee,
                                           line=line,
                                           type='line')

            # TODO: Add costs, #45

            target_obj_result = branch_tee

    # node ist nearest connection point
    else:

        # what kind of node is to be connected? (which type is node of?)
        #   LVStation: Connect to LVStation or BranchTee
        #   Generator: Connect to LVStation, BranchTee or Generator
        if isinstance(node, LVStation):
            valid_conn_objects = (LVStation, BranchTee)
        elif isinstance(node, Generator):
            valid_conn_objects = (LVStation, BranchTee, Generator)
        else:
            raise ValueError('Oops, the node you are trying to connect is not a valid connection object')

        # if target is generator or Load, check if it is aggregated (=> connection not allowed)
        if isinstance(target_obj['obj'], (Generator, Load)):
            target_is_aggregated = any([_ for _ in network.mv_grid.graph.edge[target_obj['obj']].values()
                                        if _['type'] == 'line_aggr'])
        else:
            target_is_aggregated = False

        # target node is not a load area of type aggregated
        if isinstance(target_obj['obj'], valid_conn_objects) and not target_is_aggregated:

            # add new branch for satellite (station to station)
            line_length = calc_geo_dist_vincenty(network=network,
                                                 node_source=node,
                                                 node_target=target_obj['obj'])

            line = Line(id=random.randint(10 ** 8, 10 ** 9),
                        type=std_line_type,
                        kind=std_line_kind,
                        quantity=1,
                        length=line_length / 1e3,
                        grid=network.mv_grid)

            network.mv_grid.graph.add_edge(node,
                                           target_obj['obj'],
                                           line=line,
                                           type='line')

            # TODO: Add costs, #45

            target_obj_result = target_obj['obj']

    return target_obj_result
