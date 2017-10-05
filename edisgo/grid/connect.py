from ..grid.components import Line, MVStation
from ..grid.tools import select_cable
from ..tools.geo import calc_geo_dist_vincenty, calc_geo_lines_in_buffer, proj2equidistant

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

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    generators : :pandas:`pandas.DataFrame<dataframe>`
        List of generators
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

                length = calc_geo_dist_vincenty(geno, network.mv_grid.station)

                cable_type, cable_count = select_cable(network=network,
                                                       level='mv',
                                                       apparent_power=geno.nominal_capacity / pfac_mv_gen)

                line = {'line': Line(id=random.randint(10**8, 10**9),
                                     type=cable_type,
                                     quantity=cable_count,
                                     length=length / 1e3,
                                     grid=network.mv_grid)
                        }

                network.mv_grid.graph.add_edge(network.mv_grid.station, geno, line, type='line')

            # ===== voltage level 5: generator has to be connected to MV grid (next-neighbor) =====
            elif geno.v_level == 5:

                # get branches within a the predefined radius `generator_buffer_radius`
                branches = calc_geo_lines_in_buffer(geno,
                                                    network.mv_grid,
                                                    buffer_radius,
                                                    buffer_radius_inc)

                # calc distance between generator and grid's lines -> find nearest line
                conn_objects_min_stack = _find_nearest_conn_objects(geno,
                                                                    branches)

                # connect!
                # go through the stack (from nearest to most far connection target object)
                generator_connected = False
                for dist_min_obj in conn_objects_min_stack:
                    # Note 1: conn_dist_ring_mod=0 to avoid re-routing of existent lines
                    # Note 2: In connect_node(), the default cable/line type of grid is used. This is reasonable since
                    #         the max. allowed power of the smallest possible cable/line type (3.64 MVA for overhead
                    #         line of type 48-AL1/8-ST1A) exceeds the max. allowed power of a generator (4.5 MVA (dena))
                    #         (if connected separately!)
                    target_obj_result = connect_node(generator,
                                                     generator_shp,
                                                     mv_grid_district.mv_grid,
                                                     dist_min_obj,
                                                     proj2,
                                                     graph,
                                                     conn_dist_ring_mod=0,
                                                     debug=debug)

                    if target_obj_result is not None:
                        if debug:
                            logger.debug(
                                'Generator {0} was connected to {1}'.format(
                                    generator, target_obj_result))
                        generator_connected = True
                        break

                if not generator_connected and debug:
                    logger.debug(
                        'Generator {0} could not be connected, try to '
                        'increase the parameter `generator_buffer_radius` in '
                        'config file `config_calc.cfg` to gain more possible '
                        'connection points.'.format(generator))


def _find_nearest_conn_objects(network, node, branches):
    """ Searches all `branches` for the nearest possible connection object per branch (picks out 1 object out of 3
        possible objects: 2 branch-adjacent stations and 1 potentially created cable distributor on the line
        (perpendicular projection)). The resulting stack (list) is sorted ascending by distance from node.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object

    node: XXXXXXXXXXX
    branches: BranchDing0 objects of MV region

    Returns
    -------
    conn_objects_min_stack: List of connection objects (each object is represented by dict with Ding0 object,
                                shapely object and distance to node.

    """
    # TODO: Update docstring

    # threshold which is used to determine if 2 objects are on the same position (see below for details on usage)
    conn_diff_tolerance = network.config['connect']['conn_diff_tolerance']

    conn_objects_min_stack = []

    node_shp = transform(proj2equidistant(), node.geo_data)

    for branch in branches:
        stations = branch['adj_nodes']

        # create shapely objects for 2 stations and line between them, transform to equidistant CRS
        station1_shp = transform(proj2equidistant(), stations[0].geo_data)
        station2_shp = transform(proj2equidistant(), stations[1].geo_data)
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
        #if not branches_only:
        #    conn_objects_min_stack.append(conn_objects_min)
        #elif isinstance(conn_objects_min['shp'], LineString):
        #    conn_objects_min_stack.append(conn_objects_min)
        conn_objects_min_stack.append(conn_objects_min)

    # sort all objects by distance from node
    conn_objects_min_stack = [_ for _ in sorted(conn_objects_min_stack, key=lambda x: x['dist'])]

    return conn_objects_min_stack


def connect_node(node, node_shp, mv_grid, target_obj, proj, graph, conn_dist_ring_mod, debug):
    """ Connects `node` to `target_obj`

    Args:
        node: origin node - Ding0 object (e.g. LVLoadAreaCentreDing0)
        node_shp: Shapely Point object of origin node
        target_obj: object that node shall be connected to
        proj: pyproj projection object: equidistant CRS to conformal CRS (e.g. ETRS -> WGS84)
        graph: NetworkX graph object with nodes and newly created branches
        conn_dist_ring_mod: Max. distance when nodes are included into route instead of creating a new line,
                            see mv_connect() for details.
        debug: If True, information is printed during process

    Returns:
        target_obj_result: object that node was connected to (instance of LVLoadAreaCentreDing0 or
                           MVCableDistributorDing0). If node is included into line instead of creating a new line (see arg
                           `conn_dist_ring_mod`), `target_obj_result` is None.
    """

    target_obj_result = None

    # MV line is nearest connection point
    if isinstance(target_obj['shp'], LineString):

        adj_node1 = target_obj['obj']['adj_nodes'][0]
        adj_node2 = target_obj['obj']['adj_nodes'][1]

        # find nearest point on MV line
        conn_point_shp = target_obj['shp'].interpolate(target_obj['shp'].project(node_shp))
        conn_point_shp = transform(proj, conn_point_shp)

        # target MV line does currently not connect a load area of type aggregated
        if not target_obj['obj']['branch'].connects_aggregated:

            # Node is close to line
            # -> insert node into route (change existing route)
            if (target_obj['dist'] < conn_dist_ring_mod):
                # backup kind and type of branch
                branch_type = graph.edge[adj_node1][adj_node2]['branch'].type
                branch_kind = graph.edge[adj_node1][adj_node2]['branch'].kind
                branch_ring = graph.edge[adj_node1][adj_node2]['branch'].ring

                # check if there's a circuit breaker on current branch,
                # if yes set new position between first node (adj_node1) and newly inserted node
                circ_breaker = graph.edge[adj_node1][adj_node2]['branch'].circuit_breaker
                if circ_breaker is not None:
                    circ_breaker.geo_data = calc_geo_centre_point(adj_node1, node)

                # split old ring main route into 2 segments (delete old branch and create 2 new ones
                # along node)
                graph.remove_edge(adj_node1, adj_node2)

                branch_length = calc_geo_dist_vincenty(adj_node1, node)
                branch = BranchDing0(length=branch_length,
                                     circuit_breaker=circ_breaker,
                                     kind=branch_kind,
                                     type=branch_type,
                                     ring=branch_ring)
                if circ_breaker is not None:
                    circ_breaker.branch = branch
                graph.add_edge(adj_node1, node, branch=branch)

                branch_length = calc_geo_dist_vincenty(adj_node2, node)
                graph.add_edge(adj_node2, node, branch=BranchDing0(length=branch_length,
                                                                   kind=branch_kind,
                                                                   type=branch_type,
                                                                   ring=branch_ring))

                target_obj_result = 're-routed'

                if debug:
                    logger.debug('Ring main route modified to include '
                                 'node {}'.format(node))

            # Node is too far away from route
            # => keep main route and create new line from node to (cable distributor on) route.
            else:

                # create cable distributor and add it to grid
                cable_dist = MVCableDistributorDing0(geo_data=conn_point_shp,
                                                     grid=mv_grid)
                mv_grid.add_cable_distributor(cable_dist)

                # check if there's a circuit breaker on current branch,
                # if yes set new position between first node (adj_node1) and newly created cable distributor
                circ_breaker = graph.edge[adj_node1][adj_node2]['branch'].circuit_breaker
                if circ_breaker is not None:
                    circ_breaker.geo_data = calc_geo_centre_point(adj_node1, cable_dist)

                # split old branch into 2 segments (delete old branch and create 2 new ones along cable_dist)
                # ===========================================================================================

                # backup kind and type of branch
                branch_kind = graph.edge[adj_node1][adj_node2]['branch'].kind
                branch_type = graph.edge[adj_node1][adj_node2]['branch'].type
                branch_ring = graph.edge[adj_node1][adj_node2]['branch'].ring

                graph.remove_edge(adj_node1, adj_node2)

                branch_length = calc_geo_dist_vincenty(adj_node1, cable_dist)
                branch = BranchDing0(length=branch_length,
                                     circuit_breaker=circ_breaker,
                                     kind=branch_kind,
                                     type=branch_type,
                                     ring=branch_ring)
                if circ_breaker is not None:
                    circ_breaker.branch = branch
                graph.add_edge(adj_node1, cable_dist, branch=branch)

                branch_length = calc_geo_dist_vincenty(adj_node2, cable_dist)
                graph.add_edge(adj_node2, cable_dist, branch=BranchDing0(length=branch_length,
                                                                         kind=branch_kind,
                                                                         type=branch_type,
                                                                         ring=branch_ring))

                # add new branch for satellite (station to cable distributor)
                # ===========================================================

                # get default branch kind and type from grid to use it for new branch
                branch_kind = mv_grid.default_branch_kind
                branch_type = mv_grid.default_branch_type

                branch_length = calc_geo_dist_vincenty(node, cable_dist)
                graph.add_edge(node, cable_dist, branch=BranchDing0(length=branch_length,
                                                                    kind=branch_kind,
                                                                    type=branch_type,
                                                                    ring=branch_ring))
                target_obj_result = cable_dist

                # debug info
                if debug:
                    logger.debug('Nearest connection point for object {0} '
                                 'is branch {1} (distance={2} m)'.format(
                        node, target_obj['obj']['adj_nodes'], target_obj['dist']))

    # node ist nearest connection point
    else:

        # what kind of node is to be connected? (which type is node of?)
        #   LVLoadAreaCentreDing0: Connect to LVLoadAreaCentreDing0 only
        #   LVStationDing0: Connect to LVLoadAreaCentreDing0, LVStationDing0 or MVCableDistributorDing0
        #   GeneratorDing0: Connect to LVLoadAreaCentreDing0, LVStationDing0, MVCableDistributorDing0 or GeneratorDing0
        if isinstance(node, LVLoadAreaCentreDing0):
            valid_conn_objects = LVLoadAreaCentreDing0
        elif isinstance(node, LVStationDing0):
            valid_conn_objects = (LVLoadAreaCentreDing0, LVStationDing0, MVCableDistributorDing0)
        elif isinstance(node, GeneratorDing0):
            valid_conn_objects = (LVLoadAreaCentreDing0, LVStationDing0, MVCableDistributorDing0, GeneratorDing0)
        else:
            raise ValueError('Oops, the node you are trying to connect is not a valid connection object')

        # if target is Load Area centre or LV station, check if it belongs to a load area of type aggregated
        # (=> connection not allowed)
        if isinstance(target_obj['obj'], (LVLoadAreaCentreDing0, LVStationDing0)):
            target_is_aggregated = target_obj['obj'].lv_load_area.is_aggregated
        else:
            target_is_aggregated = False

        # target node is not a load area of type aggregated
        if isinstance(target_obj['obj'], valid_conn_objects) and not target_is_aggregated:

            # get default branch kind and type from grid to use it for new branch
            branch_kind = mv_grid.default_branch_kind
            branch_type = mv_grid.default_branch_type

            # get branch ring obj
            branch_ring = mv_grid.get_ring_from_node(target_obj['obj'])

            # add new branch for satellite (station to station)
            branch_length = calc_geo_dist_vincenty(node, target_obj['obj'])
            graph.add_edge(node, target_obj['obj'], branch=BranchDing0(length=branch_length,
                                                                       kind=branch_kind,
                                                                       type=branch_type,
                                                                       ring=branch_ring))
            target_obj_result = target_obj['obj']

            # debug info
            if debug:
                logger.debug('Nearest connection point for object {0} is station {1} '
                      '(distance={2} m)'.format(
                    node, target_obj['obj'], target_obj['dist']))

    return target_obj_result
