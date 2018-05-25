from ..grid.components import Line, MVStation, LVStation, MVDisconnectingPoint, Generator, Load, BranchTee
from ..tools.geo import calc_geo_dist_vincenty, \
                        calc_geo_lines_in_buffer, \
                        proj2equidistant, \
                        proj2conformal

import networkx as nx
import random
import pandas as pd

import os
if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString
    from shapely.ops import transform

import logging
logger = logging.getLogger('edisgo')


def connect_mv_generators(network):
    """Connect MV generators to existing grids.

    This function searches for unconnected generators in MV grids and connects them.

    It connects

        * generators of voltage level 4
            * to HV-MV station

        * generators of voltage level 5
            * with a nom. capacity of <=30 kW to LV loads of type residential
            * with a nom. capacity of >30 kW and <=100 kW to LV loads of type
                retail, industrial or agricultural
            * to the MV-LV station if no appropriate load is available (fallback)

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object

    Notes
    -----
    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/grid/mv_grid/mv_connect.py#L820>`_.
    """

    # get params from config
    buffer_radius = int(network.config[
                            'grid_connection']['conn_buffer_radius'])
    buffer_radius_inc = int(network.config[
                                'grid_connection']['conn_buffer_radius_inc'])

    # get standard equipment
    std_line_type = network.equipment_data['mv_cables'].loc[
        network.config['grid_expansion_standard_equipment']['mv_line']]

    for geno in sorted(network.mv_grid.graph.nodes_by_attribute('generator'),
                       key=lambda _: repr(_)):
        if nx.is_isolate(network.mv_grid.graph, geno):

            # ===== voltage level 4: generator has to be connected to MV station =====
            if geno.v_level == 4:

                line_length = calc_geo_dist_vincenty(network=network,
                                                     node_source=geno,
                                                     node_target=network.mv_grid.station)

                line = Line(id=random.randint(10**8, 10**9),
                            type=std_line_type,
                            kind='cable',
                            quantity=1,
                            length=line_length / 1e3,
                            grid=network.mv_grid)

                network.mv_grid.graph.add_edge(network.mv_grid.station,
                                               geno,
                                               line=line,
                                               type='line')

                # add line to equipment changes to track costs
                _add_cable_to_equipment_changes(network=network,
                                                line=line)

            # ===== voltage level 5: generator has to be connected to MV grid (next-neighbor) =====
            elif geno.v_level == 5:

                # get branches within a the predefined radius `generator_buffer_radius`
                branches = calc_geo_lines_in_buffer(network=network,
                                                    node=geno,
                                                    grid=network.mv_grid,
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
                    target_obj_result = _connect_mv_node(network=network,
                                                         node=geno,
                                                         target_obj=dist_min_obj)

                    if target_obj_result is not None:
                        generator_connected = True
                        break

                if not generator_connected:
                    logger.debug(
                        'Generator {0} could not be connected, try to '
                        'increase the parameter `conn_buffer_radius` in '
                        'config file `config_grid.cfg` to gain more possible '
                        'connection points.'.format(geno))


def connect_lv_generators(network, allow_multiple_genos_per_load=True):
    """Connect LV generators to existing grids.

    This function searches for unconnected generators in all LV grids and connects them.

    It connects

        * generators of voltage level 6
            * to MV-LV station

        * generators of voltage level 7
            * with a nom. capacity of <=30 kW to LV loads of type residential
            * with a nom. capacity of >30 kW and <=100 kW to LV loads of type
                retail, industrial or agricultural
            * to the MV-LV station if no appropriate load is available (fallback)

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    allow_multiple_genos_per_load : :obj:`bool`
        If True, more than one generator can be connected to one load

    Notes
    -----
    For the allocation, loads are selected randomly (sector-wise) using a predefined seed
    to ensure reproducibility.

    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/grid/lv_grid/lv_connect.py#L27>`_.
    """

    # get predefined random seed and initialize random generator
    seed = int(network.config['grid_connection']['random_seed'])
    #random.seed(a=seed)
    random.seed(a=1234)
    # TODO: Switch back to 'seed' as soon as line ids are finished, #58

    # get standard equipment
    std_line_type = network.equipment_data['lv_cables'].loc[
        network.config['grid_expansion_standard_equipment']['lv_line']]
    std_line_kind = 'cable'

    # # TEMP: DEBUG STUFF
    # lv_grid_stats = pd.DataFrame(columns=('lv_grid',
    #                                       'load_count',
    #                                       'geno_count',
    #                                       'more_genos_than_loads')
    #                             )

    # iterate over all LV grids
    for lv_grid in network.mv_grid.lv_grids:

        lv_loads = lv_grid.graph.nodes_by_attribute('load')

        # counter for genos in v_level 7
        log_geno_count_vlevel7 = 0

        # generate random list (without replacement => unique elements)
        # of loads (residential) to connect genos (P <= 30kW) to.
        lv_loads_res = sorted([lv_load for lv_load in lv_loads
                               if 'residential' in list(lv_load.consumption.keys())],
                              key=lambda _: repr(_))

        if len(lv_loads_res) > 0:
            lv_loads_res_rnd = set(random.sample(lv_loads_res,
                                                 len(lv_loads_res)))
        else:
            lv_loads_res_rnd = None

        # generate random list (without replacement => unique elements)
        # of loads (retail, industrial, agricultural) to connect genos
        # (30kW < P <= 100kW) to.
        lv_loads_ria = sorted([lv_load for lv_load in lv_loads
                               if any([_ in list(lv_load.consumption.keys())
                                       for _ in ['retail', 'industrial', 'agricultural']])],
                              key=lambda _: repr(_))

        if len(lv_loads_ria) > 0:
            lv_loads_ria_rnd = set(random.sample(lv_loads_ria,
                                                 len(lv_loads_ria)))
        else:
            lv_loads_ria_rnd = None

        for geno in sorted(lv_grid.graph.nodes_by_attribute('generator'), key=lambda x: repr(x)):
            if nx.is_isolate(lv_grid.graph, geno):

                lv_station = lv_grid.station

                # generator is of v_level 6 -> connect to LV station
                if geno.v_level == 6:

                    line_length = calc_geo_dist_vincenty(network=network,
                                                         node_source=geno,
                                                         node_target=lv_station)

                    line = Line(id=random.randint(10 ** 8, 10 ** 9),
                                length=line_length / 1e3,
                                quantity=1,
                                kind=std_line_kind,
                                type=std_line_type,
                                grid=lv_grid)

                    lv_grid.graph.add_edge(geno,
                                           lv_station,
                                           line=line,
                                           type='line')

                    # add line to equipment changes to track costs
                    _add_cable_to_equipment_changes(network=network,
                                                    line=line)

                # generator is of v_level 7 -> assign geno to load
                elif geno.v_level == 7:
                    # counter for genos in v_level 7
                    log_geno_count_vlevel7 += 1

                    # connect genos with P <= 30kW to residential loads, if available
                    if (geno.nominal_capacity <= 30) and (lv_loads_res_rnd is not None):
                        if len(lv_loads_res_rnd) > 0:
                            lv_load = lv_loads_res_rnd.pop()
                        # if random load list is empty, create new one
                        else:
                            lv_loads_res_rnd = set(random.sample(lv_loads_res,
                                                                 len(lv_loads_res))
                                                   )
                            lv_load = lv_loads_res_rnd.pop()

                        # get cable distributor of building
                        lv_conn_target = lv_grid.graph.neighbors(lv_load)[0]

                        if not allow_multiple_genos_per_load:
                            # check if there's an existing generator connected to the load
                            # if so, select next load. If no load is available, connect to station.
                            while any([isinstance(_, Generator)
                                       for _ in lv_grid.graph.neighbors(
                                    lv_grid.graph.neighbors(lv_load)[0])]):
                                if len(lv_loads_res_rnd) > 0:
                                    lv_load = lv_loads_res_rnd.pop()

                                    # get cable distributor of building
                                    lv_conn_target = lv_grid.graph.neighbors(lv_load)[0]
                                else:
                                    lv_conn_target = lv_grid.station

                                    logger.debug(
                                        'No valid conn. target found for {}. '
                                        'Connected to {}.'.format(
                                            repr(geno),
                                            repr(lv_conn_target)
                                        )
                                    )
                                    break

                    # connect genos with 30kW <= P <= 100kW to residential loads
                    # to retail, industrial, agricultural loads, if available
                    elif (geno.nominal_capacity > 30) and (lv_loads_ria_rnd is not None):
                        if len(lv_loads_ria_rnd) > 0:
                            lv_load = lv_loads_ria_rnd.pop()
                        # if random load list is empty, create new one
                        else:
                            lv_loads_ria_rnd = set(random.sample(lv_loads_ria,
                                                                 len(lv_loads_ria))
                                                   )
                            lv_load = lv_loads_ria_rnd.pop()

                        # get cable distributor of building
                        lv_conn_target = lv_grid.graph.neighbors(lv_load)[0]

                        if not allow_multiple_genos_per_load:
                            # check if there's an existing generator connected to the load
                            # if so, select next load. If no load is available, connect to station.
                            while any([isinstance(_, Generator)
                                       for _ in lv_grid.graph.neighbors(
                                    lv_grid.graph.neighbors(lv_load)[0])]):
                                if len(lv_loads_ria_rnd) > 0:
                                    lv_load = lv_loads_ria_rnd.pop()

                                    # get cable distributor of building
                                    lv_conn_target = lv_grid.graph.neighbors(lv_load)[0]
                                else:
                                    lv_conn_target = lv_grid.station

                                    logger.debug(
                                        'No valid conn. target found for {}. '
                                        'Connected to {}.'.format(
                                            repr(geno),
                                            repr(lv_conn_target)
                                        )
                                    )
                                    break

                    # fallback: connect to station
                    else:
                        lv_conn_target = lv_grid.station

                        logger.debug(
                            'No valid conn. target found for {}. '
                            'Connected to {}.'.format(
                                repr(geno),
                                repr(lv_conn_target)
                            )
                        )

                    line = Line(id=random.randint(10 ** 8, 10 ** 9),
                                length=1e-3,
                                quantity=1,
                                kind=std_line_kind,
                                type=std_line_type,
                                grid=lv_grid)

                    lv_grid.graph.add_edge(geno,
                                           lv_station,
                                           line=line,
                                           type='line')

                    # add line to equipment changes to track costs
                    _add_cable_to_equipment_changes(network=network,
                                                    line=line)

        # warn if there're more genos than loads in LV grid
        if log_geno_count_vlevel7 > len(lv_loads):
            logger.debug('The count of newly connected generators in voltage level 7 ({}) '
                         'exceeds the count of loads ({}) in LV grid {}.'
                         .format(str(log_geno_count_vlevel7),
                                 str(len(lv_loads)),
                                 repr(lv_grid)
                                 )
                         )
        # # TEMP: DEBUG STUFF
        # lv_grid_stats.loc[len(lv_grid_stats)] = [repr(lv_grid),
        #                                          len(lv_loads),
        #                                          log_geno_count_vlevel7,
        #                                          log_geno_count_vlevel7 > len(lv_loads)]


def _add_cable_to_equipment_changes(network, line):
    """Add cable to the equipment changes

    All changes of equipment are stored in network.results.equipment_changes
    which is used later to determine grid expansion costs.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    line : class:`~.grid.components.Line`
        Line instance which is to be added
    """
    network.results.equipment_changes = \
        network.results.equipment_changes.append(
            pd.DataFrame(
                {'iteration_step': [0],
                 'change': ['added'],
                 'equipment': [line.type.name],
                 'quantity': [1]
                 },
                index=[line]
            )
        )


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

    Notes
    -----
    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/grid/mv_grid/mv_connect.py#L38>`_.
    """

    # threshold which is used to determine if 2 objects are on the same position (see below for details on usage)
    conn_diff_tolerance = network.config['grid_connection'][
        'conn_diff_tolerance']

    conn_objects_min_stack = []

    node_shp = transform(proj2equidistant(network), node.geom)

    for branch in branches:
        stations = branch['adj_nodes']

        # create shapely objects for 2 stations and line between them, transform to equidistant CRS
        station1_shp = transform(proj2equidistant(network), stations[0].geom)
        station2_shp = transform(proj2equidistant(network), stations[1].geom)
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

    Notes
    -----
    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/grid/mv_grid/mv_connect.py#L311>`_.
    """

    # get standard equipment
    std_line_type = network.equipment_data['mv_cables'].loc[
        network.config['grid_expansion_standard_equipment']['mv_line']]
    std_line_kind = 'cable'

    target_obj_result = None

    node_shp = transform(proj2equidistant(network), node.geom)

    # MV line is nearest connection point
    if isinstance(target_obj['shp'], LineString):

        adj_node1 = target_obj['obj']['adj_nodes'][0]
        adj_node2 = target_obj['obj']['adj_nodes'][1]

        # find nearest point on MV line
        conn_point_shp = target_obj['shp'].interpolate(target_obj['shp'].project(node_shp))
        conn_point_shp = transform(proj2conformal(network), conn_point_shp)

        line = network.mv_grid.graph.edge[adj_node1][adj_node2]

        # target MV line does currently not connect a load area of type aggregated
        if not line['type'] == 'line_aggr':

            # create branch tee and add it to grid
            branch_tee = BranchTee(geom=conn_point_shp,
                                   grid=network.mv_grid,
                                   in_building=False)
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
                        type=line_type,
                        grid=network.mv_grid)
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
                        type=line_type,
                        grid=network.mv_grid)
            network.mv_grid.graph.add_edge(adj_node2,
                                           branch_tee,
                                           line=line,
                                           type='line')

            # add new branch for new node (node to branch tee)
            # ================================================
            line_length = calc_geo_dist_vincenty(network=network,
                                                 node_source=node,
                                                 node_target=branch_tee)
            line = Line(id=random.randint(10 ** 8, 10 ** 9),
                        length=line_length / 1e3,
                        quantity=1,
                        kind=std_line_kind,
                        type=std_line_type,
                        grid=network.mv_grid)
            network.mv_grid.graph.add_edge(node,
                                           branch_tee,
                                           line=line,
                                           type='line')

            # add line to equipment changes to track costs
            _add_cable_to_equipment_changes(network=network,
                                            line=line)

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

            # add line to equipment changes to track costs
            _add_cable_to_equipment_changes(network=network,
                                            line=line)

            target_obj_result = target_obj['obj']

    return target_obj_result
