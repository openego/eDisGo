import networkx as nx
import random
import pandas as pd
import os

if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString, Point
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads

from edisgo.network.components import Generator, Load
from edisgo.tools.geo import \
    calc_geo_dist_vincenty, calc_geo_lines_in_buffer, \
    proj2equidistant, proj2conformal

import logging
logger = logging.getLogger('edisgo')


def add_and_connect_mv_generator(edisgo_object, generator):
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

    # get standard equipment
    std_line_type = edisgo_object.topology.equipment_data['mv_cables'].loc[
        edisgo_object.config['grid_expansion_standard_equipment']['mv_line']]

    # add generator bus
    gen_bus = 'Bus_Generator_{}'.format(generator.name)
    geom = wkt_loads(generator.geom)
    edisgo_object.topology.add_bus(
        bus_name=gen_bus,
        v_nom=edisgo_object.topology.mv_grid.nominal_voltage,
        x=geom.x,
        y=geom.y)

    # add generator
    edisgo_object.topology.add_generator(
        generator_id=generator.name,
        bus=gen_bus,
        p_nom=generator.electrical_capacity / 1e3,
        generator_type=generator.generation_type,
        subtype=generator.generation_subtype,
        weather_cell_id=generator.w_id)

    # ===== voltage level 4: generator is connected to MV station =====
    if generator.voltage_level == 4:

        # add line

        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=edisgo_object.topology.buses_df.loc[gen_bus, :],
            bus_target=edisgo_object.topology.mv_grid.station.iloc[0, :])

        line_name = edisgo_object.topology.add_line(
            bus0=edisgo_object.topology.mv_grid.station.index[0],
            bus1=gen_bus,
            length=line_length,
            kind='cable',
            type_info=std_line_type.name
        )

        # add line to equipment changes to track costs
        _add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name])

    # == voltage level 5: generator is connected to MV grid (next-neighbor) ==
    elif generator.voltage_level == 5:

        # get branches within a the predefined radius `generator_buffer_radius`
        # get params from config
        lines = calc_geo_lines_in_buffer(
            edisgo_object=edisgo_object,
            bus=edisgo_object.topology.buses_df.loc[gen_bus, :],
            grid=edisgo_object.topology.mv_grid)

        # calc distance between generator and grid's lines -> find nearest line
        conn_objects_min_stack = _find_nearest_conn_objects(
            edisgo_object=edisgo_object,
            bus=edisgo_object.topology.buses_df.loc[gen_bus, :],
            lines=lines)

        # connect
        # go through the stack (from nearest to most far connection target
        # object)
        generator_connected = False
        for dist_min_obj in conn_objects_min_stack:
            target_obj_result = _connect_mv_node(
                edisgo_object=edisgo_object,
                bus=edisgo_object.topology.buses_df.loc[gen_bus, :],
                target_obj=dist_min_obj)

            if target_obj_result is not None:
                generator_connected = True
                break

        if not generator_connected:
            logger.debug(
                'Generator {} could not be connected, try to '
                'increase the parameter `conn_buffer_radius` in '
                'config file `config_grid.cfg` to gain more possible '
                'connection points.'.format(generator.name))


def connect_lv_generators(network, allow_multiple_genos_per_load=True):
    """Connect LV generators to existing grids.

    This function searches for unconnected generators in all LV grids and
    connects them.

    It connects

        * generators of voltage level 6
            * to MV-LV station

        * generators of voltage level 7
            * with a nom. capacity of <=30 kW to LV loads of type residential
            * with a nom. capacity of >30 kW and <=100 kW to LV loads of type
                retail, industrial or agricultural
            * to the MV-LV station if no appropriate load is available
              (fallback)

    Parameters
    ----------
    network : :class:`~.network.topology.Topology`
        The eDisGo container object
    allow_multiple_genos_per_load : :obj:`bool`
        If True, more than one generator can be connected to one load

    Notes
    -----
    For the allocation, loads are selected randomly (sector-wise) using a
    predefined seed to ensure reproducibility.

    """

    # get predefined random seed and initialize random generator
    seed = int(network.config['grid_connection']['random_seed'])
    #random.seed(a=seed)
    random.seed(a=1234)
    # ToDo: Switch back to 'seed' as soon as line ids are finished, #58

    # get standard equipment
    std_line_type = edisgo_object.topology.equipment_data['lv_cables'].loc[
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
                                length=line_length,
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
                        lv_conn_target = list(lv_grid.graph.neighbors(lv_load))[0]

                        if not allow_multiple_genos_per_load:
                            # check if there's an existing generator connected to the load
                            # if so, select next load. If no load is available, connect to station.
                            while any([isinstance(_, Generator)
                                       for _ in lv_grid.graph.neighbors(
                                    list(lv_grid.graph.neighbors(lv_load))[0])]):
                                if len(lv_loads_res_rnd) > 0:
                                    lv_load = lv_loads_res_rnd.pop()

                                    # get cable distributor of building
                                    lv_conn_target = list(lv_grid.graph.neighbors(lv_load))[0]
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
                        lv_conn_target = list(lv_grid.graph.neighbors(lv_load))[0]

                        if not allow_multiple_genos_per_load:
                            # check if there's an existing generator connected to the load
                            # if so, select next load. If no load is available, connect to station.
                            while any([isinstance(_, Generator)
                                       for _ in lv_grid.graph.neighbors(
                                    list(lv_grid.graph.neighbors(lv_load))[0])]):
                                if len(lv_loads_ria_rnd) > 0:
                                    lv_load = lv_loads_ria_rnd.pop()

                                    # get cable distributor of building
                                    lv_conn_target = list(lv_grid.graph.neighbors(lv_load))[0]
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

        # warn if there're more genos than loads in LV network
        if log_geno_count_vlevel7 > len(lv_loads):
            logger.debug('The count of newly connected generators in voltage level 7 ({}) '
                         'exceeds the count of loads ({}) in LV network {}.'
                         .format(str(log_geno_count_vlevel7),
                                 str(len(lv_loads)),
                                 repr(lv_grid)
                                 )
                         )


def _add_line_to_equipment_changes(edisgo_object, line):
    """
    Add line to the equipment changes.

    All changes of equipment are stored in edisgo.results.equipment_changes
    which is used later to determine network expansion costs.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    line : pd.Series
        Data of line to add.
        Series has same rows as columns of topology.lines_df. Line
        representative is the series name.

    """
    edisgo_object.results.equipment_changes = \
        edisgo_object.results.equipment_changes.append(
            pd.DataFrame(
                {'iteration_step': [0],
                 'change': ['added'],
                 'equipment': [line.type_info],
                 'quantity': [1]
                 },
                index=[line.name]
            )
        )


def _del_line_from_equipment_changes(edisgo_object, line_repr):
    """
    Delete line from the equipment changes if it exists.

    This is needed when a line was already added to
    Results.equipment_changes but another component is later connected
    to this line. Therefore, the line needs to be split which changes the
    representative of the line and the line data.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    line_repr : str
        Line representative as in topology.lines_df.index.

    """
    if line_repr in edisgo_object.results.equipment_changes.index:
        edisgo_object.results.equipment_changes = \
            edisgo_object.results.equipment_changes.drop(line_repr)


def _find_nearest_conn_objects(edisgo_object, bus, lines):
    """
    Searches all lines for the nearest possible connection object per line.

    It picks out 1 object out of 3 possible objects: 2 branch-adjacent buses
    and 1 potentially created branch tee on the line (using perpendicular
    projection). The resulting stack (list) is sorted ascending by distance
    from bus.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    bus : pandas Series
        Data of bus to connect.
        Series has same rows as columns of topology.buses_df.
    lines : list(str)
        List of line representatives from topology.lines_df.index

    Returns
    -------
    :obj:`list` of :obj:`dict`
        List of connection objects (each object is represented by dict with
        representative, shapely object and distance to node.

    Notes
    -----
    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/network/mv_grid/mv_connect.py#L38>`_.

    """

    # threshold which is used to determine if 2 objects are at the same
    # position (see below for details on usage)
    conn_diff_tolerance = edisgo_object.config['grid_connection'][
        'conn_diff_tolerance']

    conn_objects_min_stack = []

    srid = edisgo_object.topology.grid_district['srid']
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))

    for line in lines:

        line_bus0 = edisgo_object.topology.buses_df.loc[
            edisgo_object.topology.lines_df.loc[line, 'bus0']]
        line_bus1 = edisgo_object.topology.buses_df.loc[
            edisgo_object.topology.lines_df.loc[line, 'bus1']]

        # create shapely objects for 2 buses and line between them,
        # transform to equidistant CRS
        line_bus0_shp = transform(proj2equidistant(srid),
                                 Point(line_bus0.x, line_bus0.y))
        line_bus1_shp = transform(proj2equidistant(srid),
                                 Point(line_bus1.x, line_bus1.y))
        line_shp = LineString([line_bus0_shp, line_bus1_shp])

        # create dict with line & 2 adjacent buses and their shapely objects
        # and distances
        conn_objects = {'s1': {'repr': line_bus0.name,
                               'shp': line_bus0_shp,
                               'dist': bus_shp.distance(line_bus0_shp) *
                                       0.999},
                        's2': {'repr': line_bus1.name,
                               'shp': line_bus1_shp,
                               'dist': bus_shp.distance(line_bus1_shp) *
                                       0.999},
                        'b': {'repr': line,
                              'shp': line_shp,
                              'dist': bus_shp.distance(line_shp)}}

        # remove line from the dict of possible conn. objects if it is too
        # close to the bus (necessary to assure that connection target is
        # reproducible)
        if (abs(conn_objects['s1']['dist'] - conn_objects['b']['dist']) <
                conn_diff_tolerance or
                abs(conn_objects['s2']['dist'] - conn_objects['b']['dist']) <
                conn_diff_tolerance):
            del conn_objects['b']

        # remove MV station as possible connection point
        if conn_objects['s1']['repr'] == \
                edisgo_object.topology.mv_grid.station.index[0]:
            del conn_objects['s1']
        elif conn_objects['s2']['repr'] == \
                edisgo_object.topology.mv_grid.station.index[0]:
            del conn_objects['s2']

        # find nearest connection point in conn_objects
        conn_objects_min = min(conn_objects.values(), key=lambda v: v['dist'])

        conn_objects_min_stack.append(conn_objects_min)

    # sort all objects by distance from node
    conn_objects_min_stack = [_ for _ in sorted(conn_objects_min_stack,
                                                key=lambda x: x['dist'])]

    return conn_objects_min_stack


def _connect_mv_node(edisgo_object, bus, target_obj):
    """
    Connects MV generators to target object in MV network

    If the target object is a bus, a new line is created to it.
    If the target object is a line, the node is connected to a newly created
    bus (using perpendicular projection) on this line.
    New lines are created using standard equipment.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    bus : pandas Series
        Data of bus to connect.
        Series has same rows as columns of topology.buses_df.
    target_obj : :class:`~.network.components.Component`
        Object that node shall be connected to

    Returns
    -------
    :class:`~.network.components.Component` or None
        Node that node was connected to

    Notes
    -----
    Adapted from `Ding0 <https://github.com/openego/ding0/blob/\
        21a52048f84ec341fe54e0204ac62228a9e8a32a/\
        ding0/network/mv_grid/mv_connect.py#L311>`_.

    """

    # get standard equipment
    std_line_type = edisgo_object.topology.equipment_data['mv_cables'].loc[
        edisgo_object.config['grid_expansion_standard_equipment']['mv_line']]
    std_line_kind = 'cable'

    target_obj_result = None

    srid = edisgo_object.topology.grid_district['srid']
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))

    # MV line is nearest connection point
    if isinstance(target_obj['shp'], LineString):

        line_data = edisgo_object.topology.lines_df.loc[target_obj['repr'], :]

        # find nearest point on MV line
        conn_point_shp = target_obj['shp'].interpolate(
            target_obj['shp'].project(bus_shp))
        conn_point_shp = transform(proj2conformal(srid), conn_point_shp)

        # create new branch tee bus
        branch_tee_repr = 'BranchTee_{}'.format(target_obj['repr'])
        edisgo_object.topology.add_bus(
            bus_name=branch_tee_repr,
            v_nom=edisgo_object.topology.mv_grid.nominal_voltage,
            x=conn_point_shp.x,
            y=conn_point_shp.y)

        # split old branch into 2 segments
        # (delete old branch and create 2 new ones along cable_dist)
        # ==========================================================

        # remove line from graph and equipment changes
        edisgo_object.topology.remove_line(line_data.name)

        _del_line_from_equipment_changes(
            edisgo_object=edisgo_object,
            line_repr=line_data.name)

        # add new line between newly created branch tee and line's bus0

        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=line_data.bus0,
            bus_target=branch_tee_repr)

        line_name_bus0 = edisgo_object.topology.add_line(
            bus0=branch_tee_repr,
            bus1=line_data.bus0,
            length=line_length,
            kind=line_data.kind,
            type_info=line_data.type_info)

        # add line to equipment changes to track costs
        _add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name_bus0, :])

        # add new line between newly created branch tee and line's bus0

        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=line_data.bus1,
            bus_target=branch_tee_repr)

        line_name_bus1 = edisgo_object.topology.add_line(
            bus0=branch_tee_repr,
            bus1=line_data.bus1,
            length=line_length,
            kind=line_data.kind,
            type_info=line_data.type_info)

        # add line to equipment changes to track costs
        _add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name_bus1, :])

        # add new branch for new node (node to branch tee)
        # ================================================
        line_length = calc_geo_dist_vincenty(network=network,
                                             node_source=node,
                                             node_target=branch_tee)
        line = Line(id=random.randint(10 ** 8, 10 ** 9),
                    length=line_length,
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
        if isinstance(bus, LVStation):
            valid_conn_objects = (LVStation, BranchTee)
        elif isinstance(node, Generator):
            valid_conn_objects = (LVStation, BranchTee, Generator)
        else:
            raise ValueError('Oops, the node you are trying to connect is not a valid connection object')

        # if target is generator or Load, check if it is aggregated (=> connection not allowed)
        if isinstance(target_obj['obj'], (Generator, Load)):
            target_is_aggregated = any([_ for _ in network.mv_grid.graph.adj[target_obj['obj']].values()
                                        if _['type'] == 'line_aggr'])
        else:
            target_is_aggregated = False

        # target node is not a load area of type aggregated
        if isinstance(target_obj['obj'], valid_conn_objects) and not target_is_aggregated:

            # add new branch for satellite (station to station)
            line_length = calc_geo_dist_vincenty(
                edisgo_object=edisgo_object,
                bus_source=bus,
                bus_target=target_obj['obj'])

            line = Line(id=random.randint(10 ** 8, 10 ** 9),
                        type=std_line_type,
                        kind=std_line_kind,
                        quantity=1,
                        length=line_length,
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
