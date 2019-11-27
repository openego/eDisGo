import random
import pandas as pd
import os

if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString, Point
    from shapely.ops import transform
    from shapely.wkt import loads as wkt_loads

from edisgo.tools.geo import \
    calc_geo_dist_vincenty, calc_geo_lines_in_buffer, \
    proj2equidistant, proj2equidistant_reverse

import logging

logger = logging.getLogger('edisgo')


def add_and_connect_mv_generator(edisgo_object, generator):
    """
    Add and connect new MV generator to existing grid.

    This function connects

        * generators of voltage level 4
            * to HV-MV station

        * generators of voltage level 5
            * to nearest MV bus or line
            * in case generator is connected to a line, the line is split and
              a new branch tee is added to connect new generator to

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    generator : pd.Series
        Pandas series with generator information such as electrical_capacity
        in MW and generation_type.

    """

    # ToDo use select_cable instead of standard line?

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
        p_nom=generator.electrical_capacity,
        generator_type=generator.generation_type,
        subtype=generator.generation_subtype,
        weather_cell_id=generator.w_id)

    # ===== voltage level 4: generator is connected to MV station =====
    if generator.voltage_level == 4:

        # add line

        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=gen_bus,
            bus_target=edisgo_object.topology.mv_grid.station.index[0])

        line_name = edisgo_object.topology.add_line(
            bus0=edisgo_object.topology.mv_grid.station.index[0],
            bus1=gen_bus,
            length=line_length,
            kind='cable',
            type_info=std_line_type.name)

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


def add_and_connect_lv_generator(edisgo_object, generator,
                                 allow_multiple_genos_per_load=True):
    """
    Add and connect new LV generator to existing grids.

    It connects

        * generators with an MV-LV station ID that does not exist (i.e.
          generators in an aggregated load area)
            * to MV-LV station

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
    edisgo_object : :class:`~.EDisGo`
    generator : pd.Series
        Pandas series with generator information such as electrical_capacity
        in MW and generation_type.
    allow_multiple_genos_per_load : :obj:`bool`
        If True, more than one generator can be connected to one load.

    Notes
    -----
    For the allocation, loads are selected randomly (sector-wise) using a
    predefined seed to ensure reproducibility.

    """

    # get list of LV grid IDs
    lv_grid_ids = [_.id for _ in edisgo_object.topology.mv_grid.lv_grids]

    add_generator_data = {
        'generator_id': generator.name,
        'p_nom': generator.electrical_capacity,
        'generator_type': generator.generation_type,
        'subtype': generator.generation_subtype,
        'weather_cell_id': generator.w_id}

    # determine LV grid the generator should be connected in

    # if substation ID (= LV grid ID) is given but it does not match an
    # existing LV grid ID (i.e. it is an aggregated LV grid), connect
    # generator to HV-MV substation
    if (generator.mvlv_subst_id and
            generator.mvlv_subst_id not in lv_grid_ids):
        # add generator
        edisgo_object.topology.add_generator(
            bus=edisgo_object.topology.mv_grid.station.index[0],
            **add_generator_data)
        return

    # if substation ID (= LV grid ID) is given and it matches an existing LV
    # grid ID (i.e. it is not an aggregated LV grid), set grid to connect
    # generator to to specified grid (in case the generator has no geometry
    # it is connected to the grid's station)
    elif generator.mvlv_subst_id and generator.mvlv_subst_id in lv_grid_ids:

        # get LV grid
        lv_grid = edisgo_object.topology._grids[
            'LVGrid_{}'.format(generator.mvlv_subst_id)]

        # if no geom is given, connect to LV grid's station
        if not generator.geom:
            # add generator
            edisgo_object.topology.add_generator(
                bus=lv_grid.station.index[0],
                **add_generator_data)
            logger.debug(
                "Generator {} has no geom entry and will be connected to "
                "grid's LV stations.".format(generator.id))
            return

    # if no MV-LV substation ID is given, choose random LV grid and connect
    # to station
    else:
        random.seed(a=generator.name)
        lv_grid = random.choice(lv_grid_ids)
        edisgo_object.topology.add_generator(
            bus=lv_grid.station.index[0],
            **add_generator_data)
        logger.warning(
            'Generator {} has no mvlv_subst_id. It is therefore allocated to '
            'a random LV Grid ({}); geom was set to stations\' geom.'.format(
                generator.id, lv_grid.id))
        return

    # generator is of v_level 6 -> connect to grid's LV station
    if generator.voltage_level == 6:

        gen_bus = 'Bus_Generator_{}'.format(generator.name)
        geom = wkt_loads(generator.geom)
        edisgo_object.topology.add_bus(
            bus_name=gen_bus,
            v_nom=lv_grid.nominal_voltage,
            x=geom.x,
            y=geom.y,
            lv_grid_id=lv_grid.id)

        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=gen_bus,
            bus_target=lv_grid.station.index[0])

        # get standard equipment
        std_line_type = edisgo_object.topology.equipment_data['lv_cables'].loc[
            edisgo_object.config[
                'grid_expansion_standard_equipment']['lv_line']]
        line_name = edisgo_object.topology.add_line(
            bus0=lv_grid.station.index[0],
            bus1=gen_bus,
            length=line_length,
            kind='cable',
            type_info=std_line_type.name)

        # add line to equipment changes to track costs
        _add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name])

        # add generator
        edisgo_object.topology.add_generator(
            bus=gen_bus,
            **add_generator_data)

    # generator is of v_level 7 -> assign generator to load
    # generators with P <= 30kW are connected to residential loads, if
    # available; generators with 30kW <= P <= 100kW are connected to
    # retail, industrial, or agricultural load, if available
    # in case no load is available the generator is connected to random
    # bus in LV grid
    # if load to connect to is available the generator is connected to
    # load with less than two generators
    # if every load has two or more generators choose the first load
    # from random sample
    elif generator.voltage_level == 7:

        lv_loads = lv_grid.loads_df
        if generator.electrical_capacity <= 30:
            target_loads = lv_loads[lv_loads.sector == 'residential']
        else:
            target_loads = lv_loads[lv_loads.sector.isin(
                ['industrial', 'agricultural', 'retail'])]

        # generate random list (unique elements) of possible target loads
        # to connect generators to
        random.seed(a=generator.name)
        if len(target_loads) > 0:
            lv_loads_rnd = set(random.sample(
                list(target_loads.index), len(target_loads)))
        else:
            logger.debug(
                'No load to connect LV generator to. The '
                'generator is therefore connected to random LV bus.')
            gen_bus = random.choice(
                lv_grid.buses_df[lv_grid.buses_df.in_building].index)
            # add generator
            edisgo_object.topology.add_generator(
                bus=gen_bus,
                **add_generator_data)
            return

        # search through list of loads for load with less
        # than two generators
        lv_conn_target = None
        while len(lv_loads_rnd) > 0 and lv_conn_target is None:

            lv_load = lv_loads_rnd.pop()

            # determine number of generators of LV load
            load_bus = target_loads.at[lv_load, 'bus']
            if edisgo_object.topology.buses_df.at[
                load_bus, 'in_building'] is not True:
                neighbours = \
                    list(edisgo_object.topology.get_neighbours(load_bus))
                branch_tee_in_building = neighbours[0]
                #ToDo handle boolean true/false
                if len(neighbours) > 1 or \
                        edisgo_object.topology.buses_df.at[
                            branch_tee_in_building, 'in_building'] is not \
                        True:
                    raise ValueError(
                        "Expected neighbour to be branch tee in building.")
            else:
                branch_tee_in_building = load_bus
            generators_at_load = edisgo_object.topology.generators_df[
                edisgo_object.topology.generators_df.bus.isin(
                    [load_bus, branch_tee_in_building])]
            if len(generators_at_load) < 2:
                lv_conn_target = branch_tee_in_building

        if lv_conn_target is None:
            logger.debug(
                'No valid connection target found for generator {}. '
                'Connected to LV station.'.format(
                    generator.name))

            station_bus = lv_grid.station.index[0]

            gen_bus = 'Bus_Generator_{}'.format(generator.name)
            lv_conn_target = gen_bus
            geom = wkt_loads(generator.geom)
            edisgo_object.topology.add_bus(
                bus_name=gen_bus,
                v_nom=lv_grid.nominal_voltage,
                x=geom.x,
                y=geom.y,
                lv_grid_id=lv_grid.id)

            line_length = calc_geo_dist_vincenty(
                edisgo_object=edisgo_object,
                bus_source=gen_bus,
                bus_target=station_bus)

            # get standard equipment
            std_line_type = \
            edisgo_object.topology.equipment_data['lv_cables'].loc[
                edisgo_object.config[
                    'grid_expansion_standard_equipment']['lv_line']]
            line_name = edisgo_object.topology.add_line(
                bus0=station_bus,
                bus1=gen_bus,
                length=line_length,
                kind='cable',
                type_info=std_line_type.name)

            # add line to equipment changes to track costs
            _add_line_to_equipment_changes(
                edisgo_object=edisgo_object,
                line=edisgo_object.topology.lines_df.loc[line_name])

        # add generator
        edisgo_object.topology.add_generator(
            bus=lv_conn_target,
            **add_generator_data)
        return


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

    srid = edisgo_object.topology.grid_district['srid']
    bus_shp = transform(proj2equidistant(srid), Point(bus.x, bus.y))

    # MV line is nearest connection point => split old line into 2 segments
    # (delete old line and create 2 new ones)
    if isinstance(target_obj['shp'], LineString):

        line_data = edisgo_object.topology.lines_df.loc[target_obj['repr'], :]

        # find nearest point on MV line
        conn_point_shp = target_obj['shp'].interpolate(
            target_obj['shp'].project(bus_shp))
        conn_point_shp = transform(proj2equidistant_reverse(srid),
                                   conn_point_shp)

        # create new branch tee bus
        branch_tee_repr = 'BranchTee_{}'.format(target_obj['repr'])
        edisgo_object.topology.add_bus(
            bus_name=branch_tee_repr,
            v_nom=edisgo_object.topology.mv_grid.nominal_voltage,
            x=conn_point_shp.x,
            y=conn_point_shp.y)

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

        # add line to equipment changes
        # ToDo @Anya?
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

        # add line to equipment changes
        _add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[line_name_bus1, :])

        # add new line for new bus
        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=bus.name,
            bus_target=branch_tee_repr)

        new_line_name = edisgo_object.topology.add_line(
            bus0=branch_tee_repr,
            bus1=bus.name,
            length=line_length,
            kind=std_line_kind,
            type_info=std_line_type.name)

        # add line to equipment changes
        _add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[new_line_name, :])

        # remove old line from topology and equipment changes
        edisgo_object.topology.remove_line(line_data.name)

        _del_line_from_equipment_changes(
            edisgo_object=edisgo_object,
            line_repr=line_data.name)

        return branch_tee_repr

    # node ist nearest connection point
    else:

        # add new branch for satellite (station to station)
        line_length = calc_geo_dist_vincenty(
            edisgo_object=edisgo_object,
            bus_source=bus.name,
            bus_target=target_obj['repr'])

        new_line_name = edisgo_object.topology.add_line(
            bus0=target_obj['repr'],
            bus1=bus.name,
            length=line_length,
            kind=std_line_kind,
            type_info=std_line_type.name)

        # add line to equipment changes
        _add_line_to_equipment_changes(
            edisgo_object=edisgo_object,
            line=edisgo_object.topology.lines_df.loc[new_line_name, :])

        return target_obj['repr']


def _check_mvlv_subst_id(generator, mvlv_subst_id, lv_grid_dict):
    """
    Checks if MV/LV substation id of single LV generator is valid.

    In case it is not valid or missing, a random one from existing stations
    in LV grids will be assigned.

    Parameters
    ----------
    generator : :class:`~.network.components.Generator`
        LV generator
    mvlv_subst_id : :obj:`int`
        MV-LV substation id
    lv_grid_dict : :obj:`dict`
        Dict of existing LV grids
        Format: {:obj:`int`: :class:`~.network.grids.LVGrid`}

    Returns
    -------
    :class:`~.network.grids.LVGrid`
        LV network of generator

    """

    if mvlv_subst_id and not isnan(mvlv_subst_id):
        # assume that given LA exists
        try:
            # get LV grid
            lv_grid = lv_grid_dict[mvlv_subst_id]

            # if no geom, use geom of station
            if not generator.geom:
                generator.geom = lv_grid.station.geom
                logger.debug(
                    "Generator {} has no geom entry, stations' geom will "
                    "be used.".format(generator.id))
            return lv_grid

        # if LA/LVGD does not exist, choose random LVGD and move generator
        # to station of LVGD
        # this occurs due to exclusion of LA with peak load < 1kW
        except:
            lv_grid = random.choice(list(lv_grid_dict.values()))
            generator.geom = lv_grid.station.geom

            logger.warning('Generator {} cannot be assigned to '
                           'non-existent LV Grid and was '
                           'allocated to a random LV Grid ({}); '
                           'geom was set to stations\' geom.'
                           .format(repr(generator),
                                   repr(lv_grid)))
            return lv_grid

    else:
        lv_grid = random.choice(list(lv_grid_dict.values()))
        generator.geom = lv_grid.station.geom

        logger.warning('Generator {} has no mvlv_subst_id and was '
                       'allocated to a random LV Grid ({}); '
                       'geom was set to stations\' geom.'
                       .format(repr(generator),
                               repr(lv_grid)))
        return lv_grid
