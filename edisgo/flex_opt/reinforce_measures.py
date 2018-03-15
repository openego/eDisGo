import copy
import math
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra as \
    dijkstra_shortest_path_length

from edisgo.grid.components import Transformer, BranchTee, Generator, Load, \
    LVStation
from edisgo.grid.grids import LVGrid
from edisgo.flex_opt import exceptions

import logging
logger = logging.getLogger('edisgo')


def extend_distribution_substation_overloading(network, critical_stations):
    """
    Reinforce MV/LV substations due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    critical_stations : dict
        Dictionary with critical :class:`~.grid.components.LVStation`
        Format: {lv_station_1: overloading_1, ..., lv_station_n: overloading_n}

    Returns
    -------
    dict
        Dictionary with lists of added and removed transformers.

    """

    # get parameters for standard transformer
    try:
        standard_transformer = network.equipment_data['lv_trafos'].loc[
            network.config['grid_expansion_standard_equipment'][
                'mv_lv_transformer']]
    except KeyError:
        print('Standard MV/LV transformer is not in equipment list.')

    #ToDo: differentiate between load and feed-in case!
    load_factor = network.config['grid_expansion_load_factors'][
        'lv_feedin_case_transformer']

    transformers_changes = {'added': {}, 'removed': {}}
    for station in critical_stations:

        # list of maximum power of each transformer in the station
        s_max_per_trafo = [_.type.S_nom for _ in station.transformers]

        # maximum station load from power flow analysis
        s_station_pfa = critical_stations[station]

        # determine missing transformer power to solve overloading issue
        s_trafo_missing = s_station_pfa - (sum(s_max_per_trafo) * load_factor)

        # check if second transformer of the same kind is sufficient
        # if true install second transformer, otherwise install as many
        # standard transformers as needed
        if max(s_max_per_trafo) >= s_trafo_missing:
            # if station has more than one transformer install a new
            # transformer of the same kind as the transformer that best
            # meets the missing power demand
            duplicated_transformer = min(
                [_ for _ in station.transformers
                 if _.type.S_nom > s_trafo_missing],
                key=lambda j: j.type.S_nom - s_trafo_missing)

            new_transformer = Transformer(
                id='LVStation_{}_transformer_{}'.format(
                    str(station.id), str(len(station.transformers) + 1)),
                geom=duplicated_transformer.geom,
                mv_grid=duplicated_transformer.mv_grid,
                grid=duplicated_transformer.grid,
                voltage_op=duplicated_transformer.voltage_op,
                type=copy.deepcopy(duplicated_transformer.type))

            # add transformer to station and return value
            station.add_transformer(new_transformer)
            transformers_changes['added'][station] = [new_transformer]

        else:
            # get any transformer to get attributes for new transformer from
            station_transformer = station.transformers[0]

            # calculate how many parallel standard transformers are needed
            number_transformers = math.ceil(
                s_station_pfa / standard_transformer.S_nom)

            # add transformer to station
            new_transformers = []
            for i in range(number_transformers):
                new_transformer = Transformer(
                    id='LVStation_{}_transformer_{}'.format(
                        str(station.id), str(i + 1)),
                    geom=station_transformer.geom,
                    mv_grid=station_transformer.mv_grid,
                    grid=station_transformer.grid,
                    voltage_op=station_transformer.voltage_op,
                    type=copy.deepcopy(standard_transformer))
                new_transformers.append(new_transformer)
            transformers_changes['added'][station] = new_transformers
            transformers_changes['removed'][station] = station.transformers
            station.transformers = new_transformers
    return transformers_changes


def extend_distribution_substation_overvoltage(network, critical_stations):
    """
    Reinforce MV/LV substations due to voltage issues.

    A parallel standard transformer is installed.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    critical_stations : dict
        Dictionary with critical :class:`~.grid.components.LVStation`
        Format: {lv_station_1: overloading_1, ..., lv_station_n: overloading_n}

    Returns
    -------
    Dictionary with lists of added transformers.

    """

    # get parameters for standard transformer
    try:
        standard_transformer = network.equipment_data['lv_trafos'].loc[
            network.config['grid_expansion_standard_equipment'][
                'mv_lv_transformer']]
    except KeyError:
        print('Standard MV/LV transformer is not in equipment list.')

    transformers_changes = {'added': {}}
    for grid, voltage_deviation in critical_stations.items():

        # get any transformer to get attributes for new transformer from
        station_transformer = grid.station.transformers[0]

        new_transformer = Transformer(
            id='LVStation_{}_transformer_{}'.format(
                str(grid.station.id), str(len(grid.station.transformers) + 1)),
            geom=station_transformer.geom,
            mv_grid=station_transformer.mv_grid,
            grid=station_transformer.grid,
            voltage_op=station_transformer.voltage_op,
            type=copy.deepcopy(standard_transformer))

        # add standard transformer to station and return value
        grid.station.add_transformer(new_transformer)
        transformers_changes['added'][grid.station] = [new_transformer]

    if transformers_changes['added']:
        logger.debug("==> {} LV station(s) has/have been reinforced ".format(
            str(len(transformers_changes['added']))) +
                    "due to overloading issues.")

    return transformers_changes


def extend_substation_overloading(network, critical_stations):
    """
    Reinforce HV/MV station due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    critical_stations : dict
        Dictionary with critical :class:`~.grid.components.MVStation` and
        maximum apparent power from power flow analysis.
        Format: {MVStation: S_max}

    Returns
    -------
    Dictionary with lists of added and removed transformers.

    """

    # get parameters for standard transformer
    try:
        standard_transformer = network.equipment_data['mv_trafos'].loc[
            network.config['grid_expansion_standard_equipment'][
                'hv_mv_transformer']]
    except KeyError:
        print('Standard HV/MV transformer is not in equipment list.')

    # ToDo: differentiate between load and feed-in case!
    load_factor = \
        network.config['grid_expansion_load_factors'][
            'mv_feedin_case_transformer']

    transformers_changes = {'added': {}, 'removed': {}}
    for station in critical_stations:

        # list of maximum power of each transformer in the station
        s_max_per_trafo = [_.type.S_nom for _ in station.transformers]

        # maximum station load from power flow analysis
        s_station_pfa = critical_stations[station]

        # determine missing transformer power to solve overloading issue
        s_trafo_missing = s_station_pfa - (sum(s_max_per_trafo) * load_factor)

        # check if second transformer of the same kind is sufficient
        # if true install second transformer, otherwise install as many
        # standard transformers as needed
        if max(s_max_per_trafo) >= s_trafo_missing:
            # if station has more than one transformer install a new
            # transformer of the same kind as the transformer that best
            # meets the missing power demand
            duplicated_transformer = min(
                [_ for _ in station.transformers
                 if _.type.S_nom > s_trafo_missing],
                key=lambda j: j.type.S_nom - s_trafo_missing)

            new_transformer = Transformer(
                id='MVStation_{}_transformer_{}'.format(
                    str(station.id), str(len(station.transformers) + 1)),
                geom=duplicated_transformer.geom,
                grid=duplicated_transformer.grid,
                voltage_op=duplicated_transformer.voltage_op,
                type=copy.deepcopy(duplicated_transformer.type))

            # add transformer to station and return value
            station.add_transformer(new_transformer)
            transformers_changes['added'][station] = [new_transformer]

        else:
            # get any transformer to get attributes for new transformer from
            station_transformer = station.transformers[0]

            # calculate how many parallel standard transformers are needed
            number_transformers = math.ceil(
                s_station_pfa / standard_transformer.S_nom)

            # add transformer to station
            new_transformers = []
            for i in range(number_transformers):
                new_transformer = Transformer(
                    id='MVStation_{}_transformer_{}'.format(
                        str(station.id), str(i + 1)),
                    geom=station_transformer.geom,
                    grid=station_transformer.grid,
                    voltage_op=station_transformer.voltage_op,
                    type=copy.deepcopy(standard_transformer))
                new_transformers.append(new_transformer)
            transformers_changes['added'][station] = new_transformers
            transformers_changes['removed'][station] = station.transformers
            station.transformers = new_transformers

    if transformers_changes['added']:
        logger.debug("==> MV station has been reinforced due to overloading "
                     "issues.")

    return transformers_changes


def reinforce_branches_overvoltage(network, grid, crit_nodes):
    """
    Reinforce MV and LV grid due to voltage issues.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    grid : :class:`~.grid.grids.MVGrid` or :class:`~.grid.grids.LVGrid`
    crit_nodes : :pandas:`pandas.Series<series>`
        Series with critical nodes of one grid and corresponding voltage
        deviation, sorted descending by voltage deviation.

    Returns
    -------
    Dictionary with :class:`~.grid.components.Line` and the number of Lines
    added.

    Notes
    -----
    Reinforce measures:

    1. Disconnect line at 2/3 of the length between station and critical node
    farthest away from the station and install new standard line
    2. Install parallel standard line

    In LV grids only lines outside buildings are reinforced; loads and
    generators in buildings cannot be directly connected to the MV/LV station.

    In MV grids lines can only be disconnected at LVStations because they
    have switch disconnectors needed to operate the lines as half rings (loads
    in MV would be suitable as well because they have a switch bay (Schaltfeld)
    but loads in dingo are only connected to MV busbar). If there is no
    suitable LV station the generator is directly connected to the MV busbar.
    There is no need for a switch disconnector in that case because generators
    don't need to be n-1 safe.

    References
    ----------

    The method of grid reinforce as implemented here bases on
    [VerteilnetzstudieBW]_ and [EAMS]_.

    """

    # load standard line data
    if isinstance(grid, LVGrid):
        try:
            standard_line = network.equipment_data['lv_cables'].loc[
                network.config['grid_expansion_standard_equipment']['lv_line']]
        except KeyError:
            print('Chosen standard LV line is not in equipment list.')
    else:
        try:
            standard_line = network.equipment_data['mv_cables'].loc[
                network.config['grid_expansion_standard_equipment']['mv_line']]
        except KeyError:
            print('Chosen standard MV line is not in equipment list.')

    # find first nodes of every main line as representatives
    rep_main_line = list(
        nx.predecessor(grid.graph, grid.station, cutoff=1).keys())
    # list containing all representatives of main lines that have already been
    # reinforced
    main_line_reinforced = []

    lines_changes = {}
    for i in range(len(crit_nodes)):
        path = nx.shortest_path(grid.graph, grid.station,
                                crit_nodes.index[i])
        # raise exception if voltage issue occurs at station's secondary side
        # because voltage issues should have been solved during extension of
        # distribution substations due to overvoltage issues.
        if len(path) == 1:
            logging.error("Voltage issues at busbar in LV grid {} should have "
                          "been solved in previous steps.".format(grid))
        else:
            # check if representative of line is already in list
            # main_line_reinforced; if it is, the main line the critical node
            # is connected to has already been reinforced in this iteration
            # step
            if not path[1] in main_line_reinforced:

                main_line_reinforced.append(path[1])
                # get path length from station to critical node
                get_weight = lambda u, v, data: data['line'].length
                path_length = dijkstra_shortest_path_length(
                    grid.graph, grid.station, get_weight,
                    target=crit_nodes.index[i])
                # find first node in path that exceeds 2/3 of the line length
                # from station to critical node farthest away from the station
                node_2_3 = next(j for j in path if
                                path_length[j] >= path_length[
                                    crit_nodes.index[i]] * 2 / 3)

                # if LVGrid: check if node_2_3 is outside of a house
                # and if not find next BranchTee outside the house
                if isinstance(grid, LVGrid):
                    if isinstance(node_2_3, BranchTee):
                        if node_2_3.in_building:
                            # ToDo more generic (new function)
                            try:
                                node_2_3 = path[path.index(node_2_3) - 1]
                            except IndexError:
                                print('BranchTee outside of building is not ' +
                                      'in path.')
                    elif (isinstance(node_2_3, Generator) or
                              isinstance(node_2_3, Load)):
                        pred_node = path[path.index(node_2_3) - 1]
                        if isinstance(pred_node, BranchTee):
                            if pred_node.in_building:
                                # ToDo more generic (new function)
                                try:
                                    node_2_3 = path[path.index(node_2_3) - 2]
                                except IndexError:
                                    print('BranchTee outside of building is ' +
                                          'not in path.')
                    else:
                        logging.error("Not implemented for {}.".format(
                            str(type(node_2_3))))
                # if MVGrid: check if node_2_3 is LV station and if not find
                # next LV station
                else:
                    if not isinstance(node_2_3, LVStation):
                        next_index = path.index(node_2_3) + 1
                        try:
                            # try to find LVStation behind node_2_3
                            while not isinstance(node_2_3, LVStation):
                                node_2_3 = path[next_index]
                                next_index += 1
                        except IndexError:
                            # if no LVStation between node_2_3 and node with
                            # voltage problem, connect node directly to
                            # MVStation
                            node_2_3 = crit_nodes.index[i]

                # if node_2_3 is a representative (meaning it is already
                # directly connected to the station), line cannot be
                # disconnected and must therefore be reinforced
                if node_2_3 in rep_main_line:
                    crit_line = grid.graph.get_edge_data(
                        grid.station, node_2_3)['line']

                    # if critical line is already a standard line install one
                    # more parallel line
                    if crit_line.type.name == standard_line.name:
                        crit_line.quantity += 1
                        lines_changes[crit_line] = 1

                    # if critical line is not yet a standard line replace old
                    # line by a standard line
                    else:
                        # number of parallel standard lines could be calculated
                        # following [2] p.103; for now number of parallel
                        # standard lines is iterated
                        crit_line.type = standard_line.copy()
                        crit_line.quantity = 1
                        crit_line.kind = 'cable'
                        lines_changes[crit_line] = 1

                # if node_2_3 is not a representative, disconnect line
                else:
                    # get line between node_2_3 and predecessor node (that is
                    # closer to the station)
                    pred_node = path[path.index(node_2_3) - 1]
                    crit_line = grid.graph.get_edge_data(
                        node_2_3, pred_node)['line']
                    # add new edge between node_2_3 and station
                    new_line_data = {'line': crit_line,
                                     'type': 'line'}
                    grid.graph.add_edge(grid.station, node_2_3, new_line_data)
                    # remove old edge
                    grid.graph.remove_edge(pred_node, node_2_3)
                    # change line length and type
                    crit_line.length = path_length[node_2_3]
                    crit_line.type = standard_line.copy()
                    crit_line.kind = 'cable'
                    crit_line.quantity = 1
                    lines_changes[crit_line] = 1
                    # add node_2_3 to representatives list to not further
                    # reinforce this part off the grid in this iteration step
                    rep_main_line.append(node_2_3)
                    main_line_reinforced.append(node_2_3)

            else:
                logger.debug(
                    '==> Main line of node {} in grid {} '.format(
                        str(crit_nodes.index[i]), str(grid)) +
                    'has already been reinforced.')

    if main_line_reinforced:
        logger.debug('==> {} branche(s) was/were reinforced '.format(
            str(len(lines_changes))) + 'due to over-voltage issues.')

    return lines_changes


def reinforce_branches_overloading(network, crit_lines):
    """
    Reinforce MV or LV grid due to overloading.
    
    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    crit_lines : dict
        Dictionary of critical :class:`~.grid.components.Line` with max.
        relative overloading
        Format: {line_1: rel_overloading_1, ..., line_n: rel_overloading_n}

    Returns
    -------
    Dictionary with :class:`~.grid.components.Line` and the number of Lines
    added.
        
    Notes
    -----
    Reinforce measures:

    1. Install parallel line of the same type as the existing line (Only if
       line is a cable, not an overhead line. Otherwise a standard equipment
       cable is installed right away.)
    2. Remove old line and install as many parallel standard lines as
       needed.

    """

    # load standard line data
    try:
        standard_line_lv = network.equipment_data['lv_cables'].loc[
            network.config['grid_expansion_standard_equipment']['lv_line']]
    except KeyError:
        print('Chosen standard LV line is not in equipment list.')
    try:
        standard_line_mv = network.equipment_data['mv_cables'].loc[
            network.config['grid_expansion_standard_equipment']['mv_line']]
    except KeyError:
        print('Chosen standard MV line is not in equipment list.')

    lines_changes = {}
    for crit_line, rel_overload in crit_lines.items():
        # check if line is in LV or MV and set standard line accordingly
        if isinstance(crit_line.grid, LVGrid):
            standard_line = standard_line_lv
        else:
            standard_line = standard_line_mv

        if crit_line.type.name == standard_line.name:
            # check how many parallel standard lines are needed
            number_parallel_lines = math.ceil(
                rel_overload * crit_line.quantity)
            lines_changes[crit_line] = (number_parallel_lines -
                                        crit_line.quantity)
            crit_line.quantity = number_parallel_lines
        else:
            # check if parallel line of the same kind is sufficient
            if (crit_line.quantity == 1 and rel_overload <= 2
                    and crit_line.kind == 'cable'):
                crit_line.quantity = 2
                lines_changes[crit_line] = 1
            else:
                number_parallel_lines = math.ceil(
                    crit_line.type['I_max_th'] * rel_overload /
                    standard_line['I_max_th'])
                lines_changes[crit_line] = number_parallel_lines
                crit_line.type = standard_line.copy()
                crit_line.quantity = number_parallel_lines
                crit_line.kind = 'cable'

    if crit_lines:
        logger.debug('==> {} branche(s) was/were reinforced '.format(
            str(len(crit_lines))) + 'due to over-loading issues.')

    return lines_changes
