import copy
import math
import pandas as pd
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra as \
    dijkstra_shortest_path_length

from edisgo.grid.components import Transformer
from edisgo.grid.grids import LVGrid

import logging

logger = logging.getLogger('ding0')


# ToDo: Return reinforced components to results object
def extend_distribution_substation(network, critical_stations):
    """
    Reinforce MV/LV substations.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    network : edisgo network object
    critical_stations : dict
        Dictionary with key holding the station name and values the
        corresponding station load.

    Returns
    -------
    Dictionary with lists of added and removed transformers.

    """

    # get parameters for standard transformer
    try:
        standard_transformer = network.equipment_data['LV_trafos'].loc[
            network.config['grid_expansion']['std_mv_lv_transformer']]
    except KeyError:
        print('Standard MV/LV transformer is not in equipment list.')

    load_factor_mv_lv_transformer = float(network.config['grid_expansion'][
        'load_factor_mv_lv_transformer'])

    transformers_changes = {'added': [], 'removed': []}
    for station in critical_stations:

        # list of maximum power of each transformer in the station
        s_max_per_trafo = [_.type.s for _ in station.transformers]

        # maximum station load
        s_max_station = critical_stations[station]

        # determine missing trafo power to solve overloading issue
        s_trafo_missing = s_max_station - (
            sum(s_max_per_trafo) * load_factor_mv_lv_transformer)

        # check if second transformer of the same kind is sufficient
        # if true install second transformer, otherwise install as many
        # standard transformers as needed
        if max(s_max_per_trafo) >= s_trafo_missing:
            # if station has more than one transformer install a new
            # transformer of the same kind as the transformer that best
            # meets the missing power demand
            duplicated_transformer = min(
                [_ for _ in station.transformers
                 if _.type.s > s_trafo_missing],
                key=lambda j: j.type.s - s_trafo_missing)

            new_transformer = Transformer(
                id='LV_station_{}_transformer_{}'.format(
                    str(station.id), str(len(station.transformers) + 1)),
                geom=duplicated_transformer.geom,
                grid=duplicated_transformer.grid,
                voltage_op=duplicated_transformer.voltage_op,
                type=copy.deepcopy(duplicated_transformer.type))

            # add transformer to station and return value
            station.add_transformer(new_transformer)
            transformers_changes['added'].append(new_transformer)

        else:
            # get any transformer to get attributes for new transformer from
            station_transformer = station.transformers[0]

            # calculate how many parallel standard transformers are needed
            number_transformers = math.ceil(
                s_max_station / standard_transformer.s_nom)

            # add transformer to station
            new_transformers = []
            for i in range(number_transformers):
                new_transformer = Transformer(
                    id='LV_station_{}_transformer_{}'.format(
                        str(station.id), str(i)),
                    geom=station_transformer.geom,
                    grid=station_transformer.grid,
                    voltage_op=station_transformer.voltage_op,
                    type=copy.deepcopy(standard_transformer))
                new_transformers.append(new_transformer)
            transformers_changes['added'].extend(new_transformers)
            transformers_changes['removed'].extend(station.transformers)
            station.transformers = new_transformers

    logger.info("{} have been reinforced due to overloading "
                "issues.".format(str(len(critical_stations))))

    return transformers_changes


def reinforce_branches_voltage(network, crit_nodes):
    """
    Reinforce MV and LV grid due to voltage issues.

    Parameters
    ----------
    network : edisgo network object
    crit_nodes : pd.Series
        pd.Series with critical nodes of one grid as index and corresponding
        voltage deviation as values

    Returns
    -------
    Dictionary with lists of added and removed lines.

    Notes
    -----
    Reinforce measures:
    1. Disconnect line at 2/3 of the length between station and critical node
    farthest away from the station and install new standard line
    2. Install parallel standard line

    References
    ----------
    .. [1] "Verteilnetzstudie für das Land Baden-Württemberg"
    .. [2] "Technische Richtlinie Erzeugungsanlagen am Mittelspannungsnetz -
            Richtlinie für Anschluss und Parallelbetrieb von
            Erzeugungsanlagen am Mittelspannungsnetz, Juni 2008"

    """

    # ToDo: gilt Methodik auch für die MS?

    # load standard line data
    grid = crit_nodes.index[0].grid
    if isinstance(grid, LVGrid):
        try:
            standard_line = network.equipment_data['LV_cables'].loc[
                network.config['grid_expansion']['std_lv_line']]
        except KeyError:
            print('Chosen standard LV line is not in equipment list.')
    else:
        try:
            standard_line = network.equipment_data['MV_cables'].loc[
                network.config['grid_expansion']['std_mv_line']]
        except KeyError:
            print('Chosen standard MV line is not in equipment list.')

    # find first nodes of every main line as representatives
    rep_main_line = nx.predecessor(grid.graph, grid.station, cutoff=1)
    # list containing all representatives of main lines that have already been
    # reinforced
    main_line_reinforced = []

    lines_changes = {'added': [], 'removed': []}
    for i in range(len(crit_nodes)):
        path = nx.shortest_path(grid.graph, grid.station,
                                crit_nodes.index[i])

        # check if representative of line is already in list
        # main_line_reinforced, if it is the main line the critical node is
        # connected to has already been reinforced in this iteration step
        if not path[1] in main_line_reinforced:

            main_line_reinforced.append(path[1])
            # get path length from station to critical node
            get_weight = lambda u, v, data: data['line'].length
            path_length = dijkstra_shortest_path_length(
                grid.graph, grid.station, get_weight,
                target=crit_nodes.index[i])
            # find first node in path that exceeds 2/3 of the line length
            # from station to critical node farthest away from the station
            node_2_3 = next(j for j in path if path_length[j] >= path_length[
                crit_nodes.index[i]] * 2 / 3)

            # if node_2_3 is a representative (meaning it is already directly
            # connected to the station), line cannot be disconnected and must
            # therefore be reinforced
            if node_2_3 in rep_main_line:
                crit_line = grid.graph.get_edge_data(
                    grid.station, node_2_3)['line']

                # if critical line is already a standard line install one more
                # parallel line
                if crit_line.type.name == standard_line.name:
                    crit_line.quantity += 1
                    lines_changes['added'].append(crit_line.type)

                # if critical line is not yet a standard line replace old line
                # by a standard line
                else:
                    # number of parallel standard lines could be calculated
                    # following [2] p.103; for now number of parallel standard
                    # lines is iterated
                    lines_changes['removed'].append(crit_line.type)
                    crit_line.type = standard_line.copy()
                    crit_line.quantity = 1
                    lines_changes['added'].append(crit_line.type)

            # if node_2_3 is not a representative, disconnect line
            else:
                # get line between node_2_3 and predecessor node (that is
                # closer to the station)
                pred_node = path[path.index(node_2_3) - 1]
                crit_line = grid.graph.get_edge_data(
                    node_2_3, pred_node)['line']
                lines_changes['removed'].append(crit_line.type)
                # add new edge between node_2_3 and station
                new_line_data = {'line': crit_line,
                                 'type': 'line'}
                grid.graph.add_edge(grid.station, node_2_3, new_line_data)
                # remove old edge
                grid.graph.remove_edge(pred_node, node_2_3)
                # change line length and type
                crit_line.length = path_length[node_2_3]
                crit_line.type = standard_line.copy()
                lines_changes['added'].append(crit_line.type)

        else:
            logger.debug(
                '==> Main line of node {} '.format(str(crit_nodes.index[i])) +
                'in LV grid {} '.format(str(grid)) +
                'has already been reinforced.')

    if main_line_reinforced:
        logger.info('==> {} branche(s) was/were reinforced.'.format(
            str(len(main_line_reinforced))))

    return lines_changes


def reinforce_branches_current(network, crit_lines):
    """
    Reinforce MV or LV grid due to overloading.
    
    Parameters
    ----------
    network : edisgo network object
    crit_lines : dict
        Dict with critical lines as keys and their max. relative
        overloading.

    Returns
    -------
    type 
        #TODO: Description of return.
        
    Notes
    -----
        Reinforce measures:
        1. Install parallel line of the same type as the existing line
        2. Remove old line and install as many parallel standard lines as
           needed
        The branch type to be installed is determined per branch using the rel.
        overloading. According to [2]_  only cables are installed.

    """

    # load standard line data
    try:
        standard_line_lv = network.equipment_data['LV_cables'].loc[
            network.config['grid_expansion']['std_lv_line']]
    except KeyError:
        print('Chosen standard LV line is not in equipment list.')
    try:
        standard_line_mv = network.equipment_data['MV_cables'].loc[
            network.config['grid_expansion']['std_mv_line']]
    except KeyError:
        print('Chosen standard MV line is not in equipment list.')

    for crit_line, rel_overload in crit_lines.items():
        # check if line is in LV or MV and set standard line accordingly
        if isinstance(crit_line.grid, LVGrid):
            standard_line = standard_line_lv
        else:
            standard_line = standard_line_mv

        if crit_line.type.name == standard_line.name:
            # check how many parallel standard lines are needed
            number_parallel_lines = math.ceil(
                crit_line.type['I_max_th'] * rel_overload /
                standard_line['I_max_th'])
            crit_line.quantity = number_parallel_lines
        else:
            # check if parallel line of the same kind is sufficient
            if (crit_line.type['I_max_th'] * rel_overload <=
                    crit_line.type['I_max_th'] * 2):
                crit_line.quantity = 2
            else:
                number_parallel_lines = math.ceil(
                    crit_line.type['I_max_th'] * rel_overload /
                    standard_line['I_max_th'])
                crit_line.type = standard_line.copy()
                crit_line.quantity = number_parallel_lines

    if crit_lines:
        logger.info('==> {} branches were reinforced.'.format(
            str(len(crit_lines))))
