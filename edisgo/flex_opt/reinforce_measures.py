import copy
import math
import pandas as pd
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra as \
    dijkstra_shortest_path_length

import ding0
from edisgo.grid.components import Transformer
from edisgo import flex_opt
# from ding0.tools import config as cfg_ding0
# from ding0.grid.lv_grid.build_grid import select_transformers
# from ding0.flexopt.check_tech_constraints import get_voltage_at_bus_bar
# import networkx as nx
import logging

package_path = ding0.__path__[0]
logger = logging.getLogger('ding0')


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
        corresponding station load

    """

    # get parameters for standard transformer
    try:
        standard_transformer = network.equipment_data['LV_trafos'].loc[
            network.config['grid_expansion']['std_mv_lv_transformer']]
    except KeyError:
        print('Standard MV/LV transformer is not in equipment list.')

    load_factor_mv_lv_transformer = float(network.config['grid_expansion'][
        'load_factor_mv_lv_transformer'])

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
                key=lambda i: i.type.s - s_trafo_missing)

            new_transformer = Transformer(
                id='LV_station_{}_transformer_{}'.format(
                    str(station.id), str(len(station.transformers))),
                geom=duplicated_transformer.geom,
                grid=duplicated_transformer.grid,
                voltage_op=duplicated_transformer.voltage_op,
                type=copy.deepcopy(duplicated_transformer.type))

            # add transformer to station
            # ToDo: Methode in Station hierfür einführen?
            station._transformers.append(new_transformer)

        else:
            # get any transformer to get attributes for new transformer from
            station_transformer = station.transformers[0]


            # calculate how many parallel standard transformers are needed
            number_transformers = math.ceil(s_max_gc / standard_transformer.s)

            # add transformer to station
            # ToDo: Methode in Station hierfür einführen?
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
            station.transformers = new_transformers

    logger.info("{stations_cnt} have been reinforced due to overloading "
                "issues.".format(stations_cnt=len(critical_stations)))


def reinforce_branches_voltage(network, crit_nodes):
    """ Reinforce MV or LV grid by installing a new branch/line type

    Parameters
    ----------
    network : edisgo network object
    crit_nodes : List of nodes objects with critical voltages sorted by voltage
                 (descending)

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

    # find first nodes of every main line as representatives
    rep_main_line = nx.predecessor(network.grid._graph, network.grid.station,
                                   cutoff=1)
    # list containing all representatives of main lines that have already been
    # reinforced
    main_line_reinforced = []

    for crit_node in crit_nodes:
        path = nx.shortest_path(network.grid._graph, network.grid.station,
                                crit_node)

        # check if representative of line is already in list
        # main_line_reinforced, if it is the main line the critical node is
        # connected to has already been reinforced in this iteration step
        if not path[1] in main_line_reinforced:

            main_line_reinforced.append(path[1])
            # get path length from station to critical node
            get_weight = lambda u, v, data: data['line']._length
            path_length = dijkstra_shortest_path_length(
                network.grid._graph, network.grid.station, get_weight,
                target=crit_node)
            # find first node in path that exceeds 2/3 of the line length
            # from station to critical node farthest away from the station
            node_2_3 = next(i for i in path if
                              path_length[i] >= path_length[crit_node] * 2 / 3)

            # if node_2_3 is a representative (meaning it is already directly
            # connected to the station), line cannot be disconnected and must
            # therefore be reinforced
            if node_2_3 in rep_main_line:
                crit_line = network.grid._graph.get_edge_data(
                    network.grid.station, node_2_3)['line']

                # if critical line is already a standard line install one more
                # parallel line
                if crit_line._type.name == 'NAYY 4x150':
                    crit_line._quantity = crit_line._quantity + 1

                # if critical line is not yet a standard line check if one or
                # several standard lines are needed
                else:
                    # number of parallel standard lines could be calculated
                    # following [2] p.103, for now number of parallel standard
                    # lines is iterated
                    #ToDo: LV or MV
                    crit_line._type = standard_line_lv.copy()
                    crit_line._quantity = 1

            # if node_2_3 is not a representative, disconnect line
            else:
                # get line between node_2_3 and predecessor node (that is
                # closer to the station)
                pred_node = path[path.index(node_2_3) - 1]
                crit_line = network.grid._graph.get_edge_data(
                    node_2_3, pred_node)['line']
                # add new edge between node_2_3 and station
                new_line_data = {'line': crit_line,
                                 'type': 'line'}
                network.grid._graph.add_edge(network.grid.station, node_2_3,
                                             new_line_data)
                # remove old edge
                network.grid._graph.remove_edge(pred_node, node_2_3)
                # change line length and type
                #ToDo: MV or LV
                crit_line._length = path_length[node_2_3]
                crit_line._type = standard_line_lv.copy()

        else:
            logger.debug('==> Main line of node {} '.format(str(crit_node)) +
                         'in LV grid {} '.format(str(crit_node._grid)) +
                         'has already been reinforced.')

    if main_line_reinforced:
        logger.info('==> {} branches were reinforced.'.format(
            str(len(main_line_reinforced))))


def reinforce_branches_current(network, crit_lines):
    """ Reinforce MV or LV grid by installing a new branch/line type
    
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
        #ToDo: MV or LV
        if crit_line._type.name == standard_line_lv.name:
            # check how many parallel standard lines are needed
            number_parallel_lines = math.ceil(crit_line._type['I_max_th'] *
                                              rel_overload /
                                              standard_line_lv['I_max_th'])
            crit_line._quantity = number_parallel_lines
        else:
            # check if parallel line of the same kind is sufficient
            if (crit_line._type['I_max_th'] * rel_overload <=
                        crit_line._type['I_max_th'] * 2):
                crit_line._quantity = 2
            else:
                number_parallel_lines = math.ceil(crit_line._type['I_max_th'] *
                                                  rel_overload /
                                                  standard_line_lv['I_max_th'])
                crit_line._type = standard_line_lv.copy()
                crit_line._quantity = number_parallel_lines

    if crit_lines:
        logger.info('==> {} branches were reinforced.'.format(
            str(len(crit_lines))))