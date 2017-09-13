import copy
import math
import pandas as pd
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra as \
    dijkstra_shortest_path_length
if not 'READTHEDOCS' in os.environ:
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


def extend_distribution_substation(critical_stations):
    """
    Reinforce MV/LV substations.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
        critical_stations : list
            List of stations with overloading

    """

    # ToDo: get parameters from config
    # ToDo: Einheiten klären
    # trafo_params = grid.network._static_data['{grid_level}_trafos'.format(
    #     grid_level=grid_level)]
    # get parameters for standard transformer
    standard_transformer = pd.Series({'r': 0.01, 's': 630, 'x': 0.04})
    lf_lv_trans_normal = 1

    for station in critical_stations:

        # list of maximum power of each transformer in the station
        s_max_per_trafo = [_._type.s for _ in station._transformers]

        # maximum power in generation case
        # ToDo: critical_stations[station] anpassen wenn Datenstruktur geändert
        # werden sollte
        s_max_gc = critical_stations[station]

        # determine missing trafo power to solve overloading issue
        s_trafo_missing = s_max_gc - (
            sum(s_max_per_trafo) * lf_lv_trans_normal)

        # check if second transformer of the same kind is sufficient
        # if true install second transformer, otherwise install as many
        # standard transformers as needed
        if max(s_max_per_trafo) >= s_trafo_missing:
            # if station has more than one transformer install a new
            # transformer of the same kind as the transformer that best
            # meets the missing power demand
            duplicated_transformer = min(
                [_ for _ in station.transformers
                 if _._type.s > s_trafo_missing],
                key=lambda i: i._type.s - s_trafo_missing)

            # ToDo: id?
            new_transformer = Transformer(
                id=99,
                geom=duplicated_transformer.geom,
                grid=duplicated_transformer.grid,
                voltage_op=duplicated_transformer._voltage_op,
                type=copy.deepcopy(duplicated_transformer._type))

            # add transformer to station
            # ToDo: Methode in Station hierfür einführen?
            station._transformers.append(new_transformer)

        else:
            # get any transformer to get attributes for new transformer from
            station_transformer = station.transformers[0]

            # ToDo: id?
            new_transformer = Transformer(
                id=99,
                geom=station_transformer.geom,
                grid=station_transformer.grid,
                voltage_op=station_transformer._voltage_op,
                type=copy.deepcopy(standard_transformer))

            # calculate how many parallel standard transformers are needed
            number_transformers = math.ceil(s_max_gc / standard_transformer.s)

            # add transformer to station
            # ToDo: Methode in Station hierfür einführen?
            new_transformers = []
            for i in range(number_transformers):
                new_transformers.append(new_transformer)
            station._transformers = new_transformers

    logger.info("{stations_cnt} have been reinforced due to overloading "
                "issues.".format(stations_cnt=len(critical_stations)))


def reinforce_branches_voltage(grid, crit_nodes):
    """ Reinforce MV or LV grid by installing a new branch/line type

    Parameters
    ----------
    grid: Grid object
    crit_nodes: List of nodes objects with critical voltages sorted by voltage
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

    # load cable data, file_names and parameter
    # branch_parameters = grid.network.static_data['{gridlevel}_cables'.format(
    #     gridlevel=grid_level)]
    # branch_parameters = branch_parameters[
    #     branch_parameters['U_n'] == grid.v_level].sort_values('I_max_th')
    # ToDo: get parameters from config
    standard_line = pd.Series({
        'U_n': 400, 'I_max_th': 270, 'R': 0.1, 'L': 0.28, 'C': None},
         name='NAYY 4x150')

    # find first nodes of every main line as representatives
    rep_main_line = nx.predecessor(grid._graph, grid.station, cutoff=1)
    # list containing all representatives of main lines that have already been
    # reinforced
    main_line_reinforced = []

    for crit_node in crit_nodes:
        path = nx.shortest_path(grid._graph, grid.station, crit_node)

        # check if representative of line is already in list
        # main_line_reinforced, if it is the main line the critical node is
        # connected to has already been reinforced in this iteration step
        if not path[1] in main_line_reinforced:

            main_line_reinforced.append(path[1])
            # get path length from station to critical node
            get_weight = lambda u, v, data: data['line']._length
            path_length = dijkstra_shortest_path_length(
                grid._graph, grid.station, get_weight, target=crit_node)
            # find first node in path that exceeds 2/3 of the line length
            # from station to critical node farthest away from the station
            node_2_3 = next(i for i in path if
                              path_length[i] >= path_length[crit_node] * 2 / 3)

            # if node_2_3 is a representative (meaning it is already directly
            # connected to the station), line cannot be disconnected and must
            # therefore be reinforced
            if node_2_3 in rep_main_line:
                crit_line = grid._graph.get_edge_data(
                    grid.station, node_2_3)['line']

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
                    crit_line._type = standard_line.copy()
                    crit_line._quantity = 1

            # if node_2_3 is not a representative, disconnect line
            else:
                # get line between node_2_3 and predecessor node (that is
                # closer to the station)
                pred_node = path[path.index(node_2_3) - 1]
                crit_line = grid._graph.get_edge_data(
                    node_2_3, pred_node)['line']
                # add new edge between node_2_3 and station
                new_line_data = {'line': crit_line,
                                 'type': 'line'}
                grid._graph.add_edge(grid.station, node_2_3, new_line_data)
                # remove old edge
                grid._graph.remove_edge(pred_node, node_2_3)
                # change line length and type
                crit_line._length = path_length[node_2_3]
                crit_line._type = standard_line.copy()

        else:
            logger.debug('==> Main line of node {} '.format(str(crit_node)) +
                         'in LV grid {} '.format(str(crit_node._grid)) +
                         'has already been reinforced.')

    if main_line_reinforced:
        logger.info('==> {} branches were reinforced.'.format(
            str(len(main_line_reinforced))))


def reinforce_branches_current(crit_lines):
    """ Reinforce MV or LV grid by installing a new branch/line type
    
    Parameters
    ----------
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
    # load cable data, file_names and parameter
    # branch_parameters = grid.network.static_data['MV_cables']
    # branch_parameters = branch_parameters[branch_parameters['U_n'] ==
    #                                       grid.v_level].sort_values('I_max_th')
    # ToDo: get parameters from config
    standard_line = pd.Series({
        'U_n': 400, 'I_max_th': 270, 'R': 0.1, 'L': 0.28, 'C': None},
         name='NAYY 4x150')

    for crit_line, rel_overload in crit_lines.items():
        if crit_line._type.name == standard_line.name:
            # check how many parallel standard lines are needed
            number_parallel_lines = math.ceil(crit_line._type['I_max_th'] *
                                              rel_overload /
                                              standard_line['I_max_th'])
            crit_line._quantity = number_parallel_lines
        else:
            # check if parallel line of the same kind is sufficient
            if (crit_line._type['I_max_th'] * rel_overload <=
                        crit_line._type['I_max_th'] * 2):
                crit_line._quantity = 2
            else:
                number_parallel_lines = math.ceil(crit_line._type['I_max_th'] *
                                                  rel_overload /
                                                  standard_line['I_max_th'])
                crit_line._type = standard_line.copy()
                crit_line._quantity = number_parallel_lines

    if crit_lines:
        logger.info('==> {} branches were reinforced.'.format(
            str(len(crit_lines))))