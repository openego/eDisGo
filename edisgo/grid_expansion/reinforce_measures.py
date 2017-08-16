import copy
import math
import pandas as pd
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra as \
    dijkstra_shortest_path_length

import ding0
from edisgo.grid.components import Transformer
from edisgo import grid_expansion
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
                    # ToDo: increment number of parallel lines
                    pass

                # if critical line is not yet a standard line check if one or
                # several standard lines are needed
                else:
                    # check if new standard line might solve the voltage
                    # problem
                    # ToDo: welche Kenngröße verwenden?
                    if crit_line._type['I_max_th'] < standard_line['I_max_th']:
                        crit_line._type = standard_line.copy()
                    else:
                        # ToDo: wie viele Standardbetriebsmittel?
                        crit_line._type = standard_line.copy()

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


def reinforce_branches_current(grid, crit_branches):
    """ Reinforce MV or LV grid by installing a new branch/line type
    
    Parameters
    ----------
        grid : GridDing0
            Grid identifier.
        crit_branches : dict
            Dict of critical branches with max. relative overloading.

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
        
    Notes
    -----
        The branch type to be installed is determined per branch using the rel. overloading. According to [2]_ 
        only cables are installed.
        
    See Also
    --------
    ding0.flexopt.check_tech_constraints.check_load :
    ding0.flexopt.reinforce_measures.reinforce_branches_voltage :
    """
    # # load cable data, file_names and parameter
    # branch_parameters = grid.network.static_data['MV_cables']
    # branch_parameters = branch_parameters[branch_parameters['U_n'] == grid.v_level].sort_values('I_max_th')
    #
    # branch_ctr = 0
    #
    # for branch, rel_overload in crit_branches.items():
    #     try:
    #         type = branch_parameters.ix[branch_parameters[branch_parameters['I_max_th'] >=
    #                                     branch['branch'].type['I_max_th'] * rel_overload]['I_max_th'].idxmin()]
    #         branch['branch'].type = type
    #         branch_ctr += 1
    #     except:
    #         logger.warning('Branch {} could not be reinforced (current '
    #                        'issues) as there is no appropriate cable type '
    #                        'available. Original type is retained.'.format(
    #             branch))
    #         pass
    #
    # if branch_ctr:
    #     logger.info('==> {} branches were reinforced.'.format(str(branch_ctr)))


def reinforce_lv_branches_overloading(grid, crit_branches):
    """
    Choose appropriate cable type for branches with line overloading

    Parameters
    ----------
    grid : ding0.core.network.grids.LVGridDing0
        Ding0 LV grid object
    crit_branches : list
        List of critical branches incl. its line loading

    Notes
    -----
    If maximum size cable is not capable to resolve issue due to line
    overloading largest available cable type is assigned to branch.

    Returns
    -------

        unsolved_branches : :obj:`list`
            List of braches no suitable cable could be found
    """
    # unsolved_branches = []
    #
    # cable_lf = cfg_ding0.get('assumptions',
    #                          'load_factor_lv_cable_lc_normal')
    #
    # cables = grid.network.static_data['LV_cables']
    #
    # # resolve overloading issues for each branch segment
    # for branch in crit_branches:
    #     I_max_branch_load = branch['s_max'][0]
    #     I_max_branch_gen = branch['s_max'][1]
    #     I_max_branch = max([I_max_branch_load, I_max_branch_gen])
    #
    #     suitable_cables = cables[(cables['I_max_th'] * cable_lf)
    #                       > I_max_branch]
    #
    #     if not suitable_cables.empty:
    #         cable_type = suitable_cables.ix[suitable_cables['I_max_th'].idxmin()]
    #         branch['branch'].type = cable_type
    #         crit_branches.remove(branch)
    #     else:
    #         cable_type_max = cables.ix[cables['I_max_th'].idxmax()]
    #         unsolved_branches.append(branch)
    #         branch['branch'].type = cable_type_max
    #         logger.error("No suitable cable type could be found for {branch} "
    #                       "with I_th_max = {current}. "
    #                       "Cable of type {cable} is chosen during "
    #                       "reinforcement.".format(
    #             branch=branch['branch'],
    #             cable=cable_type_max.name,
    #             current=I_max_branch
    #         ))
    #
    # return unsolved_branches