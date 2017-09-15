import numpy as np

import logging


logger = logging.getLogger('edisgo')


def check_line_load(network, results_lines):
    """ Checks for over-loading of branches and transformers for MV or LV grid

    Parameters
    ----------
    network: edisgo Network object
    results_lines: pandas.DataFrame
        power flow analysis results (pfa_edges) from edisgo Results object

    Returns
    -------
    Dict of critical lines (edisgo Line objects) with max. relative overloading
    Format: {line_1: rel_overloading_1, ..., line_n: rel_overloading_n}

    Notes
    -----
    According to [1]_ load factors in feed-in case of all equipment in MV and
    LV is set to 1.

    References
    ----------
    .. [1] dena VNS

    """

    crit_lines = {}

    load_factor_mv_line = network.config['grid_expansion'][
        'load_factor_mv_line']
    load_factor_lv_line = network.config['grid_expansion'][
        'load_factor_lv_line']

    # ToDo: Einheiten klären (results und Werte in type)
    # ToDo: generischer, sodass nicht unbedingt LV und MV gerechnet werden? + HV/MV transformer immer raus lassen?
    # ToDo: Zugriff auf Attribute mit _

    mw2kw = 1e3

    # ToDo: p0 und p1 Ergebnisse am Anfang und Ende einer Line?
    # ToDo: Wo soll S berechnet werden? In results objekt schon s_max berechnen?
    # ToDo: mit Skalar testen
    results_lines['s0'] = (results_lines['p0'].apply(lambda x: x ** 2) +
                           results_lines['q0'].apply(lambda x: x ** 2)).apply(
                               lambda x: np.sqrt(x))
    # results_lines['s0'] = results_lines['p0'].apply(
    #     lambda x: [i ** 2 for i in x])

    # MV
    for line in list(network.mv_grid.graph.graph_edges()):
        s_max_th = (3 ** 0.5 * line['line']._type['U_n'] *
                    line['line']._type['I_max_th']) * \
                   load_factor_mv_line * line['line']._quantity
        try:
            # check if maximum s_0 from power flow analysis exceeds allowed
            # values
            if max(results_lines.loc[line, 's0']) > s_max_th:
                crit_lines[line] = (max(results_lines.loc[line, 's0']) *
                                    mw2kw / s_max_th)
        except:
            logger.debug('No results for line {} '.format(str(line)) +
                         'to check overloading.')
    # LV
    for lv_grid in network.mv_grid.lv_grids:
        for line in list(lv_grid.graph.graph_edges()):
            line['line']._quantity = 1
            s_max_th = (3 ** 0.5 * line['line']._type['U_n'] *
                        line['line']._type['I_max_th']) * \
                       load_factor_lv_line * line['line']._quantity
            try:
                # check if maximum s_0 from power flow analysis exceeds allowed
                # values
                if max(results_lines.loc[line, 's0']) > s_max_th:
                    crit_lines[line] = (max(results_lines.loc[line, 's0']) *
                                        mw2kw / s_max_th)
            except:
                logger.debug('No results for line {} '.format(str(line)) +
                             'to check overloading.')

    if crit_lines:
        logger.info('==> {} lines have load issues.'.format(
            len(crit_lines)))

    return crit_lines


def check_station_load(network):
    """
    Checks for over-loading of MV/LV transformers.

    Parameters
    ----------
    network: edisgo Network object

    Returns
    -------
    Dict of critical stations (edisgo Station objects)
    Format: {station_1: overloading_1, ..., station_n: overloading_n}

    Notes
    -----
    According to [1]_ load factors in feed-in case of all equipment in MV and
    LV is set to 1.

    HV/MV transformers are not checked.

    References
    ----------
    .. [1] dena VNS

    """

    # ToDo: lv stations not yet in pfa_p and pfa_q
    crit_stations = {}

    load_factor_mv_lv_transformer = network.config['grid_expansion'][
        'load_factor_mv_lv_transformer']

    mw2kw = 1e3

    print(network.results.s_res(None))
        #network.mv_grid.graph.nodes_by_attribute('lv_station')))

    # MV/LV station
    for lv_grid in network.mv_grid.lv_grids:
        station = lv_grid.station
        s_max_th = sum([_._type.s for _ in station.transformers])
        # find line station is connected to
        # ToDo: Was ist load der station? Summe der loads der lines die zur Station führen?
        # edges_station = nx.edges(lv_grid.graph, station)
        # max(results_lines.loc[line, 's0'])
        # try:
        #     # check if maximum s_0 from power flow analysis exceeds allowed
        #     # values
        #     if max(results_lines.loc[line, 's0']) > s_max_th:
        #         crit_lines[line] = (max(results_lines.loc[line, 's0']) *
        #                             mw2kw / s_max_th)
        # except:
        #     logger.debug('No results for line {} '.format(str(line)) +
        #                 'to check overloading.')

    if crit_stations:
        logger.info('==> {} stations have load issues.'.format(
            len(crit_stations)))

    return crit_stations


def check_voltage_mv(network, results_v_mag_pu):
    """
    Checks for voltage stability issues at all nodes of MV grid.

    Parameters
    ----------
    network: edisgo Network object
    results_nodes: pandas.DataFrame
        power flow analysis results (pfa_nodes) from edisgo Results object

    Returns
    -------
    Dict of critical nodes, sorted descending by voltage difference
    Format: {grid_1: [node_1A, node_1B, ...], ..., grid_n: [node_nA, ...]}

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.
    """

    # ToDo: delta_U wird auch benötigt, deshalb crit_nodes als dict mit series
    # ToDo: crit_nodes innerhalb einer Series sortieren
    crit_nodes = {}

    # load max. voltage deviation for load and feedin case
    mv_max_v_deviation = network.config['grid_expansion']['mv_max_v_deviation']
    lv_max_v_deviation = network.config['grid_expansion']['lv_max_v_deviation']

    # check nodes' voltages
    # MV
    voltage_station = network.mv_grid.station.transformers[0]._voltage_op
    for node in network.mv_grid.graph.nodes():
        try:
            # check if maximum deviation in v_mag_pu exceeds allowed deviation
            if (max(results_nodes.loc[node, 'v_mag_pu']) >
                    1 + mv_max_v_deviation or
                min(results_nodes.loc[
                    node, 'v_mag_pu']) < 1 - mv_max_v_deviation):
                try:
                    crit_nodes[network.mv_grid].append(node)
                except:
                    crit_nodes[network.mv_grid] = [node]
        except:
            logger.debug('No results for node {} '.format(str(node)) +
                         'to check overvoltage.')

    if crit_nodes:
        logger.info(
            '==> {} nodes have voltage issues.'.format(len(crit_nodes)))

    return crit_nodes


def check_voltage_lv(network, results_nodes):
    """
    Checks for voltage stability issues at all nodes of LV grids.

    Parameters
    ----------
    network: edisgo Network object
    results_nodes: pandas.DataFrame
        power flow analysis results (pfa_nodes) from edisgo Results object

    Returns
    -------
    Dict of grids and their critical nodes as index of pd.Series, with voltage
    deviation, sorted descending by voltage difference
    Format: {grid_1: pd.Series(data=[v_mag_pu_node_1A, v_mag_pu_node_1B],
                               index=[node_1A, node_1B]), ...}

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.
    """

    # ToDo: delta_U wird auch benötigt, deshalb crit_nodes als dict mit series
    # ToDo: crit_nodes innerhalb einer Series sortieren
    crit_nodes = {}

    # load max. voltage deviation for load and feedin case
    mv_max_v_deviation = network.config['grid_expansion']['mv_max_v_deviation']
    lv_max_v_deviation = network.config['grid_expansion']['lv_max_v_deviation']

    # check nodes' voltages
    # MV
    voltage_station = network.mv_grid.station.transformers[0]._voltage_op
    for node in network.mv_grid.graph.nodes():
        try:
            # check if maximum deviation in v_mag_pu exceeds allowed deviation
            if (max(results_nodes.loc[node, 'v_mag_pu']) >
                    1 + mv_max_v_deviation or
                min(results_nodes.loc[
                    node, 'v_mag_pu']) < 1 - mv_max_v_deviation):
                try:
                    crit_nodes[network.mv_grid].append(node)
                except:
                    crit_nodes[network.mv_grid] = [node]
        except:
            logger.debug('No results for node {} '.format(str(node)) +
                         'to check overvoltage.')

    # LV
    for lv_grid in network.mv_grid.lv_grids:
        for node in lv_grid.graph.nodes():
            try:
                # check if maximum deviation in v_mag_pu exceeds allowed
                # deviation
                if (max(results_nodes.loc[node, 'v_mag_pu']) >
                        1 + lv_max_v_deviation or
                    min(results_nodes.loc[
                        node, 'v_mag_pu']) < 1 - lv_max_v_deviation):
                    try:
                        crit_nodes[lv_grid].append(node)
                    except:
                        crit_nodes[lv_grid] = [node]
            except:
                logger.debug('No results for node {} '.format(str(node)) +
                             'in LV grid {} '.format(str(lv_grid)) +
                             'to check overvoltage.')

    if crit_nodes:
        logger.info(
            '==> {} nodes have voltage issues.'.format(len(crit_nodes)))

    return crit_nodes
