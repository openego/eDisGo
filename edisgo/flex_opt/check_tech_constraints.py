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
    network : :class:`~.grid.network.Network`

    Returns
    -------
    Dictionary with critical :class:`~.grid.components.LVStation`
    Format: {lv_station_1: overloading_1, ..., lv_station_n: overloading_n}

    Notes
    -----
    According to [1]_ load factors in feed-in case of all equipment in MV and
    LV is set to 1.

    HV/MV transformers are not checked.

    References
    ----------
    .. [1] Verteilnetzstudie für das Land Baden-Württemberg

    """

    crit_stations = {}

    load_factor_mv_lv_transformer = float(network.config['grid_expansion'][
        'load_factor_mv_lv_transformer'])

    for lv_grid in network.mv_grid.lv_grids:
        station = lv_grid.station
        # maximum allowed apparent power of station
        # ToDo: change s to S
        s_station_max = sum([_.type.s for _ in station.transformers]) * \
                        load_factor_mv_lv_transformer
        try:
            # check if maximum allowed apparent power of station exceeds
            # apparent power from power flow analysis
            s_station_pfa = max(network.results.s_res(
                station.transformers).sum(axis=1))
            if s_station_max < s_station_pfa:
                crit_stations[station] = s_station_pfa
        except:
            logger.debug('No results for LV station {} '.format(str(station)) +
                         'to check overloading.')

    if crit_stations:
        logger.info('==> {} LV stations have load issues.'.format(
            len(crit_stations)))

    return crit_stations


def check_voltage_mv(network):
    """
    Checks for voltage stability issues in MV grid.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    :pandas:`pandas.Series<series>`
        Critical nodes with corresponding maximum voltage deviation.
        Format: pd.Series(data=[v_mag_pu_node_A, v_mag_pu_node_B, ...],
                          index=[node_A, node_B, ...])

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.

    """

    # load max. voltage deviation
    mv_max_v_deviation = float(
        network.config['grid_expansion']['mv_max_v_deviation'])

    v_mag_pu_pfa = network.results.v_res(nodes=network.mv_grid.graph.nodes(),
                                         level='mv')
    # check for overvoltage
    v_max = v_mag_pu_pfa.max()
    crit_nodes_max = v_max[(v_max > (1 + mv_max_v_deviation))] - 1
    # check for undervoltage
    v_min = v_mag_pu_pfa.min()
    crit_nodes_min = 1 - v_min[(v_min < (1 - mv_max_v_deviation))]
    # combine critical nodes and keep highest voltage deviation at each
    # node
    crit_nodes = crit_nodes_max.append(crit_nodes_min).max(level=0)
    if len(crit_nodes) > 0:
        crit_nodes.sort_values(ascending=False, inplace=True)
        logger.info(
            '==> {} nodes in MV grid have voltage issues.'.format(
                len(crit_nodes)))
    else:
        crit_nodes = None

    return crit_nodes


def check_voltage_lv(network):
    """
    Checks for voltage stability issues in LV grids.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    Dict of LV grids with critical nodes as :pandas:`pandas.Series<series>`,
    sorted descending by voltage deviation.
    Format: {grid_1: pd.Series(data=[v_mag_pu_node_1A, v_mag_pu_node_1B],
                               index=[node_1A, node_1B]), ...}

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.

    """

    crit_nodes = {}

    # load max. voltage deviation
    lv_max_v_deviation = float(
        network.config['grid_expansion']['lv_max_v_deviation'])

    for lv_grid in network.mv_grid.lv_grids:
        v_mag_pu_pfa = network.results.v_res(nodes=lv_grid.graph.nodes(),
                                             level='lv')
        # check for overvoltage
        v_max = v_mag_pu_pfa.max()
        crit_nodes_max = v_max[(v_max > (1 + lv_max_v_deviation))] - 1
        # check for undervoltage
        v_min = v_mag_pu_pfa.min()
        crit_nodes_min = 1 - v_min[(v_min < (1 - lv_max_v_deviation))]
        # combine critical nodes and keep highest voltage deviation at each
        # node
        crit_nodes_grid = crit_nodes_max.append(crit_nodes_min).max(level=0)
        if len(crit_nodes_grid) > 0:
            crit_nodes[lv_grid] = crit_nodes_grid.sort_values(
                ascending=False)

    if crit_nodes:
        logger.info(
            '==> {} nodes in LV grids have voltage issues.'.format(
                len(crit_nodes)))

    return crit_nodes
