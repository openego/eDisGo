import logging

logger = logging.getLogger('edisgo')


def mv_line_load(network):
    """
    Checks for over-loading issues in MV grid.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    Dictionary of critical :class:`~.grid.components.Line` with max. relative
    overloading
    Format: {line_1: rel_overloading_1, ..., line_n: rel_overloading_n}

    Notes
    -----
    According to [1]_ load factors in feed-in case of all cables and lines in
    MV grids are set to 1.

    References
    ----------
    .. [1] Verteilnetzstudie für das Land Baden-Württemberg

    """

    crit_lines = {}

    load_factor_mv_line = network.scenario.parameters.load_factor_mv_line

    for line in list(network.mv_grid.graph.lines()):
        i_line_max = line['line'].type['I_max_th'] * \
                     load_factor_mv_line * line['line'].quantity
        try:
            # check if maximum current from power flow analysis exceeds
            # allowed maximum current
            i_line_pfa = max(network.results.i_res[repr(line['line'])])
            if i_line_pfa > i_line_max:
                crit_lines[line['line']] = i_line_pfa / i_line_max
        except KeyError:
            logger.debug('No results for line {} '.format(str(line)) +
                         'to check overloading.')

    if crit_lines:
        logger.debug('==> {} line(s) in MV grid has/have load issues.'.format(
            len(crit_lines)))
    else:
        logger.debug('==> No line load issues in MV grid.')

    return crit_lines


def lv_line_load(network):
    """
    Checks for over-loading issues in LV grids.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    Dictionary of critical :class:`~.grid.components.Line` with max. relative
    overloading
    Format: {line_1: rel_overloading_1, ..., line_n: rel_overloading_n}

    Notes
    -----
    According to [1]_ load factors in feed-in case of all cables and lines in
    LV grids are set to 1.

    References
    ----------
    .. [1] Verteilnetzstudie für das Land Baden-Württemberg

    """

    crit_lines = {}

    load_factor_lv_line = network.scenario.parameters.load_factor_lv_line

    for lv_grid in network.mv_grid.lv_grids:
        for line in list(lv_grid.graph.lines()):
            i_line_max = line['line'].type['I_max_th'] * \
                         load_factor_lv_line * line['line'].quantity
            try:
                # check if maximum current from power flow analysis exceeds
                # allowed maximum current
                i_line_pfa = max(network.results.i_res[repr(line['line'])])
                if i_line_pfa > i_line_max:
                    crit_lines[line['line']] = i_line_pfa / i_line_max
            except KeyError:
                logger.debug('No results for line {} '.format(str(line)) +
                             'to check overloading.')

    if crit_lines:
        logger.debug('==> {} line(s) in LV grids has/have load issues.'.format(
            len(crit_lines)))
    else:
        logger.debug('==> No line load issues in LV grids.')

    return crit_lines


def mv_lv_station_load(network):
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

    load_factor_mv_lv_transformer = \
        network.scenario.parameters.load_factor_mv_lv_transformer

    for lv_grid in network.mv_grid.lv_grids:
        station = lv_grid.station
        # maximum allowed apparent power of station
        s_station_max = (sum([_.type.S_nom for _ in station.transformers]) *
                         load_factor_mv_lv_transformer)
        try:
            # check if maximum allowed apparent power of station exceeds
            # apparent power from power flow analysis
            s_station_pfa = max(network.results.s_res(
                station.transformers).sum(axis=1))
            if s_station_max < s_station_pfa:
                crit_stations[station] = s_station_pfa
        except KeyError:
            logger.debug('No results for LV station {} '.format(str(station)) +
                         'to check overloading.')

    if crit_stations:
        logger.debug('==> {} MV/LV station(s) has/have load issues.'.format(
            len(crit_stations)))
    else:
        logger.debug('==> No MV/LV station load issues.')

    return crit_stations


def mv_voltage_deviation(network):
    """
    Checks for voltage stability issues in MV grid.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    Dict with :class:`~.grid.grids.MVGrid` with critical nodes as
    :pandas:`pandas.Series<series>`, sorted descending by voltage deviation.
    Format: {MV_grid: pd.Series(data=[v_mag_pu_node_A, v_mag_pu_node_B, ...],
                                index=[node_A, node_B, ...])}

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.

    """

    crit_nodes = {}

    # load max. voltage deviation
    mv_max_v_deviation = network.scenario.parameters.mv_max_v_deviation

    v_mag_pu_pfa = network.results.v_res(nodes=network.mv_grid.graph.nodes(),
                                         level='mv')
    # check for over-voltage
    v_max = v_mag_pu_pfa.max()
    crit_nodes_max = v_max[(v_max > (1 + mv_max_v_deviation))] - 1
    # check for under-voltage
    v_min = v_mag_pu_pfa.min()
    crit_nodes_min = 1 - v_min[(v_min < (1 - mv_max_v_deviation))]
    # combine critical nodes and keep highest voltage deviation at each
    # node
    crit_nodes_grid = crit_nodes_max.append(crit_nodes_min).max(level=0)
    if len(crit_nodes_grid) > 0:
        crit_nodes[network.mv_grid] = crit_nodes_grid.sort_values(
            ascending=False)
        logger.debug(
            '==> {} node(s) in MV grid has/have voltage issues.'.format(
                len(crit_nodes[network.mv_grid])))
    else:
        crit_nodes = None
        logger.debug('==> No voltage issues in MV grid.')

    return crit_nodes


def lv_voltage_deviation(network):
    """
    Checks for voltage stability issues in LV grids.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    Dict of :class:`~.grid.grids.LVGrid` with critical nodes as
    :pandas:`pandas.Series<series>`, sorted descending by voltage deviation.
    Format: {grid_1: pd.Series(data=[v_mag_pu_node_1A, v_mag_pu_node_1B],
                               index=[node_1A, node_1B]), ...}

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.

    """

    crit_nodes = {}

    # load max. voltage deviation
    lv_max_v_deviation = network.scenario.parameters.lv_max_v_deviation

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
        logger.debug(
            '==> {} LV grid(s) has/have voltage issues.'.format(
                len(crit_nodes)))
    else:
        logger.debug('==> No voltage issues in LV grids.')

    return crit_nodes
