import pandas as pd
import logging

from edisgo.grid.grids import LVGrid

logger = logging.getLogger('edisgo')


def mv_line_load(network):
    """
    Checks for over-loading issues in MV grid.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded MV lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.grid.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Notes
    -----
    Line over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_lines = pd.DataFrame()
    crit_lines = _line_load(network, network.mv_grid, crit_lines)

    if not crit_lines.empty:
        logger.debug('==> {} line(s) in MV grid has/have load issues.'.format(
            crit_lines.shape[0]))
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
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded LV lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.grid.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Notes
    -----
    Line over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_lines = pd.DataFrame()

    for lv_grid in network.mv_grid.lv_grids:
        crit_lines = _line_load(network, lv_grid, crit_lines)

    if not crit_lines.empty:
        logger.debug('==> {} line(s) in LV grids has/have load issues.'.format(
            crit_lines.shape[0]))
    else:
        logger.debug('==> No line load issues in LV grids.')

    return crit_lines


def _line_load(network, grid, crit_lines):
    """
    Checks for over-loading issues in MV grid.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    grid : :class:`~.grid.grids.LVGrid` or :class:`~.grid.grids.MVGrid`
    crit_lines : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.grid.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.grid.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    """
    if isinstance(grid, LVGrid):
        grid_level = 'lv'
    else:
        grid_level = 'mv'

    for line in list(grid.graph.lines()):
        i_line_allowed_per_case = {}
        i_line_allowed_per_case['feedin_case'] = \
            line['line'].type['I_max_th'] * line['line'].quantity * \
            network.config['grid_expansion_load_factors'][
                '{}_feedin_case_line'.format(grid_level)]
        i_line_allowed_per_case['load_case'] = \
            line['line'].type['I_max_th'] * line['line'].quantity * \
            network.config['grid_expansion_load_factors'][
                '{}_load_case_line'.format(grid_level)]
        # maximum allowed apparent power of station in each time step
        i_line_allowed = \
            network.timeseries.timesteps_load_feedin_case.case.apply(
                lambda _: i_line_allowed_per_case[_])
        try:
            # check if maximum current from power flow analysis exceeds
            # allowed maximum current
            i_line_pfa = network.results.i_res[repr(line['line'])]
            if any((i_line_allowed - i_line_pfa) < 0):
                # find out largest relative deviation
                relative_i_res = i_line_pfa / i_line_allowed
                crit_lines = crit_lines.append(pd.DataFrame(
                    {'max_rel_overload': relative_i_res.max(),
                     'time_index': relative_i_res.idxmax()},
                    index=[line['line']]))
        except KeyError:
            logger.debug('No results for line {} '.format(str(line)) +
                         'to check overloading.')

    return crit_lines


def hv_mv_station_load(network):
    """
    Checks for over-loading of HV/MV station.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    Dictionary with critical :class:`~.grid.components.MVStation` and maximum
    apparent power from power flow analysis. Format: {mv_station: S_max}

    """

    crit_stations = {}

    #ToDo: differentiate between load and feed-in case!
    load_factor = network.config['grid_expansion_load_factors'][
        'mv_feedin_case_transformer']

    # maximum allowed apparent power of station
    s_station_max = (sum(
        [_.type.S_nom for _ in network.mv_grid.station.transformers]) *
                     load_factor)
    try:
        # check if maximum allowed apparent power of station exceeds
        # apparent power from power flow analysis
        s_station_pfa = network.results.s_res(
            [network.mv_grid.station]).sum(axis=1).max()
        if s_station_max < s_station_pfa:
            crit_stations[network.mv_grid.station] = s_station_pfa
    except KeyError:
        logger.debug('No results for MV station to check overloading.')

    if crit_stations:
        logger.debug('==> HV/MV station has load issues.')
    else:
        logger.debug('==> No HV/MV station load issues.')

    return crit_stations


def mv_lv_station_load(network):
    """
    Checks for over-loading of MV/LV stations.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    -------
    Dictionary with critical :class:`~.grid.components.LVStation` and maximum
    apparent power from power flow analysis.
    Format: {lv_station_1: S_max_1, ..., lv_station_n: S_max_n}

    Notes
    -----
    According to [VerteilnetzstudieBW]_ load factors in feed-in case of all equipment in MV and
    LV is set to 1.

    HV/MV transformers are not checked.

    """

    crit_stations = {}

    # ToDo: differentiate between load and feed-in case!
    load_factor = network.config['grid_expansion_load_factors'][
        'lv_feedin_case_transformer']

    for lv_grid in network.mv_grid.lv_grids:
        station = lv_grid.station
        # maximum allowed apparent power of station
        s_station_max = (sum([_.type.S_nom for _ in station.transformers]) *
                         load_factor)
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
    #ToDo: for now only voltage deviation for the combined calculation of MV
    # and LV is considered (load and feed-in case for seperate consideration
    # of MV and LV needs to be implemented)
    max_v_dev = network.config['grid_expansion_allowed_voltage_deviations'][
        'mv_lv_max_v_deviation']

    v_mag_pu_pfa = network.results.v_res(nodes=network.mv_grid.graph.nodes(),
                                         level='mv')
    # check for over-voltage
    v_max = v_mag_pu_pfa.max()
    crit_nodes_max = v_max[(v_max > (1 + max_v_dev))] - 1
    # check for under-voltage
    v_min = v_mag_pu_pfa.min()
    crit_nodes_min = 1 - v_min[(v_min < (1 - max_v_dev))]
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


def lv_voltage_deviation(network, mode=None):
    """
    Checks for voltage stability issues in LV grids.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    mode : None or String
        If None voltage at all nodes in LV grid is checked. If mode is set to
        'stations' only voltage at busbar is checked.

    Returns
    -------
    Dict with :class:`~.grid.grids.LVGrid` as keys.
    If mode is None values of dictionary are critical nodes of grid as
    :pandas:`pandas.Series<series>`, sorted descending by voltage deviation.
    (Format: {grid_1: pd.Series(data=[v_mag_pu_node_1A, v_mag_pu_node_1B],
                               index=[node_1A, node_1B]), ...}).
    If mode is 'stations' values are maximum voltage deviation at secondary
    side of station (Format: {grid_1: v_mag_pu_station_grid_1, ...,
                              grid_n: v_mag_pu_station_grid_n}).

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.

    """

    #ToDo: devide this function into several functions to not have so many
    # if statements
    crit_nodes = {}

    # load max. voltage deviation
    # ToDo: for now only voltage deviation for the combined calculation of MV
    # and LV is considered (load and feed-in case for seperate consideration
    # of MV and LV needs to be implemented)
    max_v_dev = network.config['grid_expansion_allowed_voltage_deviations'][
        'mv_lv_max_v_deviation']

    for lv_grid in network.mv_grid.lv_grids:
        if mode:
            if mode == 'stations':
                v_mag_pu_pfa = network.results.v_res(
                    nodes=[lv_grid.station], level='lv')
            else:
                raise ValueError(
                    "{} is not a valid option for input variable 'mode' in "
                    "function lv_voltage_deviation. Try 'stations' or "
                    "None".format(mode))
        else:
            v_mag_pu_pfa = network.results.v_res(nodes=lv_grid.graph.nodes(),
                                                 level='lv')
        # check for overvoltage
        v_max = v_mag_pu_pfa.max()
        crit_nodes_max = v_max[(v_max > (1 + max_v_dev))] - 1
        # check for undervoltage
        v_min = v_mag_pu_pfa.min()
        crit_nodes_min = 1 - v_min[(v_min < (1 - max_v_dev))]
        # combine critical nodes and keep highest voltage deviation at each
        # node
        crit_nodes_grid = crit_nodes_max.append(crit_nodes_min).max(level=0)
        if len(crit_nodes_grid) > 0:
            if not mode:
                crit_nodes[lv_grid] = crit_nodes_grid.sort_values(
                    ascending=False)
            else:
                crit_nodes[lv_grid] = crit_nodes_grid[repr(lv_grid.station)]

    if crit_nodes:
        logger.debug(
            '==> {} LV grid(s) has/have voltage issues.'.format(
                len(crit_nodes)))
    else:
        logger.debug('==> No voltage issues in LV grids.')

    return crit_nodes
