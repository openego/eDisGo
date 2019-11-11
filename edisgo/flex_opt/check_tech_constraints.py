import pandas as pd
import logging

from edisgo.network.grids import LVGrid, MVGrid

logger = logging.getLogger('edisgo')


def mv_line_load(network):
    """
    Checks for over-loading issues in MV topology.

    Parameters
    ----------
    network : :class:`~.network.topology.Topology`

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded MV lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
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
        logger.debug('==> {} line(s) in MV topology has/have load issues.'.format(
            crit_lines.shape[0]))
    else:
        logger.debug('==> No line load issues in MV topology.')

    return crit_lines


def lv_line_load(network):
    """
    Checks for over-loading issues in LV grids.

    Parameters
    ----------
    network : :class:`~.network.topology.Topology`

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded LV lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
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
    Checks for over-loading issues of lines.

    Parameters
    ----------
    network : :class:`~.network.network.Network`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`
    crit_lines : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
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
        # maximum allowed line load in each time step
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


def hv_mv_station_load(edisgo):
    """
    Checks for over-loading of HV/MV station.

    Parameters
    ----------
    edisgo : :class:`~.topology.topology.Edisgo`

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded HV/MV stations, their apparent power
        at maximal over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded stations of type
        :class:`~.network.components.MVStation`. Columns are 's_pfa'
        containing the apparent power at maximal over-loading as float and
        'time_index' containing the corresponding time step the over-loading
        occured in as :pandas:`pandas.Timestamp<timestamp>`.

    Notes
    -----
    Over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """
    crit_stations = pd.DataFrame()
    crit_stations = _station_load(edisgo, edisgo.topology.mv_grid,
                                  crit_stations)
    if not crit_stations.empty:
        logger.debug('==> HV/MV station has load issues.')
    else:
        logger.debug('==> No HV/MV station load issues.')

    return crit_stations


def mv_lv_station_load(edisgo_obj):
    """
    Checks for over-loading of MV/LV stations.

    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded MV/LV stations, their apparent power
        at maximal over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded stations of type
        :class:`~.network.components.LVStation`. Columns are 's_pfa'
        containing the apparent power at maximal over-loading as float and
        'time_index' containing the corresponding time step the over-loading
        occured in as :pandas:`pandas.Timestamp<timestamp>`.

    Notes
    -----
    Over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_stations = pd.DataFrame()

    for lv_grid in edisgo_obj.topology.mv_grid.lv_grids:
        crit_stations = _station_load(edisgo_obj, lv_grid,
                                      crit_stations)
    if not crit_stations.empty:
        logger.debug('==> {} MV/LV station(s) has/have load issues.'.format(
            crit_stations.shape[0]))
    else:
        logger.debug('==> No MV/LV station load issues.')

    return crit_stations


def _station_load(edisgo, grid, crit_stations):
    """
    Checks for over-loading of stations.

    Parameters
    ----------
    edisgo : :class:`~.network.network.Edisgo`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`
    crit_stations : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded stations, their apparent power at
        maximal over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded stations either of type
        :class:`~.network.components.LVStation` or
        :class:`~.network.components.MVStation`. Columns are 's_pfa'
        containing the apparent power at maximal over-loading as float and
        'time_index' containing the corresponding time step the over-loading
        occured in as :pandas:`pandas.Timestamp<timestamp>`.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing over-loaded stations, their apparent power at
        maximal over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded stations either of type
        :class:`~.network.components.LVStation` or
        :class:`~.network.components.MVStation`. Columns are 's_pfa'
        containing the apparent power at maximal over-loading as float and
        'time_index' containing the corresponding time step the over-loading
        occured in as :pandas:`pandas.Timestamp<timestamp>`.

    """
    if len(grid.transformers_df) < 1:
        logger.warning('No station found, cannot check station.')
        return crit_stations

    if isinstance(grid, LVGrid):
        grid_level = 'lv'
    elif isinstance(grid, MVGrid):
        grid_level = 'mv'
    else:
        raise ValueError('Inserted topology of unknown type.')

    # maximum allowed apparent power of station for feed-in and load case
    s_station = sum(grid.transformers_df.s_nom)
    s_station_allowed_per_case = {}
    s_station_allowed_per_case['feedin_case'] = s_station * edisgo.config[
        'grid_expansion_load_factors']['{}_feedin_case_transformer'.format(
        grid_level)]
    s_station_allowed_per_case['load_case'] = s_station * edisgo.config[
        'grid_expansion_load_factors']['{}_load_case_transformer'.format(
        grid_level)]
    # maximum allowed apparent power of station in each time step
    s_station_allowed = \
        edisgo.timeseries.timesteps_load_feedin_case.apply(
            lambda _: s_station_allowed_per_case[_])

    try:
        if grid_level == 'lv':
            s_station_pfa = edisgo.results.s_res(
                grid.transformers_df).sum(axis=1)
        elif grid_level == 'mv':
            s_station_pfa = (edisgo.results.hv_mv_exchanges.p ** 2 + \
                edisgo.results.hv_mv_exchanges.q ** 2) ** 0.5
        else:
            raise ValueError('Unknown grid level. Please check.')
        s_res = s_station_allowed - s_station_pfa
        s_res = s_res[s_res < 0]
        # check if maximum allowed apparent power of station exceeds
        # apparent power from power flow analysis at any time step
        if not s_res.empty:
            # find out largest relative deviation
            load_factor = \
                edisgo.timeseries.timesteps_load_feedin_case.apply(
                    lambda _: edisgo.config[
                        'grid_expansion_load_factors'][
                        '{}_{}_transformer'.format(grid_level, _)])
            relative_s_res = load_factor * s_res
            crit_stations = crit_stations.append(pd.DataFrame(
                {'s_pfa': s_station_pfa.loc[relative_s_res.idxmin()],
                 'time_index': relative_s_res.idxmin()},
                index=grid.transformers_df.bus1.unique()))

    except KeyError:
        logger.debug('No results for {} station to check overloading.'.format(
            grid_level.upper()))

    return crit_stations


def mv_voltage_deviation(network, voltage_levels='mv_lv'):
    """
    Checks for voltage stability issues in MV topology.

    Parameters
    ----------
    network : :class:`~.network.topology.Topology`
    voltage_levels : :obj:`str`
        Specifies which allowed voltage deviations to use. Possible options
        are:

        * 'mv_lv'
          This is the default. The allowed voltage deviation for nodes in the
          MV topology is the same as for nodes in the LV topology. Further load and
          feed-in case are not distinguished.
        * 'mv'
          Use this to handle allowed voltage deviations in the MV and LV topology
          differently. Here, load and feed-in case are differentiated as well.

    Returns
    -------
    :obj:`dict`
        Dictionary with :class:`~.network.grids.MVGrid` as key and a
        :pandas:`pandas.DataFrame<dataframe>` with its critical nodes, sorted
        descending by voltage deviation, as value.
        Index of the dataframe are all nodes (of type
        :class:`~.network.components.Generator`, :class:`~.network.components.Load`,
        etc.) with over-voltage issues. Columns are 'v_mag_pu' containing the
        maximum voltage deviation as float and 'time_index' containing the
        corresponding time step the over-voltage occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Notes
    -----
    Voltage issues are determined based on allowed voltage deviations defined
    in the config file 'config_grid_expansion' in section
    'grid_expansion_allowed_voltage_deviations'.

    """

    crit_nodes = {}

    v_dev_allowed_per_case = {}
    v_dev_allowed_per_case['feedin_case_lower'] = 0.9
    v_dev_allowed_per_case['load_case_upper'] = 1.1
    offset = network.config[
        'grid_expansion_allowed_voltage_deviations']['hv_mv_trafo_offset']
    control_deviation = network.config[
        'grid_expansion_allowed_voltage_deviations'][
        'hv_mv_trafo_control_deviation']
    if voltage_levels == 'mv_lv':
        v_dev_allowed_per_case['feedin_case_upper'] = \
            1 + offset + control_deviation + network.config[
                'grid_expansion_allowed_voltage_deviations'][
                'mv_lv_feedin_case_max_v_deviation']
        v_dev_allowed_per_case['load_case_lower'] = \
            1 + offset - control_deviation - network.config[
                'grid_expansion_allowed_voltage_deviations'][
                'mv_lv_load_case_max_v_deviation']
    elif voltage_levels == 'mv':
        v_dev_allowed_per_case['feedin_case_upper'] = \
            1 + offset + control_deviation + network.config[
                'grid_expansion_allowed_voltage_deviations'][
                'mv_feedin_case_max_v_deviation']
        v_dev_allowed_per_case['load_case_lower'] = \
            1 + offset - control_deviation - network.config[
                'grid_expansion_allowed_voltage_deviations'][
                'mv_load_case_max_v_deviation']
    else:
        raise ValueError(
            'Specified mode {} is not a valid option.'.format(voltage_levels))
    # maximum allowed apparent power of station in each time step
    v_dev_allowed_upper = \
        network.timeseries.timesteps_load_feedin_case.case.apply(
            lambda _: v_dev_allowed_per_case['{}_upper'.format(_)])
    v_dev_allowed_lower = \
        network.timeseries.timesteps_load_feedin_case.case.apply(
            lambda _: v_dev_allowed_per_case['{}_lower'.format(_)])

    nodes = list(network.mv_grid.graph.nodes())

    crit_nodes_grid = _voltage_deviation(
        network, nodes, v_dev_allowed_upper, v_dev_allowed_lower,
        voltage_level='mv')

    if not crit_nodes_grid.empty:
        crit_nodes[network.mv_grid] = crit_nodes_grid.sort_values(
            by=['v_mag_pu'], ascending=False)
        logger.debug(
            '==> {} node(s) in MV topology has/have voltage issues.'.format(
                crit_nodes[network.mv_grid].shape[0]))
    else:
        logger.debug('==> No voltage issues in MV topology.')

    return crit_nodes


def lv_voltage_deviation(network, mode=None, voltage_levels='mv_lv'):
    """
    Checks for voltage stability issues in LV grids.

    Parameters
    ----------
    network : :class:`~.network.topology.Topology`
    mode : None or String
        If None voltage at all nodes in LV topology is checked. If mode is set to
        'stations' only voltage at busbar is checked.
    voltage_levels : :obj:`str`
        Specifies which allowed voltage deviations to use. Possible options
        are:

        * 'mv_lv'
          This is the default. The allowed voltage deviation for nodes in the
          MV topology is the same as for nodes in the LV topology. Further load and
          feed-in case are not distinguished.
        * 'lv'
          Use this to handle allowed voltage deviations in the MV and LV topology
          differently. Here, load and feed-in case are differentiated as well.

    Returns
    -------
    :obj:`dict`
        Dictionary with :class:`~.network.grids.LVGrid` as key and a
        :pandas:`pandas.DataFrame<dataframe>` with its critical nodes, sorted
        descending by voltage deviation, as value.
        Index of the dataframe are all nodes (of type
        :class:`~.network.components.Generator`, :class:`~.network.components.Load`,
        etc.) with over-voltage issues. Columns are 'v_mag_pu' containing the
        maximum voltage deviation as float and 'time_index' containing the
        corresponding time step the over-voltage occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    Notes
    -----
    Voltage issues are determined based on allowed voltage deviations defined
    in the config file 'config_grid_expansion' in section
    'grid_expansion_allowed_voltage_deviations'.

    """

    crit_nodes = {}

    v_dev_allowed_per_case = {}
    if voltage_levels == 'mv_lv':
        offset = network.config[
            'grid_expansion_allowed_voltage_deviations']['hv_mv_trafo_offset']
        control_deviation = network.config[
            'grid_expansion_allowed_voltage_deviations'][
            'hv_mv_trafo_control_deviation']
        v_dev_allowed_per_case['feedin_case_upper'] = \
            1 + offset + control_deviation + network.config[
                'grid_expansion_allowed_voltage_deviations'][
                'mv_lv_feedin_case_max_v_deviation']
        v_dev_allowed_per_case['load_case_lower'] = \
            1 + offset - control_deviation - network.config[
                'grid_expansion_allowed_voltage_deviations'][
                'mv_lv_load_case_max_v_deviation']

        v_dev_allowed_per_case['feedin_case_lower'] = 0.9
        v_dev_allowed_per_case['load_case_upper'] = 1.1

        v_dev_allowed_upper = \
                network.timeseries.timesteps_load_feedin_case.case.apply(
                lambda _: v_dev_allowed_per_case['{}_upper'.format(_)])
        v_dev_allowed_lower = \
            network.timeseries.timesteps_load_feedin_case.case.apply(
                lambda _: v_dev_allowed_per_case['{}_lower'.format(_)])
    elif voltage_levels == 'lv':
        pass
    else:
        raise ValueError(
            'Specified mode {} is not a valid option.'.format(voltage_levels))

    for lv_grid in network.mv_grid.lv_grids:

        if mode:
            if mode == 'stations':
                nodes = [lv_grid.station]
            else:
                raise ValueError(
                    "{} is not a valid option for input variable 'mode' in "
                    "function lv_voltage_deviation. Try 'stations' or "
                    "None".format(mode))
        else:
            nodes = list(lv_grid.graph.nodes())

        if voltage_levels == 'lv':
            if mode == 'stations':
                # get voltage at primary side to calculate upper bound for
                # feed-in case and lower bound for load case
                v_lv_station_primary = network.results.v_res(
                    nodes=[lv_grid.station], level='mv').iloc[:, 0]
                timeindex = v_lv_station_primary.index
                v_dev_allowed_per_case['feedin_case_upper'] = \
                    v_lv_station_primary + network.config[
                        'grid_expansion_allowed_voltage_deviations'][
                        'mv_lv_station_feedin_case_max_v_deviation']
                v_dev_allowed_per_case['load_case_lower'] = \
                    v_lv_station_primary - network.config[
                        'grid_expansion_allowed_voltage_deviations'][
                        'mv_lv_station_load_case_max_v_deviation']
            else:
                # get voltage at secondary side to calculate upper bound for
                # feed-in case and lower bound for load case
                v_lv_station_secondary = network.results.v_res(
                    nodes=[lv_grid.station], level='lv').iloc[:, 0]
                timeindex = v_lv_station_secondary.index
                v_dev_allowed_per_case['feedin_case_upper'] = \
                    v_lv_station_secondary + network.config[
                        'grid_expansion_allowed_voltage_deviations'][
                        'lv_feedin_case_max_v_deviation']
                v_dev_allowed_per_case['load_case_lower'] = \
                    v_lv_station_secondary - network.config[
                        'grid_expansion_allowed_voltage_deviations'][
                        'lv_load_case_max_v_deviation']
            v_dev_allowed_per_case['feedin_case_lower'] = pd.Series(
                0.9, index=timeindex)
            v_dev_allowed_per_case['load_case_upper'] = pd.Series(
                1.1, index=timeindex)
            # maximum allowed voltage deviation in each time step
            v_dev_allowed_upper = []
            v_dev_allowed_lower = []
            for t in timeindex:
                case = \
                    network.timeseries.timesteps_load_feedin_case.loc[
                        t, 'case']
                v_dev_allowed_upper.append(
                    v_dev_allowed_per_case[
                        '{}_upper'.format(case)].loc[t])
                v_dev_allowed_lower.append(
                    v_dev_allowed_per_case[
                        '{}_lower'.format(case)].loc[t])
            v_dev_allowed_upper = pd.Series(v_dev_allowed_upper,
                                            index=timeindex)
            v_dev_allowed_lower = pd.Series(v_dev_allowed_lower,
                                            index=timeindex)

        crit_nodes_grid = _voltage_deviation(
            network, nodes, v_dev_allowed_upper, v_dev_allowed_lower,
            voltage_level='lv')

        if not crit_nodes_grid.empty:
            crit_nodes[lv_grid] = crit_nodes_grid.sort_values(
                by=['v_mag_pu'], ascending=False)

    if crit_nodes:
        if mode == 'stations':
            logger.debug(
                '==> {} LV station(s) has/have voltage issues.'.format(
                    len(crit_nodes)))
        else:
            logger.debug(
                '==> {} LV topology(s) has/have voltage issues.'.format(
                    len(crit_nodes)))
    else:
        if mode == 'stations':
            logger.debug('==> No voltage issues in LV stations.')
        else:
            logger.debug('==> No voltage issues in LV grids.')

    return crit_nodes


def _voltage_deviation(network, nodes, v_dev_allowed_upper,
                       v_dev_allowed_lower, voltage_level):
    """
    Checks for voltage stability issues in LV grids.
    Parameters
    ----------
    network : :class:`~.network.topology.Topology`
    nodes : :obj:`list`
        List of nodes (of type :class:`~.network.components.Generator`,
        :class:`~.network.components.Load`, etc.) to check voltage deviation for.
    v_dev_allowed_upper : :pandas:`pandas.Series<series>`
        Series with time steps (of type :pandas:`pandas.Timestamp<timestamp>`)
        power flow analysis was conducted for and the allowed upper limit of
        voltage deviation for each time step as float.
    v_dev_allowed_lower : :pandas:`pandas.Series<series>`
        Series with time steps (of type :pandas:`pandas.Timestamp<timestamp>`)
        power flow analysis was conducted for and the allowed lower limit of
        voltage deviation for each time step as float.
    voltage_levels : :obj:`str`
        Specifies which voltage level to retrieve power flow analysis results
        for. Possible options are 'mv' and 'lv'.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with critical nodes, sorted descending by voltage deviation.
        Index of the dataframe are all nodes (of type
        :class:`~.network.components.Generator`, :class:`~.network.components.Load`,
        etc.) with over-voltage issues. Columns are 'v_mag_pu' containing the
        maximum voltage deviation as float and 'time_index' containing the
        corresponding time step the over-voltage occured in as
        :pandas:`pandas.Timestamp<timestamp>`.

    """

    def _append_crit_node(series):
        return pd.DataFrame({'v_mag_pu': series.max(),
                             'time_index': series.idxmax()},
                            index=[node])

    crit_nodes_grid = pd.DataFrame()

    v_mag_pu_pfa = network.results.v_res(nodes=nodes, level=voltage_level)

    for node in nodes:
        # check for over- and under-voltage
        overvoltage = v_mag_pu_pfa[repr(node)][
            (v_mag_pu_pfa[repr(node)] > (v_dev_allowed_upper.loc[
                v_mag_pu_pfa.index]))]
        undervoltage = v_mag_pu_pfa[repr(node)][
            (v_mag_pu_pfa[repr(node)] < (v_dev_allowed_lower.loc[
                v_mag_pu_pfa.index]))]

        # write greatest voltage deviation to dataframe
        if not overvoltage.empty:
            overvoltage_diff = overvoltage - v_dev_allowed_upper.loc[
                overvoltage.index]
            if not undervoltage.empty:
                undervoltage_diff = v_dev_allowed_lower.loc[
                    undervoltage.index] - undervoltage
                if overvoltage_diff.max() > undervoltage_diff.max():
                    crit_nodes_grid = crit_nodes_grid.append(
                        _append_crit_node(overvoltage_diff))
                else:
                    crit_nodes_grid = crit_nodes_grid.append(
                        _append_crit_node(undervoltage_diff))
            else:
                crit_nodes_grid = crit_nodes_grid.append(
                    _append_crit_node(overvoltage_diff))
        elif not undervoltage.empty:
            undervoltage_diff = v_dev_allowed_lower.loc[
                                    undervoltage.index] - undervoltage
            crit_nodes_grid = crit_nodes_grid.append(
                _append_crit_node(undervoltage_diff))

    return crit_nodes_grid


def check_ten_percent_voltage_deviation(network):
    """
    Checks if 10% criteria is exceeded.

    Parameters
    ----------
    network : :class:`~.network.topology.Topology`

    """

    v_mag_pu_pfa = network.results.v_res()
    if (v_mag_pu_pfa > 1.1).any().any() or (v_mag_pu_pfa < 0.9).any().any():
        message = "Maximum allowed voltage deviation of 10% exceeded."
        raise ValueError(message)
