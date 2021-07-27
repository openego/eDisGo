import pandas as pd
import logging
from math import sqrt
import numpy as np
import itertools

from edisgo.network.grids import LVGrid, MVGrid

logger = logging.getLogger("edisgo")


def mv_line_load(edisgo_obj):
    """
    Checks for over-loading issues in MV network.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV lines, their maximum relative
        over-loading (maximum calculated current over allowed current) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    Notes
    -----
    Line over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_lines = _line_load(edisgo_obj, voltage_level="mv")

    if not crit_lines.empty:
        logger.debug(
            "==> {} line(s) in MV network has/have load issues.".format(
                crit_lines.shape[0]
            )
        )
    else:
        logger.debug("==> No line load issues in MV network.")

    return crit_lines


def lv_line_load(edisgo_obj):
    """
    Checks for over-loading issues in LV network.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded LV lines, their maximum relative
        over-loading (maximum calculated current over allowed current) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    Notes
    -----
    Line over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_lines = _line_load(edisgo_obj, voltage_level="lv")

    if not crit_lines.empty:
        logger.debug(
            "==> {} line(s) in LV networks has/have load issues.".format(
                crit_lines.shape[0]
            )
        )
    else:
        logger.debug("==> No line load issues in LV networks.")

    return crit_lines


def lines_allowed_load(edisgo_obj, voltage_level):
    """
    Get allowed maximum current per line per time step

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_level : str
        Grid level, allowed line load is returned for. Possible options are
        "mv" or "lv".

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the maximum allowed current per line and time step
        in kA. Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Columns are line names of all lines in the specified voltage level.

    """
    # get lines and nominal voltage
    mv_grid = edisgo_obj.topology.mv_grid
    if voltage_level == "lv":
        lines_df = edisgo_obj.topology.lines_df[
            ~edisgo_obj.topology.lines_df.index.isin(mv_grid.lines_df.index)
        ]
        if len(list(mv_grid.lv_grids)) > 0:
            nominal_voltage = list(mv_grid.lv_grids)[0].nominal_voltage
        else:
            nominal_voltage = np.NaN
    elif voltage_level == "mv":
        lines_df = mv_grid.lines_df
        nominal_voltage = mv_grid.nominal_voltage
    else:
        raise ValueError(
            "{} is not a valid option for input variable 'voltage_level' in "
            "function lines_allowed_load. Try 'mv' or "
            "'lv'.".format(voltage_level)
        )

    i_lines_allowed_per_case = {}
    i_lines_allowed_per_case["feedin_case"] = (
        lines_df.s_nom
        / sqrt(3)
        / nominal_voltage
        * edisgo_obj.config["grid_expansion_load_factors"][
            "{}_feedin_case_line".format(voltage_level)
        ]
    )

    # adapt i_lines_allowed for radial feeders
    buses_in_cycles = list(
        set(itertools.chain.from_iterable(edisgo_obj.topology.rings)))

    # Find lines in cycles
    lines_in_cycles = list(
        lines_df.loc[lines_df[[
            'bus0', 'bus1']].isin(buses_in_cycles).all(axis=1)].index.values)
    lines_radial_feeders = list(
        lines_df.loc[~lines_df.index.isin(lines_in_cycles)].index.values)

    # lines in cycles have to be n-1 secure
    i_lines_allowed_per_case["load_case"] = (
        lines_df.loc[lines_in_cycles].s_nom
        / sqrt(3)
        / nominal_voltage
        * edisgo_obj.config["grid_expansion_load_factors"][
            "{}_load_case_line".format(voltage_level)
        ]
    )

    # lines in radial feeders are not n-1 secure anyways
    i_lines_allowed_per_case["load_case"] = \
        i_lines_allowed_per_case["load_case"].append(
            lines_df.loc[lines_radial_feeders].s_nom / sqrt(3) / nominal_voltage)

    i_lines_allowed = edisgo_obj.timeseries.timesteps_load_feedin_case.loc[
        edisgo_obj.results.i_res.index
    ].apply(lambda _: i_lines_allowed_per_case[_])
    return i_lines_allowed


def lines_relative_load(edisgo_obj, lines_allowed_load):
    """
    Calculates relative line load based on specified allowed line load.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    lines_allowed_load : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the maximum allowed current per line and time step
        in kA. Index of the dataframe are time steps of type
        :pandas:`pandas.Timestamp<Timestamp>` and columns are line names.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the relative line load per line and time step.
        Index and columns of the dataframe are the same as those of parameter
        `lines_allowed_load`.

    """
    # get line load from power flow analysis
    i_lines_pfa = edisgo_obj.results.i_res.loc[lines_allowed_load.index,
                                               lines_allowed_load.columns]

    return i_lines_pfa / lines_allowed_load


def _line_load(edisgo_obj, voltage_level):
    """
    Checks for over-loading issues of lines.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_level : str
        Voltage level, over-loading is checked for. Possible options are
        "mv" or "lv".

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading (maximum calculated current over allowed current) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    """
    if edisgo_obj.results.i_res.empty:
        raise Exception(
            "No power flow results to check over-load for. Please perform "
            "power flow analysis first."
        )

    # get allowed line load
    i_lines_allowed = lines_allowed_load(edisgo_obj, voltage_level)

    # calculate relative line load and keep maximum over-load of each line
    relative_i_res = lines_relative_load(edisgo_obj, i_lines_allowed)

    crit_lines_relative_load = (
        relative_i_res[relative_i_res > 1].max().dropna()
    )
    if len(crit_lines_relative_load) > 0:
        crit_lines = pd.concat(
            [
                crit_lines_relative_load,
                relative_i_res.idxmax()[crit_lines_relative_load.index],
            ],
            axis=1,
            keys=["max_rel_overload", "time_index"],
            sort=True
        )
        crit_lines.loc[:, "voltage_level"] = voltage_level
    else:
        crit_lines = pd.DataFrame()

    return crit_lines


def hv_mv_station_load(edisgo_obj):
    """
    Checks for over-loading of HV/MV station.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded HV/MV station, their apparent power
        at maximal over-loading and the corresponding time step.
        Index of the dataframe is the representative of the MVGrid.
        Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """
    crit_stations = _station_load(edisgo_obj, edisgo_obj.topology.mv_grid)
    if not crit_stations.empty:
        logger.debug("==> HV/MV station has load issues.")
    else:
        logger.debug("==> No HV/MV station load issues.")

    return crit_stations


def mv_lv_station_load(edisgo_obj):
    """
    Checks for over-loading of MV/LV stations.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV/LV stations, their missing apparent
        power at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_stations = pd.DataFrame()
    for lv_grid in edisgo_obj.topology.mv_grid.lv_grids:
        crit_stations = crit_stations.append(
            _station_load(edisgo_obj, lv_grid)
        )
    if not crit_stations.empty:
        logger.debug(
            "==> {} MV/LV station(s) has/have load issues.".format(
                crit_stations.shape[0]
            )
        )
    else:
        logger.debug("==> No MV/LV station load issues.")

    return crit_stations


def _station_load(edisgo_obj, grid):
    """
    Checks for over-loading of stations.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded stations, their missing apparent
        power at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    """
    # get apparent power over station from power flow analysis
    if isinstance(grid, LVGrid):
        voltage_level = "lv"
        transformers_df = grid.transformers_df
        s_station_pfa = edisgo_obj.results.s_res.loc[
            :, transformers_df.index
        ].sum(axis=1)
    elif isinstance(grid, MVGrid):
        voltage_level = "mv"
        transformers_df = edisgo_obj.topology.transformers_hvmv_df
        # ensure that power flow was conducted for MV
        mv_lines = edisgo_obj.topology.mv_grid.lines_df.index
        if not any(mv_lines.isin(edisgo_obj.results.i_res.columns)):
            raise ValueError(
                "MV was not included in power flow analysis, wherefore load "
                "of HV/MV station cannot be calculated.")
        s_station_pfa = np.hypot(
            edisgo_obj.results.pfa_slack.p,
            edisgo_obj.results.pfa_slack.q,
        )
    else:
        raise ValueError("Inserted grid is invalid.")

    # get maximum allowed apparent power of station in each time step
    s_station = sum(transformers_df.s_nom)
    load_factor = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
        lambda _: edisgo_obj.config["grid_expansion_load_factors"][
            "{}_{}_transformer".format(voltage_level, _)
        ]
    )
    s_station_allowed = s_station * load_factor

    # calculate residual apparent power (if negative, station is over-loaded)
    s_res = s_station_allowed - s_station_pfa
    s_res = s_res[s_res < 0]

    if not s_res.empty:
        # calculate greatest apparent power missing (residual apparent power is
        # devided by the load factor to account for load factors smaller than
        # one, which lead to a higher needed additional capacity)
        s_missing = (s_res / load_factor).dropna()
        crit_stations = pd.DataFrame(
            {
                "s_missing": abs(s_missing.min()),
                "time_index": s_missing.idxmin(),
            },
            index=[repr(grid)],
        )
    else:
        crit_stations = pd.DataFrame()

    return crit_stations


def mv_voltage_deviation(edisgo_obj, voltage_levels="mv_lv"):
    """
    Checks for voltage stability issues in MV network.

    Returns buses with voltage issues and their maximum voltage deviation.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_levels : :obj:`str`
        Specifies which allowed voltage deviations to use. Possible options
        are:

        * 'mv_lv'
          This is the default. The allowed voltage deviations for buses in the
          MV is the same as for buses in the LV. Further, load and feed-in case
          are not distinguished.
        * 'mv'
          Use this to handle allowed voltage limits in the MV and LV
          topology differently. In that case, load and feed-in case are
          differentiated.

    Returns
    -------
    :obj:`dict`
        Dictionary with representative of :class:`~.network.grids.MVGrid` as
        key and a :pandas:`pandas.DataFrame<DataFrame>` with voltage
        deviations from allowed lower or upper voltage limits, sorted
        descending from highest to lowest voltage deviation, as value.
        Index of the dataframe are all buses with voltage issues.
        Columns are 'v_diff_max' containing the maximum voltage deviation as
        float and 'time_index' containing the corresponding time step the
        voltage issue occured in as :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Voltage issues are determined based on allowed voltage deviations defined
    in the config file 'config_grid_expansion' in section
    'grid_expansion_allowed_voltage_deviations'.

    """

    crit_buses = {}

    # get allowed lower and upper voltage limits
    v_limits_upper, v_limits_lower = _mv_allowed_voltage_limits(
        edisgo_obj, voltage_levels
    )

    # find buses with voltage issues and their maximum voltage deviation
    crit_buses_grid = _voltage_deviation(
        edisgo_obj,
        edisgo_obj.topology.mv_grid.buses_df.index,
        v_limits_upper,
        v_limits_lower,
    )

    if not crit_buses_grid.empty:
        crit_buses[repr(edisgo_obj.topology.mv_grid)] = crit_buses_grid
        logger.debug(
            "==> {} bus(es) in MV topology has/have voltage issues.".format(
                crit_buses_grid.shape[0]
            )
        )
    else:
        logger.debug("==> No voltage issues in MV topology.")

    return crit_buses


def lv_voltage_deviation(edisgo_obj, mode=None, voltage_levels="mv_lv"):
    """
    Checks for voltage stability issues in LV networks.

    Returns buses with voltage issues and their maximum voltage deviation.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    mode : None or :obj:`str`
        If None voltage at all buses in LV networks is checked. If mode is set
        to 'stations' only voltage at bus bar is checked. Default: None.
    voltage_levels : :obj:`str`
        Specifies which allowed voltage deviations to use. Possible options
        are:

        * 'mv_lv'
          This is the default. The allowed voltage deviations for buses in the
          LV is the same as for buses in the MV. Further, load and feed-in case
          are not distinguished.
        * 'lv'
          Use this to handle allowed voltage limits in the MV and LV
          topology differently. In that case, load and feed-in case are
          differentiated.

    Returns
    -------
    :obj:`dict`
        Dictionary with representative of :class:`~.network.grids.LVGrid` as
        key and a :pandas:`pandas.DataFrame<DataFrame>` with voltage
        deviations from allowed lower or upper voltage limits, sorted
        descending from highest to lowest voltage deviation, as value.
        Index of the dataframe are all buses with voltage issues.
        Columns are 'v_diff_max' containing the maximum voltage deviation as
        float and 'time_index' containing the corresponding time step the
        voltage issue occured in as :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Voltage issues are determined based on allowed voltage deviations defined
    in the config file 'config_grid_expansion' in section
    'grid_expansion_allowed_voltage_deviations'.

    """

    crit_buses = {}

    if voltage_levels == "mv_lv":
        v_limits_upper, v_limits_lower = _mv_allowed_voltage_limits(
            edisgo_obj, "mv_lv"
        )
    elif not "lv" == voltage_levels:
        raise ValueError(
            "{} is not a valid option for input variable 'voltage_levels' in "
            "function lv_voltage_deviation. Try 'mv_lv' or "
            "'lv'.".format(voltage_levels)
        )

    for lv_grid in edisgo_obj.topology.mv_grid.lv_grids:

        if mode:
            if mode == "stations":
                buses = lv_grid.station.index
            else:
                raise ValueError(
                    "{} is not a valid option for input variable 'mode' in "
                    "function lv_voltage_deviation. Try 'stations' or "
                    "None.".format(mode)
                )
        else:
            buses = lv_grid.buses_df.index

        if voltage_levels == "lv":
            v_limits_upper, v_limits_lower = _lv_allowed_voltage_limits(
                edisgo_obj, lv_grid, mode
            )

        crit_buses_grid = _voltage_deviation(
            edisgo_obj, buses, v_limits_upper, v_limits_lower
        )

        if not crit_buses_grid.empty:
            crit_buses[repr(lv_grid)] = crit_buses_grid

    if crit_buses:
        if mode == "stations":
            logger.debug(
                "==> {} LV station(s) has/have voltage issues.".format(
                    len(crit_buses)
                )
            )
        else:
            logger.debug(
                "==> {} LV topology(s) has/have voltage issues.".format(
                    len(crit_buses)
                )
            )
    else:
        if mode == "stations":
            logger.debug("==> No voltage issues in LV stations.")
        else:
            logger.debug("==> No voltage issues in LV grids.")

    return crit_buses


def _mv_allowed_voltage_limits(edisgo_obj, voltage_levels):
    """
    Calculates allowed upper and lower MV voltage limits in p.u..

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_levels : :obj:`str`
        Specifies which allowed voltage limits to use. Possible options
        are:

        * 'mv_lv'
          The allowed voltage deviations for buses in the MV are the same as
          for buses in the LV, namely $pm$ 10 %.
        * 'mv'
          Use this to handle allowed voltage limits in the MV and LV
          differently. In that case load and feed-in case are differentiated.

    Returns
    -------
    :pandas:`pandas.Series<Series>`
        Series containing the allowed upper voltage limits in p.u..
        Index of the series are all time steps power flow was last conducted
        for of type :pandas:`pandas.Timestamp<Timestamp>`.

    :pandas:`pandas.Series<Series>`
        Series containing the allowed lower voltage limits in p.u..
        Index of the series are all time steps power flow was last conducted
        for of type :pandas:`pandas.Timestamp<Timestamp>`.

    """
    v_allowed_per_case = {}

    # get config values for lower voltage limit in feed-in case and upper
    # voltage limit in load case
    v_allowed_per_case["feedin_case_lower"] = edisgo_obj.config[
        "grid_expansion_allowed_voltage_deviations"
    ]["feedin_case_lower"]
    v_allowed_per_case["load_case_upper"] = edisgo_obj.config[
        "grid_expansion_allowed_voltage_deviations"
    ]["load_case_upper"]

    # calculate upper voltage limit in feed-in case and lower voltage limit in
    # load case
    offset = edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
        "hv_mv_trafo_offset"
    ]
    control_deviation = edisgo_obj.config[
        "grid_expansion_allowed_voltage_deviations"
    ]["hv_mv_trafo_control_deviation"]

    if voltage_levels == "mv_lv" or voltage_levels == "mv":
        v_allowed_per_case["feedin_case_upper"] = (
            1
            + offset
            + control_deviation
            + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "{}_feedin_case_max_v_deviation".format(voltage_levels)
            ]
        )
        v_allowed_per_case["load_case_lower"] = (
            1
            + offset
            - control_deviation
            - edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "{}_load_case_max_v_deviation".format(voltage_levels)
            ]
        )
    else:
        raise ValueError(
            "Specified mode {} is not a valid option.".format(voltage_levels)
        )

    # create series with upper and lower voltage limits for each time step
    v_limits_upper = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
        lambda _: v_allowed_per_case["{}_upper".format(_)]
    )
    v_limits_lower = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
        lambda _: v_allowed_per_case["{}_lower".format(_)]
    )

    return v_limits_upper, v_limits_lower


def _lv_allowed_voltage_limits(edisgo_obj, lv_grid, mode):
    """
    Calculates allowed upper and lower voltage limits for given LV grid.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    lv_grid : :class:`~.network.grids.LVGrid`
        LV grid to get voltage limits for.
    mode : None or :obj:`str`
        If None, voltage limits for buses in the LV network are returned. In
        that case the reference bus is the LV stations' secondary side.
        If mode is set to 'stations', voltage limits for stations' secondary
        side (LV bus bar) are returned; the reference bus is the stations'
        primary side.

    Returns
    -------
    :pandas:`pandas.Series<Series>`
        Series containing the allowed upper voltage limits in p.u..
        Index of the series are all time steps power flow was last conducted
        for of type :pandas:`pandas.Timestamp<Timestamp>`.

    :pandas:`pandas.Series<Series>`
        Series containing the allowed lower voltage limits in p.u..
        Index of the series are all time steps power flow was last conducted
        for of type :pandas:`pandas.Timestamp<Timestamp>`.

    """
    v_allowed_per_case = {}

    # get reference voltages for different modes
    if mode == "stations":
        # reference voltage is voltage at stations' primary side
        bus_station_primary = lv_grid.transformers_df.iloc[0].bus0
        voltage_base = edisgo_obj.results.v_res.loc[:, bus_station_primary]
        config_string = "mv_lv_station"
    else:
        # reference voltage is voltage at stations' secondary side
        voltage_base = edisgo_obj.results.v_res.loc[
            :, lv_grid.station.index.values[0]
        ]
        config_string = "lv"

    # calculate upper voltage limit in feed-in case and lower voltage limit in
    # load case
    v_allowed_per_case["feedin_case_upper"] = (
        voltage_base
        + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "{}_feedin_case_max_v_deviation".format(config_string)
        ]
    )
    v_allowed_per_case["load_case_lower"] = (
        voltage_base
        - edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "{}_load_case_max_v_deviation".format(config_string)
        ]
    )

    timeindex = voltage_base.index
    v_allowed_per_case["feedin_case_lower"] = pd.Series(
        edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "feedin_case_lower"
        ],
        index=timeindex,
    )
    v_allowed_per_case["load_case_upper"] = pd.Series(
        edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "load_case_upper"
        ],
        index=timeindex,
    )

    # create series with upper and lower voltage limits for each time step
    v_limits_upper = []
    v_limits_lower = []
    load_feedin_case = edisgo_obj.timeseries.timesteps_load_feedin_case
    for t in timeindex:
        case = load_feedin_case.loc[t]
        v_limits_upper.append(
            v_allowed_per_case["{}_upper".format(case)].loc[t]
        )
        v_limits_lower.append(
            v_allowed_per_case["{}_lower".format(case)].loc[t]
        )
    v_limits_upper = pd.Series(v_limits_upper, index=timeindex)
    v_limits_lower = pd.Series(v_limits_lower, index=timeindex)

    return v_limits_upper, v_limits_lower


def voltage_diff(edisgo_obj, buses, v_dev_allowed_upper, v_dev_allowed_lower):
    """
    Function to detect under- and overvoltage at buses.

    The function returns both under- and overvoltage deviations in p.u. from
    the allowed lower and upper voltage limit, respectively, in separate
    dataframes. In case of both under- and overvoltage issues at one bus,
    only the highest voltage deviation is returned.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    buses : list(str)
        List of buses to check voltage deviation for.
    v_dev_allowed_upper : :pandas:`pandas.Series<Series>`
        Series with time steps (of type :pandas:`pandas.Timestamp<Timestamp>`)
        power flow analysis was conducted for and the allowed upper limit of
        voltage deviation for each time step as float in p.u..
    v_dev_allowed_lower : :pandas:`pandas.Series<Series>`
        Series with time steps (of type :pandas:`pandas.Timestamp<Timestamp>`)
        power flow analysis was conducted for and the allowed lower limit of
        voltage deviation for each time step as float in p.u..

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with deviations from allowed lower voltage level.
        Columns of the dataframe are all time steps power flow analysis was
        conducted for of type :pandas:`pandas.Timestamp<Timestamp>`; in the
        index are all buses for which undervoltage was detected. In case of
        a higher over- than undervoltage deviation for a bus, the bus does
        not appear in this dataframe, but in the dataframe with overvoltage
        deviations.
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with deviations from allowed upper voltage level.
        Columns of the dataframe are all time steps power flow analysis was
        conducted for of type :pandas:`pandas.Timestamp<Timestamp>`; in the
        index are all buses for which overvoltage was detected. In case of
        a higher under- than overvoltage deviation for a bus, the bus does
        not appear in this dataframe, but in the dataframe with undervoltage
        deviations.

    """
    v_mag_pu_pfa = edisgo_obj.results.v_res.loc[:, buses]

    v_dev_allowed_upper_format = np.tile(
        (v_dev_allowed_upper.loc[v_mag_pu_pfa.index]).values,
        (v_mag_pu_pfa.shape[1], 1),
    )
    v_dev_allowed_lower_format = np.tile(
        (v_dev_allowed_lower.loc[v_mag_pu_pfa.index]).values,
        (v_mag_pu_pfa.shape[1], 1),
    )
    overvoltage = v_mag_pu_pfa.T[
        v_mag_pu_pfa.T > v_dev_allowed_upper_format
    ].dropna(how="all")
    undervoltage = v_mag_pu_pfa.T[
        v_mag_pu_pfa.T < v_dev_allowed_lower_format
    ].dropna(how="all")
    # sort buses with under- and overvoltage issues in a way that
    # worst case is saved
    buses_both = v_mag_pu_pfa[
        overvoltage[overvoltage.index.isin(undervoltage.index)].index
    ]
    voltage_diff_ov = (
        buses_both.T - v_dev_allowed_upper.loc[v_mag_pu_pfa.index].values
    )
    voltage_diff_uv = (
        -buses_both.T + v_dev_allowed_lower.loc[v_mag_pu_pfa.index].values
    )
    voltage_diff_ov = voltage_diff_ov.loc[
        voltage_diff_ov.max(axis=1) > voltage_diff_uv.max(axis=1)
    ]
    voltage_diff_uv = voltage_diff_uv.loc[
        ~voltage_diff_uv.index.isin(voltage_diff_ov.index)
    ]
    # handle buses with overvoltage issues and append to voltage_diff_ov
    buses_ov = v_mag_pu_pfa[
        overvoltage[~overvoltage.index.isin(buses_both.columns)].index
    ]
    voltage_diff_ov = voltage_diff_ov.append(
        buses_ov.T - v_dev_allowed_upper.loc[v_mag_pu_pfa.index].values
    )

    # handle buses with undervoltage issues and append to voltage_diff_uv
    buses_uv = v_mag_pu_pfa[
        undervoltage[~undervoltage.index.isin(buses_both.columns)].index
    ]
    voltage_diff_uv = voltage_diff_uv.append(
        -buses_uv.T + v_dev_allowed_lower.loc[v_mag_pu_pfa.index].values
    )

    return voltage_diff_uv, voltage_diff_ov


def _voltage_deviation(edisgo_obj, buses, v_limits_upper, v_limits_lower):
    """
    Function to detect voltage issues at buses.

    The function returns the highest voltage deviation from allowed lower
    or upper voltage limit in p.u. for all buses with voltage issues.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    buses : list(str)
        List of buses to check voltage deviation for.
    v_limits_upper : :pandas:`pandas.Series<Series>`
        Series with time steps (of type :pandas:`pandas.Timestamp<Timestamp>`)
        power flow analysis was conducted for and the allowed upper limit of
        voltage deviation for each time step as float in p.u..
    v_limits_lower : :pandas:`pandas.Series<Series>`
        Series with time steps (of type :pandas:`pandas.Timestamp<Timestamp>`)
        power flow analysis was conducted for and the allowed lower limit of
        voltage deviation for each time step as float in p.u..

    Returns
    -------
    pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with deviations from allowed lower or upper voltage limits
        sorted descending from highest to lowest voltage deviation
        (it is not distinguished between over- or undervoltage).
        Columns of the dataframe are 'v_diff_max' containing the maximum
        absolute voltage deviation as float and 'time_index' containing the
        corresponding time step the voltage issue occured in as
        :pandas:`pandas.Timestamp<Timestamp>`. Index of the dataframe are the
        names of all buses with voltage issues.

    """

    def _append_crit_buses(df):
        return pd.DataFrame(
            {
                "v_diff_max": df.max(axis=1).values,
                "time_index": df.idxmax(axis=1).values,
            },
            index=df.index,
        )

    crit_buses_grid = pd.DataFrame()

    voltage_diff_uv, voltage_diff_ov = voltage_diff(
        edisgo_obj, buses, v_limits_upper, v_limits_lower
    )

    # append to crit buses dataframe
    if not voltage_diff_ov.empty:
        crit_buses_grid = crit_buses_grid.append(
            _append_crit_buses(voltage_diff_ov)
        )
    if not voltage_diff_uv.empty:
        crit_buses_grid = crit_buses_grid.append(
            _append_crit_buses(voltage_diff_uv)
        )

    if not crit_buses_grid.empty:
        crit_buses_grid.sort_values(
            by=["v_diff_max"], ascending=False, inplace=True
        )

    return crit_buses_grid


def check_ten_percent_voltage_deviation(edisgo_obj):
    """
    Checks if 10% criteria is exceeded.

    Through the 10% criteria it is ensured that voltage is kept between 0.9
    and 1.1 p.u.. In case of higher or lower voltages a ValueError is raised.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    """

    v_mag_pu_pfa = edisgo_obj.results.v_res
    if (v_mag_pu_pfa > 1.1).any().any() or (v_mag_pu_pfa < 0.9).any().any():
        message = "Maximum allowed voltage deviation of 10% exceeded."
        raise ValueError(message)
