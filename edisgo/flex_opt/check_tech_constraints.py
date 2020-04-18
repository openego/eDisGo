import pandas as pd
import logging
from math import sqrt
import numpy as np

from edisgo.network.grids import LVGrid, MVGrid

logger = logging.getLogger("edisgo")


def mv_line_load(edisgo_obj):
    """
    Checks for over-loading issues in MV topology.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Line over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_lines = pd.DataFrame()
    crit_lines = _line_load(
        edisgo_obj, edisgo_obj.topology.mv_grid, crit_lines
    )

    if not crit_lines.empty:
        logger.debug(
            "==> {} line(s) in MV topology has/have load issues.".format(
                crit_lines.shape[0]
            )
        )
    else:
        logger.debug("==> No line load issues in MV topology.")

    return crit_lines


def lv_line_load(edisgo_obj):
    """
    Checks for over-loading issues in LV grids.

    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded LV lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Line over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_lines = pd.DataFrame()

    for lv_grid in edisgo_obj.topology.mv_grid.lv_grids:
        crit_lines = _line_load(edisgo_obj, lv_grid, crit_lines)

    if not crit_lines.empty:
        logger.debug(
            "==> {} line(s) in LV grids has/have load issues.".format(
                crit_lines.shape[0]
            )
        )
    else:
        logger.debug("==> No line load issues in LV grids.")

    return crit_lines


def lines_allowed_load(edisgo_obj, grid, grid_level):
    """
    Get allowed maximum current per line per time step

    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`
    grid_level : :String: `mv` or `lv`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the maximum allowed current per line per time step
        Index of the dataframe are the timesteps of the supplied network of type
        :pandas:`pandas.Timestamp<Timestamp>`.

        Columns are the network lines of type :class:`~.network.components.Line`.
        They contain the maximum allowed current flow for each line per time step

    """
    i_lines_allowed_per_case = {}
    i_lines_allowed_per_case["feedin_case"] = (
        grid.lines_df.s_nom
        / sqrt(3)
        / grid.nominal_voltage
        * grid.lines_df.num_parallel
        * edisgo_obj.config["grid_expansion_load_factors"][
            "{}_feedin_case_line".format(grid_level)
        ]
    )
    i_lines_allowed_per_case["load_case"] = (
        grid.lines_df.s_nom
        / sqrt(3)
        / grid.nominal_voltage
        * grid.lines_df.num_parallel
        * edisgo_obj.config["grid_expansion_load_factors"][
            "{}_load_case_line".format(grid_level)
        ]
    )
    i_lines_allowed = edisgo_obj.timeseries.timesteps_load_feedin_case.loc[
        edisgo_obj.results.i_res.index
    ].apply(lambda _: i_lines_allowed_per_case[_])
    return i_lines_allowed


def _line_load(edisgo_obj, grid, crit_lines):
    """
    Checks for over-loading issues of lines.

    Parameters
    ----------
    network : :class:`~.network.network.Network`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`
    crit_lines : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading and the corresponding time step.
        Index of the dataframe are the over-loaded lines of type
        :class:`~.network.components.Line`. Columns are 'max_rel_overload'
        containing the maximum relative over-loading as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    """
    if edisgo_obj.results.i_res is None:
        raise Exception(
            "No results i_res to check. " "Please analyze grid first."
        )

    if isinstance(grid, LVGrid):
        grid_level = "lv"
    elif isinstance(grid, MVGrid):
        grid_level = "mv"
    else:
        raise ValueError("Inserted grid of unknown type.")

    i_lines_allowed = lines_allowed_load(edisgo_obj, grid, grid_level)

    try:
        i_lines_pfa = edisgo_obj.results.i_res[grid.lines_df.index]
        relative_i_res = i_lines_pfa / i_lines_allowed
        crit_lines_relative_load = (
            relative_i_res[relative_i_res > 1].max().dropna()
        )
        if len(crit_lines_relative_load) > 0:
            tmp_lines = pd.concat(
                [
                    crit_lines_relative_load,
                    relative_i_res.idxmax()[crit_lines_relative_load.index],
                ],
                axis=1,
                keys=["max_rel_overload", "time_index"],
            )
            tmp_lines.loc[:, "grid_level"] = grid_level
            crit_lines = crit_lines.append(tmp_lines)
    except KeyError:
        logger.debug(
            "No results for line to check overloading. Checking lines "
            "one by one"
        )
        for line_name, line in grid.lines_df.iterrows():
            try:
                i_line_pfa = edisgo_obj.results.i_res[line_name]
                i_line_allowed = i_line_allowed[line_name]
                if any((i_line_allowed - i_line_pfa) < 0):
                    # find out largest relative deviation
                    relative_i_res = i_line_pfa / i_line_allowed
                    crit_lines = crit_lines.append(
                        pd.DataFrame(
                            {
                                "max_rel_overload": relative_i_res.max(),
                                "grid_level": grid_level,
                                "time_index": relative_i_res.idxmax(),
                            },
                            index=[line_name],
                        )
                    )
            except KeyError:
                logger.debug(
                    "No results for line {} ".format(line.name)
                    + "to check overloading."
                )

    return crit_lines


def hv_mv_station_load(edisgo):
    """
    Checks for over-loading of HV/MV station.

    Parameters
    ----------
    edisgo : :class:`~.edisgo.Edisgo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded HV/MV stations, their apparent power
        at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the MVGrid of type
        :class:'~.network.grids.MVGrid' where over-loaded stations occur.
        Columns are 's_pfa' containing the apparent power at maximal
        over-loading as float and 'time_index' containing the corresponding
        time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """
    crit_stations = pd.DataFrame()
    crit_stations = _station_load(
        edisgo, edisgo.topology.mv_grid, crit_stations
    )
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
    edisgo_obj : :class:`~.edisgo.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV/LV stations, their apparent power
        at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of LVGrids of type
        :class:'~.network.grids.LVGrid' with over-loaded stations.
        Columns are 's_pfa' containing the apparent power at maximal
        over-loading as float and 'time_index' containing  the corresponding
        time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_stations = pd.DataFrame()

    for lv_grid in edisgo_obj.topology.mv_grid.lv_grids:
        crit_stations = _station_load(edisgo_obj, lv_grid, crit_stations)
    if not crit_stations.empty:
        logger.debug(
            "==> {} MV/LV station(s) has/have load issues.".format(
                crit_stations.shape[0]
            )
        )
    else:
        logger.debug("==> No MV/LV station load issues.")

    return crit_stations


def _station_load(edisgo_obj, grid, crit_stations):
    """
    Checks for over-loading of stations.

    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.Edisgo`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`
    crit_stations : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded stations, their apparent power at
        maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_pfa' containing the apparent power
        at maximal over-loading as float and 'time_index' containing the
        corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded stations, their apparent power at
        maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_pfa' containing the apparent power
        at maximal over-loading as float and 'time_index' containing the
        corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    """

    if isinstance(grid, LVGrid):
        grid_level = "lv"
        transformers_df = grid.transformers_df
    elif isinstance(grid, MVGrid):
        grid_level = "mv"
        transformers_df = edisgo_obj.topology.transformers_hvmv_df
    else:
        raise ValueError("Inserted grid of unknown type.")

    if len(transformers_df) < 1:
        logger.warning("No transformers found, cannot check station.")
        return crit_stations

    # maximum allowed apparent power of station for feed-in and load case
    s_station = sum(transformers_df.s_nom)
    s_station_allowed_per_case = {}
    s_station_allowed_per_case["feedin_case"] = (
        s_station
        * edisgo_obj.config["grid_expansion_load_factors"][
            "{}_feedin_case_transformer".format(grid_level)
        ]
    )
    s_station_allowed_per_case["load_case"] = (
        s_station
        * edisgo_obj.config["grid_expansion_load_factors"][
            "{}_load_case_transformer".format(grid_level)
        ]
    )
    # maximum allowed apparent power of station in each time step
    s_station_allowed = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
        lambda _: s_station_allowed_per_case[_]
    )

    try:
        if grid_level == "lv":
            s_station_pfa = edisgo_obj.results.s_res(transformers_df).sum(
                axis=1
            )
        elif grid_level == "mv":
            s_station_pfa = (
                edisgo_obj.results.hv_mv_exchanges.p ** 2
                + edisgo_obj.results.hv_mv_exchanges.q ** 2
            ) ** 0.5
        else:
            raise ValueError("Unknown grid level. Please check.")
        s_res = s_station_allowed - s_station_pfa
        s_res = s_res[s_res < 0]
        # check if maximum allowed apparent power of station exceeds
        # apparent power from power flow analysis at any time step
        if not s_res.empty:
            # find out largest relative deviation
            load_factor = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
                lambda _: edisgo_obj.config["grid_expansion_load_factors"][
                    "{}_{}_transformer".format(grid_level, _)
                ]
            )
            relative_s_res = load_factor * s_res
            crit_stations = crit_stations.append(
                pd.DataFrame(
                    {
                        "s_pfa": s_station_pfa.loc[relative_s_res.idxmin()],
                        "time_index": relative_s_res.idxmin(),
                    },
                    index=[repr(grid)],
                )
            )

    except KeyError:
        logger.debug(
            "No results for {} station to check overloading.".format(
                grid_level.upper()
            )
        )

    return crit_stations


def mv_allowed_deviations(edisgo_obj, voltage_levels):
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
          for buses in the LV. Further load and feed-in case are not
          distinguished.
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


def mv_voltage_deviation(edisgo_obj, voltage_levels="mv_lv"):
    """
    Checks for voltage stability issues in MV topology.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_levels : :obj:`str`
        Specifies which allowed voltage deviations to use. Possible options
        are:

        * 'mv_lv'
          This is the default. The allowed voltage deviation for nodes in the
          MV topology is the same as for nodes in the LV topology. Further load
          and feed-in case are not distinguished.
        * 'mv'
          Use this to handle allowed voltage deviations in the MV and LV
          topology differently. Here, load and feed-in case are differentiated
          as well.

    Returns
    -------
    :obj:`dict`
        Dictionary with representative of :class:`~.network.grids.MVGrid` as
        key and a :pandas:`pandas.DataFrame<DataFrame>` with its critical
        nodes, sorted descending by voltage deviation, as value.
        Index of the dataframe are all buses with voltage issues.
        Columns are 'v_mag_pu' containing the maximum voltage deviation as
        float and 'time_index' containing the corresponding time step the
        voltage issue occured in as :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Voltage issues are determined based on allowed voltage deviations defined
    in the config file 'config_grid_expansion' in section
    'grid_expansion_allowed_voltage_deviations'.

    """

    crit_nodes = {}

    buses = edisgo_obj.topology.mv_grid.buses_df.index

    v_dev_allowed_upper, v_dev_allowed_lower = mv_allowed_deviations(
        edisgo_obj, voltage_levels
    )

    crit_nodes_grid = _voltage_deviation(
        edisgo_obj,
        buses,
        v_dev_allowed_upper,
        v_dev_allowed_lower
    )

    if not crit_nodes_grid.empty:
        crit_nodes[
            repr(edisgo_obj.topology.mv_grid)
        ] = crit_nodes_grid.sort_values(by=["v_mag_pu"], ascending=False)
        logger.debug(
            "==> {} node(s) in MV topology has/have voltage issues.".format(
                crit_nodes[repr(edisgo_obj.topology.mv_grid)].shape[0]
            )
        )
    else:
        logger.debug("==> No voltage issues in MV topology.")

    return crit_nodes


def lv_voltage_deviation(edisgo_obj, mode=None, voltage_levels="mv_lv"):
    """
    Checks for voltage stability issues in LV grids.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    mode : None or :obj:`str`
        If None voltage at all nodes in LV topology is checked. If mode is set
        to 'stations' only voltage at bus bar is checked.
    voltage_levels : :obj:`str`
        Specifies which allowed voltage deviations to use. Possible options
        are:

        * 'mv_lv'
          This is the default. The allowed voltage deviation for nodes in the
          MV topology is the same as for nodes in the LV topology. Further load
          and feed-in case are not distinguished.
        * 'lv'
          Use this to handle allowed voltage deviations in the MV and LV
          topology differently. Here, load and feed-in case are differentiated.

    Returns
    -------
    :obj:`dict`
        Dictionary with representative of :class:`~.network.grids.LVGrid` as
        key and a :pandas:`pandas.DataFrame<DataFrame>` with its critical
        buses, sorted descending by voltage deviation, as value.
        Index of the dataframe are all nodes with voltage issues.
        Columns are 'v_mag_pu' containing the maximum voltage deviation as
        float and 'time_index' containing the corresponding time step the
        voltage issue occured in as :pandas:`pandas.Timestamp<Timestamp>`.

    Notes
    -----
    Voltage issues are determined based on allowed voltage deviations defined
    in the config file 'config_grid_expansion' in section
    'grid_expansion_allowed_voltage_deviations'.

    """

    crit_nodes = {}

    v_dev_allowed_per_case = {}
    if voltage_levels == "mv_lv":
        offset = edisgo_obj.config[
            "grid_expansion_allowed_voltage_deviations"
        ]["hv_mv_trafo_offset"]
        control_deviation = edisgo_obj.config[
            "grid_expansion_allowed_voltage_deviations"
        ]["hv_mv_trafo_control_deviation"]
        v_dev_allowed_per_case["feedin_case_upper"] = (
            1
            + offset
            + control_deviation
            + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "mv_lv_feedin_case_max_v_deviation"
            ]
        )
        v_dev_allowed_per_case["load_case_lower"] = (
            1
            + offset
            - control_deviation
            - edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "mv_lv_load_case_max_v_deviation"
            ]
        )

        v_dev_allowed_per_case["feedin_case_lower"] = edisgo_obj.config[
            "grid_expansion_allowed_voltage_deviations"
        ]["feedin_case_lower"]
        v_dev_allowed_per_case["load_case_upper"] = edisgo_obj.config[
            "grid_expansion_allowed_voltage_deviations"
        ]["load_case_upper"]

        v_dev_allowed_upper = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
            lambda _: v_dev_allowed_per_case["{}_upper".format(_)]
        )
        v_dev_allowed_lower = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
            lambda _: v_dev_allowed_per_case["{}_lower".format(_)]
        )
    elif voltage_levels == "lv":
        pass
    else:
        raise ValueError(
            "Specified mode {} is not a valid option.".format(voltage_levels)
        )

    for lv_grid in edisgo_obj.topology.mv_grid.lv_grids:

        if mode:
            if mode == "stations":
                nodes = lv_grid.station.index
            else:
                raise ValueError(
                    "{} is not a valid option for input variable 'mode' in "
                    "function lv_voltage_deviation. Try 'stations' or "
                    "None".format(mode)
                )
        else:
            nodes = lv_grid.buses_df.index

        if voltage_levels == "lv":
            if mode == "stations":
                # get voltage at primary side to calculate upper bound for
                # feed-in case and lower bound for load case
                bus_station_primary = lv_grid.transformers_df.iloc[0].bus0
                v_lv_station_primary = edisgo_obj.results.v_res.loc[
                                       :, bus_station_primary]
                timeindex = v_lv_station_primary.index
                v_dev_allowed_per_case["feedin_case_upper"] = (
                    v_lv_station_primary
                    + edisgo_obj.config[
                        "grid_expansion_allowed_voltage_deviations"
                    ]["mv_lv_station_feedin_case_max_v_deviation"]
                )
                v_dev_allowed_per_case["load_case_lower"] = (
                    v_lv_station_primary
                    - edisgo_obj.config[
                        "grid_expansion_allowed_voltage_deviations"
                    ]["mv_lv_station_load_case_max_v_deviation"]
                )
            else:
                # get voltage at secondary side to calculate upper bound for
                # feed-in case and lower bound for load case
                v_lv_station_secondary = edisgo_obj.results.v_res.loc[
                                         :, lv_grid.station]
                timeindex = v_lv_station_secondary.index
                v_dev_allowed_per_case["feedin_case_upper"] = (
                    v_lv_station_secondary
                    + edisgo_obj.config[
                        "grid_expansion_allowed_voltage_deviations"
                    ]["lv_feedin_case_max_v_deviation"]
                )
                v_dev_allowed_per_case["load_case_lower"] = (
                    v_lv_station_secondary
                    - edisgo_obj.config[
                        "grid_expansion_allowed_voltage_deviations"
                    ]["lv_load_case_max_v_deviation"]
                )
            v_dev_allowed_per_case["feedin_case_lower"] = pd.Series(
                0.9, index=timeindex
            )
            v_dev_allowed_per_case["load_case_upper"] = pd.Series(
                1.1, index=timeindex
            )
            # maximum allowed voltage deviation in each time step
            v_dev_allowed_upper = []
            v_dev_allowed_lower = []
            for t in timeindex:
                case = edisgo_obj.timeseries.timesteps_load_feedin_case.loc[t]
                v_dev_allowed_upper.append(
                    v_dev_allowed_per_case["{}_upper".format(case)].loc[t]
                )
                v_dev_allowed_lower.append(
                    v_dev_allowed_per_case["{}_lower".format(case)].loc[t]
                )
            v_dev_allowed_upper = pd.Series(
                v_dev_allowed_upper, index=timeindex
            )
            v_dev_allowed_lower = pd.Series(
                v_dev_allowed_lower, index=timeindex
            )

        crit_nodes_grid = _voltage_deviation(
            edisgo_obj,
            nodes,
            v_dev_allowed_upper,
            v_dev_allowed_lower
        )

        if not crit_nodes_grid.empty:
            crit_nodes[lv_grid] = crit_nodes_grid.sort_values(
                by=["v_mag_pu"], ascending=False
            )

    if crit_nodes:
        if mode == "stations":
            logger.debug(
                "==> {} LV station(s) has/have voltage issues.".format(
                    len(crit_nodes)
                )
            )
        else:
            logger.debug(
                "==> {} LV topology(s) has/have voltage issues.".format(
                    len(crit_nodes)
                )
            )
    else:
        if mode == "stations":
            logger.debug("==> No voltage issues in LV stations.")
        else:
            logger.debug("==> No voltage issues in LV grids.")

    return crit_nodes


def voltage_diff(
    edisgo_obj, buses, v_dev_allowed_upper, v_dev_allowed_lower
):
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


def _voltage_deviation(
    edisgo_obj, buses, v_dev_allowed_upper, v_dev_allowed_lower
):
    """
    Function to detect voltage issues at buses.

    The function returns the highest voltage deviation from allowed lower
    or upper voltage limit in p.u. for all buses with voltage issues.

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
    pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with deviations from allowed lower or upper voltage level
        sorted descending from highest to lowest voltage deviation
        (it is not distinguished between over- or undervoltage).
        Columns of the dataframe are 'v_mag_pu' containing the maximum absolute
        voltage deviation as float and 'time_index' containing the
        corresponding time step the voltage issue occured in as
        :pandas:`pandas.Timestamp<Timestamp>`. Index of the dataframe are the
        names of all buses with voltage issues.

    """

    def _append_crit_buses(df):
        return pd.DataFrame(
            {
                "v_mag_pu": df.max(axis=1).values,
                "time_index": df.idxmax(axis=1).values,
            },
            index=df.index,
        )

    crit_buses_grid = pd.DataFrame()

    voltage_diff_uv, voltage_diff_ov = voltage_diff(
        edisgo_obj,
        buses,
        v_dev_allowed_upper,
        v_dev_allowed_lower
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
            by=["v_mag_pu"],
            ascending=False,
            inplace=True)

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
