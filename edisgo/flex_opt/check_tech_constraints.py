import logging

import numpy as np
import pandas as pd

from edisgo.network.grids import LVGrid, MVGrid

logger = logging.getLogger(__name__)


def mv_line_max_relative_overload(edisgo_obj, n_minus_one=False):
    """
    Returns time step and value of most severe overloading of lines in MV network.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    n_minus_one : bool
        Determines which allowed load factors to use (see :py:attr:`~lines_allowed_load`
        for more information). Currently, n-1 security cannot be handled correctly,
        wherefore the case where this parameter is set to True will lead to an error
        being raised.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV lines, their maximum relative over-loading
        in p.u. (maximum calculated apparent power over allowed apparent power) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occurred in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    Notes
    -----
    Line over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_lines = _line_max_relative_overload(
        edisgo_obj, voltage_level="mv", n_minus_one=n_minus_one
    )

    if not crit_lines.empty:
        logger.debug(
            "==> {} line(s) in MV network has/have load issues.".format(
                crit_lines.shape[0]
            )
        )
    else:
        logger.debug("==> No line load issues in MV network.")

    return crit_lines


def lv_line_max_relative_overload(edisgo_obj, n_minus_one=False, lv_grid_id=None):
    """
    Returns time step and value of most severe overloading of lines in LV networks.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    n_minus_one : bool
        Determines which allowed load factors to use (see :py:attr:`~lines_allowed_load`
        for more information). Currently, n-1 security cannot be handled correctly,
        wherefore the case where this parameter is set to True will lead to an error
        being raised.
    lv_grid_id : str or int or None
        If None, checks overloading for all LV lines. Otherwise, only lines in given
        LV grid are checked. Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded LV lines, their maximum relative over-loading
        in p.u. (maximum calculated apparent power over allowed apparent power) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occurred in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    Notes
    -----
    Line over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """

    crit_lines = _line_max_relative_overload(
        edisgo_obj, voltage_level="lv", n_minus_one=n_minus_one, lv_grid_id=lv_grid_id
    )

    if not crit_lines.empty:
        logger.debug(
            "==> {} line(s) in LV networks has/have load issues.".format(
                crit_lines.shape[0]
            )
        )
    else:
        logger.debug("==> No line load issues in LV networks.")

    return crit_lines


def _line_max_relative_overload(
    edisgo_obj, voltage_level, n_minus_one=False, lv_grid_id=None
):
    """
    Returns time step and value of most severe overloading of lines.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_level : str
        Voltage level, over-loading is checked for. Possible options are 'mv' or 'lv'.
    n_minus_one : bool
        Determines which allowed load factors to use. See :py:attr:`~lines_allowed_load`
        for more information.
    lv_grid_id : str or int or None
        This parameter is only used in case `voltage_level` is "lv".
        If None, checks overloading for all LV lines. Otherwise, only lines in given
        LV grid are checked. Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded lines, their maximum relative over-loading
        in p.u. (maximum calculated apparent power over allowed apparent power) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occurred in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    """
    if edisgo_obj.results.i_res.empty:
        raise Exception(
            "No power flow results to check over-load for. Please perform "
            "power flow analysis first."
        )

    # get lines in voltage level
    mv_grid = edisgo_obj.topology.mv_grid
    if voltage_level == "lv":
        if lv_grid_id is None:
            lines = edisgo_obj.topology.lines_df[
                ~edisgo_obj.topology.lines_df.index.isin(mv_grid.lines_df.index)
            ].index
        else:
            lv_grid = edisgo_obj.topology.get_lv_grid(lv_grid_id)
            lines = lv_grid.lines_df.index
    elif voltage_level == "mv":
        lines = mv_grid.lines_df.index
    else:
        raise ValueError(
            "{} is not a valid option for input variable 'voltage_level'. "
            "Try 'mv' or 'lv'.".format(voltage_level)
        )

    # calculate relative line load and keep maximum over-load of each line
    relative_i_res = lines_relative_load(edisgo_obj, lines, n_minus_one=n_minus_one)

    crit_lines_relative_load = relative_i_res[relative_i_res > 1].max().dropna()
    if len(crit_lines_relative_load) > 0:
        crit_lines = pd.concat(
            [
                crit_lines_relative_load,
                relative_i_res.idxmax()[crit_lines_relative_load.index],
            ],
            axis=1,
            keys=["max_rel_overload", "time_index"],
            sort=True,
        )
        crit_lines.loc[:, "voltage_level"] = voltage_level
    else:
        crit_lines = pd.DataFrame(dtype=float)

    return crit_lines


def lines_allowed_load(edisgo_obj, lines=None, n_minus_one=False):
    """
    Returns allowed loading of specified lines per time step in MVA.

    Allowed loading is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    lines : list(str)
        List of line names to get allowed loading for. Per default
        allowed loading is returned for all lines in the network. Default: None.
    n_minus_one : bool
        Determines which allowed load factors to use. In case it is set to False,
        allowed load factors defined in the config file 'config_grid_expansion' in
        section 'grid_expansion_load_factors' are used. This is the default.
        In case it is set to True, allowed load factors defined in the config file
        'config_grid_expansion' in section 'grid_expansion_load_factors_n_minus_one'
        are used. This case is currently not implemented.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the maximum allowed apparent power per line and time step
        in MVA. Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Columns are line names as in index of
        :attr:`~.network.topology.Topology.loads_df`.

    """
    allowed_load_lv = _lines_allowed_load_voltage_level(
        edisgo_obj, voltage_level="lv", n_minus_one=n_minus_one
    )
    allowed_load_mv = _lines_allowed_load_voltage_level(
        edisgo_obj, voltage_level="mv", n_minus_one=n_minus_one
    )
    allowed_load = pd.concat([allowed_load_lv, allowed_load_mv], axis=1)
    if lines is None:
        return allowed_load
    else:
        return allowed_load.loc[:, lines]


def _lines_allowed_load_voltage_level(edisgo_obj, voltage_level, n_minus_one=False):
    """
    Returns allowed loading per line in the specified voltage level in MVA.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_level : str
        Grid level, allowed line load is returned for. Possible options are
        "mv" or "lv".
    n_minus_one : bool
        Determines which allowed load factors to use. In case it is set to False,
        allowed load factors defined in the config file 'config_grid_expansion' in
        section 'grid_expansion_load_factors' are used. This is the default.
        In case it is set to True, allowed load factors defined in the config file
        'config_grid_expansion' in section 'grid_expansion_load_factors_n_minus_one'
        are used. This case is currently not implemented.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the maximum allowed apparent power per line and time step
        in MVA. Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Columns are line names as in index of
        :attr:`~.network.topology.Topology.loads_df` of all lines in the specified
        voltage level.

    """
    # get lines in voltage level
    mv_grid = edisgo_obj.topology.mv_grid
    if voltage_level == "lv":
        lines_df = edisgo_obj.topology.lines_df[
            ~edisgo_obj.topology.lines_df.index.isin(mv_grid.lines_df.index)
        ]
    elif voltage_level == "mv":
        lines_df = mv_grid.lines_df
    else:
        raise ValueError(
            "{} is not a valid option for input variable 'voltage_level' in "
            "function lines_allowed_load_voltage_level. Try 'mv' or "
            "'lv'.".format(voltage_level)
        )

    allowed_load_per_case = {}

    # get allowed loads per case
    if n_minus_one is True:
        raise NotImplementedError("n-1 security can currently not be checked.")
        # # handle lines in cycles differently from lines in stubs
        # for case in ["feed-in_case", "load_case"]:
        #     if (
        #         edisgo_obj.config["grid_expansion_load_factors_n_minus_one"][
        #             f"{voltage_level}_{case}_line"
        #         ]
        #         != 1.0
        #     ):
        #
        #         buses_in_cycles = list(
        #             set(itertools.chain.from_iterable(edisgo_obj.topology.rings))
        #         )
        #
        #         # Find lines in cycles
        #         lines_in_cycles = list(
        #             lines_df.loc[
        #                 lines_df[["bus0", "bus1"]].isin(buses_in_cycles).all(axis=1)
        #             ].index.values
        #         )
        #         lines_radial_feeders = list(
        #             lines_df.loc[~lines_df.index.isin(lines_in_cycles)].index.values
        #         )
        #
        #         # lines in cycles have to be n-1 secure
        #         allowed_load_per_case[case] = (
        #             lines_df.loc[lines_in_cycles].s_nom
        #             * edisgo_obj.config["grid_expansion_load_factors_n_minus_one"][
        #                 f"{voltage_level}_{case}_line"
        #             ]
        #         )
        #
        #         # lines in radial feeders are not n-1 secure anyway
        #         allowed_load_per_case[case] = pd.concat(
        #             [
        #                 allowed_load_per_case[case],
        #                 lines_df.loc[lines_radial_feeders].s_nom,
        #             ]
        #         )
    else:
        for case in ["feed-in_case", "load_case"]:
            allowed_load_per_case[case] = (
                lines_df.s_nom
                * edisgo_obj.config["grid_expansion_load_factors"][
                    f"{voltage_level}_{case}_line"
                ]
            )

    return edisgo_obj.timeseries.timesteps_load_feedin_case.loc[
        edisgo_obj.results.s_res.index
    ].apply(lambda _: allowed_load_per_case[_])


def lines_relative_load(edisgo_obj, lines=None, n_minus_one=False):
    """
    Returns relative line load.

    The relative line load is here defined as the apparent power over a line, obtained
    from power flow analysis, divided by the allowed load of a line, which is the
    nominal apparent power times a security factor (see :py:attr:`~lines_allowed_load`
    for more information).

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    lines : list(str) or None
        List of line names to get relative loading for. Per default relative loading
        is returned for all lines included in the power flow analysis. Default: None.
    n_minus_one : bool
        Determines which allowed load factors to use. See :py:attr:`~lines_allowed_load`
        for more information.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the relative loading per line and time step
        in p.u.. Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Columns are line names as in index of
        :attr:`~.network.topology.Topology.loads_df`.

    """
    if lines is None:
        lines = edisgo_obj.results.s_res.columns.drop(
            edisgo_obj.topology.transformers_df.index, errors="ignore"
        )

    # get allowed loading
    allowed_loading = lines_allowed_load(edisgo_obj, lines, n_minus_one=n_minus_one)

    # get line load from power flow analysis
    loading = edisgo_obj.results.s_res.loc[:, lines]

    return loading / allowed_loading


def hv_mv_station_max_overload(edisgo_obj):
    """
    Checks for over-loading of HV/MV station.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        In case there are no over-loading problems returns an empty dataframe.
        In case of over-loading problems the dataframe contains the name of the
        over-loaded station (grid's name with the extension '_station') in the index.
        Columns are 's_missing' containing the missing apparent power at maximal
        over-loading in MVA as float, 'time_index' containing the corresponding time
        step the over-loading occurred in as :pandas:`pandas.Timestamp<Timestamp>`,
        and 'grid' containing the grid object as :class:`~.network.grids.MVGrid`.

    Notes
    -----
    Over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """
    crit_stations = _station_max_overload(edisgo_obj, edisgo_obj.topology.mv_grid)
    if not crit_stations.empty:
        logger.debug("==> HV/MV station has load issues.")
    else:
        logger.debug("==> No HV/MV station load issues.")

    return crit_stations


def mv_lv_station_max_overload(edisgo_obj, lv_grid_id=None):
    """
    Checks for over-loading of MV/LV stations.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    lv_grid_id : str or int or None
        If None, checks overloading for all MV/LV stations. Otherwise, only station
        in given LV grid is checked. Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        In case there are no over-loading problems returns an empty dataframe.
        In case of over-loading problems the dataframe contains the name of the
        over-loaded station (grid's name with the extension '_station') in the index.
        Columns are 's_missing' containing the missing apparent power at maximal
        over-loading in MVA as float, 'time_index' containing the corresponding time
        step the over-loading occurred in as :pandas:`pandas.Timestamp<Timestamp>`,
        and 'grid' containing the grid object as :class:`~.network.grids.LVGrid`.

    Notes
    -----
    Over-load is determined based on allowed load factors for feed-in and
    load cases that are defined in the config file 'config_grid_expansion' in
    section 'grid_expansion_load_factors'.

    """
    crit_stations = pd.DataFrame(dtype=float)

    if lv_grid_id is not None:
        lv_grids = [edisgo_obj.topology.get_lv_grid(lv_grid_id)]
    else:
        lv_grids = list(edisgo_obj.topology.lv_grids)
    for lv_grid in lv_grids:
        crit_stations = pd.concat(
            [
                crit_stations,
                _station_max_overload(edisgo_obj, lv_grid),
            ]
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


def _station_max_overload(edisgo_obj, grid):
    """
    Checks for over-loading of stations.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        In case there are no over-loading problems returns an empty dataframe.
        In case of over-loading problems the dataframe contains the name of the
        over-loaded station (grid's name with the extension '_station') in the index.
        Columns are 's_missing' containing the missing apparent power at maximal
        over-loading in MVA as float, 'time_index' containing the corresponding time
        step the over-loading occurred in as :pandas:`pandas.Timestamp<Timestamp>`,
        and 'grid' containing the grid object as :class:`~.network.grids.Grid`.

    """
    if isinstance(grid, LVGrid):
        voltage_level = "lv"
    elif isinstance(grid, MVGrid):
        voltage_level = "mv"
    else:
        raise ValueError("Inserted grid is invalid.")

    # get apparent power over station from power flow analysis
    s_station_pfa = _station_load(edisgo_obj, grid)

    # get maximum allowed apparent power of station in each time step
    s_station_allowed = _station_allowed_load(edisgo_obj, grid)

    # calculate residual apparent power (if negative, station is over-loaded)
    s_res = s_station_allowed - s_station_pfa
    s_res = s_res[s_res < 0]

    if not s_res.dropna().empty:
        load_factor = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
            lambda _: edisgo_obj.config["grid_expansion_load_factors"][
                f"{voltage_level}_{_}_transformer"
            ]
        )

        # calculate the greatest apparent power missing (residual apparent power is
        # divided by the load factor to account for load factors smaller than
        # one, which lead to a higher needed additional capacity)
        s_missing = (s_res.iloc[:, 0] / load_factor).dropna()
        return pd.DataFrame(
            {
                "s_missing": abs(s_missing.min()),
                "time_index": s_missing.idxmin(),
                "grid": grid,
            },
            index=[grid.station_name],
        )

    else:
        return pd.DataFrame(dtype=float)


def _station_load(edisgo_obj, grid):
    """
    Returns loading of stations per time step from power flow analysis in MVA.

    In case of HV/MV transformers, which are not included in power flow analysis,
    loading is determined using slack results.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing loading of grid's station to the overlying voltage level
        per time step in MVA.
        Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Column name is grid's name with the extension '_station'.

    """
    # get apparent power over station from power flow analysis
    if isinstance(grid, LVGrid):
        return pd.DataFrame(
            {
                grid.station_name: edisgo_obj.results.s_res.loc[
                    :, grid.transformers_df.index
                ].sum(axis=1)
            }
        )
    elif isinstance(grid, MVGrid):
        # ensure that power flow was conducted for MV as slack could also be at MV/LV
        # station's secondary side
        mv_lines = edisgo_obj.topology.mv_grid.lines_df.index
        if not any(mv_lines.isin(edisgo_obj.results.i_res.columns)):
            raise ValueError(
                "MV was not included in power flow analysis, wherefore load "
                "of HV/MV station cannot be calculated."
            )
        return pd.DataFrame(
            {
                grid.station_name: np.hypot(
                    edisgo_obj.results.pfa_slack.p,
                    edisgo_obj.results.pfa_slack.q,
                )
            }
        )
    else:
        raise ValueError("Inserted grid is invalid.")


def _station_allowed_load(edisgo_obj, grid):
    """
    Returns allowed loading of grid's station to the overlying voltage level per time
    step in MVA.

    Allowed loading considers allowed load factors in heavy load flow case ('load case')
    and reverse power flow case ('feed-in case') that are defined in the config file
    'config_grid_expansion' in section 'grid_expansion_load_factors'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.LVGrid` or :class:`~.network.grids.MVGrid`
        Grid to get allowed station loading for.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the maximum allowed apparent power over the grid's
        transformers to the overlying voltage level per time step in MVA.
        Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Column name is grid's name with the extension '_station'.

    """
    # get grid's voltage level and transformers to the overlying voltage level
    if isinstance(grid, LVGrid):
        voltage_level = "lv"
        transformers_df = grid.transformers_df
    elif isinstance(grid, MVGrid):
        voltage_level = "mv"
        transformers_df = edisgo_obj.topology.transformers_hvmv_df
    else:
        raise ValueError("Inserted grid is invalid.")

    # get maximum allowed apparent power of station in each time step
    s_station = sum(transformers_df.s_nom)
    load_factor = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
        lambda _: edisgo_obj.config["grid_expansion_load_factors"][
            f"{voltage_level}_{_}_transformer"
        ]
    )

    return pd.DataFrame(
        {grid.station_name: s_station * load_factor}, index=load_factor.index
    )


def stations_allowed_load(edisgo_obj, grids=None):
    """
    Returns allowed loading of specified grids stations to the overlying voltage level
    per time step in MVA.

    Allowed loading considers allowed load factors in heavy load flow case ('load case')
    and reverse power flow case ('feed-in case') that are defined in the config file
    'config_grid_expansion' in section 'grid_expansion_load_factors'.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grids : list(:class:`~.network.grids.Grid`)
        List of MV and LV grids to get allowed station loading for. Per default
        allowed loading is returned for all stations in the network. Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the maximum allowed apparent power over the grid's
        transformers to the overlying voltage level per time step in MVA.
        Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Column names are the respective grid's name with the extension '_station'.

    """
    if grids is None:
        grids = edisgo_obj.topology.grids

    allowed_loading = pd.DataFrame()
    for grid in grids:
        allowed_loading = pd.concat(
            [allowed_loading, _station_allowed_load(edisgo_obj, grid)], axis=1
        )
    return allowed_loading


def stations_relative_load(edisgo_obj, grids=None):
    """
    Returns relative loading of specified grids stations to the overlying voltage level
    per time step in p.u..

    Stations relative loading is determined by dividing the stations loading (from
    power flow analysis) by the allowed loading (considering allowed load factors in
    heavy load flow case ('load case') and reverse power flow case ('feed-in case')
    from config files).

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grids : list(:class:`~.network.grids.Grid`)
        List of MV and LV grids to get relative station loading for. Per default
        relative loading is returned for all stations in the network that were
        included in the power flow analysis. Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the relative loading of the grid's
        transformers to the overlying voltage level per time step in p.u..
        Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Column names are the respective grid's name with the extension '_station'.

    """
    if grids is None:
        grids = edisgo_obj.topology.grids

    # get allowed loading
    allowed_loading = stations_allowed_load(edisgo_obj, grids)

    # get loading from power flow results
    loading = pd.DataFrame()
    for grid in grids:
        # check that grid was included in power flow analysis
        try:
            loading = pd.concat([loading, _station_load(edisgo_obj, grid)], axis=1)
        except Exception:
            pass

    return loading / allowed_loading.loc[:, loading.columns]


def components_relative_load(edisgo_obj, n_minus_one=False):
    """
    Returns relative loading of all lines and stations included in power flow analysis.

    The component's relative loading is determined by dividing the stations loading
    (from power flow analysis) by the allowed loading (considering allowed load factors
    in heavy load flow case ('load case') and reverse power flow case ('feed-in case')
    from config files).

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    n_minus_one : bool
        Determines which allowed load factors to use. See :py:attr:`~lines_allowed_load`
        for more information.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the relative loading of lines and stations power flow
        results are available for per time step in p.u..
        Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Columns are line names (as in index of
        :attr:`~.network.topology.Topology.loads_df`) and station names (respective
        grid's name with the extension '_station', see
        :attr:`~.network.grids.Grid.station_name`).

    """
    stations_rel_load = stations_relative_load(edisgo_obj)
    lines_rel_load = lines_relative_load(
        edisgo_obj, lines=None, n_minus_one=n_minus_one
    )
    return pd.concat([lines_rel_load, stations_rel_load], axis=1)


def voltage_issues(edisgo_obj, voltage_level, split_voltage_band=True, lv_grid_id=None):
    """
    Gives buses with voltage issues and their maximum voltage deviation in p.u..

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_level : None or str
        Specifies voltage level for which to determine voltage issues. Possible options
        are 'mv' to check voltage deviations at MV buses, 'mv_lv' to check voltage
        deviations at MV-LV stations, and 'lv' to check voltage deviations at LV buses.
        If None voltage deviations in all voltage levels are checked.
    split_voltage_band : bool
        If True the allowed voltage band of +/-10 percent is allocated to the different
        voltage levels MV, MV/LV and LV according to config values set in section
        `grid_expansion_allowed_voltage_deviations`. If False, the same voltage limits
        are used for all voltage levels. Default: True.
    lv_grid_id : str or int or None
        This parameter is only used in case `voltage_level` is "mv_lv" or "lv".
        If None, checks voltage issues for all LV buses. Otherwise, only buses
        in given LV grid are checked. Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with maximum deviations from allowed lower or upper voltage limits
        in p.u. sorted descending from highest to lowest voltage deviation
        (it is not distinguished between over- or undervoltage).
        Columns of the dataframe are 'abs_max_voltage_dev' containing the maximum
        absolute voltage deviation as float, 'time_index' containing the
        corresponding time step the maximum voltage issue occured in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'lv_grid_id' giving the LV grid ID
        the bus is in as integer. Index of the dataframe are the
        names of all buses with voltage issues as in index of
        :attr:`~.network.topology.Topology.buses_df`.

    Notes
    -----
    Voltage issues are determined based on allowed voltage deviations defined
    in the config file 'config_grid_expansion' in section
    'grid_expansion_allowed_voltage_deviations'.

    """

    if voltage_level:
        if voltage_level == "mv_lv":
            if lv_grid_id is None:
                buses = edisgo_obj.topology.transformers_df.bus1.unique()
            else:
                lv_grid = edisgo_obj.topology.get_lv_grid(lv_grid_id)
                buses = lv_grid.transformers_df.bus1.unique()
        elif voltage_level == "lv":
            if lv_grid_id is None:
                buses = edisgo_obj.topology.buses_df.index
                # drop MV buses and buses of stations secondary sides
                station_buses = edisgo_obj.topology.transformers_df.bus1.unique()
                buses = buses.drop(
                    edisgo_obj.topology.mv_grid.buses_df.index.append(
                        pd.Index(station_buses)
                    )
                )
            else:
                lv_grid = edisgo_obj.topology.get_lv_grid(lv_grid_id)
                buses = lv_grid.buses_df.index
                # drop buses of station's secondary side
                station_buses = lv_grid.transformers_df.bus1.unique()
                buses = buses.drop(station_buses)
        elif voltage_level == "mv":
            buses = edisgo_obj.topology.mv_grid.buses_df.index
        else:
            raise ValueError(
                "{} is not a valid option for input variable 'voltage_level' in "
                "function voltage_issue. Possible options are 'mv', 'mv_lv', 'lv', "
                "or None.".format(voltage_level)
            )
    else:
        mv_issues = voltage_issues(
            edisgo_obj, voltage_level="mv", split_voltage_band=split_voltage_band
        )
        mv_lv_issues = voltage_issues(
            edisgo_obj, voltage_level="mv_lv", split_voltage_band=split_voltage_band
        )
        lv_issues = voltage_issues(
            edisgo_obj, voltage_level="lv", split_voltage_band=split_voltage_band
        )
        crit_buses = pd.concat([mv_issues, mv_lv_issues, lv_issues])
        if not crit_buses.empty:
            crit_buses.sort_values(
                by=["abs_max_voltage_dev"], ascending=False, inplace=True
            )
        return crit_buses

    crit_buses = _voltage_issues_helper(edisgo_obj, buses, split_voltage_band)

    # join LV grid information
    if voltage_level == "mv_lv" or voltage_level == "lv":
        crit_buses["lv_grid_id"] = edisgo_obj.topology.buses_df.loc[
            crit_buses.index, "lv_grid_id"
        ]
    else:
        crit_buses["lv_grid_id"] = None

    if not crit_buses.empty:
        if voltage_level == "mv_lv":
            message = "==> {} MV-LV station(s) has/have voltage issues."
        elif voltage_level == "lv":
            message = "==> {} LV bus(es) has/have voltage issues."
        else:
            message = "==> {} bus(es) in MV topology has/have voltage issues."
        logger.debug(message.format(len(crit_buses)))
    else:
        if voltage_level == "mv_lv":
            message = "==> No voltage issues in MV-LV stations."
        elif voltage_level == "lv":
            message = "==> No voltage issues in LV grids."
        else:
            message = "==> No voltage issues in MV topology."
        logger.debug(message)
    return crit_buses


def _voltage_issues_helper(edisgo_obj, buses, split_voltage_band):
    """
    Function to detect voltage issues at buses.

    The function returns the highest voltage deviation from allowed lower
    or upper voltage limit in p.u. for all buses with voltage issues.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    buses : list(str)
        List of buses to check voltage deviation for.
    split_voltage_band : bool
        If True the allowed voltage band of +/-10 percent is allocated to the different
        voltage levels MV, MV/LV and LV according to config values set in section
        `grid_expansion_allowed_voltage_deviations`. If False, the same voltage limits
        are used for all voltage levels.

    Returns
    -------
    pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with maximum deviations from allowed lower or upper voltage limits
        in p.u. sorted descending from highest to lowest voltage deviation
        (it is not distinguished between over- or undervoltage).
        Columns of the dataframe are 'abs_max_voltage_dev' containing the maximum
        absolute voltage deviation as float and 'time_index' containing the
        corresponding time step the maximum voltage issue occured in as
        :pandas:`pandas.Timestamp<Timestamp>`. Index of the dataframe are the
        names of all buses with voltage issues as in index of
        :attr:`~.network.topology.Topology.buses_df`.

    """
    crit_buses = pd.DataFrame(dtype=float)
    # get voltage deviations
    voltage_dev = voltage_deviation_from_allowed_voltage_limits(
        edisgo_obj, buses=buses, split_voltage_band=split_voltage_band
    )
    # drop buses without voltage issues
    voltage_dev = voltage_dev[voltage_dev != 0].dropna(how="all", axis=1).abs()
    # determine absolute maximum voltage deviation and time step it occurs
    crit_buses["abs_max_voltage_dev"] = voltage_dev.max()
    crit_buses["time_index"] = voltage_dev.idxmax()
    # sort descending by maximum voltage deviation
    if not crit_buses.empty:
        crit_buses.sort_values(
            by=["abs_max_voltage_dev"], ascending=False, inplace=True
        )
    return crit_buses


def allowed_voltage_limits(edisgo_obj, buses=None, split_voltage_band=True):
    """
    Calculates allowed upper and lower voltage limits.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    buses : list(str)
        List of bus names to get allowed voltage limits for. Per default
        allowed voltage limits are returned for all buses in the network. Default: None.
    split_voltage_band : bool
        If True the allowed voltage band of +/-10 percent is allocated to the different
        voltage levels MV, MV/LV and LV according to config values set in section
        `grid_expansion_allowed_voltage_deviations`. If False, the same voltage limits
        are used for all voltage levels. Default: True.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the maximum allowed apparent power per line and time step
        in MVA. Index of the dataframe are all time steps power flow analysis
        was conducted for of type :pandas:`pandas.Timestamp<Timestamp>`.
        Columns are bus names as in index of
        :attr:`~.network.topology.Topology.buses_df`.

    """
    if buses is None:
        buses = edisgo_obj.results.v_res.columns

    if split_voltage_band:
        # MV limits
        mv_buses = edisgo_obj.topology.mv_grid.buses_df.index
        mv_upper, mv_lower = _mv_allowed_voltage_limits(edisgo_obj)
        mv_upper = pd.DataFrame(
            mv_upper, columns=mv_buses, index=edisgo_obj.results.v_res.index
        )
        mv_lower = pd.DataFrame(
            mv_lower, columns=mv_buses, index=edisgo_obj.results.v_res.index
        )

        # station limits
        stations_upper, stations_lower = _lv_allowed_voltage_limits(
            edisgo_obj, mode="stations"
        )

        # LV limits
        lv_upper, lv_lower = _lv_allowed_voltage_limits(edisgo_obj, mode=None)

        # concat results and select relevant buses
        upper = pd.concat([mv_upper, stations_upper, lv_upper], axis=1)
        lower = pd.concat([mv_lower, stations_lower, lv_lower], axis=1)

        # check if allowed voltage limits could be determined for all specified buses
        allowed_buses = upper.columns
        buses_not_incl = list(set(buses) - set(allowed_buses))
        if buses_not_incl:
            logger.warning(
                f"Allowed voltage limits cannot be determined for all given buses as "
                f"voltage information from power flow analysis is needed to calculate "
                f"allowed voltage for the MV/LV and LV level but the buses were not "
                f"included in the power flow analysis. "
                f"This concerns the following buses: {buses_not_incl}."
            )
            buses = list(set(buses) - set(buses_not_incl))

        return upper.loc[:, buses], lower.loc[:, buses]
    else:
        upper = pd.DataFrame(1.1, columns=buses, index=edisgo_obj.results.v_res.index)
        lower = pd.DataFrame(0.9, columns=buses, index=edisgo_obj.results.v_res.index)
        return upper, lower


def _mv_allowed_voltage_limits(edisgo_obj):
    """
    Calculates allowed lower and upper voltage limits for MV nodes in p.u..

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    (float, float)
        Lower and upper voltage limit for MV nodes.

    """
    # get values from config
    offset = edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
        "hv_mv_trafo_offset"
    ]
    control_deviation = edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
        "hv_mv_trafo_control_deviation"
    ]

    upper_limit = (
        1
        + offset
        - control_deviation
        + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "mv_max_v_rise"
        ]
    )
    lower_limit = (
        1
        + offset
        + control_deviation
        - edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "mv_max_v_drop"
        ]
    )

    return upper_limit, lower_limit


def _lv_allowed_voltage_limits(edisgo_obj, lv_grids=None, mode=None):
    """
    Calculates allowed lower and upper voltage limits for either buses or transformers
    in given LV grids.

    Voltage limits are determined relative to the station's secondary side, in case
    limits are determined for buses in the LV grid (default), or relative to the
    station's primary side, in case limits are determined for transformers.

    Limits can only be determined for grids included in power flow analysis.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    lv_grids : list(:class:`~.network.grids.LVGrid`) or None
        LV grids to get voltage limits for. If None, limits for all LV grids
        included in last power flow analysis are returned.
    mode : None or str
        If None, voltage limits for buses in the LV network are returned. In
        that case the reference bus is the LV stations' secondary side.
        If mode is set to 'stations', voltage limits for stations' secondary
        side (LV bus bar) are returned; the reference bus is the stations'
        primary side.

    Returns
    -------
    (:pandas:`pandas.DataFrame<DataFrame>`, :pandas:`pandas.DataFrame<DataFrame>`)
        Dataframe containing the allowed lower and upper voltage limits in p.u..
        Index of the dataframe are all time steps power flow was last conducted
        for of type :pandas:`pandas.Timestamp<Timestamp>`. Columns are bus names as in
        index of :attr:`~.network.topology.Topology.buses_df` for all buses power flow
        results are available. If mode is 'stations' columns contain bus names
        of the stations secondary sides.

    """
    if lv_grids is None:
        lv_grids = list(edisgo_obj.topology.mv_grid.lv_grids)

    upper_limits_df = pd.DataFrame()
    lower_limits_df = pd.DataFrame()
    voltages_pfa = edisgo_obj.results.v_res
    buses_in_pfa = voltages_pfa.columns

    if mode == "stations":
        config_string = "mv_lv_station"

        # get base voltage (voltage at primary side) for each station
        voltage_base = pd.DataFrame()
        for grid in lv_grids:
            transformers_df = grid.transformers_df
            primary_side = transformers_df.iloc[0].bus0
            secondary_side = transformers_df.iloc[0].bus1
            if primary_side in buses_in_pfa:
                voltage_base[secondary_side] = voltages_pfa.loc[:, primary_side]

        upper_limits_df = (
            voltage_base
            + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "{}_max_v_rise".format(config_string)
            ]
        )
        lower_limits_df = (
            voltage_base
            - edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "{}_max_v_drop".format(config_string)
            ]
        )
    else:
        config_string = "lv"

        # get all secondary sides and buses in grids
        buses_dict = {}
        secondary_sides_dict = {}
        for grid in lv_grids:
            secondary_side = grid.station.index[0]
            if secondary_side in buses_in_pfa:
                secondary_sides_dict[grid] = secondary_side
                buses_dict[grid.station.index[0]] = grid.buses_df.index.drop(
                    grid.station.index[0]
                )
        secondary_sides = pd.Series(secondary_sides_dict)

        voltage_base = voltages_pfa.loc[:, secondary_sides.values]

        upper_limits_df_tmp = (
            voltage_base
            + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "{}_max_v_rise".format(config_string)
            ]
        )
        lower_limits_df_tmp = (
            voltage_base
            - edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "{}_max_v_drop".format(config_string)
            ]
        )

        # rename columns to secondary side
        for colname, values in upper_limits_df_tmp.items():
            tmp = pd.DataFrame(
                data=np.tile(values, (len(buses_dict[colname]), 1)).T,
                columns=buses_dict[colname],
                index=values.index,
            )
            upper_limits_df = pd.concat([upper_limits_df, tmp], axis=1)
        for colname, values in lower_limits_df_tmp.items():
            tmp = pd.DataFrame(
                data=np.tile(values, (len(buses_dict[colname]), 1)).T,
                columns=buses_dict[colname],
                index=values.index,
            )
            lower_limits_df = pd.concat([lower_limits_df, tmp], axis=1)

    return upper_limits_df, lower_limits_df


def voltage_deviation_from_allowed_voltage_limits(
    edisgo_obj, buses=None, split_voltage_band=True
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
    buses : list(str) or None
        List of buses to check voltage deviation for. Per default voltage deviation
        is returned for all buses included in the power flow analysis. Default: None.
    split_voltage_band : bool
        If True the allowed voltage band of +/-10 percent is allocated to the different
        voltage levels MV, MV/LV and LV according to config values set in section
        `grid_expansion_allowed_voltage_deviations`. If False, the same voltage limits
        are used for all voltage levels. Default: True.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with deviations from allowed lower voltage level in p.u.. Positive
        values signify an overvoltage whereas negative values signify an undervoltage.
        Zero values signify that the voltage is within the allowed limits.
        Index of the dataframe are all time steps power flow analysis was conducted for
        of type :pandas:`pandas.Timestamp<Timestamp>`. Columns are bus names as in index
        of :attr:`~.network.topology.Topology.buses_df`.

    """
    if buses is None:
        buses = edisgo_obj.results.v_res.columns

    # get allowed voltage deviations
    v_dev_allowed_upper, v_dev_allowed_lower = allowed_voltage_limits(
        edisgo_obj, buses=buses, split_voltage_band=split_voltage_band
    )

    # get voltages from power flow analysis
    v_mag_pu_pfa = edisgo_obj.results.v_res.loc[:, buses]

    # make all entries without voltage issues NaN values
    overvoltage = v_mag_pu_pfa[v_mag_pu_pfa > v_dev_allowed_upper]
    undervoltage = v_mag_pu_pfa[v_mag_pu_pfa < v_dev_allowed_lower]

    # determine deviation from allowed voltage limits for times with voltage issues
    # overvoltage deviations are positive, undervoltage deviations negative
    overvoltage_dev = overvoltage - v_dev_allowed_upper
    undervoltage_dev = undervoltage - v_dev_allowed_lower

    # combine overvoltage and undervoltage issues and set NaN values to zero
    voltage_dev = overvoltage_dev.fillna(0) + undervoltage_dev.fillna(0)

    return voltage_dev
