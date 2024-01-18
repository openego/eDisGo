from __future__ import annotations

import logging
import os

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from edisgo.flex_opt import check_tech_constraints
from edisgo.flex_opt.costs import line_expansion_costs

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def _scored_most_critical_loading(edisgo_obj: EDisGo) -> pd.Series:
    """
    Method to get time steps where at least one component shows its highest overloading.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    --------
    :pandas:`pandas.Series<Series>`
        Series with time index and corresponding sum of maximum relative overloadings
        of lines and transformers. The series only contains time steps, where at least
        one component is maximally overloaded, and is sorted descending by the
        sum of maximum relative overloadings.

    """

    # Get current relative to allowed current
    relative_i_res = check_tech_constraints.components_relative_load(edisgo_obj)

    # Get lines that have violations
    crit_lines_score = relative_i_res[relative_i_res > 1]

    # Get most critical timesteps per component
    crit_lines_score = (
        (crit_lines_score[crit_lines_score == crit_lines_score.max()])
        .dropna(how="all")
        .dropna(how="all", axis=1)
    )

    # Sort according to highest cumulated relative overloading
    crit_lines_score = (crit_lines_score - 1).sum(axis=1)
    return crit_lines_score.sort_values(ascending=False)


def _scored_most_critical_voltage_issues(edisgo_obj: EDisGo) -> pd.Series:
    """
    Method to get time steps where at least one bus shows its highest deviation from
    allowed voltage boundaries.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    --------
    :pandas:`pandas.Series<Series>`
        Series with time index and corresponding sum of maximum voltage deviations.
        The series only contains time steps, where at least one bus has its highest
        deviation from the allowed voltage limits, and is sorted descending by the
        sum of maximum voltage deviations.

    """
    voltage_diff = check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(
        edisgo_obj
    )

    # Get score for nodes that are over or under the allowed deviations
    voltage_diff = voltage_diff.abs()[voltage_diff.abs() > 0]
    # get only most critical events for component
    # Todo: should there be different ones for over and undervoltage?
    voltage_diff = (
        (voltage_diff[voltage_diff.abs() == voltage_diff.abs().max()])
        .dropna(how="all")
        .dropna(how="all", axis=1)
    )

    voltage_diff = voltage_diff.sum(axis=1)

    return voltage_diff.sort_values(ascending=False)


def _scored_most_critical_loading_time_interval(
    edisgo_obj,
    time_steps_per_time_interval=168,
    time_steps_per_day=24,
    time_step_day_start=0,
    overloading_factor=0.95,
):
    """
    Get time intervals sorted by severity of overloadings.

    The overloading is weighed by the estimated expansion costs of each respective line
    and transformer.
    The length of the time intervals and hour of day at which the time intervals should
    begin can be set through the parameters `time_steps_per_time_interval` and
    `time_step_day_start`.

    This function currently only works for an hourly resolution!

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    time_steps_per_time_interval : int
        Amount of continuous time steps in an interval that violation is determined for.
        Currently, these can only be multiples of 24.
        Default: 168.
    time_steps_per_day : int
        Number of time steps in one day. In case of an hourly resolution this is 24.
        As currently only an hourly resolution is possible, this value should always be
        24.
        Default: 24.
    time_step_day_start : int
        Time step of the day at which each interval should start. If you want it to
        start at midnight, this should be set to 0. Default: 0.
    overloading_factor : float
        Factor at which an overloading of a component is considered to be close enough
        to the highest overloading of that component. This is used to determine the
        number of components that reach their highest overloading in each time interval.
        Per default, it is set to 0.95, which means that if the highest overloading of
        a component is 2, it will be considered maximally overloaded at an overloading
        of higher or equal to 2*0.95.
        Default: 0.95.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Contains time intervals in which grid expansion needs due to overloading of
        lines and transformers are detected. The time intervals are sorted descending
        by the expected cumulated grid expansion costs, so that the time interval with
        the highest expected costs corresponds to index 0. The time steps in the
        respective time interval are given in column "time_steps" and the share
        of components for which the maximum overloading is reached during the time
        interval is given in column "percentage_max_overloaded_components".

    """

    # Get current relative to allowed current
    relative_i_res = check_tech_constraints.components_relative_load(edisgo_obj)

    # Get lines that have violations and replace nan values with 0
    crit_lines_score = relative_i_res[relative_i_res > 1].fillna(0)

    # weight line violations with expansion costs
    costs_lines = (
        line_expansion_costs(edisgo_obj).drop(columns="voltage_level").sum(axis=1)
    )
    costs_trafos_lv = pd.Series(
        index=[
            str(lv_grid) + "_station"
            for lv_grid in list(edisgo_obj.topology.mv_grid.lv_grids)
        ],
        data=edisgo_obj.config["costs_transformers"]["lv"],
    )
    costs_trafos_mv = pd.Series(
        index=["MVGrid_" + str(edisgo_obj.topology.id) + "_station"],
        data=edisgo_obj.config["costs_transformers"]["mv"],
    )
    costs = pd.concat([costs_lines, costs_trafos_lv, costs_trafos_mv])
    crit_lines_cost = crit_lines_score * costs

    # Get highest overloading in each window for each component and sum it up
    crit_timesteps = (
        crit_lines_cost.rolling(
            window=int(time_steps_per_time_interval), closed="right"
        )
        .max()
        .sum(axis=1)
    )
    # select each nth time window to only consider windows starting at a certain time
    # of day and sort time intervals in descending order
    # ToDo: To make function work for frequencies other than hourly, the following
    #  needs to be adapted to index based on time index instead of iloc
    crit_timesteps = (
        crit_timesteps.iloc[int(time_steps_per_time_interval) - 1 :]
        .iloc[time_step_day_start + 1 :: time_steps_per_day]
        .sort_values(ascending=False)
    )
    # move time index as rolling gives the end of the time interval, but we want the
    # beginning
    timesteps = crit_timesteps.index - pd.DateOffset(
        hours=int(time_steps_per_time_interval)
    )
    time_intervals = [
        pd.date_range(
            start=timestep, periods=int(time_steps_per_time_interval), freq="h"
        )
        for timestep in timesteps
    ]

    # make dataframe with time steps in each time interval and the percentage of
    # components that reach their maximum overloading
    time_intervals_df = pd.DataFrame(
        index=range(len(time_intervals)),
        columns=["time_steps", "percentage_max_overloaded_components"],
    )
    time_intervals_df["time_steps"] = time_intervals
    lines_no_max = crit_lines_score.columns.values
    total_lines = len(lines_no_max)
    max_per_line = crit_lines_score.max()
    for i in range(len(time_intervals)):
        # check if worst overloading of every line is included in time interval
        max_per_line_ti = crit_lines_score.loc[time_intervals[i]].max()
        time_intervals_df["percentage_max_overloaded_components"][i] = (
            len(max_per_line_ti[max_per_line_ti >= max_per_line * overloading_factor])
            / total_lines
        )

    return time_intervals_df


def _scored_most_critical_voltage_issues_time_interval(
    edisgo_obj,
    time_steps_per_time_interval=168,
    time_steps_per_day=24,
    time_step_day_start=0,
    voltage_deviation_factor=0.95,
):
    """
    Get time intervals sorted by severity of voltage issues.

    The voltage issues are weighed by the estimated expansion costs in each respective
    feeder.
    The length of the time intervals and hour of day at which the time intervals should
    begin can be set through the parameters `time_steps_per_time_interval` and
    `time_step_day_start`.

    This function currently only works for an hourly resolution!

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    time_steps_per_time_interval : int
        Amount of continuous time steps in an interval that violation is determined for.
        Currently, these can only be multiples of 24.
        Default: 168.
    time_steps_per_day : int
        Number of time steps in one day. In case of an hourly resolution this is 24.
        As currently only an hourly resolution is possible, this value should always be
        24.
        Default: 24.
    time_step_day_start : int
        Time step of the day at which each interval should start. If you want it to
        start at midnight, this should be set to 0. Default: 0.
    voltage_deviation_factor : float
        Factor at which a voltage deviation at a bus is considered to be close enough
        to the highest voltage deviation at that bus. This is used to determine the
        number of buses that reach their highest voltage deviation in each time
        interval. Per default, it is set to 0.95. This means that if the highest voltage
        deviation at a bus is 0.2, it will be included in the determination of number
        of buses that reach their maximum voltage deviation in a certain time interval
        at a voltage deviation of higher or equal to 0.2*0.95.
        Default: 0.95.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Contains time intervals in which grid expansion needs due to voltage issues
        are detected. The time intervals are sorted descending
        by the expected cumulated grid expansion costs, so that the time interval with
        the highest expected costs corresponds to index 0. The time steps in the
        respective time interval are given in column "time_steps" and the share
        of buses for which the maximum voltage deviation is reached during the time
        interval is given in column "percentage_buses_max_voltage_deviation". Each bus
        is only considered once. That means if its maximum voltage deviation was
        already considered in an earlier time interval, it is not considered again.

    """

    # Get voltage deviation from allowed voltage limits
    voltage_diff = check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(
        edisgo_obj
    )
    voltage_diff = voltage_diff.abs()[voltage_diff.abs() > 0]

    # determine costs per feeder
    lv_station_buses = [
        lv_grid.station.index[0] for lv_grid in edisgo_obj.topology.mv_grid.lv_grids
    ]
    costs_lines = (
        line_expansion_costs(edisgo_obj).drop(columns="voltage_level").sum(axis=1)
    )
    costs_trafos_lv = pd.Series(
        index=lv_station_buses,
        data=edisgo_obj.config._data["costs_transformers"]["lv"],
    )
    costs = pd.concat([costs_lines, costs_trafos_lv])

    # set feeder using MV feeder for MV components and LV feeder for LV components
    edisgo_obj.topology.assign_feeders(mode="grid_feeder")
    # feeders of buses at MV/LV station's secondary sides are set to the name of the
    # station bus to have them as separate feeders
    edisgo_obj.topology.buses_df.loc[lv_station_buses, "grid_feeder"] = lv_station_buses

    feeder_lines = edisgo_obj.topology.lines_df.grid_feeder
    feeder_trafos_lv = pd.Series(
        index=lv_station_buses,
        data=lv_station_buses,
    )
    feeder = pd.concat([feeder_lines, feeder_trafos_lv])
    costs_per_feeder = (
        pd.concat([costs.rename("costs"), feeder.rename("feeder")], axis=1)
        .groupby(by="feeder")[["costs"]]
        .sum()
    )

    # check for every feeder if any of the buses within violate the allowed voltage
    # deviation, by grouping voltage_diff per feeder
    feeder_buses = edisgo_obj.topology.buses_df.grid_feeder
    columns = [feeder_buses.loc[col] for col in voltage_diff.columns]
    voltage_diff_copy = deepcopy(voltage_diff).fillna(0)
    voltage_diff.columns = columns
    voltage_diff_feeder = (
        voltage_diff.transpose().reset_index().groupby(by="index").sum().transpose()
    )
    voltage_diff_feeder[voltage_diff_feeder != 0] = 1

    # weigh feeder voltage violation with costs per feeder
    voltage_diff_feeder = voltage_diff_feeder * costs_per_feeder.squeeze()

    # Get the highest voltage issues in each window for each feeder and sum it up
    crit_timesteps = (
        voltage_diff_feeder.rolling(
            window=int(time_steps_per_time_interval), closed="right"
        )
        .max()
        .sum(axis=1)
    )
    # select each nth time window to only consider windows starting at a certain time
    # of day and sort time intervals in descending order
    # ToDo: To make function work for frequencies other than hourly, the following
    #  needs to be adapted to index based on time index instead of iloc
    crit_timesteps = (
        crit_timesteps.iloc[int(time_steps_per_time_interval) - 1 :]
        .iloc[time_step_day_start + 1 :: time_steps_per_day]
        .sort_values(ascending=False)
    )
    timesteps = crit_timesteps.index - pd.DateOffset(
        hours=int(time_steps_per_time_interval)
    )
    time_intervals = [
        pd.date_range(
            start=timestep, periods=int(time_steps_per_time_interval), freq="h"
        )
        for timestep in timesteps
    ]

    # make dataframe with time steps in each time interval and the percentage of
    # buses that reach their maximum voltage deviation
    time_intervals_df = pd.DataFrame(
        index=range(len(time_intervals)),
        columns=["time_steps", "percentage_buses_max_voltage_deviation"],
    )
    time_intervals_df["time_steps"] = time_intervals

    max_per_bus = voltage_diff_copy.max().fillna(0)
    buses_no_max = max_per_bus.index.values
    total_buses = len(buses_no_max)
    for i in range(len(time_intervals)):
        # check if worst voltage deviation of every bus is included in time interval
        max_per_bus_ti = voltage_diff_copy.loc[time_intervals[i]].max()
        time_intervals_df["percentage_buses_max_voltage_deviation"][i] = (
            len(
                max_per_bus_ti[max_per_bus_ti >= max_per_bus * voltage_deviation_factor]
            )
            / total_buses
        )

    return time_intervals_df


def _troubleshooting_mode(
    edisgo_obj,
    mode=None,
    timesteps=None,
    lv_grid_id=None,
    scale_timeseries=None,
):
    """
    Handles non-convergence issues in power flow by iteratively reducing load and
    feed-in until the power flow converges.

    Load and feed-in is reduced in steps of 10% down to 20% of the original load
    and feed-in. The most critical time intervals / time steps can then be determined
    based on the power flow results with the reduced load and feed-in.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    mode : str or None
        Allows to toggle between power flow analysis for the whole network or just
        the MV or one LV grid. See parameter `mode` in function
        :attr:`~.EDisGo.analyze` for more information.
    timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
        :pandas:`pandas.Timestamp<Timestamp>`
        Timesteps specifies from which time steps to select most critical ones. It
        defaults to None in which case all time steps in
        :attr:`~.network.timeseries.TimeSeries.timeindex` are used.
    lv_grid_id : int or str
        ID (e.g. 1) or name (string representation, e.g. "LVGrid_1") of LV grid
        to analyze in case mode is 'lv'. Default: None.
    scale_timeseries : float or None
        See parameter `scale_timeseries` in function :attr:`~.EDisGo.analyze` for more
        information.

    """
    try:
        logger.debug("Running initial power flow for temporal complexity reduction.")
        edisgo_obj.analyze(
            mode=mode,
            timesteps=timesteps,
            lv_grid_id=lv_grid_id,
            scale_timeseries=scale_timeseries,
        )
    # Exception is used, as non-convergence can also lead to RuntimeError, not only
    # ValueError
    except Exception:
        # if power flow did not converge for all time steps, run again with smaller
        # loading - loading is decreased, until all time steps converge
        logger.warning(
            "When running power flow to determine most critical time intervals, "
            "not all time steps converged. Power flow is run again with reduced "
            "network load."
        )
        if isinstance(scale_timeseries, float):
            iter_start = scale_timeseries - 0.1
        else:
            iter_start = 0.8
        for fraction in np.arange(iter_start, 0.0, step=-0.1):
            try:
                edisgo_obj.analyze(
                    mode=mode,
                    timesteps=timesteps,
                    lv_grid_id=lv_grid_id,
                    scale_timeseries=fraction,
                )
                logger.info(
                    f"Power flow fully converged for a reduction factor "
                    f"of {fraction}."
                )
                break
            except Exception:
                if fraction == 0.1:
                    raise ValueError(
                        f"Power flow did not converge for smallest reduction "
                        f"factor of {fraction}. Most critical time intervals "
                        f"can therefore not be determined."
                    )
                else:
                    logger.info(
                        f"Power flow did not fully converge for a reduction factor "
                        f"of {fraction}."
                    )
    return edisgo_obj


def get_most_critical_time_intervals(
    edisgo_obj,
    num_time_intervals=None,
    percentage=1.0,
    time_steps_per_time_interval=168,
    time_step_day_start=0,
    save_steps=False,
    path="",
    use_troubleshooting_mode=True,
    overloading_factor=0.95,
    voltage_deviation_factor=0.95,
):
    """
    Get time intervals sorted by severity of overloadings as well as voltage issues.

    The overloading and voltage issues are weighed by the estimated expansion costs
    solving the issue would require.
    The length of the time intervals and hour of day at which the time intervals should
    begin can be set through the parameters `time_steps_per_time_interval` and
    `time_step_day_start`.

    This function currently only works for an hourly resolution!

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    num_time_intervals : int
        The number of time intervals of most critical line loading and voltage issues
        to select. If None, `percentage` is used. Default: None.
    percentage : float
        The percentage of most critical time intervals to select. The default is 1.0, in
        which case all most critical time steps are selected.
        Default: 1.0.
    time_steps_per_time_interval : int
        Amount of continuous time steps in an interval that violation is determined for.
        Currently, these can only be multiples of 24.
        Default: 168.
    time_step_day_start : int
        Time step of the day at which each interval should start. If you want it to
        start at midnight, this should be set to 0. Default: 0.
    save_steps : bool
        If set to True, dataframe with time intervals is saved to csv file. The path
        can be specified through parameter `path`.
        Default: False.
    path : str
        Directory the csv file is saved to. Per default, it takes the current
        working directory.
    use_troubleshooting_mode : bool
        If set to True, non-convergence issues in power flow are tried to be handled
        by reducing load and feed-in in steps of 10% down to 20% of the original load
        and feed-in until the power flow converges. The most critical time intervals
        are then determined based on the power flow results with the reduced load and
        feed-in. If False, an error will be raised in case time steps do not converge.
        Default: True.
    overloading_factor : float
        Factor at which an overloading of a component is considered to be close enough
        to the highest overloading of that component. This is used to determine the
        number of components that reach their highest overloading in each time interval.
        Per default, it is set to 0.95, which means that if the highest overloading of
        a component is 2, it will be considered maximally overloaded at an overloading
        of higher or equal to 2*0.95.
        Default: 0.95.
    voltage_deviation_factor : float
        Factor at which a voltage deviation at a bus is considered to be close enough
        to the highest voltage deviation at that bus. This is used to determine the
        number of buses that reach their highest voltage deviation in each time
        interval. Per default, it is set to 0.95. This means that if the highest voltage
        deviation at a bus is 0.2, it will be included in the determination of number
        of buses that reach their maximum voltage deviation in a certain time interval
        at a voltage deviation of higher or equal to 0.2*0.95.
        Default: 0.95.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Contains time intervals in which grid expansion needs due to overloading and
        voltage issues are detected. The time intervals are determined independently
        for overloading and voltage issues and sorted descending by the expected
        cumulated grid expansion costs, so that the time intervals with the highest
        expected costs correspond to index 0.
        In case of overloading, the time steps in the respective time interval are given
        in column "time_steps_overloading" and the share of components for which the
        maximum overloading is reached during the time interval is given in column
        "percentage_max_overloaded_components".
        For voltage issues, the time steps in the respective time interval are given
        in column "time_steps_voltage_issues" and the share of  buses for which the
        maximum voltage deviation is reached during the time interval is given in column
        "percentage_buses_max_voltage_deviation".

    """
    # check frequency of time series data
    timeindex = edisgo_obj.timeseries.timeindex
    timedelta = timeindex[1] - timeindex[0]
    if timedelta != pd.Timedelta("1h"):
        logger.warning(
            "The function 'get_most_critical_time_intervals' can currently only be "
            "applied to time series data in an hourly resolution."
        )

    # Run power flow
    if use_troubleshooting_mode:
        edisgo_obj = _troubleshooting_mode(edisgo_obj)
    else:
        logger.debug("Running initial power flow for temporal complexity reduction.")
        edisgo_obj.analyze()

    # Select most critical time intervals based on current violations
    loading_scores = _scored_most_critical_loading_time_interval(
        edisgo_obj,
        time_steps_per_time_interval,
        time_step_day_start=time_step_day_start,
        overloading_factor=overloading_factor,
    )
    if num_time_intervals is None:
        num_time_intervals = int(np.ceil(len(loading_scores) * percentage))
    else:
        if num_time_intervals > len(loading_scores):
            logger.info(
                f"The number of time intervals with highest overloading "
                f"({len(loading_scores)}) is lower than the defined number of "
                f"loading time intervals ({num_time_intervals}). Therefore, only "
                f"{len(loading_scores)} time intervals are exported."
            )
            num_time_intervals = len(loading_scores)
    steps = loading_scores.iloc[:num_time_intervals]

    # Select most critical steps based on voltage violations
    voltage_scores = _scored_most_critical_voltage_issues_time_interval(
        edisgo_obj,
        time_steps_per_time_interval,
        time_step_day_start=time_step_day_start,
        voltage_deviation_factor=voltage_deviation_factor,
    )
    if num_time_intervals is None:
        num_time_intervals = int(np.ceil(len(voltage_scores) * percentage))
    else:
        if num_time_intervals > len(voltage_scores):
            logger.info(
                f"The number of time steps with highest voltage issues "
                f"({len(voltage_scores)}) is lower than the defined number of "
                f"voltage time steps ({num_time_intervals}). Therefore, only "
                f"{len(voltage_scores)} time steps are exported."
            )
            num_time_intervals = len(voltage_scores)

    # merge time intervals
    steps = pd.merge(
        steps,
        voltage_scores.iloc[:num_time_intervals],
        left_index=True,
        right_index=True,
        suffixes=("_overloading", "_voltage_issues"),
    )
    if len(steps) == 0:
        logger.info("No critical steps detected. No network expansion required.")

    if save_steps:
        abs_path = os.path.abspath(path)
        steps.to_csv(
            os.path.join(
                abs_path,
                f"{edisgo_obj.topology.id}_t_{time_steps_per_time_interval}.csv",
            )
        )
    return steps


def get_most_critical_time_steps(
    edisgo_obj: EDisGo,
    mode=None,
    timesteps=None,
    lv_grid_id=None,
    scale_timeseries=None,
    num_steps_loading=None,
    num_steps_voltage=None,
    percentage: float = 1.0,
    use_troubleshooting_mode=True,
    run_initial_analyze=True,
) -> pd.DatetimeIndex:
    """
    Get the time steps with the most critical overloading and voltage issues.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    mode : str or None
        Allows to toggle between power flow analysis for the whole network or just
        the MV or one LV grid. See parameter `mode` in function
        :attr:`~.EDisGo.analyze` for more information.
    timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
        :pandas:`pandas.Timestamp<Timestamp>`
        Timesteps specifies from which time steps to select most critical ones. It
        defaults to None in which case all time steps in
        :attr:`~.network.timeseries.TimeSeries.timeindex` are used.
    lv_grid_id : int or str
        ID (e.g. 1) or name (string representation, e.g. "LVGrid_1") of LV grid
        to analyze in case mode is 'lv'. Default: None.
    scale_timeseries : float or None
        See parameter `scale_timeseries` in function :attr:`~.EDisGo.analyze` for more
        information.
    num_steps_loading : int
        The number of most critical overloading events to select. If None, `percentage`
        is used. Default: None.
    num_steps_voltage : int
        The number of most critical voltage issues to select. If None, `percentage`
        is used. Default: None.
    percentage : float
        The percentage of most critical time steps to select. The default is 1.0, in
        which case all most critical time steps are selected.
        Default: 1.0.
    use_troubleshooting_mode : bool
        If set to True, non-convergence issues in power flow are tried to be handled
        by reducing load and feed-in in steps of 10% down to 20% of the original load
        and feed-in until the power flow converges. The most critical time steps
        are then determined based on the power flow results with the reduced load and
        feed-in. If False, an error will be raised in case time steps do not converge.
        Default: True.
    run_initial_analyze : bool
        This parameter can be used to specify whether to run an initial analyze to
        determine most critical time steps or to use existing results. If set to False,
        `use_troubleshooting_mode` is ignored. Default: True.

    Returns
    --------
    :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
        Time index with unique time steps where maximum overloading or maximum
        voltage deviation is reached for at least one component respectively bus.

    """
    # Run power flow
    if run_initial_analyze:
        if use_troubleshooting_mode:
            edisgo_obj = _troubleshooting_mode(
                edisgo_obj,
                mode=mode,
                timesteps=timesteps,
                lv_grid_id=lv_grid_id,
                scale_timeseries=scale_timeseries,
            )
        else:
            logger.debug(
                "Running initial power flow for temporal complexity reduction."
            )
            edisgo_obj.analyze(
                mode=mode,
                timesteps=timesteps,
                lv_grid_id=lv_grid_id,
                scale_timeseries=scale_timeseries,
            )

    # Select most critical steps based on current violations
    loading_scores = _scored_most_critical_loading(edisgo_obj)
    if num_steps_loading is None:
        num_steps_loading = int(len(loading_scores) * percentage)
    else:
        if num_steps_loading > len(loading_scores):
            logger.info(
                f"The number of time steps with highest overloading "
                f"({len(loading_scores)}) is lower than the defined number of "
                f"loading time steps ({num_steps_loading}). Therefore, only "
                f"{len(loading_scores)} time steps are exported."
            )
            num_steps_loading = len(loading_scores)
    steps = loading_scores[:num_steps_loading].index

    # Select most critical steps based on voltage violations
    voltage_scores = _scored_most_critical_voltage_issues(edisgo_obj)
    if num_steps_voltage is None:
        num_steps_voltage = int(len(voltage_scores) * percentage)
    else:
        if num_steps_voltage > len(voltage_scores):
            logger.info(
                f"The number of time steps with highest voltage issues "
                f"({len(voltage_scores)}) is lower than the defined number of "
                f"voltage time steps ({num_steps_voltage}). Therefore, only "
                f"{len(voltage_scores)} time steps are exported."
            )
            num_steps_voltage = len(voltage_scores)
    steps = steps.append(voltage_scores[:num_steps_voltage].index)

    if len(steps) == 0:
        logger.warning("No critical steps detected. No network expansion required.")

    return pd.DatetimeIndex(steps.unique())
