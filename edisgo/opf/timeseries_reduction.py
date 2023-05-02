import logging
import os

from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from edisgo.flex_opt import check_tech_constraints
from edisgo.flex_opt.costs import line_expansion_costs
from edisgo.tools.tools import assign_feeder

logger = logging.getLogger(__name__)


def _scored_critical_loading(edisgo_obj):

    # Get current relative to allowed current
    relative_s_res = check_tech_constraints.lines_relative_load(
        edisgo_obj, lines=edisgo_obj.topology.mv_grid.lines_df.index
    )

    # Get lines that have violations
    crit_lines_score = relative_s_res[relative_s_res > 1]

    # Remove time steps with no violations
    crit_lines_score = crit_lines_score.dropna(how="all", axis=0)

    # Cumulate violations over all lines per time step
    crit_lines_score = crit_lines_score.sum(axis=1)

    return crit_lines_score.sort_values(ascending=False)


def _scored_most_critical_loading(edisgo_obj):
    """
    Method to get time steps where at least one component
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


def _scored_most_critical_loading_time_interval(edisgo_obj, window_days):
    """
    Get the time steps with the most critical overloadings for flexibility
    optimization.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    window_days : int
        Amount of continuous days that violation is determined for. Default: 7

    Returns
    --------
    `pandas.DataFrame`
        Contains time intervals and first time step of these intervals with worst
        overloadings for given timeframe. Also contains information of how many
        lines and transformers have had their worst overloading within the considered
        timeframes.
    """

    # Get current relative to allowed current
    relative_i_res = check_tech_constraints.components_relative_load(edisgo_obj)

    # Get lines that have violations and replace nan values with 0
    crit_lines_score = relative_i_res[relative_i_res > 1].fillna(0)
    max_per_line = crit_lines_score.max()
    # weight line violations with expansion costs
    costs_lines = (
        line_expansion_costs(edisgo_obj).drop(columns="voltage_level").sum(axis=1)
    )
    costs_trafos_lv = pd.Series(
        index=[
            str(lv_grid) + "_station"
            for lv_grid in list(edisgo_obj.topology.mv_grid.lv_grids)
        ],
        data=edisgo_obj.config._data["costs_transformers"]["lv"],
    )
    costs_trafos_mv = pd.Series(
        index=["MVGrid_" + str(edisgo_obj.topology.id) + "_station"],
        data=edisgo_obj.config._data["costs_transformers"]["mv"],
    )
    costs = pd.concat([costs_lines, costs_trafos_lv, costs_trafos_mv])

    crit_lines_cost = crit_lines_score * costs
    # Get most "expensive" time intervall over all components
    crit_timesteps = (
        crit_lines_cost.rolling(window=int(window_days * 24), closed="right")
        .max()
        .sum(axis=1)
    )
    # time intervall starts at 4am on every considered day
    crit_timesteps = (
        crit_timesteps.iloc[int(window_days * 24) - 1 :]
        .iloc[5::24]
        .sort_values(ascending=False)
    )
    timesteps = crit_timesteps.index - pd.DateOffset(hours=int(window_days * 24))
    time_intervals = [
        pd.date_range(start=timestep, periods=int(window_days * 24) + 1, freq="h")
        for timestep in timesteps
    ]
    time_intervals_df = pd.DataFrame(
        index=range(len(time_intervals)), columns=["OL_ts", "OL_t1", "OL_max"]
    )
    time_intervals_df["OL_t1"] = timesteps
    for i in range(len(time_intervals)):
        time_intervals_df["OL_ts"][i] = time_intervals[i]
    lines_no_max = crit_lines_score.columns.values
    total_lines = len(lines_no_max)
    # check if worst overloading of every line is included in worst three time intervals
    for i in range(len(time_intervals)):
        max_per_lin_ti = crit_lines_score.loc[time_intervals[i]].max()
        time_intervals_df["OL_max"][i] = (
            len(
                np.intersect1d(
                    lines_no_max,
                    max_per_lin_ti[max_per_lin_ti >= max_per_line * 0.95].index.values,
                )
            )
            / total_lines
        )
        lines_no_max = np.intersect1d(
            lines_no_max,
            max_per_lin_ti[max_per_lin_ti < max_per_line * 0.95].index.values,
        )

        if i == 2:
            if len(lines_no_max) > 0:
                logger.warning(
                    "Highest overloading of following lines does not lie within the "
                    "overall worst three time intervals: " + str(lines_no_max)
                )

    return time_intervals_df


def _scored_critical_overvoltage(edisgo_obj):

    voltage_dev = check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(
        edisgo_obj,
        buses=edisgo_obj.topology.mv_grid.buses_df.index,
    )

    # Get score for nodes that are over the allowed deviations
    voltage_dev_ov = (
        voltage_dev[voltage_dev > 0.0].dropna(axis=1, how="all").sum(axis=1)
    )
    return voltage_dev_ov.sort_values(ascending=False)


def _scored_most_critical_voltage_issues(edisgo_obj):
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


def _scored_most_critical_voltage_issues_time_interval(edisgo_obj, window_days):
    """
    Get the time steps with the most critical voltage violations for flexibilities
    optimization.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    window_days : int
        Amount of continuous days that violation is determined for. Default: 7

    Returns
    --------
    `pandas.DataFrame`
        Contains time intervals and first time step of these intervals with worst
        voltage violations for given timeframe. Also contains information of how many
        lines and transformers have had their worst voltage violation within the
        considered timeframes.
    """
    voltage_diff = check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(
        edisgo_obj
    ).fillna(0)

    # Get score for nodes that are over or under the allowed deviations
    voltage_diff = voltage_diff.abs()[voltage_diff.abs() > 0]
    max_per_bus = voltage_diff.max().fillna(0)

    assign_feeder(edisgo_obj, mode="mv_feeder")

    # determine costs per feeder
    costs_lines = (
        line_expansion_costs(edisgo_obj).drop(columns="voltage_level").sum(axis=1)
    )
    costs_trafos_lv = pd.Series(
        index=[
            lv_grid.station.index[0] for lv_grid in edisgo_obj.topology.mv_grid.lv_grids
        ],
        data=edisgo_obj.config._data["costs_transformers"]["lv"],
    )
    costs_trafos_mv = pd.Series(
        index=[edisgo_obj.topology.mv_grid.station.index[0]],
        data=edisgo_obj.config._data["costs_transformers"]["mv"],
    )
    costs = pd.concat([costs_lines, costs_trafos_lv, costs_trafos_mv])

    feeder_lines = edisgo_obj.topology.lines_df.mv_feeder
    feeder_trafos_lv = pd.Series(
        index=[
            lv_grid.station.index[0] for lv_grid in edisgo_obj.topology.mv_grid.lv_grids
        ],
        data=[
            lv_grid.station.mv_feeder[0]
            for lv_grid in edisgo_obj.topology.mv_grid.lv_grids
        ],
    )
    feeder_trafos_mv = pd.Series(
        index=[edisgo_obj.topology.mv_grid.station.index[0]],
        data=[edisgo_obj.topology.mv_grid.station.mv_feeder[0]],
    )
    feeder = pd.concat([feeder_lines, feeder_trafos_lv, feeder_trafos_mv])
    costs_per_feeder = (
        pd.concat([costs.rename("costs"), feeder.rename("feeder")], axis=1)
        .groupby(by="feeder")[["costs"]]
        .sum()
    )

    # check vor every feeder if any of the buses within violate the allowed voltage
    # deviation
    feeder_buses = edisgo_obj.topology.buses_df.mv_feeder
    columns = [
        feeder_buses.loc[voltage_diff.columns[i]]
        for i in range(len(voltage_diff.columns))
    ]
    voltage_diff_copy = deepcopy(voltage_diff).fillna(0)
    voltage_diff.columns = columns
    voltage_diff_feeder = (
        voltage_diff.transpose().reset_index().groupby(by="index").sum().transpose()
    )
    voltage_diff_feeder[voltage_diff_feeder != 0] = 1

    # weigth feeder voltage violation with costs per feeder
    voltage_diff_feeder = voltage_diff_feeder * costs_per_feeder.squeeze()
    # Todo: should there be different ones for over and undervoltage?
    # Get most "expensive" time intervall over all feeders
    crit_timesteps = (
        voltage_diff_feeder.rolling(window=int(window_days * 24), closed="right")
        .max()
        .sum(axis=1)
    )
    # time intervall starts at 4am on every considered day
    crit_timesteps = (
        crit_timesteps.iloc[int(window_days * 24) - 1 :]
        .iloc[5::24]
        .sort_values(ascending=False)
    )
    timesteps = crit_timesteps.index - pd.DateOffset(hours=int(window_days * 24))
    time_intervals = [
        pd.date_range(start=timestep, periods=int(window_days * 24) + 1, freq="h")
        for timestep in timesteps
    ]
    time_intervals_df = pd.DataFrame(
        index=range(len(time_intervals)), columns=["V_ts", "V_t1", "V_max"]
    )
    time_intervals_df["V_t1"] = timesteps
    for i in range(len(time_intervals)):
        time_intervals_df["V_ts"][i] = time_intervals[i]

    buses_no_max = max_per_bus.index.values
    total_buses = len(buses_no_max)

    # check if worst voltage deviation of every bus is included in worst three time
    # intervals
    for i in range(len(time_intervals)):
        max_per_bus_ti = voltage_diff_copy.loc[time_intervals[i]].max()
        time_intervals_df["V_max"][i] = (
            len(
                np.intersect1d(
                    buses_no_max,
                    max_per_bus_ti[max_per_bus_ti >= max_per_bus * 0.95].index.values,
                )
            )
            / total_buses
        )
        buses_no_max = np.intersect1d(
            buses_no_max,
            max_per_bus_ti[max_per_bus_ti < max_per_bus * 0.95].index.values,
        )
        if i == 2:
            if len(buses_no_max) > 0:
                logger.warning(
                    "Highest voltage deviation of following buses does not lie within "
                    "the overall worst three time intervals: " + str(buses_no_max)
                )

    return time_intervals_df


def get_steps_reinforcement(
    edisgo_obj, num_steps_loading=None, num_steps_voltage=None, percentage=1.0
):
    """
    Get the time steps with the most critical violations for curtailment
    optimization.
    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    num_steps_loading: int
        The number of most critical overloading events to select, if None percentage
        is used
    num_steps_voltage: int
        The number of most critical voltage issues to select, if None percentage is used
    percentage : float
        The percentage of most critical time steps to select
    Returns
    --------
    `pandas.DatetimeIndex`
        the reduced time index for modeling curtailment
    """
    # Run power flow if not available
    if edisgo_obj.results.i_res is None or edisgo_obj.results.i_res.empty:
        logger.debug("Running initial power flow")
        edisgo_obj.analyze(raise_not_converged=False)  # Todo: raise warning?

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
    steps = steps.append(
        voltage_scores[:num_steps_voltage].index
    )  # Todo: Can this cause duplicated?

    if len(steps) == 0:
        logger.warning("No critical steps detected. No network expansion required.")

    return pd.DatetimeIndex(steps.unique())


def get_steps_curtailment(edisgo_obj, percentage=0.5):
    """
    Get the time steps with the most critical violations for curtailment
    optimization.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    percentage : float
        The percentage of most critical time steps to select

    Returns
    --------
    `pandas.DatetimeIndex`
        the reduced time index for modeling curtailment

    """
    # Run power flow if not available
    if edisgo_obj.results.i_res is None:
        logger.debug("Running initial power flow")
        edisgo_obj.analyze(mode="mv")

    # Select most critical steps based on current violations
    current_scores = _scored_critical_loading(edisgo_obj)
    num_steps_current = int(len(current_scores) * percentage)
    steps = current_scores[:num_steps_current].index

    # Select most critical steps based on voltage violations
    voltage_scores = _scored_critical_overvoltage(edisgo_obj)
    num_steps_voltage = int(len(voltage_scores) * percentage)
    steps = steps.append(voltage_scores[:num_steps_voltage].index)

    # Always add worst cases
    steps = steps.append(get_steps_storage(edisgo_obj, window=0))

    if len(steps) == 0:
        logger.warning("No critical steps detected. No network expansion required.")

    # Strip duplicates
    return pd.DatetimeIndex(steps.unique())


def get_steps_storage(edisgo_obj, window=5):
    """
    Get the most critical time steps from series for storage problems.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    window : int
        The additional hours to include before and after each critical time
        step.

    Returns
    -------
    `pandas.DatetimeIndex`
        the reduced time index for modeling storage

    """
    # Run power flow if not available
    if edisgo_obj.results.i_res is None:
        logger.debug("Running initial power flow")
        edisgo_obj.analyze(mode="mv")

    # Get periods with voltage violations
    crit_nodes = check_tech_constraints.voltage_issues(
        edisgo_obj, voltage_level="mv", split_voltage_band=True
    )
    # Get periods with current violations
    crit_lines = check_tech_constraints.mv_line_max_relative_overload(edisgo_obj)

    crit_periods = crit_nodes["time_index"].append(crit_lines["time_index"]).unique()

    reduced = []
    window_period = pd.Timedelta(window, unit="h")
    for step in crit_periods:
        reduced.extend(
            pd.date_range(
                start=step - window_period, end=step + window_period, freq="h"
            )
        )

    # strip duplicates
    reduced = set(reduced)

    if len(reduced) == 0:
        logger.warning("No critical steps detected. No network expansion required.")

    return pd.DatetimeIndex(reduced)


def get_steps_flex_opf(
    edisgo_obj,
    num_ti=None,
    percentage=0.1,
    window_days=7,
    save_steps=False,
    path="",
):
    """
    Get the time steps with the most critical violations for curtailment
    optimization.
    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object
    num_ti: int
        The number of most critical line loading and voltage issues to select. If None
        percentage is used. Default: None
    percentage : float
        The percentage of most critical time intervals to select. Default: 0.1
    window_days : int
        Amount of continuous days that violation is determined for. Default: 7
    save_steps : bool
        If set to True, dataframe with time intervals is saved to csv file.
        Default: False
    path:
        Directory the csv file is saved to. Per default it takes the current
        working directory.

    Returns
    --------
    `pandas.DataFrame`
        Contains time intervals and first time step of these intervals with worst grid
        violations for given timeframe. Also contains information of how many lines and
        transformers have had their worst violation within the considered timeframes.
    """
    # Run power flow if not available
    if edisgo_obj.results.i_res is None or edisgo_obj.results.i_res.empty:
        logger.debug("Running initial power flow")
        edisgo_obj.analyze(raise_not_converged=False)  # Todo: raise warning?

    # Select most critical time intervalls based on current violations
    loading_scores = _scored_most_critical_loading_time_interval(
        edisgo_obj, window_days
    )
    if num_ti is None:
        num_ti = int(np.ceil(len(loading_scores) * percentage))
    else:
        if num_ti > len(loading_scores):
            logger.info(
                f"The number of time intervals with highest overloading "
                f"({len(loading_scores)}) is lower than the defined number of "
                f"loading time intervals ({num_ti}). Therefore, only "
                f"{len(loading_scores)} time intervals are exported."
            )
            num_ti = len(loading_scores)
    steps = loading_scores.iloc[:num_ti]

    # Select most critical steps based on voltage violations
    voltage_scores = _scored_most_critical_voltage_issues_time_interval(
        edisgo_obj, window_days
    )
    if num_ti is None:
        num_ti = int(np.ceil(len(voltage_scores) * percentage))
    else:
        if num_ti > len(voltage_scores):
            logger.info(
                f"The number of time steps with highest voltage issues "
                f"({len(voltage_scores)}) is lower than the defined number of "
                f"voltage time steps ({num_ti}). Therefore, only "
                f"{len(voltage_scores)} time steps are exported."
            )
            num_ti = len(voltage_scores)
    steps = pd.concat([steps, voltage_scores.iloc[:num_ti]], axis=1)

    if len(steps) == 0:
        logger.warning("No critical steps detected. No network expansion required.")

    if save_steps:
        abs_path = os.path.abspath(path)
        steps.to_csv(
            os.path.join(
                abs_path,
                str(edisgo_obj.topology.id) + "_t" + str(window_days * 24 + 1) + ".csv",
            )
        )
    return steps


def get_linked_steps(cluster_params, num_steps=24, keep_steps=[]):
    """
    Use provided data to identify representative time steps and create mapping
    Dict that can be passed to optimization

    Parameters
    -----------
    cluster_params : :pandas:`pandas.DataFrame<DataFrame>`
        Time series containing the parameters to be considered for distance
        between points.
    num_steps : int
        The number of representative time steps to be selected.
    keep_steps : Iterable of the same type as cluster_params.index
        Time steps to retain with full resolution, regardless of
        clustering result.

    Returns
    -------
    dict
        Dictionary where each represented time step is a key and its
        representative time step is a value.

    """

    # From all values, find the subvector with the smallest SSD to a given
    # cluster center and return its index
    def get_representative(center, values):
        temp = (values - center) ** 2
        temp = temp.sum(axis=1)
        return temp.argmin()

    # Make values comparable and run k-Means
    sc = StandardScaler()
    X = sc.fit_transform(cluster_params.values)
    km = KMeans(n_clusters=num_steps).fit(X)

    # k-Means returns synthetic points which do not exist in the original time series.
    # We need to link to existing steps, so we pick the point that is closest
    # to each cluster center as a cluster representative instead
    representatives = []
    for c in km.cluster_centers_:
        r = get_representative(c, X)
        representatives.append(r)
    representatives = np.array(representatives)

    # Create list with numerical values of steps to be ignored
    ignore = [cluster_params.index.get_loc(i) for i in keep_steps]
    ignore = list(dict.fromkeys(ignore))

    linked_steps = {}
    for step, cluster_id in enumerate(km.labels_):
        if step in ignore:
            continue
        # current step was not identified as representative
        if not np.isin(representatives, step).any():
            # find representative and link to it.
            # Also add offset for one-based indexing
            linked_steps[step + 1] = representatives[cluster_id] + 1

    return linked_steps


def distribute_overlying_grid_timeseries(edisgo_obj):
    """
    Distributes overlying grid timeseries of flexibilities for temporal complexity
    reduction.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object

    Returns
    --------
    :class:`~.EDisGo`
        Contains adjusted timeseries for flexibilities.
    """
    edisgo_copy = deepcopy(edisgo_obj)
    if not edisgo_copy.overlying_grid.electromobility_active_power.empty:
        cp_loads = edisgo_obj.topology.loads_df.index[
            edisgo_obj.topology.loads_df.type == "charging_point"
        ]
        # scale flexibility band upper power timeseries
        scaling_df = (
            edisgo_obj.electromobility.flexibility_bands["upper_power"].transpose()
            / edisgo_obj.electromobility.flexibility_bands["upper_power"].sum(axis=1)
        ).transpose()
        edisgo_copy.timeseries._loads_active_power.loc[:, cp_loads] = (
            scaling_df.transpose()
            * edisgo_obj.overlying_grid.electromobility_active_power
        ).transpose()
    if not edisgo_copy.overlying_grid.storage_units_active_power.empty:
        scaling_factor = (
            edisgo_obj.topology.storage_units_df.p_nom
            / edisgo_obj.topology.storage_units_df.p_nom.sum()
        )
        scaling_df = pd.DataFrame(
            columns=scaling_factor.index,
            index=edisgo_copy.timeseries.timeindex,
            data=pd.concat(
                [scaling_factor] * len(edisgo_copy.timeseries.timeindex), axis=1
            )
            .transpose()
            .values,
        )
        edisgo_copy.timeseries._storage_units_active_power = (
            scaling_df.transpose()
            * edisgo_obj.overlying_grid.storage_units_active_power
        ).transpose()
    if not edisgo_copy.overlying_grid.heat_pump_central_active_power.empty:
        hp_district = edisgo_obj.topology.loads_df[
            (edisgo_obj.topology.loads_df.type == "heat_pump")
            & (edisgo_obj.topology.loads_df.sector == "district_heating")
        ]
        scaling_factor = hp_district.p_set / hp_district.p_set.sum()
        scaling_df = pd.DataFrame(
            columns=scaling_factor.index,
            index=edisgo_copy.timeseries.timeindex,
            data=pd.concat(
                [scaling_factor] * len(edisgo_copy.timeseries.timeindex), axis=1
            )
            .transpose()
            .values,
        )
        edisgo_copy.timeseries._loads_active_power.loc[:, hp_district.index] = (
            scaling_df.transpose()
            * edisgo_obj.overlying_grid.heat_pump_central_active_power.sum(axis=1)[0]
        ).transpose()

    if not edisgo_copy.overlying_grid.heat_pump_decentral_active_power.empty:
        hp_individual = edisgo_obj.topology.loads_df.index[
            (edisgo_obj.topology.loads_df.type == "heat_pump")
            & (edisgo_obj.topology.loads_df.sector == "individual_heating")
        ]
        # scale with heat pump upper power
        scaling_factor = (
            edisgo_obj.topology.loads_df.p_set.loc[hp_individual]
            / edisgo_obj.topology.loads_df.p_set.loc[hp_individual].sum()
        )
        scaling_df = pd.DataFrame(
            columns=scaling_factor.index,
            index=edisgo_copy.timeseries.timeindex,
            data=pd.concat(
                [scaling_factor] * len(edisgo_copy.timeseries.timeindex), axis=1
            )
            .transpose()
            .values,
        )
        edisgo_copy.timeseries._loads_active_power.loc[:, hp_individual] = (
            scaling_df.transpose()
            * edisgo_obj.overlying_grid.heat_pump_decentral_active_power
        ).transpose()
    if not edisgo_copy.overlying_grid.dsm_active_power.empty:
        try:
            dsm_loads = edisgo_copy.dsm.p_max.columns
            scaling_df_max = (
                edisgo_copy.dsm.p_max.transpose() / edisgo_copy.dsm.p_max.sum(axis=1)
            ).transpose()
            scaling_df_min = (
                edisgo_copy.dsm.p_min.transpose() / edisgo_copy.dsm.p_min.sum(axis=1)
            ).transpose()
            edisgo_copy.timeseries._loads_active_power.loc[:, dsm_loads] = (
                edisgo_obj.timeseries._loads_active_power.loc[:, dsm_loads]
                + (
                    scaling_df_min.transpose()
                    * edisgo_obj.overlying_grid.dsm_active_power.clip(upper=0)
                ).transpose()
                + (
                    scaling_df_max.transpose()
                    * edisgo_obj.overlying_grid.dsm_active_power.clip(lower=0)
                ).transpose()
            )
        except AttributeError:
            logger.warning(
                "'EDisGo' object has no attribute 'dsm'. DSM timeseries from"
                " overlying grid can not be distributed."
            )
    if not edisgo_copy.overlying_grid.renewables_curtailment.empty:
        gens = edisgo_obj.topology.generators_df[
            (edisgo_obj.topology.generators_df.type == "solar")
            | (edisgo_obj.topology.generators_df.type == "wind")
        ].index
        gen_per_ts = edisgo_obj.timeseries.generators_active_power.loc[:, gens].sum(
            axis=1
        )
        scaling_factor = (
            (
                edisgo_obj.timeseries.generators_active_power.loc[:, gens].transpose()
                * 1
                / gen_per_ts
            )
            .transpose()
            .fillna(0)
        )
        curtailment = (
            scaling_factor.transpose()
            * edisgo_obj.overlying_grid.renewables_curtailment
        ).transpose()
        edisgo_copy.timeseries._generators_active_power.loc[:, gens] = (
            edisgo_obj.timeseries.generators_active_power.loc[:, gens] - curtailment
        )

    edisgo_copy.set_time_series_reactive_power_control()
    return edisgo_copy
