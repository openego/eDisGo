import logging

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from edisgo.flex_opt import check_tech_constraints

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
