import logging
import pandas as pd
import numpy as np

from edisgo.flex_opt import check_tech_constraints

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def _scored_critical_current(edisgo_obj, grid):
    # Get allowed current per line per time step
    i_lines_allowed = check_tech_constraints.lines_allowed_load(
        edisgo_obj, "mv"
    )
    i_lines_pfa = edisgo_obj.results.i_res[grid.lines_df.index]

    # Get current relative to allowed current
    relative_i_res = i_lines_pfa / i_lines_allowed

    # Get lines that have violations
    crit_lines_score = relative_i_res[relative_i_res > 1]

    # Remove time steps with no violations
    crit_lines_score = crit_lines_score.dropna(how="all", axis=0)

    # Cumulate violations over all lines per time step
    crit_lines_score = crit_lines_score.sum(axis=1)

    return crit_lines_score.sort_values(ascending=False)


def _scored_critical_overvoltage(edisgo_obj, grid):
    nodes = grid.buses_df.index

    # Get allowed deviations per time step
    (
        v_dev_allowed_upper,
        v_dev_allowed_lower,
    ) = check_tech_constraints._mv_allowed_voltage_limits(
        edisgo_obj, voltage_levels="mv"
    )
    _, voltage_diff_ov = check_tech_constraints.voltage_diff(
        edisgo_obj,
        nodes,
        v_dev_allowed_upper,
        v_dev_allowed_lower
    )

    # Get score for nodes that are over or under the allowed deviations
    voltage_diff_ov = (
        voltage_diff_ov[voltage_diff_ov > 0]
        .dropna(axis=1, how="all")
        .sum(axis=0)
    )
    return voltage_diff_ov.sort_values(ascending=False)


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

    grid = edisgo_obj.topology.mv_grid

    # Select most critical steps based on current violations
    current_scores = _scored_critical_current(edisgo_obj, grid)
    num_steps_current = int(len(current_scores) * percentage)
    steps = current_scores[:num_steps_current].index.tolist()

    # Select most critical steps based on voltage violations
    voltage_scores = _scored_critical_overvoltage(edisgo_obj, grid)
    num_steps_voltage = int(len(voltage_scores) * percentage)
    steps.extend(voltage_scores[:num_steps_voltage].index.tolist())

    # Always add worst cases
    steps.extend(get_steps_storage(edisgo_obj, window=0).tolist())

    if len(steps) == 0:
        logger.warning(
            "No critical steps detected. No network expansion required."
        )

    # Strip duplicates
    steps = list(dict.fromkeys(steps))

    return pd.DatetimeIndex(steps)


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

    crit_periods = []

    # Get periods with voltage violations
    crit_nodes = check_tech_constraints.mv_voltage_deviation(
        edisgo_obj, voltage_levels="mv"
    )
    for v in crit_nodes.values():
        nodes = pd.DataFrame(v)
        if "time_index" in nodes:
            for step in nodes["time_index"]:
                if not step in crit_periods:
                    crit_periods.append(step)

    # Get periods with current violations
    crit_lines = check_tech_constraints.mv_line_load(edisgo_obj)
    if "time_index" in crit_lines:
        for step in crit_lines["time_index"]:
            if not step in crit_periods:
                crit_periods.append(step)

    reduced = []
    window_period = pd.Timedelta(window, unit="h")
    for step in crit_periods:
        reduced.extend(
            pd.date_range(
                start=step - window_period, end=step + window_period, freq="h"
            )
        )

    # strip duplicates
    reduced = list(dict.fromkeys(reduced))

    if len(reduced) == 0:
        logger.warning(
            "No critical steps detected. No network expansion required."
        )

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
