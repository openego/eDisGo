import logging
import pandas as pd

from edisgo.flex_opt import check_tech_constraints

logger = logging.getLogger(__name__)


def _scored_critical_current(edisgo_obj, grid):
    # Get allowed current per line per time step
    i_lines_allowed = check_tech_constraints.lines_allowed_load(
        edisgo_obj, grid, 'mv')
    i_lines_pfa = edisgo_obj.results.i_res[grid.lines_df.index]

    # Get current relative to allowed current
    relative_i_res = i_lines_pfa / i_lines_allowed

    # Get lines that have violations
    crit_lines_score = relative_i_res[relative_i_res > 1]

    # Remove time steps with no violations
    crit_lines_score = crit_lines_score.dropna(how='all', axis=0)

    # Cumulate violations over all lines per time step
    crit_lines_score = crit_lines_score.sum(axis=1)

    return crit_lines_score.sort_values(ascending=False)


def _scored_critical_voltage(edisgo_obj, grid):
    nodes = grid.buses_df

    # Get allowed deviations per time step
    v_dev_allowed_upper, v_dev_allowed_lower = check_tech_constraints.mv_allowed_deviations(
        edisgo_obj, voltage_levels='mv')
    voltage_diff_uv, voltage_diff_ov = check_tech_constraints.voltage_diff(
        edisgo_obj, nodes, v_dev_allowed_upper, v_dev_allowed_lower, voltage_level='mv')

    # Get score for nodes that are over or under the allowed deviations
    voltage_diff_uv = voltage_diff_uv[voltage_diff_uv > 0].dropna(
        axis=1, how='all').sum(axis=0)
    voltage_diff_ov = voltage_diff_ov[voltage_diff_ov > 0].dropna(
        axis=1, how='all').sum(axis=0)
    return (voltage_diff_ov + voltage_diff_uv).sort_values(ascending=False)


def get_steps_curtailment(edisgo_obj, percentage=0.5):
    '''
    :param edisgo_obj: The eDisGo API object
    :type name: :class:`~.network.network.EDisGo`
    :param percentage: The percentage of most critical time steps to select
    :type percentage: float
    :returns: `pandas.DatetimeIndex` -- the reduced time index for modeling curtailment
    '''

    # Run power flow if not available
    if edisgo_obj.results.i_res is None:
        logger.debug('Running initial power flow')
        edisgo_obj.analyze(mode='mv')

    grid = edisgo_obj.topology.mv_grid

    # Select most critical steps based on current viaolations
    current_scores = scored_critical_current(edisgo_obj, grid)
    num_steps_current = int(len(current_scores) * percentage)
    steps = current_scores[:num_steps_current].index.tolist()

    # Select most critical steps based on voltage viaolations
    voltage_scores = scored_critical_voltage(edisgo_obj, grid)
    num_steps_voltage = int(len(voltage_scores) * percentage)
    steps.extend(voltage_scores[:num_steps_voltage].index.tolist())

    if len(steps) == 0:
        logger.warning(
            "No critical steps detected. No network expansion required.")

    # Strip duplicates
    steps = list(dict.fromkeys(steps))

    return pd.DatetimeIndex(steps)


def get_steps_storage(edisgo_obj, window=5):
    '''
    :param edisgo_obj: The eDisGo API object
    :type name: :class:`~.network.network.EDisGo`
    :param window: The additional hours to include before and after each critical time step
    :type window: int
    :returns:  `pandas.DatetimeIndex` -- the reduced time index for modeling storage
    '''
    # Run power flow if not available
    if edisgo_obj.results.i_res is None:
        logger.debug('Running initial power flow')
        edisgo_obj.analyze(mode='mv')

    crit_periods = []

    # Get periods with voltage violations
    crit_nodes = check_tech_constraints.mv_voltage_deviation(
        edisgo_obj, voltage_levels='mv')
    for v in crit_nodes.values():
        nodes = pd.DataFrame(v)
        if 'time_index' in nodes:
            for step in nodes['time_index']:
                if not step in crit_periods:
                    crit_periods.append(step)

    # Get periods with current violations
    crit_lines = check_tech_constraints.mv_line_load(edisgo_obj)
    if 'time_index' in crit_lines:
        for step in crit_lines['time_index']:
            if not step in crit_periods:
                crit_periods.append(step)

    reduced = []
    window_period = pd.Timedelta(window, unit='h')
    for step in crit_periods:
        reduced.extend(
            pd.date_range(
                start=step -
                window_period,
                end=step +
                window_period,
                freq='h'))

    # strip duplicates
    reduced = list(dict.fromkeys(reduced))

    if len(reduced) == 0:
        logger.warning(
            "No critical steps detected. No network expansion required.")

    return pd.DatetimeIndex(reduced)
