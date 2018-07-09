import pandas as pd


def select_worstcase_snapshots(network):
    """
    Select two worst-case snapshots from time series

    Two time steps in a time series represent worst-case snapshots. These are

    1. Load case: refers to the point in the time series where the
        (load - generation) achieves its maximum and is greater than 0.
    2. Feed-in case: refers to the point in the time series where the
        (load - generation) achieves its minimum and is smaller than 0.

    These two points are identified based on the generation and load time
    series. In case load or feed-in case don't exist None is returned.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        Network for which worst-case snapshots are identified.

    Returns
    -------
    :obj:`dict`
        Dictionary with keys 'load_case' and 'feedin_case'. Values are
        corresponding worst-case snapshots of type
        :pandas:`pandas.Timestamp<timestamp>` or None.

    """

    grids = [network.mv_grid] + list(network.mv_grid.lv_grids)

    gens = []
    loads = []
    for grid in grids:
        gens.extend(list(grid.graph.nodes_by_attribute('generator')))
        loads.extend(list(grid.graph.nodes_by_attribute('load')))

    generation_timeseries = pd.Series(0, index=network.timeseries.timeindex)
    for gen in gens:
        generation_timeseries += gen.timeseries.p

    load_timeseries = pd.Series(0, index=network.timeseries.timeindex)
    for load in loads:
        load_timeseries += load.timeseries.p

    residual_load = load_timeseries - generation_timeseries

    timestamp = {}
    timestamp['load_case'] = (
        residual_load.idxmax() if max(residual_load) > 0 else None)
    timestamp['feedin_case'] = (
        residual_load.idxmin() if min(residual_load) < 0 else None)

    return timestamp

