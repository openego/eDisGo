import pandas as pd


def select_worstcase_snapshots(network):
    """
    Select two worst-case snapshot from time series

    Two time steps in a time series represent worst-case snapshots. These are

    1. Maximum residual load: refers to the point in the time series where the
        (load - generation) achieves its maximum
    2. Minimum residual load: refers to the point in the time series where the
        (load - generation) achieves its minimum

    These to points are identified based on the generation and load time series.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo overall container

    Returns
    -------
    type
        Timestamp of snapshot maximum residual load
    type
        Timestamp of snapshot minimum residual load
    """

    grids = [network.mv_grid] + list(network.mv_grid.lv_grids)

    peak_generation = pd.concat(
        [_.peak_generation_per_technology for _ in grids], axis=1).fillna(
        0).sum(axis=1)

    non_solar_wind = [_ for _ in list(peak_generation.index)
                      if _ not in ['wind', 'solar']]
    peak_generation['other'] = peak_generation[non_solar_wind].sum()
    peak_generation.drop(non_solar_wind, inplace=True)

    peak_load = pd.concat(
        [_.consumption for _ in grids], axis=1).fillna(
        0).sum(axis=1)

    residual_load = (
    (network.timeseries.load * peak_load).sum(axis=1) - (
        network.timeseries.generation * peak_generation).sum(
        axis=1))

    return residual_load.idxmax(), residual_load.idxmin()

