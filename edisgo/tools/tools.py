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
    timeseries_load_feedin_case = network.timeseries.timesteps_load_feedin_case

    timestamp = {}
    timestamp['load_case'] = (
        timeseries_load_feedin_case.residual_load.idxmax()
        if max(timeseries_load_feedin_case.residual_load) > 0 else None)
    timestamp['feedin_case'] = (
        timeseries_load_feedin_case.residual_load.idxmin()
        if min(timeseries_load_feedin_case.residual_load) < 0 else None)
    return timestamp


def get_residual_load_from_pypsa_network(pypsa_network):
    """
    Calculates residual load in MW in MV grid and underlying LV grids.

    Parameters
    ----------
    pypsa_network : :pypsa:`pypsa.Network<network>`
        The `PyPSA network
        <https://www.pypsa.org/doc/components.html#network>`_ container,
        containing load flow results.

    Returns
    -------
    :pandas:`pandas.Series<series>`
        Series with residual load in MW for each time step. Positiv values
        indicate a higher demand than generation and vice versa. Index of the
        series is a :pandas:`pandas.DatetimeIndex<datetimeindex>`

    """
    residual_load = \
        pypsa_network.loads_t.p_set.sum(axis=1) - (
                pypsa_network.generators_t.p_set.loc[
                :, pypsa_network.generators_t.p_set.columns !=
                   'Generator_slack'].sum(axis=1) +
                pypsa_network.storage_units_t.p_set.sum(axis=1))
    return residual_load


def assign_load_feedin_case(network):
    """
    For each time step evaluate whether it is a feed-in or a load case.

    Feed-in and load case are identified based on the
    generation and load time series and defined as follows:

    1. Load case: positive (load - generation) at HV/MV substation
    2. Feed-in case: negative (load - generation) at HV/MV substation

    Output of this function is written to `timesteps_load_feedin_case`
    attribute of the network.timeseries (see
    :class:`~.grid.network.TimeSeries`).

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        Network for which worst-case snapshots are identified.

    Returns
    --------
    :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with information on whether time step is handled as load case
        ('load_case') or feed-in case ('feedin_case') for each time step in
        `timeindex` attribute of network.timeseries.
        Index of the dataframe is network.timeseries.timeindex. Columns of the
        dataframe are 'residual_load' with (load - generation) in kW at HV/MV
        substation and 'case' with 'load_case' for positive residual load and
        'feedin_case' for negative residual load.

    """

    if network.pypsa is not None:
        residual_load = get_residual_load_from_pypsa_network(network.pypsa) * \
                        1e3

    else:
        grids = [network.mv_grid] + list(network.mv_grid.lv_grids)

        gens = []
        loads = []
        for grid in grids:
            gens.extend(grid.generators)
            gens.extend(list(grid.graph.nodes_by_attribute('storage')))
            loads.extend(list(grid.graph.nodes_by_attribute('load')))

        generation_timeseries = pd.Series(
            0, index=network.timeseries.timeindex)
        for gen in gens:
            generation_timeseries += gen.timeseries.p

        load_timeseries = pd.Series(0, index=network.timeseries.timeindex)
        for load in loads:
            load_timeseries += load.timeseries.p

        residual_load = load_timeseries - generation_timeseries

    timeseries_load_feedin_case = residual_load.rename(
        'residual_load').to_frame()

    timeseries_load_feedin_case['case'] = \
        timeseries_load_feedin_case.residual_load.apply(
            lambda _: 'feedin_case' if _ < 0 else 'load_case')

    return timeseries_load_feedin_case
