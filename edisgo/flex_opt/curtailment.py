import pandas as pd
import logging


def curtail_all(total_curtailment_ts, network):
    """
    Implements curtailment methodology 'curtail_all'.

    The curtailment that has to be met in each time step is allocated
    equally to all generators depending on their share of total
    feed-in in that time step.

    Parameters
    ----------
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    network : :class:`~.grid.network.Network`

    """

    def _calculate_curtailment(feedin_df, curtailment_series):
        """
        Gets installed capacities of wind and solar generators.

        Parameters
        -----------
        feedin_df : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with feed-in time series the curtailment is
            allocated to.
        curtailment_series : :pandas:`pandas.Series<series>`
            Series with curtailment time series that needs to be
            allocated to solar and wind generators and/or weather cells.

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with the allocated curtailment time series.

        """
        curtailment = (feedin_df.divide(
            feedin_df.sum(axis=1), axis=0)).multiply(
            curtailment_series, axis=0)
        return curtailment

    # get aggregated capacities either by technology or technology and
    # weather cell
    if isinstance(network.timeseries.generation_fluctuating.columns,
                  pd.MultiIndex):
        dict_capacities = _get_capacities_by_type_and_weather_cell(network)
    else:
        dict_capacities = _get_capacities_by_type(network)

    # calculate absolute feed-in
    feedin_df = network.timeseries._generation_fluctuating.multiply(
        pd.Series(dict_capacities))
    feedin_df.dropna(axis=1, how='all', inplace=True)

    # allocate curtailment if feed-in time series are in a higher
    # resolution than the curtailment time series (e.g. curtailment
    # specified by technology and feed-in by technology and weather cell)
    if isinstance(total_curtailment_ts, pd.Series):
        network.timeseries.curtailment = _calculate_curtailment(
            feedin_df, total_curtailment_ts)
    elif isinstance(total_curtailment_ts, pd.DataFrame):
        if isinstance(total_curtailment_ts.columns, pd.MultiIndex):
            # if both feed-in and curtailment are differentiated by
            # technology and weather cell the curtailment time series
            # can be used directly
            if isinstance(feedin_df.columns, pd.MultiIndex):
                network.timeseries.curtailment = total_curtailment_ts
            else:
                message = 'Curtailment time series are provided for ' \
                          'different weather cells but feed-in time ' \
                          'series are not.'
                logging.error(message)
                raise KeyError(message)
        else:
            if isinstance(feedin_df.columns, pd.MultiIndex):
                # allocate curtailment to weather cells
                if 'wind' in total_curtailment_ts.columns:
                    try:
                        curtailment_wind = _calculate_curtailment(
                            feedin_df.loc[:, 'wind': 'wind'],
                            total_curtailment_ts['wind'])
                    except:
                        message = 'Curtailment time series for wind ' \
                                  'generators provided but no wind ' \
                                  'feed-in time series.'
                        logging.error(message)
                        raise KeyError(message)
                else:
                    curtailment_wind = pd.DataFrame()
                if 'solar' in total_curtailment_ts.columns:
                    try:
                        curtailment_solar = _calculate_curtailment(
                            feedin_df.loc[:, 'solar': 'solar'],
                            total_curtailment_ts['solar'])
                    except:
                        message = 'Curtailment time series for solar ' \
                                  'generators provided but no solar ' \
                                  'feed-in time series.'
                        logging.error(message)
                        raise KeyError(message)
                else:
                    curtailment_solar = pd.DataFrame()
                network.timeseries.curtailment = \
                    curtailment_wind.join(curtailment_solar, how='outer')
            else:
                # if both feed-in and curtailment are only differentiated
                # by technology the curtailment time series can be used
                # directly
                network.timeseries.curtailment = total_curtailment_ts
    else:
        message = 'Unallowed type {} of provided curtailment time ' \
                  'series. Must either be pandas.Series or ' \
                  'pandas.DataFrame.'.format(type(total_curtailment_ts))
        logging.error(message)
        raise TypeError(message)

    # check if curtailment exceeds feed-in
    _check_curtailment(feedin_df, network)


def _get_capacities_by_type_and_weather_cell(network):
    """
    Gets installed capacities of wind and solar generators by weather
    cell ID.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    --------
    dict
        Dictionary with keys being a tuple of technology and weather
        cell ID (e.g. ('solar', '1')) and the values containing the
        corresponding installed capacity.

    """
    # get all generators
    gens = list(network.mv_grid.graph.nodes_by_attribute('generator'))
    for lv_grid in network.mv_grid.lv_grids:
        gens.extend(list(lv_grid.graph.nodes_by_attribute('generator')))

    dict_capacities = {}
    for gen in gens:
        if gen.type in ['solar', 'wind']:
            if gen.weather_cell_id:
                if (gen.type, gen.weather_cell_id) in \
                        dict_capacities.keys():
                    dict_capacities[
                        (gen.type, gen.weather_cell_id)] = \
                        dict_capacities[
                            (gen.type, gen.weather_cell_id)] + \
                        gen.nominal_capacity
                else:
                    dict_capacities[
                        (gen.type, gen.weather_cell_id)] = \
                        gen.nominal_capacity
            else:
                message = 'Please provide a weather cell ID for ' \
                          'generator {}.'.format(repr(gen))
                logging.error(message)
                raise KeyError(message)
    return dict_capacities


def _get_capacities_by_type(network):
    """
    Gets installed capacities of wind and solar generators.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`

    Returns
    --------
    dict
        Dictionary with keys 'solar' and 'wind' and the values
        containing the corresponding installed capacity.

    """
    # get all generators
    gens = list(network.mv_grid.graph.nodes_by_attribute('generator'))
    for lv_grid in network.mv_grid.lv_grids:
        gens.extend(list(lv_grid.graph.nodes_by_attribute('generator')))

    dict_capacities = {'solar': 0, 'wind': 0}
    for gen in gens:
        if gen.type in ['solar', 'wind']:
            dict_capacities[gen.type] = gen.nominal_capacity
    return dict_capacities


def _check_curtailment(feedin_df, network):
    """
    Raises an error if the curtailment at any time step exceeds the
    feed-in at that time.

    Parameters
    -----------
    feedin_df : :pandas:`pandas.DataFrame<dataframe>`
        DataFrame with feed-in time series in kW. The DataFrame needs to have
        the same columns as the curtailment DataFrame.
    network : :class:`~.grid.network.Network`

    """
    if not (feedin_df.loc[:, network.timeseries.curtailment.columns]
            >= network.timeseries.curtailment).all().all():
        message = 'Curtailment exceeds feed-in.'
        logging.error(message)
        raise TypeError(message)