import pandas as pd
import numpy as np
import logging

from edisgo.grid.tools import get_gen_info, \
    get_capacities_by_type, \
    get_capacities_by_type_and_weather_cell


def curtail_voltage(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
    Implements curtailment methodology 'curtail_voltage'.

    The curtailment that has to be met in each step is allocated
    depending on the voltage at the nodes of the generators

    Parameters
    ----------
    feedin : : pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object created
    voltage_threshold_lower: :float

    **kwargs : :class:`~.grid.network.Network`
    """
    voltage_threshold_lower = kwargs.get('voltage_threshold_lower', 0.96)

    # get the results of a load flow
    # get the voltages at the nodes
    try:
        v_pu = edisgo_object.network.results.v_res()

    except TypeError:
        # if the load flow hasn't been done,
        # do it!
        edisgo_object.analyze()
        v_pu = edisgo_object.network.results.v_res()
        #TODO : This is a temporary fix but this must be fixed for later
        edisgo_object.network.pypsa = None

    # get only the GeneratorFluctuating objects
    gen_columns = list(filter(lambda x: 'GeneratorFluctuating' in x, v_pu.columns.levels[1]))
    v_pu.sort_index(axis=1, inplace=True)
    v_pu_gen = v_pu.loc[:, (slice(None), gen_columns)]

    # curtailment calculation by inducing a reduced or increased feedin
    # find out the difference from lower threshold
    feedin_factor = v_pu_gen - voltage_threshold_lower + 1
    feedin_factor.columns = feedin_factor.columns.droplevel(0)  # drop the 'mv' 'lv' labels

    # multiply feedin_factor to feedin to modify the amount of curtailment
    modified_feedin = feedin_factor.multiply(feedin, level=1)

    # total_curtailment
    curtailment = modified_feedin.divide(modified_feedin.sum(axis=1), axis=0). \
        multiply(total_curtailment_ts, axis=0)
    for r in range(len(curtailment.columns.levels)-1):
        curtailment.columns = curtailment.columns.droplevel(1)

    # assign curtailment to individual generators
    assign_curtailment(curtailment, edisgo_object)


def curtail_loading(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
        Implements curtailment methodology 'curtail_loading'.

        The curtailment that has to be met in each step is allocated
        depending on the voltage at the nodes of the generators

        Parameters
        ----------
        feedin : : pandas:`pandas.DataFrame<dataframe>`
            See class definition for further information.
        total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
            See class definition for further information.
        edisgo_object : :class:`edisgo.EDisGo`
            The edisgo object created
        **kwargs : :class:`~.grid.network.Network`
        """

    return None


def curtail_all(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
    Implements curtailment methodology 'curtail_all'.

    The curtailment that has to be met in each time step is allocated
    equally to all generators depending on their share of total
    feed-in in that time step.

    Parameters
    ----------
    feedin : : pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object created
    **kwargs : :class:`~.grid.network.Network`

    """
    if 'feedin_factor' in kwargs:
        feedin_factor = kwargs.get('feedin_factor')
    else:
        feedin_factor = total_curtailment_ts.copy()
        feedin_factor = feedin_factor / feedin_factor
        feedin_factor.fillna(1.0, inplace=True)

    # feedin = feedin.sum(axis=1, level='type')
    feedin.mul(feedin_factor, axis=0, level=1)

    # total_curtailment
    curtailment = feedin.divide(feedin.sum(axis=1), axis=0). \
        multiply(total_curtailment_ts, axis=0)

    # Drop columns where there were 0/0 divisions due to feedin being 0
    curtailment.dropna(axis=1, how='all', inplace=True)
    # fill the remaining nans if there are any with 0s
    curtailment.fillna(0, inplace=True)

    # drop all the feedin column multiindex levels keeping the
    # Generator Objects as the only column label
    for r in range(len(curtailment.columns.levels)-1):
        curtailment.columns = curtailment.columns.droplevel(1)

    # assign curtailment to individual generators
    assign_curtailment(curtailment, edisgo_object)


def assign_curtailment(curtailment, edisgo_object):
    """
    Implements curtailment helper function to assign the curtailment time series
    to each and every individual generator and ensure that they get processed
    and included in the edisgo_object.timeseries.curtailment correctly

    Parameters
    ----------
    curtailment : : pandas:`pandas.DataFrame<dataframe>`
        final curtailment dataframe with generator objects as column
        labels and a DatetimeIndex as the index
    edisgo_object : :class:`edisgo.EDisGo`
        The edisgo object created
    **kwargs : :class:`~.grid.network.Network`

    """
    # assign curtailment to individual generators
    for gen in curtailment.columns:
        gen.curtailment = curtailment.loc[:, gen]

    if not edisgo_object.network.timeseries._curtailment:
        edisgo_object.network.timeseries._curtailment = list(curtailment.columns)
    else:
        edisgo_object.network.timeseries._curtailment.extend(list(curtailment.columns))
