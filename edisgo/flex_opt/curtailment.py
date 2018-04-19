import pandas as pd
import numpy as np
import logging

from edisgo.grid.tools import get_gen_info, \
    get_capacities_by_type, \
    get_capacities_by_type_and_weather_cell


def curtail_all(feedin_df, total_curtailment_ts, network, **kwargs):
    """
    Implements curtailment methodology 'curtail_all'.

    The curtailment that has to be met in each time step is allocated
    equally to all generators depending on their share of total
    feed-in in that time step.

    Parameters
    ----------
    feedin_df : : pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    network : :class:`~.grid.network.Network`
    **kwargs : :class:`~.grid.network.Network`

    """
    if 'curtailment_factor' in kwargs:
        curtailment_factor = kwargs.get('curtailment_factor')
    else:
        curtailment_factor = total_curtailment_ts.copy()
        curtailment_factor = curtailment_factor / curtailment_factor
        curtailment_factor.fillna(1.0, inplace=True)

    feedin_df = feedin_df.T.groupby('type').sum().T

    total_curtailment = total_curtailment_ts.multiply(curtailment_factor)
    curtailment = (feedin_df.divide(
        feedin_df.sum(axis=1), axis=0)).multiply(
        total_curtailment, axis=0)
    return curtailment

def curtail_voltage_(feedin_df, total_curtailment_ts, network, **kwargs):
    """
    Implements curtailment methodology 'curtail_voltage'.

    The curtailment that has to be met in each step is allocated
    depending on the voltage at the nodes of the generators

    Parameters
    ----------
    feedin_df : : pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        See class definition for further information.
    network : :class:`~.grid.network.Network`
    **kwargs : :class:`~.grid.network.Network`
    """
    # get the generators:
    gens = get_gen_info(network)

    # get the results of a load flow
    # get the voltages at the nodes
    try:
        v_pu = network.results.v_res()

    except AttributeError:
        # if the load flow hasn't been done,
        # do it!
        network.analyse()
        v_pu = network.results.v_res()

    # create a series of curtailment factors df
    # curtailment_factors_df =
    # network.mv_grid

    return None