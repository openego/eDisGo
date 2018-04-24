import pandas as pd
import numpy as np
import logging

from edisgo.grid.tools import get_gen_info, \
    get_capacities_by_type, \
    get_capacities_by_type_and_weather_cell



def curtail_all(feedin_df, total_curtailment_ts, edisgo_object, **kwargs):
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

    # write the curtailment timeseries in the right location
    edisgo_object.network.timeseries.curtailment = curtailment


def curtail_voltage(feedin_df, total_curtailment_ts, edisgo_object, **kwargs):
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

    # get only the GeneratorFluctuating objects
    gen_columns = list(filter(lambda x: 'GeneratorFluctuating' in x, v_pu.columns.levels[1]))
    v_pu.sort_index(axis=1, inplace=True)
    v_pu_gen = v_pu.loc[:, (['mv', 'lv'], gen_columns)]

    # curtailment calculation by inducing a reduced or increased feedin
    # find out the difference from lower threshold
    feedin_factor = v_pu_gen - voltage_threshold_lower + 1
    feedin_factor.columns = feedin_factor.columns.droplevel(0)  # drop the 'mv' 'lv' labels

    # multiply feedin_factor to feedin to modify the amount of curtailment
    modified_feedin = feedin_factor.multiply(feedin_df, level=0)

    # total_curtailment
    curtailment = modified_feedin.divide(modified_feedin.sum(axis=1), axis=0).\
        multiply(total_curtailment_ts, axis=0)
    curtailment.columns = curtailment.columns.droplevel(1)

    # get all generators from the network
    gens = get_gen_info(edisgo_object.network, 'mvlv')
    gens = gens.loc[(gens.type == 'solar') | (gens.type == 'wind')]
    gens.set_index('gen_repr', inplace=True)

    # assign curtailment to individual generators
    for gen in curtailment.columns:
        gens.loc[gen, 'generator'].curtailment = curtailment.loc[:, gen]

    edisgo_object.network.timeseries.curtailment = list(gens.loc[curtailment.columns,
                                                                 'generator'])


def curtail_loading(feedin_df, total_curtailment_ts, network, **kwargs):
    """
        Implements curtailment methodology 'curtail_loading'.

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

    return None