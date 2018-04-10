import pandas as pd
import logging


def curtail_all(feedin_df, total_curtailment_ts):
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

    """
    curtailment = (feedin_df.divide(
        feedin_df.sum(axis=1), axis=0)).multiply(
        total_curtailment_ts, axis=0)
    return curtailment
