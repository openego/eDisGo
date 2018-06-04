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
    depending on the voltage at the nodes of the generators. The voltage
    at the node is used as an input to calculate a feedin_factor that changes
    the curtailment by modifying the feed-in at the points where
    there are very high voltages. This is only used to manipulate the resulting
    curtailment and does not change the feed-in timeseries itself.

    The lower voltage threshold is the node voltage below which no
    curtailment is assigned to the respective generator connected
    to the node. This assignment can be done by using the keyword
    argument 'voltage_threshold_lower'. By default, this voltage
    is set to 1.0 per unit.

    Above the lower voltage threshold, the curtailment is proportional
    to the difference between the node voltage and the lower
    voltage threshold. Thus the higher the voltage, the greater the
    difference from the lower voltage threshold and thereby the higher the
    curtailment.

    Lowering this voltage will increase the amount of curtailment to
    generators with higher node voltages as well as possibly increase the
    number of generators the curtailment is spread over.

    This method runs an edisgo_object.analyze internally to find out
    the voltage at the nodes if an :meth:`edisgo.grid.network.EDisGo.analyze`
    has not already been performed and the results saved in
    :meth:`edisgo.grid.network.Results.v_res`.

    Parameters
    ----------
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        Obtains the feedin of each and every generator in the grid from the
        :class:`edisgo.grid.network.CurtailmentControl` class. The feedin
        dataframe has a Datetimeindex which is the same as the timeindex
        in the edisgo_object for simulation. The columns of the feedin dataframe
        are a MultiIndex column which consists of the following levels:

        * generator : :class:`edisgo.grid.components.GeneratorFluctuating`,
          essentially all the generator objects in the MV grid and the LV grid
        * gen_repr : :obj:`str`
          the repr strings of the generator objects from above
        * type : :obj:`str`
          the type of the generator object e.g. 'solar' or 'wind'
        * weather_cell_id : :obj:`int`
          the weather_cell_id that the generator object belongs to.

        All the information is gathered in the
        :class:`edisgo.grid.network.CurtailmentControl` class and available
        through the :attr:`edisgo.grid.network.CurtailmentControl.feedin`
        attribute. See :class:`edisgo.grid.network.CurtailmentControl` for more
        details.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        The curtailment to be distributed amongst the generators in the
        edisgo_objects' network. This is input through the edisgo_object.
        See class definition for further information.
    edisgo_object : :class:`edisgo.grid.network.EDisGo`
        The edisgo object in which this function was called through the
        respective :class:`edisgo.grid.network.CurtailmentControl` instance.
    voltage_threshold_lower: :obj:`float`
        The node voltage below which no curtailment would be assigned to the
        respective generator.
    """
    voltage_threshold_lower = kwargs.get('voltage_threshold_lower', 1.0)

    # get the results of a load flow
    # get the voltages at the nodes
    feedin_gen_reprs = feedin.columns.levels[1].values.copy()

    try:
        v_pu = edisgo_object.network.results.v_res()

    except AttributeError:
        # if the load flow hasn't been done,
        # do it!
        edisgo_object.analyze()
        v_pu = edisgo_object.network.results.v_res()

    if not(v_pu.empty):
        # get only the specific feedin objects
        v_pu = v_pu.loc[:, (slice(None), feedin_gen_reprs)]

        # curtailment calculation by inducing a reduced or increased feedin
        # find out the difference from lower threshold
        feedin_factor = v_pu - voltage_threshold_lower
        # make sure the difference is positive
        # zero the curtailment of those generators below the voltage_threshold_lower
        feedin_factor = feedin_factor[feedin_factor >= 0].fillna(0)
        # after being normalized to maximum difference being 1 and minimum being 0
        feedin_factor = feedin_factor.divide(feedin_factor.max(axis=1), axis=0)
        # the curtailment here would be directly multplied the difference


        feedin_factor.columns = feedin_factor.columns.droplevel(0)  # drop the 'mv' 'lv' labels

        # multiply feedin_factor to feedin to modify the amount of curtailment
        modified_feedin = feedin_factor.multiply(feedin, level=1)

        # total_curtailment
        curtailment = modified_feedin.divide(modified_feedin.sum(axis=1), axis=0). \
            multiply(total_curtailment_ts, axis=0)

        # assign curtailment to individual generators
        assign_curtailment(curtailment, edisgo_object)
    else:
        message = "There is no resulting node voltages after the PFA calculation" +\
            " which correspond to the generators in the columns of the given feedin data"
        logging.warning(message)


def curtail_all(feedin, total_curtailment_ts, edisgo_object, **kwargs):
    """
    Implements curtailment methodology 'curtail_all'.

    The curtailment that has to be met in each time step is allocated
    equally to all generators depending on their share of total
    feed-in in that time step. This is a simple curtailment method where
    the feedin is summed up and normalized, multiplied with `total_curtailment_ts`
    and assigned to each generator directly based on the columns in
    `total_curtailment_ts`.

    Parameters
    ----------
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        Obtains the feedin of each and every generator in the grid from the
        :class:`edisgo.grid.network.CurtailmentControl` class. The feedin
        dataframe has a Datetimeindex which is the same as the timeindex
        in the edisgo_object for simulation. The columns of the feedin dataframe
        are a MultiIndex column which consists of the following levels:

        * generator : :class:`edisgo.grid.components.GeneratorFluctuating`,
          essentially all the generator objects in the MV grid and the LV grid
        * gen_repr : :obj:`str`
          the repr strings of the generator objects from above
        * type : :obj:`str`
          the type of the generator object e.g. 'solar' or 'wind'
        * weather_cell_id : :obj:`int`
          the weather_cell_id that the generator object belongs to.

        All the information is gathered in the
        :class:`edisgo.grid.network.CurtailmentControl` class and available
        through the :attr:`edisgo.grid.network.CurtailmentControl.feedin`
        attribute. See :class:`edisgo.grid.network.CurtailmentControl` for more
        details.
    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        The curtailment to be distributed amongst the generators in the
        edisgo_objects' network. This is input through the edisgo_object.
        See class definition for further information.
    edisgo_object : :class:`edisgo.grid.network.EDisGo`
        The edisgo object in which this function was called through the
        respective :class:`edisgo.grid.network.CurtailmentControl` instance.
    """
    # create a feedin factor of 1
    # make sure the nans are filled in
    # this is a work around to ensure the
    # type of total_curtailment_ts (either series or dataframe)
    # doesn't affect the calculation depending on the input
    # and the timestamps are maintained

    feedin_factor = total_curtailment_ts.copy()
    feedin_factor = feedin_factor / feedin_factor
    feedin_factor.fillna(1.0, inplace=True)

    feedin.mul(feedin_factor, axis=0, level=1)

    # total_curtailment
    curtailment = feedin.divide(feedin.sum(axis=1), axis=0). \
        multiply(total_curtailment_ts, axis=0)

    # assign curtailment to individual generators
    assign_curtailment(curtailment, edisgo_object)


def assign_curtailment(curtailment, edisgo_object):
    """
    Implements curtailment helper function to assign the curtailment time series
    to each and every individual generator and ensure that they get processed
    and included in the :meth:`edisgo.grid.network.TimeSeries.curtailment` correctly

    Parameters
    ----------
    curtailment : :pandas:`pandas.DataFrame<dataframe>`
        final curtailment dataframe with generator objects as column
        labels and a DatetimeIndex as the index
    edisgo_object : :class:`edisgo.grid.network.EDisGo`
        The edisgo object in which this function was called through the
        respective :class:`edisgo.grid.network.CurtailmentControl` instance.
    """
    # pre-process curtailment before assigning it to generators
    curtailment.fillna(0, inplace=True)

    # drop extra column levels that were present in feedin
    for r in range(len(curtailment.columns.levels) - 1):
        curtailment.columns = curtailment.columns.droplevel(1)

    # assign curtailment to individual generators
    for gen in curtailment.columns:
        gen.curtailment = curtailment.loc[:, gen]

    if not edisgo_object.network.timeseries._curtailment:
        edisgo_object.network.timeseries._curtailment = list(curtailment.columns)
    else:
        edisgo_object.network.timeseries._curtailment.extend(list(curtailment.columns))
