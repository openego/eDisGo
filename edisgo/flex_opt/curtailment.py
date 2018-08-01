import pandas as pd
import logging

from pyomo.environ import ConcreteModel, Set, Param, Objective, Constraint, minimize, Var
from pyomo.opt import SolverFactory


def curtail_voltage(feedin, generators, total_curtailment_ts, edisgo,
                    **kwargs):
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
    argument 'voltage_threshold'. By default, this voltage
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
    voltage_threshold: :obj:`float`
        The node voltage below which no curtailment would be assigned to the
        respective generator. Default: 1.0.
    """
    voltage_threshold = pd.Series(kwargs.get('voltage_threshold', 1.0),
                                  index=total_curtailment_ts.index)

    # get the voltages at the nodes
    voltage_pu_lv = edisgo.network.results.v_res(
        nodes=generators.loc[(generators.voltage_level == 'lv')].index,
        level='lv')
    voltage_pu_mv = edisgo.network.results.v_res(
        nodes=generators.loc[(generators.voltage_level == 'mv')].index,
        level='mv')
    voltage_pu = voltage_pu_lv.join(voltage_pu_mv)

    # for every time step check if curtailment can be fulfilled, otherwise
    # reduce voltage threshold; set feed-in of generators below voltage
    # threshold to zero, so that they cannot be curtailed
    for ts in total_curtailment_ts.index:
        # get generators with voltage higher than threshold
        gen_pool = voltage_pu.loc[
            ts, voltage_pu.loc[ts, :] > voltage_threshold.loc[ts]].index
        # if curtailment cannot be fulfilled lower voltage threshold
        while sum(feedin.loc[ts, gen_pool]) < total_curtailment_ts.loc[ts]:
            voltage_threshold.loc[ts] = voltage_threshold.loc[ts] - 0.01
            gen_pool = voltage_pu.loc[
                ts, voltage_pu.loc[ts, :] > voltage_threshold.loc[ts]].index
        # set feed-in of generators below voltage threshold to zero, so that
        # they cannot be curtailed
        gen_pool_out = voltage_pu.loc[
            ts, voltage_pu.loc[ts, :] <= voltage_threshold.loc[ts]].index
        feedin.loc[ts, gen_pool_out] = 0

    # only optimize for time steps where curtailment is greater than zero
    timeindex = total_curtailment_ts[total_curtailment_ts > 0].index
    if not timeindex.empty:
        curtailment = _optimize_curtail_voltage(
            feedin, voltage_pu, total_curtailment_ts, voltage_threshold,
            timeindex)
    else:
        curtailment = pd.DataFrame()

    # set curtailment for other time steps to zero
    curtailment = curtailment.append(pd.DataFrame(
        0, columns=feedin.columns, index=total_curtailment_ts[
            total_curtailment_ts <= 0].index))

    # assign curtailment to individual generators
    assign_curtailment(curtailment, edisgo, generators)


def _optimize_curtail_voltage(feedin, voltage_pu, total_curtailment,
                              voltage_threshold, timeindex):
    """
    Parameters
    ------------
    feedin : index is time index, columns are generators that will be curtailed, feed-in in kW
    voltage_pu : index is time index, columns are generators, voltage from power flow in p.u.
    total_curtailment : index is time index, total curtailment in time step in kW
    voltage_threshold : index is time index, voltage threshold in time step in p.u.

    """

    logging.info("Start curtailment optimization.")
        
    v_max = voltage_pu.max(axis=1)
    generators = feedin.columns

    # additional curtailment factors
    cf_add = pd.DataFrame(index=timeindex)
    for gen in generators:
        cf_add[gen] = abs(
            (voltage_pu.loc[timeindex, gen] - v_max[timeindex]) / (
                    voltage_threshold[timeindex] - v_max[timeindex]))

    # curtailment factors
    cf = pd.DataFrame(index=timeindex)
    for gen in generators:
        cf[gen] = abs(
            (voltage_pu.loc[timeindex, gen] - voltage_threshold[timeindex]) / (
                    v_max[timeindex] - voltage_threshold[timeindex]))

    ### initialize model
    model = ConcreteModel()

    ### add sets
    model.T = Set(initialize=timeindex)
    model.G = Set(initialize=generators)

    ### add parameters
    def feedin_init(model, t, g):
        return feedin.loc[t, g]

    model.feedin = Param(model.T, model.G, initialize=feedin_init)

    def voltage_pu_init(model, t, g):
        return voltage_pu.loc[t, g]

    model.voltage_pu = Param(model.T, model.G, initialize=voltage_pu_init)

    def cf_add_init(model, t, g):
        return cf_add.loc[t, g]

    model.cf_add = Param(model.T, model.G, initialize=cf_add_init)

    def cf_init(model, t, g):
        return cf.loc[t, g]

    model.cf = Param(model.T, model.G, initialize=cf_init)

    def total_curtailment_init(model, t):
        return total_curtailment.loc[t]

    model.total_curtailment = Param(model.T, initialize=total_curtailment_init)

    ### add variables
    model.offset = Var(model.T, bounds=(0, 1))
    model.cf_max = Var(model.T, bounds=(0, 1))

    def curtailment_init(model, t, g):
        return (0, feedin.loc[t, g])

    model.c = Var(model.T, model.G, bounds=curtailment_init)

    ### add objective
    def obj_rule(model):
        expr = (sum(model.offset[t] * 100
                    for t in model.T))
        return expr

    model.obj = Objective(rule=obj_rule, sense=minimize)

    ### add constraints
    # curtailment per generator constraints
    def curtail(model, t, g):
        return (
        model.cf[t, g] * model.cf_max[t] * model.feedin[t, g] + model.cf_add[
            t, g] * model.offset[t] * model.feedin[t, g] == model.c[t, g])

    model.curtailment = Constraint(model.T, model.G, rule=curtail)

    # total curtailment constraint
    def total_curtailment(model, t):
        return (
        sum(model.c[t, g] for g in model.G) == model.total_curtailment[t])

    model.sum_curtailment = Constraint(model.T, rule=total_curtailment)

    # solve
    solver = SolverFactory('cbc')
    results = solver.solve(model, tee=True)

    # load results back into model
    model.solutions.load_from(results)

    c = pd.DataFrame({g: [model.c[t, g].value for t in model.T]
                      for g in model.G},
                     index=model.T.value)
    return c


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


def assign_curtailment(curtailment, edisgo, generators):
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

    # assign curtailment to individual generators
    gen_object_list = []
    for gen in curtailment.columns:
        # get object from representative
        gen_object = generators.loc[generators.gen_repr == gen].index[0]
        gen_object.curtailment = curtailment.loc[:, gen]
        gen_object_list.append(gen_object)

    if edisgo.network.timeseries._curtailment:
        edisgo.network.timeseries._curtailment.extend(gen_object_list)
    else:
        edisgo.network.timeseries._curtailment = gen_object_list
