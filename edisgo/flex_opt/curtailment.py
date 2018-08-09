import pandas as pd
import logging

from pyomo.environ import ConcreteModel, Set, Param, Objective, Constraint, minimize, Var
from pyomo.opt import SolverFactory


def curtail_voltage(feedin, generators, total_curtailment_ts, edisgo,
                    assigned_curtailment_key,
                    **kwargs):
    """
    Implements curtailment methodology 'curtail_voltage'.

    The curtailment that has to be met in each step is allocated
    depending on the voltage at the nodes of the generators. The voltage
    at the node is used as an input to calculate a feedin_factor that changes
    the curtailment by curtailing the feed-in at the points where
    there are very high voltages. This is only used to manipulate the resulting
    curtailment and does not change the feed-in timeseries itself.

    The lower voltage threshold is the node voltage below which no
    curtailment is assigned to the respective generator connected
    to the node. This assignment can be done by using the keyword
    argument 'voltage_threshold'. By default, this voltage
    is set to 1.0 per unit.

    Above the voltage threshold, the curtailment is proportional
    to the difference between the node voltage and the
    voltage threshold. Thus the higher the voltage, the greater the
    difference from the voltage threshold and thereby the higher the
    curtailment.

    Lowering this voltage will increase the amount of curtailment to
    generators with higher node voltages as well as possibly increase the
    number of generators the curtailment is spread over.

    The method builds the curtailment distribution to the generators as
    and optimization problem and passes through an available solver, like
    'glpk' or 'cbc', etc.

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
    assigned_curtailment_key: :obj:`tuple` or :obj:`str`
        The type and weather cell ID if :obj:`tuple` or only
        the type if :obj:`str` in which the generators are
        being curtailed. This is used to separate the
        resulting assigned curtailment dataframes in the
        :class:`edisgo.grid.network.Results` objects
        accordingly.
    voltage_threshold: :obj:`float`
        The node voltage below which no curtailment would be assigned to the
        respective generator. Default: 1.0.
    solver: :obj:`str`
        The solver used to optimize the curtailment assigned to the generator.
        The string depends upon the installed or available solver.

        * 'cbc' - coin-or branch and cut solver
        * 'glpk' - gnu linear programming kit solver
        * any other available compatible with 'pyomo' like 'gurobi'
          or 'cplex'
        Default: 'cbc'
    """
    # get the optimization_method
    optimization_method = kwargs.get('optimization_method', 'lp')

    voltage_threshold = pd.Series(kwargs.get('voltage_threshold', 1.0),
                                  index=total_curtailment_ts.index)

    # get the voltages at the nodes
    # need to get generators separately due to non-lex-sorted column
    # levels/labels
    voltage_pu_lv = edisgo.network.results.v_res(
        nodes=generators.loc[(generators.voltage_level == 'lv')].index,
        level='lv')
    voltage_pu_mv = edisgo.network.results.v_res(
        nodes=generators.loc[(generators.voltage_level == 'mv')].index,
        level='mv')
    voltage_pu = voltage_pu_lv.join(voltage_pu_mv)

    if optimization_method == 'weighted':
        # save feedin in case curtail all needs to be done as a last resort
        feedin_orig = feedin.copy()

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
            logging.debug("The 'voltage_threshold' chosen is too high, "
                         "the feedin from the generators with such high "
                         "voltages is insufficient for the required "
                         "curtailment. Voltage reduced to {}".format(
                voltage_threshold.loc[ts]))
        # set feed-in of generators below voltage threshold to zero, so that
        # they cannot be curtailed
        gen_pool_out = voltage_pu.loc[
            ts, voltage_pu.loc[ts, :] <= voltage_threshold.loc[ts]].index
        feedin.loc[ts, gen_pool_out] = 0

    if optimization_method == 'lp':
        # get the appropriate solver available
        solver = kwargs.get('solver', 'cbc')

        # only optimize for time steps where curtailment is greater than zero
        timeindex = total_curtailment_ts[total_curtailment_ts > 0].index
        if not timeindex.empty:
            curtailment = _optimize_curtail_voltage(
                feedin, voltage_pu, total_curtailment_ts, voltage_threshold,
                timeindex, solver)
        else:
            curtailment = pd.DataFrame()

        # set curtailment for other time steps to zero
        curtailment = curtailment.append(pd.DataFrame(
            0, columns=feedin.columns, index=total_curtailment_ts[
                total_curtailment_ts <= 0].index))
    elif optimization_method == 'weighted':
        # do weighted curtailment with no solver
        curtailment = _weighted_curtail_voltage(
            feedin, voltage_pu, total_curtailment_ts, voltage_threshold, feedin_orig)
    else:
        message = 'Unknown Optimization method {}'.format(optimization_method)
        raise RuntimeError(message)

    # assign curtailment to individual generators
    assign_curtailment(curtailment, edisgo, generators, assigned_curtailment_key)


def _optimize_curtail_voltage(feedin, voltage_pu, total_curtailment,
                              voltage_threshold, timeindex, solver='cbc'):
    """
    Implements the curtailment method based on voltage by formulating the
    strategy as a liner optimization problem. This method uses a linear
    programming solver to find the most optimal distribution of curtailment
    among the generators.

    Parameters
    ------------
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        This contains the feedin of the specific set of generators that either:

        * of no specific grouping if curtailment is provided
          as a :pandas:`pandas.Series<series>`
        * of a given type if curtailment is provided
          as a :pandas:`pandas.DataFrame<dataframe`
        * of a give type and in a given weather cell if
          curtailment is provided as a :pandas:`pandas.DataFrame<dataframe>`
          with a :pandas.`pandas.MultiIndex` column with two levels, type
          and weather cell ID.

        The feedin that is expected as an input here already has the
        columns corresponding to the 'not-to-be-curtailed' generators
        set to zero so that no curtailment is assigned to them.
        The index is a time index and the columns are generators that will be curtailed,
        The feed-in is in kW.
    voltage_pu : :pandas:`pandas.DataFrame<dataframe>
        The dataframe that contains the voltages at the
        various generator nodes after doing a power flow calculation.
        This stucture's index is a :pandas:`pandas.DateTimeIndex`
        and the columns are generators.
        The voltage from power flow in p.u.
    total_curtailment : :pandas:`pandas.Series<series>`
        The series containing the specific curtailment for the type of
        generators to be curtailed and the weather cells they belong to.
        This is the specific column of the input curtailment.
        The index is a :pandas:`pandas.DateTimeIndex` and the
        total curtailment in time step in kW
    voltage_threshold : :pandas:`pandas.Series<series>`
        A series containing the voltage thresholds below which no
        generator curtailment will occur.The series contains the
        threshold calculated for each and every timestep that is
        being calculated. The index is a :pandas:`pandas.DateTimeIndex`,
        and the voltage threshold in time step in p.u.
    solver: :obj:`str`
        The solver used to optimize the curtailment assigned to the generator.
        The string depends upon the installed or available solver.

        * 'cbc' - coin-or branch and cut solver
        * 'glpk' - gnu linear programming kit solver
        * any other available compatible with 'pyomo' like 'gurobi'
          or 'cplex'
        Default: 'cbc'

    Returns
    -------
    :pandas:`pandas:DataFrame<dataframe>`
        A dataframe containing the curtailment in kW per generator
        of the provided 'feedin' dataframe per timestep.
        The index is a :pandas:`pandas.DateTimeIndex` and the
        columns are :py:mod:`edisgo.network.GeneratorFluctuating`
        objects.
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
            t, g] * model.offset[t] * model.feedin[t, g] - model.c[t, g] == 0)

    model.curtailment = Constraint(model.T, model.G, rule=curtail)

    # total curtailment constraint
    def total_curtailment(model, t):
        return (
        sum(model.c[t, g] for g in model.G) == model.total_curtailment[t])

    model.sum_curtailment = Constraint(model.T, rule=total_curtailment)

    # solve
    solver = SolverFactory(solver)
    results = solver.solve(model, tee=True)

    # load results back into model
    model.solutions.load_from(results)

    c = pd.DataFrame({g: [model.c[t, g].value for t in model.T]
                      for g in model.G},
                     index=model.T.value)
    return c


def _weighted_curtail_voltage(feedin,
                              voltage_pu,
                              total_curtailment,
                              voltage_threshold,
                              feedin_original):
    """
    Implements curtailment method 'curtail_voltage'
    using weighting factors to influence the
    amount of curtailment assigned to the generators.
    This strategy assigns weights and checks if the
    assigned curtailment is still above the limits
    of the individual generator feedins. If the
    curtailment exceeds the feedins then a small
    kink is introduced in the curtailment vs
    voltage characteristic to increase the curtailment
    of the generators with voltages close to the
    voltage_threshold. The kink is increases till
    the point where the individual generator
    curtailments do not exceed their feedins.
    If during the calculation the kink becomes too much,
    the curtailment_voltage strategy is replaced with
    the curtail_all strategy, this case should not be met
    in any typical case with curtailment < 100% of feed-in.
    This is a last resort measure if the strategy fails.
    The method also checks to ensure that the total curtailment
    assigned remains the same as the curtailment
    provided in the input.

    Parameters
    ----------
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        This contains the feedin of the specific set
        of generators that either:

        * of no specific grouping if curtailment is provided
          as a :pandas:`pandas.Series<series>`
        * of a given type if curtailment is provided
          as a :pandas:`pandas.DataFrame<dataframe`
        * of a give type and in a given weather cell if
          curtailment is provided as a :pandas:`pandas.DataFrame<dataframe>`
          with a :pandas.`pandas.MultiIndex` column with two levels, type
          and weather cell ID.

        The feedin that is expected as an input here already has the
        columns corresponding to the 'not-to-be-curtailed' generators
        set to zero so that no curtailment is assigned to them.
        The index is a time index and thecolumns are generators
        that will be curtailed.
        The feed-in is in kW.
    voltage_pu : :pandas:`pandas.DataFrame<dataframe>
        The dataframe that contains the voltages at the
        various generator nodes after doing a power flow calculation.
        This stucture's index is a :pandas:`pandas.DateTimeIndex`
        and the columns are generators.
        The voltage from power flow in p.u.
    total_curtailment : :pandas:`pandas.Series<series>`
        The series containing the specific curtailment for the type of
        generators to be curtailed and the weather cells they belong to.
        This is the specific column of the input curtailment.
        The index is a :pandas:`pandas.DateTimeIndex` and the
        total curtailment in time step in kW
    voltage_threshold : :pandas:`pandas.Series<series>`
        A series containing the voltage thresholds below which no
        generator curtailment will occur.The series contains the
        threshold calculated for each and every timestep that is
        being calculated. The index is a :pandas:`pandas.DateTimeIndex`,
        and the voltage threshold in time step in p.u.
    feedin_original: :pandas:`pandas.DataFrame<dataframe>`
        The same as 'feedin' with the only difference being that this
        dataframe would not have the 'not-to-be-curtailed' generators'
        feedin zeroed out. The dataframe would be used as a last resort
        only when the curtailment based on voltage fails. This dataframe
        would be used to calculate using a curtail_all algorithm
        if voltage weighted curtailment fails.

    Return
    ------
    :pandas:`pandas:DataFrame<dataframe>`
        A dataframe containing the curtailment in kW per generator
        of the provided 'feedin' dataframe per timestep.
        The index is a :pandas:`pandas.DateTimeIndex` and the
        columns are :py:mod:`edisgo.network.GeneratorFluctuating`
        objects.
    """

    # curtailment calculation by inducing a reduced or increased feedin
    # find out the difference from lower threshold
    # add a tilt factor of zero initially
    tilt_factor = pd.Series(0, index=voltage_threshold.index.copy())
    curtailment = _calculate_weighted_curtailment(feedin, voltage_pu, total_curtailment,
                                                  voltage_threshold, tilt_factor)

    # make sure the no single generator overshoots its feed-in,
    # if it does, introduce a small tilt in the voltage characteristic iteratively
    # to allow higher curtailment at nodes with lower voltages to ensure that
    # lower curtailment is assigned to nodes at higher voltages

    while not ((feedin - curtailment) > -1e-3).all().all():
        # The actual algorithm is below this break
        # if the tilt factor is greater than 1.0 then curtailment voltage through
        # this algorithm has failed due to low feed-in and high curtailment requirement
        # in this case curtail_all algorithm will be used
        if tilt_factor.any() >= 1.0:
            logging.warning("Maximum tilt reached, doing curtail_all")
            # NOTE: this curtail all uses the zeroed out feedin at the voltages below
            # voltage threshold
            curtailment = feedin.divide(feedin.sum(axis=1), axis=0). \
                multiply(total_curtailment, axis=0)
            curtailment.fillna(0, inplace=True)

            # need to check again inside the if, if the curtailment overshoots feedin
            # if it does then use the non-zeroed out feedin below voltage_threshold
            if not ((feedin - curtailment) > -1e-3).all().all():
                curtailment = feedin_original.divide(feedin_original.sum(axis=1), axis=0). \
                    multiply(total_curtailment, axis=0)
                curtailment.fillna(0, inplace=True)
            break
        # increase the tilt factor in small amounts at the indices where there is an
        # over shoot of feedin
        for ts in curtailment[(feedin - curtailment) <= -1e-3].dropna(axis=1).index:
            tilt_factor.loc[ts] += 1e-4
        curtailment = _calculate_weighted_curtailment(feedin, voltage_pu, total_curtailment,
                                                      voltage_threshold, tilt_factor)

    # check if curtailment exceeds feedin
    # check if overall curtailment
    if ((abs(curtailment.sum(axis=1) - total_curtailment)) >= 1e-3).any():
        raise ValueError("Total_curtailment sum does not match the assigned curtailment")

    return curtailment


def _calculate_weighted_curtailment(feedin, voltage_pu,
                                    total_curtailment,
                                    voltage_threshold,
                                    tilt_factor):
    """
    The raw calculation method to obtain the weighted curtailment
    without checking if the individual generator feedins are
    overshot. This check happens in the function calling this
    method in :py:mod:`edisgo.flex_opt.curtailment._weighted_curtail_voltage`.
    Parameters
    ----------
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        This contains the feedin of the specific set
        of generators that either:

        * of no specific grouping if curtailment is provided
          as a :pandas:`pandas.Series<series>`
        * of a given type if curtailment is provided
          as a :pandas:`pandas.DataFrame<dataframe`
        * of a give type and in a given weather cell if
          curtailment is provided as a :pandas:`pandas.DataFrame<dataframe>`
          with a :pandas.`pandas.MultiIndex` column with two levels, type
          and weather cell ID.

        The feedin that is expected as an input here already has the
        columns corresponding to the 'not-to-be-curtailed' generators
        set to zero so that no curtailment is assigned to them.
        The index is a time index and thecolumns are generators
        that will be curtailed.
        The feed-in is in kW.
    voltage_pu : :pandas:`pandas.DataFrame<dataframe>
        The dataframe that contains the voltages at the
        various generator nodes after doing a power flow calculation.
        This stucture's index is a :pandas:`pandas.DateTimeIndex`
        and the columns are generators.
        The voltage from power flow in p.u.
    total_curtailment : :pandas:`pandas.Series<series>`
        The series containing the specific curtailment for the type of
        generators to be curtailed and the weather cells they belong to.
        This is the specific column of the input curtailment.
        The index is a :pandas:`pandas.DateTimeIndex` and the
        total curtailment in time step in kW
    voltage_threshold : :pandas:`pandas.Series<series>`
        A series containing the voltage thresholds below which no
        generator curtailment will occur.The series contains the
        threshold calculated for each and every timestep that is
        being calculated. The index is a :pandas:`pandas.DateTimeIndex`,
        and the voltage threshold in time step in p.u.
    tilt_factor: :pandas:`pandas.Series<series>`
        The tilt factor per time step controls the size of the kink
        introduced between the 'not-to-be-curtailed' generators and
        the generators lying just above the voltage_threshold. This
        effectively increases the curtailment assigned to these
        generators. The slope of the voltage characterisitic is hence
        changed. The factor is bounded to a value between 0 and 1.
        In the calculation routines, if this value becomes greater
        than 1, the curtail_all strategy is used.
        The index is a :pandas:`pandas.DateTimeIndex`,
        and the tilt factor is a value between 0 and 1.

    Return
    ------
    :pandas:`pandas:DataFrame<dataframe>`
        A dataframe containing the curtailment in kW per generator
        of the provided 'feedin' dataframe per timestep.
        The index is a :pandas:`pandas.DateTimeIndex` and the
        columns are :py:mod:`edisgo.network.GeneratorFluctuating`
        objects.
    """
    feedin_factor = voltage_pu.subtract(voltage_threshold, axis='rows').add(tilt_factor, axis='rows')
    # make sure the difference is positive
    # after being normalized to maximum difference being 1 and minimum being 0
    feedin_factor = feedin_factor.divide(feedin_factor.max(axis=1), axis=0)
    # the curtailment here would be directly multplied the difference

    # multiply feedin_factor to feedin to modify the amount of curtailment
    modified_feedin = feedin_factor.multiply(feedin)

    # normalize the feedin
    normalized_feedin = modified_feedin.divide(modified_feedin.sum(axis=1), axis=0)
    # fill the nans with zeros typically filling in the x/0 cases with 0
    normalized_feedin.fillna(0, inplace=True)
    # total_curtailment
    curtailment = normalized_feedin.multiply(total_curtailment, axis=0)
    return curtailment


def curtail_all(feedin, generators, total_curtailment, edisgo,
                assigned_curtailment_key,
                    **kwargs):
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
    generators: :pandas:`pandas.DataFrame<dataframe>`
        This contains a dataframe of all the generators selected for the assignment
        of the curtailment. The typical structure of this dataframe can be obtained
        from :py:mod:`edigo.grid.tools.get_gen_info`. The stucture essentially
        contains 5 columns namely:

        * 'gen_repr': The repr string of the generator with the asset name and the asset id
        * 'type': the generator type, e. g. 'solar' or 'wind' typically
        * 'voltage_level': the voltage level, either 'mv' or 'lv'
        * 'nominal_capacity': the nominal capacity of the generator
        * 'weather_cell_id': the id of the weather cell the generator is located in.

    total_curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        The curtailment to be distributed amongst the generators in the
        edisgo_objects' network. This is input through the edisgo_object.
        See class definition for further information.
    edisgo : :class:`edisgo.grid.network.EDisGo`
        The edisgo object in which this function was called through the
        respective :class:`edisgo.grid.network.CurtailmentControl` instance.
    assigned_curtailment_key: :obj:`tuple` or :obj:`str`
        The type and weather cell ID if :obj:`tuple` or only
        the type if :obj:`str` in which the generators are
        being curtailed. This is used to separate the
        resulting assigned curtailment dataframes in the
        :class:`edisgo.grid.network.Results` objects
        accordingly.
    """
    # total_curtailment
    curtailment = feedin.divide(feedin.sum(axis=1), axis=0). \
        multiply(total_curtailment, axis=0)

    # make sure that the feedin isn't zero, as if it is
    # dividing by zero makes a lot of Nans which makes it harder
    # to check the curtailment correctly
    # just fillna with 0s
    curtailment.fillna(0, inplace=True)

    # assign curtailment to individual generators
    assign_curtailment(curtailment, edisgo, generators, assigned_curtailment_key)


def assign_curtailment(curtailment, edisgo, generators, assigned_curtailment_key):
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
    assigned_curtailment_key: :obj:`tuple` or :obj:`str`
        The type and weather cell ID if :obj:`tuple` or only
        the type if :obj:`str` in which the generators are
        being curtailed. This is used to separate the
        resulting assigned curtailment dataframes in the
        :class:`edisgo.grid.network.Results` objects
        accordingly.
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
        edisgo.network.results._assigned_curtailment[assigned_curtailment_key] = \
            gen_object_list
    else:
        # if gen_object_list isn't copied here, then
        # the object is saved in _curtailment
        # and in edisgo.network.results._assigned_curtailment[1st_key]
        # and after every _curtailment.extend,
        # the object simply get added to the ._assigned_curtailment[1st_key]
        # which is bad
        edisgo.network.timeseries._curtailment = gen_object_list
        edisgo.network.results._assigned_curtailment = \
            {assigned_curtailment_key: gen_object_list.copy()}

