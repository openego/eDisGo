import pandas as pd
import logging

from pyomo.environ import ConcreteModel, Set, Param, Objective, Constraint, \
    minimize, Var
from pyomo.opt import SolverFactory


def voltage_based(feedin, generators, curtailment_timeseries, edisgo,
                  curtailment_key, **kwargs):
    """
    Implements curtailment methodology 'voltage-based'.

    The curtailment that has to be met in each step is allocated depending on
    the voltage at the nodes of the generators.

    In a first step it is for each time step checked whether the required
    curtailment can be met by all generators with node voltages above the
    specified voltage threshold. If this is not the case the voltage threshold
    is lowered in steps of 0.01 p.u. until curtailment demand can be met.

    In a second step the curtailment demand is allocated to all generators with
    node voltages above the defined voltage threshold. Generators with node
    voltages below the threshold will not be curtailed. Above the voltage
    threshold, the curtailment is proportional to the difference between the
    node voltage and the voltage threshold. Thus the higher the voltage, the
    higher the curtailment. In order to find the linear relation between
    the curtailment and the voltage difference a linear problem is formulated
    and solved using the python package pyomo. See documentation for further
    information.

    Parameters
    ----------
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe holding the feed-in of each generator in kW for the
        technology (and weather cell) specified in `curtailment_key` parameter.
        Index of the dataframe is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`. Columns are the
        representatives of the fluctuating generators.
    generators : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with all generators of the type (and in weather cell)
        specified in `curtailment_key` parameter. See return value of
        :func:`edisgo.grid.tools.get_gen_info` for more information.
    curtailment_timeseries : :pandas:`pandas.Series<series>`
        The curtailment in kW to be distributed amongst the generators in
        `generators` parameter. Index of the series is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    edisgo : :class:`edisgo.grid.network.EDisGo`
    curtailment_key : :obj:`str` or :obj:`tuple` with :obj:`str`
        The technology and weather cell ID if :obj:`tuple` or only
        the technology if :obj:`str` the curtailment is specified for.
    voltage_threshold: :obj:`float`
        The node voltage below which no curtailment is assigned to the
        respective generator if not necessary. Default: 1.0.
    solver: :obj:`str`
        The solver used to optimize the curtailment assigned to the generator.
        Possible options are:

        * 'cbc'
          coin-or branch and cut solver
        * 'glpk'
          gnu linear programming kit solver
        * any other available compatible with 'pyomo' like 'gurobi'
          or 'cplex'

        Default: 'cbc'

    """

    voltage_threshold = pd.Series(kwargs.get('voltage_threshold', 0.0),
                                  index=curtailment_timeseries.index)
    solver = kwargs.get('solver', 'cbc')
    combined_analysis = kwargs.get('combined_analysis', False)

    # get the voltages at the generators
    voltages_lv_gens = edisgo.network.results.v_res(
        nodes=generators.loc[(generators.voltage_level == 'lv')].index,
        level='lv')
    voltages_mv_gens = edisgo.network.results.v_res(
        nodes=generators.loc[(generators.voltage_level == 'mv')].index,
        level='mv')
    voltages_gens = voltages_lv_gens.join(voltages_mv_gens)

    # get voltages at stations
    grids = list(set(generators.grid))
    lv_stations = [_.station for _ in grids if 'LVStation' in repr(_.station)]
    voltage_lv_stations = edisgo.network.results.v_res(
        nodes=lv_stations, level='lv')
    voltages_mv_station = edisgo.network.results.v_res(
        nodes=[edisgo.network.mv_grid.station], level='mv')
    voltages_stations = voltage_lv_stations.join(voltages_mv_station)

    # get allowed voltage deviations
    if not combined_analysis:
        allowed_voltage_dev_mv = edisgo.network.config[
            'grid_expansion_allowed_voltage_deviations'][
            'mv_feedin_case_max_v_deviation']
        allowed_voltage_diff_lv = edisgo.network.config[
            'grid_expansion_allowed_voltage_deviations'][
            'lv_feedin_case_max_v_deviation']
    else:
        allowed_voltage_dev_mv = edisgo.network.config[
            'grid_expansion_allowed_voltage_deviations'][
            'mv_lv_feedin_case_max_v_deviation']
        allowed_voltage_diff_lv = edisgo.network.config[
            'grid_expansion_allowed_voltage_deviations'][
            'mv_lv_feedin_case_max_v_deviation']
    generators['allowed_voltage_dev'] = generators.voltage_level.apply(
        lambda _: allowed_voltage_diff_lv if _ == 'lv'
        else allowed_voltage_dev_mv)

    # calculate voltage difference from generator node to station
    voltage_gens_diff = pd.DataFrame()
    for gen in voltages_gens.columns:
        station = generators[generators.gen_repr==gen].grid.values[0].station
        voltage_gens_diff[gen] = voltages_gens.loc[:, gen] - \
                                 voltages_stations.loc[:, repr(station)] - \
                                 generators[generators.gen_repr ==
                                            gen].allowed_voltage_dev.iloc[0]

    # for every time step check if curtailment can be fulfilled, otherwise
    # reduce voltage threshold; set feed-in of generators below voltage
    # threshold to zero, so that they cannot be curtailed
    for ts in curtailment_timeseries.index:
        # get generators with voltage higher than threshold
        gen_pool = voltage_gens_diff.loc[
            ts, voltage_gens_diff.loc[ts, :] > voltage_threshold.loc[ts]].index
        # if curtailment cannot be fulfilled lower voltage threshold
        while sum(feedin.loc[ts, gen_pool]) < curtailment_timeseries.loc[ts]:
            voltage_threshold.loc[ts] = voltage_threshold.loc[ts] - 0.01
            gen_pool = voltage_gens_diff.loc[
                ts, voltage_gens_diff.loc[ts, :] >
                voltage_threshold.loc[ts]].index
        # set feed-in of generators below voltage threshold to zero, so that
        # they cannot be curtailed
        gen_pool_out = voltage_gens_diff.loc[
            ts, voltage_gens_diff.loc[ts, :] <=
            voltage_threshold.loc[ts]].index
        feedin.loc[ts, gen_pool_out] = 0

    # only optimize for time steps where curtailment is greater than zero
    timeindex = curtailment_timeseries[curtailment_timeseries > 0].index
    if not timeindex.empty:
        curtailment = _optimize_voltage_based_curtailment(
            feedin, voltage_gens_diff, curtailment_timeseries,
            voltage_threshold, timeindex, solver)
    else:
        curtailment = pd.DataFrame()

    # set curtailment for other time steps to zero
    curtailment = curtailment.append(pd.DataFrame(
        0, columns=feedin.columns, index=curtailment_timeseries[
            curtailment_timeseries <= 0].index))

    # check if curtailment target was met
    _check_curtailment_target(curtailment, curtailment_timeseries,
                              curtailment_key)

    # assign curtailment to individual generators
    _assign_curtailment(curtailment, edisgo, generators, curtailment_key)


def _optimize_voltage_based_curtailment(feedin, voltage_pu, total_curtailment,
                                        voltage_threshold, timeindex, solver):
    """
    Formulates and solves linear problem to find linear relation between
    curtailment and node voltage.

    Parameters
    ------------
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        See `feedin` parameter in
        :func:`edisgo.flex_opt.curtailment.voltage_based` for more information.
    voltage_pu : :pandas:`pandas.DataFrame<dataframe>
        Dataframe containing voltages in p.u. at the generator nodes. Index
        of the dataframe is a :pandas:`pandas.DatetimeIndex<datetimeindex>`,
        columns are the generator representatives.
    total_curtailment : :pandas:`pandas.Series<series>`
        Series containing the specific curtailment in kW to be allocated to the
        generators. The index is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    voltage_threshold : :pandas:`pandas.Series<series>`
        Series containing the voltage thresholds in p.u. below which no
        generator curtailment will occur. The index is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    solver : :obj:`str`
        The solver used to optimize the linear problem. Default: 'cbc'.

    Returns
    -------
    :pandas:`pandas:DataFrame<dataframe>`
        Dataframe containing the curtailment in kW per generator and time step
        feed-in was provided for in `feedin` parameter. Index is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`, columns are the
        generator representatives.

    """

    logging.debug("Start curtailment optimization.")
  
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

    # initialize model
    model = ConcreteModel()

    # add sets
    model.T = Set(initialize=timeindex)
    model.G = Set(initialize=generators)

    # add parameters
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

    # add variables
    model.offset = Var(model.T, bounds=(0, 1))
    model.cf_max = Var(model.T, bounds=(0, 1))

    def curtailment_init(model, t, g):
        return (0, feedin.loc[t, g])

    model.c = Var(model.T, model.G, bounds=curtailment_init)

    # add objective
    def obj_rule(model):
        expr = (sum(model.offset[t] * 100
                    for t in model.T))
        return expr

    model.obj = Objective(rule=obj_rule, sense=minimize)

    # add constraints
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
    results = solver.solve(model, tee=False)

    # load results back into model
    model.solutions.load_from(results)

    return pd.DataFrame({g: [model.c[t, g].value for t in model.T]
                         for g in model.G}, index=model.T.value)


def feedin_proportional(feedin, generators, curtailment_timeseries, edisgo,
                        curtailment_key, **kwargs):
    """
    Implements curtailment methodology 'feedin-proportional'.

    The curtailment that has to be met in each time step is allocated
    equally to all generators depending on their share of total
    feed-in in that time step.

    Parameters
    ----------
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe holding the feed-in of each generator in kW for the
        technology (and weather cell) specified in `curtailment_key` parameter.
        Index of the dataframe is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`. Columns are the
        representatives of the fluctuating generators.
    generators : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with all generators of the type (and in weather cell)
        specified in `curtailment_key` parameter. See return value of
        :func:`edisgo.grid.tools.get_gen_info` for more information.
    curtailment_timeseries : :pandas:`pandas.Series<series>`
        The curtailment in kW to be distributed amongst the generators in
        `generators` parameter. Index of the series is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    edisgo : :class:`edisgo.grid.network.EDisGo`
    curtailment_key::obj:`str` or :obj:`tuple` with :obj:`str`
        The technology and weather cell ID if :obj:`tuple` or only
        the technology if :obj:`str` the curtailment is specified for.

    """
    # calculate curtailment in each time step of each generator
    curtailment = feedin.divide(feedin.sum(axis=1), axis=0). \
        multiply(curtailment_timeseries, axis=0)

    # substitute NaNs from division with 0 by 0
    curtailment.fillna(0, inplace=True)

    # check if curtailment target was met
    _check_curtailment_target(curtailment, curtailment_timeseries,
                              curtailment_key)

    # assign curtailment to individual generators
    _assign_curtailment(curtailment, edisgo, generators, curtailment_key)


def _check_curtailment_target(curtailment, curtailment_target,
                              curtailment_key):
    """
    Raises an error if curtailment target was not met in any time step.

    Parameters
    -----------
    curtailment : :pandas:`pandas:DataFrame<dataframe>`
        Dataframe containing the curtailment in kW per generator and time step.
        Index is a :pandas:`pandas.DatetimeIndex<datetimeindex>`, columns are
        the generator representatives.
    curtailment_target : :pandas:`pandas.Series<series>`
        The curtailment in kW that was to be distributed amongst the
        generators. Index of the series is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    curtailment_key : :obj:`str` or :obj:`tuple` with :obj:`str`
        The technology and weather cell ID if :obj:`tuple` or only
        the technology if :obj:`str` the curtailment was specified for.

    """
    if not (abs(curtailment.sum(axis=1) - curtailment_target) < 1e-3).all():
        message = 'Curtailment target not met for {}.'.format(curtailment_key)
        logging.error(message)
        raise TypeError(message)


def _assign_curtailment(curtailment, edisgo, generators, curtailment_key):
    """
    Helper function to write curtailment time series to generator objects.

    This function also writes a list of the curtailed generators to curtailment
    in :class:`edisgo.grid.network.TimeSeries` and
    :class:`edisgo.grid.network.Results`.

    Parameters
    ----------
    curtailment : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing the curtailment in kW per generator and time step
        for all generators of the type (and in weather cell) specified in
        `curtailment_key` parameter. Index is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`, columns are the
        generator representatives.
    edisgo : :class:`edisgo.grid.network.EDisGo`
    generators : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe with all generators of the type (and in weather cell)
        specified in `curtailment_key` parameter. See return value of
        :func:`edisgo.grid.tools.get_gen_info` for more information.
    curtailment_key : :obj:`str` or :obj:`tuple` with :obj:`str`
        The technology and weather cell ID if :obj:`tuple` or only
        the technology if :obj:`str` the curtailment is specified for.

    """

    gen_object_list = []
    for gen in curtailment.columns:
        # get generator object from representative
        gen_object = generators.loc[generators.gen_repr == gen].index[0]
        # assign curtailment to individual generators
        gen_object.curtailment = curtailment.loc[:, gen]
        gen_object_list.append(gen_object)

    # set timeseries.curtailment
    if edisgo.network.timeseries._curtailment:
        edisgo.network.timeseries._curtailment.extend(gen_object_list)
        edisgo.network.results._curtailment[curtailment_key] = \
            gen_object_list
    else:
        edisgo.network.timeseries._curtailment = gen_object_list
        # list needs to be copied, otherwise it will be extended every time
        # a new key is added to results._curtailment
        edisgo.network.results._curtailment = \
            {curtailment_key: gen_object_list.copy()}
