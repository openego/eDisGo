import pandas as pd
import logging

from pyomo.environ import (
    ConcreteModel,
    Set,
    Param,
    Objective,
    Constraint,
    minimize,
    Var,
)
from pyomo.opt import SolverFactory

from edisgo.io import pypsa_io


def voltage_based(
    feedin,
    generators,
    curtailment_timeseries,
    edisgo,
    curtailment_key,
    **kwargs
):
    """
    Implements curtailment methodology 'voltage-based'.

    ToDo: adapt to refactored code!

    The curtailment that has to be met in each time step is allocated depending
    on the exceedance of the allowed voltage deviation at the nodes of the
    generators. The higher the exceedance, the higher the curtailment.

    The optional parameter `voltage_threshold` specifies the threshold for the
    exceedance of the allowed voltage deviation above which a generator is
    curtailed. By default it is set to zero, meaning that all generators at
    nodes with voltage deviations that exceed the allowed voltage deviation are
    curtailed. Generators at nodes where the allowed voltage deviation is not
    exceeded are not curtailed. In the case that the required curtailment
    exceeds the weather-dependent availability of all generators with voltage
    deviations above the specified threshold, the voltage threshold is lowered
    in steps of 0.01 p.u. until the curtailment target can be met.

    Above the threshold, the curtailment is proportional to the exceedance of
    the allowed voltage deviation. In order to find the linear relation between
    the curtailment and the voltage difference a linear problem is formulated
    and solved using the python package pyomo. See documentation for further
    information.

    Parameters
    ----------
    feedin : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe holding the feed-in of each generator in kW for the
        technology (and weather cell) specified in `curtailment_key` parameter.
        Index of the dataframe is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`. Columns are the
        representatives of the fluctuating generators.
    generators : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with all generators of the type (and in weather cell)
        specified in `curtailment_key` parameter. See return value of
        :func:`edisgo.network.tools.get_gen_info` for more information.
    curtailment_timeseries : :pandas:`pandas.Series<Series>`
        The curtailment in kW to be distributed amongst the generators in
        `generators` parameter. Index of the series is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
    edisgo : :class:`~.edisgo.EDisGo`
    curtailment_key : :obj:`str` or :obj:`tuple` with :obj:`str`
        The technology and weather cell ID if :obj:`tuple` or only
        the technology if :obj:`str` the curtailment is specified for.
    voltage_threshold: :obj:`float`
        The node voltage below which no curtailment is assigned to the
        respective generator if not necessary. Default: 0.0.
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

    raise NotImplementedError

    voltage_threshold = pd.Series(
        kwargs.get("voltage_threshold", 0.0),
        index=curtailment_timeseries.index,
    )
    solver = kwargs.get("solver", "cbc")
    combined_analysis = kwargs.get("combined_analysis", False)

    # get the voltages at the generators
    if not edisgo.network.pypsa.edisgo_mode:
        voltages_lv_gens = edisgo.network.results.v_res(
            nodes_df=generators.loc[(generators.voltage_level == "lv")].index,
            level="lv",
        )
    else:
        # if only MV topology was analyzed (edisgo_mode = 'mv') all LV
        # generators are assigned the voltage at the corresponding station's
        # primary side
        lv_gens = generators[generators.voltage_level == "lv"]
        voltages_lv_stations = edisgo.network.results.v_res(
            nodes_df=[_.station for _ in lv_gens.grid.unique()], level="mv"
        )
        voltages_lv_gens = pd.DataFrame()
        for lv_gen in lv_gens.index:
            voltages_lv_gens[repr(lv_gen)] = voltages_lv_stations[
                repr(lv_gen.grid.station)
            ]
    voltages_mv_gens = edisgo.network.results.v_res(
        nodes_df=generators.loc[(generators.voltage_level == "mv")].index,
        level="mv",
    )
    voltages_gens = voltages_lv_gens.join(voltages_mv_gens)

    # get allowed voltage deviations from config
    if not combined_analysis:
        allowed_voltage_dev_mv = edisgo.network.config[
            "grid_expansion_allowed_voltage_deviations"
        ]["mv_feedin_case_max_v_deviation"]
        allowed_voltage_diff_lv = edisgo.network.config[
            "grid_expansion_allowed_voltage_deviations"
        ]["lv_feedin_case_max_v_deviation"]
    else:
        allowed_voltage_dev_mv = edisgo.network.config[
            "grid_expansion_allowed_voltage_deviations"
        ]["mv_lv_feedin_case_max_v_deviation"]
        allowed_voltage_diff_lv = edisgo.network.config[
            "grid_expansion_allowed_voltage_deviations"
        ]["mv_lv_feedin_case_max_v_deviation"]

    # assign allowed voltage deviation to each generator
    if not edisgo.network.pypsa.edisgo_mode:
        # for edisgo_mode = None

        # get voltages at stations
        grids = list(set(generators.grid))
        lv_stations = [
            _.station for _ in grids if "LVStation" in repr(_.station)
        ]
        voltage_lv_stations = edisgo.network.results.v_res(
            nodes_df=lv_stations, level="lv"
        )
        voltages_mv_station = edisgo.network.results.v_res(
            nodes_df=[edisgo.network.mv_grid.station], level="mv"
        )
        voltages_stations = voltage_lv_stations.join(voltages_mv_station)

        # assign allowed voltage deviation
        generators["allowed_voltage_dev"] = generators.voltage_level.apply(
            lambda _: allowed_voltage_diff_lv
            if _ == "lv"
            else allowed_voltage_dev_mv
        )

        # calculate voltage difference from generator node to station
        voltage_gens_diff = pd.DataFrame()
        for gen in voltages_gens.columns:
            station = (
                generators[generators.gen_repr == gen].grid.values[0].station
            )
            voltage_gens_diff[gen] = (
                voltages_gens.loc[:, gen]
                - voltages_stations.loc[:, repr(station)]
                - generators[
                    generators.gen_repr == gen
                ].allowed_voltage_dev.iloc[0]
            )

    else:
        # for edisgo_mode = 'mv'

        station = edisgo.network.mv_grid.station
        # get voltages at HV/MV station
        voltages_station = edisgo.network.results.v_res(
            nodes_df=[station], level="mv"
        )

        # assign allowed voltage deviation
        generators["allowed_voltage_dev"] = allowed_voltage_dev_mv

        # calculate voltage difference from generator node to station
        voltage_gens_diff = pd.DataFrame()
        for gen in voltages_gens.columns:
            voltage_gens_diff[gen] = (
                voltages_gens.loc[:, gen]
                - voltages_station.loc[:, repr(station)]
                - generators[
                    generators.gen_repr == gen
                ].allowed_voltage_dev.iloc[0]
            )

    # for every time step check if curtailment can be fulfilled, otherwise
    # reduce voltage threshold; set feed-in of generators below voltage
    # threshold to zero, so that they cannot be curtailed
    for ts in curtailment_timeseries.index:
        # get generators with voltage higher than threshold
        gen_pool = voltage_gens_diff.loc[
            ts, voltage_gens_diff.loc[ts, :] > voltage_threshold.loc[ts]
        ].index
        # if curtailment cannot be fulfilled lower voltage threshold
        while sum(feedin.loc[ts, gen_pool]) < curtailment_timeseries.loc[ts]:
            voltage_threshold.loc[ts] = voltage_threshold.loc[ts] - 0.01
            gen_pool = voltage_gens_diff.loc[
                ts, voltage_gens_diff.loc[ts, :] > voltage_threshold.loc[ts]
            ].index
        # set feed-in of generators below voltage threshold to zero, so that
        # they cannot be curtailed
        gen_pool_out = voltage_gens_diff.loc[
            ts, voltage_gens_diff.loc[ts, :] <= voltage_threshold.loc[ts]
        ].index
        feedin.loc[ts, gen_pool_out] = 0

    # only optimize for time steps where curtailment is greater than zero
    timeindex = curtailment_timeseries[curtailment_timeseries > 0].index
    if not timeindex.empty:
        curtailment = _optimize_voltage_based_curtailment(
            feedin,
            voltage_gens_diff,
            curtailment_timeseries,
            voltage_threshold,
            timeindex,
            solver,
        )
    else:
        curtailment = pd.DataFrame()

    # set curtailment for other time steps to zero
    curtailment = curtailment.append(
        pd.DataFrame(
            0,
            columns=feedin.columns,
            index=curtailment_timeseries[curtailment_timeseries <= 0].index,
        )
    )

    # check if curtailment target was met
    _check_curtailment_target(
        curtailment, curtailment_timeseries, curtailment_key
    )

    # assign curtailment to individual generators
    _assign_curtailment(curtailment, edisgo, generators, curtailment_key)


def _optimize_voltage_based_curtailment(
    feedin, voltage_pu, total_curtailment, voltage_threshold, timeindex, solver
):
    """
    Formulates and solves linear problem to find linear relation between
    curtailment and node voltage.

    ToDo: adapt to refactored code!

    Parameters
    ------------
    feedin : :pandas:`pandas.DataFrame<DataFrame>`
        See `feedin` parameter in
        :func:`edisgo.flex_opt.curtailment.voltage_based` for more information.
    voltage_pu : :pandas:`pandas.DataFrame<DataFrame>
        Dataframe containing voltages in p.u. at the generator nodes. Index
        of the dataframe is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`,
        columns are the generator representatives.
    total_curtailment : :pandas:`pandas.Series<Series>`
        Series containing the specific curtailment in kW to be allocated to the
        generators. The index is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
    voltage_threshold : :pandas:`pandas.Series<Series>`
        Series containing the voltage thresholds in p.u. below which no
        generator curtailment will occur. The index is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
    solver : :obj:`str`
        The solver used to optimize the linear problem. Default: 'cbc'.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the curtailment in kW per generator and time step
        feed-in was provided for in `feedin` parameter. Index is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`, columns are the
        generator representatives.

    """

    raise NotImplementedError

    logging.debug("Start curtailment optimization.")

    v_max = voltage_pu.max(axis=1)
    generators = feedin.columns

    # additional curtailment factors
    cf_add = pd.DataFrame(index=timeindex)
    for gen in generators:
        cf_add[gen] = abs(
            (voltage_pu.loc[timeindex, gen] - v_max[timeindex])
            / (voltage_threshold[timeindex] - v_max[timeindex])
        )

    # curtailment factors
    cf = pd.DataFrame(index=timeindex)
    for gen in generators:
        cf[gen] = abs(
            (voltage_pu.loc[timeindex, gen] - voltage_threshold[timeindex])
            / (v_max[timeindex] - voltage_threshold[timeindex])
        )

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
        expr = sum(model.offset[t] * 100 for t in model.T)
        return expr

    model.obj = Objective(rule=obj_rule, sense=minimize)

    # add constraints
    # curtailment per generator constraints
    def curtail(model, t, g):
        return (
            model.cf[t, g] * model.cf_max[t] * model.feedin[t, g]
            + model.cf_add[t, g] * model.offset[t] * model.feedin[t, g]
            - model.c[t, g]
            == 0
        )

    model.curtailment = Constraint(model.T, model.G, rule=curtail)

    # total curtailment constraint
    def total_curtailment(model, t):
        return (
            sum(model.c[t, g] for g in model.G) == model.total_curtailment[t]
        )

    model.sum_curtailment = Constraint(model.T, rule=total_curtailment)

    # solve
    solver = SolverFactory(solver)
    results = solver.solve(model, tee=False)

    # load results back into model
    model.solutions.load_from(results)

    return pd.DataFrame(
        {g: [model.c[t, g].value for t in model.T] for g in model.G},
        index=model.T,
    )


def feedin_proportional(
    feedin,
    generators,
    curtailment_timeseries,
    edisgo,
    curtailment_key,
    **kwargs
):
    """
    Implements curtailment methodology 'feedin-proportional'.

    ToDo: adapt to refactored code!

    The curtailment that has to be met in each time step is allocated
    equally to all generators depending on their share of total
    feed-in in that time step.

    Parameters
    ----------
    feedin : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe holding the feed-in of each generator in kW for the
        technology (and weather cell) specified in `curtailment_key` parameter.
        Index of the dataframe is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`. Columns are the
        representatives of the fluctuating generators.
    generators : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with all generators of the type (and in weather cell)
        specified in `curtailment_key` parameter. See return value of
        :func:`edisgo.network.tools.get_gen_info` for more information.
    curtailment_timeseries : :pandas:`pandas.Series<Series>`
        The curtailment in kW to be distributed amongst the generators in
        `generators` parameter. Index of the series is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
    edisgo : :class:`~.edisgo.EDisGo`
    curtailment_key::obj:`str` or :obj:`tuple` with :obj:`str`
        The technology and weather cell ID if :obj:`tuple` or only
        the technology if :obj:`str` the curtailment is specified for.

    """
    raise NotImplementedError

    # calculate curtailment in each time step of each generator
    curtailment = feedin.divide(feedin.sum(axis=1), axis=0).multiply(
        curtailment_timeseries, axis=0
    )

    # substitute NaNs from division with 0 by 0
    curtailment.fillna(0, inplace=True)

    # check if curtailment target was met
    _check_curtailment_target(
        curtailment, curtailment_timeseries, curtailment_key
    )

    # assign curtailment to individual generators
    _assign_curtailment(curtailment, edisgo, generators, curtailment_key)


def _check_curtailment_target(
    curtailment, curtailment_target, curtailment_key
):
    """
    Raises an error if curtailment target was not met in any time step.

    ToDo: adapt to refactored code!

    Parameters
    -----------
    curtailment : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the curtailment in kW per generator and time step.
        Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`, columns are
        the generator representatives.
    curtailment_target : :pandas:`pandas.Series<Series>`
        The curtailment in kW that was to be distributed amongst the
        generators. Index of the series is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
    curtailment_key : :obj:`str` or :obj:`tuple` with :obj:`str`
        The technology and weather cell ID if :obj:`tuple` or only
        the technology if :obj:`str` the curtailment was specified for.

    """
    raise NotImplementedError

    if not (abs(curtailment.sum(axis=1) - curtailment_target) < 1e-1).all():
        message = "Curtailment target not met for {}.".format(curtailment_key)
        logging.error(message)
        raise TypeError(message)


def _assign_curtailment(curtailment, edisgo, generators, curtailment_key):
    """
    Helper function to write curtailment time series to generator objects.

    ToDo: adapt to refactored code!

    This function also writes a list of the curtailed generators to curtailment
    in :class:`edisgo.network.network.TimeSeries` and
    :class:`edisgo.network.network.Results`.

    Parameters
    ----------
    curtailment : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing the curtailment in kW per generator and time step
        for all generators of the type (and in weather cell) specified in
        `curtailment_key` parameter. Index is a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`, columns are the
        generator representatives.
    edisgo : :class:`~.edisgo.EDisGo`
    generators : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with all generators of the type (and in weather cell)
        specified in `curtailment_key` parameter. See return value of
        :func:`edisgo.network.tools.get_gen_info` for more information.
    curtailment_key : :obj:`str` or :obj:`tuple` with :obj:`str`
        The technology and weather cell ID if :obj:`tuple` or only
        the technology if :obj:`str` the curtailment is specified for.

    """
    raise NotImplementedError

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
        edisgo.network.results._curtailment[curtailment_key] = gen_object_list
    else:
        edisgo.network.timeseries._curtailment = gen_object_list
        # list needs to be copied, otherwise it will be extended every time
        # a new key is added to results._curtailment
        edisgo.network.results._curtailment = {
            curtailment_key: gen_object_list.copy()
        }


class CurtailmentControl:
    """
    Allocates given curtailment targets to solar and wind generators.

    ToDo: adapt to refactored code!

    Parameters
    ----------
    edisgo: :class:`edisgo.EDisGo`
        The parent EDisGo object that this instance is a part of.
    methodology : :obj:`str`
        Defines the curtailment strategy. Possible options are:

        * 'feedin-proportional'
          The curtailment that has to be met in each time step is allocated
          equally to all generators depending on their share of total
          feed-in in that time step. For more information see
          :func:`edisgo.flex_opt.curtailment.feedin_proportional`.
        * 'voltage-based'
          The curtailment that has to be met in each time step is allocated
          based on the voltages at the generator connection points and a
          defined voltage threshold. Generators at higher voltages
          are curtailed more. The default voltage threshold is 1.0 but
          can be changed by providing the argument 'voltage_threshold'. This
          method formulates the allocation of curtailment as a linear
          optimization problem using :py:mod:`Pyomo` and requires a linear
          programming solver like coin-or cbc (cbc) or gnu linear programming
          kit (glpk). The solver can be specified through the parameter
          'solver'. For more information see
          :func:`edisgo.flex_opt.curtailment.voltage_based`.

    curtailment_timeseries : :pandas:`pandas.Series<Series>` or \
        :pandas:`pandas.DataFrame<DataFrame>`, optional
        Series or DataFrame containing the curtailment time series in kW. Index
        needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
        Provide a Series if the curtailment time series applies to wind and
        solar generators. Provide a DataFrame if the curtailment time series
        applies to a specific technology and optionally weather cell. In the
        first case columns of the DataFrame are e.g. 'solar' and 'wind'; in the
        second case columns need to be a
        :pandas:`pandas.MultiIndex<multiindex>` with the first level containing
        the type and the second level the weather cell ID. Default: None.
    solver: :obj:`str`
        The solver used to optimize the curtailment assigned to the generators
        when 'voltage-based' curtailment methodology is chosen.
        Possible options are:

        * 'cbc'
        * 'glpk'
        * any other available solver compatible with 'pyomo' such as 'gurobi'
          or 'cplex'

        Default: 'cbc'.
    voltage_threshold : :obj:`float`
        Voltage below which no curtailment is assigned to the respective
        generator if not necessary when 'voltage-based' curtailment methodology
        is chosen. See :func:`edisgo.flex_opt.curtailment.voltage_based` for
        more information. Default: 1.0.
    mode : :obj:`str`
        The `mode` is only relevant for curtailment method 'voltage-based'.
        Possible options are None and 'mv'. Per default `mode` is None in which
        case a power flow is conducted for both the MV and LV. In case `mode`
        is set to 'mv' components in underlying LV grids are considered
        aggregative. Default: None.

    """

    # ToDo move some properties from topology here (e.g. peak_load, generators,...)
    def __init__(
        self, edisgo, methodology, curtailment_timeseries, mode=None, **kwargs
    ):

        raise NotImplementedError

        logging.info("Start curtailment methodology {}.".format(methodology))

        self._check_timeindex(curtailment_timeseries, edisgo.topology)

        if methodology == "feedin-proportional":
            curtailment_method = feedin_proportional
        elif methodology == "voltage-based":
            curtailment_method = voltage_based
        else:
            raise ValueError(
                "{} is not a valid curtailment methodology.".format(
                    methodology
                )
            )

        # check if provided mode is valid
        if mode and mode is not "mv":
            raise ValueError("Provided mode {} is not a valid mode.")

        # get all fluctuating generators and their attributes (weather ID,
        # type, etc.)
        generators = get_gen_info(edisgo.topology, "mvlv", fluctuating=True)

        # do analyze to get all voltages at generators and feed-in dataframe
        edisgo.analyze(mode=mode)

        # get feed-in time series of all generators
        if not mode:
            feedin = edisgo.topology.pypsa.generators_t.p * 1000
            # drop dispatchable generators and slack generator
            drop_labels = [
                _ for _ in feedin.columns if "GeneratorFluctuating" not in _
            ] + ["Generator_slack"]
        else:
            feedin = edisgo.topology.mv_grid.generators_timeseries()
            for grid in edisgo.topology.mv_grid.lv_grids:
                feedin = pd.concat(
                    [feedin, grid.generators_timeseries()], axis=1
                )
            feedin.rename(columns=lambda _: repr(_), inplace=True)
            # drop dispatchable generators
            drop_labels = [
                _ for _ in feedin.columns if "GeneratorFluctuating" not in _
            ]
        feedin.drop(labels=drop_labels, axis=1, inplace=True)

        if isinstance(curtailment_timeseries, pd.Series):
            # check if curtailment exceeds feed-in
            self._precheck(
                curtailment_timeseries, feedin, "all_fluctuating_generators"
            )

            # do curtailment
            curtailment_method(
                feedin,
                generators,
                curtailment_timeseries,
                edisgo,
                "all_fluctuating_generators",
                **kwargs
            )

        elif isinstance(curtailment_timeseries, pd.DataFrame):
            for col in curtailment_timeseries.columns:
                logging.debug("Calculating curtailment for {}".format(col))

                # filter generators
                if isinstance(curtailment_timeseries.columns, pd.MultiIndex):
                    selected_generators = generators.loc[
                        (generators.type == col[0])
                        & (generators.weather_cell_id == col[1])
                    ]
                else:
                    selected_generators = generators.loc[
                        (generators.type == col)
                    ]

                # check if curtailment exceeds feed-in
                feedin_selected_generators = feedin.loc[
                    :, selected_generators.gen_repr.values
                ]
                self._precheck(
                    curtailment_timeseries.loc[:, col],
                    feedin_selected_generators,
                    col,
                )

                # do curtailment
                if not feedin_selected_generators.empty:
                    curtailment_method(
                        feedin_selected_generators,
                        selected_generators,
                        curtailment_timeseries.loc[:, col],
                        edisgo,
                        col,
                        **kwargs
                    )

        # check if curtailment exceeds feed-in
        self._postcheck(edisgo.topology, feedin)

        # update generator time series in pypsa topology
        if edisgo.topology.pypsa is not None:
            pypsa_io.update_pypsa_generator_timeseries(edisgo.topology)

        # add measure to Results object
        edisgo.results.measures = "curtailment"

    def _check_timeindex(self, curtailment_timeseries, network):
        """
        Raises an error if time index of curtailment time series does not
        comply with the time index of load and feed-in time series.

        Parameters
        -----------
        curtailment_timeseries : :pandas:`pandas.Series<Series>` or \
            :pandas:`pandas.DataFrame<DataFrame>`
            See parameter `curtailment_timeseries` in class definition for more
            information.

        """
        raise NotImplementedError

        if curtailment_timeseries is None:
            message = "No curtailment given."
            logging.error(message)
            raise KeyError(message)
        try:
            curtailment_timeseries.loc[network.timeseries.timeindex]
        except:
            message = (
                "Time index of curtailment time series does not match "
                "with load and feed-in time series."
            )
            logging.error(message)
            raise KeyError(message)

    def _precheck(self, curtailment_timeseries, feedin_df, curtailment_key):
        """
        Raises an error if the curtailment at any time step exceeds the
        total feed-in of all generators curtailment can be distributed among
        at that time.

        Parameters
        -----------
        curtailment_timeseries : :pandas:`pandas.Series<Series>`
            Curtailment time series in kW for the technology (and weather
            cell) specified in `curtailment_key`.
        feedin_df : :pandas:`pandas.Series<Series>`
            Feed-in time series in kW for all generators of type (and in
            weather cell) specified in `curtailment_key`.
        curtailment_key : :obj:`str` or :obj:`tuple` with :obj:`str`
            Technology (and weather cell) curtailment is given for.

        """
        raise NotImplementedError

        if not feedin_df.empty:
            feedin_selected_sum = feedin_df.sum(axis=1)
            diff = feedin_selected_sum - curtailment_timeseries
            # add tolerance (set small negative values to zero)
            diff[diff.between(-1, 0)] = 0
            if not (diff >= 0).all():
                bad_time_steps = [_ for _ in diff.index if diff[_] < 0]
                message = (
                    "Curtailment demand exceeds total feed-in in time "
                    "steps {}.".format(bad_time_steps)
                )
                logging.error(message)
                raise ValueError(message)
        else:
            bad_time_steps = [
                _
                for _ in curtailment_timeseries.index
                if curtailment_timeseries[_] > 0
            ]
            if bad_time_steps:
                message = (
                    "Curtailment given for time steps {} but there "
                    "are no generators to meet the curtailment target "
                    "for {}.".format(bad_time_steps, curtailment_key)
                )
                logging.error(message)
                raise ValueError(message)

    def _postcheck(self, network, feedin):
        """
        Raises an error if the curtailment of a generator exceeds the
        feed-in of that generator at any time step.

        Parameters
        -----------
        network : :class:`~.network.topology.Topology`
        feedin : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with feed-in time series in kW. Columns of the dataframe
            are :class:`~.network.components.GeneratorFluctuating`, index is
            time index.

        """
        raise NotImplementedError

        curtailment = network.timeseries.curtailment
        gen_repr = [repr(_) for _ in curtailment.columns]
        feedin_repr = feedin.loc[:, gen_repr]
        curtailment_repr = curtailment
        curtailment_repr.columns = gen_repr
        if not ((feedin_repr - curtailment_repr) > -1e-1).all().all():
            message = "Curtailment exceeds feed-in."
            logging.error(message)
            raise TypeError(message)
