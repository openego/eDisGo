import pandas as pd
import logging
import csv


from edisgo.tools import pypsa_io
from edisgo.flex_opt import storage_integration, storage_operation, \
    curtailment, storage_positioning
from edisgo.network.components import Generator, Load
from edisgo.network.tools import get_gen_info


logger = logging.getLogger('edisgo')


class Network:
    """
    Used as container for all data related to a single
    :class:`~.network.grids.MVGrid`.

    Parameters
    ----------
    ding0_grid : :obj:`str`
        Path to directory containing csv files of network to be loaded.
    config_path : None or :obj:`str` or :obj:`dict`, optional
        See :class:`~.network.network.Config` for further information.
        Default: None.
    generator_scenario : :obj:`str`
        Defines which scenario of future generator park to use.

    Attributes
    -----------


    _grid_district : :obj:`dict`
        Contains the following information about the supplied
        region (network district) of the network:
        'geom': Shape of network district as MultiPolygon.
        'population': Number of inhabitants.
    _grids : dict
    generators_t : enthält auch curtailment dataframe (muss bei Erstellung von
        pypsa Netzwerk beachtet werden)

    """

    def __init__(self, **kwargs):

        self._generator_scenario = kwargs.get('generator_scenario', None)


    @property
    def buses_df(self):
        """
        Dataframe with all buses in MV network and underlying LV grids.

        Parameters
        ----------
        buses_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in MV network and underlying LV grids.
            Index of the dataframe are bus names. Columns of the dataframe are:
            v_nom
            x
            y
            mv_grid_id
            lv_grid_id
            in_building

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all buses in MV network and underlying LV grids.

        """
        return self._buses_df

    @buses_df.setter
    def buses_df(self, buses_df):
        self._buses_df = buses_df

    @property
    def generators_df(self):
        """
        Dataframe with all generators in MV network and underlying LV grids.

        Parameters
        ----------
        generators_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all generators in MV network and underlying LV grids.
            Index of the dataframe are generator names. Columns of the
            dataframe are:
            bus
            control
            p_nom
            type
            weather_cell_id	subtype

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all generators in MV network and underlying LV grids.
            Slack generator is excluded.

        """
        return self._generators_df.drop(labels=['Generator_slack'])

    @generators_df.setter
    def generators_df(self, generators_df):
        self._generators_df = generators_df

    @property
    def loads_df(self):
        """
        Dataframe with all loads in MV network and underlying LV grids.

        Parameters
        ----------
        loads_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all loads in MV network and underlying LV grids.
            Index of the dataframe are load names. Columns of the
            dataframe are:
            bus
            peak_load
            sector
            annual_consumption

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all loads in MV network and underlying LV grids.

        """
        return self._loads_df

    @loads_df.setter
    def loads_df(self, loads_df):
        self._loads_df = loads_df

    @property
    def transformers_df(self):
        """
        Dataframe with all transformers.

        Parameters
        ----------
        transformers_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all transformers.
            Index of the dataframe are transformer names. Columns of the
            dataframe are:
            bus0
            bus1
            x_pu
            r_pu
            s_nom
            type

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all transformers.

        """
        return self._transformers_df

    @transformers_df.setter
    def transformers_df(self, transformers_df):
        self._transformers_df = transformers_df

    @property
    def lines_df(self):
        """
        Dataframe with all lines in MV network and underlying LV grids.

        Parameters
        ----------
        lines_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all lines in MV network and underlying LV grids.
            Index of the dataframe are line names. Columns of the
            dataframe are:
            bus0
            bus1
            length
            x
            r
            s_nom
            num_parallel
            type

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all lines in MV network and underlying LV grids.

        """
        return self._lines_df

    @lines_df.setter
    def lines_df(self, lines_df):
        self._lines_df = lines_df

    @property
    def switches_df(self):
        """
        Dataframe with all switches in MV network and underlying LV grids.

        Parameters
        ----------
        switches_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all switches in MV network and underlying LV grids.
            Index of the dataframe are switch names. Columns of the
            dataframe are:
            bus_open
            bus_closed
            branch
            type

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all switches in MV network and underlying LV grids.

        """
        return self._switches_df

    @switches_df.setter
    def switches_df(self, switches_df):
        self._switches_df = switches_df

    @property
    def storages_df(self):
        """
        Dataframe with all storages in MV network and underlying LV grids.

        Parameters
        ----------
        storages_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all storages in MV network and underlying LV grids.
            Index of the dataframe are storage names. Columns of the
            dataframe are:
            bus
            control
            p_nom
            capacity
            efficiency_store
            efficiency_dispatch

        Returns
        --------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with all storages in MV network and underlying LV grids.

        """
        return self._storages_df

    @storages_df.setter
    def storages_df(self, storages_df):
        self._storages_df = storages_df

    @property
    def generators(self):
        """
        Connected generators within the network.

        Returns
        -------
        list(:class:`~.network.components.Generator`)
            List of generators within the network.

        """
        for gen in self.generators_df.drop(labels=['Generator_slack']).index:
            yield Generator(id=gen)

    @property
    def loads(self):
        """
        Connected loads within the network.

        Returns
        -------
        list(:class:`~.network.components.Load`)
            List of loads within the network.

        """
        for l in self.loads_df.index:
            yield Load(id=l)

    @property
    def id(self):
        """
        MV network ID

        Returns
        --------
        :obj:`str`
            MV network ID

        """

        return self.mv_grid.id

    @property
    def generator_scenario(self):
        """
        Defines which scenario of future generator park to use.

        Parameters
        ----------
        generator_scenario_name : :obj:`str`
            Name of scenario of future generator park

        Returns
        --------
        :obj:`str`
            Name of scenario of future generator park

        """
        return self._generator_scenario

    @generator_scenario.setter
    def generator_scenario(self, generator_scenario_name):
        self._generator_scenario = generator_scenario_name

    @property
    def mv_grid(self):
        """
        Medium voltage (MV) network

        Parameters
        ----------
        mv_grid : :class:`~.network.grids.MVGrid`
            Medium voltage (MV) network

        Returns
        --------
        :class:`~.network.grids.MVGrid`
            Medium voltage (MV) network

        """
        return self._mv_grid

    @mv_grid.setter
    def mv_grid(self, mv_grid):
        self._mv_grid = mv_grid

    @property
    def grid_district(self):
        """
        Medium voltage (MV) network

        Parameters
        ----------
        mv_grid : :class:`~.network.grids.MVGrid`
            Medium voltage (MV) network

        Returns
        --------
        :class:`~.network.grids.MVGrid`
            Medium voltage (MV) network

        """
        return self._grid_district

    @grid_district.setter
    def grid_district(self, grid_district):
        self._grid_district = grid_district
    #
    # @timeseries.setter
    # def timeseries(self, timeseries):
    #     self._timeseries = timeseries

    #ToDo still needed?
    # @property
    # def dingo_import_data(self):
    #     """
    #     Temporary data from ding0 import needed for OEP generator update
    #
    #     """
    #     return self._dingo_import_data
    #
    # @dingo_import_data.setter
    # def dingo_import_data(self, dingo_data):
    #     self._dingo_import_data = dingo_data



    def __repr__(self):
        return 'Network ' + str(self.id)


class CurtailmentControl:
    """
    Allocates given curtailment targets to solar and wind generators.

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

    curtailment_timeseries : :pandas:`pandas.Series<series>` or \
        :pandas:`pandas.DataFrame<dataframe>`, optional
        Series or DataFrame containing the curtailment time series in kW. Index
        needs to be a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
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
    # ToDo move some properties from network here (e.g. peak_load, generators,...)
    def __init__(self, edisgo, methodology, curtailment_timeseries, mode=None,
                 **kwargs):

        logging.info("Start curtailment methodology {}.".format(methodology))

        self._check_timeindex(curtailment_timeseries, edisgo.network)

        if methodology == 'feedin-proportional':
            curtailment_method = curtailment.feedin_proportional
        elif methodology == 'voltage-based':
            curtailment_method = curtailment.voltage_based
        else:
            raise ValueError(
                '{} is not a valid curtailment methodology.'.format(
                    methodology))

        # check if provided mode is valid
        if mode and mode is not 'mv':
            raise ValueError("Provided mode {} is not a valid mode.")

        # get all fluctuating generators and their attributes (weather ID,
        # type, etc.)
        generators = get_gen_info(edisgo.network, 'mvlv', fluctuating=True)

        # do analyze to get all voltages at generators and feed-in dataframe
        edisgo.analyze(mode=mode)

        # get feed-in time series of all generators
        if not mode:
            feedin = edisgo.network.pypsa.generators_t.p * 1000
            # drop dispatchable generators and slack generator
            drop_labels = [_ for _ in feedin.columns
                           if 'GeneratorFluctuating' not in _] \
                          + ['Generator_slack']
        else:
            feedin = edisgo.network.mv_grid.generators_timeseries()
            for grid in edisgo.network.mv_grid.lv_grids:
                feedin = pd.concat([feedin, grid.generators_timeseries()],
                                   axis=1)
            feedin.rename(columns=lambda _: repr(_), inplace=True)
            # drop dispatchable generators
            drop_labels = [_ for _ in feedin.columns
                           if 'GeneratorFluctuating' not in _]
        feedin.drop(labels=drop_labels, axis=1, inplace=True)

        if isinstance(curtailment_timeseries, pd.Series):
            # check if curtailment exceeds feed-in
            self._precheck(curtailment_timeseries, feedin,
                           'all_fluctuating_generators')

            # do curtailment
            curtailment_method(
                feedin, generators, curtailment_timeseries, edisgo,
                'all_fluctuating_generators', **kwargs)

        elif isinstance(curtailment_timeseries, pd.DataFrame):
            for col in curtailment_timeseries.columns:
                logging.debug('Calculating curtailment for {}'.format(col))

                # filter generators
                if isinstance(curtailment_timeseries.columns, pd.MultiIndex):
                    selected_generators = generators.loc[
                        (generators.type == col[0]) &
                        (generators.weather_cell_id == col[1])]
                else:
                    selected_generators = generators.loc[
                        (generators.type == col)]

                # check if curtailment exceeds feed-in
                feedin_selected_generators = \
                    feedin.loc[:, selected_generators.gen_repr.values]
                self._precheck(curtailment_timeseries.loc[:, col],
                               feedin_selected_generators, col)

                # do curtailment
                if not feedin_selected_generators.empty:
                    curtailment_method(
                        feedin_selected_generators, selected_generators,
                        curtailment_timeseries.loc[:, col], edisgo,
                        col, **kwargs)

        # check if curtailment exceeds feed-in
        self._postcheck(edisgo.network, feedin)

        # update generator time series in pypsa network
        if edisgo.network.pypsa is not None:
            pypsa_io.update_pypsa_generator_timeseries(edisgo.network)

        # add measure to Results object
        edisgo.results.measures = 'curtailment'

    def _check_timeindex(self, curtailment_timeseries, network):
        """
        Raises an error if time index of curtailment time series does not
        comply with the time index of load and feed-in time series.

        Parameters
        -----------
        curtailment_timeseries : :pandas:`pandas.Series<series>` or \
            :pandas:`pandas.DataFrame<dataframe>`
            See parameter `curtailment_timeseries` in class definition for more
            information.

        """
        if curtailment_timeseries is None:
            message = 'No curtailment given.'
            logging.error(message)
            raise KeyError(message)
        try:
            curtailment_timeseries.loc[network.timeseries.timeindex]
        except:
            message = 'Time index of curtailment time series does not match ' \
                      'with load and feed-in time series.'
            logging.error(message)
            raise KeyError(message)

    def _precheck(self, curtailment_timeseries, feedin_df, curtailment_key):
        """
        Raises an error if the curtailment at any time step exceeds the
        total feed-in of all generators curtailment can be distributed among
        at that time.

        Parameters
        -----------
        curtailment_timeseries : :pandas:`pandas.Series<series>`
            Curtailment time series in kW for the technology (and weather
            cell) specified in `curtailment_key`.
        feedin_df : :pandas:`pandas.Series<series>`
            Feed-in time series in kW for all generators of type (and in
            weather cell) specified in `curtailment_key`.
        curtailment_key : :obj:`str` or :obj:`tuple` with :obj:`str`
            Technology (and weather cell) curtailment is given for.

        """
        if not feedin_df.empty:
            feedin_selected_sum = feedin_df.sum(axis=1)
            diff = feedin_selected_sum - curtailment_timeseries
            # add tolerance (set small negative values to zero)
            diff[diff.between(-1, 0)] = 0
            if not (diff >= 0).all():
                bad_time_steps = [_ for _ in diff.index
                                  if diff[_] < 0]
                message = 'Curtailment demand exceeds total feed-in in time ' \
                          'steps {}.'.format(bad_time_steps)
                logging.error(message)
                raise ValueError(message)
        else:
            bad_time_steps = [_ for _ in curtailment_timeseries.index
                              if curtailment_timeseries[_] > 0]
            if bad_time_steps:
                message = 'Curtailment given for time steps {} but there ' \
                          'are no generators to meet the curtailment target ' \
                          'for {}.'.format(bad_time_steps, curtailment_key)
                logging.error(message)
                raise ValueError(message)

    def _postcheck(self, network, feedin):
        """
        Raises an error if the curtailment of a generator exceeds the
        feed-in of that generator at any time step.

        Parameters
        -----------
        network : :class:`~.network.network.Network`
        feedin : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with feed-in time series in kW. Columns of the dataframe
            are :class:`~.network.components.GeneratorFluctuating`, index is
            time index.

        """
        curtailment = network.timeseries.curtailment
        gen_repr = [repr(_) for _ in curtailment.columns]
        feedin_repr = feedin.loc[:, gen_repr]
        curtailment_repr = curtailment
        curtailment_repr.columns = gen_repr
        if not ((feedin_repr - curtailment_repr) > -1e-1).all().all():
            message = 'Curtailment exceeds feed-in.'
            logging.error(message)
            raise TypeError(message)


class StorageControl:
    """
    Integrates storages into the network.

    Parameters
    ----------
    edisgo : :class:`~.network.network.EDisGo`
    timeseries : :obj:`str` or :pandas:`pandas.Series<series>` or :obj:`dict`
        Parameter used to obtain time series of active power the
        storage(s) is/are charged (negative) or discharged (positive) with. Can
        either be a given time series or an operation strategy.
        Possible options are:

        * :pandas:`pandas.Series<series>`
          Time series the storage will be charged and discharged with can be
          set directly by providing a :pandas:`pandas.Series<series>` with
          time series of active charge (negative) and discharge (positive)
          power in kW. Index needs to be a
          :pandas:`pandas.DatetimeIndex<datetimeindex>`.
          If no nominal power for the storage is provided in
          `parameters` parameter, the maximum of the time series is
          used as nominal power.
          In case of more than one storage provide a :obj:`dict` where each
          entry represents a storage. Keys of the dictionary have to match
          the keys of the `parameters dictionary`, values must
          contain the corresponding time series as
          :pandas:`pandas.Series<series>`.
        * 'fifty-fifty'
          Storage operation depends on actual power of generators. If
          cumulative generation exceeds 50% of the nominal power, the storage
          will charge. Otherwise, the storage will discharge.
          If you choose this option you have to provide a nominal power for
          the storage. See `parameters` for more information.

        Default: None.
    position : None or :obj:`str` or :class:`~.network.components.Station` or :class:`~.network.components.BranchTee`  or :class:`~.network.components.Generator` or :class:`~.network.components.Load` or :obj:`dict`
        To position the storage a positioning strategy can be used or a
        node in the network can be directly specified. Possible options are:

        * 'hvmv_substation_busbar'
          Places a storage unit directly at the HV/MV station's bus bar.
        * :class:`~.network.components.Station` or :class:`~.network.components.BranchTee` or :class:`~.network.components.Generator` or :class:`~.network.components.Load`
          Specifies a node the storage should be connected to. In the case
          this parameter is of type :class:`~.network.components.LVStation` an
          additional parameter, `voltage_level`, has to be provided to define
          which side of the LV station the storage is connected to.
        * 'distribute_storages_mv'
          Places one storage in each MV feeder if it reduces network expansion
          costs. This method needs a given time series of active power.
          ToDo: Elaborate

        In case of more than one storage provide a :obj:`dict` where each
        entry represents a storage. Keys of the dictionary have to match
        the keys of the `timeseries` and `parameters`
        dictionaries, values must contain the corresponding positioning
        strategy or node to connect the storage to.
    parameters : :obj:`dict`, optional
        Dictionary with the following optional storage parameters:

        .. code-block:: python

            {
                'nominal_power': <float>, # in kW
                'max_hours': <float>, # in h
                'soc_initial': <float>, # in kWh
                'efficiency_in': <float>, # in per unit 0..1
                'efficiency_out': <float>, # in per unit 0..1
                'standing_loss': <float> # in per unit 0..1
            }

        See :class:`~.network.components.Storage` for more information on storage
        parameters.
        In case of more than one storage provide a :obj:`dict` where each
        entry represents a storage. Keys of the dictionary have to match
        the keys of the `timeseries` dictionary, values must
        contain the corresponding parameters dictionary specified above.
        Note: As edisgo currently only provides a power flow analysis storage
        parameters don't have any effect on the calculations, except of the
        nominal power of the storage.
        Default: {}.
    voltage_level : :obj:`str` or :obj:`dict`, optional
        This parameter only needs to be provided if any entry in `position` is
        of type :class:`~.network.components.LVStation`. In that case
        `voltage_level` defines which side of the LV station the storage is
        connected to. Valid options are 'lv' and 'mv'.
        In case of more than one storage provide a :obj:`dict` specifying the
        voltage level for each storage that is to be connected to an LV
        station. Keys of the dictionary have to match the keys of the
        `timeseries` dictionary, values must contain the corresponding
        voltage level.
        Default: None.
    timeseries_reactive_power : :pandas:`pandas.Series<series>` or :obj:`dict`
        By default reactive power is set through the config file
        `config_timeseries` in sections `reactive_power_factor` specifying
        the power factor and `reactive_power_mode` specifying if inductive
        or capacitive reactive power is provided.
        If you want to over-write this behavior you can provide a reactive
        power time series in kvar here. Be aware that eDisGo uses the generator
        sign convention for storages (see `Definitions and units` section of
        the documentation for more information). Index of the series needs to
        be a  :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        In case of more than one storage provide a :obj:`dict` where each
        entry represents a storage. Keys of the dictionary have to match
        the keys of the `timeseries` dictionary, values must contain the
        corresponding time series as :pandas:`pandas.Series<series>`.

    """

    def __init__(self, edisgo, timeseries, position, **kwargs):

        self.edisgo = edisgo
        voltage_level = kwargs.pop('voltage_level', None)
        parameters = kwargs.pop('parameters', {})
        timeseries_reactive_power = kwargs.pop(
            'timeseries_reactive_power', None)
        if isinstance(timeseries, dict):
            # check if other parameters are dicts as well if provided
            if voltage_level is not None:
                if not isinstance(voltage_level, dict):
                    message = 'Since storage `timeseries` is a dictionary, ' \
                              '`voltage_level` has to be provided as a ' \
                              'dictionary as well.'
                    logging.error(message)
                    raise KeyError(message)
            if parameters is not None:
                if not all(isinstance(value, dict) == True
                           for value in parameters.values()):
                    message = 'Since storage `timeseries` is a dictionary, ' \
                              'storage parameters of each storage have to ' \
                              'be provided as a dictionary as well.'
                    logging.error(message)
                    raise KeyError(message)
            if timeseries_reactive_power is not None:
                if not isinstance(timeseries_reactive_power, dict):
                    message = 'Since storage `timeseries` is a dictionary, ' \
                              '`timeseries_reactive_power` has to be ' \
                              'provided as a dictionary as well.'
                    logging.error(message)
                    raise KeyError(message)
            for storage, ts in timeseries.items():
                try:
                    pos = position[storage]
                except KeyError:
                    message = 'Please provide position for storage {}.'.format(
                        storage)
                    logging.error(message)
                    raise KeyError(message)
                try:
                    voltage_lev = voltage_level[storage]
                except:
                    voltage_lev = None
                try:
                    params = parameters[storage]
                except:
                    params = {}
                try:
                    reactive_power = timeseries_reactive_power[storage]
                except:
                    reactive_power = None
                self._integrate_storage(ts, pos, params,
                                        voltage_lev, reactive_power, **kwargs)
        else:
            self._integrate_storage(timeseries, position, parameters,
                                    voltage_level, timeseries_reactive_power,
                                    **kwargs)

        # add measure to Results object
        self.edisgo.results.measures = 'storage_integration'

    def _integrate_storage(self, timeseries, position, params, voltage_level,
                           reactive_power_timeseries, **kwargs):
        """
        Integrate storage units in the network.

        Parameters
        ----------
        timeseries : :obj:`str` or :pandas:`pandas.Series<series>`
            Parameter used to obtain time series of active power the storage
            storage is charged (negative) or discharged (positive) with. Can
            either be a given time series or an operation strategy. See class
            definition for more information
        position : :obj:`str` or :class:`~.network.components.Station` or :class:`~.network.components.BranchTee` or :class:`~.network.components.Generator` or :class:`~.network.components.Load`
            Parameter used to place the storage. See class definition for more
            information.
        params : :obj:`dict`
            Dictionary with storage parameters for one storage. See class
            definition for more information on what parameters must be
            provided.
        voltage_level : :obj:`str` or None
            `voltage_level` defines which side of the LV station the storage is
            connected to. Valid options are 'lv' and 'mv'. Default: None. See
            class definition for more information.
        reactive_power_timeseries : :pandas:`pandas.Series<series>` or None
            Reactive power time series in kvar (generator sign convention).
            Index of the series needs to be a
            :pandas:`pandas.DatetimeIndex<datetimeindex>`.

        """
        # place storage
        params = self._check_nominal_power(params, timeseries)
        if isinstance(position, Station) or isinstance(position, BranchTee) \
                or isinstance(position, Generator) \
                or isinstance(position, Load):
            storage = storage_integration.set_up_storage(
                node=position, parameters=params, voltage_level=voltage_level)
            line = storage_integration.connect_storage(storage, position)
        elif isinstance(position, str) \
                and position == 'hvmv_substation_busbar':
            storage, line = storage_integration.storage_at_hvmv_substation(
                self.edisgo.network.mv_grid, params)
        elif isinstance(position, str) \
                and position == 'distribute_storages_mv':
            # check active power time series
            if not isinstance(timeseries, pd.Series):
                raise ValueError(
                    "Storage time series needs to be a pandas Series if "
                    "`position` is 'distribute_storages_mv'.")
            else:
                timeseries = pd.DataFrame(data={'p': timeseries},
                                          index=timeseries.index)
                self._check_timeindex(timeseries)
            # check reactive power time series
            if reactive_power_timeseries is not None:
                self._check_timeindex(reactive_power_timeseries)
                timeseries['q'] = reactive_power_timeseries.loc[
                    timeseries.index]
            else:
                timeseries['q'] = 0
            # start storage positioning method
            storage_positioning.one_storage_per_feeder(
                edisgo=self.edisgo, storage_timeseries=timeseries,
                storage_nominal_power=params['nominal_power'], **kwargs)
            return
        else:
            message = 'Provided storage position option {} is not ' \
                      'valid.'.format(timeseries)
            logging.error(message)
            raise KeyError(message)

        # implement operation strategy (active power)
        if isinstance(timeseries, pd.Series):
            timeseries = pd.DataFrame(data={'p': timeseries},
                                      index=timeseries.index)
            self._check_timeindex(timeseries)
            storage.timeseries = timeseries
        elif isinstance(timeseries, str) and timeseries == 'fifty-fifty':
            storage_operation.fifty_fifty(self.edisgo.network, storage)
        else:
            message = 'Provided storage timeseries option {} is not ' \
                      'valid.'.format(timeseries)
            logging.error(message)
            raise KeyError(message)

        # reactive power
        if reactive_power_timeseries is not None:
            self._check_timeindex(reactive_power_timeseries)
            storage.timeseries = pd.DataFrame(
                {'p': storage.timeseries.p,
                 'q': reactive_power_timeseries.loc[storage.timeseries.index]},
                index=storage.timeseries.index)

        # update pypsa representation
        if self.edisgo.network.pypsa is not None:
            pypsa_io.update_pypsa_storage(
                self.edisgo.network.pypsa,
                storages=[storage], storages_lines=[line])

    def _check_nominal_power(self, storage_parameters, timeseries):
        """
        Tries to assign a nominal power to the storage.

        Checks if nominal power is provided through `storage_parameters`,
        otherwise tries to return the absolute maximum of `timeseries`. Raises
        an error if it cannot assign a nominal power.

        Parameters
        ----------
        timeseries : :obj:`str` or :pandas:`pandas.Series<series>`
            See parameter `timeseries` in class definition for more
            information.
        storage_parameters : :obj:`dict`
            See parameter `parameters` in class definition for more
            information.

        Returns
        --------
        :obj:`dict`
            The given `storage_parameters` is returned extended by an entry for
            'nominal_power', if it didn't already have that key.

        """
        if storage_parameters.get('nominal_power', None) is None:
            try:
                storage_parameters['nominal_power'] = max(abs(timeseries))
            except:
                raise ValueError("Could not assign a nominal power to the "
                                 "storage. Please provide either a nominal "
                                 "power or an active power time series.")
        return storage_parameters

    def _check_timeindex(self, timeseries):
        """
        Raises an error if time index of storage time series does not
        comply with the time index of load and feed-in time series.

        Parameters
        -----------
        timeseries : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power the storage is charged (negative)
            and discharged (positive) with in kW in column 'p' and
            reactive power in kVA in column 'q'.

        """
        try:
            timeseries.loc[self.edisgo.network.timeseries.timeindex]
        except:
            message = 'Time index of storage time series does not match ' \
                      'with load and feed-in time series.'
            logging.error(message)
            raise KeyError(message)


class NetworkReimport:
    """
    Network class created from saved results.

    """
    def __init__(self, results_path, **kwargs):

        # import configs
        self.config = {}
        with open('{}/configs.csv'.format(results_path), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                a = iter(row[1:])
                self.config[row[0]] = dict(zip(a, a))