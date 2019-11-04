import os
import pandas as pd
import numpy as np
from math import sqrt
import logging
import datetime
import csv


from edisgo.tools import config
from edisgo.tools import pypsa_io
from edisgo.flex_opt import storage_integration, storage_operation, \
    curtailment, storage_positioning
from edisgo.network.components import Generator, Load
from edisgo.network.tools import get_gen_info
from edisgo.network.grids import MVGrid


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


class Config:
    """
    Container for all configurations.

    Parameters
    -----------
    config_path : None or :obj:`str` or :obj:`dict`
        Path to the config directory. Options are:

        * None
          If `config_path` is None configs are loaded from the edisgo
          default config directory ($HOME$/.edisgo). If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.
        * :obj:`str`
          If `config_path` is a string configs will be loaded from the
          directory specified by `config_path`. If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.
        * :obj:`dict`
          A dictionary can be used to specify different paths to the
          different config files. The dictionary must have the following
          keys:
          * 'config_db_tables'
          * 'config_grid'
          * 'config_grid_expansion'
          * 'config_timeseries'

          Values of the dictionary are paths to the corresponding
          config file. In contrast to the other two options the directories
          and config files must exist and are not automatically created.

        Default: None.

    Notes
    -----
    The Config object can be used like a dictionary. See example on how to use
    it.

    Examples
    --------
    Create Config object from default config files

    >>> from edisgo.network.network import Config
    >>> config = Config()

    Get reactive power factor for generators in the MV network

    >>> config['reactive_power_factor']['mv_gen']

    """

    def __init__(self, **kwargs):
        self._data = self._load_config(kwargs.get('config_path', None))

    @staticmethod
    def _load_config(config_path=None):
        """
        Load config files.

        Parameters
        -----------
        config_path : None or :obj:`str` or dict
            See class definition for more information.

        Returns
        -------
        :obj:`collections.OrderedDict`
            eDisGo configuration data from config files.

        """

        config_files = ['config_db_tables', 'config_grid',
                        'config_grid_expansion', 'config_timeseries']

        # load configs
        if isinstance(config_path, dict):
            for conf in config_files:
                config.load_config(filename='{}.cfg'.format(conf),
                                   config_dir=config_path[conf],
                                   copy_default_config=False)
        else:
            for conf in config_files:
                config.load_config(filename='{}.cfg'.format(conf),
                                   config_dir=config_path)

        config_dict = config.cfg._sections

        # convert numeric values to float
        for sec, subsecs in config_dict.items():
            for subsec, val in subsecs.items():
                # try str -> float conversion
                try:
                    config_dict[sec][subsec] = float(val)
                except:
                    pass

        # convert to time object
        config_dict['demandlib']['day_start'] = datetime.datetime.strptime(
            config_dict['demandlib']['day_start'], "%H:%M")
        config_dict['demandlib']['day_start'] = datetime.time(
            config_dict['demandlib']['day_start'].hour,
            config_dict['demandlib']['day_start'].minute)
        config_dict['demandlib']['day_end'] = datetime.datetime.strptime(
            config_dict['demandlib']['day_end'], "%H:%M")
        config_dict['demandlib']['day_end'] = datetime.time(
            config_dict['demandlib']['day_end'].hour,
            config_dict['demandlib']['day_end'].minute)

        return config_dict

    def __getitem__(self, key1, key2=None):
        if key2 is None:
            try:
                return self._data[key1]
            except:
                raise KeyError(
                    "Config does not contain section {}.".format(key1))
        else:
            try:
                return self._data[key1][key2]
            except:
                raise KeyError("Config does not contain value for {} or "
                               "section {}.".format(key2, key1))

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


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


class Results:
    """
    Power flow analysis results management

    Includes raw power flow analysis results, history of measures to increase
    the network's hosting capacity and information about changes of equipment.

    Attributes
    ----------
    network : :class:`~.network.network.Network`
        The network is a container object holding all data.

    """

    def __init__(self, network):
        self.network = network
        self._measures = ['original']
        self._pfa_p = None
        self._pfa_q = None
        self._pfa_v_mag_pu = None
        self._i_res = None
        self._equipment_changes = pd.DataFrame()
        self._grid_expansion_costs = None
        self._grid_losses = None
        self._hv_mv_exchanges = None
        self._curtailment = None
        self._storage_integration = None
        self._unresolved_issues = {}
        self._storages_costs_reduction = None

    @property
    def measures(self):
        """
        List with the history of measures to increase network's hosting capacity.

        Parameters
        ----------
        measure : :obj:`str`
            Measure to increase network's hosting capacity. Possible options are
            'grid_expansion', 'storage_integration', 'curtailment'.

        Returns
        -------
        measures : :obj:`list`
            A stack that details the history of measures to increase network's
            hosting capacity. The last item refers to the latest measure. The
            key `original` refers to the state of the network topology as it was
            initially imported.

        """
        return self._measures

    @measures.setter
    def measures(self, measure):
        self._measures.append(measure)

    @property
    def pfa_p(self):
        """
        Active power results from power flow analysis in kW.

        Holds power flow analysis results for active power for the last
        iteration step. Index of the DataFrame is a DatetimeIndex indicating
        the time period the power flow analysis was conducted for; columns
        of the DataFrame are the edges as well as stations of the network
        topology.

        Parameters
        ----------
        pypsa : :pandas:`pandas.DataFrame<dataframe>`
            Results time series of active power P in kW from the
            `PyPSA network <https://www.pypsa.org/doc/components.html#network>`_

            Provide this if you want to set values. For retrieval of data do
            not pass an argument

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Active power results from power flow analysis

        """
        return self._pfa_p

    @pfa_p.setter
    def pfa_p(self, pypsa):
        self._pfa_p = pypsa

    @property
    def pfa_q(self):
        """
        Reactive power results from power flow analysis in kvar.

        Holds power flow analysis results for reactive power for the last
        iteration step. Index of the DataFrame is a DatetimeIndex indicating
        the time period the power flow analysis was conducted for; columns
        of the DataFrame are the edges as well as stations of the network
        topology.

        Parameters
        ----------
        pypsa : :pandas:`pandas.DataFrame<dataframe>`
            Results time series of reactive power Q in kvar from the
            `PyPSA network <https://www.pypsa.org/doc/components.html#network>`_

            Provide this if you want to set values. For retrieval of data do not
            pass an argument

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Reactive power results from power flow analysis

        """
        return self._pfa_q

    @pfa_q.setter
    def pfa_q(self, pypsa):
        self._pfa_q = pypsa

    @property
    def pfa_v_mag_pu(self):
        """
        Voltage deviation at node in p.u.

        Holds power flow analysis results for relative voltage deviation for
        the last iteration step. Index of the DataFrame is a DatetimeIndex
        indicating the time period the power flow analysis was conducted for;
        columns of the DataFrame are the nodes as well as stations of the network
        topology.

        Parameters
        ----------
        pypsa : :pandas:`pandas.DataFrame<dataframe>`
            Results time series of voltage deviation in p.u. from the
            `PyPSA network <https://www.pypsa.org/doc/components.html#network>`_

            Provide this if you want to set values. For retrieval of data do
            not pass an argument

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Voltage level nodes of network

        """
        return self._pfa_v_mag_pu

    @pfa_v_mag_pu.setter
    def pfa_v_mag_pu(self, pypsa):
        self._pfa_v_mag_pu = pypsa

    @property
    def i_res(self):
        """
        Current results from power flow analysis in A.

        Holds power flow analysis results for current for the last
        iteration step. Index of the DataFrame is a DatetimeIndex indicating
        the time period the power flow analysis was conducted for; columns
        of the DataFrame are the edges as well as stations of the network
        topology.

        Parameters
        ----------
        pypsa : :pandas:`pandas.DataFrame<dataframe>`
            Results time series of current in A from the
            `PyPSA network <https://www.pypsa.org/doc/components.html#network>`_

            Provide this if you want to set values. For retrieval of data do
            not pass an argument

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Current results from power flow analysis

        """
        return self._i_res

    @i_res.setter
    def i_res(self, pypsa):
        self._i_res = pypsa

    @property
    def equipment_changes(self):
        """
        Tracks changes in the equipment (e.g. replaced or added cable, etc.)

        The DataFrame is indexed by the component(
        :class:`~.network.components.Line`, :class:`~.network.components.Station`,
        etc.) and has the following columns:

        equipment : detailing what was changed (line, station, storage,
        curtailment). For ease of referencing we take the component itself.
        For lines we take the line-dict, for stations the transformers, for
        storages the storage-object itself and for curtailment
        either a dict providing the details of curtailment or a curtailment
        object if this makes more sense (has to be defined).

        change : :obj:`str`
            Specifies if something was added or removed.

        iteration_step : :obj:`int`
            Used for the update of the pypsa network to only consider changes
            since the last power flow analysis.

        quantity : :obj:`int`
            Number of components added or removed. Only relevant for
            calculation of network expansion costs to keep track of how many
            new standard lines were added.

        Parameters
        ----------
        changes : :pandas:`pandas.DataFrame<dataframe>`
            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Equipment changes

        """
        return self._equipment_changes

    @equipment_changes.setter
    def equipment_changes(self, changes):
        self._equipment_changes = changes

    @property
    def grid_expansion_costs(self):
        """
        Holds network expansion costs in kEUR due to network expansion measures
        tracked in self.equipment_changes and calculated in
        edisgo.flex_opt.costs.grid_expansion_costs()

        Parameters
        ----------
        total_costs : :pandas:`pandas.DataFrame<dataframe>`

            DataFrame containing type and costs plus in the case of lines the
            line length and number of parallel lines of each reinforced
            transformer and line. Provide this if you want to set
            grid_expansion_costs. For retrieval of costs do not pass an
            argument.

            Index of the DataFrame is the respective object
            that can either be a :class:`~.network.components.Line` or a
            :class:`~.network.components.Transformer`. Columns are the following:

            type : :obj:`str`
                Transformer size or cable name

            total_costs : :obj:`float`
                Costs of equipment in kEUR. For lines the line length and
                number of parallel lines is already included in the total
                costs.

            quantity : :obj:`int`
                For transformers quantity is always one, for lines it specifies
                the number of parallel lines.

            line_length : :obj:`float`
                Length of line or in case of parallel lines all lines in km.

            voltage_level : :obj:`str`
                Specifies voltage level the equipment is in ('lv', 'mv' or
                'mv/lv').

            mv_feeder : :class:`~.network.components.Line`
                First line segment of half-ring used to identify in which
                feeder the network expansion was conducted in.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Costs of each reinforced equipment in kEUR.

        Notes
        -------
        Total network expansion costs can be obtained through
        costs.total_costs.sum().

        """
        return self._grid_expansion_costs

    @grid_expansion_costs.setter
    def grid_expansion_costs(self, total_costs):
        self._grid_expansion_costs = total_costs

    @property
    def grid_losses(self):
        """
        Holds active and reactive network losses in kW and kvar, respectively.

        Parameters
        ----------
        pypsa_grid_losses : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive network losses in columns 'p'
            and 'q' and in kW and kvar, respectively. Index is a
            :pandas:`pandas.DatetimeIndex<datetimeindex>`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive network losses in columns 'p'
            and 'q' and in kW and kvar, respectively. Index is a
            :pandas:`pandas.DatetimeIndex<datetimeindex>`.

        Notes
        ------
        Grid losses are calculated as follows:

        .. math::
            P_{loss} = \sum{feed-in} - \sum{load} + P_{slack}
            Q_{loss} = \sum{feed-in} - \sum{load} + Q_{slack}

        As the slack is placed at the secondary side of the HV/MV station
        losses do not include losses of the HV/MV transformers.

        """

        return self._grid_losses

    @grid_losses.setter
    def grid_losses(self, pypsa_grid_losses):
        self._grid_losses = pypsa_grid_losses

    @property
    def hv_mv_exchanges(self):
        """
        Holds active and reactive power exchanged with the HV network.

        The exchanges are essentially the slack results. As the slack is placed
        at the secondary side of the HV/MV station, this gives the energy
        transferred to and taken from the HV network at the secondary side of the
        HV/MV station.

        Parameters
        ----------
        hv_mv_exchanges : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive power exchanged with the HV
            network in columns 'p' and 'q' and in kW and kvar, respectively. Index
            is a :pandas:`pandas.DatetimeIndex<datetimeindex>`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>
            Dataframe holding active and reactive power exchanged with the HV
            network in columns 'p' and 'q' and in kW and kvar, respectively. Index
            is a :pandas:`pandas.DatetimeIndex<datetimeindex>`.

        """

        return self._hv_mv_exchanges

    @hv_mv_exchanges.setter
    def hv_mv_exchanges(self, hv_mv_exchanges):
        self._hv_mv_exchanges = hv_mv_exchanges

    @property
    def curtailment(self):
        """
        Holds curtailment assigned to each generator per curtailment target.

        Returns
        -------
        :obj:`dict` with :pandas:`pandas.DataFrame<dataframe>`
            Keys of the dictionary are generator types (and weather cell ID)
            curtailment targets were given for. E.g. if curtailment is provided
            as a :pandas:`pandas.DataFrame<dataframe>` with
            :pandas.`pandas.MultiIndex` columns with levels 'type' and
            'weather cell ID' the dictionary key is a tuple of
            ('type','weather_cell_id').
            Values of the dictionary are dataframes with the curtailed power in
            kW per generator and time step. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<datetimeindex>`. Columns are the
            generators of type
            :class:`edisgo.network.components.GeneratorFluctuating`.

        """
        if self._curtailment is not None:
            result_dict = {}
            for key, gen_list in self._curtailment.items():
                curtailment_df = pd.DataFrame()
                for gen in gen_list:
                    curtailment_df[gen] = gen.curtailment
                result_dict[key] = curtailment_df
            return result_dict
        else:
            return None

    @property
    def storages(self):
        """
        Gathers relevant storage results.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`

            Dataframe containing all storages installed in the MV network and
            LV grids. Index of the dataframe are the storage representatives,
            columns are the following:

            nominal_power : :obj:`float`
                Nominal power of the storage in kW.

            voltage_level : :obj:`str`
                Voltage level the storage is connected to. Can either be 'mv'
                or 'lv'.

        """
        grids = [self.network.mv_grid] + list(self.network.mv_grid.lv_grids)
        storage_results = {}
        storage_results['storage_id'] = []
        storage_results['nominal_power'] = []
        storage_results['voltage_level'] = []
        storage_results['grid_connection_point'] = []
        for grid in grids:
            for storage in grid.graph.nodes_by_attribute('storage'):
                storage_results['storage_id'].append(repr(storage))
                storage_results['nominal_power'].append(storage.nominal_power)
                storage_results['voltage_level'].append(
                    'mv' if isinstance(grid, MVGrid) else 'lv')
                storage_results['grid_connection_point'].append(
                     list(grid.graph.neighbors(storage))[0])

        return pd.DataFrame(storage_results).set_index('storage_id')

    def storages_timeseries(self):
        """
        Returns a dataframe with storage time series.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`

            Dataframe containing time series of all storages installed in the
            MV network and LV grids. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<datetimeindex>`. Columns are the
            storage representatives.

        """
        storages_p = pd.DataFrame()
        storages_q = pd.DataFrame()
        grids = [self.network.mv_grid] + list(self.network.mv_grid.lv_grids)
        for grid in grids:
            for storage in grid.graph.nodes_by_attribute('storage'):
                ts = storage.timeseries
                storages_p[repr(storage)] = ts.p
                storages_q[repr(storage)] = ts.q

        return storages_p, storages_q

    @property
    def storages_costs_reduction(self):
        """
        Contains costs reduction due to storage integration.

        Parameters
        ----------
        costs_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe containing network expansion costs in kEUR before and after
            storage integration in columns 'grid_expansion_costs_initial' and
            'grid_expansion_costs_with_storages', respectively. Index of the
            dataframe is the MV network id.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`

            Dataframe containing network expansion costs in kEUR before and after
            storage integration in columns 'grid_expansion_costs_initial' and
            'grid_expansion_costs_with_storages', respectively. Index of the
            dataframe is the MV network id.

        """
        return self._storages_costs_reduction

    @storages_costs_reduction.setter
    def storages_costs_reduction(self, costs_df):
        self._storages_costs_reduction = costs_df

    @property
    def unresolved_issues(self):
        """
        Holds lines and nodes where over-loading or over-voltage issues
        could not be solved in network reinforcement.

        In case over-loading or over-voltage issues could not be solved
        after maximum number of iterations, network reinforcement is not
        aborted but network expansion costs are still calculated and unresolved
        issues listed here.

        Parameters
        ----------
        issues : dict

            Dictionary of critical lines/stations with relative over-loading
            and critical nodes with voltage deviation in p.u.. Format:

            .. code-block:: python

                {crit_line_1: rel_overloading_1, ...,
                 crit_line_n: rel_overloading_n,
                 crit_node_1: v_mag_pu_node_1, ...,
                 crit_node_n: v_mag_pu_node_n}

            Provide this if you want to set unresolved_issues. For retrieval
            of unresolved issues do not pass an argument.

        Returns
        -------
        Dictionary
            Dictionary of critical lines/stations with relative over-loading
            and critical nodes with voltage deviation in p.u.

        """
        return self._unresolved_issues

    @unresolved_issues.setter
    def unresolved_issues(self, issues):
        self._unresolved_issues = issues

    def s_res(self, components=None):
        """
        Get resulting apparent power in kVA at line(s) and transformer(s).

        The apparent power at a line (or transformer) is determined from the
        maximum values of active power P and reactive power Q.

        .. math::

            S = max(\sqrt{p_0^2 + q_0^2}, \sqrt{p_1^2 + q_1^2})

        Parameters
        ----------
        components : :obj:`list`
            List with all components (of type :class:`~.network.components.Line`
            or :class:`~.network.components.Transformer`) to get apparent power
            for. If not provided defaults to return apparent power of all lines
            and transformers in the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Apparent power in kVA for lines and/or transformers.

        """

        if components is not None:
            labels_included = []
            labels_not_included = []
            labels = [repr(l) for l in components]
            for label in labels:
                if (label in list(self.pfa_p.columns) and
                        label in list(self.pfa_q.columns)):
                    labels_included.append(label)
                else:
                    labels_not_included.append(label)
            if labels_not_included:
                logging.warning(
                    "Apparent power for {lines} are not returned from "
                    "PFA".format(lines=labels_not_included))
        else:
            labels_included = self.pfa_p.columns

        s_res = ((self.pfa_p[labels_included] ** 2 + self.pfa_q[
            labels_included] ** 2)).applymap(sqrt)

        return s_res

    def v_res(self, nodes=None, level=None):
        """
        Get voltage results (in p.u.) from power flow analysis.

        Parameters
        ----------
        nodes : :class:`~.network.components.Load`, \
            :class:`~.network.components.Generator`, etc. or :obj:`list`
            Grid topology component or list of network topology components.
            If not provided defaults to column names available in network level
            `level`.
        level : str
            Either 'mv' or 'lv' or None (default). Depending on which network
            level results you are interested in. It is required to provide this
            argument in order to distinguish voltage levels at primary and
            secondary side of the transformer/LV station.
            If not provided (respectively None) defaults to ['mv', 'lv'].

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Resulting voltage levels obtained from power flow analysis

        Notes
        -----
        Limitation:  When power flow analysis is performed for MV only
        (with aggregated LV loads and generators) this methods only returns
        voltage at secondary side busbar and not at load/generator.

        """
        # First check if results are available:
        if hasattr(self, 'pfa_v_mag_pu'):
            # unless index is lexsorted, it cannot be sliced
            self.pfa_v_mag_pu.sort_index(axis=1, inplace=True)
        else:
            message = "No Power Flow Calculation has be done yet, " \
                      "so there are no results yet."
            raise AttributeError(message)

        if level is None:
            level = ['mv', 'lv']

        if nodes is None:
            return self.pfa_v_mag_pu.loc[:, (level, slice(None))]
        else:
            labels = list(map(repr, list(nodes).copy()))
            not_included = [_ for _ in labels
                            if _ not in list(self.pfa_v_mag_pu[level].columns)]
            labels_included = [_ for _ in labels if _ not in not_included]

            if not_included:
                logging.warning("Voltage levels for {nodes} are not returned "
                                "from PFA".format(
                nodes=not_included))
            return self.pfa_v_mag_pu[level][labels_included]

    def save(self, directory, parameters='all'):
        """
        Saves results to disk.

        Depending on which results are selected and if they exist, the
        following directories and files are created:

        * `powerflow_results` directory

          * `voltages_pu.csv`

            See :py:attr:`~pfa_v_mag_pu` for more information.
          * `currents.csv`

            See :func:`~i_res` for more information.
          * `active_powers.csv`

            See :py:attr:`~pfa_p` for more information.
          * `reactive_powers.csv`

            See :py:attr:`~pfa_q` for more information.
          * `apparent_powers.csv`

            See :func:`~s_res` for more information.
          * `grid_losses.csv`

            See :py:attr:`~grid_losses` for more information.
          * `hv_mv_exchanges.csv`

            See :py:attr:`~hv_mv_exchanges` for more information.

        * `pypsa_network` directory

          See :py:func:`pypsa.Network.export_to_csv_folder`

        * `grid_expansion_results` directory

          * `grid_expansion_costs.csv`

            See :py:attr:`~grid_expansion_costs` for more information.
          * `equipment_changes.csv`

            See :py:attr:`~equipment_changes` for more information.
          * `unresolved_issues.csv`

            See :py:attr:`~unresolved_issues` for more information.

        * `curtailment_results` directory

          Files depend on curtailment specifications. There will be one file
          for each curtailment specification, that is for every key in
          :py:attr:`~curtailment` dictionary.

        * `storage_integration_results` directory

          * `storages.csv`

            See :func:`~storages` for more information.

        Parameters
        ----------
        directory : :obj:`str`
            Directory to save the results in.
        parameters : :obj:`str` or :obj:`list` of :obj:`str`
            Specifies which results will be saved. By default all results are
            saved. To only save certain results set `parameters` to one of the
            following options or choose several options by providing a list:

            * 'pypsa_network'
            * 'powerflow_results'
            * 'grid_expansion_results'
            * 'curtailment_results'
            * 'storage_integration_results'

        """
        def _save_power_flow_results(target_dir):
            if self.pfa_v_mag_pu is not None:
                # create directory
                os.makedirs(target_dir, exist_ok=True)

                # voltage
                self.pfa_v_mag_pu.to_csv(
                    os.path.join(target_dir, 'voltages_pu.csv'))

                # current
                self.i_res.to_csv(
                    os.path.join(target_dir, 'currents.csv'))

                # active power
                self.pfa_p.to_csv(
                    os.path.join(target_dir, 'active_powers.csv'))

                # reactive power
                self.pfa_q.to_csv(
                    os.path.join(target_dir, 'reactive_powers.csv'))

                # apparent power
                self.s_res().to_csv(
                    os.path.join(target_dir, 'apparent_powers.csv'))

                # network losses
                self.grid_losses.to_csv(
                    os.path.join(target_dir, 'grid_losses.csv'))

                # network exchanges
                self.hv_mv_exchanges.to_csv(os.path.join(
                    target_dir, 'hv_mv_exchanges.csv'))

        def _save_pypsa_network(target_dir):
            if self.network.pypsa:
                # create directory
                os.makedirs(target_dir, exist_ok=True)
                self.network.pypsa.export_to_csv_folder(target_dir)

        def _save_grid_expansion_results(target_dir):
            if self.grid_expansion_costs is not None:
                # create directory
                os.makedirs(target_dir, exist_ok=True)

                # network expansion costs
                self.grid_expansion_costs.to_csv(os.path.join(
                    target_dir, 'grid_expansion_costs.csv'))

                # unresolved issues
                pd.DataFrame(self.unresolved_issues).to_csv(os.path.join(
                    target_dir, 'unresolved_issues.csv'))

                # equipment changes
                self.equipment_changes.to_csv(os.path.join(
                    target_dir, 'equipment_changes.csv'))

        def _save_curtailment_results(target_dir):
            if self.curtailment is not None:
                # create directory
                os.makedirs(target_dir, exist_ok=True)

                for key, curtailment_df in self.curtailment.items():
                    if type(key) == tuple:
                        type_prefix = '-'.join([key[0], str(key[1])])
                    elif type(key) == str:
                        type_prefix = key
                    else:
                        raise KeyError("Unknown key type {} for key {}".format(
                            type(key), key))

                    filename = os.path.join(
                        target_dir, '{}.csv'.format(type_prefix))

                    curtailment_df.to_csv(filename, index_label=type_prefix)

        def _save_storage_integration_results(target_dir):
            storages = self.storages
            if not storages.empty:
                # create directory
                os.makedirs(target_dir, exist_ok=True)

                # general storage information
                storages.to_csv(os.path.join(target_dir, 'storages.csv'))

                # storages time series
                ts_p, ts_q = self.storages_timeseries()
                ts_p.to_csv(os.path.join(
                    target_dir, 'storages_active_power.csv'))
                ts_q.to_csv(os.path.join(
                    target_dir, 'storages_reactive_power.csv'))

                if not self.storages_costs_reduction is None:
                    self.storages_costs_reduction.to_csv(
                        os.path.join(target_dir,
                                     'storages_costs_reduction.csv'))

        # dictionary with function to call to save each parameter
        func_dict = {
            'powerflow_results': _save_power_flow_results,
            'pypsa_network': _save_pypsa_network,
            'grid_expansion_results': _save_grid_expansion_results,
            'curtailment_results': _save_curtailment_results,
            'storage_integration_results': _save_storage_integration_results
        }

        # if string is given convert to list
        if isinstance(parameters, str):
            if parameters == 'all':
                parameters = ['powerflow_results', 'pypsa_network',
                              'grid_expansion_results', 'curtailment_results',
                              'storage_integration_results']
            else:
                parameters = [parameters]

        # save each parameter
        for parameter in parameters:
            try:
                func_dict[parameter](os.path.join(directory, parameter))
            except KeyError:
                message = "Invalid input {} for `parameters` when saving " \
                          "results. Must be any or a list of the following: " \
                          "'pypsa_network', 'powerflow_results', " \
                          "'grid_expansion_results', 'curtailment_results', " \
                          "'storage_integration_results'.".format(parameter)
                logger.error(message)
                raise KeyError(message)
            except:
                raise
        # save measures
        pd.DataFrame(data={'measure': self.measures}).to_csv(
            os.path.join(directory, 'measures.csv'))
        # save configs
        with open(os.path.join(directory, 'configs.csv'), 'w') as f:
            writer = csv.writer(f)
            rows = [
                ['{}'.format(key)] + [value for item in values.items()
                                      for value in item]
                for key, values in self.network.config._data.items()]
            writer.writerows(rows)


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




class ResultsReimport:
    """
    Results class created from saved results.

    """
    def __init__(self, results_path, parameters='all'):

        # measures
        measures_df = pd.read_csv(os.path.join(results_path, 'measures.csv'),
                                  index_col=0)
        self.measures = list(measures_df.measure.values)

        # if string is given convert to list
        if isinstance(parameters, str):
            if parameters == 'all':
                parameters = ['powerflow_results', 'grid_expansion_results',
                              'curtailment_results',
                              'storage_integration_results']
            else:
                parameters = [parameters]

        # import power flow results
        if 'powerflow_results' in parameters and os.path.isdir(os.path.join(
                results_path, 'powerflow_results')):
            # line loading
            self.i_res = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'currents.csv'),
                index_col=0, parse_dates=True)
            # voltage
            self.pfa_v_mag_pu = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'voltages_pu.csv'),
                index_col=0, parse_dates=True, header=[0, 1])
            # active power
            self.pfa_p = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'active_powers.csv'),
                index_col=0, parse_dates=True)
            # reactive power
            self.pfa_q = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'reactive_powers.csv'),
                index_col=0, parse_dates=True)
            # apparent power
            self.apparent_power = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'apparent_powers.csv'),
                index_col=0, parse_dates=True)
            # network losses
            self.grid_losses = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'grid_losses.csv'),
                index_col=0, parse_dates=True)
            # network exchanges
            self.hv_mv_exchanges = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'hv_mv_exchanges.csv'),
                index_col=0, parse_dates=True)
        else:
            self.i_res = None
            self.pfa_v_mag_pu = None
            self.pfa_p = None
            self.pfa_q = None
            self.apparent_power = None
            self.grid_losses = None
            self.hv_mv_exchanges = None

        # import network expansion results
        if 'grid_expansion_results' in parameters and os.path.isdir(
                os.path.join(results_path, 'grid_expansion_results')):
            # network expansion costs
            self.grid_expansion_costs = pd.read_csv(
                os.path.join(
                    results_path, 'grid_expansion_results',
                    'grid_expansion_costs.csv'),
                index_col=0)
            # equipment changes
            self.equipment_changes = pd.read_csv(
                os.path.join(
                    results_path, 'grid_expansion_results',
                    'equipment_changes.csv'),
                index_col=0)
        else:
            self.grid_expansion_costs = None
            self.equipment_changes = None

        # import curtailment results
        if 'curtailment_results' in parameters and os.path.isdir(
                os.path.join(results_path, 'curtailment_results')):
            self.curtailment = {}
            for file in os.listdir(os.path.join(
                    results_path, 'curtailment_results')):
                if file.endswith(".csv"):
                    try:
                        key = file[0:-4]
                        if '-' in key:
                            # make tuple if curtailment was given for generator
                            # type and weather cell id
                            tmp = key.split('-')
                            key = (tmp[0], float(tmp[1]))
                        self.curtailment[key] = pd.read_csv(
                            os.path.join(
                                results_path, 'curtailment_results', file),
                            index_col=0, parse_dates=True)
                    except Exception as e:
                        logging.warning(
                            'The following error occured when trying to '
                            'import curtailment results: {}'.format(e))
        else:
            self.curtailment = None

        # import storage results
        if 'storage_integration_results' in parameters and os.path.isdir(
                os.path.join(results_path, 'storage_integration_results')):
            # storages
            self.storages = pd.read_csv(
                os.path.join(results_path, 'storage_integration_results',
                             'storages.csv'),
                index_col=0)
            # storages costs reduction
            try:
                self.storages_costs_reduction = pd.read_csv(
                    os.path.join(
                        results_path, 'storage_integration_results',
                        'storages_costs_reduction.csv'),
                    index_col=0)
            except:
                pass
            # storages time series
            self.storages_p = pd.read_csv(
                os.path.join(
                    results_path, 'storage_integration_results',
                    'storages_active_power.csv'),
                index_col=0, parse_dates=True)
            # storages time series
            self.storages_q = pd.read_csv(
                os.path.join(
                    results_path, 'storage_integration_results',
                    'storages_reactive_power.csv'),
                index_col=0, parse_dates=True)

        else:
            self.storages = None
            self.storages_costs_reduction = None
            self.storages_p = None
            self.storages_q = None

    def v_res(self, nodes=None, level=None):
        """
        Get resulting voltage level at node.

        Parameters
        ----------
        nodes : :obj:`list`
            List of string representatives of network topology components, e.g.
            :class:`~.network.components.Generator`. If not provided defaults to
            all nodes available in network level `level`.
        level : :obj:`str`
            Either 'mv' or 'lv' or None (default). Depending on which network
            level results you are interested in. It is required to provide this
            argument in order to distinguish voltage levels at primary and
            secondary side of the transformer/LV station.
            If not provided (respectively None) defaults to ['mv', 'lv'].

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Resulting voltage levels obtained from power flow analysis

        """
        # check if voltages are available:
        if hasattr(self, 'pfa_v_mag_pu'):
            self.pfa_v_mag_pu.sort_index(axis=1, inplace=True)
        else:
            message = "No voltage results available."
            raise AttributeError(message)

        if level is None:
            level = ['mv', 'lv']

        if nodes is None:
            return self.pfa_v_mag_pu.loc[:, (level, slice(None))]
        else:
            not_included = [_ for _ in nodes
                            if _ not in list(self.pfa_v_mag_pu[level].columns)]
            labels_included = [_ for _ in nodes if _ not in not_included]

            if not_included:
                logging.warning("Voltage levels for {nodes} are not returned "
                                "from PFA".format(nodes=not_included))
            return self.pfa_v_mag_pu[level][labels_included]

    def s_res(self, components=None):
        """
        Get apparent power in kVA at line(s) and transformer(s).

        Parameters
        ----------
        components : :obj:`list`
            List of string representatives of :class:`~.network.components.Line`
            or :class:`~.network.components.Transformer`. If not provided defaults
            to return apparent power of all lines and transformers in the network.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Apparent power in kVA for lines and/or transformers.

        """
        if components is None:
            return self.apparent_power
        else:
            not_included = [_ for _ in components
                            if _ not in self.apparent_power.index]
            labels_included = [_ for _ in components if _ not in not_included]

            if not_included:
                logging.warning(
                    "No apparent power results available for: {}".format(
                        not_included))
            return self.apparent_power.loc[:, labels_included]

    def storages_timeseries(self):
        """
        Returns a dataframe with storage time series.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`

            Dataframe containing time series of all storages installed in the
            MV network and LV grids. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<datetimeindex>`. Columns are the
            storage representatives.

        """
        return self.storages_p, self.storages_q
