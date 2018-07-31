import os
import pandas as pd
import numpy as np
from math import sqrt
import logging
import datetime

from matplotlib import pyplot as plt

import edisgo
from edisgo.tools import config, pypsa_io, tools
from edisgo.data.import_data import import_from_ding0, import_generators, \
    import_feedin_timeseries, import_load_timeseries
from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.flex_opt import storage_integration, storage_operation, curtailment
from edisgo.grid.components import Station, BranchTee
from edisgo.grid.tools import get_gen_info

logger = logging.getLogger('edisgo')


class EDisGo:
    """
    Provides the top-level API for invocation of data import, analysis of
    hosting capacity, grid reinforcement and flexibility measures.

    Parameters
    ----------
    worst_case_analysis : None or :obj:`str`, optional
        If not None time series for feed-in and load will be generated
        according to the chosen worst case analysis
        Possible options are:

        * 'worst-case'
          feed-in for the two worst-case scenarios feed-in case and load case
          are generated
        * 'worst-case-feedin'
          feed-in for the worst-case scenario feed-in case is generated
        * 'worst-case-load'
          feed-in for the worst-case scenario load case is generated

        Be aware that if you choose to conduct a worst-case analysis your
        input for the following parameters will not be used:
        * `timeseries_generation_fluctuating`
        * `timeseries_generation_dispatchable`
        * `timeseries_load`

    mv_grid_id : :obj:`str`
        MV grid ID used in import of ding0 grid.

        .. ToDo: explain where MV grid IDs come from

    ding0_grid : file: :obj:`str` or :class:`ding0.core.NetworkDing0`
        If a str is provided it is assumed it points to a pickle with Ding0
        grid data. This file will be read. If an object of the type
        :class:`ding0.core.NetworkDing0` data will be used directly from this
        object.
        This will probably be removed when ding0 grids are in oedb.
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
    scenario_description : None or :obj:`str`
        Can be used to describe your scenario but is not used for anything
        else. Default: None.
    timeseries_generation_fluctuating : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`
        Parameter used to obtain time series for active power feed-in of
        fluctuating renewables wind and solar.
        Possible options are:

        * 'oedb'
          Time series for the year 2011 are obtained from the OpenEnergy
          DataBase.
        * :pandas:`pandas.DataFrame<dataframe>`
          DataFrame with time series, normalized with corresponding capacity.
          Time series can either be aggregated by technology type or by type
          and weather cell ID. In the first case columns of the DataFrame are
          'solar' and 'wind'; in the second case columns need to be a
          :pandas:`pandas.MultiIndex<multiindex>` with the first level
          containing the type and the second level the weather cell id.
          Index needs to be a :pandas:`pandas.DatetimeIndex<datetimeindex>`.

         .. ToDo: explain how to obtain weather cell id,

         .. ToDo: add link to explanation of weather cell id

    timeseries_generation_dispatchable : :pandas:`pandas.DataFrame<dataframe>`
        DataFrame with time series for active power of each (aggregated)
        type of dispatchable generator normalized with corresponding capacity.
        Index needs to be a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        Columns represent generator type:

        * 'gas'
        * 'coal'
        * 'biomass'
        * 'other'
        * ...

        Use 'other' if you don't want to explicitly provide every possible
        type.
    timeseries_generation_reactive_power : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with time series of normalized reactive power (normalized by
        the rated nominal active power) per technology and weather cell. Index
        needs to be a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        Columns represent generator type and can be a MultiIndex column
        containing the weather cell ID in the second level. If the technology
        doesn't contain weather cell information i.e. if it is other than solar
        and wind generation, this second level can be left as a numpy Nan or a
        None.
        Default: None.
        If no time series for the technology or technology and weather cell ID
        is given, reactive power will be calculated from power factor and
        power factor mode in the config sections `reactive_power_factor` and
        `reactive_power_mode` and a warning will be raised. See
        :class:`~.grid.components.Generator` and
        :class:`~.grid.components.GeneratorFluctuating` for more information.
    timeseries_load : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`
        Parameter used to obtain time series of active power of (cumulative)
        loads.
        Possible options are:

        * 'demandlib'
          Time series for the year specified in `timeindex` are
          generated using the oemof demandlib.
        * :pandas:`pandas.DataFrame<dataframe>`
          DataFrame with load time series of each (cumulative) type of load
          normalized with corresponding annual energy demand. Index needs to
          be a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
          Columns represent load type:
          * 'residential'
          * 'retail'
          * 'industrial'
          * 'agricultural'

    timeseries_load_reactive_power : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with time series of normalized reactive power (normalized by
        annual energy demand) per load sector. Index needs to be a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        Columns represent load type:

          * 'residential'
          * 'retail'
          * 'industrial'
          * 'agricultural'

        Default: None.
        If no time series for the load sector is given, reactive power will be
        calculated from power factor and power factor mode in the config
        sections `reactive_power_factor` and `reactive_power_mode` and a
        warning will be raised. See :class:`~.grid.components.Load` for
        more information.
    generator_scenario : None or :obj:`str`
        If provided defines which scenario of future generator park to use
        and invokes import of these generators. Possible options are 'nep2035'
        and 'ego100'.

        .. ToDo: Add link to explanation of scenarios.

    timeindex : None or :pandas:`pandas.DatetimeIndex<datetimeindex>`
        Can be used to select time ranges of the feed-in and load time series
        that will be used in the power flow analysis. Also defines the year
        load time series are obtained for when choosing the 'demandlib' option
        to generate load time series.

    Attributes
    ----------
    network : :class:`~.grid.network.Network`
        The network is a container object holding all data.

    Examples
    --------
    Assuming you have the Ding0 `ding0_data.pkl` in CWD

    Create eDisGo Network object by loading Ding0 file

    >>> from edisgo.grid.network import EDisGo
    >>> edisgo = EDisGo(ding0_grid='ding0_data.pkl', mode='worst-case-feedin')

    Analyze hosting capacity for MV and LV grid level

    >>> edisgo.analyze()

    Print LV station secondary side voltage levels returned by PFA

    >>> lv_stations = edisgo.network.mv_grid.graph.nodes_by_attribute(
    >>>     'lv_station')
    >>> print(edisgo.network.results.v_res(lv_stations, 'lv'))

    """

    def __init__(self, **kwargs):

        # create network
        self.network = Network(
            generator_scenario=kwargs.get('generator_scenario', None),
            config_path=kwargs.get('config_path', None),
            scenario_description=kwargs.get('scenario_description', None))

        # load grid
        # ToDo: should at some point work with only MV grid ID
        self.import_from_ding0(kwargs.get('ding0_grid', None))

        # set up time series for feed-in and load
        # worst-case time series
        if kwargs.get('worst_case_analysis', None):
            self.network.timeseries = TimeSeriesControl(
                network=self.network,
                mode=kwargs.get('worst_case_analysis', None)).timeseries
        else:
            self.network.timeseries = TimeSeriesControl(
                network=self.network,
                timeseries_generation_fluctuating=kwargs.get(
                    'timeseries_generation_fluctuating', None),
                timeseries_generation_dispatchable=kwargs.get(
                    'timeseries_generation_dispatchable', None),
                timeseries_generation_reactive_power=kwargs.get(
                    'timeseries_generation_reactive_power', None),
                timeseries_load=kwargs.get(
                    'timeseries_load', None),
                timeseries_load_reactive_power = kwargs.get(
                    'timeseries_load_reactive_power', None),
                timeindex=kwargs.get('timeindex', None)).timeseries

        # import new generators
        if self.network.generator_scenario is not None:
            self.import_generators()

    def curtail(self, **kwargs):
        """
        Sets up curtailment time series.

        Curtailment time series are written into
        :class:`~.grid.network.TimeSeries`. See
        :class:`~.grid.network.CurtailmentControl` for more information on
        parameters and methodologies.

        """
        CurtailmentControl(edisgo_object=self, **kwargs)

    def import_from_ding0(self, file, **kwargs):
        """Import grid data from DINGO file

        For details see
        :func:`edisgo.data.import_data.import_from_ding0`

        """
        import_from_ding0(file=file, network=self.network)

    def import_generators(self, generator_scenario=None):
        """Import generators

        For details see
        :func:`edisgo.data.import_data.import_generators`

        """
        if generator_scenario:
            self.network.generator_scenario = generator_scenario
        data_source = 'oedb'
        import_generators(network=self.network, data_source=data_source)

    def analyze(self, mode=None, timesteps=None):
        """Analyzes the grid by power flow analysis

        Analyze the grid for violations of hosting capacity. Means, perform a
        power flow analysis and obtain voltages at nodes (load, generator,
        stations/transformers and branch tees) and active/reactive power at
        lines.

        The power flow analysis can currently only be performed for both grid
        levels MV and LV. See ToDos section for more information.

        A static `non-linear power flow analysis is performed using PyPSA
        <https://www.pypsa.org/doc/power_flow.html#full-non-linear-power-flow>`_.
        The high-voltage to medium-voltage transformer are not included in the
        analysis. The slack bus is defined at secondary side of these
        transformers assuming an ideal tap changer. Hence, potential
        overloading of the transformers is not studied here.

        Parameters
        ----------
        mode : str
            Allows to toggle between power flow analysis (PFA) on the whole
            grid topology (MV + LV), only MV or only LV. Defaults to None which
            equals power flow analysis for MV + LV which is the only
            implemented option at the moment. See ToDos section for
            more information.
        timesteps : :pandas:`pandas.DatetimeIndex<datetimeindex>` or :pandas:`pandas.Timestamp<timestamp>`
            Timesteps specifies for which time steps to conduct the power flow
            analysis. It defaults to None in which case the time steps in
            timeseries.timeindex (see :class:`~.grid.network.TimeSeries`) are
            used.

        Notes
        -----
        The current implementation always translates the grid topology
        representation to the PyPSA format and stores it to
        :attr:`self.network.pypsa`.

        ToDos
        ------
        The option to export only the edisgo MV grid (mode = 'mv') to conduct
        a power flow analysis is implemented in
        :func:`~.tools.pypsa_io.to_pypsa` but NotImplementedError is raised
        since the rest of edisgo does not handle this option yet. The analyze
        function will throw an error since
        :func:`~.tools.pypsa_io.process_pfa_results`
        does not handle aggregated loads and generators in the LV grids. Also,
        grid reinforcement, pypsa update of time series, and probably other
        functionalities do not work when only the MV grid is analysed.

        Further ToDos are:
        * explain how power plants are modeled, if possible use a link
        * explain where to find and adjust power flow analysis defining
        parameters

        See Also
        --------
        :func:`~.tools.pypsa_io.to_pypsa`
            Translator to PyPSA data format

        """
        if timesteps is None:
            timesteps = self.network.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]

        if self.network.pypsa is None:
            # Translate eDisGo grid topology representation to PyPSA format
            self.network.pypsa = pypsa_io.to_pypsa(
                self.network, mode, timesteps)
        else:
            if self.network.pypsa.edisgo_mode is not mode:
                # Translate eDisGo grid topology representation to PyPSA format
                self.network.pypsa = pypsa_io.to_pypsa(
                    self.network, mode, timesteps)

        # check if all timesteps are in pypsa.snapshots, if not update time
        # series
        if False in [True if _ in self.network.pypsa.snapshots else False
                     for _ in timesteps]:
            pypsa_io.update_pypsa_timeseries(self.network, timesteps=timesteps)
        # run power flow analysis
        pf_results = self.network.pypsa.pf(timesteps)

        if all(pf_results['converged']['0'].tolist()):
            pypsa_io.process_pfa_results(self.network, self.network.pypsa)
        else:
            raise ValueError("Power flow analysis did not converge.")

    def reinforce(self, **kwargs):
        """
        Reinforces the grid and calculates grid expansion costs.

        See :meth:`~.flex_opt.reinforce_grid` for more information.

        """
        return reinforce_grid(
            self, max_while_iterations=kwargs.get(
                'max_while_iterations', 10),
            copy_graph=kwargs.get('copy_graph', False),
            timesteps_pfa=kwargs.get('timesteps_pfa', None),
            combined_analysis=kwargs.get('combined_analysis', False))

    def integrate_storage(self, **kwargs):
        """
        Integrates storage into grid.

        See :class:`~.grid.network.StorageControl` for more information.

        """
        StorageControl(network=self.network,
                       timeseries_battery=kwargs.get('timeseries_battery',
                                                     None),
                       battery_parameters=kwargs.get('battery_parameters',
                                                     None),
                       battery_position=kwargs.get('battery_position',
                                                   None))


class Network:
    """
    Used as container for all data related to a single
    :class:`~.grid.grids.MVGrid`.

    Parameters
    ----------
    scenario_description : :obj:`str`, optional
        Can be used to describe your scenario but is not used for anything
        else. Default: None.
    config_path : None or :obj:`str` or :obj:`dict`, optional
        See :class:`~.grid.network.Config` for further information.
        Default: None.
    metadata : :obj:`dict`
        Metadata of Network such as ?
    data_sources : :obj:`dict` of :obj:`str`
        Data Sources of grid, generators etc.
        Keys: 'grid', 'generators', ?
    mv_grid : :class:`~.grid.grids.MVGrid`
        Medium voltage (MV) grid
    generator_scenario : :obj:`str`
        Defines which scenario of future generator park to use.

    Attributes
    ----------
    results : :class:`~.grid.network.Results`
        Object with results from power flow analyses

    """

    def __init__(self, **kwargs):
        self._scenario_description = kwargs.get('scenario_description', None)
        self._config = Config(config_path=kwargs.get('config_path', None))
        self._equipment_data = self._load_equipment_data()
        self._metadata = kwargs.get('metadata', None)
        self._data_sources = kwargs.get('data_sources', {})
        self._generator_scenario = kwargs.get('generator_scenario', None)

        self._mv_grid = kwargs.get('mv_grid', None)
        self._pypsa = None
        self.results = Results()

        self._dingo_import_data = []

    def _load_equipment_data(self):
        """Load equipment data for transformers, cables etc.

        Returns
        -------
        :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`

        """

        package_path = edisgo.__path__[0]
        equipment_dir = self.config['system_dirs']['equipment_dir']

        data = {}
        equipment = {'mv': ['trafos', 'lines', 'cables'],
                     'lv': ['trafos', 'cables']}

        for voltage_level, eq_list in equipment.items():
            for i in eq_list:
                equipment_parameters = self.config['equipment'][
                    'equipment_{}_parameters_{}'.format(voltage_level, i)]
                data['{}_{}'.format(voltage_level, i)] = pd.read_csv(
                    os.path.join(package_path, equipment_dir,
                                 equipment_parameters),
                    comment='#', index_col='name',
                    delimiter=',', decimal='.')

        return data

    @property
    def id(self):
        """
        MV grid ID

        Returns
        --------
        :obj:`str`
            MV grid ID

        """
        return self._id

    @property
    def config(self):
        """
        eDisGo configuration data.

        Returns
        -------
        :obj:`collections.OrderedDict`
            Configuration data from config files.

        """
        return self._config

    @config.setter
    def config(self, config_path):
        self._config = Config(config_path=config_path)

    @property
    def metadata(self):
        """
        Metadata of Network

        Returns
        --------
        :obj:`dict`
            Metadata of Network

        """
        return self._metadata

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
    def scenario_description(self):
        """
        Used to describe your scenario but not used for anything else.

        Parameters
        ----------
        scenario_description : :obj:`str`
            Description of scenario

        Returns
        --------
        :obj:`str`
            Scenario name

        """
        return self._scenario_description

    @scenario_description.setter
    def scenario_description(self, scenario_description):
        self._scenario_description = scenario_description

    @property
    def equipment_data(self):
        """
        Technical data of electrical equipment such as lines and transformers

        Returns
        --------
        :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
            Data of electrical equipment

        """
        return self._equipment_data

    @property
    def mv_grid(self):
        """
        Medium voltage (MV) grid

        Parameters
        ----------
        mv_grid : :class:`~.grid.grids.MVGrid`
            Medium voltage (MV) grid

        Returns
        --------
        :class:`~.grid.grids.MVGrid`
            Medium voltage (MV) grid

        """
        return self._mv_grid

    @mv_grid.setter
    def mv_grid(self, mv_grid):
        self._mv_grid = mv_grid

    @property
    def timeseries(self):
        """
        Object containing load and feed-in time series.

        Parameters
        ----------
        timeseries : :class:`~.grid.network.TimeSeries`
            Object containing load and feed-in time series.

        Returns
        --------
        :class:`~.grid.network.TimeSeries`
            Object containing load and feed-in time series.

        """
        return self._timeseries

    @timeseries.setter
    def timeseries(self, timeseries):
        self._timeseries = timeseries

    @property
    def data_sources(self):
        """
        Dictionary with data sources of grid, generators etc.

        Returns
        --------
        :obj:`dict` of :obj:`str`
            Data Sources of grid, generators etc.

        """
        return self._data_sources

    def set_data_source(self, key, data_source):
        """
        Set data source for key (e.g. 'grid')

        Parameters
        ----------
        key : :obj:`str`
            Specifies data
        data_source : :obj:`str`
            Specifies data source

        """
        self._data_sources[key] = data_source

    @property
    def dingo_import_data(self):
        """
        Temporary data from ding0 import needed for OEP generator update

        """
        return self._dingo_import_data

    @dingo_import_data.setter
    def dingo_import_data(self, dingo_data):
        self._dingo_import_data = dingo_data

    @property
    def pypsa(self):
        """
        PyPSA grid representation

        A grid topology representation based on
        :pandas:`pandas.DataFrame<dataframe>`. The overall container object of
        this data model, the :pypsa:`pypsa.Network<network>`,
        is assigned to this attribute.

        Parameters
        ----------
        pypsa : :pypsa:`pypsa.Network<network>`
            The `PyPSA network
            <https://www.pypsa.org/doc/components.html#network>`_ container.

        Returns
        -------
        :pypsa:`pypsa.Network<network>`
            PyPSA grid representation. The attribute `edisgo_mode` is added
            to specify if pypsa representation of the edisgo network
            was created for the whole grid topology (MV + LV), only MV or only
            LV. See parameter `mode` in
            :meth:`~.grid.network.EDisGo.analyze` for more information.

        """
        return self._pypsa

    @pypsa.setter
    def pypsa(self, pypsa):
        self._pypsa = pypsa

    def __repr__(self):
        return 'Network ' + str(self._id)


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

    >>> from edisgo.grid.network import Config
    >>> config = Config()

    Get reactive power factor for generators in the MV grid

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


class TimeSeriesControl:
    """
    Sets up TimeSeries Object.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo data container
    mode : :obj:`str`, optional
        Mode must be set in case of worst-case analyses and can either be
        'worst-case' (both feed-in and load case), 'worst-case-feedin' (only
        feed-in case) or 'worst-case-load' (only load case). All other
        parameters except of `config-data` will be ignored. Default: None.
    timeseries_generation_fluctuating : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`, optional
        Parameter used to obtain time series for active power feed-in of
        fluctuating renewables wind and solar.
        Possible options are:

        * 'oedb'
          Time series for 2011 are obtained from the OpenEnergy DataBase.
          `mv_grid_id` and `scenario_description` have to be provided when
          choosing this option.
        * :pandas:`pandas.DataFrame<dataframe>`
          DataFrame with time series, normalized with corresponding capacity.
          Time series can either be aggregated by technology type or by type
          and weather cell ID. In the first case columns of the DataFrame are
          'solar' and 'wind'; in the second case columns need to be a
          :pandas:`pandas.MultiIndex<multiindex>` with the first level
          containing the type and the second level the weather cell ID.

        Default: None.
    timeseries_generation_dispatchable : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with time series for active power of each (aggregated)
        type of dispatchable generator normalized with corresponding capacity.
        Columns represent generator type:

        * 'gas'
        * 'coal'
        * 'biomass'
        * 'other'
        * ...

        Use 'other' if you don't want to explicitly provide every possible
        type. Default: None.
    timeseries_generation_reactive_power : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with time series of normalized reactive power (normalized by
        the rated nominal active power) per technology and weather cell. Index
        needs to be a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        Columns represent generator type and can be a MultiIndex column
        containing the weather cell ID in the second level. If the technology
        doesn't contain weather cell information i.e. if it is other than solar
        and wind generation, this second level can be left as an empty string ''.

        Default: None.
    timeseries_load : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`, optional
        Parameter used to obtain time series of active power of (cumulative)
        loads.
        Possible options are:

        * 'demandlib'
          Time series are generated using the oemof demandlib.
        * :pandas:`pandas.DataFrame<dataframe>`
          DataFrame with load time series of each (cumulative) type of load
          normalized with corresponding annual energy demand.
          Columns represent load type:

          * 'residential'
          * 'retail'
          * 'industrial'
          * 'agricultural'

        Default: None.
    timeseries_load_reactive_power : :pandas:`pandas.DataFrame<dataframe>`, optional
        Parameter to get the time series of the reactive power of loads. It should be a
        DataFrame with time series of normalized reactive power (normalized by
        annual energy demand) per load sector. Index needs to be a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        Columns represent load type:

          * 'residential'
          * 'retail'
          * 'industrial'
          * 'agricultural'

        Default: None.
    timeindex : :pandas:`pandas.DatetimeIndex<datetimeindex>`
        Can be used to define a time range for which to obtain load time series
        and feed-in time series of fluctuating renewables or to define time
        ranges of the given time series that will be used in the analysis.

    """

    def __init__(self, network, **kwargs):

        self.timeseries = TimeSeries(network=network)
        mode = kwargs.get('mode', None)
        config_data = network.config
        weather_cell_ids = network.mv_grid.weather_cells

        if mode:
            if mode == 'worst-case':
                modes = ['feedin_case', 'load_case']
            elif mode == 'worst-case-feedin' or mode == 'worst-case-load':
                modes = ['{}_case'.format(mode.split('-')[-1])]
            else:
                raise ValueError('{} is not a valid mode.'.format(mode))

            # set random timeindex
            self.timeseries._timeindex = pd.date_range(
                '1/1/1970', periods=len(modes), freq='H')
            self._worst_case_generation(config_data['worst_case_scale_factor'],
                                        modes)
            self._worst_case_load(config_data['worst_case_scale_factor'],
                                  config_data['peakload_consumption_ratio'],
                                  modes)

        else:
            # feed-in time series of fluctuating renewables
            ts = kwargs.get('timeseries_generation_fluctuating', None)
            if isinstance(ts, pd.DataFrame):
                self.timeseries.generation_fluctuating = ts
            elif isinstance(ts, str) and ts == 'oedb':
                self.timeseries.generation_fluctuating = \
                    import_feedin_timeseries(config_data,
                                             weather_cell_ids)
            else:
                raise ValueError('Your input for '
                                 '"timeseries_generation_fluctuating" is not '
                                 'valid.'.format(mode))
            # feed-in time series for dispatchable generators
            ts = kwargs.get('timeseries_generation_dispatchable', None)
            if isinstance(ts, pd.DataFrame):
                self.timeseries.generation_dispatchable = ts
            else:
                raise ValueError('Your input for '
                                 '"timeseries_generation_dispatchable" is not '
                                 'valid.'.format(mode))
            # reactive power time series for all generators
            ts = kwargs.get('timeseries_generation_reactive_power', None)
            if isinstance(ts, pd.DataFrame):
                self.timeseries.generation_reactive_power = ts
            # set time index
            if kwargs.get('timeindex', None) is not None:
                self.timeseries._timeindex = kwargs.get('timeindex')
            else:
                self.timeseries._timeindex = \
                    self.timeseries._generation_fluctuating.index

            # load time series
            ts = kwargs.get('timeseries_load', None)
            if isinstance(ts, pd.DataFrame):
                self.timeseries.load = ts
            elif ts == 'demandlib':
                self.timeseries.load = import_load_timeseries(
                    config_data, ts, year=self.timeseries.timeindex[0].year)
            else:
                raise ValueError('Your input for "timeseries_load" is not '
                                 'valid.'.format(mode))
            # reactive power timeseries for loads
            ts = kwargs.get('timeseries_load_reactive_power', None)
            if isinstance(ts, pd.DataFrame):
                self.timeseries.load_reactive_power = ts

            # check if time series for the set time index can be obtained
            self._check_timeindex()

    def _check_timeindex(self):
        """
        Check function to check if all feed-in and load time series contain
        values for the specified time index.

        """
        try:
            self.timeseries.generation_fluctuating
            self.timeseries.generation_dispatchable
            self.timeseries.load
            self.timeseries.generation_reactive_power
            self.timeseries.load_reactive_power
        except:
            message = 'Time index of feed-in and load time series does ' \
                      'not match.'
            logging.error(message)
            raise KeyError(message)

    def _worst_case_generation(self, worst_case_scale_factors, modes):
        """
        Define worst case generation time series for fluctuating and
        dispatchable generators.

        Parameters
        ----------
        worst_case_scale_factors : dict
            Scale factors defined in config file 'config_timeseries.cfg'.
            Scale factors describe actual power to nominal power ratio of in
            worst-case scenarios.
        modes : list
            List with worst-cases to generate time series for. Can be
            'feedin_case', 'load_case' or both.

        """

        self.timeseries.generation_fluctuating = pd.DataFrame(
            {'solar': [worst_case_scale_factors[
                           '{}_feedin_pv'.format(mode)] for mode in modes],
             'wind': [worst_case_scale_factors[
                          '{}_feedin_other'.format(mode)] for mode in modes]},
            index=self.timeseries.timeindex)

        self.timeseries.generation_dispatchable = pd.DataFrame(
            {'other': [worst_case_scale_factors[
                           '{}_feedin_other'.format(mode)] for mode in modes]},
            index=self.timeseries.timeindex)

    def _worst_case_load(self, worst_case_scale_factors,
                         peakload_consumption_ratio, modes):
        """
        Define worst case load time series for each sector.

        Parameters
        ----------
        worst_case_scale_factors : dict
            Scale factors defined in config file 'config_timeseries.cfg'.
            Scale factors describe actual power to nominal power ratio of in
            worst-case scenarios.
        peakload_consumption_ratio : dict
            Ratios of peak load to annual consumption per sector, defined in
            config file 'config_timeseries.cfg'
        modes : list
            List with worst-cases to generate time series for. Can be
            'feedin_case', 'load_case' or both.

        """

        sectors = ['residential', 'retail', 'industrial', 'agricultural']
        lv_power_scaling = np.array(
            [worst_case_scale_factors['lv_{}_load'.format(mode)]
             for mode in modes])
        mv_power_scaling = np.array(
            [worst_case_scale_factors['mv_{}_load'.format(mode)]
             for mode in modes])

        lv = {(sector, 'lv'): peakload_consumption_ratio[sector] *
                              lv_power_scaling
              for sector in sectors}
        mv = {(sector, 'mv'): peakload_consumption_ratio[sector] *
                              mv_power_scaling
              for sector in sectors}
        self.timeseries.load = pd.DataFrame({**lv, **mv},
                                            index=self.timeseries.timeindex)


class CurtailmentControl:
    """
    Sets up curtailment time series for solar and wind generators.

    Parameters
    ----------
    edisgo_object : :class:`edisgo.EDisGo`
        The parent EDisGo object that this instance is a part of.
    curtailment_methodology : :obj:`str`
        Mode defines the curtailment strategy. Possible options are:

        * 'curtail_all'
          The curtailment that has to be met in each time step is allocated
          equally to all generators depending on their share of total
          feed-in in that time step. For more information see
          :meth:`edisgo.flex_opt.curtailment.curtail_all()`.
        * 'curtail_voltage'
          The curtailment that has to be met in each time step is allocated
          based on the voltages at the generator connection points and a
          defined voltage threshold. Generators at higher voltages
          are curtailed more. The default voltage threshold is 1.0 but
          can be changed by providing the argument 'voltage_threshold'. For
          more information see
          :meth:`edisgo.flex_opt.curtailment.curtail_voltage()`.

    timeseries_curtailment : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`, optional
        Series or DataFrame containing the curtailment time series in kW. Index
        needs to be a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        Provide a Series if the curtailment time series applies to wind and
        solar generators. Provide a DataFrame if the curtailment time series
        applies to a specific technology and/or weather cell. In the first case
        columns of the DataFrame are e.g. 'solar' and 'wind'; in the second
        case columns need to be a :pandas:`pandas.MultiIndex<multiindex>` with
        the first level containing the type and the second level the weather
        cell ID.
        Curtailment time series cannot be more specific than the feed-in time
        series (e.g. if feed-in is given by technology curtailment cannot be
        given by technology and weather cell).
        Default: None.

    Attributes
    ----------

    mode : :obj:`str`
        Contains the string given by the `curtailment_methodology`
        keyword argument.
    curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`,
        Contains the *total_curtailment_ts* input object
    capacities : :pandas:`pandas.Series<series>`
        This is a series containing the nominal capacities of every single
        generator in the MV grid and the underlying LV grids. The series has a
        MultiIndex index with the following levels:

        * generator : :class:`edisgo.grid.components.GeneratorFluctuating`,
          essentially all the generator objects in the MV grid and the LV grid
        * gen_repr : :obj:`str`
          the repr strings of the generator objects from above
        * type : :obj:`str`
          the type of the generator object e.g. 'solar' or 'wind'
        * weather_cell_id : :obj:`int`
          the weather_cell_id that the generator object belongs to.
    feedin : :pandas:`pandas.DataFrame<dataframe>`
        This is a dataframe that is essentially a multiplication of the feedin timeseries
        obtained from the parameter `timeseries_generation_fluctuating` in :class:`edisgo.grid.network.EDisGo`
        and the *capacities* attribute above. Upon multiplication, this dataframe's
        columns come from the indexes of *capacities* and the dataframe's index comes
        from the `timeseries_generation_fluctuating`'s Datetimeindex. This dataframe
        is further passed on to the curtailment methodology functions.
    """

    def __init__(self, edisgo_object, **kwargs):

        self.mode = kwargs.get('curtailment_methodology', None)
        self.curtailment_ts = kwargs.get('timeseries_curtailment', None)

        if self.curtailment_ts is not None:
            self._check_timeindex(edisgo_object.network)

        # get generation fluctuating time series
        gen_fluct_ts = edisgo_object.network.timeseries.generation_fluctuating.copy()

        # get aggregated capacities by technology and weather cell id
        self.capacities = get_gen_info(edisgo_object.network, 'mvlv')
        self.capacities = self.capacities.loc[(self.capacities.type == 'solar') | (self.capacities.type == 'wind')]
        self.capacities = self.capacities.reset_index()
        self.capacities.set_index(['generator', 'gen_repr', 'type', 'weather_cell_id'], inplace=True)
        self.capacities = self.capacities.loc[:, 'nominal_capacity']

        # calculate absolute feed-in timeseries including technology and weather cell id
        self.feedin = pd.DataFrame(self.capacities).T
        self.feedin = self.feedin.append([self.feedin] * (gen_fluct_ts.index.size - 1),
                                         ignore_index=True)
        self.feedin.index = gen_fluct_ts.index.copy()
        self.feedin.columns = self.feedin.columns.remove_unused_levels()

        # multiply feedin per type/weather cell id or both to capacities
        # this is a workaround for pandas currently not allowing multiindex dataframes
        # to be multiplied  and broadcast on two levels (type and weather_cell_id)
        for x in self.feedin.columns.levels[0]:
            try:
                self.feedin.loc[:, (x, str(x), x.type, x.weather_cell_id)] = \
                    self.feedin.loc[:, (x, str(x), x.type, x.weather_cell_id)] * \
                    gen_fluct_ts.loc[:, (x.type, x.weather_cell_id)]
            except AttributeError:
                # when either weather_cell_id or type attribute is missing
                # meaning this could be a Generator Object instead of a Generator Fluctuating
                if type(x) == edisgo.grid.components.GeneratorFluctuating:
                    message = "One or both of the attributes, \'type\' or \'weather_cell_id\'" +\
                              "of the {} object is missing even though ".format(x) +\
                              "its a GeneratorFluctuating Object."
                    logging.warning(message)
                    #raise Warning(message)
                else:
                    message = "Generator Object found instead of GeneratorFluctuating Object " +\
                              "in {}".format(x)
                    logging.warning(message)
                    # raise Warning(message)
                pass
            except KeyError:
                # when one of the keys are missing in either the feedin or the capacities
                message = "One of the keys of {} are absent in either the feedin or the".format(x) +\
                          " generator fluctuating timeseries  object is missing"
                logging.warning(message)
                # raise Warning(message)
                pass

        # get mode of curtailment and the arguments necessary
        if self.mode == 'curtail_all':
            curtail_function = curtailment.curtail_all
        elif self.mode == 'curtail_voltage':
            curtail_function = curtailment.curtail_voltage
        else:
            raise ValueError('{} is not a valid mode.'.format(self.mode))

        # perform the mode of curtailment with some relevant checks
        # as to what the inputs are
        if isinstance(self.curtailment_ts, pd.Series):
            curtail_function(self.feedin,
                             self.curtailment_ts,
                             edisgo_object,
                             **kwargs)
        elif isinstance(self.curtailment_ts, pd.DataFrame):
            if isinstance(self.curtailment_ts.columns, pd.MultiIndex):
                col_tuple_list = self.curtailment_ts.columns.tolist()
                for col_slice in col_tuple_list:
                    feedin_slice = self.feedin.loc[:, (slice(None),
                                                       slice(None),
                                                       col_slice[0],
                                                       col_slice[1])]
                    feedin_slice.columns = feedin_slice.columns.remove_unused_levels()
                    if feedin_slice.size > 0:
                        curtail_function(feedin_slice,
                                         self.curtailment_ts[col_slice],
                                         edisgo_object,
                                         **kwargs)
                    else:
                        message = "In this grid there seems to be no feedin time series" +\
                            " or generators corresponding to the combination of {}".format(col_slice)
                        logging.warning(message)
            else:
                # when there is no multi-index then we assume that this is only
                # curtailed through technology or with weather cell id only
                if self.curtailment_ts.columns.dtype == object:
                    # this is when technology is given as strings
                    for tech in self.curtailment_ts.columns:
                        curtail_function(self.feedin.loc[:, (slice(None),
                                                             slice(None),
                                                             tech,
                                                             slice(None))],
                                         self.curtailment_ts[tech],
                                         edisgo_object,
                                         **kwargs)
                elif self.curtailment_ts.columns.dtype == int:
                    # this is when weather_cell_id is given as strings
                    for w_id in self.curtailment_ts.columns:
                        curtail_function(self.feedin.loc[:, (slice(None),
                                                             slice(None),
                                                             slice(None),
                                                             w_id)],
                                         self.curtailment_ts[w_id],
                                         edisgo_object,
                                         **kwargs)
                else:
                    message = 'Unallowed type {} of provided curtailment time ' \
                              'series labels. Must either be string (like \'solar\') or ' \
                              'integer (like w_id 933).'.format(type(self.curtailment_ts))
                    logging.error(message)
                    raise TypeError(message)
        else:
            message = 'Unallowed type {} of provided curtailment time ' \
                      'series. Must either be pandas.Series or ' \
                      'pandas.DataFrame.'.format(type(self.curtailment_ts))
            logging.error(message)
            raise TypeError(message)

        # update generator time series in pypsa network
        if edisgo_object.network.pypsa is not None:
            pypsa_io.update_pypsa_generator_timeseries(edisgo_object.network)

        # check if curtailment exceeds feed-in
        self._check_curtailment(edisgo_object.network)

        # write curtailment to results to be able to put it out as files for
        # result checking
        # make sure you don't overwrite existing curtailment data
        edisgo_object.network.results.assigned_curtailment =\
            edisgo_object.network.timeseries.curtailment.copy()

    def _check_timeindex(self, network):
        """
        Raises an error if time index of curtailment time series does not
        comply with the time index of load and feed-in time series.

        Parameters
        -----------
        curtailment_ts : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
            See parameter `total_curtailment_ts` in class definition for more
            information.

        """

        try:
            self.curtailment_ts.loc[network.timeseries.timeindex]
        except:
            message = 'Time index of curtailment time series does not match ' \
                      'with load and feed-in time series.'
            logging.error(message)
            raise KeyError(message)

    def _check_curtailment(self, network):
        """
        Raises an error if the curtailment at any time step exceeds the
        feed-in at that time.

        Parameters
        -----------
        feedin : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with feed-in time series in kW. The DataFrame needs to have
            the same columns as the curtailment DataFrame.
        network : :class:`~.grid.network.Network`

        """

        feedin_ts_compare = self.feedin.copy()
        for r in range(len(feedin_ts_compare.columns.levels) - 1):
            feedin_ts_compare.columns = feedin_ts_compare.columns.droplevel(1)
        # need an if condition to remove the weather_cell_id level too

        if not ((feedin_ts_compare.loc[
                 :, network.timeseries.curtailment.columns] -
                 network.timeseries.curtailment) > -1e-3).all().all():
            message = 'Curtailment exceeds feed-in.'
            logging.error(message)
            raise TypeError(message)


class StorageControl:
    """
    Integrates storages into the grid.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    timeseries_battery : :obj:`str` or :pandas:`pandas.Series<series>` or :obj:`dict`
        Parameter used to obtain time series of active power the battery
        storage(s) is/are charged (negative) or discharged (positive) with. Can
        either be a given time series or an operation strategy.
        Possible options are:

        * Time series
          Time series the storage will be charged and discharged with can be
          set directly by providing a :pandas:`pandas.Series<series>` with
          time series of active charge (negative) and discharge (positive)
          power, normalized with corresponding storage capacity. Index needs
          to be a :pandas:`pandas.DatetimeIndex<datetimeindex>`.
          In case of more than one storage provide a :obj:`dict` where each
          entry represents a storage. Keys of the dictionary have to match
          the keys of the `battery_parameters dictionary`, values must
          contain the corresponding time series as
          :pandas:`pandas.Series<series>`.
        * 'fifty-fifty'
          Storage operation depends on actual power of generators. If
          cumulative generation exceeds 50% of the nominal power, the storage
          will charge. Otherwise, the storage will discharge.

        Default: None.
    battery_parameters : :obj:`dict`
        Dictionary with storage parameters. Format must be as follows:

        .. code-block:: python

            {
                'nominal_capacity': <float>, # in kWh
                'soc_initial': <float>, # in kWh
                'efficiency_in': <float>, # in per unit 0..1
                'efficiency_out': <float>, # in per unit 0..1
                'standing_loss': <float> # in per unit 0..1
            }

        In case of more than one storage provide a :obj:`dict` where each
        entry represents a storage. Keys of the dictionary have to match
        the keys of the `timeseries_battery` dictionary, values must
        contain the corresponding parameters dictionary specified above.
    battery_position : None or :obj:`str` or :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee` or :obj:`dict`
        To position the storage a positioning strategy can be used or a
        node in the grid can be directly specified. Possible options are:

        * 'hvmv_substation_busbar'
          Places a storage unit directly at the HV/MV station's bus bar.
        * :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee`
          Specifies a node the storage should be connected to.

        In case of more than one storage provide a :obj:`dict` where each
        entry represents a storage. Keys of the dictionary have to match
        the keys of the `timeseries_battery` and `battery_parameters`
        dictionaries, values must contain the corresponding positioning
        strategy or node to connect the storage to.

    """

    def __init__(self, network, timeseries_battery, battery_parameters,
                 battery_position):

        self.network = network

        if isinstance(timeseries_battery, dict):
            for storage, ts in timeseries_battery.items():
                try:
                    params = battery_parameters[storage]
                    position = battery_position[storage]
                except KeyError:
                    message = 'Please provide storage parameters or ' \
                              'position for storage {}.'.format(storage)
                    logging.error(message)
                    raise KeyError(message)
                self._integrate_storage(ts, params, position)
        else:
            self._integrate_storage(timeseries_battery, battery_parameters,
                                    battery_position)

    def _integrate_storage(self, timeseries, params, position):
        """
        Integrate storage units in the grid and specify its operational mode.

        Parameters
        ----------
        timeseries : :obj:`str` or :pandas:`pandas.Series<series>`
            Parameter used to obtain time series of active power the battery
            storage is charged (negative) or discharged (positive) with. Can
            either be a given time series or an operation strategy. See class
            definition for more information
        params : :obj:`dict`
            Dictionary with storage parameters for one storage. See class
            definition for more information on what parameters must be
            provided.
        position : :obj:`str` or :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee`
            Parameter used to place the storage. See class definition for more
            information.

        """
        # place storage
        if position == 'hvmv_substation_busbar':
            storage = storage_integration.storage_at_hvmv_substation(
                self.network.mv_grid, params)
        elif isinstance(position, Station) or isinstance(position, BranchTee):
            storage = storage_integration.set_up_storage(
                params, position)
            storage_integration.connect_storage(storage, position)
        else:
            message = 'Provided battery position option {} is not ' \
                      'valid.'.format(timeseries)
            logging.error(message)
            raise KeyError(message)

        # implement operation strategy
        if isinstance(timeseries, pd.Series):
            # ToDo: Eingabe von Blindleistung auch ermöglichen?
            timeseries = pd.DataFrame(data={'p': timeseries,
                                            'q': [0] * len(timeseries)},
                                      index=timeseries.index)
            self._check_timeindex(timeseries)
            storage.timeseries = timeseries
        elif timeseries == 'fifty-fifty':
            storage_operation.fifty_fifty(storage)
        else:
            message = 'Provided battery timeseries option {} is not ' \
                      'valid.'.format(timeseries)
            logging.error(message)
            raise KeyError(message)

    def _check_timeindex(self, timeseries):
        """
        Raises an error if time index of battery time series does not
        comply with the time index of load and feed-in time series.

        Parameters
        -----------
        timeseries : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power the storage is charged (negative)
            and discharged (positive) with in kW in column 'p' and
            reactive power in kVA in column 'q'.

        """
        try:
            timeseries.loc[self.network.timeseries.timeindex]
        except:
            message = 'Time index of battery time series does not match ' \
                      'with load and feed-in time series.'
            logging.error(message)
            raise KeyError(message)


class TimeSeries:
    """
    Defines time series for all loads and generators in network (if set).

    Contains time series for loads (sector-specific), generators
    (technology-specific), and curtailment (technology-specific).

    Attributes
    ----------
    generation_fluctuating : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with active power feed-in time series for fluctuating
        renewables solar and wind, normalized with corresponding capacity.
        Time series can either be aggregated by technology type or by type
        and weather cell ID. In the first case columns of the DataFrame are
        'solar' and 'wind'; in the second case columns need to be a
        :pandas:`pandas.MultiIndex<multiindex>` with the first level
        containing the type and the second level the weather cell ID.
        Default: None.
    generation_dispatchable : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with time series for active power of each (aggregated)
        type of dispatchable generator normalized with corresponding capacity.
        Columns represent generator type:

        * 'gas'
        * 'coal'
        * 'biomass'
        * 'other'
        * ...

        Use 'other' if you don't want to explicitly provide every possible
        type. Default: None.
    generation_reactive_power : :pandas: `pandasDataFrame<dataframe>`, optional
        DataFrame with reactive power per technology and weather cell ID,
        normalized with the nominal active power.
        Time series can either be aggregated by technology type or by type
        and weather cell ID. In the first case columns of the DataFrame are
        'solar' and 'wind'; in the second case columns need to be a
        :pandas:`pandas.MultiIndex<multiindex>` with the first level
        containing the type and the second level the weather cell ID.
        If the technology doesn't contain weather cell information, i.e.
        if it is other than solar or wind generation,
        this second level can be left as a numpy Nan or a None.
        Default: None.
    load : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with active power of load time series of each (cumulative)
        type of load, normalized with corresponding annual energy demand.
        Columns represent load type:

        * 'residential'
        * 'retail'
        * 'industrial'
        * 'agricultural'

         Default: None.
    load_reactive_power : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with time series of normalized reactive power (normalized by
        annual energy demand) per load sector. Index needs to be a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
        Columns represent load type:

          * 'residential'
          * 'retail'
          * 'industrial'
          * 'agricultural'

        Default: None.
    curtailment : :pandas:`pandas.DataFrame<dataframe>` or List, optional
        In the case curtailment is applied to all fluctuating renewables
        this needs to be a DataFrame with active power curtailment time series.
        Time series can either be aggregated by technology type or by type
        and weather cell ID. In the first case columns of the DataFrame are
        'solar' and 'wind'; in the second case columns need to be a
        :pandas:`pandas.MultiIndex<multiindex>` with the first level
        containing the type and the second level the weather cell ID.
        In the case curtailment is only applied to specific generators, this
        parameter needs to be a list of all generators that are curtailed.
        Default: None.
    timeindex : :pandas:`pandas.DatetimeIndex<datetimeindex>`, optional
        Can be used to define a time range for which to obtain the provided
        time series and run power flow analysis. Default: None.

    See also
    --------
    `timeseries` getter in :class:`~.grid.components.Generator`,
    :class:`~.grid.components.GeneratorFluctuating` and
    :class:`~.grid.components.Load`.

    """

    def __init__(self, network, **kwargs):
        self.network = network
        self._generation_dispatchable = kwargs.get('generation_dispatchable',
                                                   None)
        self._generation_fluctuating = kwargs.get('generation_fluctuating',
                                                  None)
        self._generation_reactive_power = kwargs.get(
            'generation_reactive_power', None)
        self._load = kwargs.get('load', None)
        self._load_reactive_power = kwargs.get('load_reacitve_power', None)
        self._curtailment = kwargs.get('curtailment', None)
        self._timeindex = kwargs.get('timeindex', None)
        self._timesteps_load_feedin_case = None

    @property
    def generation_dispatchable(self):
        """
        Get generation time series of dispatchable generators (only active
        power)

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        try:
            return self._generation_dispatchable.loc[[self.timeindex], :]
        except:
            return self._generation_dispatchable.loc[self.timeindex, :]

    @generation_dispatchable.setter
    def generation_dispatchable(self, generation_dispatchable_timeseries):
        self._generation_dispatchable = generation_dispatchable_timeseries

    @property
    def generation_fluctuating(self):
        """
        Get generation time series of fluctuating renewables (only active
        power)

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        try:
            return self._generation_fluctuating.loc[[self.timeindex], :]
        except:
            return self._generation_fluctuating.loc[self.timeindex, :]

    @generation_fluctuating.setter
    def generation_fluctuating(self, generation_fluc_timeseries):
        self._generation_fluctuating = generation_fluc_timeseries

    @property
    def generation_reactive_power(self):
        """
        Get reactive power time series for generators normalized by nominal
        active power.

        Returns
        -------
        :pandas: `pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._generation_reactive_power is not None:
            return self._generation_reactive_power.loc[self.timeindex, :]
        else:
            return None

    @generation_reactive_power.setter
    def generation_reactive_power(self, generation_reactive_power_timeseries):
        self._generation_reactive_power = generation_reactive_power_timeseries

    @property
    def load(self):
        """
        Get load time series (only active power)

        Returns
        -------
        dict or :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        try:
            return self._load.loc[[self.timeindex], :]
        except:
            return self._load.loc[self.timeindex, :]

    @load.setter
    def load(self, load_timeseries):
        self._load = load_timeseries

    @property
    def load_reactive_power(self):
        """
        Get reactive power time series for load normalized by annual
        consumption.

        Returns
        -------
        :pandas: `pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._load_reactive_power is not None:
            return self._load_reactive_power.loc[self.timeindex, :]
        else:
            return None

    @load_reactive_power.setter
    def load_reactive_power(self, load_reactive_power_timeseries):
        self._load_reactive_power = load_reactive_power_timeseries

    @property
    def timeindex(self):
        """
        Parameters
        ----------
        time_range : :pandas:`pandas.DatetimeIndex<datetimeindex>`
            Time range of power flow analysis

        Returns
        -------
        :pandas:`pandas.DatetimeIndex<datetimeindex>`
            See class definition for details.

        """
        return self._timeindex

    @property
    def curtailment(self):
        """
        Get curtailment time series of dispatchable generators (only active
        power)

        Parameters
        ----------
        curtailment : list or :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            In the case curtailment is applied to all solar and wind generators
            curtailment time series either aggregated by technology type or by
            type and weather cell ID are returnded. In the first case columns
            of the DataFrame are 'solar' and 'wind'; in the second case columns
            need to be a :pandas:`pandas.MultiIndex<multiindex>` with the
            first level containing the type and the second level the weather
            cell ID.
            In the case curtailment is only applied to specific generators,
            curtailment time series of all curtailed generators, specified in
            by the column name are returned.

        """
        if self._curtailment is not None:
            if isinstance(self._curtailment, pd.DataFrame):
                try:
                    return self._curtailment.loc[[self.timeindex], :]
                except:
                    return self._curtailment.loc[self.timeindex, :]
            elif isinstance(self._curtailment, list):
                try:
                    curtailment = pd.DataFrame()
                    for gen in self._curtailment:
                        curtailment[gen] = gen.curtailment
                    return curtailment
                except:
                    raise
        else:
            return None

    @curtailment.setter
    def curtailment(self, curtailment):
        self._curtailment = curtailment

    @property
    def timesteps_load_feedin_case(self):
        """
        Contains residual load and information on feed-in and load case.

        Residual load is calculated from total (load - generation) in the grid.
        Grid losses are not considered.

        Feed-in and load case are identified based on the
        generation and load time series and defined as follows:

        1. Load case: positive (load - generation) at HV/MV substation
        2. Feed-in case: negative (load - generation) at HV/MV substation

        See also :func:`~.tools.tools.assign_load_feedin_case`.

        Parameters
        -----------
        timeseries_load_feedin_case : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with information on whether time step is handled as load
            case ('load_case') or feed-in case ('feedin_case') for each time
            step in :py:attr:`~timeindex`. Index of the series is the
            :py:attr:`~timeindex`.

        Returns
        -------
        :pandas:`pandas.Series<series>`
            Series with information on whether time step is handled as load
            case ('load_case') or feed-in case ('feedin_case') for each time
            step in :py:attr:`~timeindex`.
            Index of the dataframe is :py:attr:`~timeindex`. Columns of the
            dataframe are 'residual_load' with (load - generation) in kW at
            HV/MV substation and 'case' with 'load_case' for positive residual
            load and 'feedin_case' for negative residual load.

        """
        if self._timesteps_load_feedin_case is None:
            tools.assign_load_feedin_case(self.network)
        return self._timesteps_load_feedin_case

    @timesteps_load_feedin_case.setter
    def timesteps_load_feedin_case(self, timeseries_load_feedin_case):
        self._timesteps_load_feedin_case = timeseries_load_feedin_case


class Results:
    """
    Power flow analysis results management

    Includes raw power flow analysis results, history of measures to increase
    the grid's hosting capacity and information about changes of equipment.

    Attributes
    ----------
    measures: list
        A stack that details the history of measures to increase grid's hosting
        capacity. The last item refers to the latest measure. The key
        `original` refers to the state of the grid topology as it was initially
        imported.

    """

    # ToDo: maybe add setter to alter list of measures

    def __init__(self):
        self._measures = ['original']
        self._pfa_p = None
        self._pfa_q = None
        self._pfa_v_mag_pu = None
        self._i_res = None
        self._equipment_changes = pd.DataFrame()
        self._grid_expansion_costs = None
        self._grid_losses = None
        self._grid_exchanges = None
        self._assigned_curtailment = None
        self._unresolved_issues = {}

    @property
    def pfa_p(self):
        """
        Active power results from power flow analysis in kW.

        Holds power flow analysis results for active power for the last
        iteration step. Index of the DataFrame is a DatetimeIndex indicating
        the time period the power flow analysis was conducted for; columns
        of the DataFrame are the edges as well as stations of the grid
        topology.

        Parameters
        ----------
        pypsa: `pandas.DataFrame<dataframe>`
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
        of the DataFrame are the edges as well as stations of the grid
        topology.

        Parameters
        ----------
        pypsa: `pandas.DataFrame<dataframe>`
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
        columns of the DataFrame are the nodes as well as stations of the grid
        topology.

        Parameters
        ----------
        pypsa: `pandas.DataFrame<dataframe>`
            Results time series of voltage deviation in p.u. from the
            `PyPSA network <https://www.pypsa.org/doc/components.html#network>`_

            Provide this if you want to set values. For retrieval of data do
            not pass an argument

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Voltage level nodes of grid

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
        of the DataFrame are the edges as well as stations of the grid
        topology.

        Parameters
        ----------
        pypsa: `pandas.DataFrame<dataframe>`
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
        :class:`~.grid.components.Line`, :class:`~.grid.components.Station`,
        etc.) and has the following columns:

        equipment : detailing what was changed (line, station, battery,
        curtailment). For ease of referencing we take the component itself.
        For lines we take the line-dict, for stations the transformers, for
        batteries the battery-object itself and for curtailment
        either a dict providing the details of curtailment or a curtailment
        object if this makes more sense (has to be defined).

        change : :obj:`str` {'added' | 'removed'}
            says if something was added or removed

        iteration_step : int
            Used for the update of the pypsa network to only consider changes
            since the last power flow analysis.

        quantity : int
            Number of components added or removed. Only relevant for
            calculation of grid expansion costs to keep track of how many
            new standard lines were added.

        Parameters
        ----------
        changes: `pandas.DataFrame<dataframe>`
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
        Holds grid expansion costs in kEUR due to grid expansion measures
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
            that can either be a :class:`~.grid.components.Line` or a
            :class:`~.grid.components.Transformer`. Columns are the following:

            type: String
                Transformer size or cable name

            total_costs: float
                Costs of equipment in kEUR. For lines the line length and
                number of parallel lines is already included in the total
                costs.

            quantity: int
                For transformers quantity is always one, for lines it specifies
                the number of parallel lines.

            line_length: float
                Length of line or in case of parallel lines all lines in km.

            voltage_level : :obj:`str` {'lv' | 'mv' | 'mv/lv'}
                Specifies voltage level the equipment is in.

            mv_feeder : :class:`~.grid.components.Line`
                First line segment of half-ring used to identify in which
                feeder the grid expansion was conducted in.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Costs of each reinforced equipment in kEUR.

        Notes
        -------
        Total grid expansion costs can be obtained through
        costs.total_costs.sum().

        """
        return self._grid_expansion_costs

    @grid_expansion_costs.setter
    def grid_expansion_costs(self, total_costs):
        self._grid_expansion_costs = total_costs

    @property
    def grid_losses(self):
        """
        Holds the losses in the grid obtained from the slack bus in
        kW and kvar.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Total Losses, both active and reactive power losses
            per timestep
        """

        return self._grid_losses

    @grid_losses.setter
    def grid_losses(self, pypsa_grid_losses):
        self._grid_losses = pypsa_grid_losses

    @property
    def grid_exchanges(self):
        """
        Holds the grid powers (active and reactive) transfered to the higher voltage
        level through the slack

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>
            Total power exchanged to the higher voltage network through slack not
            including the grid losses
        """

        return self._grid_exchanges

    @grid_exchanges.setter
    def grid_exchanges(self, grid_exchanges):
        self._grid_exchanges = grid_exchanges

    @property
    def assigned_curtailment(self):
        """
        Holds the curtailment assigned to each generator.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>
            curtailment per generator (in columns) in timesteps(rows).
        """

        return self._assigned_curtailment

    @assigned_curtailment.setter
    def assigned_curtailment(self, assigned_curtailment):
        self._assigned_curtailment = assigned_curtailment
        self.assigned_curtailment.sort_index(inplace=True)

    @property
    def unresolved_issues(self):
        """
        Holds lines and nodes where over-loading or over-voltage issues
        could not be solved in grid reinforcement.

        In case over-loading or over-voltage issues could not be solved
        after maximum number of iterations, grid reinforcement is not
        aborted but grid expansion costs are still calculated and unresolved
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

            S = max(\sqrt{p0^2 + q0^2}, \sqrt{p1^2 + q1^2})

        Parameters
        ----------
        components : :class:`~.grid.components.Line` or
            :class:`~.grid.components.Transformer`
            Could be a list of instances of these classes

            Line or Transformers objects of grid topology. If not provided
            (respectively None) defaults to return `s_res` of all lines and
            transformers in the grid.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Apparent power for `lines` and/or `transformers`

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
                print("Apparent power for {lines} are not returned from "
                      "PFA".format(lines=labels_not_included))
        else:
            labels_included = self.pfa_p.columns

        s_res = ((self.pfa_p[labels_included] ** 2 + self.pfa_q[
            labels_included] ** 2)).applymap(sqrt)

        return s_res

    def v_res(self, nodes=None, generators=None, level=None):
        """
        Get resulting voltage level at node

        Parameters
        ----------
        nodes :  {:class:`~.grid.components.Load`, :class:`~.grid.components.Generator`, ...} or :obj:`list` of
            grid topology component or `list` grid topology components
            If not provided defaults to column names available in grid level
            `level`
        level : str
            Either 'mv' or 'lv' or None (default). Depending which grid level results you are
            interested in. It is required to provide this argument in order
            to distinguish voltage levels at primary and secondary side of the
            transformer/LV station.
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
            message = "No Power Flow Calculation has be done yet, so there are no results yet."
            raise AttributeError

        if level is None:
            level = ['mv', 'lv']

        if nodes is None:
            return self.pfa_v_mag_pu.loc[:, (level, slice(None))]
        else:
            labels = list(map(repr, nodes.copy()))
            not_included = [_ for _ in labels
                            if _ not in list(self.pfa_v_mag_pu[level].columns)]
            labels_included = [_ for _ in labels if _ not in not_included]

            if not_included:
                logging.info("Voltage levels for {nodes} are not returned from PFA".format(
                nodes=not_included))
            return self.pfa_v_mag_pu[level][labels_included]

    def save(self, directory, create_plots=False, **kwargs):
        """
        Save all results to disk in a folder.

        Parameters
        ----------
        directory: :obj:`str
            path to save the plots
        """
        powerflow_results_dir = os.path.join(directory, 'powerflow_results')
        calculated_results_dir = os.path.join(directory, 'calculated_results')

        os.makedirs(powerflow_results_dir, exist_ok=True)
        os.makedirs(calculated_results_dir, exist_ok=True)

        # put out important information at the top level

        # put out all relevant power_flow results
        # voltage
        voltage_pu_file = os.path.join(powerflow_results_dir, 'voltages_pu.csv')
        self.pfa_v_mag_pu.to_csv(voltage_pu_file)

        # current
        current_file = os.path.join(powerflow_results_dir, 'currents.csv')
        self.i_res.to_csv(current_file)

        # active power
        acitve_power_file = os.path.join(powerflow_results_dir, 'active_powers.csv')
        self.pfa_p.to_csv(acitve_power_file)

        # reactive power
        reacitve_power_file = os.path.join(powerflow_results_dir, 'reactive_powers.csv')
        self.pfa_q.to_csv(reacitve_power_file)

        # apparent power
        apparent_power_file = os.path.join(powerflow_results_dir, 'apparent_powers.csv')
        self.s_res().to_csv(apparent_power_file)

        # put out all relevant calculated results
        # grid losses
        grid_losses_file = os.path.join(calculated_results_dir, 'grid_losses.csv')
        self.grid_losses.to_csv(grid_losses_file)

        # grid exchanges
        grid_exchanges_file = os.path.join(calculated_results_dir, 'grid_exchanges.csv')
        self.grid_exchanges.to_csv(grid_exchanges_file)

        # assigned curtailment
        if self.assigned_curtailment is not None:
            assigned_curtailment_file = os.path.join(calculated_results_dir, 'assigned_curtailment.csv')
            self.assigned_curtailment.to_csv(assigned_curtailment_file)

        # equipment_changes
        equipment_changes_file = os.path.join(calculated_results_dir, 'equipment_changes.csv')
        self.equipment_changes.to_csv(equipment_changes_file)

        # grid_expansion_costs
        if self.grid_expansion_costs is not None:
            grid_expansion_costs_file = os.path.join(calculated_results_dir, 'grid_expansion_costs.csv')
            self.grid_expansion_costs.to_csv(grid_expansion_costs_file)

        # unresolved_issues
        unresolved_issues_file = os.path.join(calculated_results_dir, 'unresolved_issues.csv')
        pd.DataFrame(self.unresolved_issues).to_csv(unresolved_issues_file)
