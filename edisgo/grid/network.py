from os import path
import pandas as pd
from math import sqrt
import logging

import edisgo
from edisgo.tools import config, pypsa_io
from edisgo.data.import_data import import_from_ding0, import_generators, \
    import_feedin_timeseries, import_load_timeseries
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.flex_opt.storage_integration import integrate_storage


logger = logging.getLogger('edisgo')

class Network:
    """Defines the eDisGo Network

    Used as container for all data related to a single
    :class:`~.grid.grids.MVGrid`.
    Provides the top-level API for invocation of data import, analysis of
    hosting capacity, grid reinforce and flexibility measures.

    Examples
    --------
    Assuming you the Ding0 `ding0_data.pkl` in CWD

    Create eDisGo Network object by loading Ding0 file

    >>> from edisgo.grid.network import Network
    >>> network = Network.import_from_ding0('ding0_data.pkl'))

    Analyze hosting capacity for MV grid level

    >>> network.analyze(mode='mv')

    Print LV station secondary side voltage levels returned by PFA

    >>> lv_stations = network.mv_grid.graph.nodes_by_attribute('lv_station')
    >>> print(network.results.v_res(lv_stations, 'lv'))

    Attributes
    ----------
    _id : :obj:`str`
        Name of network
    _equipment_data : :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
        Electrical equipment such as lines and transformers
    _config : ???
        #TODO: TBD
    _metadata : :obj:`dict`
        Metadata of Network such as ?
    _data_sources : :obj:`dict` of :obj:`str`
        Data Sources of grid, generators etc.
        Keys: 'grid', 'generators', ?
    _scenario : :class:`~.grid.grids.Scenario`
        Scenario which is used for calculations
    _pypsa : :pypsa:`pypsa.Network<network>`
        PyPSA representation of grid topology
    _dingo_import_data :
        Temporary data from ding0 import which are needed for OEP generator update
    """

    def __init__(self, **kwargs):
        if 'pypsa' not in kwargs.keys():
            self._id = kwargs.get('id', None)
            self._metadata = kwargs.get('metadata', None)
            self._data_sources = kwargs.get('data_sources', {})
            self._scenario = kwargs.get('scenario', None)

            if self._scenario is not None:
                self._scenario.network = self

            self._mv_grid = kwargs.get('mv_grid', None)
            self._pypsa = None
            self.results = Results()
        else:
            self._pypsa = kwargs.get('pypsa', None)

        self._config = kwargs.get('config', None)
        if self._config is None:
            try:
                self._config = self._scenario.config
            except:
                self._config = Config()
        self._equipment_data = self._load_equipment_data()

        self._dingo_import_data = []

    def _load_equipment_data(self):
        """Load equipment data for transformers, cables etc.

        Returns
        -------
        :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
        """

        package_path =  edisgo.__path__[0]
        equipment_dir = self.config['system_dirs']['equipment_dir']

        data = {}

        equipment_mv_parameters_trafos = self.config['equipment'][
            'equipment_mv_parameters_trafos']
        data['MV_trafos'] = pd.read_csv(
            path.join(package_path, equipment_dir,
                      equipment_mv_parameters_trafos),
            comment='#', index_col='name',
            delimiter=',', decimal='.')

        equipment_mv_parameters_lines = self.config['equipment'][
            'equipment_mv_parameters_lines']
        data['MV_lines'] = pd.read_csv(
            path.join(package_path, equipment_dir,
                      equipment_mv_parameters_lines),
            comment='#', index_col='name',
            delimiter=',', decimal='.')

        equipment_mv_parameters_cables = self.config['equipment'][
            'equipment_mv_parameters_cables']
        data['MV_cables'] = pd.read_csv(
            path.join(package_path, equipment_dir,
                      equipment_mv_parameters_cables),
            comment='#', index_col='name',
            delimiter=',', decimal='.')

        equipment_lv_parameters_cables = self.config['equipment'][
            'equipment_lv_parameters_cables']
        data['LV_cables'] = pd.read_csv(
            path.join(package_path, equipment_dir,
                      equipment_lv_parameters_cables),
            comment='#', index_col='name',
            delimiter=',', decimal='.')

        equipment_lv_parameters_trafos = self.config['equipment'][
            'equipment_lv_parameters_trafos']
        data['LV_trafos'] = pd.read_csv(
            path.join(package_path, equipment_dir,
                      equipment_lv_parameters_trafos),
            comment='#', index_col='name',
            delimiter=',', decimal='.')

        return data

    @classmethod
    def import_from_ding0(cls, file, **kwargs):
        """Import grid data from DINGO file

        For details see
        :func:`edisgo.data.import_data.import_from_ding0`
        """

        # create the network instance
        network = cls(**kwargs)

        # call the importer
        import_from_ding0(file=file,
                          network=network)

        # integrate storage into grid in case ETraGo Specs are given
        scenario = kwargs.get('scenario', None)
        if scenario and scenario.etrago_specs and \
            scenario.etrago_specs.battery_capacity:
            integrate_storage(network,
                              position='hvmv_substation_busbar',
                              operational_mode='etrago-specs',
                              parameters={
                                  'nominal_capacity': \
                                      scenario.etrago_specs.battery_capacity,
                                  'soc_initial': 0.0,
                                  'efficiency_in': 1.0,
                                  'efficiency_out': 1.0,
                                  'standing_loss': 0})

        return network

    def import_generators(self, types=None):
        """Import generators

        For details see
        :func:`edisgo.data.import_data.import_generators`
        """
        data_source = data_source=self.config['data']['data_source']
        import_generators(network=self,
                          data_source=data_source,
                          types=types)

    def analyze(self, mode=None):
        """Analyzes the grid by power flow analysis

        Analyze the grid for violations of hosting capacity. Means, perform a
        power flow analysis and obtain voltages at nodes (load, generator,
        stations/transformers and branch tees) and active/reactive power at
        lines.

        The power flow analysis can be performed for both grid levels MV and LV
        and for both of them individually. Use `mode` to choose (defaults to
        MV + LV).

        A static `non-linear power flow analysis is performed using PyPSA
        <https://www.pypsa.org/doc/power_flow.html#full-non-linear-power-flow>`_.
        The high-voltage to medium-voltage transformer are not included in the
        analysis. The slack bus is defined at secondary side of these
        transformers assuming an ideal tap changer. Hence, potential overloading
        of the transformers is not studied here.

        Parameters
        ----------
        mode: str
            Allows to toggle between power flow analysis (PFA) on the whole grid
            topology (MV + LV), only MV or only LV. Therefore, either specify
            `mode='mv'` for PFA of the MV grid topology or `mode='lv'`
            for PFA of the LV grid topology.
            Defaults to None which equals power flow analysis for MV + LV.

        Notes
        -----
        The current implementation always translates the grid topology
        representation to the PyPSA format and stores it to :attr:`self._pypsa`.

        TODO: extend doctring by

        * How power plants are modeled, if possible use a link
        * Where to find and adjust power flow analysis defining parameters

        See Also
        --------
        :func:~.tools.pypsa_io.to_pypsa
            Translator to PyPSA data format

        """
        if self.pypsa is None:
            # Translate eDisGo grid topology representation to PyPSA format
            self.pypsa = pypsa_io.to_pypsa(self, mode)
        else:
            # Update PyPSA data with equipment changes
            pypsa_io.update_pypsa(self)

        # run power flow analysis
        pf_results = self.pypsa.pf(self.pypsa.snapshots)

        if all(pf_results['converged']['0'].tolist()) == True:
            pypsa_io.process_pfa_results(self, self.pypsa)
        else:
            raise ValueError("Power flow analysis did not converge.")

    def reinforce(self, **kwargs):
        """Reinforces the grid and calculates grid expansion costs"""
        reinforce_grid(
            self, max_while_iterations=kwargs.get('max_while_iterations', 10))
        self.results.grid_expansion_costs = grid_expansion_costs(self)

    def integrate_storage(self, **kwargs):
        """Integrate storage in grid"""
        integrate_storage(network=self,
                          position=kwargs.get('position', None),
                          operational_mode=kwargs.get(
                              'operational_mode', None),
                          parameters=kwargs.get('parameters', None))

    @property
    def id(self):
        """Returns id of network"""
        return self._id

    @property
    def config(self):
        """Returns config object"""
        return self._config.data

    @property
    def equipment_data(self):
        """Returns equipment data object

        Electrical equipment such as lines and transformers
        :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
        """
        return self._equipment_data

    @property
    def mv_grid(self):
        """:class:`~.grid.grids.MVGrid` : Medium voltage (MV) grid

        Retrieve the instance of the loaded MV grid
        """
        return self._mv_grid

    @mv_grid.setter
    def mv_grid(self, mv_grid):
        self._mv_grid = mv_grid

    @property
    def data_sources(self):
        """:obj:`dict` of :obj:`str` : Data Sources

        """
        return self._data_sources

    def set_data_source(self, key, data_source):
        """Set data source for key (e.g. 'grid')
        """
        self._data_sources[key] = data_source

    @property
    def dingo_import_data(self):
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
        this data model the :pypsa:`pypsa.Network<network>`
        is assigned to this attribute.
        This allows as well to overwrite data.

        Parameters
        ----------
        pypsa:
            The `PyPSA network <https://www.pypsa.org/doc/components.html#network>`_
            container

        Returns
        -------
        :pypsa:`pypsa.Network<network>`
            PyPSA grid representation

        """
        return self._pypsa

    @pypsa.setter
    def pypsa(self, pypsa):
        self._pypsa = pypsa

    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, scenario):
        self._scenario = scenario

    def __repr__(self):
        return 'Network ' + str(self._id)


class Config:
    """Defines the configurations

    Used as container for all configurations.

    """
    #ToDo: add docstring
    def __init__(self, **kwargs):
        self._data = self._load_config()

    @staticmethod
    def _load_config():
        """Load config files

        Returns
        -------
        config object
        """

        # load config
        config.load_config('config_db_tables.cfg')
        config.load_config('config_data.cfg')
        config.load_config('config_flexopt.cfg')
        config.load_config('config_misc.cfg')
        config.load_config('config_scenario.cfg')
        config.load_config('config_costs.cfg')

        confic_dict = config.cfg._sections

        # convert numeric values to float
        for sec, subsecs in confic_dict.items():
            for subsec, val in subsecs.items():
                # try str -> float conversion
                try:
                    confic_dict[sec][subsec] = float(val)
                except:
                    pass

        # modify structure of config data
        confic_dict['data']['peakload_consumption_ratio'] = {
            'residential': confic_dict['data'][
                'residential_peakload_consumption'],
            'retail': confic_dict['data'][
                'retail_peakload_consumption'],
            'industrial': confic_dict['data'][
                'residential_peakload_consumption'],
            'agricultural': confic_dict['data'][
                'agricultural_peakload_consumption']}

        del (confic_dict['data']['residential_peakload_consumption'])
        del (confic_dict['data']['retail_peakload_consumption'])
        del (confic_dict['data']['industrial_peakload_consumption'])
        del (confic_dict['data']['agricultural_peakload_consumption'])

        return confic_dict

    @property
    def data(self):
        return self._data


class Scenario:
    """Defines an eDisGo scenario

    It contains parameters and links to further data that is used for
    calculations within eDisGo.

    Parameters
    ----------
    power_flow : str or tuple of two :obj:`datetime` objects.
        Define time range of power flow analysis. Either choose 'worst-case' to
        analyze feedin worst-case. Or analyze a timerange based on actual power
        generation and demand data.

        For input of type str only 'worst-case' is a valid input.
        To specify the time range for a power flow analysis provide the start
        and end time as 2-tuple of :obj:`datetime`

    Optional Parameters
    --------------------
    timeseries : :obj:`list` of :class:`~.grid.grids.TimeSeries`
        Time series associated with a scenario. Only specify if you don't
        want to do a worst-case analysis and are not using etrago
        specifications.
    pfac_mv_gen : :obj:`float`
        Power factor for medium voltage generators
    pfac_mv_load : :obj:`float`
        Power factor for medium voltage loads
    pfac_lv_gen : :obj:`float`
        Power factor for low voltage generators
    pfac_lv_load : :obj:`float`
        Power factor for low voltage loads

    Attributes
    ----------
    _name : :obj:`str`
        Scenario name (e.g. "feedin case weather 2011")
    _mv_grid_id : :obj:`str`
        ID of MV grid district
    _mode : :obj:`str`
        'worst-case' or 'time-range'
    _config : :class:~.grid.network.Config`
        Configuration parameters
    _timeseries : :obj:`list` of :class:`~.grid.grids.TimeSeries`
        Time series associated with a scenario.
    _etrago_specs : :class:`~.grid.grids.ETraGoSpecs`
        Specifications which are to be fulfilled at transition point (HV-MV
        substation)
    _parameters : :class:`~.grid.network.Parameters`
        Parameters for power flow analysis and grid expansion.
    scenario_name : str
        Specify a scenario that is used to distinguish data, assumptions and
        parameter.

    """

    def __init__(self, power_flow, mv_grid_id, **kwargs):
        self._mv_grid_id = mv_grid_id
        self._name = kwargs.get('name', None)
        self._config = kwargs.get('config', None)
        self._timeseries = kwargs.get('timeseries', None)
        self._etrago_specs = kwargs.get('etrago_specs', None)
        self._parameters = Parameters(self, **kwargs)
        self.scenario_name = kwargs.get('scenario_name', None)
        self._curtailment = kwargs.get('curtailment', None)

        # get config parameters if not provided
        if self._config is None:
            self._config = Config()
        # populate timeseries attribute
        self.set_timeseries(power_flow)

    @property
    def mv_grid_id(self):
        return self._mv_grid_id

    @property
    def timeseries(self):
        return self._timeseries

    @property
    def parameters(self):
        return self._parameters

    @property
    def config(self):
        return self._config

    @property
    def mode(self):
        return self._mode

    @property
    def etrago_specs(self):
        return self._etrago_specs

    def set_timeseries(self, power_flow):
        if isinstance(power_flow, str):
            if power_flow != 'worst-case':
                raise ValueError(
                    "{} is not a valid specification for type of power flow "
                    "analysis. Try 'worst-case'".format(power_flow))
            else:
                self._mode = 'worst-case'
                if self._etrago_specs:
                    logger.warning("Dispatch specifications from etrago are "
                                   "overwritten when power_flow is set to "
                                   "'worst-case'.")
                if self._timeseries:
                    logger.warning("Timeseries are overwritten when "
                                   "power_flow is set to 'worst-case'.")
                self._timeseries = TimeSeries()
                self._timeseries.generation = \
                    self._timeseries.worst_case_generation_ts()
                self._timeseries.load = self._timeseries.worst_case_load_ts(
                    self)
        elif isinstance(power_flow, tuple):
            if self._etrago_specs:
                if self._etrago_specs.dispatch is not None:
                    self._mode = 'time-range'
                    self._timeseries = TimeSeries()
                    if not power_flow:
                        self._timeseries.timeindex = \
                            self._etrago_specs.dispatch.index
                        self._timeseries.generation = \
                            self._etrago_specs.dispatch
                        self._timeseries.load = \
                            self._timeseries.import_load_timeseries(self)
                    else:
                        self._timeseries.timeindex = pd.date_range(
                            power_flow[0], power_flow[1], freq='H')
                        self._timeseries.generation = \
                            self._etrago_specs.dispatch.loc[
                                self._timeseries.timeindex]
                        self._timeseries.load = \
                            self._timeseries.import_load_timeseries(self).loc[
                                self._timeseries.timeindex]
                else:
                    logger.error("Etrago specifications must contain dispatch "
                                 "timeseries. Please provide them.")
            elif not self._timeseries:
                self._mode = 'time-range'
                self._timeseries = TimeSeries()
                if not power_flow:
                    self._timeseries.generation = \
                        self._timeseries.import_feedin_timeseries(self)
                    self._timeseries.load = \
                        self._timeseries.import_load_timeseries(self)
                    self._timeseries.timeindex = \
                        self._timeseries.generation.index
                else:
                    self._timeseries.timeindex = pd.date_range(
                        power_flow[0], power_flow[1], freq='H')
                    self._timeseries.generation = \
                        self._timeseries.import_feedin_timeseries(self).loc[
                            self._timeseries.timeindex]
                    self._timeseries.load = \
                        self._timeseries.import_load_timeseries(self).loc[
                            self._timeseries.timeindex]

    @property
    def curtailment(self):
        """
        Return technology specific curtailment factors
        """
        return self._curtailment

    def __repr__(self):
        return 'Scenario ' + self._name


class Parameters:
    """
    Contains model parameters for power flow analysis and grid expansion.

    Attributes
    ----------
    _pfac_mv_gen : :obj:`float`
        Power factor for medium voltage generators
    _pfac_mv_load : :obj:`float`
        Power factor for medium voltage loads
    _pfac_lv_gen : :obj:`float`
        Power factor for low voltage generators
    _pfac_lv_load : :obj:`float`
        Power factor for low voltage loads
    _hv_mv_trafo_offset : :obj:`float`
        Offset at substation
    _hv_mv_trafo_control_deviation : :obj:`float`
        Voltage control deviation at substation
    _load_factor_hv_mv_transformer : :obj:`float`
        Allowed load of transformers at substation, retrieved from config
        files depending on analyzed case (feed-in or load).
    _load_factor_mv_lv_transformer : :obj:`float`
        Allowed load of transformers at distribution substation, retrieved from
        config files depending on analyzed case (feed-in or load).
    _load_factor_mv_line : :obj:`float`
        Allowed load of MV line, retrieved from config files depending on
        analyzed case (feed-in or load).
    _load_factor_lv_line : :obj:`float`
        Allowed load of LV line, retrieved from config files depending on
        analyzed case (feed-in or load).
    _mv_max_v_deviation : :obj:`float`
        Allowed voltage deviation in MV grid, retrieved from config files
        depending on analyzed case (feed-in or load).
    _lv_max_v_deviation : :obj:`float`
        Allowed voltage deviation in LV grid, retrieved from config files
        depending on analyzed case (feed-in or load).

    """

    def __init__(self, scenario_class, **kwargs):
        self._scenario = scenario_class
        self._pfac_mv_gen = kwargs.get('pfac_mv_gen', None)
        self._pfac_mv_load = kwargs.get('pfac_mv_load', None)
        self._pfac_lv_gen = kwargs.get('pfac_lv_gen', None)
        self._pfac_lv_load = kwargs.get('pfac_lv_load', None)
        self._hv_mv_transformer_offset = None
        self._hv_mv_transformer_control_deviation = None
        self._load_factor_hv_mv_transformer = None
        self._load_factor_mv_lv_transformer = None
        self._load_factor_mv_line = None
        self._load_factor_lv_line = None
        self._mv_max_v_deviation = None
        self._lv_max_v_deviation = None

    @property
    def scenario(self):
        return self._scenario

    @property
    def pfac_mv_gen(self):
        if not self._pfac_mv_gen:
            self._pfac_mv_gen = float(
                self.scenario.network.config['scenario']['pfac_mv_gen'])
        return self._pfac_mv_gen
    
    @property
    def pfac_mv_load(self):
        if not self._pfac_mv_load:
            self._pfac_mv_load = float(
                self.scenario.network.config['scenario']['pfac_mv_load'])
        return self._pfac_mv_load
    
    @property
    def pfac_lv_gen(self):
        if not self._pfac_lv_gen:
            self._pfac_lv_gen = float(
                self.scenario.network.config['scenario']['pfac_lv_gen'])
        return self._pfac_lv_gen
    
    @property
    def pfac_lv_load(self):
        if not self._pfac_lv_load:
            self._pfac_lv_load = float(
                self.scenario.network.config['scenario']['pfac_lv_load'])
        return self._pfac_lv_load

    @property
    def hv_mv_transformer_offset(self):
        if not self._hv_mv_transformer_offset:
            self._hv_mv_transformer_offset = float(
                self.scenario.network.config['grid_expansion'][
                    'hv_mv_trafo_offset'])
        return self._hv_mv_transformer_offset

    @property
    def hv_mv_transformer_control_deviation(self):
        if not self._hv_mv_transformer_control_deviation:
            self._hv_mv_transformer_control_deviation = float(
                self.scenario.network.config['grid_expansion'][
                    'hv_mv_trafo_control_deviation'])
        return self._hv_mv_transformer_control_deviation

    @property
    # ToDo: for now only feed-in case is considered
    def load_factor_hv_mv_transformer(self):
        if not self._load_factor_hv_mv_transformer:
            self._load_factor_hv_mv_transformer = float(
                self.scenario.network.config['grid_expansion'][
                    'load_factor_hv_mv_transformer'])
        return self._load_factor_hv_mv_transformer

    @property
    # ToDo: for now only feed-in case is considered
    def load_factor_mv_lv_transformer(self):
        if not self._load_factor_mv_lv_transformer:
            self._load_factor_mv_lv_transformer = float(
                self.scenario.network.config['grid_expansion'][
                    'load_factor_mv_lv_transformer'])
        return self._load_factor_mv_lv_transformer

    @property
    # ToDo: for now only feed-in case is considered
    def load_factor_mv_line(self):
        if not self._load_factor_mv_line:
            self._load_factor_mv_line = float(
                self.scenario.network.config['grid_expansion'][
                    'load_factor_mv_line'])
        return self._load_factor_mv_line

    @property
    # ToDo: for now only feed-in case is considered
    def load_factor_lv_line(self):
        if not self._load_factor_lv_line:
            self._load_factor_lv_line = float(
                self.scenario.network.config['grid_expansion'][
                    'load_factor_lv_line'])
        return self._load_factor_lv_line

    @property
    # ToDo: for now only voltage deviation for the combined calculation of MV
    # and LV is considered (load and feed-in case for seperate consideration
    # of MV and LV needs to be implemented)
    def mv_max_v_deviation(self):
        if not self._mv_max_v_deviation:
            self._mv_max_v_deviation = float(
                self.scenario.network.config['grid_expansion'][
                    'mv_lv_max_v_deviation'])
        return self._mv_max_v_deviation

    @property
    # ToDo: for now only voltage deviation for the combined calculation of MV
    # and LV is considered (load and feed-in case for seperate consideration
    # of MV and LV needs to be implemented)
    def lv_max_v_deviation(self):
        if not self._lv_max_v_deviation:
            self._lv_max_v_deviation = float(
                self.scenario.network.config['grid_expansion'][
                    'mv_lv_max_v_deviation'])
        return self._lv_max_v_deviation


class TimeSeries:
    """Defines an eDisGo time series

    Contains time series for loads (sector-specific) and generators
    (technology-specific), e.g. tech. solar.

    Attributes
    ----------
    _generation : :pandas:`pandas.DataFrame<dataframe>`
        Time series of active power of generators. Columns represent generator
        type:

         * 'solar'
         * 'wind'
         * 'coal'
         * ...

        In case of worst-case analysis generator type is distinguished so that
        the DataFrame contains only one column for all generators.

    _load : :pandas:`pandas.DataFrame<dataframe>`
        Time series of active power of (cumulative) loads. Columns represent
        load sectors:

         * 'residential'
         * 'retail'
         * 'industrial'
         * 'agricultural'

    See also
    --------
    edisgo.grid.components.Generator : Usage details of :meth:`_generation`
    edisgo.grid.components.Load : Usage details of :meth:`_load`

    """

    def __init__(self, **kwargs):
        self._generation = kwargs.get('generation', None)
        self._load = kwargs.get('load', None)
        self._timeindex = kwargs.get('timeindex', None)

    @property
    def generation(self):
        """
        Get generation timeseries (only active power)

        Returns
        -------
        dict or :pandas:`pandas.Series<series>`
            See class definition for details.
        """
        return self._generation

    @generation.setter
    def generation(self, generation_timeseries):
        self._generation = generation_timeseries
        
    @property
    def load(self):
        """
        Get load timeseries (only active power)

        Provides normalized load for each sector. Simply access sectoral load
        time series by :code:`Timeseries.load['residential']`.

        Returns
        -------
        dict or :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        return self._load

    @load.setter
    def load(self, load_timeseries):
        self._load = load_timeseries

    @property
    def timeindex(self):
        """
        Parameters
        ----------
        timerange : :pandas:`pandas.DatetimeIndex<datetimeindex>`
            Time range of power flow analysis

        Returns
        -------
        :pandas:`pandas.DatetimeIndex<datetimeindex>`
            Time range of power flow analysis

        """
        return self._timeindex

    @timeindex.setter
    def timeindex(self, time_range):
        self._timeindex = time_range

    def worst_case_generation_ts(self):
        """
        Define worst case generation time series.

        Parameters
        ----------
        network : :class:~.grid.network.Network`

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Normalized active power (1 kW) in column 'p' with random time index

        """
        # set random timeindex
        self.timeindex = pd.date_range('1/1/1970', periods=1, freq='H')
        return pd.DataFrame({'p': 1}, index=self.timeindex)

    def worst_case_load_ts(self, scenario):
        """
        Define worst case load time series

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Normalized active power (1 kW) for each load sector with
            random time index

        """
        # set random timeindex
        self.timeindex = pd.date_range('1/1/1970', periods=1, freq='H')
        #ToDo: remove hard coded sectors?
        return pd.DataFrame({
            'residential': 1 * float(
                scenario.config.data['data']['peakload_consumption_ratio'][
                    'residential']),
            'retail': 1 * float(
                scenario.config.data['data']['peakload_consumption_ratio'][
                    'retail']),
            'industrial': 1 * float(
                scenario.config.data['data']['peakload_consumption_ratio'][
                    'industrial']),
            'agricultural': 1 * float(
                scenario.config.data['data']['peakload_consumption_ratio'][
                    'agricultural'])},
            index=self.timeindex)

    def import_feedin_timeseries(self, scenario):
        """
        Import feedin timeseries from oedb
        """
        #ToDo: add docstring
        generation_df = import_feedin_timeseries(scenario)
        #ToDo: remove hard coded value
        if generation_df is not None:
            generation_df = pd.concat(
                [generation_df, pd.DataFrame({'other': 0.9},
                                             index=generation_df.index)],
                axis=1)
        #ToDo remove hard coded index?
        generation_df.index = pd.date_range('1/1/2011', periods=8760, freq='H')
        return generation_df

    def import_load_timeseries(self, scenario, data_source='demandlib'):
        """
        Import load timeseries

        Parameters
        ----------
        data_source : str
            Specify type of data source. Available data sources are

             * 'oedb': retrieves load time series cumulated across sectors
             * 'demandlib': determine a load time series with the use of the
                demandlib. This calculated standard load profiles for 4
                different sectors.

        """
        # ToDo: add docstring
        #ToDo: find better place for input data_source (in config?)
        return import_load_timeseries(scenario, data_source)


class ETraGoSpecs:
    """Defines an eTraGo object used in project open_eGo

    Contains specifications which are to be fulfilled at transition point
    (superiorHV-MV substation) for a specific scenario.

    Attributes
    ----------
    _battery_capacity: :obj:`float`
        Capacity of virtual battery at Transition Point in kWh.
    _battery_active_power : :pandas:`pandas.Series<series>`
        Time series of active power the (virtual) battery (at Transition Point)
        is charged (negative) or discharged (positive) with in kW.
    _conv_dispatch : :pandas:`pandas.DataFrame<dataframe>`
        Time series of active power for each (aggregated) type of flexible
        generators normalized with corresponding capacity.
        Columns represent generator type:
         * 'gas'
         * 'coal'
         * 'biomass'
         * ...
    _ren_dispatch : :pandas:`pandas.DataFrame<dataframe>`
        Time series of active power of wind and solar aggregates,
        normalized with corresponding capacity.
        Columns represent ren_id (see _renewables):
         * '0'
         * '1'
         * ...
    _curtailment : :pandas:`pandas.DataFrame<dataframe>`
        Time series of curtailed power for wind and solar aggregates,
        normalized with corresponding capacity.
        Columns represent ren_id (see _renewables):
         * '0'
         * '1'
         * ...
    _renewables : :pandas:`pandas.DataFrame<dataframe>`
        Dataframe containing `ren_id` specifying type (wind or solar) and
        weather cell ID.
        Columns are:
         * 'name' (type, e.g. 'solar')
         * 'w_id' (weather cell ID)
         * 'ren_id'

    """

    def __init__(self, **kwargs):
        self._battery_capacity = kwargs.get('battery_capacity', None)
        self._battery_active_power = kwargs.get('battery_active_power', None)
        self._conv_dispatch = kwargs.get('conv_dispatch', None)
        self._ren_dispatch = kwargs.get('ren_dispatch', None)
        self._curtailment = kwargs.get('curtailment', None)
        self._renewables = kwargs.get('renewables', None)

    @property
    def battery_capacity(self):
        return self._battery_capacity

    @property
    def battery_active_power(self):
        return self._battery_active_power

    @property
    def dispatch(self):
        """
        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Time series of active power for each type of generator normalized
            with corresponding capacity.
            Columns represent generator type:
             * 'solar'
             * 'wind'
             * 'coal'
             * ...

        """
        # this is temporary until feed-in of renewables is distinguished by
        # weather ID
        wind = self._renewables[self._renewables['name'] == 'wind']['ren_id']
        solar = self._renewables[self._renewables['name'] == 'solar']['ren_id']
        ren_dispatch_aggr_wind = self._ren_dispatch[wind].mean(axis=1).rename(
            'wind')
        ren_dispatch_aggr_solar = self._ren_dispatch[solar].mean(
            axis=1).rename('solar')
        ren_dispatch_aggr = pd.DataFrame(ren_dispatch_aggr_wind).join(
            ren_dispatch_aggr_solar)
        return ren_dispatch_aggr.join(self._conv_dispatch)


class Results:
    """
    Power flow analysis results management

    Includes raw power flow analysis results, history of measures to increase
    the grid's hosting capacity and information about changes of equipment.

    Attributes
    ----------
    measures: list
        A stack that details the history of measures to increase grid's hosting
        capacity. The last item refers to the latest measure. The key `original`
        refers to the state of the grid topology as it was initially imported.

    """

    # TODO: maybe add setter to alter list of measures

    def __init__(self):
        self._measures = ['original']
        self._pfa_p = None
        self._pfa_q = None
        self._pfa_v_mag_pu = None
        self._i_res = None
        self._equipment_changes = pd.DataFrame()
        self._grid_expansion_costs = None
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

            Provide this if you want to set values. For retrieval of data do not
            pass an argument

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
        ToDo: add unit

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

        equipment: detailing what was changed (line, station, battery,
        curtailment). For ease of referencing we take the component itself.
        For lines we take the line-dict, for stations the transformers, for
        batteries the battery-object itself and for curtailment
        either a dict providing the details of curtailment or a curtailment
        object if this makes more sense (has to be defined).

        change: string {'added' | 'removed'}
            says if something was added or removed

        iteration_step: int
            Used for the update of the pypsa network to only consider changes
            since the last power flow analysis.

        quantity: int
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

            The DataFrame has the following columns:

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

    def v_res(self, nodes=None, level=None):
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
        if level is None:
            level = ['mv', 'lv']

        if nodes is None:
            labels = list(self.pfa_v_mag_pu[level])
        else:
            labels = [repr(_) for _ in nodes]

        not_included = [_ for _ in labels
                        if _ not in list(self.pfa_v_mag_pu[level].columns)]

        labels_included = [_ for _ in labels if _ not in not_included]

        if not_included:
            print("Voltage levels for {nodes} are not returned from PFA".format(
                nodes=not_included))


        return self.pfa_v_mag_pu[level][labels_included]

