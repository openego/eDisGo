from os import path
import pandas as pd
import numpy as np
from math import sqrt
import logging
import datetime

import edisgo
from edisgo.tools import config, pypsa_io
from edisgo.data.import_data import import_from_ding0, import_generators, \
    import_feedin_timeseries, import_load_timeseries
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.flex_opt.storage_integration import integrate_storage


logger = logging.getLogger('edisgo')

class EDisGoAPI:
    """
    Handles user inputs.
    #ToDo: Doch noch mode paramater einführen, damit der User nur einmal den
    worst case Fall definieren muss und nicht für jede Zeitreihe?

    Parameters
    ----------
    mv_grid_id : :obj:`str`
        MV grid ID used in import of ding0 grid.
        ToDo: explain where MV grid IDs come from
    ding0_grid : file: :obj:`str` or :class:`ding0.core.NetworkDing0`
        If a str is provided it is assumed it points to a pickle with Ding0
        grid data. This file will be read. If an object of the type
        :class:`ding0.core.NetworkDing0` data will be used directly from this
        object.
        This will probably be removed when ding0 grids are in oedb.
    config_path : dict
        Dictionary specifying which config files to use. If not specified
        config files in $HOME/.edisgo/config/ are used. Keys of the dictionary
        are the config files, values contain the corresponding path to the
        file.
        ToDo: list allowed keys
    scenario_name : None or :obj:`str`
        Can be used to describe your scenario but is not used for anything
        else. Default: None.
    timeseries_generation_fluc : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`
        Parameter used to obtain time series for active power feed-in of
        fluctuating renewables wind and solar.
        Possible options are:
         * 'worst-case'
            feed-in for the two worst-case scenarios feed-in case and load case
            are generated
         * 'worst-case-feedin'
            feed-in for the worst-case scenario feed-in case is generated
         * 'worst-case-load'
            feed-in for the worst-case scenario load case is generated
         * 'oedb'
            time series for the time steps specified in `timeindex` are
            obtained from the OpenEnergy DataBase
         * :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with time series, normalized with corresponding capacity.
            Time series can either be aggregated by technology type or by type
            and weather cell ID. In the first case columns of the DataFrame are
            'solar' and 'wind'; in the second case columns need to be a
            :pandas:`pandas.MultiIndex<multiindex>` with the first level
            containing the type and the second level the weather cell ID.
        ToDo: explain how to obtain weather cell ID, add link to explanation
        of worst-case analyses
    timeseries_generation_flex : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`
        Parameter used to obtain time series for active power feed-in of
        flexible generators such as coal and biomass generators.
        Possible options are:
         * 'worst-case'
            feed-in for the two worst-case scenarios feed-in case and load case
            are generated
         * 'worst-case-feedin'
            feed-in for the worst-case scenario feed-in case is generated
         * 'worst-case-load'
            feed-in for the worst-case scenario load case is generated
         * :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with time series for active power of each (aggregated)
            type of flexible generator normalized with corresponding capacity.
            Columns represent generator type:
             * 'gas'
             * 'coal'
             * 'biomass'
             * 'other'
             * ...
            Use 'other' if you don't want to explicitly provide every possible
            type.
    timeseries_load : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`
        Parameter used to obtain time series of active power of (cumulative)
        loads.
        Possible options are:
         * 'worst-case'
            load time series for the two worst-case scenarios feed-in case and
            load case are generated
         * 'worst-case-feedin'
            load time series for the worst-case scenario feed-in case is
            generated
         * 'worst-case-load'
            load time series for the worst-case scenario load case is generated
         * 'oedb'
            time series for the time steps specified in `timeindex` are
            obtained from the OpenEnergy DataBase
         * 'demandlib'
            time series for the time steps specified in `timeindex` are
            generated using the oemof demandlib
         * :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with load time series of each (cumulative) type of load
            normalized with corresponding annual energy demand.
            Columns represent load type:
             * 'residential'
             * 'retail'
             * 'industrial'
             * 'agricultural'
    timeseries_battery : None or :obj:`str` or :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`
        Parameter used to obtain time series of active power the battery
        storage(s) is/are charged (negative) or discharged (positive) with.
        Possible options are:
         * 'worst-case'
            time series for the two worst-case scenarios feed-in case and
            load case are generated
         * 'worst-case-feedin'
            time series for the worst-case scenario feed-in case is
            generated
         * 'worst-case-load'
            time series for the worst-case scenario load case is generated
         * :pandas:`pandas.Series<series>`/:pandas:`pandas.DataFrame<dataframe>`
            Time series of active power the battery storage is charged
            (negative) or discharged (positive) with, normalized with
            corresponding capacity. In case of more than one storage provide
            a DataFrame where each column represents one storage.
    battery_parameters : None or dict or list
        In case of one battery storage a dictionary needs to be provided. In
        case of more than one battery storage a list of dictionaries needs to
        be provided. Default: None.
        #ToDo: Add description of the dictionary.
        #ToDo: Besser DataFrame mit gleichen columns wie timeseries_battery?
    timeseries_curtailment : None or :pandas:`pandas.DataFrame<dataframe>`
        DataFrame with time series of curtailed power of wind and solar
        aggregates in kW.
        Time series can either be aggregated by technology type or by type
        and weather cell ID. In the first case columns of the DataFrame are
        'solar' and 'wind'; in the second case columns need to be a
        :pandas:`pandas.MultiIndex<multiindex>` with the first level
        containing the type and the second level the weather cell ID. See
        `timeseries_fluc` parameter for further explanation of the weather
        cell ID. Default: None.
    curtailment_methodology : None or :obj:`str`
        Specifies the methodology used to allocate the curtailment time
        series to single generators. Needs to be set when curtailment time
        series are given. Default: None.
        #ToDo: Add possible options and links to them once we defined them.
    import_generator_scenario : None or :obj:`str`
        If provided defines which scenario of future generator park to use.
        Possible options are 'nep2035' and 'ego100'.
        #ToDo: Add link to explanation of scenarios.
    timeindex : None or :pandas:`pandas.DatetimeIndex<datetimeindex>`
        Can be used to define a time range for which to obtain load time series
        and feed-in time series of fluctuating renewables or to define time
        ranges of the given time series that will be used in the analysis.
    """

    def __init__(self, grid_id):
        pass


class Network:
    """Defines the eDisGo Network

    Used as container for all data related to a single
    :class:`~.grid.grids.MVGrid`.
    Provides the top-level API for invocation of data import, analysis of
    hosting capacity, grid reinforcement and flexibility measures.

    Examples
    --------
    Assuming you have the Ding0 `ding0_data.pkl` in CWD

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
    id : :obj:`str`
        Name of network
    equipment_data : :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
        Electrical equipment such as lines and transformers
    config : :class:`~.grid.network.Config`
        Data from config files
    metadata : :obj:`dict`
        Metadata of Network such as ?
    data_sources : :obj:`dict` of :obj:`str`
        Data Sources of grid, generators etc.
        Keys: 'grid', 'generators', ?
    pypsa : :pypsa:`pypsa.Network<network>`
        PyPSA representation of grid topology
    dingo_import_data :
        Temporary data from ding0 import which are needed for OEP generator
        update
    results : :class:`~.grid.network.Results`
        Object with results from power flow analyses
    timeseries : :class:`~.grid.network.TimeSeries`
        Object containing time series
    mv_grid : :class:`~.grid.grids.MVGrid`
    scenario : :obj:`str`
        Defines which scenario of future generator park to use.
    """

    def __init__(self, **kwargs):
        if 'pypsa' not in kwargs.keys():
            self._id = kwargs.get('id', None)
            self._metadata = kwargs.get('metadata', None)
            self._data_sources = kwargs.get('data_sources', {})

            self._mv_grid = kwargs.get('mv_grid', None)
            self._scenario = kwargs.get('scenario', None)
            self._pypsa = None
            self.results = Results()
        else:
            self._pypsa = kwargs.get('pypsa', None)

        self._config = Config()
        self._equipment_data = self._load_equipment_data()

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
                    path.join(package_path, equipment_dir,
                              equipment_parameters),
                    comment='#', index_col='name',
                    delimiter=',', decimal='.')

        return data

    @classmethod
    def import_from_ding0(cls, file, **kwargs):
        """Import grid data from DINGO file

        For details see
        :func:`edisgo.data.import_data.import_from_ding0`
        """
        #ToDo: Should also work when only an MV grid ID is provided.
        # create the network instance
        network = cls(**kwargs)

        # call the importer
        import_from_ding0(file=file,
                          network=network)

        return network

    def import_generators(self, types=None, data_source='oedb'):
        """Import generators

        For details see
        :func:`edisgo.data.import_data.import_generators`
        """
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
        transformers assuming an ideal tap changer. Hence, potential
        overloading of the transformers is not studied here.

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
        """Returns config object data"""
        return self._config.data

    @property
    def metadata(self):
        """Returns meta data"""
        return self._metadata

    @property
    def scenario(self):
        """Returns scenario name"""
        return self._scenario

    @property
    def equipment_data(self):
        """Returns equipment data

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
    def timeseries(self):
        """:class:`~.grid.network.TimeSeries`
        """
        return self._timeseries

    @timeseries.setter
    def timeseries(self, timeseries):
        self._timeseries = timeseries

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
        config.load_config('config_grid.cfg')
        config.load_config('config_grid_expansion.cfg')
        config.load_config('config_timeseries.cfg')

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

    @property
    def data(self):
        return self._data


class TimeSeriesControl:
    """
    Sets up TimeSeries Object.

    #ToDo: Default for kwargs in docstring? None for kwargs in docstring?
    #ToDo: How to avoid redundant docstrings?

    Parameters
    ----------
    mode : :obj:`str`, optional
        Mode must be set in case of worst-case analyses and can either be
        'worst-case' (both feed-in and load case), 'worst-case-feedin' (only
        feed-in case) or 'worst-case-load' (only load case). All other
        parameters except of `config-data` will be ignored. Default: None.
    mv_grid_id : :obj:`str`, optional
        MV grid ID as used in oedb. Default: None.
    scenario_name : :obj:`str`
        Defines which scenario of future generator park to use. Possible
        options are 'nep2035' and 'ego100'. Default: None.
    timeseries_generation_fluc : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`, optional
        Parameter used to obtain time series for active power feed-in of
        fluctuating renewables wind and solar.
        Possible options are:
         * 'oedb'
            time series are obtained from the OpenEnergy DataBase
         * :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with time series, normalized with corresponding capacity.
            Time series can either be aggregated by technology type or by type
            and weather cell ID. In the first case columns of the DataFrame are
            'solar' and 'wind'; in the second case columns need to be a
            :pandas:`pandas.MultiIndex<multiindex>` with the first level
            containing the type and the second level the weather cell ID.
        Default: None.
    timeseries_generation_flex : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with time series for active power of each (aggregated)
        type of flexible generator normalized with corresponding capacity.
        Columns represent generator type:
         * 'gas'
         * 'coal'
         * 'biomass'
         * 'other'
         * ...
        Use 'other' if you don't want to explicitly provide every possible
        type.
        Default: None.
    timeseries_load : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`, optional
        Parameter used to obtain time series of active power of (cumulative)
        loads.
        Possible options are:
         * 'oedb'
            time series are obtained from the OpenEnergy DataBase
         * 'demandlib'
            time series are generated using the oemof demandlib
         * :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with load time series of each (cumulative) type of load
            normalized with corresponding annual energy demand.
            Columns represent load type:
             * 'residential'
             * 'retail'
             * 'industrial'
             * 'agricultural'
         Default: None.
    config_data : dict, optional
        Dictionary containing config data from config files. See
        :class:`~.grid.network.Config` data attribute for more information.
        Default: None.

    """
    def __init__(self, **kwargs):

        self.timeseries = TimeSeries()
        mode = kwargs.get('mode', None)
        config_data = kwargs.get('config_data', None)

        if mode:
            if mode == 'worst-case':
                modes = ['feedin_case', 'load_case']
            elif mode == 'worst-case-feedin' or mode == 'worst-case-load':
                modes = ['{}_case'.format(mode.split('-')[-1])]
            else:
                raise ValueError('{} is not a valid mode.'.format(mode))

            # set random timeindex
            self.timeseries.timeindex = pd.date_range(
                '1/1/1970', periods=len(modes), freq='H')
            self._worst_case_generation(config_data['worst_case_scale_factor'],
                                        modes)
            self._worst_case_load(config_data['worst_case_scale_factor'],
                                  config_data['peakload_consumption_ratio'],
                                  modes)

        else:
            # feed-in time series of fluctuating renewables
            ts = kwargs.get('timeseries_generation_fluc', None)
            if isinstance(ts, pd.DataFrame):
                self.timeseries.generation_fluctuating = ts
            elif isinstance(ts, str) and ts == 'oedb':
                self._import_feedin_timeseries(
                    config_data, kwargs.get('mv_grid_id', None),
                    kwargs.get('scenario_name', None))
            else:
                raise ValueError('Your input for "timeseries_fluc" is not '
                                 'valid.'.format(mode))
            # feed-in time series for flexible generators
                ts = kwargs.get('timeseries_generation_flex', None)
            if isinstance(ts, pd.DataFrame):
                self.timeseries.generation_flexible = ts
            else:
                raise ValueError('Your input for "timeseries_flex" is not '
                                 'valid.'.format(mode))
            # load time series
            ts = kwargs.get('timeseries_load', None)
            if isinstance(ts, pd.DataFrame):
                self.timeseries.load = ts_flex
            elif isinstance(ts, str) and (ts == 'oedb' or ts == 'demandlib'):
                self.timeseries.load = import_load_timeseries(
                    config_data, ts)
            else:
                raise ValueError('Your input for "timeseries_flex" is not '
                                 'valid.'.format(mode))

            #ToDo: check if time series have the same index (or timeindex in
            #TimeSeries object)

    def _worst_case_generation(self, worst_case_scale_factors, modes):
        """
        Define worst case generation time series for fluctuating and flexible
        generators.

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

        self.timeseries.generation_flexible = pd.DataFrame(
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

    def _import_feedin_timeseries(self, config_data, mv_grid_id,
                                  scenario_name):
        """
        Import feed-in time series for wind and solar for the year 2011
        from oedb

        Parameters
        ----------
        config_data : dict
            Dictionary containing config data from config files. See
            :class:`~.grid.network.Config` data attribute for more information.
        mv_grid_id : :obj:`str`
            MV grid ID as used in oedb.
        scenario_name : None or :obj:`str`
            Defines which scenario of future generator park to use.

        """

        #ToDo: remove this function if it just calls another function
        self.timeseries.generation_fluctuating = import_feedin_timeseries(
            config_data, mv_grid_id, scenario_name)
        #ToDo: remove hard coded value
        # if generation_df is not None:
        #     generation_df = pd.concat(
        #         [generation_df, pd.DataFrame({'other': 0.9},
        #                                      index=generation_df.index)],
        #         axis=1)


class TimeSeries:
    """Defines an eDisGo time series

    Contains time series for loads (sector-specific) and generators
    (technology-specific), e.g. tech. solar.

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
    generation_flexible : :pandas:`pandas.DataFrame<dataframe>`, optional
        DataFrame with time series for active power of each (aggregated)
        type of flexible generator normalized with corresponding capacity.
        Columns represent generator type:
         * 'gas'
         * 'coal'
         * 'biomass'
         * 'other'
         * ...
        Use 'other' if you don't want to explicitly provide every possible
        type.
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
    timeindex : :pandas:`pandas.DatetimeIndex<datetimeindex>`, optional
        Can be used to define a time range for which to obtain the provided
        time series and run power flow analysis. Default: None.

    See also
    --------
    edisgo.grid.components.Generator : Usage details of :meth:`_generation`
    edisgo.grid.components.Load : Usage details of :meth:`_load`

    """

    def __init__(self, **kwargs):
        self._generation_flexible = kwargs.get('generation_flexible', None)
        self._generation_fluctuating = kwargs.get('generation_fluctuating',
                                                  None)
        self._load = kwargs.get('load', None)
        self._timeindex = kwargs.get('timeindex', None)

    @property
    def generation_flexible(self):
        """
        Get generation time series of flexible generators (only active power)

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.
        """
        try:
            return self._generation_flexible.loc[[self.timeindex], :]
        except:
            return self._generation_flexible.loc[self.timeindex, :]

    @generation_flexible.setter
    def generation_flexible(self, generation_flex_timeseries):
        self._generation_flexible = generation_flex_timeseries

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
    def load(self):
        """
        Get load timeseries (only active power)

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

    @timeindex.setter
    def timeindex(self, time_range):
        self._timeindex = time_range


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

            Index of the DataFrame is the representation of the respective
            object, columns are the following:

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

