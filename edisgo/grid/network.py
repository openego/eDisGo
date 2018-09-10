import os
import pandas as pd
import numpy as np
from math import sqrt
import logging
import datetime
from pyomo.environ import Constraint
import networkx as nx
import csv

import edisgo
from edisgo.tools import config, tools
from edisgo.tools import pypsa_io_lopf, pypsa_io
from edisgo.data.import_data import import_from_ding0, import_generators, \
    import_feedin_timeseries, import_load_timeseries
from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.flex_opt import storage_integration, storage_operation, \
    curtailment, storage_positioning
from edisgo.grid.components import Station, BranchTee, Generator, Load
from edisgo.grid.tools import get_gen_info, disconnect_storage
from edisgo.grid.grids import MVGrid
from edisgo.tools import plots

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

    def curtail(self, methodology, curtailment_timeseries, **kwargs):
        """
        Sets up curtailment time series.

        Curtailment time series are written into
        :class:`~.grid.network.TimeSeries`. See
        :class:`~.grid.network.CurtailmentControl` for more information on
        parameters and methodologies.

        """
        CurtailmentControl(edisgo=self, methodology=methodology,
                           curtailment_timeseries=curtailment_timeseries,
                           **kwargs)

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
            pypsa_io.process_pfa_results(
                self.network, self.network.pypsa, timesteps)
        else:
            raise ValueError("Power flow analysis did not converge.")

    def analyze_lopf(self, mode=None, timesteps=None,
                     etrago_max_storage_size=None):
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

        # Translate eDisGo grid topology representation to PyPSA format
        logging.debug('Translate eDisGo grid topology representation to '
                      'PyPSA format.')
        self.network.pypsa_lopf = pypsa_io_lopf.to_pypsa(
            self.network, mode, timesteps)
        logging.debug('Translating eDisGo grid topology representation to '
                      'PyPSA format finished.')

        # add total storage capacity constraint
        def extra_functionality(network, snapshots):
            model = network.model
            # total installed capacity
            model.storages_p_nom = Constraint(
                rule=lambda model: sum(
                    model.generator_p_nom[s]
                    for s in self.network.pypsa_lopf.generators[
                        self.network.pypsa_lopf.generators.type ==
                        'Storage'].index) <= etrago_max_storage_size)

        # run power flow analysis
        self.network.pypsa_lopf.lopf(
            snapshots=timesteps, solver_name='cbc', keep_files=False,
            extra_functionality=extra_functionality,
            solver_options={'tee': True})

        # self.network.pypsa.model.write(
        #     io_options={'symbolic_solver_labels': True})

        print('objective: {}'.format(self.network.pypsa_lopf.objective))

        # relevant outputs
        # plot MV grid
        plots.storage_size(self.network.mv_grid, self.network.pypsa_lopf,
                           filename='storage_results_{}.pdf'.format(
                               self.network.id))

        storages = self.network.mv_grid.graph.nodes_by_attribute('storage')
        storages_repr = [repr(_) for _ in storages]
        print('Installed storage capacity: {} MW'.format(
            self.network.pypsa_lopf.generators.loc[
                storages_repr, 'p_nom_opt'].sum()))

        # export storage results (pypsa and path to storage)
        pypsa_storages_df = self.network.pypsa_lopf.generators.loc[
            storages_repr, :].sort_values(by=['p_nom_opt'], ascending=False)

        storage_repr = []
        storage_path = []
        for s in storages:
            storage_repr.append(repr(s))
            storage_path.append(nx.shortest_path(self.network.mv_grid.graph,
                                    self.network.mv_grid.station, s))
        graph_storages_df = pd.DataFrame({'path': storage_path},
                                         index=storage_repr)
        pypsa_storages_df.join(graph_storages_df).to_csv(
            'storage_results_{}.csv'.format(self.network.id))

        # take largest 8 storages and remove the rest
        keep_storages = pypsa_storages_df.iloc[:8, :].index
        remove_storages = pypsa_storages_df.iloc[8:, :].index
        # write time series to kept storages
        for s in keep_storages:
            keep_storage_obj = [_ for _ in storages if repr(_)==s][0]
            ts = self.network.pypsa_lopf.generators_t.p.loc[:, s]
            keep_storage_obj.timeseries = pd.DataFrame({'p': ts * 1000,
                                                        'q': [0] * len(ts)},
                                                        index=ts.index)
        # delete small storages
        for s in remove_storages:
            disconnect_storage(self.network,
                               [_ for _ in storages if repr(_)==s][0])

    def reinforce(self, **kwargs):
        """
        Reinforces the grid and calculates grid expansion costs.

        See :meth:`~.flex_opt.reinforce_grid` for more information.

        """
        results = reinforce_grid(
            self, max_while_iterations=kwargs.get(
                'max_while_iterations', 10),
            copy_graph=kwargs.get('copy_graph', False),
            timesteps_pfa=kwargs.get('timesteps_pfa', None),
            combined_analysis=kwargs.get('combined_analysis', False))

        # add measure to Results object
        if not kwargs.get('copy_graph', False):
            self.network.results.measures = 'grid_expansion'

        return results

    def integrate_storage(self, timeseries, position, **kwargs):
        """
        Integrates storage into grid.

        See :class:`~.grid.network.StorageControl` for more information.

        """
        StorageControl(edisgo=self, timeseries=timeseries,
                       position=position, **kwargs)


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
        self.results = Results(self)

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

    curtailment_timeseries : :pandas:`pandas.Series<series>` or :pandas:`pandas.DataFrame<dataframe>`, optional
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

    """

    def __init__(self, edisgo, methodology, curtailment_timeseries, **kwargs):

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

        # get all fluctuating generators and their attributes (weather ID,
        # type, etc.)
        generators = get_gen_info(edisgo.network, 'mvlv', fluctuating=True)

        # do analyze to get all voltages at generators and feed-in dataframe
        edisgo.analyze()

        # get feed-in time series of all generators
        feedin = edisgo.network.pypsa.generators_t.p * 1000
        # drop dispatchable generators and slack generator
        drop_labels = [_ for _ in feedin.columns
                       if 'GeneratorFluctuating' not in _] \
                      + ['Generator_slack']
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
        edisgo.network.results.measures = 'curtailment'

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
            bad_time_steps = [_ for _ in curtailment_timeseries.index
                              if curtailment_timeseries[_] >
                              feedin_selected_sum[_]]
            if bad_time_steps:
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
        network : :class:`~.grid.network.Network`
        feedin : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with feed-in time series in kW. Columns of the dataframe
            are :class:`~.grid.components.GeneratorFluctuating`, index is
            time index.

        """
        curtailment = network.timeseries.curtailment
        gen_repr = [repr(_) for _ in curtailment.columns]
        feedin_repr = feedin.loc[:, gen_repr]
        curtailment_repr = curtailment
        curtailment_repr.columns = gen_repr
        if not ((feedin_repr - curtailment_repr) > -1e-3).all().all():
            message = 'Curtailment exceeds feed-in.'
            logging.error(message)
            raise TypeError(message)


class StorageControl:
    """
    Integrates storages into the grid.

    Parameters
    ----------
    edisgo : :class:`~.grid.network.EDisGo`
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
    position : None or :obj:`str` or :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee`  or :class:`~.grid.components.Generator` or :class:`~.grid.components.Load` or :obj:`dict`
        To position the storage a positioning strategy can be used or a
        node in the grid can be directly specified. Possible options are:

        * 'hvmv_substation_busbar'
          Places a storage unit directly at the HV/MV station's bus bar.
        * :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee` or :class:`~.grid.components.Generator` or :class:`~.grid.components.Load`
          Specifies a node the storage should be connected to. In the case
          this parameter is of type :class:`~.grid.components.LVStation` an
          additional parameter, `voltage_level`, has to be provided to define
          which side of the LV station the storage is connected to.
        * 'distribute_storages_mv'
          Places one storage in each MV feeder if it reduces grid expansion
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

        See :class:`~.grid.components.Storage` for more information on storage
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
        of type :class:`~.grid.components.LVStation`. In that case
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
        self.edisgo.network.results.measures = 'storage_integration'

    def _integrate_storage(self, timeseries, position, params, voltage_level,
                           reactive_power_timeseries, **kwargs):
        """
        Integrate storage units in the grid.

        Parameters
        ----------
        timeseries : :obj:`str` or :pandas:`pandas.Series<series>`
            Parameter used to obtain time series of active power the storage
            storage is charged (negative) or discharged (positive) with. Can
            either be a given time series or an operation strategy. See class
            definition for more information
        position : :obj:`str` or :class:`~.grid.components.Station` or :class:`~.grid.components.BranchTee` or :class:`~.grid.components.Generator` or :class:`~.grid.components.Load`
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
        return tools.assign_load_feedin_case(self.network)


class Results:
    """
    Power flow analysis results management

    Includes raw power flow analysis results, history of measures to increase
    the grid's hosting capacity and information about changes of equipment.

    Attributes
    ----------
    network : :class:`~.grid.network.Network`
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
        List with the history of measures to increase grid's hosting capacity.

        Parameters
        ----------
        measure : :obj:`str`
            Measure to increase grid's hosting capacity. Possible options are
            'grid_expansion', 'storage_integration', 'curtailment'.

        Returns
        -------
        measures : :obj:`list`
            A stack that details the history of measures to increase grid's
            hosting capacity. The last item refers to the latest measure. The
            key `original` refers to the state of the grid topology as it was
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
        of the DataFrame are the edges as well as stations of the grid
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
        of the DataFrame are the edges as well as stations of the grid
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
        columns of the DataFrame are the nodes as well as stations of the grid
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
        :class:`~.grid.components.Line`, :class:`~.grid.components.Station`,
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
            calculation of grid expansion costs to keep track of how many
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
        Holds active and reactive grid losses in kW and kvar, respectively.

        Parameters
        ----------
        pypsa_grid_losses : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive grid losses in columns 'p'
            and 'q' and in kW and kvar, respectively. Index is a
            :pandas:`pandas.DatetimeIndex<datetimeindex>`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive grid losses in columns 'p'
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
        Holds active and reactive power exchanged with the HV grid.

        The exchanges are essentially the slack results. As the slack is placed
        at the secondary side of the HV/MV station, this gives the energy
        transferred to and taken from the HV grid at the secondary side of the
        HV/MV station.

        Parameters
        ----------
        hv_mv_exchanges : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive power exchanged with the HV
            grid in columns 'p' and 'q' and in kW and kvar, respectively. Index
            is a :pandas:`pandas.DatetimeIndex<datetimeindex>`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>
            Dataframe holding active and reactive power exchanged with the HV
            grid in columns 'p' and 'q' and in kW and kvar, respectively. Index
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
            :class:`edisgo.grid.components.GeneratorFluctuating`.

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

            Dataframe containing all storages installed in the MV grid and
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
        for grid in grids:
            for storage in grid.graph.nodes_by_attribute('storage'):
                storage_results['storage_id'].append(repr(storage))
                storage_results['nominal_power'].append(storage.nominal_power)
                storage_results['voltage_level'].append(
                    'mv' if isinstance(grid, MVGrid) else 'lv')

        return pd.DataFrame(storage_results).set_index('storage_id')

    @property
    def storages_costs_reduction(self):
        """
        Contains costs reduction due to storage integration.

        Parameters
        ----------
        costs_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe containing grid expansion costs in kEUR before and after
            storage integration in columns 'grid_expansion_costs_initial' and
            'grid_expansion_costs_with_storages', respectively. Index of the
            dataframe is the MV grid id.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`

            Dataframe containing grid expansion costs in kEUR before and after
            storage integration in columns 'grid_expansion_costs_initial' and
            'grid_expansion_costs_with_storages', respectively. Index of the
            dataframe is the MV grid id.

        """
        return self._storages_costs_reduction

    @storages_costs_reduction.setter
    def storages_costs_reduction(self, costs_df):
        self._storages_costs_reduction = costs_df

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

            S = max(\sqrt{p_0^2 + q_0^2}, \sqrt{p_1^2 + q_1^2})

        Parameters
        ----------
        components : :obj:`list`
            List with all components (of type :class:`~.grid.components.Line`
            or :class:`~.grid.components.Transformer`) to get apparent power
            for. If not provided defaults to return apparent power of all lines
            and transformers in the grid.

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
                print("Apparent power for {lines} are not returned from "
                      "PFA".format(lines=labels_not_included))
        else:
            labels_included = self.pfa_p.columns

        s_res = ((self.pfa_p[labels_included] ** 2 + self.pfa_q[
            labels_included] ** 2)).applymap(sqrt)

        return s_res

    def v_res(self, nodes=None, level=None):
        """
        Get resulting voltage level at node.

        Parameters
        ----------
        nodes : :class:`~.grid.components.Load`, \
            :class:`~.grid.components.Generator`, etc. or :obj:`list`
            Grid topology component or list of grid topology components.
            If not provided defaults to column names available in grid level
            `level`.
        level : str
            Either 'mv' or 'lv' or None (default). Depending on which grid
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
            labels = list(map(repr, nodes.copy()))
            not_included = [_ for _ in labels
                            if _ not in list(self.pfa_v_mag_pu[level].columns)]
            labels_included = [_ for _ in labels if _ not in not_included]

            if not_included:
                logging.info("Voltage levels for {nodes} are not returned "
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

                # grid losses
                self.grid_losses.to_csv(
                    os.path.join(target_dir, 'grid_losses.csv'))

                # grid exchanges
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

                # grid expansion costs
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

                # grid expansion costs
                storages.to_csv(os.path.join(target_dir, 'storages.csv'))

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
                          "'storage_integration_results'."
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
