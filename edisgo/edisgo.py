import logging
import os
import pickle

import pandas as pd

from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.io import pypsa_io
from edisgo.io.ding0_import import import_ding0_grid
from edisgo.io.generators_import import oedb as import_generators_oedb
from edisgo.network import timeseries
from edisgo.network.results import Results
from edisgo.network.topology import Topology
from edisgo.opf.results.opf_result_class import OPFResults
from edisgo.opf.run_mp_opf import run_mp_opf
from edisgo.tools import plots, tools
from edisgo.tools.config import Config
from edisgo.tools.geo import find_nearest_bus

if "READTHEDOCS" not in os.environ:
    from shapely.geometry import Point

logger = logging.getLogger("edisgo")


class EDisGo:
    """
    Provides the top-level API for invocation of data import, power flow
    analysis, network reinforcement, flexibility measures, etc..

    Parameters
    ----------
    ding0_grid : :obj:`str`
        Path to directory containing csv files of network to be loaded.
    generator_scenario : None or :obj:`str`, optional
        If None, the generator park of the imported grid is kept as is.
        Otherwise defines which scenario of future generator park to use
        and invokes grid integration of these generators. Possible options are
        'nep2035' and 'ego100'. These are scenarios from the research project
        `open_eGo <https://openegoproject.wordpress.com/>`_ (see
        `final report <https://www.uni-flensburg.de/fileadmin/content/\
        abteilungen/industrial/dokumente/downloads/veroeffentlichungen/\
        forschungsergebnisse/20190426endbericht-openego-fkz0325881-final.pdf>`_
        for more information on the scenarios).
        See :attr:`~.EDisGo.import_generators` for further information on how
        generators are integrated and what further options there are.
        Default: None.
    timeindex : None or :pandas:`pandas.DatetimeIndex<DatetimeIndex>`, optional
        Defines the time steps feed-in and demand time series of all generators, loads
        and storage units need to be set.
        The time index is for example used as default for time steps considered in
        the power flow analysis and when checking the integrity of the network.
        Providing a time index is only optional in case a worst case analysis is set
        up using :func:`~set_time_series_worst_case_analysis`.
        In all other cases a time index needs to be set manually.
    config_path : None or :obj:`str` or :obj:`dict`
        Path to the config directory. Options are:

        * None

          If `config_path` is None, configs are loaded from the edisgo
          default config directory ($HOME$/.edisgo). If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.

        * :obj:`str`

          If `config_path` is a string, configs will be loaded from the
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
          config file. In contrast to the other two options, the directories
          and config files must exist and are not automatically created.

        Default: None.

    Attributes
    ----------
    topology : :class:`~.network.topology.Topology`
        The topology is a container object holding the topology of the grids.
    timeseries : :class:`~.network.timeseries.TimeSeries`
        Container for component time series.
    results : :class:`~.network.results.Results`
        This is a container holding all calculation results from power flow
        analyses, curtailment, storage integration, etc.

    """

    def __init__(self, **kwargs):

        # load configuration
        self._config = Config(config_path=kwargs.get("config_path", None))

        # instantiate topology object and load grid data
        self.topology = Topology(config=self.config)
        self.import_ding0_grid(path=kwargs.get("ding0_grid", None))

        # set up results and time series container
        self.results = Results(self)
        self.opf_results = OPFResults()
        self.timeseries = timeseries.TimeSeries(
            timeindex=kwargs.get("timeindex", pd.DatetimeIndex([]))
        )

        # import new generators
        if kwargs.get("generator_scenario", None) is not None:
            self.import_generators(
                generator_scenario=kwargs.pop("generator_scenario"), **kwargs
            )

    @property
    def config(self):
        """
        eDisGo configuration data.

        Returns
        -------
        :class:`~.tools.config.Config`
            Config object with configuration data from config files.

        """
        return self._config

    @config.setter
    def config(self, config_path):
        self._config = Config(config_path=config_path)

    def import_ding0_grid(self, path):
        """
        Import ding0 topology data from csv files in the format as
        `Ding0 <https://github.com/openego/ding0>`_ provides it.

        Parameters
        -----------
        path : str
            Path to directory containing csv files of network to be loaded.

        """
        if path is not None:
            import_ding0_grid(path, self)

    def set_timeindex(self, timeindex):
        """
        Sets :py:attr:`~.network.timeseries.TimeSeries.timeindex` all time-dependent
        attributes are indexed by.

        The time index is for example used as default for time steps considered in
        the power flow analysis and when checking the integrity of the network.

        Parameters
        -----------
        timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
            Time index to set.

        """
        self.timeseries.timeindex = timeindex

    def set_time_series_manual(
        self,
        generators_p=None,
        loads_p=None,
        storage_units_p=None,
        generators_q=None,
        loads_q=None,
        storage_units_q=None,
    ):
        """
        Sets given component time series.

        If time series for a component were already set before, they are overwritten.

        Parameters
        -----------
        generators_p : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series in MW of generators. Index of the data frame is
            a datetime index. Columns contain generators names of generators to set
            time series for. Default: None.
        loads_p : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series in MW of loads. Index of the data frame is
            a datetime index. Columns contain load names of loads to set
            time series for. Default: None.
        storage_units_p : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series in MW of storage units. Index of the data frame is
            a datetime index. Columns contain storage unit names of storage units to set
            time series for. Default: None.
        generators_q : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series in MVA of generators. Index of the data frame is
            a datetime index. Columns contain generators names of generators to set
            time series for. Default: None.
        loads_q : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series in MVA of loads. Index of the data frame is
            a datetime index. Columns contain load names of loads to set
            time series for. Default: None.
        storage_units_q : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series in MVA of storage units. Index of the data frame
            is a datetime index. Columns contain storage unit names of storage units to
            set time series for. Default: None.

        """
        # check if time index is already set, otherwise raise warning
        if self.timeseries.timeindex.empty:
            logger.warning(
                "When setting time series manually a time index is not automatically "
                "set but needs to be set by the user. You can set the time index "
                "upon initialisation of the EDisGo object by providing the input "
                "parameter 'timeindex' or using the function EDisGo.set_timeindex()."
            )
        self.timeseries.set_active_power_manual(
            self,
            ts_generators=generators_p,
            ts_loads=loads_p,
            ts_storage_units=storage_units_p,
        )
        self.timeseries.set_reactive_power_manual(
            self,
            ts_generators=generators_q,
            ts_loads=loads_q,
            ts_storage_units=storage_units_q,
        )

    def set_time_series_worst_case_analysis(self, cases=None):
        """
        Sets demand and feed-in of all loads, generators and storage units for the
        specified worst cases.

        See :func:`~.network.timeseries.TimeSeries.set_worst_case` for more information.

        Parameters
        -----------
        cases : str or list(str)
            List with worst-cases to generate time series for. Can be
            'feed-in_case', 'load_case' or both. Defaults to None in which case both
            'feed-in_case' and 'load_case' are set up.

        """
        if cases is None:
            cases = ["load_case", "feed-in_case"]
        if isinstance(cases, str):
            cases = [cases]

        self.timeseries.set_worst_case(self, cases)

    def set_time_series_active_power_predefined(
        self,
        fluctuating_generators_ts=None,
        fluctuating_generators_names=None,
        dispatchable_generators_ts=None,
        dispatchable_generators_names=None,
        conventional_loads_ts=None,
        conventional_loads_names=None,
        charging_points_ts=None,
        charging_points_names=None,
    ):
        """
        Uses predefined feed-in or demand profiles.

        Predefined profiles comprise i.e. standard electric conventional load profiles
        for different sectors generated using the oemof
        `demandlib <https://github.com/oemof/demandlib/>`_ or feed-in time series of
        fluctuating solar and wind generators provided on the OpenEnergy DataBase for
        the weather year 2011.

        This function can also be used to provide your own profiles per technology or
        load sector.

        Parameters
        -----------
        fluctuating_generators_ts : str or :pandas:`pandas.DataFrame<DataFrame>`
            Defines which technology-specific (or technology and weather cell specific)
            time series to use to set active power time series of fluctuating
            generators. See parameter `ts_generators` in
            :func:`~.network.timeseries.TimeSeries.predefined_fluctuating_generators_by_technology`
            for more information. If None, no time series of fluctuating generators
            are set. Default: None.
        fluctuating_generators_names : list(str)
            Defines for which fluctuating generators to apply technology-specific time
            series. See parameter `generator_names` in
            :func:`~.network.timeseries.TimeSeries.predefined_dispatchable_generators_by_technology`
            for more information. Default: None.
        dispatchable_generators_ts : :pandas:`pandas.DataFrame<DataFrame>`
            Defines which technology-specific time series to use to set active power
            time series of dispatchable generators.
            See parameter `ts_generators` in
            :func:`~.network.timeseries.TimeSeries.predefined_dispatchable_generators_by_technology`
            for more information. If None, no time series of dispatchable generators
            are set. Default: None.
        dispatchable_generators_names : list(str)
            Defines for which dispatchable generators to apply technology-specific time
            series. See parameter `generator_names` in
            :func:`~.network.timeseries.TimeSeries.predefined_dispatchable_generators_by_technology`
            for more information. Default: None.
        conventional_loads_ts : :pandas:`pandas.DataFrame<DataFrame>`
            Defines which sector-specific time series to use to set active power
            time series of conventional loads.
            See parameter `ts_loads` in
            :func:`~.network.timeseries.TimeSeries.predefined_conventional_loads_by_sector`
            for more information. If None, no time series of conventional loads
            are set. Default: None.
        conventional_loads_names : list(str)
            Defines for which conventional loads to apply technology-specific time
            series. See parameter `load_names` in
            :func:`~.network.timeseries.TimeSeries.predefined_conventional_loads_by_sector`
            for more information. Default: None.
        charging_points_ts : :pandas:`pandas.DataFrame<DataFrame>`
            Defines which use-case-specific time series to use to set active power
            time series of charging points.
            See parameter `ts_loads` in
            :func:`~.network.timeseries.TimeSeries.predefined_charging_points_by_use_case`
            for more information. If None, no time series of charging points
            are set. Default: None.
        charging_points_names : list(str)
            Defines for which charging points to apply use-case-specific time
            series. See parameter `load_names` in
            :func:`~.network.timeseries.TimeSeries.predefined_charging_points_by_use_case`
            for more information. Default: None.

        """
        if self.timeseries.timeindex.empty:
            logger.warning(
                "When setting time series using predefined profiles a time index is "
                "not automatically set but needs to be set by the user. In some cases "
                "not setting a time index prior to calling this function may lead "
                "to errors. You can set the time index upon initialisation of the "
                "EDisGo object by providing the input parameter 'timeindex' or using "
                "the function EDisGo.set_timeindex()."
            )
        if fluctuating_generators_ts is not None:
            self.timeseries.predefined_fluctuating_generators_by_technology(
                self, fluctuating_generators_ts, fluctuating_generators_names
            )
        if dispatchable_generators_ts is not None:
            self.timeseries.predefined_dispatchable_generators_by_technology(
                self, dispatchable_generators_ts, dispatchable_generators_names
            )
        if conventional_loads_ts is not None:
            self.timeseries.predefined_conventional_loads_by_sector(
                self, conventional_loads_ts, conventional_loads_names
            )
        if charging_points_ts is not None:
            self.timeseries.predefined_charging_points_by_use_case(
                self, charging_points_ts, charging_points_names
            )

    def set_time_series_reactive_power_control(
        self,
        control="fixed_cosphi",
        generators_parametrisation="default",
        loads_parametrisation="default",
        storage_units_parametrisation="default",
    ):
        """
        Parameters
        -----------
        control : str
            Type of reactive power control to apply. Currently the only option is
            'fixed_coshpi'. See :func:`~.network.timeseries.TimeSeries.fixed_cosphi`
            for further information.
        generators_parametrisation : str or :pandas:`pandas.DataFrame<dataframe>`
            See parameter `generators_parametrisation` in
            :func:`~.network.timeseries.TimeSeries.fixed_cosphi` for further
            information. Here, per default, the option 'default' is used.
        loads_parametrisation : str or :pandas:`pandas.DataFrame<dataframe>`
            See parameter `loads_parametrisation` in
            :func:`~.network.timeseries.TimeSeries.fixed_cosphi` for further
            information. Here, per default, the option 'default' is used.
        storage_units_parametrisation : str or :pandas:`pandas.DataFrame<dataframe>`
            See parameter `storage_units_parametrisation` in
            :func:`~.network.timeseries.TimeSeries.fixed_cosphi` for further
            information. Here, per default, the option 'default' is used.

        """
        if control == "fixed_cosphi":
            self.timeseries.fixed_cosphi(
                self,
                generators_parametrisation=generators_parametrisation,
                loads_parametrisation=loads_parametrisation,
                storage_units_parametrisation=storage_units_parametrisation,
            )
        else:
            raise ValueError("'control' must be 'fixed_cosphi'.")

    def to_pypsa(self, **kwargs):
        """
        Convert to PyPSA :pypsa:`pypsa.Network<network>` representation.

        Parameters
        ----------
        kwargs :
            See :func:`~.io.pypsa_io.to_pypsa` for further information.

        Other Parameters
        -----------------
        mode : str
            Determines network levels that are translated to
            `PyPSA network representation
            <https://www.pypsa.org/doc/components.html#network>`_. Specify

            * None to export MV and LV network levels. None is the default.
            * 'mv' to export MV network level only. This includes cumulative load
              and generation from underlying LV network aggregated at respective LV
              station's primary side.
            * 'mvlv' to export MV network level only. This includes cumulative load
              and generation from underlying LV network aggregated at respective LV
              station's secondary side.
            * 'lv' to export specified LV network only.

        Returns
        -------
        :pypsa:`pypsa.Network<network>`
            PyPSA network representation.

        """
        timesteps = kwargs.pop("timesteps", None)
        mode = kwargs.get("mode", None)

        if timesteps is None:
            timesteps = self.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]
        # export grid
        # ToDo: Move to pypsa_io.to_pypsa
        if not mode:
            return pypsa_io.to_pypsa(self, timesteps, **kwargs)
        elif "mv" in mode:
            return pypsa_io.to_pypsa(self.topology.mv_grid, timesteps, **kwargs)
        elif mode == "lv":
            lv_grid_name = kwargs.get("lv_grid_name", None)
            if not lv_grid_name:
                raise ValueError(
                    "For exporting lv grids, name of lv_grid has to be provided."
                )
            return pypsa_io.to_pypsa(
                self.topology._grids[lv_grid_name],
                mode=mode,
                timesteps=timesteps,
            )
        else:
            raise ValueError("The entered mode is not a valid option.")

    def to_graph(self):
        """
        Returns networkx graph representation of the grid.

        Returns
        -------
        :networkx:`networkx.Graph<network.Graph>`
            Graph representation of the grid as networkx Ordered Graph,
            where lines are represented by edges in the graph, and buses and
            transformers are represented by nodes.

        """

        return self.topology.to_graph()

    def import_generators(self, generator_scenario=None, **kwargs):
        """
        Gets generator park for specified scenario and integrates them into
        the grid.

        Currently, the only supported data source is scenario data generated
        in the research project
        `open_eGo <https://openegoproject.wordpress.com/>`_. You can choose
        between two scenarios: 'nep2035' and 'ego100'. You can get more
        information on the scenarios in the
        `final report <https://www.uni-flensburg.de/fileadmin/content/\
        abteilungen/industrial/dokumente/downloads/veroeffentlichungen/\
        forschungsergebnisse/20190426endbericht-openego-fkz0325881-final\
        .pdf>`_.

        The generator data is retrieved from the
        `open energy platform <https://openenergy-platform.org/>`_
        from tables for
        `conventional power plants <https://openenergy-platform.org/dataedit/\
        view/supply/ego_dp_conv_powerplant>`_ and
        `renewable power plants <https://openenergy-platform.org/dataedit/\
        view/supply/ego_dp_res_powerplant>`_.

        When the generator data is retrieved, the following steps are
        conducted:

            * Step 1: Update capacity of existing generators if `
              update_existing` is True, which it is by default.
            * Step 2: Remove decommissioned generators if
              `remove_decommissioned` is True, which it is by default.
            * Step 3: Integrate new MV generators.
            * Step 4: Integrate new LV generators.

        For more information on how generators are integrated, see
        :attr:`~.network.topology.Topology.connect_to_mv` and
        :attr:`~.network.topology.Topology.connect_to_lv`.

        After the generator park is changed there may be grid issues due to the
        additional in-feed. These are not solved automatically. If you want to
        have a stable grid without grid issues you can invoke the automatic
        grid expansion through the function :attr:`~.EDisGo.reinforce`.

        Parameters
        ----------
        generator_scenario : str
            Scenario for which to retrieve generator data. Possible options
            are 'nep2035' and 'ego100'.

        Other Parameters
        ----------------
        kwargs :
            See :func:`edisgo.io.generators_import.oedb`.

        """
        import_generators_oedb(
            edisgo_object=self, generator_scenario=generator_scenario, **kwargs
        )

    def analyze(self, mode=None, timesteps=None, raise_not_converged=True, **kwargs):
        """
        Conducts a static, non-linear power flow analysis.

        Conducts a static, non-linear power flow analysis using
        `PyPSA <https://www.pypsa.org/doc/power_flow.html#full-non-linear-power-flow>`_
        and writes results (active, reactive and apparent power as well as
        current on lines and voltages at buses) to :class:`~.network.results.Results`
        (e.g. :attr:`~.network.results.Results.v_res` for voltages).

        Parameters
        ----------
        mode : str or None
            Allows to toggle between power flow analysis for the whole network or just
            the MV or one LV grid. Possible options are:

            * None (default)

                Power flow analysis is conducted for the whole network including MV and
                LV level.

            * 'mv'

                Power flow analysis is conducted for the MV level only. LV loads and
                generators are aggregated at the respective MV/LV stations' primary
                side.

            * 'mvlv'

                Power flow analysis is conducted for the MV level only. In contrast to
                mode 'mv' LV loads and generators are in this case aggregated at the
                respective MV/LV stations' secondary side.

            * 'lv'

                Power flow analysis is conducted for one LV grid only. Name of the LV
                grid to conduct power flow analysis for needs to be provided through
                keyword argument 'lv_grid_name' as string.

        timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
            :pandas:`pandas.Timestamp<Timestamp>`
            Timesteps specifies for which time steps to conduct the power flow
            analysis. It defaults to None in which case all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex` are
            used.
        raise_not_converged : bool
            If True, an error is raised in case power flow analysis did not converge
            for all time steps. I
            Default: True.

        Returns
        --------
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
            Returns the time steps for which power flow analysis did not converge.

        Other Parameters
        -----------------
        Possible other parameters comprise all other parameters that can be set in
        :attr:`~.io.pypsa_io.to_pypsa`.

        """
        if timesteps is None:
            timesteps = self.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]

        pypsa_network = self.to_pypsa(mode=mode, timesteps=timesteps, **kwargs)

        # run power flow analysis
        pf_results = pypsa_network.pf(timesteps, use_seed=kwargs.get("use_seed", False))

        # get converged and not converged time steps
        timesteps_converged = pf_results["converged"][
            pf_results["converged"]["0"]
        ].index
        timesteps_not_converged = pf_results["converged"][
            ~pf_results["converged"]["0"]
        ].index

        if raise_not_converged and len(timesteps_not_converged) > 0:
            raise ValueError(
                "Power flow analysis did not converge for the "
                "following time steps: {}.".format(timesteps_not_converged)
            )

        # handle converged time steps
        pypsa_io.process_pfa_results(self, pypsa_network, timesteps_converged)

        return timesteps_not_converged

    def reinforce(self, **kwargs):
        """
        Reinforces the network and calculates network expansion costs.

        See :func:`edisgo.flex_opt.reinforce_grid.reinforce_grid` for more
        information.

        """
        results = reinforce_grid(
            self,
            max_while_iterations=kwargs.get("max_while_iterations", 10),
            copy_grid=kwargs.get("copy_grid", False),
            timesteps_pfa=kwargs.get("timesteps_pfa", None),
            combined_analysis=kwargs.get("combined_analysis", False),
            mode=kwargs.get("mode", None),
        )

        # add measure to Results object
        if not kwargs.get("copy_grid", False):
            self.results.measures = "grid_expansion"

        return results

    def perform_mp_opf(self, timesteps, storage_series=None, **kwargs):
        """
        Run optimal power flow with julia.

        Parameters
        -----------
        timesteps : list
            List of timesteps to perform OPF for.
        kwargs :
            See :func:`~.opf.run_mp_opf.run_mp_opf` for further
            information.

        Returns
        --------
        str
            Status of optimization.

        """
        if storage_series is None:
            storage_series = []
        status = run_mp_opf(self, timesteps, storage_series=storage_series, **kwargs)
        return status

    def add_component(
        self,
        comp_type,
        add_ts=True,
        ts_active_power=None,
        ts_reactive_power=None,
        **kwargs
    ):
        """
        Adds single component to network.

        Components can be lines or buses as well as generators, loads, or storage units.
        If add_ts is set to True, time series of elements are set as well. Currently,
        time series need to be provided.

        Parameters
        ----------
        comp_type : str
            Type of added component. Can be 'bus', 'line', 'load', 'generator', or
            'storage_unit'.
        add_ts : bool
            Indicator if time series for component are added as well. If True, active
            and reactive power time series need to be provided through parameters
            `ts_active_power` and `ts_reactive_power`. Default: True.
        ts_active_power : :pandas:`pandas.Series<series>`
            Active power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are active power per time step in MW.
        ts_reactive_power : :pandas:`pandas.Series<series>`
            Reactive power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are reactive power per time step in MVA.
        **kwargs: dict
            Attributes of added component. See respective functions for required
            entries.

            * 'bus' : :attr:`~.network.topology.Topology.add_bus`

            * 'line' : :attr:`~.network.topology.Topology.add_line`

            * 'load' : :attr:`~.network.topology.Topology.add_load`

            * 'generator' : :attr:`~.network.topology.Topology.add_generator`

            * 'storage_unit' : :attr:`~.network.topology.Topology.add_storage_unit`

        """
        # ToDo: Add option to add transformer.
        # Todo: change into add_components to allow adding of several components
        #    at a time, change topology.add_load etc. to add_loads, where
        #    lists of parameters can be inserted

        if comp_type == "bus":
            comp_name = self.topology.add_bus(**kwargs)

        elif comp_type == "line":
            comp_name = self.topology.add_line(**kwargs)

        elif comp_type == "generator":
            comp_name = self.topology.add_generator(**kwargs)
            if add_ts:
                self.set_time_series_manual(
                    generators_p=pd.DataFrame({comp_name: ts_active_power}),
                    generators_q=pd.DataFrame({comp_name: ts_reactive_power}),
                )

        elif comp_type == "storage_unit":
            comp_name = self.topology.add_storage_unit(**kwargs)
            if add_ts:
                self.set_time_series_manual(
                    storage_units_p=pd.DataFrame({comp_name: ts_active_power}),
                    storage_units_q=pd.DataFrame({comp_name: ts_reactive_power}),
                )

        elif comp_type == "load":
            comp_name = self.topology.add_load(**kwargs)
            if add_ts:
                self.set_time_series_manual(
                    loads_p=pd.DataFrame({comp_name: ts_active_power}),
                    loads_q=pd.DataFrame({comp_name: ts_reactive_power}),
                )

        else:
            raise ValueError(
                "Invalid input for parameter 'comp_type'. Must either be "
                "'line', 'bus', 'generator', 'load' or 'storage_unit'."
            )
        return comp_name

    def integrate_component(
        self,
        comp_type,
        geolocation,
        voltage_level=None,
        add_ts=True,
        ts_active_power=None,
        ts_reactive_power=None,
        **kwargs
    ):
        """
        Adds single component to topology based on geolocation.

        Currently components can be generators or charging points.

        Parameters
        ----------
        comp_type : str
            Type of added component. Can be 'generator' or 'charging_point'.
        geolocation : :shapely:`shapely.Point<Point>` or tuple
            Geolocation of the new component. In case of tuple, the geolocation
            must be given in the form (longitude, latitude).
        voltage_level : int, optional
            Specifies the voltage level the new component is integrated in.
            Possible options are 4 (MV busbar), 5 (MV grid), 6 (LV busbar) or
            7 (LV grid). If no voltage level is provided the voltage level
            is determined based on the nominal power `p_nom` (given as kwarg)
            as follows:

            * voltage level 4 (MV busbar): nominal power between 4.5 MW and
              17.5 MW
            * voltage level 5 (MV grid) : nominal power between 0.3 MW and
              4.5 MW
            * voltage level 6 (LV busbar): nominal power between 0.1 MW and
              0.3 MW
            * voltage level 7 (LV grid): nominal power below 0.1 MW

        add_ts : bool, optional
            Indicator if time series for component are added as well.
            Default: True.
        ts_active_power : :pandas:`pandas.Series<Series>`, optional
            Active power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are active power per time step in MW. Currently, if you want
            to add time series (if `add_ts` is True), you must provide a
            time series. It is not automatically retrieved.
        ts_reactive_power : :pandas:`pandas.Series<Series>`, optional
            Reactive power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are reactive power per time step in MVA. Currently, if you
            want to add time series (if `add_ts` is True), you must provide a
            time series. It is not automatically retrieved.

        Other Parameters
        ------------------
        kwargs :
            Attributes of added component.
            See :attr:`~.network.topology.Topology.add_generator` respectively
            :attr:`~.network.topology.Topology.add_charging_point` methods
            for more information on required and optional parameters of
            generators and charging points.

        """
        supported_voltage_levels = {4, 5, 6, 7}
        p_nom = kwargs.get("p_nom", None)
        if voltage_level not in supported_voltage_levels:
            if p_nom is None:
                raise ValueError(
                    "Neither appropriate voltage level nor nominal power "
                    "were supplied."
                )
            # Determine voltage level manually from nominal power
            if 4.5 < p_nom <= 17.5:
                voltage_level = 4
            elif 0.3 < p_nom <= 4.5:
                voltage_level = 5
            elif 0.1 < p_nom <= 0.3:
                voltage_level = 6
            elif 0 < p_nom <= 0.1:
                voltage_level = 7
            else:
                raise ValueError("Unsupported voltage level")

        # check if geolocation is given as shapely Point, otherwise transform
        # to shapely Point
        if not type(geolocation) is Point:
            geolocation = Point(geolocation)

        # Connect in MV
        if voltage_level in [4, 5]:
            kwargs["voltage_level"] = voltage_level
            kwargs["geom"] = geolocation
            comp_name = self.topology.connect_to_mv(self, kwargs, comp_type)

        # Connect in LV
        else:
            substations = self.topology.buses_df.loc[
                self.topology.transformers_df.bus1.unique()
            ]
            nearest_substation, _ = find_nearest_bus(geolocation, substations)
            kwargs["mvlv_subst_id"] = int(nearest_substation.split("_")[-2])
            kwargs["geom"] = geolocation
            kwargs["voltage_level"] = voltage_level
            comp_name = self.topology.connect_to_lv(self, kwargs, comp_type)

        if add_ts:
            if comp_type == "generator":
                self.set_time_series_manual(
                    generators_p=pd.DataFrame({comp_name: ts_active_power}),
                    generators_q=pd.DataFrame({comp_name: ts_reactive_power}),
                )
            else:
                self.set_time_series_manual(
                    loads_p=pd.DataFrame({comp_name: ts_active_power}),
                    loads_q=pd.DataFrame({comp_name: ts_reactive_power}),
                )

        return comp_name

    def remove_component(self, comp_type, comp_name, drop_ts=True):
        """
        Removes single component from network.

        Components can be lines or buses as well as generators, loads, or storage units.
        If drop_ts is set to True, time series of elements are deleted as well.

        Parameters
        ----------
        comp_type : str
            Type of removed component.  Can be 'bus', 'line', 'load', 'generator', or
            'storage_unit'.
        comp_name : str
            Name of component to be removed.
        drop_ts : bool
            Indicator if time series for component are removed as well. Defaults
            to True.

        """
        # Todo: change into remove_components, when add_component is changed into
        #    add_components, to allow removal of several components at a time

        if comp_type == "bus":
            self.topology.remove_bus(comp_name)

        elif comp_type == "line":
            self.topology.remove_line(comp_name)

        elif comp_type == "load":
            self.topology.remove_load(comp_name)
            if drop_ts:
                for ts in ["active_power", "reactive_power"]:
                    timeseries.drop_component_time_series(
                        obj=self.timeseries,
                        df_name="loads_{}".format(ts),
                        comp_names=comp_name,
                    )

        elif comp_type == "generator":
            self.topology.remove_generator(comp_name)
            if drop_ts:
                for ts in ["active_power", "reactive_power"]:
                    timeseries.drop_component_time_series(
                        obj=self.timeseries,
                        df_name="generators_{}".format(ts),
                        comp_names=comp_name,
                    )

        elif comp_type == "storage_unit":
            self.topology.remove_storage_unit(comp_name)
            if drop_ts:
                for ts in ["active_power", "reactive_power"]:
                    timeseries.drop_component_time_series(
                        obj=self.timeseries,
                        df_name="storage_units_{}".format(ts),
                        comp_names=comp_name,
                    )

        else:
            raise ValueError("Component type is not correct.")

    def aggregate_components(
        self,
        aggregate_generators_by_cols=None,
        aggregate_loads_by_cols=None,
    ):
        """
        Aggregates generators and loads at the same bus.

        By default all generators respectively loads at the same bus are aggregated.
        You can specify further columns to consider in the aggregation, such as the
        generator type or the load sector. Make sure to always include the bus in the
        list of columns to aggregate by, as otherwise the topology would change.

        Be aware that by aggregating components you loose some information
        e.g. on load sector or charging point use case.

        Parameters
        -----------
        aggregate_generators_by_cols : list(str) or None
            List of columns to aggregate generators at the same bus by. Valid
            columns are all columns in
            :attr:`~.network.topology.Topology.generators_df`. If an empty list is
            given, generators are not aggregated. Defaults to None, in
            which case all generators at the same bus are aggregated.
        aggregate_loads_by_cols : list(str)
            List of columns to aggregate loads at the same bus by. Valid
            columns are all columns in
            :attr:`~.network.topology.Topology.loads_df`. If an empty list is
            given, generators are not aggregated. Defaults to None, in
            which case all loads at the same bus are aggregated.

        """

        def _aggregate_time_series(attribute, groups, naming):
            return pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format("_".join(k))
                            if isinstance(k, tuple)
                            else naming.format(k): getattr(self.timeseries, attribute)
                            .loc[:, v]
                            .sum(axis=1)
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )

        if aggregate_generators_by_cols is None:
            aggregate_generators_by_cols = ["bus"]
        if aggregate_loads_by_cols is None:
            aggregate_loads_by_cols = ["bus"]

        # aggregate generators
        if (
            len(aggregate_generators_by_cols) > 0
            and not self.topology.generators_df.empty
        ):

            gens_groupby = self.topology.generators_df.groupby(
                aggregate_generators_by_cols
            )
            naming = "Generators_{}"

            # set up new generators_df
            gens_df_grouped = gens_groupby.sum().reset_index()
            gens_df_grouped["name"] = gens_df_grouped.apply(
                lambda _: naming.format("_".join(_.loc[aggregate_generators_by_cols])),
                axis=1,
            )
            gens_df_grouped["control"] = "PQ"
            if "weather_cell_id" in gens_df_grouped.columns:
                gens_df_grouped.drop(columns=["weather_cell_id"], inplace=True)
            self.topology.generators_df = gens_df_grouped.set_index("name")

            # set up new generator time series
            self.timeseries.generators_active_power = _aggregate_time_series(
                "generators_active_power", gens_groupby.groups, naming
            )
            self.timeseries.generators_reactive_power = _aggregate_time_series(
                "generators_reactive_power", gens_groupby.groups, naming
            )

        # aggregate loads
        if len(aggregate_loads_by_cols) > 0 and not self.topology.loads_df.empty:

            loads_groupby = self.topology.loads_df.groupby(aggregate_loads_by_cols)
            naming = "Loads_{}"

            # set up new loads_df
            loads_df_grouped = loads_groupby.sum().reset_index()
            loads_df_grouped["name"] = loads_df_grouped.apply(
                lambda _: naming.format("_".join(_.loc[aggregate_loads_by_cols])),
                axis=1,
            )
            self.topology.loads_df = loads_df_grouped.set_index("name")

            # set up new loads time series
            self.timeseries.loads_active_power = _aggregate_time_series(
                "loads_active_power", loads_groupby.groups, naming
            )
            self.timeseries.loads_reactive_power = _aggregate_time_series(
                "loads_reactive_power", loads_groupby.groups, naming
            )

    def plot_mv_grid_topology(self, technologies=False, **kwargs):
        """
        Plots plain MV network topology and optionally nodes by technology type
        (e.g. station or generator).

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        Parameters
        ----------
        technologies : bool
            If True plots stations, generators, etc. in the topology in
            different colors. If False does not plot any nodes. Default: False.

        """

        plots.mv_grid_topology(
            self,
            node_color="technology" if technologies is True else None,
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            title=kwargs.get("title", ""),
        )

    def plot_mv_voltages(self, **kwargs):
        """
        Plots voltages in MV network on network topology plot.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.v_res is None:
                logging.warning(
                    "Voltages from power flow "
                    "analysis must be available to plot them."
                )
                return
        except AttributeError:
            logging.warning(
                "Results must be available to plot voltages. "
                "Please analyze grid first."
            )
            return
        except ValueError:
            pass

        plots.mv_grid_topology(
            self,
            timestep=kwargs.get("timestep", None),
            node_color="voltage",
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            voltage=self.results.v_res,
            limits_cb_nodes=kwargs.get("limits_cb_nodes", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            title=kwargs.get("title", ""),
        )

    def plot_mv_line_loading(self, **kwargs):
        """
        Plots relative line loading (current from power flow analysis to
        allowed current) of MV lines.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.i_res is None:
                logging.warning(
                    "Currents `i_res` from power flow analysis "
                    "must be available to plot line loading."
                )
                return
        except AttributeError:
            logging.warning(
                "Results must be available to plot line loading. "
                "Please analyze grid first."
            )
            return

        plots.mv_grid_topology(
            self,
            timestep=kwargs.get("timestep", None),
            line_color="loading",
            node_color=kwargs.get("node_color", None),
            line_load=self.results.i_res,
            filename=kwargs.get("filename", None),
            arrows=kwargs.get("arrows", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            voltage=self.results.v_res,
            limits_cb_lines=kwargs.get("limits_cb_lines", None),
            limits_cb_nodes=kwargs.get("limits_cb_nodes", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            lines_cmap=kwargs.get("lines_cmap", "inferno_r"),
            title=kwargs.get("title", ""),
            scaling_factor_line_width=kwargs.get("scaling_factor_line_width", None),
            curtailment_df=kwargs.get("curtailment_df", None),
        )

    def plot_mv_grid_expansion_costs(self, **kwargs):
        """
        Plots grid expansion costs per MV line.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.grid_expansion_costs is None:
                logging.warning(
                    "Grid expansion cost results needed to plot "
                    "them. Please do grid reinforcement."
                )
                return
        except AttributeError:
            logging.warning(
                "Results of MV topology needed to  plot topology "
                "expansion costs. Please reinforce first."
            )
            return

        plots.mv_grid_topology(
            self,
            line_color="expansion_costs",
            grid_expansion_costs=self.results.grid_expansion_costs,
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            limits_cb_lines=kwargs.get("limits_cb_lines", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            lines_cmap=kwargs.get("lines_cmap", "inferno_r"),
            title=kwargs.get("title", ""),
            scaling_factor_line_width=kwargs.get("scaling_factor_line_width", None),
        )

    def plot_mv_storage_integration(self, **kwargs):
        """
        Plots storage position in MV topology of integrated storage units.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        plots.mv_grid_topology(self, node_color="storage_integration", **kwargs)

    def plot_mv_grid(self, **kwargs):
        """
        General plotting function giving all options of function
        :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        plots.mv_grid_topology(self, **kwargs)

    def histogram_voltage(self, timestep=None, title=True, **kwargs):
        """
        Plots histogram of voltages.

        For more information on the histogram plot and possible configurations
        see :func:`edisgo.tools.plots.histogram`.

        Parameters
        ----------
        timestep : :pandas:`pandas.Timestamp<Timestamp>` or \
            list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
            Specifies time steps histogram is plotted for. If timestep is None
            all time steps voltages are calculated for are used. Default: None.
        title : :obj:`str` or :obj:`bool`, optional
            Title for plot. If True title is auto generated. If False plot has
            no title. If :obj:`str`, the provided title is used. Default: True.

        """
        try:
            data = self.results.v_res
            if data is None:
                logger.warning(
                    "Results for voltages are required for "
                    "voltage histogramm. Please analyze first."
                )
                return
        except AttributeError:
            logger.warning(
                "Results are required for voltage histogramm. Please analyze first."
            )
            return

        if timestep is None:
            timestep = data.index
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timestep, "__len__"):
            timestep = [timestep]

        if title is True:
            if len(timestep) == 1:
                title = "Voltage histogram for time step {}".format(timestep[0])
            else:
                title = "Voltage histogram \nfor time steps {} to {}".format(
                    timestep[0], timestep[-1]
                )
        elif title is False:
            title = None
        plots.histogram(data=data, title=title, timeindex=timestep, **kwargs)

    def histogram_relative_line_load(
        self, timestep=None, title=True, voltage_level="mv_lv", **kwargs
    ):
        """
        Plots histogram of relative line loads.

        For more information on how the relative line load is calculated see
        :func:`edisgo.tools.tools.get_line_loading_from_network`.
        For more information on the histogram plot and possible configurations
        see :func:`edisgo.tools.plots.histogram`.

        Parameters
        ----------
        timestep : :pandas:`pandas.Timestamp<Timestamp>` or \
            list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
            Specifies time step(s) histogram is plotted for. If `timestep` is
            None all time steps currents are calculated for are used.
            Default: None.
        title : :obj:`str` or :obj:`bool`, optional
            Title for plot. If True title is auto generated. If False plot has
            no title. If :obj:`str`, the provided title is used. Default: True.
        voltage_level : :obj:`str`
            Specifies which voltage level to plot voltage histogram for.
            Possible options are 'mv', 'lv' and 'mv_lv'. 'mv_lv' is also the
            fallback option in case of wrong input. Default: 'mv_lv'

        """
        try:
            if self.results.i_res is None:
                logger.warning(
                    "Currents `i_res` from power flow analysis "
                    "must be available to plot histogram line "
                    "loading."
                )
                return
        except AttributeError:
            logger.warning(
                "Results must be available to plot histogram line "
                "loading. Please analyze grid first."
            )
            return

        if voltage_level == "mv":
            lines = self.topology.mv_grid.lines_df
        elif voltage_level == "lv":
            lines = self.topology.lines_df[
                ~self.topology.lines_df.index.isin(self.topology.mv_grid.lines_df.index)
            ]
        else:
            lines = self.topology.lines_df

        rel_line_loading = tools.calculate_relative_line_load(
            self, lines.index, timestep
        )

        if timestep is None:
            timestep = rel_line_loading.index
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timestep, "__len__"):
            timestep = [timestep]

        if title is True:
            if len(timestep) == 1:
                title = "Relative line load histogram for time step {}".format(
                    timestep[0]
                )
            else:
                title = (
                    "Relative line load histogram \nfor time steps "
                    "{} to {}".format(timestep[0], timestep[-1])
                )
        elif title is False:
            title = None
        plots.histogram(data=rel_line_loading, title=title, **kwargs)

    def save(
        self,
        directory,
        save_results=True,
        save_topology=True,
        save_timeseries=True,
        **kwargs
    ):
        """
        Saves EDisGo object to csv.

        It can be chosen if results, topology and timeseries should be saved.
        For each one, a separate directory is created.

        Parameters
        ----------
        directory : str
            Main directory to save EDisGo object to.
        save_results : bool, optional
            Indicates whether to save :class:`~.network.results.Results`
            object. Per default it is saved. See
            :attr:`~.network.results.Results.to_csv` for more information.
        save_topology : bool, optional
            Indicates whether to save :class:`~.network.topology.Topology`.
            Per default it is saved. See
            :attr:`~.network.topology.Topology.to_csv` for more information.
        save_timeseries : bool, optional
            Indicates whether to save :class:`~.network.timeseries.Timeseries`.
            Per default it is saved. See
            :attr:`~.network.timeseries.Timeseries.to_csv` for more
            information.

        Other Parameters
        ------------------
        reduce_memory : bool, optional
            If True, size of dataframes containing time series in
            :class:`~.network.results.Results` and
            :class:`~.network.timeseries.TimeSeries`
            is reduced. See :attr:`~.network.results.Results.reduce_memory`
            and :attr:`~.network.timeseries.TimeSeries.reduce_memory` for more
            information. Type to convert to can be specified by providing
            `to_type` as keyword argument. Further parameters of reduce_memory
            functions cannot be passed here. Call these functions directly to
            make use of further options. Default: False.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        """
        os.makedirs(directory, exist_ok=True)
        if save_results:
            self.results.to_csv(
                os.path.join(directory, "results"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
                parameters=kwargs.get("parameters", None),
            )
        if save_topology:
            self.topology.to_csv(os.path.join(directory, "topology"))
        if save_timeseries:
            self.timeseries.to_csv(
                os.path.join(directory, "timeseries"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
            )

    def save_edisgo_to_pickle(self, path="", filename=None):
        abs_path = os.path.abspath(path)
        if filename is None:
            filename = "edisgo_object_{ext}.pkl".format(ext=self.topology.mv_grid.id)
        pickle.dump(self, open(os.path.join(abs_path, filename), "wb"))

    def reduce_memory(self, **kwargs):
        """
        Reduces size of dataframes containing time series to save memory.

        Per default, float data is stored as float64. As this precision is
        barely needed, this function can be used to convert time series data
        to a data subtype with less memory usage, such as float32.

        Other Parameters
        -----------------
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".
        results_attr_to_reduce : list(str), optional
            See `attr_to_reduce` parameter in
            :attr:`~.network.results.Results.reduce_memory` for more
            information.
        timeseries_attr_to_reduce : list(str), optional
            See `attr_to_reduce` parameter in
            :attr:`~.network.timeseries.TimeSeries.reduce_memory` for more
            information.

        """
        # time series
        self.timeseries.reduce_memory(
            to_type=kwargs.get("to_type", "float32"),
            attr_to_reduce=kwargs.get("timeseries_attr_to_reduce", None),
        )
        # results
        self.results.reduce_memory(
            to_type=kwargs.get("to_type", "float32"),
            attr_to_reduce=kwargs.get("results_attr_to_reduce", None),
        )


def import_edisgo_from_pickle(filename, path=""):
    abs_path = os.path.abspath(path)
    return pickle.load(open(os.path.join(abs_path, filename), "rb"))


def import_edisgo_from_files(
    directory="",
    import_topology=True,
    import_timeseries=False,
    import_results=False,
    **kwargs
):
    edisgo_obj = EDisGo(import_timeseries=False)
    if import_topology:
        topology_dir = kwargs.get(
            "topology_directory", os.path.join(directory, "topology")
        )
        if os.path.exists(topology_dir):
            edisgo_obj.topology.from_csv(topology_dir, edisgo_obj)
        else:
            logging.warning("No topology directory found. Topology not imported.")
    if import_timeseries:
        if os.path.exists(os.path.join(directory, "timeseries")):
            edisgo_obj.timeseries.from_csv(os.path.join(directory, "timeseries"))
        else:
            logging.warning("No timeseries directory found. Timeseries not imported.")
    if import_results:
        parameters = kwargs.get("parameters", None)
        if os.path.exists(os.path.join(directory, "results")):
            edisgo_obj.results.from_csv(os.path.join(directory, "results"), parameters)
        else:
            logging.warning("No results directory found. Results not imported.")
    if kwargs.get("import_residual_load", False):
        if os.path.exists(os.path.join(directory, "time_series_sums.csv")):
            residual_load = (
                pd.read_csv(os.path.join(directory, "time_series_sums.csv"))
                .rename(columns={"Unnamed: 0": "timeindex"})
                .set_index("timeindex")["residual_load"]
            )
            residual_load.index = pd.to_datetime(residual_load.index)
            edisgo_obj.timeseries._residual_load = residual_load
    return edisgo_obj
