from __future__ import annotations

import copy
import logging
import os
import pickle
import shutil

from numbers import Number
from pathlib import PurePath

import numpy as np
import pandas as pd

from edisgo.flex_opt.charging_strategies import charging_strategy
from edisgo.flex_opt.heat_pump_operation import (
    operating_strategy as hp_operating_strategy,
)
from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.io import pypsa_io
from edisgo.io.ding0_import import import_ding0_grid
from edisgo.io.electromobility_import import (
    distribute_charging_demand,
    import_electromobility,
    integrate_charging_parks,
)
from edisgo.io.generators_import import oedb as import_generators_oedb

# from edisgo.io.heat_pump_import import oedb as import_heat_pumps_oedb
from edisgo.network import timeseries
from edisgo.network.electromobility import Electromobility
from edisgo.network.heat import HeatPump
from edisgo.network.results import Results
from edisgo.network.topology import Topology
from edisgo.opf.results.opf_result_class import OPFResults
from edisgo.opf.run_mp_opf import run_mp_opf
from edisgo.tools import plots, tools
from edisgo.tools.config import Config
from edisgo.tools.geo import find_nearest_bus

if "READTHEDOCS" not in os.environ:
    from shapely.geometry import Point

logger = logging.getLogger(__name__)


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
    config_path : None or str or dict
        Path to the config directory. Options are:

        * 'default' (default)
            If `config_path` is set to 'default', the provided default config files
            are used directly.
        * str
            If `config_path` is a string, configs will be loaded from the
            directory specified by `config_path`. If the directory
            does not exist, it is created. If config files don't exist, the
            default config files are copied into the directory.
        * dict
            A dictionary can be used to specify different paths to the
            different config files. The dictionary must have the following
            keys:

            * 'config_db_tables'

            * 'config_grid'

            * 'config_grid_expansion'

            * 'config_timeseries'

            Values of the dictionary are paths to the corresponding
            config file. In contrast to the other options, the directories
            and config files must exist and are not automatically created.
        * None
            If `config_path` is None, configs are loaded from the edisgo
            default config directory ($HOME$/.edisgo). If the directory
            does not exist, it is created. If config files don't exist, the
            default config files are copied into the directory.

        Default: "default".

    Attributes
    ----------
    topology : :class:`~.network.topology.Topology`
        The topology is a container object holding the topology of the grids including
        buses, lines, transformers, switches and components connected to the grid
        including generators, loads and storage units.
    timeseries : :class:`~.network.timeseries.TimeSeries`
        Container for active and reactive power time series of generators, loads and
        storage units.
    results : :class:`~.network.results.Results`
        This is a container holding all calculation results from power flow
        analyses and grid reinforcement.
    electromobility : :class:`~.network.electromobility.Electromobility`
        This class holds data on charging processes (how long cars are parking at a
        charging station, how much they need to charge, etc.) necessary to apply
        different charging strategies, as well as information on potential charging
        sites and integrated charging parks.
    heat_pump : :class:`~.network.heat.HeatPump`
        This is a container holding heat pump data such as COP, heat demand to be
        served and heat storage information.

    """

    def __init__(self, **kwargs):

        # load configuration
        self._config = Config(**kwargs)

        # instantiate topology object and load grid data
        self.topology = Topology(config=self.config)
        self.import_ding0_grid(path=kwargs.get("ding0_grid", None))

        # set up results and time series container
        self.results = Results(self)
        self.opf_results = OPFResults()
        self.timeseries = timeseries.TimeSeries(
            timeindex=kwargs.get("timeindex", pd.DatetimeIndex([]))
        )

        # instantiate electromobility and heat pump object
        self.electromobility = Electromobility(edisgo_obj=self)
        self.heat_pump = HeatPump()

        # import new generators
        if kwargs.get("generator_scenario", None) is not None:
            self.import_generators(
                generator_scenario=kwargs.pop("generator_scenario"), **kwargs
            )

    @property
    def config(self):
        """
        eDisGo configuration data.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keyword arguments to set up Config object. See parameters
            of :class:`~.tools.config.Config` class for more information on possible
            input parameters.

        Returns
        -------
        :class:`~.tools.config.Config`
            Config object with configuration data from config files.

        """
        return self._config

    @config.setter
    def config(self, kwargs):
        self._config = Config(**kwargs)

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

        Notes
        ------
        This function raises a warning in case a time index was not previously set.
        You can set the time index upon initialisation of the EDisGo object by
        providing the input parameter 'timeindex' or using the function
        :attr:`~.edisgo.EDisGo.set_timeindex`.
        Also make sure that the time steps for which time series are provided include
        the set time index.

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

    def set_time_series_worst_case_analysis(
        self,
        cases=None,
        generators_names=None,
        loads_names=None,
        storage_units_names=None,
    ):
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
        generators_names : list(str)
            Defines for which generators to set worst case time series. If None,
            time series are set for all generators. Default: None.
        loads_names : list(str)
            Defines for which loads to set worst case time series. If None,
            time series are set for all loads. Default: None.
        storage_units_names : list(str)
            Defines for which storage units to set worst case time series. If None,
            time series are set for all storage units. Default: None.

        """
        if cases is None:
            cases = ["load_case", "feed-in_case"]
        if isinstance(cases, str):
            cases = [cases]

        self.timeseries.set_worst_case(
            self, cases, generators_names, loads_names, storage_units_names
        )

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

        Notes
        ------
        This function raises a warning in case a time index was not previously set.
        You can set the time index upon initialisation of the EDisGo object by
        providing the input parameter 'timeindex' or using the function
        :attr:`~.edisgo.EDisGo.set_timeindex`.
        Also make sure that the time steps for which time series are provided include
        the set time index.

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
            return
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
        Set reactive power time series of components.

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

        Notes
        ------
        Be careful to set parametrisation of other component types to None if you only
        want to set reactive power of certain components. See example below for further
        information.

        Examples
        --------
        To only set reactive power time series of one generator using default
        configurations you can do the following:

        >>> self.set_time_series_reactive_power_control(
        >>>     generators_parametrisation=pd.DataFrame(
        >>>        {
        >>>            "components": [["Generator_1"]],
        >>>            "mode": ["default"],
        >>>            "power_factor": ["default"],
        >>>        },
        >>>        index=[1],
        >>>     ),
        >>>     loads_parametrisation=None,
        >>>     storage_units_parametrisation=None
        >>> )

        In the example above, `loads_parametrisation` and
        `storage_units_parametrisation` need to be set to None, otherwise already
        existing time series would be overwritten.

        To only change configuration of one load and for all other components use
        default configurations you can do the following:

        >>> self.set_time_series_reactive_power_control(
        >>>     loads_parametrisation=pd.DataFrame(
        >>>        {
        >>>            "components": [["Load_1"],
        >>>                           self.topology.loads_df.index.drop(["Load_1"])],
        >>>            "mode": ["capacitive", "default"],
        >>>            "power_factor": [0.98, "default"],
        >>>        },
        >>>        index=[1, 2],
        >>>     )
        >>> )

        In the example above, `generators_parametrisation` and
        `storage_units_parametrisation` do not need to be set as default configurations
        are per default used for all generators and storage units anyways.

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

    def to_pypsa(
        self, mode=None, timesteps=None, check_edisgo_integrity=False, **kwargs
    ):
        """
        Convert grid to :pypsa:`PyPSA.Network<network>` representation.

        You can choose between translation of the MV and all underlying LV grids
        (mode=None (default)), the MV network only (mode='mv' or mode='mvlv') or a
        single LV network (mode='lv').

        Parameters
        -----------
        mode : str
            Determines network levels that are translated to
            :pypsa:`PyPSA.Network<network>`.
            Possible options are:

            * None

                MV and underlying LV networks are exported. This is the default.

            * 'mv'

                Only MV network is exported. MV/LV transformers are not exported in
                this mode. Loads, generators and storage units in underlying LV grids
                are connected to the respective MV/LV station's primary side. Per
                default, they are all connected separately, but you can also choose to
                aggregate them. See parameters `aggregate_loads`, `aggregate_generators`
                and `aggregate_storages` for more information.

            * 'mvlv'

                This mode works similar as mode 'mv', with the difference that MV/LV
                transformers are as well exported and LV components connected to the
                respective MV/LV station's secondary side. Per default, all components
                are connected separately, but you can also choose to aggregate them.
                See parameters `aggregate_loads`, `aggregate_generators`
                and `aggregate_storages` for more information.

            * 'lv'

                Single LV network topology including the MV/LV transformer is exported.
                The LV grid to export is specified through the parameter `lv_grid_id`.
                The slack is positioned at the secondary side of the MV/LV station.

        timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
            :pandas:`pandas.Timestamp<Timestamp>`
            Specifies which time steps to export to pypsa representation to e.g.
            later on use in power flow analysis. It defaults to None in which case
            all time steps in :attr:`~.network.timeseries.TimeSeries.timeindex`
            are used.
            Default: None.
        check_edisgo_integrity : bool
            Check integrity of edisgo object before translating to pypsa. This option is
            meant to help the identification of possible sources of errors if the power
            flow calculations fail. See :attr:`~.edisgo.EDisGo.check_integrity` for
            more information.

        Other Parameters
        -------------------
        use_seed : bool
            Use a seed for the initial guess for the Newton-Raphson algorithm.
            Only available when MV level is included in the power flow analysis.
            If True, uses voltage magnitude results of previous power flow
            analyses as initial guess in case of PQ buses. PV buses currently do
            not occur and are therefore currently not supported.
            Default: False.
        lv_grid_id : int or str
            ID (e.g. 1) or name (string representation, e.g. "LVGrid_1") of LV grid
            to export in case mode is 'lv'.
        aggregate_loads : str
            Mode for load aggregation in LV grids in case mode is 'mv' or 'mvlv'.
            Can be 'sectoral' aggregating the loads sector-wise, 'all' aggregating all
            loads into one or None, not aggregating loads but appending them to the
            station one by one. Default: None.
        aggregate_generators : str
            Mode for generator aggregation in LV grids in case mode is 'mv' or 'mvlv'.
            Can be 'type' aggregating generators per generator type, 'curtailable'
            aggregating 'solar' and 'wind' generators into one and all other generators
            into another one, or None, where no aggregation is undertaken
            and generators are added to the station one by one. Default: None.
        aggregate_storages : str
            Mode for storage unit aggregation in LV grids in case mode is 'mv' or
            'mvlv'. Can be 'all' where all storage units in an LV grid are aggregated to
            one storage unit or None, in which case no aggregation is conducted and
            storage units are added to the station. Default: None.

        Returns
        -------
        :pypsa:`PyPSA.Network<network>`
            :pypsa:`PyPSA.Network<network>` representation.

        """
        # possibly execute consistency check
        if check_edisgo_integrity or logger.level == logging.DEBUG:
            self.check_integrity()
        return pypsa_io.to_pypsa(self, mode, timesteps, **kwargs)

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

    def analyze(
        self,
        mode: str | None = None,
        timesteps: pd.Timestamp | pd.DatetimeIndex | None = None,
        raise_not_converged: bool = True,
        troubleshooting_mode: str | None = None,
        range_start: Number = 0.1,
        range_num: int = 10,
        **kwargs,
    ):
        """
        Conducts a static, non-linear power flow analysis.

        Conducts a static, non-linear power flow analysis using
        `PyPSA <https://pypsa.readthedocs.io/en/latest/power_flow.html#\
        full-non-linear-power-flow>`_
        and writes results (active, reactive and apparent power as well as
        current on lines and voltages at buses) to :class:`~.network.results.Results`
        (e.g. :attr:`~.network.results.Results.v_res` for voltages).

        Parameters
        ----------
        mode : str or None
            Allows to toggle between power flow analysis for the whole network or just
            the MV or one LV grid. Possible options are:

            * None (default)

                Power flow analysis is conducted for the whole network including MV grid
                and underlying LV grids.

            * 'mv'

                Power flow analysis is conducted for the MV level only. LV loads,
                generators and storage units are aggregated at the respective MV/LV
                stations' primary side. Per default, they are all connected separately,
                but you can also choose to aggregate them. See parameters
                `aggregate_loads`, `aggregate_generators` and `aggregate_storages`
                in :attr:`~.edisgo.EDisGo.to_pypsa` for more information.

            * 'mvlv'

                Power flow analysis is conducted for the MV level only. In contrast to
                mode 'mv' LV loads, generators and storage units are in this case
                aggregated at the respective MV/LV stations' secondary side. Per
                default, they are all connected separately, but you can also choose to
                aggregate them. See parameters `aggregate_loads`, `aggregate_generators`
                and `aggregate_storages` in :attr:`~.edisgo.EDisGo.to_pypsa` for more
                information.

            * 'lv'

                Power flow analysis is conducted for one LV grid only. ID or name of
                the LV grid to conduct power flow analysis for needs to be provided
                through keyword argument 'lv_grid_id' as integer or string.
                See parameter `lv_grid_id` in :attr:`~.edisgo.EDisGo.to_pypsa` for more
                information.
                The slack is positioned at the secondary side of the MV/LV station.

        timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
            :pandas:`pandas.Timestamp<Timestamp>`
            Timesteps specifies for which time steps to conduct the power flow
            analysis. It defaults to None in which case all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex` are used.
        raise_not_converged : bool
            If True, an error is raised in case power flow analysis did not converge
            for all time steps.
            Default: True.

        troubleshooting_mode : str or None
            Two optional troubleshooting methods in case of nonconvergence of nonlinear
            power flow (cf. [1])

            * None (default)
                Power flow analysis is conducted using nonlinear power flow method.
            * 'lpf'
                Non-linear power flow initial guess is seeded with the voltage angles
                from the linear power flow.
            * 'iteration'
                Power flow analysis is conducted by reducing all power values of
                generators and loads to a fraction, e.g. 10%, solving the load flow and
                using it as a seed for the power at 20%, iteratively up to 100%.

        range_start : float, optional
            Specifies the minimum fraction that power values are set to when using
            troubleshooting_mode 'iteration'. Must be between 0 and 1.
            Default: 0.1.

        range_num : int, optional
            Specifies the number of fraction samples to generate when using
            troubleshooting_mode 'iteration'. Must be non-negative.
            Default: 10.

        Other Parameters
        -----------------
        kwargs : dict
            Possible other parameters comprise all other parameters that can be set in
            :func:`edisgo.io.pypsa_io.to_pypsa`.

        Returns
        --------
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
            Returns the time steps for which power flow analysis did not converge.

        References
        --------
        [1] https://pypsa.readthedocs.io/en/latest/troubleshooting.html

        """

        def _check_convergence():
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
                    "following {} time steps: {}.".format(
                        len(timesteps_not_converged), timesteps_not_converged
                    )
                )
            elif len(timesteps_not_converged) > 0:
                logger.warning(
                    "Power flow analysis did not converge for the "
                    "following {} time steps: {}.".format(
                        len(timesteps_not_converged), timesteps_not_converged
                    )
                )
            return timesteps_converged, timesteps_not_converged

        if timesteps is None:
            timesteps = self.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]

        pypsa_network = self.to_pypsa(mode=mode, timesteps=timesteps, **kwargs)

        if troubleshooting_mode == "lpf":
            # run linear power flow analysis
            pypsa_network.lpf()
            # run power flow analysis
            pf_results = pypsa_network.pf(timesteps, use_seed=True)
            # get converged and not converged time steps
            timesteps_converged, timesteps_not_converged = _check_convergence()
        elif troubleshooting_mode == "iteration":
            pypsa_network_copy = pypsa_network.copy()
            for fraction in np.linspace(range_start, 1, range_num):
                # Reduce power values of generators, loads and storages to fraction of
                # original value
                for obj1, obj2 in [
                    (pypsa_network.generators_t, pypsa_network_copy.generators_t),
                    (pypsa_network.loads_t, pypsa_network_copy.loads_t),
                    (pypsa_network.storage_units_t, pypsa_network_copy.storage_units_t),
                ]:
                    for attr in ["p_set", "q_set"]:
                        setattr(obj1, attr, getattr(obj2, attr) * fraction)
                # run power flow analysis
                pf_results = pypsa_network.pf(timesteps, use_seed=True)
                logging.warning(
                    "Current fraction in iterative process: {}.".format(fraction)
                )
                # get converged and not converged time steps
                timesteps_converged, timesteps_not_converged = _check_convergence()
        else:
            # run power flow analysis
            pf_results = pypsa_network.pf(
                timesteps, use_seed=kwargs.get("use_seed", False)
            )
            # get converged and not converged time steps
            timesteps_converged, timesteps_not_converged = _check_convergence()

        # handle converged time steps
        pypsa_io.process_pfa_results(self, pypsa_network, timesteps_converged)

        return timesteps_not_converged

    def reinforce(
        self,
        timesteps_pfa: str | pd.DatetimeIndex | pd.Timestamp | None = None,
        copy_grid: bool = False,
        max_while_iterations: int = 20,
        combined_analysis: bool = False,
        mode: str | None = None,
        without_generator_import: bool = False,
        **kwargs,
    ) -> Results:
        """
        Reinforces the network and calculates network expansion costs.

        If the :attr:`edisgo.network.timeseries.TimeSeries.is_worst_case` is
        True input for `timesteps_pfa` and `mode` are overwritten and therefore
        ignored.

        See :func:`edisgo.flex_opt.reinforce_grid.reinforce_grid` for more
        information on input parameters and methodology.

        Other Parameters
        -----------------
        is_worst_case : bool
            Is used to overwrite the return value from
            :attr:`edisgo.network.timeseries.TimeSeries.is_worst_case`. If True
            reinforcement is calculated for worst-case MV and LV cases separately.

        """
        if kwargs.get("is_worst_case", self.timeseries.is_worst_case):

            logger.debug(
                "Running reinforcement in worst-case mode by differentiating between "
                "MV and LV load and feed-in cases."
            )

            if copy_grid:
                edisgo_obj = copy.deepcopy(self)
            else:
                edisgo_obj = self

            timeindex_worst_cases = self.timeseries.timeindex_worst_cases

            if mode != "lv":

                timesteps_pfa = pd.DatetimeIndex(
                    timeindex_worst_cases.loc[
                        timeindex_worst_cases.index.str.contains("mv")
                    ]
                )
                reinforce_grid(
                    edisgo_obj,
                    max_while_iterations=max_while_iterations,
                    copy_grid=False,
                    timesteps_pfa=timesteps_pfa,
                    combined_analysis=combined_analysis,
                    mode="mv",
                    without_generator_import=without_generator_import,
                )

            if mode != "mv":
                timesteps_pfa = pd.DatetimeIndex(
                    timeindex_worst_cases.loc[
                        timeindex_worst_cases.index.str.contains("lv")
                    ]
                )
                reinforce_mode = mode if mode == "mvlv" else "lv"
                reinforce_grid(
                    edisgo_obj,
                    max_while_iterations=max_while_iterations,
                    copy_grid=False,
                    timesteps_pfa=timesteps_pfa,
                    combined_analysis=combined_analysis,
                    mode=reinforce_mode,
                    without_generator_import=without_generator_import,
                )

            if mode not in ["mv", "lv"]:
                edisgo_obj.analyze(mode=mode)
            results = edisgo_obj.results

        else:
            results = reinforce_grid(
                self,
                max_while_iterations=max_while_iterations,
                copy_grid=copy_grid,
                timesteps_pfa=timesteps_pfa,
                combined_analysis=combined_analysis,
                mode=mode,
                without_generator_import=without_generator_import,
            )

        # add measure to Results object
        if not copy_grid:
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
        return run_mp_opf(self, timesteps, storage_series=storage_series, **kwargs)

    def add_component(
        self,
        comp_type,
        ts_active_power=None,
        ts_reactive_power=None,
        **kwargs,
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
        ts_active_power : :pandas:`pandas.Series<series>` or None
            Active power time series of added component.
            Index of the series must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are active power per time step in MW.
            Defaults to None in which case no time series is set.
        ts_reactive_power : :pandas:`pandas.Series<series>` or str or None
            Possible options are:

            * :pandas:`pandas.Series<series>`

                Reactive power time series of added component. Index of the series must
                contain all time steps in
                :attr:`~.network.timeseries.TimeSeries.timeindex`. Values are reactive
                power per time step in MVA.

            * "default"

                Reactive power time series is determined based on assumptions on fixed
                power factor of the component. To this end, the power factors set in the
                config section `reactive_power_factor` and the power factor mode,
                defining whether components behave inductive or capacitive, given in the
                config section `reactive_power_mode`, are used.
                This option requires you to provide an active power time series. In case
                it was not provided, reactive power cannot be set and a warning is
                raised.

            * None

                No reactive power time series is set.

            Default: None
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

        def _get_q_default_df(comp_name):
            return pd.DataFrame(
                {
                    "components": [[comp_name]],
                    "mode": ["default"],
                    "power_factor": ["default"],
                },
                index=["comp"],
            )

        def _set_timeseries():
            if ts_active_power is not None:
                self.set_time_series_manual(
                    **{f"{comp_type}s_p": pd.DataFrame({comp_name: ts_active_power})}
                )
            if ts_reactive_power is not None:
                if isinstance(ts_reactive_power, pd.Series):
                    self.set_time_series_manual(
                        **{
                            f"{comp_type}s_q": pd.DataFrame(
                                {comp_name: ts_reactive_power}
                            )
                        }
                    )
                elif ts_reactive_power == "default":
                    if ts_active_power is None:
                        logging.warning(
                            f"Default reactive power time series of {comp_name} cannot "
                            "be set as active power time series was not provided."
                        )
                    else:
                        other_comps = [
                            _
                            for _ in ["generator", "load", "storage_unit"]
                            if _ != comp_type
                        ]
                        parameter_dict = {
                            f"{t}s_parametrisation": None for t in other_comps
                        }
                        parameter_dict.update(
                            {
                                f"{comp_type}s_parametrisation": _get_q_default_df(
                                    comp_name
                                )
                            }
                        )
                        self.set_time_series_reactive_power_control(**parameter_dict)

        if comp_type == "bus":
            comp_name = self.topology.add_bus(**kwargs)

        elif comp_type == "line":
            comp_name = self.topology.add_line(**kwargs)

        elif comp_type == "generator":
            comp_name = self.topology.add_generator(**kwargs)
            _set_timeseries()

        elif comp_type == "storage_unit":
            comp_name = self.topology.add_storage_unit(**kwargs)
            _set_timeseries()

        elif comp_type == "load":
            comp_name = self.topology.add_load(**kwargs)
            _set_timeseries()

        else:
            raise ValueError(
                "Invalid input for parameter 'comp_type'. Must either be "
                "'line', 'bus', 'generator', 'load' or 'storage_unit'."
            )
        return comp_name

    def integrate_component_based_on_geolocation(
        self,
        comp_type,
        geolocation,
        voltage_level=None,
        add_ts=True,
        ts_active_power=None,
        ts_reactive_power=None,
        **kwargs,
    ):
        """
        Adds single component to topology based on geolocation.

        Currently components can be generators, charging points and heat pumps.

        See :attr:`~.network.topology.Topology.connect_to_mv` and
        :attr:`~.network.topology.Topology.connect_to_lv` for more information.

        Parameters
        ----------
        comp_type : str
            Type of added component. Can be 'generator', 'charging_point' or
            'heat_pump'.
        geolocation : :shapely:`shapely.Point<Point>` or tuple
            Geolocation of the new component. In case of tuple, the geolocation
            must be given in the form (longitude, latitude).
        voltage_level : int, optional
            Specifies the voltage level the new component is integrated in.
            Possible options are 4 (MV busbar), 5 (MV grid), 6 (LV busbar) or
            7 (LV grid). If no voltage level is provided the voltage level
            is determined based on the nominal power `p_nom` or `p_set` (given as kwarg)
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
            Values are active power per time step in MW. If you want
            to add time series (if `add_ts` is True), you must provide a
            time series. It is not automatically retrieved.
        ts_reactive_power : :pandas:`pandas.Series<Series>`, optional
            Reactive power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are reactive power per time step in MVA. If you
            want to add time series (if `add_ts` is True), you must provide a
            time series. It is not automatically retrieved.

        Other Parameters
        ------------------
        kwargs :
            Attributes of added component.
            See :attr:`~.network.topology.Topology.add_generator` respectively
            :attr:`~.network.topology.Topology.add_load` methods
            for more information on required and optional parameters of
            generators respectively charging points and heat pumps.

        """
        supported_voltage_levels = {4, 5, 6, 7}
        p_nom = kwargs.get("p_nom", None)
        p_set = kwargs.get("p_set", None)

        p = p_nom if p_set is None else p_set

        kwargs["p"] = p

        if voltage_level not in supported_voltage_levels:
            if p is None:
                raise ValueError(
                    "Neither appropriate voltage level nor nominal power "
                    "were supplied."
                )
            # Determine voltage level manually from nominal power
            if 4.5 < p <= 17.5:
                voltage_level = 4
            elif 0.3 < p <= 4.5:
                voltage_level = 5
            elif 0.1 < p <= 0.3:
                voltage_level = 6
            elif 0 < p <= 0.1:
                voltage_level = 7
            else:
                raise ValueError("Unsupported voltage level")

        # check if geolocation is given as shapely Point, otherwise transform
        # to shapely Point
        if type(geolocation) is not Point:
            geolocation = Point(geolocation)

        # Connect in MV
        if voltage_level in [4, 5]:
            kwargs["voltage_level"] = voltage_level
            kwargs["geom"] = geolocation
            comp_name = self.topology.connect_to_mv(self, kwargs, comp_type)

        # Connect in LV
        else:
            if kwargs.get("mvlv_subst_id", None) is None:
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
                    self.timeseries.drop_component_time_series(
                        df_name=f"loads_{ts}", comp_names=comp_name
                    )

        elif comp_type == "generator":
            self.topology.remove_generator(comp_name)
            if drop_ts:
                for ts in ["active_power", "reactive_power"]:
                    self.timeseries.drop_component_time_series(
                        df_name=f"generators_{ts}",
                        comp_names=comp_name,
                    )

        elif comp_type == "storage_unit":
            self.topology.remove_storage_unit(comp_name)
            if drop_ts:
                for ts in ["active_power", "reactive_power"]:
                    self.timeseries.drop_component_time_series(
                        df_name=f"storage_units_{ts}",
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

    def import_electromobility(
        self,
        simbev_directory: PurePath | str,
        tracbev_directory: PurePath | str,
        import_electromobility_data_kwds=None,
        allocate_charging_demand_kwds=None,
    ):
        """
        Imports electromobility data and integrates charging points into grid.

        So far, this function requires electromobility data from
        `SimBEV <https://github.com/rl-institut/simbev>`_ (required version:
        `3083c5a <https://github.com/rl-institut/simbev/commit/
        86076c936940365587c9fba98a5b774e13083c5a>`_) and
        `TracBEV <https://github.com/rl-institut/tracbev>`_ (required version:
        `14d864c <https://github.com/rl-institut/tracbev/commit/
        03e335655770a377166c05293a966052314d864c>`_) to be stored in the
        directories specified through the parameters `simbev_directory` and
        `tracbev_directory`. SimBEV provides data on standing times, charging demand,
        etc. per vehicle, whereas TracBEV provides potential charging point locations.

        After electromobility data is loaded, the charging demand from SimBEV is
        allocated to potential charging points from TracBEV. Afterwards,
        all potential charging points with charging demand allocated to them are
        integrated into the grid.

        Be aware that this function does not yield charging time series per charging
        point but only charging processes (see
        :attr:`~.network.electromobility.Electromobility.charging_processes_df` for
        more information). The actual charging time series are determined through
        applying a charging strategy using the function
        :attr:`~.edisgo.EDisGo.charging_strategy`.

        Parameters
        ----------
        simbev_directory : str
            SimBEV directory holding SimBEV data.
        tracbev_directory : str
            TracBEV directory holding TracBEV data.
        import_electromobility_data_kwds : dict
            These may contain any further attributes you want to specify when calling
            the function to import electromobility data from SimBEV and TracBEV using
            :func:`~.io.electromobility_import.import_electromobility`.

            gc_to_car_rate_home : float
                Specifies the minimum rate between potential charging parks
                points for the use case "home" and the total number of cars.
                Default 0.5.
            gc_to_car_rate_work : float
                Specifies the minimum rate between potential charging parks
                points for the use case "work" and the total number of cars.
                Default 0.25.
            gc_to_car_rate_public : float
                Specifies the minimum rate between potential charging parks
                points for the use case "public" and the total number of cars.
                Default 0.1.
            gc_to_car_rate_hpc : float
                Specifies the minimum rate between potential charging parks
                points for the use case "hpc" and the total number of cars.
                Default 0.005.
            mode_parking_times : str
                If the mode_parking_times is set to "frugal" only parking times
                with any charging demand are imported. Any other input will lead
                to all parking and driving events being imported. Default "frugal".
            charging_processes_dir : str
                Charging processes sub-directory. Default None.
            simbev_config_file : str
                Name of the simbev config file. Default "metadata_simbev_run.json".

        allocate_charging_demand_kwds :
            These may contain any further attributes you want to specify when calling
            the function that allocates charging processes from SimBEV to potential
            charging points from TracBEV using
            :func:`~.io.electromobility_import.distribute_charging_demand`.

            mode : str
                Distribution mode. If the mode is set to "user_friendly" only the
                simbev weights are used for the distribution. If the mode is
                "grid_friendly" also grid conditions are respected.
                Default "user_friendly".
            generators_weight_factor : float
                Weighting factor of the generators weight within an LV grid in
                comparison to the loads weight. Default 0.5.
            distance_weight : float
                Weighting factor for the distance between a potential charging park
                and its nearest substation in comparison to the combination of
                the generators and load factors of the LV grids.
                Default 1 / 3.
            user_friendly_weight : float
                Weighting factor of the user friendly weight in comparison to the
                grid friendly weight. Default 0.5.

        """
        if import_electromobility_data_kwds is None:
            import_electromobility_data_kwds = {}

        import_electromobility(
            self,
            simbev_directory,
            tracbev_directory,
            **import_electromobility_data_kwds,
        )

        if allocate_charging_demand_kwds is None:
            allocate_charging_demand_kwds = {}

        distribute_charging_demand(self, **allocate_charging_demand_kwds)

        integrate_charging_parks(self)

    def apply_charging_strategy(self, strategy="dumb", **kwargs):
        """
        Applies charging strategy to set EV charging time series at charging parks.

        This function requires that standing times, charging demand, etc. at
        charging parks were previously set using
        :attr:`~.edisgo.EDisGo.import_electromobility`.

        It is assumed that only 'private' charging processes at 'home' or at 'work' can
        be flexibilized. 'public' charging processes will always be 'dumb'.

        The charging time series at each charging parks are written to
        :attr:`~.network.timeseries.TimeSeries.loads_active_power`. Reactive power
        in :attr:`~.network.timeseries.TimeSeries.loads_reactive_power` is
        set to 0 Mvar.

        Parameters
        ----------
        strategy : str
            Defines the charging strategy to apply. The following charging
            strategies are valid:

            * 'dumb'

                The cars are charged directly after arrival with the
                maximum possible charging capacity.

            * 'reduced'

                The cars are charged directly after arrival with the
                minimum possible charging power. The minimum possible charging
                power is determined by the parking time and the parameter
                `minimum_charging_capacity_factor`.

            * 'residual'

                The cars are charged when the residual load in the MV
                grid is lowest (high generation and low consumption).
                Charging processes with a low flexibility are given priority.

            Default: 'dumb'.

        Other Parameters
        ------------------
        timestamp_share_threshold : float
            Percental threshold of the time required at a time step for charging
            the vehicle. If the time requirement is below this limit, then the
            charging process is not mapped into the time series. If, however, it is
            above this limit, the time step is mapped to 100% into the time series.
            This prevents differences between the charging strategies and creates a
            compromise between the simultaneity of charging processes and an
            artificial increase in the charging demand. Default: 0.2.
        minimum_charging_capacity_factor : float
            Technical minimum charging power of charging points in p.u. used in case of
            charging strategy 'reduced'. E.g. for a charging point with a nominal
            capacity of 22 kW and a minimum_charging_capacity_factor of 0.1 this would
            result in a minimum charging power of 2.2 kW. Default: 0.1.

        """
        charging_strategy(self, strategy=strategy, **kwargs)

    def import_heat_pumps(self, scenario=None, **kwargs):
        """
        Gets heat pump capacities for specified scenario from oedb and integrates them
        into the grid.

        Besides heat pump capacity the heat pump's COP and heat demand to be served
        are as well retrieved.

        Currently, the only supported data source is scenario data generated
        in the research project `eGo^n <https://ego-n.org/>`_. You can choose
        between two scenarios: 'eGon2035' and 'eGon100RE'.

        The data is retrieved from the
        `open energy platform <https://openenergy-platform.org/>`_.

        # ToDo Add information on scenarios and from which tables data is retrieved.

        The following steps are conducted in this function:

            * Spatially disaggregated data on heat pump capacities in individual and
              district heating are obtained from the database for the specified
              scenario.
            * Heat pumps are integrated into the grid (added to
              :attr:`~.network.topology.Topology.loads_df`).

              * Grid connection points of heat pumps for individual heating are
                determined based on the corresponding building ID.
              * Grid connection points of heat pumps for district heating are determined
                based on their geolocation and installed capacity.
                See :attr:`~.network.topology.Topology.connect_to_mv` and
                :attr:`~.network.topology.Topology.connect_to_lv` for more information.
            * COP and heat demand for each heat pump are retrieved from the database
              and stored in the :class:`~.network.heat.HeatPump` class that can be
              accessed through :attr:`~.edisgo.EDisGo.heat_pump`.

        Be aware that this function does not yield electricity load time series for the
        heat pumps. The actual time series are determined through applying an
        operation strategy or optimising heat pump dispatch.

        After the heat pumps are integrated there may be grid issues due to the
        additional load. These are not solved automatically. If you want to
        have a stable grid without grid issues you can invoke the automatic
        grid expansion through the function :attr:`~.EDisGo.reinforce`.

        Parameters
        ----------
        scenario : str
            Scenario for which to retrieve heat pump data. Possible options
            are 'eGon2035' and 'eGon100RE'.

        Other Parameters
        ----------------
        kwargs :
            See :func:`edisgo.io.heat_pump_import.oedb`.

        """
        raise NotImplementedError
        # integrated_heat_pumps = import_heat_pumps_oedb(
        #     edisgo_object=self, scenario=scenario, **kwargs
        # )
        # self.heat_pump.set_heat_demand(
        #     self, "oedb", heat_pump_names=integrated_heat_pumps
        # )
        # self.heat_pump.set_cop(self, "oedb", heat_pump_names=integrated_heat_pumps)

    def apply_heat_pump_operating_strategy(
        self, strategy="uncontrolled", heat_pump_names=None, **kwargs
    ):
        """
        Applies operating strategy to set electrical load time series of heat pumps.

        This function requires that COP and heat demand time series, and depending on
        the operating strategy also information on thermal storage units,
        were previously set in :attr:`~.edisgo.EDisGo.heat_pump`. COP and heat demand
        information is automatically set when using
        :attr:`~.edisgo.EDisGo.import_heat_pumps`. When not using this function it can
        be manually set using :attr:`~.network.heat.HeatPump.set_cop` and
        :attr:`~.network.heat.HeatPump.set_heat_demand`.

        The electrical load time series of each heat pump are written to
        :attr:`~.network.timeseries.TimeSeries.loads_active_power`. Reactive power
        in :attr:`~.network.timeseries.TimeSeries.loads_reactive_power` is
        set to 0 Mvar.

        Parameters
        ----------
        strategy : str
            Defines the operating strategy to apply. The following strategies are valid:

            * 'uncontrolled'

                The heat demand is directly served by the heat pump without buffering
                heat using a thermal storage. The electrical load of the heat pump is
                determined as follows:

                .. math::

                    P_{el} = P_{th} / COP

            Default: 'uncontrolled'.

        heat_pump_names : list(str) or None
            Defines for which heat pumps to apply operating strategy. If None, all heat
            pumps for which COP information in :attr:`~.edisgo.EDisGo.heat_pump` is
            given are used. Default: None.

        """
        hp_operating_strategy(self, strategy=strategy, heat_pump_names=heat_pump_names)

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
                title = f"Voltage histogram for time step {timestep[0]}"
            else:
                title = (
                    f"Voltage histogram \nfor time steps {timestep[0]} to "
                    f"{timestep[-1]}"
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
                title = f"Relative line load histogram for time step {timestep[0]}"
            else:
                title = (
                    "Relative line load histogram \nfor time steps "
                    f"{timestep[0]} to {timestep[-1]}"
                )
        elif title is False:
            title = None
        plots.histogram(data=rel_line_loading, title=title, **kwargs)

    def save(
        self,
        directory,
        save_topology=True,
        save_timeseries=True,
        save_results=True,
        save_electromobility=False,
        save_heatpump=False,
        **kwargs,
    ):
        """
        Saves EDisGo object to csv files.

        It can be chosen what is included in the csv export (e.g. power flow results,
        electromobility flexibility, etc.). Further, in order to save disk storage space
        the data type of time series data can be reduced, e.g. to float32 and data
        can be archived, e.g. in a zip archive.

        Parameters
        ----------
        directory : str
            Main directory to save EDisGo object to.
        save_topology : bool, optional
            Indicates whether to save :class:`~.network.topology.Topology` object.
            Per default it is saved to sub-directory 'topology'. See
            :attr:`~.network.topology.Topology.to_csv` for more information.
            Default: True.
        save_timeseries : bool, optional
            Indicates whether to save :class:`~.network.timeseries.Timeseries` object.
            Per default it is saved to subdirectory 'timeseries'.
            Through the keyword arguments `reduce_memory`
            and `to_type` it can be chosen if memory should be reduced. See
            :attr:`~.network.timeseries.Timeseries.to_csv` for more
            information.
            Default: True.
        save_results : bool, optional
            Indicates whether to save :class:`~.network.results.Results`
            object. Per default it is saved to subdirectory 'results'.
            Through the keyword argument `parameters` the results that should
            be stored can be specified. Further, through the keyword parameters
            `reduce_memory` and `to_type` it can be chosen if memory should be reduced.
            See :attr:`~.network.results.Results.to_csv` for more information.
            Default: True.
        save_electromobility : bool, optional
            Indicates whether to save
            :class:`~.network.electromobility.Electromobility` object. Per default it is
            not saved. If set to True, it is saved to subdirectory 'electromobility'.
            See :attr:`~.network.electromobility.Electromobility.to_csv` for more
            information.
        save_heatpump : bool, optional
            Indicates whether to save
            :class:`~.network.heat.HeatPump` object. Per default it is not saved.
            If set to True, it is saved to subdirectory 'heat_pump'.
            See :attr:`~.network.heat.HeatPump.to_csv` for more information.

        Other Parameters
        ------------------
        reduce_memory : bool, optional
            If True, size of dataframes containing time series in
            :class:`~.network.results.Results`, :class:`~.network.timeseries.TimeSeries`
            and :class:`~.network.heat.HeatPump`
            is reduced. See respective classes `reduce_memory` functions for more
            information. Type to convert to can be specified by providing
            `to_type` as keyword argument. Further parameters of reduce_memory
            functions cannot be passed here. Call these functions directly to
            make use of further options. Default: False.
        to_type : str, optional
            Data type to convert time series data to. This is a trade-off
            between precision and memory. Default: "float32".
        parameters : None or dict
            Specifies which results to store. By default this is set to None,
            in which case all available results are stored.
            To only store certain results provide a dictionary. See function docstring
            `parameters` parameter in :func:`~.network.results.Results.to_csv`
            for more information.
        electromobility_attributes : None or list(str)
            Specifies which electromobility attributes to store. By default this is set
            to None, in which case all attributes are stored.
            See function docstring `attributes` parameter in
            :attr:`~.network.electromobility.Electromobility.to_csv` for more
            information.
        archive : bool, optional
            Save disk storage capacity by archiving the csv files. The
            archiving takes place after the generation of the CSVs and
            therefore temporarily the storage needs are higher. Default: False.
        archive_type : str, optional
            Set archive type. Default: "zip".
        drop_unarchived : bool, optional
            Drop the unarchived data if parameter archive is set to True.
            Default: True.

        """
        os.makedirs(directory, exist_ok=True)

        if save_topology:
            self.topology.to_csv(os.path.join(directory, "topology"))

        if save_timeseries:
            self.timeseries.to_csv(
                os.path.join(directory, "timeseries"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
            )

        if save_results:
            self.results.to_csv(
                os.path.join(directory, "results"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
                parameters=kwargs.get("parameters", None),
            )

        if save_electromobility:
            self.electromobility.to_csv(
                os.path.join(directory, "electromobility"),
                attributes=kwargs.get("electromobility_attributes", None),
            )

        # save configs
        self.config.to_json(directory)

        if save_heatpump:
            self.heat_pump.to_csv(
                os.path.join(directory, "heat_pump"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
            )

        if kwargs.get("archive", False):
            archive_type = kwargs.get("archive_type", "zip")
            shutil.make_archive(directory, archive_type, directory)

            dir_size = tools.get_directory_size(directory)
            zip_size = os.path.getsize(directory + ".zip")

            reduction = (1 - zip_size / dir_size) * 100

            drop_unarchived = kwargs.get("drop_unarchived", True)

            if drop_unarchived:
                shutil.rmtree(directory)

            logger.info(
                f"Archived files in a {archive_type} archive and reduced "
                f"storage needs by {reduction:.2f} %. The unarchived files"
                f" were dropped: {drop_unarchived}"
            )

    def save_edisgo_to_pickle(self, path="", filename=None):
        """
        Saves EDisGo object to pickle file.

        Parameters
        -----------
        path : str
            Directory the pickle file is saved to. Per default it takes the current
            working directory.
        filename : str or None
            Filename the pickle file is saved under. If None, filename is
            'edisgo_object_{grid_id}.pkl'.

        """
        abs_path = os.path.abspath(path)
        if filename is None:
            filename = f"edisgo_object_{self.topology.mv_grid.id}.pkl"
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

    def check_integrity(self):
        """
        Method to check the integrity of the EDisGo object.

        Checks for consistency of topology (see
        :func:`edisgo.topology.check_integrity`), timeseries (see
        :func:`edisgo.timeseries.check_integrity`) and the interplay of both.

        """
        self.topology.check_integrity()
        self.timeseries.check_integrity()

        # check consistency of topology and timeseries
        comp_types = ["generators", "loads", "storage_units"]

        for comp_type in comp_types:
            comps = getattr(self.topology, comp_type + "_df")

            for ts in ["active_power", "reactive_power"]:
                comp_ts_name = f"{comp_type}_{ts}"
                comp_ts = getattr(self.timeseries, comp_ts_name)

                # check whether all components in topology have an entry in the
                # respective active and reactive power timeseries
                missing = comps.index[~comps.index.isin(comp_ts.columns)]
                if len(missing) > 0:
                    logger.warning(
                        f"The following {comp_type} are missing in {comp_ts_name}: "
                        f"{missing.values}"
                    )

                # check whether all elements in timeseries have an entry in the topology
                missing_ts = comp_ts.columns[~comp_ts.columns.isin(comps.index)]
                if len(missing_ts) > 0:
                    logger.warning(
                        f"The following {comp_type} have entries in {comp_ts_name}, but"
                        f" not in {comp_type}_df: {missing_ts.values}"
                    )

            # check if the active powers inside the timeseries exceed the given nominal
            # or peak power of the component
            if comp_type in ["generators", "storage_units"]:
                attr = "p_nom"
            else:
                attr = "p_set"

            active_power = getattr(self.timeseries, f"{comp_type}_active_power")
            comps_complete = comps.index[comps.index.isin(active_power.columns)]
            exceeding = comps_complete[
                (active_power[comps_complete].max() > comps.loc[comps_complete, attr])
            ]

            if len(exceeding) > 0:
                logger.warning(
                    f"Values of active power in the timeseries object exceed {attr} for"
                    f" the following {comp_type}: {exceeding.values}"
                )

            logging.info("Integrity check finished. Please pay attention to warnings.")

    def resample_timeseries(self, method: str = "ffill", freq: str = "15min"):
        """
        Resamples all generator, load and storage time series to a desired resolution.

        The following time series are affected by this:

        * :attr:`~.network.timeseries.TimeSeries.generators_active_power`

        * :attr:`~.network.timeseries.TimeSeries.loads_active_power`

        * :attr:`~.network.timeseries.TimeSeries.storage_units_active_power`

        * :attr:`~.network.timeseries.TimeSeries.generators_reactive_power`

        * :attr:`~.network.timeseries.TimeSeries.loads_reactive_power`

        * :attr:`~.network.timeseries.TimeSeries.storage_units_reactive_power`

        Both up- and down-sampling methods are possible.

        Parameters
        ----------
        method : str, optional
            Method to choose from to fill missing values when resampling.
            Possible options are:

            * 'ffill'
                Propagate last valid observation forward to next valid
                observation. See :pandas:`pandas.DataFrame.ffill<DataFrame.ffill>`.
            * 'bfill'
                Use next valid observation to fill gap. See
                :pandas:`pandas.DataFrame.bfill<DataFrame.bfill>`.
            * 'interpolate'
                Fill NaN values using an interpolation method. See
                :pandas:`pandas.DataFrame.interpolate<DataFrame.interpolate>`.

            Default: 'ffill'.
        freq : str, optional
            Frequency that time series is resampled to. Offset aliases can be found
            here:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Default: '15min'.

        """
        self.timeseries.resample_timeseries(method=method, freq=freq)


def import_edisgo_from_pickle(filename, path=""):
    """
    Restores EDisGo object from pickle file.

    Parameters
    -----------
    filename : str
        Filename the pickle file is saved under.
    path : str
        Directory the pickle file is restored from. Per default it takes the current
        working directory.

    """
    abs_path = os.path.abspath(path)
    return pickle.load(open(os.path.join(abs_path, filename), "rb"))


def import_edisgo_from_files(
    edisgo_path,
    import_topology=True,
    import_timeseries=False,
    import_results=False,
    import_electromobility=False,
    import_heat_pump=False,
    from_zip_archive=False,
    **kwargs,
):
    """
    Sets up EDisGo object from csv files.

    This is the reverse function of :func:`~.edisgo.EDisGo.save` and if not specified
    differently assumes all data in the default sub-directories created in the
    :func:`~.edisgo.EDisGo.save` function.

    Parameters
    -----------
    edisgo_path : str
        Main directory to restore EDisGo object from. This directory must contain the
        config files. Further, if not specified differently,
        it is assumed to be the main directory containing sub-directories with
        e.g. topology data. In case `from_zip_archive` is set to True, `edisgo_path`
        is the name of the archive.
    import_topology : bool
        Indicates whether to import :class:`~.network.topology.Topology` object.
        Per default it is set to True, in which case topology data is imported.
        The default directory topology data is imported from is the sub-directory
        'topology'. A different directory can be specified through keyword argument
        `topology_directory`.
        Default: True.
    import_timeseries : bool
        Indicates whether to import :class:`~.network.timeseries.Timeseries` object.
        Per default it is set to False, in which case timeseries data is not imported.
        The default directory time series data is imported from is the sub-directory
        'timeseries'. A different directory can be specified through keyword argument
        `timeseries_directory`.
        Default: False.
    import_results : bool
        Indicates whether to import :class:`~.network.results.Results` object.
        Per default it is set to False, in which case results data is not imported.
        The default directory results data is imported from is the sub-directory
        'results'. A different directory can be specified through keyword argument
        `results_directory`.
        Default: False.
    import_electromobility : bool
        Indicates whether to import :class:`~.network.electromobility.Electromobility`
        object. Per default it is set to False, in which case electromobility data is
        not imported.
        The default directory electromobility data is imported from is the sub-directory
        'electromobility'. A different directory can be specified through keyword
        argument `electromobility_directory`.
        Default: False.
    import_heat_pump : bool
        Indicates whether to import :class:`~.network.heat.HeatPump` object.
        Per default it is set to False, in which case heat pump data containing
        information on COP, heat demand time series, etc. is not imported.
        The default directory heat pump data is imported from is the sub-directory
        'heat_pump'. A different directory can be specified through keyword
        argument `heat_pump_directory`.
        Default: False.
    from_zip_archive : bool
        Set to True if data needs to be imported from an archive, e.g. a zip
        archive. Default: False.

    Other Parameters
    -----------------
    topology_directory : str
        Indicates directory :class:`~.network.topology.Topology` object is imported
        from. Per default topology data is imported from `edisgo_path` sub-directory
        'topology'.
    timeseries_directory : str
        Indicates directory :class:`~.network.timeseries.Timeseries` object is imported
        from. Per default time series data is imported from `edisgo_path` sub-directory
        'timeseries'.
    results_directory : str
        Indicates directory :class:`~.network.results.Results` object is imported
        from. Per default results data is imported from `edisgo_path` sub-directory
        'results'.
    electromobility_directory : str
        Indicates directory :class:`~.network.electromobility.Electromobility` object is
        imported from. Per default electromobility data is imported from `edisgo_path`
        sub-directory 'electromobility'.
    heat_pump_directory : str
        Indicates directory :class:`~.network.heat.HeatPump` object is
        imported from. Per default heat pump data is imported from `edisgo_path`
        sub-directory 'heat_pump'.
    dtype : str
        Numerical data type for time series and results data to be imported,
        e.g. "float32". Per default this is None in which case data type is inferred.
    parameters : None or dict
        Specifies which results to restore. By default this is set to None,
        in which case all available results are restored.
        To only restore certain results provide a dictionary. See function docstring
        `parameters` parameter in :func:`~.network.results.Results.to_csv`
        for more information.

    Results
    ---------
    :class:`~.EDisGo`
        Restored EDisGo object.

    """

    if not from_zip_archive and str(edisgo_path).endswith(".zip"):
        from_zip_archive = True
        logging.info("Given path is a zip archive. Setting 'from_zip_archive' to True.")

    edisgo_obj = EDisGo()
    try:
        edisgo_obj.config = {
            "from_json": True,
            "config_path": edisgo_path,
            "from_zip_archive": from_zip_archive,
        }
    except FileNotFoundError:
        logging.info(
            "Configuration data could not be loaded from json wherefore "
            "the default configuration data is loaded."
        )
    except Exception:
        raise Exception

    if from_zip_archive:
        directory = edisgo_path

    if import_topology:
        if not from_zip_archive:
            directory = kwargs.get(
                "topology_directory", os.path.join(edisgo_path, "topology")
            )

        if os.path.exists(directory):
            edisgo_obj.topology.from_csv(directory, edisgo_obj, from_zip_archive)
        else:
            logging.warning("No topology data found. Topology not imported.")

    if import_timeseries:
        dtype = kwargs.get("dtype", None)

        if not from_zip_archive:
            directory = kwargs.get(
                "timeseries_directory", os.path.join(edisgo_path, "timeseries")
            )

        if os.path.exists(directory):
            edisgo_obj.timeseries.from_csv(
                directory, dtype=dtype, from_zip_archive=from_zip_archive
            )
        else:
            logging.warning("No time series data found. Timeseries not imported.")

    if import_results:
        parameters = kwargs.get("parameters", None)
        dtype = kwargs.get("dtype", None)

        if not from_zip_archive:
            directory = kwargs.get(
                "results_directory", os.path.join(edisgo_path, "results")
            )

        if os.path.exists(directory):
            edisgo_obj.results.from_csv(
                directory, parameters, dtype=dtype, from_zip_archive=from_zip_archive
            )
        else:
            logging.warning("No results data found. Results not imported.")

    if import_electromobility:
        if not from_zip_archive:
            directory = kwargs.get(
                "electromobility_directory",
                os.path.join(edisgo_path, "electromobility"),
            )

        if os.path.exists(directory):
            edisgo_obj.electromobility.from_csv(
                directory, edisgo_obj, from_zip_archive=from_zip_archive
            )
        else:
            logging.warning(
                "No electromobility data found. Electromobility not imported."
            )

    if import_heat_pump:
        if not from_zip_archive:
            directory = kwargs.get(
                "heat_pump_directory",
                os.path.join(edisgo_path, "heat_pump"),
            )

        if os.path.exists(directory):
            edisgo_obj.heat_pump.from_csv(directory, from_zip_archive=from_zip_archive)
        else:
            logging.warning("No heat pump data found. Heat pump data not imported.")

    return edisgo_obj
