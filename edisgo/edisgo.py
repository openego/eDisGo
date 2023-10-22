from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import shutil

from numbers import Number
from pathlib import PurePath

import numpy as np
import pandas as pd

from sqlalchemy.engine.base import Engine

from edisgo.flex_opt.charging_strategies import charging_strategy
from edisgo.flex_opt.check_tech_constraints import lines_relative_load
from edisgo.flex_opt.heat_pump_operation import (
    operating_strategy as hp_operating_strategy,
)
from edisgo.flex_opt.reinforce_grid import (
    catch_convergence_reinforce_grid,
    reinforce_grid,
)
from edisgo.io import (
    dsm_import,
    generators_import,
    powermodels_io,
    pypsa_io,
    timeseries_import,
)
from edisgo.io.ding0_import import import_ding0_grid
from edisgo.io.electromobility_import import (
    distribute_charging_demand,
    import_electromobility_from_dir,
    import_electromobility_from_oedb,
    integrate_charging_parks,
)
from edisgo.io.heat_pump_import import oedb as import_heat_pumps_oedb
from edisgo.io.storage_import import home_batteries_oedb
from edisgo.network import timeseries
from edisgo.network.dsm import DSM
from edisgo.network.electromobility import Electromobility
from edisgo.network.heat import HeatPump
from edisgo.network.overlying_grid import OverlyingGrid
from edisgo.network.results import Results
from edisgo.network.topology import Topology
from edisgo.opf import powermodels_opf
from edisgo.opf.results.opf_result_class import OPFResults
from edisgo.tools import plots, tools
from edisgo.tools.config import Config
from edisgo.tools.geo import find_nearest_bus
from edisgo.tools.spatial_complexity_reduction import spatial_complexity_reduction
from edisgo.tools.tools import determine_grid_integration_voltage_level

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
    legacy_ding0_grids : bool
        Allow import of old ding0 grids. Default: True.

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
    overlying_grid : :class:`~.network.overlying_grid.OverlyingGrid`
        This is a container holding data from the overlying grid such as curtailment
        requirements or power plant dispatch.
    dsm : :class:`~.network.dsm.DSM`
        This is a container holding data on demand side management potential.

    """

    def __init__(self, **kwargs):
        # load configuration
        self._config = Config(**kwargs)

        # instantiate topology object and load grid data
        self.topology = Topology(config=self.config)
        self.import_ding0_grid(
            path=kwargs.get("ding0_grid", None),
            legacy_ding0_grids=kwargs.get("legacy_ding0_grids", True),
        )
        self.legacy_grids = kwargs.get("legacy_ding0_grids", True)

        # instantiate other data classes
        self.results = Results(self)
        self.opf_results = OPFResults()
        self.timeseries = timeseries.TimeSeries(
            timeindex=kwargs.get("timeindex", pd.DatetimeIndex([]))
        )
        self.electromobility = Electromobility(edisgo_obj=self)
        self.heat_pump = HeatPump()
        self.dsm = DSM()
        self.overlying_grid = OverlyingGrid()

        # import new generators
        if kwargs.get("generator_scenario", None) is not None:
            self.import_generators(
                generator_scenario=kwargs.pop("generator_scenario"), **kwargs
            )

        # add MVGrid id to logging messages of logger "edisgo"
        log_grid_id = kwargs.get("log_grid_id", True)
        if log_grid_id:

            def add_grid_id_filter(record):
                record.grid_id = self.topology.id
                return True

            logger_edisgo = logging.getLogger("edisgo")
            for handler in logger_edisgo.handlers:
                fmt = handler.formatter._fmt
                colon_idx = fmt.index(":")
                formatter_str = (
                    f"{fmt[:colon_idx]} - MVGrid(%(grid_id)s){fmt[colon_idx:]}"
                )
                formatter = logging.Formatter(formatter_str)
                handler.setFormatter(formatter)
                handler.addFilter(add_grid_id_filter)

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

    def import_ding0_grid(self, path, legacy_ding0_grids=True):
        """
        Import ding0 topology data from csv files in the format as
        `Ding0 <https://github.com/openego/ding0>`_ provides it.

        Parameters
        -----------
        path : str
            Path to directory containing csv files of network to be loaded.
        legacy_ding0_grids : bool
            Allow import of old ding0 grids. Default: True.

        """
        if path is not None:
            import_ding0_grid(path, self, legacy_ding0_grids)

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

        See :attr:`~.network.timeseries.TimeSeries.set_worst_case` for more information.

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
        **kwargs,
    ):
        """
        Uses predefined feed-in or demand profiles to set active power time series.

        Predefined profiles comprise i.e. standard electric conventional load profiles
        for different sectors generated using the oemof
        `demandlib <https://github.com/oemof/demandlib/>`_ or feed-in time series of
        fluctuating solar and wind generators provided on the OpenEnergy DataBase.
        This function can also be used to provide your own profiles per technology or
        load sector.

        The active power time series are written to
        :attr:`~.network.timeseries.TimeSeries.generators_active_power` or
        :attr:`~.network.timeseries.TimeSeries.loads_active_power`.
        As data in :class:`~.network.timeseries.TimeSeries` is indexed by
        :attr:`~.network.timeseries.TimeSeries.timeindex` it is better to set
        :attr:`~.network.timeseries.TimeSeries.timeindex` before calling this function.
        You can set the time index upon initialisation of the EDisGo object by
        providing the input parameter 'timeindex' or using the function
        :attr:`~.edisgo.EDisGo.set_timeindex`.
        Also make sure that the time steps of self-provided time series include
        the set time index.

        Parameters
        -----------
        fluctuating_generators_ts : str or :pandas:`pandas.DataFrame<DataFrame>` or None
            Defines option to set technology-specific or technology- and weather cell
            specific active power time series of wind and solar generators.
            Possible options are:

            * 'oedb'

                Technology- and weather cell-specific hourly feed-in time series are
                obtained from the
                `OpenEnergy DataBase
                <https://openenergy-platform.org/dataedit/schemas>`_. See
                :func:`edisgo.io.timeseries_import.feedin_oedb` for more information.

                This option requires that the parameter `engine` is provided in case
                new ding0 grids with geo-referenced LV grids are used. For further
                settings, the parameter `timeindex` can also be provided.

            * :pandas:`pandas.DataFrame<DataFrame>`

                DataFrame with self-provided feed-in time series per technology or
                per technology and weather cell ID normalized to a nominal capacity
                of 1.
                In case time series are provided only by technology, columns of the
                DataFrame contain the technology type as string.
                In case time series are provided by technology and weather cell ID
                columns need to be a :pandas:`pandas.MultiIndex<MultiIndex>` with the
                first level containing the technology as string and the second level
                the weather cell ID as integer.
                Index needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

                When importing a ding0 grid and/or using predefined scenarios
                of the future generator park,
                each generator has an assigned weather cell ID that identifies the
                weather data cell from the weather data set used in the research
                project `open_eGo <https://openegoproject.wordpress.com/>`_ to
                determine feed-in profiles. The weather cell ID can be retrieved
                from column `weather_cell_id` in
                :attr:`~.network.topology.Topology.generators_df` and could be
                overwritten to use own weather cells.

            * None

                If None, time series are not set.

            Default: None.

        fluctuating_generators_names : list(str) or None
            Defines for which fluctuating generators to apply technology-specific time
            series. See parameter `generator_names` in
            :attr:`~.network.timeseries.TimeSeries.predefined_fluctuating_generators_by_technology`
            for more information. Default: None.
        dispatchable_generators_ts : :pandas:`pandas.DataFrame<DataFrame>` or None
            Defines which technology-specific time series to use to set active power
            time series of dispatchable generators.
            See parameter `ts_generators` in
            :attr:`~.network.timeseries.TimeSeries.predefined_dispatchable_generators_by_technology`
            for more information. If None, no time series of dispatchable generators
            are set. Default: None.
        dispatchable_generators_names : list(str) or None
            Defines for which dispatchable generators to apply technology-specific time
            series. See parameter `generator_names` in
            :attr:`~.network.timeseries.TimeSeries.predefined_dispatchable_generators_by_technology`
            for more information. Default: None.
        conventional_loads_ts : str or :pandas:`pandas.DataFrame<DataFrame>` or None
            Defines option to set active power time series of conventional loads.
            Possible options are:

            * 'oedb'

                Sets active power demand time series using individual hourly electricity
                load time series for one year obtained from the `OpenEnergy DataBase
                <https://openenergy-platform.org/dataedit/schemas>`_.

                This option requires that the parameters `engine` and `scenario` are
                provided. For further settings, the parameter `timeindex` can also be
                provided.

            * 'demandlib'

                Sets active power demand time series using hourly electricity load time
                series obtained using standard electric load profiles from
                the oemof `demandlib <https://github.com/oemof/demandlib/>`_.
                The demandlib provides sector-specific time series for the sectors
                'residential', 'cts', 'industrial', and 'agricultural'.

                For further settings, the parameter `timeindex` can also be provided.

            * :pandas:`pandas.DataFrame<DataFrame>`

                Sets active power demand time series using sector-specific demand
                time series provided in this DataFrame.
                The load time series per sector need to be normalized to an annual
                consumption of 1. Index needs to
                be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
                Columns need to contain the sector as string.
                In the current grid existing load types can be retrieved from column
                `sector` in :attr:`~.network.topology.Topology.loads_df` (make sure to
                select `type` 'conventional_load').
                In ding0 grids the differentiated sectors are 'residential', 'cts',
                and 'industrial'.

            * None

                If None, conventional load time series are not set.

            Default: None.
        conventional_loads_names : list(str) or None
            Defines for which conventional loads to set time series. In case
            `conventional_loads_ts` is 'oedb' see parameter `load_names` in
            :func:`edisgo.io.timeseries_import.electricity_demand_oedb` for more
            information. For other cases see parameter `load_names` in
            :attr:`~.network.timeseries.TimeSeries.predefined_conventional_loads_by_sector`
            for more information. Default: None.
        charging_points_ts : :pandas:`pandas.DataFrame<DataFrame>` or None
            Defines which use-case-specific time series to use to set active power
            time series of charging points.
            See parameter `ts_loads` in
            :attr:`~.network.timeseries.TimeSeries.predefined_charging_points_by_use_case`
            for more information. If None, no time series of charging points
            are set. Default: None.
        charging_points_names : list(str) or None
            Defines for which charging points to apply use-case-specific time
            series. See parameter `load_names` in
            :attr:`~.network.timeseries.TimeSeries.predefined_charging_points_by_use_case`
            for more information. Default: None.

        Other Parameters
        ------------------
        engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine. This parameter is only required in case
            `conventional_loads_ts` or `fluctuating_generators_ts` is 'oedb'.
        scenario : str
            Scenario for which to retrieve demand data. Possible options are 'eGon2035'
            and 'eGon100RE'. This parameter is only required in case
            `conventional_loads_ts` is 'oedb'.
        timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
            This parameter can optionally be provided in case `conventional_loads_ts`
            is 'oedb' or 'demandlib' and in case `fluctuating_generators_ts` is
            'oedb'. It is used to specify time steps for which to set active power data.
            Leap years can currently not be handled when data is retrieved from the
            oedb. In case the given timeindex contains a leap year, the data will
            be indexed using a default year and set for the whole year.
            If no timeindex is provided, the timeindex set in
            :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
            If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
            is indexed using a default year and set for the whole year.

        """
        if self.timeseries.timeindex.empty:
            logger.warning(
                "When setting time series using predefined profiles it is better to "
                "set a time index as all data in TimeSeries class is indexed by the"
                "time index. You can set the time index upon initialisation of "
                "the EDisGo object by providing the input parameter 'timeindex' or by "
                "using the function EDisGo.set_timeindex()."
            )
        if fluctuating_generators_ts is not None:
            self.timeseries.predefined_fluctuating_generators_by_technology(
                self,
                fluctuating_generators_ts,
                fluctuating_generators_names,
                engine=kwargs.get("engine"),
                timeindex=kwargs.get("timeindex", None),
            )
        if dispatchable_generators_ts is not None:
            self.timeseries.predefined_dispatchable_generators_by_technology(
                self, dispatchable_generators_ts, dispatchable_generators_names
            )
        if conventional_loads_ts is not None:
            if (
                isinstance(conventional_loads_ts, str)
                and conventional_loads_ts == "oedb"
            ):
                loads_ts_df = timeseries_import.electricity_demand_oedb(
                    edisgo_obj=self,
                    scenario=kwargs.get("scenario"),
                    engine=kwargs.get("engine"),
                    timeindex=kwargs.get("timeindex", None),
                    load_names=conventional_loads_names,
                )
                # concat new time series with existing ones and drop any duplicate
                # entries
                self.timeseries.loads_active_power = tools.drop_duplicated_columns(
                    pd.concat([self.timeseries.loads_active_power, loads_ts_df], axis=1)
                )
            else:
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
            Type of reactive power control to apply. Currently, the only option is
            'fixed_coshpi'. See :attr:`~.network.timeseries.TimeSeries.fixed_cosphi`
            for further information.
        generators_parametrisation : str or :pandas:`pandas.DataFrame<DataFrame>`
            See parameter `generators_parametrisation` in
            :attr:`~.network.timeseries.TimeSeries.fixed_cosphi` for further
            information. Here, per default, the option 'default' is used.
        loads_parametrisation : str or :pandas:`pandas.DataFrame<DataFrame>`
            See parameter `loads_parametrisation` in
            :attr:`~.network.timeseries.TimeSeries.fixed_cosphi` for further
            information. Here, per default, the option 'default' is used.
        storage_units_parametrisation : str or :pandas:`pandas.DataFrame<DataFrame>`
            See parameter `storage_units_parametrisation` in
            :attr:`~.network.timeseries.TimeSeries.fixed_cosphi` for further
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
        are per default used for all generators and storage units anyway.

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
            more information. Default: False.

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
            to export in case mode is 'lv'. Default: None.
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

    def to_powermodels(
        self,
        s_base=1,
        flexible_cps=None,
        flexible_hps=None,
        flexible_loads=None,
        flexible_storage_units=None,
        opf_version=1,
    ):
        """
        Convert eDisGo representation of the network topology and timeseries to
        PowerModels network data format.

        Parameters
        ----------
        s_base : int
            Base value of apparent power for per unit system.
            Default: 1 MVA
        flexible_cps : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all charging points that allow for flexible charging.
        flexible_hps : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all heat pumps that allow for flexible operation due to an
            attached heat storage.
        flexible_loads : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all flexible loads that allow for application of demand
            side management strategy.
        flexible_storage_units : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all flexible storages. Non-flexible storages operate to
            optimize self consumption.
            Default: None.
        opf_version : int
            Version of optimization models to choose from. Must be one of [1, 2, 3, 4].
            For more information see :func:`edisgo.opf.powermodels_opf.pm_optimize`.
            Default: 1.

        Returns
        -------
        dict
            Dictionary that contains all network data in PowerModels network data
            format.

        """
        return powermodels_io.to_powermodels(
            self,
            s_base=s_base,
            flexible_cps=flexible_cps,
            flexible_hps=flexible_hps,
            flexible_loads=flexible_loads,
            flexible_storage_units=flexible_storage_units,
            opf_version=opf_version,
        )

    def pm_optimize(
        self,
        s_base=1,
        flexible_cps=None,
        flexible_hps=None,
        flexible_loads=None,
        flexible_storage_units=None,
        opf_version=1,
        method="soc",
        warm_start=False,
        silence_moi=False,
        save_heat_storage=True,
        save_slack_gen=True,
        save_slacks=True,
    ):
        """
        Run OPF in julia subprocess and write results of OPF back to edisgo object.

        Results of OPF are time series of operation schedules of flexibilities.

        Parameters
        ----------
        s_base : int
            Base value of apparent power for per unit system.
            Default: 1 MVA.
        flexible_cps : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all charging points that allow for flexible charging.
            Default: None.
        flexible_hps : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all heat pumps that allow for flexible operation due to an
            attached heat storage.
            Default: None.
        flexible_loads : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all flexible loads that allow for application of demand
            side management strategy.
            Default: None.
        flexible_storage_units: :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all flexible storages. Non-flexible storages operate to
            optimize self consumption.
            Default: None.
        opf_version : int
            Version of optimization models to choose from. Must be one of [1, 2, 3, 4].
            For more information see :func:`edisgo.opf.powermodels_opf.pm_optimize`.
            Default: 1.
        method : str
            Optimization method to use. Must be either "soc" (Second Order Cone) or "nc"
            (Non Convex). For more information see
            :func:`edisgo.opf.powermodels_opf.pm_optimize`.
            Default: "soc".
        warm_start : bool
            If set to True and if method is set to "soc", non-convex IPOPT OPF will be
            run additionally and will be warm started with Gurobi SOC solution.
            Warm-start will only be run if results for Gurobi's SOC relaxation is exact.
            Default: False.
        silence_moi : bool
            If set to True, MathOptInterface's optimizer attribute "MOI.Silent" is set
            to True in julia subprocess. This attribute is for silencing the output of
            an optimizer. When set to True, it requires the solver to produce no output,
            hence there will be no logging coming from julia subprocess in python
            process.
            Default: False.
        """
        return powermodels_opf.pm_optimize(
            self,
            s_base=s_base,
            flexible_cps=flexible_cps,
            flexible_hps=flexible_hps,
            flexible_loads=flexible_loads,
            flexible_storage_units=flexible_storage_units,
            opf_version=opf_version,
            method=method,
            warm_start=warm_start,
            silence_moi=silence_moi,
        )

    def to_graph(self):
        """
        Returns networkx graph representation of the grid.

        Returns
        -------
        :networkx:`networkx.Graph<networkx.Graph>`
            Graph representation of the grid as networkx Ordered Graph,
            where lines are represented by edges in the graph, and buses and
            transformers are represented by nodes.

        """

        return self.topology.to_graph()

    def import_generators(self, generator_scenario=None, **kwargs):
        """
        Gets generator park for specified scenario and integrates generators into grid.

        The generator data is retrieved from the
        `open energy platform <https://openenergy-platform.org/>`_. Decommissioned
        generators are removed from the grid, generators with changed capacity
        updated and new generators newly integrated into the grid.

        In case you are using new ding0 grids, where the LV is geo-referenced, the
        supported data source is scenario data generated in the research project
        `eGo^n <https://ego-n.org/>`_. You can choose between two scenarios:
        'eGon2035' and 'eGon100RE'. For more information on database tables used and
        how generator park is adapted see :func:`~.io.generators_import.oedb`.

        In case you are using old ding0 grids, where the LV is not geo-referenced,
        the supported data source is scenario data generated in the research project
        `open_eGo <https://openegoproject.wordpress.com/>`_. You can choose
        between two scenarios: 'nep2035' and 'ego100'. You can get more
        information on the scenarios in the
        `final report <https://www.uni-flensburg.de/fileadmin/content/\
        abteilungen/industrial/dokumente/downloads/veroeffentlichungen/\
        forschungsergebnisse/20190426endbericht-openego-fkz0325881-final\
        .pdf>`_. For more information on database tables used and
        how generator park is adapted see :func:`~.io.generators_import.oedb_legacy`.

        After the generator park is adapted there may be grid issues due to the
        additional feed-in. These are not solved automatically. If you want to
        have a stable grid without grid issues you can invoke the automatic
        grid expansion through the function :attr:`~.EDisGo.reinforce`.

        Parameters
        ----------
        generator_scenario : str
            Scenario for which to retrieve generator data. In case you are using new
            ding0 grids, where the LV is geo-referenced, possible options are
            'eGon2035' and 'eGon100RE'. In case you are using old ding0 grids, where
            the LV is not geo-referenced, possible options are 'nep2035' and 'ego100'.

        Other Parameters
        ----------------
        kwargs :
            In case you are using new ding0 grids, where the LV is geo-referenced, a
            database engine needs to be provided through keyword argument `engine`.
            In case you are using old ding0 grids, where the LV is not geo-referenced,
            you can check :func:`edisgo.io.generators_import.oedb_legacy` for possible
            keyword arguments.

        """
        if self.legacy_grids is True:
            generators_import.oedb_legacy(
                edisgo_object=self, generator_scenario=generator_scenario, **kwargs
            )
        else:
            generators_import.oedb(
                edisgo_object=self,
                engine=kwargs.get("engine"),
                scenario=generator_scenario,
            )

    def analyze(
        self,
        mode: str | None = None,
        timesteps: pd.Timestamp | pd.DatetimeIndex | None = None,
        raise_not_converged: bool = True,
        troubleshooting_mode: str | None = None,
        range_start: Number = 0.1,
        range_num: int = 10,
        scale_timeseries: float | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            Two optional troubleshooting methods in case of non-convergence of nonlinear
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
                Using parameters `range_start` and `range_num` you can define at what
                scaling factor the iteration should start and how many iterations
                should be conducted.

        range_start : float, optional
            Specifies the minimum fraction that power values are set to when using
            `troubleshooting_mode` 'iteration'. Must be between 0 and 1.
            Default: 0.1.
        range_num : int, optional
            Specifies the number of fraction samples to generate when using
            `troubleshooting_mode` 'iteration'. Must be non-negative.
            Default: 10.
        scale_timeseries : float or None, optional
            If a value is given, the timeseries in the pypsa object are scaled with
            this factor (values between 0 and 1 will scale down the time series and
            values above 1 will scale the timeseries up). Downscaling of time series
            can be used to check if power flow converges for smaller
            grid loads. If None, timeseries are not scaled. In case of
            `troubleshooting_mode` 'iteration' this parameter is ignored.
            Default: None.

        Other Parameters
        -----------------
        kwargs : dict
            Possible other parameters comprise all other parameters that can be set in
            :func:`edisgo.io.pypsa_io.to_pypsa`.

        Returns
        --------
        tuple(:pandas:`pandas.DatetimeIndex<DatetimeIndex>`,\
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>`)
            First index contains time steps for which power flow analysis did converge.
            Second index contains time steps for which power flow analysis did not
            converge.

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

        def _scale_timeseries(pypsa_network_copy, fraction):
            # Scales the timeseries in the pypsa object, the pypsa_network_copy is
            # the network with the original time series
            # Reduce power values of generators, loads and storages to given fraction
            for obj1, obj2 in [
                (pypsa_network.generators_t, pypsa_network_copy.generators_t),
                (pypsa_network.loads_t, pypsa_network_copy.loads_t),
                (pypsa_network.storage_units_t, pypsa_network_copy.storage_units_t),
            ]:
                for attr in ["p_set", "q_set"]:
                    setattr(obj1, attr, getattr(obj2, attr) * fraction)

            return pypsa_network

        if timesteps is None:
            timesteps = self.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]

        pypsa_network = self.to_pypsa(mode=mode, timesteps=timesteps, **kwargs)

        if scale_timeseries is not None and troubleshooting_mode != "iteration":
            pypsa_network = _scale_timeseries(pypsa_network, scale_timeseries)

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
                pypsa_network = _scale_timeseries(pypsa_network_copy, fraction)
                # run power flow analysis
                pf_results = pypsa_network.pf(timesteps, use_seed=True)
                logger.info(
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

        return timesteps_converged, timesteps_not_converged

    def reinforce(
        self,
        timesteps_pfa: str | pd.DatetimeIndex | pd.Timestamp | None = None,
        reduced_analysis: bool = False,
        copy_grid: bool = False,
        max_while_iterations: int = 20,
        split_voltage_band: bool = True,
        mode: str | None = None,
        without_generator_import: bool = False,
        n_minus_one: bool = False,
        catch_convergence_problems: bool = False,
        **kwargs,
    ) -> Results:
        """
        Reinforces the network and calculates network expansion costs.

        If the :attr:`edisgo.network.timeseries.TimeSeries.is_worst_case` is
        True input for `timesteps_pfa` is overwritten and therefore ignored.

        See :ref:`features-in-detail` for more information on how network
        reinforcement is conducted.

        Parameters
        -----------
        timesteps_pfa : str or \
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
            :pandas:`pandas.Timestamp<Timestamp>`
            timesteps_pfa specifies for which time steps power flow analysis is
            conducted and therefore which time steps to consider when checking
            for over-loading and over-voltage issues.
            It defaults to None in which case all timesteps in
            :attr:`~.network.timeseries.TimeSeries.timeindex` are used.
            Possible options are:

            * None
              Time steps in :attr:`~.network.timeseries.TimeSeries.timeindex` are used.
            * 'snapshot_analysis'
              Reinforcement is conducted for two worst-case snapshots. See
              :meth:`edisgo.tools.tools.select_worstcase_snapshots()` for further
              explanation on how worst-case snapshots are chosen.
              Note: If you have large time series, choosing this option will save
              calculation time since power flow analysis is only conducted for two
              time steps. If your time series already represents the worst-case,
              keep the default value of None because finding the worst-case
              snapshots takes some time.
            * :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
              :pandas:`pandas.Timestamp<Timestamp>`
              Use this option to explicitly choose which time steps to consider.
        reduced_analysis : bool
              If True, reinforcement is conducted for all time steps at which at least
              one branch shows its highest overloading or one bus shows its highest
              voltage violation. Time steps to consider are specified through parameter
              `timesteps_pfa`. If False, all time steps in parameter `timesteps_pfa`
              are used. Default: False.
        copy_grid : bool
            If True, reinforcement is conducted on a copied grid and discarded.
            Default: False.
        max_while_iterations : int
            Maximum number of times each while loop is conducted. Default: 20.
        split_voltage_band : bool
            If True the allowed voltage band of +/-10 percent is allocated to the
            different voltage levels MV, MV/LV and LV according to config values set
            in section `grid_expansion_allowed_voltage_deviations`. If False, the same
            voltage limits are used for all voltage levels. Be aware that this does
            currently not work correctly. Default: True.
        mode : str
            Determines network levels reinforcement is conducted for. Specify

            * None to reinforce MV and LV network levels. None is the default.
            * 'mv' to reinforce MV level only, neglecting MV/LV stations,
              and LV network topology. LV load and generation is aggregated per
              LV network and directly connected to the primary side of the
              respective MV/LV station.
            * 'mvlv' to reinforce MV network level only, including MV/LV stations,
              and neglecting LV network topology. LV load and generation is
              aggregated per LV network and directly connected to the secondary
              side of the respective MV/LV station.
            * 'lv' to reinforce LV networks. In case an LV grid is specified through
              parameter `lv_grid_id`, the grid's MV/LV station is not included. In case
              no LV grid ID is given, all MV/LV stations are included.
        without_generator_import : bool
            If True, excludes lines that were added in the generator import to connect
            new generators from calculation of network expansion costs. Default: False.
        n_minus_one : bool
            Determines whether n-1 security should be checked. Currently, n-1 security
            cannot be handled correctly, wherefore the case where this parameter is set
            to True will lead to an error being raised. Default: False.
        catch_convergence_problems : bool
            Uses reinforcement strategy to reinforce not converging grid.
            Reinforces first with only converging timesteps. Reinforce again with at
            start not converging timesteps. If still not converging, scale timeseries.
            Default: False

        Other Parameters
        -----------------
        is_worst_case : bool
            Is used to overwrite the return value from
            :attr:`edisgo.network.timeseries.TimeSeries.is_worst_case`. If True,
            reinforcement is calculated for worst-case MV and LV cases separately.
        lv_grid_id : str or int or None
            LV grid id to specify the grid to check, if mode is "lv". If no grid is
            specified, all LV grids are checked. In that case, the power flow analysis
            is conducted including the MV grid, in order to check loading and voltage
            drop/rise of MV/LV stations.
        skip_mv_reinforcement : bool
            If True, MV is not reinforced, even if `mode` is "mv", "mvlv" or None.
            This is used in case worst-case grid reinforcement is conducted in order to
            reinforce MV/LV stations for LV worst-cases.
            Default: False.
        num_steps_loading : int
            In case `timesteps_pfa` is set to 'reduced_analysis', this parameter can be
            used to specify the number of most critical overloading events to consider.
            If None, `percentage` is used. Default: None.
        num_steps_voltage : int
            In case `timesteps_pfa` is set to 'reduced_analysis', this parameter can be
            used to specify the number of most critical voltage issues to select. If
            None, `percentage` is used. Default: None.
        percentage : float
            In case `timesteps_pfa` is set to 'reduced_analysis', this parameter can be
            used to specify the percentage of most critical time steps to select. The
            default is 1.0, in which case all most critical time steps are selected.
            Default: 1.0.
        use_troubleshooting_mode : bool
            In case `timesteps_pfa` is set to 'reduced_analysis', this parameter can be
            used to specify how to handle non-convergence issues in the power flow
            analysis. If set to True, non-convergence issues are tried to be
            circumvented by reducing load and feed-in until the power flow converges.
            The most critical time steps are then determined based on the power flow
            results with the reduced load and feed-in. If False, an error will be
            raised in case time steps do not converge. Default: True.

        Returns
        --------
        :class:`~.network.results.Results`
            Returns the Results object holding network expansion costs, equipment
            changes, etc.

        """
        if copy_grid:
            edisgo_obj = copy.deepcopy(self)
        else:
            edisgo_obj = self

        # Build reinforce run settings
        if kwargs.get("is_worst_case", self.timeseries.is_worst_case):
            logger.debug(
                "Running reinforcement in worst-case mode by differentiating between "
                "MV and LV load and feed-in cases."
            )
            timeindex_worst_cases = self.timeseries.timeindex_worst_cases
            timesteps_mv = pd.DatetimeIndex(
                timeindex_worst_cases.loc[
                    timeindex_worst_cases.index.str.contains("mv")
                ]
            )
            timesteps_lv = pd.DatetimeIndex(
                timeindex_worst_cases.loc[
                    timeindex_worst_cases.index.str.contains("lv")
                ]
            )
            # Run the analyze-method at the end, to get a power flow for all
            # timesteps for reinforced components
            run_analyze_at_the_end = True
            if mode is None:
                kwargs_mv = kwargs.copy()
                kwargs_mv.update({"mode": "mv", "timesteps_pfa": timesteps_mv})
                kwargs_mvlv = kwargs.copy()
                kwargs_mvlv.update(
                    {
                        "mode": "mvlv",
                        "timesteps_pfa": timesteps_lv,
                        "skip_mv_reinforcement": True,
                    }
                )
                kwargs_lv = kwargs.copy()
                kwargs_lv.update({"mode": "lv", "timesteps_pfa": timesteps_lv})
                kwargs.update({"mode": "mv", "timesteps_pfa": timesteps_mv})
                setting_list = [
                    kwargs_mv,
                    kwargs_mvlv,
                    kwargs_lv,
                ]
            elif mode == "mv":
                kwargs.update({"mode": "mv", "timesteps_pfa": timesteps_mv})
                setting_list = [kwargs]
            elif mode == "mvlv":
                kwargs.update(
                    {
                        "mode": "mvlv",
                        "timesteps_pfa": timesteps_lv,
                        "skip_mv_reinforcement": True,
                    }
                )
                setting_list = [kwargs]
            elif mode == "lv":
                kwargs.update({"mode": "lv", "timesteps_pfa": timesteps_lv})
                setting_list = [kwargs]
            else:
                raise ValueError(f"Mode {mode} does not exist.")
        else:
            kwargs.update({"mode": mode, "timesteps_pfa": timesteps_pfa})
            setting_list = [kwargs]
            run_analyze_at_the_end = False

        logger.info(f"Run the following reinforcements: {setting_list=}")

        for setting in setting_list:
            logger.info(f"Run the following reinforcement: {setting=}")
            func = (
                catch_convergence_reinforce_grid
                if catch_convergence_problems
                else reinforce_grid
            )

            func(
                edisgo_obj,
                max_while_iterations=max_while_iterations,
                split_voltage_band=split_voltage_band,
                without_generator_import=without_generator_import,
                n_minus_one=n_minus_one,
                **setting,
            )

        if run_analyze_at_the_end:
            lv_grid_id = kwargs.get("lv_grid_id", None)

            if mode == "lv" and lv_grid_id:
                analyze_mode = "lv"
            elif mode == "lv":
                analyze_mode = None
            else:
                analyze_mode = mode

            edisgo_obj.analyze(
                mode=analyze_mode, lv_grid_id=lv_grid_id, timesteps=timesteps_pfa
            )

        # add measure to Results object
        if not copy_grid:
            self.results.measures = "grid_expansion"

        return edisgo_obj.results

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
        ts_active_power : :pandas:`pandas.Series<Series>` or None
            Active power time series of added component.
            Index of the series must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are active power per time step in MW.
            Defaults to None in which case no time series is set.
        ts_reactive_power : :pandas:`pandas.Series<Series>` or str or None
            Possible options are:

            * :pandas:`pandas.Series<Series>`

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
        **kwargs : dict
            Attributes of added component. See respective functions for required
            entries.

            * 'bus' : :attr:`~.network.topology.Topology.add_bus`

            * 'line' : :attr:`~.network.topology.Topology.add_line`

            * 'load' : :attr:`~.network.topology.Topology.add_load`

            * 'generator' : :attr:`~.network.topology.Topology.add_generator`

            * 'storage_unit' : :attr:`~.network.topology.Topology.add_storage_unit`

        Returns
        --------
        str
            The identifier of the newly integrated component as in index of
            :attr:`~.network.topology.Topology.generators_df`,
            :attr:`~.network.topology.Topology.loads_df`, etc., depending on component
            type.

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
                        logger.warning(
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

        Currently, components can be generators, charging points, heat pumps and
        storage units.

        See :attr:`~.network.topology.Topology.connect_to_mv`,
        :attr:`~.network.topology.Topology.connect_to_lv` and
        :attr:`~.network.topology.Topology.connect_to_lv_based_on_geolocation` for more
        information.

        Parameters
        ----------
        comp_type : str
            Type of added component. Can be 'generator', 'charging_point', 'heat_pump'
            or 'storage_unit'.
        geolocation : :shapely:`shapely.Point<Point>` or tuple
            Geolocation of the new component. In case of tuple, the geolocation
            must be given in the form (longitude, latitude).
        voltage_level : int, optional
            Specifies the voltage level the new component is integrated in.
            Possible options are 4 (MV busbar), 5 (MV grid), 6 (LV busbar) or
            7 (LV grid). If no voltage level is provided the voltage level
            is determined based on the nominal power `p_nom` or `p_set` (given as
            kwarg). For this, upper limits up to which capacity a component is
            integrated into a certain voltage level (set in the config section
            `grid_connection` through the parameters 'upper_limit_voltage_level_{4:7}')
            are used.
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
            See :attr:`~.network.topology.Topology.add_generator`,
            :attr:`~.network.topology.Topology.add_storage_unit` respectively
            :attr:`~.network.topology.Topology.add_load` methods
            for more information on required and optional parameters.

        Returns
        -------
        str
            The identifier of the newly integrated component as in index of
            :attr:`~.network.topology.Topology.generators_df`,
            :attr:`~.network.topology.Topology.loads_df` or
            :attr:`~.network.topology.Topology.storage_units_df`, depending on component
            type.

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
            # determine voltage level manually from nominal power
            voltage_level = determine_grid_integration_voltage_level(self, p)

        # check if geolocation is given as shapely Point, otherwise transform
        # to shapely Point
        if type(geolocation) is not Point:
            geolocation = Point(geolocation)

        # write voltage level and geolocation to kwargs
        kwargs["geom"] = geolocation
        kwargs["voltage_level"] = voltage_level

        # Connect in MV
        if voltage_level in [4, 5]:
            comp_name = self.topology.connect_to_mv(self, kwargs, comp_type)

        # Connect in LV
        else:
            # check if LV is geo-referenced or not
            lv_buses = self.topology.buses_df.drop(self.topology.mv_grid.buses_df.index)
            lv_buses_dropna = lv_buses.dropna(axis=0, subset=["x", "y"])

            # if there are some LV buses without geo-reference, use function where
            # components are not integrated based on geolocation
            if len(lv_buses_dropna) < len(lv_buses):
                if kwargs.get("mvlv_subst_id", None) is None:
                    substations = self.topology.buses_df.loc[
                        self.topology.transformers_df.bus1.unique()
                    ]
                    nearest_substation, _ = find_nearest_bus(geolocation, substations)
                    kwargs["mvlv_subst_id"] = int(nearest_substation.split("_")[-2])
                comp_name = self.topology.connect_to_lv(self, kwargs, comp_type)

            else:
                max_distance_from_target_bus = kwargs.pop(
                    "max_distance_from_target_bus", 0.02
                )
                comp_name = self.topology.connect_to_lv_based_on_geolocation(
                    self, kwargs, comp_type, max_distance_from_target_bus
                )

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
        data_source: str,
        scenario: str = None,
        engine: Engine = None,
        charging_processes_dir: PurePath | str = None,
        potential_charging_points_dir: PurePath | str = None,
        import_electromobility_data_kwds=None,
        allocate_charging_demand_kwds=None,
    ):
        """
        Imports electromobility data and integrates charging points into grid.

        Electromobility data can be obtained from the `OpenEnergy DataBase
        <https://openenergy-platform.org/dataedit/schemas>`_ or from self-provided
        data. In case you want to use self-provided data, it needs to be generated
        using the tools
        `SimBEV <https://github.com/rl-institut/simbev>`_ (required version:
        `3083c5a <https://github.com/rl-institut/simbev/commit/
        86076c936940365587c9fba98a5b774e13083c5a>`_) and
        `TracBEV <https://github.com/rl-institut/tracbev>`_ (required version:
        `14d864c <https://github.com/rl-institut/tracbev/commit/
        03e335655770a377166c05293a966052314d864c>`_). SimBEV provides data on standing
        times, charging demand, etc. per vehicle, whereas TracBEV provides potential
        charging point locations.

        After electromobility data is loaded, the charging demand from is allocated to
        potential charging points. Afterwards, all potential charging points with
        charging demand allocated to them are integrated into the grid.

        Be aware that this function does not yield charging time series per charging
        point but only charging processes (see
        :attr:`~.network.electromobility.Electromobility.charging_processes_df` for
        more information). The actual charging time series are determined through
        applying a charging strategy using the function
        :attr:`~.edisgo.EDisGo.charging_strategy`.

        Parameters
        ----------
        data_source : str
            Specifies source from where to obtain electromobility data.
            Possible options are:

            * "oedb"

                Electromobility data is obtained from the `OpenEnergy DataBase
                <https://openenergy-platform.org/dataedit/schemas>`_.

                This option requires that the parameters `scenario` and `engine` are
                provided.

            * "directory"

                Electromobility data is obtained from directories specified through
                parameters `charging_processes_dir` and `potential_charging_points_dir`.

        scenario : str
            Scenario for which to retrieve electromobility data in case `data_source` is
            set to "oedb". Possible options are "eGon2035" and "eGon100RE".
        engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine. Needs to be provided in case `data_source` is set to
            "oedb".
        charging_processes_dir : str or pathlib.PurePath
            Directory holding data on charging processes (standing times, charging
            demand, etc. per vehicle), including metadata, from SimBEV.
        potential_charging_points_dir : str or pathlib.PurePath
            Directory holding data on potential charging point locations from TracBEV.
        import_electromobility_data_kwds : dict
            These may contain any further attributes you want to specify when importing
            electromobility data.

            gc_to_car_rate_home : float
                Specifies the minimum rate between potential charging points for the
                use case "home" and the total number of cars. Default: 0.5.
            gc_to_car_rate_work : float
                Specifies the minimum rate between potential charging points for the
                use case "work" and the total number of cars. Default: 0.25.
            gc_to_car_rate_public : float
                Specifies the minimum rate between potential charging points for the
                use case "public" and the total number of cars. Default: 0.1.
            gc_to_car_rate_hpc : float
                Specifies the minimum rate between potential charging points for the
                use case "hpc" and the total number of cars. Default: 0.005.
            mode_parking_times : str
                If the mode_parking_times is set to "frugal" only parking times
                with any charging demand are imported. Any other input will lead
                to all parking and driving events being imported. Default: "frugal".
            charging_processes_dir : str
                Charging processes sub-directory. Only used when `data_source` is
                set to "directory". Default: None.
            simbev_config_file : str
                Name of the simbev config file. Only used when `data_source` is
                set to "directory". Default: "metadata_simbev_run.json".

        allocate_charging_demand_kwds :
            These may contain any further attributes you want to specify when calling
            the function :func:`~.io.electromobility_import.distribute_charging_demand`
            that allocates charging processes to potential charging points.

            mode : str
                Distribution mode. If the mode is set to "user_friendly" only the
                simbev weights are used for the distribution. If the mode is
                "grid_friendly" also grid conditions are respected.
                Default: "user_friendly".
            generators_weight_factor : float
                Weighting factor of the generators weight within an LV grid in
                comparison to the loads weight. Default: 0.5.
            distance_weight : float
                Weighting factor for the distance between a potential charging park
                and its nearest substation in comparison to the combination of
                the generators and load factors of the LV grids. Default: 1 / 3.
            user_friendly_weight : float
                Weighting factor of the user-friendly weight in comparison to the
                grid friendly weight. Default: 0.5.

        """
        if import_electromobility_data_kwds is None:
            import_electromobility_data_kwds = {}

        if data_source == "oedb":
            import_electromobility_from_oedb(
                self,
                scenario=scenario,
                engine=engine,
                **import_electromobility_data_kwds,
            )
        elif data_source == "directory":
            import_electromobility_from_dir(
                self,
                charging_processes_dir,
                potential_charging_points_dir,
                **import_electromobility_data_kwds,
            )
        else:
            raise ValueError(
                "Invalid input for parameter 'data_source'. Possible options are "
                "'oedb' and 'directory'."
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

        Notes
        ------
        If the frequency of time series data in :class:`~.network.timeseries.TimeSeries`
        (checked using :attr:`~.network.timeseries.TimeSeries.timeindex`) differs from
        the frequency of SimBEV data, then the time series in
        :class:`~.network.timeseries.TimeSeries` is first automatically resampled to
        match the SimBEV data frequency and after determining the charging demand time
        series resampled back to the original frequency.

        """
        charging_strategy(self, strategy=strategy, **kwargs)

    def import_heat_pumps(self, scenario, engine, timeindex=None, import_types=None):
        """
        Gets heat pump data for specified scenario from oedb and integrates the heat
        pumps into the grid.

        Besides heat pump capacity the heat pump's COP and heat demand to be served
        are as well retrieved.

        Currently, the only supported data source is scenario data generated
        in the research project `eGo^n <https://ego-n.org/>`_. You can choose
        between two scenarios: 'eGon2035' and 'eGon100RE'.

        The data is retrieved from the
        `open energy platform <https://openenergy-platform.org/>`_.

        # ToDo Add information on scenarios and from which tables data is retrieved.

        The following steps are conducted in this function:

            * Heat pump capacities for individual and district heating per building
              respectively district heating area are obtained from the database for the
              specified scenario and integrated into the grid using the function
              :func:`~.io.heat_pump_import.oedb`.
            * Heat pumps are integrated into the grid (added to
              :attr:`~.network.topology.Topology.loads_df`) as follows.

              * Grid connection points of heat pumps for individual heating are
                determined based on the corresponding building ID. In case the heat
                pump is too large to use the same grid connection point, they are
                connected via their own grid connection point.
              * Grid connection points of heat pumps for district heating are determined
                based on their geolocation and installed capacity.
                See :attr:`~.network.topology.Topology.connect_to_mv` and
                :attr:`~.network.topology.Topology.connect_to_lv_based_on_geolocation`
                for more information.
            * COP and heat demand for each heat pump are retrieved from the database,
              using the functions :func:`~.io.timeseries_import.cop_oedb` respectively
              :func:`~.io.timeseries_import.heat_demand_oedb`, and stored in the
              :class:`~.network.heat.HeatPump` class that can be accessed through
              :attr:`~.edisgo.EDisGo.heat_pump`.

        Be aware that this function does not yield electricity load time series for the
        heat pumps. The actual time series are determined through applying an
        operation strategy or optimising heat pump dispatch. Further, the heat pumps
        do not yet have a thermal storage and can therefore not yet be used as a
        flexibility. Thermal storage units need to be added manually to
        :attr:`~.edisgo.EDisGo.thermal_storage_units_df`.

        After the heat pumps are integrated there may be grid issues due to the
        additional load. These are not solved automatically. If you want to
        have a stable grid without grid issues you can invoke the automatic
        grid expansion through the function :attr:`~.EDisGo.reinforce`.

        Parameters
        ----------
        scenario : str
            Scenario for which to retrieve heat pump data. Possible options
            are 'eGon2035' and 'eGon100RE'.
        engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine.
        timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
            Specifies time steps for which to set COP and heat demand data. Leap years
            can currently not be handled. In case the given
            timeindex contains a leap year, the data will be indexed using the default
            year (2035 in case of the 'eGon2035' and to 2045 in case of the
            'eGon100RE' scenario) and returned for the whole year.
            If no timeindex is provided, the timeindex set in
            :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
            If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
            is indexed using the default year and returned for the whole year.
        import_types : list(str) or None
            Specifies which technologies to import. Possible options are
            "individual_heat_pumps", "central_heat_pumps" and
            "central_resistive_heaters". If None, all are imported.

        """
        # set up year to index data by
        # first try to get index from time index
        if timeindex is None:
            timeindex = self.timeseries.timeindex
            # if time index is not set get year from scenario
            if timeindex.empty:
                year = tools.get_year_based_on_scenario(scenario)
                # if year is still None, scenario is not valid
                if year is None:
                    raise ValueError(
                        "Invalid input for parameter 'scenario'. Possible options are "
                        "'eGon2035' and 'eGon100RE'."
                    )
                timeindex = pd.date_range(f"1/1/{year}", periods=8760, freq="H")
        # if year is leap year set year according to scenario
        if pd.Timestamp(timeindex.year[0], 1, 1).is_leap_year:
            logger.warning(
                "A leap year was given to 'heat_demand_oedb' function. This is "
                "currently not valid. The year the data is indexed by is therefore set "
                "according to the given scenario."
            )
            year = tools.get_year_based_on_scenario(scenario)
            return self.import_heat_pumps(
                scenario,
                engine,
                timeindex=pd.date_range(f"1/1/{year}", periods=8760, freq="H"),
                import_types=import_types,
            )

        integrated_heat_pumps = import_heat_pumps_oedb(
            edisgo_object=self,
            scenario=scenario,
            engine=engine,
            import_types=import_types,
        )
        if len(integrated_heat_pumps) > 0:
            self.heat_pump.set_heat_demand(
                self,
                "oedb",
                heat_pump_names=integrated_heat_pumps,
                engine=engine,
                scenario=scenario,
                timeindex=timeindex,
            )
            self.heat_pump.set_cop(
                self,
                "oedb",
                heat_pump_names=integrated_heat_pumps,
                engine=engine,
                timeindex=timeindex,
            )

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

    def import_dsm(self, scenario: str, engine: Engine, timeindex=None):
        """
        Gets industrial and CTS DSM profiles from the
        `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

        Profiles comprise minimum and maximum load increase in MW as well as maximum
        energy pre- and postponing in MWh. The data is written to the
        :class:`~.network.dsm.DSM` object.

        Currently, the only supported data source is scenario data generated
        in the research project `eGo^n <https://ego-n.org/>`_. You can choose
        between two scenarios: 'eGon2035' and 'eGon100RE'.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        scenario : str
            Scenario for which to retrieve DSM data. Possible options
            are 'eGon2035' and 'eGon100RE'.
        engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine.
        timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
            Specifies time steps for which to get data. Leap years can currently not be
            handled. In case the given timeindex contains a leap year, the data will be
            indexed using the default year (2035 in case of the 'eGon2035' and to 2045
            in case of the 'eGon100RE' scenario) and returned for the whole year.
            If no timeindex is provided, the timeindex set in
            :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
            If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
            is indexed using the default year and returned for the whole year.

        """
        dsm_profiles = dsm_import.oedb(
            edisgo_obj=self, scenario=scenario, engine=engine, timeindex=timeindex
        )
        self.dsm.p_min = dsm_profiles["p_min"]
        self.dsm.p_max = dsm_profiles["p_max"]
        self.dsm.e_min = dsm_profiles["e_min"]
        self.dsm.e_max = dsm_profiles["e_max"]

    def import_home_batteries(
        self,
        scenario: str,
        engine: Engine,
    ):
        """
        Gets home battery data for specified scenario and integrates the batteries into
        the grid.

        Currently, the only supported data source is scenario data generated
        in the research project `eGo^n <https://ego-n.org/>`_. You can choose
        between two scenarios: 'eGon2035' and 'eGon100RE'.

        The data is retrieved from the
        `open energy platform <https://openenergy-platform.org/>`_.

        The batteries are integrated into the grid (added to
        :attr:`~.network.topology.Topology.storage_units_df`) based on their building
        ID. In case the battery is too large to use the same grid connection point as
        the generator or, if no generator is allocated at the same building ID, the
        load, they are connected via their own grid connection point, based on their
        geolocation and installed capacity.

        Be aware that this function does not yield time series for the batteries. The
        actual time series can be determined through a dispatch optimisation.

        Parameters
        ----------
        scenario : str
            Scenario for which to retrieve home battery data. Possible options
            are 'eGon2035' and 'eGon100RE'.
        engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine.

        """
        home_batteries_oedb(
            edisgo_obj=self,
            scenario=scenario,
            engine=engine,
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
                logger.warning(
                    "Voltages from power flow "
                    "analysis must be available to plot them."
                )
                return
        except AttributeError:
            logger.warning(
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
                logger.warning(
                    "Currents `i_res` from power flow analysis "
                    "must be available to plot line loading."
                )
                return
        except AttributeError:
            logger.warning(
                "Results must be available to plot line loading. "
                "Please analyze grid first."
            )
            return

        plots.mv_grid_topology(
            self,
            timestep=kwargs.get("timestep", None),
            line_color="loading",
            node_color=kwargs.get("node_color", None),
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
                logger.warning(
                    "Grid expansion cost results needed to plot "
                    "them. Please do grid reinforcement."
                )
                return
        except AttributeError:
            logger.warning(
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
        :func:`edisgo.tools.tools.calculate_relative_line_load`.
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

        rel_line_loading = lines_relative_load(self, lines.index)

        if timestep is None:
            timestep = rel_line_loading.index
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timestep, "__len__"):
            timestep = [timestep]
        rel_line_loading = rel_line_loading.loc[timestep, :]

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
        save_opf_results=False,
        save_heatpump=False,
        save_overlying_grid=False,
        save_dsm=False,
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
            Per default, it is saved to sub-directory 'topology'. See
            :attr:`~.network.topology.Topology.to_csv` for more information.
            Default: True.
        save_timeseries : bool, optional
            Indicates whether to save :class:`~.network.timeseries.TimeSeries` object.
            Per default it is saved to subdirectory 'timeseries'.
            Through the keyword arguments `reduce_memory`
            and `to_type` it can be chosen if memory should be reduced. See
            :attr:`~.network.timeseries.TimeSeries.to_csv` for more
            information.
            Default: True.
        save_results : bool, optional
            Indicates whether to save :class:`~.network.results.Results`
            object. Per default, it is saved to subdirectory 'results'.
            Through the keyword argument `parameters` the results that should
            be stored can be specified. Further, through the keyword parameters
            `reduce_memory` and `to_type` it can be chosen if memory should be reduced.
            See :attr:`~.network.results.Results.to_csv` for more information.
            Default: True.
        save_electromobility : bool, optional
            Indicates whether to save
            :class:`~.network.electromobility.Electromobility` object. Per default, it
            is not saved. If set to True, it is saved to subdirectory 'electromobility'.
            See :attr:`~.network.electromobility.Electromobility.to_csv` for more
            information.
        save_opf_results : bool, optional
            Indicates whether to save
            :class:`~.opf.results.opf_result_class.OPFResults` object. Per default, it
            is not saved. If set to True, it is saved to subdirectory 'opf_results'.
            See :attr:`~.opf.results.opf_result_class.OPFResults.to_csv` for more
            information.
        save_heatpump : bool, optional
            Indicates whether to save
            :class:`~.network.heat.HeatPump` object. Per default, it is not saved.
            If set to True, it is saved to subdirectory 'heat_pump'.
            See :attr:`~.network.heat.HeatPump.to_csv` for more information.
        save_overlying_grid : bool, optional
            Indicates whether to save
            :class:`~.network.overlying_grid.OverlyingGrid` object. Per default, it is
            not saved. If set to True, it is saved to subdirectory 'overlying_grid'.
            See :attr:`~.network.overlying_grid.OverlyingGrid.to_csv` for more
            information.
        save_dsm : bool, optional
            Indicates whether to save :class:`~.network.dsm.DSM` object. Per default,
            it is not saved. If set to True, it is saved to subdirectory 'dsm'. See
            :attr:`~.network.dsm.DSM.to_csv` for more information.

        Other Parameters
        ------------------
        reduce_memory : bool, optional
            If True, size of dataframes containing time series in
            :class:`~.network.results.Results`,
            :class:`~.network.timeseries.TimeSeries`,
            :class:`~.network.heat.HeatPump`,
            :class:`~.network.overlying_grid.OverlyingGrid` and
            :class:`~.network.dsm.DSM`
            is reduced. See respective classes `reduce_memory` functions for more
            information. Type to convert to can be specified by providing
            `to_type` as keyword argument. Further parameters of reduce_memory
            functions cannot be passed here. Call these functions directly to
            make use of further options. Default: False.
        to_type : str, optional
            Data type to convert time series data to. This is a trade-off
            between precision and memory. Default: "float32".
        parameters : None or dict
            Specifies which results to store. By default, this is set to None,
            in which case all available results are stored.
            To only store certain results provide a dictionary. See function docstring
            `parameters` parameter in :attr:`~.network.results.Results.to_csv`
            for more information.
        electromobility_attributes : None or list(str)
            Specifies which electromobility attributes to store. By default, this is set
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

        if save_opf_results:
            self.opf_results.to_csv(
                os.path.join(directory, "opf_results"),
                attributes=kwargs.get("opf_results_attributes", None),
            )

        # save configs
        self.config.to_json(directory)

        if save_heatpump:
            self.heat_pump.to_csv(
                os.path.join(directory, "heat_pump"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
            )

        if save_dsm:
            self.dsm.to_csv(
                os.path.join(directory, "dsm"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
            )

        if save_overlying_grid:
            self.overlying_grid.to_csv(
                os.path.join(directory, "overlying_grid"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
            )

        if kwargs.get("archive", False):
            archive_type = kwargs.get("archive_type", "zip")
            shutil.make_archive(directory, archive_type, directory)

            dir_size = tools.get_directory_size(directory)
            zip_size = os.path.getsize(str(directory) + ".zip")

            reduction = (1 - zip_size / dir_size) * 100

            drop_unarchived = kwargs.get("drop_unarchived", True)

            if drop_unarchived:
                shutil.rmtree(directory)

            logger.info(
                f"Archived files in a {archive_type} archive and reduced "
                f"storage needs by {reduction:.2f} %. The unarchived files "
                f"were dropped: {drop_unarchived}"
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

    def save_edisgo_to_json(
        self,
        filename=None,
        path="",
        s_base=1,
        flexible_cps=None,
        flexible_hps=None,
        flexible_loads=None,
        flexible_storage_units=None,
        opf_version=1,
    ):
        """
        Saves EDisGo object in PowerModels network data format to json file.

        Parameters
        -----------
        filename : str or None
            Filename the json file is saved under. If None, filename is
            'ding0_{grid_id}_t_{#timesteps}.json'.
        path : str
            Directory the json file is saved to. Per default, it takes the current
            working directory.
        s_base : int
            Base value of apparent power for per unit system.
            Default: 1 MVA
        flexible_cps : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all charging points that allow for flexible charging.
        flexible_hps : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all heat pumps that allow for flexible operation due to an
            attached heat storage.
        flexible_loads : :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all flexible loads that allow for application of demand
            side management strategy.
        flexible_storage_units: :numpy:`numpy.ndarray<ndarray>` or None
            Array containing all flexible storages. Non-flexible storages operate to
            optimize self consumption.
            Default: None
        opf_version: Int
            Version of optimization models to choose from. Must be one of [1, 2, 3, 4].
            For more information see :func:`edisgo.opf.powermodels_opf.pm_optimize`.
            Default: 1.

        Returns
        -------
        dict
            Dictionary that contains all network data in PowerModels network data
            format.

        """
        abs_path = os.path.abspath(path)
        pm, hv_flex_dict = self.to_powermodels(
            s_base=s_base,
            flexible_cps=flexible_cps,
            flexible_hps=flexible_hps,
            flexible_loads=flexible_loads,
            flexible_storage_units=flexible_storage_units,
            opf_version=opf_version,
        )

        def _convert(o):
            """
            Helper function for json dump, as int64 cannot be dumped.

            """
            if isinstance(o, np.int64):
                return int(o)
            raise TypeError

        if filename is None:
            filename = "{}.json".format(pm["name"])

        with open(
            os.path.join(abs_path, filename),
            "w",
        ) as outfile:
            json.dump(pm, outfile, default=_convert)
        return pm

    def reduce_memory(self, **kwargs):
        """
        Reduces size of time series data to save memory.

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
            :attr:`~.network.results.Results.reduce_memory` for more information.
        timeseries_attr_to_reduce : list(str), optional
            See `attr_to_reduce` parameter in
            :attr:`~.network.timeseries.TimeSeries.reduce_memory` for more information.
        heat_pump_attr_to_reduce : list(str), optional
            See `attr_to_reduce` parameter in
            :attr:`~.network.heat.HeatPump.reduce_memory` for more information.
        overlying_grid_attr_to_reduce : list(str), optional
            See `attr_to_reduce` parameter in
            :attr:`~.network.overlying_grid.OverlyingGrid.reduce_memory` for more
            information.
        dsm_attr_to_reduce : list(str), optional
            See `attr_to_reduce` parameter in
            :attr:`~.network.overlying_grid.OverlyingGrid.reduce_memory` for more
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
        # heat pump data
        self.heat_pump.reduce_memory(
            to_type=kwargs.get("to_type", "float32"),
            attr_to_reduce=kwargs.get("heat_pump_attr_to_reduce", None),
        )
        # overlying grid
        self.overlying_grid.reduce_memory(
            to_type=kwargs.get("to_type", "float32"),
            attr_to_reduce=kwargs.get("overlying_grid_attr_to_reduce", None),
        )

    def spatial_complexity_reduction(
        self,
        copy_edisgo: bool = False,
        mode: str = "kmeansdijkstra",
        cluster_area: str = "feeder",
        reduction_factor: float = 0.25,
        reduction_factor_not_focused: bool | float = False,
        apply_pseudo_coordinates: bool = True,
        **kwargs,
    ) -> tuple[EDisGo, pd.DataFrame, pd.DataFrame]:
        """
        Reduces the number of busses and lines by applying a spatial clustering.

        Per default, this function creates pseudo coordinates for all busses in the LV
        grids (see function :func:`~.tools.pseudo_coordinates.make_pseudo_coordinates`).
        In case LV grids are not geo-referenced, this is a necessary step. If they are
        already geo-referenced it can still be useful to obtain better results.

        Which busses are clustered is determined in function
        :func:`~.tools.spatial_complexity_reduction.make_busmap`.
        The clustering method used can be specified through the parameter `mode`.
        Further, the clustering can be applied to different areas such as the whole grid
        or the separate feeders, which is specified through the parameter
        `cluster_area`, and to different degrees, specified through the parameter
        `reduction_factor`.

        The actual spatial reduction of the EDisGo object is conducted in function
        :func:`~.tools.spatial_complexity_reduction.apply_busmap`. The changes, such as
        dropping of lines connecting the same buses and adapting buses loads, generators
        and storage units are connected to, are applied directly in the Topology object.
        If you want to keep information on the original grid, hand a copy of the EDisGo
        object to this function. You can also set how loads and generators at clustered
        busses are aggregated through the keyword arguments
        `load_aggregation_mode` and `generator_aggregation_mode`.

        Parameters
        ----------
        copy_edisgo : bool
            Defines whether to apply the spatial complexity reduction directly on the
            EDisGo object or on a copy. Per default, the complexity reduction is
            directly applied.
        mode : str
            Clustering method to use. Possible options are "kmeans", "kmeansdijkstra",
            "aggregate_to_main_feeder" or "equidistant_nodes". The clustering methods
            "aggregate_to_main_feeder" and "equidistant_nodes" only work for the cluster
            area "main_feeder".

            - "kmeans":
                Perform the k-means algorithm on the cluster area and then map the buses
                to the cluster centers.
            - "kmeansdijkstra":
                Perform the k-means algorithm and then map the nodes to the cluster
                centers through the shortest distance in the graph. The distances are
                calculated using the dijkstra algorithm.
            - "aggregate_to_main_feeder":
                Aggregate the nodes in the feeder to the longest path in the feeder,
                here called main feeder.
            - "equidistant_nodes":
                Uses the method "aggregate_to_main_feeder" and then reduces the nodes
                again through a reduction of the nodes by the specified reduction factor
                and distributing the remaining nodes on the graph equidistantly.

            Default: "kmeansdijkstra".
        cluster_area : str
            The cluster area is the area the different clustering methods are applied
            to. Possible options are 'grid', 'feeder' or 'main_feeder'.
            Default: "feeder".
        reduction_factor : float
            Factor to reduce number of nodes by. Must be between 0 and 1. Default: 0.25.
        reduction_factor_not_focused : bool or float
            If False, uses the same reduction factor for all cluster areas. If between 0
            and 1, this sets the reduction factor for buses not of interest (these are
            buses without voltage or overloading issues, that are determined through a
            worst case power flow analysis). When selecting 0, the nodes of the
            clustering area are aggregated to the transformer bus. This parameter is
            only used when parameter `cluster_area` is set to 'feeder' or 'main_feeder'.
            Default: False.
        apply_pseudo_coordinates : bool
            If True pseudo coordinates are applied. The spatial complexity reduction
            method is only tested with pseudo coordinates. Default: True.

        Other Parameters
        -----------------
        line_naming_convention : str
            Determines how to set "type_info" and "kind" in case two or more lines are
            aggregated. Possible options are "standard_lines" or "combined_name".
            If "standard_lines" is selected, the values of the standard line of the
            respective voltage level are used to set "type_info" and "kind".
            If "combined_name" is selected, "type_info" and "kind" contain the
            concatenated values of the merged lines. x and r of the lines are not
            influenced by this as they are always determined from the x and r values of
            the aggregated lines.
            Default: "standard_lines".
        aggregation_mode : bool
            Specifies, whether to aggregate loads and generators at the same bus or not.
            If True, loads and generators at the same bus are aggregated
            according to their selected modes (see parameters `load_aggregation_mode`
            and `generator_aggregation_mode`). Default: False.
        load_aggregation_mode : str
            Specifies, how to aggregate loads at the same bus, in case parameter
            `aggregation_mode` is set to True. Possible options are "bus" or "sector".
            If "bus" is chosen, loads are aggregated per bus. When "sector" is chosen,
            loads are aggregated by bus, type and sector. Default: "sector".
        generator_aggregation_mode : str
            Specifies, how to aggregate generators at the same bus, in case parameter
            `aggregation_mode` is set to True. Possible options are "bus" or "type".
            If "bus" is chosen, generators are aggregated per bus. When "type" is
            chosen, generators are aggregated by bus and type.
        mv_pseudo_coordinates : bool, optional
            If True pseudo coordinates are also generated for MV grid.
            Default: False.

        Returns
        -------
        tuple(:class:`~.EDisGo`, :pandas:`pandas.DataFrame<DataFrame>`,\
            :pandas:`pandas.DataFrame<DataFrame>`)
            Returns the EDisGo object (which is only relevant in case the parameter
            `copy_edisgo` was set to True), as well as the busmap and linemap
            dataframes.
            The busmap maps the original busses to the new busses with new coordinates.
            Columns are "new_bus" with new bus name, "new_x" with new x-coordinate and
            "new_y" with new y-coordinate. Index of the dataframe holds bus names of
            original buses as in buses_df.
            The linemap maps the original line names (in the index of the dataframe) to
            the new line names (in column "new_line_name").

        """
        if copy_edisgo is True:
            edisgo_obj = copy.deepcopy(self)
        else:
            edisgo_obj = self
        busmap_df, linemap_df = spatial_complexity_reduction(
            edisgo_obj=edisgo_obj,
            mode=mode,
            cluster_area=cluster_area,
            reduction_factor=reduction_factor,
            reduction_factor_not_focused=reduction_factor_not_focused,
            apply_pseudo_coordinates=apply_pseudo_coordinates,
            **kwargs,
        )
        return edisgo_obj, busmap_df, linemap_df

    def check_integrity(self):
        """
        Method to check the integrity of the EDisGo object.

        Checks for consistency of topology (see
        :func:`edisgo.network.topology.Topology.check_integrity`), timeseries (see
        :func:`edisgo.network.timeseries.TimeSeries.check_integrity`) and the interplay
        of both.
        Further, checks integrity of electromobility object (see
        :func:`edisgo.network.electromobility.Electromobility.check_integrity`),
        the heat pump object (see :func:`edisgo.network.heat.HeatPump.check_integrity`)
        and the DSM object (see :func:`edisgo.network.dsm.DSM.check_integrity`).
        Additionally, checks whether time series data in
        :class:`~.network.heat.HeatPump`,
        :class:`~.network.electromobility.Electromobility`,
        :class:`~.network.overlying_grid.OverlyingGrid` and :class:`~.network.dsm.DSM`
        contains all time steps in
        :attr:`edisgo.network.timeseries.TimeSeries.timeindex`.

        """
        self.topology.check_integrity()
        self.timeseries.check_integrity()
        self.electromobility.check_integrity()
        self.dsm.check_integrity()
        self.heat_pump.check_integrity()

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

            if len(exceeding) > 1e-6:
                logger.warning(
                    f"Values of active power in the timeseries object exceed {attr} for"
                    f" the following {comp_type}: {exceeding.values}"
                )

        def _check_timeindex(check_df, param_name):
            if not check_df.empty:
                missing_indices = [
                    _ for _ in self.timeseries.timeindex if _ not in check_df.index
                ]
                if len(missing_indices) > 0:
                    logger.warning(
                        f"There are time steps in timeindex of TimeSeries object that "
                        f"are not in the index of {param_name}. This may lead "
                        f"to problems."
                    )

        # check if time index of other time series data contains all time steps
        # in TimeSeries.timeindex
        if len(self.timeseries.timeindex) > 0:
            # check time index of electromobility flexibility bands
            flex_band = list(self.electromobility.flexibility_bands.values())[0]
            _check_timeindex(flex_band, "Electromobility.flexibility_bands")
            # check time index of HeatPump data
            for param_name in self.heat_pump._timeseries_attributes:
                _check_timeindex(
                    getattr(self.heat_pump, param_name), f"HeatPump.{param_name}"
                )
            # check time index of OverlyingGrid data
            for param_name in self.overlying_grid._attributes:
                _check_timeindex(
                    getattr(self.overlying_grid, param_name),
                    f"OverlyingGrid.{param_name}",
                )
            # check time index of DSM data
            for param_name in self.dsm._attributes:
                _check_timeindex(
                    getattr(self.dsm, param_name),
                    f"DSM.{param_name}",
                )

        # check if heat demand can be met by corresponding heatpump at all times.
        if len(self.heat_pump.cop_df.columns) > len(
            self.heat_pump.heat_demand_df.columns
        ):
            # If there are heat pumps with heat demand but no COP time series, or the
            # other way around, a warning is raised in HeatPump.check_integrity
            pass
        else:
            hp_cop = self.heat_pump.cop_df
            hp_p_nom = self.topology.loads_df.loc[
                self.heat_pump.heat_demand_df.columns.values
            ][["p_set"]]
            heat_demand = self.heat_pump.heat_demand_df
            comparison = (
                heat_demand[hp_p_nom.index] > hp_cop * hp_p_nom.squeeze()
            ).any()
            if comparison.any():
                logger.warning(
                    "Heat demand is higher than rated heatpump power"
                    " of heatpumps: {}. Demand can not be covered if no sufficient"
                    " heat storage capacities are available.".format(
                        comparison.index[comparison.values].values
                    )
                )

        logging.info("Integrity check finished. Please pay attention to warnings.")

    def resample_timeseries(
        self, method: str = "ffill", freq: str | pd.Timedelta = "15min"
    ):
        """
        Resamples time series data in
        :class:`~.network.timeseries.TimeSeries`, :class:`~.network.heat.HeatPump`,
        :class:`~.network.electromobility.Electromobility` and
        :class:`~.network.overlying_grid.OverlyingGrid`.

        Both up- and down-sampling methods are possible.

        The following time series are affected by this:

        * :attr:`~.network.timeseries.TimeSeries.generators_active_power`

        * :attr:`~.network.timeseries.TimeSeries.loads_active_power`

        * :attr:`~.network.timeseries.TimeSeries.storage_units_active_power`

        * :attr:`~.network.timeseries.TimeSeries.generators_reactive_power`

        * :attr:`~.network.timeseries.TimeSeries.loads_reactive_power`

        * :attr:`~.network.timeseries.TimeSeries.storage_units_reactive_power`

        * :attr:`~.network.electromobility.Electromobility.flexibility_bands`

        * :attr:`~.network.heat.HeatPump.cop_df`

        * :attr:`~.network.heat.HeatPump.heat_demand_df`

        * All data in :class:`~.network.overlying_grid.OverlyingGrid`

        Parameters
        ----------
        method : str, optional
            Method to choose from to fill missing values when resampling time series
            data (only exception from this is for flexibility bands in
            :class:`~.network.electromobility.Electromobility` object where method
            cannot be chosen to assure consistency of flexibility band data).
            Possible options are:

            * 'ffill' (default)
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
        self.timeseries.resample(method=method, freq=freq)
        self.electromobility.resample(freq=freq)
        self.heat_pump.resample_timeseries(method=method, freq=freq)
        self.overlying_grid.resample(method=method, freq=freq)


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
    edisgo_path: str | PurePath,
    import_topology: bool = True,
    import_timeseries: bool = False,
    import_results: bool = False,
    import_electromobility: bool = False,
    import_opf_results: bool = False,
    import_heat_pump: bool = False,
    import_dsm: bool = False,
    import_overlying_grid: bool = False,
    from_zip_archive: bool = False,
    **kwargs,
):
    """
    Sets up EDisGo object from csv files.

    This is the reverse function of :attr:`~.edisgo.EDisGo.save` and if not specified
    differently assumes all data in the default sub-directories created in the
    :attr:`~.edisgo.EDisGo.save` function.

    Parameters
    -----------
    edisgo_path : str or pathlib.PurePath
        Main directory to restore EDisGo object from. This directory must contain the
        config files. Further, if not specified differently,
        it is assumed to be the main directory containing sub-directories with
        e.g. topology data. In case `from_zip_archive` is set to True, `edisgo_path`
        is the name of the archive.
    import_topology : bool
        Indicates whether to import :class:`~.network.topology.Topology` object.
        Per default, it is set to True, in which case topology data is imported.
        The default directory topology data is imported from is the sub-directory
        'topology'. A different directory can be specified through keyword argument
        `topology_directory`.
        Default: True.
    import_timeseries : bool
        Indicates whether to import :class:`~.network.timeseries.TimeSeries` object.
        Per default, it is set to False, in which case timeseries data is not imported.
        The default directory time series data is imported from is the sub-directory
        'timeseries'. A different directory can be specified through keyword argument
        `timeseries_directory`.
        Default: False.
    import_results : bool
        Indicates whether to import :class:`~.network.results.Results` object.
        Per default, it is set to False, in which case results data is not imported.
        The default directory results data is imported from is the sub-directory
        'results'. A different directory can be specified through keyword argument
        `results_directory`.
        Default: False.
    import_electromobility : bool
        Indicates whether to import :class:`~.network.electromobility.Electromobility`
        object. Per default, it is set to False, in which case electromobility data is
        not imported.
        The default directory electromobility data is imported from is the sub-directory
        'electromobility'. A different directory can be specified through keyword
        argument `electromobility_directory`.
        Default: False.
    import_opf_results : bool
        Indicates whether to import :class:`~.opf.results.opf_result_class.OPFResults`
        object. Per default, it is set to False, in which case opf results data is not
        imported. The default directory results data is imported from is the
        sub-directory 'opf_results'. A different directory can be specified through
        keyword argument `opf_results_directory`.
        Default: False.
    import_heat_pump : bool
        Indicates whether to import :class:`~.network.heat.HeatPump` object.
        Per default, it is set to False, in which case heat pump data containing
        information on COP, heat demand time series, etc. is not imported.
        The default directory heat pump data is imported from is the sub-directory
        'heat_pump'. A different directory can be specified through keyword
        argument `heat_pump_directory`.
        Default: False.
    import_overlying_grid : bool
        Indicates whether to import :class:`~.network.overlying_grid.OverlyingGrid`
        object. Per default, it is set to False, in which case overlying grid data
        containing information on renewables curtailment requirements, generator
        dispatch, etc. is not imported.
        The default directory overlying grid data is imported from is the sub-directory
        'overlying_grid'. A different directory can be specified through keyword
        argument `overlying_grid_directory`.
        Default: False.
    import_dsm : bool
        Indicates whether to import :class:`~.network.dsm.DSM`
        object. Per default, it is set to False, in which case DSM data is not imported.
        The default directory DSM data is imported from is the sub-directory
        'dsm'. A different directory can be specified through keyword
        argument `dsm_directory`.
        Default: False.
    from_zip_archive : bool
        Set to True if data needs to be imported from an archive, e.g. a zip
        archive. Default: False.

    Other Parameters
    -----------------
    topology_directory : str
        Indicates directory :class:`~.network.topology.Topology` object is imported
        from. Per default, topology data is imported from `edisgo_path` sub-directory
        'topology'.
    timeseries_directory : str
        Indicates directory :class:`~.network.timeseries.TimeSeries` object is imported
        from. Per default, time series data is imported from `edisgo_path` sub-directory
        'timeseries'.
    results_directory : str
        Indicates directory :class:`~.network.results.Results` object is imported
        from. Per default, results data is imported from `edisgo_path` sub-directory
        'results'.
    electromobility_directory : str
        Indicates directory :class:`~.network.electromobility.Electromobility` object is
        imported from. Per default, electromobility data is imported from `edisgo_path`
        sub-directory 'electromobility'.
    opf_results_directory : str
        Indicates directory :class:`~.opf.results.opf_result_class.OPFResults` object is
        imported from. Per default, results data is imported from `edisgo_path`
        sub-directory 'opf_results'.
    heat_pump_directory : str
        Indicates directory :class:`~.network.heat.HeatPump` object is
        imported from. Per default, heat pump data is imported from `edisgo_path`
        sub-directory 'heat_pump'.
    overlying_grid_directory : str
        Indicates directory :class:`~.network.overlying_grid.OverlyingGrid` object is
        imported from. Per default, overlying grid data is imported from `edisgo_path`
        sub-directory 'overlying_grid'.
    dsm_directory : str
        Indicates directory :class:`~.network.dsm.DSM` object is imported from. Per
        default, DSM data is imported from `edisgo_path` sub-directory 'dsm'.
    dtype : str
        Numerical data type for time series and results data to be imported,
        e.g. "float32". Per default, this is None in which case data type is inferred.
    parameters : None or dict
        Specifies which results to restore. By default, this is set to None,
        in which case all available results are restored.
        To only restore certain results provide a dictionary. See function docstring
        `parameters` parameter in :func:`~.network.results.Results.to_csv`
        for more information.

    Returns
    ---------
    :class:`~.EDisGo`
        Restored EDisGo object.

    """

    if not os.path.exists(edisgo_path):
        raise ValueError("Given edisgo_path does not exist.")

    if not from_zip_archive and str(edisgo_path).endswith(".zip"):
        from_zip_archive = True
        logger.info("Given path is a zip archive. Setting 'from_zip_archive' to True.")

    edisgo_obj = EDisGo()
    try:
        edisgo_obj.config = {
            "from_json": True,
            "config_path": edisgo_path,
            "from_zip_archive": from_zip_archive,
        }
    except FileNotFoundError:
        logger.info(
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
            logger.warning("No topology data found. Topology not imported.")

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
            logger.warning("No time series data found. Timeseries not imported.")

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
            logger.warning("No results data found. Results not imported.")

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
            logger.warning(
                "No electromobility data found. Electromobility not imported."
            )

    if import_opf_results:
        if not from_zip_archive:
            directory = kwargs.get(
                "opf_results_directory", os.path.join(edisgo_path, "opf_results")
            )

        if os.path.exists(directory):
            edisgo_obj.opf_results.from_csv(
                directory, from_zip_archive=from_zip_archive
            )
        else:
            logger.warning("No opf results data found. OPF results not imported.")

    if import_heat_pump:
        if not from_zip_archive:
            directory = kwargs.get(
                "heat_pump_directory",
                os.path.join(edisgo_path, "heat_pump"),
            )

        if os.path.exists(directory):
            edisgo_obj.heat_pump.from_csv(directory, from_zip_archive=from_zip_archive)
        else:
            logger.warning("No heat pump data found. Heat pump data not imported.")

    if import_dsm:
        if not from_zip_archive:
            directory = kwargs.get(
                "dsm_directory",
                os.path.join(edisgo_path, "dsm"),
            )

        if os.path.exists(directory):
            edisgo_obj.dsm.from_csv(directory, from_zip_archive=from_zip_archive)
        else:
            logger.warning("No DSM data found. DSM data not imported.")

    if import_overlying_grid:
        if not from_zip_archive:
            directory = kwargs.get(
                "overlying_grid_directory",
                os.path.join(edisgo_path, "overlying_grid"),
            )

        if os.path.exists(directory):
            edisgo_obj.overlying_grid.from_csv(
                directory, from_zip_archive=from_zip_archive
            )
        else:
            logger.warning(
                "No overlying grid data found. Overlying grid data not imported."
            )

    return edisgo_obj
