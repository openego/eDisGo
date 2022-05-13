from __future__ import annotations

import itertools
import logging
import os

import numpy as np
import pandas as pd

from edisgo.flex_opt import q_control
from edisgo.io import timeseries_import
from edisgo.tools.tools import (
    assign_voltage_level_to_component,
    get_weather_cells_intersecting_with_grid_district,
)

logger = logging.getLogger(__name__)


class TimeSeries:
    """
    Holds component-specific active and reactive power time series.

    All time series are fixed time series that in case of flexibilities result after
    application of a heuristic or optimisation. They can be used for power flow
    calculations.

    Also holds any raw time series data that was used to generate component-specific
    time series in attribute `time_series_raw`. See
    :class:`~.network.timeseries.TimeSeriesRaw` for more information.

    Other Parameters
    -----------------
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`, optional
        Can be used to define a time range for which to obtain the provided
        time series and run power flow analysis. Default: None.

    Attributes
    -----------
    time_series_raw : :class:`~.network.timeseries.TimeSeriesRaw`
        Raw time series. See :class:`~.network.timeseries.TimeSeriesRaw` for  more
        information.

    """

    def __init__(self, **kwargs):

        self._timeindex = kwargs.get("timeindex", pd.DatetimeIndex([]))
        self.time_series_raw = TimeSeriesRaw()

    @property
    def is_worst_case(self) -> bool:
        """
        Time series mode.

        Is used to distinguish between normal time series analysis and worst-case
        analysis. Is determined by checking if the timindex starts before 1971 as the
        default for worst-case is 1970. Be mindful when creating your own worst-cases.

        Returns
        -------
        bool
            Indicates if current time series is worst-case time series with different
            assumptions for mv and lv simultaneities.
        """
        if len(self.timeindex) > 0:
            return self.timeindex[0] < pd.Timestamp("1971-01-01")
        return False

    @property
    def timeindex(self):
        """
        Time index all time-dependent attributes are indexed by.

        Is used as default time steps in e.g. power flow analysis.

        Parameters
        -----------
        ind : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
            Time index all time-dependent attributes are indexed by.

        Returns
        -------
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
            Time index all time-dependent attributes are indexed by.

        """
        return self._timeindex

    @timeindex.setter
    def timeindex(self, ind):
        if len(self._timeindex) > 0 and not ind.isin(self._timeindex).all():
            logger.warning(
                "Not all time steps of new time index lie within existing "
                "time index. This may cause problems later on."
            )
        self._timeindex = ind

    def _internal_getter(self, attribute):
        try:
            return getattr(self, f"_{attribute}").loc[self.timeindex, :]
        except AttributeError:
            return pd.DataFrame(index=self.timeindex)
        except KeyError:
            logger.warning(
                f"Timeindex and {attribute} have deviating indices. "
                "Empty dataframe will be returned."
            )
            return pd.DataFrame(index=self.timeindex)

    @property
    def generators_active_power(self):
        """
        Active power time series of generators in MW.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series of all generators in topology in MW. Index of the
            dataframe is a time index and column names are names of generators.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series of all generators in topology in MW for time steps
            given in :py:attr:`~timeindex`. For more information on the dataframe see
            input parameter `df`.

        """
        return self._internal_getter("generators_active_power")

    @generators_active_power.setter
    def generators_active_power(self, df):
        self._generators_active_power = df

    @property
    def generators_reactive_power(self):
        """
        Reactive power time series of generators in MVA.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series of all generators in topology in MVA. Index of
            the dataframe is a time index and column names are names of generators.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series of all generators in topology in MVA for time
            steps given in :py:attr:`~timeindex`. For more information on the dataframe
            see input parameter `df`.

        """
        return self._internal_getter("generators_reactive_power")

    @generators_reactive_power.setter
    def generators_reactive_power(self, df):
        self._generators_reactive_power = df

    @property
    def loads_active_power(self):
        """
        Active power time series of loads in MW.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series of all loads in topology in MW. Index of the
            dataframe is a time index and column names are names of loads.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series of all loads in topology in MW for time steps
            given in :py:attr:`~timeindex`. For more information on the dataframe see
            input parameter `df`.

        """
        return self._internal_getter("loads_active_power")

    @loads_active_power.setter
    def loads_active_power(self, df):
        self._loads_active_power = df

    @property
    def loads_reactive_power(self):
        """
        Reactive power time series of loads in MVA.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series of all loads in topology in MVA. Index of
            the dataframe is a time index and column names are names of loads.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series of all loads in topology in MVA for time
            steps given in :py:attr:`~timeindex`. For more information on the dataframe
            see input parameter `df`.

        """
        return self._internal_getter("loads_reactive_power")

    @loads_reactive_power.setter
    def loads_reactive_power(self, df):
        self._loads_reactive_power = df

    @property
    def storage_units_active_power(self):
        """
        Active power time series of storage units in MW.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series of all storage units in topology in MW. Index of
            the dataframe is a time index and column names are names of storage units.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series of all storage units in topology in MW for time
            steps given in :py:attr:`~timeindex`. For more information on the dataframe
            see input parameter `df`.

        """
        return self._internal_getter("storage_units_active_power")

    @storage_units_active_power.setter
    def storage_units_active_power(self, df):
        self._storage_units_active_power = df

    @property
    def storage_units_reactive_power(self):
        """
        Reactive power time series of storage units in MVA.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series of all storage units in topology in MVA. Index of
            the dataframe is a time index and column names are names of storage units.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series of all storage units in topology in MVA for time
            steps given in :py:attr:`~timeindex`. For more information on the dataframe
            see input parameter `df`.

        """
        return self._internal_getter("storage_units_reactive_power")

    @storage_units_reactive_power.setter
    def storage_units_reactive_power(self, df):
        self._storage_units_reactive_power = df

    def reset(self):
        """
        Resets all time series.

        Active and reactive power time series of all loads, generators and storage units
        are deleted, as well as everything stored in :py:attr:`~time_series_raw`.

        """
        self.generators_active_power = None
        self.loads_active_power = None
        self.storage_units_active_power = None
        self.time_series_raw = TimeSeriesRaw()

    def set_active_power_manual(
        self, edisgo_object, ts_generators=None, ts_loads=None, ts_storage_units=None
    ):
        """
        Sets given component active power time series.

        If time series for a component were already set before, they are overwritten.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_generators : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series in MW of generators. Index of the data frame is
            a datetime index. Columns contain generators names of generators to set
            time series for.
        ts_loads : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series in MW of loads. Index of the data frame is
            a datetime index. Columns contain load names of loads to set
            time series for.
        ts_storage_units : :pandas:`pandas.DataFrame<DataFrame>`
            Active power time series in MW of storage units. Index of the data frame is
            a datetime index. Columns contain storage unit names of storage units to set
            time series for.

        """
        self._set_manual(
            edisgo_object,
            "active",
            ts_generators=ts_generators,
            ts_loads=ts_loads,
            ts_storage_units=ts_storage_units,
        )

    def set_reactive_power_manual(
        self, edisgo_object, ts_generators=None, ts_loads=None, ts_storage_units=None
    ):
        """
        Sets given component reactive power time series.

        If time series for a component were already set before, they are overwritten.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_generators : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series in MVA of generators. Index of the data frame is
            a datetime index. Columns contain generators names of generators to set
            time series for.
        ts_loads : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series in MVA of loads. Index of the data frame is
            a datetime index. Columns contain load names of loads to set
            time series for.
        ts_storage_units : :pandas:`pandas.DataFrame<DataFrame>`
            Reactive power time series in MVA of storage units. Index of the data frame
            is a datetime index. Columns contain storage unit names of storage units to
            set time series for.

        """
        self._set_manual(
            edisgo_object,
            "reactive",
            ts_generators=ts_generators,
            ts_loads=ts_loads,
            ts_storage_units=ts_storage_units,
        )

    def _set_manual(
        self,
        edisgo_object,
        mode,
        ts_generators=None,
        ts_loads=None,
        ts_storage_units=None,
    ):
        """
        Sets given component time series.

        If time series for a component were already set before, they are overwritten.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        mode : str
            Defines whether to set active or reactive power time series. Possible
            options are "active" and "reactive".
        ts_generators : :pandas:`pandas.DataFrame<DataFrame>`
            Active or reactive power time series in MW or MVA of generators.
            Index of the data frame is a datetime index. Columns contain generator
            names of generators to set time series for.
        ts_loads : :pandas:`pandas.DataFrame<DataFrame>`
            Active or reactive power time series in MW or MVA of loads.
            Index of the data frame is a datetime index. Columns contain load names of
            loads to set time series for.
        ts_storage_units : :pandas:`pandas.DataFrame<DataFrame>`
            Active or reactive power time series in MW or MVA of storage units.
            Index of the data frame is a datetime index. Columns contain storage unit
            names of storage units to set time series for.

        """
        if ts_generators is not None:
            # check if all generators time series are provided for exist in the network
            # and only set time series for those that do
            comps_in_network = _check_if_components_exist(
                edisgo_object, ts_generators.columns, "generators"
            )
            ts_generators = ts_generators.loc[:, comps_in_network]

            # drop generators time series from self.generators_(re)active_power that may
            # already exist for some of the given generators
            df_name = f"generators_{mode}_power"
            drop_component_time_series(
                obj=self, df_name=df_name, comp_names=ts_generators.columns
            )
            # set (re)active power
            _add_component_time_series(obj=self, df_name=df_name, ts_new=ts_generators)

        if ts_loads is not None:
            # check if all loads time series are provided for exist in the network
            # and only set time series for those that do
            comps_in_network = _check_if_components_exist(
                edisgo_object, ts_loads.columns, "loads"
            )
            ts_loads = ts_loads.loc[:, comps_in_network]

            # drop load time series from self.loads_(re)active_power that may
            # already exist for some of the given loads
            df_name = f"loads_{mode}_power"
            drop_component_time_series(
                obj=self, df_name=df_name, comp_names=ts_loads.columns
            )
            # set (re)active power
            _add_component_time_series(obj=self, df_name=df_name, ts_new=ts_loads)

        if ts_storage_units is not None:
            # check if all storage units time series are provided for exist in the
            # network and only set time series for those that do
            comps_in_network = _check_if_components_exist(
                edisgo_object, ts_storage_units.columns, "storage_units"
            )
            ts_storage_units = ts_storage_units.loc[:, comps_in_network]

            # drop storage unit time series from self.storage_units_(re)active_power
            # that may already exist for some of the given storage units
            df_name = f"storage_units_{mode}_power"
            drop_component_time_series(
                obj=self, df_name=df_name, comp_names=ts_storage_units.columns
            )
            # set (re)active power
            _add_component_time_series(
                obj=self, df_name=df_name, ts_new=ts_storage_units
            )

    def set_worst_case(self, edisgo_object, cases):
        """
        Sets demand and feed-in of all loads, generators and storage units for the
        specified worst cases.

        Possible worst cases are 'load_case' (heavy load flow case) and 'feed-in_case'
        (reverse power flow case). Each case is set up once for dimensioning of the MV
        grid ('load_case_mv'/'feed-in_case_mv') and once for the dimensioning of the LV
        grid ('load_case_lv'/'feed-in_case_lv'), as different simultaneity factors are
        assumed for the different voltage levels.

        Assumed simultaneity factors specified in the config section
        `worst_case_scale_factor` are used to generate active power demand or feed-in.
        For the reactive power behavior fixed cosphi is assumed. The power factors
        set in the config section `reactive_power_factor` and the power factor
        mode, defining whether components behave inductive or capacitive, given
        in the config section `reactive_power_mode`, are used.

        Component specific information is given below:

        * Generators

            Worst case feed-in time series are distinguished by technology (PV, wind
            and all other) and whether it is a load or feed-in case.
            In case of generator worst case time series it is not distinguished by
            whether it is used to analyse the MV or LV. However, both options are
            generated as it is distinguished in the case of loads.
            Worst case scaling factors for generators are specified in
            the config section `worst_case_scale_factor` through the parameters:
            'feed-in_case_feed-in_pv', 'feed-in_case_feed-in_wind',
            'feed-in_case_feed-in_other',
            'load_case_feed-in_pv', load_case_feed-in_wind', and
            'load_case_feed-in_other'.

            For reactive power a fixed cosphi is assumed. A different reactive power
            factor is used for generators in the MV and generators in the LV.
            The reactive power factors for generators are specified in
            the config section `reactive_power_factor` through the parameters:
            'mv_gen' and 'lv_gen'.

        * Conventional loads

            Worst case load time series are distinguished by whether it
            is a load or feed-in case and whether it used to analyse the MV or LV.
            Worst case scaling factors for conventional loads are specified in
            the config section `worst_case_scale_factor` through the parameters:
            'mv_feed-in_case_load', 'lv_feed-in_case_load', 'mv_load_case_load', and
            'lv_load_case_load'.

            For reactive power a fixed cosphi is assumed. A different reactive power
            factor is used for loads in the MV and loads in the LV.
            The reactive power factors for conventional loads are specified in
            the config section `reactive_power_factor` through the parameters:
            'mv_load' and 'lv_load'.

        * Charging points

            Worst case demand time series are distinguished by use case (home charging,
            work charging, public (slow) charging and HPC), by whether it is a load or
            feed-in case and by whether it used to analyse the MV or LV.
            Worst case scaling factors for charging points are specified in
            the config section `worst_case_scale_factor` through the parameters:
            'mv_feed-in_case_cp_home', 'mv_feed-in_case_cp_work',
            'mv_feed-in_case_cp_public', and 'mv_feed-in_case_cp_hpc',
            'lv_feed-in_case_cp_home', 'lv_feed-in_case_cp_work',
            'lv_feed-in_case_cp_public', and 'lv_feed-in_case_cp_hpc',
            'mv_load-in_case_cp_home', 'mv_load-in_case_cp_work',
            'mv_load-in_case_cp_public', and 'mv_load-in_case_cp_hpc',
            'lv_load-in_case_cp_home', 'lv_load-in_case_cp_work',
            'lv_load-in_case_cp_public', and 'lv_load-in_case_cp_hpc'.

            For reactive power a fixed cosphi is assumed. A different reactive power
            factor is used for charging points in the MV and charging points in the LV.
            The reactive power factors for charging points are specified in
            the config section `reactive_power_factor` through the parameters:
            'mv_cp' and 'lv_cp'.

        * Heat pumps

            Worst case demand time series are distinguished by whether it is a load or
            feed-in case and by whether it used to analyse the MV or LV.
            Worst case scaling factors for heat pumps are specified in
            the config section `worst_case_scale_factor` through the parameters:
            'mv_feed-in_case_hp', 'lv_feed-in_case_hp', 'mv_load_case_hp', and
            'lv_load_case_hp'.

            For reactive power a fixed cosphi is assumed. A different reactive power
            factor is used for heat pumps in the MV and heat pumps in the LV.
            The reactive power factors for heat pumps are specified in
            the config section `reactive_power_factor` through the parameters:
            'mv_hp' and 'lv_hp'.

        * Storage units

            Worst case feed-in time series are distinguished by whether it is a load or
            feed-in case.
            In case of storage units worst case time series it is not distinguished by
            whether it is used to analyse the MV or LV. However, both options are
            generated as it is distinguished in the case of loads.
            Worst case scaling factors for storage units are specified in
            the config section `worst_case_scale_factor` through the parameters:
            'feed-in_case_storage' and 'load_case_storage'.

            For reactive power a fixed cosphi is assumed. A different reactive power
            factor is used for storage units in the MV and storage units in the LV.
            The reactive power factors for storage units are specified in
            the config section `reactive_power_factor` through the parameters:
            'mv_storage' and 'lv_storage'.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        cases : list(str)
            List with worst-cases to generate time series for. Can be
            'feed-in_case', 'load_case' or both.

        Notes
        -----
        Loads for which type information is not set are handled as conventional loads.

        """
        # reset all time series
        self.reset()

        # create a mapping from worst case cases to time stamps needed for pypsa
        worst_cases = [
            "_".join(case) for case in itertools.product(cases, ["mv", "lv"])
        ]
        time_stamps = pd.date_range("1/1/1970", periods=len(worst_cases), freq="H")
        self.timeindex_worst_cases = pd.Series(time_stamps, index=worst_cases)
        self.timeindex = time_stamps

        if not edisgo_object.topology.generators_df.empty:
            # assign voltage level for reactive power
            df = assign_voltage_level_to_component(
                edisgo_object.topology.generators_df, edisgo_object.topology.buses_df
            )
            p, q = self._worst_case_generators(cases, df, edisgo_object.config)
            # change index and set p and q
            self.generators_active_power = p.rename(index=self.timeindex_worst_cases)
            self.generators_reactive_power = q.rename(index=self.timeindex_worst_cases)

        if not edisgo_object.topology.loads_df.empty:
            # assign voltage level for reactive power
            df = assign_voltage_level_to_component(
                edisgo_object.topology.loads_df, edisgo_object.topology.buses_df
            )
            # conventional loads
            df_tmp = df[df.type == "conventional_load"]
            if not df_tmp.empty:
                p, q = self._worst_case_conventional_load(
                    cases, df_tmp, edisgo_object.config
                )
                # change index and set p and q
                self.loads_active_power = p.rename(index=self.timeindex_worst_cases)
                self.loads_reactive_power = q.rename(index=self.timeindex_worst_cases)
            # charging points
            df_tmp = df[df.type == "charging_point"]
            if not df_tmp.empty:
                p, q = self._worst_case_charging_points(
                    cases, df_tmp, edisgo_object.config
                )
                # change index and set p and q
                p = p.rename(index=self.timeindex_worst_cases)
                q = q.rename(index=self.timeindex_worst_cases)
                self.loads_active_power = pd.concat(
                    [self.loads_active_power, p], axis=1
                )
                self.loads_reactive_power = pd.concat(
                    [self.loads_reactive_power, q], axis=1
                )
            # heat pumps
            df_tmp = df[df.type == "heat_pump"]
            if not df_tmp.empty:
                p, q = self._worst_case_heat_pumps(cases, df_tmp, edisgo_object.config)
                # change index and set p and q
                p = p.rename(index=self.timeindex_worst_cases)
                q = q.rename(index=self.timeindex_worst_cases)
                self.loads_active_power = pd.concat(
                    [self.loads_active_power, p], axis=1
                )
                self.loads_reactive_power = pd.concat(
                    [self.loads_reactive_power, q], axis=1
                )
            # check if there are loads without time series remaining and if so, handle
            # them as conventional loads
            loads_without_ts = list(
                set(df.index) - set(self.loads_active_power.columns)
            )
            if loads_without_ts:
                logging.warning(
                    "There are loads where information on type of load is missing. "
                    "Handled types are 'conventional_load', 'charging_point', and "
                    "'heat_pump'. Loads with missing type information are handled as "
                    "conventional loads. If this is not the wanted behavior, please "
                    "set type information. This concerns the following "
                    f"loads: {loads_without_ts}."
                )
                p, q = self._worst_case_conventional_load(
                    cases, df.loc[loads_without_ts, :], edisgo_object.config
                )

                # change index and set p and q
                p = p.rename(index=self.timeindex_worst_cases)
                q = q.rename(index=self.timeindex_worst_cases)
                self.loads_active_power = pd.concat(
                    [self.loads_active_power, p], axis=1
                )
                self.loads_reactive_power = pd.concat(
                    [self.loads_reactive_power, q], axis=1
                )
        if not edisgo_object.topology.storage_units_df.empty:
            # assign voltage level for reactive power
            df = assign_voltage_level_to_component(
                edisgo_object.topology.storage_units_df, edisgo_object.topology.buses_df
            )
            p, q = self._worst_case_storage_units(cases, df, edisgo_object.config)
            # change index and set p and q
            self.storage_units_active_power = p.rename(index=self.timeindex_worst_cases)
            self.storage_units_reactive_power = q.rename(
                index=self.timeindex_worst_cases
            )

    def _worst_case_generators(self, cases, df, configs):
        """
        Get feed-in of generators for worst case analyses.

        See :py:attr:`~set_worst_case` for further information.

        Parameters
        ----------
        cases : list(str)
            List with worst-cases to generate time series for. Can be
            'feed-in_case', 'load_case' or both.
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with information on generators in the format of
            :attr:`~.network.topology.Topology.generators_df` with additional column
            "voltage_level".
        configs : :class:`~.tools.config.Config`
            Configuration data with assumed simultaneity factors and reactive power
            behavior.

        Returns
        -------
        (:pandas:`pandas.DataFrame<DataFrame>`, :pandas:`pandas.DataFrame<DataFrame>`)
            Active and reactive power (in MW and MVA, respectively) in each case for
            each generator. The index of the dataframe contains the case and the columns
            are the generator names.

        """
        # check that all generators have information on nominal power, technology type,
        # and voltage level they are in
        df = df.loc[:, ["p_nom", "voltage_level", "type"]]
        check = df.isnull().any(axis=1)
        if check.any():
            raise AttributeError(
                f"The following generators have missing information on nominal power, "
                f"technology type or voltage level: {check[check].index.values}."
            )

        # active power
        # get worst case configurations
        worst_case_scale_factors = configs["worst_case_scale_factor"]
        # get power scaling factors for different technologies, voltage levels and
        # feed-in/load case
        types = ["pv", "wind", "other"]
        power_scaling = pd.DataFrame(columns=types)
        for t in types:
            for case in cases:
                power_scaling.at[f"{case}_mv", t] = worst_case_scale_factors[
                    f"{case}_feed-in_{t}"
                ]

                power_scaling.at[f"{case}_lv", t] = power_scaling.at[f"{case}_mv", t]

        # calculate active power of generators
        active_power = pd.concat(
            [
                power_scaling.pv.to_frame("p_nom").dot(
                    df[df.type == "solar"].loc[:, ["p_nom"]].T
                ),
                power_scaling.wind.to_frame("p_nom").dot(
                    df[df.type == "wind"].loc[:, ["p_nom"]].T
                ),
                power_scaling.other.to_frame("p_nom").dot(
                    df[~df.type.isin(["solar", "wind"])].loc[:, ["p_nom"]].T
                ),
            ],
            axis=1,
        )

        # reactive power
        # get worst case configurations for each generator
        power_factor = q_control._fixed_cosphi_default_power_factor(
            df, "generators", configs
        )
        q_sign = q_control._fixed_cosphi_default_reactive_power_sign(
            df, "generators", configs
        )
        # write reactive power configuration to TimeSeriesRaw
        self.time_series_raw.q_control.drop(df.index, errors="ignore", inplace=True)
        self.time_series_raw.q_control = pd.concat(
            [
                self.time_series_raw.q_control,
                pd.DataFrame(
                    index=df.index,
                    data={
                        "type": "fixed_cosphi",
                        "q_sign": q_sign,
                        "power_factor": power_factor,
                    },
                ),
            ]
        )
        # calculate reactive power of generators
        reactive_power = q_control.fixed_cosphi(active_power, q_sign, power_factor)
        return active_power, reactive_power

    def _worst_case_conventional_load(self, cases, df, configs):
        """
        Get demand of conventional loads for worst case analyses.

        See :py:attr:`~set_worst_case` for further information.

        Parameters
        ----------
        cases : list(str)
            List with worst-cases to generate time series for. Can be
            'feed-in_case', 'load_case' or both.
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with information on conventional loads in the format of
            :attr:`~.network.topology.Topology.loads_df` with additional column
            "voltage_level".
        configs : :class:`~.tools.config.Config`
            Configuration data with assumed simultaneity factors and reactive power
            behavior.

        Returns
        -------
        (:pandas:`pandas.DataFrame<DataFrame>`, :pandas:`pandas.DataFrame<DataFrame>`)
            Active and reactive power (in MW and MVA, respectively) in each case for
            each load. The index of the dataframe contains the case and the columns
            are the load names.

        """
        # check that all loads have information on nominal power (grid connection power)
        # and voltage level they are in
        df = df.loc[:, ["p_set", "voltage_level"]]
        check = df.isnull().any(axis=1)
        if check.any():
            raise AttributeError(
                f"The following loads have missing information on grid connection power"
                f" or voltage level: {check[check].index.values}."
            )

        # active power
        # get worst case configurations
        worst_case_scale_factors = configs["worst_case_scale_factor"]
        # get power scaling factors for different voltage levels and feed-in/load case
        power_scaling = pd.Series(dtype=float)
        for case in cases:
            for voltage_level in ["mv", "lv"]:
                power_scaling.at[f"{case}_{voltage_level}"] = worst_case_scale_factors[
                    f"{voltage_level}_{case}_load"
                ]

        # calculate active power of loads
        active_power = power_scaling.to_frame("p_set").dot(df.loc[:, ["p_set"]].T)

        # reactive power
        # get worst case configurations for each load
        power_factor = q_control._fixed_cosphi_default_power_factor(
            df, "loads", configs
        )
        q_sign = q_control._fixed_cosphi_default_reactive_power_sign(
            df, "loads", configs
        )
        # write reactive power configuration to TimeSeriesRaw
        self.time_series_raw.q_control.drop(df.index, errors="ignore", inplace=True)
        self.time_series_raw.q_control = pd.concat(
            [
                self.time_series_raw.q_control,
                pd.DataFrame(
                    index=df.index,
                    data={
                        "type": "fixed_cosphi",
                        "q_sign": q_sign,
                        "power_factor": power_factor,
                    },
                ),
            ]
        )
        # calculate reactive power of loads
        reactive_power = q_control.fixed_cosphi(active_power, q_sign, power_factor)
        return active_power, reactive_power

    def _worst_case_charging_points(self, cases, df, configs):
        """
        Get demand of charging points for worst case analyses.

        See :py:attr:`~set_worst_case` for further information.

        Parameters
        ----------
        cases : list(str)
            List with worst-cases to generate time series for. Can be
            'feed-in_case', 'load_case' or both.
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with information on charging points in the format of
            :attr:`~.network.topology.Topology.loads_df` with additional column
            "voltage_level".
        configs : :class:`~.tools.config.Config`
            Configuration data with assumed simultaneity factors and reactive power
            behavior.

        Returns
        -------
        (:pandas:`pandas.DataFrame<DataFrame>`, :pandas:`pandas.DataFrame<DataFrame>`)
            Active and reactive power (in MW and MVA, respectively) in each case for
            each charging point. The index of the dataframe contains the case and the
            columns are the charging point names.

        """
        # check that all charging points have information on nominal power,
        # sector (use case), and voltage level they are in
        df = df.loc[:, ["p_set", "voltage_level", "sector"]]
        check = df.isnull().any(axis=1)
        if check.any():
            raise AttributeError(
                "The following charging points have missing information on nominal "
                f"power, use case or voltage level: {check[check].index.values}."
            )

        # check that there is no invalid sector (only "home", "work", "public", and
        # "hpc" allowed)
        use_cases = ["home", "work", "public", "hpc"]
        sectors = df.sector.unique()
        diff = list(set(sectors) - set(use_cases))
        if diff:
            raise AttributeError(
                "The following charging points have a use case no worst case "
                "simultaneity factor is defined for: "
                f"{df[df.sector.isin(diff)].index.values}."
            )

        # active power
        # get worst case configurations
        worst_case_scale_factors = configs["worst_case_scale_factor"]
        # get power scaling factors for different use cases, voltage levels and
        # feed-in/load case
        power_scaling = pd.DataFrame(columns=sectors)
        for s in sectors:
            for case in cases:
                for voltage_level in ["mv", "lv"]:
                    power_scaling.at[
                        f"{case}_{voltage_level}", s
                    ] = worst_case_scale_factors[f"{voltage_level}_{case}_cp_{s}"]

        # calculate active power of charging points
        active_power = pd.concat(
            [
                power_scaling.loc[:, s]
                .to_frame("p_set")
                .dot(df[df.sector == s].loc[:, ["p_set"]].T)
                for s in sectors
            ],
            axis=1,
        )

        # reactive power
        # get worst case configurations for each charging point
        power_factor = q_control._fixed_cosphi_default_power_factor(
            df, "charging_points", configs
        )
        q_sign = q_control._fixed_cosphi_default_reactive_power_sign(
            df, "charging_points", configs
        )
        # write reactive power configuration to TimeSeriesRaw
        self.time_series_raw.q_control.drop(df.index, errors="ignore", inplace=True)
        self.time_series_raw.q_control = pd.concat(
            [
                self.time_series_raw.q_control,
                pd.DataFrame(
                    index=df.index,
                    data={
                        "type": "fixed_cosphi",
                        "q_sign": q_sign,
                        "power_factor": power_factor,
                    },
                ),
            ]
        )
        # calculate reactive power of charging points
        reactive_power = q_control.fixed_cosphi(active_power, q_sign, power_factor)
        return active_power, reactive_power

    def _worst_case_heat_pumps(self, cases, df, configs):
        """
        Get demand of heat pumps for worst case analyses.

        See :py:attr:`~set_worst_case` for further information.

        Parameters
        ----------
        cases : list(str)
            List with worst-cases to generate time series for. Can be
            'feed-in_case', 'load_case' or both.
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with information on heat pumps in the format of
            :attr:`~.network.topology.Topology.loads_df` with additional column
            "voltage_level".
        configs : :class:`~.tools.config.Config`
            Configuration data with assumed simultaneity factors and reactive power
            behavior.

        Returns
        -------
        (:pandas:`pandas.DataFrame<DataFrame>`, :pandas:`pandas.DataFrame<DataFrame>`)
            Active and reactive power (in MW and MVA, respectively) in each case for
            each heat pump. The index of the dataframe contains the case and the columns
            are the heat pump names.

        """
        # check that all heat pumps have information on nominal power, and voltage level
        # they are in
        df = df.loc[:, ["p_set", "voltage_level"]]
        check = df.isnull().any(axis=1)
        if check.any():
            raise AttributeError(
                f"The following heat pumps have missing information on nominal power or"
                f" voltage level: {check[check].index.values}."
            )

        # active power
        # get worst case configurations
        worst_case_scale_factors = configs["worst_case_scale_factor"]
        # get power scaling factors for different voltage levels and feed-in/load case
        power_scaling = pd.Series()
        for case in cases:
            for voltage_level in ["mv", "lv"]:
                power_scaling.at[f"{case}_{voltage_level}"] = worst_case_scale_factors[
                    f"{voltage_level}_{case}_hp"
                ]

        # calculate active power of heat pumps
        active_power = power_scaling.to_frame("p_set").dot(df.loc[:, ["p_set"]].T)

        # reactive power
        # get worst case configurations for each heat pump
        power_factor = q_control._fixed_cosphi_default_power_factor(
            df, "heat_pumps", configs
        )
        q_sign = q_control._fixed_cosphi_default_reactive_power_sign(
            df, "heat_pumps", configs
        )
        # write reactive power configuration to TimeSeriesRaw
        self.time_series_raw.q_control.drop(df.index, errors="ignore", inplace=True)
        self.time_series_raw.q_control = pd.concat(
            [
                self.time_series_raw.q_control,
                pd.DataFrame(
                    index=df.index,
                    data={
                        "type": "fixed_cosphi",
                        "q_sign": q_sign,
                        "power_factor": power_factor,
                    },
                ),
            ]
        )
        # calculate reactive power of heat pumps
        reactive_power = q_control.fixed_cosphi(active_power, q_sign, power_factor)
        return active_power, reactive_power

    def _worst_case_storage_units(self, cases, df, configs):
        """
        Get charging and discharging of storage units for worst case analyses.

        See :py:attr:`~set_worst_case` for further information.

        Parameters
        ----------
        cases : list(str)
            List with worst-cases to generate time series for. Can be
            'feed-in_case', 'load_case' or both.
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with information on generators in the format of
            :attr:`~.network.topology.Topology.generators_df` with additional column
            "voltage_level".
        configs : :class:`~.tools.config.Config`
            Configuration data with assumed simultaneity factors and reactive power
            behavior.

        Returns
        -------
        (:pandas:`pandas.DataFrame<DataFrame>`, :pandas:`pandas.DataFrame<DataFrame>`)
            Active and reactive power (in MW and MVA, respectively) in each case for
            each storage. The index of the dataframe contains the case and the columns
            are the storage names.

        """
        # check that all storage units have information on nominal power
        # and voltage level they are in
        df = df.loc[:, ["p_nom", "voltage_level"]]
        check = df.isnull().any(axis=1)
        if check.any():
            raise AttributeError(
                "The following storage units have missing information on nominal power"
                f" or voltage level: {check[check].index.values}."
            )

        # active power
        # get worst case configurations
        worst_case_scale_factors = configs["worst_case_scale_factor"]
        # get power scaling factors for different voltage levels and feed-in/load case
        power_scaling = pd.Series()
        for case in cases:
            power_scaling.at[f"{case}_mv"] = worst_case_scale_factors[f"{case}_storage"]
            power_scaling.at[f"{case}_lv"] = power_scaling.at[f"{case}_mv"]

        # calculate active power of loads
        active_power = power_scaling.to_frame("p_nom").dot(df.loc[:, ["p_nom"]].T)

        # reactive power
        # get worst case configurations for each load
        power_factor = q_control._fixed_cosphi_default_power_factor(
            df, "storage_units", configs
        )
        q_sign = q_control._fixed_cosphi_default_reactive_power_sign(
            df, "storage_units", configs
        )
        # write reactive power configuration to TimeSeriesRaw
        self.time_series_raw.q_control.drop(df.index, errors="ignore", inplace=True)
        self.time_series_raw.q_control = pd.concat(
            [
                self.time_series_raw.q_control,
                pd.DataFrame(
                    index=df.index,
                    data={
                        "type": "fixed_cosphi",
                        "q_sign": q_sign,
                        "power_factor": power_factor,
                    },
                ),
            ]
        )
        # calculate reactive power of loads
        reactive_power = q_control.fixed_cosphi(active_power, q_sign, power_factor)
        return active_power, reactive_power

    def predefined_fluctuating_generators_by_technology(
        self, edisgo_object, ts_generators, generator_names=None
    ):
        """
        Set active power feed-in time series for fluctuating generators by technology.

        In case time series are provided per technology and weather cell ID, active
        power feed-in time series are also set by technology and weather cell ID.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_generators : str or :pandas:`pandas.DataFrame<dataframe>`
            Defines which technology-specific or technology and weather cell specific
            active power time series to use.
            Possible options are:

            * 'oedb'

                Technology and weather cell specific hourly feed-in time series are
                obtained from the OpenEnergy DataBase for the weather year 2011. See
                :func:`edisgo.io.timeseries_import.import_feedin_timeseries` for more
                information.

            * :pandas:`pandas.DataFrame<dataframe>`

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

        generator_names : list(str)
            Defines for which fluctuating generators to use technology-specific time
            series. If None, all generators technology (and weather cell) specific time
            series are provided for are used. In case the time series are retrieved from
            the oedb, all solar and wind generators are used.

        """
        # in case time series from oedb are used, retrieve oedb time series
        if isinstance(ts_generators, str) and ts_generators == "oedb":
            weather_cell_ids = get_weather_cells_intersecting_with_grid_district(
                edisgo_object
            )
            ts_generators = timeseries_import.feedin_oedb(
                edisgo_object.config, weather_cell_ids, self.timeindex
            )
        elif not isinstance(ts_generators, pd.DataFrame):
            raise ValueError(
                "'ts_generators' must either be a pandas DataFrame or 'oedb'."
            )

        # write to TimeSeriesRaw
        self.time_series_raw.fluctuating_generators_active_power_by_technology = (
            ts_generators
        )

        # set generator_names if None
        if generator_names is None:
            if isinstance(ts_generators.columns, pd.MultiIndex):
                groups = edisgo_object.topology.generators_df.groupby(
                    ["type", "weather_cell_id"]
                ).groups
                combinations = ts_generators.columns
                generator_names = np.concatenate(
                    [groups[_].values for _ in combinations if _ in groups.keys()]
                )
            else:
                technologies = ts_generators.columns.unique()
                generator_names = edisgo_object.topology.generators_df[
                    edisgo_object.topology.generators_df.type.isin(technologies)
                ].index
        generator_names = _check_if_components_exist(
            edisgo_object, generator_names, "generators"
        )
        generators_df = edisgo_object.topology.generators_df.loc[generator_names, :]

        # drop existing time series
        drop_component_time_series(
            obj=self, df_name="generators_active_power", comp_names=generator_names
        )

        # scale time series by nominal power
        if isinstance(ts_generators.columns, pd.MultiIndex):
            ts_scaled = generators_df.apply(
                lambda x: ts_generators[x.type][x.weather_cell_id].T * x.p_nom,
                axis=1,
            ).T
        else:
            ts_scaled = generators_df.apply(
                lambda x: ts_generators[x.type].T * x.p_nom,
                axis=1,
            ).T
        if not ts_scaled.empty:
            self.generators_active_power = pd.concat(
                [
                    self.generators_active_power,
                    ts_scaled,
                ],
                axis=1,
                sort=False,
            )

    def predefined_dispatchable_generators_by_technology(
        self, edisgo_object, ts_generators, generator_names=None
    ):
        """
        Set active power feed-in time series for dispatchable generators by technology.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_generators : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with self-provided active power time series of each
            type of dispatchable generator normalized to a nominal capacity of 1.
            Columns contain the technology type as string, e.g. 'gas', 'coal'.
            Use 'other' if you don't want to explicitly provide a time series for every
            possible technology. In the current grid existing generator technologies
            can be retrieved from column `type` in
            :attr:`~.network.topology.Topology.generators_df`.
            Index needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
        generator_names : list(str)
            Defines for which dispatchable generators to use technology-specific time
            series. If None, all dispatchable generators technology-specific time series
            are provided for are used. In case `ts_generators` contains a column
            'other', all dispatchable generators in the network (i.e. all but solar and
            wind generators) are used.

        """
        if not isinstance(ts_generators, pd.DataFrame):
            raise ValueError("'ts_generators' must be a pandas DataFrame.")

        # write to TimeSeriesRaw
        self.time_series_raw.dispatchable_generators_active_power_by_technology = (
            ts_generators
        )

        # set generator_names if None
        if generator_names is None:
            if "other" in ts_generators.columns:
                generator_names = edisgo_object.topology.generators_df[
                    ~edisgo_object.topology.generators_df.type.isin(["solar", "wind"])
                ].index
            else:
                generator_names = edisgo_object.topology.generators_df[
                    edisgo_object.topology.generators_df.type.isin(
                        ts_generators.columns
                    )
                ].index
        generator_names = _check_if_components_exist(
            edisgo_object, generator_names, "generators"
        )
        generators_df = edisgo_object.topology.generators_df.loc[generator_names, :]

        # drop existing time series
        drop_component_time_series(
            obj=self, df_name="generators_active_power", comp_names=generator_names
        )

        # scale time series by nominal power
        ts_scaled = generators_df.apply(
            lambda x: ts_generators[x.type] * x.p_nom
            if x.type in ts_generators.columns
            else ts_generators["other"] * x.p_nom,
            axis=1,
        ).T
        if not ts_scaled.empty:
            self.generators_active_power = pd.concat(
                [
                    self.generators_active_power,
                    ts_scaled,
                ],
                axis=1,
                sort=False,
            )

    def predefined_conventional_loads_by_sector(
        self, edisgo_object, ts_loads, load_names=None
    ):
        """
        Set active power demand time series for conventional loads by sector.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_loads : str or :pandas:`pandas.DataFrame<DataFrame>`
            Defines which sector-specific active power time series to use.
            Possible options are:

            * 'demandlib'

                Time series for the year specified :py:attr:`~timeindex` are
                generated using standard electric load profiles from the oemof
                `demandlib <https://github.com/oemof/demandlib/>`_.
                The demandlib provides sector-specific time series for the sectors
                'residential', 'retail', 'industrial', and 'agricultural'.

            * :pandas:`pandas.DataFrame<DataFrame>`

                DataFrame with load time series per sector normalized to an annual
                consumption of 1. Index needs to
                be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
                Columns contain the sector as string.
                In the current grid existing load types can be retrieved from column
                `sector` in :attr:`~.network.topology.Topology.loads_df` (make sure to
                select `type` 'conventional_load').
                In ding0 grid the differentiated sectors are 'residential', 'retail',
                'industrial', and 'agricultural'.
        load_names : list(str)
            Defines for which conventional loads to use sector-specific time series.
            If None, all loads of sectors for which sector-specific time series are
            provided are used. In case the demandlib is used, all loads of sectors
            'residential', 'retail', 'industrial', and 'agricultural' are used.

        """
        # in case time series from demandlib are used, retrieve demandlib time series
        if isinstance(ts_loads, str) and ts_loads == "demandlib":
            ts_loads = timeseries_import.load_time_series_demandlib(
                edisgo_object.config, timeindex=self.timeindex
            )
        elif not isinstance(ts_loads, pd.DataFrame):
            raise ValueError(
                "'ts_loads' must either be a pandas DataFrame or 'demandlib'."
            )
        elif ts_loads.empty:
            raise Warning("The profile you entered is empty. Method is skipped.")
            return

        # write to TimeSeriesRaw
        if self.time_series_raw.conventional_loads_active_power_by_sector is not None:
            for col in ts_loads:
                self.time_series_raw.conventional_loads_active_power_by_sector[
                    col
                ] = ts_loads[col]
        else:
            self.time_series_raw.conventional_loads_active_power_by_sector = (
                ts_loads.copy()
            )

        # set load_names if None
        if load_names is None:
            sectors = ts_loads.columns.unique()
            load_names = edisgo_object.topology.loads_df[
                edisgo_object.topology.loads_df.sector.isin(sectors)
            ].index
        load_names = _check_if_components_exist(edisgo_object, load_names, "loads")
        loads_df = edisgo_object.topology.loads_df.loc[load_names, :]

        # drop existing time series
        drop_component_time_series(
            obj=self, df_name="loads_active_power", comp_names=load_names
        )

        # scale time series by annual consumption
        self.loads_active_power = pd.concat(
            [
                self.loads_active_power,
                loads_df.apply(
                    lambda x: ts_loads[x.sector] * x.annual_consumption,
                    axis=1,
                ).T,
            ],
            axis=1,
        )

    def predefined_charging_points_by_use_case(
        self, edisgo_object, ts_loads, load_names=None
    ):
        """
        Set active power demand time series for charging points by their use case.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_loads : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with self-provided load time series per use case normalized to
            a nominal power of the charging point of 1.
            Index needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
            Columns contain the use case as string.
            In the current grid existing use case types can be retrieved from column
            `sector` in :attr:`~.network.topology.Topology.loads_df` (make sure to
            select `type` 'charging_point').
            When using charging point input from SimBEV the differentiated use cases are
            'home', 'work', 'public' and 'hpc'.
        load_names : list(str)
            Defines for which charging points to use use-case-specific time series.
            If None, all charging points of use cases for which use-case-specific time
            series are provided are used.

        """
        if not isinstance(ts_loads, pd.DataFrame):
            raise ValueError("'ts_loads' must be a pandas DataFrame.")
        elif ts_loads.empty:
            raise Warning("The profile you entered is empty. Method is skipped.")
            return

        # write to TimeSeriesRaw
        if self.time_series_raw.charging_points_active_power_by_use_case is not None:
            for col in ts_loads:
                self.time_series_raw.charging_points_active_power_by_use_case[
                    col
                ] = ts_loads[col]
        else:
            self.time_series_raw.charging_points_active_power_by_use_case = (
                ts_loads.copy()
            )

        # set load_names if None
        if load_names is None:
            sectors = ts_loads.columns.unique()
            load_names = edisgo_object.topology.loads_df[
                edisgo_object.topology.loads_df.sector.isin(sectors)
            ].index
        load_names = _check_if_components_exist(edisgo_object, load_names, "loads")
        loads_df = edisgo_object.topology.loads_df.loc[load_names, :]

        # check if all loads are charging points and throw warning if not
        if not all(loads_df.type.isin(["charging_point"])):
            raise Warning(
                "Not all affected loads are charging points. Please check and"
                " adapt if necessary."
            )

        # drop existing time series
        drop_component_time_series(
            obj=self, df_name="loads_active_power", comp_names=load_names
        )

        # scale time series by nominal power
        self.loads_active_power = pd.concat(
            [
                self.loads_active_power,
                loads_df.apply(
                    lambda x: ts_loads[x.sector] * x.p_set,
                    axis=1,
                ).T,
            ],
            axis=1,
        )

    def fixed_cosphi(
        self,
        edisgo_object,
        generators_parametrisation=None,
        loads_parametrisation=None,
        storage_units_parametrisation=None,
    ):
        """
        Sets reactive power of specified components assuming a fixed power factor.

        Overwrites reactive power time series in case they already exist.

        Parameters
        -----------
        generators_parametrisation : str or :pandas:`pandas.DataFrame<dataframe>`
            Sets fixed cosphi parameters for generators.
            Possible options are:

            * 'default'

                Default configuration is used for all generators in the grid.
                To this end, the power factors set in the config section
                `reactive_power_factor` and the power factor mode, defining whether
                components behave inductive or capacitive, given in the config section
                `reactive_power_mode`, are used.

            * :pandas:`pandas.DataFrame<dataframe>`

                DataFrame with fix cosphi parametrisation for specified generators.
                Columns are:

                    * 'components' : list(str)
                        List with generators to apply parametrisation for.

                    * 'mode' : str
                        Defines whether generators behave inductive or capacitive.
                        Possible options are 'inductive', 'capacitive' or 'default'.
                        In case of 'default', configuration from config section
                        `reactive_power_mode` is used.

                    * 'power_factor' : float or str
                        Defines the fixed cosphi power factor. The power factor can
                        either be directly provided as float or it can be set to
                        'default', in which case configuration from config section
                        `reactive_power_factor` is used.

                Index of the dataframe is ignored.
        loads_parametrisation : str or :pandas:`pandas.DataFrame<dataframe>`
            Sets fixed cosphi parameters for loads. The same options as for parameter
            `generators_parametrisation` apply.
        storage_units_parametrisation : str or :pandas:`pandas.DataFrame<dataframe>`
            Sets fixed cosphi parameters for storage units. The same options as for
            parameter `generators_parametrisation` apply.

        """

        def _get_q_sign_and_power_factor_per_component(
            parametrisation, components_df, type, q_sign_func
        ):
            # default configuration
            if isinstance(parametrisation, str) and parametrisation == "default":
                # get default parametrisation from config
                df = assign_voltage_level_to_component(
                    components_df, edisgo_object.topology.buses_df
                )
                components_names = df.index
                q_sign = q_control._fixed_cosphi_default_reactive_power_sign(
                    df, type, edisgo_object.config
                )
                power_factor = q_control._fixed_cosphi_default_power_factor(
                    df, type, edisgo_object.config
                )
            elif isinstance(parametrisation, pd.DataFrame):
                # check if all given components exist in network and only use existing
                components_names = list(
                    itertools.chain.from_iterable(parametrisation.components)
                )
                components_names = _check_if_components_exist(
                    edisgo_object, components_names, type
                )
                # set up series with sign of reactive power and power factors
                q_sign = pd.Series()
                power_factor = pd.Series()
                for index, row in parametrisation.iterrows():
                    # get only components that exist in the network
                    comps = [_ for _ in row["components"] if _ in components_names]
                    if len(comps) > 0:
                        # get q_sign (default or given)
                        if row["mode"] == "default":
                            df = assign_voltage_level_to_component(
                                components_df.loc[comps, :],
                                edisgo_object.topology.buses_df,
                            )
                            q_sign = pd.concat(
                                [
                                    q_sign,
                                    q_control._fixed_cosphi_default_reactive_power_sign(
                                        df, type, edisgo_object.config
                                    ),
                                ]
                            )
                        else:
                            q_sign = pd.concat(
                                [
                                    q_sign,
                                    pd.Series(q_sign_func(row["mode"]), index=comps),
                                ]
                            )
                        # get power factor (default or given)
                        if row["power_factor"] == "default":
                            df = assign_voltage_level_to_component(
                                components_df.loc[comps, :],
                                edisgo_object.topology.buses_df,
                            )
                            power_factor = pd.concat(
                                [
                                    power_factor,
                                    q_control._fixed_cosphi_default_power_factor(
                                        df, type, edisgo_object.config
                                    ),
                                ]
                            )
                        else:
                            power_factor = pd.concat(
                                [
                                    power_factor,
                                    pd.Series(row["power_factor"], index=comps),
                                ]
                            )
            else:
                raise ValueError(
                    f"'{type}_parametrisation' must either be a pandas DataFrame or "
                    f"'default'."
                )

            # write reactive power configuration to TimeSeriesRaw
            # delete existing previous settings
            self.time_series_raw.q_control.drop(
                index=self.time_series_raw.q_control.index[
                    self.time_series_raw.q_control.index.isin(components_names)
                ],
                inplace=True,
            )
            self.time_series_raw.q_control = pd.concat(
                [
                    self.time_series_raw.q_control,
                    pd.DataFrame(
                        index=components_names,
                        data={
                            "type": "fixed_cosphi",
                            "q_sign": q_sign,
                            "power_factor": power_factor,
                        },
                    ),
                ]
            )

            # drop existing time series
            drop_component_time_series(
                obj=self, df_name=f"{type}_reactive_power", comp_names=components_names
            )

            return q_sign, power_factor

        # set reactive power for generators
        if (
            generators_parametrisation is not None
            and not edisgo_object.topology.generators_df.empty
        ):
            q_sign, power_factor = _get_q_sign_and_power_factor_per_component(
                parametrisation=generators_parametrisation,
                components_df=edisgo_object.topology.generators_df,
                type="generators",
                q_sign_func=q_control.get_q_sign_generator,
            )
            # calculate reactive power
            reactive_power = q_control.fixed_cosphi(
                self.generators_active_power, q_sign, power_factor
            )
            self.generators_reactive_power = pd.concat(
                [self.generators_reactive_power, reactive_power], axis=1
            )
        if (
            loads_parametrisation is not None
            and not edisgo_object.topology.loads_df.empty
        ):
            q_sign, power_factor = _get_q_sign_and_power_factor_per_component(
                parametrisation=loads_parametrisation,
                components_df=edisgo_object.topology.loads_df,
                type="loads",
                q_sign_func=q_control.get_q_sign_load,
            )
            # calculate reactive power
            reactive_power = q_control.fixed_cosphi(
                self.loads_active_power, q_sign, power_factor
            )
            self.loads_reactive_power = pd.concat(
                [self.loads_reactive_power, reactive_power], axis=1
            )
        if (
            storage_units_parametrisation is not None
            and not edisgo_object.topology.storage_units_df.empty
        ):
            q_sign, power_factor = _get_q_sign_and_power_factor_per_component(
                parametrisation=storage_units_parametrisation,
                components_df=edisgo_object.topology.storage_units_df,
                type="storage_units",
                q_sign_func=q_control.get_q_sign_generator,
            )
            # calculate reactive power
            reactive_power = q_control.fixed_cosphi(
                self.storage_units_active_power, q_sign, power_factor
            )
            self.storage_units_reactive_power = pd.concat(
                [self.storage_units_reactive_power, reactive_power], axis=1
            )

    @property
    def residual_load(self):
        """
        Returns residual load in network.

        Residual load for each time step is calculated from total load
        minus total generation minus storage active power (discharge is
        positive).
        A positive residual load represents a load case while a negative
        residual load here represents a feed-in case.
        Grid losses are not considered.

        Returns
        -------
        :pandas:`pandas.Series<Series>`
            Series with residual load in MW.

        """
        return (
            self.loads_active_power.sum(axis=1)
            - self.generators_active_power.sum(axis=1)
            - self.storage_units_active_power.sum(axis=1)
        )

    @property
    def timesteps_load_feedin_case(self):
        """
        Contains residual load and information on feed-in and load case.

        Residual load is calculated from total (load - generation) in the
        network. Grid losses are not considered.

        Feed-in and load case are identified based on the
        generation, load and storage time series and defined as follows:

        1. Load case: positive (load - generation - storage) at HV/MV
           substation
        2. Feed-in case: negative (load - generation - storage) at HV/MV
           substation

        Returns
        -------
        :pandas:`pandas.Series<Series>`

            Series with information on whether time step is handled as load
            case ('load_case') or feed-in case ('feed-in_case') for each time
            step in :py:attr:`~timeindex`.

        """

        return self.residual_load.apply(
            lambda _: "feed-in_case" if _ < 0.0 else "load_case"
        )

    @property
    def _attributes(self):
        return [
            "loads_active_power",
            "loads_reactive_power",
            "generators_active_power",
            "generators_reactive_power",
            "storage_units_active_power",
            "storage_units_reactive_power",
        ]

    def reduce_memory(
        self, attr_to_reduce=None, to_type="float32", time_series_raw=True, **kwargs
    ):
        """
        Reduces size of dataframes to save memory.

        See :attr:`EDisGo.reduce_memory` for more information.

        Parameters
        -----------
        attr_to_reduce : list(str), optional
            List of attributes to reduce size for. Per default, all active
            and reactive power time series of generators, loads, and storage units
            are reduced.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".
        time_series_raw : bool, optional
            If True raw time series data in :py:attr:`~time_series_raw` is reduced
            as well. Default: True.

        Other Parameters
        ------------------
        attr_to_reduce_raw : list(str), optional
            List of attributes in :class:`~.network.timeseries.TimeSeriesRaw` to reduce
            size for. See :attr:`~.network.timeseries.TimeSeriesRaw.reduce_memory`
            for default.

        """
        if attr_to_reduce is None:
            attr_to_reduce = self._attributes
        for attr in attr_to_reduce:
            setattr(
                self,
                attr,
                getattr(self, attr).apply(lambda _: _.astype(to_type)),
            )
        if time_series_raw:
            self.time_series_raw.reduce_memory(
                kwargs.get("attr_to_reduce_raw", None), to_type=to_type
            )

    def to_csv(self, directory, reduce_memory=False, time_series_raw=False, **kwargs):
        """
        Saves component time series to csv.

        Saves the following time series to csv files with the same file name
        (if the time series dataframe is not empty):

        * loads_active_power and loads_reactive_power
        * generators_active_power and generators_reactive_power
        * storage_units_active_power and  storage_units_reactive_power

        If parameter `time_series_raw` is set to True, raw time series data is saved
        to csv as well. See :attr:`~.network.timeseries.TimeSeriesRaw.to_csv`
        for more information.

        Parameters
        ----------
        directory : str
            Directory to save time series in.
        reduce_memory : bool, optional
            If True, size of dataframes is reduced using
            :attr:`~.network.timeseries.TimeSeries.reduce_memory`.
            Optional parameters of
            :attr:`~.network.timeseries.TimeSeries.reduce_memory`
            can be passed as kwargs to this function. Default: False.
        time_series_raw : bool, optional
            If True raw time series data in :py:attr:`~time_series_raw` is saved to csv
            as well. Per default all raw time series data is then stored in a
            subdirectory of the specified `directory` called "time_series_raw". Further,
            if `reduce_memory` is set to True, raw time series data is reduced as well.
            To change this default behavior please call
            :attr:`~.network.timeseries.TimeSeriesRaw.to_csv` separately.
            Default: False.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain arguments of
            :attr:`~.network.timeseries.TimeSeries.reduce_memory`.

        """
        if reduce_memory is True:
            self.reduce_memory(**kwargs)

        os.makedirs(directory, exist_ok=True)

        for attr in self._attributes:
            if not getattr(self, attr).empty:
                getattr(self, attr).to_csv(os.path.join(directory, f"{attr}.csv"))

        if time_series_raw:
            self.time_series_raw.to_csv(
                directory=os.path.join(directory, "time_series_raw"),
                reduce_memory=reduce_memory,
            )

    def from_csv(self, directory, time_series_raw=False, **kwargs):
        """
        Restores time series from csv files.

        See :func:`~to_csv` for more information on which time series can be saved and
        thus restored.

        Parameters
        ----------
        directory : str
            Directory time series are saved in.
        time_series_raw : bool, optional
            If True raw time series data is as well read in (see
            :attr:`~.network.timeseries.TimeSeriesRaw.from_csv` for further
            information). Directory data is restored from can be specified through
            kwargs.
            Default: False.

        Other Parameters
        ------------------
        directory_raw : str, optional
            Directory to read raw time series data from. Per default this is a
            subdirectory of the specified `directory` called "time_series_raw".

        """
        timeindex = None
        for attr in self._attributes:
            path = os.path.join(directory, f"{attr}.csv")
            if os.path.exists(path):
                setattr(
                    self,
                    attr,
                    pd.read_csv(path, index_col=0, parse_dates=True),
                )
                if timeindex is None:
                    timeindex = getattr(self, "f_{attr}").index
        if timeindex is None:
            timeindex = pd.DatetimeIndex([])
        self._timeindex = timeindex

        if time_series_raw:
            self.time_series_raw.from_csv(
                directory=kwargs.get(
                    "directory_raw", os.path.join(directory, "time_series_raw")
                )
            )


class TimeSeriesRaw:
    """
    Holds raw time series data, e.g. sector-specific demand and standing times of EV.

    Normalised time series are e.g. sector-specific demand time series or
    technology-specific feed-in time series. Time series needed for
    flexibilities are e.g. heat time series or curtailment time series.

    Attributes
    ------------
    q_control : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with information on applied reactive power control or in case of
        conventional loads assumed reactive power behavior. Index of the dataframe are
        the component names as in index of
        :attr:`~.network.topology.Topology.generators_df`,
        :attr:`~.network.topology.Topology.loads_df`, and
        :attr:`~.network.topology.Topology.storage_units_df`. Columns are
        "type" with the type of Q-control applied (can be "fixed_cosphi", "cosphi(P)",
        or "Q(V)"),
        "power_factor" with the (maximum) power factor,
        "q_sign" giving the sign of the reactive power (only applicable to
        "fixed_cosphi"),
        "parametrisation" with the parametrisation of the
        respective Q-control (only applicable to "cosphi(P)" and "Q(V)").
    fluctuating_generators_active_power_by_technology : \
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with feed-in time series per technology or technology and
        weather cell ID normalized to a nominal capacity of 1.
        Columns can either just contain the technology type as string or
        be a :pandas:`pandas.MultiIndex<MultiIndex>` with the
        first level containing the technology as string and the second level
        the weather cell ID as integer.
        Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
    dispatchable_generators_active_power_by_technology : \
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with feed-in time series per technology normalized to a nominal
        capacity of 1.
        Columns contain the technology type as string.
        Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
    conventional_loads_active_power_by_sector : :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with load time series of each type of conventional load
        normalized to an annual consumption of 1. Index needs to
        be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
        Columns represent load type. In ding0 grids the
        differentiated sectors are 'residential', 'retail', 'industrial', and
        'agricultural'.
    charging_points_active_power_by_use_case : :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with charging demand time series per use case normalized to a nominal
        capacity of 1.
        Columns contain the use case as string.
        Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

    """

    def __init__(self):
        self.q_control = pd.DataFrame(
            columns=["type", "q_sign", "power_factor", "parametrisation"]
        )
        self.fluctuating_generators_active_power_by_technology = None
        self.dispatchable_generators_active_power_by_technology = None
        self.conventional_loads_active_power_by_sector = None
        self.charging_points_active_power_by_use_case = None

    @property
    def _attributes(self):
        return [
            "q_control",
            "fluctuating_generators_active_power_by_technology",
            "dispatchable_generators_active_power_by_technology",
            "conventional_loads_active_power_by_sector",
            "charging_points_active_power_by_use_case",
        ]

    def reduce_memory(self, attr_to_reduce=None, to_type="float32"):
        """
        Reduces size of dataframes to save memory.

        See :attr:`EDisGo.reduce_memory` for more information.

        Parameters
        -----------
        attr_to_reduce : list(str), optional
            List of attributes to reduce size for. Attributes need to be
            dataframes containing only time series. Per default, all active
            and reactive power time series of generators, loads, storage units
            and charging points are reduced.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        """
        if attr_to_reduce is None:
            attr_to_reduce = self._attributes
        # remove attributes that do not contain only floats
        if "q_control" in attr_to_reduce:
            attr_to_reduce.remove("q_control")
        for attr in attr_to_reduce:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(
                    self, attr, getattr(self, attr).apply(lambda _: _.astype(to_type))
                )

    def to_csv(self, directory, reduce_memory=False, **kwargs):
        """
        Saves time series to csv.

        Saves all attributes that are set to csv files with the same file name.
        See class definition for possible attributes.

        Parameters
        ----------
        directory: str
            Directory to save time series in.
        reduce_memory : bool, optional
            If True, size of dataframes is reduced using
            :attr:`~.network.timeseries.TimeSeriesRaw.reduce_memory`. Optional
            parameters of
            :attr:`~.network.timeseries.TimeSeriesRaw.reduce_memory`
            can be passed as kwargs to this function. Default: False.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain optional arguments of
            :attr:`~.network.timeseries.TimeSeriesRaw.reduce_memory`.

        """
        if reduce_memory is True:
            self.reduce_memory(**kwargs)

        os.makedirs(directory, exist_ok=True)

        for attr in self._attributes:
            if hasattr(self, attr) and not getattr(self, attr).empty:
                getattr(self, attr).to_csv(os.path.join(directory, f"{attr}.csv"))

    def from_csv(self, directory):
        """
        Restores time series from csv files.

        See :func:`~to_csv` for more information on which time series are
        saved.

        Parameters
        ----------
        directory : str
            Directory time series are saved in.

        """
        timeindex = None
        for attr in self._attributes:
            path = os.path.join(directory, f"{attr}.csv")
            if os.path.exists(path):
                setattr(
                    self,
                    attr,
                    pd.read_csv(path, index_col=0, parse_dates=True),
                )
                if timeindex is None:
                    timeindex = getattr(self, f"_{attr}").index
        if timeindex is None:
            timeindex = pd.DatetimeIndex([])
        self._timeindex = timeindex


def drop_component_time_series(obj, df_name, comp_names):
    """
    Drop component time series.

    Parameters
    ----------
    obj : obj
        Object with attr `df_name` to remove columns from. Can e.g. be
        :class:`~.network.timeseries.TimeSeries`.
    df_name : str
        Name of attribute of given object holding the dataframe to remove columns from.
        Can e.g. be "generators_active_power" if time series should be removed from
        :attr:`~.network.timeseries.TimeSeries.generators_active_power`.
    comp_names: str or list(str)
        Names of components to drop.

    """
    if isinstance(comp_names, str):
        comp_names = [comp_names]
    # drop existing time series of component
    setattr(
        obj,
        df_name,
        getattr(obj, df_name).drop(
            getattr(obj, df_name).columns[
                getattr(obj, df_name).columns.isin(comp_names)
            ],
            axis=1,
        ),
    )


def _add_component_time_series(obj, df_name, ts_new):
    """
    Add component time series.

    Parameters
    ----------
    obj : obj
        Object with attr `df_name` to add columns to. Can e.g. be
        :class:`~.network.timeseries.TimeSeries`.
    df_name : str
        Name of attribute of given object holding the dataframe to add columns to.
        Can e.g. be "generators_active_power" if time series should be added to
        :attr:`~.network.timeseries.TimeSeries.generators_active_power`.
    ts_new : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with new time series to add to existing time series dataframe.

    """
    setattr(
        obj,
        df_name,
        pd.concat(
            [getattr(obj, df_name), ts_new],
            axis=1,
        ),
    )


def _check_if_components_exist(edisgo_object, component_names, component_type):
    """
    Checks if all provided components exist in the network.

    Raises warning if there any provided components that are not in the network.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    component_names : list(str)
        Names of components for which time series are added.
    component_type : str
        The component type for which time series are added.
        Possible options are 'generators', 'storage_units', 'loads'.

    Returns
    --------
    set(str)
        Returns a set of all provided components that are in the network.

    """
    comps_in_network = getattr(edisgo_object.topology, f"{component_type}_df").index

    comps_not_in_network = list(set(component_names) - set(comps_in_network))

    if comps_not_in_network:
        logging.warning(
            f"Some of the provided {component_type} are not in the network. This "
            f"concerns the following components: {comps_not_in_network}."
        )

        return set(component_names) - set(comps_not_in_network)
    return component_names
