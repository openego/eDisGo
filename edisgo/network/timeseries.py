from itertools import product
import logging
import os

import numpy as np
import pandas as pd

from edisgo.io import timeseries_import
from edisgo.flex_opt import q_control
from edisgo.tools.tools import (
    assign_voltage_level_to_component,
    drop_duplicated_columns,
    get_weather_cells_intersecting_with_grid_district,
)

logger = logging.getLogger("edisgo")


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
        if len(self._timeindex) > 0:
            # check if new time index is subset of existing time index
            if not ind.isin(self._timeindex).all():
                logger.warning(
                    "Not all time steps of new time index lie within existing "
                    "time index. This may cause problems later on."
                )
        self._timeindex = ind

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
        try:
            return self._generators_active_power.loc[self.timeindex, :]
        except:
            return pd.DataFrame(index=self.timeindex)

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
        try:
            return self._generators_reactive_power.loc[self.timeindex, :]
        except:
            return pd.DataFrame(index=self.timeindex)

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
        try:
            return self._loads_active_power.loc[self.timeindex, :]
        except:
            return pd.DataFrame(index=self.timeindex)

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
        try:
            return self._loads_reactive_power.loc[self.timeindex, :]
        except:
            return pd.DataFrame(index=self.timeindex)

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
        try:
            return self._storage_units_active_power.loc[self.timeindex, :]
        except:
            return pd.DataFrame(index=self.timeindex)

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
        try:
            return self._storage_units_reactive_power.loc[self.timeindex, :]
        except:
            return pd.DataFrame(index=self.timeindex)

    @storage_units_reactive_power.setter
    def storage_units_reactive_power(self, df):
        self._storage_units_reactive_power = df

    def set_active_power_manual(self, ts_generators=None, ts_loads=None,
                                ts_storage_units=None):
        """
        Sets given component active power time series.

        If time series for a component were already set before, they are overwritten.

        Parameters
        ----------
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
        self._set_manual("active", ts_generators=ts_generators, ts_loads=ts_loads,
                         ts_storage_units=ts_storage_units)

    def set_reactive_power_manual(self, ts_generators=None, ts_loads=None,
                                  ts_storage_units=None):
        """
        Sets given component reactive power time series.

        If time series for a component were already set before, they are overwritten.

        Parameters
        ----------
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
        self._set_manual("reactive", ts_generators=ts_generators, ts_loads=ts_loads,
                         ts_storage_units=ts_storage_units)

    def _set_manual(self, mode, ts_generators=None, ts_loads=None,
                    ts_storage_units=None):
        """
        Sets given component time series.

        If time series for a component were already set before, they are overwritten.

        Parameters
        ----------
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
            df_name = "generators_{}_power".format(mode)
            # drop generators time series from self.generators_(re)active_power that may
            # already exist for some of the given generators
            _drop_component_time_series(
                obj=self, df_name=df_name,
                comp_names=ts_generators.columns
            )
            # set (re)active power
            _add_component_time_series(obj=self, df_name=df_name,
                                       ts_new=ts_generators)

        if ts_loads is not None:
            df_name = "loads_{}_power".format(mode)
            # drop load time series from self.loads_(re)active_power that may
            # already exist for some of the given loads
            _drop_component_time_series(
                obj=self, df_name=df_name, comp_names=ts_loads.columns
            )
            # set (re)active power
            _add_component_time_series(obj=self, df_name=df_name,
                                       ts_new=ts_loads)

        if ts_storage_units is not None:
            df_name = "storage_units_{}_power".format(mode)
            # drop storage unit time series from self.storage_units_(re)active_power
            # that may already exist for some of the given storage units
            _drop_component_time_series(
                obj=self, df_name=df_name,
                comp_names=ts_storage_units.columns
            )
            # set (re)active power
            _add_component_time_series(obj=self, df_name=df_name,
                                       ts_new=ts_storage_units)

    def set_worst_case(self, edisgo_object, cases):
        """
        Worst case scaling factors for loads and generators are specified in
        the config section `worst_case_scale_factor`.

        For each case time series for MV analysis and LV analysis are returned, as
        different simultaneity factors are assumed.

        For conventional loads simultaneity factors from config used. Fixed coshphi assumed.
        Values for cosphi also taken from config and it is distinguished between loads
        in the MV and LV.

        """
        #ToDo: Check if index needs to be time index
        # self.timeindex = pd.date_range(
        #     "1/1/1970", periods=len(modes), freq="H"
        # )
        self.timeindex = ["_".join(case) for case in product(cases, ["mv", "lv"])]

        if not edisgo_object.topology.generators_df.empty:
            # assign voltage level for reactive power
            df = assign_voltage_level_to_component(
                edisgo_object.topology.generators_df, edisgo_object.topology.buses_df)
            self._worst_case_generators(
                cases, df, edisgo_object.config)
        if not edisgo_object.topology.loads_df.empty:
            # assign voltage level for reactive power
            df = assign_voltage_level_to_component(
                edisgo_object.topology.loads_df, edisgo_object.topology.buses_df)
            # conventional loads
            self._worst_case_conventional_load(
                cases, df[df.type == "conventional_load"],
                edisgo_object.config)

            # charging points

            # heat pumps

            # other?
        print("x")

    def _worst_case_generators(self, cases, df, configs):
        """
        Get feed-in of generators for worst case analyses.

        Worst case feed-in time series are distinguished by technology (PV, wind
        and all other) and whether it is a load or feed-in case.
        In case of generator worst case time series it is not distinguished by whether
        it is used to analyse the MV or LV. However, both options are generated as it
        is distinguished in the case of loads.

        For reactive power a fixed cosphi is assumed.

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

        """
        # check that all generators have information on nominal power, technology type,
        # and voltage level they are in
        df = df.loc[:, ["p_nom", "voltage_level", "type"]]
        check = df.isnull().any(axis=1)
        if check.any():
            raise AttributeError(
                "The following generators have missing information on "
                "nominal power, technology type or voltage level: {}.".format(
                    check[check].index.values)
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
                power_scaling.at["{}_{}".format(case, "mv"), t] = (
                    worst_case_scale_factors[
                        "{}_feed-in_{}".format(case, t)]
                )
                power_scaling.at["{}_{}".format(case, "lv"), t] = power_scaling.at[
                    "{}_{}".format(case, "mv"), t]
        # calculate active power of generators
        self.generators_active_power = pd.concat(
            [power_scaling.pv.to_frame("p_nom").dot(
                df[df.type == "solar"].loc[:, ["p_nom"]].T),
             power_scaling.wind.to_frame("p_nom").dot(
                 df[df.type == "wind"].loc[:, ["p_nom"]].T),
                power_scaling.other.to_frame("p_nom").dot(
                    df[~df.type.isin(["solar", "wind"])].loc[:, ["p_nom"]].T)
            ], axis=1
        )

        # reactive power
        # get worst case configurations for each load
        q_sign, power_factor = _reactive_power_factor_and_mode_default(
            df, "generators", configs)
        # write reactive power configuration to TimeSeriesRaw
        self.time_series_raw.q_control = pd.concat([
            self.time_series_raw.q_control,
            pd.DataFrame(
                index=df.index,
                data={"type": "fixed_cosphi",
                      "q_sign": q_sign,
                      "power_factor": power_factor
                      }
            )]
        )
        # calculate reactive power of loads
        self.generators_reactive_power = q_control.fixed_cosphi(
            self.generators_active_power, q_sign, power_factor)

    def _worst_case_conventional_load(self, cases, df, configs):
        """
        Get demand of conventional loads for worst case analyses.

        Worst case load time series are distinguished by whether it
        is a load or feed-in case and whether it used to analyse the MV or LV.

        For reactive power a fixed cosphi is assumed.

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

        """
        # check that all loads have information on nominal power (grid connection power)
        # and voltage level they are in
        df = df.loc[:, ["p_nom", "voltage_level"]]
        check = df.isnull().any(axis=1)
        if check.any():
            raise AttributeError(
                "The following loads have missing information on "
                "grid connection power or voltage level: {}.".format(
                    check[check].index.values)
            )

        # active power
        # get worst case configurations
        worst_case_scale_factors = configs["worst_case_scale_factor"]
        # get power scaling factors for different voltage levels and feed-in/load case
        power_scaling = pd.Series()
        for case in cases:
            for voltage_level in ["mv", "lv"]:
                power_scaling.at["{}_{}".format(case, voltage_level)] = (
                    worst_case_scale_factors[
                        "{}_{}_load".format(voltage_level, case)]
                )
        # calculate active power of loads
        self.loads_active_power = power_scaling.to_frame("p_nom").dot(
            df.loc[:, ["p_nom"]].T)

        # reactive power
        # get worst case configurations for each load
        q_sign, power_factor = _reactive_power_factor_and_mode_default(
            df, "loads", configs)
        # write reactive power configuration to TimeSeriesRaw
        self.time_series_raw.q_control = pd.concat([
            self.time_series_raw.q_control,
            pd.DataFrame(
                index=df.index,
                data={"type": "fixed_cosphi",
                      "q_sign": q_sign,
                      "power_factor": power_factor
                      }
            )]
        )
        # calculate reactive power of loads
        self.loads_reactive_power = q_control.fixed_cosphi(
            self.loads_active_power, q_sign, power_factor)


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
                self.loads_active_power.sum(axis=1) -
                self.generators_active_power.sum(axis=1) -
                self.storage_units_active_power.sum(axis=1)
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
            "loads_active_power", "loads_reactive_power",
            "generators_active_power", "generators_reactive_power",
            "storage_units_active_power", "storage_units_reactive_power"
        ]

    def reduce_memory(self, attr_to_reduce=None, to_type="float32",
                      time_series_raw=True, **kwargs):
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
                kwargs.get("attr_to_reduce_raw", None),
                to_type=to_type
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
                getattr(self, attr).to_csv(
                    os.path.join(directory, "{}.csv".format(attr))
                )

        if time_series_raw:
            self.time_series_raw.to_csv(
                directory=os.path.join(directory, "time_series_raw"),
                reduce_memory=reduce_memory
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
            path = os.path.join(directory, "{}.csv".format(attr))
            if os.path.exists(path):
                setattr(
                    self,
                    attr,
                    pd.read_csv(path, index_col=0, parse_dates=True),
                )
                if timeindex is None:
                    timeindex = getattr(self, "_{}".format(attr)).index
        if timeindex is None:
            timeindex = pd.DatetimeIndex([])
        self._timeindex = timeindex

        if time_series_raw:
            self.time_series_raw.from_csv(
                directory=kwargs.get(
                    "directory_raw", os.path.join(directory, "time_series_raw"))
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

    Notes
    -----
    Can also hold the following attributes when specific mode of
    :meth:`get_component_timeseries` is called: mode, generation_fluctuating,
    generation_dispatchable, generation_reactive_power, load,
    load_reactive_power. See description of meth:`get_component_timeseries` for
    format of these.

    """

    def __init__(self, **kwargs):
        self.q_control = pd.DataFrame(
            columns=["type", "q_sign", "power_factor", "parametrisation"])

    @property
    def _attributes(self):
        return [
            "curtailment_target",
            "generators_active_power_normalised"
        ]

    @property
    def curtailment_target(self):
        """
        Generator- or technology-specific active power curtailment target.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with generator- or technology-specific curtailment
            target in MW.
            In the case of generator-specific curtailment targets columns
            of the DataFrame hold the generator name.
            In the case of technology-specific curtailment targets columns
            hold the technology type. It is also possible to provide
            curtailment targets by generator type and weather cell ID, in which
            case columns are a :pandas:`pandas.MultiIndex<multiindex>`
            with the first level containing the technology type and the second
            level the weather cell ID.
            Index of the DataFrame is a time index.

        """
        try:
            return self._curtailment_target
        except:
            return pd.DataFrame()

    @curtailment_target.setter
    def curtailment_target(self, curtailment_target):
        self._curtailment_target = curtailment_target

    @property
    def generators_active_power_normalised(self):
        """
        Technology-specific active power feed-in time series.

        In case of wind and solar generators time series can further be
        specified by weather cell ID, in which case columns of the DataFrame
        are a :pandas:`pandas.MultiIndex<multiindex>` with
        the first level containing the technology type and the second level
        the weather cell ID.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with technology-specific active power feed-in time series
            in MW.
            In the case of only technology-specific time series columns
            hold the technology type. In the case of technology- and weather
            cell-specific time series columns are a
            :pandas:`pandas.MultiIndex<multiindex>` with the first level
            containing the technology type and the second level
            the weather cell ID.
            Index of the DataFrame is a time index.

        """
        try:
            return self._generators_active_power_normalised
        except:
            return pd.DataFrame()

    @generators_active_power_normalised.setter
    def generators_active_power_normalised(
            self, generators_active_power_normalised):
        self._generators_active_power_normalised = \
            generators_active_power_normalised

    @property
    def loads_active_power_normalised(self):
        """
        Sector-specific active power demand time series.

        Holds e.g. sector-specific electricity demand time series as generated
        using the demandlib (see .

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            DataFrame with technology-specific active power feed-in time series
            in MW.
            In the case of only technology-specific time series columns
            hold the technology type. In the case of technology- and weather
            cell-specific time series columns are a
            :pandas:`pandas.MultiIndex<multiindex>` with the first level
            containing the technology type and the second level
            the weather cell ID.
            Index of the DataFrame is a time index.

        """
        try:
            return self._generators_active_power_normalised
        except:
            return pd.DataFrame()

    @generators_active_power_normalised.setter
    def generators_active_power_normalised(
            self, generators_active_power_normalised):
        self._generators_active_power_normalised = \
            generators_active_power_normalised

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
        # ToDo: Adapt once format of raw data is defined
        if attr_to_reduce is None:
            attr_to_reduce = self._attributes
        for attr in attr_to_reduce:
            setattr(
                self,
                attr,
                getattr(self, attr).apply(
                    lambda _: _.astype(to_type)
                )
            )

    def to_csv(self, directory, reduce_memory=False, **kwargs):
        """
        Saves time series to csv.

        Saves the following time series to csv files with the same file name
        (if the time series dataframe is not empty):

        * curtailment

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
            if not getattr(self, attr).empty:
                getattr(self, attr).to_csv(
                    os.path.join(directory, "{}.csv".format(attr))
                )

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
            path = os.path.join(directory, "{}.csv".format(attr))
            if os.path.exists(path):
                setattr(
                    self,
                    attr,
                    pd.read_csv(path, index_col=0, parse_dates=True),
                )
                if timeindex is None:
                    timeindex = getattr(self, "_{}".format(attr)).index
        if timeindex is None:
            timeindex = pd.DatetimeIndex([])
        self._timeindex = timeindex


def get_component_timeseries(edisgo_obj, **kwargs):
    """
    Sets up TimeSeries Object.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo data container
    mode : :obj:`str`, optional
        Mode must be set in case of worst-case analyses and can either be
        'worst-case' (both feed-in and load case), 'worst-case-feedin' (only
        feed-in case) or 'worst-case-load' (only load case). All other
        parameters except of `config-data` will be ignored. Default: None.
        Mode can also be set to manual in order to give standard timeseries,
        that are not obtained from oedb or demandlib.
    timeseries_generation_fluctuating : :obj:`str` or :pandas:`pandas.DataFrame<dataframe>`, optional
        Parameter used to obtain time series for active power feed-in of
        fluctuating renewables wind and solar.
        Possible options are:

        * 'oedb'
          Time series for 2011 are obtained from the OpenEnergy DataBase.
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
        needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
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
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
        Columns represent load type:

          * 'residential'
          * 'retail'
          * 'industrial'
          * 'agricultural'

        Default: None.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
        Can be used to define a time range for which to obtain load time series
        and feed-in time series of fluctuating renewables or to define time
        ranges of the given time series that will be used in the analysis.

    """
    mode = kwargs.get("mode", None)
    timeindex = kwargs.get("timeindex", edisgo_obj.timeseries.timeindex)
    # reset TimeSeries
    edisgo_obj.timeseries = TimeSeries(
        timeindex=timeindex)
    edisgo_obj.timeseries.mode = mode

    config_data = edisgo_obj.config

    weather_cell_ids = get_weather_cells_intersecting_with_grid_district(edisgo_obj)

    # feed-in time series of fluctuating renewables
    ts = kwargs.get("timeseries_generation_fluctuating", None)
    if isinstance(ts, pd.DataFrame):
        edisgo_obj.timeseries.generation_fluctuating = ts
    elif isinstance(ts, str) and ts == "oedb":
        edisgo_obj.timeseries.generation_fluctuating = \
            timeseries_import.feedin_oedb(
                config_data, weather_cell_ids, kwargs.get(
                    "timeindex", None))
    else:
        raise ValueError(
            "Your input for "
            '"timeseries_generation_fluctuating" is not '
            "valid.".format(mode)
        )
    # feed-in time series for dispatchable generators
    ts = kwargs.get("timeseries_generation_dispatchable", None)
    if isinstance(ts, pd.DataFrame):
        edisgo_obj.timeseries.generation_dispatchable = ts
    else:
        # check if there are any dispatchable generators, and
        # throw error if there are
        gens = edisgo_obj.topology.generators_df
        if not (gens.type.isin(["solar", "wind"])).all():
            raise ValueError(
                'Your input for "timeseries_generation_dispatchable" '
                "is not valid.".format(mode)
            )

    # load time series
    ts = kwargs.get("timeseries_load", None)
    if isinstance(ts, pd.DataFrame):
        edisgo_obj.timeseries.load = ts
    elif ts == "demandlib":
        edisgo_obj.timeseries.load = \
            timeseries_import.load_time_series_demandlib(
                config_data,
                year=edisgo_obj.timeseries.timeindex[0].year
            )
    else:
        raise ValueError(
            "Your input for 'timeseries_load' is not valid.".format(mode)
        )

    # create generator active and reactive power timeseries
    _generation_from_timeseries(edisgo_obj=edisgo_obj)

    # create load active and reactive power timeseries
    load_by_sector(edisgo_obj=edisgo_obj)

    # create storage active and reactive power timeseries
    _storage_from_timeseries(
        edisgo_obj=edisgo_obj,
        ts_active_power=kwargs.get("timeseries_storage_units", None),
        ts_reactive_power=kwargs.get(
            "timeseries_storage_units_reactive_power", None
        ),
    )

    # check if time series for the set time index can be obtained
    _check_timeindex(edisgo_obj=edisgo_obj)


def load_by_sector(edisgo_obj, load_names=None):
    """
    Set active and reactive load time series for specified loads by sector.

    If loads are not specified, sets time series of all existing loads.
    In case reactive power time series are not provided, a fixed power factor
    as specified in config file 'config_timeseries' in section
    'reactive_power_factor' is assumed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    load_names : list(str)

    """
    # get all requested loads and drop existing timeseries
    if load_names is None:
        load_names = edisgo_obj.topology.loads_df.index
    loads = edisgo_obj.topology.loads_df.loc[load_names]
    _drop_existing_component_timeseries(
        edisgo_obj=edisgo_obj, comp_type="loads", comp_names=load_names
    )
    # set active power
    edisgo_obj.timeseries.loads_active_power = pd.concat(
        [
            edisgo_obj.timeseries.loads_active_power,
            loads.apply(
                lambda x: edisgo_obj.timeseries.load[x.sector] * x.annual_consumption
                if x.sector in edisgo_obj.timeseries.load.columns
                else edisgo_obj.timeseries.load["other"] * x.annual_consumption,
                axis=1,
            ).T,
        ],
        axis=1,
    )

    # if reactive power is given as attribute set with inserted timeseries
    if hasattr(edisgo_obj.timeseries, "load_reactive_power"):
        edisgo_obj.timeseries.loads_reactive_power = pd.concat(
            [
                edisgo_obj.timeseries.loads_reactive_power,
                loads.apply(
                    lambda x: edisgo_obj.timeseries.load_reactive_power[x.sector]
                    * x.annual_consumption
                    if x.sector in edisgo_obj.timeseries.load_reactive_power.columns
                    else edisgo_obj.timeseries.load_reactive_power["other"]
                    * x.annual_consumption,
                    axis=1,
                ),
            ],
            axis=1,
        )
    # set default reactive load
    else:
        q_control._set_reactive_power_time_series_for_fixed_cosphi_using_config(
            edisgo_obj=edisgo_obj,
            df=loads,
            component_type="loads"
        )


def _generation_from_timeseries(edisgo_obj, generator_names=None):
    def _timeseries_fluctuating():
        if isinstance(
            edisgo_obj.timeseries.generation_fluctuating.columns, pd.MultiIndex
        ):
            return gens_fluctuating.apply(
                lambda x: edisgo_obj.timeseries.generation_fluctuating[x.type][
                    x.weather_cell_id
                ].T
                * x.p_nom,
                axis=1,
            ).T
        else:
            return gens_fluctuating.apply(
                lambda x: edisgo_obj.timeseries.generation_fluctuating[x.type].T
                * x.p_nom,
                axis=1,
            ).T

    def _timeseries_dispatchable():
        return gens_dispatchable.apply(
            lambda x: edisgo_obj.timeseries.generation_dispatchable[x.type] * x.p_nom
            if x.type in edisgo_obj.timeseries.generation_dispatchable.columns
            else edisgo_obj.timeseries.generation_dispatchable["other"] * x.p_nom,
            axis=1,
        ).T

    if generator_names is None:
        generator_names = edisgo_obj.topology.generators_df.index
    # get all generators
    gens = edisgo_obj.topology.generators_df.loc[generator_names]
    # drop existing timeseries
    _drop_existing_component_timeseries(edisgo_obj, "generators", generator_names)
    # handling of fluctuating generators
    gens_fluctuating = gens[gens.type.isin(["solar", "wind"])]
    gens_dispatchable = gens[~gens.index.isin(gens_fluctuating.index)]
    if gens_dispatchable.empty and gens_fluctuating.empty:
        logger.debug("No generators provided to add timeseries for.")
        return
    if not gens_dispatchable.empty:
        edisgo_obj.timeseries.generators_active_power = pd.concat(
            [
                edisgo_obj.timeseries.generators_active_power,
                _timeseries_dispatchable(),
            ],
            axis=1,
            sort=False,
        )
    if not gens_fluctuating.empty:
        edisgo_obj.timeseries.generators_active_power = pd.concat(
            [
                edisgo_obj.timeseries.generators_active_power,
                _timeseries_fluctuating(),
            ],
            axis=1,
            sort=False,
        )

    # set reactive power if given as attribute
    if (
        hasattr(edisgo_obj.timeseries, "generation_reactive_power")
        and gens.index.isin(
            edisgo_obj.timeseries.generation_reactive_power.columns
        ).all()
    ):

        edisgo_obj.timeseries.generators_reactive_power = pd.concat(
            [
                edisgo_obj.timeseries.generators_reactive_power,
                edisgo_obj.timeseries.generation_reactive_power.loc[:, gens.index],
            ],
            axis=1,
        )
    # set default reactive power by cos_phi
    else:
        logger.debug("Reactive power calculated by cos(phi).")
        q_control._set_reactive_power_time_series_for_fixed_cosphi_using_config(
            edisgo_obj=edisgo_obj,
            df=gens,
            component_type="generators"
        )


def _storage_from_timeseries(
    edisgo_obj, ts_active_power, ts_reactive_power, name_storage_units=None
):
    """
    Sets up storage timeseries for mode=None in get_component_timeseries.
    Timeseries with the right timeindex and columns with storage unit names
    have to be provided.

    Overwrites active and reactive power time series of storage units

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    ts_active_power: :pandas:`pandas.DataFrame<dataframe>`
        Timeseries of active power with index=timeindex,
        columns=name_storage_units
    ts_reactive_power: :pandas:`pandas.DataFrame<dataframe>`
        Timeseries of active power with index=timeindex,
        columns=name_storage_units
    name_storage_units: str or list of str
        Names of storage units to add timeseries for. Default None, timeseries
        for all storage units of edisgo_obj are set then.
    """
    if name_storage_units is None:
        name_storage_units = edisgo_obj.topology.storage_units_df.index
    storage_units_df = edisgo_obj.topology.storage_units_df.loc[name_storage_units]
    _drop_existing_component_timeseries(edisgo_obj, "storage_units", name_storage_units)

    if len(storage_units_df) == 0:
        edisgo_obj.timeseries.storage_units_active_power = pd.DataFrame(
            {}, index=edisgo_obj.timeseries.timeindex
        )
        edisgo_obj.timeseries.storage_units_reactive_power = pd.DataFrame(
            {}, index=edisgo_obj.timeseries.timeindex
        )
    elif ts_active_power is None:
        # Todo: move up to check at the start
        raise ValueError("No timeseries for storage units provided.")
    else:
        try:
            # check if indices and columns are correct
            if (ts_active_power.index == edisgo_obj.timeseries.timeindex).all():
                edisgo_obj.timeseries.storage_units_active_power = (
                    drop_duplicated_columns(
                        pd.concat(
                            [
                                edisgo_obj.timeseries.storage_units_active_power,
                                ts_active_power.loc[:, name_storage_units],
                            ],
                            axis=1,
                        )
                    )
                )
                # check if reactive power is given
                if (
                    ts_reactive_power is not None
                    and (ts_active_power.index == edisgo_obj.timeseries.timeindex).all()
                ):
                    edisgo_obj.timeseries.storage_units_reactive_power = (
                        drop_duplicated_columns(
                            pd.concat(
                                [
                                    edisgo_obj.timeseries.storage_units_reactive_power,
                                    ts_reactive_power.loc[:, name_storage_units],
                                ],
                                axis=1,
                            )
                        )
                    )
                else:
                    q_control._set_reactive_power_time_series_for_fixed_cosphi_using_config(
                        edisgo_obj=edisgo_obj,
                        df=storage_units_df,
                        component_type="storage_units"
                    )
            else:
                raise ValueError(
                    "Index of provided storage active power "
                    "timeseries does not match timeindex of "
                    "TimeSeries class."
                )
        except ValueError:
            raise ValueError(
                "Columns or indices of inserted storage "
                "timeseries do not match topology and "
                "timeindex."
            )


def _worst_case_storage(edisgo_obj, modes, storage_names=None):
    """
    Define worst case storage unit time series.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    modes : list
        List with worst-cases to generate time series for. Can be
        'feed-in_case', 'load_case' or both.
    storage_names: str or list of str
        Names of storage units to add timeseries for. Default None,
        timeseries for all storage units of edisgo_obj are set then.

    """
    if len(edisgo_obj.topology.storage_units_df) == 0:
        edisgo_obj.timeseries.storage_units_active_power = pd.DataFrame(
            {}, index=edisgo_obj.timeseries.timeindex
        )
        edisgo_obj.timeseries.storage_units_reactive_power = pd.DataFrame(
            {}, index=edisgo_obj.timeseries.timeindex
        )
    else:
        if storage_names is None:
            storage_names = edisgo_obj.topology.storage_units_df.index
        storage_df = edisgo_obj.topology.storage_units_df.loc[
            storage_names, ["bus", "p_nom"]
        ]

        # check that all storage units have bus, nominal power
        check_storage = storage_df.isnull().any(axis=1)
        if check_storage.any():
            raise AttributeError(
                "The following storage units have either missing bus or "
                "nominal power: {}.".format(
                    check_storage[check_storage].index.values
                )
            )

        # active power
        # get worst case configurations
        worst_case_scale_factors = edisgo_obj.config["worst_case_scale_factor"]

        # get worst case scaling factors for feed-in/load case
        worst_case_ts = pd.DataFrame(
            np.transpose(
                [
                    [
                        worst_case_scale_factors["{}_storage".format(mode)]
                        for mode in modes
                    ]
                ]
                * len(storage_df)
            ),
            index=edisgo_obj.timeseries.timeindex,
            columns=storage_df.index,
        )
        edisgo_obj.timeseries.storage_units_active_power = drop_duplicated_columns(
            pd.concat(
                [
                    edisgo_obj.timeseries.storage_units_active_power,
                    (worst_case_ts * storage_df.p_nom),
                ],
                axis=1,
            ),
            keep="last",
        )

        q_control._set_reactive_power_time_series_for_fixed_cosphi_using_config(
            edisgo_obj=edisgo_obj,
            df=storage_df,
            component_type="storage_units",
        )


def _check_timeindex(edisgo_obj):
    """
    Check function to check if all feed-in and load time series contain
    values for the specified time index.

    """
    try:
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.generators_reactive_power.index
        ).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.generators_active_power.index
        ).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.loads_reactive_power.index
        ).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.loads_active_power.index
        ).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.storage_units_reactive_power.index
        ).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.storage_units_active_power.index
        ).all()
    except:
        message = "Time index of feed-in and load time series does not match."
        logging.error(message)
        raise KeyError(message)


def _drop_component_time_series(obj, df_name, comp_names):
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
                getattr(
                    obj, df_name
                ).columns.isin(comp_names)
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
            [
                getattr(obj, df_name),
                ts_new
            ],
            axis=1,
        ),
    )


def check_timeseries_for_index_and_cols(edisgo_obj, timeseries, component_names):
    """
    Checks index and column names of inserted timeseries to make sure, they
    have the right format.

    Parameters
    ----------
    timeseries:  :pandas:`pandas.DataFrame<dataframe>`
        inserted timeseries
    component_names: list of str
        names of components of which timeseries are to be added
    """
    if (~edisgo_obj.timeseries.timeindex.isin(timeseries.index)).any():
        raise ValueError(
            "Inserted timeseries for the following "
            "components have the a wrong time index: "
            "{}. Values are missing.".format(component_names)
        )
    if any(comp not in timeseries.columns for comp in component_names):
        raise ValueError(
            "Columns of inserted timeseries are not the same "
            "as names of components to be added. Timeseries "
            "for the following components were tried to be "
            "added: {}".format(component_names)
        )


def _reactive_power_factor_and_mode_default(comp_df, component_type, configs):
    """
    Gets default values for sign of reactive power and power factor for each component.

    Parameters
    -----------
    comp_df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with component names (in the index) of all components
        reactive power factor and mode needs to be set. Only required column is
        column 'voltage_level', giving the voltage level the component is in (the
        voltage level can be set using the function
        :func:`~.tools.tools.assign_voltage_level_to_component`).
        All components must have the same `component_type`.
    component_type : str
        The component type determines the reactive power factor and mode used.
        Possible options are 'generators', 'storage_units' and 'loads'.
    configs : :class:`~.tools.config.Config`
        eDisGo configuration data.

    Returns
    --------
    (:pandas:`pandas.Series<Series>`, :pandas:`pandas.Series<Series>`)
        Series with sign of reactive power (positive or negative) and series with
        reactive power factor are returned.

    """
    # get default configurations
    reactive_power_mode = configs["reactive_power_mode"]
    reactive_power_factor = configs["reactive_power_factor"]

    # write series with sign of reactive power and power factor for each component
    q_sign = pd.Series(index=comp_df.index)
    power_factor = pd.Series(index=comp_df.index)
    if component_type == "generators":
        get_q_sign = q_control.get_q_sign_generator
        comp = "gen"
    elif component_type == "storage_units":
        get_q_sign = q_control.get_q_sign_generator
        comp = "storage"
    elif component_type == "loads":
        get_q_sign = q_control.get_q_sign_load
        comp = "load"
    else:
        raise ValueError(
            "Given 'component_type' is not valid. Valid options are "
            "'generators','storage_units' and 'loads'.")
    for voltage_level in comp_df.voltage_level.unique():
        cols = comp_df.index[comp_df.voltage_level == voltage_level]
        if len(cols) > 0:
            q_sign[cols] = get_q_sign(
                reactive_power_mode[
                    "{}_{}".format(voltage_level, comp)
                ]
            )
            power_factor[cols] = reactive_power_factor[
                "{}_{}".format(voltage_level, comp)
            ]
    return q_sign, power_factor
