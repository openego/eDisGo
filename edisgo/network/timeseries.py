import logging
import pandas as pd
import numpy as np
import datetime

from workalendar.europe import Germany
from demandlib import bdew as bdew, particular_profiles as profiles

from edisgo.io.timeseries_import import import_feedin_timeseries

logger = logging.getLogger('edisgo')


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
    `timeseries` getter in :class:`~.network.components.Generator`,
    :class:`~.network.components.GeneratorFluctuating` and
    :class:`~.network.components.Load`.

    """

    def __init__(self, **kwargs):

        self._timeindex = kwargs.get('timeindex', None)
        self._generators_active_power = kwargs.get(
            'generators_active_power', None)
        self._generators_reactive_power = kwargs.get(
            'generators_reactive_power', None)
        self._loads_active_power = kwargs.get(
            'loads_active_power', None)
        self._loads_reactive_power = kwargs.get(
            'loads_reacitve_power', None)
        self._storage_units_active_power = kwargs.get(
            'storage_units_active_power', None)
        self._storage_units_reactive_power = kwargs.get(
            'storage_units_reacitve_power', None)
        self._curtailment = kwargs.get('curtailment', None)

    @property
    def generators_active_power(self):
        """
        #ToDo docstring
        Get generation time series of dispatchable generators (only active
        power)

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._generators_active_power is None:
            return None
        else:
            try:
                return self._generators_active_power.loc[[self.timeindex], :]
            except:
                return self._generators_active_power.loc[self.timeindex, :]

    @generators_active_power.setter
    def generators_active_power(self, generators_active_power_ts):
        self._generators_active_power = generators_active_power_ts

    @property
    def generators_reactive_power(self):
        """
        #ToDo docstring
        Get reactive power time series for generators normalized by nominal
        active power.

        Returns
        -------
        :pandas: `pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._generators_reactive_power is None:
            return None
        else:
            try:
                return self._generators_reactive_power.loc[[self.timeindex], :]
            except:
                return self._generators_reactive_power.loc[self.timeindex, :]

    @generators_reactive_power.setter
    def generators_reactive_power(self, generators_reactive_power_ts):
        self._generators_reactive_power = generators_reactive_power_ts

    @property
    def loads_active_power(self):
        """
        #ToDo docstring
        Get load time series (only active power)

        Returns
        -------
        dict or :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._loads_active_power is None:
            return None
        else:
            try:
                return self._loads_active_power.loc[[self.timeindex], :]
            except:
                return self._loads_active_power.loc[self.timeindex, :]

    @loads_active_power.setter
    def loads_active_power(self, loads_active_power_ts):
        self._loads_active_power = loads_active_power_ts

    @property
    def loads_reactive_power(self):
        """
        #ToDo docstring
        Get reactive power time series for load normalized by annual
        consumption.

        Returns
        -------
        :pandas: `pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._loads_reactive_power is None:
            return None
        else:
            try:
                return self._loads_reactive_power.loc[[self.timeindex], :]
            except:
                return self._loads_reactive_power.loc[self.timeindex, :]

    @loads_reactive_power.setter
    def loads_reactive_power(self, loads_reactive_power_ts):
        self._loads_reactive_power = loads_reactive_power_ts

    @property
    def storage_units_active_power(self):
        """
        #ToDo docstring
        Get load time series (only active power)

        Returns
        -------
        dict or :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._storage_units_active_power is None:
            return None
        else:
            try:
                return self._storage_units_active_power.loc[[self.timeindex],
                       :]
            except:
                return self._storage_units_active_power.loc[self.timeindex, :]

    @storage_units_active_power.setter
    def storage_units_active_power(self, storage_units_active_power_ts):
        self._storage_units_active_power = storage_units_active_power_ts

    @property
    def storage_units_reactive_power(self):
        """
        #ToDo docstring
        Get reactive power time series for load normalized by annual
        consumption.

        Returns
        -------
        :pandas: `pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._storage_units_reactive_power is None:
            return None
        else:
            try:
                return self._storage_units_reactive_power.loc[[self.timeindex],
                ]
            except:
                return self._storage_units_reactive_power.loc[self.timeindex, :]

    @storage_units_reactive_power.setter
    def storage_units_reactive_power(self, storage_units_reactive_power_ts):
        self._storage_units_reactive_power = storage_units_reactive_power_ts

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

        Residual load is calculated from total (load - generation) in the network.
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
        residual_load = self.generators_active_power.sum(axis=1) - \
                        self.loads_active_power.sum(axis=1)

        return residual_load.apply(
            lambda _: 'feedin_case' if _ < 0 else 'load_case')


class TimeSeriesControl:
    """
    Sets up TimeSeries Object.

    Parameters
    ----------
    network : :class:`~.network.topology.Topology`
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

    def __init__(self, edisgo_obj, **kwargs):

        self.edisgo_obj = edisgo_obj
        mode = kwargs.get('mode', None)

        if mode:
            if mode == 'worst-case':
                modes = ['feedin_case', 'load_case']
            elif mode == 'worst-case-feedin' or mode == 'worst-case-load':
                modes = ['{}_case'.format(mode.split('-')[-1])]
            else:
                raise ValueError('{} is not a valid mode.'.format(mode))

            # set random timeindex
            self.edisgo_obj.timeseries._timeindex = pd.date_range(
                '1/1/1970', periods=len(modes), freq='H')
            self._worst_case_generation(modes)
            self._worst_case_load(modes)

        else:
            config_data = edisgo_obj.config
            weather_cell_ids = edisgo_obj.topology.mv_grid.weather_cells
            # feed-in time series of fluctuating renewables
            ts = kwargs.get('timeseries_generation_fluctuating', None)
            if isinstance(ts, pd.DataFrame):
                self.edisgo_obj.timeseries.generation_fluctuating = ts
            elif isinstance(ts, str) and ts == 'oedb':
                self.edisgo_obj.timeseries.generation_fluctuating = \
                    import_feedin_timeseries(config_data,
                                             weather_cell_ids)
            else:
                raise ValueError('Your input for '
                                 '"timeseries_generation_fluctuating" is not '
                                 'valid.'.format(mode))
            # feed-in time series for dispatchable generators
            ts = kwargs.get('timeseries_generation_dispatchable', None)
            if isinstance(ts, pd.DataFrame):
                self.edisgo_obj.timeseries.generation_dispatchable = ts
            else:
                # check if there are any dispatchable generators, and
                # throw error if there are
                gens = edisgo_obj.topology.generators_df
                if not (gens.type.isin(['solar', 'wind'])).all():
                    raise ValueError(
                        'Your input for "timeseries_generation_dispatchable" '
                        'is not valid.'.format(mode))
            # reactive power time series for all generators
            ts = kwargs.get('timeseries_generation_reactive_power', None)
            if isinstance(ts, pd.DataFrame):
                self.edisgo_obj.timeseries.generation_reactive_power = ts
            # set time index
            if kwargs.get('timeindex', None) is not None:
                self.edisgo_obj.timeseries._timeindex = kwargs.get('timeindex')
            else:
                self.edisgo_obj.timeseries._timeindex = \
                    self.edisgo_obj.timeseries.generation_fluctuating.index

            # load time series
            ts = kwargs.get('timeseries_load', None)
            if isinstance(ts, pd.DataFrame):
                self.edisgo_obj.timeseries.load = ts
            elif ts == 'demandlib':
                self.edisgo_obj.timeseries.load = import_load_timeseries(
                    config_data, ts,
                    year=self.edisgo_obj.timeseries.timeindex[0].year)
            else:
                raise ValueError('Your input for "timeseries_load" is not '
                                 'valid.'.format(mode))
            # reactive power timeseries for loads
            ts = kwargs.get('timeseries_load_reactive_power', None)
            if isinstance(ts, pd.DataFrame):
                self.edisgo_obj.timeseries.load_reactive_power = ts

            # create generator active and reactive power timeseries
            self._generation_from_timeseries()

            # create load active and reactive power timeseries
            self._load_from_timeseries()

            # check if time series for the set time index can be obtained
            self._check_timeindex()

    def _load_from_timeseries(self):
        # get all loads and set active power
        loads = self.edisgo_obj.topology.loads_df
        self.edisgo_obj.timeseries.loads_active_power = \
            loads.apply(lambda x: self.edisgo_obj.timeseries.load[x.sector] *
                                  x.annual_consumption, axis=1).T
        # if reactive power is given as attribute set with inserted timeseries
        if hasattr(self.edisgo_obj.timeseries, 'load_reactive_power'):
            self.edisgo_obj.timeseries.loads_reactive_power = \
                loads.apply(
                    lambda x: self.edisgo_obj.timeseries.load_reactive_power
                              [x.sector] * x.annual_consumption, axis=1).T
        # set default reactive load
        else:
            # assign voltage level to loads
            loads['voltage_level'] = loads.apply(
                lambda _: 'lv' if self.edisgo_obj.topology.buses_df.at[
                                      _.bus, 'v_nom'] < 1
                else 'mv', axis=1)
            self._reactive_power_load_by_cos_phi(loads)

    def _generation_from_timeseries(self):
        # get all generators
        gens = self.edisgo_obj.topology.generators_df
        # handling of fluctuating generators
        gens_fluctuating = gens[gens.type.isin(['solar', 'wind'])]
        self.edisgo_obj.timeseries.generators_active_power = pd.concat(
            [gens_fluctuating.apply(lambda x:
                self.edisgo_obj.timeseries.generation_fluctuating[x.type]
                [x.weather_cell_id].T*x.p_nom, axis=1).T,
            self.edisgo_obj.timeseries.generation_dispatchable], axis=1)
        # set reactive power if given as attribute
        if hasattr(self.edisgo_obj.timeseries, 'generation_reactive_power'):
            gens_dispatchable = gens[~gens.index.isin(gens_fluctuating.index)]
            self.edisgo_obj.timeseries.generators_reactive_power = pd.concat([
                gens_fluctuating.apply(lambda x:
                self.edisgo_obj.timeseries.generation_reactive_power[x.type]
                [x.weather_cell_id]*x.p_nom, axis=1),
                gens_dispatchable.apply(lambda x:
                self.edisgo_obj.timeseries.generation_reactive_power[x.type]
                                         * x.p_nom, axis=1)], axis=1)
        # set default reactive power by cos_phi
        else:
            self._reactive_power_gen_by_cos_phi(gens)

    def _worst_case_generation(self, modes):
        """
        #ToDo: docstring
        Define worst case generation time series for fluctuating and
        dispatchable generators.

        Overwrites active and reactive power time series of generators

        Parameters
        ----------
        modes : list
            List with worst-cases to generate time series for. Can be
            'feedin_case', 'load_case' or both.

        """
        gens_df = self.edisgo_obj.topology.generators_df.loc[
                  :, ['bus', 'type', 'p_nom']]

        # check that all generators have bus, type, nominal power
        check_gens = gens_df.isnull().any(axis=1)
        if check_gens.any():
            raise AttributeError(
                "The following generators have either missing bus, type or "
                "nominal power: {}.".format(
                    check_gens[check_gens].index.values))

        # active power
        # get worst case configurations
        worst_case_scale_factors = self.edisgo_obj.config[
            'worst_case_scale_factor']

        # get worst case scaling factors for different generator types and
        # feed-in/load case
        worst_case_ts = pd.DataFrame(
            {'solar': [worst_case_scale_factors[
                           '{}_feedin_pv'.format(mode)] for mode in modes],
             'other': [worst_case_scale_factors[
                           '{}_feedin_other'.format(mode)] for mode in modes]
             },
            index=self.edisgo_obj.timeseries.timeindex)

        gen_ts = pd.DataFrame(index=self.edisgo_obj.timeseries.timeindex,
                              columns=gens_df.index, dtype='float64')
        # assign normalized active power time series to solar generators
        cols = gen_ts[gens_df.index[gens_df.type == 'solar']].columns
        if len(cols) > 0:
            gen_ts[cols] = pd.concat(
                [worst_case_ts.loc[:, ['solar']]] * len(cols), axis=1)
        # assign normalized active power time series to other generators
        cols = gen_ts[gens_df.index[gens_df.type != 'solar']].columns
        if len(cols) > 0:
            gen_ts[cols] = pd.concat(
                [worst_case_ts.loc[:, ['other']]] * len(cols), axis=1)

        # multiply normalized time series by nominal power of generator
        self.edisgo_obj.timeseries.generators_active_power = gen_ts.mul(
            gens_df.p_nom)

        # calculate reactive power
        self._reactive_power_gen_by_cos_phi(gens_df)

    def _reactive_power_gen_by_cos_phi(self, gens_df):
        # reactive power
        # assign voltage level to generators
        gens_df['voltage_level'] = gens_df.apply(
            lambda _: 'lv'
            if self.edisgo_obj.topology.buses_df.at[_.bus, 'v_nom'] < 1
            else 'mv', axis=1)
        # write dataframes with sign of reactive power and power factor
        # for each generator
        q_sign = pd.Series(index=gens_df.index)
        power_factor = pd.Series(index=gens_df.index)
        for voltage_level in ['mv', 'lv']:
            cols = gens_df.index[gens_df.voltage_level == voltage_level]
            if len(cols) > 0:
                q_sign[cols] = self._get_q_sign_generator(
                    self.edisgo_obj.config['reactive_power_mode'][
                        '{}_gen'.format(voltage_level)])
                power_factor[cols] = self.edisgo_obj.config[
                    'reactive_power_factor']['{}_gen'.format(voltage_level)]

        # calculate reactive power time series for each generator
        self.edisgo_obj.timeseries.generators_reactive_power = \
            self._fixed_cosphi(
                self.edisgo_obj.timeseries.generators_active_power,
                q_sign, power_factor)

    def _worst_case_load(self, modes):
        """
        #ToDo: docstring
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
        voltage_levels = ['mv', 'lv']

        loads_df = self.edisgo_obj.topology.loads_df.loc[
                   :, ['bus', 'sector', 'peak_load']]

        # check that all loads have bus, sector, annual consumption
        check_loads = loads_df.isnull().any(axis=1)
        if check_loads.any():
            raise AttributeError(
                "The following loads have either missing bus, sector or "
                "annual consumption: {}.".format(
                    check_loads[check_loads].index.values))

        # assign voltage level to loads
        loads_df['voltage_level'] = loads_df.apply(
            lambda _: 'lv' if self.edisgo_obj.topology.buses_df.at[
                                  _.bus, 'v_nom'] < 1
            else 'mv', axis=1)

        # active power
        # get worst case configurations
        worst_case_scale_factors = self.edisgo_obj.config[
            'worst_case_scale_factor']

        # get power scaling factors for different voltage levels and feed-in/
        # load case
        power_scaling = {}
        for voltage_level in voltage_levels:
            power_scaling[voltage_level] = [
                worst_case_scale_factors['{}_{}_load'.format(
                    voltage_level, mode)] for mode in modes]

        # assign power scaling factor to each load
        power_scaling_df = pd.DataFrame(data=np.transpose(
            [power_scaling[loads_df.at[col, 'voltage_level']] for col in
             loads_df.index]),
                     index=self.edisgo_obj.timeseries.timeindex,
                     columns=loads_df.index)

        # calculate active power of loads
        self.edisgo_obj.timeseries.loads_active_power = \
            power_scaling_df * loads_df.loc[:, 'peak_load']

        self._reactive_power_load_by_cos_phi(loads_df)

    def _reactive_power_load_by_cos_phi(self, loads_df):
        # reactive power
        # get default configurations
        reactive_power_mode = self.edisgo_obj.config['reactive_power_mode']
        reactive_power_factor = self.edisgo_obj.config[
            'reactive_power_factor']
        voltage_levels = loads_df.voltage_level.unique()
        # write dataframes with sign of reactive power and power factor
        # for each load
        q_sign = pd.Series(index=self.edisgo_obj.topology.loads_df.index)
        power_factor = pd.Series(index=self.edisgo_obj.topology.loads_df.index)
        for voltage_level in voltage_levels:
            cols = loads_df.index[loads_df.voltage_level == voltage_level]
            if len(cols) > 0:
                q_sign[cols] = self._get_q_sign_load(
                    reactive_power_mode['{}_load'.format(voltage_level)])
                power_factor[cols] = reactive_power_factor[
                    '{}_load'.format(voltage_level)]

        # calculate reactive power time series for each load
        self.edisgo_obj.timeseries.loads_reactive_power = self._fixed_cosphi(
            self.edisgo_obj.timeseries.loads_active_power,
            q_sign, power_factor)

    def _get_q_sign_generator(self, reactive_power_mode):
        """
        Get the sign of reactive power in generator sign convention.

        In the generator sign convention the reactive power is negative in
        inductive operation (`reactive_power_mode` is 'inductive') and positive
        in capacitive operation (`reactive_power_mode` is 'capacitive').

        Parameters
        ----------
        reactive_power_mode : str
            Possible options are 'inductive' and 'capacitive'.

        Returns
        --------
        int
            Sign of reactive power in generator sign convention.

        """
        if reactive_power_mode.lower() == 'inductive':
            return -1
        elif reactive_power_mode.lower() == 'capacitive':
            return 1
        else:
            raise ValueError("reactive_power_mode must either be 'capacitive' "
                             "or 'inductive' but is {}.".format(
                                reactive_power_mode))

    def _get_q_sign_load(self, reactive_power_mode):
        """
        Get the sign of reactive power in load sign convention.

        In the load sign convention the reactive power is positive in
        inductive operation (`reactive_power_mode` is 'inductive') and negative
        in capacitive operation (`reactive_power_mode` is 'capacitive').

        Parameters
        ----------
        reactive_power_mode : str
            Possible options are 'inductive' and 'capacitive'.

        Returns
        --------
        int
            Sign of reactive power in load sign convention.

        """
        if reactive_power_mode.lower() == 'inductive':
            return 1
        elif reactive_power_mode.lower() == 'capacitive':
            return -1
        else:
            raise ValueError("reactive_power_mode must either be 'capacitive' "
                             "or 'inductive' but is {}.".format(
                                reactive_power_mode))

    def _fixed_cosphi(self, active_power, q_sign, power_factor):
        """
        Calculates reactive power for a fixed cosphi operation.

        Parameters
        ----------
        active_power : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with active power time series.
        q_sign : int
            `q_sign` defines whether the reactive power is positive or
            negative and must either be -1 or +1.
        power_factor :
            Ratio of real to apparent power.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe with the same format as the `active_power` dataframe,
            containing the reactive power.

        """
        return active_power * q_sign * np.tan(np.arccos(power_factor))

    # Generator
    # @property
    # def timeseries(self):
    #     """
    #     Feed-in time series of generator
    #
    #     It returns the actual dispatch time series used in power flow analysis.
    #     If :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
    #     :meth:`timeseries` looks for time series of the according type of
    #     technology in :class:`~.network.network.TimeSeries`. If the reactive
    #     power time series is provided through :attr:`_timeseries_reactive`,
    #     this is added to :attr:`_timeseries`. When :attr:`_timeseries_reactive`
    #     is not set, the reactive power is also calculated in
    #     :attr:`_timeseries` using :attr:`power_factor` and
    #     :attr:`reactive_power_mode`. The :attr:`power_factor` determines the
    #     magnitude of the reactive power based on the power factor and active
    #     power provided and the :attr:`reactive_power_mode` determines if the
    #     reactive power is either consumed (inductive behaviour) or provided
    #     (capacitive behaviour).
    #
    #     Returns
    #     -------
    #     :pandas:`pandas.DataFrame<dataframe>`
    #         DataFrame containing active power in kW in column 'p' and
    #         reactive power in kvar in column 'q'.
    #
    #     """
    #     if self._timeseries is None:
    #         # calculate time series for active and reactive power
    #         try:
    #             timeseries = \
    #                 self.network.network.timeseries.generation_dispatchable[
    #                     self.type].to_frame('p')
    #         except KeyError:
    #             try:
    #                 timeseries = \
    #                     self.network.network.timeseries.generation_dispatchable[
    #                         'other'].to_frame('p')
    #             except KeyError:
    #                 logger.exception("No time series for type {} "
    #                                  "given.".format(self.type))
    #                 raise
    #
    #         timeseries = timeseries * self.nominal_capacity
    #         if self.timeseries_reactive is not None:
    #             timeseries['q'] = self.timeseries_reactive
    #         else:
    #             timeseries['q'] = timeseries['p'] * self.q_sign * tan(acos(
    #                 self.power_factor))
    #
    #         return timeseries
    #     else:
    #         return self._timeseries.loc[
    #                self.network.network.timeseries.timeindex, :]

    # @property
    # def timeseries_reactive(self):
    #     """
    #     Reactive power time series in kvar.
    #
    #     Parameters
    #     -----------
    #     timeseries_reactive : :pandas:`pandas.Seriese<series>`
    #         Series containing reactive power in kvar.
    #
    #     Returns
    #     -------
    #     :pandas:`pandas.Series<series>` or None
    #         Series containing reactive power time series in kvar. If it is not
    #         set it is tried to be retrieved from `generation_reactive_power`
    #         attribute of global TimeSeries object. If that is not possible
    #         None is returned.
    #
    #     """
    #     if self._timeseries_reactive is None:
    #         if self.network.network.timeseries.generation_reactive_power \
    #                 is not None:
    #             try:
    #                 timeseries = \
    #                     self.network.network.timeseries.generation_reactive_power[
    #                         self.type].to_frame('q')
    #             except (KeyError, TypeError):
    #                 try:
    #                     timeseries = \
    #                         self.network.network.timeseries.generation_reactive_power[
    #                             'other'].to_frame('q')
    #                 except:
    #                     logger.warning(
    #                         "No reactive power time series for type {} given. "
    #                         "Reactive power time series will be calculated from "
    #                         "assumptions in config files and active power "
    #                         "timeseries.".format(self.type))
    #                     return None
    #             self.power_factor = 'not_applicable'
    #             self.reactive_power_mode = 'not_applicable'
    #             return timeseries * self.nominal_capacity
    #         else:
    #             return None
    #     else:
    #         return self._timeseries_reactive.loc[
    #                self.network.network.timeseries.timeindex, :]


    # @property
    # def power_factor(self):
    #     """
    #     Power factor of generator
    #
    #     Parameters
    #     -----------
    #     power_factor : :obj:`float`
    #         Ratio of real power to apparent power.
    #
    #     Returns
    #     --------
    #     :obj:`float`
    #         Ratio of real power to apparent power. If power factor is not set
    #         it is retrieved from the network config object depending on the
    #         network level the generator is in.
    #
    #     """
    #     if self._power_factor is None:
    #         if isinstance(self.topology, MVGrid):
    #             self._power_factor = self.topology.topology.config[
    #                 'reactive_power_factor']['mv_gen']
    #         elif isinstance(self.topology, LVGrid):
    #             self._power_factor = self.topology.topology.config[
    #                 'reactive_power_factor']['lv_gen']
    #     return self._power_factor

    # Load
    # @property
    # def timeseries(self):
    #     """
    #     Load time series
    #
    #     It returns the actual time series used in power flow analysis. If
    #     :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
    #     :meth:`timeseries()` looks for time series of the according sector in
    #     :class:`~.network.network.TimeSeries` object.
    #
    #     Returns
    #     -------
    #     :pandas:`pandas.DataFrame<dataframe>`
    #         DataFrame containing active power in kW in column 'p' and
    #         reactive power in kVA in column 'q'.
    #
    #     """
    #     if self._timeseries is None:
    #
    #         if isinstance(self.topology, MVGrid):
    #             voltage_level = 'mv'
    #         elif isinstance(self.topology, LVGrid):
    #             voltage_level = 'lv'
    #
    #         ts_total = None
    #         for sector in self.consumption.keys():
    #             consumption = self.consumption[sector]
    #
    #             # check if load time series for MV and LV are differentiated
    #             try:
    #                 ts = self.network.network.timeseries.load[
    #                     sector, voltage_level].to_frame('p')
    #             except KeyError:
    #                 try:
    #                     ts = self.network.network.timeseries.load[
    #                         sector].to_frame('p')
    #                 except KeyError:
    #                     logger.exception(
    #                         "No timeseries for load of type {} "
    #                         "given.".format(sector))
    #                     raise
    #             ts = ts * consumption
    #             ts_q = self.timeseries_reactive
    #             if ts_q is not None:
    #                 ts['q'] = ts_q.q
    #             else:
    #                 ts['q'] = ts['p'] * self.q_sign * tan(
    #                     acos(self.power_factor))
    #
    #             if ts_total is None:
    #                 ts_total = ts
    #             else:
    #                 ts_total.p += ts.p
    #                 ts_total.q += ts.q
    #
    #         return ts_total
    #     else:
    #         return self._timeseries
    #
    # @property
    # def timeseries_reactive(self):
    #     """
    #     Reactive power time series in kvar.
    #
    #     Parameters
    #     -----------
    #     timeseries_reactive : :pandas:`pandas.Seriese<series>`
    #         Series containing reactive power in kvar.
    #
    #     Returns
    #     -------
    #     :pandas:`pandas.Series<series>` or None
    #         Series containing reactive power time series in kvar. If it is not
    #         set it is tried to be retrieved from `load_reactive_power`
    #         attribute of global TimeSeries object. If that is not possible
    #         None is returned.
    #
    #     """
    #     if self._timeseries_reactive is None:
    #         # if normalized reactive power time series are given, they are
    #         # scaled by the annual consumption; if none are given reactive
    #         # power time series are calculated timeseries getter using a given
    #         # power factor
    #         if self.network.network.timeseries.load_reactive_power is not None:
    #             self.power_factor = 'not_applicable'
    #             self.reactive_power_mode = 'not_applicable'
    #             ts_total = None
    #             for sector in self.consumption.keys():
    #                 consumption = self.consumption[sector]
    #
    #                 try:
    #                     ts = self.network.network.timeseries.load_reactive_power[
    #                         sector].to_frame('q')
    #                 except KeyError:
    #                     logger.exception(
    #                         "No timeseries for load of type {} "
    #                         "given.".format(sector))
    #                     raise
    #                 ts = ts * consumption
    #                 if ts_total is None:
    #                     ts_total = ts
    #                 else:
    #                     ts_total.q += ts.q
    #             return ts_total
    #         else:
    #             return None
    #
    #     else:
    #         return self._timeseries_reactive
    #
    # @timeseries_reactive.setter
    # def timeseries_reactive(self, timeseries_reactive):
    #     if isinstance(timeseries_reactive, pd.Series):
    #         self._timeseries_reactive = timeseries_reactive
    #         self._power_factor = 'not_applicable'
    #         self._reactive_power_mode = 'not_applicable'
    #     else:
    #         raise ValueError(
    #             "Reactive power time series of load {} needs to be a pandas "
    #             "Series.".format(repr(self)))

    # @property
    # def power_factor(self):
    #     """
    #     Power factor of load
    #
    #     Parameters
    #     -----------
    #     power_factor : :obj:`float`
    #         Ratio of real power to apparent power.
    #
    #     Returns
    #     --------
    #     :obj:`float`
    #         Ratio of real power to apparent power. If power factor is not set
    #         it is retrieved from the network config object depending on the
    #         network level the load is in.
    #
    #     """
    #     if self._power_factor is None:
    #         if isinstance(self.topology, MVGrid):
    #             self._power_factor = self.topology.topology.config[
    #                 'reactive_power_factor']['mv_load']
    #         elif isinstance(self.topology, LVGrid):
    #             self._power_factor = self.topology.topology.config[
    #                 'reactive_power_factor']['lv_load']
    #     return self._power_factor

    def _check_timeindex(self):
        """
        Check function to check if all feed-in and load time series contain
        values for the specified time index.

        """
        try:
            self.edisgo_obj.timeseries.generators_reactive_power
            self.edisgo_obj.timeseries.generators_active_power
            self.edisgo_obj.timeseries.loads_active_power
            self.edisgo_obj.timeseries.loads_reactive_power
        except:
            message = 'Time index of feed-in and load time series does ' \
                      'not match.'
            logging.error(message)
            raise KeyError(message)

def import_load_timeseries(config_data, data_source, year=2018):
    """
    Import load time series

    Parameters
    ----------
    config_data : dict
        Dictionary containing config data from config files.
    data_source : str
        Specify type of data source. Available data sources are

         * 'demandlib'
            Determine a load time series with the use of the demandlib.
            This calculates standard load profiles for 4 different sectors.

    mv_grid_id : :obj:`str`
        MV grid ID as used in oedb. Provide this if `data_source` is 'oedb'.
        Default: None.
    year : int
        Year for which to generate load time series. Provide this if
        `data_source` is 'demandlib'. Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Load time series

    """

    def _load_timeseries_demandlib(config_data, year):
        """
        Get normalized sectoral load time series

        Time series are normalized to 1 kWh consumption per year

        Parameters
        ----------
        config_data : dict
            Dictionary containing config data from config files.
        year : int
            Year for which to generate load time series.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Load time series

        """

        sectoral_consumption = {'h0': 1, 'g0': 1, 'i0': 1, 'l0': 1}

        cal = Germany()
        holidays = dict(cal.holidays(year))

        e_slp = bdew.ElecSlp(year, holidays=holidays)

        # multiply given annual demand with timeseries
        elec_demand = e_slp.get_profile(sectoral_consumption)

        # Add the slp for the industrial group
        ilp = profiles.IndustrialLoadProfile(e_slp.date_time_index,
                                             holidays=holidays)

        # Beginning and end of workday, weekdays and weekend days, and scaling
        # factors by default
        elec_demand['i0'] = ilp.simple_profile(
            sectoral_consumption['i0'],
            am=datetime.time(config_data['demandlib']['day_start'].hour,
                             config_data['demandlib']['day_start'].minute, 0),
            pm=datetime.time(config_data['demandlib']['day_end'].hour,
                             config_data['demandlib']['day_end'].minute, 0),
            profile_factors=
            {'week': {'day': config_data['demandlib']['week_day'],
                      'night': config_data['demandlib']['week_night']},
             'weekend': {'day': config_data['demandlib']['weekend_day'],
                         'night': config_data['demandlib']['weekend_night']}})

        # Resample 15-minute values to hourly values and sum across sectors
        elec_demand = elec_demand.resample('H').mean()

        return elec_demand

    if data_source == 'demandlib':
        try:
            float(year)
            if year > datetime.date.today().year:
                raise TypeError
        except TypeError:
            year = datetime.date.today().year - 1
            logger.warning('No valid year inserted. Year set to previous year.')
        load = _load_timeseries_demandlib(config_data, year)
        load.rename(columns={'g0': 'retail', 'h0': 'residential',
                             'l0': 'agricultural', 'i0': 'industrial'},
                    inplace=True)
    else:
        raise NotImplementedError
    return load