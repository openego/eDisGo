import logging
import pandas as pd
import numpy as np
import datetime
import os

from workalendar.europe import Germany
from demandlib import bdew as bdew, particular_profiles as profiles

from edisgo.io.timeseries_import import import_feedin_timeseries
from edisgo.tools.tools import drop_duplicated_indices

logger = logging.getLogger('edisgo')


class TimeSeries:
    """
    Defines time series for all loads, generators and storage units in network
    (if set).

    Can also contain time series for loads (sector-specific), generators
    (technology-specific), and curtailment (technology-specific).

    Attributes
    ----------
    timeindex : :pandas:`pandas.DatetimeIndex<datetimeindex>`, optional
        Can be used to define a time range for which to obtain the provided
        time series and run power flow analysis. Default: None.
    generators_active_power: :pandas:`pandas.DataFrame<dataframe>`, optional
        Active power timeseries of all generators in topology. Index of
        DataFrame has to contain timeindex and column names are names of
        generators.
    generators_reactive_power: :pandas:`pandas.DataFrame<dataframe>`, optional
        Reactive power timeseries of all generators in topology. Format is the
        same as for generators_active power.
    loads_active_power: :pandas:`pandas.DataFrame<dataframe>`, optional
        Active power timeseries of all loads in topology. Index of DataFrame
        has to contain timeindex and column names are names of loads.
    loads_reactive_power: :pandas:`pandas.DataFrame<dataframe>`, optional
        Reactive power timeseries of all loads in topology. Format is the
        same as for loads_active power.
    storage_units_active_power: :pandas:`pandas.DataFrame<dataframe>`, optional
        Active power timeseries of all storage units in topology. Index of
        DataFrame has to contain timeindex and column names are names of
        storage units.
    storage_units_reactive_power: :pandas:`pandas.DataFrame<dataframe>`, optional
        Reactive power timeseries of all storage_units in topology. Format is
        the same as for storage_units_active power.
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

    Notes
    -----
    Can also hold the following attributes when specific mode of
    :meth:`get_component_timeseries` is called: mode, generation_fluctuating,
    generation_dispatchable, generation_reactive_power, load,
    load_reactive_power. See description of meth:`get_component_timeseries` for
    format of these.

    See also
    --------
    `timeseries` getter in :class:`~.network.components.Generator`,
    :class:`~.network.components.GeneratorFluctuating` and
    :class:`~.network.components.Load`.

    """

    def __init__(self, **kwargs):

        self._timeindex = kwargs.get('timeindex', None)
        self._generators_active_power = kwargs.get(
            'generators_active_power', pd.DataFrame(index=self.timeindex))
        self._generators_reactive_power = kwargs.get(
            'generators_reactive_power', pd.DataFrame(index=self.timeindex))
        self._loads_active_power = kwargs.get(
            'loads_active_power', pd.DataFrame(index=self.timeindex))
        self._loads_reactive_power = kwargs.get(
            'loads_reactive_power', pd.DataFrame(index=self.timeindex))
        self._storage_units_active_power = kwargs.get(
            'storage_units_active_power', pd.DataFrame(index=self.timeindex))
        self._storage_units_reactive_power = kwargs.get(
            'storage_units_reactive_power', pd.DataFrame(index=self.timeindex))
        self._curtailment = kwargs.get('curtailment', pd.DataFrame(
            index=self.timeindex))

    @property
    def generators_active_power(self):
        """
        Active power timeseries of generators [MW].

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._generators_active_power.empty:
            return self._generators_active_power
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
        Reactive power timeseries of generators in [MVA].

        Returns
        -------
        :pandas: `pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._generators_reactive_power.empty:
            return self._generators_reactive_power
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
        Active power timeseries of loads in [MW].

        Returns
        -------
        dict or :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._loads_active_power.empty:
            return self._loads_active_power
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
        Reactive power timeseries in [MVA].

        Returns
        -------
        :pandas: `pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._loads_reactive_power.empty:
            return self._loads_reactive_power
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
        Active power timeseries of storage units in [MW].

        Returns
        -------
        dict or :pandas:`pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._storage_units_active_power.empty:
            return self._storage_units_active_power
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
        Reactive power timeseries of storage_units in [MVA].

        Returns
        -------
        :pandas: `pandas.DataFrame<dataframe>`
            See class definition for details.

        """
        if self._storage_units_reactive_power.empty:
            return self._storage_units_reactive_power
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
        Time range of power flow analysis

        Returns
        -------
        :pandas:`pandas.DatetimeIndex<datetimeindex>`
            See class definition for details.

        """
        return self._timeindex

    @timeindex.setter
    def timeindex(self, new_index):
        if self._timeindex is not None:
            # check if new time index is subset of existing time index
            if not new_index.isin(self.timeindex).all():
                logger.warning(
                    "Not all time steps of new time index lie within existing "
                    "time index. This may cause problems later on.")
        self._timeindex = new_index

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
    def residual_load(self):
        """
        Returns residual load.

        Residual load for each time step is calculated from total generation
        plus storage active power (discharge is positive) minus total load.
        A positive residual load represents a feed-in case while a negative
        residual load here represents a load case.
        Grid losses are not considered.

        Returns
        -------
        :pandas:`pandas.Series<series>`

            Series with residual load in MW.

        """
        return (self.generators_active_power.sum(axis=1) +
                self.storage_units_active_power.sum(axis=1) -
                self.loads_active_power.sum(axis=1)).loc[self.timeindex]

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

        return self.residual_load.apply(
            lambda _: 'feedin_case' if _ > 0 else 'load_case')

    def to_csv(self, directory):
        """
        Exports topology to csv files with names loads_active_power,
        loads_reactive_power, generators_active_power, generators_reactive_power
        storage_units_active_power, storage_units_reactive_power. A sub-
        folder named "timeseries" is added to the provided directory.

        Parameters
        ----------
        directory: str
            path to save timeseries to
        """
        os.makedirs(directory, exist_ok=True)
        ts_dir = os.path.join(directory, 'timeseries')
        os.makedirs(ts_dir, exist_ok=True)
        if self.loads_active_power is not None:
            self.loads_active_power.to_csv(
                os.path.join(ts_dir, 'loads_active_power.csv'))
        if self.loads_reactive_power is not None:
            self.loads_reactive_power.to_csv(
                os.path.join(ts_dir, 'loads_reactive_power.csv'))
        if self.generators_active_power is not None:
            self.generators_active_power.to_csv(
                os.path.join(ts_dir, 'generators_active_power.csv'))
        if self.generators_reactive_power is not None:
            self.generators_reactive_power.to_csv(
                os.path.join(ts_dir, 'generators_reactive_power.csv'))
        if self.storage_units_active_power is not None:
            self.storage_units_active_power.to_csv(
                os.path.join(ts_dir, 'storage_units_active_power.csv'))
        if self.storage_units_reactive_power is not None:
            self.storage_units_reactive_power.to_csv(
                os.path.join(ts_dir, 'storage_units_reactive_power.csv'))
        logger.debug("Timeseries exported.")


def get_component_timeseries(edisgo_obj, **kwargs):
    """
    Sets up TimeSeries Object.

    Parameters
    ----------
    edisgo_obj : :class:`~.edisgo.EDisGo`
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

    mode = kwargs.get('mode', None)
    edisgo_obj.timeseries.mode = mode
    _reset_timeseries(edisgo_obj.timeseries)
    if mode:
        if 'worst-case' in mode:
            modes = _get_worst_case_modes(mode)
            # set random timeindex
            edisgo_obj.timeseries._timeindex = pd.date_range(
                '1/1/1970', periods=len(modes), freq='H')
            _worst_case_generation(edisgo_obj=edisgo_obj, modes=modes)
            _worst_case_load(edisgo_obj=edisgo_obj, modes=modes)
            _worst_case_storage(edisgo_obj=edisgo_obj, modes=modes)

        elif mode == 'manual':
            edisgo_obj.timeseries._timeindex = kwargs.get('timeindex',
                                                               None)
            edisgo_obj.timeseries.loads_active_power = \
                kwargs.get('loads_active_power', None)
            edisgo_obj.timeseries.loads_reactive_power = \
                kwargs.get('loads_reactive_power', None)
            edisgo_obj.timeseries.generators_active_power = \
                kwargs.get('generators_active_power', None)
            edisgo_obj.timeseries.generators_reactive_power = \
                kwargs.get('generators_reactive_power', None)
            edisgo_obj.timeseries.storage_units_active_power = \
                kwargs.get('storage_units_active_power', None)
            edisgo_obj.timeseries.storage_units_reactive_power = \
                kwargs.get('storage_units_reactive_power', None)
        else:
            raise ValueError('{} is not a valid mode.'.format(mode))
    else:
        config_data = edisgo_obj.config
        weather_cell_ids = \
            edisgo_obj.topology.generators_df.weather_cell_id.dropna().unique()
        # feed-in time series of fluctuating renewables
        ts = kwargs.get('timeseries_generation_fluctuating', None)
        if isinstance(ts, pd.DataFrame):
            edisgo_obj.timeseries.generation_fluctuating = ts
        elif isinstance(ts, str) and ts == 'oedb':
            edisgo_obj.timeseries.generation_fluctuating = \
                import_feedin_timeseries(config_data,
                                         weather_cell_ids,
                                         kwargs.get('timeindex',
                                                    None))
        else:
            raise ValueError('Your input for '
                             '"timeseries_generation_fluctuating" is not '
                             'valid.'.format(mode))
        # feed-in time series for dispatchable generators
        ts = kwargs.get('timeseries_generation_dispatchable', None)
        if isinstance(ts, pd.DataFrame):
            edisgo_obj.timeseries.generation_dispatchable = ts
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
            edisgo_obj.timeseries.generation_reactive_power = ts
        # set time index
        if kwargs.get('timeindex', None) is not None:
            edisgo_obj.timeseries._timeindex = kwargs.get('timeindex')
        else:
            edisgo_obj.timeseries._timeindex = \
                edisgo_obj.timeseries.generation_fluctuating.index

        # load time series
        ts = kwargs.get('timeseries_load', None)
        if isinstance(ts, pd.DataFrame):
            edisgo_obj.timeseries.load = ts
        elif ts == 'demandlib':
            edisgo_obj.timeseries.load = import_load_timeseries(
                config_data, ts,
                year=edisgo_obj.timeseries.timeindex[0].year)
        else:
            raise ValueError('Your input for "timeseries_load" is not '
                             'valid.'.format(mode))
        # reactive power timeseries for loads
        ts = kwargs.get('timeseries_load_reactive_power', None)
        if isinstance(ts, pd.DataFrame):
            edisgo_obj.timeseries.load_reactive_power = ts

        # create generator active and reactive power timeseries
        _generation_from_timeseries(edisgo_obj=edisgo_obj)

        # create load active and reactive power timeseries
        _load_from_timeseries(edisgo_obj=edisgo_obj)

        # create storage active and reactive power timeseries
        _storage_from_timeseries(edisgo_obj=edisgo_obj,
            ts_active_power=kwargs.get('timeseries_storage_units', None),
            ts_reactive_power=
            kwargs.get('timeseries_storage_units_reactive_power', None))

        # check if time series for the set time index can be obtained
        _check_timeindex(edisgo_obj=edisgo_obj)


def _load_from_timeseries(edisgo_obj, load_names=None):
    # get all requested loads and drop existing timeseries
    if load_names is None:
        load_names = edisgo_obj.topology.loads_df.index
    loads = edisgo_obj.topology.loads_df.loc[load_names]
    _drop_existing_component_timeseries(edisgo_obj=edisgo_obj,
                                        comp_type='loads', comp_names=load_names)
    # set active power
    edisgo_obj.timeseries.loads_active_power = \
        edisgo_obj.timeseries.loads_active_power.T.append(
            loads.apply(
                lambda x: edisgo_obj.timeseries.load[x.sector] *
                          x.annual_consumption, axis=1)).T
    # if reactive power is given as attribute set with inserted timeseries
    if hasattr(edisgo_obj.timeseries, 'load_reactive_power'):
        edisgo_obj.timeseries.loads_reactive_power = \
            edisgo_obj.timeseries.loads_reactive_power.T.append(
            loads.apply(
                lambda x: edisgo_obj.timeseries.load_reactive_power
                          [x.sector] * x.annual_consumption, axis=1)).T
    # set default reactive load
    else:
        # assign voltage level to loads
        loads['voltage_level'] = loads.apply(
            lambda _: 'lv' if edisgo_obj.topology.buses_df.at[
                                  _.bus, 'v_nom'] < 1
            else 'mv', axis=1)
        _reactive_power_load_by_cos_phi(edisgo_obj=edisgo_obj, loads_df=loads)


def _generation_from_timeseries(edisgo_obj, generator_names=None):

    def _timeseries_fluctuating():
        if isinstance(edisgo_obj.timeseries.generation_fluctuating.columns,
                      pd.MultiIndex):
            return gens_fluctuating.apply(
                lambda x: edisgo_obj.timeseries.generation_fluctuating[
                              x.type][x.weather_cell_id].T * x.p_nom, axis=1).T
        else:
            return gens_fluctuating.apply(
                lambda x: edisgo_obj.timeseries.generation_fluctuating[
                              x.type].T * x.p_nom, axis=1).T

    def _timeseries_dispatchable():
        return gens_dispatchable.apply(
            lambda x: edisgo_obj.timeseries.generation_dispatchable[x.type].T *
                      x.p_nom
            if x.type in edisgo_obj.timeseries.generation_dispatchable.columns
            else edisgo_obj.timeseries.generation_dispatchable['other'].T *
                 x.p_nom,
            axis=1).T

    if generator_names is None:
        generator_names = edisgo_obj.topology.generators_df.index
    # get all generators
    gens = edisgo_obj.topology.generators_df.loc[generator_names]
    # drop existing timeseries
    _drop_existing_component_timeseries(edisgo_obj, 'generators',
                                        generator_names)
    # handling of fluctuating generators
    gens_fluctuating = gens[gens.type.isin(['solar', 'wind'])]
    gens_dispatchable = gens[~gens.index.isin(gens_fluctuating.index)]
    if gens_dispatchable.empty and gens_fluctuating.empty:
        logger.debug("No generators provided to add timeseries for.")
        return
    if not gens_dispatchable.empty:
        edisgo_obj.timeseries.generators_active_power = \
            pd.concat([edisgo_obj.timeseries.generators_active_power,
                       _timeseries_dispatchable()], axis=1)
    if not gens_fluctuating.empty:
        edisgo_obj.timeseries.generators_active_power = \
            pd.concat([edisgo_obj.timeseries.generators_active_power,
                       _timeseries_fluctuating()], axis=1)

    # set reactive power if given as attribute
    if hasattr(edisgo_obj.timeseries, 'generation_reactive_power')\
        and gens.index.isin(
            edisgo_obj.timeseries.generation_reactive_power.columns)\
            .all():

        edisgo_obj.timeseries.generators_reactive_power = \
            edisgo_obj.timeseries.generators_reactive_power.T.append(
                edisgo_obj.timeseries.generation_reactive_power.loc[
                    :, gens.index].T).T
    # set default reactive power by cos_phi
    else:
        logger.debug("Reactive power calculated by cos(phi).")
        _reactive_power_gen_by_cos_phi(edisgo_obj=edisgo_obj, gens_df=gens)


def _storage_from_timeseries(edisgo_obj,  ts_active_power, ts_reactive_power,
                             name_storage_units=None):
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
        name_storage_units = \
            edisgo_obj.topology.storage_units_df.index
    storage_units_df = \
        edisgo_obj.topology.storage_units_df.loc[name_storage_units]
    _drop_existing_component_timeseries(edisgo_obj, 'storage_units',
                                        name_storage_units)

    if len(storage_units_df) == 0:
        edisgo_obj.timeseries.storage_units_active_power = \
            pd.DataFrame({}, index=edisgo_obj.timeseries.timeindex)
        edisgo_obj.timeseries.storage_units_reactive_power = \
            pd.DataFrame({}, index=edisgo_obj.timeseries.timeindex)
    elif ts_active_power is None:
        # Todo: move up to check at the start
        raise ValueError("No timeseries for storage units provided.")
    else:
        try:
            # check if indices and columns are correct
            if (ts_active_power.index == \
                    edisgo_obj.timeseries.timeindex).all():
                edisgo_obj.timeseries.storage_units_active_power = \
                    drop_duplicated_indices(
                        edisgo_obj.timeseries.
                        storage_units_active_power.T.append(
                            ts_active_power.loc[:, name_storage_units].T)).T
                # check if reactive power is given
                if ts_reactive_power is not None and \
                    (ts_active_power.index == \
                        edisgo_obj.timeseries.timeindex).all():
                    edisgo_obj.timeseries.storage_units_reactive_power = \
                        drop_duplicated_indices(
                            edisgo_obj.timeseries.
                            storage_units_reactive_power.T.append(
                                ts_reactive_power.loc[:, name_storage_units].T)).T
                else:
                    _reactive_power_storage_by_cos_phi(edisgo_obj=edisgo_obj,
                        storage_units_df=storage_units_df)
            else:
                raise ValueError("Index of provided storage active power "
                                 "timeseries does not match timeindex of "
                                 "TimeSeries class.")
        except ValueError:
            raise ValueError("Columns or indices of inserted storage "
                             "timeseries do not match topology and "
                             "timeindex.")


def _worst_case_generation(edisgo_obj, modes, generator_names=None):
    """
    Define worst case generation time series for fluctuating and
    dispatchable generators.

    Overwrites active and reactive power time series of generators

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    modes : list
        List with worst-cases to generate time series for. Can be
        'feedin_case', 'load_case' or both.
    generator_names: str or list of str
        Names of generators to add timeseries for. Default None, timeseries
        for all generators of edisgo_obj are set then.
    """
    if generator_names is None:
        generator_names = edisgo_obj.topology.generators_df.index

    gens_df = edisgo_obj.topology.generators_df.loc[
              generator_names, ['bus', 'type', 'p_nom']]

    # check that all generators have bus, type, nominal power
    check_gens = gens_df.isnull().any(axis=1)
    if check_gens.any():
        raise AttributeError(
            "The following generators have either missing bus, type or "
            "nominal power: {}.".format(
                check_gens[check_gens].index.values))

    # active power
    # get worst case configurations
    worst_case_scale_factors = edisgo_obj.config[
        'worst_case_scale_factor']

    # get worst case scaling factors for different generator types and
    # feed-in/load case
    worst_case_ts = pd.DataFrame(
        {'solar': [worst_case_scale_factors[
                       '{}_feedin_pv'.format(mode)] for mode in modes],
         'other': [worst_case_scale_factors[
                       '{}_feedin_other'.format(mode)] for mode in modes]
         },
        index=edisgo_obj.timeseries.timeindex)

    gen_ts = pd.DataFrame(index=edisgo_obj.timeseries.timeindex,
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

    # drop existing timeseries
    _drop_existing_component_timeseries(edisgo_obj, 'generators',
                                        generator_names)

    # multiply normalized time series by nominal power of generator
    edisgo_obj.timeseries.generators_active_power = \
        edisgo_obj.timeseries.generators_active_power.T.append(
            gen_ts.mul(gens_df.p_nom).T).T

    # calculate reactive power
    _reactive_power_gen_by_cos_phi(edisgo_obj=edisgo_obj, gens_df=gens_df)


def _reactive_power_gen_by_cos_phi(edisgo_obj, gens_df):
    if gens_df.empty:
        return
    # reactive power
    # assign voltage level to generators
    gens_df['voltage_level'] = gens_df.apply(
        lambda _: 'lv'
        if edisgo_obj.topology.buses_df.at[_.bus, 'v_nom'] < 1
        else 'mv', axis=1)
    # write dataframes with sign of reactive power and power factor
    # for each generator
    q_sign = pd.Series(index=gens_df.index)
    power_factor = pd.Series(index=gens_df.index)
    for voltage_level in ['mv', 'lv']:
        cols = gens_df.index[gens_df.voltage_level == voltage_level]
        if len(cols) > 0:
            q_sign[cols] = _get_q_sign_generator(
                edisgo_obj.config['reactive_power_mode'][
                    '{}_gen'.format(voltage_level)])
            power_factor[cols] = edisgo_obj.config[
                'reactive_power_factor']['{}_gen'.format(voltage_level)]

    # calculate reactive power time series for each generator
    edisgo_obj.timeseries.generators_reactive_power = \
        edisgo_obj.timeseries.generators_reactive_power.T.append(
            _fixed_cosphi(
                edisgo_obj.timeseries.generators_active_power.loc[
                    :, gens_df.index], q_sign, power_factor).T).T


def _worst_case_load(edisgo_obj, modes, load_names=None):
    """
    Define worst case load time series for each sector.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    modes : list
        List with worst-cases to generate time series for. Can be
        'feedin_case', 'load_case' or both.
    load_names: str or list of str
        Names of loads to add timeseries for. Default None, timeseries
        for all loads of edisgo_obj are set then.

    """

    voltage_levels = ['mv', 'lv']

    if load_names is None:
        load_names = edisgo_obj.topology.loads_df.index
    loads_df = edisgo_obj.topology.loads_df.loc[
               load_names, ['bus', 'sector', 'peak_load']]

    # check that all loads have bus, sector, annual consumption
    check_loads = loads_df.isnull().any(axis=1)
    if check_loads.any():
        raise AttributeError(
            "The following loads have either missing bus, sector or "
            "annual consumption: {}.".format(
                check_loads[check_loads].index.values))

    # assign voltage level to loads
    if loads_df.empty:
        return
    loads_df['voltage_level'] = loads_df.apply(
        lambda _: 'lv' if edisgo_obj.topology.buses_df.at[
                              _.bus, 'v_nom'] < 1
        else 'mv', axis=1)

    # active power
    # get worst case configurations
    worst_case_scale_factors = edisgo_obj.config[
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
                 index=edisgo_obj.timeseries.timeindex,
                 columns=loads_df.index)

    # drop existing timeseries
    _drop_existing_component_timeseries(edisgo_obj=edisgo_obj, comp_type='loads',
                                        comp_names=load_names)

    # calculate active power of loads
    edisgo_obj.timeseries.loads_active_power = \
        edisgo_obj.timeseries.loads_active_power.T.append(
            (power_scaling_df * loads_df.loc[:, 'peak_load']).T,
            sort=False).T

    _reactive_power_load_by_cos_phi(edisgo_obj=edisgo_obj, loads_df=loads_df)


def _reactive_power_load_by_cos_phi(edisgo_obj, loads_df):
    # reactive power
    # get default configurations
    reactive_power_mode = edisgo_obj.config['reactive_power_mode']
    reactive_power_factor = edisgo_obj.config[
        'reactive_power_factor']
    voltage_levels = loads_df.voltage_level.unique()
    # write dataframes with sign of reactive power and power factor
    # for each load
    q_sign = pd.Series(index=loads_df.index)
    power_factor = pd.Series(index=loads_df.index)
    for voltage_level in voltage_levels:
        cols = loads_df.index[loads_df.voltage_level == voltage_level]
        if len(cols) > 0:
            q_sign[cols] = _get_q_sign_load(
                reactive_power_mode['{}_load'.format(voltage_level)])
            power_factor[cols] = reactive_power_factor[
                '{}_load'.format(voltage_level)]

    # calculate reactive power time series for each load
    edisgo_obj.timeseries.loads_reactive_power = \
        edisgo_obj.timeseries.loads_reactive_power.T.append(
            _fixed_cosphi(
                edisgo_obj.timeseries.loads_active_power.loc[
                    :, loads_df.index],
                q_sign, power_factor).T, sort=False).T


def _get_q_sign_generator(reactive_power_mode):
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


def _get_q_sign_load(reactive_power_mode):
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


def _fixed_cosphi(active_power, q_sign, power_factor):
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


def _worst_case_storage(edisgo_obj, modes, storage_names=None):
    """
    Define worst case storage unit time series.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    modes : list
        List with worst-cases to generate time series for. Can be
        'feedin_case', 'load_case' or both.
    storage_namess: str or list of str
        Names of storage units to add timeseries for. Default None,
        timeseries for all storage units of edisgo_obj are set then.

    """
    if len(edisgo_obj.topology.storage_units_df) == 0:
        edisgo_obj.timeseries.storage_units_active_power = \
            pd.DataFrame({}, index=edisgo_obj.timeseries.timeindex)
        edisgo_obj.timeseries.storage_units_reactive_power = \
            pd.DataFrame({}, index=edisgo_obj.timeseries.timeindex)
    else:
        if storage_names is None:
            storage_names = edisgo_obj.topology.storage_units_df.index
        storage_df = \
            edisgo_obj.topology.storage_units_df.loc[storage_names,
            ['bus', 'p_nom']]

        # check that all storage units have bus, nominal power
        check_storage = storage_df.isnull().any(axis=1)
        if check_storage.any():
            raise AttributeError(
                "The following storage units have either missing bus or "
                "nominal power: {}.".format(
                    check_storage[check_storage].index.values))

        # active power
        # get worst case configurations
        worst_case_scale_factors = edisgo_obj.config[
            'worst_case_scale_factor']

        # get worst case scaling factors for feed-in/load case
        worst_case_ts = pd.DataFrame(
            np.transpose([[worst_case_scale_factors[
                               '{}_storage'.format(mode)] for mode
                           in modes]] * len(storage_df)),
            index=edisgo_obj.timeseries.timeindex,
            columns=storage_df.index)

        edisgo_obj.timeseries.storage_units_active_power = \
            drop_duplicated_indices(
                edisgo_obj.timeseries.storage_units_active_power.T.
                append((worst_case_ts*edisgo_obj.topology.
                        storage_units_df.p_nom).T), keep='last').T

        _reactive_power_storage_by_cos_phi(edisgo_obj=edisgo_obj,
                                           storage_units_df=storage_df)


def _reactive_power_storage_by_cos_phi(edisgo_obj, storage_units_df):
    # reactive power
    # assign voltage level to storage units
    if storage_units_df.empty:
        return
    storage_units_df['voltage_level'] = storage_units_df.apply(
        lambda _: 'lv'
        if edisgo_obj.topology.buses_df.at[_.bus, 'v_nom'] < 1
        else 'mv', axis=1)
    # write dataframes with sign of reactive power and power factor
    # for each storage unit
    q_sign = pd.Series(index=storage_units_df.index)
    power_factor = pd.Series(index=storage_units_df.index)
    for voltage_level in ['mv', 'lv']:
        cols = storage_units_df.index[storage_units_df.voltage_level
                                      == voltage_level]
        if len(cols) > 0:
            # storage units are handled like generators in pypsa, therefore
            # use same sign convention as for generators
            q_sign[cols] = _get_q_sign_generator(
                edisgo_obj.config['reactive_power_mode'][
                    '{}_storage'.format(voltage_level)])
            power_factor[cols] = edisgo_obj.config[
                'reactive_power_factor']['{}_storage'.format(voltage_level)]

    # calculate reactive power time series for each storage unit
    edisgo_obj.timeseries.storage_units_reactive_power = \
        drop_duplicated_indices(
            edisgo_obj.timeseries.storage_units_reactive_power.T.
            append(_fixed_cosphi(
                edisgo_obj.timeseries.storage_units_active_power.loc[
                    :, storage_units_df.index], q_sign, power_factor).T)).T


def _check_timeindex(edisgo_obj):
    """
    Check function to check if all feed-in and load time series contain
    values for the specified time index.

    """
    try:
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.generators_reactive_power.index).\
            all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.generators_active_power.index).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.loads_reactive_power.index).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.loads_active_power.index).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.storage_units_reactive_power.
            index).all()
        assert edisgo_obj.timeseries.timeindex.isin(
            edisgo_obj.timeseries.storage_units_active_power.index).\
            all()
    except:
        message = 'Time index of feed-in and load time series does ' \
                  'not match.'
        logging.error(message)
        raise KeyError(message)


def add_loads_timeseries(edisgo_obj, load_names, **kwargs):
    """
    Define load time series for active and reactive power. For more information
    on required and optional parameters see description of
    :func:`get_component_timeseries`. The mode initially set within
    get_component_timeseries is used here to set new timeseries. If a different
    mode is required, change edisgo_obj.timeseries.mode to the desired mode and
    provide respective parameters.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    load_names: str or list of str
        Names of loads to add timeseries for. Default None, timeseries
        for all loads of edisgo_obj are set then.

    """
    # If timeseries have not yet been filled, it is not
    # necessary to add timeseries
    if not hasattr(edisgo_obj.timeseries, 'mode'):
        logger.debug('Timeseries have not been set yet. Please call'
                     'get_component_timeseries to create '
                     'timeseries.')
        return
    # turn single name to list
    if isinstance(load_names, str):
        load_names = [load_names]
    # append timeseries of respective mode
    if edisgo_obj.timeseries.mode:
        if 'worst-case' in edisgo_obj.timeseries.mode:
            modes = _get_worst_case_modes(edisgo_obj.timeseries.mode)
            # set random timeindex
            _worst_case_load(edisgo_obj=edisgo_obj, modes=modes,
                             load_names=load_names)
        elif edisgo_obj.timeseries.mode == 'manual':
            loads_active_power = kwargs.get('loads_active_power', None)
            if loads_active_power is not None:
                check_timeseries_for_index_and_cols(edisgo_obj,
                    loads_active_power, load_names)
            loads_reactive_power = kwargs.get('loads_reactive_power', None)
            if loads_reactive_power is not None:
                check_timeseries_for_index_and_cols(edisgo_obj,
                    loads_reactive_power, load_names)
            _drop_existing_component_timeseries(
                edisgo_obj=edisgo_obj, comp_type='loads',comp_names=load_names)
            # add new load timeseries
            edisgo_obj.timeseries.loads_active_power = \
                edisgo_obj.timeseries.loads_active_power.T.append(
                    loads_active_power.T.loc[load_names]).T
            edisgo_obj.timeseries.loads_reactive_power = \
                edisgo_obj.timeseries.loads_reactive_power.T.append(
                    loads_reactive_power.T.loc[load_names]).T
        else:
            raise ValueError('{} is not a valid mode.'.format(
                edisgo_obj.timeseries.mode))
    else:
        # create load active and reactive power timeseries
        _load_from_timeseries(edisgo_obj=edisgo_obj, load_names=load_names)


def add_generators_timeseries(edisgo_obj, generator_names, **kwargs):
    """
    Define generator time series for active and reactive power. For more
    information on required and optional parameters see description of
    :func:`get_component_timeseries`.The mode initially set within
    get_component_timeseries is used here to set new timeseries. If a different
    mode is required, change edisgo_obj.timeseries.mode to the desired mode and
    provide respective parameters.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    generator_names: str or list of str
        Names of generators to add timeseries for. Default None, timeseries
        for all generators of edisgo_obj are set then.

    """
    # If timeseries have not been set yet, it is not
    # necessary to add timeseries
    if not hasattr(edisgo_obj.timeseries, 'mode'):
        logger.debug('Timeseries have not been set yet. Please call '
                     'get_component_timeseries to create '
                     'timeseries.')
        return
    # turn single name to list
    if isinstance(generator_names, str):
        generator_names = [generator_names]
    # append timeseries of respective mode
    if edisgo_obj.timeseries.mode:
        if 'worst-case' in edisgo_obj.timeseries.mode:
            modes = _get_worst_case_modes(edisgo_obj.timeseries.mode)
            # set random timeindex
            _worst_case_generation(edisgo_obj=edisgo_obj,
                                   modes=modes,
                                   generator_names=generator_names)
        elif edisgo_obj.timeseries.mode == 'manual':
            # check inserted timeseries and drop existing generators
            gens_active_power = kwargs.get('generators_active_power', None)
            if gens_active_power is not None:
                check_timeseries_for_index_and_cols(edisgo_obj,
                                                    gens_active_power,
                                                    generator_names)
            gens_reactive_power = kwargs.get('generators_reactive_power',
                                             None)
            if gens_reactive_power is not None:
                check_timeseries_for_index_and_cols(edisgo_obj,
                    gens_reactive_power, generator_names)
            _drop_existing_component_timeseries(edisgo_obj, 'generators',
                                                generator_names)
            # add new timeseries
            edisgo_obj.timeseries.generators_active_power = \
                edisgo_obj.timeseries.generators_active_power.T.\
                append(gens_active_power.T.loc[generator_names]).T
            edisgo_obj.timeseries.generators_reactive_power = \
                edisgo_obj.timeseries.generators_reactive_power.T.\
                append(gens_reactive_power.T.loc[generator_names]).T
        else:
            raise ValueError('{} is not a valid mode.'.format(
                edisgo_obj.timeseries.mode))
    else:
        ts_dispatchable = kwargs.get('timeseries_generation_dispatchable',
                                     None)
        if ts_dispatchable is not None:
            if hasattr(edisgo_obj.timeseries,
                       'generation_dispatchable'):
                edisgo_obj.timeseries.generation_dispatchable = \
                    drop_duplicated_indices(edisgo_obj.timeseries.
                        generation_dispatchable.T.append(
                        ts_dispatchable.T), keep='last').T
            else:
                edisgo_obj.timeseries.generation_dispatchable = \
                    ts_dispatchable

        ts_reactive_power = kwargs.get('generation_reactive_power', None)
        if ts_reactive_power is not None:
            if hasattr(edisgo_obj.timeseries,
                       'generation_reactive_power'):
                edisgo_obj.timeseries.generation_reactive_power = \
                    drop_duplicated_indices(edisgo_obj.timeseries.
                                            generation_reactive_power.T.
                                            append(ts_reactive_power.T),
                                            keep='last').T
            else:
                edisgo_obj.timeseries.generation_reactive_power = \
                    ts_reactive_power
        # create load active and reactive power timeseries
        _generation_from_timeseries(edisgo_obj=edisgo_obj,
                                    generator_names=generator_names)


def add_storage_units_timeseries(edisgo_obj, storage_unit_names, **kwargs):
    """
    Define storage unit time series for active and reactive power. For more
    information on required and optional parameters see description of
    :func:`get_component_timeseries`. The mode initially set within
    get_component_timeseries is used here to set new timeseries. If a different
    mode is required, change edisgo_obj.timeseries.mode to the desired mode and
    provide respective parameters.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    storage_unit_names: str or list of str
        Names of storage units to add timeseries for. Default None, timeseries
        for all storage units of edisgo_obj are set then.

    """
    # if timeseries have not been set yet, it is not
    # necessary to add timeseries
    if not hasattr(edisgo_obj.timeseries, 'mode'):
        logger.debug('Timeseries have not been set yet. Please call'
                     'get_components_timeseries to create timeseries.')
        return
    # turn single name to list
    if isinstance(storage_unit_names, str):
        storage_unit_names = [storage_unit_names]
    # append timeseries of respective mode
    if edisgo_obj.timeseries.mode:
        if 'worst-case' in edisgo_obj.timeseries.mode:
            modes = _get_worst_case_modes(edisgo_obj.timeseries.mode)
            # set random timeindex
            _worst_case_storage(edisgo_obj=edisgo_obj, modes=modes,
                                storage_names=storage_unit_names)
        elif edisgo_obj.timeseries.mode == 'manual':
            storage_units_active_power = kwargs.get(
                'storage_units_active_power', None)
            if storage_units_active_power is not None:
                check_timeseries_for_index_and_cols(edisgo_obj,
                    storage_units_active_power, storage_unit_names)
            storage_units_reactive_power = \
                kwargs.get('storage_units_reactive_power', None)
            if storage_units_reactive_power is not None:
                check_timeseries_for_index_and_cols(edisgo_obj,
                    storage_units_reactive_power, storage_unit_names)
            _drop_existing_component_timeseries(edisgo_obj, 'storage_units',
                                                storage_unit_names)
            # add new storage timeseries
            edisgo_obj.timeseries.storage_units_active_power = \
                edisgo_obj.timeseries.storage_units_active_power.T.\
                append(storage_units_active_power.T.loc[
                                               storage_unit_names]).T
            edisgo_obj.timeseries.storage_units_reactive_power = \
                edisgo_obj.timeseries.storage_units_reactive_power.T.\
                append(storage_units_reactive_power.T.loc[
                                               storage_unit_names]).T
        else:
            raise ValueError('{} is not a valid mode.'.format(
                edisgo_obj.timeseries.mode))
    else:
        # create load active and reactive power timeseries
        _storage_from_timeseries(edisgo_obj=edisgo_obj,
            name_storage_units=storage_unit_names,
            ts_active_power=kwargs.get('timeseries_storage_units', None),
            ts_reactive_power=
            kwargs.get('timeseries_storage_units_reactive_power', None))


def _drop_existing_component_timeseries(edisgo_obj, comp_type, comp_names):
    """
    Drop columns of active and reactive power timeseries of 'comp_type'
    components with names 'comp_names'.

    Parameters
    ----------
    edisgo_obj: :class:`~.self.edisgo.EDisGo`
        The eDisGo model overall container
    comp_type: str
        Specification of component type, either 'loads', 'generators' or
        'storage_units'
    comp_names: list of str
        List of names of components that are to be dropped

    """
    if isinstance(comp_names, str):
        comp_names = [comp_names]
    # drop existing timeseries of component
    setattr(edisgo_obj.timeseries, comp_type+'_active_power',
            getattr(edisgo_obj.timeseries, comp_type+'_active_power').drop(
        getattr(edisgo_obj.timeseries, comp_type + '_active_power').columns[
            getattr(edisgo_obj.timeseries, comp_type + '_active_power').
            columns.isin(comp_names)], axis=1))
    setattr(edisgo_obj.timeseries, comp_type + '_reactive_power',
            getattr(edisgo_obj.timeseries, comp_type + '_reactive_power').drop(
        getattr(edisgo_obj.timeseries, comp_type + '_reactive_power').columns[
            getattr(edisgo_obj.timeseries, comp_type + '_reactive_power').
                columns.isin(comp_names)], axis=1))


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
        raise ValueError("Inserted timeseries for the following "
                         "components have the a wrong time index: "
                         "{}. Values are missing.".format(component_names))
    if any(comp not in timeseries.columns for comp in component_names):
        raise ValueError("Columns of inserted timeseries are not the same "
                         "as names of components to be added. Timeseries "
                         "for the following components were tried to be "
                         "added: {}".format(component_names))


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


def _get_worst_case_modes(mode):
    """
    Returns list of modes to be handled in worst case analysis.

    Parameters
    ----------
    mode: str
        string containing 'worst-case' and specifies case

    Returns
    -------
    modes: list of str
        list which can contains 'feedin-case', 'load_case' or both
    """
    if mode == 'worst-case':
        modes = ['feedin_case', 'load_case']
    elif mode == 'worst-case-feedin' or mode == 'worst-case-load':
        modes = ['{}_case'.format(mode.split('-')[-1])]
    else:
        raise ValueError('{} is not a valid mode.'.format(mode))
    return modes


def _reset_timeseries(timeseries):
    """
    Resets all relevant timeseries to empty DataFrames with index timeindex
    of inserted TimeSeries object.

    Parameters
    ----------
    timeseries: :class:'~.edisgo.network.timeseries.TimeSeries'
    """
    timeseries.generators_active_power = pd.DataFrame(
        index=timeseries.timeindex)
    timeseries.generators_reactive_power = pd.DataFrame(
        index=timeseries.timeindex)
    timeseries.loads_active_power = pd.DataFrame(
        index=timeseries.timeindex)
    timeseries.loads_reactive_power = pd.DataFrame(
        index=timeseries.timeindex)
    timeseries.storage_units_active_power = pd.DataFrame(
        index=timeseries.timeindex)
    timeseries.storage_units_reactive_power = pd.DataFrame(
        index=timeseries.timeindex)
