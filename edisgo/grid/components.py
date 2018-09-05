import os
import logging
import pandas as pd
from math import acos, tan

if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString
from .grids import LVGrid, MVGrid

logger = logging.getLogger('edisgo')


class Component:
    """Generic component

    Notes
    -----
    In case of a MV-LV voltage station, :attr:`grid` refers to the LV grid.
    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._geom = kwargs.get('geom', None)
        self._grid = kwargs.get('grid', None)

    @property
    def id(self):
        """
        Unique ID of component

        Returns
        --------
        :obj:`int`
            Unique ID of component

        """
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def geom(self):
        """
        Location of component

        Returns
        --------
        :shapely:`Shapely Point object<points>` or :shapely:`Shapely LineString object<linestrings>`
            Location of the :class:`Component` as Shapely Point or LineString

        """
        return self._geom

    @geom.setter
    def geom(self, geom):
        self._geom = geom

    @property
    def grid(self):
        """
        Grid the component belongs to

        Returns
        --------
        :class:`~.grid.grids.MVGrid` or :class:`~.grid.grids.LVGrid`
            The MV or LV grid the component belongs to

        """
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid

    def __repr__(self):
        return '_'.join([self.__class__.__name__, str(self._id)])


class Station(Component):
    """Station object (medium or low voltage)

    Represents a station, contains transformers.

    Attributes
    ----------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._transformers = kwargs.get('transformers', None)

    @property
    def transformers(self):
        """:obj:`list` of :class:`Transformer` : Transformers located in
        station"""
        return self._transformers

    @transformers.setter
    def transformers(self, transformer):
        """
        Parameters
        ----------
        transformer : :obj:`list` of :class:`Transformer`
        """
        self._transformers = transformer

    def add_transformer(self, transformer):
        self._transformers.append(transformer)


class Transformer(Component):
    """Transformer object

    Attributes
    ----------
    _voltage_op : :obj:`float`
        Operational voltage
    _type : :pandas:`pandas.DataFrame<dataframe>`
        Specification of type, refers to  ToDo: ADD CORRECT REF TO (STATIC) DATA
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mv_grid = kwargs.get('mv_grid', None)
        self._voltage_op = kwargs.get('voltage_op', None)
        self._type = kwargs.get('type', None)

    @property
    def mv_grid(self):
        return self._mv_grid

    @property
    def voltage_op(self):
        return self._voltage_op

    @property
    def type(self):
        return self._type

    def __repr__(self):
        return str(self._id)


class Load(Component):
    """
    Load object

    Attributes
    ----------
    _timeseries : :pandas:`pandas.Series<series>`, optional
        See `timeseries` getter for more information.
    _consumption : :obj:`dict`, optional
        See `consumption` getter for more information.
    _timeseries_reactive : :pandas:`pandas.Series<series>`, optional
        See `timeseries_reactive` getter for more information.
    _power_factor : :obj:`float`, optional
        See `power_factor` getter for more information.
    _reactive_power_mode : :obj:`str`, optional
        See `reactive_power_mode` getter for more information.
    _q_sign : :obj:`int`, optional
        See `q_sign` getter for more information.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timeseries = kwargs.get('timeseries', None)
        self._consumption = kwargs.get('consumption', None)
        self._timeseries_reactive = kwargs.get('timeseries_reactive', None)
        self._power_factor = kwargs.get('power_factor', None)
        self._reactive_power_mode = kwargs.get('reactive_power_mode', None)
        self._q_sign = None

    @property
    def timeseries(self):
        """
        Load time series

        It returns the actual time series used in power flow analysis. If
        :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
        :meth:`timeseries()` looks for time series of the according sector in
        :class:`~.grid.network.TimeSeries` object.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power in kW in column 'p' and
            reactive power in kVA in column 'q'.

        """
        if self._timeseries is None:

            if isinstance(self.grid, MVGrid):
                voltage_level = 'mv'
            elif isinstance(self.grid, LVGrid):
                voltage_level = 'lv'

            ts_total = None
            for sector in self.consumption.keys():
                consumption = self.consumption[sector]

                # check if load time series for MV and LV are differentiated
                try:
                    ts = self.grid.network.timeseries.load[
                        sector, voltage_level].to_frame('p')
                except KeyError:
                    try:
                        ts = self.grid.network.timeseries.load[
                            sector].to_frame('p')
                    except KeyError:
                        logger.exception(
                            "No timeseries for load of type {} "
                            "given.".format(sector))
                        raise
                ts = ts * consumption
                if self.timeseries_reactive is not None:
                    ts['q'] = self.timeseries_reactive
                else:
                    ts['q'] = ts['p'] * self.q_sign * tan(
                        acos(self.power_factor))

                if not ts_total:
                    ts_total = ts
                else:
                    ts_total.p += ts.p
                    ts_total.q += ts.q

                return ts_total
        else:
            return self._timeseries

    @property
    def timeseries_reactive(self):
        """
        Reactive power time series in kvar.

        Parameters
        -----------
        timeseries_reactive : :pandas:`pandas.Seriese<series>`
            Series containing reactive power in kvar.

        Returns
        -------
        :pandas:`pandas.Series<series>` or None
            Series containing reactive power time series in kvar. If it is not
            set it is tried to be retrieved from `load_reactive_power`
            attribute of global TimeSeries object. If that is not possible
            None is returned.

        """
        if self._timeseries_reactive is None:
            
            # work around until retail and industrial are separate sectors
            # ToDo: remove once Ding0 data changed to single sector consumption
            sector = list(self.consumption.keys())[0]
            if len(list(self.consumption.keys())) > 1:
                consumption = sum([v for k, v in self.consumption.items()])
            else:
                consumption = self.consumption[sector]

            try:
                timeseries = \
                    self.grid.network.timeseries.load_reactive_power[
                        sector].to_frame('q')
            except (KeyError, TypeError):
                return None

            self.power_factor = 'not_applicable'
            self.reactive_power_mode = 'not_applicable'

            return timeseries * consumption
        else:
            return self._timeseries_reactive

    @timeseries_reactive.setter
    def timeseries_reactive(self, timeseries_reactive):
        if isinstance(timeseries_reactive, pd.Series):
            self._timeseries_reactive = timeseries_reactive
            self._power_factor = 'not_applicable'
            self._reactive_power_mode = 'not_applicable'
        else:
            raise ValueError(
                "Reactive power time series of load {} needs to be a pandas "
                "Series.".format(repr(self)))

    def pypsa_timeseries(self, attr):
        """Return time series in PyPSA format

        Parameters
        ----------
        attr : str
            Attribute name (PyPSA conventions). Choose from {p_set, q_set}
        """

        return self.timeseries[attr] / 1e3

    @property
    def consumption(self):
        """:obj:`dict` : Annual consumption per sector in kWh

        Sectors

            - retail/industrial
            - agricultural
            - residential

        The format of the :obj:`dict` is as follows::

            {
                'residential': 453.4
            }

        """
        return self._consumption

    @consumption.setter
    def consumption(self, cons_dict):
        self._consumption = cons_dict

    @property
    def peak_load(self):
        """
        Get sectoral peak load
        """
        peak_load = pd.Series(self.consumption).mul(pd.Series(
            self.grid.network.config['peakload_consumption_ratio']).astype(
            float), fill_value=0)

        return peak_load

    @property
    def power_factor(self):
        """
        Power factor of load

        Parameters
        -----------
        power_factor : :obj:`float`
            Ratio of real power to apparent power.

        Returns
        --------
        :obj:`float`
            Ratio of real power to apparent power. If power factor is not set
            it is retrieved from the network config object depending on the
            grid level the load is in.

        """
        if self._power_factor is None:
            if isinstance(self.grid, MVGrid):
                self._power_factor = self.grid.network.config[
                    'reactive_power_factor']['mv_load']
            elif isinstance(self.grid, LVGrid):
                self._power_factor = self.grid.network.config[
                    'reactive_power_factor']['lv_load']
        return self._power_factor

    @power_factor.setter
    def power_factor(self, power_factor):
        self._power_factor = power_factor

    @property
    def reactive_power_mode(self):
        """
        Power factor mode of Load.

        This information is necessary to make the load behave in an inductive
        or capacitive manner. Essentially this changes the sign of the reactive
        power.

        The convention used here in a load is that:
        - when `reactive_power_mode` is 'inductive' then Q is positive
        - when `reactive_power_mode` is 'capacitive' then Q is negative

        Parameters
        ----------
        reactive_power_mode : :obj:`str` or None
            Possible options are 'inductive', 'capacitive' and
            'not_applicable'. In the case of 'not_applicable' a reactive
            power time series must be given.

        Returns
        -------
        :obj:`str`
            In the case that this attribute is not set, it is retrieved from
            the network config object depending on the voltage level the load
            is in.

        """
        if self._reactive_power_mode is None:
            if isinstance(self.grid, MVGrid):
                self._reactive_power_mode = self.grid.network.config[
                    'reactive_power_mode']['mv_load']
            elif isinstance(self.grid, LVGrid):
                self._reactive_power_mode = self.grid.network.config[
                    'reactive_power_mode']['lv_load']

        return self._reactive_power_mode

    @reactive_power_mode.setter
    def reactive_power_mode(self, reactive_power_mode):
        self._reactive_power_mode = reactive_power_mode

    @property
    def q_sign(self):
        """
        Get the sign of reactive power based on :attr:`_reactive_power_mode`.

        Returns
        -------
        :obj:`int` or None
            In case of inductive reactive power returns +1 and in case of
            capacitive reactive power returns -1. If reactive power time
            series is given, `q_sign` is set to None.

        """
        if self.reactive_power_mode.lower() == 'inductive':
            return 1
        elif self.reactive_power_mode.lower() == 'capacitive':
            return -1
        elif self.reactive_power_mode.lower() == 'not_applicable':
            return None
        else:
            raise ValueError("Unknown value {} in reactive_power_mode for "
                             "Load {}.".format(self.reactive_power_mode,
                                                    repr(self)))

    def __repr__(self):
        return '_'.join(['Load',
                         sorted(list(self.consumption.keys()))[0],
                         repr(self.grid),
                         str(self.id)])


class Generator(Component):
    """Generator object

    Attributes
    ----------
    _timeseries : :pandas:`pandas.Series<series>`, optional
        See `timeseries` getter for more information.
    _nominal_capacity : :obj:`dict`, optional
        See `nominal_capacity` getter for more information.
    _type : :pandas:`pandas.Series<series>`, optional
        See `type` getter for more information.
    _subtype : :obj:`str`, optional
        See `subtype` getter for more information.
    _v_level : :obj:`str`, optional
        See `v_level` getter for more information.
    _q_sign : :obj:`int`, optional
        See `q_sign` getter for more information.
    _power_factor : :obj:`float`, optional
        See `power_factor` getter for more information.
    _reactive_power_mode : :obj:`str`, optional
        See `reactive_power_mode` getter for more information.
    _q_sign : :obj:`int`, optional
        See `q_sign` getter for more information.

    Notes
    -----
    The attributes :attr:`_type` and :attr:`_subtype` have to match the
    corresponding types in :class:`~.grid.network.Timeseries` to
    allow allocation of time series to generators.

    See also
    --------
    edisgo.network.TimeSeries : Details of global
        :class:`~.grid.network.TimeSeries`

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._nominal_capacity = kwargs.get('nominal_capacity', None)
        self._type = kwargs.get('type', None)
        self._subtype = kwargs.get('subtype', None)
        self._v_level = kwargs.get('v_level', None)
        self._timeseries = kwargs.get('timeseries', None)
        self._timeseries_reactive = kwargs.get('timeseries_reactive', None)
        self._power_factor = kwargs.get('power_factor', None)
        self._reactive_power_mode = kwargs.get('reactive_power_mode', None)
        self._q_sign = None

    @property
    def timeseries(self):
        """
        Feed-in time series of generator

        It returns the actual dispatch time series used in power flow analysis.
        If :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
        :meth:`timeseries` looks for time series of the according type of
        technology in :class:`~.grid.network.TimeSeries`. If the reactive
        power time series is provided through :attr:`_timeseries_reactive`,
        this is added to :attr:`_timeseries`. When :attr:`_timeseries_reactive`
        is not set, the reactive power is also calculated in
        :attr:`_timeseries` using :attr:`power_factor` and
        :attr:`reactive_power_mode`. The :attr:`power_factor` determines the
        magnitude of the reactive power based on the power factor and active
        power provided and the :attr:`reactive_power_mode` determines if the
        reactive power is either consumed (inductive behaviour) or provided
        (capacitive behaviour).

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power in kW in column 'p' and
            reactive power in kvar in column 'q'.

        """
        if self._timeseries is None:
            # calculate time series for active and reactive power
            try:
                timeseries = \
                    self.grid.network.timeseries.generation_dispatchable[
                        self.type].to_frame('p')
            except KeyError:
                try:
                    timeseries = \
                        self.grid.network.timeseries.generation_dispatchable[
                            'other'].to_frame('p')
                except KeyError:
                    logger.exception("No time series for type {} "
                                     "given.".format(self.type))
                    raise

            timeseries = timeseries * self.nominal_capacity
            if self.timeseries_reactive is not None:
                timeseries['q'] = self.timeseries_reactive
            else:
                timeseries['q'] = timeseries['p'] * self.q_sign * tan(acos(
                    self.power_factor))

            return timeseries
        else:
            return self._timeseries.loc[
                   self.grid.network.timeseries.timeindex, :]

    @property
    def timeseries_reactive(self):
        """
        Reactive power time series in kvar.

        Parameters
        -----------
        timeseries_reactive : :pandas:`pandas.Seriese<series>`
            Series containing reactive power in kvar.

        Returns
        -------
        :pandas:`pandas.Series<series>` or None
            Series containing reactive power time series in kvar. If it is not
            set it is tried to be retrieved from `generation_reactive_power`
            attribute of global TimeSeries object. If that is not possible
            None is returned.

        """
        if self._timeseries_reactive is None:
            if self.grid.network.timeseries.generation_reactive_power \
                    is not None:
                try:
                    timeseries = \
                        self.grid.network.timeseries.generation_reactive_power[
                            self.type].to_frame('q')
                except (KeyError, TypeError):
                    try:
                        timeseries = \
                            self.grid.network.timeseries.generation_reactive_power[
                                'other'].to_frame('q')
                    except:
                        logger.warning(
                            "No reactive power time series for type {} given. "
                            "Reactive power time series will be calculated from "
                            "assumptions in config files and active power "
                            "timeseries.".format(self.type))
                        return None
                self.power_factor = 'not_applicable'
                self.reactive_power_mode = 'not_applicable'
                return timeseries * self.nominal_capacity
            else:
                return None
        else:
            return self._timeseries_reactive.loc[
                   self.grid.network.timeseries.timeindex, :]

    @timeseries_reactive.setter
    def timeseries_reactive(self, timeseries_reactive):
        if isinstance(timeseries_reactive, pd.Series):
            # check if the values in time series makes sense
            if timeseries_reactive.max() <= self._nominal_capacity:
                self._timeseries_reactive = timeseries_reactive
            else:
                message = "Maximum reactive power in timeseries at index " \
                          "{} ".format(timeseries_reactive.idxmax()) + \
                          "is higher than nominal capacity."
                logger.error(message)
                raise ValueError(message)
        else:
            raise ValueError(
                "Reactive power time series of generator {} needs to be a "
                "pandas Series.".format(repr(self)))

    def pypsa_timeseries(self, attr):
        """
        Return time series in PyPSA format

        Convert from kW, kVA to MW, MVA

        Parameters
        ----------
        attr : :obj:`str`
            Attribute name (PyPSA conventions). Choose from {p_set, q_set}
        """
        return self.timeseries[attr] / 1e3

    @property
    def type(self):
        """:obj:`str` : Technology type (e.g. 'solar')"""
        return self._type

    @property
    def subtype(self):
        """:obj:`str` : Technology subtype (e.g. 'solar_roof_mounted')"""
        return self._subtype

    @property
    def nominal_capacity(self):
        """:obj:`float` : Nominal generation capacity in kW"""
        return self._nominal_capacity

    @nominal_capacity.setter
    def nominal_capacity(self, nominal_capacity):
        self._nominal_capacity = nominal_capacity

    @property
    def v_level(self):
        """:obj:`int` : Voltage level"""
        return self._v_level

    @property
    def power_factor(self):
        """
        Power factor of generator

        Parameters
        -----------
        power_factor : :obj:`float`
            Ratio of real power to apparent power.

        Returns
        --------
        :obj:`float`
            Ratio of real power to apparent power. If power factor is not set
            it is retrieved from the network config object depending on the
            grid level the generator is in.

        """
        if self._power_factor is None:
            if isinstance(self.grid, MVGrid):
                self._power_factor = self.grid.network.config[
                    'reactive_power_factor']['mv_gen']
            elif isinstance(self.grid, LVGrid):
                self._power_factor = self.grid.network.config[
                    'reactive_power_factor']['lv_gen']
        return self._power_factor

    @power_factor.setter
    def power_factor(self, power_factor):
        self._power_factor = power_factor

    @property
    def reactive_power_mode(self):
        """
        Power factor mode of generator.

        This information is necessary to make the generator behave in an
        inductive or capacitive manner. Essentially this changes the sign of
        the reactive power.

        The convention used here in a generator is that:
        - when `reactive_power_mode` is 'capacitive' then Q is positive
        - when `reactive_power_mode` is 'inductive' then Q is negative

        In the case that this attribute is not set, it is retrieved from the
        network config object depending on the voltage level the generator
        is in.

        Parameters
        ----------
        reactive_power_mode : :obj:`str` or None
            Possible options are 'inductive', 'capacitive' and
            'not_applicable'. In the case of 'not_applicable' a reactive
            power time series must be given.

        Returns
        -------
        :obj:`str` : Power factor mode
            In the case that this attribute is not set, it is retrieved from
            the network config object depending on the voltage level the
            generator is in.

        """
        if self._reactive_power_mode is None:
            if isinstance(self.grid, MVGrid):
                self._reactive_power_mode = self.grid.network.config[
                    'reactive_power_mode']['mv_gen']
            elif isinstance(self.grid, LVGrid):
                self._reactive_power_mode = self.grid.network.config[
                    'reactive_power_mode']['lv_gen']

        return self._reactive_power_mode

    @reactive_power_mode.setter
    def reactive_power_mode(self, reactive_power_mode):
        self._reactive_power_mode = reactive_power_mode

    @property
    def q_sign(self):
        """
        Get the sign of reactive power based on :attr:`_reactive_power_mode`.

        Returns
        -------
        :obj:`int` or None
            In case of inductive reactive power returns -1 and in case of
            capacitive reactive power returns +1. If reactive power time
            series is given, `q_sign` is set to None.

        """
        if self.reactive_power_mode.lower() == 'inductive':
            return -1
        elif self.reactive_power_mode.lower() == 'capacitive':
            return 1
        else:
            raise ValueError("Unknown value {} in reactive_power_mode for "
                             "Generator {}.".format(self.reactive_power_mode,
                                                    repr(self)))


class GeneratorFluctuating(Generator):
    """
    Generator object for fluctuating renewables.

    Attributes
    ----------
    _curtailment : :pandas:`pandas.Series<series>`
        See `curtailment` getter for more information.
    _weather_cell_id : :obj:`int`
        See `weather_cell_id` getter for more information.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._curtailment = kwargs.get('curtailment', None)
        self._weather_cell_id = kwargs.get('weather_cell_id', None)

    @property
    def timeseries(self):
        """
        Feed-in time series of generator

        It returns the actual time series used in power flow analysis. If
        :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
        :meth:`timeseries` looks for generation and curtailment time series
        of the according type of technology (and weather cell) in
        :class:`~.grid.network.TimeSeries`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power in kW in column 'p' and
            reactive power in kVA in column 'q'.

        """
        if self._timeseries is None:

            # get time series for active power depending on if they are
            # differentiated by weather cell ID or not
            if isinstance(self.grid.network.timeseries.generation_fluctuating.
                                  columns, pd.MultiIndex):
                if self.weather_cell_id:
                    try:
                        timeseries = self.grid.network.timeseries.\
                            generation_fluctuating[
                            self.type, self.weather_cell_id].to_frame('p')
                    except KeyError:
                        logger.exception("No time series for type {} and "
                                         "weather cell ID {} given.".format(
                            self.type, self.weather_cell_id))
                        raise
                else:
                    logger.exception("No weather cell ID provided for "
                                     "fluctuating generator {}.".format(
                        repr(self)))
                    raise KeyError
            else:
                try:
                    timeseries = self.grid.network.timeseries.\
                        generation_fluctuating[self.type].to_frame('p')
                except KeyError:
                    logger.exception("No time series for type {} "
                                     "given.".format(self.type))
                    raise

            timeseries = timeseries * self.nominal_capacity

            # subtract curtailment
            if self.curtailment is not None:
                timeseries = timeseries.join(
                    self.curtailment.to_frame('curtailment'), how='left')
                timeseries.p = timeseries.p - timeseries.curtailment.fillna(0)

            if self.timeseries_reactive is not None:
                timeseries['q'] = self.timeseries_reactive
            else:
                timeseries['q'] = timeseries['p'] * self.q_sign * tan(acos(
                    self.power_factor))

            return timeseries
        else:
            #ToDo: should curtailment be subtracted from timeseries?
            return self._timeseries.loc[
                   self.grid.network.timeseries.timeindex, :]

    @property
    def timeseries_reactive(self):
        """
        Reactive power time series in kvar.

        Parameters
        -------
        :pandas:`pandas.Series<series>`
            Series containing reactive power time series in kvar.

        Returns
        ----------
        :pandas:`pandas.DataFrame<dataframe>` or None
            Series containing reactive power time series in kvar. If it is not
            set it is tried to be retrieved from `generation_reactive_power`
            attribute of global TimeSeries object. If that is not possible
            None is returned.

        """

        if self._timeseries_reactive is None:
            # try to get time series for reactive power depending on if they
            # are differentiated by weather cell ID or not
            # raise warning if no time series for generator type (and weather
            # cell ID) can be retrieved
            if self.grid.network.timeseries.generation_reactive_power \
                    is not None:
                if isinstance(
                        self.grid.network.timeseries.generation_reactive_power.
                                columns, pd.MultiIndex):
                    if self.weather_cell_id:
                        try:
                            timeseries = self.grid.network.timeseries. \
                                generation_reactive_power[
                                self.type, self.weather_cell_id].to_frame('q')
                            return timeseries * self.nominal_capacity
                        except (KeyError, TypeError):
                            logger.warning("No time series for type {} and "
                                           "weather cell ID {} given. "
                                           "Reactive power time series will "
                                           "be calculated from assumptions "
                                           "in config files and active power "
                                           "timeseries.".format(
                                self.type, self.weather_cell_id))
                            return None
                    else:
                        raise ValueError(
                            "No weather cell ID provided for fluctuating "
                            "generator {}, but reactive power is given as a "
                            "MultiIndex suggesting that it is differentiated "
                            "by weather cell ID.".format(repr(self)))
                else:
                    try:
                        timeseries = self.grid.network.timeseries. \
                            generation_reactive_power[self.type].to_frame('q')
                        return timeseries * self.nominal_capacity
                    except (KeyError, TypeError):
                        logger.warning("No reactive power time series for "
                                       "type {} given. Reactive power time "
                                       "series will be calculated from "
                                       "assumptions in config files and "
                                       "active power timeseries.".format(
                            self.type))
                        return None
            else:
                return None
        else:
            return self._timeseries_reactive.loc[
                   self.grid.network.timeseries.timeindex, :]

    @timeseries_reactive.setter
    def timeseries_reactive(self, timeseries_reactive):
        if isinstance(timeseries_reactive, pd.Series):
            if timeseries_reactive.max() <= self._nominal_capacity:
                self._timeseries_reactive = timeseries_reactive
                self._power_factor = 'not_applicable'
                self._reactive_power_mode = 'not_applicable'
            else:
                message = "Maximum reactive power in time series at " + \
                          "index {} ".format(timeseries_reactive.idxmax()) + \
                          "is higher than nominal capacity."
                logger.error(message)
                raise ValueError(message)

    @property
    def curtailment(self):
        """
        Parameters
        ----------
        curtailment_ts : :pandas:`pandas.Series<series>`
            See class definition for details.

        Returns
        -------
        :pandas:`pandas.Series<series>`
            If self._curtailment is set it returns that. Otherwise, if
            curtailment in :class:`~.grid.network.TimeSeries` for the
            corresponding technology type (and if given, weather cell ID)
            is set this is returned.

        """
        if self._curtailment is not None:
            return self._curtailment
        elif isinstance(self.grid.network.timeseries._curtailment,
                        pd.DataFrame):
            if isinstance(self.grid.network.timeseries.curtailment.
                                  columns, pd.MultiIndex):
                if self.weather_cell_id:
                    try:
                        return self.grid.network.timeseries.curtailment[
                            self.type, self.weather_cell_id]
                    except KeyError:
                        logger.exception("No curtailment time series for type "
                                         "{} and weather cell ID {} "
                                         "given.".format(self.type,
                                                         self.weather_cell_id))
                        raise
                else:
                    logger.exception("No weather cell ID provided for "
                                     "fluctuating generator {}.".format(
                        repr(self)))
                    raise KeyError
            else:
                try:
                    return self.grid.network.timeseries.curtailment[self.type]
                except KeyError:
                    logger.exception("No curtailment time series for type "
                                     "{} given.".format(self.type))
                    raise
        else:
            return None

    @curtailment.setter
    def curtailment(self, curtailment_ts):
        self._curtailment = curtailment_ts

    @property
    def weather_cell_id(self):
        """
        Get weather cell ID

        Returns
        -------
        :obj:`str`
            See class definition for details.

        """
        return self._weather_cell_id

    @weather_cell_id.setter
    def weather_cell_id(self, weather_cell):
        self._weather_cell_id = weather_cell


class Storage(Component):
    """Storage object

    Describes a single storage instance in the eDisGo grid. Includes technical
    parameters such as :attr:`Storage.efficiency_in` or
    :attr:`Storage.standing_loss` as well as its time series of operation
    :meth:`Storage.timeseries`.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timeseries = kwargs.get('timeseries', None)
        self._nominal_power = kwargs.get('nominal_power', None)
        self._power_factor = kwargs.get('power_factor', None)
        self._reactive_power_mode = kwargs.get('reactive_power_mode', None)

        self._max_hours = kwargs.get('max_hours', None)
        self._soc_initial = kwargs.get('soc_initial', None)
        self._efficiency_in = kwargs.get('efficiency_in', None)
        self._efficiency_out = kwargs.get('efficiency_out', None)
        self._standing_loss = kwargs.get('standing_loss', None)
        self._operation = kwargs.get('operation', None)
        self._reactive_power_mode = kwargs.get('reactive_power_mode', None)
        self._q_sign = None

    @property
    def timeseries(self):
        """
        Time series of storage operation

        Parameters
        ----------
        ts : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power the storage is charged (negative)
            and discharged (positive) with (on the grid side) in kW in column 
            'p' and reactive power in kvar in column 'q'. When 'q' is positive,
            reactive power is supplied (behaving as a capacitor) and when 'q'
            is negative reactive power is consumed (behaving as an inductor).

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            See parameter `timeseries`.

        """
        # check if time series for reactive power is given, otherwise
        # calculate it
        if 'q' in self._timeseries.columns:
            return self._timeseries
        else:
            self._timeseries['q'] = abs(self._timeseries.p) * self.q_sign * \
                                    tan(acos(self.power_factor))
            return self._timeseries.loc[
                   self.grid.network.timeseries.timeindex, :]

    @timeseries.setter
    def timeseries(self, ts):
        self._timeseries = ts

    def pypsa_timeseries(self, attr):
        """Return time series in PyPSA format

        Convert from kW, kVA to MW, MVA

        Parameters
        ----------
        attr : str
            Attribute name (PyPSA conventions). Choose from {p_set, q_set}

        """
        return self.timeseries[attr] / 1e3

    @property
    def nominal_power(self):
        """
        Nominal charging and discharging power of storage instance in kW.

        Returns
        -------
        float
            Storage nominal power

        """
        return self._nominal_power

    @property
    def max_hours(self):
        """
        Maximum state of charge capacity in terms of hours at full discharging
        power `nominal_power`.

        Returns
        -------
        float
            Hours storage can be discharged for at nominal power

        """
        return self._max_hours

    @property
    def nominal_capacity(self):
        """
        Nominal storage capacity in kWh.

        Returns
        -------
        float
            Storage nominal capacity

        """
        return self._max_hours * self._nominal_power

    @property
    def soc_initial(self):
        """Initial state of charge in kWh.

        Returns
        -------
        float
            Initial state of charge

        """
        return self._soc_initial

    @property
    def efficiency_in(self):
        """Storage charging efficiency in per unit.

        Returns
        -------
        float
            Charging efficiency in range of 0..1

        """
        return self._efficiency_in

    @property
    def efficiency_out(self):
        """Storage discharging efficiency in per unit.

        Returns
        -------
        float
            Discharging efficiency in range of 0..1

        """
        return self._efficiency_out

    @property
    def standing_loss(self):
        """Standing losses of storage in %/100 / h

        Losses relative to SoC per hour. The unit is pu (%/100%). Hence, it
        ranges from 0..1.

        Returns
        -------
        float
            Standing losses in pu.

        """
        return self._standing_loss

    @property
    def operation(self):
        """
        Storage operation definition

        Returns
        -------
        :obj:`str`

        """
        self._operation

    @property
    def power_factor(self):
        """
        Power factor of storage

        If power factor is not set it is retrieved from the network config
        object depending on the grid level the storage is in.

        Returns
        --------
        :obj:`float` : Power factor
            Ratio of real power to apparent power.

        """
        if self._power_factor is None:
            if isinstance(self.grid, MVGrid):
                self._power_factor = self.grid.network.config[
                    'reactive_power_factor']['mv_storage']
            elif isinstance(self.grid, LVGrid):
                self._power_factor = self.grid.network.config[
                    'reactive_power_factor']['lv_storage']
        return self._power_factor

    @power_factor.setter
    def power_factor(self, power_factor):
        self._power_factor = power_factor

    @property
    def reactive_power_mode(self):
        """
        Power factor mode of storage.

        If the power factor is set, then it is necessary to know whether
        it is leading or lagging. In other words this information is necessary
        to make the storage behave in an inductive or capacitive manner.
        Essentially this changes the sign of the reactive power Q.

        The convention used here in a storage is that:
        - when `reactive_power_mode` is 'capacitive' then Q is positive
        - when `reactive_power_mode` is 'inductive' then Q is negative

        In the case that this attribute is not set, it is retrieved from the
        network config object depending on the voltage level the storage
        is in.

        Returns
        -------
        :obj: `str` : Power factor mode
            Either 'inductive' or 'capacitive'

        """
        if self._reactive_power_mode is None:
            if isinstance(self.grid, MVGrid):
                self._reactive_power_mode = self.grid.network.config[
                    'reactive_power_mode']['mv_storage']
            elif isinstance(self.grid, LVGrid):
                self._reactive_power_mode = self.grid.network.config[
                    'reactive_power_mode']['lv_storage']

        return self._reactive_power_mode

    @reactive_power_mode.setter
    def reactive_power_mode(self, reactive_power_mode):
        """
        Set the power factor mode of the generator.
        Should be either 'inductive' or 'capacitive'
        """
        self._reactive_power_mode = reactive_power_mode

    @property
    def q_sign(self):
        """
        Get the sign reactive power based on the
        :attr: `_reactive_power_mode`

        Returns
        -------
        :obj: `int` : +1 or -1
        """
        if self.reactive_power_mode.lower() == 'inductive':
            return -1
        elif self.reactive_power_mode.lower() == 'capacitive':
            return 1
        else:
            raise ValueError("Unknown value {} in reactive_power_mode".format(
                self.reactive_power_mode))

    def __repr__(self):
        return str(self._id)


class MVDisconnectingPoint(Component):
    """Disconnecting point object

    Medium voltage disconnecting points = points where MV rings are split under
    normal operation conditions (= switch disconnectors in DINGO).

    Attributes
    ----------
    _nodes : tuple
        Nodes of switch disconnector line segment
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._state = kwargs.get('state', None)
        self._line = kwargs.get('line', None)
        self._nodes = kwargs.get('nodes', None)

    def open(self):
        """Toggle state to open switch disconnector"""
        if self._state != 'open':
            if self._line is not None:
                self._state = 'open'
                self._nodes = self.grid.graph.nodes_from_line(self._line)
                self.grid.graph.remove_edge(
                    self._nodes[0], self._nodes[1])
            else:
                raise ValueError('``line`` is not set')

    def close(self):
        """Toggle state to closed switch disconnector"""
        self._state = 'closed'
        self.grid.graph.add_edge(
            self._nodes[0], self._nodes[1], {'line': self._line})

    @property
    def state(self):
        """
        Get state of switch disconnector

        Returns
        -------
        str or None
            State of MV ring disconnector: 'open' or 'closed'.

            Returns `None` if switch disconnector line segment is not set. This
            refers to an open ring, but it's unknown if the grid topology was
            built correctly.
        """
        return self._state

    @property
    def line(self):
        """
        Get or set line segment that belongs to the switch disconnector

        The setter allows only to set the respective line initially. Once the
        line segment representing the switch disconnector is set, it cannot be
        changed.

        Returns
        -------
        Line
            Line segment that is part of the switch disconnector model
        """
        return self._line

    @line.setter
    def line(self, line):
        if self._line is None:
            if isinstance(line, Line):
                self._line = line
            else:
                raise TypeError('``line`` must be of type {}'.format(Line))
        else:
            raise ValueError('``line`` can only be set initially. Too late '
                             'dude!')


class BranchTee(Component):
    """Branch tee object

    A branch tee is used to branch off a line to connect another node
    (german: Abzweigmuffe)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_building = kwargs.get('in_building', None)

        # set id of BranchTee automatically if not provided
        if not self._id:
            ids = [_.id for _ in
                   self.grid.graph.nodes_by_attribute('branch_tee')]
            if ids:
                self._id = max(ids) + 1
            else:
                self._id = 1

    def __repr__(self):
        return '_'.join(
            [self.__class__.__name__, repr(self.grid), str(self._id)])


class MVStation(Station):
    """MV Station object"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self, side=None):
        repr_base = super().__repr__()

        # As we don't consider HV-MV transformers in PFA, we don't have to care
        # about primary side bus of MV station. Hence, the general repr()
        # currently returned, implicitely refers to the secondary side (MV level)
        # if side == 'hv':
        #     return '_'.join(['primary', repr_base])
        # elif side == 'mv':
        #     return '_'.join(['secondary', repr_base])
        # else:
        #     return repr_base
        return repr_base


class LVStation(Station):
    """LV Station object"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mv_grid = kwargs.get('mv_grid', None)

    @property
    def mv_grid(self):
        return self._mv_grid

    def __repr__(self, side=None):
        repr_base = super().__repr__()

        if side == 'mv':
            return '_'.join(['primary', repr_base])
        elif side == 'lv':
            return '_'.join(['secondary', repr_base])
        else:
            return repr_base


class Line(Component):
    """
    Line object

    Parameters
    ----------
    _type: :pandas:`pandas.Series<series>`
        Equipment specification including R and X for power flow analysis
        Columns:

        ======== ================== ====== =========
        Column   Description        Unit   Data type
        ======== ================== ====== =========
        name     Name (e.g. NAYY..) -      str
        U_n      Nominal voltage    kV     int
        I_max_th Max. th. current   A      float
        R        Resistance         Ohm/km float
        L        Inductance         mH/km  float
        C        Capacitance        uF/km  float
        Source   Data source        -      str
        ============================================

    _length: float
        Length of the line calculated in linear distance. Unit: m
    _quantity: float
        Quantity of parallel installed lines.
    _kind: String
        Specifies whether the line is an underground cable ('cable') or an
        overhead line ('line').
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = kwargs.get('type', None)
        self._length = kwargs.get('length', None)
        self._quantity = kwargs.get('quantity', 1)
        self._kind = kwargs.get('kind', None)

    @property
    def geom(self):
        """Provide :shapely:`Shapely LineString object<linestrings>` geometry of
        :class:`Line`"""
        adj_nodes = self._grid._graph.nodes_from_line(self)

        return LineString([adj_nodes[0].geom, adj_nodes[1].geom])

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        self._type = new_type

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, new_length):
        self._length = new_length

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, new_quantity):
        self._quantity = new_quantity

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, new_kind):
        self._kind = new_kind

