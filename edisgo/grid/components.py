import os
import logging
import pandas as pd
from math import acos, tan, sqrt

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
        Specification of type, refers to #TODO: ADD CORRECT REF TO (STATIC) DATA
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
    """Load object

    Attributes
    ----------
    _timeseries : :pandas:`pandas.Series<series>`
        Contains time series for load

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timeseries = kwargs.get('timeseries', None)
        self._consumption = kwargs.get('consumption', None)

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
            # work around until retail and industrial are separate sectors
            # TODO: remove once Ding0 data changed to single sector consumption
            sector = list(self.consumption.keys())[0]
            if len(list(self.consumption.keys())) > 1:
                consumption = sum([v for k, v in self.consumption.items()])
            else:
                consumption = self.consumption[sector]

            if isinstance(self.grid, MVGrid):
                q_factor = tan(acos(self.grid.network.config[
                                        'reactive_power_factor']['mv_load']))
                voltage_level = 'mv'
            elif isinstance(self.grid, LVGrid):
                q_factor = tan(acos(self.grid.network.config[
                                        'reactive_power_factor']['lv_load']))
                voltage_level = 'lv'
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
            ts['q'] = ts['p'] * q_factor
            self._timeseries = ts * consumption

        return self._timeseries

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

    def __repr__(self):
        return '_'.join(['Load',
                         sorted(list(self.consumption.keys()))[0],
                         repr(self.grid),
                         str(self.id)])


class Generator(Component):
    """Generator object

    Attributes
    ----------
    _timeseries : :pandas:`pandas.Series<series>`
        Contains time series for generator

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

    @property
    def timeseries(self):
        """
        Feed-in time series of generator

        It returns the actual time series used in power flow analysis. If
        :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
        :meth:`timeseries` looks for time series of the according type of
        technology in :class:`~.grid.network.TimeSeries`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power in kW in column 'p' and
            reactive power in kVA in column 'q'.

        """
        if self._timeseries is None:
            # calculate share of reactive power
            if isinstance(self.grid, MVGrid):
                q_factor = tan(acos(self.grid.network.config[
                                        'reactive_power_factor']['mv_gen']))
            elif isinstance(self.grid, LVGrid):
                q_factor = tan(acos(self.grid.network.config[
                                        'reactive_power_factor']['lv_gen']))
            # set time series for active and reactive power
            try:
                ts = self.grid.network.timeseries.generation_dispatchable[
                    self.type].to_frame('p')
            except KeyError:
                try:
                    ts = self.grid.network.timeseries.generation_dispatchable[
                        'other'].to_frame('p')
                except KeyError:
                    logger.exception("No time series for type {} "
                                     "given.".format(self.type))
                    raise
            ts['q'] = ts['p'] * q_factor
            self._timeseries = ts * self.nominal_capacity

        return self._timeseries.loc[self.grid.network.timeseries.timeindex, :]

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
    def type(self):
        """:obj:`str` : Technology type (e.g. 'solar')"""
        return self._type

    @property
    def subtype(self):
        """:obj:`str` : Technology subtype (e.g. 'solar_roof_mounted')"""
        return self._subtype

    @property
    def nominal_capacity(self):
        """:obj:`float` : Nominal generation capacity"""
        return self._nominal_capacity

    @nominal_capacity.setter
    def nominal_capacity(self, nominal_capacity):
        self._nominal_capacity = nominal_capacity

    @property
    def v_level(self):
        """:obj:`int` : Voltage level"""
        return self._v_level


class GeneratorFluctuating(Generator):
    """
    Generator object for fluctuating renewables.

    Attributes
    ----------
    _curtailment : :pandas:`pandas.Series<series>`
        Contains time series for curtailment in kW
    _weather_cell_id : :obj:`str`
        ID of the weather cell used to generate feed-in time series

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
                        ts = self.grid.network.timeseries.\
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
                    ts = self.grid.network.timeseries.generation_fluctuating[
                        self.type].to_frame('p')
                except KeyError:
                    logger.exception("No time series for type {} "
                                     "given.".format(self.type))
                    raise

            # subtract curtailment
            if self.curtailment is not None:
                ts = ts.join(self.curtailment.to_frame('curtailment'),
                             how='left')
                ts.p = ts.p - ts.curtailment.fillna(0)

            # calculate share of reactive power
            if isinstance(self.grid, MVGrid):
                q_factor = tan(acos(self.grid.network.config[
                                        'reactive_power_factor'][
                                        'mv_gen']))
            elif isinstance(self.grid, LVGrid):
                q_factor = tan(acos(self.grid.network.config[
                                        'reactive_power_factor'][
                                        'lv_gen']))
            ts['q'] = ts['p'] * q_factor
            self._timeseries = ts * self.nominal_capacity

        return self._timeseries.loc[self.grid.network.timeseries.timeindex, :]

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
        if self._curtailment:
            return self._curtailment
        elif isinstance(self.grid.network.timeseries.curtailment,
                        pd.DataFrame):
            if isinstance(self.grid.network.timeseries.curtailment.
                          columns, pd.MultiIndex):
                if self.weather_cell_id:
                    try:
                        return self.grid.network.timeseries.curtailment[
                            self.type, self.weather_cell_id]
                    except KeyError:
                        logger.exception("No curtailment time series for type "
                                         "{} and  weather cell ID {} "
                                         "given.".format(self.type,
                                                         self.weather_cell_id))
                        raise
                else:
                    logger.exception("No weather cell ID provided for "
                                     "fluctuating generator {}.".format(
                                        repr(self)))
                    raise KeyError
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

    Examples
    --------
    In order to define a storage that operates in mode "fifty-fifty"
    (see :class:`~.grid.network.StorageControl` `timeseries_battery` parameter
    for details about modes) provide the following when instantiating a
    storage:

    >>> from edisgo.grid.components import Storage
    >>> from edisgo.flex_opt import storage_operation
    >>> storage_parameters = {'nominal_capacity': 100,
    >>>                       'soc_initial': 0,
    >>>                       'efficiency_in': .9,
    >>>                       'efficiency_out': .9,
    >>>                       'standing_loss': 0}
    >>> storage = Storage(storage_parameters)
    >>> storage_operation.fifty_fifty(storage)

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timeseries = kwargs.get('timeseries', None)
        self._nominal_capacity = kwargs.get('nominal_capacity', None)
        self._soc_initial = kwargs.get('soc_initial', None)
        self._efficiency_in = kwargs.get('efficiency_in', None)
        self._efficiency_out = kwargs.get('efficiency_out', None)
        self._standing_loss = kwargs.get('standing_loss', None)
        self._operation = kwargs.get('operation', None)

    @property
    def timeseries(self):
        """
        Time series of storage operation

        Parameters
        ----------
        timeseries : :pandas:`pandas.DataFrame<dataframe>`
            DataFrame containing active power the storage is charged (negative)
            and discharged (positive) with in kW in column 'p' and
            reactive power in kVA in column 'q'.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            See parameter `timeseries`.

        """
        # ToDo: Consider efficiencies
        return self._timeseries.loc[self.grid.network.timeseries.timeindex, :]

    @timeseries.setter
    def timeseries(self, timeseries):
        self._timeseries = timeseries

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
    def nominal_capacity(self):
        """
        Nominal capacity of storage instance in kW.

        Returns
        -------
        float
            Storage nominal capacity

        """
        return self._nominal_capacity

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
        return '_'.join([self.__class__.__name__, repr(self.grid), str(self._id)])


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

