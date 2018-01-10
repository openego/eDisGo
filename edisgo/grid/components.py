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

    _id : :obj:`int`
        Unique ID

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
        """Returns id of component"""
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def geom(self):
        """:shapely:`Shapely Point object<points>` or
        :shapely:`Shapely LineString object<linestrings>` : Location of the
        :class:`Component` as Shapely Point or LineString"""
        return self._geom

    @geom.setter
    def geom(self, geom):
        self._geom = geom

    @property
    def grid(self):
        """:class:`~.grid.grids.MVGrid` or :class:`~.grid.grids.LVGrid` : The MV or LV grid this component belongs to"""
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
        """Return time series of load

        It returns the actual time series used in power flow analysis. If
        :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
        :meth:`timeseries()` looks for time series of the according sector in
        :class:`~.grid.network.TimeSeries` object.

        See also
        --------
        edisgo.network.TimeSeries : Details of global TimeSeries
        """
        if self._timeseries is None:
            sector = list(self.consumption.keys())[0]
            peak_load_consumption_ratio = float(self.grid.network.config['data'][
                'peakload_consumption_ratio'][sector])

            if isinstance(self.grid, MVGrid):
                q_factor = tan(acos(
                    self.grid.network.scenario.parameters.pfac_mv_load))
                power_scaling = float(self.grid.network.config['scenario'][
                                          'scale_factor_mv_load'])
            elif isinstance(self.grid, LVGrid):
                q_factor = tan(acos(
                    self.grid.network.scenario.parameters.pfac_lv_load))
                power_scaling = float(self.grid.network.config['scenario'][
                                          'scale_factor_lv_load'])

            # work around until retail and industrial are separate sectors
            # TODO: remove once Ding0 data changed to single sector consumption
            sector = list(self.consumption.keys())[0]
            if len(list(self.consumption.keys())) > 1:
                consumption = sum([v for k, v in self.consumption.items()])
            else:
                consumption = self.consumption[sector]

            # set timeseries for active and reactive power
            if self.grid.network.scenario.mode == 'worst-case':
                if isinstance(self.grid, MVGrid):
                    power_scaling = float(self.grid.network.config['scenario'][
                                              'scale_factor_mv_load'])
                elif isinstance(self.grid, LVGrid):
                    power_scaling = float(self.grid.network.config['scenario'][
                                              'scale_factor_lv_load'])
                ts = (self.grid.network.scenario.timeseries.load[
                          sector]).to_frame('p')
                ts['q'] = (self.grid.network.scenario.timeseries.load[sector] *
                           q_factor)
                self._timeseries = (ts * consumption * power_scaling)
            else:
                try:
                    ts = pd.DataFrame()
                    ts['p'] = self.grid.network.scenario.timeseries.load[
                        sector]
                    ts['q'] = ts['p'] * q_factor
                    self._timeseries = ts * consumption
                except KeyError:
                    logger.exception("No timeseries for load of type {}"
                                     "given.".format(sector))
                    raise
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
            self.grid.network.config['data'][
                'peakload_consumption_ratio']).astype(float), fill_value=0)

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
        """Return time series of generator

        It returns the actual time series used in power flow analysis. If
        :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
        :meth:`timeseries` looks for time series of the according weather cell
        and type of technology in :class:`~.grid.network.TimeSeries` object and
        considers for predefined curtailment as well.
        """
        if self._timeseries is None:
            # calculate share of reactive power
            if isinstance(self.grid, MVGrid):
                q_factor = tan(acos(
                    self.grid.network.scenario.parameters.pfac_mv_gen))
            elif isinstance(self.grid, LVGrid):
                q_factor = tan(acos(
                    self.grid.network.scenario.parameters.pfac_lv_gen))
            # set timeseries for active and reactive power
            if self.grid.network.scenario.mode == 'worst-case':
                ts = self.grid.network.scenario.timeseries.generation.copy()
                ts['q'] = ts['p'] * q_factor
                if self.type == 'solar':
                    power_scaling = float(self.grid.network.config['scenario'][
                                              'scale_factor_feedin_pv'])
                else:
                    power_scaling = float(self.grid.network.config['scenario'][
                                              'scale_factor_feedin_other'])
                self._timeseries = ts * self.nominal_capacity * power_scaling
            else:
                try:
                    ts = pd.DataFrame()
                    ts['p'] = self.grid.network.scenario.timeseries.generation[
                        self.type]
                    ts['q'] = ts['p'] * q_factor
                    self._timeseries = ts * self.nominal_capacity
                except KeyError:
                    try:
                        ts['p'] = self.grid.network.scenario.timeseries.\
                            generation['other']
                        ts['q'] = ts['p'] * q_factor
                        self._timeseries = ts * self.nominal_capacity
                    except KeyError:
                        logger.exception("No timeseries for type {} "
                                         "given.".format(self.type))
                        raise

        curtailment = self.grid.network.scenario.curtailment
        if curtailment:
            if self.type in list(curtailment.keys()):
                self._timeseries = self._timeseries * curtailment[self._type]

        return self._timeseries

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


class Storage(Component):
    """Storage object

    Describes a single storage instance in the eDisGo grid. Includes technical
    parameters like :attr:`Storage.efficiency_in` or
    :attr:`Storage.standing_loss` as
    well as its time series of operation :meth:`Storage.timeseries`.
    The storage's operation is defined by :class:`StorageOperation`.

    Examples
    --------
    In order to define a storage that operates in mode "fifty-fifty"
    (see :ref:`storage-operation` for details about modes)
    provide the following when instantiating a storage.

    >>> from edisgo.grid.components import Storage
    >>> storage_parameters = {'soc_initial': 0,
    >>>                       'efficiency_in': .9,
    >>>                       'efficiency_out': .9,
    >>>                       'standing_loss': 0}
    >>> network.integrate_storage(position='hvmv_substation_busbar',
    >>>                           parameters=storage_parameters)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timeseries = kwargs.get('timeseries', None)
        self._nominal_capacity = kwargs.get('nominal_capacity', None)
        self._soc_initial = kwargs.get('soc_initial', None)
        self._efficiency_in = kwargs.get('efficiency_in', None)
        self._efficiency_out = kwargs.get('efficiency_out', None)
        self._standing_loss = kwargs.get('standing_loss', None)

        operation = kwargs.get('operation', None)
        if operation is not None:
            self._operation = StorageOperation(storage=self,
                                               mode=operation['mode'])
        else:
            self._operation = None

    @property
    def timeseries(self):
        """
        Get time series of storage operation

        Returns time series defined by :attr:`StorageOperation.timeseries` if
        :attr:`operation` is available. Otherwise, time series stored in
        :attr:`timeseries` is returned.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Storage operational time series
        """
        if self._operation is not None:
            return self._operation.timeseries
        else:
            return self._timeseries


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
        Get nominal capacity of storage instance

        Returns
        -------
        float
            Storage nominal capacity
        """
        return self._nominal_capacity

    @property
    def soc_initial(self):
        """Initial state of charge in kWh

        Returns
        -------
        float
            Initial state of charge
        """
        return self._soc_initial

    @property
    def efficiency_in(self):
        """Storage charging efficiency in per unit

        Returns
        -------
        float
            Charging efficiency in range of 0..1
        """
        return self._efficiency_in

    @property
    def efficiency_out(self):
        """Storage discharging efficiency in per unit

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
        StorageOperation
            Class defining operation of a :class:`Storage`
        """


class StorageOperation():
    """
    Define storage operation mode and time series for power flow analysis
    """

    def __init__(self, **kwargs):
        self._timeseries = kwargs.get('timeseries', None)
        self._storage = kwargs.get('storage', None)

        mode = kwargs.get('mode', None)

        if mode is not None:
            self.define_timeseries(mode)

    def define_timeseries(self, mode, feedin_threshold=.5):
        """
        Define time series for :class:`Storage`

        Determine the actual storage time series and save it to
        :attr:`timeseries`.

        Parameters
        ----------
        mode : str
            Choose way of time series definition. Available ``mode`` 's are

             * **'fifty-fifty'**: the storage operation depends on actual power
               by generators. If cumulative generation exceeds 50 % of nominal
               power, the storage will charge. Otherwise, the storage will
               charge.
             * **'etrago-specs'**: the storage operation is given by ETraGo
               specification

        """
        if mode == 'etrago-specs':
            if self._timeseries is None:
                self._timeseries = pd.DataFrame()
                self._timeseries['p'] = self.storage.grid.network.scenario.\
                    etrago_specs.battery_active_power
                self._timeseries['q'] = (self.storage.grid.network.scenario.\
                                            etrago_specs.battery_active_power *
                                         0)
        elif 'fifty-fifty':
            # determine generators cumulative apparent power output
            generators = self.storage.grid.graph.nodes_by_attribute(
                'generator') + [generators for lv_grid in
                                self.storage.grid.lv_grids for generators in
                                lv_grid.graph.nodes_by_attribute('generator')]
            generators_p = pd.concat([_.timeseries['p'] for _ in generators],
                                     axis=1).sum(axis=1).rename('p')
            generators_q = pd.concat([_.timeseries['q'] for _ in generators],
                                     axis=1).sum(axis=1).rename('q')
            generation = pd.concat([generators_p, generators_q], axis=1)
            generation['s'] = generation.apply(
                lambda x: sqrt(x['p'] ** 2 + x['q'] ** 2), axis=1)
            generators_nom_capacity = sum(
                [_.nominal_capacity for _ in generators])
            feedin_bool = generation['s'] > (
                feedin_threshold * generators_nom_capacity)
            feedin = feedin_bool.apply(
                lambda x: self.storage.nominal_capacity if x
                else -self.storage.nominal_capacity).rename('p').to_frame()
            feedin['q'] = 0
            self._timeseries = feedin * self.storage.nominal_capacity
        else:
            raise ValueError('The mode {} is not know as valid storage '
                             'operational mode'.format(mode))

    @property
    def timeseries(self):
        """
        Storage's operational time series

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Storage's operational time series as p and q
        """
        return self._timeseries

    @property
    def storage(self):
        """
        Reference to storage instance

        Returns
        -------
        Storage
            Storage instance this object is associated to
        """
        return self._storage


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

