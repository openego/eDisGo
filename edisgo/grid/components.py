import os
import logging
import pandas as pd
from math import acos, tan

if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString

logger = logging.getLogger('edisgo')


class Component:
    """
    Generic component

    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._network = kwargs.get('network', None)
        self._grid = kwargs.get('grid', None)

    @property
    def id(self):
        """
        Unique identifier of component as used in component dataframes in
        :class:`~.grid.network.Network`.

        Returns
        --------
        :obj:`str`
            Unique identifier of component.

        """
        return self._id

    @property
    def network(self):
        """
        Network container

        Returns
        --------
        :class:`~.grid.network.Network`

        """
        return self._network

    @property
    def grid(self):
        """
        Grid component is in.

        Returns
        --------
        :class:`~.grid.components.Grid`

        """
        return self._grid

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

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def peak_load(self):
        """
        Peak load in MW.

        Parameters
        -----------
        peak_load : :obj:`float`
            Peak load in MW.

        Returns
        --------
        :obj:`float`
            Peak load in MW.

        """
        return self.network.loads.loc[self.id, 'peak_load']

    @peak_load.setter
    def peak_load(self, peak_load):
        # ToDo: Maybe perform type check before setting it.
        self.network.loads.loc[self.id, 'peak_load'] = peak_load

    @property
    def consumption(self):
        """
        #ToDo Wo soll consumption hergeholt werden?
        """
        return None

    @consumption.setter
    def consumption(self, consumption):
        self._consumption = consumption

    @property
    def sector(self):
        """
        Sector load is associated with.

        The sector is e.g. used to assign load time series to a load using the
        demandlib. The following four sectors are considered:
        'agricultural', 'retail', 'residential', 'industrial'.

        Parameters
        -----------
        sector : :obj:`str`

        Returns
        --------
        :obj:`str`
            Load sector

        #ToDo: Maybe return 'not specified' in case sector is None?

        """
        return self.network.loads.loc[self.id, 'sector']

    @sector.setter
    def sector(self, sector):
        # ToDo: Maybe perform type check before setting it.
        self.network.loads.loc[self.id, 'sector'] = sector

    @property
    def grid(self):
        """
        Grid load is in.

        Returns
        --------
        :class:`~.grid.components.Grid`
            Grid load is in.

        """
        if self._grid is None:
            grid = self.network.buses.loc[
                self.network.loads.loc[self.id, 'bus'],
                ['mv_grid_id', 'lv_grid_id']]
            if grid.lv_grid_id is None:
                return self.network.mv_grid
            else:
                return self.network._grids['LVGrid_{}'.format(grid.lv_grid_id)]
        else:
            return self._grid

    @property
    def voltage_level(self):
        """
        Voltage level the load is connected to ('mv' or 'lv').

        Returns
        --------
        :obj:`str`
            Voltage level

        """
        return 'lv' if self.grid.nominal_voltage < 1 else 'mv'

    @property
    def active_power_timeseries(self):
        """
        Active power time series of load in MW.

        Returns
        --------
        :pandas:`pandas.Series<series>`
            Active power time series of load in MW.

        """
        return self.network.loads_t.p_set.loc[self.id]

    @property
    def reactive_power_timeseries(self):
        """
        Reactive power time series of load in Mvar.

        Returns
        --------
        :pandas:`pandas.Series<series>`
            Reactive power time series of load in Mvar.

        """
        return self.network.loads_t.q_set.loc[self.id]

    @property
    def geom(self):
        """
        Geo location of generator.

        Returns
        --------
        shapely Point

        """
        geom = self.network.buses.loc[
            self.network.loads.loc[self.id, 'bus'], ['x', 'y']]
        # ToDo return shapely point
        pass


class Generator(Component):
    """
    Generator object

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def nominal_power(self):
        """
        Nominal power of generator in MW.

        Parameters
        -----------
        nominal_power : :obj:`float`
            Nominal power of generator in MW.

        Returns
        --------
        :obj:`float`
            Nominal power of generator in MW.

        """
        return self.network.generators_df.loc[self.id, 'nominal_power']

    @nominal_power.setter
    def nominal_power(self, nominal_power):
        # ToDo: Maybe perform type check before setting it.
        self.network.generators_df.loc[
            self.id, 'nominal_power'] = nominal_power

    @property
    def type(self):
        """
        Technology type of generator (e.g. 'solar').

        Parameters
        -----------
        type : :obj:`str`

        Returns
        --------
        :obj:`str`
            Technology type

        #ToDo: Maybe return 'not specified' in case type is None?

        """
        return self.network.generators_df.loc[self.id, 'type']

    @type.setter
    def type(self, type):
        #ToDo: Maybe perform type check before setting it.
        self.network.generators_df.loc[self.id, 'type'] = type

    @property
    def subtype(self):
        """
        Technology subtype of generator (e.g. 'solar_roof_mounted').

        Parameters
        -----------
        subtype : :obj:`str`

        Returns
        --------
        :obj:`str`
            Technology subtype

        #ToDo: Maybe return 'not specified' in case subtype is None?

        """
        return self.network.generators_df.loc[self.id, 'subtype']

    @subtype.setter
    def subtype(self, subtype):
        #ToDo: Maybe perform type check before setting it.
        self.network.generators_df.loc[self.id, 'subtype'] = subtype

    @property
    def grid(self):
        """
        Grid generator is in.

        Returns
        --------
        :class:`~.grid.components.Grid`
            Grid generator is in.

        """
        if self._grid is None:
            grid = self.network.buses_df.loc[
                self.network.generators_df.loc[self.id, 'bus'],
                ['mv_grid_id', 'lv_grid_id']]
            if grid.lv_grid_id is None:
                return self.network.mv_grid
            else:
                return self.network._grids['LVGrid_{}'.format(grid.lv_grid_id)]
        else:
            return self._grid

    @property
    def voltage_level(self):
        """
        Voltage level the generator is connected to ('mv' or 'lv').

        Returns
        --------
        :obj:`str`
            Voltage level

        """
        return 'lv' if self.grid.nominal_voltage < 1 else 'mv'

    @property
    def active_power_timeseries(self):
        """
        Active power time series of generator in MW.

        Returns
        --------
        :pandas:`pandas.Series<series>`
            Active power time series of generator in MW.

        """
        return self.network.generators_t.p_set.loc[self.id]

    @property
    def reactive_power_timeseries(self):
        """
        Reactive power time series of generator in Mvar.

        Returns
        --------
        :pandas:`pandas.Series<series>`
            Reactive power time series of generator in Mvar.

        """
        return self.network.generators_t.q_set.loc[self.id]

    @property
    def weather_cell_id(self):
        """
        Weather cell ID of generator.

        The weather cell ID is only used to obtain generator feed-in time
        series for solar and wind generators.

        Returns
        --------
        :obj:`int`
            Weather cell ID of generator.

        """
        return self.network.generators_df.loc[self.id, 'weather_cell_id']

    @property
    def geom(self):
        """
        Geo location of generator.

        Returns
        --------
        shapely Point

        """
        geom = self.network.buses_df.loc[
            self.network.generators_df.loc[self.id, 'bus'], ['x', 'y']]
        #ToDo return shapely point
        pass


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

    def pypsa_timeseries(self):
        """Return time series in PyPSA format

        Convert from kW, kVA to MW, MVA

        """
        return self.timeseries / 1e3

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

    # @property
    # def power_factor(self):
    #     """
    #     Power factor of storage
    #
    #     If power factor is not set it is retrieved from the network config
    #     object depending on the grid level the storage is in.
    #
    #     Returns
    #     --------
    #     :obj:`float` : Power factor
    #         Ratio of real power to apparent power.
    #
    #     """
    #     if self._power_factor is None:
    #         if isinstance(self.grid, MVGrid):
    #             self._power_factor = self.grid.network.config[
    #                 'reactive_power_factor']['mv_storage']
    #         elif isinstance(self.grid, LVGrid):
    #             self._power_factor = self.grid.network.config[
    #                 'reactive_power_factor']['lv_storage']
    #     return self._power_factor
    #
    # @power_factor.setter
    # def power_factor(self, power_factor):
    #     self._power_factor = power_factor

    # @property
    # def reactive_power_mode(self):
    #     """
    #     Power factor mode of storage.
    #
    #     If the power factor is set, then it is necessary to know whether
    #     it is leading or lagging. In other words this information is necessary
    #     to make the storage behave in an inductive or capacitive manner.
    #     Essentially this changes the sign of the reactive power Q.
    #
    #     The convention used here in a storage is that:
    #     - when `reactive_power_mode` is 'capacitive' then Q is positive
    #     - when `reactive_power_mode` is 'inductive' then Q is negative
    #
    #     In the case that this attribute is not set, it is retrieved from the
    #     network config object depending on the voltage level the storage
    #     is in.
    #
    #     Returns
    #     -------
    #     :obj: `str` : Power factor mode
    #         Either 'inductive' or 'capacitive'
    #
    #     """
    #     if self._reactive_power_mode is None:
    #         if isinstance(self.grid, MVGrid):
    #             self._reactive_power_mode = self.grid.network.config[
    #                 'reactive_power_mode']['mv_storage']
    #         elif isinstance(self.grid, LVGrid):
    #             self._reactive_power_mode = self.grid.network.config[
    #                 'reactive_power_mode']['lv_storage']
    #
    #     return self._reactive_power_mode

    # @reactive_power_mode.setter
    # def reactive_power_mode(self, reactive_power_mode):
    #     """
    #     Set the power factor mode of the generator.
    #     Should be either 'inductive' or 'capacitive'
    #     """
    #     self._reactive_power_mode = reactive_power_mode

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
            self._nodes[0], self._nodes[1], line=self._line)

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

