from shapely.geometry import LineString
from math import acos, tan
import pandas as pd


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

    @property
    def geom(self):
        """:shapely:`Shapely Point object<points>` or
        :shapely:`Shapely LineString object<linestrings>` : Location of the
        :class:`Component` as Shapely Point or LineString"""
        return self._geom

    @property
    def grid(self):
        """:class:`~.grid.grids.MVGrid` or :class:`~.grid.grids.LVGrid` : The MV or LV grid this component belongs to"""
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
        Specification of type, refers to #TODO: ADD CORRECT REF TO (STATIC) DATA
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._voltage_op = kwargs.get('voltage_op', None)
        self._type = kwargs.get('type', None)

    @property
    def voltage_op(self):
        return self._voltage_op

    @property
    def type(self):
        return self._type


class Load(Component):
    """Load object

    Attributes
    ----------
    _timeseries : :pandas:`pandas.Series<series>`
        Contains time series for load
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._timeseries = kwargs.get('timeseries', None)
        self._consumption = kwargs.get('consumption', None)

        # TODO: replace below dummy timeseries
        hours_of_the_year = 8760

        cos_phi = 0.95

        q_factor = tan(acos(0.95))

        avg_hourly_load = {k: v / hours_of_the_year / 1e3
                           for k, v in self.consumption.items()}

        rng = pd.date_range('1/1/2011', periods=hours_of_the_year, freq='H')

        ts_dict_p = {
            (k, 'p'): [avg_hourly_load[k] * (
            1 - q_factor)] * hours_of_the_year
            for k in avg_hourly_load.keys()}
        ts_dict_q = {
            (k, 'q'): [avg_hourly_load[k] * (
                q_factor)] * hours_of_the_year
            for k in avg_hourly_load.keys()}
        ts_dict = {**ts_dict_p, **ts_dict_q}

        self._timeseries = pd.DataFrame(ts_dict, index=rng)

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

        return self._timeseries

    # @property
    def pypsa_timeseries(self, sector, attr):
        """Return time series in PyPSA format

        Parameters
        ----------
        sector : str
            Sectoral load that is of interest. Valid sectors {residential,
            retail, agricultural, industrial}
        attr : str
            Attribute name (PyPSA conventions). Choose from {p_set, q_set}
        """

        pypsa_component_name = '_'.join([repr(self), sector])

        return self._timeseries[(sector, attr)]

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

        # TODO: replace below dummy timeseries
        hours_of_the_year = 8760

        cos_phi = 0.95

        q_factor = tan(acos(0.95))

        rng = pd.date_range('1/1/2011', periods=hours_of_the_year, freq='H')

        ts_dict = {
            'p': [self.nominal_capacity / 1e3 * (1 - q_factor)] * hours_of_the_year,
            'q': [self.nominal_capacity / 1e3 * q_factor] * hours_of_the_year}

        self._timeseries = pd.DataFrame(ts_dict, index=rng)
        # self._timeseries = kwargs.get('timeseries', None)

    def timeseries(self):
        """Return time series of generator

        It returns the actual time series used in power flow analysis. If
        :attr:`_timeseries` is not :obj:`None`, it is returned. Otherwise,
        :meth:`timeseries` looks for time series of the according weather cell
        and type of technology in :class:`~.grid.network.TimeSeries` object and
        considers for predefined curtailment as well.
        """
        return self._timeseries

    def pypsa_timeseries(self, attr):
        """Return time series in PyPSA format

        Parameters
        ----------
        attr : str
            Attribute name (PyPSA conventions). Choose from {p_set, q_set}
        """

        return self._timeseries[attr]

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


class Storage(Component):
    """Storage object

    Attributes
    ----------
    TBC
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MVDisconnectingPoint(Component):
    """Disconnecting point object

    Medium voltage disconnecting points = points where MV rings are split under
    normal operation conditions (= switch disconnectors in DINGO).

    Attributes
    ----------
    _state : :obj:`str`
        State of switch ('open' or 'closed')
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._state = kwargs.get('state', None)

    def open(self):
        """Toggle state to opened switch disconnector"""
        raise NotImplementedError

    def close(self):
        """Toggle state to closed switch disconnector"""
        raise NotImplementedError


class BranchTee(Component):
    """Branch tee object

    A branch tee is used to branch off a line to connect another node
    (german: Abzweigmuffe)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MVStation(Station):
    """MV Station object"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


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
        U_n      Nominal voltage    V      int
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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = kwargs.get('type', None)
        self._length = kwargs.get('length', None)
        self._quantity = kwargs.get('quantity', 1)

    @property
    def geom(self):
        """Provide :shapely:`Shapely LineString object<linestrings>` geometry of
        :class:`Line`"""
        adj_nodes = self._grid._graph.nodes_from_line(self)

        return LineString([adj_nodes[0].geom, adj_nodes[1].geom])

    @property
    def type(self):
        return self._type

    @property
    def length(self):
        return self._length

