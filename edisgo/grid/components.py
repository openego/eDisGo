from shapely.geometry import LineString


class Component:
    """Generic component

    _id : :obj:`int`
        Unique ID
    _geom : :shapely:`Shapely Point object<points>` or `Shapely LineString object<linestrings>`
        Location as Shapely Point object
    _grid : #TODO: ADD CORRECT REF
        The MV or LV grid this component belongs to

    Notes
    -----
    In case of a MV-LV voltage station, `_grid` refers to the LV grid
    """
    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._geom = kwargs.get('geom', None)
        self._grid = kwargs.get('grid', None)

    @property
    def geom(self):
        """Provide access to geom"""
        return self._geom


class Station(Component):
    """Station object (medium or low voltage)

    Represents a station, contains transformers.

    Attributes
    ----------
    _transformers : :obj:`list` of Transformer
        Transformers located in station
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._transformers = kwargs.get('transformers', None)


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


class Load(Component):
    """Load object

    Attributes
    ----------
    _timeseries : :pandas:`pandas.Series<series>`
        Contains time series for load

    _consumption : :obj:`dict`
        Contains annual consumption in
        #TODO: To implement consumption, DINGO #208 has to be solved first:

    Notes
    -----
    The
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._timeseries = kwargs.get('timeseries', None)
        self._consumption = kwargs.get('consumption', None)

    def timeseries(self):
        """Return time series of load

        It returns the actual time series used in power flow analysis. If `_timeseries` is not None,
        it is returned. Otherwise, timeseries() looks for time series of the according sector in
        `TimeSeries` object.

        See also
        --------
        edisgo.network.TimeSeries : Details of global TimeSeries

        #TODO: CHECK REFS IN TEXT -> MAKE LINKS WORK
        """
        raise NotImplementedError


class Generator(Component):
    """Generator object

    Attributes
    ----------
    _nominal_capacity : :obj:`float`
        Nominal generation capacity
    _type : :obj:`str`
        Technology type (e.g. 'solar')
    _subtype : :obj:`str`
        Technology subtype (e.g. 'solar rooftop')
    _timeseries : :pandas:`pandas.Series<series>`
        Contains time series for generator

    Notes
    -----
    The attributes `_type` and `_subtype` have to match the corresponding types in Timeseries to
    allow allocation of time series to generators.

    #TODO: CHECK REFS IN TEXT -> MAKE LINKS WORK
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._nominal_capacity = kwargs.get('nominal_capacity', None)
        self._type = kwargs.get('type', None)
        self._subtype = kwargs.get('subtype', None)
        self._timeseries = kwargs.get('timeseries', None)

    def timeseries(self):
        """Return time series of generator

        It returns the actual time series used in power flow analysis. If `_timeseries` is not None,
        it is returned. Otherwise, timeseries() looks for time series of the according weather and
        type of technology in `TimeSeries` object and considers for predefined curtailment as well.

        See also
        --------
        edisgo.network.TimeSeries : Details of global TimeSeries

        #TODO: CHECK REFS IN TEXT -> MAKE LINKS WORK
        """
        raise NotImplementedError


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

    A branch tee is used to branch off a line to connect another node (german: Abzweigmuffe)
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
        .. code-block:: python

            {
                'name': name (str),
                'U_n': 'nominal voltage' (int),
                'I_max_th' (A),
                'R': resistance (ohm/km),
                'L': inductivity (mH/km),
                'C': capacity (uF/km),
                'Source': data source (str)
            }

            name: str
            ,U_n,I_max_th,R,L,C,Source
    _length: float
        Length of the line calculated in linear distance. Unit: m
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = kwargs.get('type', None)
        self._length = kwargs.get('length', None)

    @property
    def geom(self):
        """Provide LineString geometry of line object"""
        adj_nodes = self._grid._graph.nodes_from_line(self)

        return LineString([adj_nodes[0].geom, adj_nodes[1].geom])

