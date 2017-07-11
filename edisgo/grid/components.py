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
    _timeseries : #TODO
    _consumption : #TODO

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._timeseries = kwargs.get('timeseries', None)
        self._consumption = kwargs.get('consumption', None)


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
    _timeseries : #TODO

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

