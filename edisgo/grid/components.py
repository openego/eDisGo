class Station:
    """Station object (medium or low voltage)

    Represents a station, contains transformers.

    Attributes
    ----------
    _id : :obj:`int`
        Unique ID
    _geom : :shapely:`Shapely Point object<points>`
        Location as Shapely Point object
    _transformers : :obj:`list` of Tran
        Unique ID
    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._geom = kwargs.get('geom', None)
        self._transformers = kwargs.get('transformers', None)


class Load:
    """Load object

    Attributes
    ----------
    _id : :obj:`str`
        Name of network
    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._timeseries = kwargs.get('timeseries', None)
        self._geom = kwargs.get('geom', None)
        self._consumption = kwargs.get('consumption', None)


class Generator:
    """Generator object"""

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._geom = kwargs.get('geom', None)
        self._nominal_capacity = kwargs.get('nominal_capacity', None)
        self._type = kwargs.get('type', None)
        self._subtype = kwargs.get('subtype', None)
        self._timeseries = kwargs.get('timeseries', None)


class MVDisconnectingPoint:
    """Location in MV grid ring where switch disconnector is placed"""

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._geom = kwargs.get('geom', None)
        self._state = kwargs.get('state', None)

    def open(self):
        """Toggle state to opened switch disconnector"""
        raise NotImplementedError

    def close(self):
        """Toggle state to closed switch disconnector"""
        raise NotImplementedError


class BranchTee:
    """Branch tee for branching lines"""

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._geom = kwargs.get('geom', None)


