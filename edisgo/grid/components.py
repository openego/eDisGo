class Load:
    """Load object """

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
