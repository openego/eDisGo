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


class Station:
    """Station object"""

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._geom = kwargs.get('geo', None)
        self._transformers = kwargs.get('transformers', None)


class LVStation(Station):
    """LV Station object"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MVStation(Station):
    """MV Station object"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)