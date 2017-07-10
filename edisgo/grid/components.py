class Load:
    """Load object """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._timeseries = kwargs.get('timeseries', None)
        self._geom = kwargs.get('geom', None)
        self._consumption = kwargs.get('consumption', None)


