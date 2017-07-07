
class Network:
    """Defines the eDisGo Network

    Used as container for all data related to a single MV grid.

    Attributes
    ----------
    _id : :obj:`str`
        Name of network
    _equipment_data : :obj:`dict` of `pandas.core.frame.DataFrame`
        Electrical equipment such as lines and transformers
    _config : ???
    _metadata : :obj:`dict`
        Metadata of Network such as
    _data_source : :obj:`str`
        Data Source of grid data (e.g. "dingo")
    _scenario : `edisgo.grid.network.Scenario`
        Scenario which is used for calculations
    _mv_grid : `edisgo.grid.grids.MVGrid`
        Medium voltage (MV) grid
    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._equipment_data = kwargs.get('equipment_data', None)
        self._config = kwargs.get('config', None)
        self._metadata = kwargs.get('metadata', None)
        self._data_source = kwargs.get('data_source', None)
        self._scenarios = kwargs.get('scenarios', None)
        self._mv_grid = kwargs.get('mv_grid', None)


class Scenario:
    """Defines an eDisGo scenario

    It contains

    Attributes
    ----------
    _scenario : :obj:`str`
        Scenario name (e.g. "feedin case weather 2011")
    _timeseries : :obj:`list` of `edisgo.grid.network.TimeSeries`
        Time series associated to a scenario
    _etrago_spec : `edisgo.grid.network.eTraGoSpec`
        Specifications which are to be fulfilled at transition point (HV-MV substation)
    _pfac_mv_gen : :obj:`float`
        Power factor for medium voltage generators
    _pfac_mv_load : :obj:`float`
        Power factor for medium voltage loads
    _pfac_lv_gen : :obj:`float`
        Power factor for low voltage generators
    _pfac_lv_load : :obj:`float`
        Power factor for low voltage loads
    """

    def __init__(self, **kwargs):
        self._scenario = kwargs.get('scenario', None)
        self._timeseries = kwargs.get('timeseries', None)
        self._etrago_spec = kwargs.get('etrago_spec', None)
        self._pfac_mv_gen = kwargs.get('pfac_mv_gen', None)
        self._pfac_mv_load = kwargs.get('pfac_mv_load', None)
        self._pfac_lv_gen = kwargs.get('pfac_lv_gen', None)
        self._pfac_lv_load = kwargs.get('pfac_lv_load', None)


class TimeSeries:
    """Defines an eDisGo time series

    Attributes
    ----------
    _scenario : :obj:`str`
        Scenario name (e.g. "feedin case weather 2011")
    _pfac_mv_gen : :obj:`float`
        Power factor for medium voltage generators
    _pfac_mv_load : :obj:`float`
        Power factor for medium voltage loads
    _pfac_lv_gen : :obj:`float`
        Power factor for low voltage generators
    _pfac_lv_load : :obj:`float`
        Power factor for low voltage loads
    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)


class eTraGoSpec:
    """

    Specifications which are to be fulfilled at transition point (HV-MV substation)

    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)