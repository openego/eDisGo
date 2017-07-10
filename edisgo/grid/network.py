from edisgo.data.import_data import import_from_dingo


class Network:
    """Defines the eDisGo Network

    Used as container for all data related to a single MV grid.

    Attributes
    ----------
    _id : :obj:`str`
        Name of network
    _equipment_data : :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
        Electrical equipment such as lines and transformers
    _config : ???
        #TODO: TBD
    _metadata : :obj:`dict`
        Metadata of Network such as ?
    _data_source : :obj:`str`
        Data Source of grid data (e.g. "dingo")
    _scenario : Scenario
        Scenario which is used for calculations
    _mv_grid : MVGrid
        Medium voltage (MV) grid
    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._equipment_data = kwargs.get('equipment_data', None)
        self._config = kwargs.get('config', None)
        self._metadata = kwargs.get('metadata', None)
        self._data_source = kwargs.get('data_source', None)
        self._scenario = kwargs.get('scenario', None)
        self._mv_grid = kwargs.get('mv_grid', None)

    @classmethod
    def import_from_dingo(cls, file):
        """Import grid data from DINGO grid data saved as pickle

        This includes grid elements such as lines, transformers, branch tees, loads and generators.

        """
        import_from_dingo(file)

        # TODO: finalize instantiation call
        # TODO: try to move most of the code outside this function. This is maybe not possible for the network itself (then use the cls() call
        # return cls(id='id')
        # raise NotImplementedError

    def import_generators(self):
        """Imports generators

        TBD

        """
        raise NotImplementedError

    def analyze(self):
        """Analyzes the grid

        TBD

        """
        raise NotImplementedError

    def reinforce(self):
        """Reinforces the grid

        TBD

        """
        raise NotImplementedError

    def __repr__(self):
        return 'Network ' + self._id


class Scenario:
    """Defines an eDisGo scenario

    It contains parameters and links to further data that is used for calculations within eDisGo.

    Attributes
    ----------
    _name : :obj:`str`
        Scenario name (e.g. "feedin case weather 2011")
    _network : Network
        Network which this scenario is associated with
    _timeseries : :obj:`list` of TimeSeries
        Time series associated to a scenario
    _etrago_specs : ETraGoSpecs
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
        self._name = kwargs.get('name', None)
        self._network = kwargs.get('network', None)
        self._timeseries = kwargs.get('timeseries', None)
        self._etrago_specs = kwargs.get('etrago_specs', None)
        self._pfac_mv_gen = kwargs.get('pfac_mv_gen', None)
        self._pfac_mv_load = kwargs.get('pfac_mv_load', None)
        self._pfac_lv_gen = kwargs.get('pfac_lv_gen', None)
        self._pfac_lv_load = kwargs.get('pfac_lv_load', None)

    def __repr__(self):
        return 'Scenario ' + self._name


class TimeSeries:
    """Defines an eDisGo time series

    Contains time series for loads and generators (technology-specific), e.g. tech. solar, sub-tech. rooftop.

    Attributes
    ----------
    _generation : :obj:`dict` of :obj:`dict` of :pandas:`pandas.Series<series>`
        Time series of active power of generators for technologies and sub-technologies,
        format: {tech_1: {sub-tech_1_1: timeseries_1_1, ..., sub-tech_1_n: timeseries_1_n},
                 ...,
                 tech_m: {sub-tech_m_1: timeseries_m_1, ..., sub-tech_m_n: timeseries_m_n}
                 }
    _load : :pandas:`pandas.Series<series>`
        Time series of active power of (cumulative) loads
    """

    def __init__(self, **kwargs):
        self._generation = kwargs.get('generation', None)
        self._load = kwargs.get('load', None)


class ETraGoSpecs:
    """Defines an eTraGo object used in project open_eGo

    Contains specifications which are to be fulfilled at transition point (superiorHV-MV substation)
    for a specific scenario.

    Attributes
    ----------
    _active_power : :pandas:`pandas.Series<series>`
        Time series of active power at Transition Point
    _reactive_power : :pandas:`pandas.Series<series>`
        Time series of reactive power at Transition Point
    _battery_capacity: :obj:`float`
        Capacity of virtual battery at Transition Point
    _battery_active_power : :pandas:`pandas.Series<series>`
        Time series of active power the (virtual) battery (at Transition Point) is charged (negative)
        or discharged (positive) with
    _curtailment : :obj:`dict` of :obj:`dict` of :pandas:`pandas.Series<series>`
        #TODO: Is this really an active power value or a ratio (%) ?
        Time series of active power curtailment of generators for technologies and sub-technologies,
        format: {tech_1: {sub-tech_1_1: timeseries_1_1, ..., sub-tech_1_n: timeseries_1_n},
                 ...,
                 tech_m: {sub-tech_m_1: timeseries_m_1, ..., sub-tech_m_n: timeseries_m_n}
                 }
    """

    def __init__(self, **kwargs):
        self._active_power = kwargs.get('active_power', None)
        self._reactive_power = kwargs.get('reactive_power', None)
        self._battery_capacity = kwargs.get('battery_capacity', None)
        self._battery_active_power = kwargs.get('battery_active_power', None)
        self._curtailment = kwargs.get('curtailment', None)


