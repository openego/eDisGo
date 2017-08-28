from edisgo.tools import config
from edisgo.data.import_data import import_from_dingo, import_generators
import pandas as pd


class Network:
    """Defines the eDisGo Network

    Used as container for all data related to a single
    :class:`~.grid.grids.MVGrid`.

    Attributes
    ----------
    _id : :obj:`str`
        Name of network
    _metadata : :obj:`dict`
        Metadata of Network such as ?
    _data_sources : :obj:`dict` of :obj:`str`
        Data Sources of grid, generators etc.
        Keys: 'grid', 'generators', ?
    _scenario : :class:`~.grid.grids.Scenario`
        Scenario which is used for calculations
    _config :
        #TODO: TBD
    _equipment_data : :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
        Electrical equipment such as lines and transformers
    # TODO: Add remaining attributes
    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._metadata = kwargs.get('metadata', None)
        self._data_sources = kwargs.get('data_sources', {})
        self._scenario = kwargs.get('scenario', None)
        self._mv_grid = kwargs.get('mv_grid', None)
        self.results = Results()

        self._config = self._load_config()
        self._equipment_data = self._load_equipment_data()

    @staticmethod
    def _load_config():
        """Load config files

        Returns
        -------
        config object
        """

        config.load_config('config_db_tables.cfg')
        config.load_config('config_data.cfg')
        config.load_config('config_scenario.cfg')

        return config.cfg.sections()

    @staticmethod
    def _load_equipment_data():
        """Load equipment data for transformers, cables etc.

        Returns
        -------
        :obj:`dict` of :pandas:`pandas.DataFrame<dataframe>`
        """

        raise NotImplementedError

    @classmethod
    def import_from_dingo(cls, file):
        """Import grid data from DINGO file

        For details see
        :func:`edisgo.data.import_data.import_from_dingo`
        """

        # create the network instance
        network = cls()

        # call the importer
        import_from_dingo(file, network)

        return network

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

    @property
    def mv_grid(self):
        """:class:`~.grid.grids.MVGrid` : Medium voltage (MV) grid

        Retrieve the instance of the loaded MV grid
        """
        return self._mv_grid

    @mv_grid.setter
    def mv_grid(self, mv_grid):
        self._mv_grid = mv_grid

    @property
    def data_sources(self):
        """:obj:`dict` of :obj:`str` : Data Sources

        """
        return self._data_sources

    def set_data_source(self, key, data_source):
        """Set data source for key (e.g. 'grid')
        """
        self._data_sources[key] = data_source

    def __repr__(self):
        return 'Network ' + self._id


class Scenario:
    """Defines an eDisGo scenario

    It contains parameters and links to further data that is used for
    calculations within eDisGo.

    Attributes
    ----------
    _name : :obj:`str`
        Scenario name (e.g. "feedin case weather 2011")
    _network : :class:~.grid.network.Network`
        Network which this scenario is associated with
    _timeseries : :obj:`list` of :class:`~.grid.grids.TimeSeries`
        Time series associated to a scenario
    _etrago_specs : :class:`~.grid.grids.ETraGoSpecs`
        Specifications which are to be fulfilled at transition point (HV-MV
        substation)
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

    Contains time series for loads (sector-specific) and generators
    (technology-specific), e.g. tech. solar, sub-tech. rooftop.

    Attributes
    ----------
    _generation : :obj:`dict` of :obj:`dict` of :pandas:`pandas.Series<series>`
        Time series of active power of generators for technologies and
        sub-technologies, format:

        .. code-block:: python

            {tech_1: {
                sub-tech_1_1: timeseries_1_1,
                ...,
                sub-tech_1_n: timeseries_1_n},
                 ...,
            tech_m: {
                sub-tech_m_1: timeseries_m_1,
                ...,
                sub-tech_m_n: timeseries_m_n}
            }

    _load : :obj:`dict` of :pandas:`pandas.Series<series>`
        Time series of active power of (cumulative) loads,
        format:

        .. code-block:: python

            {
                sector_1:
                    timeseries_1,
                    ...,
                sector_n:
                    timeseries_n
            }

    See also
    --------
    edisgo.grid.components.Generator : Usage details of :meth:`_generation`
    edisgo.grid.components.Load : Usage details of :meth:`_load`
    """

    def __init__(self, **kwargs):
        self._generation = kwargs.get('generation', None)
        self._load = kwargs.get('load', None)


class ETraGoSpecs:
    """Defines an eTraGo object used in project open_eGo

    Contains specifications which are to be fulfilled at transition point
    (superiorHV-MV substation) for a specific scenario.

    Attributes
    ----------
    _active_power : :pandas:`pandas.Series<series>`
        Time series of active power at Transition Point
    _reactive_power : :pandas:`pandas.Series<series>`
        Time series of reactive power at Transition Point
    _battery_capacity: :obj:`float`
        Capacity of virtual battery at Transition Point
    _battery_active_power : :pandas:`pandas.Series<series>`
        Time series of active power the (virtual) battery (at Transition Point)
        is charged (negative) or discharged (positive) with
    _curtailment : :obj:`dict` of :obj:`dict` of :pandas:`pandas.Series<series>`
        Time series of active power curtailment of generators for technologies
        and sub-technologies, format::

            {
                tech_1: {
                    sub-tech_1_1:
                        timeseries_1_1,
                        ...,
                    sub-tech_1_n:
                    timeseries_1_n
                    },
                ...,
                tech_m: {
                    sub-tech_m_1:
                        timeseries_m_1,
                        ...,
                    sub-tech_m_n:
                        timeseries_m_n
                        }
                 }

        .. TODO: Is this really an active power value or a ratio (%) ?
    """

    def __init__(self, **kwargs):
        self._active_power = kwargs.get('active_power', None)
        self._reactive_power = kwargs.get('reactive_power', None)
        self._battery_capacity = kwargs.get('battery_capacity', None)
        self._battery_active_power = kwargs.get('battery_active_power', None)
        self._curtailment = kwargs.get('curtailment', None)


class Results:
    """
    Power flow analysis results managment

    Includes raw power flow analysis results, history of measures to increase
    the grid's hosting capacity and information about changes of equipment.

    Attributes
    ----------
    measures: list
        A stack that details the history of measures to increase grid's hosting
        capacity. The last item refers to the latest measure. The key `original`
        refers to the state of the grid topology as it was initially imported.
    pfa_nodes: :pandas:`pandas.DataFrame<dataframe>`
        Holds power flow analysis results for nodes in the grid topology from
        several runs. Each run corresponds to and is indexed by an item of the
        stack `measures`.
    pfa_edges: :pandas:`pandas.DataFrame<dataframe>`
        Holds power flow analysis results for edges in the grid topology from
        several runs. Each run corresponds to and is indexed by an item of the
        stack `measures`.
    equipment_changes: :pandas:`pandas.DataFrame<dataframe>`
        Tracks changes in the equipment (replaced or added cable, batteries
        added, curtailment set to a generator, ...). This is indexed by the
        components (nodes or edges) and has following columns:

        equipment: detailing what was changed (line, battery, curtailment). For
        ease of referencing we take the component itself. For lines we take the
        line-dict, for batteries the battery-object itself and for curtailment
        either a dict providing the details of curtailment or a curtailment
        object if this makes more sense (has to be defined).

        change: {added | removed} - says if something was added or removed
    """

    # TODO: maybe add setter to alter list of measures

    # TODO: maybe initialize DataFrames `pfa_nodes` different. Like with index of all components of similarly

    def __init__(self):
        self.measures = ['original']
        self.pfa_nodes = pd.DataFrame()
        self.pfa_edges = pd.DataFrame()
        self.equipment_changes = pd.DataFrame()
