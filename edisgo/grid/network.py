from edisgo.data.import_data import import_from_ding0
import pandas as pd
from edisgo.tools import interfaces


class Network:
    """Defines the eDisGo Network

    Used as container for all data related to a single
    :class:`~.grid.grids.MVGrid`.

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
        Data Source of grid data (e.g. "ding0")
    _scenario : :class:`~.grid.grids.Scenario`
        Scenario which is used for calculations
    _pypsa : :pypsa:`pypsa.Network<network>`
        PyPSA representation of grid topology
    """

    def __init__(self, **kwargs):
        # TODO: sort out realistic use cases for data from PyPSA network
        if 'pypsa' not in kwargs.keys():
            self._id = kwargs.get('id', None)
            self._equipment_data = kwargs.get('equipment_data', None)
            self._config = kwargs.get('config', None)
            self._metadata = kwargs.get('metadata', None)
            self._data_source = kwargs.get('data_source', None)
            self._scenario = kwargs.get('scenario', None)
            self._mv_grid = kwargs.get('mv_grid', None)
            self._pypsa = None
            self.results = Results()
        else:
            self._pypsa = kwargs.get('pypsa', None)

    @classmethod
    def import_from_ding0(cls, file, **kwargs):
        """Import grid data from DINGO file

        For details see
        :func:`edisgo.data.import_data.import_from_ding0`
        """

        # create the network instance
        network = cls(**kwargs)

        # call the importer
        import_from_ding0(file, network)

        return network

    def import_generators(self):
        """Imports generators

        TBD

        """
        raise NotImplementedError

    def analyze(self, mode=None):
        """Analyzes the grid by power flow analysis

        .. TODO: explain a bunch of details and example how to use
        * what type of power flow is performed
        * how to select time range for analysis
        * representation of grid in PFA
         * No HV-MV transformer
         * ...

        Parameters
        ----------
        mode: str
            Allows to toggle between power flow analysis (PFA) on the whole grid
            topology (MV + LV), only MV or only LV. Therefore, either specify
            `mode='mv'` for PFA of the MV grid topology or `mode='lv'`
            for PFA of the LV grid topology.
            Defaults to None which equals power flow analysis for MV + LV.

        Notes
        -----
        The current implementation always translates the grid topology
        representation to the PyPSA format and stores it to :attr:`self._pypsa`.

        """
        self.pypsa = mode

        # TODO: remove export prior to merge
        self.pypsa.export_to_csv_folder('edisgo2pypsa_export')

        # TODO: check if timeseries dataframes contain all data
        # TODO: maybe 'v_mag_pu_set' is required for buses
        # TODO: maybe there are lv station without load and generation at secondary side
        # TODO: if missing, add slack generator
        self.pypsa.pf(self.pypsa.snapshots)


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
    def pypsa(self):
        return self._pypsa

    @pypsa.setter
    def pypsa(self, mode=None):
        """
        Convert NetworkX based grid topology representation to PyPSA grid
        representation based on :pandas:`pandas.DataFrame<dataframe>`

        Parameters
        ----------
        mode: str
            Allows to toggle between converting the whole grid topology
            (MV + LV), only MV or only LV. Therefore, either specify `mode='mv'`
            for the conversion of the MV grid topology or `mode='lv'`
            for the conversion of the LV grid topology.
            Defaults to None which equals converting MV + LV.

        Notes
        -----
        Tell about
         * How the PyPSA interface is constructed, i.e. splitted in MV and LV
            with combination or attachment of aggregated LV load and generation
            to MV part
         * How power plants are modeled, if possible use a link
         * Recommendations for further development:
            https://github.com/openego/eDisGo/issues/18
         * Where to find and adjust power flow analysis defining parameters

        Returns
        -------
        .. TODO: describe return
        """
        self._pypsa = interfaces.to_pypsa(self, mode)

    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, scenario):
        self._scenario = scenario

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

    @property
    def timeseries(self):
        return self._timeseries

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

    @property
    def timeindex(self):
        # TODO: replace this dummy when time series are ready. Replace by the index of one of the DataFrames
        hours_of_the_year = 8760
        return pd.date_range('12/4/2011', periods=2, freq='H')


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
