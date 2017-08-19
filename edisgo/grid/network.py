from edisgo.data.import_data import import_from_dingo
from ..utils import interfaces


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
        Data Source of grid data (e.g. "dingo")
    _scenario : :class:`~.grid.grids.Scenario`
        Scenario which is used for calculations
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
         * How power plants are modeled, if possible use a link
         * Recommendations for further development
         * Where to find and adjust power flow analysis defining parameters

        Returns
        -------
        .. TODO: describe return
        """
        if mode is None:
            mv_components = interfaces.mv_to_pypsa(self)
            lv_components = interfaces.lv_to_pypsa(self)
            components = interfaces.combine_mv_and_lv(mv_components,
                                                      lv_components)
        elif mode is 'mv':
            interfaces.mv_to_pypsa(self)
        elif mode is 'lv':
            interfaces.lv_to_pypsa(self)
        else:
            raise ValueError("Provide proper mode or leave it empty to export "
                             "entire grid topology.")


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
