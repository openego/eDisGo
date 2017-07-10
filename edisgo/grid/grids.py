import networkx as nx
class Grid:
    """Defines a basic grid in eDisGo

    Attributes
    ----------
    _network : Network #TODO: ADD CORRECT REF
        Network which this scenario is associated with
    _voltage_nom : int
        Nominal voltage
    _peak_load : :obj:`float`
        Cumulative peak load of grid
    _peak_generation : :obj:`float`
        Cumulative peak generation of grid
    _grid_district : :obj:`dict`
        Contains information about grid district (supplied region) of grid,
        format: #TODO: DEFINE FORMAT
    """

    def __init__(self, **kwargs):
        self._network = kwargs.get('network', None)
        self._voltage_nom = kwargs.get('voltage_nom', None)
        self._peak_load = kwargs.get('peak_load', None)
        self._peak_generation = kwargs.get('peak_generation', None)
        self._grid_district = kwargs.get('grid_district', None)

        self._graph = nx.Graph()

class MVGrid(Grid):
    """Defines a medium voltage grid in eDisGo

    Attributes
    ----------
    _mv_disconn_points : :obj:`list` of
        Medium voltage disconnecting points = points where MV rings are split under
        normal operation conditions (= switch disconnectors in DINGO).
    """

    def __init__(self, **kwargs):
        self._mv_disconn_points = kwargs.get('mv_disconn_points', None)


# TODO: Go on here..

