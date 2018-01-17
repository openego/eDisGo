import networkx as nx
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


class Grid:
    """Defines a basic grid in eDisGo

    Attributes
    ----------
    _id : :obj:`str`
        Identifier
    _network : :class:`~.grid.network.Network`
        Network which this grid is associated with
    _voltage_nom : int
        Nominal voltage
    _peak_load : :obj:`float`
        Cumulative peak load of grid
    _peak_generation : :obj:`float`
        Cumulative peak generation of grid
    _grid_district : :obj:`dict`
        Contains information about grid district (supplied region) of grid,
        format: #TODO: DEFINE FORMAT
    _station : :class:`~.grid.components.Station`
        The station the grid is fed by
    """

    def __init__(self, **kwargs):
        self._id = kwargs.get('id', None)
        self._network = kwargs.get('network', None)
        self._voltage_nom = kwargs.get('voltage_nom', None)
        self._peak_load = kwargs.get('peak_load', None)
        self._peak_generation = kwargs.get('peak_generation', None)
        self._grid_district = kwargs.get('grid_district', None)
        self._station = kwargs.get('station', None)

        self._graph = Graph()

    def connect_generators(self, generators):
        """Connects generators to grid

        Parameters
        ----------
        generators: :pandas:`pandas.DataFrame<dataframe>`
            Generators to be connected

        """
        raise NotImplementedError

    @property
    def id(self):
        """Returns id of grid"""
        return self._id

    @property
    def graph(self):
        """Provide access to the graph"""
        return self._graph

    @property
    def station(self):
        """Provide access to station"""
        return self._station

    @property
    def voltage_nom(self):
        """Provide access to nominal voltage"""
        return self._voltage_nom

    @property
    def id(self):
        return self._id

    @property
    def network(self):
        return self._network

    @property
    def grid_district(self):
        """Provide access to the grid_district"""
        return self._grid_district

    @property
    def peak_generation(self):
        """
        Cumulative peak generation capacity of generators of this grid

        Returns
        -------
        float
            Ad-hoc calculated or cached peak generation capacity
        """
        if self._peak_generation is None:
            self._peak_generation = sum(
                [_.nominal_capacity
                 for _ in self.graph.nodes_by_attribute('generator')])

        return self._peak_generation

    @property
    def peak_generation_per_technology(self):
        """
        Peak generation of each technology in the grid

        Returns
        -------
        :pandas:`pandas.Series<series>`
            Peak generation index by technology
        """
        peak_generation = defaultdict(float)
        for gen in self.graph.nodes_by_attribute('generator'):
            peak_generation[gen.type] += gen.nominal_capacity

        return pd.Series(peak_generation)

    @property
    def peak_load(self):
        """
        Cumulative peak load capacity of generators of this grid

        Returns
        -------
        float
            Ad-hoc calculated or cached peak load capacity
        """
        if self._peak_load is None:
            self._peak_load = sum(
                [_.peak_load.sum()
                 for _ in self.graph.nodes_by_attribute('load')])

        return self._peak_load

    @property
    def consumption(self):
        """
        Consumption in kWh per sector for whole grid

        Returns
        -------
        :pandas:`pandas.Series<series>`
            Indexed by demand sector
        """
        consumption = defaultdict(float)
        for load in self.graph.nodes_by_attribute('load'):
            for sector, val in load.consumption.items():
                consumption[sector] += val

        return pd.Series(consumption)


    def __repr__(self):
        return '_'.join([self.__class__.__name__, str(self._id)])


class MVGrid(Grid):
    """Defines a medium voltage grid in eDisGo

    Attributes
    ----------
    _mv_disconn_points : :obj:`list` of
        :class:`~.grid.components.MVDisconnectingPoint`

        Medium voltage disconnecting points = points where MV rings are split under
        normal operation conditions (= switch disconnectors in DINGO).
    _aggregates : :obj:`list` of :obj:`dict`
        This attribute is used for DINGO-imported data only. It contains data from
        DINGO's Aggregated Load Areas. Each list element represents one aggregated
        Load Area.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._mv_disconn_points = kwargs.get('mv_disconn_points', None)
        self._aggregates = kwargs.get('aggregates', None)
        self._lv_grids = kwargs.get('aggregates', None)

    @property
    def lv_grids(self):
        """LV grids associated to this MV grid : :obj:`list` of
        :class:`LVGrid`"""
        for lv_grid in self._lv_grids:
            yield lv_grid

    @lv_grids.setter
    def lv_grids(self, lv_grids):
        self._lv_grids = lv_grids

    def draw(self):
        """ Draw MV grid's graph using the geo data of nodes

        Notes
        -----
        This method uses the coordinates stored in the nodes' geoms which
        are usually conformal, not equidistant. Therefore, the plot might
        be distorted and does not (fully) reflect the real positions or
        distances between nodes.
        """

        # get nodes' positions
        nodes_pos = {}
        for node in self.graph.nodes():
            nodes_pos[node] = (node.geom.x, node.geom.y)

        plt.figure()
        nx.draw_networkx(self.graph, nodes_pos, node_size=16, font_size=8)
        plt.show()


class LVGrid(Grid):
        """Defines a low voltage grid in eDisGo

        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)


class Graph(nx.Graph):
    """Graph object

    This graph is an object subclassed from `networkX.Graph` extended by extra
    functionality and specific methods.
    """

    def nodes_from_line(self, line):
        """
        Get nodes adjacent to line

        Here, line refers to the object behind the key 'line' of the attribute
        dict attached to each edge.

        Parameters
        ----------
        line: edisgo.grid.components.Line
            A eDisGo line object

        Returns
        -------
        tuple
            Nodes adjacent to this edge
        """

        return dict([(v, k) for k, v in
              nx.get_edge_attributes(self, 'line').items()])[line]

    def line_from_nodes(self, u, v):
        """
        Get line between two nodes ``u`` and ``v``.

        Parameters
        ----------
        u : :class:`~.grid.components.Component`
            One adjacent node
        v : :class:`~.grid.components.Component`
            The other adjacent node

        Returns
        -------
        Line
            Line segment connecting ``u`` and ``v``.
        """
        try:
            line = nx.get_edge_attributes(self, 'line')[(u, v)]
        except:
            try:
                line = nx.get_edge_attributes(self, 'line')[(v, u)]
            except:
                raise nx.NetworkXError('Line between ``u`` and ``v`` not '
                                       'included in the graph.')

        return line

    def nodes_by_attribute(self, attr_val, attr='type'):
        """
        Select Graph's nodes by attribute value

        Get all nodes that share the same attribute. By default, the attr 'type'
        is used to specify the nodes type (generator, load, etc.).

        Examples
        --------
        >>> import edisgo
        >>> G = edisgo.grids.Graph()
        >>> G.add_node(1, type='generator')
        >>> G.add_node(2, type='load')
        >>> G.add_node(3, type='generator')
        >>> G.nodes_by_attribute('generator')
        [1, 3]

        Parameters
        ----------
        attr_val: str
            Value of the `attr` nodes should be selected by
        attr: str, default: 'type'
            Attribute key which is 'type' by default

        Returns
        -------
        list
            A list containing nodes elements that match the given attribute
            value
        """

        temp_nodes = getattr(self, 'node')
        nodes = list(filter(None, map(lambda x: x if temp_nodes[x][attr] == attr_val else None,
                                      temp_nodes.keys())))

        return nodes

    def lines_by_attribute(self, attr_val=None, attr='type'):
        """ Returns a generator for iterating over Graph's lines by attribute value.

        Get all lines that share the same attribute. By default, the attr 'type'
        is used to specify the lines' type (line, agg_line, etc.).

        The edge of a graph is described by the two adjacent nodes and the line
        object itself. Whereas the line object is used to hold all relevant
        power system parameters.

        Examples
        --------
        >>> import edisgo
        >>> G = edisgo.grids.Graph()
        >>> G.add_node(1, type='generator')
        >>> G.add_node(2, type='load')
        >>> G.add_edge(1, 2, type='line')
        >>> lines = G.lines_by_attribute('line')
        >>> list(lines)[0]
        <class 'tuple'>: ((node1, node2), line)

        Parameters
        ----------
        attr_val: str
            Value of the `attr` lines should be selected by
        attr: str, default: 'type'
            Attribute key which is 'type' by default

        Returns
        -------
        Generator of :obj:`dict`
            A list containing line elements that match the given attribute
            value

        Notes
        -----
        There are generator functions for nodes (`Graph.nodes()`) and edges
        (`Graph.edges()`) in NetworkX but unlike graph nodes, which can be
        represented by objects, branch objects can only be accessed by using an
        edge attribute ('line' is used here)

        To make access to attributes of the line objects simpler and more
        intuitive for the user, this generator yields a dictionary for each edge
        that contains information about adjacent nodes and the line object.

        Note, the construction of the dictionary highly depends on the structure
        of the in-going tuple (which is defined by the needs of networkX). If
        this changes, the code will break.

        Adapted from `Dingo <https://github.com/openego/dingo/blob/\
            ee237e37d4c228081e1e246d7e6d0d431c6dda9e/dingo/core/network/\
            __init__.py>`_.
        """

        # get all lines that have the attribute 'type' set
        lines_attributes = nx.get_edge_attributes(self, attr).items()

        # attribute value provided?
        if attr_val:
            # extract lines where 'type' == attr_val
            lines_attributes = [(k, self[k[0]][k[1]]['line'])
                                for k, v in lines_attributes if v == attr_val]
        else:
            # get all lines
            lines_attributes = [(k, self[k[0]][k[1]]['line'])
                                for k, v in lines_attributes]

        # sort them according to connected nodes
        lines_sorted = sorted(list(lines_attributes), key=lambda _: repr(_[1]))

        for line in lines_sorted:
            yield {'adj_nodes': line[0], 'line': line[1]}

    def lines(self):
        """ Returns a generator for iterating over Graph's lines

        Returns
        -------
        Generator of :obj:`dict`
            A list containing line elements

        Notes
        -----
        For a detailed description see lines_by_attribute()
        """
        return self.lines_by_attribute()
