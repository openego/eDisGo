import networkx as nx
import os
if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import Point
from .components import LVStation, BranchTee, Generator, Load, \
    MVDisconnectingPoint, Line

import logging
logger = logging.getLogger('edisgo')


def position_switch_disconnectors(mv_grid, mode='load', status='open'):
    """
    Determine position of switch disconnector in MV grid rings

    Determination of the switch disconnector location is motivated by placing
    it to minimized load flows in both parts of the ring (half-rings).
    Use the parameter mode

    Parameters
    ----------
    mv_grid : :class:`~.grid.grids.MVGrid`
        MV grid instance
    mode : str
        Define modus switch disconnector positioning: can be performed based of
        'load', 'generation' or both 'loadgen'. Defaults to 'load'
    status : str
        Either 'open' or 'closed'. Define which status is should be set
        initially. Defaults to 'open' (which refers to conditions of normal
        grid operation).

    Returns
    -------
    tuple
        A tuple of size 2 specifying their pair of nodes between which the
        switch disconnector is located. The first node specifies the node that
        actually includes the switch disconnector.

    Notes
    -----
    This function uses `nx.algorithms.find_cycle()` to identify nodes that are
    part of the MV grid ring(s). Make sure grid topology data that is provided
    has closed rings. Otherwise, no location for a switch disconnector can be
    identified.

    """

    def peak_load_gen_at_node(node):
        """Return peak load and peak generation capacity for ``node``

        Parameters
        ----------
        node : object
            Node instance in the grid topology graph

        Returns
        -------
        tuple
            Tuple of size two. First item is the peak load at ``node``; second
            parameters reflects peak generation capacity at ``node``.
            Returned peak_load and generation capacity is given as apparent
            power in kVA.

        """
        if isinstance(node, LVStation):
            node_peak_load = node.grid.peak_load
            node_peak_gen = node.grid.peak_generation
        elif isinstance(node, Generator):
            node_peak_load = 0
            node_peak_gen = node.nominal_capacity
        elif isinstance(node, Load):
            node_peak_load = node.peak_load.sum()
            node_peak_gen = 0
        elif isinstance(node, BranchTee):
            node_peak_load = 0
            node_peak_gen = 0

        return (node_peak_load / cos_phi_load, node_peak_gen / cos_phi_gen)

    def load_gen_from_subtree(graph, ring, node):
        """
        Accumulate load and generation capacity in branches to connecting node
        on the ring.

        Includes peak_load and generation capacity at ``node```itself.

        Parameters
        ----------
        graph : networkx.Graph
            The graph representing the MV grid topology
        ring : list
            A list of ring nodes
        node : networkx.Node
            A member of the ring

        Returns
        -------
        tuple
            Tuple of size two. First item is the peak load of subtree at
            ``node``; second parameter reflects peak generation capacity from
            subtree at ``node``.

        """
        ring_nodes_except_node = [_ for _ in ring if _ is not node]
        non_ring_nodes = [n for n in
                          [_ for _ in graph.nodes()
                           if _ is not mv_grid.station]
                          if n not in ring_nodes_except_node]
        subgraph = graph.subgraph(non_ring_nodes)

        nodes_subtree = nx.dfs_tree(subgraph, source=node)

        if len(nodes_subtree) > 1:
            peak_load_subtree = 0
            peak_gen_subtree = 0
            for n in nodes_subtree.nodes():
                peak_load_subtree_tmp, peak_gen_subtree_tmp = \
                    peak_load_gen_at_node(n)
                peak_load_subtree += peak_load_subtree_tmp
                peak_gen_subtree += peak_gen_subtree_tmp
            return (peak_load_subtree, peak_gen_subtree)
        else:
            return (0, 0)

    cos_phi_load = mv_grid.network.config['reactive_power_factor']['mv_load']
    cos_phi_gen = mv_grid.network.config['reactive_power_factor']['mv_gen']

    # Identify position of switch disconnector (SD)
    rings = nx.algorithms.cycle_basis(mv_grid.graph, root=mv_grid.station)

    for ring in rings:
        ring = [_ for _ in ring if _ is not mv_grid.station]

        node_peak_load = []
        node_peak_gen = []

        # Collect peak load and generation along the ring
        for node in ring:
            if len(mv_grid.graph.edges(nbunch=node)) > 2:
                peak_load, peak_gen = load_gen_from_subtree(
                    mv_grid.graph, ring, node)
            else:
                peak_load, peak_gen = peak_load_gen_at_node(node)

            node_peak_load.append(peak_load)
            node_peak_gen.append(peak_gen)

            # Choose if SD is placed 'load' or 'generation' oriented
            if mode == 'load':
                node_peak_data = node_peak_load
            elif mode == 'generation':
                node_peak_data = node_peak_gen
            elif mode == 'loadgen':
                node_peak_data = node_peak_load if sum(node_peak_load) > sum(
                    node_peak_gen) else node_peak_gen
            else:
                raise ValueError("Mode {mode} is not known!".format(mode=mode))

        # Set start value for difference in ring halfs
        diff_min = 10e9

        # Identify nodes where switch disconnector is located in between
        for ctr in range(len(node_peak_data)):
            # check if node that owns the switch disconnector is of type
            # LVStation
            if isinstance(ring[ctr - 2], LVStation):
                # Iteratively split route and calc peak load difference
                route_data_part1 = sum(node_peak_data[0:ctr])
                route_data_part2 = sum(node_peak_data[ctr:len(node_peak_data)])
                diff = abs(route_data_part1 - route_data_part2)

                # stop walking through the ring when load/generation is almost
                # equal
                if diff <= diff_min:
                    diff_min = diff
                    position = ctr
                else:
                    break

        # find position of switch disconnector
        node1 = ring[position - 1]
        node2 = ring[position]

        implement_switch_disconnector(mv_grid, node1, node2)

    # open all switch disconnectors
    if status == 'open':
        for sd in mv_grid.graph.nodes_by_attribute('mv_disconnecting_point'):
            sd.open()
    elif status == 'close':
        for sd in mv_grid.graph.nodes_by_attribute('mv_disconnecting_point'):
            sd.close()


def implement_switch_disconnector(mv_grid, node1, node2):
    """
    Install switch disconnector in grid topology

    The graph that represents the grid's topology is altered in such way that
    it explicitly includes a switch disconnector.
    The switch disconnector is always located at ``node1``. Technically, it
    does not make any difference. This is just an convention ensuring
    consistency of multiple runs.

    The ring is still closed after manipulations of this function.

    Parameters
    ----------
    mv_grid : :class:`~.grid.grids.MVGrid`
        MV grid instance
    node1
        A rings node
    node2
        Another rings node

    """
    # Get disconnecting point's location
    line = mv_grid.graph.edge[node1][node2]['line']

    length_sd_line = .75e-3 # in km

    x_sd = node1.geom.x * (length_sd_line / line.length) * (
        node1.geom.x - node2.geom.x)
    y_sd = node1.geom.y * (length_sd_line / line.length) * (
        node1.geom.y - node2.geom.y)

    # Instantiate disconnecting point
    mv_dp_number = len(mv_grid.graph.nodes_by_attribute(
        'mv_disconnecting_point'))
    disconnecting_point = MVDisconnectingPoint(
        id=mv_dp_number + 1,
        geom=Point(x_sd, y_sd),
        grid=mv_grid)
    mv_grid.graph.add_node(disconnecting_point, type='mv_disconnecting_point')

    # Replace original line by a new line
    new_line_attr = {
        'line': Line(
            id=line.id,
            type=line.type,
            length=line.length - length_sd_line,
            grid=mv_grid),
        'type': 'line'}
    mv_grid.graph.remove_edge(node1, node2)
    mv_grid.graph.add_edge(disconnecting_point, node2, new_line_attr)

    # Add disconnecting line segment
    switch_disconnector_line_attr = {
        'line': Line(
                  id="switch_disconnector_line_{}".format(
                      str(mv_dp_number + 1)),
                  type=line.type,
                  length=length_sd_line,
                  grid=mv_grid),
        'type': 'line'}

    mv_grid.graph.add_edge(node1, disconnecting_point,
                           switch_disconnector_line_attr)

    # Set line to switch disconnector
    disconnecting_point.line = mv_grid.graph.line_from_nodes(
        disconnecting_point, node1)


def select_cable(network, level, apparent_power):
    """Selects an appropriate cable type and quantity using given apparent
    power.

    Considers load factor.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    level : :obj:`str`
        Grid level ('mv' or 'lv')
    apparent_power : :obj:`float`
        Apparent power the cable must carry in kVA

    Returns
    -------
    :pandas:`pandas.Series<series>`
        Cable type
    :obj:`ìnt`
        Cable count

    Notes
    ------
    Cable is selected to be able to carry the given `apparent_power`, no load
    factor is considered.

    """

    cable_count = 1

    if level == 'mv':

        available_cables = network.equipment_data['mv_cables'][
            network.equipment_data['mv_cables']['U_n'] ==
            network.mv_grid.voltage_nom]

        suitable_cables = available_cables[
            available_cables['I_max_th'] *
            network.mv_grid.voltage_nom > apparent_power]

        # increase cable count until appropriate cable type is found
        while suitable_cables.empty:
            cable_count += 1
            suitable_cables = available_cables[
                available_cables['I_max_th'] *
                network.mv_grid.voltage_nom *
                cable_count > apparent_power]

        cable_type = suitable_cables.ix[suitable_cables['I_max_th'].idxmin()]

    elif level == 'lv':

        suitable_cables = network.equipment_data['lv_cables'][
            network.equipment_data['lv_cables']['I_max_th'] *
            network.equipment_data['lv_cables']['U_n'] > apparent_power]

        # increase cable count until appropriate cable type is found
        while suitable_cables.empty:
            cable_count += 1
            suitable_cables = network.equipment_data['lv_cables'][
                network.equipment_data['lv_cables']['I_max_th'] *
                network.equipment_data['lv_cables']['U_n'] *
                cable_count > apparent_power]

        cable_type = suitable_cables.ix[suitable_cables['I_max_th'].idxmin()]

    else:
        raise ValueError('Please supply a level (either \'mv\' or \'lv\').')

    return cable_type, cable_count
