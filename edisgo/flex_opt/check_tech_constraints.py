import numpy as np

import logging


logger = logging.getLogger('edisgo')


def check_line_load(network, results_lines, **kwargs):
    """ Checks for over-loading of branches and transformers for MV or LV grid

    Parameters
    ----------
    network: edisgo Network object
    results_lines: pandas.DataFrame
        power flow analysis results (pfa_edges) from edisgo Results object
    **kwargs:
        load_factor_mv_line: float (optional)
            allowed load of MV line in uninterrupted operation
        load_factor_lv_line: float (optional)
            allowed load of LV line in uninterrupted operation

    Returns
    -------
    Dict of critical lines (edisgo Line objects) with max. relative overloading
    Format: {line_1: rel_overloading_1, ..., line_n: rel_overloading_n}

    Notes
    -----
    According to [1]_ load factors in feed-in case of all equipment in MV and
    LV is set to 1.

    References
    ----------
    .. [1] dena VNS

    """

    crit_lines = {}

    # ToDo: Unterscheidung cables/lines?
    # ToDo: Unterscheidung lc/fc? Nur durch load factors, results dürfen dann nur den einen Fall beinhalten (wie ist das bei PyPsa?)
    # ToDo: Einheiten klären (results und Werte in type)
    # ToDo: generischer, sodass nicht unbedingt LV und MV gerechnet werden? + HV/MV transformer immer raus lassen?
    # ToDo: Zugriff auf Attribute mit _
    load_factor_mv_line = kwargs.get('load_factor_mv_line', 1)
    load_factor_lv_line = kwargs.get('load_factor_mv_line', 1)

    mw2kw = 1e3

    # ToDo: p0 und p1 Ergebnisse am Anfang und Ende einer Line?
    # ToDo: Wo soll S berechnet werden? In results objekt schon s_max berechnen?
    # ToDo: mit Skalar testen
    results_lines['s0'] = (results_lines['p0'].apply(lambda x: x ** 2) +
                           results_lines['q0'].apply(lambda x: x ** 2)).apply(
                               lambda x: np.sqrt(x))
    # results_lines['s0'] = results_lines['p0'].apply(
    #     lambda x: [i ** 2 for i in x])

    # MV
    for line in list(network.mv_grid.graph.graph_edges()):
        s_max_th = (3 ** 0.5 * line['line']._type['U_n'] *
                    line['line']._type['I_max_th']) * \
                   load_factor_mv_line * line['line']._quantity
        try:
            # check if maximum s_0 from power flow analysis exceeds allowed
            # values
            if max(results_lines.loc[line, 's0']) > s_max_th:
                crit_lines[line] = (max(results_lines.loc[line, 's0']) *
                                    mw2kw / s_max_th)
        except:
            logger.debug('No results for line {} '.format(str(line)) +
                         'to check overloading.')
    # LV
    for lv_grid in network.mv_grid.lv_grids:
        for line in list(lv_grid.graph.graph_edges()):
            line['line']._quantity = 1
            s_max_th = (3 ** 0.5 * line['line']._type['U_n'] *
                        line['line']._type['I_max_th']) * \
                       load_factor_lv_line * line['line']._quantity
            try:
                # check if maximum s_0 from power flow analysis exceeds allowed
                # values
                if max(results_lines.loc[line, 's0']) > s_max_th:
                    crit_lines[line] = (max(results_lines.loc[line, 's0']) *
                                        mw2kw / s_max_th)
            except:
                logger.debug('No results for line {} '.format(str(line)) +
                             'to check overloading.')

    if crit_lines:
        logger.info('==> {} lines have load issues.'.format(
            len(crit_lines)))

    return crit_lines


def check_station_load(network, results_lines, **kwargs):
    """ Checks for over-loading of branches and transformers for MV or LV grid

    Parameters
    ----------
    network: edisgo Network object
    results_lines: pandas.DataFrame
        power flow analysis results (pfa_edges) from edisgo Results object
    **kwargs:
        load_factor_mv_lv_transformer: float (optional)
            allowed load of MV/LV transformer in uninterrupted operation

    Returns
    -------
    Dict of critical stations (edisgo Station objects)
    Format: {station_1: overloading_1, ..., station_n: overloading_n}

    Notes
    -----
    According to [1]_ load factors in feed-in case of all equipment in MV and
    LV is set to 1.

    HV/MV transformers are not checked.

    References
    ----------
    .. [1] dena VNS

    """

    # ToDo: line load an der Seite wo Station angeschlossen ist?
    crit_stations = {}

    load_factor_mv_lv_transformer = kwargs.get('load_factor_mv_lv_transformer',
                                               1)

    mw2kw = 1e3

    # ToDo: Wo soll S berechnet werden?
    # ToDo: mit Skalar testen
    results_lines['s0'] = (results_lines['p0'].apply(lambda x: x ** 2) +
                           results_lines['q0'].apply(lambda x: x ** 2)).apply(
                               lambda x: np.sqrt(x))

    # MV/LV station
    for lv_grid in network.mv_grid.lv_grids:
        station = lv_grid.station
        s_max_th = sum([_._type.s for _ in station.transformers])
        # find line station is connected to
        # ToDo: Was ist load der station? Summe der loads der lines die zur Station führen?
        # edges_station = nx.edges(lv_grid.graph, station)
        # max(results_lines.loc[line, 's0'])
        # try:
        #     # check if maximum s_0 from power flow analysis exceeds allowed
        #     # values
        #     if max(results_lines.loc[line, 's0']) > s_max_th:
        #         crit_lines[line] = (max(results_lines.loc[line, 's0']) *
        #                             mw2kw / s_max_th)
        # except:
        #     logger.debug('No results for line {} '.format(str(line)) +
        #                 'to check overloading.')

    if crit_stations:
        logger.info('==> {} stations have load issues.'.format(
            len(crit_stations)))

    return crit_stations


def check_voltage(network, results_nodes, **kwargs):
    """ Checks for voltage stability issues at all nodes for MV and LV grid

    Parameters
    ----------
    network: edisgo Network object
    results_nodes: pandas.DataFrame
        power flow analysis results (pfa_nodes) from edisgo Results object
    **kwargs:
        mv_max_v_deviation: float (optional)
            allowed voltage deviation in MV grid
        lv_max_v_deviation: float (optional)
            allowed voltage deviation in LV grid

    Returns
    -------
    List of critical nodes, sorted descending by voltage difference
    Format: {grid_1: [node_1A, node_1B, ...], ..., grid_n: [node_nA, ...]}

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.
    """

    # ToDo: delta_U wird auch benötigt, deshalb crit_nodes als dict mit series
    # ToDo: crit_nodes innerhalb einer Series sortieren
    crit_nodes = {}

    # load max. voltage deviation for load and feedin case
    mv_max_v_deviation = kwargs.get('mv_max_v_deviation', 0.1)
    lv_max_v_deviation = kwargs.get('lv_max_v_deviation', 0.1)

    # check nodes' voltages
    # MV
    voltage_station = network.mv_grid.station.transformers[0]._voltage_op
    for node in network.mv_grid.graph.nodes():
        try:
            # check if maximum deviation in v_mag_pu exceeds allowed deviation
            if (max(results_nodes.loc[node, 'v_mag_pu']) >
                    1 + mv_max_v_deviation or
                min(results_nodes.loc[
                    node, 'v_mag_pu']) < 1 - mv_max_v_deviation):
                try:
                    crit_nodes[network.mv_grid].append(node)
                except:
                    crit_nodes[network.mv_grid] = [node]
        except:
            logger.debug('No results for node {} '.format(str(node)) +
                         'to check overvoltage.')

    # LV
    for lv_grid in network.mv_grid.lv_grids:
        for node in lv_grid.graph.nodes():
            try:
                # check if maximum deviation in v_mag_pu exceeds allowed
                # deviation
                if (max(results_nodes.loc[node, 'v_mag_pu']) >
                        1 + lv_max_v_deviation or
                    min(results_nodes.loc[
                        node, 'v_mag_pu']) < 1 - lv_max_v_deviation):
                    try:
                        crit_nodes[lv_grid].append(node)
                    except:
                        crit_nodes[lv_grid] = [node]
            except:
                logger.debug('No results for node {} '.format(str(node)) +
                             'in LV grid {} '.format(str(lv_grid)) +
                             'to check overvoltage.')

    if crit_nodes:
        logger.info(
            '==> {} nodes have voltage issues.'.format(len(crit_nodes)))

    return crit_nodes


def get_critical_voltage_at_nodes(grid):
    """
    Estimate voltage drop/increase induced by loads/generators connected to the
    grid.

    Based on voltage level at each node of the grid critical nodes in terms
    of exceed tolerable voltage drop/increase are determined.
    The tolerable voltage drop/increase is defined by [VDE-AR]_ a adds up to
    3 % of nominal voltage.
    The longitudinal voltage drop at each line segment is estimated by a
    simplified approach (neglecting the transverse voltage drop) described in
    [VDE-AR]_.

    Two equations are available for assessing voltage drop/ voltage increase.

    The first is used to assess a voltage drop in the load case

    .. math::
        \Delta u = \frac{S_{Amax} \cdot ( R_{kV} \cdot cos(\phi) + X_{kV} \cdot sin(\phi) )}{U_{nom}}

    The second equation can be used to assess the voltage increase in case of
    feedin. The only difference is the negative sign before X. This is related
    to consider a voltage drop due to inductive operation of generators.

    .. math::
        \Delta u = \frac{S_{Amax} \cdot ( R_{kV} \cdot cos(\phi) - X_{kV} \cdot sin(\phi) )}{U_{nom}}

    .. TODO: correct docstring such that documentation builds properly

    ================  =============================
    Symbol            Description
    ================  =============================
    :math:`\Delta u`  Voltage drop/increase at node
    :math:`S_{Amax}`  Apparent power
    :math:`R_{kV}`    Short-circuit resistance
    :math:`X_{kV}`    Short-circuit reactance
    :math:`cos(\phi)` Power factor
    :math:`U_{nom}`   Nominal voltage
    ================  =============================

    Parameters
    ----------
    grid : ding0.core.network.grids.LVGridDing0
        Ding0 LV grid object

    Notes
    -----
    The implementation highly depends on topology of LV grid. This must not
    change its topology from radial grid with stubs branching from radial
    branches. In general, the approach of [VDE-AR]_ is only applicable to grids of
    radial topology.

    We consider the transverse voltage drop/increase by applying the same
    methodology successively on results of main branch. The voltage
    drop/increase at each house connection branch (aka. stub branch or grid
    connection point) is estimated by superposition based on voltage level
    in the main branch cable distributor.
    """

    # v_delta_tolerable_fc = cfg_ding0.get('assumptions',
    #                                   'lv_max_v_level_fc_diff_normal')
    # v_delta_tolerable_lc = cfg_ding0.get('assumptions',
    #                                   'lv_max_v_level_lc_diff_normal')
    #
    # omega = 2 * math.pi * 50
    #
    # crit_nodes = []
    #
    # # get list of nodes of main branch in right order
    # tree = nx.dfs_tree(grid._graph, grid._station)
    #
    # # list for nodes of main branch
    # main_branch = []
    #
    # # list of stub cable distributors branching from main branch
    # grid_conn_points = []
    #
    # # fill two above lists
    # for node in list(nx.descendants(tree, grid._station)):
    #     successors = tree.successors(node)
    #     if successors and all(isinstance(successor, LVCableDistributorDing0)
    #            for successor in successors):
    #         main_branch.append(node)
    #     elif (isinstance(node, LVCableDistributorDing0) and
    #         all(isinstance(successor, (GeneratorDing0, LVLoadDing0))
    #            for successor in successors)):
    #         grid_conn_points.append(node)
    #
    # # voltage at substation bus bar
    # r_mv_grid, x_mv_grid = get_mv_impedance(grid)
    #
    # r_trafo = sum([tr.r for tr in grid._station._transformers])
    # x_trafo = sum([tr.x for tr in grid._station._transformers])
    #
    # v_delta_load_case_bus_bar, \
    # v_delta_gen_case_bus_bar = get_voltage_at_bus_bar(grid, tree)
    #
    # if (abs(v_delta_gen_case_bus_bar) > v_delta_tolerable_fc
    #     or abs(v_delta_load_case_bus_bar) > v_delta_tolerable_lc):
    #     crit_nodes.append({'node': grid._station,
    #                        'v_diff': [v_delta_load_case_bus_bar,
    #                                   v_delta_gen_case_bus_bar]})
    #
    #
    #
    # # voltage at main route nodes
    # for first_node in [b for b in tree.successors(grid._station)
    #                if b in main_branch]:
    #
    #     # cumulative resistance/reactance at bus bar
    #     r = r_mv_grid + r_trafo
    #     x = x_mv_grid + x_trafo
    #
    #     # roughly estimate transverse voltage drop
    #     stub_node = [_ for _ in tree.successors(first_node) if
    #                  _ not in main_branch][0]
    #     v_delta_load_stub, v_delta_gen_stub = voltage_delta_stub(
    #         grid,
    #         tree,
    #         first_node,
    #         stub_node,
    #         r,
    #         x)
    #
    #     # cumulative voltage drop/increase at substation bus bar
    #     v_delta_load_cum = v_delta_load_case_bus_bar
    #     v_delta_gen_cum = v_delta_gen_case_bus_bar
    #
    #     # calculate voltage at first node of branch
    #     voltage_delta_load, voltage_delta_gen, r, x = \
    #         get_voltage_delta_branch(grid, tree, first_node, r, x)
    #
    #     v_delta_load_cum += voltage_delta_load
    #     v_delta_gen_cum += voltage_delta_gen
    #
    #     if (abs(v_delta_gen_cum) > (v_delta_tolerable_fc - v_delta_gen_stub)
    #         or abs(v_delta_load_cum) > (v_delta_tolerable_lc - v_delta_load_stub)):
    #         crit_nodes.append({'node': first_node,
    #                            'v_diff': [v_delta_load_cum,
    #                                       v_delta_gen_cum]})
    #         crit_nodes.append({'node': stub_node,
    #                            'v_diff': [
    #                                v_delta_load_cum + v_delta_load_stub,
    #                                v_delta_gen_cum + v_delta_gen_stub]})
    #
    #     # get next neighboring nodes down the tree
    #     successor = [x for x in tree.successors(first_node)
    #                   if x in main_branch]
    #     if successor:
    #         successor = successor[0] # simply unpack
    #
    #     # successively determine voltage levels for succeeding nodes
    #     while successor:
    #         voltage_delta_load, voltage_delta_gen, r, x = \
    #             get_voltage_delta_branch(grid, tree, successor, r, x)
    #
    #         v_delta_load_cum += voltage_delta_load
    #         v_delta_gen_cum += voltage_delta_gen
    #
    #         # roughly estimate transverse voltage drop
    #         stub_node = [_ for _ in tree.successors(successor) if
    #                      _ not in main_branch][0]
    #         v_delta_load_stub, v_delta_gen_stub = voltage_delta_stub(
    #             grid,
    #             tree,
    #             successor,
    #             stub_node,
    #             r,
    #             x)
    #
    #         if (abs(v_delta_gen_cum) > (v_delta_tolerable_fc - v_delta_gen_stub)
    #             or abs(v_delta_load_cum) > (
    #                         v_delta_tolerable_lc - v_delta_load_stub)):
    #             crit_nodes.append({'node': successor,
    #                                'v_diff': [v_delta_load_cum,
    #                                           v_delta_gen_cum]})
    #             crit_nodes.append({'node': stub_node,
    #                                'v_diff': [
    #                                    v_delta_load_cum + v_delta_load_stub,
    #                                    v_delta_gen_cum + v_delta_gen_stub]})
    #
    #         successor = [_ for _ in tree.successors(successor)
    #                      if _ in main_branch]
    #         if successor:
    #             successor = successor[0]
    #
    # return crit_nodes


def voltage_delta_vde(v_nom, s_max, r, x, cos_phi):
    """
    Estimate voltrage drop/increase

    The VDE [VDE-AR]_ proposes a simplified method to estimate voltage drop or
     increase in radial grids.

    Parameters
    ----------
    v_nom : int
        Nominal voltage
    s_max : numeric
        Apparent power
    r : numeric
        Short-circuit resistance from node to HV/MV substation (in ohm)
    x : numeric
        Short-circuit reactance from node to HV/MV substation (in ohm). Must
        be a signed number indicating (+) inductive reactive consumer (load
        case) or (-) inductive reactive supplier (generation case)
    cos_phi : numeric

    References
    ----------
    .. [VDE-AR] VDE Anwenderrichtlinie: Erzeugungsanlagen am Niederspannungsnetz –
        Technische Mindestanforderungen für Anschluss und Parallelbetrieb von
        Erzeugungsanlagen am Niederspannungsnetz, 2011

    Returns
    -------
    voltage_delta : numeric
        Voltage drop or increase
    """
    # delta_v = (s_max * (
    #     r * cos_phi + x * math.sin(math.acos(cos_phi)))) / v_nom ** 2
    # return delta_v


def get_house_conn_gen_load(graph, node):
    """
    Get generation capacity/ peak load of neighboring house connected to main
    branch

    Parameter
    ---------
    graph : networkx.DiGraph
        Directed graph
    node : graph node
        Node of the main branch of LV grid

    Return
    ------
    generation_peak_load : list
        A list containing two items
            # peak load of connected house branch
            # generation capacity of connected generators
    """
    # generation = 0
    # peak_load = 0
    #
    # for cus_1 in graph.successors(node):
    #     for cus_2 in graph.successors(cus_1):
    #         if not isinstance(cus_2, list):
    #             cus_2 = [cus_2]
    #         generation += sum([gen.capacity for gen in cus_2
    #                       if isinstance(gen, GeneratorDing0)])
    #         peak_load += sum([load.peak_load for load in cus_2
    #                       if isinstance(load, LVLoadDing0)])
    #
    # return [peak_load, generation]


def get_voltage_delta_branch(grid, tree, node, r_preceeding, x_preceeding):
    """
    Determine voltage for a preceeding branch (edge) of node

    Parameters
    ----------
    grid : ding0.core.network.grids.LVGridDing0
        Ding0 grid object
    tree : networkx.DiGraph
        Tree of grid topology
    node : graph node
        Node to determine voltage level at
    r_preceeding : float
        Resitance of preceeding grid
    x_preceeding : float
        Reactance of preceeding grid

    Return
    ------
    delta_voltage : float
        Delta voltage for node
    """
    # cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    # cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    # v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage')
    # omega = 2 * math.pi * 50
    #
    # # add resitance/ reactance to preceeding
    # in_edge = [_ for _ in grid.graph_branches_from_node(node) if
    #            _[0] in tree.predecessors(node)][0][1]
    # r = r_preceeding + (in_edge['branch'].type['R'] *
    #                  in_edge['branch'].length)
    # x = x_preceeding + (in_edge['branch'].type['L'] / 1e3 * omega *
    #                  in_edge['branch'].length)
    #
    # # get apparent power for load and generation case
    # peak_load, gen_capacity = get_house_conn_gen_load(tree, node)
    # s_max_load = peak_load / cos_phi_load
    # s_max_feedin = gen_capacity / cos_phi_feedin
    #
    # # determine voltage increase/ drop a node
    # voltage_delta_load = voltage_delta_vde(v_nom, s_max_load, r, x,
    #                                        cos_phi_load)
    # voltage_delta_gen = voltage_delta_vde(v_nom, s_max_feedin, r, -x,
    #                                       cos_phi_feedin)
    #
    # return [voltage_delta_load, voltage_delta_gen, r, x]


def get_mv_impedance(grid):
    """
    Determine MV grid impedance (resistance and reactance separately)

    Parameters
    ----------
    grid : ding0.core.network.grids.LVGridDing0

    Returns
    -------
    List containing resistance and reactance of MV grid
    """

    # omega = 2 * math.pi * 50
    #
    # mv_grid = grid.grid_district.lv_load_area.mv_grid_district.mv_grid
    # edges = mv_grid.find_path(grid._station, mv_grid._station, type='edges')
    # r_mv_grid = sum([e[2]['branch'].type['R'] * e[2]['branch'].length / 1e3
    #                  for e in edges])
    # x_mv_grid = sum([e[2]['branch'].type['L'] / 1e3 * omega * e[2][
    #     'branch'].length / 1e3 for e in edges])
    #
    # return [r_mv_grid, x_mv_grid]


def voltage_delta_stub(grid, tree, main_branch_node, stub_node, r_preceeding,
                       x_preceedig):
    """
    Determine voltage for stub branches

    Parameters
    ----------
    grid : ding0.core.network.grids.LVGridDing0
        Ding0 grid object
    tree : networkx.DiGraph
        Tree of grid topology
    main_branch_node : graph node
        Node of main branch that stub branch node in connected to
    main_branch : dict
        Nodes of main branch
    r_preceeding : float
        Resitance of preceeding grid
    x_preceeding : float
        Reactance of preceeding grid

    Return
    ------
    delta_voltage : float
        Delta voltage for node
    """
    # cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    # cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    # v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage')
    # omega = 2 * math.pi * 50
    #
    # stub_branch = [_ for _ in grid.graph_branches_from_node(main_branch_node) if
    #                _[0] == stub_node][0][1]
    # r_stub = stub_branch['branch'].type['R'] * stub_branch[
    #     'branch'].length / 1e3
    # x_stub = stub_branch['branch'].type['L'] / 1e3 * omega * \
    #          stub_branch['branch'].length / 1e3
    # s_max_gen = [_.capacity / cos_phi_feedin
    #              for _ in tree.successors(stub_node)
    #              if isinstance(_, GeneratorDing0)]
    # if s_max_gen:
    #     s_max_gen = s_max_gen[0]
    #     v_delta_stub_gen = voltage_delta_vde(v_nom, s_max_gen, r_stub + r_preceeding,
    #                                          x_stub + x_preceedig, cos_phi_feedin)
    # else:
    #     v_delta_stub_gen = 0
    #
    # s_max_load = [_.peak_load / cos_phi_load
    #               for _ in tree.successors(stub_node)
    #               if isinstance(_, LVLoadDing0)]
    # if s_max_load:
    #     s_max_load = s_max_load[0]
    #     v_delta_stub_load = voltage_delta_vde(v_nom, s_max_load, r_stub + r_preceeding,
    #                                           x_stub + x_preceedig, cos_phi_load)
    # else:
    #     v_delta_stub_load = 0
    #
    # return [v_delta_stub_load, v_delta_stub_gen]


def get_voltage_at_bus_bar(grid, tree):
    """
    Determine voltage level at bus bar of MV-LV substation

    grid : ding0.core.network.grids.LVGridDing0
        Ding0 grid object
    tree : networkx.DiGraph
        Tree of grid topology:

    Returns
    -------
    voltage_levels : list
        Voltage at bus bar. First item refers to load case, second item refers
        to voltage in feedin (generation) case
    """

    # # voltage at substation bus bar
    # r_mv_grid, x_mv_grid = get_mv_impedance(grid)
    #
    # r_trafo = sum([tr.r for tr in grid._station._transformers])
    # x_trafo = sum([tr.x for tr in grid._station._transformers])
    #
    # cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    # cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    # v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage')
    #
    # # loads and generators connected to bus bar
    # bus_bar_load = sum(
    #     [node.peak_load for node in tree.successors(grid._station)
    #      if isinstance(node, LVLoadDing0)]) / cos_phi_load
    # bus_bar_generation = sum(
    #     [node.capacity for node in tree.successors(grid._station)
    #      if isinstance(node, GeneratorDing0)]) / cos_phi_feedin
    #
    # v_delta_load_case_bus_bar = voltage_delta_vde(v_nom,
    #                                               bus_bar_load,
    #                                               (r_mv_grid + r_trafo),
    #                                               (x_mv_grid + x_trafo),
    #                                               cos_phi_load)
    # v_delta_gen_case_bus_bar = voltage_delta_vde(v_nom,
    #                                              bus_bar_generation,
    #                                              (r_mv_grid + r_trafo),
    #                                              -(x_mv_grid + x_trafo),
    #                                              cos_phi_feedin)
    #
    # return v_delta_load_case_bus_bar, v_delta_gen_case_bus_bar
