import pandas as pd
import networkx as nx
from math import sqrt
from edisgo.grid.grids import LVGrid
from edisgo.grid.components import LVStation, Transformer
from edisgo.flex_opt import storage_integration


def one_storage_per_feeder(edisgo, storage_parameters):
    # ToDo: add parameters from etrago specs
    # ToDo: document

    def feeder_ranking(grid_expansion_costs):
        """
        Get feeder ranking from grid expansion costs DataFrame.

        Parameters
        ----------
        grid_expansion_costs : :pandas:`pandas.DataFrame<dataframe>`
            grid_expansion_costs DataFrame from :class:`~.grid.network.Results`

        Returns
        -------
        :pandas:`pandas.Series<series>`
            Series with ranked MV feeders. Feeders are ranked by total grid
            expansion costs of all measures conducted in the feeder. The
            feeder with the highest costs is in the first row and the feeder
            with the lowest costs in the last row.

        """
        return grid_expansion_costs.groupby(['mv_feeder'], sort=False).sum().\
            reset_index().sort_values(by=['total_costs'], ascending=False)[
            'mv_feeder']

    def find_battery_node(components):
        """
        Evaluates where to install the storage

        Returns
        --------
        Node where storage is installed.

        """
        # nodes_mv will contain all nodes of MV lines with over-loading or
        # over-voltage issues; lv_stations will contain all stations of LV
        # grids with over-loading or over-voltage issues
        nodes_mv = []
        lv_stations = []
        for comp in components:
            if isinstance(comp, Transformer) or isinstance(comp.grid, LVGrid):
                lv_stations.append(comp.grid.station)
            else:
                nodes_mv.extend(list(comp.grid.graph.nodes_from_line(comp)))

        # if there are nodes with issues in the MV grid the battery storage
        # will be installed at the node farthest away from the MV station
        if nodes_mv:
            # dictionary with nodes and their corresponding path length to
            # MV station
            path_length_dict = {}
            for node in nodes_mv:
                if isinstance(node, LVStation):
                    path_length_dict[node] = len(
                        nx.shortest_path(node.mv_grid.graph,
                                         node.mv_grid.station, node))
                else:
                    path_length_dict[node] = len(
                        nx.shortest_path(node.grid.graph, node.grid.station,
                                         node))
            # find node farthest away
            battery_node = [_ for _ in path_length_dict.keys()
                            if path_length_dict[_] ==
                            max(path_length_dict.values())][0]
        # if there are only issues in the LV grids find LV station with issues
        # closest to MV station
        else:
            # calculate path lengths
            path_length_dict = {}
            for node in lv_stations:
                path_length_dict[node] = len(nx.shortest_path(
                    node.mv_grid.graph, node.mv_grid.station, node))
            # find closest node
            battery_node = [_ for _ in path_length_dict.keys()
                            if path_length_dict[_] ==
                            min(path_length_dict.values())][0]
        # ToDo: Return node in original edisgo object instead of copied one
        return battery_node

    # rank MV feeders by grid expansion costs
    # conduct grid reinforcement on copied edisgo object on worst-case time
    # steps
    # ToDo: reinforce for worst-case once implemented
    grid_expansion_results = edisgo.reinforce(copy_graph=True)
    # ToDo: delete once implementation is finished
    # import pickle
    # edisgo_reinforce = pickle.load(open('edisgo.pkl', 'rb'))
    # grid_expansion_results = edisgo_reinforce.network.results
    ranked_feeders = feeder_ranking(
        grid_expansion_results.grid_expansion_costs)

    for feeder in ranked_feeders.values:

        # first step: find node where storage will be installed
        # get all reinforced components in respective feeder
        components = grid_expansion_results.grid_expansion_costs.loc[
            grid_expansion_results.grid_expansion_costs.mv_feeder == feeder]. \
            index
        battery_node = find_battery_node(components)

        # second step: estimate residual line load to calculate size of battery
        # ToDo: find battery line in original edisgo object instead of copy
        path_to_battery_node = nx.shortest_path(
            feeder.grid.graph, feeder.grid.station, battery_node)
        battery_line = feeder.grid.graph.line_from_nodes(
            path_to_battery_node[-1], path_to_battery_node[-2])
        # analyze (probably not needed here since first analyze in conducted
        # in reinforce_grid) for every time step
        #edisgo.analyze()

        # allowed line load
        # ToDo: Differentiate between load and feedin case
        load_factor_line = edisgo.network.config[
            'grid_expansion_load_factors']['mv_feedin_case_line']
        i_allowed = battery_line.type['I_max_th'] * load_factor_line * \
                    battery_line.quantity
        u_allowed = battery_line.type['U_n']
        # ToDo: correct?
        s_allowed = sqrt(3) * i_allowed * u_allowed

        # maximum power of the line
        pfa_q = edisgo.network.results.pfa_q[repr(battery_line)]
        # ToDo: storage efficiency
        p_max = sqrt(s_allowed ** 2 - pfa_q ** 2)

        # third step: integrate storage
        storage_parameters['nominal_capacity'] = p_max
        storage_integration.set_up_storage(storage_parameters, battery_node)

        # fourth step: if there is storage capacity left, rerun the past steps
        # for the next feeder in the ranking list
