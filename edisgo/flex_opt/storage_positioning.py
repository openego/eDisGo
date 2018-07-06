import pandas as pd
import networkx as nx
from math import sqrt
from edisgo.grid.grids import LVGrid
from edisgo.grid.components import LVStation
from edisgo.flex_opt import storage_integration


def one_storage_per_feeder(edisgo, storage_parameters):

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

    # rank MV feeders by grid expansion costs
    if edisgo.network.results.grid_expansion_costs is not None:
        grid_expansion_results = edisgo.network.results
    else:
        # conduct grid reinforcement
        #grid_expansion_results = edisgo.reinforce(copy_graph=True)
        import pickle
        edisgo_reinforce = pickle.load(open('edisgo.pkl', 'rb'))
        grid_expansion_results = edisgo_reinforce.network.results

    # rank feeders
    ranked_feeders = feeder_ranking(
        grid_expansion_results.grid_expansion_costs)

    for feeder in ranked_feeders.values:
        ### install storage at the node of the overloaded lines farthest away
        ### from the HV/MV transformer

        # get the nodes farthest away from HV/MV transformer
        lines = grid_expansion_results.grid_expansion_costs.loc[
            grid_expansion_results.grid_expansion_costs.mv_feeder==feeder].\
            index
        nodes_mv = []
        lv_stations = []
        for line in lines:
            if isinstance(line.grid, LVGrid):
                lv_stations.append(line.grid.station)
            else:
                nodes_mv.extend(list(line.grid.graph.nodes_from_line(line)))
        # if there are nodes with issues in the MV grid find node farthest away from station
        if nodes_mv:
            # dictionary with path length as key and corresponding node as value
            path_length_dict = {}
            for node in nodes_mv:
                if isinstance(node, LVStation):
                    path_length_dict[node] = len(nx.shortest_path(node.mv_grid.graph, node.mv_grid.station, node))
                else:
                    path_length_dict[node] = len(nx.shortest_path(node.grid.graph, node.grid.station, node))
            # find node farthest away
            battery_node = [_ for _ in path_length_dict.keys()
                            if path_length_dict[_]==
                            max(path_length_dict.values())][0]
        # if there are only issues in the LV grids find LV station with issues closest to
        # MV station
        else:
            # dictionary with path lengths
            path_length_dict = {}
            for node in lv_stations:
                path_length_dict[node] = len(nx.shortest_path(
                    node.mv_grid.graph, node.mv_grid.station, node))
            # find closest node
            battery_node = [_ for _ in path_length_dict.keys()
                            if path_length_dict[_] ==
                            min(path_length_dict.values())][0]

        ### estimate residual line load (neglect grid losses) to identify all time steps where line is over-loaded
        # get line
        line = None
        counter = 0
        while line is None:
            if battery_node in lines[counter].grid.graph.nodes_from_line(lines[counter]):
                line = lines[counter]
            counter += 1
        # analyze
        edisgo.analyze()
        # ToDo: Differentiate between load and feedin case
        load_factor_line = edisgo.network.config['grid_expansion_load_factors'][
            'mv_feedin_case_line']
        i_allowed = line.type['I_max_th'] * load_factor_line * line.quantity
        u_allowed = line.type['U_n']
        i_pf = edisgo.network.results.i_res[repr(line)]

        # to calculate max p of line before last reinforcement
        S_ol_line_nom = sqrt(3) * i_allowed * u_allowed  # rated apparent power S of line kVA

        # maximum power of the line
        pfa_q = edisgo.network.results.pfa_q[repr(line)]
        # ToDo: storage efficiency
        p_max = sqrt(S_ol_line_nom ** 2 - pfa_q ** 2)

        # integrate storage
        storage_parameters['nominal_capacity'] = p_max
        storage_integration.set_up_storage(storage_parameters, battery_node)



    # for every feeder:
    # ** install storage at the node of the overloaded lines farthest away from the HV/MV transformer
    # ** estimate residual line load (neglect grid losses) to identify all time steps where line is over-loaded
    # ** determine storage capacity by adding storage dispatch time series to residual line load and checking if line load can be reduced; maximum needed reduction to avoid all line over-loadings determines the storage capacity
    # ** if there is storage capacity left, rerun the past steps for the next feeder in the ranking list
