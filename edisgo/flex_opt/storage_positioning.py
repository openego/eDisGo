import pandas as pd
import networkx as nx
from edisgo.flex_opt.check_tech_constraints import mv_line_load


def one_storage_per_feeder(edisgo):

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

    # # rank MV feeders by grid expansion costs calculated with feed-in case
    # if edisgo.network.results.grid_expansion_costs is not None:
    #     ranked_feeders = feeder_ranking(
    #         edisgo.network.results.grid_expansion_costs)
    # else:
    #     # conduct grid reinforcement
    #     edisgo.reinforce()
    #     # rank feeders
    #     ranked_feeders = feeder_ranking(
    #         edisgo.network.results.grid_expansion_costs)
    #     # reset edisgo object (new grid -> moved to reinforce_grid)
    #     #edisgo.import_from_ding0(kwargs.get('ding0_grid', None))

    # work around for now: random feeder ranking
    # get nodes adjacent to HV/MV station
    station_adj_nodes = list(edisgo.network.mv_grid.graph.edge[
        edisgo.network.mv_grid.station].keys())
    # for each adjacent node check if
    ranked_feeders = []
    for node in station_adj_nodes:
        if len(edisgo.network.mv_grid.graph.edge[node]) > 1:
            ranked_feeders.append(edisgo.network.mv_grid.graph.line_from_nodes(
                edisgo.network.mv_grid.station, node))

    # get list of overloaded lines
    critical_lines = mv_line_load(edisgo.network)

    print(critical_lines)

    # find the longest nx. shortest path from all the nodes
    # get all nodes in the mv_grid
    graph_without_station = edisgo.network.mv_grid.graph.copy()
    mv_station = graph_without_station.nodes_by_attribute('mv_station')
    station_nearest_nodes = list(graph_without_station.edge[
                                     mv_station.keys()])

    graph_without_station.remove_node(mv_station)
    all_nodes = list(graph_without_station.nodes)
    shortest_paths = {}
    for node in all_nodes:
        for station_near_node in station_nearest_nodes:
            shortest_paths[(station_near_node, node)] = \
                nx.shortest_path(graph_without_station, station_near_node, node)

    print(shortest_paths)




    # for feeder in ranked_feeders:
        # get the nodes farthest away from HV/MV transformer



    # for every feeder:
    # ** install storage at the node of the overloaded lines farthest away from the HV/MV transformer
    # ** estimate residual line load (neglect grid losses) to identify all time steps where line is over-loaded
    # ** determine storage capacity by adding storage dispatch time series to residual line load and checking if line load can be reduced; maximum needed reduction to avoid all line over-loadings determines the storage capacity
    # ** if there is storage capacity left, rerun the past steps for the next feeder in the ranking list
