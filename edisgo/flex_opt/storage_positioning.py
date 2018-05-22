import pandas as pd


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

    # rank MV feeders by grid expansion costs calculated with feed-in case
    if edisgo.network.results.grid_expansion_costs is not None:
        ranked_feeders = feeder_ranking(
            edisgo.network.results.grid_expansion_costs)
    else:
        # conduct grid reinforcement
        edisgo.reinforce()
        # rank feeders
        ranked_feeders = feeder_ranking(
            edisgo.network.results.grid_expansion_costs)
        # reset edisgo object (new grid -> moved to reinforce_grid)
        #edisgo.import_from_ding0(kwargs.get('ding0_grid', None))
    # for every feeder:
    # ** install storage at the node of the overloaded lines farthest away from the HV/MV transformer
    # ** estimate residual line load (neglect grid losses) to identify all time steps where line is over-loaded
    # ** determine storage capacity by adding storage dispatch time series to residual line load and checking if line load can be reduced; maximum needed reduction to avoid all line over-loadings determines the storage capacity
    # ** if there is storage capacity left, rerun the past steps for the next feeder in the ranking list
