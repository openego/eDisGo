from __future__ import annotations

from networkx import Graph
from pandas import DataFrame


def translate_df_to_graph(
    buses_df: DataFrame,
    lines_df: DataFrame,
    transformers_df: DataFrame | None = None,
) -> Graph:
    """
    Translate DataFrames to networkx Graph Object.

    Parameters
    ----------
    buses_df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with all buses to use as Graph nodes. For more information about the
        Dataframe see :attr:`~.network.topology.Topology.buses_df`.
    lines_df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with all lines to use as Graph branches. For more information about
        the Dataframe see :attr:`~.network.topology.Topology.lines_df`
    transformers_df : :pandas:`pandas.DataFrame<DataFrame>`, optional
        Dataframe with all transformers to use as additional Graph nodes. For more
        information about the Dataframe see
        :attr:`~.network.topology.Topology.transformers_df`

    Returns
    -------
    :networkx:`networkx.Graph<network.Graph>`
            Graph representation of the grid as networkx Ordered Graph,
            where lines are represented by edges in the graph, and buses and
            transformers are represented by nodes.

    """
    graph = Graph()

    # add nodes
    buses = [
        (bus_name, {"pos": (x, y)})
        for bus_name, x, y in buses_df[["x", "y"]].itertuples()
    ]

    graph.add_nodes_from(buses)

    # add branches
    branches = [
        (bus0, bus1, {"branch_name": line_name, "length": length})
        for line_name, bus0, bus1, length in lines_df[
            ["bus0", "bus1", "length"]
        ].itertuples()
    ]

    if transformers_df is not None:
        branches.extend(
            (bus0, bus1, {"branch_name": trafo_name, "length": 0})
            for trafo_name, bus0, bus1 in transformers_df[["bus0", "bus1"]].itertuples()
        )

    graph.add_edges_from(branches)

    return graph
