from networkx import Graph


def translate_df_to_graph(buses_df, lines_df, transformers_df=None):
    graph = Graph()

    buses = []
    for bus_name, bus in buses_df.iterrows():
        pos = (bus.x, bus.y)
        buses.append((bus_name, {'pos': pos}))
        # add nodes
    graph.add_nodes_from(buses)
    # add branches
    branches = []
    for line_name, line in lines_df.iterrows():
        branches.append(
            (
                line.bus0,
                line.bus1,
                {"branch_name": line_name, "length": line.length},
            )
        )
    if transformers_df is not None:
        for trafo_name, trafo in transformers_df.iterrows():
            branches.append(
                (
                    trafo.bus0,
                    trafo.bus1,
                    {"branch_name": trafo_name, "length": 0},
                )
            )
    graph.add_edges_from(branches)
    return graph
