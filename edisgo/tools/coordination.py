import copy
import logging
import math

from time import time

import networkx as nx
import plotly.graph_objects as go

from dash import dcc, html
from dash.dependencies import Input, Output
from pyproj import Transformer


def draw_plotly(
    edisgo_obj,
    G,
    mode_lines=False,
    mode_nodes="adjecencies",
    grid=False,
    busmap_df=None,
):
    """
    Plot the graph and shows information of the grid

    Parameters
    ----------
    edisgo_obj : :class:`~edisgo.EDisGo`
        EDisGo object which contains data of the grid

    G : :networkx:`Graph`
        Transfer the graph of the grid to plot, the graph must contain the positions

    mode_lines : :obj:`str`
        Defines the color of the lines

        * 'relative_loading'
          - shows the line loading relative to the s_nom of the line
        * 'loading'
          - shows the loading
        * 'reinforce'
          - shows the reinforced lines in green

    mode_nodes : :obj:`str`

        * 'voltage_deviation'
          - shows the deviation of the node voltage relative to 1 p.u.
        * 'adjecencies'
          - shows the the number of connections of the graph

    grid : :class:`~.network.grids.Grid` or :obj:`False`

        * :class:`~.network.grids.Grid`
          - transfer the grid of the graph, to set the coordinate
          origin to the first bus of the grid
        * :obj:`False`
          - the coordinates are not modified

    """

    # initialization
    transformer_4326_to_3035 = Transformer.from_crs(
        "EPSG:4326", "EPSG:3035", always_xy=True
    )
    data = []
    if not grid:
        x_root = 0
        y_root = 0
    elif grid is None:
        node_root = edisgo_obj.topology.transformers_hvmv_df.bus1[0]
        x_root, y_root = G.nodes[node_root]["pos"]
    else:
        node_root = grid.transformers_df.bus1[0]
        x_root, y_root = G.nodes[node_root]["pos"]

    x_root, y_root = transformer_4326_to_3035.transform(x_root, y_root)

    # line text
    middle_node_x = []
    middle_node_y = []
    middle_node_text = []
    for edge in G.edges(data=True):
        x0, y0 = G.nodes[edge[0]]["pos"]
        x1, y1 = G.nodes[edge[1]]["pos"]
        x0, y0 = transformer_4326_to_3035.transform(x0, y0)
        x1, y1 = transformer_4326_to_3035.transform(x1, y1)
        middle_node_x.append((x0 - x_root + x1 - x_root) / 2)
        middle_node_y.append((y0 - y_root + y1 - y_root) / 2)

        text = str(edge[2]["branch_name"])
        try:
            loading = edisgo_obj.results.s_res.T.loc[
                edge[2]["branch_name"]
            ].max()  # * 1000
            text = text + "<br>" + "Loading = " + str(loading)
        except KeyError:
            text = text

        try:
            text = text + "<br>" + "GRAPH_LOAD = " + str(edge[2]["load"])
        except KeyError:
            text = text

        try:
            line_parameters = edisgo_obj.topology.lines_df.loc[
                edge[2]["branch_name"], :
            ]
            for index, value in line_parameters.iteritems():
                text = text + "<br>" + str(index) + " = " + str(value)
        except KeyError:
            text = text

        try:
            r = edisgo_obj.topology.lines_df.r.loc[edge[2]["branch_name"]]
            x = edisgo_obj.topology.lines_df.x.loc[edge[2]["branch_name"]]
            s_nom = edisgo_obj.topology.lines_df.s_nom.loc[edge[2]["branch_name"]]
            length = edisgo_obj.topology.lines_df.length.loc[edge[2]["branch_name"]]
            bus_0 = edisgo_obj.topology.lines_df.bus0.loc[edge[2]["branch_name"]]
            v_nom = edisgo_obj.topology.buses_df.loc[bus_0, "v_nom"]
            import math

            text = text + "<br>" + "r/length = " + str(r / length)
            text = (
                text
                + "<br>"
                + "x/length = "
                + str(x / length / 2 / math.pi / 50 * 1000)
            )
            text = text + "<br>" + "i_max_th = " + str(s_nom / math.sqrt(3) / v_nom)
        except KeyError:
            text = text

        middle_node_text.append(text)

    middle_node_trace = go.Scatter(
        x=middle_node_x,
        y=middle_node_y,
        text=middle_node_text,
        mode="markers",
        hoverinfo="text",
        marker=dict(opacity=0.0, size=10, color="white"),
    )
    data.append(middle_node_trace)

    # line plot
    import matplotlib as matplotlib
    import matplotlib.cm as cm

    if mode_lines == "loading":
        s_res_view = edisgo_obj.results.s_res.T.index.isin(
            [edge[2]["branch_name"] for edge in G.edges.data()]
        )
        color_min = edisgo_obj.results.s_res.T.loc[s_res_view].T.min().max()
        color_max = edisgo_obj.results.s_res.T.loc[s_res_view].T.max().max()
    elif mode_lines == "relative_loading":
        color_min = 0
        color_max = 1

    if (mode_lines != "reinforce") and not mode_lines:

        def color_map_color(
            value, cmap_name="coolwarm", vmin=color_min, vmax=color_max
        ):
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap(cmap_name)
            rgb = cmap(norm(abs(value)))[:3]
            color = matplotlib.colors.rgb2hex(rgb)
            return color

    for edge in G.edges(data=True):
        edge_x = []
        edge_y = []

        x0, y0 = G.nodes[edge[0]]["pos"]
        x1, y1 = G.nodes[edge[1]]["pos"]
        x0, y0 = transformer_4326_to_3035.transform(x0, y0)
        x1, y1 = transformer_4326_to_3035.transform(x1, y1)
        edge_x.append(x0 - x_root)
        edge_x.append(x1 - x_root)
        edge_x.append(None)
        edge_y.append(y0 - y_root)
        edge_y.append(y1 - y_root)
        edge_y.append(None)

        if mode_lines == "reinforce":
            if edisgo_obj.results.grid_expansion_costs.index.isin(
                [edge[2]["branch_name"]]
            ).any():
                color = "lightgreen"
            else:
                color = "black"
        elif mode_lines == "loading":
            loading = edisgo_obj.results.s_res.T.loc[edge[2]["branch_name"]].max()
            color = color_map_color(loading)
        elif mode_lines == "relative_loading":
            loading = edisgo_obj.results.s_res.T.loc[edge[2]["branch_name"]].max()
            s_nom = edisgo_obj.topology.lines_df.s_nom.loc[edge[2]["branch_name"]]
            color = color_map_color(loading / s_nom)
            if loading > s_nom:
                color = "green"
        else:
            color = "black"

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            hoverinfo="none",
            opacity=0.4,
            mode="lines",
            line=dict(width=2, color=color),
        )
        data.append(edge_trace)

    # node plot
    node_x = []
    node_y = []

    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        x, y = transformer_4326_to_3035.transform(x, y)
        node_x.append(x - x_root)
        node_y.append(y - y_root)

    colors = []
    if mode_nodes == "adjecencies":
        for node, adjacencies in enumerate(G.adjacency()):
            colors.append(len(adjacencies[1]))
        colorscale = "YlGnBu"
        cmid = None
        colorbar = dict(
            thickness=15, title="Node Connections", xanchor="left", titleside="right"
        )
    elif mode_nodes == "voltage_deviation":
        for node in G.nodes():
            v_min = edisgo_obj.results.v_res.T.loc[node].min()
            v_max = edisgo_obj.results.v_res.T.loc[node].max()
            if abs(v_min - 1) > abs(v_max - 1):
                color = v_min - 1
            else:
                color = v_max - 1
            colors.append(color)
        colorscale = "RdBu"
        cmid = 0
        colorbar = dict(
            thickness=15,
            title="Node Voltage Deviation",
            xanchor="left",
            titleside="right",
        )

    node_text = []
    for node in G.nodes():
        text = str(node)
        try:
            peak_load = edisgo_obj.topology.loads_df.loc[
                edisgo_obj.topology.loads_df.bus == node
            ].peak_load.sum()
            text = text + "<br>" + "peak_load = " + str(peak_load)
            p_nom = edisgo_obj.topology.generators_df.loc[
                edisgo_obj.topology.generators_df.bus == node
            ].p_nom.sum()
            text = text + "<br>" + "p_nom_gen = " + str(p_nom)
            p_charge = edisgo_obj.topology.charging_points_df.loc[
                edisgo_obj.topology.charging_points_df.bus == node
            ].p_nom.sum()
            text = text + "<br>" + "p_nom_charge = " + str(p_charge)
        except ValueError:
            text = text

        try:
            s_tran_1 = edisgo_obj.topology.transformers_df.loc[
                edisgo_obj.topology.transformers_df.bus0 == node, "s_nom"
            ].sum()
            s_tran_2 = edisgo_obj.topology.transformers_df.loc[
                edisgo_obj.topology.transformers_df.bus1 == node, "s_nom"
            ].sum()
            s_tran = s_tran_1 + s_tran_2
            text = text + "<br>" + "s_transformer = {:.2f}kVA".format(s_tran * 1000)
        except KeyError:
            text = text

        try:
            v_min = edisgo_obj.results.v_res.T.loc[node].min()
            v_max = edisgo_obj.results.v_res.T.loc[node].max()
            if abs(v_min - 1) > abs(v_max - 1):
                text = text + "<br>" + "v = " + str(v_min)
            else:
                text = text + "<br>" + "v = " + str(v_max)
        except KeyError:
            text = text

        try:
            text = text + "<br>" + "Neighbors = " + str(G.degree(node))
        except KeyError:
            text = text

        try:
            node_parameters = edisgo_obj.topology.buses_df.loc[node]
            for index, value in node_parameters.iteritems():
                text = text + "<br>" + str(index) + " = " + str(value)
        except KeyError:
            text = text

        if busmap_df is not None:
            text = text + "<br>" + "new_bus_name = " + busmap_df.loc[node, "new_bus"]

        node_text.append(text)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            reversescale=True,
            color=colors,
            size=8,
            cmid=cmid,
            line_width=2,
            colorbar=colorbar,
        ),
    )

    data.append(node_trace)

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            height=500,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
            yaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
        ),
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig.update_yaxes(tick0=0, dtick=1000)
    # fig.update_xaxes(tick0=0, dtick=1000)
    return fig


def dash_plot(**kwargs):
    """
    Uses the :func:`draw_plotly` for interactive plotting.

    Shows different behavior for different number of parameters.
    One edisgo object creates one large plot.
    Two or more edisgo objects create two adjacent plots,
    the objects to be plotted are selected in the dropdown menu.

    **Example run:**

        | app = dash_plot(edisgo_obj_1=edisgo_obj_1,edisgo_obj_2=edisgo_obj_2,...)
        | app.run_server(mode="inline",debug=True)

    """

    from jupyter_dash import JupyterDash

    def chosen_graph(edisgo_obj, selected_grid):
        lv_grids = list(edisgo_obj.topology.mv_grid.lv_grids)
        lv_grid_name_list = list(map(str, lv_grids))
        # selected_grid = "LVGrid_452669"
        # selected_grid = lv_grid_name_list[0]
        try:
            lv_grid_id = lv_grid_name_list.index(selected_grid)
        except ValueError:
            lv_grid_id = False

        mv_grid = edisgo_obj.topology.mv_grid
        lv_grid = lv_grids[lv_grid_id]

        if selected_grid == "Grid":
            G = edisgo_obj.to_graph()
            grid = None
        elif selected_grid == str(mv_grid):
            G = mv_grid.graph
            grid = mv_grid
        elif selected_grid.split("_")[0] == "LVGrid":
            G = lv_grid.graph
            grid = lv_grid
        else:
            raise ValueError("False Grid")

        return G, grid

    edisgo_obj = list(kwargs.values())[0]
    mv_grid = edisgo_obj.topology.mv_grid
    lv_grids = list(edisgo_obj.topology.mv_grid.lv_grids)

    edisgo_name_list = list(kwargs.keys())

    lv_grid_name_list = list(map(str, lv_grids))

    grid_name_list = ["Grid", str(mv_grid)] + lv_grid_name_list

    line_plot_modes = ["reinforce", "loading", "relative_loading"]
    node_plot_modes = ["adjecencies", "voltage_deviation"]

    app = JupyterDash(__name__)
    if len(kwargs) > 1:
        app.layout = html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id="dropdown_edisgo_object_1",
                            options=[
                                {"label": i, "value": i} for i in edisgo_name_list
                            ],
                            value=edisgo_name_list[0],
                        ),
                        dcc.Dropdown(
                            id="dropdown_edisgo_object_2",
                            options=[
                                {"label": i, "value": i} for i in edisgo_name_list
                            ],
                            value=edisgo_name_list[1],
                        ),
                        dcc.Dropdown(
                            id="dropdown_grid",
                            options=[{"label": i, "value": i} for i in grid_name_list],
                            value=grid_name_list[1],
                        ),
                        dcc.Dropdown(
                            id="dropdown_line_plot_mode",
                            options=[{"label": i, "value": i} for i in line_plot_modes],
                            value=line_plot_modes[0],
                        ),
                        dcc.Dropdown(
                            id="dropdown_node_plot_mode",
                            options=[{"label": i, "value": i} for i in node_plot_modes],
                            value=node_plot_modes[0],
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div([dcc.Graph(id="fig_1")], style={"flex": "auto"}),
                        html.Div([dcc.Graph(id="fig_2")], style={"flex": "auto"}),
                    ],
                    style={"display": "flex", "flex-direction": "row"},
                ),
            ],
            style={"display": "flex", "flex-direction": "column"},
        )

        @app.callback(
            Output("fig_1", "figure"),
            Output("fig_2", "figure"),
            Input("dropdown_grid", "value"),
            Input("dropdown_edisgo_object_1", "value"),
            Input("dropdown_edisgo_object_2", "value"),
            Input("dropdown_line_plot_mode", "value"),
            Input("dropdown_node_plot_mode", "value"),
        )
        def update_figure(
            selected_grid,
            selected_edisgo_object_1,
            selected_edisgo_object_2,
            selected_line_plot_mode,
            selected_node_plot_mode,
        ):

            edisgo_obj = kwargs[selected_edisgo_object_1]
            (G, grid) = chosen_graph(edisgo_obj, selected_grid)
            fig_1 = draw_plotly(
                edisgo_obj,
                G,
                selected_line_plot_mode,
                selected_node_plot_mode,
                grid=grid,
            )

            edisgo_obj = kwargs[selected_edisgo_object_2]
            (G, grid) = chosen_graph(edisgo_obj, selected_grid)
            fig_2 = draw_plotly(
                edisgo_obj,
                G,
                selected_line_plot_mode,
                selected_node_plot_mode,
                grid=grid,
            )

            return fig_1, fig_2

    else:
        app.layout = html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id="dropdown_grid",
                            options=[{"label": i, "value": i} for i in grid_name_list],
                            value=grid_name_list[1],
                        ),
                        dcc.Dropdown(
                            id="dropdown_line_plot_mode",
                            options=[{"label": i, "value": i} for i in line_plot_modes],
                            value=line_plot_modes[0],
                        ),
                        dcc.Dropdown(
                            id="dropdown_node_plot_mode",
                            options=[{"label": i, "value": i} for i in node_plot_modes],
                            value=node_plot_modes[0],
                        ),
                    ]
                ),
                html.Div(
                    [html.Div([dcc.Graph(id="fig")], style={"flex": "auto"})],
                    style={"display": "flex", "flex-direction": "row"},
                ),
            ],
            style={"display": "flex", "flex-direction": "column"},
        )

        @app.callback(
            Output("fig", "figure"),
            Input("dropdown_grid", "value"),
            Input("dropdown_line_plot_mode", "value"),
            Input("dropdown_node_plot_mode", "value"),
        )
        def update_figure(
            selected_grid, selected_line_plot_mode, selected_node_plot_mode
        ):

            edisgo_obj = list(kwargs.values())[0]
            (G, grid) = chosen_graph(edisgo_obj, selected_grid)
            fig = draw_plotly(
                edisgo_obj,
                G,
                selected_line_plot_mode,
                selected_node_plot_mode,
                grid=grid,
            )
            return fig

    return app


# Functions for other functions
coor_transform = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
coor_transform_back = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


# Pseudo coordinates
def make_pseudo_coordinates(edisgo_root):
    def make_coordinates(graph_root):
        def coordinate_source(pos_start, length, node_numerator, node_total_numerator):
            length = length / 1.3
            angle = node_numerator * 360 / node_total_numerator
            x0, y0 = pos_start
            x1 = x0 + 1000 * length * math.cos(math.radians(angle))
            y1 = y0 + 1000 * length * math.sin(math.radians(angle))
            pos_end = (x1, y1)
            origin_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            return pos_end, origin_angle

        def coordinate_branch(
            pos_start, angle_offset, length, node_numerator, node_total_numerator
        ):
            length = length / 1.3
            angle = (
                node_numerator * 180 / (node_total_numerator + 1) + angle_offset - 90
            )
            x0, y0 = pos_start
            x1 = x0 + 1000 * length * math.cos(math.radians(angle))
            y1 = y0 + 1000 * length * math.sin(math.radians(angle))
            origin_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            pos_end = (x1, y1)
            return pos_end, origin_angle

        def coordinate_longest_path(pos_start, angle_offset, length):
            length = length / 1.3
            angle = angle_offset
            x0, y0 = pos_start
            x1 = x0 + 1000 * length * math.cos(math.radians(angle))
            y1 = y0 + 1000 * length * math.sin(math.radians(angle))
            origin_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            pos_end = (x1, y1)
            return pos_end, origin_angle

        def coordinate_longest_path_neighbor(
            pos_start, angle_offset, length, direction
        ):
            length = length / 1.3
            if direction:
                angle_random_offset = 90
            else:
                angle_random_offset = -90
            angle = angle_offset + angle_random_offset
            x0, y0 = pos_start
            x1 = x0 + 1000 * length * math.cos(math.radians(angle))
            y1 = y0 + 1000 * length * math.sin(math.radians(angle))
            origin_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            pos_end = (x1, y1)

            return pos_end, origin_angle

        start_node = list(nx.nodes(graph_root))[0]
        graph_root.nodes[start_node]["pos"] = (0, 0)
        graph_copy = graph_root.copy()

        long_paths = []
        next_nodes = []

        for i in range(1, 30):
            path_length_to_transformer = []
            for node in graph_copy.nodes():
                try:
                    paths = list(nx.shortest_simple_paths(graph_copy, start_node, node))
                except ValueError:
                    paths = [[]]
                path_length_to_transformer.append(len(paths[0]))
            index = path_length_to_transformer.index(max(path_length_to_transformer))
            path_to_max_distance_node = list(
                nx.shortest_simple_paths(
                    graph_copy, start_node, list(nx.nodes(graph_copy))[index]
                )
            )[0]
            path_to_max_distance_node.remove(start_node)
            graph_copy.remove_nodes_from(path_to_max_distance_node)
            for node in path_to_max_distance_node:
                long_paths.append(node)

        path_to_max_distance_node = long_paths
        n = 0

        for node in list(nx.neighbors(graph_root, start_node)):
            n = n + 1
            pos, origin_angle = coordinate_source(
                graph_root.nodes[start_node]["pos"],
                graph_root.edges[start_node, node]["length"],
                n,
                len(list(nx.neighbors(graph_root, start_node))),
            )
            graph_root.nodes[node]["pos"] = pos
            graph_root.nodes[node]["origin_angle"] = origin_angle
            next_nodes.append(node)

        graph_copy = graph_root.copy()
        graph_copy.remove_node(start_node)
        while graph_copy.number_of_nodes() > 0:
            next_node = next_nodes[0]
            n = 0
            for node in list(nx.neighbors(graph_copy, next_node)):
                n = n + 1
                if node in path_to_max_distance_node:
                    pos, origin_angle = coordinate_longest_path(
                        graph_root.nodes[next_node]["pos"],
                        graph_root.nodes[next_node]["origin_angle"],
                        graph_root.edges[next_node, node]["length"],
                    )
                elif next_node in path_to_max_distance_node:
                    direction = math.fmod(
                        len(
                            list(
                                nx.shortest_simple_paths(
                                    graph_root, start_node, next_node
                                )
                            )[0]
                        ),
                        2,
                    )
                    pos, origin_angle = coordinate_longest_path_neighbor(
                        graph_root.nodes[next_node]["pos"],
                        graph_root.nodes[next_node]["origin_angle"],
                        graph_root.edges[next_node, node]["length"],
                        direction,
                    )
                else:
                    pos, origin_angle = coordinate_branch(
                        graph_root.nodes[next_node]["pos"],
                        graph_root.nodes[next_node]["origin_angle"],
                        graph_root.edges[next_node, node]["length"],
                        n,
                        len(list(nx.neighbors(graph_copy, next_node))),
                    )

                graph_root.nodes[node]["pos"] = pos
                graph_root.nodes[node]["origin_angle"] = origin_angle
                next_nodes.append(node)

            graph_copy.remove_node(next_node)
            next_nodes.remove(next_node)

        return graph_root

    logger = logging.getLogger("edisgo.cr_make_pseudo_coor")
    start_time = time()
    logger.info(
        "Start - Making pseudo coordinates for grid: {}".format(
            str(edisgo_root.topology.mv_grid)
        )
    )

    edisgo_obj = copy.deepcopy(edisgo_root)
    lv_grids = list(edisgo_obj.topology.mv_grid.lv_grids)

    for lv_grid in lv_grids:
        logger.debug("Make pseudo coordinates for: {}".format(lv_grid))
        G = lv_grid.graph
        x0, y0 = G.nodes[list(nx.nodes(G))[0]]["pos"]
        G = make_coordinates(G)
        x0, y0 = coor_transform.transform(x0, y0)
        for node in G.nodes():
            x, y = G.nodes[node]["pos"]
            x, y = coor_transform_back.transform(x + x0, y + y0)
            edisgo_obj.topology.buses_df.loc[node, "x"] = x
            edisgo_obj.topology.buses_df.loc[node, "y"] = y

    logger.info("Finished in {}s".format(time() - start_time))
    return edisgo_obj
