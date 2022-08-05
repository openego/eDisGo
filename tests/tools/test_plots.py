import copy

import pytest

from edisgo import EDisGo
from edisgo.tools.plots import chosen_graph, dash_plot, draw_plotly


class TestPlots:
    @classmethod
    def setup_class(cls):
        cls.edisgo_root = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        cls.edisgo_root.set_time_series_worst_case_analysis()
        cls.edisgo_analyzed = copy.deepcopy(cls.edisgo_root)
        cls.edisgo_reinforced = copy.deepcopy(cls.edisgo_root)
        cls.edisgo_analyzed.analyze()
        cls.edisgo_reinforced.reinforce()

    def test_draw_plotly(self):
        # test
        edisgo_obj = self.edisgo_root
        grid = edisgo_obj.topology.mv_grid
        G = grid.graph

        mode_lines = "reinforce"
        mode_nodes = "adjacencies"
        fig = draw_plotly(
            edisgo_obj, G, line_color=mode_lines, node_color=mode_nodes, grid=grid
        )
        fig.show()

        edisgo_obj = self.edisgo_reinforced
        grid = edisgo_obj.topology.mv_grid
        G = grid.graph

        mode_lines = "relative_loading"
        mode_nodes = "voltage_deviation"
        fig = draw_plotly(
            edisgo_obj, G, line_color=mode_lines, node_color=mode_nodes, grid=grid
        )
        fig.show()

        # plotting loading and voltage deviation, with unchanged coordinates
        mode_lines = "loading"
        mode_nodes = "voltage_deviation"
        fig = draw_plotly(
            edisgo_obj, G, line_color=mode_lines, node_color=mode_nodes, grid=False
        )
        fig.show()

        # plotting reinforced lines and node adjacencies
        edisgo_obj = self.edisgo_reinforced
        edisgo_obj.results.equipment_changes.loc["Line_10006", "change"] = "added"
        G = edisgo_obj.topology.mv_grid.graph

        mode_lines = "reinforce"
        mode_nodes = "adjacencies"
        fig = draw_plotly(
            edisgo_obj, G, line_color=mode_lines, node_color=mode_nodes, grid=False
        )
        fig.show()

    def test_chosen_graph(self):
        chosen_graph(edisgo_obj=self.edisgo_root, selected_grid="Grid")
        grid = str(self.edisgo_root.topology.mv_grid)
        chosen_graph(edisgo_obj=self.edisgo_root, selected_grid=grid)
        grid = list(map(str, self.edisgo_root.topology.mv_grid.lv_grids))[0]
        chosen_graph(edisgo_obj=self.edisgo_root, selected_grid=grid)

    def test_dash_plot(self):
        # TODO: at the moment this doesn't really test anything. Add meaningful tests.
        # test if any errors occur when only passing one edisgo object
        app = dash_plot(
            edisgo_objects=self.edisgo_root,
        )

        # test if any errors occur when passing multiple edisgo objects
        app = dash_plot(  # noqa: F841
            edisgo_objects={
                "edisgo_1": self.edisgo_root,
                "edisgo_2": self.edisgo_reinforced,
            }
        )
