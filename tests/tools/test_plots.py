import copy

import numpy as np
import pandas as pd
import pytest

from matplotlib import pyplot as plt

from edisgo import EDisGo
from edisgo.tools import plots
from edisgo.tools.plots import chosen_graph, plot_dash_app, plot_plotly


class TestPlots:
    @classmethod
    def setup_class(cls):
        cls.edisgo_root = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        cls.edisgo_root.set_time_series_worst_case_analysis()
        cls.edisgo_analyzed = copy.deepcopy(cls.edisgo_root)
        cls.edisgo_reinforced = copy.deepcopy(cls.edisgo_root)
        cls.edisgo_analyzed.analyze()
        cls.edisgo_reinforced.reinforce()
        cls.edisgo_reinforced.results.equipment_changes.loc["Line_10006", "change"] = (
            "added"
        )

    @pytest.mark.parametrize(
        "line_color,"
        "node_color,"
        "line_result_selection,"
        "node_result_selection,"
        "plot_map,"
        "pseudo_coordinates",
        [
            ("loading", "voltage_deviation", "min", "min", True, True),
            ("relative_loading", "adjacencies", "max", "max", False, False),
            ("reinforce", "adjacencies", "max", "min", True, False),
        ],
    )
    @pytest.mark.parametrize(
        "selected_timesteps",
        [
            None,
            "1970-01-01 01:00:00",
            ["1970-01-01 01:00:00", "1970-01-01 03:00:00"],
        ],
    )
    @pytest.mark.parametrize(
        "node_selection", [False, ["Bus_MVStation_1", "Bus_BranchTee_MVGrid_1_5"]]
    )
    @pytest.mark.parametrize(
        "edisgo_obj_name", ["edisgo_root", "edisgo_analyzed", "edisgo_reinforced"]
    )
    @pytest.mark.parametrize("grid_name", ["None", "LVGrid"])
    def test_plot_plotly(
        self,
        edisgo_obj_name,
        grid_name,
        line_color,
        node_color,
        line_result_selection,
        node_result_selection,
        selected_timesteps,
        plot_map,
        pseudo_coordinates,
        node_selection,
    ):
        if edisgo_obj_name == "edisgo_root":
            edisgo_obj = self.edisgo_root
        elif edisgo_obj_name == "edisgo_analyzed":
            edisgo_obj = self.edisgo_analyzed
        elif edisgo_obj_name == "edisgo_reinforced":
            edisgo_obj = self.edisgo_reinforced

        if grid_name == "None":
            grid = None
        elif grid_name == "LVGrid":
            grid = list(edisgo_obj.topology.mv_grid.lv_grids)[1]

        if (grid_name == "LVGrid") and (node_selection is not False):
            with pytest.raises(ValueError):
                plot_plotly(
                    edisgo_obj=edisgo_obj,
                    grid=grid,
                    line_color=line_color,
                    node_color=node_color,
                    line_result_selection=line_result_selection,
                    node_result_selection=node_result_selection,
                    selected_timesteps=selected_timesteps,
                    plot_map=plot_map,
                    pseudo_coordinates=pseudo_coordinates,
                    node_selection=node_selection,
                )
        else:
            plot_plotly(
                edisgo_obj=edisgo_obj,
                grid=grid,
                line_color=line_color,
                node_color=node_color,
                line_result_selection=line_result_selection,
                node_result_selection=node_result_selection,
                selected_timesteps=selected_timesteps,
                plot_map=plot_map,
                pseudo_coordinates=pseudo_coordinates,
                node_selection=node_selection,
            )

    def test_plot_plotly_custom_colors(self):
        def line_colors(fig):
            return {t.line.color for t in fig.data if t.mode == "lines"}

        def node_colors(fig):
            # the node trace has one color per node, other marker traces only one
            return [
                c
                for t in fig.data
                if t.mode == "markers" and isinstance(t.marker.color, tuple)
                for c in t.marker.color
            ]

        # lines and nodes that are not in the Series or dict are grey
        fig = plot_plotly(
            edisgo_obj=self.edisgo_root,
            line_color=pd.Series({"Line_10003": "red", "Line_10004": "blue"}),
            node_color=pd.Series({"Bus_MVStation_1": "orange"}),
        )
        assert line_colors(fig) == {"red", "blue", "grey"}
        assert "orange" in node_colors(fig)
        assert set(node_colors(fig)) == {"orange", "grey"}

        fig = plot_plotly(
            edisgo_obj=self.edisgo_root,
            line_color={"Line_10003": "green"},
            node_color={"Bus_MVStation_1": "cyan"},
        )
        assert line_colors(fig) == {"green", "grey"}
        assert set(node_colors(fig)) == {"cyan", "grey"}

    def test_chosen_graph(self):
        chosen_graph(edisgo_obj=self.edisgo_root, selected_grid="Grid")
        grid = str(self.edisgo_root.topology.mv_grid)
        chosen_graph(edisgo_obj=self.edisgo_root, selected_grid=grid)
        grid = list(map(str, self.edisgo_root.topology.mv_grid.lv_grids))[0]
        chosen_graph(edisgo_obj=self.edisgo_root, selected_grid=grid)

    def test_plot_dash_app(self):
        # TODO: at the moment this doesn't really test anything. Add meaningful tests.
        # test if any errors occur when only passing one edisgo object
        plot_dash_app(
            edisgo_objects=self.edisgo_root,
        )

        # test if any errors occur when passing multiple edisgo objects
        plot_dash_app(  # noqa: F841
            edisgo_objects={
                "edisgo_1": self.edisgo_root,
                "edisgo_2": self.edisgo_reinforced,
            }
        )

    def test_add_basemap_timeout(self, monkeypatch):
        # add_basemap passes the timeout to contextily and passes a failed
        # download on to the caller
        calls = {}

        def bounds2img(*args, timeout=None, **kwargs):
            calls["timeout"] = timeout
            raise TimeoutError("tile download timed out")

        monkeypatch.setattr(plots.ctx, "bounds2img", bounds2img)
        fig, ax = plt.subplots()
        with pytest.raises(TimeoutError):
            plots.add_basemap(ax)
        plt.close(fig)
        assert calls["timeout"] == 10

    def test_add_basemap_source_note(self, monkeypatch):
        # the TopPlusOpen licence requires the BKG source note on the map
        def bounds2img(*args, **kwargs):
            return np.zeros((256, 256, 4)), (0, 1, 0, 1)

        monkeypatch.setattr(plots.ctx, "bounds2img", bounds2img)
        fig, ax = plt.subplots()
        plots.add_basemap(ax)
        texts = [text.get_text() for text in ax.texts]
        plt.close(fig)
        assert any(text.startswith("© BKG (") for text in texts)
        assert any("dl-de/by-2-0" in text for text in texts)
