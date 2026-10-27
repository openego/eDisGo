import copy

import pandas as pd
import pytest

from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints
from edisgo.flex_opt.expansion_path import reinforce_expansion_path


def _grow_loads_by(factor):
    """Stage function that grows the loads by `factor`."""

    def apply_stage(edisgo_obj):
        edisgo_obj.topology.loads_df["p_set"] *= factor
        edisgo_obj.set_time_series_worst_case_analysis()

    return apply_stage


class TestReinforceExpansionPath:
    @classmethod
    def setup_class(cls):
        cls.edisgo_original = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        cls.edisgo_original.set_time_series_worst_case_analysis()
        # loads grow to 1.5 and 2 times their original size
        cls.stages = {
            2035: _grow_loads_by(1.5),
            2030: _grow_loads_by(1.0),
            2040: _grow_loads_by(2.0 / 1.5),
        }

    def setup_method(self):
        self.edisgo = copy.deepcopy(self.edisgo_original)

    @pytest.mark.slow
    def test_expansion_path(self):
        results = reinforce_expansion_path(
            self.edisgo, self.stages, discount_rate=0.1, compare_direct_reinforce=True
        )
        summary = results.summary

        # stages are processed in the order of their years
        assert list(summary.index) == [2030, 2035, 2040]
        assert list(summary.columns) == [
            "costs",
            "costs_mv",
            "costs_lv",
            "present_value",
            "cumulative_costs",
            "cumulative_present_value",
            "direct_reinforce_costs",
        ]
        # every stage needs reinforcement in this grid
        assert (summary.costs > 0).all()
        pd.testing.assert_series_equal(
            summary.costs,
            results.grid_expansion_costs.groupby("year").total_costs.sum(),
            check_names=False,
        )
        costs = results.grid_expansion_costs
        is_mv = costs.voltage_level.isin(["mv", "hv/mv"])
        pd.testing.assert_series_equal(
            summary.costs_mv,
            costs[is_mv]
            .groupby("year")
            .total_costs.sum()
            .reindex(summary.index, fill_value=0.0),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            summary.costs_lv,
            costs[~is_mv]
            .groupby("year")
            .total_costs.sum()
            .reindex(summary.index, fill_value=0.0),
            check_names=False,
        )
        # discounting to the first stage
        assert summary.at[2030, "present_value"] == pytest.approx(
            summary.at[2030, "costs"]
        )
        assert summary.at[2040, "present_value"] == pytest.approx(
            summary.at[2040, "costs"] / 1.1**10
        )
        assert summary.cumulative_costs.iloc[-1] == pytest.approx(summary.costs.sum())
        # the first stage is the same as reinforcing the original grid directly
        assert summary.at[2030, "direct_reinforce_costs"] == pytest.approx(
            summary.at[2030, "costs"]
        )
        # reinforcing directly for 2040 applies the stage functions of all years
        edisgo_direct = copy.deepcopy(self.edisgo_original)
        edisgo_direct.topology.loads_df["p_set"] *= 2.0
        edisgo_direct.set_time_series_worst_case_analysis()
        edisgo_direct.reinforce()
        assert summary.at[2040, "direct_reinforce_costs"] == pytest.approx(
            edisgo_direct.results.grid_expansion_costs.total_costs.sum()
        )

        # equipment changes of all stages are in the EDisGo object
        pd.testing.assert_frame_equal(
            results.equipment_changes.drop(columns="year"),
            self.edisgo.results.equipment_changes,
        )
        assert set(results.equipment_changes.year) == {2030, 2035, 2040}
        # lines reinforced again in a later stage pay earthworks again, while the
        # costs in the EDisGo object count the earthworks of every line once
        assert summary.costs.sum() > (
            self.edisgo.results.grid_expansion_costs.total_costs.sum()
        )

        # the grid is reinforced for the last stage
        assert self.edisgo.topology.loads_df.p_set.sum() == pytest.approx(
            2 * self.edisgo_original.topology.loads_df.p_set.sum()
        )
        self.edisgo.analyze()
        assert (
            check_tech_constraints.components_relative_load(self.edisgo).max().max()
            <= 1.0 + 1e-6
        )

    def test_stage_without_growth_costs_nothing(self):
        results = self.edisgo.reinforce_expansion_path(
            {
                2030: _grow_loads_by(1.5),
                2035: _grow_loads_by(1.0),
            },
            base_year=2025,
        )
        assert results.summary.at[2030, "costs"] > 0
        assert results.summary.at[2035, "costs"] == 0
        assert results.summary.at[2030, "present_value"] == pytest.approx(
            results.summary.at[2030, "costs"] / 1.05**5
        )
        assert "direct_reinforce_costs" not in results.summary.columns
        assert (results.equipment_changes.year == 2030).all()
        assert (results.grid_expansion_costs.year == 2030).all()

    def test_line_reinforced_before_is_split(self):
        # Line_10006 is reinforced in 2030, in 2035 a generator is connected to it,
        # which splits the line and removes it from the equipment changes
        line = "Line_10006"
        buses = self.edisgo.topology.buses_df
        bus0, bus1 = self.edisgo.topology.lines_df.loc[line, ["bus0", "bus1"]]
        geolocation = Point(
            (buses.at[bus0, "x"] + buses.at[bus1, "x"]) / 2 + 1e-5,
            (buses.at[bus0, "y"] + buses.at[bus1, "y"]) / 2,
        )

        def connect_generator(edisgo_obj):
            edisgo_obj.integrate_component_based_on_geolocation(
                comp_type="generator",
                generator_type="solar",
                geolocation=geolocation,
                voltage_level=5,
                p_nom=0.5,
                add_ts=False,
                generator_id=None,
            )
            edisgo_obj.set_time_series_worst_case_analysis()

        results = self.edisgo.reinforce_expansion_path(
            {2030: _grow_loads_by(1.0), 2035: connect_generator}
        )

        changes_2030 = results.equipment_changes.query("year == 2030")
        assert line in changes_2030.index
        # the two parts of the split line and the line to the generator
        lines = self.edisgo.topology.lines_df.index
        new_lines = lines[lines.str.startswith(f"Line_BranchTee_{line}_")]
        assert len(new_lines) == 3
        changes_2035 = results.equipment_changes.query("year == 2035")
        assert set(new_lines) == set(
            changes_2035[changes_2035.iteration_step == 0].index
        )
        costs_2035 = results.grid_expansion_costs.query("year == 2035")
        assert new_lines.isin(costs_2035.index).all()

    def test_stage_without_time_series(self):
        def add_load_without_time_series(edisgo_obj):
            edisgo_obj.add_component(
                "load",
                bus=edisgo_obj.topology.loads_df.bus.iloc[0],
                p_set=0.01,
                type="heat_pump",
            )

        with pytest.raises(ValueError, match="2030.*no time series"):
            reinforce_expansion_path(self.edisgo, {2030: add_load_without_time_series})

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="At least one stage"):
            reinforce_expansion_path(self.edisgo, {})
        with pytest.raises(TypeError, match="not callable"):
            reinforce_expansion_path(self.edisgo, {2030: None})
        with pytest.raises(ValueError, match="base_year"):
            reinforce_expansion_path(self.edisgo, self.stages, base_year=2031)
        with pytest.raises(ValueError, match="copy_grid"):
            reinforce_expansion_path(
                self.edisgo, self.stages, reinforce_kwargs={"copy_grid": True}
            )
