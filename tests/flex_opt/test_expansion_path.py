import copy

import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints
from edisgo.flex_opt.expansion_path import reinforce_expansion_path


def _grow_loads_to(p_set_original, factor):
    """Stage function that sets the loads to `factor` times their original size."""

    def apply_stage(edisgo_obj):
        edisgo_obj.topology.loads_df["p_set"] = p_set_original * factor
        edisgo_obj.set_time_series_worst_case_analysis()

    return apply_stage


class TestReinforceExpansionPath:
    @classmethod
    def setup_class(cls):
        cls.edisgo_original = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        cls.edisgo_original.set_time_series_worst_case_analysis()
        p_set = cls.edisgo_original.topology.loads_df.p_set.copy()
        cls.stages = {
            2035: _grow_loads_to(p_set, 1.5),
            2030: _grow_loads_to(p_set, 1.0),
            2040: _grow_loads_to(p_set, 2.0),
        }

    def setup_method(self):
        self.edisgo = copy.deepcopy(self.edisgo_original)

    def test_expansion_path(self):
        results = reinforce_expansion_path(
            self.edisgo, self.stages, discount_rate=0.1, greenfield=True
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
            "greenfield_costs",
        ]
        # every stage needs reinforcement in this grid
        assert (summary.costs > 0).all()
        pd.testing.assert_series_equal(
            summary.costs, summary.costs_mv + summary.costs_lv, check_names=False
        )
        pd.testing.assert_series_equal(
            summary.costs,
            results.grid_expansion_costs.groupby("year").total_costs.sum(),
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
        assert summary.at[2030, "greenfield_costs"] == pytest.approx(
            summary.at[2030, "costs"]
        )

        # equipment changes of all stages are in the EDisGo object
        assert len(results.equipment_changes) == len(
            self.edisgo.results.equipment_changes
        )
        assert set(results.equipment_changes.year) == {2030, 2035, 2040}

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
        p_set = self.edisgo.topology.loads_df.p_set.copy()
        results = self.edisgo.reinforce_expansion_path(
            {
                2030: _grow_loads_to(p_set, 1.5),
                2035: _grow_loads_to(p_set, 1.5),
            },
            base_year=2025,
        )
        assert results.summary.at[2030, "costs"] > 0
        assert results.summary.at[2035, "costs"] == 0
        assert results.summary.at[2030, "present_value"] == pytest.approx(
            results.summary.at[2030, "costs"] / 1.05**5
        )
        assert "greenfield_costs" not in results.summary.columns
        assert (results.equipment_changes.year == 2030).all()

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
