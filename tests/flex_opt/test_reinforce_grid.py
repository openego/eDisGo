import copy

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from edisgo import EDisGo
from edisgo.flex_opt import reinforce_measures
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.flex_opt.reinforce_grid import (
    _analyze_after_voltage_reinforcement,
    _split_lines,
    reinforce_grid,
    run_separate_lv_grids,
)


class TestReinforceGrid:
    """
    Here, currently only reinforce_grid function is tested.
    Other functions in reinforce_grid module are currently tested in test_edisgo module.
    """

    @classmethod
    def setup_class(cls):
        cls.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

        cls.edisgo.set_time_series_worst_case_analysis()

    def test_reinforce_grid(self):
        modes = [None, "mv", "mvlv", "lv"]

        results_dict = {
            mode: reinforce_grid(edisgo=copy.deepcopy(self.edisgo), mode=mode)
            for mode in modes
        }

        for mode, result in results_dict.items():
            if mode is None:
                target = ["mv/lv", "mv", "lv"]
            elif mode == "mv":
                target = ["mv"]
            elif mode == "mvlv":
                target = ["mv", "mv/lv"]
            elif mode == "lv":
                target = ["mv/lv", "lv"]
            else:
                raise ValueError("Non existing mode")

            assert_array_equal(
                np.sort(target),
                np.sort(result.grid_expansion_costs.voltage_level.unique()),
            )

            for comparison_mode, comparison_result in results_dict.items():
                if mode != comparison_mode:
                    with pytest.raises(AssertionError):
                        assert_frame_equal(
                            result.equipment_changes,
                            comparison_result.equipment_changes,
                        )
        # test reduced analysis
        res_reduced = reinforce_grid(
            edisgo=copy.deepcopy(self.edisgo),
            reduced_analysis=True,
            num_steps_loading=2,
        )
        assert len(res_reduced.i_res) == 2

    def test_analyze_after_voltage_reinforcement(self, monkeypatch):
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.analyze()

        # split three MV feeders (see test_reinforce_lines_voltage_issues in
        # test_reinforce_measures), Line_10024 is reinforced on the path
        crit_nodes = pd.DataFrame(
            {
                "abs_max_voltage_dev": [0.08, 0.06, 0.05, 0.04],
                "time_index": [edisgo.timeseries.timeindex[0]] * 4,
            },
            index=[
                "Bus_BranchTee_MVGrid_1_10",
                "Bus_BranchTee_MVGrid_1_11",
                "Bus_BranchTee_MVGrid_1_2",
                "BusBar_MVGrid_1_LVGrid_7_MV",
            ],
        )
        buses_before = edisgo.topology.lines_df[["bus0", "bus1"]].copy()
        lines_changes = reinforce_measures.reinforce_lines_voltage_issues(
            edisgo, edisgo.topology.mv_grid, crit_nodes
        )
        lines_changes["Line_10024"] = 1
        split_lines = _split_lines(edisgo, buses_before, lines_changes)
        assert sorted(split_lines) == ["Line_10007", "Line_10023", "Line_10028"]

        # power flow does not converge after the split, but after adding a
        # parallel line to the split lines
        calls = []

        def analyze(**kwargs):
            calls.append(kwargs)
            timeindex = edisgo.timeseries.timeindex
            if len(calls) == 1:
                if kwargs.get("raise_not_converged", True):
                    raise ValueError("Power flow analysis did not converge")
                return timeindex[1:], timeindex[:1]
            return timeindex, timeindex[:0]

        monkeypatch.setattr(edisgo, "analyze", analyze)
        _analyze_after_voltage_reinforcement(edisgo, split_lines, 3, mode="mv")

        assert len(calls) == 2
        assert calls[0] == {"raise_not_converged": False, "mode": "mv"}
        assert calls[1] == {"raise_not_converged": False, "mode": "mv"}
        assert (edisgo.topology.lines_df.loc[split_lines, "num_parallel"] == 2).all()
        assert edisgo.topology.lines_df.at["Line_10024", "num_parallel"] == 1
        changes = edisgo.results.equipment_changes.loc[split_lines]
        assert (changes.iteration_step == 3).all()
        assert (changes.quantity == 1).all()

        # in LV, only split lines in LV grids that do not converge on their own
        # get a parallel line
        lines_lv_1 = edisgo.topology.get_lv_grid(1).lines_df.index[:2].tolist()
        lines_lv_5 = edisgo.topology.get_lv_grid(5).lines_df.index[:1].tolist()
        num_parallel_before = edisgo.topology.lines_df.loc[
            lines_lv_1 + lines_lv_5, "num_parallel"
        ]
        calls.clear()

        def analyze_lv(**kwargs):
            calls.append(kwargs)
            timeindex = edisgo.timeseries.timeindex
            if len(calls) == 1 or kwargs.get("lv_grid_id") == 5:
                return timeindex[1:], timeindex[:1]
            return timeindex, timeindex[:0]

        monkeypatch.setattr(edisgo, "analyze", analyze_lv)
        _analyze_after_voltage_reinforcement(edisgo, lines_lv_1 + lines_lv_5, 4)

        assert len(calls) == 4
        assert sorted(c["lv_grid_id"] for c in calls[1:3]) == [1, 5]
        assert all(c["mode"] == "lv" for c in calls[1:3])
        assert all(
            c["timesteps"].equals(edisgo.timeseries.timeindex[:1]) for c in calls[1:3]
        )
        assert calls[3] == {"raise_not_converged": False}
        num_parallel = edisgo.topology.lines_df.loc[
            lines_lv_1 + lines_lv_5, "num_parallel"
        ]
        assert (num_parallel[lines_lv_1] == num_parallel_before[lines_lv_1]).all()
        assert (num_parallel[lines_lv_5] == num_parallel_before[lines_lv_5] + 1).all()
        assert (edisgo.results.equipment_changes.iteration_step == 4).sum() == len(
            lines_lv_5
        )

        # at most four parallel lines are added, then non-convergence is raised
        calls.clear()

        def analyze_not_converged(**kwargs):
            calls.append(kwargs)
            if kwargs.get("raise_not_converged", True):
                raise ValueError("Power flow analysis did not converge")
            timeindex = edisgo.timeseries.timeindex
            return timeindex[1:], timeindex[:1]

        monkeypatch.setattr(edisgo, "analyze", analyze_not_converged)
        with pytest.raises(ValueError, match="did not converge"):
            _analyze_after_voltage_reinforcement(edisgo, lines_lv_5, 5)
        # full grid and LV grid 5 four times each, plus the last full grid
        assert len(calls) == 9
        assert edisgo.topology.lines_df.loc[lines_lv_5, "num_parallel"].equals(
            num_parallel_before[lines_lv_5] + 5
        )
        assert (edisgo.results.equipment_changes.iteration_step == 5).sum() == 4 * len(
            lines_lv_5
        )

        # without split lines, non-convergence is raised right away
        monkeypatch.setattr(edisgo, "analyze", analyze)
        calls.clear()
        with pytest.raises(ValueError, match="did not converge"):
            _analyze_after_voltage_reinforcement(edisgo, [], 4, mode="mv")
        assert calls == [{"raise_not_converged": True, "mode": "mv"}]

    def test_run_separate_lv_grids(self):
        edisgo = copy.deepcopy(self.edisgo)

        edisgo.timeseries.scale_timeseries(p_scaling_factor=5, q_scaling_factor=5)

        lv_grids = [copy.deepcopy(g) for g in edisgo.topology.mv_grid.lv_grids]

        run_separate_lv_grids(edisgo)

        edisgo.results.grid_expansion_costs = grid_expansion_costs(edisgo)
        lv_grids_new = list(edisgo.topology.mv_grid.lv_grids)

        # check that no new lv grid only consist of the station
        for g in lv_grids_new:
            if g.id != 0:
                assert len(g.buses_df) > 1

        assert len(lv_grids_new) == 26
        assert np.isclose(edisgo.results.grid_expansion_costs.total_costs.sum(), 440.06)

        # check if all generators are still present
        assert np.isclose(
            sum(g.generators_df.p_nom.sum() for g in lv_grids),
            sum(g.generators_df.p_nom.sum() for g in lv_grids_new),
        )

        # check if all loads are still present
        assert np.isclose(
            sum(g.loads_df.p_set.sum() for g in lv_grids),
            sum(g.loads_df.p_set.sum() for g in lv_grids_new),
        )

        # check if all storages are still present
        assert np.isclose(
            sum(g.storage_units_df.p_nom.sum() for g in lv_grids),
            sum(g.storage_units_df.p_nom.sum() for g in lv_grids_new),
        )

        # test if power flow works
        edisgo.set_time_series_worst_case_analysis()
        edisgo.analyze()
