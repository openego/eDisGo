import copy

import numpy as np
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from edisgo import EDisGo
from edisgo.flex_opt import reinforce_measures
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.flex_opt.reinforce_grid import (
    _analyze_after_voltage_reinforcement,
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

    @pytest.mark.parametrize("choose_cheaper", [False, True])
    def test_reinforce_choose_cheaper_voltage_measure(
        self, monkeypatch, choose_cheaper
    ):
        calls = []
        reinforce_lines_voltage_issues = (
            reinforce_measures.reinforce_lines_voltage_issues
        )

        def record(edisgo_obj, grid, crit_nodes, choose_cheaper_measure=False):
            calls.append(choose_cheaper_measure)
            return reinforce_lines_voltage_issues(
                edisgo_obj, grid, crit_nodes, choose_cheaper_measure
            )

        monkeypatch.setattr(
            reinforce_measures, "reinforce_lines_voltage_issues", record
        )
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.timeseries.loads_active_power *= 5
        edisgo.reinforce(choose_cheaper_voltage_measure=choose_cheaper)
        assert calls
        assert set(calls) == {choose_cheaper}

    @staticmethod
    def _split(edisgo, grid, lines):
        """Connect `lines` to the station of `grid`, as splitting a feeder does."""
        buses_before = edisgo.topology.lines_df[["bus0", "bus1"]].copy()
        station = grid.transformers_df.bus1.iloc[0]
        edisgo.topology._lines_df.loc[lines, "bus0"] = station
        return buses_before

    @staticmethod
    def _fake_analyze(edisgo, calls, rounds_diverging, lv_grids_diverging=(5,)):
        """
        Fake of EDisGo.analyze that records its calls. The power flow diverges in the
        first `rounds_diverging` calls for the whole grid and always for the LV grids
        in `lv_grids_diverging` on their own.
        """

        def analyze(**kwargs):
            calls.append(kwargs)
            if kwargs.get("mode") == "lv":
                diverges = kwargs["lv_grid_id"] in lv_grids_diverging
            else:
                diverges = sum(c.get("mode") != "lv" for c in calls) <= rounds_diverging
            timeindex = edisgo.timeseries.timeindex
            if not diverges:
                return timeindex, timeindex[:0]
            if kwargs.get("raise_not_converged", True):
                raise ValueError("Power flow analysis did not converge")
            return timeindex[1:], timeindex[:1]

        return analyze

    def test_analyze_after_voltage_reinforcement_without_split(self, monkeypatch):
        edisgo = copy.deepcopy(self.edisgo)
        buses_before = edisgo.topology.lines_df[["bus0", "bus1"]].copy()
        calls = []
        monkeypatch.setattr(edisgo, "analyze", self._fake_analyze(edisgo, calls, 1))

        # without split lines, non-convergence is raised right away
        with pytest.raises(ValueError, match="did not converge"):
            _analyze_after_voltage_reinforcement(edisgo, buses_before, 1, mode="mv")
        assert calls == [{"raise_not_converged": True, "mode": "mv"}]

    def test_analyze_after_voltage_reinforcement_mv(self, monkeypatch):
        edisgo = copy.deepcopy(self.edisgo)
        grid = edisgo.topology.mv_grid
        lines = ["Line_10007", "Line_10023"]
        buses_before = self._split(edisgo, grid, lines)
        calls = []
        monkeypatch.setattr(edisgo, "analyze", self._fake_analyze(edisgo, calls, 1))

        _analyze_after_voltage_reinforcement(edisgo, buses_before, 3, mode="mv")

        # all split lines are in one grid, so no grid is analysed on its own
        assert calls == [{"raise_not_converged": False, "mode": "mv"}] * 2
        assert (edisgo.topology.lines_df.loc[lines, "num_parallel"] == 2).all()
        changes = edisgo.results.equipment_changes
        assert sorted(changes.index) == lines
        assert (changes.iteration_step == 3).all() and (changes.quantity == 1).all()

    @pytest.mark.parametrize(
        "lv_grids_diverging, lines_with_parallel_line",
        [((5,), ["lv_5"]), ((), ["lv_1", "lv_5"])],
    )
    def test_analyze_after_voltage_reinforcement_lv(
        self, monkeypatch, lv_grids_diverging, lines_with_parallel_line
    ):
        edisgo = copy.deepcopy(self.edisgo)
        lines = {
            "lv_1": edisgo.topology.get_lv_grid(1).lines_df.index[1:3].tolist(),
            "lv_5": edisgo.topology.get_lv_grid(5).lines_df.index[1:2].tolist(),
        }
        buses_before = self._split(
            edisgo, edisgo.topology.get_lv_grid(1), lines["lv_1"]
        )
        self._split(edisgo, edisgo.topology.get_lv_grid(5), lines["lv_5"])
        num_parallel_before = edisgo.topology.lines_df.num_parallel.copy()
        calls = []
        monkeypatch.setattr(
            edisgo,
            "analyze",
            self._fake_analyze(edisgo, calls, 1, lv_grids_diverging),
        )

        _analyze_after_voltage_reinforcement(edisgo, buses_before, 4)

        # LV grids 1 and 5 are analysed on their own, for the diverging time step
        assert sorted(c["lv_grid_id"] for c in calls[1:3]) == [1, 5]
        assert all(
            c["timesteps"].equals(edisgo.timeseries.timeindex[:1]) for c in calls[1:3]
        )
        # if every LV grid converges on its own, all split lines get a parallel line
        expected = [line for key in lines_with_parallel_line for line in lines[key]]
        added = edisgo.topology.lines_df.num_parallel - num_parallel_before
        assert sorted(added[added > 0].index) == sorted(expected)
        assert sorted(edisgo.results.equipment_changes.index) == sorted(expected)

    def test_analyze_after_voltage_reinforcement_max_rounds(self, monkeypatch):
        edisgo = copy.deepcopy(self.edisgo)
        grid = edisgo.topology.get_lv_grid(5)
        line = grid.lines_df.index[1]
        buses_before = self._split(edisgo, grid, [line])
        num_parallel_before = edisgo.topology.lines_df.at[line, "num_parallel"]
        calls = []
        monkeypatch.setattr(edisgo, "analyze", self._fake_analyze(edisgo, calls, 99))

        with pytest.raises(ValueError, match="did not converge"):
            _analyze_after_voltage_reinforcement(edisgo, buses_before, 5)

        # first power flow and four rounds, the last one raises
        assert len(calls) == 5
        assert calls[-1] == {"raise_not_converged": True}
        num_parallel = edisgo.topology.lines_df.at[line, "num_parallel"]
        assert num_parallel == num_parallel_before + 4
        assert len(edisgo.results.equipment_changes) == 4

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
        # includes the parallel standard lines to the new stations
        assert np.isclose(edisgo.results.grid_expansion_costs.total_costs.sum(), 444.26)

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
