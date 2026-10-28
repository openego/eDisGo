import copy

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from edisgo import EDisGo
from edisgo.flex_opt import reinforce_measures
from edisgo.flex_opt.costs import grid_expansion_costs, line_expansion_costs
from edisgo.flex_opt.reinforce_grid import (
    _add_lines_changes_to_equipment_changes,
    _add_reinforcement_log_entries,
    _analyze_after_voltage_reinforcement,
    _separate_lv_grid_not_converged,
    catch_convergence_reinforce_grid,
    reinforce_grid,
    run_separate_lv_grids,
)
from edisgo.network.results import Results


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

    def test_reinforce_log_costs_match_grid_expansion_costs(self):
        # verify Step 5: reinforce_log's per-measure costs add up to the same
        # total as grid_expansion_costs, which is the established, separately
        # tested source of truth for total network expansion costs (see
        # e.g. test_run_separate_lv_grids using
        # grid_expansion_costs.total_costs.sum())
        edisgo = copy.deepcopy(self.edisgo)
        results = reinforce_grid(edisgo=edisgo)

        reinforce_log_total = results.reinforce_log["costs"].sum()
        grid_expansion_total = results.grid_expansion_costs["total_costs"].sum()

        # both sums are built from the same line_expansion_costs() /
        # transformer_expansion_costs() cost tables, just aggregated
        # differently (reinforce_log: once per (violation, changed
        # component) row; grid_expansion_costs: once per changed component
        # across the whole equipment_changes history), so a tight
        # floating-point tolerance covers summation-order effects only
        assert reinforce_log_total == pytest.approx(grid_expansion_total)

    def test_reinforce_log_to_csv_from_csv_roundtrip(self, tmp_path):
        # verify Step 5: reinforce_log persists and restores correctly via
        # Results.to_csv / Results.from_csv
        edisgo = copy.deepcopy(self.edisgo)
        results = reinforce_grid(edisgo=edisgo)

        original_reinforce_log = results.reinforce_log.copy()
        assert not original_reinforce_log.empty

        results.to_csv(str(tmp_path))

        fresh_results = Results(edisgo)
        fresh_results.from_csv(str(tmp_path))
        loaded_reinforce_log = fresh_results.reinforce_log.copy()

        # read_csv's parse_dates=True (used generically by Results.from_csv
        # for all grid expansion results) only parses the index column, not
        # regular data columns, so the 'time_index' column round-trips as a
        # string instead of a Timestamp; this is a generic limitation that
        # would equally affect any other Results attribute with a Timestamp
        # *column* (e.g. unresolved_issues), not something specific to
        # reinforce_log, so it is fixed up here rather than in the property
        loaded_reinforce_log["time_index"] = pd.to_datetime(
            loaded_reinforce_log["time_index"]
        )
        # for the same generic reason, an all-None 'lv_grid_id' column
        # (as here, since none of the triggered measures are voltage issues
        # at a LV bus/station) round-trips as NaN/float64 instead of
        # None/object, since CSV cannot distinguish "no value" from "no
        # value of this dtype"
        loaded_reinforce_log["lv_grid_id"] = (
            loaded_reinforce_log["lv_grid_id"]
            .astype(object)
            .where(loaded_reinforce_log["lv_grid_id"].notna(), None)
        )

        assert_frame_equal(loaded_reinforce_log, original_reinforce_log)

    @pytest.fixture(scope="class")
    @classmethod
    def network_2_reinforced(cls):
        # shared across all tests that need a scenario with voltage issues
        # and lines reinforced across multiple iterations, neither of which
        # ding0_test_network_1's worst-case scenario (used in setup_class)
        # happens to produce; computed once and reused (read-only) since
        # reinforce_grid() takes ~90s on this larger network
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        edisgo.set_time_series_worst_case_analysis()
        edisgo.analyze()
        results = edisgo.reinforce()
        return edisgo, results

    @pytest.mark.slow
    def test_reinforce_log_earthworks_charged_once_per_line(self, network_2_reinforced):
        # regression test: a line reinforced across several iterations (e.g.
        # repeated parallel-line additions to resolve a persistent voltage
        # issue) must only be charged earthworks once, not once per
        # iteration - the trench is only dug once, no matter how many
        # parallel cables end up in it (see costs.line_expansion_costs()).
        edisgo, results = network_2_reinforced

        log = results.reinforce_log
        line_rows = log[log["changed_component"].isin(edisgo.topology.lines_df.index)]
        counts = line_rows["changed_component"].value_counts()
        repeated_lines = counts[counts > 1].index

        # sanity check that this test actually exercises the scenario it is
        # meant to guard, and isn't silently vacuous
        assert len(repeated_lines) > 0

        checked_at_least_one_line = False
        for line in repeated_lines:
            rows = line_rows[line_rows["changed_component"] == line]
            # skip lines whose type was changed to something new between
            # calls - accumulating cost for those has to additionally
            # account for a separate, pre-existing quirk in
            # grid_expansion_costs() (it only counts quantity added under
            # the line's *final* type, silently dropping cost attributed to
            # a since-superseded type), which is out of scope here
            if rows["equipment"].nunique() > 1:
                continue
            checked_at_least_one_line = True

            total_quantity = rows["quantity"].sum()
            costs = line_expansion_costs(edisgo, [line])
            expected_total_costs = (
                costs.at[line, "costs_earthworks"]
                + costs.at[line, "costs_cable"] * total_quantity
            )
            assert rows["costs"].sum() == pytest.approx(expected_total_costs)

        assert checked_at_least_one_line

    @pytest.mark.slow
    def test_reinforce_log_voltage_trigger_reached_and_consistent(
        self, network_2_reinforced
    ):
        # coverage gap found during an audit: none of the other
        # reinforce_log tests exercise trigger == "voltage" at all
        # (ding0_test_network_1's worst-case scenario only produces
        # overloading violations), and "mv_lv_voltage" specifically (the
        # station-voltage branch fixed for Bug D, resolved via
        # reinforce_mv_lv_station_voltage_issues()) was reachable by no test
        # at all. ding0_test_network_2 naturally exercises both.
        _, results = network_2_reinforced

        voltage_rows = results.reinforce_log[
            results.reinforce_log["trigger"] == "voltage"
        ]
        assert not voltage_rows.empty

        # all three voltage-triggered stages must actually be reached
        assert {"mv_voltage", "mv_lv_voltage", "lv_voltage"}.issubset(
            set(voltage_rows["reinforcement_stage"])
        )

        # value/limit must be real, plausible p.u. voltages (unlike station
        # overloading, voltage issues always have a derivable limit)
        assert voltage_rows["value"].between(0.5, 1.5).all()
        assert voltage_rows["limit"].between(0.5, 1.5).all()
        assert voltage_rows["limit"].notna().all()

        # issue_type must be consistent with which of value/limit is larger;
        # this scenario's worst case only produces undervoltage (heavily
        # loaded feeders, not enough feed-in for overvoltage), so only that
        # direction is asserted as actually occurring
        undervoltage = voltage_rows[voltage_rows["issue_type"] == "undervoltage"]
        overvoltage = voltage_rows[voltage_rows["issue_type"] == "overvoltage"]
        assert not undervoltage.empty
        assert (undervoltage["value"] < undervoltage["limit"]).all()
        assert (overvoltage["value"] > overvoltage["limit"]).all()

    @pytest.mark.parametrize("choose_cheaper", [False, True])
    def test_reinforce_choose_cheaper_voltage_measure(
        self, monkeypatch, choose_cheaper
    ):
        calls = []
        reinforce_lines_voltage_issues = (
            reinforce_measures.reinforce_lines_voltage_issues
        )

        def record(
            edisgo_obj, grid, crit_nodes, choose_cheaper_measure=False, **kwargs
        ):
            calls.append(choose_cheaper_measure)
            return reinforce_lines_voltage_issues(
                edisgo_obj, grid, crit_nodes, choose_cheaper_measure, **kwargs
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

    def test_catch_convergence_reinforce_grid_keeps_mode(self, monkeypatch):
        # the first reinforcement does not converge, the time steps that converge
        # and those that do not are then reinforced separately; all three runs
        # must keep mode "lv", although the grid is analyzed in mode None
        edisgo = copy.deepcopy(self.edisgo)
        timesteps = edisgo.timeseries.timeindex
        modes = []

        def fake_reinforce_grid(edisgo_obj, **kwargs):
            modes.append(kwargs["mode"])
            if len(modes) == 1:
                raise ValueError("Power flow analysis did not converge for the")

        analyze_modes = []

        def fake_analyze(**kwargs):
            analyze_modes.append(kwargs["mode"])
            return timesteps[:2], timesteps[2:]

        monkeypatch.setattr(
            "edisgo.flex_opt.reinforce_grid.reinforce_grid", fake_reinforce_grid
        )
        monkeypatch.setattr(edisgo, "analyze", fake_analyze)
        catch_convergence_reinforce_grid(edisgo, mode="lv")
        assert modes == ["lv", "lv", "lv"]
        assert analyze_modes == [None]

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

    def test_analyze_after_voltage_reinforcement_reinforce_log(self, monkeypatch):
        edisgo = copy.deepcopy(self.edisgo)
        grid = edisgo.topology.mv_grid
        lines = ["Line_10007", "Line_10023"]
        buses_before = self._split(edisgo, grid, lines)
        calls = []
        monkeypatch.setattr(edisgo, "analyze", self._fake_analyze(edisgo, calls, 1))

        # without a run_id, the parallel lines are not logged
        _analyze_after_voltage_reinforcement(edisgo, buses_before, 3, mode="mv")
        assert edisgo.results.reinforce_log.empty

        edisgo = copy.deepcopy(self.edisgo)
        buses_before = self._split(edisgo, grid, lines)
        calls = []
        monkeypatch.setattr(edisgo, "analyze", self._fake_analyze(edisgo, calls, 1))
        _analyze_after_voltage_reinforcement(
            edisgo, buses_before, 3, stage="mv_voltage", run_id=0, mode="mv"
        )

        log = edisgo.results.reinforce_log
        assert sorted(log.changed_component) == lines
        assert sorted(log.criterion_component) == lines
        assert (log.reinforcement_stage == "mv_voltage_not_converged").all()
        assert (log.trigger == "non_convergence").all()
        # the fake power flow does not converge for one time step
        assert (log.value == 1).all()
        assert (log.time_index == edisgo.timeseries.timeindex[0]).all()
        assert log.costs.sum() == pytest.approx(
            grid_expansion_costs(edisgo).total_costs.sum()
        )

    def test_reinforce_log_one_info_line_per_violation(self, caplog):
        # a voltage issue solved by reinforcing several lines on the path to the
        # critical node gives one row per line, but one INFO line
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.analyze()
        node = edisgo.topology.mv_grid.buses_df.index[-1]
        lines = edisgo.topology.mv_grid.lines_df.index[:2].tolist()
        time_step = edisgo.timeseries.timeindex[0]
        issues = pd.DataFrame(
            {
                "abs_max_voltage_dev": [0.01],
                "max_voltage_dev": [-0.01],
                "time_index": [time_step],
                "lv_grid_id": [None],
            },
            index=[node],
        )

        with caplog.at_level("INFO", logger="edisgo.flex_opt.reinforce_grid"):
            _add_reinforcement_log_entries(
                edisgo,
                1,
                "mv_voltage",
                "voltage",
                issues,
                dict.fromkeys(lines, 1),
                run_id=0,
                mapping=dict.fromkeys(lines, node),
            )

        log = edisgo.results.reinforce_log
        assert sorted(log.changed_component) == sorted(lines)
        assert (log.criterion_component == node).all()
        info_lines = [r for r in caplog.records if r.getMessage().startswith("Iter")]
        assert len(info_lines) == 1
        assert "2 line(s)" in info_lines[0].getMessage()

    def test_reinforce_log_earthworks_once_over_runs(self):
        # a line reinforced in two calls of reinforce_grid is charged earthworks
        # only once, as in grid_expansion_costs
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.analyze()
        line = edisgo.topology.mv_grid.lines_df.index[0]
        issues = pd.DataFrame(
            {
                "max_rel_overload": [1.5],
                "time_index": [edisgo.timeseries.timeindex[0]],
            },
            index=[line],
        )
        for run_id in (0, 1):
            edisgo.topology.update_number_of_parallel_lines(
                edisgo.topology.lines_df.loc[[line], "num_parallel"] + 1
            )
            _add_lines_changes_to_equipment_changes(edisgo, {line: 1}, 1)
            _add_reinforcement_log_entries(
                edisgo,
                1,
                "mv_line_overload",
                "overloading",
                issues,
                {line: 1},
                run_id=run_id,
            )

        log = edisgo.results.reinforce_log
        costs = line_expansion_costs(edisgo, [line])
        assert log.costs.tolist() == pytest.approx(
            [
                costs.at[line, "costs_earthworks"] + costs.at[line, "costs_cable"],
                costs.at[line, "costs_cable"],
            ]
        )
        assert log.costs.sum() == pytest.approx(
            grid_expansion_costs(edisgo).total_costs.sum()
        )

    def test_separate_lv_grid_not_converged_reinforce_log(self):
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.timeseries.scale_timeseries(p_scaling_factor=5, q_scaling_factor=5)
        lv_grid = max(edisgo.topology.mv_grid.lv_grids, key=lambda g: len(g.loads_df))
        timesteps = edisgo.timeseries.timeindex[:2]

        _separate_lv_grid_not_converged(edisgo, lv_grid, timesteps, True)

        log = edisgo.results.reinforce_log
        changes = edisgo.results.equipment_changes
        assert not changes.empty
        # every new line and transformer is logged, all caused by the station of
        # the separated grid
        new_lines = changes.index[changes.index.isin(edisgo.topology.lines_df.index)]
        new_transformers = changes.loc[changes.change == "added", "equipment"]
        assert sorted(log.changed_component) == sorted(
            list(new_lines) + list(new_transformers)
        )
        assert (log.criterion_component == f"{lv_grid}_station").all()
        assert (log.trigger == "non_convergence").all()
        assert (log.value == 2).all()
        assert (log.lv_grid_id == lv_grid.id).all()
        assert log.costs.sum() == pytest.approx(
            grid_expansion_costs(edisgo).total_costs.sum()
        )

        # with log_reinforcement=False, nothing is logged
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.timeseries.scale_timeseries(p_scaling_factor=5, q_scaling_factor=5)
        lv_grid = edisgo.topology.get_lv_grid(lv_grid.id)
        _separate_lv_grid_not_converged(edisgo, lv_grid, timesteps, False)
        assert not edisgo.results.equipment_changes.empty
        assert edisgo.results.reinforce_log.empty

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
