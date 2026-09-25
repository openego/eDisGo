import copy

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from edisgo import EDisGo
from edisgo.flex_opt.costs import grid_expansion_costs, line_expansion_costs
from edisgo.flex_opt.reinforce_grid import reinforce_grid, run_separate_lv_grids
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
    def test_reinforce_log_earthworks_charged_once_per_line(
        self, network_2_reinforced
    ):
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
