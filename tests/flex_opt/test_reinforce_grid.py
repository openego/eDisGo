import copy

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.flex_opt.reinforce_grid import reinforce_grid, run_separate_lv_grids
from edisgo.tools import tools


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

    def test_reinforce_grid_enable_ront_idempotent_across_iterations(self):
        # A grid that keeps reappearing in crit_nodes across several
        # while-iterations (simulated via monkeypatching
        # check_tech_constraints.voltage_issues with a call counter) must
        # get RONT installed exactly once -- not once per iteration. This
        # is the idempotency guard 'not tools.is_ront(...)' added to the
        # enable_ront branch in reinforce_grid() (see CONCEPT_ront.md).
        edisgo = copy.deepcopy(self.edisgo)
        edisgo.analyze()

        lv_grid_1 = edisgo.topology.get_lv_grid(1)
        transformer_name = lv_grid_1.transformers_df.index[0]
        original_type_info = edisgo.topology.transformers_df.at[
            transformer_name, "type_info"
        ]
        timestep = edisgo.timeseries.timeindex[0]

        crit_nodes_df = pd.DataFrame(
            {
                "abs_max_voltage_dev": [0.05],
                "time_index": [timestep],
                "lv_grid_id": [1],
            },
            index=["Bus_BranchTee_LVGrid_1_1"],
        )

        call_state = {"count": 0}
        real_voltage_issues = check_tech_constraints.voltage_issues
        real_lv_grid_ront_feasible = check_tech_constraints.lv_grid_ront_feasible

        def fake_voltage_issues(edisgo_obj, voltage_level, **kwargs):
            if voltage_level == "lv":
                call_state["count"] += 1
                # keep reporting the same grid as critical for the first
                # two calls, forcing a second while-iteration to revisit it
                if call_state["count"] <= 2:
                    return crit_nodes_df.copy()
                return pd.DataFrame(dtype=float)
            return real_voltage_issues(edisgo_obj, voltage_level=voltage_level, **kwargs)

        def fake_feasible(edisgo_obj, lv_grid, ront_voltage_range):
            return True  # always feasible -> RONT triggers

        check_tech_constraints.voltage_issues = fake_voltage_issues
        check_tech_constraints.lv_grid_ront_feasible = fake_feasible
        try:
            result = reinforce_grid(edisgo, mode="lv", enable_ront=True)
        finally:
            check_tech_constraints.voltage_issues = real_voltage_issues
            check_tech_constraints.lv_grid_ront_feasible = real_lv_grid_ront_feasible

        # sanity check that the test actually forced multiple iterations
        assert call_state["count"] >= 3

        assert edisgo.topology.transformers_df.at[
            transformer_name, "type_info"
        ] == tools.ront_type_name(original_type_info)

        changed_rows = result.equipment_changes[
            (result.equipment_changes.index == str(lv_grid_1))
            & (result.equipment_changes.change == "changed")
            & (result.equipment_changes.equipment == transformer_name)
        ]
        assert len(changed_rows) == 1

    def test_reinforce_grid_enable_ront_infeasible_falls_back(self):
        # A grid for which lv_grid_ront_feasible() reports infeasible (e.g.
        # spread too large, or v_unreg too far for the control range -- see
        # CONCEPT_ront.md, "Befund 1") must NOT get RONT -- the existing
        # line-based voltage kaskade must run unchanged, bit-identical to
        # enable_ront=False.
        edisgo_ront = copy.deepcopy(self.edisgo)
        edisgo_baseline = copy.deepcopy(self.edisgo)
        edisgo_ront.analyze()
        edisgo_baseline.analyze()

        timestep = edisgo_ront.timeseries.timeindex[0]
        crit_nodes_df = pd.DataFrame(
            {
                "abs_max_voltage_dev": [0.08],
                "time_index": [timestep],
                "lv_grid_id": [1],
            },
            index=["Bus_BranchTee_LVGrid_1_1"],
        )

        real_voltage_issues = check_tech_constraints.voltage_issues
        real_lv_grid_ront_feasible = check_tech_constraints.lv_grid_ront_feasible

        def make_fake_voltage_issues():
            state = {"count": 0}

            def fake(edisgo_obj, voltage_level, **kwargs):
                if voltage_level == "lv":
                    state["count"] += 1
                    if state["count"] == 1:
                        return crit_nodes_df.copy()
                return real_voltage_issues(
                    edisgo_obj, voltage_level=voltage_level, **kwargs
                )

            return fake

        def fake_infeasible(edisgo_obj, lv_grid, ront_voltage_range):
            return False

        check_tech_constraints.voltage_issues = make_fake_voltage_issues()
        check_tech_constraints.lv_grid_ront_feasible = fake_infeasible
        try:
            result_ront = reinforce_grid(edisgo_ront, mode="lv", enable_ront=True)
        finally:
            check_tech_constraints.voltage_issues = real_voltage_issues
            check_tech_constraints.lv_grid_ront_feasible = real_lv_grid_ront_feasible

        check_tech_constraints.voltage_issues = make_fake_voltage_issues()
        try:
            result_baseline = reinforce_grid(
                edisgo_baseline, mode="lv", enable_ront=False
            )
        finally:
            check_tech_constraints.voltage_issues = real_voltage_issues

        transformer_name = edisgo_ront.topology.get_lv_grid(1).transformers_df.index[0]
        assert not tools.is_ront(
            edisgo_ront.topology.transformers_df.at[transformer_name, "type_info"]
        )
        assert_frame_equal(
            edisgo_ront.topology.lines_df.sort_index(),
            edisgo_baseline.topology.lines_df.sort_index(),
        )
        assert_frame_equal(
            result_ront.equipment_changes.sort_index(),
            result_baseline.equipment_changes.sort_index(),
        )

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
