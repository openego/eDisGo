"""
Property-based tests for the EDisGo facade (power flow, reinforcement, persistence
and data handling methods in :mod:`edisgo.edisgo`).

Each test states the physical or domain property it checks. Fixtures and helpers
live in ``tests/properties/conftest.py``.
"""

from __future__ import annotations

import copy
import logging
import math
import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files, import_edisgo_from_pickle

from .conftest import (
    ALL_CHECKS,
    POWER_ATOL,
    TIMESERIES_ATTRIBUTES,
    assert_timeseries_equal,
    assert_topology_equal,
    bus_injections,
    edisgo_warnings,
    examples,
    limit_violations,
    max_abs_diff,
    net_injection_totals,
    power_balance,
    reinforce_checks_per_worst_case,
    run_pypsa_power_flow,
    worst_case_timesteps,
)

PF_SETTINGS = settings(
    deadline=None,
    max_examples=examples(6),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Voltage tolerance in p.u. when comparing two power flows of the same physical
#: state: Newton-Raphson stops at a mismatch of 1e-6, which moves voltages by far
#: less than 1e-8 p.u. in practice on this grid; 1e-8 still detects any change in
#: injections (1 kW in the LV shifts voltages by ~1e-5 p.u.).
V_ATOL_SAME_STATE = 1e-8


def _scale_components(edisgo, load_k, gen_k, storage_k, seed):
    """Scale P and Q of every component by a random factor around the given one."""
    rng = np.random.default_rng(seed)
    ts = edisgo.timeseries
    for comp, k in (
        ("loads", load_k),
        ("generators", gen_k),
        ("storage_units", storage_k),
    ):
        p = getattr(ts, f"{comp}_active_power")
        factors = k * rng.uniform(0.5, 1.5, size=p.shape[1])
        # same factor for P and Q keeps the power factor of every component
        setattr(ts, f"{comp}_active_power", p * factors)
        q = getattr(ts, f"{comp}_reactive_power")
        setattr(ts, f"{comp}_reactive_power", q * factors)


def _branch_flows(pypsa_network, attr):
    """Sum of signed branch flows leaving each bus (``attr`` is "p" or "q")."""
    flows = []
    for comp in ("lines", "transformers"):
        static = getattr(pypsa_network, comp)
        dynamic = getattr(pypsa_network, f"{comp}_t")
        flows.append(dynamic[f"{attr}0"].T.groupby(static.bus0).sum().T)
        flows.append(dynamic[f"{attr}1"].T.groupby(static.bus1).sum().T)
    total = pd.DataFrame(0.0, index=flows[0].index, columns=pypsa_network.buses.index)
    for f in flows:
        total = total.add(f, fill_value=0.0)
    return total


def _branch_losses(pypsa_network, attr="p"):
    """Total branch losses per time step from signed PyPSA flows."""
    losses = [
        (getattr(pypsa_network, f"{c}_t")[f"{attr}0"])
        + getattr(pypsa_network, f"{c}_t")[f"{attr}1"]
        for c in ("lines", "transformers")
    ]
    return pd.concat(losses, axis=1).sum(axis=1)


# ======================================================================================
# power flow analysis
# ======================================================================================


class TestPowerFlowPhysics:
    def test_power_balance_equals_branch_losses(self, analyzed_grid):
        """
        Energy conservation: slack + generation + storage - load equals the losses
        dissipated in lines and transformers (from signed PyPSA branch flows), and
        these losses are non-negative.
        """
        balance = power_balance(analyzed_grid)
        pypsa_network = run_pypsa_power_flow(analyzed_grid)
        branch_losses = _branch_losses(pypsa_network, "p")

        np.testing.assert_allclose(
            balance.p, branch_losses.loc[balance.index], atol=POWER_ATOL
        )
        assert (balance.p >= -POWER_ATOL).all()
        # Results.grid_losses is defined via abs(); with positive losses it must
        # equal the raw balance
        np.testing.assert_allclose(
            analyzed_grid.results.grid_losses.p, balance.p, atol=POWER_ATOL
        )

    @PF_SETTINGS
    @given(
        load_k=st.floats(0.0, 2.0),
        gen_k=st.floats(0.0, 2.0),
        storage_k=st.floats(-1.0, 1.0),
        seed=st.integers(0, 2**16),
        mode=st.sampled_from([None, "mv", "mvlv"]),
    )
    def test_losses_non_negative_for_random_injections(
        self, wc_grid_factory, load_k, gen_k, storage_k, seed, mode
    ):
        """
        A passive grid never generates active power: for arbitrary injections the
        losses implied by slack power and time series are >= 0, in the full grid
        and in the MV-only power flow modes (LV aggregated at the stations).
        """
        edisgo = wc_grid_factory()
        _scale_components(edisgo, load_k, gen_k, storage_k, seed)
        converged, _ = edisgo.analyze(mode=mode, raise_not_converged=False)
        assume(len(converged) > 0)

        balance = power_balance(edisgo).loc[converged]
        assert (balance.p >= -POWER_ATOL).all(), balance
        # losses cannot exceed the power exchanged by the slack and all components
        ts = edisgo.timeseries
        throughput = (
            ts.loads_active_power.loc[converged].abs().sum(axis=1)
            + ts.generators_active_power.loc[converged].abs().sum(axis=1)
            + ts.storage_units_active_power.loc[converged].abs().sum(axis=1)
            + edisgo.results.pfa_slack.p.loc[converged].abs()
        )
        assert (balance.p <= throughput + POWER_ATOL).all()

    def test_kirchhoff_current_law_per_bus(self, analyzed_grid):
        """
        Kirchhoff's current law at every bus: the net injection of all eDisGo
        components connected to the bus (from topology and time series, plus the
        slack at its bus) equals the sum of signed branch flows leaving the bus.
        This also verifies sign conventions and the bus mapping of ``to_pypsa``.
        """
        pypsa_network = run_pypsa_power_flow(analyzed_grid)
        slack_bus = pypsa_network.generators.at["Generator_slack", "bus"]
        p_inj, q_inj = bus_injections(analyzed_grid)
        slack = pypsa_network.generators_t
        for attr, inj in (("p", p_inj), ("q", q_inj)):
            inj = inj.reindex(columns=pypsa_network.buses.index, fill_value=0.0)
            inj[slack_bus] += slack[attr]["Generator_slack"]
            flows = _branch_flows(pypsa_network, attr)
            # NR tolerance 1e-6 p.u. per bus (base 1 MVA)
            assert max_abs_diff(inj, flows) < 1e-6, attr

    def test_apparent_power_consistent_with_current_and_voltage(self, analyzed_grid):
        """
        |S| = sqrt(3) * U * I: the current in ``i_res`` times the bus0 voltage
        gives the apparent power at bus0. ``s_res`` is the larger of both ends,
        so it is >= that value and exceeds it at most by the branch losses
        |S0 + S1| (triangle inequality).
        """
        results = analyzed_grid.results
        topo = analyzed_grid.topology
        pypsa_network = run_pypsa_power_flow(analyzed_grid)
        for comp in ("lines", "transformers"):
            df = getattr(topo, f"{comp}_df")
            v_nom = topo.buses_df.loc[df.bus0, "v_nom"].to_numpy()
            u0 = math.sqrt(3) * v_nom * results.v_res.loc[:, df.bus0].to_numpy()
            s_from_i = results.i_res.loc[:, df.index] * u0
            s_res = results.s_res.loc[:, df.index]
            dyn = getattr(pypsa_network, f"{comp}_t")
            s_loss = np.hypot(
                dyn.p0[df.index] + dyn.p1[df.index], dyn.q0[df.index] + dyn.q1[df.index]
            )
            assert ((s_res - s_from_i) >= -1e-9).all().all(), comp
            assert ((s_res - s_from_i) <= s_loss + 1e-9).all().all(), comp

    def test_load_case_voltage_drops_along_every_path(self, analyzed_grid):
        """
        Without generation (worst-case load cases) voltage can only drop from the
        slack towards the grid ends: every bus is at most at slack voltage, and
        along every path from the slack the voltage does not rise. Tolerance
        1e-4 p.u. covers the capacitive voltage rise on lightly loaded cables.
        """
        ts = analyzed_grid.timeseries
        load_steps = ts.timeindex_worst_cases[
            ts.timeindex_worst_cases.index.str.startswith("load_case")
        ]
        assert (ts.generators_active_power.loc[load_steps] == 0).all().all()
        v = analyzed_grid.results.v_res.loc[load_steps]
        slack_bus = analyzed_grid.topology.mv_grid.station.index[0]
        assert (v.sub(v[slack_bus], axis=0) <= 1e-4).all(axis=None)
        graph = analyzed_grid.to_graph()
        for parent, child in nx.bfs_edges(graph, slack_bus):
            assert (v[child] <= v[parent] + 1e-4).all(), (parent, child)

    @settings(deadline=None, max_examples=examples(5))
    @given(steps=st.sets(st.integers(0, 3), min_size=1, max_size=3))
    def test_analyze_subset_of_timesteps_matches_full_run(
        self, _analyzed_grid_original, steps
    ):
        """
        Time steps are independent in a static power flow: analysing a subset of
        time steps gives the same voltages and currents as the full run.
        """
        edisgo = copy.deepcopy(_analyzed_grid_original)
        full_v = edisgo.results.v_res.copy()
        full_i = edisgo.results.i_res.copy()
        subset = edisgo.timeseries.timeindex[sorted(steps)]
        edisgo.analyze(timesteps=subset)
        assert max_abs_diff(edisgo.results.v_res, full_v.loc[subset]) < 1e-8
        assert max_abs_diff(edisgo.results.i_res, full_i.loc[subset]) < 1e-8

    @PF_SETTINGS
    @given(k=st.floats(0.1, 1.5))
    def test_analyze_scale_timeseries_equals_scaled_time_series(
        self, wc_grid_factory, k
    ):
        """
        Metamorphic: ``analyze(scale_timeseries=k)`` gives the same power flow as
        scaling all component time series by k with ``scale_timeseries`` first.
        """
        a = wc_grid_factory()
        a.analyze(scale_timeseries=k)
        b = wc_grid_factory()
        b.timeseries.scale_timeseries(p_scaling_factor=k, q_scaling_factor=k)
        b.analyze()
        assert max_abs_diff(a.results.v_res, b.results.v_res) < V_ATOL_SAME_STATE
        assert max_abs_diff(a.results.pfa_slack, b.results.pfa_slack) < 1e-8

    def test_scale_timeseries_one_is_identity(self, analyzed_grid):
        """Scaling all time series by 1 does not change the power flow."""
        v_before = analyzed_grid.results.v_res.copy()
        analyzed_grid.analyze(scale_timeseries=1.0)
        assert max_abs_diff(analyzed_grid.results.v_res, v_before) < 1e-12

    @PF_SETTINGS
    @given(k1=st.floats(0.2, 1.4), dk=st.floats(0.1, 0.6))
    def test_losses_increase_with_uniform_scaling(self, wc_grid_factory, k1, dk):
        """
        Monotonicity: scaling all injections uniformly by a larger factor
        increases the currents and thus the active power losses in every time
        step, and lowers the minimum voltage in the load cases.
        """
        k2 = k1 + dk
        res = {}
        for k in (k1, k2):
            edisgo = wc_grid_factory()
            # scale the eDisGo time series themselves (not only the PyPSA export via
            # analyze(scale_timeseries=k)), so that power_balance sees the same
            # injections as the power flow
            edisgo.timeseries.scale_timeseries(p_scaling_factor=k, q_scaling_factor=k)
            edisgo.analyze()
            res[k] = (power_balance(edisgo).p, edisgo.results.v_res)
        assert (res[k2][0] > res[k1][0]).all()
        load_steps = (
            wc_grid_factory()
            .timeseries.timeindex_worst_cases.loc[["load_case_mv", "load_case_lv"]]
            .values
        )
        assert (
            res[k2][1].loc[load_steps].min(axis=1)
            <= res[k1][1].loc[load_steps].min(axis=1) + 1e-12
        ).all()

    def test_to_pypsa_conserves_injections_across_modes(self, wc_grid):
        """
        Exporting MV only (LV aggregated at the stations) must not create or lose
        power: per time step, the total load, generation and storage set points
        in PyPSA equal those of the eDisGo time series for every mode and
        aggregation option.
        """
        expected = net_injection_totals(wc_grid)
        variants = [
            {"mode": None},
            {"mode": "mv"},
            {"mode": "mvlv"},
            {
                "mode": "mv",
                "aggregate_loads": "sectoral",
                "aggregate_generators": "type",
                "aggregate_storages": "all",
            },
            {
                "mode": "mvlv",
                "aggregate_loads": "all",
                "aggregate_generators": "curtailable",
                "aggregate_storages": "all",
            },
        ]
        for kwargs in variants:
            pn = wc_grid.to_pypsa(**kwargs)
            gens = pn.generators_t.p_set.drop(
                columns="Generator_slack", errors="ignore"
            )
            p = (
                gens.sum(axis=1)
                + pn.storage_units_t.p_set.sum(axis=1)
                - pn.loads_t.p_set.sum(axis=1)
            )
            q = (
                pn.generators_t.q_set.drop(
                    columns="Generator_slack", errors="ignore"
                ).sum(axis=1)
                + pn.storage_units_t.q_set.sum(axis=1)
                - pn.loads_t.q_set.sum(axis=1)
            )
            np.testing.assert_allclose(p, expected.p, atol=1e-9, err_msg=str(kwargs))
            np.testing.assert_allclose(q, expected.q, atol=1e-9, err_msg=str(kwargs))


# ======================================================================================
# grid reinforcement
# ======================================================================================


class TestReinforce:
    def test_reinforcement_steps_leave_no_violation(self, reinforced_grid):
        """
        After worst-case reinforcement, each of the three reinforcement steps
        (MV on MV cases with LV aggregated, MV/LV stations and LV on LV cases)
        finds no remaining overloading or voltage violation when its power flow
        is repeated on the reinforced grid.
        """
        violations = reinforce_checks_per_worst_case(reinforced_grid)
        assert violations == {"mv": {}, "mvlv": {}, "lv": {}}
        assert reinforced_grid.results.unresolved_issues.empty

    def test_final_power_flow_within_limits_per_worst_case(self, reinforced_grid):
        """
        The full-grid power flow stored after reinforcement fulfils the MV limits
        in the MV worst cases and the LV limits (incl. MV/LV stations) in the LV
        worst cases, i.e. the reinforced grid is feasible including all voltage
        levels at once.
        """
        steps = worst_case_timesteps(reinforced_grid)
        mv = limit_violations(
            reinforced_grid,
            ("mv_line", "hv_mv_station", "mv_voltage"),
            timesteps=steps["mv"],
        )
        lv = limit_violations(
            reinforced_grid,
            ("lv_line", "mv_lv_station", "mv_lv_voltage", "lv_voltage"),
            timesteps=steps["lv"],
        )
        assert mv == {}
        assert lv == {}

    def test_original_grid_has_violations(self, analyzed_grid):
        """Sanity check of the helper: the unreinforced test grid is infeasible."""
        assert set(limit_violations(analyzed_grid, ALL_CHECKS)) >= {
            "mv_line",
            "lv_line",
            "mv_lv_station",
        }

    def test_costs_non_negative_and_consistent_with_equipment_changes(
        self, reinforced_grid
    ):
        """
        Every reinforced component is costed exactly once and costs are >= 0:
        the cost table covers the added transformers that still exist and the
        lines whose new type is recorded in ``equipment_changes``; the total
        equals the sum over the per-measure ``reinforce_log``.
        """
        results = reinforced_grid.results
        topo = reinforced_grid.topology
        costs = results.grid_expansion_costs
        changes = results.equipment_changes
        assert (costs.total_costs >= 0).all()
        assert not costs.index.duplicated().any()

        all_trafos = pd.concat([topo.transformers_df, topo.transformers_hvmv_df])
        is_trafo = changes.equipment.str.contains("ransformer")
        added = set(changes[is_trafo & (changes.change == "added")].equipment)
        removed = set(changes[is_trafo & (changes.change == "removed")].equipment)
        expected_trafos = {t for t in added - removed if t in all_trafos.index}
        line_changes = changes[changes.index.isin(topo.lines_df.index)]
        expected_lines = set(
            line_changes.index[
                line_changes.equipment.values
                == topo.lines_df.loc[line_changes.index, "type_info"].values
            ]
        )
        assert set(costs.index) == expected_trafos | expected_lines
        # lines: total costs = earthworks + cable costs * reported quantity
        assert (costs.loc[list(expected_lines), "quantity"] >= 1).all()
        assert costs.total_costs.sum() == pytest.approx(
            results.reinforce_log["costs"].sum()
        )

    def test_reinforce_is_idempotent(self, reinforced_grid):
        """
        A grid that was just reinforced for the worst cases needs no further
        measures: a second ``reinforce`` adds no equipment change, leaves the
        topology unchanged and does not change the costs.
        """
        topo_before = copy.deepcopy(reinforced_grid)
        n_changes = len(reinforced_grid.results.equipment_changes)
        costs_before = reinforced_grid.results.grid_expansion_costs.total_costs.sum()

        reinforced_grid.reinforce()

        assert len(reinforced_grid.results.equipment_changes) == n_changes
        assert_topology_equal(reinforced_grid, topo_before)
        assert reinforced_grid.results.grid_expansion_costs.total_costs.sum() == (
            pytest.approx(costs_before)
        )

    def test_reinforce_copy_grid_leaves_object_untouched(self, analyzed_grid):
        """
        ``copy_grid=True`` works on a copy: the topology and the power flow
        results of the original object stay exactly as they were, while the
        returned results contain the reinforcement.
        """
        before = copy.deepcopy(analyzed_grid)
        results = analyzed_grid.reinforce(copy_grid=True)

        assert not results.equipment_changes.empty
        assert_topology_equal(analyzed_grid, before)
        assert analyzed_grid.results.equipment_changes.empty
        pd.testing.assert_frame_equal(analyzed_grid.results.v_res, before.results.v_res)
        pd.testing.assert_frame_equal(analyzed_grid.results.i_res, before.results.i_res)

    @pytest.mark.slow
    @settings(
        deadline=None,
        max_examples=examples(4),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(k=st.floats(0.3, 2.0), seed=st.integers(0, 2**16))
    def test_reinforce_resolves_violations_for_random_injections(
        self, wc_grid_factory, k, seed
    ):
        """
        For randomly scaled worst-case injections, the reinforced grid fulfils all
        limits checked by the respective worst-case reinforcement step.
        """
        edisgo = wc_grid_factory()
        _scale_components(edisgo, k, k, k, seed)
        edisgo.reinforce()
        violations = reinforce_checks_per_worst_case(edisgo)
        assert violations == {"mv": {}, "mvlv": {}, "lv": {}}

    @pytest.mark.slow
    def test_reinforcement_costs_grow_with_load(self, wc_grid_factory):
        """
        Monotonicity: scaling all worst-case injections up never makes the
        reinforcement cheaper or the number of changed components smaller.
        """
        totals, counts = [], []
        for k in (0.5, 1.0, 1.5, 2.0):
            edisgo = wc_grid_factory()
            edisgo.timeseries.scale_timeseries(k, k)
            results = edisgo.reinforce(copy_grid=True)
            totals.append(results.grid_expansion_costs.total_costs.sum())
            counts.append(len(results.grid_expansion_costs))
        assert totals == sorted(totals), totals
        assert counts == sorted(counts), counts


# ======================================================================================
# persistence
# ======================================================================================


def _load_all(path, **kwargs):
    return import_edisgo_from_files(
        path,
        import_topology=True,
        import_timeseries=True,
        import_results=True,
        import_electromobility=True,
        import_heat_pump=True,
        import_dsm=True,
        import_overlying_grid=True,
        **kwargs,
    )


class TestPersistence:
    @pytest.mark.parametrize("archive", [False, True])
    def test_save_load_roundtrip_is_complete(self, reinforced_grid, tmp_path, archive):
        """
        ``save`` followed by ``import_edisgo_from_files`` restores topology, time
        series (incl. the worst-case labels), power flow results, reinforcement
        results and configuration; a zip archive restores the same as a directory.
        """
        directory = tmp_path / "edisgo"
        reinforced_grid.save(directory, archive=archive)
        path = str(directory) + ".zip" if archive else directory
        loaded = _load_all(path)

        assert_topology_equal(loaded, reinforced_grid, check_dtype=False)
        assert_timeseries_equal(
            loaded, reinforced_grid, check_freq=False, check_names=False
        )
        assert loaded.timeseries.is_worst_case
        pd.testing.assert_series_equal(
            loaded.timeseries.timeindex_worst_cases,
            reinforced_grid.timeseries.timeindex_worst_cases,
            check_freq=False,
            check_names=False,
        )
        res_l, res_o = loaded.results, reinforced_grid.results
        for attr in ("v_res", "i_res", "pfa_p", "pfa_q", "pfa_slack", "grid_losses"):
            pd.testing.assert_frame_equal(
                getattr(res_l, attr),
                getattr(res_o, attr),
                check_freq=False,
                check_names=False,
                obj=attr,
            )
        pd.testing.assert_frame_equal(
            res_l.equipment_changes, res_o.equipment_changes, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            res_l.grid_expansion_costs.sort_index(),
            res_o.grid_expansion_costs.sort_index(),
            check_dtype=False,
        )
        assert loaded.config._data == reinforced_grid.config._data

    def test_loaded_object_behaves_like_original(self, reinforced_grid, tmp_path):
        """
        Functional roundtrip: on the reloaded object a power flow gives the same
        voltages as on the original, and the reloaded reinforced grid needs no
        further reinforcement.
        """
        reinforced_grid.save(tmp_path / "edisgo")
        loaded = _load_all(tmp_path / "edisgo")
        reinforced_grid.analyze()
        loaded.analyze()
        assert (
            max_abs_diff(loaded.results.v_res, reinforced_grid.results.v_res)
            < V_ATOL_SAME_STATE
        )
        n_changes = len(loaded.results.equipment_changes)
        loaded.reinforce()
        assert len(loaded.results.equipment_changes) == n_changes

    def test_save_with_reduced_memory_keeps_values(self, analyzed_grid, tmp_path):
        """
        Saving with ``reduce_memory=True`` stores float32; all time series and
        results are restored within float32 precision (rtol 1e-6).
        """
        analyzed_grid.save(tmp_path / "edisgo", reduce_memory=True)
        loaded = _load_all(tmp_path / "edisgo")
        for attr in TIMESERIES_ATTRIBUTES:
            pd.testing.assert_frame_equal(
                getattr(loaded.timeseries, attr),
                getattr(analyzed_grid.timeseries, attr),
                check_dtype=False,
                check_freq=False,
                check_names=False,
                rtol=1e-6,
                atol=1e-9,
            )
        pd.testing.assert_frame_equal(
            loaded.results.v_res,
            analyzed_grid.results.v_res,
            check_dtype=False,
            check_freq=False,
            check_names=False,
            rtol=1e-6,
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: save(archive_type='tar') looks for '<dir>.zip' and raises "
        "FileNotFoundError after the tar archive was written",
    )
    def test_save_tar_archive(self, wc_grid, tmp_path):
        """Archiving works for every archive type ``shutil`` supports."""
        directory = tmp_path / "edisgo"
        wc_grid.save(directory, archive=True, archive_type="tar")
        assert os.path.isfile(str(directory) + ".tar")
        assert not directory.exists()

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: legacy_grids is not saved; import_edisgo_from_files always "
        "returns legacy_grids=True",
    )
    def test_legacy_flag_survives_roundtrip(self, wc_grid, tmp_path):
        """
        Whether the grid is a legacy ding0 grid decides which data import paths
        are used later; it must survive saving and loading.
        """
        wc_grid.legacy_grids = False
        wc_grid.save(tmp_path / "edisgo")
        loaded = _load_all(tmp_path / "edisgo")
        assert loaded.legacy_grids is False

    def test_pickle_roundtrip(self, reinforced_grid, tmp_path):
        """Pickling restores topology, time series and results unchanged."""
        reinforced_grid.save_edisgo_to_pickle(path=str(tmp_path), filename="e.pkl")
        loaded = import_edisgo_from_pickle("e.pkl", path=str(tmp_path))
        assert_topology_equal(loaded, reinforced_grid)
        assert_timeseries_equal(loaded, reinforced_grid)
        pd.testing.assert_frame_equal(
            loaded.results.v_res, reinforced_grid.results.v_res
        )
        pd.testing.assert_frame_equal(
            loaded.results.equipment_changes, reinforced_grid.results.equipment_changes
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: import_edisgo_from_files replaces any config error by a bare "
        "'raise Exception' without message",
    )
    def test_config_import_error_keeps_cause(self, wc_grid, tmp_path):
        """
        A broken saved configuration raises an error that still names the cause
        instead of an empty ``Exception``.
        """
        wc_grid.save(tmp_path / "edisgo")
        config_files = [
            f for f in os.listdir(tmp_path / "edisgo") if f.endswith(".json")
        ]
        assert config_files
        (tmp_path / "edisgo" / config_files[0]).write_text("{not json")
        with pytest.raises(Exception) as excinfo:
            _load_all(tmp_path / "edisgo")
        assert str(excinfo.value) != ""


# ======================================================================================
# data handling: aggregation, memory reduction, resampling, manual time series
# ======================================================================================


class TestDataHandling:
    def test_aggregate_components_does_not_change_power_flow(self, analyzed_grid):
        """
        Aggregating loads and generators at the same bus keeps every bus's net
        injection, so voltages and currents of the power flow stay the same and
        total P and Q per time step are conserved.
        """
        v_before = analyzed_grid.results.v_res.copy()
        i_before = analyzed_grid.results.i_res.copy()
        totals_before = net_injection_totals(analyzed_grid)
        p_set_before = analyzed_grid.topology.loads_df.p_set.sum()
        p_nom_before = analyzed_grid.topology.generators_df.p_nom.sum()

        analyzed_grid.aggregate_components()
        analyzed_grid.analyze()

        assert max_abs_diff(analyzed_grid.results.v_res, v_before) < V_ATOL_SAME_STATE
        assert max_abs_diff(analyzed_grid.results.i_res, i_before) < 1e-8
        assert max_abs_diff(net_injection_totals(analyzed_grid), totals_before) < 1e-9
        assert analyzed_grid.topology.loads_df.p_set.sum() == pytest.approx(
            p_set_before
        )
        assert analyzed_grid.topology.generators_df.p_nom.sum() == pytest.approx(
            p_nom_before
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: aggregate_components with NaN in a grouping column: "
        "groupby.sum (dropna=True) drops the component from loads_df, the time "
        "series aggregation over groupby.groups then raises TypeError",
    )
    def test_aggregate_components_keeps_loads_with_missing_sector(self, wc_grid):
        """
        Aggregation by bus and sector conserves total load also if a load has no
        sector information.
        """
        load = wc_grid.topology.loads_df.index[0]
        wc_grid.topology.loads_df.loc[load, "sector"] = np.nan
        p_set_before = wc_grid.topology.loads_df.p_set.sum()
        p_before = wc_grid.timeseries.loads_active_power.sum(axis=1)

        wc_grid.aggregate_components(aggregate_loads_by_cols=["bus", "sector"])

        assert wc_grid.topology.loads_df.p_set.sum() == pytest.approx(p_set_before)
        np.testing.assert_allclose(
            wc_grid.timeseries.loads_active_power.sum(axis=1), p_before
        )

    def test_reduce_memory_keeps_values_and_power_flow(self, analyzed_grid):
        """
        ``reduce_memory`` only changes precision: all time series and results
        stay within float32 precision, and a subsequent power flow deviates by
        less than 1e-5 p.u. (float32 rounding of injections).
        """
        before = copy.deepcopy(analyzed_grid)
        analyzed_grid.reduce_memory()
        for attr in TIMESERIES_ATTRIBUTES:
            reduced = getattr(analyzed_grid.timeseries, attr)
            assert (reduced.dtypes == "float32").all(), attr
            np.testing.assert_allclose(
                reduced, getattr(before.timeseries, attr), rtol=1e-6, atol=1e-9
            )
        np.testing.assert_allclose(
            analyzed_grid.results.v_res, before.results.v_res, rtol=1e-6
        )
        analyzed_grid.analyze()
        assert max_abs_diff(analyzed_grid.results.v_res, before.results.v_res) < 1e-5

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: check_integrity compares max(P) > p_set without tolerance "
        "(the 1e-6 is applied to the count), float32 rounding triggers false "
        "'exceed' warnings",
    )
    def test_reduce_memory_causes_no_integrity_warnings(self, wc_grid, caplog):
        """
        Reducing the precision of consistent time series must not make
        ``check_integrity`` report components exceeding their nominal power.
        """
        assert not [
            m for m in edisgo_warnings(caplog, wc_grid.check_integrity) if "exceed" in m
        ]
        wc_grid.reduce_memory()
        warnings = edisgo_warnings(caplog, wc_grid.check_integrity)
        assert not [m for m in warnings if "exceed" in m], warnings

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: EDisGo.reduce_memory documents dsm_attr_to_reduce but never "
        "calls DSM.reduce_memory",
    )
    def test_reduce_memory_includes_dsm(self, wc_grid):
        """``reduce_memory`` reduces all time series containers, including DSM."""
        ti = wc_grid.timeseries.timeindex
        loads = wc_grid.topology.loads_df.index[:2]
        for attr, sign in (("p_min", -1), ("p_max", 1), ("e_min", -1), ("e_max", 1)):
            setattr(
                wc_grid.dsm, attr, pd.DataFrame(sign * 0.1, index=ti, columns=loads)
            )
        wc_grid.reduce_memory()
        for attr in ("p_min", "p_max", "e_min", "e_max"):
            assert (getattr(wc_grid.dsm, attr).dtypes == "float32").all(), attr

    @settings(
        deadline=None,
        max_examples=examples(15),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        values=st.lists(
            st.floats(-5.0, 5.0, allow_nan=False), min_size=24, max_size=24
        ),
        freq=st.sampled_from(["15min", "30min"]),
    )
    def test_resample_conserves_energy_and_roundtrips(
        self, _wc_grid_original, values, freq
    ):
        """
        Upsampling hourly data with ``ffill`` keeps the energy (sum of P * dt) of
        every component, downsampling back to hourly (mean) restores the
        original series, and all time series containers (incl. DSM) share the
        new time index afterwards.
        """
        edisgo = copy.deepcopy(_wc_grid_original)
        ti = pd.date_range("2011-01-01", periods=24, freq="h")
        edisgo.timeseries.reset()
        edisgo.set_timeindex(ti)
        gens = edisgo.topology.generators_df.index[:3]
        p = pd.DataFrame(
            {g: np.roll(values, i) for i, g in enumerate(gens)}, index=ti, dtype=float
        )
        edisgo.set_time_series_manual(generators_p=p, generators_q=-0.5 * p)
        loads = edisgo.topology.loads_df.index[:2]
        edisgo.dsm.p_min = pd.DataFrame(-1.0, index=ti, columns=loads)
        edisgo.dsm.p_max = pd.DataFrame(1.0, index=ti, columns=loads)
        edisgo.dsm.e_min = pd.DataFrame(-2.0, index=ti, columns=loads)
        edisgo.dsm.e_max = pd.DataFrame(2.0, index=ti, columns=loads)

        edisgo.resample_timeseries(method="ffill", freq=freq)
        new_ti = edisgo.timeseries.timeindex
        dt_new = pd.Timedelta(freq) / pd.Timedelta("1h")
        up = edisgo.timeseries.generators_active_power
        np.testing.assert_allclose(up.sum() * dt_new, p.sum(), rtol=1e-12, atol=1e-12)
        for attr in ("p_min", "p_max", "e_min", "e_max"):
            pd.testing.assert_index_equal(
                getattr(edisgo.dsm, attr).index, new_ti, check_names=False
            )

        down = up.resample("h").mean()
        pd.testing.assert_frame_equal(down, p, check_freq=False, check_names=False)

    @settings(
        deadline=None,
        max_examples=examples(15),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data())
    def test_set_time_series_manual_changes_only_given_columns(
        self, _wc_grid_original, data
    ):
        """
        Non-interference: setting time series manually for some components
        changes exactly these columns; all other time series stay bit-identical,
        and no column for unknown components is created.
        """
        edisgo = copy.deepcopy(_wc_grid_original)
        before = copy.deepcopy(edisgo.timeseries)
        ti = edisgo.timeseries.timeindex
        loads = data.draw(
            st.lists(
                st.sampled_from(list(edisgo.topology.loads_df.index)),
                min_size=1,
                max_size=5,
                unique=True,
            )
        )
        values = data.draw(
            st.lists(
                st.floats(0, 1, allow_nan=False),
                min_size=len(ti) * len(loads),
                max_size=len(ti) * len(loads),
            )
        )
        new = pd.DataFrame(
            np.reshape(values, (len(ti), len(loads))), index=ti, columns=loads
        )
        edisgo.set_time_series_manual(loads_p=new)

        after = edisgo.timeseries
        pd.testing.assert_frame_equal(after.loads_active_power[loads], new)
        others = after.loads_active_power.columns.drop(loads)
        pd.testing.assert_frame_equal(
            after.loads_active_power[others], before.loads_active_power[others]
        )
        for attr in TIMESERIES_ATTRIBUTES:
            if attr == "loads_active_power":
                continue
            pd.testing.assert_frame_equal(getattr(after, attr), getattr(before, attr))
        assert set(after.loads_active_power.columns) <= set(
            edisgo.topology.loads_df.index
        )


# ======================================================================================
# worst-case and reactive power time series
# ======================================================================================


class TestWorstCaseTimeSeries:
    def test_worst_case_bounds_and_signs(self, wc_grid):
        """
        Worst-case time series respect the component ratings and sign conventions:
        generators 0 <= P <= p_nom, loads 0 <= P <= p_set, storage |P| = p_nom
        (charging in load cases, discharging in feed-in cases), and every worst
        case has its own time step.
        """
        ts, topo = wc_grid.timeseries, wc_grid.topology
        gen = ts.generators_active_power
        assert (gen >= 0).all(axis=None)
        assert gen.le(topo.generators_df.p_nom.loc[gen.columns] + 1e-12).all(axis=None)
        load = ts.loads_active_power
        assert (load >= 0).all(axis=None)
        assert load.le(topo.loads_df.p_set.loc[load.columns] + 1e-12).all(axis=None)
        sto = ts.storage_units_active_power
        p_nom = topo.storage_units_df.p_nom.loc[sto.columns]
        twc = ts.timeindex_worst_cases
        for case, step in twc.items():
            expected = -p_nom if case.startswith("load_case") else p_nom
            np.testing.assert_allclose(sto.loc[step], expected)
        assert ts.is_worst_case
        assert twc.is_unique
        assert set(pd.DatetimeIndex(twc.values)) == set(ts.timeindex)

    @pytest.mark.parametrize("case", ["load_case", "feed-in_case"])
    def test_single_case_equals_part_of_both_cases(self, wc_grid, case):
        """
        Setting only one worst case yields the same values as the corresponding
        time steps when both cases are set (cases are independent).
        """
        single = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        single.set_time_series_worst_case_analysis(cases=[case])
        twc_all = wc_grid.timeseries.timeindex_worst_cases
        twc_single = single.timeseries.timeindex_worst_cases
        for label, step in twc_single.items():
            for attr in TIMESERIES_ATTRIBUTES:
                a = getattr(single.timeseries, attr).loc[step]
                b = getattr(wc_grid.timeseries, attr).loc[twc_all[label]]
                pd.testing.assert_series_equal(
                    a.sort_index(), b.sort_index(), check_names=False
                )

    def test_reactive_power_control_reproduces_worst_case_q(self, wc_grid):
        """
        Worst-case time series already use the fixed power factors of the
        default reactive power control: applying the control afterwards changes
        nothing, and applying it twice is idempotent.
        """
        before = copy.deepcopy(wc_grid)
        wc_grid.set_time_series_reactive_power_control()
        assert_timeseries_equal(wc_grid, before, check_exact=False, rtol=1e-12)
        once = copy.deepcopy(wc_grid)
        wc_grid.set_time_series_reactive_power_control()
        assert_timeseries_equal(wc_grid, once)


# ======================================================================================
# topology modifications and integrity
# ======================================================================================


class TestTopologyAndIntegrity:
    @settings(
        deadline=None,
        max_examples=examples(10),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        comp_type=st.sampled_from(["load", "generator", "storage_unit"]),
        p=st.floats(0.001, 1.0),
        bus_idx=st.integers(0, 141),
    )
    def test_add_then_remove_component_restores_object(
        self, _wc_grid_original, comp_type, p, bus_idx
    ):
        """
        Roundtrip: adding a component with time series and removing it again
        leaves topology and time series exactly as before. Buses that are already
        empty dead ends are excluded: there ``remove_*`` deletes bus and line by
        design (see the Klaerung test below).
        """
        edisgo = copy.deepcopy(_wc_grid_original)
        before = copy.deepcopy(edisgo)
        bus = edisgo.topology.buses_df.index[bus_idx]
        assume(not edisgo.topology._check_bus_for_removal(bus))
        ts = pd.Series(p, index=edisgo.timeseries.timeindex)
        kwargs = {
            "load": {"p_set": p, "type": "conventional_load", "sector": "cts"},
            "generator": {"p_nom": p, "generator_type": "solar"},
            "storage_unit": {"p_nom": p},
        }[comp_type]
        name = edisgo.add_component(
            comp_type=comp_type,
            bus=bus,
            ts_active_power=ts,
            ts_reactive_power="default",
            **kwargs,
        )
        edisgo.remove_component(comp_type=comp_type, comp_name=name, drop_ts=True)
        assert_topology_equal(edisgo, before, check_names=False)
        assert_timeseries_equal(edisgo, before, check_names=False)

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: remove_load/generator/storage_unit also delete a bus and "
        "line that already existed as empty dead end before the component was "
        "added (test grid 1: virtual MV busbars of LV grids 4 and 9); documented "
        "cleanup, but add -> remove then changes the original grid",
    )
    def test_remove_component_keeps_preexisting_dead_end_bus(self, wc_grid):
        """
        Removing a component only removes what was added for it: a bus of the
        original grid without components survives adding and removing a load.
        """
        bus = "virtual_BusBar_MVGrid_1_LVGrid_9_MV"
        assert wc_grid.topology._check_bus_for_removal(bus)
        n_lines = len(wc_grid.topology.lines_df)
        name = wc_grid.add_component(
            comp_type="load",
            bus=bus,
            p_set=0.1,
            type="conventional_load",
            sector="cts",
            ts_active_power=pd.Series(0.1, index=wc_grid.timeseries.timeindex),
            ts_reactive_power="default",
        )
        wc_grid.remove_component(comp_type="load", comp_name=name)
        assert bus in wc_grid.topology.buses_df.index
        assert len(wc_grid.topology.lines_df) == n_lines

    def test_graph_stays_connected(self, wc_grid, reinforced_grid):
        """
        The grid is one connected network after import, after reinforcement
        (split feeders, parallel lines) and after aggregating components.
        """
        assert nx.is_connected(wc_grid.to_graph())
        assert nx.is_connected(reinforced_grid.to_graph())
        wc_grid.aggregate_components()
        assert nx.is_connected(wc_grid.to_graph())

    def test_standard_chain_passes_integrity_check(self, reinforced_grid, caplog):
        """
        A database-free standard chain (import, worst case, reactive power
        control, reinforcement, aggregation) produces a consistent object:
        ``check_integrity`` only reports the known very short lines of the test
        grid.
        """
        reinforced_grid.set_time_series_reactive_power_control()
        reinforced_grid.aggregate_components()
        warnings = edisgo_warnings(caplog, reinforced_grid.check_integrity)
        assert [m for m in warnings if "very short line lengths" not in m] == []

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: EDisGo.__init__ adds a filter and another 'MVGrid(...)' to the "
        "formatter of every handler of the 'edisgo' logger per instance",
    )
    def test_logger_setup_does_not_accumulate(self):
        """
        Creating several EDisGo objects must not keep modifying the handlers of
        the ``edisgo`` logger: the format string contains the grid id once and
        each handler carries at most one filter.
        """
        logger = logging.getLogger("edisgo")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logger.addHandler(handler)
        try:
            for _ in range(2):
                EDisGo(ding0_grid=pytest.ding0_test_network_path)
            assert handler.formatter._fmt.count("MVGrid(") == 1
            assert len(handler.filters) <= 1
        finally:
            logger.removeHandler(handler)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: set_time_series_active_power_predefined always calls "
        "egon_engine() when no engine kwarg is given, even for DataFrame input",
    )
    def test_predefined_time_series_from_dataframes_need_no_database(
        self, wc_grid, monkeypatch
    ):
        """
        Setting predefined time series from given profiles must not open a
        database connection (or SSH tunnel).
        """

        def _no_db(*args, **kwargs):
            raise AssertionError("database engine requested")

        monkeypatch.setattr("edisgo.edisgo.egon_engine", _no_db)
        ti = pd.date_range("2011-01-01", periods=4, freq="h")
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path, timeindex=ti)
        sectors = edisgo.topology.loads_df.sector.unique()
        profiles = pd.DataFrame(1.0 / 8760, index=ti, columns=sectors)
        edisgo.set_time_series_active_power_predefined(conventional_loads_ts=profiles)
        assert not edisgo.timeseries.loads_active_power.empty
