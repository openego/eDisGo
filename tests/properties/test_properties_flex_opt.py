"""
Property-based tests for :mod:`edisgo.flex_opt`: technical constraint checks,
reinforcement measures, costs, reactive power control and the operating
strategies of storage units, heat pumps and charging points.

Most tests avoid power flows: the checks get injected voltages and branch loads,
the reinforcement measures get synthetic critical components, and the operating
strategies get synthetic time series. Each test states the property it checks.
Fixtures and helpers live in ``tests/properties/conftest.py``.
"""

from __future__ import annotations

import copy
import math

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import costs as costs_mod
from edisgo.flex_opt import q_control
from edisgo.flex_opt import reinforce_measures as measures
from edisgo.flex_opt.battery_storage_operation import (
    _reference_operation,
    apply_reference_operation,
)
from edisgo.flex_opt.charging_strategies import (
    charging_strategy,
    harmonize_charging_processes_df,
)
from edisgo.flex_opt.heat_pump_operation import operating_strategy
from edisgo.flex_opt.reinforce_grid import (
    _add_lines_changes_to_equipment_changes,
    _add_transformer_changes_to_equipment_changes,
    _next_reinforce_log_run_id,
    run_separate_lv_grids,
)

from .conftest import assert_radial, examples, inject_power_flow_results

#: Settings for tests that only work on small synthetic data (milliseconds).
SYN_SETTINGS = settings(deadline=None, max_examples=examples(60))

#: Settings for tests on a copy of test network 1 without power flow. The
#: function scoped fixtures are safe to reuse across examples because every
#: example either overwrites all injected data or works on a fresh copy.
GRID_SETTINGS = settings(
    deadline=None,
    max_examples=examples(20),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Relative tolerance for quantities that are computed with a handful of float
#: operations from the same inputs (products, quotients, sums of < 200 terms).
#: Rounding errors are ~1e-16 per operation, 1e-9 leaves ample margin and still
#: catches any wrong factor, unit or off-by-one.
RTOL = 1e-9

#: Absolute tolerance in p.u. for voltage identities (values ~1, see RTOL).
V_ATOL = 1e-12

LOAD_FACTOR_KEYS = [
    f"{level}_{case}_{comp}"
    for level in ("mv", "lv")
    for case in ("load_case", "feed-in_case")
    for comp in ("line", "transformer")
]

#: Active power in MW, either zero or at least 1 W in magnitude: subnormal floats
#: would make relative comparisons meaningless without describing any real power.
powers = st.one_of(st.just(0.0), st.floats(1e-6, 50.0), st.floats(-50.0, -1e-6))

load_factors = st.fixed_dictionaries({k: st.floats(0.2, 1.0) for k in LOAD_FACTOR_KEYS})
seeds = st.integers(0, 2**32 - 1)


# ======================================================================================
# helpers
# ======================================================================================


def _random_voltages(edisgo, seed, center, spread):
    """Random voltages for all buses and time steps of the stored power flow."""
    rng = np.random.default_rng(seed)
    v = edisgo.results.v_res
    data = 1 + center + spread * rng.uniform(-1, 1, size=v.shape)
    return pd.DataFrame(data, index=v.index, columns=v.columns)


def _random_branch_loads(edisgo, seed, kmax):
    """Random apparent power in [0, kmax * s_nom] for all lines and transformers."""
    rng = np.random.default_rng(seed)
    s_res = edisgo.results.s_res
    topo = edisgo.topology
    s_nom = pd.concat([topo.lines_df.s_nom, topo.transformers_df.s_nom])
    s_nom = s_nom.loc[s_res.columns].to_numpy()
    data = kmax * s_nom * rng.uniform(0, 1, size=s_res.shape)
    return pd.DataFrame(data, index=s_res.index, columns=s_res.columns)


def _random_slack(edisgo, seed, scale):
    """Random HV/MV station power flow of up to ``scale`` times its rating."""
    rng = np.random.default_rng(seed)
    s_nom = edisgo.topology.transformers_hvmv_df.s_nom.sum()
    idx = edisgo.results.v_res.index
    return pd.DataFrame(
        {
            "p": scale * s_nom * rng.uniform(-1, 1, len(idx)),
            "q": 0.5 * scale * s_nom * rng.uniform(-1, 1, len(idx)),
        },
        index=idx,
    )


def _load_factor(edisgo, level, comp):
    """Load factor per time step for ``level`` (mv/lv) and ``comp`` (line/...)."""
    lf = edisgo.config["grid_expansion_load_factors"]
    cases = edisgo.timeseries.timesteps_load_feedin_case
    return cases.map(lambda case: lf[f"{level}_{case}_{comp}"])


def _set_load_factors(edisgo, factors):
    edisgo.config["grid_expansion_load_factors"].update(factors)


def _voltage_band(edisgo):
    return edisgo.config["grid_expansion_allowed_voltage_deviations"]


def _station_buses(edisgo):
    return pd.Index(edisgo.topology.transformers_df.bus1.unique())


def _lv_buses(edisgo):
    topo = edisgo.topology
    return topo.buses_df.index.drop(
        topo.mv_grid.buses_df.index.append(_station_buses(edisgo))
    )


def _line_equipment(edisgo, type_info):
    data = edisgo.topology.equipment_data
    for key in ("lv_cables", "mv_cables", "mv_overhead_lines"):
        if key in data and type_info in data[key].index:
            return data[key].loc[type_info]
    return None


# ======================================================================================
# check_tech_constraints: voltages
# ======================================================================================


class TestVoltageChecks:
    @GRID_SETTINGS
    @given(
        seed=seeds,
        center=st.floats(-0.08, 0.08),
        spread=st.floats(0.0, 0.15),
        split=st.booleans(),
    )
    def test_voltage_deviation_is_distance_to_allowed_band(
        self, analyzed_grid, seed, center, spread, split
    ):
        """
        Clip identity: for every bus and time step ``v - dev == clip(v, lower,
        upper)``, i.e. the deviation is positive exactly above the upper limit,
        negative exactly below the lower limit and zero inside the band; every bus
        of the power flow gets limits and ``lower < upper``.
        """
        edisgo = analyzed_grid
        v = _random_voltages(edisgo, seed, center, spread)
        inject_power_flow_results(edisgo, v_res=v)

        upper, lower = checks.allowed_voltage_limits(edisgo, split_voltage_band=split)
        dev = checks.voltage_deviation_from_allowed_voltage_limits(
            edisgo, split_voltage_band=split
        )
        assert set(dev.columns) == set(v.columns)
        assert not dev.columns.duplicated().any()
        cols = dev.columns
        v, upper, lower = v[cols], upper[cols], lower[cols]
        assert (lower < upper).all().all()
        clipped = np.clip(v.to_numpy(), lower.to_numpy(), upper.to_numpy())
        np.testing.assert_allclose((v - dev).to_numpy(), clipped, rtol=0, atol=V_ATOL)
        assert ((dev > 0) == (v > upper)).all().all()
        assert ((dev < 0) == (v < lower)).all().all()

    @GRID_SETTINGS
    @given(
        seed=seeds,
        center=st.floats(-0.08, 0.08),
        spread=st.floats(0.0, 0.15),
        grid_id=st.integers(1, 9),
    )
    def test_voltage_issues_partition_and_match_deviation(
        self, analyzed_grid, seed, center, spread, grid_id
    ):
        """
        ``voltage_issues(None)`` is the disjoint union of the MV, MV/LV and LV
        results; a bus is reported iff its deviation is non-zero in some time
        step, with ``abs_max_voltage_dev == max |dev|``, ``max_voltage_dev ==
        dev[time_index]`` and rows sorted descending; a query for one LV grid
        returns exactly the rows of that grid.
        """
        edisgo = analyzed_grid
        topo = edisgo.topology
        inject_power_flow_results(
            edisgo, v_res=_random_voltages(edisgo, seed, center, spread)
        )
        dev = checks.voltage_deviation_from_allowed_voltage_limits(edisgo)
        total = checks.voltage_issues(edisgo, None)
        parts = {
            lvl: checks.voltage_issues(edisgo, lvl) for lvl in ("mv", "mv_lv", "lv")
        }

        assert sum(len(p) for p in parts.values()) == len(total)
        assert set(total.index) == set().union(*(set(p.index) for p in parts.values()))
        assert set(parts["mv"].index) <= set(topo.mv_grid.buses_df.index)
        assert set(parts["mv_lv"].index) <= set(_station_buses(edisgo))
        assert set(parts["lv"].index) <= set(_lv_buses(edisgo))

        assert set(total.index) == set(dev.columns[(dev != 0).any()])
        assert total.abs_max_voltage_dev.is_monotonic_decreasing
        assert (total.abs_max_voltage_dev > 0).all()
        for bus, row in total.iterrows():
            assert row.abs_max_voltage_dev == dev[bus].abs().max()
            assert row.max_voltage_dev == dev.at[row.time_index, bus]
            assert abs(row.max_voltage_dev) == row.abs_max_voltage_dev

        grid = topo.get_lv_grid(int(grid_id))
        for lvl in ("mv_lv", "lv"):
            sub = checks.voltage_issues(edisgo, lvl, lv_grid_id=grid.id)
            expected = parts[lvl].index[parts[lvl].index.isin(grid.buses_df.index)]
            assert set(sub.index) == set(expected)
            assert (sub.lv_grid_id == grid.id).all()

    @GRID_SETTINGS
    @given(seed=seeds, center=st.floats(-0.08, 0.08), spread=st.floats(0.0, 0.15))
    def test_allowed_limits_follow_reference_voltages(
        self, analyzed_grid, seed, center, spread
    ):
        """
        With a split voltage band, MV limits are constants from the config, the
        limits of a station's secondary side follow the voltage at its primary
        side and the limits of an LV bus follow the voltage at its own station's
        secondary side, each shifted by the configured rise and drop; without a
        split band the limits are ``1 +- mv_lv_max_v_rise/drop``.
        """
        edisgo = analyzed_grid
        topo = edisgo.topology
        band = _voltage_band(edisgo)
        v = _random_voltages(edisgo, seed, center, spread)
        inject_power_flow_results(edisgo, v_res=v)
        upper, lower = checks.allowed_voltage_limits(edisgo)

        mv_buses = topo.mv_grid.buses_df.index
        offset, ctrl = band["hv_mv_trafo_offset"], band["hv_mv_trafo_control_deviation"]
        np.testing.assert_allclose(
            upper[mv_buses], 1 + offset - ctrl + band["mv_max_v_rise"], atol=V_ATOL
        )
        np.testing.assert_allclose(
            lower[mv_buses], 1 + offset + ctrl - band["mv_max_v_drop"], atol=V_ATOL
        )
        for grid in topo.lv_grids:
            prim, sec = grid.transformers_df.iloc[0][["bus0", "bus1"]]
            rise, drop = (
                band["mv_lv_station_max_v_rise"],
                band["mv_lv_station_max_v_drop"],
            )
            np.testing.assert_allclose(upper[sec], v[prim] + rise, atol=V_ATOL)
            np.testing.assert_allclose(lower[sec], v[prim] - drop, atol=V_ATOL)
            for bus in grid.buses_df.index.drop(sec):
                np.testing.assert_allclose(
                    upper[bus], v[sec] + band["lv_max_v_rise"], atol=V_ATOL
                )
                np.testing.assert_allclose(
                    lower[bus], v[sec] - band["lv_max_v_drop"], atol=V_ATOL
                )

        upper, lower = checks.allowed_voltage_limits(edisgo, split_voltage_band=False)
        np.testing.assert_allclose(upper, 1 + band["mv_lv_max_v_rise"], atol=V_ATOL)
        np.testing.assert_allclose(lower, 1 - band["mv_lv_max_v_drop"], atol=V_ATOL)

    @GRID_SETTINGS
    @given(seed=seeds, lv_spread=st.floats(0.0, 0.3))
    def test_band_split_keeps_lv_limits_within_combined_band(
        self, analyzed_grid, seed, lv_spread
    ):
        """
        The voltage band split adds up: if all MV buses lie within their MV
        limits and every station's secondary side within its station limits, the
        resulting limits of all LV buses lie within the combined band
        ``[1 - mv_lv_max_v_drop, 1 + mv_lv_max_v_rise]`` (DIN EN 50160).
        """
        edisgo = analyzed_grid
        topo = edisgo.topology
        band = _voltage_band(edisgo)
        rng = np.random.default_rng(seed)
        mv_upper, mv_lower = checks._mv_allowed_voltage_limits(edisgo)
        v = _random_voltages(edisgo, seed, 0.0, lv_spread)
        mv_buses = topo.mv_grid.buses_df.index
        v[mv_buses] = rng.uniform(mv_lower, mv_upper, size=(len(v), len(mv_buses)))
        for grid in topo.lv_grids:
            prim, sec = grid.transformers_df.iloc[0][["bus0", "bus1"]]
            v[sec] = v[prim] + rng.uniform(
                -band["mv_lv_station_max_v_drop"],
                band["mv_lv_station_max_v_rise"],
                size=len(v),
            )
        inject_power_flow_results(edisgo, v_res=v)

        upper, lower = checks.allowed_voltage_limits(edisgo, buses=_lv_buses(edisgo))
        assert (upper.to_numpy() <= 1 + band["mv_lv_max_v_rise"] + V_ATOL).all()
        assert (lower.to_numpy() >= 1 - band["mv_lv_max_v_drop"] - V_ATOL).all()

    @SYN_SETTINGS
    @given(
        offset=st.floats(-0.03, 0.03),
        ctrl=st.floats(0.0, 0.01),
        rise=st.floats(0.02, 0.08),
        drop=st.floats(0.02, 0.08),
    )
    def test_mv_band_width_is_rise_plus_drop_minus_control_deviation(
        self, offset, ctrl, rise, drop
    ):
        """
        The MV band is centred on the HV/MV offset and its width is ``rise +
        drop - 2 * control_deviation``; the function returns (upper, lower).
        """
        cfg = {
            "grid_expansion_allowed_voltage_deviations": {
                "hv_mv_trafo_offset": offset,
                "hv_mv_trafo_control_deviation": ctrl,
                "mv_max_v_rise": rise,
                "mv_max_v_drop": drop,
            }
        }
        upper, lower = checks._mv_allowed_voltage_limits(SimpleNamespace(config=cfg))
        assert upper > lower
        assert upper - lower == pytest.approx(rise + drop - 2 * ctrl, abs=V_ATOL)
        assert upper == pytest.approx(1 + offset - ctrl + rise, abs=V_ATOL)


# ======================================================================================
# check_tech_constraints: loading
# ======================================================================================


class TestLoadingChecks:
    @GRID_SETTINGS
    @given(seed=seeds, kmax=st.floats(0.1, 3.0), lf=load_factors)
    def test_line_loading_consistent_with_load_factors(
        self, analyzed_grid, seed, kmax, lf
    ):
        """
        Allowed line load is ``s_nom * load_factor(level, case(t))`` and lies in
        ``(0, s_nom]``; relative load times allowed load reproduces ``s_res``;
        the MV and LV overload tables are disjoint and contain exactly the lines
        whose relative load exceeds 1, with the maximum and its time step; per-LV
        grid queries return the rows of their grid.
        """
        edisgo = analyzed_grid
        topo = edisgo.topology
        _set_load_factors(edisgo, lf)
        s_res = _random_branch_loads(edisgo, seed, kmax)
        inject_power_flow_results(edisgo, s_res=s_res)
        mv_lines = topo.mv_grid.lines_df.index
        lv_lines = topo.lines_df.index.drop(mv_lines)

        allowed = checks.lines_allowed_load(edisgo)
        assert sorted(allowed.columns) == sorted(topo.lines_df.index)
        for level, lines in (("mv", mv_lines), ("lv", lv_lines)):
            factor = _load_factor(edisgo, level, "line").to_numpy()[:, None]
            expected = factor * topo.lines_df.s_nom[lines].to_numpy()[None, :]
            np.testing.assert_allclose(allowed[lines], expected, rtol=RTOL)
        assert (allowed > 0).all().all()
        assert (allowed <= topo.lines_df.s_nom[allowed.columns]).all().all()

        rel = checks.lines_relative_load(edisgo)
        assert set(rel.columns) == set(topo.lines_df.index)
        assert (rel >= 0).all().all()
        np.testing.assert_allclose(
            rel * allowed[rel.columns], s_res[rel.columns], rtol=RTOL
        )

        crit = {
            "mv": checks.mv_line_max_relative_overload(edisgo),
            "lv": checks.lv_line_max_relative_overload(edisgo),
        }
        for level, lines in (("mv", mv_lines), ("lv", lv_lines)):
            peak = rel[lines].max()
            expected = peak[peak > 1]
            assert set(crit[level].index) == set(expected.index)
            if expected.empty:
                continue
            c = crit[level].loc[expected.index]
            np.testing.assert_allclose(c.max_rel_overload, expected, rtol=RTOL)
            assert (c.time_index == rel[expected.index].idxmax()).all()
            assert (c.voltage_level == level).all()
        assert not set(crit["mv"].index) & set(crit["lv"].index)

        for grid in topo.lv_grids:
            sub = checks.lv_line_max_relative_overload(edisgo, lv_grid_id=grid.id)
            expected = crit["lv"].index[crit["lv"].index.isin(grid.lines_df.index)]
            assert set(sub.index) == set(expected)

    @GRID_SETTINGS
    @given(seed=seeds, k=st.integers(2, 4))
    def test_relative_load_scales_inversely_with_parallel_lines(
        self, analyzed_grid_factory, seed, k
    ):
        """
        Raising the number of parallel lines by a factor ``k`` (without a new
        power flow) divides the relative load of exactly these lines by ``k``.
        """
        edisgo = analyzed_grid_factory()
        topo = edisgo.topology
        inject_power_flow_results(edisgo, s_res=_random_branch_loads(edisgo, seed, 2))
        rel_before = checks.lines_relative_load(edisgo)
        rng = np.random.default_rng(seed)
        lines = rng.choice(topo.lines_df.index, size=10, replace=False)
        topo.update_number_of_parallel_lines(topo.lines_df.num_parallel[lines] * k)
        rel_after = checks.lines_relative_load(edisgo)
        np.testing.assert_allclose(rel_after[lines], rel_before[lines] / k, rtol=RTOL)
        others = rel_before.columns.drop(lines)
        pd.testing.assert_frame_equal(rel_after[others], rel_before[others])

    @GRID_SETTINGS
    @given(
        seed=seeds,
        kmax=st.floats(0.1, 3.0),
        slack_scale=st.floats(0.0, 3.0),
        lf=load_factors,
    )
    def test_station_overload_and_reinforcement_with_unchanged_flows(
        self, analyzed_grid_factory, seed, kmax, slack_scale, lf
    ):
        """
        A station is reported overloaded iff ``S(t) > lf(t) * sum(s_nom)`` in
        some time step, with ``s_missing = max_t S(t)/lf(t) - sum(s_nom)`` taken
        at ``time_index``; the relative loads equal ``S / allowed``; after the
        station reinforcement measure the new transformers carry the unchanged
        station flows in every time step.
        """
        edisgo = analyzed_grid_factory()
        topo = edisgo.topology
        _set_load_factors(edisgo, lf)
        s_res = _random_branch_loads(edisgo, seed, kmax)
        slack = _random_slack(edisgo, seed, slack_scale)
        inject_power_flow_results(edisgo, s_res=s_res, pfa_slack=slack)

        loads = {topo.mv_grid.station_name: np.hypot(slack.p, slack.q)}
        ratings = {topo.mv_grid.station_name: topo.transformers_hvmv_df.s_nom.sum()}
        factors = {topo.mv_grid.station_name: _load_factor(edisgo, "mv", "transformer")}
        for grid in topo.lv_grids:
            name = grid.station_name
            loads[name] = s_res[grid.transformers_df.index].sum(axis=1)
            ratings[name] = grid.transformers_df.s_nom.sum()
            factors[name] = _load_factor(edisgo, "lv", "transformer")

        crit_mv = checks.hv_mv_station_max_overload(edisgo)
        crit_lv = checks.mv_lv_station_max_overload(edisgo)
        crit = pd.concat([crit_mv, crit_lv])
        rel = checks.stations_relative_load(edisgo)
        for name in loads:
            np.testing.assert_allclose(
                rel[name], loads[name] / (factors[name] * ratings[name]), rtol=RTOL
            )
            overloaded = (loads[name] > factors[name] * ratings[name]).any()
            assert (name in crit.index) == overloaded
            if not overloaded:
                continue
            missing = loads[name] / factors[name] - ratings[name]
            row = crit.loc[name]
            assert row.s_missing == pytest.approx(missing.max(), rel=RTOL)
            assert missing[row.time_index] == pytest.approx(row.s_missing, rel=RTOL)
        for grid in topo.lv_grids:
            sub = checks.mv_lv_station_max_overload(edisgo, lv_grid_id=grid.id)
            assert set(sub.index) == set(crit_lv.index) & {grid.station_name}

        measures.reinforce_hv_mv_station_overloading(edisgo, crit_mv)
        measures.reinforce_mv_lv_station_overloading(edisgo, crit_lv)
        assert checks.hv_mv_station_max_overload(edisgo).empty
        for grid in topo.lv_grids:
            name = grid.station_name
            capacity = factors[name] * grid.transformers_df.s_nom.sum()
            assert (loads[name] <= capacity * (1 + RTOL)).all(), name


# ======================================================================================
# costs
# ======================================================================================


@st.composite
def line_changes(draw, lines):
    """Lines with a new number of parallel lines and matching change rows."""
    chosen = draw(st.lists(st.sampled_from(lines), min_size=1, max_size=8, unique=True))
    num_parallel, rows = {}, []
    for line in chosen:
        n = draw(st.integers(1, 4))
        parts = draw(st.lists(st.integers(1, n), min_size=1, max_size=3))
        parts = [p for i, p in enumerate(parts) if sum(parts[: i + 1]) <= n]
        num_parallel[line] = n
        rows += [(line, q, draw(st.integers(0, 3))) for q in parts]
    return num_parallel, rows


def _equipment_changes(edisgo, rows):
    lines = [r[0] for r in rows]
    return pd.DataFrame(
        {
            "iteration_step": [r[2] for r in rows],
            "change": "changed",
            "equipment": edisgo.topology.lines_df.loc[lines, "type_info"].values,
            "quantity": [r[1] for r in rows],
        },
        index=lines,
    )


class TestCosts:
    @GRID_SETTINGS
    @given(data=st.data())
    def test_line_costs_use_reported_quantity(self, wc_grid_factory, data):
        """
        For every reinforced line ``total_costs == earthworks + cable costs *
        quantity`` with the reported quantity, ``length == quantity * line
        length``, costs are >= 0 and ``without_generator_import`` (ignoring
        iteration step 0) never costs more per line.
        """
        edisgo = wc_grid_factory()
        topo = edisgo.topology
        num_parallel, rows = data.draw(line_changes(sorted(topo.lines_df.index)))
        topo.update_number_of_parallel_lines(pd.Series(num_parallel))
        edisgo.results.equipment_changes = _equipment_changes(edisgo, rows)

        costs = costs_mod.grid_expansion_costs(edisgo)
        lines = list(num_parallel)
        assert set(costs.index) == set(lines)
        ref = costs_mod.line_expansion_costs(edisgo, lines)
        quantity = pd.Series(
            {line: sum(r[1] for r in rows if r[0] == line) for line in lines}
        )
        costs = costs.loc[lines]
        np.testing.assert_array_equal(costs.quantity, quantity[lines])
        np.testing.assert_allclose(
            costs.total_costs,
            ref.costs_earthworks + ref.costs_cable * costs.quantity,
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            costs.length, costs.quantity * topo.lines_df.length[lines], rtol=RTOL
        )
        assert (costs.voltage_level == ref.voltage_level).all()
        assert (costs.total_costs >= 0).all()

        without = costs_mod.grid_expansion_costs(edisgo, without_generator_import=True)
        without = without.drop("No reinforced equipment.", errors="ignore")
        assert set(without.index) <= set(costs.index)
        assert (
            without.total_costs <= costs.total_costs[without.index] * (1 + RTOL)
        ).all()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Bug: grid_expansion_costs costs lines with the summed quantity from "
            "equipment_changes but reports quantity = min(quantity, num_parallel) "
            "and length = quantity * length (costs.py, _get_line_costs)"
        ),
    )
    def test_line_costs_match_reported_quantity_after_type_change(self, wc_grid):
        """
        Costs and reported quantity stay consistent when a line that got parallel
        standard lines is later replaced by a single standard line (as the
        voltage measure does when it reconnects a feeder): ``total_costs ==
        earthworks + cable costs * quantity``.
        """
        edisgo = wc_grid
        topo = edisgo.topology
        line = "Line_10007"
        standard = edisgo.config["grid_expansion_standard_equipment"]["mv_line_20kv"]
        assert topo.lines_df.at[line, "type_info"] == standard
        crit = pd.DataFrame(
            {"max_rel_overload": [2.5], "voltage_level": ["mv"]}, index=[line]
        )
        changes = measures.reinforce_lines_overloading(edisgo, crit)
        _add_lines_changes_to_equipment_changes(edisgo, changes, 1)
        topo.change_line_type([line], standard)
        _add_lines_changes_to_equipment_changes(edisgo, {line: 1}, 2)

        costs = costs_mod.grid_expansion_costs(edisgo).loc[line]
        ref = costs_mod.line_expansion_costs(edisgo, [line]).loc[line]
        assert costs.total_costs == pytest.approx(
            ref.costs_earthworks + ref.costs_cable * costs.quantity, rel=RTOL
        )

    def test_reinforced_grid_line_costs_use_reported_quantity(self, reinforced_grid):
        """
        After a real worst-case reinforcement every costed line satisfies
        ``total_costs == earthworks + cable costs * quantity`` and ``length ==
        quantity * line length``; transformers cost the configured price of
        their level.
        """
        edisgo = reinforced_grid
        topo = edisgo.topology
        costs = edisgo.results.grid_expansion_costs
        lines = costs.index[costs.index.isin(topo.lines_df.index)]
        assert len(lines) > 0
        ref = costs_mod.line_expansion_costs(edisgo, lines)
        np.testing.assert_allclose(
            costs.total_costs[lines],
            ref.costs_earthworks + ref.costs_cable * costs.quantity[lines],
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            costs.length[lines],
            costs.quantity[lines] * topo.lines_df.length[lines],
            rtol=RTOL,
        )
        trafos = costs.index.drop(lines)
        price = {
            "hv/mv": float(edisgo.config["costs_transformers"]["mv"]),
            "mv/lv": float(edisgo.config["costs_transformers"]["lv"]),
        }
        for trafo in trafos:
            row = costs.loc[trafo]
            assert row.quantity == 1
            assert row.total_costs == price[row.voltage_level]

    @GRID_SETTINGS
    @given(k=st.floats(0.01, 100.0), data=st.data())
    def test_line_expansion_costs_linear_in_length(self, wc_grid_factory, k, data):
        """
        Line costs are linear in the line length (no unit mix-up), non-negative,
        earthworks + cable costs >= cable costs, ``voltage_level == 'mv'`` iff
        the line is in the MV grid, and a subset query returns the same rows as
        the full query.
        """
        edisgo = wc_grid_factory()
        topo = edisgo.topology
        full = costs_mod.line_expansion_costs(edisgo)
        assert list(full.index) == list(topo.lines_df.index)
        assert (full[["costs_earthworks", "costs_cable"]] >= 0).all().all()
        is_mv = full.index.isin(topo.mv_grid.lines_df.index)
        assert ((full.voltage_level == "mv") == is_mv).all()

        subset = data.draw(
            st.lists(st.sampled_from(list(full.index)), min_size=1, unique=True)
        )
        pd.testing.assert_frame_equal(
            costs_mod.line_expansion_costs(edisgo, subset), full.loc[subset]
        )

        topo._lines_df["length"] *= k
        scaled = costs_mod.line_expansion_costs(edisgo)
        for col in ("costs_earthworks", "costs_cable"):
            np.testing.assert_allclose(scaled[col], k * full[col], rtol=RTOL)

    def test_urban_line_costs_not_below_rural(self, wc_grid):
        """
        Raising the population density from rural to urban never makes a line
        cheaper (urban earthworks are at least as expensive as rural ones).
        """
        edisgo = wc_grid
        district = edisgo.topology.grid_district
        district["population"] = 1
        rural = costs_mod.line_expansion_costs(edisgo)
        district["population"] = 1e9
        urban = costs_mod.line_expansion_costs(edisgo)
        total = lambda df: df.costs_earthworks + df.costs_cable  # noqa: E731
        assert (total(urban) >= total(rural)).all()
        pd.testing.assert_series_equal(urban.costs_cable, rural.costs_cable)

    @GRID_SETTINGS
    @given(data=st.data())
    def test_transformer_costs_depend_only_on_level(self, wc_grid, data):
        """
        Transformer costs are the positive configured price of the transformer's
        level, independent of its type; a subset query returns the same rows as
        the full query and an empty query returns an empty table.
        """
        edisgo = wc_grid
        topo = edisgo.topology
        full = costs_mod.transformer_expansion_costs(edisgo)
        hvmv = topo.transformers_hvmv_df.index
        assert set(full.index) == set(hvmv) | set(topo.transformers_df.index)
        assert (full.costs > 0).all()
        expected_level = np.where(full.index.isin(hvmv), "hv/mv", "mv/lv")
        assert (full.voltage_level == expected_level).all()
        price = {
            "hv/mv": float(edisgo.config["costs_transformers"]["mv"]),
            "mv/lv": float(edisgo.config["costs_transformers"]["lv"]),
        }
        assert (full.costs == full.voltage_level.map(price)).all()

        subset = data.draw(
            st.lists(st.sampled_from(sorted(full.index)), min_size=1, unique=True)
        )
        sub = costs_mod.transformer_expansion_costs(edisgo, subset)
        pd.testing.assert_frame_equal(sub.sort_index(), full.loc[subset].sort_index())
        empty = costs_mod.transformer_expansion_costs(edisgo, [])
        assert empty.empty
        assert list(empty.columns) == ["costs", "voltage_level"]


# ======================================================================================
# q_control
# ======================================================================================

COMPONENT_TYPES = (
    "generator",
    "storage_unit",
    "conventional_load",
    "charging_point",
    "heat_pump",
)
GENERATOR_LIKE = ("generator", "storage_unit")


class TestQControl:
    @SYN_SETTINGS
    @given(
        mode=st.sampled_from(["inductive", "capacitive"]),
        case=st.sampled_from([str.lower, str.upper, str.title]),
    )
    def test_q_sign_generator_mirrors_load(self, mode, case):
        """
        Generator and load sign conventions are exact mirror images, the result
        is +-1 and independent of the spelling of the mode.
        """
        gen = q_control.get_q_sign_generator(case(mode))
        load = q_control.get_q_sign_load(case(mode))
        assert gen == -load
        assert gen in (-1, 1)
        assert gen == q_control.get_q_sign_generator(mode)
        assert (gen == 1) == (mode == "capacitive")

    @SYN_SETTINGS
    @given(mode=st.text(max_size=12))
    def test_q_sign_rejects_unknown_mode(self, mode):
        """Every mode other than inductive/capacitive raises a ValueError."""
        assume(mode.lower() not in ("inductive", "capacitive"))
        with pytest.raises(ValueError):
            q_control.get_q_sign_generator(mode)
        with pytest.raises(ValueError):
            q_control.get_q_sign_load(mode)

    @SYN_SETTINGS
    @given(
        p=st.lists(powers, min_size=1, max_size=6),
        n_t=st.integers(1, 4),
        pf=st.lists(st.floats(0.5, 1.0), min_size=6, max_size=6),
        signs=st.lists(st.sampled_from([-1, 1]), min_size=6, max_size=6),
        perm_seed=seeds,
    )
    def test_fixed_cosphi_keeps_power_factor_and_sign(
        self, p, n_t, pf, signs, perm_seed
    ):
        """
        ``Q = P * q_sign * tan(arccos(pf))``: ``|P| / hypot(P, Q) == pf`` where
        P != 0, ``sign(Q) == q_sign * sign(P)`` for pf < 1, Q == 0 for pf == 1,
        the result is linear in P (MW vs. kW) and independent of the column order.
        """
        cols = [f"c{i}" for i in range(len(p))]
        rng = np.random.default_rng(perm_seed)
        active = pd.DataFrame(
            np.array(p)[None, :] * rng.uniform(0.5, 1.5, (n_t, len(p))), columns=cols
        )
        pf_s = pd.Series(pf[: len(p)], index=cols)
        sign_s = pd.Series(signs[: len(p)], index=cols)
        q = q_control.fixed_cosphi(active, sign_s, pf_s)

        assert not q.isna().any().any()
        nonzero = active != 0
        ratio = active.abs() / np.hypot(active, q)
        np.testing.assert_allclose(
            ratio.where(nonzero),
            pd.DataFrame(np.broadcast_to(pf_s, active.shape), columns=cols).where(
                nonzero
            ),
            rtol=1e-9,
        )
        unity = pf_s == 1.0
        assert (q.loc[:, unity] == 0).all().all()
        expected_sign = np.sign(active) * sign_s
        assert (np.sign(q.loc[:, ~unity]) == expected_sign.loc[:, ~unity]).all().all()

        np.testing.assert_allclose(
            q_control.fixed_cosphi(1000 * active, sign_s, pf_s), 1000 * q, rtol=1e-12
        )
        perm = list(rng.permutation(cols))
        permuted = q_control.fixed_cosphi(
            active[perm], sign_s.loc[perm[::-1]], pf_s.loc[perm[::-1]]
        )
        pd.testing.assert_frame_equal(permuted[cols], q)

    @GRID_SETTINGS
    @given(
        modes=st.fixed_dictionaries(
            {
                f"{lvl}_{t}": st.sampled_from(["inductive", "capacitive"])
                for lvl in ("mv", "lv")
                for t in COMPONENT_TYPES
            }
        ),
        levels=st.lists(st.sampled_from(["mv", "lv"]), min_size=1, max_size=8),
        perm_seed=seeds,
    )
    def test_default_sign_and_power_factor(self, wc_grid, modes, levels, perm_seed):
        """
        Default reactive power signs are +-1 per component and follow the
        configured mode with the generator convention for generators and storage
        units and the load convention for loads, charging points and heat pumps;
        default power factors are in (0, 1]; both are independent of row order.
        """
        configs = {
            "reactive_power_mode": modes,
            "reactive_power_factor": wc_grid.config["reactive_power_factor"],
        }
        comp_df = pd.DataFrame(
            {"voltage_level": levels}, index=[f"c{i}" for i in range(len(levels))]
        )
        rng = np.random.default_rng(perm_seed)
        shuffled = comp_df.iloc[rng.permutation(len(comp_df))]
        for comp in COMPONENT_TYPES:
            sign = q_control._fixed_cosphi_default_reactive_power_sign(
                comp_df, comp, configs
            )
            pf = q_control._fixed_cosphi_default_power_factor(comp_df, comp, configs)
            gen_like = 1 if comp in GENERATOR_LIKE else -1
            for name, lvl in comp_df.voltage_level.items():
                capacitive = modes[f"{lvl}_{comp}"] == "capacitive"
                assert sign[name] == gen_like * (1 if capacitive else -1)
            assert ((pf > 0) & (pf <= 1)).all()
            pd.testing.assert_series_equal(
                q_control._fixed_cosphi_default_reactive_power_sign(
                    shuffled, comp, configs
                ).loc[comp_df.index],
                sign,
            )
            pd.testing.assert_series_equal(
                q_control._fixed_cosphi_default_power_factor(
                    shuffled, comp, configs
                ).loc[comp_df.index],
                pf,
            )
        for func in (
            q_control._fixed_cosphi_default_reactive_power_sign,
            q_control._fixed_cosphi_default_power_factor,
        ):
            with pytest.raises(ValueError):
                func(comp_df, "transformer", configs)


# ======================================================================================
# battery_storage_operation
# ======================================================================================


def _run_reference(fmd, soe_init, soe_max, p_nom, freq, eta_s, eta_d):
    return _reference_operation(
        pd.DataFrame({"feedin_minus_demand": fmd}),
        soe_init=soe_init,
        soe_max=soe_max,
        storage_p_nom=p_nom,
        freq=freq,
        efficiency_store=eta_s,
        efficiency_dispatch=eta_d,
    )


storage_params = st.fixed_dictionaries(
    {
        "p_nom": st.floats(0.01, 2.0),
        "max_hours": st.floats(0.1, 6.0),
        "eta_s": st.floats(0.5, 1.0),
        "eta_d": st.floats(0.5, 1.0),
        "soe_share": st.floats(0.0, 1.0),
    }
)

#: Tolerance for the storage operation, which rounds its output to 6 decimals:
#: power and SoE are each off by at most 5e-7, which shifts a SoE difference by
#: at most 1e-6 plus 5e-7 * freq / eta (eta >= 0.5, freq <= 2) -> 4e-6 in total.
STORAGE_ATOL = 4e-6


@pytest.fixture(scope="module")
def storage_grid():
    """Test network 1 with four home storage units in buildings with/without PV."""
    timeindex = pd.date_range("1/1/2011", periods=24, freq="h")
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path, timeindex=timeindex)
    topo = edisgo.topology
    bus = "Bus_BranchTee_LVGrid_2_4"
    topo.storage_units_df = pd.DataFrame(
        {
            "bus": bus,
            "control": "PQ",
            "p_nom": [0.01, 0.005, 0.02, 0.01],
            "max_hours": [2.0, 4.0, 1.0, 3.0],
            "efficiency_store": [0.9, 0.95, 0.8, 1.0],
            "efficiency_dispatch": [0.9, 0.85, 0.8, 1.0],
            "building_id": [1, 2, 3, 4],
        },
        index=["stor_pv_load", "stor_two_each", "stor_no_pv", "stor_no_load"],
    )
    loads = topo.loads_df.index[topo.loads_df.bus.str.contains("LVGrid")][:5]
    gens = topo.generators_df.index[topo.generators_df.bus.str.contains("LVGrid")][:4]
    topo.loads_df["building_id"] = np.nan
    topo.generators_df["building_id"] = np.nan
    topo.loads_df.loc[loads, "building_id"] = [1, 2, 2, 3, 99]
    topo.generators_df.loc[gens, "building_id"] = [1, 2, 2, 4]
    edisgo.set_time_series_manual(
        generators_p=pd.DataFrame(0.0, index=timeindex, columns=gens),
        loads_p=pd.DataFrame(0.0, index=timeindex, columns=loads),
    )
    return edisgo, list(loads), list(gens)


class TestStorageOperation:
    @SYN_SETTINGS
    @given(
        fmd=st.lists(st.floats(-3.0, 3.0), min_size=1, max_size=48),
        freq=st.sampled_from([0.25, 0.5, 1.0, 2.0]),
        params=storage_params,
    )
    def test_reference_operation_soe_power_and_energy_balance(self, fmd, freq, params):
        """
        Reference operation: the SoE stays in ``[0, soe_max]``; every step obeys
        the energy balance ``dSoE = -P * eta_store * freq`` (charging, P < 0) or
        ``-P / eta_dispatch * freq`` (discharging); the storage never reverses
        the building's exchange with the grid (``P * fmd <= 0``, ``|P| <=
        |fmd|``) and respects ``-p_nom <= P <= p_nom * eta_dispatch``.
        """
        p_nom, eta_s, eta_d = params["p_nom"], params["eta_s"], params["eta_d"]
        soe_max = p_nom * params["max_hours"]
        soe_init = params["soe_share"] * soe_max
        res = _run_reference(fmd, soe_init, soe_max, p_nom, freq, eta_s, eta_d)
        power = res.storage_power.to_numpy()
        soe = res.storage_soe.to_numpy()
        fmd = np.asarray(fmd)

        assert (soe >= -STORAGE_ATOL).all()
        assert (soe <= soe_max + STORAGE_ATOL).all()
        d_soe = np.diff(np.concatenate([[soe_init], soe]))
        expected = np.where(power < 0, -power * eta_s * freq, -power / eta_d * freq)
        np.testing.assert_allclose(d_soe, expected, rtol=0, atol=STORAGE_ATOL)
        assert (power * fmd <= STORAGE_ATOL).all()
        assert (np.abs(power) <= np.abs(fmd) + STORAGE_ATOL).all()
        assert (power >= -p_nom - STORAGE_ATOL).all()
        assert (power <= p_nom * eta_d + STORAGE_ATOL).all()

    @SYN_SETTINGS
    @given(
        fmd=st.lists(st.floats(-3.0, 3.0), min_size=1, max_size=24),
        sub=st.sampled_from([2, 4]),
        params=storage_params,
    )
    def test_reference_operation_independent_of_time_resolution(self, fmd, sub, params):
        """
        Resolution invariance: splitting every hour into ``sub`` equal steps
        (same power, ``freq = 1/sub``) gives the same SoE at the end of every
        hour and the same mean power per hour.
        """
        p_nom, eta_s, eta_d = params["p_nom"], params["eta_s"], params["eta_d"]
        soe_max = p_nom * params["max_hours"]
        soe_init = params["soe_share"] * soe_max
        hourly = _run_reference(fmd, soe_init, soe_max, p_nom, 1.0, eta_s, eta_d)
        fine = _run_reference(
            np.repeat(fmd, sub), soe_init, soe_max, p_nom, 1 / sub, eta_s, eta_d
        )
        np.testing.assert_allclose(
            fine.storage_soe.to_numpy()[sub - 1 :: sub],
            hourly.storage_soe,
            atol=STORAGE_ATOL,
        )
        np.testing.assert_allclose(
            fine.storage_power.to_numpy().reshape(-1, sub).mean(axis=1),
            hourly.storage_power,
            atol=STORAGE_ATOL,
        )

    @GRID_SETTINGS
    @given(seed=seeds, pv_scale=st.floats(0.0, 0.05), load_scale=st.floats(0.0, 0.05))
    def test_apply_reference_operation_conventions(
        self, storage_grid, seed, pv_scale, load_scale
    ):
        """
        Applied to buildings: storage power follows the generator sign convention
        (charging, P < 0, only with PV surplus), the building's net exchange with
        the grid never grows (``|pv - load + P| <= |pv - load|``), the reactive
        power keeps the default storage power factor and sign, and storage units
        without PV or without load stay at zero and get no SoE column.
        """
        base, loads, gens = storage_grid
        edisgo = copy.deepcopy(base)
        rng = np.random.default_rng(seed)
        idx = edisgo.timeseries.timeindex
        edisgo.set_time_series_manual(
            generators_p=pd.DataFrame(
                pv_scale * rng.uniform(0, 1, (len(idx), len(gens))), idx, gens
            ),
            loads_p=pd.DataFrame(
                load_scale * rng.uniform(0, 1, (len(idx), len(loads))), idx, loads
            ),
        )
        soe = apply_reference_operation(edisgo)

        ts, topo = edisgo.timeseries, edisgo.topology
        p = ts.storage_units_active_power
        q = ts.storage_units_reactive_power
        assert set(soe.columns) == {"stor_pv_load", "stor_two_each"}
        for stor in ("stor_no_pv", "stor_no_load"):
            assert (p[stor] == 0).all()
        pf = edisgo.config["reactive_power_factor"]["lv_storage_unit"]
        q_sign = q_control.get_q_sign_generator(
            edisgo.config["reactive_power_mode"]["lv_storage_unit"]
        )
        for stor in soe.columns:
            building = topo.storage_units_df.at[stor, "building_id"]
            pv = ts.generators_active_power[
                topo.generators_df.index[topo.generators_df.building_id == building]
            ].sum(axis=1)
            load = ts.loads_active_power[
                topo.loads_df.index[topo.loads_df.building_id == building]
            ].sum(axis=1)
            surplus = pv - load
            assert (p[stor][p[stor] < 0].index.isin(surplus[surplus > 0].index)).all()
            assert ((surplus + p[stor]).abs() <= surplus.abs() + STORAGE_ATOL).all()
            nonzero = p[stor].abs() > 1e-9
            np.testing.assert_allclose(
                p[stor][nonzero].abs() / np.hypot(p[stor], q[stor])[nonzero],
                pf,
                rtol=1e-9,
            )
            assert (
                np.sign(q[stor][nonzero]) == q_sign * np.sign(p[stor][nonzero])
            ).all()


# ======================================================================================
# heat_pump_operation
# ======================================================================================


class TestHeatPumpOperation:
    @GRID_SETTINGS
    @given(
        seed=seeds,
        n_hp=st.integers(1, 5),
        heat_scale=st.one_of(st.just(0.0), st.floats(1e-6, 5.0)),
        k=st.floats(0.1, 10.0),
        subset_seed=seeds,
    )
    def test_uncontrolled_operation_conserves_heat(
        self, wc_grid_factory, seed, n_hp, heat_scale, k, subset_seed
    ):
        """
        Uncontrolled heat pumps: ``P_el * COP == heat demand`` elementwise,
        ``P_el >= 0`` and Q == 0; only the selected heat pumps change and all
        other load time series stay bit-identical; scaling the heat demand by
        ``k`` scales P_el by ``k``.
        """
        edisgo = wc_grid_factory()
        rng = np.random.default_rng(seed)
        idx = edisgo.timeseries.timeindex
        names = [f"hp_{i}" for i in range(n_hp)]
        cop = pd.DataFrame(rng.uniform(1.0, 6.0, (len(idx), n_hp)), idx, names)
        heat = pd.DataFrame(
            heat_scale * rng.uniform(0, 1, (len(idx), n_hp)), idx, names
        )
        edisgo.heat_pump.cop_df = cop
        edisgo.heat_pump.heat_demand_df = heat
        selected = list(
            np.random.default_rng(subset_seed).choice(
                names, size=max(1, n_hp // 2), replace=False
            )
        )
        before = edisgo.timeseries.loads_active_power.copy()
        before_q = edisgo.timeseries.loads_reactive_power.copy()

        operating_strategy(edisgo, heat_pump_names=selected)

        p = edisgo.timeseries.loads_active_power.copy()
        q = edisgo.timeseries.loads_reactive_power.copy()
        np.testing.assert_allclose(
            p[selected] * cop[selected], heat[selected], rtol=RTOL
        )
        assert (p[selected] >= 0).all().all()
        assert (q[selected] == 0).all().all()
        assert not set(names).difference(selected) & set(p.columns)
        pd.testing.assert_frame_equal(p[before.columns], before)
        pd.testing.assert_frame_equal(q[before_q.columns], before_q)

        edisgo.heat_pump.heat_demand_df = k * heat
        operating_strategy(edisgo, heat_pump_names=selected)
        np.testing.assert_allclose(
            edisgo.timeseries.loads_active_power[selected], k * p[selected], rtol=RTOL
        )


# ======================================================================================
# reinforce_measures
# ======================================================================================


def _standard_line(edisgo, grid_or_level):
    std = edisgo.config["grid_expansion_standard_equipment"]
    level = grid_or_level if isinstance(grid_or_level, str) else None
    if level is None:
        level = "mv" if grid_or_level is edisgo.topology.mv_grid else "lv"
    if level == "mv":
        return std[f"mv_line_{int(edisgo.topology.mv_grid.nominal_voltage)}kv"]
    return std["lv_line"]


def _grid_by_name(edisgo, name):
    topo = edisgo.topology
    return topo.mv_grid if name == str(topo.mv_grid) else topo.get_lv_grid(name)


#: Grids of test network 1 with at least one bus besides the station.
MULTI_BUS_GRIDS = ["MVGrid_1"] + [f"LVGrid_{i}" for i in range(1, 10)]


class TestLineMeasures:
    @GRID_SETTINGS
    @given(data=st.data())
    def test_line_overloading_measure_is_sufficient_and_minimal(
        self, wc_grid_factory, data
    ):
        """
        After ``reinforce_lines_overloading`` every critical line can carry its
        old load times ``max_rel_overload`` (``s_nom_new >= s_nom_old *
        max_rel_overload``), one parallel line less would not suffice, the line
        parameters match the (standard or kept) line type per parallel line, the
        reported changes equal the added parallel lines and no other line changes.
        """
        edisgo = wc_grid_factory()
        topo = edisgo.topology
        lines = data.draw(
            st.lists(
                st.sampled_from(sorted(topo.lines_df.index)),
                min_size=1,
                max_size=10,
                unique=True,
            )
        )
        overload = data.draw(
            st.lists(
                st.floats(1.0 + 1e-6, 6.0), min_size=len(lines), max_size=len(lines)
            )
        )
        mv_lines = topo.mv_grid.lines_df.index
        crit = pd.DataFrame(
            {
                "max_rel_overload": overload,
                "voltage_level": ["mv" if line in mv_lines else "lv" for line in lines],
            },
            index=lines,
        )
        before = topo.lines_df.copy()

        changes = measures.reinforce_lines_overloading(edisgo, crit)

        after = topo.lines_df
        others = before.index.drop(lines)
        pd.testing.assert_frame_equal(after.loc[others], before.loc[others])
        assert set(changes) == set(lines)
        for line in lines:
            old, new = before.loc[line], after.loc[line]
            rel = crit.at[line, "max_rel_overload"]
            single = new.s_nom / new.num_parallel
            assert new.s_nom >= old.s_nom * rel * (1 - RTOL)
            assert (new.num_parallel - 1) * single < old.s_nom * rel * (1 + RTOL)
            if new.type_info == old.type_info:
                assert single == pytest.approx(old.s_nom / old.num_parallel, rel=RTOL)
                assert new.r * new.num_parallel == pytest.approx(
                    old.r * old.num_parallel, rel=RTOL
                )
                assert changes[line] == new.num_parallel - old.num_parallel
            else:
                level = crit.at[line, "voltage_level"]
                assert new.type_info == _standard_line(edisgo, level)
                eq = _line_equipment(edisgo, new.type_info)
                v_nom = topo.buses_df.at[new.bus0, "v_nom"]
                assert new.r * new.num_parallel == pytest.approx(
                    eq.R_per_km * new.length, rel=RTOL
                )
                assert single == pytest.approx(
                    math.sqrt(3) * v_nom * eq.I_max_th, rel=RTOL
                )
                assert changes[line] == new.num_parallel
            assert changes[line] >= 1

    @GRID_SETTINGS
    @given(
        grid_name=st.sampled_from(MULTI_BUS_GRIDS),
        reductions=st.lists(st.floats(0.01, 0.99), min_size=2, max_size=2),
        seed=seeds,
        data=st.data(),
    )
    def test_plan_lines_on_path_reaches_target_and_matches_topology(
        self, analyzed_grid_factory, grid_name, reductions, seed, data
    ):
        """
        Planning the reinforcement of a path does not touch the topology, never
        raises a line resistance and, if it reports finite costs, lowers the
        current-weighted path resistance ``sum(r * I)`` by at least the requested
        share; a larger requested reduction never costs less; applying the plan
        with ``_reinforce_lines_on_path`` yields exactly the planned lines.
        """
        edisgo = analyzed_grid_factory()
        grid = _grid_by_name(edisgo, grid_name)
        station = grid.transformers_df.bus1.iloc[0]
        target = data.draw(st.sampled_from(sorted(grid.buses_df.index.drop(station))))
        path = nx.shortest_path(grid.graph, station, target)
        path_lines = [
            grid.graph.get_edge_data(a, b)["branch_name"]
            for a, b in zip(path[:-1], path[1:])
        ]
        current = pd.Series(
            np.random.default_rng(seed).uniform(1e-3, 1.0, len(path_lines)),
            index=path_lines,
        )
        standard = _standard_line(edisgo, grid)
        lines_before = edisgo.topology.lines_df.copy()
        r_old = lines_before.r[path_lines]

        plans = {}
        for red in sorted(reductions):
            changes, planned, costs = measures._plan_lines_on_path(
                edisgo, grid, path_lines, standard, red, current
            )
            pd.testing.assert_frame_equal(edisgo.topology.lines_df, lines_before)
            assert (planned.r <= r_old * (1 + RTOL)).all()
            assert set(changes) <= set(path_lines)
            assert all(v >= 1 for v in changes.values())
            if np.isfinite(costs):
                assert (planned.r * current).sum() <= (1 - red) * (
                    r_old * current
                ).sum() * (1 + RTOL)
            plans[red] = (planned, costs)
        low, high = sorted(reductions)
        assert plans[low][1] <= plans[high][1]

        planned = plans[low][0]
        measures._reinforce_lines_on_path(
            edisgo, grid, path_lines, standard, 2 * low, current
        )
        after = edisgo.topology.lines_df.loc[path_lines]
        np.testing.assert_allclose(after.r, planned.r, rtol=RTOL)
        assert (after.num_parallel == planned.num_parallel).all()
        assert (after.type_info == planned.type_info).all()

    @GRID_SETTINGS
    @given(
        grid_name=st.sampled_from(MULTI_BUS_GRIDS),
        cheaper=st.booleans(),
        data=st.data(),
    )
    def test_line_voltage_measure_keeps_grid_radial(
        self, analyzed_grid_factory, grid_name, cheaper, data
    ):
        """
        ``reinforce_lines_voltage_issues`` keeps every grid a connected radial
        tree with the same buses and number of lines; only reported lines change;
        a reconnected line now starts at the station and its length equals the
        former path length from the station to its far end; reinforced lines
        never get a higher resistance; no house connection line is touched.
        """
        edisgo = analyzed_grid_factory()
        topo = edisgo.topology
        grid = _grid_by_name(edisgo, grid_name)
        station = grid.transformers_df.bus1.iloc[0]
        nodes = data.draw(
            st.lists(
                st.sampled_from(sorted(grid.buses_df.index.drop(station))),
                min_size=1,
                max_size=4,
                unique=True,
            )
        )
        crit = pd.DataFrame(
            {
                "abs_max_voltage_dev": data.draw(
                    st.lists(
                        st.floats(1e-4, 0.1), min_size=len(nodes), max_size=len(nodes)
                    )
                ),
                "time_index": data.draw(
                    st.lists(
                        st.sampled_from(list(edisgo.timeseries.timeindex)),
                        min_size=len(nodes),
                        max_size=len(nodes),
                    )
                ),
            },
            index=nodes,
        )
        graph_before = grid.graph
        lines_before = topo.lines_df.copy()
        buses_before = topo.buses_df.copy()

        changes = measures.reinforce_lines_voltage_issues(
            edisgo, grid, crit, choose_cheaper_measure=cheaper
        )

        after = topo.lines_df
        pd.testing.assert_frame_equal(topo.buses_df, buses_before)
        assert sorted(after.index) == sorted(lines_before.index)
        for g in topo.grids:
            assert_radial(g.graph, str(g))
        assert_radial(topo.to_graph(), "topology")

        cols = ["bus0", "bus1", "length", "r", "num_parallel", "type_info"]
        changed = after.index[
            (after[cols] != lines_before.loc[after.index, cols]).any(axis=1)
        ]
        assert set(changed) <= set(changes)
        assert all(v >= 1 for v in changes.values())
        in_building = topo.buses_df.in_building.fillna(False).astype(bool)
        for line in changed:
            old, new = lines_before.loc[line], after.loc[line]
            old_buses, new_buses = {old.bus0, old.bus1}, {new.bus0, new.bus1}
            assert not in_building[list(old_buses | new_buses)].any(), line
            if old_buses != new_buses:
                assert station in new_buses
                (far_end,) = new_buses - {station}
                assert far_end in old_buses
                assert new.length == pytest.approx(
                    nx.shortest_path_length(graph_before, station, far_end, "length"),
                    rel=RTOL,
                )
                assert new.type_info == _standard_line(edisgo, grid)
            else:
                assert new.r <= old.r * (1 + RTOL)


class TestStationMeasures:
    @GRID_SETTINGS
    @given(rounds=st.integers(1, 3), data=st.data())
    def test_station_overloading_measure_covers_missing_power(
        self, wc_grid_factory, rounds, data
    ):
        """
        After a station overloading measure each critical station has at least
        ``sum(s_nom_old) + s_missing``; if transformers are replaced, the number
        of standard transformers is minimal; the reported added/removed
        transformers equal the difference of the transformer tables (checked in
        the first call, name reuse in later calls is covered by
        ``test_added_transformer_names_are_new``), all new transformers connect
        the station's buses, transformer names stay unique (also over repeated
        calls) and other stations are untouched.
        """
        edisgo = wc_grid_factory()
        topo = edisgo.topology
        std = edisgo.config["grid_expansion_standard_equipment"]
        for round_ in range(rounds):
            level = data.draw(st.sampled_from(["mv", "lv"]))
            if level == "mv":
                grids = [topo.mv_grid]
                table = lambda: topo.transformers_hvmv_df  # noqa: E731
                func = measures.reinforce_hv_mv_station_overloading
                s_std = topo.equipment_data["mv_transformers"].at[
                    std["hv_mv_transformer"], "S_nom"
                ]
            else:
                grids = data.draw(
                    st.lists(
                        st.sampled_from(list(topo.lv_grids)),
                        min_size=1,
                        max_size=4,
                        unique_by=lambda g: g.id,
                    )
                )
                table = lambda: topo.transformers_df  # noqa: E731
                func = measures.reinforce_mv_lv_station_overloading
                s_std = topo.equipment_data["lv_transformers"].at[
                    std["mv_lv_transformer"], "S_nom"
                ]
            shares = data.draw(
                st.lists(st.floats(1e-3, 5.0), min_size=len(grids), max_size=len(grids))
            )
            old = {g.station_name: g.transformers_df.copy() for g in grids}
            crit = pd.DataFrame(
                {
                    "s_missing": [
                        s * g.transformers_df.s_nom.sum() for s, g in zip(shares, grids)
                    ],
                    "time_index": edisgo.timeseries.timeindex[0],
                    "grid": grids,
                },
                index=[g.station_name for g in grids],
            )
            table_before = table().copy()

            changes = func(edisgo, crit)

            assert not table().index.duplicated().any()
            touched = set()
            for grid in grids:
                name = grid.station_name
                before, now = old[name], grid.transformers_df
                touched |= set(before.index) | set(now.index)
                need = before.s_nom.sum() + crit.at[name, "s_missing"]
                assert now.s_nom.sum() >= need * (1 - RTOL)
                added = set(changes["added"][name])
                removed = set(changes["removed"].get(name, []))
                assert set(now.index) - set(before.index) <= added <= set(now.index)
                assert set(before.index) - set(now.index) <= removed
                assert removed <= set(before.index)
                if round_ == 0:
                    assert set(now.index) - set(before.index) == added
                    assert set(before.index) - set(now.index) == removed
                assert (now.bus0 == before.bus0.iat[0]).all()
                assert (now.bus1 == before.bus1.iat[0]).all()
                if name in changes["removed"]:
                    n = len(now)
                    assert (
                        (n - 1) * s_std
                        < need * (1 + RTOL)
                        <= n * s_std * (1 + 2 * RTOL)
                    )
                else:
                    assert len(changes["added"][name]) == 1
            untouched = table_before.index.difference(list(touched))
            pd.testing.assert_frame_equal(
                table().loc[untouched], table_before.loc[untouched]
            )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Bug: replacing the transformers of a station that was reinforced "
            "before reuses the names '..._reinforced_<i>' of removed transformers; "
            "grid_expansion_costs then treats these new transformers as "
            "added-and-removed and does not cost them"
        ),
    )
    def test_added_transformer_names_are_new(self, wc_grid):
        """
        Reinforcing a station twice (first one more transformer, then replacement
        by standard transformers) only reports transformers as added that did not
        exist before, so the costs cover every transformer the station gained.
        """
        edisgo = wc_grid
        grid = edisgo.topology.get_lv_grid(1)
        names_before = set(grid.transformers_df.index)
        for step, s_missing in ((1, 0.05), (2, 2.0)):
            crit = pd.DataFrame(
                {
                    "s_missing": [s_missing],
                    "time_index": [edisgo.timeseries.timeindex[0]],
                    "grid": [grid],
                },
                index=[grid.station_name],
            )
            existing = set(grid.transformers_df.index)
            changes = measures.reinforce_mv_lv_station_overloading(edisgo, crit)
            for mode in ("added", "removed"):
                _add_transformer_changes_to_equipment_changes(
                    edisgo, changes, step, mode
                )
            assert not set(changes["added"][grid.station_name]) & existing
        costs = costs_mod.grid_expansion_costs(edisgo)
        new_trafos = set(grid.transformers_df.index) - names_before
        assert new_trafos <= set(costs.index)

    @GRID_SETTINGS
    @given(data=st.data())
    def test_station_voltage_measure_adds_one_standard_transformer(
        self, wc_grid_factory, data
    ):
        """
        The station voltage measure adds exactly one standard transformer between
        the station's buses to every critical station and changes nothing else.
        """
        edisgo = wc_grid_factory()
        topo = edisgo.topology
        grids = data.draw(
            st.lists(
                st.sampled_from(list(topo.lv_grids)),
                min_size=1,
                max_size=5,
                unique_by=lambda g: g.id,
            )
        )
        crit = pd.DataFrame(
            {
                "abs_max_voltage_dev": 0.01,
                "time_index": edisgo.timeseries.timeindex[0],
                "lv_grid_id": [float(g.id) for g in grids],
            },
            index=[g.transformers_df.bus1.iat[0] for g in grids],
        )
        before = topo.transformers_df.copy()
        std = edisgo.config["grid_expansion_standard_equipment"]["mv_lv_transformer"]

        changes = measures.reinforce_mv_lv_station_voltage_issues(edisgo, crit)

        after = topo.transformers_df
        added = after.index.difference(before.index)
        assert len(added) == len(grids)
        assert not after.index.duplicated().any()
        pd.testing.assert_frame_equal(after.loc[before.index], before)
        assert sorted(added) == sorted(sum(changes["added"].values(), []))
        assert (after.loc[added, "type_info"] == std).all()
        for grid in grids:
            new = changes["added"][str(grid)]
            assert len(new) == 1
            ref = before.loc[grid.transformers_df.index.intersection(before.index)]
            assert after.at[new[0], "bus0"] == ref.bus0.iat[0]
            assert after.at[new[0], "bus1"] == ref.bus1.iat[0]


class TestGridSeparation:
    @GRID_SETTINGS
    @given(grid_id=st.integers(1, 9))
    def test_separate_lv_grid_keeps_both_parts_radial(self, wc_grid_factory, grid_id):
        """
        ``separate_lv_grid`` either leaves the topology untouched or splits the
        grid into two connected radial grids that together hold the old buses
        plus one new station bus; the number of lines and the connection of all
        loads, generators and storage units stay the same, the new station is at
        least as strong as the old one and all other LV grids are untouched.
        """
        edisgo = wc_grid_factory()
        topo = edisgo.topology
        grid = topo.get_lv_grid(grid_id)
        old_buses = set(grid.buses_df.index)
        old_rating = grid.transformers_df.s_nom.sum()
        buses_before, lines_before = topo.buses_df.copy(), topo.lines_df.copy()
        comps_before = {
            c: getattr(topo, f"{c}_df").bus.copy()
            for c in ("loads", "generators", "storage_units")
        }
        others = {
            g.id: (set(g.buses_df.index), g.lines_df.copy())
            for g in topo.lv_grids
            if g.id != grid_id
        }

        trafo_changes, line_changes = measures.separate_lv_grid(edisgo, grid)

        if not line_changes:
            assert trafo_changes == {}
            pd.testing.assert_frame_equal(topo.buses_df, buses_before)
            pd.testing.assert_frame_equal(topo.lines_df, lines_before)
            return
        (new_name,) = trafo_changes["added"]
        new_grid = topo.get_lv_grid(new_name)
        old_grid = topo.get_lv_grid(grid_id)
        new_station = new_grid.transformers_df.bus1.iat[0]
        assert len(topo.buses_df) == len(buses_before) + 1
        assert len(topo.lines_df) == len(lines_before)
        assert set(old_grid.buses_df.index) | set(new_grid.buses_df.index) == (
            old_buses | {new_station}
        )
        assert not set(old_grid.buses_df.index) & set(new_grid.buses_df.index)
        assert_radial(old_grid.graph, "old part")
        assert_radial(new_grid.graph, "new part")
        assert_radial(topo.to_graph(), "topology")
        assert new_grid.transformers_df.s_nom.sum() >= old_rating
        # adding and removing lines drops the index name of the component
        # tables; the name carries no information and is ignored below
        for comp, buses in comps_before.items():
            pd.testing.assert_series_equal(
                getattr(topo, f"{comp}_df").bus, buses, check_names=False
            )
        for gid, (buses, lines) in others.items():
            g = topo.get_lv_grid(gid)
            assert set(g.buses_df.index) == buses
            pd.testing.assert_frame_equal(g.lines_df, lines, check_names=False)
        for line, n in line_changes.items():
            assert topo.lines_df.at[line, "num_parallel"] == n
            assert new_station in topo.lines_df.loc[line, ["bus0", "bus1"]].values

    @pytest.mark.slow
    @settings(
        deadline=None,
        max_examples=examples(6),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        thresholds=st.lists(st.floats(0.05, 3.0), min_size=2, max_size=2),
        k=st.floats(1.0, 20.0),
    )
    def test_run_separate_lv_grids_reaches_threshold(
        self, wc_grid_factory, thresholds, k
    ):
        """
        After ``run_separate_lv_grids`` every LV grid whose residual apparent
        power exceeds ``threshold * sum(s_nom)`` cannot be separated any further;
        a higher threshold never yields more LV grids.
        """
        counts = []
        for threshold in sorted(thresholds):
            edisgo = wc_grid_factory()
            edisgo.timeseries.scale_timeseries(k, k)
            run_separate_lv_grids(edisgo, threshold=threshold)
            ts, topo = edisgo.timeseries, edisgo.topology
            grids = list(topo.mv_grid.lv_grids)
            counts.append(len(grids))
            for grid in grids:
                residual = {}
                for attr in ("active_power", "reactive_power"):
                    residual[attr] = (
                        getattr(ts, f"loads_{attr}")[grid.loads_df.index].sum(axis=1)
                        - getattr(ts, f"generators_{attr}")[
                            grid.generators_df.index
                        ].sum(axis=1)
                        - getattr(ts, f"storage_units_{attr}")[
                            grid.storage_units_df.index
                        ].sum(axis=1)
                    )
                worst = np.hypot(
                    residual["active_power"], residual["reactive_power"]
                ).max()
                if worst > threshold * grid.transformers_df.s_nom.sum():
                    probe = copy.deepcopy(edisgo)
                    _, lines = measures.separate_lv_grid(
                        probe, probe.topology.get_lv_grid(grid.id)
                    )
                    assert lines == {}, f"{grid} could still be separated"
        assert counts[1] <= counts[0]


# ======================================================================================
# charging_strategies
# ======================================================================================

STEPSIZE = 15  # minutes, as in the SimBEV example scenario


@st.composite
def charging_processes(draw):
    """Feasible SimBEV-like charging processes (demand fits into the stay)."""
    n = draw(st.integers(1, 15))
    rows = []
    for _ in range(n):
        cap = draw(st.sampled_from([3.7, 11.0, 22.0, 50.0, 150.0]))
        park_time = draw(st.integers(1, 100))
        start = draw(st.integers(0, 500))
        fill = draw(st.floats(0.0, 1.0))
        rows.append(
            {
                "use_case": draw(st.sampled_from(["home", "work", "public", "hpc"])),
                "nominal_charging_capacity_kW": cap,
                "chargingdemand_kWh": fill * cap * park_time * STEPSIZE / 60,
                "park_time_timesteps": park_time,
                "park_start_timesteps": start,
                "park_end_timesteps": start + park_time - 1,
            }
        )
    return pd.DataFrame(rows)


def _em_stub():
    return SimpleNamespace(electromobility=SimpleNamespace(stepsize=STEPSIZE))


def _harmonize(df, strategy=None, threshold=0.2, factor=0.1, eta=0.9):
    return harmonize_charging_processes_df(
        df,
        _em_stub(),
        len_ts=10_000,
        timestamp_share_threshold=threshold,
        strategy=strategy,
        minimum_charging_capacity_factor=factor,
        eta_cp=eta,
    )


class TestChargingProcesses:
    @SYN_SETTINGS
    @given(
        df=charging_processes(),
        thresholds=st.lists(st.floats(0.0, 0.99), min_size=2, max_size=2),
        factor=st.floats(0.05, 1.0),
        eta=st.floats(0.5, 1.0),
    )
    def test_harmonized_demand_is_rounded_to_whole_steps(
        self, df, thresholds, factor, eta
    ):
        """
        The minimum charging time is the exact time rounded to whole steps (up
        iff the fractional part reaches the threshold) and fits into the stay;
        the harmonized demand is that time at nominal power; a higher threshold
        never increases the harmonized demand; 'reduced' keeps the energy and
        charges with at least ``factor * nominal power`` (stays of >= 2 steps);
        'residual' sorts by non-negative flexibility.
        """
        low, high = sorted(thresholds)
        h = _harmonize(df, threshold=low, eta=eta)
        exact = df.chargingdemand_kWh / df.nominal_charging_capacity_kW * 60 / STEPSIZE
        frac = exact - np.floor(exact)
        expected = np.where(frac >= low, np.ceil(exact), np.floor(exact))
        np.testing.assert_array_equal(h.minimum_charging_time, expected)
        assert (h.minimum_charging_time <= df.park_time_timesteps).all()
        np.testing.assert_allclose(
            h.harmonized_chargingdemand,
            expected * df.nominal_charging_capacity_kW * STEPSIZE / 60,
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            h.nominal_charging_capacity_mva,
            df.nominal_charging_capacity_kW / (1000 * eta),
            rtol=RTOL,
        )
        h_high = _harmonize(df, threshold=high, eta=eta)
        assert (
            h_high.harmonized_chargingdemand <= h.harmonized_chargingdemand * (1 + RTOL)
        ).all()

        red = _harmonize(df, strategy="reduced", threshold=low, factor=factor)
        sel = (red.harmonized_chargingdemand > 0) & (red.park_time_timesteps >= 2)
        red = red[sel]
        np.testing.assert_allclose(
            red.reduced_charging_capacity * red.reduced_charging_time * STEPSIZE / 60,
            red.harmonized_chargingdemand,
            rtol=RTOL,
        )
        assert (
            red.reduced_charging_capacity
            >= factor * red.nominal_charging_capacity_kW * (1 - RTOL)
        ).all()

        res = _harmonize(df, strategy="residual", threshold=low)
        assert (res.flex_time >= 0).all()
        np.testing.assert_array_equal(
            res.flex_time, res.park_time_timesteps - res.minimum_charging_time
        )
        keys = list(
            zip(res.flex_time, res.park_start_timesteps, res.park_end_timesteps)
        )
        assert keys == sorted(keys)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Bug: 'reduced' limits the charging time to park_end - park_start, one "
            "step less than the stay (park_time_timesteps); for short stays the "
            "reduced charging power exceeds the nominal power (or is inf/NaN)"
        ),
    )
    @SYN_SETTINGS
    @given(df=charging_processes(), factor=st.floats(0.05, 1.0))
    def test_reduced_charging_power_never_exceeds_nominal_power(self, df, factor):
        """
        'reduced' charging fits into the stay (``0 < reduced_charging_time <=
        park_time_timesteps``) and never charges faster than the nominal
        charging power, for every process with a non-zero harmonized demand.
        """
        red = _harmonize(df, strategy="reduced", factor=factor)
        red = red[red.harmonized_chargingdemand > 0]
        assert (red.reduced_charging_time > 0).all()
        assert (red.reduced_charging_time <= red.park_time_timesteps).all()
        assert (
            red.reduced_charging_capacity
            <= red.nominal_charging_capacity_kW * (1 + RTOL)
        ).all()


@pytest.fixture(scope="module")
def em_grid():
    """Test network 2 on the SimBEV 15 min grid with the example charging data."""
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
    edisgo.set_timeindex(
        pd.date_range("1/1/2011", periods=7 * 24 * 60 // STEPSIZE, freq="15min")
    )
    edisgo.import_electromobility(
        data_source="directory",
        charging_processes_dir=pytest.simbev_example_scenario_path,
        potential_charging_points_dir=pytest.tracbev_example_scenario_path,
    )
    return edisgo


class TestChargingStrategies:
    @pytest.mark.parametrize("strategy", ["dumb", "reduced", "residual"])
    def test_charging_within_stays_and_conserves_energy(self, em_grid, strategy):
        """
        For every charging park: the charging power is >= 0, at every time step
        at most the summed nominal power (incl. charging efficiency) of the cars
        parked there, i.e. zero outside all stays, and the charged energy equals
        the harmonized charging demand divided by the efficiency (all stays lie
        within the simulated week).
        """
        edisgo = copy.deepcopy(em_grid)
        em = edisgo.electromobility
        assert em.stepsize == STEPSIZE
        charging_strategy(edisgo, strategy=strategy)

        eta = em.eta_charging_points
        n_steps = len(edisgo.timeseries.timeindex)
        power = edisgo.timeseries.loads_active_power
        processes = em.charging_processes_df
        parks = em.integrated_charging_parks_df
        assert len(parks) > 0
        for park_id, cp in parks.edisgo_id.items():
            p = power[cp].to_numpy()
            procs = processes[processes.charging_park_id == park_id]
            assert len(procs) > 0
            assert (procs.park_end_timesteps < n_steps).all()
            parked = np.zeros(n_steps)
            for _, proc in procs.iterrows():
                start, end = (
                    int(proc.park_start_timesteps),
                    int(proc.park_end_timesteps),
                )
                parked[start : end + 1] += proc.nominal_charging_capacity_kW / (
                    1000 * eta
                )
            assert (p >= 0).all()
            assert (p <= parked * (1 + RTOL)).all(), cp
            exact = procs.chargingdemand_kWh / procs.nominal_charging_capacity_kW * 4
            frac = exact - np.floor(exact)
            steps = np.where(frac >= 0.2, np.ceil(exact), np.floor(exact))
            energy = (steps * procs.nominal_charging_capacity_kW / 4).sum()
            assert p.sum() * STEPSIZE / 60 == pytest.approx(
                energy / (1000 * eta), rel=RTOL
            )


# ======================================================================================
# reinforce_grid
# ======================================================================================


def _change_levels(edisgo, changes):
    """Classify the rows of ``equipment_changes`` by the level they touch."""
    topo = edisgo.topology
    mv_lines = topo.mv_grid.lines_df.index
    levels = set()
    for idx in changes.index:
        if idx in topo.lines_df.index:
            levels.add("mv_line" if idx in mv_lines else "lv_line")
        elif str(idx).startswith(str(topo.mv_grid)):
            levels.add("hv_mv_station")
        else:
            levels.add("mv_lv_station")
    return levels


class TestReinforceGrid:
    @pytest.mark.parametrize(
        "kwargs, allowed",
        [
            ({"mode": "mv"}, {"mv_line", "hv_mv_station"}),
            ({"mode": "mvlv"}, {"mv_line", "hv_mv_station", "mv_lv_station"}),
            ({"mode": "lv"}, {"lv_line", "mv_lv_station"}),
            ({"mode": "lv", "lv_grid_id": 5}, {"lv_line"}),
        ],
    )
    def test_reinforce_mode_only_touches_its_levels(self, wc_grid, kwargs, allowed):
        """
        The reinforcement mode isolates the levels: 'mv' changes only MV lines
        and the HV/MV station, 'mvlv' additionally MV/LV stations, 'lv' only LV
        lines and MV/LV stations, and 'lv' with an LV grid id only lines of that
        grid.
        """
        results = wc_grid.reinforce(**kwargs)
        changes = results.equipment_changes
        assert not changes.empty
        assert _change_levels(wc_grid, changes) <= allowed
        if "lv_grid_id" in kwargs:
            grid = wc_grid.topology.get_lv_grid(kwargs["lv_grid_id"])
            assert set(changes.index) <= set(grid.lines_df.index)

    @SYN_SETTINGS
    @given(run_ids=st.lists(st.integers(0, 50), max_size=20))
    def test_next_run_id_is_fresh(self, run_ids):
        """
        The next reinforcement run id is 0 for an empty log and otherwise larger
        than every id in the log, so runs never share an id.
        """
        log = pd.DataFrame({"run_id": run_ids})
        stub = SimpleNamespace(results=SimpleNamespace(reinforce_log=log))
        run_id = _next_reinforce_log_run_id(stub)
        assert run_id == (max(run_ids) + 1 if run_ids else 0)
