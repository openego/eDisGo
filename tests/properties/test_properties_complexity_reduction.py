"""
Property-based tests for the spatial and temporal complexity reduction
(:mod:`edisgo.tools.spatial_complexity_reduction`,
:mod:`edisgo.tools.temporal_complexity_reduction`).

The spatial tests reduce copies of test network 1 and check what a reduction
must keep (injections, connectivity, line ratings, an unchanged grid for the
identity busmap). The temporal tests use a time series with several hourly time
steps built from the worst cases, synthetic data for the interval helpers and
stubs for the power flow where only the selection logic is tested. Each test
states the property it checks.
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
from threadpoolctl import threadpool_limits

from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt.costs import line_expansion_costs
from edisgo.network import overlying_grid
from edisgo.tools import spatial_complexity_reduction as spatial
from edisgo.tools import temporal_complexity_reduction as temporal
from edisgo.tools.pseudo_coordinates import make_pseudo_coordinates

from .conftest import (
    POWER_ATOL,
    TIMESERIES_ATTRIBUTES,
    examples,
    inject_power_flow_results,
    limit_violations,
    undefined_references,
)

#: Settings for synthetic tests (milliseconds per example).
SYN_SETTINGS = settings(deadline=None, max_examples=examples(60))

#: Absolute tolerance in MW/Mvar for sums of the same component powers in another
#: order or grouping (~200 terms of at most a few MW, rounding ~1e-15 each).
SUM_ATOL = 1e-9

#: Absolute tolerance in p.u. for voltages of two power flows on identical grids
#: whose buses are ordered differently: both runs stop at a mismatch below 1e-6
#: p.u. (POWER_ATOL reasoning in conftest), so the voltages agree to well below
#: 1e-6.
VOLTAGE_ATOL = 1e-6

#: Number of hourly time steps of the multi-step time series.
N_STEPS = 72

#: Spatial reductions to test: (mode, cluster_area, aggregation_mode).
#: Reason of the cluster name collision between MV grid and LV grid with equal id.
NAME_COLLISION = (
    "Bug: make_busmap_feeders/make_busmap_main_feeders name clusters "
    "'Bus_mvgd_<id>_F<f>_B<b>' without '_lvgd_' if the grid id equals the MV grid "
    "id (check 'mvgd_id == grid_id'), so clusters of LV grid 1 get the same names "
    "as clusters of MV grid 1 and MV and LV buses are merged into one bus"
)

REDUCTIONS = [
    ("kmeansdijkstra", "feeder", False),
    ("kmeansdijkstra", "feeder", True),
    ("kmeans", "grid", False),
    ("kmeansdijkstra", "main_feeder", True),
    pytest.param(
        "equidistant_nodes",
        "main_feeder",
        False,
        marks=pytest.mark.xfail(
            strict=True, reason=NAME_COLLISION + " (here the MV grid falls apart)"
        ),
    ),
    ("aggregate_to_main_feeder", "main_feeder", False),
]


def _collision(mode, area):
    return pytest.param(
        mode, area, marks=pytest.mark.xfail(strict=True, reason=NAME_COLLISION)
    )


#: Modes whose clusters are connected along the grid, so radiality is kept.
RADIAL_MODES = ("kmeansdijkstra", "equidistant_nodes", "aggregate_to_main_feeder")


# ======================================================================================
# helpers
# ======================================================================================


def multi_step_grid(edisgo, n_steps=N_STEPS, seed=1):
    """
    Replace the worst-case time series by ``n_steps`` hourly time steps.

    Every time step is one of the four worst cases scaled by a random factor in
    [0.4, 1.2], so the grid has overloadings and voltage issues of different
    magnitude at many time steps.
    """
    rng = np.random.default_rng(seed)
    ts = edisgo.timeseries
    cases = rng.integers(0, len(ts.timeindex), n_steps)
    factors = rng.uniform(0.4, 1.2, n_steps)
    index = pd.date_range("2011-01-03", periods=n_steps, freq="h")
    for attr in ts._attributes:
        df = getattr(ts, attr)
        if df.empty:
            continue
        values = df.to_numpy()[cases] * factors[:, None]
        setattr(ts, attr, pd.DataFrame(values, index=index, columns=df.columns))
    ts.timeindex = index
    return edisgo


def total_powers(edisgo):
    """Total active and reactive power per component type and time step."""
    ts = edisgo.timeseries
    return pd.DataFrame(
        {attr: getattr(ts, attr).sum(axis=1) for attr in TIMESERIES_ATTRIBUTES}
    )


def strip_virtual(name):
    return name[len("virtual_") :] if name.startswith("virtual_") else name


@pytest.fixture(scope="module")
def _multi_step_analyzed(_wc_grid_original):
    edisgo = multi_step_grid(copy.deepcopy(_wc_grid_original))
    edisgo.analyze()
    return edisgo


@pytest.fixture
def multi_step_analyzed(_multi_step_analyzed):
    """Test network 1 with N_STEPS hourly time steps after ``analyze()``."""
    return copy.deepcopy(_multi_step_analyzed)


@pytest.fixture(scope="module")
def _pseudo_grid(_wc_grid_original):
    edisgo = copy.deepcopy(_wc_grid_original)
    make_pseudo_coordinates(edisgo)
    return edisgo


@pytest.fixture
def pseudo_grid(_pseudo_grid):
    """Test network 1 with LV pseudo coordinates (as used by the reduction)."""
    return copy.deepcopy(_pseudo_grid)


# ======================================================================================
# spatial complexity reduction
# ======================================================================================


class TestSpatialComplexityReduction:
    @pytest.mark.parametrize("mode, cluster_area, aggregation_mode", REDUCTIONS)
    def test_reduction_conserves_injections_and_connectivity(
        self, wc_grid, mode, cluster_area, aggregation_mode
    ):
        """
        The reduction keeps the total load, generation and storage power of every
        time step and the totals of p_set, annual consumption and p_nom; it
        reduces the buses, keeps transformers and storage units, leaves no
        dangling reference, keeps the grid connected (and radial for cluster
        modes along the grid), and every new line connects the clusters of its
        old lines and has their summed s_nom and number of parallel lines.
        """
        orig = copy.deepcopy(wc_grid)
        busmap, linemap = spatial.spatial_complexity_reduction(
            wc_grid,
            mode=mode,
            cluster_area=cluster_area,
            reduction_factor=0.25,
            aggregation_mode=aggregation_mode,
        )
        topo, otopo = wc_grid.topology, orig.topology

        pd.testing.assert_frame_equal(
            total_powers(wc_grid), total_powers(orig), atol=SUM_ATOL, rtol=0
        )
        for df, col in (
            ("loads_df", "p_set"),
            ("loads_df", "annual_consumption"),
            ("generators_df", "p_nom"),
            ("storage_units_df", "p_nom"),
        ):
            assert getattr(topo, df)[col].sum() == pytest.approx(
                getattr(otopo, df)[col].sum(), rel=1e-12
            ), (df, col)

        assert len(topo.buses_df) < len(otopo.buses_df)
        assert len(topo.transformers_df) == len(otopo.transformers_df)
        assert len(topo.storage_units_df) == len(otopo.storage_units_df)
        assert undefined_references(topo) == []
        assert nx.is_connected(topo.to_graph())
        if mode in RADIAL_MODES:
            for grid in topo.grids:
                assert nx.is_tree(grid.graph), str(grid)

        new_bus = busmap.new_bus
        slack = otopo.transformers_hvmv_df.bus1.iloc[0]
        mapped = linemap.new_line_name if not linemap.empty else pd.Series()
        for name, line in otopo.lines_df.iterrows():
            nb0, nb1 = new_bus[line.bus0], new_bus[line.bus1]
            if name in mapped.index:
                new = topo.lines_df.loc[mapped[name]]
                assert {new.bus0, new.bus1} == {nb0, nb1}, name
            else:
                # dropped: both ends in one cluster, or short-cut of a circuit
                # breaker or of the virtual slack bus
                assert strip_virtual(nb0) == strip_virtual(
                    nb1
                ) or f"virtual_{slack}" in (nb0, nb1), name
        for new_name, old in mapped.groupby(mapped).groups.items():
            assert topo.lines_df.at[new_name, "s_nom"] == pytest.approx(
                otopo.lines_df.loc[old, "s_nom"].sum(), rel=1e-12
            )
            assert topo.lines_df.at[new_name, "num_parallel"] == (
                otopo.lines_df.loc[old, "num_parallel"].sum()
            )
        assert set(topo.lines_df.index) == set(mapped)

    @pytest.mark.parametrize(
        "mode, cluster_area",
        [
            ("kmeans", "grid"),
            ("kmeansdijkstra", "grid"),
            _collision("kmeans", "feeder"),
            _collision("kmeansdijkstra", "feeder"),
            _collision("kmeansdijkstra", "main_feeder"),
            _collision("equidistant_nodes", "main_feeder"),
            ("aggregate_to_main_feeder", "main_feeder"),
        ],
    )
    def test_busmap_contract(self, pseudo_grid, mode, cluster_area):
        """
        The busmap maps every bus exactly once to finite WGS84 coordinates, never
        clusters buses of different grids or voltage levels, maps each station
        bus to itself and is deterministic.
        """
        topo = pseudo_grid.topology
        kwargs = dict(mode=mode, cluster_area=cluster_area, reduction_factor=0.25)
        # one thread as in spatial_complexity_reduction: k-means with all cores of
        # a busy server is two orders of magnitude slower
        with threadpool_limits(1):
            busmap = spatial.make_busmap(copy.deepcopy(pseudo_grid), **kwargs)
            again = spatial.make_busmap(copy.deepcopy(pseudo_grid), **kwargs)
        pd.testing.assert_frame_equal(busmap, again)

        assert busmap.index.is_unique
        assert set(busmap.index) == set(topo.buses_df.index)
        coords = busmap[["new_x", "new_y"]].astype(float)
        assert np.isfinite(coords.to_numpy()).all()
        assert coords.new_x.abs().max() <= 180 and coords.new_y.abs().max() <= 90

        buses = topo.buses_df.loc[busmap.index]
        groups = pd.DataFrame(
            {
                "new_bus": busmap.new_bus.map(strip_virtual),
                "v_nom": buses.v_nom,
                "lv_grid_id": buses.lv_grid_id.fillna(-1),
            }
        ).groupby("new_bus")
        assert (groups.v_nom.nunique() == 1).all()
        assert (groups.lv_grid_id.nunique() == 1).all()
        for grid in topo.grids:
            station = grid.station.index[0]
            assert busmap.at[station, "new_bus"] == station, str(grid)

    @pytest.mark.parametrize(
        "mode, cluster_area",
        [
            ("kmeans", "grid"),
            pytest.param(
                "kmeansdijkstra",
                "feeder",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Bug: make_busmap_feeders replaces topology.buses_df by a "
                    "copy with coordinates in EPSG:3035 and never transforms back",
                ),
            ),
            pytest.param(
                "kmeansdijkstra",
                "main_feeder",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Bug: make_busmap_main_feeders replaces topology.buses_df "
                    "by a copy with coordinates in EPSG:3035 and never transforms back",
                ),
            ),
            ("aggregate_to_main_feeder", "main_feeder"),
        ],
    )
    def test_make_busmap_leaves_input_unchanged(self, pseudo_grid, mode, cluster_area):
        """Creating a busmap does not change the buses (coordinates) of the grid."""
        before = pseudo_grid.topology.buses_df.copy()
        with threadpool_limits(1):
            spatial.make_busmap(
                pseudo_grid, mode=mode, cluster_area=cluster_area, reduction_factor=0.25
            )
        pd.testing.assert_frame_equal(
            pseudo_grid.topology.buses_df[before.columns], before
        )

    def test_identity_busmap_keeps_grid_and_power_flow(self, wc_grid):
        """
        With MV and LV pseudo coordinates (bus distance * detour factor = line
        length), applying the identity busmap keeps every line's length, r, x and
        s_nom, and the power flow gives the same voltages, line loads and slack
        power (rms of compare_voltage / compare_apparent_power ~ 0).
        """
        make_pseudo_coordinates(wc_grid, mv_coordinates=True)
        topo = wc_grid.topology
        # test network 1 stores r and x of an older cable data set; use the
        # equipment values so that the recomputed impedances are comparable
        equipment = pd.concat(
            [
                topo.equipment_data[k]
                for k in ("mv_overhead_lines", "mv_cables", "lv_cables")
            ]
        )
        lines = topo.lines_df
        v_nom = lines.bus0.map(topo.buses_df.v_nom)
        for name, line in lines.iterrows():
            # same lookup as apply_busmap: type at the grid voltage, else any
            same_u = equipment[equipment.U_n == v_nom[name]]
            data = (
                same_u.loc[line.type_info]
                if line.type_info in same_u.index
                else equipment.loc[line.type_info]
            )
            lines.at[name, "r"] = data.R_per_km * line.length / line.num_parallel
            lines.at[name, "x"] = (
                2 * math.pi * 50 * data.L_per_km * 1e-3 * line.length
            ) / line.num_parallel
        orig = copy.deepcopy(wc_grid)
        busmap = pd.DataFrame(
            {
                "new_bus": topo.buses_df.index,
                "new_x": topo.buses_df.x,
                "new_y": topo.buses_df.y,
            },
            index=topo.buses_df.index,
        )
        linemap = spatial.apply_busmap(wc_grid, busmap)

        assert set(topo.buses_df.index) == set(orig.topology.buses_df.index)
        assert len(linemap) == len(orig.topology.lines_df)
        old = orig.topology.lines_df
        new = topo.lines_df.loc[linemap.loc[old.index, "new_line_name"]]
        for attr in ("length", "r", "x", "s_nom"):
            np.testing.assert_allclose(
                new[attr].astype(float), old[attr], rtol=1e-6, err_msg=attr
            )

        orig.analyze()
        wc_grid.analyze()
        for timestep in orig.timeseries.timeindex:
            v, rms_v = spatial.compare_voltage(orig, wc_grid, busmap, timestep)
            assert rms_v < VOLTAGE_ATOL
            s, rms_s = spatial.compare_apparent_power(orig, wc_grid, linemap, timestep)
            assert rms_s < POWER_ATOL
        pd.testing.assert_frame_equal(
            wc_grid.results.pfa_slack, orig.results.pfa_slack, atol=POWER_ATOL
        )

    def test_remove_short_end_lines_keeps_components(self, wc_grid):
        """
        Removing short end lines keeps all components and their totals, leaves
        no dangling reference and the grid connected, removes as many buses as
        lines and only removes lines of at most 1 m.
        """
        orig = copy.deepcopy(wc_grid)
        spatial.remove_short_end_lines(wc_grid)
        topo, otopo = wc_grid.topology, orig.topology
        for df, col in (
            ("loads_df", "p_set"),
            ("generators_df", "p_nom"),
            ("storage_units_df", "p_nom"),
        ):
            assert getattr(topo, df).index.equals(getattr(otopo, df).index)
            assert getattr(topo, df)[col].sum() == getattr(otopo, df)[col].sum()
        removed = otopo.lines_df.index.difference(topo.lines_df.index)
        assert (otopo.lines_df.loc[removed, "length"] <= 0.001).all()
        assert len(otopo.buses_df) - len(topo.buses_df) == len(removed)
        assert undefined_references(topo) == []
        assert nx.is_connected(topo.to_graph()) == nx.is_connected(otopo.to_graph())


# ======================================================================================
# temporal complexity reduction: selection from power flow results
# ======================================================================================


class TestCriticalTimeSteps:
    def test_selection_contains_worst_step_of_every_violation(
        self, multi_step_analyzed
    ):
        """
        With percentage 1, the selected time steps contain, for every overloaded
        line/transformer and every bus with a voltage issue, the time steps of
        its maximum; the selection is the same with and without cost weighting,
        unique and within the time index, and a smaller percentage selects a
        subset.
        """
        edisgo = multi_step_analyzed
        rel = checks.components_relative_load(copy.deepcopy(edisgo))
        dev = checks.voltage_deviation_from_allowed_voltage_limits(
            copy.deepcopy(edisgo)
        ).abs()
        required = set()
        for frame, limit in ((rel, 1.0), (dev, 0.0)):
            worst = frame.max()
            for comp in worst.index[worst > limit]:
                required |= set(frame.index[frame[comp] == worst[comp]])
        assert required, "test grid must have violations"

        def select(**kwargs):
            return temporal.get_most_critical_time_steps(
                copy.deepcopy(edisgo), run_initial_analyze=False, **kwargs
            )

        weighted = select(percentage=1.0)
        unweighted = select(percentage=1.0, weight_by_costs=False)
        assert weighted.is_unique
        assert set(weighted) <= set(edisgo.timeseries.timeindex)
        assert required <= set(weighted)
        assert set(weighted) == set(unweighted)
        assert set(select(percentage=0.5)) <= set(weighted)

    def test_costs_per_feeder_and_component_cover_all_lines(self, wc_grid):
        """
        The cost estimates cover every line (finite, >= 0), and the feeder costs
        add up to the line costs plus one transformer per LV station, so no line
        is lost when grouping by feeder.
        """
        line_costs = (
            line_expansion_costs(wc_grid).drop(columns="voltage_level").sum(axis=1)
        )
        per_comp = temporal._costs_per_line_and_transformer(copy.deepcopy(wc_grid))
        assert set(wc_grid.topology.lines_df.index) <= set(per_comp.index)
        assert np.isfinite(per_comp).all() and (per_comp >= 0).all()

        per_feeder = temporal._costs_per_feeder(wc_grid)
        n_lv = len(list(wc_grid.topology.mv_grid.lv_grids))
        cost_lv = wc_grid.config["costs_transformers"]["lv"]
        assert per_feeder.sum() == pytest.approx(
            line_costs.sum() + n_lv * cost_lv, rel=1e-12
        )
        assert set(per_feeder.index) <= set(wc_grid.topology.buses_df.index)


class TestVoltageIntervals:
    """Voltage issue intervals on injected voltages (daily 24 h intervals)."""

    @staticmethod
    def _setup(edisgo):
        """Return two deep buses of one LV feeder, one of another, and the band."""
        topo = edisgo.topology
        topo.assign_feeders(mode="grid_feeder")
        stations = set(topo.transformers_df.bus1)
        candidates = {}
        for grid in topo.lv_grids:
            buses = topo.buses_df.loc[grid.buses_df.index]
            for feeder, group in buses.groupby("grid_feeder"):
                deep = [b for b in group.index if b != feeder and b not in stations]
                if feeder != "station_node" and len(deep) >= 2:
                    candidates.setdefault(str(grid), (feeder, deep))
        (f1, deep1), (f2, deep2) = list(candidates.values())[:2]
        v_res = edisgo.results.v_res
        ones = pd.DataFrame(1.0, index=v_res.index, columns=v_res.columns)
        probe = ones.copy()
        probe[deep1[0]] = 1.2
        inject_power_flow_results(edisgo, v_res=probe)
        upper = (
            1.2
            - checks.voltage_deviation_from_allowed_voltage_limits(
                copy.deepcopy(edisgo)
            )[deep1[0]].iloc[0]
        )
        return ones, (deep1[0], deep1[1]), deep2[0], upper

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: _scored_most_critical_voltage_issues_time_interval groups the "
        "voltage deviations by bus instead of by feeder (groupby('Bus') overwrites "
        "the feeder columns), so with weight_by_costs the feeder costs only align "
        "with feeder head buses and a violation at another bus scores NaN -> 0",
    )
    def test_weighted_intervals_include_violation_inside_feeder(
        self, multi_step_analyzed
    ):
        """
        With cost weighting, a day with a voltage violation at a bus inside an
        LV feeder is selected as a critical interval.
        """
        edisgo = multi_step_analyzed
        v, _, bus, upper = self._setup(edisgo)
        t = v.index[30]
        v.loc[t, bus] = upper + 0.03
        inject_power_flow_results(edisgo, v_res=v)
        result = temporal._scored_most_critical_voltage_issues_time_interval(
            edisgo, 24, weight_by_costs=True
        )
        assert any(t in iv for iv in result.time_steps)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: _scored_most_critical_voltage_issues_time_interval groups by "
        "bus instead of feeder, so without weighting the score is the sum over "
        "buses instead of the maximum per feeder (contrary to the docstring)",
    )
    def test_unweighted_intervals_rank_by_maximum_per_feeder(self, multi_step_analyzed):
        """
        Without weighting, an interval is scored by the sum over feeders of the
        maximum deviation in the feeder: two buses of one feeder with 0.02 rank
        below one bus of another feeder with 0.03.
        """
        edisgo = multi_step_analyzed
        v, (b1, b2), b3, upper = self._setup(edisgo)
        t_a, t_b = v.index[5], v.index[30]
        v.loc[t_a, [b1, b2]] = upper + 0.02
        v.loc[t_b, b3] = upper + 0.03
        inject_power_flow_results(edisgo, v_res=v)
        dev = checks.voltage_deviation_from_allowed_voltage_limits(
            copy.deepcopy(edisgo)
        ).abs()
        # precondition of the scenario: only these buses violate their band
        assert set(dev.columns[dev.max() > 1e-9]) == {b1, b2, b3}
        result = temporal._scored_most_critical_voltage_issues_time_interval(
            edisgo, 24, weight_by_costs=False
        )
        assert t_b in result.time_steps.iloc[0]


# ======================================================================================
# temporal complexity reduction: selection logic with stubs
# ======================================================================================


def _stub_grid(n_steps=N_STEPS, analyze=None):
    """Object with an hourly time index and a stubbed ``analyze``."""
    index = pd.date_range("2011-01-03", periods=n_steps, freq="h")
    return SimpleNamespace(
        timeseries=SimpleNamespace(timeindex=index),
        analyze=analyze or (lambda **kwargs: None),
    )


class TestSelectionLogic:
    @pytest.mark.xfail(
        strict=True,
        reason="Bug: _troubleshooting_mode raises only 'if fraction == 0.1', but "
        "np.arange(0.8, 0, -0.1) ends with 0.09999999999999987, so a power flow "
        "that never converges is silently ignored",
    )
    @pytest.mark.parametrize("scale", [None, 0.5])
    def test_troubleshooting_mode_raises_without_convergence(self, scale):
        """If the power flow fails for every scaling, a ValueError is raised."""

        def analyze(**kwargs):
            raise RuntimeError("power flow did not converge")

        with pytest.raises(ValueError):
            temporal._troubleshooting_mode(
                _stub_grid(analyze=analyze), scale_timeseries=scale
            )

    @SYN_SETTINGS
    @given(k=st.integers(1, 8), start=st.sampled_from([None, 1.0, 0.65]))
    def test_troubleshooting_mode_uses_largest_converging_scaling(self, k, start):
        """
        The power flow is rerun with decreasing scaling in steps of 0.1 and the
        last run uses the largest scaling that converges.
        """
        threshold = k / 10 + 0.025
        calls = []

        def analyze(scale_timeseries=None, **kwargs):
            calls.append(scale_timeseries)
            if scale_timeseries is None or scale_timeseries > threshold:
                raise RuntimeError("power flow did not converge")

        top = 0.8 if start is None else start - 0.1
        expected = top - 0.1 * max(0, math.ceil(round((top - threshold) / 0.1, 9)))
        assume(threshold <= (start or 1.0) and expected > 0)
        temporal._troubleshooting_mode(
            _stub_grid(analyze=analyze), scale_timeseries=start
        )
        assert calls[-1] == pytest.approx(expected, abs=1e-9)
        assert all(c > threshold for c in calls[1:-1] if c is not None)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: get_most_critical_time_intervals reuses num_time_intervals of "
        "the overloading intervals for the voltage intervals and joins both with "
        "an inner merge, so a grid without overloading gets no intervals at all "
        "and in general only min(#overloading, #voltage) intervals are returned",
    )
    @pytest.mark.parametrize("n_overloading", [0, 1])
    def test_voltage_intervals_independent_of_overloading(
        self, monkeypatch, n_overloading
    ):
        """
        With percentage 1, all voltage issue intervals are returned, also if
        there are fewer (or no) overloading intervals.
        """
        index = _stub_grid().timeseries.timeindex
        days = [index[24 * i : 24 * (i + 1)] for i in range(3)]

        def frame(name, n):
            return pd.DataFrame(
                {"time_steps": days[:n], f"percentage_{name}": [1.0] * n},
                index=range(n),
            )

        monkeypatch.setattr(
            temporal,
            "_scored_most_critical_loading_time_interval",
            lambda *a, **k: frame("max_overloaded_components", n_overloading),
        )
        monkeypatch.setattr(
            temporal,
            "_scored_most_critical_voltage_issues_time_interval",
            lambda *a, **k: frame("buses_max_voltage_deviation", 3),
        )
        result = temporal.get_most_critical_time_intervals(
            _stub_grid(), time_steps_per_time_interval=24, percentage=1.0
        )
        assert result["time_steps_voltage_issues"].notna().sum() == 3

    @SYN_SETTINGS
    @given(
        n_days=st.integers(1, 4),
        hour=st.integers(0, 23),
        day_start=st.integers(0, 23),
        n_index_days=st.integers(0, 10),
    )
    def test_centered_interval_contains_step(
        self, n_days, hour, day_start, n_index_days
    ):
        """
        For interval lengths of three days and more, the interval around a
        critical step has the requested length, lies in the time index, starts
        at the day start hour (if not clipped at the index start or end) and
        contains the step, but not as its last step.
        """
        n = 24 * (n_days + 2)
        index = pd.date_range("2011-01-01", periods=24 * (n_index_days + 20), freq="h")
        t = index[n + 24 + 24 * n_index_days // 2 + hour]
        iv = temporal._build_centered_interval(t, index, n, day_start)
        assert len(iv) == n
        assert iv[0] >= index[0] and iv[-1] <= index[-1]
        assert t in iv and t != iv[-1]
        assert iv[0].hour == day_start

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: _build_centered_interval snaps the start back by up to 23 "
        "steps after centering, so for 24 and 48 steps the critical step can fall "
        "out of the interval or be its last step",
    )
    @pytest.mark.parametrize("n, hour", [(24, 5), (48, 23)])
    def test_centered_interval_contains_step_for_short_intervals(self, n, hour):
        """For one- and two-day intervals the step lies inside, not at the end."""
        index = pd.date_range("2011-01-01", periods=24 * 10, freq="h")
        t = index[24 * 4 + hour]
        iv = temporal._build_centered_interval(t, index, n, 0)
        assert t in iv and t != iv[-1]

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: select_two_intervals merges overlapping top intervals with "
        "pd.infer_freq(top), which raises 'Need at least 3 dates' for intervals "
        "of one or two time steps",
    )
    def test_select_two_intervals_merges_short_intervals(self):
        """Two overlapping one-step intervals are merged into one interval."""
        iv = pd.date_range("2011-01-01", periods=1, freq="h")
        result = temporal.select_two_intervals([iv], [iv])
        assert len(result) == 1 and result[0].equals(iv)

    @SYN_SETTINGS
    @given(
        # at least 3 steps, see test_select_two_intervals_merges_short_intervals
        load=st.lists(st.tuples(st.integers(0, 200), st.integers(3, 48)), max_size=5),
        feedin=st.lists(st.tuples(st.integers(0, 200), st.integers(3, 48)), max_size=5),
    )
    def test_select_two_intervals(self, load, feedin):
        """
        The selected intervals are contiguous and pairwise non-overlapping; the
        top load interval is covered; with two intervals the second is the
        highest ranked feed-in interval not overlapping the first; a merged
        interval covers both top intervals.
        """
        base = pd.Timestamp("2011-01-01")

        def ivs(spec):
            return [
                pd.date_range(base + pd.Timedelta(hours=s), periods=n, freq="h")
                for s, n in spec
            ]

        load_iv, feedin_iv = ivs(load), ivs(feedin)
        result = temporal.select_two_intervals(load_iv, feedin_iv)
        if not load_iv and not feedin_iv:
            assert result == []
            return
        for iv in result:
            assert (np.diff(iv.asi8) == 3600 * 10**9).all()
        if len(result) == 2:
            assert not temporal.intervals_overlap(*result)
        if load_iv:
            assert set(load_iv[0]) <= set().union(*map(set, result))
        if load_iv and feedin_iv:
            free = [
                c for c in feedin_iv if not temporal.intervals_overlap(load_iv[0], c)
            ]
            if free:
                assert len(result) == 2 and result[1].equals(free[0])
            else:
                assert len(result) == 1
                assert set(feedin_iv[0]) <= set(result[0])

    @settings(SYN_SETTINGS, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        residual=st.lists(st.floats(-10, 10), min_size=1, max_size=40),
        n_load=st.one_of(st.none(), st.integers(0, 50)),
        n_feed=st.one_of(st.none(), st.integers(0, 50)),
        percentage=st.floats(0, 1),
        more=st.floats(0, 1),
    )
    def test_residual_load_steps_are_extreme(
        self, monkeypatch, residual, n_load, n_feed, percentage, more
    ):
        """
        Every selected load-case step has a residual load at least as high as
        every unselected step (feed-in case: at most as low), the numbers of
        steps are as requested, and a smaller percentage selects a subset.
        """
        index = pd.date_range("2011-01-01", periods=len(residual), freq="h")
        res = pd.Series(residual, index=index)
        distributed = SimpleNamespace(timeseries=SimpleNamespace(residual_load=res))
        monkeypatch.setattr(
            overlying_grid,
            "distribute_overlying_grid_requirements",
            lambda edisgo: distributed,
        )

        def select(**kwargs):
            return temporal.get_most_critical_time_steps(
                None, by="residual_load", **kwargs
            )

        load = select(num_steps_loading=n_load, num_steps_voltage=0, percentage=1)
        feed = select(num_steps_loading=0, num_steps_voltage=n_feed, percentage=1)
        for chosen, sign, n in ((load, 1, n_load), (feed, -1, n_feed)):
            assert len(chosen) == min(len(res) if n is None else n, len(res))
            rest = res.drop(chosen)
            if len(chosen) and len(rest):
                assert (sign * res[chosen]).min() >= (sign * rest).max()
        both = select(percentage=percentage)
        assert both.is_unique
        larger = percentage + more * (1 - percentage)
        assert set(both) <= set(select(percentage=larger))

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: _most_critical_time_interval counts components without "
        "any violation (maximum 0) as 'maximum reached' in every interval, so the "
        "share is > 0 even for intervals without any maximum",
    )
    def test_interval_share_counts_only_reached_maxima(self):
        """
        An interval that contains no component's maximum overloading has a share
        of 0 components at their maximum.
        """
        index = _stub_grid().timeseries.timeindex
        magnitude = pd.DataFrame(0.0, index=index, columns=["overloaded", "fine"])
        magnitude.loc[index[5], "overloaded"] = 1.2
        magnitude.loc[index[30], "overloaded"] = 1.5
        result = temporal._most_critical_time_interval(
            magnitude, magnitude, "overloading", time_steps_per_time_interval=24
        )
        share = result.set_index(result.time_steps.map(lambda iv: iv[0]))
        first_day = share.loc[index[0], "percentage_max_overloaded_components"]
        assert first_day == 0


# ======================================================================================
# chain: reinforcement on the selected time steps
# ======================================================================================


def test_reinforce_on_critical_steps_resolves_full_time_series(multi_step_analyzed):
    """
    Reinforcing the grid only on the most critical time steps (percentage 1)
    leaves no overloading and no voltage issue on the full time series.
    """
    edisgo = multi_step_analyzed
    steps = temporal.get_most_critical_time_steps(
        copy.deepcopy(edisgo), run_initial_analyze=False
    )
    assert 0 < len(steps) < len(edisgo.timeseries.timeindex)
    edisgo.reinforce(timesteps_pfa=steps)
    edisgo.analyze()
    assert limit_violations(edisgo) == {}
