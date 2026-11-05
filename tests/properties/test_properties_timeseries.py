"""
Property-based tests for the time series containers in :mod:`edisgo.network`:
:mod:`~edisgo.network.timeseries` (worst cases, residual load, container
operations), :mod:`~edisgo.network.electromobility` (flexibility bands),
:mod:`~edisgo.network.overlying_grid` (distribution of overlying grid
requirements), :mod:`~edisgo.network.heat`, :mod:`~edisgo.network.dsm` and
:mod:`~edisgo.network.results`.

Most tests use synthetic data frames; the worst-case and distribution tests use
test network 1. Each test states the property it checks. Fixtures and helpers
live in ``tests/properties/conftest.py``.
"""

from __future__ import annotations

import copy
import logging

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from edisgo import EDisGo
from edisgo.network.electromobility import Electromobility
from edisgo.network.overlying_grid import distribute_overlying_grid_requirements
from edisgo.network.results import Results
from edisgo.network.timeseries import TimeSeries
from edisgo.tools.config import Config

from .conftest import (
    TIMESERIES_ATTRIBUTES,
    assert_timeseries_equal,
    assert_topology_equal,
    edisgo_warnings,
    examples,
)

#: Settings for tests on small synthetic data (milliseconds per example).
SYN_SETTINGS = settings(deadline=None, max_examples=examples(60))

#: Synthetic tests that read log records: ``caplog`` is function scoped, but
#: every example clears it before it runs the checked function.
CAPLOG_SETTINGS = settings(
    deadline=None,
    max_examples=examples(60),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Settings for tests on a copy of test network 1 (no power flow).
GRID_SETTINGS = settings(
    deadline=None,
    max_examples=examples(15),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Relative tolerance for values computed with a handful of float operations
#: from the same inputs (products, tan(arccos(pf)), sums of < 200 terms).
RTOL = 1e-9

#: Absolute tolerance in MW/MWh for sums of synthetic powers of order 1.
ATOL = 1e-12

#: Tolerance of the flexibility band rounding fix (default of
#: ``get_flexibility_bands``): bands may be shifted by up to this value.
BAND_TOL = 1e-6

COMPONENT_KEYS = {
    "generator": "generators",
    "conventional_load": "loads",
    "charging_point": "loads",
    "heat_pump": "loads",
    "storage_unit": "storage_units",
}
GENERATOR_LIKE = ("generator", "storage_unit")
CP_SECTORS = ("home", "work", "public", "hpc")

reference_timeindex = pd.date_range("2011-01-03", periods=6, freq="h")


# ======================================================================================
# helpers
# ======================================================================================


def _component_kind(topology, comp, name):
    """Config key of a component: generator, storage_unit or the load type."""
    if comp == "loads":
        load_type = topology.loads_df.at[name, "type"]
        return load_type if load_type in COMPONENT_KEYS else "conventional_load"
    return {"generators": "generator", "storage_units": "storage_unit"}[comp]


def _voltage_level(topology, comp, name):
    bus = getattr(topology, f"{comp}_df").at[name, "bus"]
    return "lv" if topology.buses_df.at[bus, "v_nom"] < 1 else "mv"


def _q_sign(kind, mode):
    """Reactive power sign in the eDisGo convention, derived from the mode."""
    inductive = mode.lower() == "inductive"
    if kind in GENERATOR_LIKE:
        return -1 if inductive else 1
    return 1 if inductive else -1


def _expected_q(edisgo, comp, p):
    """Q of the default fixed cos φ for active power ``p`` of component type comp."""
    topo, config = edisgo.topology, edisgo.config
    q = pd.DataFrame(index=p.index, columns=p.columns, dtype=float)
    for name in p.columns:
        kind = _component_kind(topo, comp, name)
        level = _voltage_level(topo, comp, name)
        pf = config["reactive_power_factor"][f"{level}_{kind}"]
        mode = config["reactive_power_mode"][f"{level}_{kind}"]
        q[name] = p[name].astype(float) * _q_sign(kind, mode) * np.tan(np.arccos(pf))
    return q


def _worst_case_factor(edisgo, comp, name, label):
    """Simultaneity factor of a component in a worst case such as load_case_mv."""
    topo = edisgo.topology
    factors = edisgo.config["worst_case_scale_factor"]
    case, level = label.rsplit("_", 1)
    kind = _component_kind(topo, comp, name)
    if kind == "generator":
        gen_type = topo.generators_df.at[name, "type"]
        tech = {"solar": "pv", "wind": "wind"}.get(gen_type, "other")
        return factors[f"{case}_feed-in_{tech}"]
    if kind == "storage_unit":
        return factors[f"{case}_storage"]
    if kind == "charging_point":
        sector = topo.loads_df.at[name, "sector"]
        return factors[f"{level}_{case}_cp_{sector}"]
    if kind == "heat_pump":
        return factors[f"{level}_{case}_hp"]
    return factors[f"{level}_{case}_load"]


def _rating(topology, comp, name):
    df = getattr(topology, f"{comp}_df")
    return df.at[name, "p_set"] if comp == "loads" else df.at[name, "p_nom"]


def _add_all_component_types(edisgo):
    """Add charging points, heat pumps, LV storage and other generator types."""
    topo = edisgo.topology
    mv_bus = topo.mv_grid.buses_df.index[5]
    lv_bus = topo.get_lv_grid(3).buses_df.index[3]
    for bus in (mv_bus, lv_bus):
        for sector in CP_SECTORS:
            topo.add_load(bus, 0.011, type="charging_point", sector=sector)
        topo.add_load(bus, 0.008, type="heat_pump", sector="individual_heating")
        topo.add_load(bus, 0.05, type="heat_pump", sector="district_heating")
    topo.add_storage_unit(lv_bus, 0.01)
    topo.add_generator(lv_bus, 0.02, "biomass")
    topo.add_generator(mv_bus, 0.5, "run_of_river")


@pytest.fixture(scope="module")
def all_types_grid():
    """Test network 1 with every component type in MV and LV, no time series."""
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
    _add_all_component_types(edisgo)
    return edisgo


@pytest.fixture(scope="module")
def all_types_wc_grid(all_types_grid):
    """:func:`all_types_grid` with worst-case time series for both cases."""
    edisgo = copy.deepcopy(all_types_grid)
    edisgo.set_time_series_worst_case_analysis()
    return edisgo


def _frame(values, index, columns):
    return pd.DataFrame(
        np.reshape(values, (len(index), len(columns))), index=index, columns=columns
    )


finite = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)


# ======================================================================================
# timeseries: worst cases
# ======================================================================================


class TestWorstCase:
    def test_worst_case_power_follows_scale_factors_and_power_factors(
        self, all_types_wc_grid
    ):
        """
        For every component and worst case, ``P = factor · rating`` with the
        simultaneity factor of its type, case and the level the case dimensions
        (generators and storage units use the same factor for MV and LV), and
        ``Q = P · q_sign · tan(arccos pf)`` with pf and mode of its own voltage
        level; with the inductive default, generators have ``P·Q <= 0`` and
        loads ``P·Q >= 0``.
        """
        edisgo = all_types_wc_grid
        ts, topo = edisgo.timeseries, edisgo.topology
        twc = ts.timeindex_worst_cases
        modes = edisgo.config["reactive_power_mode"]
        assert set(modes.values()) == {"inductive"}
        for comp in ("generators", "loads", "storage_units"):
            p = getattr(ts, f"{comp}_active_power")
            q = getattr(ts, f"{comp}_reactive_power")
            assert sorted(p.columns) == sorted(getattr(topo, f"{comp}_df").index)
            assert not p.isna().any(axis=None) and not q.isna().any(axis=None)
            expected = pd.DataFrame(
                {
                    name: [
                        _worst_case_factor(edisgo, comp, name, label)
                        * _rating(topo, comp, name)
                        for label in twc.index
                    ]
                    for name in p.columns
                },
                index=pd.DatetimeIndex(twc.values),
            )
            np.testing.assert_allclose(
                p.loc[expected.index, expected.columns].astype(float),
                expected.astype(float),
                rtol=RTOL,
                atol=0,
            )
            np.testing.assert_allclose(
                q[p.columns].astype(float),
                _expected_q(edisgo, comp, p).astype(float),
                rtol=RTOL,
                atol=ATOL,
            )
            if comp == "generators":
                assert ((p * q) <= 0).all(axis=None)
            elif comp == "loads":
                assert ((p * q) >= 0).all(axis=None)

    def test_load_case_has_non_negative_residual_load(self, all_types_wc_grid):
        """
        In every load case, generators feed in nothing, loads draw power,
        storage units charge, the residual load is non-negative and the time
        step is classified as load case.
        """
        ts = all_types_wc_grid.timeseries
        twc = ts.timeindex_worst_cases
        steps = pd.DatetimeIndex(twc[twc.index.str.startswith("load_case")].values)
        assert len(steps) == 2
        assert (ts.generators_active_power.loc[steps] == 0).all(axis=None)
        assert (ts.loads_active_power.loc[steps] >= 0).all(axis=None)
        assert (ts.storage_units_active_power.loc[steps] <= 0).all(axis=None)
        assert (ts.residual_load.loc[steps] >= 0).all()
        assert (ts.timesteps_load_feedin_case.loc[steps] == "load_case").all()

    @pytest.mark.parametrize(
        "order", [["load_case", "feed-in_case"], ["feed-in_case", "load_case"]]
    )
    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: set_worst_case for an additional case replaces the "
        "components' whole time series (documented as 'overwritten'), so the "
        "time steps of the case set before become NaN while staying in timeindex",
    )
    def test_cases_set_one_after_another_equal_cases_set_together(
        self, all_types_grid, all_types_wc_grid, order
    ):
        """
        Setting the worst cases one after the other yields, per case label, the
        same values as setting both at once, and the reactive power control
        parameters contain every component once.
        """
        edisgo = copy.deepcopy(all_types_grid)
        for case in order:
            edisgo.timeseries.set_worst_case(edisgo, cases=[case])
        both = all_types_wc_grid.timeseries
        seq = edisgo.timeseries
        assert set(seq.timeindex_worst_cases.index) == set(
            both.timeindex_worst_cases.index
        )
        for label, step in seq.timeindex_worst_cases.items():
            for attr in TIMESERIES_ATTRIBUTES:
                a = getattr(seq, attr).loc[step]
                b = getattr(both, attr).loc[both.timeindex_worst_cases[label]]
                pd.testing.assert_series_equal(
                    a.sort_index(), b.sort_index(), check_names=False, obj=attr
                )
        assert seq.time_series_raw.q_control.index.is_unique

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: fixed cos φ keeps Q/P constant, so an 'inductive' storage "
        "unit consumes reactive power when discharging but supplies it when "
        "charging (worst-case load case: P = -0.4 MW, Q = +0.19 Mvar)",
    )
    def test_inductive_storage_consumes_reactive_power_in_every_state(
        self, all_types_wc_grid
    ):
        """
        A storage unit in inductive mode consumes reactive power (Q <= 0 in the
        generator sign convention) whether it charges or discharges.
        """
        q = all_types_wc_grid.timeseries.storage_units_reactive_power
        assert (q <= 0).all(axis=None)

    @SYN_SETTINGS
    @given(
        data=st.data(),
        cases=st.sampled_from(
            [["feed-in_case"], ["load_case"], ["feed-in_case", "load_case"]]
        ),
        scale=st.floats(0.1, 100.0),
    )
    def test_worst_case_helpers_are_linear_and_map_types(self, data, cases, scale):
        """
        The worst-case helpers give ``P = factor(type, case) · rating`` for
        random component tables, are linear in the rating, map only 'solar' to
        PV and 'wind' to wind (everything else to 'other'), and the reactive
        power parameters contain every component exactly once, also after
        repeated calls.
        """
        config = Config()
        factors = config["worst_case_scale_factor"]
        n = data.draw(st.integers(1, 6))
        names = [f"c{i}" for i in range(n)]
        levels = data.draw(
            st.lists(st.sampled_from(["mv", "lv"]), min_size=n, max_size=n)
        )
        ratings = data.draw(st.lists(st.floats(0.001, 5.0), min_size=n, max_size=n))
        gen_types = data.draw(
            st.lists(
                st.sampled_from(["solar", "wind", "gas", "biomass", "water", "Solar"]),
                min_size=n,
                max_size=n,
            )
        )
        sectors = data.draw(
            st.lists(st.sampled_from(CP_SECTORS), min_size=n, max_size=n)
        )
        ts = TimeSeries()
        gens = pd.DataFrame(
            {"p_nom": ratings, "voltage_level": levels, "type": gen_types}, index=names
        )
        loads = pd.DataFrame(
            {"p_set": ratings, "voltage_level": levels, "sector": sectors}, index=names
        )
        labels = [f"{c}_{lvl}" for c in cases for lvl in ("mv", "lv")]

        def expected(kind, name, label):
            case, level = label.rsplit("_", 1)
            if kind == "gen":
                tech = {"solar": "pv", "wind": "wind"}.get(
                    gens.at[name, "type"], "other"
                )
                return factors[f"{case}_feed-in_{tech}"]
            if kind == "cp":
                return factors[f"{level}_{case}_cp_{loads.at[name, 'sector']}"]
            if kind == "hp":
                return factors[f"{level}_{case}_hp"]
            if kind == "storage":
                return factors[f"{case}_storage"]
            return factors[f"{level}_{case}_load"]

        helpers = {
            "gen": (ts._worst_case_generators, gens, "p_nom"),
            "load": (ts._worst_case_conventional_load, loads, "p_set"),
            "cp": (ts._worst_case_charging_points, loads, "p_set"),
            "hp": (ts._worst_case_heat_pumps, loads, "p_set"),
            "storage": (ts._worst_case_storage_units, gens, "p_nom"),
        }
        for kind, (func, df, col) in helpers.items():
            p, _ = func(cases, df, config)
            for name in names:
                for label in labels:
                    assert p.at[label, name] == pytest.approx(
                        expected(kind, name, label) * df.at[name, col], rel=RTOL
                    )
            scaled = df.assign(**{col: df[col] * scale})
            p_scaled, q_scaled = func(cases, scaled, config)
            _, q = func(cases, df, config)
            np.testing.assert_allclose(
                p_scaled[p.columns].astype(float), (p * scale).astype(float), rtol=RTOL
            )
            np.testing.assert_allclose(
                q_scaled[q.columns].astype(float), (q * scale).astype(float), rtol=RTOL
            )
        q_control = ts.time_series_raw.q_control
        assert q_control.index.is_unique and set(q_control.index) == set(names)

    @SYN_SETTINGS
    @given(sector=st.text(min_size=1, max_size=10))
    def test_unknown_charging_point_sector_is_rejected(self, sector):
        """Charging points with a use case other than home/work/public/hpc raise."""
        assume(sector not in CP_SECTORS)
        df = pd.DataFrame(
            {
                "p_set": [0.011, 0.022],
                "voltage_level": "lv",
                "sector": ["home", sector],
            },
            index=["cp1", "cp2"],
        )
        with pytest.raises(AttributeError):
            TimeSeries()._worst_case_charging_points(["load_case"], df, Config())


# ======================================================================================
# timeseries: container operations
# ======================================================================================


def _container(data, n_t=4, n_comp=3):
    """TimeSeries with random active power of generators, loads and storage units."""
    index = pd.date_range("2011-01-01", periods=n_t, freq="h")
    ts = TimeSeries(timeindex=index)
    frames = {}
    for comp in ("generators", "loads", "storage_units"):
        cols = [f"{comp}_{i}" for i in range(n_comp)]
        values = data.draw(
            st.lists(finite, min_size=n_t * n_comp, max_size=n_t * n_comp)
        )
        frames[comp] = _frame(values, index, cols)
        setattr(ts, f"{comp}_active_power", frames[comp])
    return ts, frames


class TestTimeSeriesContainer:
    @SYN_SETTINGS
    @given(data=st.data())
    def test_residual_load_and_case_classification(self, data):
        """
        ``residual_load = ΣP_load − ΣP_gen − ΣP_storage``; a time step is a
        feed-in case iff the residual load is negative; swapping loads and
        generators and negating storage mirrors the residual load.
        """
        ts, frames = _container(data)
        expected = (
            frames["loads"].sum(axis=1)
            - frames["generators"].sum(axis=1)
            - frames["storage_units"].sum(axis=1)
        )
        pd.testing.assert_series_equal(ts.residual_load, expected, check_names=False)
        cases = ts.timesteps_load_feedin_case
        assert ((cases == "feed-in_case") == (expected < 0)).all()
        assert set(cases) <= {"load_case", "feed-in_case"}

        mirrored = TimeSeries(timeindex=ts.timeindex)
        mirrored.loads_active_power = frames["generators"]
        mirrored.generators_active_power = frames["loads"]
        mirrored.storage_units_active_power = -frames["storage_units"]
        np.testing.assert_allclose(mirrored.residual_load, -expected, atol=1e-12)

    @SYN_SETTINGS
    @given(data=st.data(), start=st.integers(0, 3), length=st.integers(1, 4))
    def test_getters_follow_timeindex_and_reset_clears(self, data, start, length):
        """
        Every power getter returns exactly the rows of ``timeindex`` (also a
        subset of the stored rows); ``reset`` empties all six frames, the time
        index and the reactive power parameters, and is idempotent.
        """
        ts, frames = _container(data, n_t=8)
        sub = ts.timeindex[start : start + length]
        ts.timeindex = sub
        for comp, df in frames.items():
            pd.testing.assert_frame_equal(
                getattr(ts, f"{comp}_active_power"), df.loc[sub]
            )
        ts.time_series_raw.q_control = pd.DataFrame({"type": ["fixed_cosphi"]})
        for _ in range(2):
            ts.reset()
            assert ts.timeindex.empty
            assert ts.time_series_raw.q_control.empty
            for attr in TIMESERIES_ATTRIBUTES:
                assert getattr(ts, attr).empty, attr

    @SYN_SETTINGS
    @given(data=st.data())
    def test_add_and_drop_component_time_series(self, data):
        """
        ``add_component_time_series`` overwrites given columns, keeps all other
        columns and is idempotent; dropping the added columns again restores the
        original frame; dropping unknown names changes nothing.
        """
        ts, frames = _container(data)
        original = frames["loads"]
        cols = data.draw(
            st.lists(
                st.sampled_from(list(original.columns) + ["new_0", "new_1"]),
                min_size=1,
                max_size=4,
                unique=True,
            )
        )
        values = data.draw(
            st.lists(finite, min_size=4 * len(cols), max_size=4 * len(cols))
        )
        new = _frame(values, ts.timeindex, cols)
        ts.add_component_time_series("loads_active_power", new)
        once = ts.loads_active_power.copy()
        ts.add_component_time_series("loads_active_power", new)
        pd.testing.assert_frame_equal(ts.loads_active_power, once)
        assert once.columns.is_unique
        pd.testing.assert_frame_equal(once[cols], new)
        others = original.columns.difference(cols)
        pd.testing.assert_frame_equal(once[others], original[others])

        added = [c for c in cols if c not in original.columns]
        ts.drop_component_time_series("loads_active_power", added + ["unknown"])
        assert sorted(ts.loads_active_power.columns) == sorted(original.columns)
        if not set(cols) & set(original.columns):
            pd.testing.assert_frame_equal(
                ts.loads_active_power[original.columns], original
            )

    @GRID_SETTINGS
    @given(data=st.data())
    def test_check_if_components_exist_returns_known_subset(
        self, wc_grid_factory, data
    ):
        """
        The component check returns exactly the given names that exist in the
        grid (as a set, without duplicates).
        """
        edisgo = wc_grid_factory()
        known = list(edisgo.topology.loads_df.index)
        names = data.draw(
            st.lists(
                st.sampled_from(known + ["unknown_a", "unknown_b"]),
                min_size=1,
                max_size=8,
                unique=True,
            )
        )
        result = edisgo.timeseries._check_if_components_exist(edisgo, names, "loads")
        assert len(result) == len(set(result))
        assert set(result) == set(names) & set(known)

    @GRID_SETTINGS
    @given(k=st.floats(0.1, 5.0))
    def test_equal_scaling_keeps_power_factor(self, wc_grid_factory, k):
        """
        Scaling P and Q by the same factor scales every time series by it and
        keeps every component's power factor.
        """
        edisgo = wc_grid_factory()
        before = copy.deepcopy(edisgo.timeseries)
        edisgo.timeseries.scale_timeseries(p_scaling_factor=k, q_scaling_factor=k)
        for attr in TIMESERIES_ATTRIBUTES:
            np.testing.assert_allclose(
                getattr(edisgo.timeseries, attr), getattr(before, attr) * k, rtol=1e-12
            )

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: scale_timeseries writes back only the rows of the current "
        "timeindex, all stored rows outside of it are silently dropped",
    )
    def test_scaling_keeps_data_outside_timeindex(self):
        """Scaling with a restricted time index keeps the other stored rows."""
        index = pd.date_range("2011-01-01", periods=4, freq="h")
        ts = TimeSeries(timeindex=index)
        ts.loads_active_power = pd.DataFrame({"l": [1.0, 2.0, 3.0, 4.0]}, index=index)
        ts.timeindex = index[:2]
        ts.scale_timeseries(p_scaling_factor=2.0)
        stored = ts._loads_active_power
        assert stored.index.equals(index), "rows outside the timeindex dropped"
        np.testing.assert_allclose(stored["l"], [2.0, 4.0, 3.0, 4.0])


# ======================================================================================
# electromobility: flexibility bands
# ======================================================================================

STEPSIZE = 15
STEPS_PER_HOUR = 60 // STEPSIZE
N_STEPS = 96  # one simulated day


def _electromobility(processes, eta=0.9, timeindex=None):
    """Electromobility container with charging points 'CP_<park id>'."""
    em = Electromobility()
    start = pd.Timestamp("2011-01-03")
    em.simbev_config_df = pd.DataFrame(
        {
            "eta_cp": [eta],
            "stepsize": [STEPSIZE],
            "start_date": [start],
            "end_date": [start],
            "days": [1],
        }
    )
    em.charging_processes_df = processes.reset_index(drop=True)
    parks = sorted(processes.charging_park_id.unique())
    cps = [f"CP_{p}" for p in parks]
    em.integrated_charging_parks_df = pd.DataFrame({"edisgo_id": cps}, index=parks)
    loads = pd.DataFrame(
        {"bus": "b", "p_set": 0.05, "type": "charging_point", "sector": "home"},
        index=cps,
    )
    obj = SimpleNamespace(
        topology=SimpleNamespace(loads_df=loads),
        timeseries=SimpleNamespace(
            timeindex=pd.DatetimeIndex([]) if timeindex is None else timeindex
        ),
        electromobility=em,
        config={"timeindex": {"reference_year": 2011}},
    )
    em._edisgo_obj = obj
    return em, obj


@st.composite
def charging_processes(draw, n_steps=N_STEPS, last_step=False):
    """
    Feasible charging processes (demand <= power · parking time) on up to 3
    charging points. Processes end before the last time step unless
    ``last_step``.
    """
    n = draw(st.integers(1, 8))
    rows = []
    for _ in range(n):
        start = draw(st.integers(0, n_steps - 3))
        end = draw(st.integers(start, n_steps - 2))
        power = draw(st.sampled_from([3.7, 11.0, 22.0, 50.0]))
        share = draw(st.floats(0.01, 1.0))
        rows.append(
            {
                "charging_park_id": draw(st.integers(1, 3)),
                "park_start_timesteps": start,
                "park_end_timesteps": end,
                "park_time_timesteps": end - start + 1,
                "nominal_charging_capacity_kW": power,
                "chargingdemand_kWh": share
                * power
                * (end - start + 1)
                / STEPS_PER_HOUR,
            }
        )
    df = pd.DataFrame(rows)
    if last_step:
        df.loc[0, "park_end_timesteps"] = n_steps - 1
    return df


def _oracle_bands(processes, eta, n_steps=N_STEPS):
    """
    Closed-form bands per charging point: charging as early as possible
    (upper energy), as late as possible (lower energy), and the summed grid-side
    power of all parked cars (upper power). In MW and MWh.
    """
    t = np.arange(n_steps)
    cps = sorted(f"CP_{p}" for p in processes.charging_park_id.unique())
    up = pd.DataFrame(0.0, index=t, columns=cps)
    low, power = up.copy(), up.copy()
    for _, proc in processes.iterrows():
        cp = f"CP_{int(proc.charging_park_id)}"
        p = proc.nominal_charging_capacity_kW / STEPS_PER_HOUR  # kWh per step
        demand = proc.chargingdemand_kWh
        start, end = int(proc.park_start_timesteps), int(proc.park_end_timesteps)
        up[cp] += np.where(t >= start, np.minimum(demand, p * (t - start + 1)), 0.0)
        low[cp] += np.where(t >= end, demand, np.maximum(0.0, demand - p * (end - t)))
        parked = (t >= start) & (t <= end)
        power[cp] += np.where(parked, proc.nominal_charging_capacity_kW / eta, 0.0)
    return {
        "upper_energy": up / 1e3,
        "lower_energy": low / 1e3,
        "upper_power": power / 1e3,
    }


def _assert_band_integrity(bands, eta, step_hours, tol=BAND_TOL, mono_tol=1e-12):
    """Monotone energy bands within each other, reachable with the power band."""
    up, low, power = bands["upper_energy"], bands["lower_energy"], bands["upper_power"]
    assert (power >= 0).all(axis=None)
    assert (low <= up + tol).all(axis=None)
    for band in (up, low):
        assert (band.diff().iloc[1:] >= -mono_tol).all(axis=None)
        assert (band.diff().iloc[1:] <= power.iloc[1:] * eta * step_hours + tol).all(
            axis=None
        )


class TestFlexibilityBands:
    @SYN_SETTINGS
    @given(processes=charging_processes(), eta=st.floats(0.8, 1.0))
    def test_bands_equal_closed_form_and_are_consistent(self, processes, eta):
        """
        The bands equal the closed-form bands of charging as early / as late as
        possible and of the parked cars' power (up to the rounding fix, which
        shifts lower energy by -tol and upper power by at most +2·tol); they are
        monotone, ``lower <= upper``, every energy step is reachable with the
        power band (``ΔE <= P·η·Δt``), both end at the total charging demand of
        each charging point, and a second rounding fix changes nothing.
        (``tol=0`` is not used: the integrity check then rejects float noise of
        1e-18.)
        """
        em, obj = _electromobility(processes, eta=eta)
        bands = em.get_flexibility_bands(obj, "home")
        oracle = _oracle_bands(processes, eta)
        for key, expected in oracle.items():
            np.testing.assert_allclose(
                bands[key][expected.columns].to_numpy(),
                expected.to_numpy(),
                atol=2 * BAND_TOL + 1e-12,
                err_msg=key,
            )
        _assert_band_integrity(bands, eta, STEPSIZE / 60)
        demand = processes.groupby("charging_park_id").chargingdemand_kWh.sum() / 1e3
        demand.index = [f"CP_{p}" for p in demand.index]
        for key in ("upper_energy", "lower_energy"):
            np.testing.assert_allclose(
                bands[key].iloc[-1][demand.index], demand, atol=BAND_TOL + 1e-12
            )
        before = copy.deepcopy(em.flexibility_bands)
        em.fix_flexibility_bands_rounding_errors()
        for key, df in before.items():
            pd.testing.assert_frame_equal(em.flexibility_bands[key], df)

    @SYN_SETTINGS
    @given(processes=charging_processes(), freq=st.sampled_from(["5min", "60min"]))
    def test_resampled_bands_stay_consistent(self, processes, freq):
        """
        After up- (5 min) or downsampling (60 min) the bands stay monotone,
        within each other and reachable with the power band (up to the rounding
        fix tol: the fix lowers the lower band to -tol at the start, upsampling
        interpolates from 0 down to it; see also the strict check below), keep
        the total charging demand
        at the end, and downsampling keeps the energy of the power band (mean
        power · Δt) and passes the strict integrity check.
        """
        eta = 0.9
        em, obj = _electromobility(processes, eta=eta)
        em.get_flexibility_bands(obj, "home")
        before = copy.deepcopy(em.flexibility_bands)
        em.resample(freq=freq)
        after = em.flexibility_bands
        step_hours = pd.Timedelta(freq) / pd.Timedelta("1h")
        tol = BAND_TOL + 1e-12
        _assert_band_integrity(after, eta, step_hours, tol=tol, mono_tol=tol)
        if freq == "60min":
            em.check_integrity()
        for key in ("upper_energy", "lower_energy"):
            np.testing.assert_allclose(
                after[key].iloc[-1], before[key].iloc[-1], atol=1e-12
            )
        if freq == "60min":
            np.testing.assert_allclose(
                after["upper_power"].sum() * 1.0,
                before["upper_power"].sum() * STEPSIZE / 60,
                rtol=1e-12,
            )

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason="Bug: get_flexibility_bands applies the rounding fix (tol) before "
        "resampling to a finer TimeSeries.timeindex and then runs the strict "
        "integrity check; float noise of 5e-20 from upsampling raises ValueError "
        "(about 7 % of random single charging processes at 5 min)",
    )
    def test_bands_on_finer_timeindex_pass_integrity_check(self):
        """
        Flexibility bands can be built for a time index finer than the SimBEV
        step size (here 5 min instead of 15 min).
        """
        processes = pd.DataFrame(
            {
                "charging_park_id": [1],
                "park_start_timesteps": [76],
                "park_end_timesteps": [79],
                "park_time_timesteps": [4],
                "nominal_charging_capacity_kW": [3.7],
                "chargingdemand_kWh": [2.020372449772423],
            }
        )
        timeindex = pd.date_range("2011-01-03", periods=3 * N_STEPS, freq="5min")
        em, obj = _electromobility(processes, timeindex=timeindex)
        em.get_flexibility_bands(obj, "home")

    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason="Bug: get_flexibility_bands only extends the time index if "
        "len(index) < max(park_end_timesteps); for a process ending at "
        "len(index) (one step past the simulated period) and no TimeSeries."
        "timeindex, d_pow[end + 1] is out of bounds",
    )
    def test_process_ending_after_simulated_period(self):
        """A charging process ending one step after the simulated day is handled."""
        processes = pd.DataFrame(
            {
                "charging_park_id": [1],
                "park_start_timesteps": [80],
                "park_end_timesteps": [N_STEPS],
                "park_time_timesteps": [N_STEPS - 80 + 1],
                "nominal_charging_capacity_kW": [11.0],
                "chargingdemand_kWh": [5.0],
            }
        )
        em, obj = _electromobility(processes)
        em.get_flexibility_bands(obj, "home")

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: get_flexibility_bands drops every charging process ending "
        "in the last time step of the band index ('can lead to problems'), so its "
        "charging demand is missing in the bands",
    )
    @SYN_SETTINGS
    @given(processes=charging_processes(last_step=True))
    def test_bands_contain_processes_ending_in_last_step(self, processes):
        """The energy bands end at the total charging demand of all processes."""
        em, obj = _electromobility(processes)
        bands = em.get_flexibility_bands(obj, "home")
        demand = processes.groupby("charging_park_id").chargingdemand_kWh.sum() / 1e3
        np.testing.assert_allclose(
            bands["upper_energy"].iloc[-1][[f"CP_{p}" for p in demand.index]],
            demand,
            atol=BAND_TOL,
        )


# ======================================================================================
# overlying grid: distribution of requirements
# ======================================================================================

FLEX_INDEX = pd.date_range("2011-01-03", periods=6, freq="h")


@pytest.fixture(scope="module")
def flex_grid():
    """
    Test network 1 on six hours with every flexibility type: three charging
    points with flexibility bands, two individual and two district heat pumps,
    two storage units, two DSM loads and random component time series.
    """
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path, timeindex=FLEX_INDEX)
    topo = edisgo.topology
    lv = topo.get_lv_grid(3).buses_df.index
    cps = [
        topo.add_load(lv[i], 0.011 * (i + 1), type="charging_point", sector="home")
        for i in range(3)
    ]
    for i, sector in enumerate(["individual_heating"] * 2 + ["district_heating"] * 2):
        topo.add_load(lv[i + 3], 0.005 * (i + 1), type="heat_pump", sector=sector)
    topo.add_storage_unit(lv[2], 0.03)
    rng = np.random.default_rng(42)
    n = len(FLEX_INDEX)
    gens, loads, sto = topo.generators_df, topo.loads_df, topo.storage_units_df
    edisgo.set_time_series_manual(
        generators_p=pd.DataFrame(
            rng.uniform(0, 1, (n, len(gens))) * gens.p_nom.to_numpy(),
            index=FLEX_INDEX,
            columns=gens.index,
        ),
        loads_p=pd.DataFrame(
            rng.uniform(0, 1, (n, len(loads))) * loads.p_set.to_numpy(),
            index=FLEX_INDEX,
            columns=loads.index,
        ),
        storage_units_p=pd.DataFrame(
            rng.uniform(-1, 1, (n, len(sto))) * sto.p_nom.to_numpy(),
            index=FLEX_INDEX,
            columns=sto.index,
        ),
    )
    edisgo.set_time_series_reactive_power_control()
    dsm_loads = list(loads[loads.type == "conventional_load"].index[:2])
    edisgo.dsm.p_min = pd.DataFrame(-0.01, index=FLEX_INDEX, columns=dsm_loads)
    edisgo.dsm.p_max = pd.DataFrame(
        [[0.02, 0.01]] * n, index=FLEX_INDEX, columns=dsm_loads
    )
    edisgo.dsm.e_min = pd.DataFrame(-0.05, index=FLEX_INDEX, columns=dsm_loads)
    edisgo.dsm.e_max = pd.DataFrame(0.05, index=FLEX_INDEX, columns=dsm_loads)
    band = pd.DataFrame(0.01, index=FLEX_INDEX, columns=cps)
    edisgo.electromobility.flexibility_bands = {
        "upper_power": band,
        "lower_energy": band.cumsum() * 0.5,
        "upper_energy": band.cumsum(),
    }
    return edisgo


def _overlying(draw_values, edisgo, renewables):
    n = len(FLEX_INDEX)
    pos = st.lists(st.floats(0.0, 0.05), min_size=n, max_size=n)
    sym = st.lists(st.floats(-0.05, 0.05), min_size=n, max_size=n)
    shares = st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n)
    og = edisgo.overlying_grid
    og.electromobility_active_power = pd.Series(draw_values(pos), index=FLEX_INDEX)
    og.storage_units_active_power = pd.Series(draw_values(sym), index=FLEX_INDEX)
    og.heat_pump_central_active_power = pd.Series(draw_values(pos), index=FLEX_INDEX)
    og.heat_pump_decentral_active_power = pd.Series(draw_values(pos), index=FLEX_INDEX)
    og.dsm_active_power = pd.Series(draw_values(sym), index=FLEX_INDEX)
    og.renewables_curtailment = pd.Series(draw_values(shares), index=FLEX_INDEX) * (
        renewables
    )


def _flex_groups(edisgo):
    loads = edisgo.topology.loads_df
    hp = loads[loads.type == "heat_pump"]
    gens = edisgo.topology.generators_df
    return {
        "cp": loads.index[loads.type == "charging_point"],
        "hp_central": hp.index[hp.sector == "district_heating"],
        "hp_decentral": hp.index[hp.sector == "individual_heating"],
        "dsm": edisgo.dsm.p_max.columns,
        "renewables": gens.index[gens.type.isin(["solar", "wind"])],
    }


class TestOverlyingGrid:
    @settings(
        deadline=None,
        max_examples=examples(8),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data(), band_seed=st.integers(0, 2**32 - 1))
    def test_distribution_conserves_requirements(self, flex_grid, data, band_seed):
        """
        Distributing overlying grid requirements keeps, per time step, the total
        of every technology (charging, storage, central and decentral heat
        pumps, DSM shift, curtailment), splits storage and heat pump power in
        proportion to the rating and charging power in proportion to the upper
        power band, never yields negative feed-in, recomputes Q with the
        default power factors, and leaves the input object unchanged.
        """
        edisgo = copy.deepcopy(flex_grid)
        groups = _flex_groups(edisgo)
        rng = np.random.default_rng(band_seed)
        band = pd.DataFrame(
            rng.uniform(0.001, 0.05, (len(FLEX_INDEX), len(groups["cp"]))),
            index=FLEX_INDEX,
            columns=groups["cp"],
        )
        edisgo.electromobility.flexibility_bands["upper_power"] = band
        renewables = edisgo.timeseries.generators_active_power[groups["renewables"]]
        _overlying(data.draw, edisgo, renewables.sum(axis=1))
        before = copy.deepcopy(edisgo)

        new = distribute_overlying_grid_requirements(edisgo)

        assert_topology_equal(edisgo, before)
        assert_timeseries_equal(edisgo, before)
        og = edisgo.overlying_grid
        ts_old, ts_new = edisgo.timeseries, new.timeseries
        loads, sto = ts_new.loads_active_power, ts_new.storage_units_active_power
        for attr in TIMESERIES_ATTRIBUTES:
            assert not getattr(ts_new, attr).isna().any(axis=None), attr

        np.testing.assert_allclose(
            loads[groups["cp"]].sum(axis=1), og.electromobility_active_power, atol=ATOL
        )
        np.testing.assert_allclose(
            (loads[groups["cp"]] / band).sub(
                og.electromobility_active_power / band.sum(axis=1), axis=0
            ),
            0.0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            sto.sum(axis=1), og.storage_units_active_power, atol=ATOL
        )
        p_nom = edisgo.topology.storage_units_df.p_nom
        ratio = sto[p_nom.index] / p_nom
        np.testing.assert_allclose(ratio.sub(ratio.iloc[:, 0], axis=0), 0.0, atol=1e-9)
        p_set = edisgo.topology.loads_df.p_set
        for key, attr in (
            ("hp_central", "heat_pump_central_active_power"),
            ("hp_decentral", "heat_pump_decentral_active_power"),
        ):
            names = groups[key]
            np.testing.assert_allclose(
                loads[names].sum(axis=1), getattr(og, attr), atol=ATOL
            )
            ratio = loads[names] / p_set[names]
            np.testing.assert_allclose(
                ratio.sub(ratio.iloc[:, 0], axis=0), 0.0, atol=1e-9
            )
        shift = loads[groups["dsm"]] - ts_old.loads_active_power[groups["dsm"]]
        np.testing.assert_allclose(shift.sum(axis=1), og.dsm_active_power, atol=ATOL)
        gens_new = ts_new.generators_active_power[groups["renewables"]]
        np.testing.assert_allclose(
            renewables.sum(axis=1) - gens_new.sum(axis=1),
            og.renewables_curtailment,
            atol=ATOL,
        )
        assert (gens_new >= -ATOL).all(axis=None)
        others = ts_old.generators_active_power.columns.difference(groups["renewables"])
        pd.testing.assert_frame_equal(
            ts_new.generators_active_power[others],
            ts_old.generators_active_power[others],
        )
        for comp in ("generators", "loads", "storage_units"):
            p = getattr(ts_new, f"{comp}_active_power")
            np.testing.assert_allclose(
                getattr(ts_new, f"{comp}_reactive_power")[p.columns],
                _expected_q(new, comp, p),
                rtol=RTOL,
                atol=ATOL,
            )

    @pytest.mark.parametrize("case", ["no_car_parked", "cp_without_band"])
    @pytest.mark.xfail(
        strict=True,
        reason="Bug: distribute_overlying_grid_requirements divides the upper power "
        "band by its row sum without fillna and assigns the result to all charging "
        "points; a time step without any parked car or a charging point without "
        "flexibility band yields NaN charging power",
    )
    def test_distribution_never_creates_nan(self, flex_grid, case):
        """
        The distributed time series contain no NaN, also when no car is parked
        in some time step (zero charging requirement there) or one charging
        point has no flexibility band.
        """
        edisgo = copy.deepcopy(flex_grid)
        cps = _flex_groups(edisgo)["cp"]
        bands = edisgo.electromobility.flexibility_bands
        og = edisgo.overlying_grid
        og.electromobility_active_power = pd.Series(0.01, index=FLEX_INDEX)
        if case == "no_car_parked":
            bands["upper_power"].iloc[2] = 0.0
            og.electromobility_active_power.iloc[2] = 0.0
        else:
            bands["upper_power"] = bands["upper_power"].drop(columns=cps[0])
        new = distribute_overlying_grid_requirements(edisgo)
        assert not new.timeseries.loads_active_power[cps].isna().any(axis=None)

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: curtailment above the renewable feed-in is distributed "
        "without check (negative feed-in), curtailment at zero feed-in is dropped "
        "silently (fillna(0))",
    )
    def test_curtailment_is_conserved_and_feasible(self, flex_grid):
        """
        Distributed curtailment equals the overlying curtailment and never
        makes a generator's feed-in negative, or the infeasible request is
        rejected.
        """
        edisgo = copy.deepcopy(flex_grid)
        gens = _flex_groups(edisgo)["renewables"]
        feed_in = edisgo.timeseries.generators_active_power[gens].sum(axis=1)
        edisgo.timeseries._generators_active_power.loc[FLEX_INDEX[0], gens] = 0.0
        curtailment = 1.5 * feed_in
        curtailment.iloc[0] = 0.1
        edisgo.overlying_grid.renewables_curtailment = curtailment
        new = distribute_overlying_grid_requirements(edisgo)
        gens_new = new.timeseries.generators_active_power[gens]
        assert (gens_new >= 0).all(axis=None)
        np.testing.assert_allclose(
            edisgo.timeseries.generators_active_power[gens].sum(axis=1)
            - gens_new.sum(axis=1),
            curtailment,
        )


# ======================================================================================
# heat and DSM
# ======================================================================================


def _heat_frames(data, n_t=4):
    cols = data.draw(
        st.lists(
            st.sampled_from([f"hp{i}" for i in range(5)]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    values = data.draw(
        st.lists(
            st.floats(0.0, 5.0), min_size=n_t * len(cols), max_size=n_t * len(cols)
        )
    )
    return _frame(values, reference_timeindex[:n_t], cols)


class TestHeatAndDSM:
    @SYN_SETTINGS
    @given(data=st.data(), attr=st.sampled_from(["cop", "heat_demand"]))
    def test_setting_heat_time_series_newest_wins(self, data, attr):
        """
        Setting COP or heat demand twice keeps the union of the heat pumps
        without duplicates, takes overlapping heat pumps from the newer input,
        keeps all others, and setting the same input again changes nothing.
        """
        edisgo = EDisGo()
        first, second = _heat_frames(data), _heat_frames(data)
        setter = getattr(edisgo.heat_pump, f"set_{attr}")
        setter(edisgo, first)
        setter(edisgo, second)
        result = getattr(edisgo.heat_pump, f"{attr}_df")
        assert result.columns.is_unique
        assert set(result.columns) == set(first.columns) | set(second.columns)
        pd.testing.assert_frame_equal(result[second.columns], second)
        rest = first.columns.difference(second.columns)
        pd.testing.assert_frame_equal(result[rest], first[rest])
        once = result.copy()
        setter(edisgo, second)
        pd.testing.assert_frame_equal(getattr(edisgo.heat_pump, f"{attr}_df"), once)

    @CAPLOG_SETTINGS
    @given(data=st.data())
    def test_heat_check_integrity_warns_iff_heat_pumps_differ(self, data, caplog):
        """
        The heat pump integrity check warns about missing entries exactly when
        COP and heat demand are given for different heat pumps.
        """
        edisgo = EDisGo()
        cop, demand = _heat_frames(data), _heat_frames(data)
        edisgo.heat_pump.cop_df = cop
        edisgo.heat_pump.heat_demand_df = demand
        warnings = edisgo_warnings(caplog, edisgo.heat_pump.check_integrity)
        missing = [w for w in warnings if "missing the following entries" in w]
        assert bool(missing) == (set(cop.columns) != set(demand.columns))

    @SYN_SETTINGS
    @given(data=st.data(), freq=st.sampled_from(["15min", "30min"]))
    def test_heat_resampling_keeps_heat_energy(self, data, freq):
        """
        Upsampling heat demand and COP with ffill keeps the heat energy
        (Σ Q·Δt) and the COP values, and downsampling back to hourly with mean
        restores the original series.
        """
        edisgo = EDisGo()
        demand, cop = _heat_frames(data), _heat_frames(data)
        edisgo.heat_pump.heat_demand_df = demand
        edisgo.heat_pump.cop_df = cop
        edisgo.heat_pump.resample_timeseries(method="ffill", freq=freq)
        up = edisgo.heat_pump.heat_demand_df
        dt = pd.Timedelta(freq) / pd.Timedelta("1h")
        np.testing.assert_allclose(up.sum() * dt, demand.sum(), rtol=1e-12, atol=1e-12)
        assert set(np.round(edisgo.heat_pump.cop_df.to_numpy().ravel(), 12)) <= set(
            np.round(cop.to_numpy().ravel(), 12)
        )
        down = up.resample("h").mean()
        pd.testing.assert_frame_equal(down, demand, check_freq=False, check_names=False)

    @CAPLOG_SETTINGS
    @given(
        signs=st.fixed_dictionaries(
            {a: st.booleans() for a in ("p_min", "p_max", "e_min", "e_max")}
        ),
        data=st.data(),
    )
    def test_dsm_check_integrity_warns_iff_sign_violated(self, signs, data, caplog):
        """
        The DSM integrity check warns about an attribute exactly when it has
        the wrong sign somewhere (p_min, e_min > 0 or p_max, e_max < 0). Since
        ``p_min <= 0 <= p_max`` follows, ``p_min <= p_max`` (and likewise for
        e) needs no check of its own.
        """
        edisgo = EDisGo()
        index = reference_timeindex[:3]
        for attr, violate in signs.items():
            sign = -1 if attr.endswith("min") else 1
            values = np.array(
                data.draw(st.lists(st.floats(0.0, 1.0), min_size=6, max_size=6))
            )
            df = _frame(sign * values, index, ["l1", "l2"])
            if violate:
                df.iloc[data.draw(st.integers(0, 2)), data.draw(st.integers(0, 1))] = (
                    -sign * data.draw(st.floats(1e-6, 1.0))
                )
            setattr(edisgo.dsm, attr, df)
        warnings = edisgo_warnings(caplog, edisgo.dsm.check_integrity)
        for attr, violate in signs.items():
            assert (
                any(f"DSM timeseries {attr} contains" in w for w in warnings) == violate
            )
        assert edisgo.dsm.p_min.le(edisgo.dsm.p_max).all(axis=None) or any(
            signs.values()
        )


# ======================================================================================
# results
# ======================================================================================


class TestResults:
    @SYN_SETTINGS
    @given(data=st.data())
    def test_apparent_power_from_active_and_reactive_power(self, data):
        """``s_res² = p² + q²`` elementwise and ``s_res >= max(|p|, |q|)``."""
        results = Results(None)
        index = reference_timeindex[:3]
        p = _frame(
            data.draw(st.lists(finite, min_size=6, max_size=6)), index, ["a", "b"]
        )
        q = _frame(
            data.draw(st.lists(finite, min_size=6, max_size=6)), index, ["a", "b"]
        )
        results.pfa_p, results.pfa_q = p, q
        s = results.s_res
        np.testing.assert_allclose(s**2, p**2 + q**2, rtol=1e-12, atol=1e-12)
        assert (s >= np.maximum(p.abs(), q.abs()) * (1 - 1e-15)).all(axis=None)

    def test_equipment_change_roundtrip(self, wc_grid):
        """
        Adding a line to ``equipment_changes`` records type and number of
        parallel systems; deleting it restores the previous table.
        """
        results = wc_grid.results
        line = wc_grid.topology.lines_df.iloc[3]
        before = results.equipment_changes.copy()
        results._add_line_to_equipment_changes(line)
        row = results.equipment_changes.loc[line.name]
        assert row.equipment == line.type_info and row.quantity == line.num_parallel
        assert row.change == "added"
        results._del_line_from_equipment_changes(line.name)
        after = results.equipment_changes
        assert list(after.index) == list(before.index)
        if not before.empty:
            pd.testing.assert_frame_equal(after, before, check_dtype=False)

    @SYN_SETTINGS
    @given(measures=st.lists(st.text(max_size=5), max_size=4))
    def test_measures_are_appended(self, measures):
        """Every assignment to ``measures`` appends; 'original' stays first."""
        results = Results(None)
        for m in measures:
            results.measures = m
        assert results.measures == ["original"] + measures

    def test_equality_check(self, analyzed_grid):
        """
        ``equality_check`` is reflexive and symmetric and detects a change of a
        single voltage value.
        """
        a = analyzed_grid.results
        b = copy.deepcopy(a)
        assert a.equality_check(b) and b.equality_check(a) and a.equality_check(a)
        v = b.v_res.copy()
        v.iloc[1, 2] += 1e-3
        b.v_res = v
        assert not a.equality_check(b) and not b.equality_check(a)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: Results.from_csv removes 's_res' from the caller's "
        "parameters['powerflow_results'] list in place",
    )
    def test_from_csv_does_not_mutate_parameters(self, analyzed_grid, tmp_path):
        """Reading results back leaves the given ``parameters`` unchanged."""
        analyzed_grid.results.to_csv(tmp_path)
        parameters = {"powerflow_results": ["v_res", "s_res"]}
        Results(analyzed_grid).from_csv(tmp_path, parameters=parameters)
        assert parameters == {"powerflow_results": ["v_res", "s_res"]}


logging.getLogger("pypsa").setLevel(logging.ERROR)
