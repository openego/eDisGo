"""
Property-based tests for the helper modules in :mod:`edisgo.tools`: line
parameters and cable selection (``tools``), time series alignment, parallel
impedances, pseudo coordinates, geo helpers, the graph translation, GeoDataFrame
export, voltage-over-distance data and the configuration container.

The spatial and temporal complexity reduction is tested in
``test_properties_complexity_reduction.py``. Most tests here are synthetic
(hypothesis, milliseconds per example); tests on the ding0 test network only
read a copy of it and run no power flow. Each test states the property it checks.
"""

from __future__ import annotations

import copy
import datetime
import math
import os

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from geopy.distance import geodesic
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from pyproj import Transformer
from shapely.geometry import Point

from edisgo.flex_opt import exceptions, q_control
from edisgo.tools import geo, tools
from edisgo.tools import voltage_over_distance as vod
from edisgo.tools.config import Config
from edisgo.tools.geopandas_helper import to_geopandas
from edisgo.tools.networkx_helper import translate_df_to_graph
from edisgo.tools.pseudo_coordinates import (
    _make_coordinates,
    make_pseudo_coordinates,
    make_pseudo_coordinates_graph,
)

from .conftest import examples

#: Settings for synthetic tests (milliseconds per example).
SYN_SETTINGS = settings(deadline=None, max_examples=examples(60))

#: Settings for hypothesis tests that read the module scoped copy of test network
#: 1 (``grid_ro``) and do not modify it.
GRID_SETTINGS = settings(
    deadline=None,
    max_examples=examples(40),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Relative tolerance for closed-form quantities computed with a handful of float
#: operations (products, quotients, sums of few terms): rounding errors are of the
#: order 1e-16 per operation, 1e-12 leaves ample margin and still catches any wrong
#: factor or unit.
RTOL = 1e-12

#: Absolute tolerance in m for coordinates after a projection roundtrip
#: (EPSG:4326 -> EPSG:3035 -> EPSG:4326 -> EPSG:3035). PROJ is accurate to well
#: below a millimetre for these transformations; 1 mm is far below any line length
#: (the shortest line in the test grids is ~0.5 m).
PROJ_ATOL_M = 1e-3

#: Absolute tolerance in degrees for WGS84 coordinates after a projection
#: roundtrip. PROJ's inverse LAEA (EPSG:3035) uses a series for the authalic
#: latitude and is accurate to ~4e-9 degrees (~0.4 mm); 1e-8 degrees are ~1 mm,
#: consistent with PROJ_ATOL_M.
DEG_ATOL = 1e-8

#: Angular frequency of the 50 Hz grid in 1/s.
OMEGA = 2 * math.pi * 50

COMPONENT_TYPES = (
    "generator",
    "conventional_load",
    "charging_point",
    "heat_pump",
    "storage_unit",
)

positive = st.floats(1e-3, 1e3, allow_nan=False, allow_infinity=False)
lengths = st.floats(1e-3, 50.0)
parallel = st.integers(1, 10)


@pytest.fixture(scope="module")
def grid_ro(_wc_grid_original):
    """Copy of test network 1 shared by the tests of this module (read only)."""
    return copy.deepcopy(_wc_grid_original)


# ======================================================================================
# line parameters, voltage difference and cable selection
# ======================================================================================


class TestLineParameters:
    @SYN_SETTINGS
    @given(
        r=positive, ind=positive, cap=positive, length=lengths, n=parallel, k=parallel
    )
    def test_line_parameters_follow_per_km_formulas(self, r, ind, cap, length, n, k):
        """
        R and X equal the per-km values times length divided by the number of
        parallel systems (L' in mH/km at 50 Hz), B equals C' (uF/km) times length
        times the number of systems, so R and X scale with l/n, B with l*n, B*X does
        not depend on n, and lines in series add up.
        """
        res = tools.calculate_line_resistance(r, length, n)
        rea = tools.calculate_line_reactance(ind, length, n)
        sus = tools.calculate_line_susceptance(cap, length, n)
        assert res == pytest.approx(r * length / n, rel=RTOL)
        assert rea == pytest.approx(OMEGA * ind * 1e-3 * length / n, rel=RTOL)
        assert sus == pytest.approx(OMEGA * cap * 1e-6 * length * n, rel=RTOL)

        for func, value, exp_l, exp_n in (
            (tools.calculate_line_resistance, r, 1, -1),
            (tools.calculate_line_reactance, ind, 1, -1),
            (tools.calculate_line_susceptance, cap, 1, 1),
        ):
            base = func(value, length, n)
            assert func(value, k * length, n) == pytest.approx(
                base * k**exp_l, rel=RTOL
            )
            assert func(value, length, k * n) == pytest.approx(
                base * k**exp_n, rel=RTOL
            )
            # two line sections in series
            assert func(value, length, n) + func(value, k * length, n) == pytest.approx(
                func(value, (1 + k) * length, n), rel=RTOL
            )
        assert tools.calculate_line_reactance(
            ind, length, k * n
        ) * tools.calculate_line_susceptance(cap, length, k * n) == pytest.approx(
            rea * sus, rel=RTOL
        )

    @SYN_SETTINGS
    @given(u=positive, i=positive, n=parallel, k=parallel)
    def test_apparent_power_is_sqrt3_u_i_n(self, u, i, n, k):
        """S = sqrt(3)*U*I*n, i.e. linear in voltage, current and number of systems."""
        s = tools.calculate_apparent_power(u, i, n)
        assert s == pytest.approx(math.sqrt(3) * u * i * n, rel=RTOL)
        assert tools.calculate_apparent_power(k * u, i, n) == pytest.approx(
            k * s, rel=RTOL
        )
        assert tools.calculate_apparent_power(u, k * i, n) == pytest.approx(
            k * s, rel=RTOL
        )
        assert tools.calculate_apparent_power(u, i, k * n) == pytest.approx(
            k * s, rel=RTOL
        )

    @SYN_SETTINGS
    @given(
        s=positive,
        r=positive,
        x=positive,
        v=st.floats(0.2, 110.0),
        pf=st.floats(0.5, 1.0),
        q_sign=st.sampled_from([-1, 1]),
        k=st.floats(0.1, 10.0),
    )
    def test_voltage_diff_is_homogeneous_and_unit_invariant(
        self, s, r, x, v, pf, q_sign, k
    ):
        """
        The VDE-AR-N 4105 voltage difference is linear in S and in the impedance,
        dimensionless (S*k^2 at U*k gives the same value), independent of X and the
        Q sign at cos(phi) = 1, and the two Q signs differ by 2*S*X*sin(phi)/U^2.
        """
        du = tools.calculate_voltage_diff_pu_per_line

        base = du(s, r, x, v, q_sign, pf)
        tol = dict(rel=1e-9, abs=1e-12 * s * (r + x) / v**2)
        assert du(k * s, r, x, v, q_sign, pf) == pytest.approx(k * base, **tol)
        assert du(s, k * r, k * x, v, q_sign, pf) == pytest.approx(k * base, **tol)
        assert du(s * k**2, r, x, v * k, q_sign, pf) == pytest.approx(base, **tol)
        assert du(s, r, x, v, q_sign, 1.0) == pytest.approx(s * r / v**2, rel=RTOL)
        assert du(s, r, 2 * x, v, -q_sign, 1.0) == pytest.approx(s * r / v**2, rel=RTOL)
        sin_phi = math.sqrt(1 - pf**2)
        assert du(s, r, x, v, 1, pf) - du(s, r, x, v, -1, pf) == pytest.approx(
            2 * s * x * sin_phi / v**2, **tol
        )

    @GRID_SETTINGS
    @given(
        data=st.data(),
        level=st.sampled_from(["mv", "lv"]),
        component_type=st.sampled_from(COMPONENT_TYPES),
        length=st.floats(1e-3, 5.0),
        n=st.integers(1, 5),
        s=st.floats(1e-3, 10.0),
        k=st.integers(2, 4),
    )
    def test_voltage_diff_from_type_uses_equipment_and_config(
        self, grid_ro, data, level, component_type, length, n, s, k
    ):
        """
        For every cable type, the voltage difference from the type equals the
        VDE formula with R and X of the cable (R' in Ohm/km, L' in mH/km), Q sign
        and power factor of the component type from the config, and therefore
        scales with the length and inversely with the number of cables.
        """
        cables = grid_ro.topology.equipment_data[f"{level}_cables"]
        name = data.draw(st.sampled_from(sorted(cables.index)))
        v_nom = float(cables.at[name, "U_n"])
        cfg = grid_ro.config
        mode = cfg["reactive_power_mode"][f"{level}_{component_type}"]
        if component_type in ("generator", "storage_unit"):
            q_sign = q_control.get_q_sign_generator(mode)
        else:
            q_sign = q_control.get_q_sign_load(mode)
        pf = cfg["reactive_power_factor"][f"{level}_{component_type}"]

        result = tools.calculate_voltage_diff_pu_per_line_from_type(
            grid_ro, name, length, n, v_nom, s, component_type
        )
        r_tot = cables.at[name, "R_per_km"] * length / n
        x_tot = OMEGA * cables.at[name, "L_per_km"] * 1e-3 * length / n
        expected = s / v_nom**2 * (r_tot * pf + q_sign * x_tot * math.sqrt(1 - pf**2))
        atol = 1e-12 * s * (r_tot + x_tot) / v_nom**2
        assert result == pytest.approx(expected, rel=1e-9, abs=atol)
        for args, factor in (((k * length, n), k), ((length, k * n), 1 / k)):
            scaled = tools.calculate_voltage_diff_pu_per_line_from_type(
                grid_ro, name, *args, v_nom, s, component_type
            )
            assert scaled == pytest.approx(factor * result, rel=1e-9, abs=atol * k)


def _available_cables(edisgo, level):
    cables = edisgo.topology.equipment_data[f"{level}_cables"]
    if level == "mv":
        cables = cables[cables.U_n == edisgo.topology.mv_grid.nominal_voltage]
    return cables


def _feasible_cables(edisgo, level, s, n, length, component_type, max_vd):
    """Cables of ``level`` that carry ``s`` with ``n`` systems within ``max_vd``."""
    cables = _available_cables(edisgo, level)
    ok = np.sqrt(3) * cables.U_n * cables.I_max_th * n > s
    if length != 0:
        vd = pd.Series(
            [
                tools.calculate_voltage_diff_pu_per_line_from_type(
                    edisgo, name, length, n, cables.at[name, "U_n"], s, component_type
                )
                for name in cables.index
            ],
            index=cables.index,
        )
        ok &= vd < max_vd
    return cables[ok]


class TestSelectCable:
    @GRID_SETTINGS
    @given(
        level=st.sampled_from(["mv", "lv"]),
        s_rel=st.floats(0.01, 8.0),
        length=st.one_of(st.just(0.0), st.floats(1e-3, 3.0)),
        component_type=st.sampled_from(COMPONENT_TYPES),
        max_vd=st.one_of(st.none(), st.floats(0.005, 0.1)),
    )
    def test_select_cable_is_sufficient_and_minimal(
        self, grid_ro, level, s_rel, length, component_type, max_vd
    ):
        """
        The selected number of cables is the smallest one for which any available
        cable meets the thermal limit (n*sqrt(3)*U_n*I_th > S) and, for a given
        length, the voltage difference limit; the selected type is the one with
        the smallest I_th among those. If no cable fits with 7 systems, a
        MaximumIterationError is raised.
        """
        cables = _available_cables(grid_ro, level)
        s_one = (np.sqrt(3) * cables.U_n * cables.I_max_th).max()
        s = s_rel * s_one
        limit = (
            max_vd
            or grid_ro.config["grid_connection"][f"{level}_max_voltage_deviation"]
        )
        feasible = {
            n: _feasible_cables(grid_ro, level, s, n, length, component_type, limit)
            for n in range(1, 8)
        }
        n_min = next((n for n, df in feasible.items() if not df.empty), None)

        kwargs = dict(
            level=level,
            apparent_power=s,
            component_type=component_type,
            length=length,
            max_voltage_diff=max_vd,
        )
        if n_min is None:
            with pytest.raises(exceptions.MaximumIterationError):
                tools.select_cable(grid_ro, **kwargs)
            return
        cable, n = tools.select_cable(grid_ro, **kwargs)
        assert n == n_min
        assert cable.name in feasible[n].index
        assert cable.I_max_th == feasible[n].I_max_th.min()

    @GRID_SETTINGS
    @given(
        level=st.sampled_from(["mv", "lv"]),
        s_rel=st.floats(0.01, 5.0),
        ds=st.floats(1.0, 3.0),
        length=st.floats(1e-3, 2.0),
        dl=st.floats(1.0, 3.0),
        component_type=st.sampled_from(COMPONENT_TYPES),
    )
    def test_select_cable_number_is_monotonic(
        self, grid_ro, level, s_rel, ds, length, dl, component_type
    ):
        """
        More power, a longer cable or a stricter voltage difference limit never
        reduces the selected number of parallel cables.
        """
        cables = _available_cables(grid_ro, level)
        s = s_rel * (np.sqrt(3) * cables.U_n * cables.I_max_th).max()
        limit = grid_ro.config["grid_connection"][f"{level}_max_voltage_deviation"]

        def count(**kw):
            args = dict(
                level=level,
                apparent_power=s,
                component_type=component_type,
                length=length,
                max_voltage_diff=limit,
            )
            args.update(kw)
            try:
                return tools.select_cable(grid_ro, **args)[1]
            except exceptions.MaximumIterationError:
                return math.inf

        n = count()
        assume(n < math.inf)
        assert count(apparent_power=s * ds) >= n
        assert count(length=length * dl) >= n
        assert count(max_voltage_diff=limit / ds) >= n


class TestParallelImpedance:
    @SYN_SETTINGS
    @given(r=positive, x=positive, s=positive, n=st.integers(1, 6))
    def test_identical_parallel_components(self, r, x, s, n):
        """n identical components in parallel give r/n, x/n and n*s_nom."""
        df = pd.DataFrame({"r": [r] * n, "x": [x] * n, "s_nom": [s] * n})
        result = tools.calculate_impedance_for_parallel_components(df)
        assert result.r == pytest.approx(r / n, rel=1e-9)
        assert result.x == pytest.approx(x / n, rel=1e-9)
        assert result.s_nom == pytest.approx(s * n, rel=1e-9)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: calculate_impedance_for_parallel_components checks "
        "(df.diff() < 1e-6).all() without abs(), so components with decreasing r, x "
        "and s_nom are treated as identical and r/n, x/n, n*s_nom of the first row "
        "is returned (result depends on the row order, e.g. 0.02+0.1j/0.63 MVA "
        "and 0.01+0.05j/0.4 MVA give 1.26 MVA instead of 0.6 MVA)",
    )
    @SYN_SETTINGS
    @given(
        comps=st.lists(
            st.tuples(positive, positive, st.floats(0.1, 50.0)),
            min_size=2,
            max_size=5,
        ),
        seed=st.integers(0, 2**32 - 1),
    )
    @example(comps=[(0.02, 0.1, 0.63), (0.01, 0.05, 0.4)], seed=0)
    def test_parallel_impedance_matches_admittance_sum_in_any_order(self, comps, seed):
        """
        The parallel impedance satisfies 1/Z = sum(1/Z_i), the rating is the
        total load at which the first component reaches its rating under the
        current divider, and the result does not depend on the row order.
        """
        df = pd.DataFrame(comps, columns=["r", "x", "s_nom"])
        y = 1 / (df.r.to_numpy() + 1j * df.x.to_numpy())
        z = 1 / y.sum()
        s_par = (df.s_nom.to_numpy() / np.abs(y / y.sum())).min()
        perm = np.random.default_rng(seed).permutation(len(df))
        for frame in (df, df.iloc[perm].reset_index(drop=True)):
            result = tools.calculate_impedance_for_parallel_components(frame)
            assert result.r == pytest.approx(z.real, rel=1e-9)
            assert result.x == pytest.approx(z.imag, rel=1e-9)
            assert result.s_nom == pytest.approx(s_par, rel=1e-9)
            assert result.s_nom <= df.s_nom.sum() * (1 + 1e-9)


# ======================================================================================
# data frame and time series helpers
# ======================================================================================


class TestFrameHelpers:
    @SYN_SETTINGS
    @given(
        labels=st.lists(st.integers(0, 5), min_size=0, max_size=15),
        keep=st.sampled_from(["first", "last"]),
    )
    def test_drop_duplicated_keeps_one_row_per_label(self, labels, keep):
        """
        After dropping duplicated indices (columns), every label is unique, no label
        is lost, the kept row (column) is the first or last occurrence, and dropping
        again changes nothing.
        """
        df = pd.DataFrame(
            {"pos": np.arange(len(labels), dtype=float)}, index=pd.Index(labels)
        )
        for result, data in (
            (tools.drop_duplicated_indices(df, keep=keep), df),
            (tools.drop_duplicated_columns(df.T, keep=keep).T, df),
        ):
            assert result.index.is_unique
            assert set(result.index) == set(labels)
            for label in result.index:
                positions = data.pos[data.index == label]
                expected = positions.iloc[0] if keep == "first" else positions.iloc[-1]
                assert result.at[label, "pos"] == expected
        again = tools.drop_duplicated_indices(
            tools.drop_duplicated_indices(df, keep=keep), keep=keep
        )
        pd.testing.assert_frame_equal(again, tools.drop_duplicated_indices(df, keep))

    @SYN_SETTINGS
    @given(
        src_year=st.integers(2010, 2021),
        tgt_year=st.integers(2010, 2021),
        freq=st.sampled_from(["15min", "h", "D"]),
        start=st.integers(0, 364),
        periods=st.integers(1, 400),
        extra_step=st.booleans(),
    )
    @example(
        src_year=2012, tgt_year=2011, freq="h", start=0, periods=400, extra_step=False
    )
    def test_align_series_keeps_month_day_and_time(
        self, src_year, tgt_year, freq, start, periods, extra_step
    ):
        """
        Aligning a full-year series to a time index of another year returns
        exactly that index (plus one step with ``extra_step``), each value is the
        one at the same month, day and time of the source year, 29 February is
        only filled if both years are the same leap year, and aligning twice
        changes nothing.
        """
        src_index = pd.date_range(
            f"{src_year}-01-01", f"{src_year}-12-31 23:59", freq=freq
        )
        ts = pd.Series(np.arange(len(src_index), dtype=float), index=src_index)
        tgt_start = pd.Timestamp(f"{tgt_year}-01-01") + pd.Timedelta(days=start)
        timeindex = pd.date_range(tgt_start, periods=periods, freq=freq)
        timeindex = timeindex[timeindex.year == tgt_year]

        result = tools.align_series_to_timeindex(ts, timeindex, extra_step=extra_step)
        expected_index = (
            timeindex.union([timeindex[-1] + timeindex.freq])
            if extra_step
            else timeindex
        )
        assert result.index.equals(expected_index)
        for t, value in result.items():
            feb_29 = t.month == 2 and t.day == 29
            if t.year != tgt_year or (feb_29 and src_year != tgt_year):
                assert np.isnan(value), t
            elif feb_29 and src_year % 4 != 0:
                assert np.isnan(value), t
            else:
                assert value == ts[t.replace(year=src_year)], t
        again = tools.align_series_to_timeindex(result, timeindex, extra_step)
        pd.testing.assert_series_equal(again, result)

    @SYN_SETTINGS
    @given(
        floats=st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=30),
        ints=st.data(),
    )
    def test_reduce_memory_usage_keeps_values(self, floats, ints):
        """
        Reducing the memory usage keeps integers exactly, floats up to float32
        precision (relative 2**-23, plus 1e-38 absolute for subnormal values) and
        never increases the memory usage.
        """
        n = len(floats)
        int_values = ints.draw(
            st.lists(st.integers(-(2**40), 2**40), min_size=n, max_size=n)
        )
        df = pd.DataFrame({"f": floats, "i": np.array(int_values, dtype="int64")})
        original = df.copy()
        mem = df.memory_usage().sum()
        result = tools.reduce_memory_usage(df)
        assert (result.i.astype("int64") == original.i).all()
        np.testing.assert_allclose(
            result.f.astype(float), original.f, rtol=2**-23, atol=1e-38
        )
        assert result.memory_usage().sum() <= mem

    @SYN_SETTINGS
    @given(
        values=st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=10),
        pos=st.data(),
    )
    def test_hash_dataframe_detects_changes(self, values, pos):
        """Equal frames have equal hashes; changing one value by 1e-3 changes it."""
        df = pd.DataFrame({"a": values})
        assert tools.hash_dataframe(df) == tools.hash_dataframe(df.copy())
        i = pos.draw(st.integers(0, len(values) - 1))
        changed = df.copy()
        changed.iloc[i, 0] += 1e-3
        assert tools.hash_dataframe(df) != tools.hash_dataframe(changed)

    def test_directory_size_counts_all_files(self, tmp_path):
        """The directory size is the sum of all file sizes found recursively."""
        assert tools.get_directory_size(str(tmp_path)) == 0
        sizes = {"a.txt": 10, "sub/b.bin": 123, "sub/deeper/c": 7}
        for rel, size in sizes.items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * size)
        files = tools.get_files_recursive(str(tmp_path))
        assert sorted(os.path.relpath(f, tmp_path) for f in files) == sorted(
            os.path.normpath(rel) for rel in sizes
        )
        assert tools.get_directory_size(str(tmp_path)) == sum(sizes.values())


# ======================================================================================
# helpers on the ding0 test network (no power flow)
# ======================================================================================


class TestGridHelpers:
    def test_worstcase_snapshots_bound_residual_load(self, grid_ro):
        """The selected snapshots are the minimum and maximum of the residual load."""
        snapshots = tools.select_worstcase_snapshots(grid_ro)
        residual = grid_ro.timeseries.residual_load
        assert residual[snapshots["min_residual_load"]] == residual.min()
        assert residual[snapshots["max_residual_load"]] == residual.max()

    def test_add_line_susceptance_modes(self, wc_grid):
        """
        For every line, b = C'(type)*length*num_parallel at 50 Hz with C' from the
        equipment data; 'no_b' sets 0, 'mv_b' only charges MV lines and 'all_b'
        uses 0.25 uF/km for LV lines, so no_b <= mv_b <= all_b elementwise.
        """
        topo = wc_grid.topology
        lines = topo.lines_df
        v_nom = lines.bus0.map(topo.buses_df.v_nom)
        equipment = pd.concat(
            [
                topo.equipment_data[k]
                for k in ("mv_overhead_lines", "mv_cables", "lv_cables")
            ]
        )
        is_lv = v_nom < 1
        cap = pd.Series(
            [
                equipment[equipment.U_n == v].at[t, "C_per_km"]
                for t, v in zip(lines.type_info, v_nom)
            ],
            index=lines.index,
        )
        b = {}
        for mode, c_lv in (("no_b", 0.0), ("mv_b", 0.0), ("all_b", 0.25)):
            tools.add_line_susceptance(wc_grid, mode=mode)
            b[mode] = topo.lines_df.b.copy()
            c = cap.where(~is_lv, c_lv) if mode != "no_b" else cap * 0
            expected = OMEGA * c * 1e-6 * lines.length * lines.num_parallel
            np.testing.assert_allclose(b[mode], expected, rtol=RTOL, atol=0)
        assert (b["no_b"] == 0).all()
        assert (b["mv_b"][is_lv] == 0).all()
        assert (b["no_b"] <= b["mv_b"]).all() and (b["mv_b"] <= b["all_b"]).all()

    def test_assign_voltage_level_follows_nominal_voltage(self, grid_ro):
        """
        A component is assigned 'lv' exactly if its bus has v_nom < 1 kV; the
        input frame is not modified and keeps its rows.
        """
        topo = grid_ro.topology
        for df in (topo.loads_df, topo.generators_df):
            before = df.copy()
            result = tools.assign_voltage_level_to_component(df, topo.buses_df)
            pd.testing.assert_frame_equal(df, before)
            assert result.index.equals(df.index)
            is_lv = topo.buses_df.loc[df.bus, "v_nom"].to_numpy() < 1
            assert ((result.voltage_level == "lv").to_numpy() == is_lv).all()
            assert set(result.voltage_level) <= {"mv", "lv"}

    @SYN_SETTINGS
    @given(p1=st.floats(1e-6, 20.0), p2=st.floats(1e-6, 20.0))
    def test_grid_integration_level_is_monotonic(self, grid_ro, p1, p2):
        """
        The integration voltage level never rises with the power, is one of
        4..7 and the configured upper limit of a level still belongs to it.
        """
        levels = [
            tools.determine_grid_integration_voltage_level(grid_ro, p)
            for p in sorted((p1, p2))
        ]
        assert set(levels) <= {4, 5, 6, 7}
        assert levels[0] >= levels[1]
        cfg = grid_ro.config["grid_connection"]
        for level in (4, 5, 6, 7):
            limit = cfg[f"upper_limit_voltage_level_{level}"]
            assert tools.determine_grid_integration_voltage_level(grid_ro, limit) == (
                level
            )
        for power in (0.0, -1.0, cfg["upper_limit_voltage_level_4"] * 1.01):
            with pytest.raises(ValueError):
                tools.determine_grid_integration_voltage_level(grid_ro, power)

    def test_bus_voltage_level_matches_topology(self, grid_ro):
        """
        Every bus is level 4/5 if v_nom >= 1 kV and 6/7 otherwise; station buses
        are 4 or 6 and buses with more than one line never are (busbar levels).
        """
        topo = grid_ro.topology
        stations = set(topo.transformers_df.bus1) | set(topo.transformers_hvmv_df.bus1)
        n_lines = pd.concat([topo.lines_df.bus0, topo.lines_df.bus1]).value_counts()
        for bus, row in topo.buses_df.iterrows():
            if n_lines.get(bus, 0) == 0 and bus not in stations:
                continue  # see A21: buses without a line raise an IndexError
            level = tools.determine_bus_voltage_level(grid_ro, bus)
            assert level in ((4, 5) if row.v_nom >= 1 else (6, 7)), bus
            if bus in stations:
                assert level in (4, 6), bus
            elif n_lines.get(bus, 0) > 1:
                assert level in (5, 7), bus

    @pytest.mark.xfail(
        strict=True,
        reason="Bug (known, #681): get_path_length_to_station counts edges for MV "
        "buses but nodes for LV buses, so every LV bus is one step farther than its "
        "edge count incl. the MV/LV transformer (two steps without it)",
    )
    def test_path_length_to_station_counts_edges(self, wc_grid):
        """
        The path length of every bus equals the number of edges on its path to
        the HV/MV station in the graph including MV/LV transformers.
        """
        lengths = tools.get_path_length_to_station(wc_grid)
        graph = wc_grid.topology.to_graph()
        station = wc_grid.topology.mv_grid.station.index[0]
        expected = pd.Series(nx.shortest_path_length(graph, station))
        # BusBar_lac_1 (LV station of an LV grid without lines) gets no value
        known = lengths.dropna().index.intersection(expected.index)
        assert (lengths.loc[known] == expected.loc[known]).all()

    def test_downstream_buses_are_nested_and_partition_the_grid(self, grid_ro):
        """
        The downstream buses of a bus contain the bus itself but never the HV/MV
        station, are nested along every line, equal all buses of the LV grid for
        an LV station, and those of the station's neighbours partition all other
        buses; for a line they equal those of its bus farther from the station.
        """
        topo = grid_ro.topology
        graph = topo.to_graph()
        station = topo.transformers_hvmv_df.bus1.iloc[0]
        dist = nx.shortest_path_length(graph, station)
        down = {
            bus: set(tools.get_downstream_buses(grid_ro, bus))
            for bus in topo.buses_df.index
            if bus != station
        }
        for bus, buses in down.items():
            assert bus in buses and station not in buses
        for name, line in topo.lines_df.iterrows():
            near, far = sorted((line.bus0, line.bus1), key=dist.get)
            if near != station:
                assert down[far] <= down[near], name
            assert set(tools.get_downstream_buses(grid_ro, name, "line")) == down[far]
        for grid in topo.lv_grids:
            assert down[grid.station.index[0]] == set(grid.buses_df.index)
        parts = [down[nb] for nb in graph.neighbors(station)]
        assert sum(len(p) for p in parts) == len(topo.buses_df) - 1
        assert set().union(*parts) == set(topo.buses_df.index) - {station}

    def test_graph_translation_matches_frames(self, grid_ro):
        """
        The graph has exactly the buses as nodes with their coordinates as
        ``pos``, one edge per line (or distinct bus pair) with the line length,
        transformer edges of length 0, and every grid's graph is a tree.
        """
        topo = grid_ro.topology
        graph = translate_df_to_graph(topo.buses_df, topo.lines_df)
        assert set(graph.nodes) == set(topo.buses_df.index)
        for bus, row in topo.buses_df.iterrows():
            # some buses have no coordinates (NaN), assert_equal treats NaN == NaN
            np.testing.assert_equal(graph.nodes[bus]["pos"], (row.x, row.y))
        pairs = {
            frozenset((b0, b1))
            for b0, b1 in zip(topo.lines_df.bus0, topo.lines_df.bus1)
        }
        assert graph.number_of_edges() == len(pairs)
        for name, line in topo.lines_df.iterrows():
            edge = graph.edges[line.bus0, line.bus1]
            if edge["branch_name"] == name:
                assert edge["length"] == line.length
        with_trafos = translate_df_to_graph(
            topo.buses_df, topo.lines_df, topo.transformers_df
        )
        for name, trafo in topo.transformers_df.iterrows():
            assert with_trafos.edges[trafo.bus0, trafo.bus1]["length"] == 0
        for grid in topo.grids:
            assert nx.is_tree(grid.graph), str(grid)


# ======================================================================================
# pseudo coordinates
# ======================================================================================

#: Transformations used to measure distances of WGS84 coordinates in metres.
TO_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


@st.composite
def random_trees(draw, max_nodes=25):
    """A random tree with node 0 first, every node attached to an earlier one."""
    n = draw(st.integers(2, max_nodes))
    graph = nx.Graph()
    graph.add_node(0, pos=(draw(st.floats(6.0, 14.0)), draw(st.floats(48.0, 54.0))))
    for node in range(1, n):
        parent = draw(st.integers(0, node - 1))
        graph.add_node(node, pos=(0.0, 0.0))
        graph.add_edge(parent, node, length=draw(st.floats(1e-3, 2.0)))
    return graph


class TestPseudoCoordinates:
    @SYN_SETTINGS
    @given(graph=random_trees(), detour=st.floats(1.0, 2.0), k=st.floats(0.5, 4.0))
    def test_make_coordinates_preserves_line_lengths(self, graph, detour, k):
        """
        In the equidistant plane the first node lies at (0, 0) and every edge is
        1000*length/detour metres long; scaling all lengths by k scales all
        coordinates by k.
        """
        result = _make_coordinates(copy.deepcopy(graph), detour)
        pos = nx.get_node_attributes(result, "pos")
        assert pos[0] == (0, 0)
        assert np.isfinite(np.array(list(pos.values()))).all()
        for u, v, length in graph.edges(data="length"):
            assert math.dist(pos[u], pos[v]) == pytest.approx(
                1000 * length / detour, rel=1e-9
            )
        scaled = copy.deepcopy(graph)
        for u, v in scaled.edges:
            scaled.edges[u, v]["length"] *= k
        pos_k = nx.get_node_attributes(_make_coordinates(scaled, detour), "pos")
        for node, (x, y) in pos.items():
            assert pos_k[node] == pytest.approx((k * x, k * y), rel=1e-9, abs=1e-9)

    @SYN_SETTINGS
    @given(graph=random_trees(), detour=st.floats(1.0, 2.0))
    def test_pseudo_coordinates_graph_keeps_root_and_lengths(self, graph, detour):
        """
        In WGS84, the first node keeps its coordinates and the EPSG:3035 distance
        of every edge times the detour factor equals its length.
        """
        root = graph.nodes[0]["pos"]
        result = make_pseudo_coordinates_graph(copy.deepcopy(graph), detour)
        assert result.nodes[0]["pos"] == pytest.approx(root, abs=DEG_ATOL)
        for u, v, length in graph.edges(data="length"):
            pu = TO_3035.transform(*result.nodes[u]["pos"])
            pv = TO_3035.transform(*result.nodes[v]["pos"])
            assert math.dist(pu, pv) == pytest.approx(
                1000 * length / detour, abs=PROJ_ATOL_M
            )

    @pytest.mark.parametrize("mv_coordinates", [False, True])
    def test_make_pseudo_coordinates_on_grid(self, wc_grid, mv_coordinates):
        """
        In the ding0 grid, every LV line (and every MV line with
        ``mv_coordinates``) is as long as the EPSG:3035 distance of its buses
        times the detour factor, the stations keep their coordinates, and MV
        buses stay unchanged without ``mv_coordinates``.
        """
        topo = wc_grid.topology
        before = topo.buses_df[["x", "y"]].copy()
        make_pseudo_coordinates(wc_grid, mv_coordinates=mv_coordinates)
        after = topo.buses_df[["x", "y"]]
        detour = wc_grid.config["grid_connection"]["branch_detour_factor"]
        grids = list(topo.lv_grids) + ([topo.mv_grid] if mv_coordinates else [])
        xy = dict(zip(after.index, zip(*TO_3035.transform(after.x, after.y))))
        for grid in grids:
            station = grid.station.index[0]
            assert after.loc[station].tolist() == pytest.approx(
                before.loc[station].tolist(), abs=DEG_ATOL
            )
            for name, line in grid.lines_df.iterrows():
                dist = math.dist(xy[line.bus0], xy[line.bus1])
                assert dist * detour / 1000 == pytest.approx(
                    line.length, abs=PROJ_ATOL_M * detour / 1000
                ), name
        if not mv_coordinates:
            mv_buses = topo.mv_grid.buses_df.index
            pd.testing.assert_frame_equal(after.loc[mv_buses], before.loc[mv_buses])


# ======================================================================================
# geo helpers and GeoDataFrames
# ======================================================================================

lons = st.floats(5.0, 15.0)
lats = st.floats(47.0, 55.0)


def _topology_stub(points):
    """Minimal topology with buses at the given (lon, lat) points."""
    df = pd.DataFrame(
        points, columns=["x", "y"], index=[f"b{i}" for i in range(len(points))]
    )
    return SimpleNamespace(buses_df=df)


class TestGeo:
    @settings(deadline=None, max_examples=examples(10))
    @given(points=st.lists(st.tuples(lons, lats), min_size=1, max_size=50))
    def test_equidistant_projection_roundtrip(self, points):
        """
        Projecting to EPSG:3035 and back is the identity (DEG_ATOL), also via
        ``proj_by_srids``, and 0.01 degrees of latitude are 1.11 km (+-1 %).
        """
        # creating a transformer takes ~0.1 s, so all points are projected at once
        lon, lat = np.array(points).T
        fwd, back = geo.proj2equidistant(4326), geo.proj2equidistant_reverse(4326)
        np.testing.assert_allclose(back(*fwd(lon, lat)), (lon, lat), atol=DEG_ATOL)
        there, home = geo.proj_by_srids(4326, 3857), geo.proj_by_srids(3857, 4326)
        np.testing.assert_allclose(home(*there(lon, lat)), (lon, lat), atol=DEG_ATOL)
        x0, y0 = fwd(lon, lat)
        x1, y1 = fwd(lon, lat + 0.01)
        np.testing.assert_allclose(np.hypot(x1 - x0, y1 - y0), 1112.0, rtol=0.01)

    @SYN_SETTINGS
    @given(
        points=st.lists(st.tuples(lons, lats), min_size=3, max_size=3),
        detour=st.floats(1.0, 2.0),
    )
    def test_geo_distance_is_a_scaled_metric(self, points, detour):
        """
        The geodesic bus distance is symmetric, non-negative, 1 m for identical
        positions, linear in the detour factor and satisfies the triangle
        inequality. (Only an exact zero is raised to 1 m, positions a few mm
        apart give lines of a few mm.)
        """
        topo = _topology_stub(points)
        d = lambda a, b, f=detour: geo.calc_geo_dist_vincenty(topo, a, b, f)  # noqa
        assert d("b0", "b1") == pytest.approx(d("b1", "b0"), rel=1e-12)
        assert d("b0", "b0") == 0.001
        for a, b in (("b0", "b1"), ("b1", "b2"), ("b0", "b2")):
            assert d(a, b) > 0
            plain = d(a, b, 1.0)
            if a != b and points[int(a[1])] != points[int(b[1])]:
                assert d(a, b) == pytest.approx(detour * plain, rel=1e-12)
        assume(len(set(points)) == 3)
        assert d("b0", "b2") <= (d("b0", "b1") + d("b1", "b2")) * (1 + 1e-12)

    @SYN_SETTINGS
    @given(
        center=st.tuples(lons, lats),
        offsets=st.lists(
            st.tuples(st.floats(-0.05, 0.05), st.floats(-0.05, 0.05)),
            min_size=1,
            max_size=20,
        ),
        point=st.tuples(st.floats(-0.05, 0.05), st.floats(-0.05, 0.05)),
        hit=st.booleans(),
    )
    def test_find_nearest_bus_is_near_optimal(self, center, offsets, point, hit):
        """
        The nearest bus is a candidate, its returned distance is the exact
        geodesic and at most 0.3 % (documented accuracy) longer than the
        distance to the truly nearest candidate; a point on a candidate is found
        with distance 0.
        """
        cands = [(center[0] + dx, center[1] + dy) for dx, dy in offsets]
        target = _topology_stub(cands).buses_df
        p = cands[0] if hit else (center[0] + point[0], center[1] + point[1])
        name, dist = geo.find_nearest_bus(Point(p), target)
        exact = [geodesic((p[1], p[0]), (y, x)).km for x, y in cands]
        assert name in target.index
        row = target.loc[name]
        assert dist == pytest.approx(geodesic((p[1], p[0]), (row.y, row.x)).km)
        assert dist <= min(exact) * (1 + 3e-3) + 1e-9
        if hit:
            assert dist == pytest.approx(0, abs=1e-9)

    def test_geopandas_geometries_match_bus_coordinates(self, grid_ro):
        """
        Bus points are the bus coordinates, every line runs from bus0 to bus1 and
        every component lies at its bus (transformers at bus1); no row is lost.
        """
        topo = grid_ro.topology
        srid = topo.grid_district["srid"]
        gdfs = to_geopandas(topo, srid)
        buses = topo.buses_df
        assert gdfs.buses_gdf.crs == f"EPSG:{srid}"
        np.testing.assert_array_equal(gdfs.buses_gdf.geometry.x, buses.x)
        np.testing.assert_array_equal(gdfs.buses_gdf.geometry.y, buses.y)
        lines = gdfs.lines_gdf
        assert set(lines.index) == set(topo.lines_df.index)
        for name, geom in lines.geometry.items():
            b0, b1 = topo.lines_df.loc[name, ["bus0", "bus1"]]
            # some buses have no coordinates (NaN), assert_equal treats NaN == NaN
            np.testing.assert_equal(
                list(geom.coords),
                [tuple(buses.loc[b0, ["x", "y"]]), tuple(buses.loc[b1, ["x", "y"]])],
            )
        for comp, col in (
            ("loads", "bus"),
            ("generators", "bus"),
            ("storage_units", "bus"),
            ("transformers", "bus1"),
        ):
            gdf = getattr(gdfs, f"{comp}_gdf")
            df = getattr(topo, f"{comp}_df")
            assert set(gdf.index) == set(df.index), comp
            for name, geom in gdf.geometry.items():
                bus = df.at[name, col]
                np.testing.assert_equal(
                    (geom.x, geom.y), tuple(buses.loc[bus, ["x", "y"]])
                )


# ======================================================================================
# voltage over distance
# ======================================================================================

bus_names = st.lists(
    st.text("abcdef", min_size=1, max_size=3), unique=True, max_size=12
)
maybe = st.one_of(st.none(), st.floats(0.8, 1.2))


class TestVoltageOverDistance:
    @SYN_SETTINGS
    @given(data=st.data(), buses=bus_names, other=st.booleans())
    def test_voltage_over_distance_rows_are_not_mixed_up(self, data, buses, other):
        """
        Every row holds the distance and voltage of its own bus, the rows are
        exactly the buses with both values known per label, and there are four
        labels with a second scenario, two otherwise.
        """

        def series():
            values = data.draw(
                st.lists(maybe, min_size=len(buses), max_size=len(buses))
            )
            return pd.Series(values, index=buses, dtype=float)

        inputs = {k: series() for k in ("dist_a", "v_a_load", "v_a_feed")}
        if other:
            inputs.update({k: series() for k in ("dist_b", "v_b_load", "v_b_feed")})
        df = vod.build_voltage_over_distance_df(buses=pd.Index(buses), **inputs)
        cases = [
            ("a", "base — load case", "load"),
            ("a", "base — feed-in case", "feed"),
        ]
        if other:
            cases += [
                ("b", "other — load case", "load"),
                ("b", "other — feed-in case", "feed"),
            ]
        for scen, label, case in cases:
            dist, v = inputs[f"dist_{scen}"], inputs[f"v_{scen}_{case}"]
            sub = df[df.label == label].set_index("bus")
            known = [b for b in buses if not (np.isnan(dist[b]) or np.isnan(v[b]))]
            assert sorted(sub.index) == sorted(known)
            for bus in known:
                assert sub.at[bus, "x"] == dist[bus]
                assert sub.at[bus, "v_pu"] == v[bus]
        assert df.label.nunique() <= (4 if other else 2)
        fig = vod.make_voltage_over_distance_figure(
            title="t", buses=pd.Index(buses), band_low=0.9, band_high=1.1, **inputs
        )
        assert len(fig.data) == df.label.nunique()
        assert sum(len(trace.x) for trace in fig.data) == len(df)

    @SYN_SETTINGS
    @given(
        values=st.lists(
            st.lists(st.floats(0.8, 1.2), min_size=3, max_size=3),
            min_size=2,
            max_size=10,
        ),
        offset=st.integers(-3, 15),
    )
    def test_infer_and_select_time_steps(self, values, offset):
        """
        The load case is the step with the lowest bus voltage, the feed-in case
        the one with the highest; a voltage row is taken at the exact or else
        the nearest time step.
        """
        idx = pd.date_range("2011-01-01", periods=len(values), freq="h")
        v_df = pd.DataFrame(values, index=idx, columns=["b0", "b1", "b2"])
        t_load, t_feed = vod._infer_load_and_feedin_timesteps(None, v_df)
        assert v_df.loc[t_load].min() == v_df.min().min()
        assert v_df.loc[t_feed].max() == v_df.max().max()
        t = idx[0] + pd.Timedelta(hours=offset) + pd.Timedelta(minutes=10)
        nearest = idx[np.argmin(np.abs((idx - t).total_seconds()))]
        pd.testing.assert_series_equal(
            vod._series_at_t(v_df, t), v_df.loc[nearest].rename(nearest)
        )


# ======================================================================================
# configuration
# ======================================================================================


class TestConfig:
    @pytest.mark.xfail(
        strict=True,
        reason="Bug: Config.from_cfg returns the _sections dict of the module global "
        "RawConfigParser, so all Config (and EDisGo) instances share one dict: "
        "changing one instance changes the others, and a new instance resets them",
    )
    def test_config_instances_are_independent(self):
        """
        Changing a value in one Config instance neither changes another instance
        nor is reset by creating a further instance.
        """
        section, key = "grid_connection", "branch_detour_factor"
        first, second = Config(), Config()
        original = second[section][key]
        try:
            first[section][key] = original + 1.0
            assert second[section][key] == original
            Config()
            assert first[section][key] == original + 1.0
        finally:
            # the shared dict is also used by the session fixtures: restore it
            first[section][key] = original
            second[section][key] = original

    @SYN_SETTINGS
    @given(
        value=st.floats(-1e9, 1e9, allow_nan=False),
        start=st.times().map(lambda t: t.replace(second=0, microsecond=0)),
        zipped=st.booleans(),
    )
    def test_json_roundtrip_keeps_values(self, tmp_path_factory, value, start, zipped):
        """
        A changed numeric value and demandlib time survive ``to_json`` and
        ``from_json``, also from a zip archive, and all other values stay equal.
        """
        cfg = Config()
        data = copy.deepcopy(cfg._data)
        data["grid_connection"]["branch_detour_factor"] = value
        data["demandlib"]["day_start"] = start
        cfg._data = data
        directory = tmp_path_factory.mktemp("cfg")
        cfg.to_json(str(directory))
        if zipped:
            import zipfile

            archive = directory / "archive.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.write(directory / "configs.json", "configs.json")
            loaded = Config(
                from_json=True, config_path=str(archive), from_zip_archive=True
            )
        else:
            loaded = Config(from_json=True, config_path=str(directory))
        assert loaded._data == data

    @SYN_SETTINGS
    @given(t=st.times().map(lambda t: t.replace(second=0, microsecond=0)))
    def test_parse_time_roundtrip(self, t):
        """Parsing the string of a time without seconds gives the same time."""
        assert Config._parse_time(str(t)) == t
        assert Config._parse_time(t.strftime("%H:%M")) == t
        assert Config._parse_time(t) is t

    @pytest.mark.parametrize("value", ["", "25:00", "noon", "12-30"])
    def test_parse_time_rejects_invalid(self, value):
        """Strings that are no time of day raise a ValueError."""
        with pytest.raises(ValueError):
            Config._parse_time(value)

    def test_default_config_types(self):
        """Numeric config values are floats and the demandlib times are times."""
        cfg = Config()
        for sec, values in cfg._data.items():
            for key, val in values.items():
                if isinstance(val, str):
                    try:
                        float(val)
                    except ValueError:
                        continue
                    pytest.fail(f"{sec}.{key} = {val!r} is numeric but a string")
        for key in ("day_start", "day_end"):
            assert isinstance(cfg["demandlib"][key], datetime.time)
