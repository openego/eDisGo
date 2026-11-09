"""
Property-based tests for the data exchange in :mod:`edisgo.io`.

Covered are the exports to PyPSA and PowerModels (units, per-unit bases, mapping of
components, round trip of OPF results without Julia), the processing of power flow
results, the ding0 grid import and the grid integration and profile helpers of the
generator, heat pump, battery, DSM, time series and electromobility imports.

Each test states the property it checks. Database (OEP) and Julia are not needed:
database queries are replaced by synthetic data via ``monkeypatch``. Fixtures and
helpers shared by all property tests live in ``tests/properties/conftest.py``.
"""

from __future__ import annotations

import copy
import json
import math
import os

import numpy as np
import pandas as pd
import pypsa
import pytest

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from edisgo import EDisGo
from edisgo.io import (
    dsm_import,
    electromobility_import,
    generators_import,
    heat_pump_import,
    powermodels_io,
    pypsa_io,
    storage_import,
    timeseries_import,
)

from .conftest import (
    POWER_ATOL,
    TIMESERIES_ATTRIBUTES,
    examples,
    run_pypsa_power_flow,
)

FIXTURE_SETTINGS = settings(
    deadline=None,
    max_examples=examples(5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Relative tolerance for values that are only rescaled (divided and multiplied by
#: s_base, unit conversions): float rounding of a handful of operations is < 1e-15
#: relative; 1e-9 leaves ample margin and still detects any wrong factor.
RTOL_RESCALE = 1e-9

#: Number of hourly time steps of the grid used for the PowerModels tests. More
#: than nine, so that a lexicographic instead of numeric ordering of the
#: PowerModels time step keys ("10" < "2") would be detected.
N_STEPS = 12


# ======================================================================================
# helpers
# ======================================================================================


def _extend_timeindex(edisgo, n_steps, start="2011-01-01"):
    """
    Replace the worst-case time series by ``n_steps`` hourly time steps.

    The four worst-case snapshots are repeated cyclically and scaled with
    reproducible random factors per time step, so that consecutive time steps
    differ. P and Q of a component get the same factor (constant power factor).
    """
    ts = edisgo.timeseries
    index = pd.date_range(start, periods=n_steps, freq="h")
    rng = np.random.default_rng(42)
    factors = rng.uniform(0.3, 1.0, size=n_steps)
    frames = {attr: getattr(ts, attr) for attr in TIMESERIES_ATTRIBUTES}
    ts.timeindex = index
    for attr, df in frames.items():
        rows = np.arange(n_steps) % len(df)
        values = df.to_numpy()[rows] * factors[:, None]
        setattr(ts, attr, pd.DataFrame(values, index=index, columns=df.columns))
    return edisgo


def _build_flex_grid():
    """
    Test network 1 with 12 hourly time steps and one flexibility of every kind.

    Adds two flexible heat pumps with thermal storage, a flexible charging point,
    one DSM load and initial states of charge for the battery and thermal storage
    units, all with time series on the grid's time index.
    """
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
    edisgo.set_time_series_worst_case_analysis()
    _extend_timeindex(edisgo, N_STEPS)
    ti = edisgo.timeseries.timeindex
    ti_ext = ti.union([ti[-1] + ti.freq])
    buses = edisgo.topology.buses_df.index
    rng = np.random.default_rng(1)

    hps = []
    for bus, p_set in ((buses[26], 0.02), (buses[30], 0.03)):
        hps.append(
            edisgo.add_component(
                comp_type="load",
                type="heat_pump",
                sector="individual_heating",
                ts_active_power=pd.Series(rng.uniform(0.0, p_set, len(ti)), index=ti),
                ts_reactive_power="default",
                bus=bus,
                p_set=p_set,
            )
        )
    cop = pd.DataFrame(rng.uniform(2.0, 4.0, (len(ti), 2)), index=ti, columns=hps)
    edisgo.heat_pump.cop_df = cop
    edisgo.heat_pump.heat_demand_df = edisgo.timeseries.loads_active_power[hps] * cop
    edisgo.heat_pump.thermal_storage_units_df = pd.DataFrame(
        {"capacity": [0.05, 0.08], "efficiency": [0.95, 0.95]}, index=hps
    )
    edisgo.overlying_grid.thermal_storage_units_decentral_soc = pd.Series(
        0.4, index=ti_ext
    )

    cp = edisgo.add_component(
        comp_type="load",
        type="charging_point",
        ts_active_power=pd.Series(0.01, index=ti),
        ts_reactive_power="default",
        bus=buses[32],
        p_set=0.022,
    )
    lower = np.cumsum(np.full(len(ti), 0.005))
    edisgo.electromobility.flexibility_bands = {
        "upper_power": pd.DataFrame({cp: 0.022}, index=ti),
        "lower_energy": pd.DataFrame({cp: lower}, index=ti),
        "upper_energy": pd.DataFrame({cp: lower + 0.03}, index=ti),
    }

    dsm_load = "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1"
    for attr, value in (("p_min", -0.1), ("p_max", 0.1), ("e_min", -0.2)):
        setattr(edisgo.dsm, attr, pd.DataFrame({dsm_load: value}, index=ti))
    edisgo.dsm.e_max = pd.DataFrame({dsm_load: 0.2}, index=ti)

    edisgo.overlying_grid.storage_units_soc = pd.Series(0.5, index=ti_ext)
    flex = {
        "flexible_cps": np.array([cp]),
        "flexible_hps": np.array(hps),
        "flexible_loads": np.array([dsm_load]),
        "flexible_storage_units": edisgo.topology.storage_units_df.index.to_numpy(),
    }
    return edisgo, flex


@pytest.fixture(scope="module")
def _flex_grid_original():
    return _build_flex_grid()


@pytest.fixture
def flex_grid(_flex_grid_original):
    """Fresh copy of the grid with all kinds of flexibilities and their names."""
    edisgo, flex = _flex_grid_original
    return copy.deepcopy(edisgo), copy.deepcopy(flex)


def _to_pm(edisgo, s_base=1, flex=None, opf_version=1):
    """Run ``to_powermodels`` on a copy and return the PowerModels dict."""
    pm, _ = powermodels_io.to_powermodels(
        copy.deepcopy(edisgo), s_base=s_base, opf_version=opf_version, **(flex or {})
    )
    return pm


def _pm_bus_name(pm, idx):
    return pm["bus"][str(idx)]["name"]


def _pm_branches(pm, transformer):
    """Real (non-virtual) branches of ``pm`` that are (not) transformers."""
    return [
        b
        for b in pm["branch"].values()
        if b["transformer"] == transformer and not b["storage"]
    ]


def _synthetic_pm_results(pm, edisgo, s_base, order_seed=0):
    """
    Build an OPF result dict that reproduces the current flexibility dispatch.

    Every time step ``nw[t]`` carries the component names of ``pm`` and, as
    "optimised" values, the active power that the flexibilities already have in
    ``edisgo.timeseries``: charging points (``pcp``) and heat pumps (``php``)
    their load, DSM a shift of zero (``pdsm``) and flexible storage units the
    negative storage power as flow on their virtual branch (``pf``), since the
    branch points from the grid bus to the storage. All other variables are zero.
    The time steps are inserted in shuffled order, as JSON objects do not
    guarantee an order.
    """
    ts = edisgo.timeseries
    loads_p = ts.loads_active_power
    storage_p = ts.storage_units_active_power
    virtual = {s["virtual_branch"]: s["name"] for s in pm["storage"].values()}
    variables = {
        "electromobility": ("pcp",),
        "heatpumps": ("php", "phps", "phps2"),
        "dsm": ("pdsm",),
        "storage": ("ps", "se"),
        "heat_storage": ("phs", "hse", "phss"),
        "gen_slack": ("pgs", "qgs"),
        "gen": ("pgens",),
        "gen_nd": ("pgc",),
        "load": ("pds",),
        "branch": ("pf", "qf", "ccm"),
    }
    steps = list(range(1, len(ts.timeindex) + 1))
    np.random.default_rng(order_seed).shuffle(steps)
    nw = {}
    for t in steps:
        time = ts.timeindex[t - 1]
        step = {"flexibilities": pm["flexibilities"], "opf_version": pm["opf_version"]}
        for comp, names in variables.items():
            step[comp] = {
                key: {**copy.deepcopy(value), **{v: 0.0 for v in names}}
                for key, value in pm[comp].items()
            }
        for value in step["electromobility"].values():
            value["pcp"] = loads_p.at[time, value["name"]] / s_base
        for value in step["heatpumps"].values():
            value["php"] = loads_p.at[time, value["name"]] / s_base
        for key, name in virtual.items():
            step["branch"][key]["pf"] = -storage_p.at[time, name] / s_base
        nw[str(t)] = step
    return {
        "solve_time": 0.1,
        "status": "LOCALLY_SOLVED",
        "solver": "Ipopt",
        "nw": nw,
    }


# ======================================================================================
# PowerModels export: units and per-unit bases
# ======================================================================================


class TestPowerModelsUnits:
    @pytest.mark.parametrize("s_base", [1, 10])
    def test_line_impedance_in_ohm(self, wc_grid, s_base):
        """
        Line impedances are converted to p.u. on the base Z = U²/S_base, i.e.
        ``br_r·U²/S_base`` and ``br_x·U²/S_base`` give r and x in Ohm. The export
        raises the smallest values (lowest 0.2 % quantile, ties included) to a
        common floor for numerical conditioning; only values below that floor
        may differ.
        """
        pm = _to_pm(wc_grid, s_base=s_base)
        topo = wc_grid.topology
        branches = _pm_branches(pm, transformer=False)
        assert {b["name"] for b in branches} == set(topo.lines_df.index)
        for par in ("r", "x"):
            pu = np.array([b[f"br_{par}"] for b in branches]) / s_base
            floor = pu.min()
            names = [b["name"] for b in branches]
            v_nom = topo.buses_df.loc[topo.lines_df.loc[names, "bus0"], "v_nom"]
            ohm = pu * v_nom.to_numpy() ** 2
            expected = topo.lines_df.loc[names, par].to_numpy()
            exact = ~np.isclose(pu, floor, rtol=RTOL_RESCALE, atol=0)
            np.testing.assert_allclose(ohm[exact], expected[exact], rtol=RTOL_RESCALE)
            # raised values: their own p.u. value lies at or below the floor
            own_pu = expected / v_nom.to_numpy() ** 2
            assert (own_pu[~exact] <= floor * (1 + RTOL_RESCALE)).all(), par

    @pytest.mark.parametrize(
        "s_base",
        [
            1,
            pytest.param(
                10,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Bug: _build_branch multiplies b_pu and g_pu with s_base; "
                    "the admittance base is S_base/U², so they must be divided.",
                ),
            ),
        ],
    )
    def test_line_susceptance_in_siemens(self, wc_grid, s_base):
        """
        Shunt susceptance is converted to p.u. on the admittance base S_base/U², so
        ``(b_fr + b_to)·U²/S_base`` gives the line's b in Siemens. The test grid
        has b = 0, so a realistic cable susceptance (0.25 µF/km at 50 Hz) is set.
        """
        topo = wc_grid.topology
        topo.lines_df["b"] = 2 * np.pi * 50 * 0.25e-6 * topo.lines_df.length
        pm = _to_pm(wc_grid, s_base=s_base)
        for b in _pm_branches(pm, transformer=False):
            line = topo.lines_df.loc[b["name"]]
            v_nom = topo.buses_df.at[line.bus0, "v_nom"]
            siemens = (b["b_fr"] + b["b_to"]) * s_base / v_nom**2
            assert siemens == pytest.approx(line.b, rel=RTOL_RESCALE), b["name"]

    @pytest.mark.parametrize("s_base", [1, 10])
    def test_single_transformer_on_common_base(self, wc_grid, s_base):
        """
        eDisGo stores transformer impedances in p.u. of their own rating. On the
        common base S_base they become ``r_pu·S_base/s_nom``, and the thermal
        limit ``rate_a`` is ``s_nom/S_base``.
        """
        pm = _to_pm(wc_grid, s_base=s_base)
        trafos = wc_grid.topology.transformers_df
        groups = trafos.groupby(["bus0", "bus1"])
        single = {k: g.iloc[0] for k, g in groups if len(g) == 1}
        assert single
        for b in _pm_branches(pm, transformer=True):
            key = (_pm_bus_name(pm, b["f_bus"]), _pm_bus_name(pm, b["t_bus"]))
            if key not in single:
                continue
            trafo = single.pop(key)
            assert b["br_r"] == pytest.approx(
                trafo.r_pu * s_base / trafo.s_nom, rel=RTOL_RESCALE
            )
            assert b["br_x"] == pytest.approx(
                trafo.x_pu * s_base / trafo.s_nom, rel=RTOL_RESCALE
            )
            assert b["rate_a"] == pytest.approx(trafo.s_nom / s_base, rel=RTOL_RESCALE)
        assert not single, f"transformers missing in pm: {list(single)}"

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: aggregate_parallel_transformers (via "
        "tools.calculate_impedance_for_parallel_components) treats the p.u. values "
        "of the transformers like Ohm: n identical transformers get r/n and n·s_nom, "
        "i.e. r_pu = r/(n²·s_nom) instead of r/(n·s_nom); for different ratings the "
        "equality check without abs() takes 40+10 kVA as identical (s_nom 80 kVA "
        "instead of 50 kVA).",
    )
    def test_parallel_transformers_are_equivalent_impedance(self, wc_grid):
        """
        Parallel transformers between the same buses are merged into one branch
        with the admittance sum on the common base, Z = 1/Σ(s_nom_i/z_pu_i), and
        - for equal p.u. impedances, i.e. load sharing in proportion to the
        rating - a thermal limit of Σ s_nom_i. Test network 1 has four identical
        1 MVA transformers and a 40 kVA plus a 10 kVA transformer in parallel.
        """
        pm = _to_pm(wc_grid)
        trafos = wc_grid.topology.transformers_df
        parallel = {k: g for k, g in trafos.groupby(["bus0", "bus1"]) if len(g) > 1}
        assert len(parallel) == 2
        errors = []
        for b in _pm_branches(pm, transformer=True):
            key = (_pm_bus_name(pm, b["f_bus"]), _pm_bus_name(pm, b["t_bus"]))
            if key not in parallel:
                continue
            g = parallel[key]
            z = 1 / sum(g.s_nom / (g.r_pu + 1j * g.x_pu))
            expected = {"br_r": z.real, "br_x": z.imag, "rate_a": g.s_nom.sum()}
            for k, v in expected.items():
                if not math.isclose(b[k], v, rel_tol=RTOL_RESCALE):
                    errors.append(f"{key[1]} {k}: {b[k]:.6g} != {v:.6g}")
        assert not errors, errors

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: aggregate_parallel_transformers makes the impedance of n "
        "identical parallel transformers n times too small (p.u. values treated "
        "like Ohm), so the merged network has a different power flow.",
    )
    def test_parallel_transformer_aggregation_keeps_power_flow(self, wc_grid):
        """
        Merging parallel transformers is a purely electrical equivalent: the power
        flow on the merged PyPSA network gives the same slack power and bus
        voltages as on the original. Tolerance: both power flows stop at a
        mismatch of 1e-6 p.u., far below the difference a wrong impedance causes
        (~1e-3 p.u. here).
        """
        original = wc_grid.to_pypsa()
        merged = wc_grid.to_pypsa()
        powermodels_io.aggregate_parallel_transformers(merged)
        assert len(merged.transformers) < len(original.transformers)
        for network in (original, merged):
            assert network.pf()["converged"]["0"].all()
        slack = "Generator_slack"
        np.testing.assert_allclose(
            merged.generators_t.p[slack], original.generators_t.p[slack], atol=1e-6
        )
        pd.testing.assert_frame_equal(
            merged.buses_t.v_mag_pu, original.buses_t.v_mag_pu, atol=1e-6
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: aggregate_parallel_transformers groups by name[:-2] but "
        "aggregates by (bus0, bus1); after reinforcement 'LVStation_1_transformer_1' "
        "and 'LVStation_1_transformer_reinforced_2' get different prefixes, so the "
        "aggregated transformer is exported twice.",
    )
    def test_one_branch_per_transformer_bus_pair_after_reinforcement(
        self, reinforced_grid
    ):
        """
        After merging parallel transformers there is exactly one transformer
        branch per (bus0, bus1) pair, independent of how the transformers are
        named (reinforcement adds transformers named ``*_reinforced_<n>``).
        """
        pm = _to_pm(reinforced_grid)
        pairs = [
            (_pm_bus_name(pm, b["f_bus"]), _pm_bus_name(pm, b["t_bus"]))
            for b in _pm_branches(pm, transformer=True)
        ]
        trafos = reinforced_grid.topology.transformers_df
        expected = set(zip(trafos.bus0, trafos.bus1))
        assert sorted(pairs) == sorted(expected)

    @pytest.mark.parametrize("k", [0.1, 10.0])
    def test_s_base_rescales_powers_and_impedances(self, flex_grid, k):
        """
        Changing the power base is a change of units only: with S_base = k all
        powers (static values and time series) and power limits are divided by
        k, impedances multiplied by k, energy capacities divided by k; voltages,
        power factors, efficiencies and COPs stay the same.
        """
        edisgo, flex = flex_grid
        pm1 = _to_pm(edisgo, s_base=1, flex=flex)
        pmk = _to_pm(edisgo, s_base=k, flex=flex)
        power_fields = {
            "gen": ("pg", "qg", "pmax", "pmin", "qmax", "qmin", "mbase"),
            "gen_nd": ("pg", "qg", "pmax", "pmin", "qmax", "qmin"),
            "load": ("pd", "qd"),
            "storage": (
                "pmax",
                "pmin",
                "qmax",
                "qmin",
                "charge_rating",
                "discharge_rating",
                "energy_rating",
                "energy",
            ),
            "electromobility": ("p_max", "q_min", "q_max", "e_min", "e_max"),
            "heatpumps": ("pd", "p_max", "q_min", "q_max"),
            "heat_storage": ("capacity",),
            "dsm": ("p_min", "p_max", "q_min", "q_max", "e_min", "e_max"),
            "branch": ("rate_a",),
        }
        for comp, fields in power_fields.items():
            assert pm1[comp].keys() == pmk[comp].keys(), comp
            for key, v1 in pm1[comp].items():
                for f in fields:
                    assert pmk[comp][key][f] * k == pytest.approx(
                        v1[f], rel=RTOL_RESCALE, abs=1e-12
                    ), (comp, key, f)
        for key, v1 in pm1["branch"].items():
            for f in ("br_r", "br_x"):
                assert pmk["branch"][key][f] / k == pytest.approx(
                    v1[f], rel=RTOL_RESCALE
                ), (key, f)
        for kind, series in pm1["time_series"].items():
            if kind == "num_steps":
                continue
            for key, values in series.items():
                for f, v1 in values.items():
                    vk = np.array(pmk["time_series"][kind][key][f])
                    scale = 1 if f == "cop" else k
                    # heat demand is rounded to 6 decimals of the p.u. value,
                    # i.e. to 0.5e-6·S_base MW
                    atol = 0.5e-6 * max(k, 1) if kind == "heatpumps" else 1e-12
                    np.testing.assert_allclose(
                        vk * scale, v1, rtol=RTOL_RESCALE, atol=atol
                    )

    @pytest.mark.parametrize(
        "comp, field",
        [
            pytest.param(
                "storage",
                "soc_initial",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Bug: soc_initial/soc_end of batteries are in MWh "
                    "(SoC·p_nom·max_hours) and not divided by s_base, while "
                    "energy_rating is.",
                ),
            ),
            pytest.param(
                "heat_storage",
                "soc_initial",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Bug: soc_initial/soc_end of thermal storage are in MWh "
                    "and not divided by s_base, while capacity is.",
                ),
            ),
            pytest.param(
                "electromobility",
                "energy",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Bug: the initial energy of charging points is not "
                    "divided by s_base, while e_min/e_max are.",
                ),
            ),
        ],
    )
    def test_s_base_rescales_storage_states(self, flex_grid, comp, field):
        """
        States of charge are energies like the capacities they are bounded by, so
        they must be divided by S_base, too; otherwise ``soc ≤ capacity`` would
        depend on the chosen power base.
        """
        edisgo, flex = flex_grid
        k = 10.0
        pm1 = _to_pm(edisgo, s_base=1, flex=flex)
        pmk = _to_pm(edisgo, s_base=k, flex=flex)
        assert pm1[comp]
        for key, v1 in pm1[comp].items():
            assert v1[field] > 0, (comp, key, field)
            assert pmk[comp][key][field] * k == pytest.approx(
                v1[field], rel=RTOL_RESCALE
            ), (comp, key, field)

    def test_storage_states_within_capacity(self, flex_grid):
        """
        For S_base = 1 MVA every initial and final state of charge lies between
        zero and the energy capacity of its storage (same unit).
        """
        edisgo, flex = flex_grid
        pm = _to_pm(edisgo, s_base=1, flex=flex)
        for comp, cap in (("storage", "energy_rating"), ("heat_storage", "capacity")):
            assert pm[comp]
            for v in pm[comp].values():
                for f in ("soc_initial", "soc_end"):
                    assert 0 <= v[f] <= v[cap] * (1 + RTOL_RESCALE), (comp, f)
        for v in pm["electromobility"].values():
            assert v["e_min"] <= v["energy"] <= v["e_max"]
            assert v["p_min"] == 0 <= v["p_max"]
        for v in pm["dsm"].values():
            assert v["p_min"] <= 0 <= v["p_max"]
            assert v["e_min"] <= 0 <= v["e_max"]

    def test_time_elapsed_matches_frequency(self):
        """
        ``time_elapsed`` is the constant step length in hours for any equidistant
        index; non-equidistant or decreasing indices are rejected.
        """

        @given(
            minutes=st.sampled_from([5, 15, 30, 60, 120, 1440]),
            n=st.integers(2, 50),
            start=st.datetimes(
                min_value=pd.Timestamp("2000-01-01").to_pydatetime(),
                max_value=pd.Timestamp("2050-01-01").to_pydatetime(),
            ),
            jitter=st.integers(1, 4),
        )
        @settings(deadline=None, max_examples=examples(30))
        def check(minutes, n, start, jitter):
            index = pd.date_range(start, periods=n, freq=f"{minutes}min")
            hours = powermodels_io._get_time_elapsed_in_hours(index)
            assert hours == pytest.approx(minutes / 60)
            with pytest.raises(ValueError):
                powermodels_io._get_time_elapsed_in_hours(index[::-1])
            if n > 3:
                irregular = index.delete(min(jitter, n - 2))
                with pytest.raises(ValueError):
                    powermodels_io._get_time_elapsed_in_hours(irregular)

        check()


# ======================================================================================
# PowerModels export: component mapping and time series
# ======================================================================================


class TestPowerModelsMapping:
    KINDS = (
        "gen",
        "gen_nd",
        "gen_slack",
        "load",
        "storage",
        "electromobility",
        "heatpumps",
        "heat_storage",
        "dsm",
        "branch",
        "bus",
    )
    BUS_KEYS = {
        "gen": "gen_bus",
        "gen_nd": "gen_bus",
        "gen_slack": "gen_bus",
        "load": "load_bus",
        "storage": "storage_bus",
        "electromobility": "cp_bus",
        "heatpumps": "hp_bus",
        "heat_storage": "storage_bus",
        "dsm": "dsm_bus",
    }

    def test_keys_indices_and_names_are_bijective(self, flex_grid):
        """
        Per component kind, the keys are "1".."n", equal to the stored index, and
        the names are unique, so key and name identify each other. Time series
        exist for exactly the keys of the static data.
        """
        edisgo, flex = flex_grid
        pm = _to_pm(edisgo, flex=flex)
        for kind in self.KINDS:
            entries = pm[kind]
            n = len(entries)
            assert set(entries) == {str(i) for i in range(1, n + 1)}, kind
            assert all(v["index"] == int(k) for k, v in entries.items()), kind
            names = [v["name"] for v in entries.values()]
            assert len(set(names)) == n, kind
        for kind in ("gen", "gen_nd", "gen_slack", "load", "electromobility", "dsm"):
            assert set(pm["time_series"][kind]) == set(pm[kind]), kind
        assert set(pm["time_series"]["heatpumps"]) == set(pm["heatpumps"])
        assert pm["time_series"]["num_steps"] == N_STEPS

    def test_components_attached_to_their_bus(self, flex_grid):
        """
        Every component refers to an existing PowerModels bus, and that bus is
        the eDisGo bus of the component (flexible batteries: their virtual bus
        ``<bus>_bss``). Exactly one bus is the reference bus, the MV station.
        """
        edisgo, flex = flex_grid
        pm = _to_pm(edisgo, flex=flex)
        topo = edisgo.topology
        comp_bus = pd.concat(
            [topo.loads_df.bus, topo.generators_df.bus, topo.storage_units_df.bus]
        )
        for kind, bus_key in self.BUS_KEYS.items():
            for v in pm[kind].values():
                bus = pm["bus"][str(v[bus_key])]
                if kind == "gen_slack":
                    expected = topo.mv_grid.station.index[0]
                elif kind == "storage":
                    expected = comp_bus[v["name"]] + "_bss"
                else:
                    expected = comp_bus[v["name"]]
                assert bus["name"] == expected, (kind, v["name"])
        for b in pm["branch"].values():
            assert str(b["f_bus"]) in pm["bus"] and str(b["t_bus"]) in pm["bus"]
        refs = [v["name"] for v in pm["bus"].values() if v["bus_type"] == 3]
        assert refs == [topo.mv_grid.station.index[0]]

    def test_components_are_partitioned(self, flex_grid):
        """
        Every generator, load and storage unit appears in exactly one place:
        solar/wind in ``gen_nd``, other generators and inflexible storage units
        in ``gen``, inflexible storage units also as charging load in ``load``,
        flexible ones in ``storage``, flexible heat pumps and charging points only
        in their own dicts, all other loads (including DSM loads) in ``load``.
        """
        edisgo, flex = flex_grid
        pm = _to_pm(edisgo, flex=flex)
        topo = edisgo.topology
        names = {k: [v["name"] for v in pm[k].values()] for k in self.KINDS}
        fluct = topo.generators_df.type.isin(["solar", "wind"])
        flex_storage = set(flex["flexible_storage_units"])
        inflex_storage = set(topo.storage_units_df.index) - flex_storage
        flex_loads = set(flex["flexible_hps"]) | set(flex["flexible_cps"])
        assert set(names["gen_nd"]) == set(topo.generators_df.index[fluct])
        assert set(names["gen"]) == set(topo.generators_df.index[~fluct]) | (
            inflex_storage
        )
        assert set(names["storage"]) == flex_storage
        assert set(names["load"]) == (set(topo.loads_df.index) - flex_loads) | (
            inflex_storage
        )
        assert set(names["heatpumps"]) == set(names["heat_storage"])
        assert set(names["heatpumps"]) == set(flex["flexible_hps"])
        assert set(names["electromobility"]) == set(flex["flexible_cps"])
        assert set(names["dsm"]) == set(flex["flexible_loads"])
        assert set(names["dsm"]) <= set(names["load"])

    @pytest.mark.parametrize("s_base", [1, 4])
    def test_time_series_equal_edisgo_time_series(self, flex_grid, s_base):
        """
        For every component, the exported time series times S_base equals its
        eDisGo time series (not just a sample): generator/load P and Q, charging
        point bands, heat demand and COP, DSM bands; inflexible storage units are
        split into discharging (``gen``) and charging (``load``) without loss.
        Heat demand is rounded to 6 decimals in the export.
        """
        edisgo, flex = flex_grid
        pm = _to_pm(edisgo, s_base=s_base, flex=flex)
        ts = edisgo.timeseries
        tsp = pm["time_series"]

        def check(actual, expected, atol=1e-12):
            np.testing.assert_allclose(
                np.array(actual) * s_base, expected, rtol=RTOL_RESCALE, atol=atol
            )

        for kind in ("gen", "gen_nd", "gen_slack"):
            for key, v in pm[kind].items():
                name = v["name"]
                if kind == "gen_slack":
                    expected_p = expected_q = np.zeros(N_STEPS)
                elif name in ts.generators_active_power:
                    expected_p = ts.generators_active_power[name]
                    expected_q = ts.generators_reactive_power[name]
                else:
                    expected_p = ts.storage_units_active_power[name].clip(lower=0)
                    expected_q = ts.storage_units_reactive_power[name].clip(upper=0)
                check(tsp[kind][key]["pg"], expected_p)
                check(tsp[kind][key]["qg"], expected_q)
        for key, v in pm["load"].items():
            name = v["name"]
            if name in ts.loads_active_power:
                expected_p = ts.loads_active_power[name]
                expected_q = ts.loads_reactive_power[name]
            else:
                expected_p = -ts.storage_units_active_power[name].clip(upper=0)
                expected_q = -ts.storage_units_reactive_power[name].clip(lower=0)
            check(tsp["load"][key]["pd"], expected_p)
            check(tsp["load"][key]["qd"], expected_q)
        bands = edisgo.electromobility.flexibility_bands
        for key, v in pm["electromobility"].items():
            for f, band in (
                ("p_max", "upper_power"),
                ("e_min", "lower_energy"),
                ("e_max", "upper_energy"),
            ):
                check(tsp["electromobility"][key][f], bands[band][v["name"]])
        for key, v in pm["heatpumps"].items():
            check(
                tsp["heatpumps"][key]["pd"],
                edisgo.heat_pump.heat_demand_df[v["name"]],
                atol=0.5e-6 * s_base,
            )
            np.testing.assert_allclose(
                tsp["heatpumps"][key]["cop"], edisgo.heat_pump.cop_df[v["name"]]
            )
        for key, v in pm["dsm"].items():
            for f in ("p_min", "p_max", "e_min", "e_max"):
                check(tsp["dsm"][key][f], getattr(edisgo.dsm, f)[v["name"]])

    def test_inflexible_storage_split_is_lossless(self, wc_grid):
        """
        An inflexible storage unit is exported as generator (discharging) and as
        load (charging); per time step, generation minus load equals its
        eDisGo active and reactive power.
        """
        pm = _to_pm(wc_grid)
        ts = wc_grid.timeseries
        gen = {v["name"]: k for k, v in pm["gen"].items()}
        load = {v["name"]: k for k, v in pm["load"].items()}
        storages = wc_grid.topology.storage_units_df.index
        assert len(storages) > 0
        for name in storages:
            g = pm["time_series"]["gen"][gen[name]]
            ld = pm["time_series"]["load"][load[name]]
            np.testing.assert_allclose(
                np.subtract(g["pg"], ld["pd"]), ts.storage_units_active_power[name]
            )
            np.testing.assert_allclose(
                np.subtract(g["qg"], ld["qd"]), ts.storage_units_reactive_power[name]
            )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: to_powermodels overwrites overlying_grid.storage_units_soc and "
        "heat_storage_units_soc with energies in MWh (SoC·capacity), so a second "
        "call on the same object multiplies by the capacity again.",
    )
    def test_to_powermodels_is_repeatable(self, flex_grid):
        """
        Exporting the same EDisGo object twice gives the same PowerModels dict
        and leaves the overlying grid input unchanged (no hidden side effects
        besides sorting the buses of lines).
        """
        edisgo, flex = flex_grid
        soc_before = edisgo.overlying_grid.storage_units_soc.copy()
        pm1, _ = powermodels_io.to_powermodels(edisgo, **flex)
        pm2, _ = powermodels_io.to_powermodels(edisgo, **flex)
        for comp in ("storage", "heat_storage"):
            for key in pm1[comp]:
                for f in ("soc_initial", "soc_end"):
                    assert pm1[comp][key][f] == pm2[comp][key][f], (comp, f)
        pd.testing.assert_series_equal(
            edisgo.overlying_grid.storage_units_soc, soc_before
        )


# ======================================================================================
# PowerModels import (round trip without Julia)
# ======================================================================================


class TestFromPowerModels:
    @given(s_base=st.sampled_from([1, 2.5, 10]), seed=st.integers(0, 2**16))
    @FIXTURE_SETTINGS
    def test_round_trip_keeps_flexibility_dispatch(self, flex_grid, s_base, seed):
        """
        Feeding the current dispatch of all flexibilities back as OPF solution
        leaves the active power time series unchanged for any S_base and any
        order of the result time steps: names are mapped correctly, powers are
        rescaled with S_base and the storage sign (virtual branch flow from grid
        to storage = charging) is converted to the generator convention.
        """
        edisgo, flex = copy.deepcopy(flex_grid)
        before = {
            attr: getattr(edisgo.timeseries, attr).copy()
            for attr in (
                "loads_active_power",
                "generators_active_power",
                "storage_units_active_power",
            )
        }
        pm, hv = powermodels_io.to_powermodels(
            copy.deepcopy(edisgo), s_base=s_base, **flex
        )
        assert set(pm["flexibilities"]) == {"cp", "hp", "dsm", "storage"}
        storage_p = edisgo.timeseries.storage_units_active_power
        assert (storage_p.abs() > 0).all().all()
        results = _synthetic_pm_results(pm, edisgo, s_base, order_seed=seed)
        powermodels_io.from_powermodels(edisgo, results, hv, s_base=s_base)
        for attr, df in before.items():
            pd.testing.assert_frame_equal(
                getattr(edisgo.timeseries, attr)[df.columns],
                df,
                rtol=RTOL_RESCALE,
                atol=1e-12,
                check_freq=False,
                obj=attr,
            )
        battery = edisgo.opf_results.battery_storage_t.p
        assert list(battery.columns) == list(flex["flexible_storage_units"])
        pd.testing.assert_index_equal(battery.index, edisgo.timeseries.timeindex)

    @given(shift=st.floats(-0.1, 0.1, allow_nan=False), s_base=st.sampled_from([1, 5]))
    @FIXTURE_SETTINGS
    def test_dsm_result_is_added_to_load(self, flex_grid, shift, s_base):
        """
        The DSM variable is a shift relative to the reference load: an OPF result
        ``pdsm = δ/S_base`` in every time step changes the DSM load by exactly δ
        MW and no other load.
        """
        edisgo, flex = copy.deepcopy(flex_grid)
        before = edisgo.timeseries.loads_active_power.copy()
        pm, hv = powermodels_io.to_powermodels(
            copy.deepcopy(edisgo), s_base=s_base, **flex
        )
        results = _synthetic_pm_results(pm, edisgo, s_base)
        for step in results["nw"].values():
            for v in step["dsm"].values():
                v["pdsm"] = shift / s_base
        powermodels_io.from_powermodels(edisgo, results, hv, s_base=s_base)
        diff = edisgo.timeseries.loads_active_power[before.columns] - before
        dsm = list(flex["flexible_loads"])
        np.testing.assert_allclose(diff[dsm], shift, atol=1e-12)
        np.testing.assert_allclose(diff.drop(columns=dsm), 0.0, atol=1e-12)

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="Bug: from_powermodels reads pm_results['nw'] instead of the parsed "
        "dict when pm_results is a path to a JSON file.",
    )
    def test_json_path_equals_dict(self, flex_grid, tmp_path):
        """
        Passing the OPF results as path to a JSON file (the documented second
        input form, a JSON-encoded string) gives the same result as the dict.
        """
        edisgo, flex = flex_grid
        pm, hv = powermodels_io.to_powermodels(copy.deepcopy(edisgo), **flex)
        results = _synthetic_pm_results(pm, edisgo, 1)
        via_dict = copy.deepcopy(edisgo)
        powermodels_io.from_powermodels(via_dict, copy.deepcopy(results), hv)
        path = tmp_path / "results.json"
        with open(path, "w") as f:
            # numpy scalars (indices, bus numbers) as plain Python numbers
            json.dump(json.dumps(results, default=lambda o: o.item()), f)
        powermodels_io.from_powermodels(edisgo, str(path), hv)
        pd.testing.assert_frame_equal(
            edisgo.timeseries.loads_active_power,
            via_dict.timeseries.loads_active_power,
        )


# ======================================================================================
# PyPSA export and power flow results
# ======================================================================================


def _lv_component_owner(edisgo):
    """Map every load, generator and storage unit to the LV grid it belongs to."""
    owner = {}
    for lv in edisgo.topology.lv_grids:
        for df in (lv.loads_df, lv.generators_df, lv.storage_units_df):
            owner.update({name: lv for name in df.index})
    return owner


class TestPyPSAExport:
    def test_full_export_reproduces_topology(self, wc_grid):
        """
        ``to_pypsa`` without mode exports the topology one-to-one: same buses
        with v_nom, same line parameters in Ohm/Siemens/MVA, transformers with
        r and x in p.u. of their own rating (PyPSA convention, i.e. r_pu = r/s_nom
        after ``calculate_dependent_values``), same ratings of loads, generators
        and storage units, and exactly one slack generator at the MV station.
        """
        topo = wc_grid.topology
        pn = wc_grid.to_pypsa()
        pd.testing.assert_series_equal(
            pn.buses.v_nom.sort_index(),
            topo.buses_df.v_nom.sort_index(),
            check_names=False,
        )
        lines = pn.lines.loc[topo.lines_df.index]
        for col in ("bus0", "bus1", "r", "x", "b", "s_nom", "length", "num_parallel"):
            pd.testing.assert_series_equal(
                lines[col], topo.lines_df[col], check_names=False, check_dtype=False
            )
        trafos = pn.transformers.loc[topo.transformers_df.index]
        for col, ecol in (("r", "r_pu"), ("x", "x_pu"), ("s_nom", "s_nom")):
            np.testing.assert_allclose(trafos[col], topo.transformers_df[ecol])
        pypsa.pf.calculate_dependent_values(pn)
        np.testing.assert_allclose(
            pn.transformers.loc[topo.transformers_df.index, "r_pu"],
            topo.transformers_df.r_pu / topo.transformers_df.s_nom,
        )
        np.testing.assert_allclose(
            pn.loads.loc[topo.loads_df.index, "p_set"], topo.loads_df.p_set
        )
        gens = pn.generators.drop("Generator_slack")
        np.testing.assert_allclose(
            gens.loc[topo.generators_df.index, "p_nom"], topo.generators_df.p_nom
        )
        assert len(gens) == len(topo.generators_df)
        sto = pn.storage_units.loc[topo.storage_units_df.index]
        np.testing.assert_allclose(sto.p_nom, topo.storage_units_df.p_nom)
        np.testing.assert_allclose(sto.max_hours, topo.storage_units_df.max_hours)
        slack = pn.generators[pn.generators.control == "Slack"]
        assert list(slack.bus) == [topo.mv_grid.station.index[0]]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": None},
            {"mode": "mv"},
            {"mode": "mvlv"},
            {
                "mode": "mv",
                "aggregate_loads": "sectoral",
                "aggregate_generators": "type",
            },
            {
                "mode": "mvlv",
                "aggregate_loads": "all",
                "aggregate_generators": "all",
                "aggregate_storages": "all",
            },
        ],
        ids=lambda kw: "-".join(str(v) for v in kw.values()),
    )
    def test_components_at_exported_buses_of_right_level(self, wc_grid, kwargs):
        """
        Every exported component and branch is connected to an exported bus. In
        mode "mv" the LV components sit at the primary (MV) side of their
        station, in mode "mvlv" at its secondary (LV) side.
        """
        pn = wc_grid.to_pypsa(**kwargs)
        buses = pn.buses.index
        for comp in ("loads", "generators", "storage_units"):
            df = getattr(pn, comp)
            assert df.bus.isin(buses).all(), comp
        for comp in ("lines", "transformers"):
            df = getattr(pn, comp)
            assert df.bus0.isin(buses).all() and df.bus1.isin(buses).all(), comp
        mode = kwargs["mode"]
        if mode is None:
            return
        for lv in wc_grid.topology.lv_grids:
            station = lv.transformers_df.iloc[0]
            expected = station.bus0 if mode == "mv" else station.bus1
            names = [
                n
                for df in (lv.loads_df, lv.generators_df, lv.storage_units_df)
                for n in df.index
            ]
            for comp in ("loads", "generators", "storage_units"):
                df = getattr(pn, comp)
                own = df.index.isin(names) | df.index.str.startswith(f"{lv!r}_")
                assert (df.bus[own] == expected).all(), (repr(lv), comp)
        v_nom = pn.buses.v_nom
        lv_level = v_nom[v_nom < 1]
        if mode == "mv":
            assert lv_level.empty
        else:
            assert set(lv_level.index) == {
                lv.transformers_df.bus1.iloc[0] for lv in wc_grid.topology.lv_grids
            }

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: _append_lv_components keeps only bus and control of LV "
        "storage units (with and without aggregation), so p_nom and max_hours are "
        "lost and PyPSA sets p_nom = 0, contrary to its docstring ('The sum of "
        "p_nom/p_set of all cumulated components is calculated').",
    )
    @pytest.mark.parametrize("aggregate_storages", [None, "all"])
    def test_lv_storage_rating_conserved(self, wc_grid, aggregate_storages):
        """
        Exporting only the MV level must not change the installed storage power
        and energy: Σ p_nom and Σ p_nom·max_hours of all exported storage units
        equal those of the topology.
        """
        lv_bus = next(iter(wc_grid.topology.lv_grids)).buses_df.index[0]
        ti = wc_grid.timeseries.timeindex
        wc_grid.add_component(
            comp_type="storage_unit",
            bus=lv_bus,
            p_nom=0.01,
            max_hours=2.0,
            ts_active_power=pd.Series(0.005, index=ti),
            ts_reactive_power=pd.Series(0.0, index=ti),
        )
        sto = wc_grid.topology.storage_units_df
        pn = wc_grid.to_pypsa(mode="mv", aggregate_storages=aggregate_storages)
        assert pn.storage_units.p_nom.sum() == pytest.approx(sto.p_nom.sum())
        assert (pn.storage_units.p_nom * pn.storage_units.max_hours).sum() == (
            pytest.approx((sto.p_nom * sto.max_hours).sum())
        )

    @given(
        n=st.integers(1, 12),
        p=st.lists(st.floats(0, 1, allow_nan=False), min_size=12, max_size=12),
        types=st.lists(
            st.sampled_from(["solar", "wind", "gas", "biomass"]),
            min_size=12,
            max_size=12,
        ),
        sectors=st.lists(
            st.sampled_from(["residential", "cts", "industrial"]),
            min_size=12,
            max_size=12,
        ),
        mode_l=st.sampled_from([None, "sectoral", "all"]),
        mode_g=st.sampled_from([None, "type", "curtailable", "all"]),
    )
    @settings(deadline=None, max_examples=examples(40))
    def test_append_lv_components_conserves_and_partitions(
        self, n, p, types, sectors, mode_l, mode_g
    ):
        """
        Aggregating the components of an LV grid conserves Σ p_set (loads) and
        Σ p_nom (generators), the aggregated elements form a partition of the
        input (each component in exactly one aggregate) and an aggregate is
        fluctuating exactly if all its members are solar or wind.
        """
        index = [f"c{i}" for i in range(n)]
        loads = pd.DataFrame(
            {"bus": "b", "p_set": p[:n], "sector": sectors[:n]}, index=index
        )
        gens = pd.DataFrame(
            {"bus": "b", "control": "PQ", "p_nom": p[:n], "type": types[:n]},
            index=index,
        )
        for comp, df, mode, col in (
            ("Load", loads, mode_l, "p_set"),
            ("Generator", gens, mode_g, "p_nom"),
        ):
            out = {comp: pd.DataFrame(dtype=float)}
            kw = {"aggregate_loads": mode_l, "aggregate_generators": mode_g}
            aggr = pypsa_io._append_lv_components(comp, df, out, "LV", **kw)
            assert out[comp][col].sum() == pytest.approx(df[col].sum())
            assert (out[comp].bus == "b").all()
            if mode is None:
                assert aggr == {}
                assert list(out[comp].index) == index
                continue
            members = [m for v in aggr.values() for m in v]
            assert sorted(members) == sorted(index)
            assert set(aggr) == set(out[comp].index)
            for name, m in aggr.items():
                assert out[comp].at[name, col] == pytest.approx(df.loc[m, col].sum())
                if comp == "Generator":
                    fluct = df.loc[m, "type"].isin(["solar", "wind"])
                    flag = out[comp].at[name, "fluctuating"]
                    if fluct.all():
                        assert flag is True or flag == True  # noqa: E712
                    elif not fluct.any():
                        assert flag is False or flag == False  # noqa: E712
                    else:
                        assert flag == "Mixed"

    def test_aggregated_time_series_are_sums_of_members(self, wc_grid):
        """
        The time series of an aggregated LV component is, per time step, the sum
        of the time series of its members; components that are not aggregated
        keep their own time series.
        """
        ts = wc_grid.timeseries
        lv = next(iter(wc_grid.topology.lv_grids))
        loads = lv.loads_df.copy()
        out = {"Load": pd.DataFrame(dtype=float)}
        aggr = pypsa_io._append_lv_components(
            "Load", loads, out, repr(lv), aggregate_loads="sectoral"
        )
        mv_loads = wc_grid.topology.mv_grid.loads_df.index
        elements = mv_loads.append(out["Load"].index)
        p, q = pypsa_io._get_timeseries_with_aggregated_elements(
            wc_grid, ts.timeindex, ["loads"], elements, aggr
        )
        assert set(p.columns) == set(elements)
        for name, members in aggr.items():
            np.testing.assert_allclose(
                p[name], ts.loads_active_power[members].sum(axis=1)
            )
            np.testing.assert_allclose(
                q[name], ts.loads_reactive_power[members].sum(axis=1)
            )
        pd.testing.assert_frame_equal(
            p[mv_loads], ts.loads_active_power[mv_loads], check_freq=False
        )

    @given(
        offset=st.floats(-0.03, 0.03, allow_nan=False),
        deviation=st.floats(0, 0.02, allow_nan=False),
    )
    @FIXTURE_SETTINGS
    def test_slack_voltage_follows_set_point(self, wc_grid_factory, offset, deviation):
        """
        The slack voltage is 1 + HV/MV offset + control deviation in feed-in
        time steps and 1 + offset − deviation in load time steps; PyPSA keeps it
        exactly, all other set points are 1 p.u. (ignored by PQ buses).
        """
        edisgo = wc_grid_factory()
        cfg = edisgo.config["grid_expansion_allowed_voltage_deviations"]
        cfg["hv_mv_trafo_offset"] = offset
        cfg["hv_mv_trafo_control_deviation"] = deviation
        pn = run_pypsa_power_flow(edisgo)
        slack = pn.generators.at["Generator_slack", "bus"]
        case = edisgo.timeseries.timesteps_load_feedin_case
        sign = np.where(case.loc[pn.snapshots] == "feedin_case", 1.0, -1.0)
        expected = 1 + offset + sign * deviation
        np.testing.assert_allclose(pn.buses_t.v_mag_pu_set[slack], expected)
        np.testing.assert_allclose(pn.buses_t.v_mag_pu[slack], expected, atol=1e-10)
        others = pn.buses_t.v_mag_pu_set.drop(columns=slack)
        assert (others == 1).all().all()

    def test_process_pfa_results_consistent_with_branch_flows(self, wc_grid):
        """
        ``pfa_p``/``pfa_q`` are |P| and |Q| of one and the same branch end, the
        one with the larger apparent power; ``i_res`` is the current at bus0,
        |S0|/(√3·U_nom·v0) in kA; ``grid_losses.p`` equals the signed sum of the
        branch losses Σ(p0 + p1) (checked without abs(), losses ≥ 0).
        """
        pn = run_pypsa_power_flow(wc_grid)
        timesteps = wc_grid.timeseries.timeindex
        pypsa_io.process_pfa_results(wc_grid, pn, timesteps)
        res = wc_grid.results
        losses = 0
        for comp in ("lines", "transformers"):
            static = getattr(pn, comp)
            dyn = getattr(pn, f"{comp}_t")
            names = static.index
            s0 = np.hypot(dyn.p0[names], dyn.q0[names])
            s1 = np.hypot(dyn.p1[names], dyn.q1[names])
            p, q = res.pfa_p[names], res.pfa_q[names]
            np.testing.assert_allclose(np.hypot(p, q), np.maximum(s0, s1), rtol=1e-12)
            end0 = np.isclose(p, dyn.p0[names].abs()) & np.isclose(
                q, dyn.q0[names].abs()
            )
            end1 = np.isclose(p, dyn.p1[names].abs()) & np.isclose(
                q, dyn.q1[names].abs()
            )
            assert (end0 | end1).all().all(), comp
            v_nom = pn.buses.v_nom[static.bus0].to_numpy()
            v0 = pn.buses_t.v_mag_pu[static.bus0].to_numpy()
            current = s0.to_numpy() / (math.sqrt(3) * v_nom * v0)
            np.testing.assert_allclose(res.i_res[names], current, rtol=1e-12)
            losses = losses + (dyn.p0[names] + dyn.p1[names]).sum(axis=1)
        assert (losses >= -POWER_ATOL).all()
        np.testing.assert_allclose(res.grid_losses.p, losses, atol=POWER_ATOL)


# ======================================================================================
# ding0 import
# ======================================================================================

DING0_NETWORKS = [
    pytest.param("ding0_test_network_path", True, id="network_1"),
    pytest.param("ding0_test_network_2_path", True, id="network_2"),
    pytest.param("ding0_test_network_3_path", False, id="network_3"),
]


@pytest.mark.parametrize("path_attr, legacy", DING0_NETWORKS)
def test_ding0_import_conserves_csv_data(path_attr, legacy):
    """
    The ding0 import keeps every component of the CSV files: same number of
    loads, generators (without slack) and storage units, same Σ p_set (legacy
    grids: ``peak_load``) and Σ p_nom; every component and branch refers to an
    existing bus, every LV bus belongs to exactly one LV grid, the HV/MV
    transformer feeds an MV bus and transformer impedances are kept as p.u.
    values (positive).
    """
    path = getattr(pytest, path_attr)
    edisgo = EDisGo(ding0_grid=path, legacy_ding0_grids=legacy)
    topo = edisgo.topology
    csv = {
        name: (
            pd.read_csv(f"{path}/{name}.csv", index_col=0)
            if os.path.isfile(f"{path}/{name}.csv")
            else pd.DataFrame(columns=["p_nom"])
        )
        for name in ("loads", "generators", "storage_units", "transformers", "lines")
    }
    gens = csv["generators"][csv["generators"].control != "Slack"]
    load_col = "peak_load" if legacy else "p_set"
    assert len(topo.loads_df) == len(csv["loads"])
    assert topo.loads_df.p_set.sum() == pytest.approx(csv["loads"][load_col].sum())
    assert len(topo.generators_df) == len(gens)
    assert topo.generators_df.p_nom.sum() == pytest.approx(gens.p_nom.sum())
    assert len(topo.storage_units_df) == len(csv["storage_units"])
    assert topo.storage_units_df.p_nom.sum() == pytest.approx(
        csv["storage_units"].p_nom.sum()
    )
    assert len(topo.lines_df) == len(csv["lines"])

    buses = topo.buses_df.index
    for df in (topo.loads_df, topo.generators_df, topo.storage_units_df):
        assert df.bus.isin(buses).all()
    for df in (topo.lines_df, topo.transformers_df):
        assert df.bus0.isin(buses).all() and df.bus1.isin(buses).all()
    assert topo.transformers_hvmv_df.bus1.isin(topo.mv_grid.buses_df.index).all()

    lv_buses = pd.Series(
        [b for lv in topo.lv_grids for b in lv.buses_df.index], dtype=object
    )
    assert not lv_buses.duplicated().any()
    assert set(lv_buses) == set(buses[topo.buses_df.v_nom < 1])

    trafos = topo.transformers_df
    raw = csv["transformers"].loc[trafos.index]
    np.testing.assert_allclose(trafos.r_pu, raw.r)
    np.testing.assert_allclose(trafos.x_pu, raw.x)
    assert (trafos.r_pu > 0).all() and (trafos.x_pu > 0).all()
    assert (
        topo.buses_df.loc[trafos.bus0, "v_nom"].to_numpy()
        > topo.buses_df.loc[trafos.bus1, "v_nom"].to_numpy()
    ).all()


# ======================================================================================
# grid integration of imported plants (generators, heat pumps, batteries)
# ======================================================================================


@pytest.fixture(scope="module")
def _grid3_original():
    return EDisGo(ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False)


@pytest.fixture
def grid3(_grid3_original):
    """Fresh copy of test network 3 (egon grid with building IDs)."""
    return copy.deepcopy(_grid3_original)


def _buildings_with_lv_load(edisgo, n):
    """First ``n`` building IDs of conventional loads at LV buses."""
    loads = edisgo.topology.loads_df
    conv = loads[loads.type == "conventional_load"]
    lv = conv[edisgo.topology.buses_df.loc[conv.bus, "v_nom"].to_numpy() < 1]
    return lv.building_id.dropna().astype(int).unique()[:n]


def _pv_df(building_ids, p_nom):
    n = len(building_ids)
    return pd.DataFrame(
        {
            "p_nom": p_nom,
            "weather_cell_id": [11051] * n,
            "building_id": building_ids,
            "generator_id": range(1, n + 1),
            "type": ["solar"] * n,
            "subtype": ["pv_rooftop"] * n,
            "source_id": [None] * n,
        },
        index=range(1, n + 1),
    )


class TestGridIntegration:
    @pytest.mark.xfail(
        strict=True,
        reason="Bug: two new PV plants in the same building get the same name "
        "Generator_pv_rooftop_<building_id>; the voltage level lookup with .at on "
        "the duplicated label already fails (ValueError), and the later "
        "rename(index={dup: new}) would rename both rows anyway.",
    )
    def test_two_new_pv_plants_in_one_building(self, grid3):
        """
        Integrating several PV rooftop plants per building keeps all of them:
        the generator index stays unique and Σ p_nom of the integrated plants
        equals the input.
        """
        b = _buildings_with_lv_load(grid3, 2)
        pv = _pv_df([b[0], b[0], b[1]], [0.004, 0.006, 0.005])
        names, _ = generators_import._integrate_new_pv_rooftop_to_buildings(grid3, pv)
        gens = grid3.topology.generators_df
        assert not gens.index.has_duplicates
        assert len(set(names)) == 3
        assert gens.loc[names, "p_nom"].sum() == pytest.approx(pv.p_nom.sum())

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: the name collision check in "
        "_integrate_new_pv_rooftop_to_buildings compares with storage_units_df "
        "instead of generators_df, so a second plant in a building that already "
        "has an integrated PV plant duplicates the generator name.",
    )
    def test_new_pv_plant_name_does_not_collide_with_existing(self, grid3):
        """
        A PV plant integrated into a building that already has an imported PV
        plant gets a new name; existing generators are never overwritten or
        duplicated.
        """
        b = _buildings_with_lv_load(grid3, 1)
        generators_import._integrate_new_pv_rooftop_to_buildings(
            grid3, _pv_df([b[0]], [0.004])
        )
        n_before = len(grid3.topology.generators_df)
        generators_import._integrate_new_pv_rooftop_to_buildings(
            grid3, _pv_df([b[0]], [0.003])
        )
        gens = grid3.topology.generators_df
        assert len(gens) == n_before + 1
        assert not gens.index.has_duplicates

    @given(
        p_nom=st.lists(st.floats(0.001, 0.09), min_size=1, max_size=6),
        seed=st.integers(0, 1000),
    )
    @FIXTURE_SETTINGS
    def test_new_pv_plants_conserve_capacity_and_level(self, grid3, p_nom, seed):
        """
        PV plants in distinct buildings are all integrated: Σ p_nom is kept, the
        names are new and unique, and every plant is connected to an existing
        bus of a voltage level at least as high as required for its size (LV
        plants at their building's bus).
        """
        from edisgo.tools.tools import (
            determine_bus_voltage_level,
            determine_grid_integration_voltage_level,
        )

        edisgo = copy.deepcopy(grid3)
        buildings = _buildings_with_lv_load(edisgo, 200)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(buildings, size=len(p_nom), replace=False)
        before = set(edisgo.topology.generators_df.index)
        names, _ = generators_import._integrate_new_pv_rooftop_to_buildings(
            edisgo, _pv_df(list(chosen), p_nom)
        )
        gens = edisgo.topology.generators_df
        assert not gens.index.has_duplicates
        assert not set(names) & before
        assert gens.loc[names, "p_nom"].sum() == pytest.approx(sum(p_nom))
        assert gens.loc[names, "bus"].isin(edisgo.topology.buses_df.index).all()
        loads = edisgo.topology.loads_df
        for name in names:
            gen = gens.loc[name]
            required = determine_grid_integration_voltage_level(edisgo, gen.p_nom)
            assert determine_bus_voltage_level(edisgo, gen.bus) <= required
            if required == 7:
                building_buses = loads.bus[loads.building_id == gen.building_id]
                assert gen.bus in set(building_buses)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: two individual heat pumps in the same building both get the "
        "name HP_<building_id>; rename(index={dup: new}) renames both rows, so "
        "loads_df gets a duplicated index.",
    )
    def test_two_heat_pumps_in_one_building(self, grid3):
        """
        Several individual heat pumps in one building are all integrated with
        unique names and their Σ p_set is kept.
        """
        b = _buildings_with_lv_load(grid3, 2)
        hp = pd.DataFrame(
            {
                "p_set": [0.003, 0.004, 0.005],
                "weather_cell_id": [11051] * 3,
                "building_id": [b[0], b[0], b[1]],
            },
            index=[1, 2, 3],
        )
        names = heat_pump_import._grid_integration(
            grid3,
            hp_individual=hp,
            hp_central=pd.DataFrame(),
            resistive_heaters_central=pd.DataFrame(),
        )
        loads = grid3.topology.loads_df
        assert not loads.index.has_duplicates
        assert loads.loc[list(names), "p_set"].sum() == pytest.approx(hp.p_set.sum())

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: an individual heat pump in a building without conventional "
        "load gets bus NaN; small heat pumps are still added to loads_df.",
    )
    def test_heat_pump_in_building_without_load_gets_a_bus(self, grid3):
        """
        Every integrated heat pump is connected to an existing bus, also if its
        building has no conventional load (then via its geolocation or an
        error, but never a missing bus).
        """
        hp = pd.DataFrame(
            {"p_set": [0.003], "weather_cell_id": [11051], "building_id": [-1]},
            index=[1],
        )
        heat_pump_import._grid_integration(
            grid3,
            hp_individual=hp,
            hp_central=pd.DataFrame(),
            resistive_heaters_central=pd.DataFrame(),
        )
        loads = grid3.topology.loads_df
        assert loads.bus.isin(grid3.topology.buses_df.index).all()

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: two home batteries in the same building both get the name "
        "Storage_<building_id>; rename(index={dup: new}) renames both rows, so "
        "storage_units_df gets a duplicated index.",
    )
    def test_two_batteries_in_one_building(self, grid3):
        """
        Several home batteries in one building are all integrated with unique
        names, and Σ p_nom and Σ capacity (p_nom·max_hours) are kept.
        """
        b = _buildings_with_lv_load(grid3, 2)
        bat = pd.DataFrame(
            {
                "p_nom": [0.003, 0.004, 0.005],
                "capacity": [0.006, 0.008, 0.01],
                "building_id": [b[0], b[0], b[1]],
            },
            index=[1, 2, 3],
        )
        storage_import._home_batteries_grid_integration(grid3, bat.copy())
        sto = grid3.topology.storage_units_df
        assert not sto.index.has_duplicates
        assert sto.p_nom.sum() == pytest.approx(bat.p_nom.sum())
        assert (sto.p_nom * sto.max_hours).sum() == pytest.approx(bat.capacity.sum())

    @given(
        p_nom=st.lists(st.floats(0.001, 0.09), min_size=1, max_size=5),
        hours=st.floats(0.5, 4),
        seed=st.integers(0, 1000),
    )
    @FIXTURE_SETTINGS
    def test_batteries_conserve_power_energy_and_bus(self, grid3, p_nom, hours, seed):
        """
        Home batteries in distinct buildings keep Σ p_nom and Σ capacity
        (max_hours = capacity/p_nom), and each one is connected at the bus of a
        PV plant or load of its own building if its size allows LV connection.
        """
        edisgo = copy.deepcopy(grid3)
        buildings = _buildings_with_lv_load(edisgo, 200)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(buildings, size=len(p_nom), replace=False)
        bat = pd.DataFrame(
            {
                "p_nom": p_nom,
                "capacity": np.array(p_nom) * hours,
                "building_id": chosen,
            },
            index=range(1, len(p_nom) + 1),
        )
        names = storage_import._home_batteries_grid_integration(edisgo, bat.copy())
        topo = edisgo.topology
        sto = topo.storage_units_df.loc[names]
        assert len(sto) == len(bat)
        assert sto.p_nom.sum() == pytest.approx(bat.p_nom.sum())
        assert (sto.p_nom * sto.max_hours).sum() == pytest.approx(bat.capacity.sum())
        lv_limit = edisgo.config["grid_connection"]["upper_limit_voltage_level_7"]
        gens = topo.generators_df
        gen_building = gens.get("building_id", pd.Series(np.nan, index=gens.index))
        for _, s in sto.iterrows():
            assert s.bus in topo.buses_df.index
            if s.p_nom <= lv_limit:
                own = pd.concat(
                    [
                        topo.loads_df.bus[topo.loads_df.building_id == s.building_id],
                        gens.bus[gen_building == s.building_id],
                    ]
                )
                assert s.bus in set(own)


# ======================================================================================
# profiles: time index, demandlib, heat/electricity demand, DSM
# ======================================================================================


class TestProfiles:
    @given(
        year=st.integers(2000, 2050),
        source=st.sampled_from(["argument", "set", "none"]),
        start_day=st.integers(0, 364),
        n=st.integers(1, 200),
    )
    @FIXTURE_SETTINGS
    def test_timeindex_helper_selects_within_full_year(
        self, wc_grid, year, source, start_day, n
    ):
        """
        For every reference year (leap years included) and every source of the
        time index (argument, already set, none), the full-year index has 8760
        unique sorted hours of one year without 29 February, and the selected
        time index is a subset of it, so ``.loc[timeindex]`` never fails.
        """
        edisgo = copy.deepcopy(wc_grid)
        edisgo.config["timeindex"]["reference_year"] = year
        data_year = 2011 if year % 2 else year
        start = pd.Timestamp(f"{data_year}-01-01") + pd.Timedelta(days=start_day)
        index = pd.date_range(start, periods=n, freq="h")
        index = index[index.year == data_year]
        assume(not index.empty and index[0].year == data_year)
        assume(not (((index.month == 2) & (index.day == 29)).all()))
        if source == "argument":
            ti, full = timeseries_import._timeindex_helper_func(edisgo, index)
        elif source == "set":
            edisgo.timeseries.timeindex = index
            ti, full = timeseries_import._timeindex_helper_func(edisgo)
        else:
            edisgo.timeseries.timeindex = pd.DatetimeIndex([])
            ti, full = timeseries_import._timeindex_helper_func(edisgo)
        assert len(full) == 8760
        assert full.is_unique and full.is_monotonic_increasing
        assert (full.year == year).all()
        assert not ((full.month == 2) & (full.day == 29)).any()
        assert ti.isin(full).all()
        assert len(ti) > 0

    @pytest.mark.parametrize("year", [2011, 2020, 2035])
    def test_demandlib_profiles_normalised(self, wc_grid, year):
        """
        For any reference year, also a leap year, the sectoral standard load
        profiles of a full year are non-negative, have 8760 hourly values and sum
        to 1 MWh each (normalised annual consumption).
        """
        edisgo = copy.deepcopy(wc_grid)
        edisgo.config["timeindex"]["reference_year"] = year
        edisgo.timeseries.timeindex = pd.DatetimeIndex([])
        profiles = timeseries_import.load_time_series_demandlib(edisgo)
        assert len(profiles) == 8760
        assert set(profiles.columns) == {
            "residential",
            "cts",
            "industrial",
            "agricultural",
        }
        assert (profiles >= 0).all().all()
        np.testing.assert_allclose(profiles.sum(), 1.0, rtol=1e-9)

    @pytest.mark.xfail(
        strict=True,
        reason="Klärung: a heat pump and a resistive heater in the same building "
        "both get the full building heat demand, so the demand is counted twice; "
        "the intended split between the two is not defined.",
    )
    def test_heat_demand_not_double_counted_per_building(self, grid3, monkeypatch):
        """
        The heat demand of a building is covered once: summed over all heat
        pumps in the building (heat pump plus resistive heater), the assigned
        heat demand equals the building's heat demand.
        """
        building = int(_buildings_with_lv_load(grid3, 1)[0])
        bus = grid3.topology.loads_df.bus[
            grid3.topology.loads_df.building_id == building
        ].iloc[0]
        for sector, p in (
            ("individual_heating", 0.003),
            ("individual_heating_resistive_heater", 0.002),
        ):
            grid3.add_component(
                comp_type="load",
                type="heat_pump",
                sector=sector,
                bus=bus,
                p_set=p,
                building_id=building,
            )
        hours = pd.RangeIndex(8760)
        demand = pd.DataFrame({building: np.linspace(0.001, 0.002, 8760)}, index=hours)
        monkeypatch.setattr(
            timeseries_import,
            "get_residential_heat_profiles_per_building",
            lambda *args, **kwargs: demand,
        )
        monkeypatch.setattr(
            timeseries_import,
            "get_cts_profiles_per_building",
            lambda *args, **kwargs: pd.DataFrame(index=hours),
        )
        grid3.timeseries.timeindex = pd.date_range("2011-01-01", periods=24, freq="h")
        grid3.config["timeindex"]["reference_year"] = 2011
        result = timeseries_import.heat_demand_oedb(grid3, "eGon2035", engine=None)
        hps = grid3.topology.loads_df.index[
            grid3.topology.loads_df.building_id == building
        ]
        hps = [h for h in hps if h in result.columns]
        assert len(hps) == 2
        np.testing.assert_allclose(
            result[hps].sum(axis=1), demand[building].iloc[:24].to_numpy()
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: electricity_demand_oedb renames the building-ID columns with "
        "a Series indexed by building ID; if two loads of a sector share a "
        "building, the column label becomes a Series object (TypeError: unhashable) "
        "and the loads get no valid profile. Same pattern in dsm_import.oedb.",
    )
    def test_electricity_profiles_for_every_load(self, grid3, monkeypatch):
        """
        Every conventional load with a building ID gets exactly one electricity
        profile column, also if several loads of a sector share a building, and
        each column is the profile of its building.
        """
        loads = grid3.topology.loads_df
        res = loads[
            (loads.type == "conventional_load") & (loads.sector == "residential")
        ]
        building = res.building_id.dropna().iloc[0]
        extra = res[res.building_id == building].iloc[0].copy()
        grid3.topology.loads_df.loc["Load_residential_second_in_building"] = extra
        loads = grid3.topology.loads_df
        res = loads[
            (loads.type == "conventional_load") & (loads.sector == "residential")
        ]
        hours = pd.RangeIndex(8760)
        buildings = res.building_id.dropna().unique()
        profiles = pd.DataFrame(
            {b: np.full(8760, i + 1.0) for i, b in enumerate(buildings)}, index=hours
        )
        monkeypatch.setattr(
            timeseries_import,
            "get_residential_electricity_profiles_per_building",
            lambda *args, **kwargs: profiles,
        )
        grid3.timeseries.timeindex = pd.date_range("2011-01-01", periods=24, freq="h")
        grid3.config["timeindex"]["reference_year"] = 2011
        result = timeseries_import.electricity_demand_oedb(
            grid3, "eGon2035", engine=None, load_names=res.index
        )
        assert result.columns.is_unique
        assert set(result.columns) == set(res.index[res.building_id.notna()])
        for name in result.columns:
            expected = profiles[res.at[name, "building_id"]].iloc[0]
            assert (result[name] == expected).all(), name

    @given(
        p_set=st.lists(st.floats(0.001, 1.0), min_size=1, max_size=8),
        profile=st.lists(st.floats(-1.0, 1.0), min_size=1, max_size=24),
    )
    @settings(deadline=None, max_examples=examples(40))
    def test_dsm_profiles_distributed_by_rating(self, p_set, profile):
        """
        Distributing the aggregated CTS DSM potential to the CTS loads conserves
        it per time step (Σ over loads = aggregate), gives every load the share
        p_set/Σ p_set at all time steps and keeps the sign.
        """
        loads = pd.DataFrame(
            {"p_set": p_set}, index=[f"Load_cts_{i}" for i in range(len(p_set))]
        )
        agg = pd.DataFrame({"cts": profile})
        dsm = {k: agg.copy() for k in ("p_min", "p_max", "e_min", "e_max")}
        out = dsm_import._distribute_dsm_profiles_to_cts_loads(dsm, loads)
        share = loads.p_set / loads.p_set.sum()
        for k, df in out.items():
            assert list(df.columns) == list(loads.index)
            np.testing.assert_allclose(df.sum(axis=1), agg.cts, atol=1e-12)
            expected = np.outer(agg.cts, share)
            np.testing.assert_allclose(df.to_numpy(), expected, atol=1e-12)
            assert (
                np.sign(df.to_numpy()) * np.sign(agg.cts.to_numpy())[:, None] >= 0
            ).all()


# ======================================================================================
# electromobility import (SimBEV/TracBEV example data)
# ======================================================================================


def _emob_grid():
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
    edisgo.set_timeindex(pd.date_range("1/1/2011", periods=24 * 7, freq="h"))
    edisgo.resample_timeseries()
    return edisgo


@pytest.fixture(scope="module")
def _emob_imported_original():
    edisgo = _emob_grid()
    electromobility_import.import_electromobility_from_dir(
        edisgo,
        pytest.simbev_example_scenario_path,
        pytest.tracbev_example_scenario_path,
    )
    return edisgo


@pytest.fixture(scope="module")
def _emob_distributed_original(_emob_imported_original):
    edisgo = copy.deepcopy(_emob_imported_original)
    electromobility_import.distribute_charging_demand(edisgo)
    return edisgo


class TestElectromobility:
    def test_charging_processes_are_feasible(self, _emob_imported_original):
        """
        Every imported charging process can be served: the charging demand does
        not exceed nominal power × parking time (time steps × step size), the
        parking ends not before it starts, and in the "frugal" mode only
        processes with positive demand are kept. The demand is
        rounded in the SimBEV output, hence a tolerance of 1e-3 kWh.
        """
        em = _emob_imported_original.electromobility
        cp = em.charging_processes_df
        assert len(cp) > 0
        hours = cp.park_time_timesteps * em.stepsize / 60
        possible = cp.nominal_charging_capacity_kW * hours
        assert (cp.chargingdemand_kWh <= possible + 1e-3).all()
        assert (cp.park_end_timesteps >= cp.park_start_timesteps).all()
        assert (cp.chargingdemand_kWh > 0).all()
        assert 0 < em.eta_charging_points <= 1

    @pytest.mark.xfail(
        strict=True,
        reason="Klärung: in the SimBEV example data park_time_timesteps is "
        "end - start for hpc processes (and once off by 30 steps), while "
        "charging_processes_from_oedb computes end - start + 1; the convention of "
        "the parking time and of park_end_timesteps (inclusive?) is not defined.",
    )
    def test_parking_time_matches_start_and_end(self, _emob_imported_original):
        """
        The parking time of every imported charging process equals the number of
        time steps from start to end (both inclusive), as in the OEDB import.
        """
        cp = _emob_imported_original.electromobility.charging_processes_df
        duration = cp.park_end_timesteps.astype(int) - cp.park_start_timesteps + 1
        assert (cp.park_time_timesteps.astype(int) == duration).all()

    @given(
        rates=st.fixed_dictionaries(
            {
                f"gc_to_car_rate_{uc}": st.floats(0.0, 1.0)
                for uc in ("home", "work", "public", "hpc")
            }
        )
    )
    @FIXTURE_SETTINGS
    def test_minimum_charging_parks_per_car(self, _emob_imported_original, rates):
        """
        For every use case there are at least ``gc_to_car_rate``·number of cars
        potential charging parks after ``assure_minimum_potential_charging_parks``;
        the original parks are kept and the index is 0..n−1.
        """
        edisgo = copy.deepcopy(_emob_imported_original)
        raw = electromobility_import.read_gpkg_potential_charging_parks(
            pytest.tracbev_example_scenario_path, edisgo
        )
        out = electromobility_import.assure_minimum_potential_charging_parks(
            edisgo, raw, **rates
        )
        num_cars = edisgo.electromobility.charging_processes_df.car_id.nunique()
        counts = out.use_case.value_counts()
        for uc in ("home", "work", "public", "hpc"):
            assert counts.get(uc, 0) >= rates[f"gc_to_car_rate_{uc}"] * num_cars, uc
            raw_uc = (raw.use_case == uc).sum()
            assert counts.get(uc, 0) >= raw_uc, uc
        assert list(out.index) == list(range(len(out)))

    def test_import_from_dir_passes_rates(self):
        """
        The ``gc_to_car_rate_*`` options of ``import_electromobility_from_dir``
        take effect: with a rate of 1 there is at least one work park per car.
        """
        edisgo = _emob_grid()
        electromobility_import.import_electromobility_from_dir(
            edisgo,
            pytest.simbev_example_scenario_path,
            pytest.tracbev_example_scenario_path,
            gc_to_car_rate_work=1.0,
        )
        em = edisgo.electromobility
        num_cars = em.charging_processes_df.car_id.nunique()
        parks = em.potential_charging_parks_gdf
        assert (parks.use_case == "work").sum() >= num_cars

    def test_distributed_processes_do_not_overlap(self, _emob_distributed_original):
        """
        After distributing the demand, every process with demand has a charging
        park and a charging point, private processes (home, work) a park of their
        own use case, and no two processes use the same charging point at the same
        time step.
        """
        em = _emob_distributed_original.electromobility
        cp = em.charging_processes_df
        cp = cp[cp.chargingdemand_kWh > 0]
        assert cp.charging_park_id.notna().all()
        assert cp.charging_point_id.notna().all()
        parks = em.potential_charging_parks_gdf
        park_uc = parks.loc[cp.charging_park_id.astype(int), "use_case"].to_numpy()
        private = cp.use_case.isin(["home", "work"]).to_numpy()
        assert (park_uc[private] == cp.use_case[private].to_numpy()).all()
        for point, group in cp.groupby("charging_point_id"):
            g = group.sort_values("park_start_timesteps")
            starts = g.park_start_timesteps.to_numpy()
            ends = g.park_end_timesteps.to_numpy()
            assert (starts[1:] > ends[:-1]).all(), point

    def test_integrated_charging_parks_match_connection_capacity(
        self, _emob_distributed_original
    ):
        """
        Integrating the charging parks adds one charging point load per
        designated park, at an existing bus, with p_set equal to the park's grid
        connection capacity.
        """
        edisgo = copy.deepcopy(_emob_distributed_original)
        electromobility_import.integrate_charging_parks(edisgo)
        topo = edisgo.topology
        parks = [
            cp
            for cp in edisgo.electromobility.potential_charging_parks
            if cp.designated_charging_point_capacity > 0 and cp.within_grid
        ]
        assert parks
        cps = topo.loads_df[topo.loads_df.type == "charging_point"]
        assert len(cps) == len(parks)
        assert cps.bus.isin(topo.buses_df.index).all()
        assert cps.p_set.sum() == pytest.approx(
            sum(cp.grid_connection_capacity for cp in parks)
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Klärung: distribute_public_charging_demand reuses a free charging "
        "point of the same nominal power regardless of its use case, so public "
        "processes end up at hpc charging parks (and vice versa).",
    )
    def test_public_processes_at_park_of_own_use_case(self, _emob_distributed_original):
        """
        Public and hpc charging processes are assigned to charging parks of their
        own use case.
        """
        em = _emob_distributed_original.electromobility
        cp = em.charging_processes_df
        cp = cp[(cp.chargingdemand_kWh > 0) & cp.use_case.isin(["public", "hpc"])]
        parks = em.potential_charging_parks_gdf
        park_uc = parks.loc[cp.charging_park_id.astype(int), "use_case"].to_numpy()
        assert (park_uc == cp.use_case.to_numpy()).all()

    @given(weights=st.lists(st.floats(0, 10, allow_nan=False), min_size=1, max_size=30))
    @settings(deadline=None, max_examples=examples(50))
    def test_normalize_gives_distribution(self, weights):
        """
        ``normalize`` returns a probability distribution: non-negative, sums to
        one; all-zero weights give the uniform distribution.
        """
        df = pd.DataFrame({"weight": weights})
        out = np.asarray(electromobility_import.normalize(df)).ravel()
        assert (out >= 0).all()
        assert out.sum() == pytest.approx(1.0)
        if sum(weights) == 0:
            np.testing.assert_allclose(out, 1 / len(weights))

    @given(
        a=st.floats(0, 5, allow_nan=False),
        b=st.floats(0, 5, allow_nan=False),
        lower=st.floats(0.05, 0.5),
        upper=st.floats(0.6, 2.0),
        factor=st.floats(0.5, 1.0),
    )
    @settings(deadline=None, max_examples=examples(60))
    def test_grid_connection_capacity_monotone(self, a, b, lower, upper, factor):
        """
        The grid connection capacity never exceeds the summed charging point
        capacity, is monotonically non-decreasing in it and continuous at both
        limits.
        """
        assume(factor * upper >= lower)
        f = electromobility_import.determine_grid_connection_capacity
        kw = {"lower_limit": lower, "upper_limit": upper, "minimum_factor": factor}
        lo, hi = sorted((a, b))
        assert f(lo, **kw) <= f(hi, **kw) + 1e-12
        assert f(a, **kw) <= a + 1e-12
        eps = 1e-9
        for x in (lower, upper):
            assert f(x - eps, **kw) == pytest.approx(f(x + eps, **kw), abs=1e-6)
