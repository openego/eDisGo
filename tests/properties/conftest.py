"""
Shared fixtures and helpers for the property-based tests in ``tests/properties``.

The fixtures follow one pattern: an expensive original (grid import, worst-case
time series, power flow, reinforcement) is built once per session and every test
gets its own :func:`copy.deepcopy`, so tests may mutate their object freely.
For hypothesis tests, which run many examples inside a single test call, use the
``*_factory`` fixtures: every call of the factory returns a fresh copy.

The helpers compute physical quantities directly from time series, topology and
power flow results instead of relying on derived eDisGo attributes (e.g.
``Results.grid_losses`` is built with ``abs()`` and would hide sign errors).
"""

from __future__ import annotations

import copy
import logging
import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hypothesis import settings

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints as checks

# --------------------------------------------------------------------------------------
# tolerances
# --------------------------------------------------------------------------------------

#: Absolute power tolerance in MW/Mvar for balances computed from power flow results.
#: PyPSA's Newton-Raphson stops at a mismatch of 1e-6 p.u. per bus (base 1 MVA), so
#: a sum over ~150 buses can deviate by ~1e-4 in the worst case; 1e-5 is reached in
#: practice and still far below the smallest load (~1e-3 MW) in the test grids.
POWER_ATOL = 1e-5

#: Relative tolerance for loading criteria (relative load <= 1 + LOAD_RTOL). The
#: reinforcement checks ``> 1`` exactly; the tolerance only absorbs float noise.
LOAD_RTOL = 1e-6

#: Absolute tolerance in p.u. for voltage criteria (same reasoning as LOAD_RTOL).
VOLTAGE_ATOL = 1e-6

#: Names of all limit checks returned by :func:`limit_violations`.
ALL_CHECKS = (
    "mv_line",
    "lv_line",
    "hv_mv_station",
    "mv_lv_station",
    "mv_voltage",
    "mv_lv_voltage",
    "lv_voltage",
)

# --------------------------------------------------------------------------------------
# hypothesis profiles
# --------------------------------------------------------------------------------------

#: Factor applied to ``max_examples`` per profile. Select the profile with the
#: environment variable ``HYPOTHESIS_PROFILE`` (``default`` or ``deep``). ``deep``
#: is meant for long runs on a server; tests keep their small example counts in the
#: default profile so the suite stays fast.
EXAMPLE_SCALE = {"default": 1, "deep": 15}

settings.register_profile("default", deadline=None)
settings.register_profile("deep", deadline=None, print_blob=True)
_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "default")
settings.load_profile(_PROFILE)


def examples(n):
    """Return the number of hypothesis examples for a test with base count ``n``."""
    return n * EXAMPLE_SCALE.get(_PROFILE, 1)


# --------------------------------------------------------------------------------------
# grid fixtures
# --------------------------------------------------------------------------------------


def build_worst_case_grid(path=None, **kwargs):
    """
    Return a new EDisGo object with worst-case time series (all four cases).

    Parameters
    ----------
    path : str or None
        ding0 grid directory, defaults to ``pytest.ding0_test_network_path``.
    kwargs :
        Passed on to :class:`~.EDisGo`.

    """
    edisgo = EDisGo(ding0_grid=path or pytest.ding0_test_network_path, **kwargs)
    edisgo.set_time_series_worst_case_analysis()
    return edisgo


@pytest.fixture(scope="session")
def _wc_grid_original():
    return build_worst_case_grid()


@pytest.fixture(scope="session")
def _analyzed_grid_original(_wc_grid_original):
    edisgo = copy.deepcopy(_wc_grid_original)
    edisgo.analyze()
    return edisgo


@pytest.fixture(scope="session")
def _reinforced_grid_original(_wc_grid_original):
    edisgo = copy.deepcopy(_wc_grid_original)
    edisgo.reinforce()
    return edisgo


@pytest.fixture
def wc_grid(_wc_grid_original):
    """Test network 1 with worst-case time series, no power flow results yet."""
    return copy.deepcopy(_wc_grid_original)


@pytest.fixture
def wc_grid_factory(_wc_grid_original):
    """Callable returning a fresh copy of :func:`wc_grid` on every call."""
    return lambda: copy.deepcopy(_wc_grid_original)


@pytest.fixture
def analyzed_grid(_analyzed_grid_original):
    """Test network 1 with worst-case time series after ``analyze()``."""
    return copy.deepcopy(_analyzed_grid_original)


@pytest.fixture
def analyzed_grid_factory(_analyzed_grid_original):
    """Callable returning a fresh copy of :func:`analyzed_grid` on every call."""
    return lambda: copy.deepcopy(_analyzed_grid_original)


@pytest.fixture
def reinforced_grid(_reinforced_grid_original):
    """Test network 1 with worst-case time series after ``reinforce()``."""
    return copy.deepcopy(_reinforced_grid_original)


@pytest.fixture
def reinforced_grid_factory(_reinforced_grid_original):
    """Callable returning a fresh copy of :func:`reinforced_grid` on every call."""
    return lambda: copy.deepcopy(_reinforced_grid_original)


# --------------------------------------------------------------------------------------
# helpers: power balance and power flow
# --------------------------------------------------------------------------------------


def net_injection_totals(edisgo, timesteps=None):
    """
    Return the total net injection of all components per time step.

    Generators and storage units count positive (generator sign convention,
    storage discharge positive), loads negative. Computed from the
    :class:`~.network.timeseries.TimeSeries` getters, i.e. independent of PyPSA.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Columns ``p`` (MW) and ``q`` (Mvar), index are the time steps.

    """
    ts = edisgo.timeseries
    if timesteps is None:
        timesteps = ts.timeindex
    p = (
        ts.generators_active_power.sum(axis=1)
        + ts.storage_units_active_power.sum(axis=1)
        - ts.loads_active_power.sum(axis=1)
    )
    q = (
        ts.generators_reactive_power.sum(axis=1)
        + ts.storage_units_reactive_power.sum(axis=1)
        - ts.loads_reactive_power.sum(axis=1)
    )
    return pd.DataFrame({"p": p, "q": q}).loc[timesteps]


def power_balance(edisgo):
    """
    Return the losses implied by slack power and component time series.

    For every time step in the last power flow, the network losses must equal
    ``P_slack + sum(P_gen) + sum(P_storage) - sum(P_load)`` (and likewise for Q).
    The values are computed from :attr:`~.network.results.Results.pfa_slack`
    and the time series, i.e. without ``Results.grid_losses``.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Columns ``p`` (active power losses in MW) and ``q`` (reactive power
        balance in Mvar, may be negative due to line capacitance).

    """
    slack = edisgo.results.pfa_slack
    injections = net_injection_totals(edisgo, slack.index)
    return slack[["p", "q"]] + injections


def bus_injections(edisgo, timesteps=None):
    """
    Return the net injection per bus and time step from the eDisGo data model.

    Sums generators and storage units (positive) and loads (negative) at each bus
    in :attr:`~.network.topology.Topology.buses_df`, using the topology to map
    components to buses. Independent of :func:`~.io.pypsa_io.to_pypsa`.

    Returns
    -------
    tuple(:pandas:`pandas.DataFrame<DataFrame>`, :pandas:`pandas.DataFrame<DataFrame>`)
        Active (MW) and reactive (Mvar) net injection; index are time steps,
        columns are bus names.

    """
    topo, ts = edisgo.topology, edisgo.timeseries
    if timesteps is None:
        timesteps = ts.timeindex
    result = []
    for attr in ("active_power", "reactive_power"):
        total = pd.DataFrame(0.0, index=timesteps, columns=topo.buses_df.index)
        for comp, sign in (("generators", 1), ("storage_units", 1), ("loads", -1)):
            df = getattr(ts, f"{comp}_{attr}").loc[timesteps]
            buses = getattr(topo, f"{comp}_df").loc[df.columns, "bus"]
            per_bus = df.T.groupby(buses.values).sum().T
            total = total.add(sign * per_bus, fill_value=0.0)
        result.append(total)
    return tuple(result)


def run_pypsa_power_flow(edisgo, **kwargs):
    """
    Export to PyPSA, run the non-linear power flow and return the network.

    Gives access to signed branch flows ``p0``/``p1``, which
    :class:`~.network.results.Results` only stores as absolute values.
    ``kwargs`` are passed to :meth:`~.EDisGo.to_pypsa`.

    """
    timesteps = kwargs.pop("timesteps", edisgo.timeseries.timeindex)
    pypsa_network = edisgo.to_pypsa(timesteps=timesteps, **kwargs)
    pf = pypsa_network.pf(timesteps)
    assert pf["converged"]["0"].all(), "power flow did not converge"
    return pypsa_network


def worst_case_timesteps(edisgo):
    """
    Return the worst-case time steps split by the level they dimension.

    Mirrors :meth:`~.EDisGo.reinforce`: time steps whose case name contains
    ``"mv"`` dimension the MV level, those containing ``"lv"`` the LV level.

    Returns
    -------
    dict
        ``{"mv": DatetimeIndex, "lv": DatetimeIndex}``

    """
    twc = edisgo.timeseries.timeindex_worst_cases
    return {
        level: pd.DatetimeIndex(twc[twc.index.str.contains(level)])
        for level in ("mv", "lv")
    }


# --------------------------------------------------------------------------------------
# helpers: technical limits
# --------------------------------------------------------------------------------------


def limit_violations(
    edisgo,
    checks_to_run=ALL_CHECKS,
    timesteps=None,
    load_rtol=LOAD_RTOL,
    voltage_atol=VOLTAGE_ATOL,
):
    """
    Return all violated technical limits of the last power flow.

    Uses the same allowed loads and voltage bands as the grid reinforcement
    (:mod:`edisgo.flex_opt.check_tech_constraints`, ``split_voltage_band=True``),
    but with an explicit tolerance and an optional restriction to time steps.
    Components that were not part of the last power flow are skipped.

    Parameters
    ----------
    edisgo : :class:`~.EDisGo`
    checks_to_run : iterable(str)
        Subset of :data:`ALL_CHECKS`.
    timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Only consider these time steps (must be in the last power flow).
    load_rtol, voltage_atol : float
        Tolerances, see module constants.

    Returns
    -------
    dict(str, :pandas:`pandas.Series<Series>`)
        Per check the maximum relative load (loading checks) or maximum absolute
        voltage deviation from the allowed band in p.u. (voltage checks) of every
        violating component. Checks without violations are omitted, so an empty
        dict means all limits are met.

    """
    results = edisgo.results
    if timesteps is None:
        timesteps = results.v_res.index
    topo = edisgo.topology
    mv_lines = topo.mv_grid.lines_df.index
    in_pf = results.s_res.columns
    violations = {}

    def _keep(name, values, limit):
        values = values.loc[timesteps]
        worst = values.max()
        worst = worst[worst > limit]
        if not worst.empty:
            violations[name] = worst

    if "mv_line" in checks_to_run:
        lines = mv_lines[mv_lines.isin(in_pf)]
        _keep("mv_line", checks.lines_relative_load(edisgo, lines), 1 + load_rtol)
    if "lv_line" in checks_to_run:
        lines = topo.lines_df.index[
            ~topo.lines_df.index.isin(mv_lines) & topo.lines_df.index.isin(in_pf)
        ]
        if len(lines) > 0:
            _keep("lv_line", checks.lines_relative_load(edisgo, lines), 1 + load_rtol)
    if "hv_mv_station" in checks_to_run:
        rel = checks.stations_relative_load(edisgo, [topo.mv_grid])
        _keep("hv_mv_station", rel, 1 + load_rtol)
    if "mv_lv_station" in checks_to_run:
        lv_grids = [
            g for g in topo.lv_grids if g.transformers_df.index.isin(in_pf).all()
        ]
        if lv_grids:
            rel = checks.stations_relative_load(edisgo, lv_grids)
            _keep("mv_lv_station", rel, 1 + load_rtol)
    # bus selection as in check_tech_constraints.voltage_issues
    station_buses = pd.Index(topo.transformers_df.bus1.unique())
    for level in ("mv", "mv_lv", "lv"):
        name = f"{level}_voltage"
        if name not in checks_to_run:
            continue
        if level == "mv":
            buses = topo.mv_grid.buses_df.index
        elif level == "mv_lv":
            buses = station_buses
        else:
            buses = topo.buses_df.index.drop(
                topo.mv_grid.buses_df.index.append(station_buses)
            )
        buses = buses[buses.isin(results.v_res.columns)]
        if len(buses) == 0:
            continue
        dev = checks.voltage_deviation_from_allowed_voltage_limits(
            edisgo, buses=buses, split_voltage_band=True
        )
        _keep(name, dev.abs(), voltage_atol)
    return violations


def reinforce_checks_per_worst_case(edisgo):
    """
    Re-run the power flows the worst-case reinforcement bases its decisions on.

    :meth:`~.EDisGo.reinforce` in worst-case mode runs three reinforcements:
    MV level (``mode="mv"``, LV aggregated at the stations' primary side) on the
    MV time steps, MV/LV stations (``mode="mvlv"``) on the LV time steps and LV
    level (whole grid) on the LV time steps. This helper repeats exactly these
    power flows on a copy and collects the limit violations the respective step
    checks.

    Returns
    -------
    dict(str, dict)
        Violations per step (``"mv"``, ``"mvlv"``, ``"lv"``) as returned by
        :func:`limit_violations`.

    """
    steps = worst_case_timesteps(edisgo)
    setup = {
        "mv": ("mv", steps["mv"], ("mv_line", "hv_mv_station", "mv_voltage")),
        "mvlv": ("mvlv", steps["lv"], ("mv_lv_station", "mv_lv_voltage")),
        "lv": (
            None,
            steps["lv"],
            ("lv_line", "mv_lv_station", "mv_lv_voltage", "lv_voltage"),
        ),
    }
    out = {}
    for name, (mode, timesteps, to_check) in setup.items():
        obj = copy.deepcopy(edisgo)
        obj.analyze(mode=mode, timesteps=timesteps)
        out[name] = limit_violations(obj, to_check)
    return out


# --------------------------------------------------------------------------------------
# helpers: comparison
# --------------------------------------------------------------------------------------

TOPOLOGY_ATTRIBUTES = (
    "buses_df",
    "lines_df",
    "transformers_df",
    "transformers_hvmv_df",
    "generators_df",
    "loads_df",
    "storage_units_df",
    "switches_df",
)

TIMESERIES_ATTRIBUTES = (
    "generators_active_power",
    "generators_reactive_power",
    "loads_active_power",
    "loads_reactive_power",
    "storage_units_active_power",
    "storage_units_reactive_power",
)


def assert_topology_equal(left, right, **kwargs):
    """Assert that all component tables of two topologies are equal."""
    for attr in TOPOLOGY_ATTRIBUTES:
        pd.testing.assert_frame_equal(
            getattr(left.topology, attr).sort_index(),
            getattr(right.topology, attr).sort_index(),
            obj=f"topology.{attr}",
            **kwargs,
        )


def assert_timeseries_equal(left, right, **kwargs):
    """Assert that all component time series of two EDisGo objects are equal."""
    pd.testing.assert_index_equal(left.timeseries.timeindex, right.timeseries.timeindex)
    for attr in TIMESERIES_ATTRIBUTES:
        pd.testing.assert_frame_equal(
            getattr(left.timeseries, attr).sort_index(axis=1),
            getattr(right.timeseries, attr).sort_index(axis=1),
            obj=f"timeseries.{attr}",
            **kwargs,
        )


def edisgo_warnings(caplog, func, *args, **kwargs):
    """Run ``func`` and return the messages of all warnings logged meanwhile."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        func(*args, **kwargs)
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def max_abs_diff(a, b):
    """Maximum absolute elementwise difference of two aligned frames."""
    a, b = a.align(b, join="outer")
    return float(np.nanmax(np.abs((a - b).to_numpy(dtype=float))))


# --------------------------------------------------------------------------------------
# helpers: injected power flow results and topology checks
# --------------------------------------------------------------------------------------


def inject_power_flow_results(edisgo, v_res=None, s_res=None, pfa_slack=None):
    """
    Overwrite power flow results without running a power flow.

    ``s_res`` is stored as active power (``pfa_p = s_res``, ``pfa_q = 0``), so
    :attr:`~.network.results.Results.s_res` returns it unchanged. Frames that are
    ``None`` are left as they are.

    """
    results = edisgo.results
    if v_res is not None:
        results.v_res = v_res
    if s_res is not None:
        results.pfa_p = s_res
        results.pfa_q = pd.DataFrame(0.0, index=s_res.index, columns=s_res.columns)
    if pfa_slack is not None:
        results.pfa_slack = pfa_slack


def assert_radial(graph, name=""):
    """Assert that ``graph`` is a connected tree (radial grid without loops)."""
    assert nx.is_connected(graph), f"{name} is not connected"
    assert nx.is_tree(graph), f"{name} is not radial"
