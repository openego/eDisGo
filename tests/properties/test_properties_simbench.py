"""
Property-based tests of the SimBench import (:mod:`edisgo.io.simbench_import`).

Instead of comparing with values stored from the test grid, the tests check
properties that must hold for every SimBench grid: the equipment data agree with
the SimBench type data, no component or power gets lost, the grid is radial with
open switches, the time series are the scaled SimBench profiles, the power flow
balances, and the reinforcement leaves no limit violation. Where inputs can be
varied cheaply, the tests use hypothesis; ``HYPOTHESIS_PROFILE=deep`` multiplies
the number of examples by 15 for long runs.
"""

from __future__ import annotations

import copy
import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.io import simbench_import as sb

EXAMPLE_SCALE = 15 if os.environ.get("HYPOTHESIS_PROFILE") == "deep" else 1

#: Absolute power tolerance in MW for balances from power flow results (see the
#: tolerances in tests/properties of the property-based tests).
POWER_ATOL = 1e-5


def examples(n: int) -> int:
    """Number of hypothesis examples for a test with base count ``n``."""
    return n * EXAMPLE_SCALE


def read_csv(table: str) -> pd.DataFrame:
    """One csv table of the SimBench test grid."""
    return pd.read_csv(
        os.path.join(pytest.simbench_test_network_path, f"{table}.csv"), sep=";"
    )


def import_test_grid(**kwargs) -> EDisGo:
    edisgo = EDisGo()
    edisgo.import_simbench_grid(pytest.simbench_test_network_path, **kwargs)
    return edisgo


@pytest.fixture(scope="module")
def _imported_original():
    return import_test_grid()


@pytest.fixture
def imported(_imported_original):
    """SimBench test grid with its time series (two days)."""
    return copy.deepcopy(_imported_original)


@pytest.fixture
def imported_factory(_imported_original):
    """Callable returning a fresh copy of :func:`imported` on every call."""
    return lambda: copy.deepcopy(_imported_original)


# ======================================================================================
# helpers of the import
# ======================================================================================


class TestHelpers:
    @given(level=st.integers(1, 9), number=st.integers(1, 999))
    @settings(deadline=None, max_examples=examples(50))
    def test_subnet_id_is_unique_per_subnet(self, level, number):
        """The grid ID encodes level and number of a subnet without collision."""
        for prefix in ("MV", "LV"):
            grid_id = sb._subnet_id(f"{prefix}{level}.{number}")
            assert divmod(grid_id, 1000) == (level, number)

    @given(
        prefix=st.sampled_from(["Air_", "Soil_", "HLS_", "H0-", "G1-", "L2-", "lv_"]),
        rest=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", max_size=8),
    )
    @settings(deadline=None, max_examples=examples(50))
    def test_load_type_depends_on_profile_prefix_only(self, prefix, rest):
        load_type, sector = sb._load_type_and_sector(prefix + rest)
        expected = {
            "Air_": ("heat_pump", "individual_heating"),
            "Soil_": ("heat_pump", "individual_heating"),
            "HLS_": ("charging_point", "home"),
            "H0-": ("conventional_load", "residential"),
            "G1-": ("conventional_load", "cts"),
            "L2-": ("conventional_load", "agricultural"),
            "lv_": ("conventional_load", "other"),
        }[prefix]
        assert (load_type, sector) == expected

    @given(
        name=st.sampled_from(["PV", "Wind", "Biomass", "Hydro"]),
        upper=st.booleans(),
        suffix=st.sampled_from(["", "_MV", "_HV", "1"]),
    )
    @settings(deadline=None, max_examples=examples(30))
    def test_generator_type_ignores_case_and_suffix(self, name, upper, suffix):
        simbench_type = (name.upper() if upper else name) + suffix
        assert sb._generator_type(simbench_type) == sb.GENERATOR_TYPES[name.lower()]

    @given(
        s_r=st.floats(0.05, 100),
        u_k=st.floats(1, 20),
        r_share=st.floats(0, 0.99),
    )
    @settings(deadline=None, max_examples=examples(50))
    def test_transformer_impedance_from_short_circuit_test(self, s_r, u_k, r_share):
        """
        From the short-circuit test, |z| = u_k and the copper losses at rated
        power are r * S_r; the reactance is real and not negative.
        """
        z_pu = u_k / 100
        p_cu_kw = r_share * z_pu * s_r * 1e3
        trafo_types = pd.DataFrame(
            {"id": ["t"], "sR": [s_r], "vmImp": [u_k], "pCu": [p_cu_kw]}
        )
        trafo = sb._transformer_impedance(pd.DataFrame({"type": ["t"]}), trafo_types)
        assert trafo.s_nom.iloc[0] == s_r
        assert np.hypot(trafo.r_pu, trafo.x_pu).iloc[0] == pytest.approx(z_pu)
        assert (trafo.r_pu * trafo.s_nom * 1e3).iloc[0] == pytest.approx(p_cu_kw)
        assert trafo.x_pu.iloc[0] >= 0

    @given(data=st.data())
    @settings(deadline=None, max_examples=examples(20))
    def test_fused_buses_are_closed_switch_components(self, data):
        """
        Nodes are fused exactly when closed switches connect them, and each group
        is named after one of its nodes that is not an auxiliary node, if any.
        """
        nodes = read_csv("Node").set_index("id")
        switches = read_csv("Switch")
        cond = data.draw(
            st.lists(
                st.sampled_from([0, 1]), min_size=len(switches), max_size=len(switches)
            )
        )
        switches["cond"] = cond
        bus = sb._fuse_closed_switches(nodes, switches)

        graph = nx.Graph()
        graph.add_nodes_from(nodes.index)
        closed = switches[switches.cond == 1]
        graph.add_edges_from(zip(closed.nodeA, closed.nodeB))
        for component in nx.connected_components(graph):
            names = {bus[n] for n in component}
            assert len(names) == 1
            (name,) = names
            assert name in component
            if (nodes.loc[list(component), "type"] != "auxiliary").any():
                assert nodes.at[name, "type"] != "auxiliary"
        assert set(bus) == set(nodes.index)

    @given(data=st.data())
    @settings(deadline=None, max_examples=examples(20))
    def test_closing_switches_to_islands_reconnects_minimally(self, data):
        """
        After opening any closed switches, the import closes open switches until
        every part of the grid that the switches could connect is connected
        again; switches that close a ring stay open, closed switches stay closed.
        """
        nodes = read_csv("Node").set_index("id")
        switches = read_csv("Switch")
        lines, trafos = read_csv("Line"), read_csv("Transformer")
        closed = switches.index[switches.cond == 1]
        opened = data.draw(st.lists(st.sampled_from(list(closed)), unique=True))
        switches.loc[opened, "cond"] = 0
        result = sb._close_switches_to_islands(nodes, switches, lines, trafos)

        assert (result.cond >= switches.cond).all()

        def graph(switch_rows):
            mvlv = trafos[trafos.voltLvl == sb.MV_LEVEL + 1]
            g = nx.Graph()
            g.add_nodes_from(nodes.index)
            for df, a, b in [
                (lines, "nodeA", "nodeB"),
                (mvlv, "nodeHV", "nodeLV"),
                (switch_rows, "nodeA", "nodeB"),
            ]:
                g.add_edges_from(zip(df[a], df[b]))
            return g

        connected = graph(result[result.cond == 1])
        assert nx.number_connected_components(connected) == (
            nx.number_connected_components(graph(result))
        )
        for sw in result[result.cond == 0].itertuples():
            assert nx.has_path(connected, sw.nodeA, sw.nodeB)


# ======================================================================================
# equipment data
# ======================================================================================


class TestEquipmentData:
    def test_line_parameters_follow_line_types(self, imported):
        """
        r = R'·l, x = X'·l, b = B'·l and s_nom = √3·U_n·I_max for every line, and
        lines connect buses of the same nominal voltage.
        """
        topo = imported.topology
        lines = topo.lines_df
        raw = read_csv("Line").set_index("id").loc[lines.index]
        types = read_csv("LineType").set_index("id").loc[raw.type]
        length = raw.length.to_numpy()
        np.testing.assert_allclose(lines.length, length)
        np.testing.assert_allclose(lines.r, types.r.to_numpy() * length)
        np.testing.assert_allclose(lines.x, types.x.to_numpy() * length)
        np.testing.assert_allclose(lines.b, types.b.to_numpy() * 1e-6 * length)
        v_nom = topo.buses_df.loc[lines.bus0, "v_nom"].to_numpy()
        np.testing.assert_allclose(topo.buses_df.loc[lines.bus1, "v_nom"], v_nom)
        np.testing.assert_allclose(
            lines.s_nom, np.sqrt(3) * v_nom * types.iMax.to_numpy() / 1e3
        )
        assert (lines.r > 0).all() and (lines.x > 0).all() and (lines.b >= 0).all()

    def test_transformer_parameters_follow_transformer_types(self, imported):
        """
        Every transformer keeps rated power, short-circuit voltage |z| = u_k and
        copper losses r·S_r of its type, and connects buses of the type's rated
        voltages.
        """
        topo = imported.topology
        trafos = pd.concat([topo.transformers_df, topo.transformers_hvmv_df])
        assert len(trafos) == len(read_csv("Transformer"))
        types = read_csv("TransformerType").set_index("id").loc[trafos.type_info]
        np.testing.assert_allclose(trafos.s_nom, types.sR)
        np.testing.assert_allclose(
            np.hypot(trafos.r_pu, trafos.x_pu), types.vmImp.to_numpy() / 100
        )
        np.testing.assert_allclose(trafos.r_pu * trafos.s_nom * 1e3, types.pCu)
        assert (trafos.x_pu > 0).all()
        v_nom = topo.buses_df.v_nom
        np.testing.assert_allclose(v_nom.loc[trafos.bus1], types.vmLV)
        mvlv = topo.transformers_df
        mvlv_types = types.iloc[: len(mvlv)]
        np.testing.assert_allclose(v_nom.loc[mvlv.bus0], mvlv_types.vmHV)


# ======================================================================================
# conservation of components and power
# ======================================================================================


class TestConservation:
    def test_every_component_once_with_its_rated_power(self, imported):
        """
        Every SimBench load, generator and PV system with home battery is
        imported exactly once, and the total rated power is kept.
        """
        topo = imported.topology
        loads, res, storage = read_csv("Load"), read_csv("RES"), read_csv("Storage")
        pv_battery = storage[storage.type.str.startswith("PV_Storage")]
        assert sorted(topo.loads_df.index) == sorted(loads.id)
        assert topo.loads_df.p_set.sum() == pytest.approx(loads.pLoad.sum())
        assert sorted(topo.generators_df.index) == sorted([*res.id, *pv_battery.id])
        assert topo.generators_df.p_nom.sum() == pytest.approx(
            res.pRES.sum() + pv_battery.sR.sum()
        )
        assert len(topo.storage_units_df) == len(storage) - len(pv_battery)

    def test_components_stay_at_their_voltage_level(self, imported):
        """Every component is at a bus with the nominal voltage of its node."""
        topo = imported.topology
        vm = read_csv("Node").set_index("id").vmR
        for df, table in [(topo.loads_df, "Load"), (topo.generators_df, "RES")]:
            raw = read_csv(table).set_index("id")
            names = df.index.intersection(raw.index)
            np.testing.assert_allclose(
                topo.buses_df.loc[df.loc[names, "bus"], "v_nom"],
                vm.loc[raw.loc[names, "node"]],
            )

    def test_every_branch_and_component_refers_to_a_bus(self, imported):
        topo = imported.topology
        buses = topo.buses_df.index
        assert buses.is_unique
        for df in (topo.loads_df, topo.generators_df, topo.storage_units_df):
            assert df.bus.isin(buses).all()
        for df in (topo.lines_df, topo.transformers_df):
            assert df.bus0.isin(buses).all() and df.bus1.isin(buses).all()
        assert topo.transformers_hvmv_df.bus1.isin(topo.mv_grid.buses_df.index).all()
        switches = topo.switches_df
        assert switches.bus_open.isin(buses).all()
        assert switches.bus_closed.isin(buses).all()
        assert switches.branch.isin(topo.lines_df.index).all()

    def test_lv_buses_partitioned_into_lv_grids(self, imported):
        """
        Every LV bus belongs to exactly one LV grid, which is fed by its own
        MV/LV transformers at its station.
        """
        topo = imported.topology
        lv_buses = [b for lv in topo.lv_grids for b in lv.buses_df.index]
        assert len(lv_buses) == len(set(lv_buses))
        assert set(lv_buses) == set(topo.buses_df.index[topo.buses_df.v_nom < 1])
        for lv in topo.lv_grids:
            station = lv.station.index
            assert len(station) == 1
            trafos = topo.transformers_df[topo.transformers_df.bus1 == station[0]]
            assert not trafos.empty
            assert trafos.index.str.startswith(f"LVStation_{lv.id}_").all()


# ======================================================================================
# topology
# ======================================================================================


class TestTopology:
    def test_grid_is_radial_with_open_switches(self, imported):
        graph = imported.topology.to_graph()
        assert nx.is_tree(graph)

    @given(data=st.data())
    @settings(
        deadline=None,
        max_examples=examples(10),
        # the factory returns a fresh copy for every example
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_each_closed_switch_closes_one_ring(self, imported_factory, data):
        """With k switch disconnectors closed, the grid has k independent rings."""
        edisgo = imported_factory()
        switches = list(edisgo.topology.mv_grid.switch_disconnectors)
        closed = data.draw(
            st.lists(st.integers(0, len(switches) - 1), unique=True), label="closed"
        )
        for i in closed:
            switches[i].close()
        graph = edisgo.topology.to_graph()
        assert len(nx.cycle_basis(graph)) == len(closed)
        # the open side of a closed switch is left without connection
        assert nx.number_connected_components(graph) == 1 + len(closed)


# ======================================================================================
# time series
# ======================================================================================


def scaled_profiles(table: str, power: str, suffix: str = "") -> pd.DataFrame:
    """SimBench profiles of all elements of `table` scaled with `power`."""
    elements = read_csv(table)
    profiles = read_csv(f"{table}Profile")
    return pd.DataFrame(
        profiles[elements.profile + suffix].to_numpy() * elements[power].to_numpy(),
        columns=elements.id,
    )


class TestTimeSeries:
    def test_power_is_rated_power_times_profile(self, imported):
        """P(t) = P_r·p(t) and Q(t) = Q_r·q(t) for every component and step."""
        ts = imported.timeseries
        storage = read_csv("Storage")
        expected = [
            (ts.loads_active_power, scaled_profiles("Load", "pLoad", "_pload")),
            (ts.loads_reactive_power, scaled_profiles("Load", "qLoad", "_qload")),
            (ts.generators_active_power, scaled_profiles("RES", "pRES")),
            (ts.generators_reactive_power, scaled_profiles("RES", "qRES")),
            # load sign convention in SimBench, generator sign convention in eDisGo
            (ts.generators_active_power, -scaled_profiles("Storage", "pStor")),
        ]
        for actual, values in expected:
            np.testing.assert_allclose(actual[values.columns].to_numpy(), values)
        assert len(ts.generators_active_power.columns) == len(read_csv("RES")) + len(
            storage[storage.type.str.startswith("PV_Storage")]
        )

    def test_generators_feed_in(self, imported):
        assert (imported.timeseries.generators_active_power >= 0).all().all()

    def test_time_steps_are_consecutive_quarter_hours(self, imported):
        """
        Profile rows become consecutive 15-minute steps in the reference year,
        starting on the date of the first SimBench label.
        """
        timeindex = imported.timeseries.timeindex
        first = pd.to_datetime(read_csv("LoadProfile").time[0], format="%d.%m.%Y %H:%M")
        assert len(timeindex) == len(read_csv("LoadProfile"))
        assert (timeindex.year == 2011).all()
        assert timeindex[0] == first.replace(year=2011)
        assert (np.diff(timeindex) == pd.Timedelta("15min")).all()

    @given(data=st.data())
    @settings(
        deadline=None,
        max_examples=examples(5),
        # the fixture is only read
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_timeindex_selects_time_steps(self, imported, data):
        """Importing with a time index gives the same values as slicing."""
        steps = data.draw(
            st.lists(
                st.sampled_from(list(imported.timeseries.timeindex)),
                min_size=1,
                max_size=20,
                unique=True,
            )
        )
        timeindex = pd.DatetimeIndex(sorted(steps))
        selected = import_test_grid(timeindex=timeindex).timeseries
        for attr in ("loads_active_power", "generators_reactive_power"):
            pd.testing.assert_frame_equal(
                getattr(selected, attr),
                getattr(imported.timeseries, attr).loc[timeindex],
                check_freq=False,
            )


# ======================================================================================
# power flow and reinforcement
# ======================================================================================


class TestPowerFlow:
    def test_losses_are_positive_and_small(self, imported):
        """
        The slack supplies load minus generation plus the losses: losses are
        positive and only a few percent of the transported power.
        """
        imported.analyze()
        ts = imported.timeseries
        net_load = (
            ts.loads_active_power.sum(axis=1)
            - ts.generators_active_power.sum(axis=1)
            - ts.storage_units_active_power.sum(axis=1)
        )
        losses = imported.results.pfa_slack.p - net_load
        assert (losses > -POWER_ATOL).all()
        transported = ts.loads_active_power.sum(
            axis=1
        ) + ts.generators_active_power.sum(axis=1)
        assert (losses < 0.1 * transported).all()


class TestReinforcement:
    @pytest.fixture(scope="class")
    @classmethod
    def reinforced(cls):
        edisgo = import_test_grid(
            timeindex=pd.date_range("2011-01-01 18:00", periods=2, freq="15min")
        )
        # overload the MV/LV station and the LV grid
        lv_loads = edisgo.topology.get_lv_grid(1108).loads_df.index
        loads_p = edisgo.timeseries.loads_active_power
        loads_p[lv_loads] *= 10
        edisgo.timeseries.loads_active_power = loads_p
        edisgo.reinforce()
        return edisgo

    def test_no_limit_violations_remain(self, reinforced):
        assert checks.mv_line_max_relative_overload(reinforced).empty
        assert checks.lv_line_max_relative_overload(reinforced).empty
        assert checks.hv_mv_station_max_overload(reinforced).empty
        assert checks.mv_lv_station_max_overload(reinforced).empty
        for level in ("mv", "mv_lv", "lv"):
            assert checks.voltage_issues(reinforced, voltage_level=level).empty

    def test_second_reinforcement_builds_nothing(self, reinforced):
        edisgo = copy.deepcopy(reinforced)
        lines = edisgo.topology.lines_df.copy()
        trafos = edisgo.topology.transformers_df.copy()
        edisgo.reinforce()
        pd.testing.assert_frame_equal(edisgo.topology.lines_df, lines)
        pd.testing.assert_frame_equal(edisgo.topology.transformers_df, trafos)

    def test_new_transformers_follow_ding0_names_and_are_paid(self, reinforced):
        """
        Transformers built by the reinforcement keep the ding0 naming scheme the
        import introduces, feed the same station, and every one of them shows up
        in the expansion costs, which are not negative.
        """
        trafos = reinforced.topology.transformers_df
        assert trafos.index.is_unique
        assert trafos.index.str.match(
            r"LVStation_1108_transformer(_reinforced)?_\d+$"
        ).all()
        assert trafos.bus1.nunique() == 1
        original = import_test_grid(time_series=False).topology.transformers_df
        costs = reinforced.results.grid_expansion_costs
        assert (costs.total_costs >= 0).all()
        assert set(trafos.index.difference(original.index)) <= set(costs.index)
