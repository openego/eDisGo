"""
Property-based tests for the grid data model in :mod:`edisgo.network`:
:mod:`~edisgo.network.topology` (line parameters, queries, modifications, grid
connection of new components), :mod:`~edisgo.network.grids` and
:mod:`~edisgo.network.components`.

Each test states the property it checks. Fixtures and helpers live in
``tests/properties/conftest.py``.
"""

from __future__ import annotations

import copy
import logging

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from geopy.distance import geodesic
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.network.components import Generator, Load, Storage, Switch

from .conftest import (
    POWER_ATOL,
    assert_radial,
    edisgo_warnings,
    examples,
    expected_line_parameters,
    undefined_references,
)

#: Settings for tests on a copy of test network 1 without power flow (a few
#: milliseconds to a second per example). The function scoped factories return a
#: fresh copy on every call, so reusing them across examples is safe.
GRID_SETTINGS = settings(
    deadline=None,
    max_examples=examples(15),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Relative tolerance for line parameters: they are products and quotients of a
#: handful of floats, so results agree to ~1e-15; 1e-12 still detects any wrong
#: factor (unit, 2πf, number of parallel lines, voltage).
LINE_RTOL = 1e-12

#: Weather cell of test network 1, needed for new solar generators.
WEATHER_CELL = 1122074

#: Warnings that test network 1 already causes on import: it contains a line of
#: 0.56 m. They are no finding of the tests below.
KNOWN_WARNINGS = ("very short line lengths",)


# ======================================================================================
# helpers
# ======================================================================================


def _grid_voltage(topology, bus):
    return topology.buses_df.at[bus, "v_nom"]


def _line_types(topology, level, v_nom=None):
    """Line types of a level, as (equipment key, name), optionally by rated voltage."""
    keys = ["lv_cables"] if level == "lv" else ["mv_cables", "mv_overhead_lines"]
    types = []
    for key in keys:
        data = topology.equipment_data[key]
        if v_nom is not None:
            data = data[data.U_n == v_nom]
        types += [(key, name) for name in data.index.unique()]
    return types


def _equipment(topology, key, name, v_nom):
    """Equipment row of a line type; for names used at several voltages, at v_nom."""
    data = topology.equipment_data[key].loc[[name]]
    if len(data) > 1:
        data = data[data.U_n == v_nom]
    return data.iloc[0]


def _assert_line_parameters(line, expected, check_b=True):
    for attr in ("r", "x", "b", "s_nom") if check_b else ("r", "x", "s_nom"):
        np.testing.assert_allclose(
            line[attr], expected[attr], rtol=LINE_RTOL, atol=0, err_msg=attr
        )


def _new_warnings(caplog, topology, before=(), ignore=()):
    """Integrity warnings of ``topology`` that are not known or already present."""
    warnings = edisgo_warnings(caplog, topology.check_integrity)
    known = KNOWN_WARNINGS + tuple(ignore)
    return [w for w in warnings if w not in before and not any(k in w for k in known)]


def _line_graph(topology):
    """Graph of buses and lines only (no transformers)."""
    graph = nx.Graph()
    graph.add_nodes_from(topology.buses_df.index)
    graph.add_edges_from(topology.lines_df[["bus0", "bus1"]].itertuples(index=False))
    return graph


def _free_leaf(topology, bus):
    """True if ``bus`` has one line and nothing else refers to it (independent)."""
    topo = topology
    lines = (topo.lines_df.bus0 == bus).sum() + (topo.lines_df.bus1 == bus).sum()
    referenced = (
        (topo.loads_df.bus == bus).any()
        or (topo.generators_df.bus == bus).any()
        or (topo.storage_units_df.bus == bus).any()
        or topo.transformers_df[["bus0", "bus1"]].eq(bus).any(axis=None)
        or topo.transformers_hvmv_df[["bus0", "bus1"]].eq(bus).any(axis=None)
        or topo.switches_df[["bus_open", "bus_closed"]].eq(bus).any(axis=None)
    )
    return lines == 1 and not referenced


def _add_chord(topology, bus0, bus1):
    """Add a short cable between two buses of the same voltage level."""
    return topology.add_line(bus0, bus1, 0.1, x=0.01, r=0.01, s_nom=1.0)


def _comp_data(comp_type, p, geom, voltage_level, idx, **extra):
    """Component data for the ``connect_to_*`` functions."""
    data = {"geom": geom, "voltage_level": voltage_level, **extra}
    if comp_type == "generator":
        data.update(
            generator_id=idx,
            p_nom=p,
            generator_type="solar",
            subtype="ground_mounted",
            weather_cell_id=WEATHER_CELL,
        )
    elif comp_type == "storage_unit":
        data.update(p_nom=p)
    elif comp_type == "charging_point":
        data.update(p_set=p, sector=extra.get("sector", "public"))
    else:
        data.update(p_set=p, sector=extra.get("sector", "district_heating"))
    return data


def _comp_df(topology, comp_type):
    return {
        "generator": topology.generators_df,
        "storage_unit": topology.storage_units_df,
    }.get(comp_type, topology.loads_df)


def _comp_power(df, name):
    return df.at[name, "p_nom"] if "p_nom" in df.columns else df.at[name, "p_set"]


@pytest.fixture(scope="module")
def grid_3():
    """Test network 3 (new ding0 format, geo-referenced LV grids)."""
    return EDisGo(ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False)


# ======================================================================================
# line parameters
# ======================================================================================


class TestLineParameters:
    @GRID_SETTINGS
    @given(
        level=st.sampled_from(["mv", "lv"]),
        data=st.data(),
        length=st.floats(0.001, 5.0),
        num_parallel=st.integers(1, 4),
    )
    def test_add_line_parameters_follow_equipment_data(
        self, wc_grid_factory, level, data, length, num_parallel
    ):
        """
        A line added with ``type_info`` of a type rated for the grid voltage
        gets ``r = R'·l/n``, ``x = 2πf·L'·l/n``, ``b = 2πf·C'·l·n`` and
        ``s_nom = √3·U·I·n`` from the equipment data.
        """
        topo = wc_grid_factory().topology
        buses = (
            topo.mv_grid.buses_df.index
            if level == "mv"
            else topo.buses_df.index.drop(topo.mv_grid.buses_df.index)
        )
        bus0 = data.draw(st.sampled_from(sorted(buses)))
        v_nom = _grid_voltage(topo, bus0)
        key, type_name = data.draw(st.sampled_from(_line_types(topo, level, v_nom)))
        bus1 = topo.add_bus(
            "Bus_new", v_nom, lv_grid_id=topo.buses_df.at[bus0, "lv_grid_id"]
        )
        name = topo.add_line(
            bus0, bus1, length, type_info=type_name, num_parallel=num_parallel
        )
        expected = expected_line_parameters(
            _equipment(topo, key, type_name, v_nom), length, num_parallel, v_nom
        )
        _assert_line_parameters(topo.lines_df.loc[name], expected)

    @GRID_SETTINGS
    @given(
        level=st.sampled_from(["mv", "lv"]),
        data=st.data(),
        prior_parallel=st.integers(1, 4),
    )
    def test_change_line_type_parameters_follow_equipment_data(
        self, wc_grid_factory, level, data, prior_parallel
    ):
        """
        ``change_line_type`` sets one cable of the new type with parameters from
        the equipment data at the grid voltage (also for MV cables rated for
        another voltage), independent of the previous number of parallel lines,
        and is idempotent.
        """
        topo = wc_grid_factory().topology
        lines = topo.mv_grid.lines_df.index if level == "mv" else None
        if lines is None:
            lines = topo.lines_df.index.drop(topo.mv_grid.lines_df.index)
        selected = data.draw(
            st.lists(
                st.sampled_from(sorted(lines)), min_size=1, max_size=4, unique=True
            )
        )
        key = "mv_cables" if level == "mv" else "lv_cables"
        type_name = data.draw(
            st.sampled_from(sorted(topo.equipment_data[key].index.unique()))
        )
        topo.update_number_of_parallel_lines(
            pd.Series(prior_parallel, index=selected, dtype=float)
        )
        topo.change_line_type(selected, type_name)
        once = topo.lines_df.loc[selected].copy()
        topo.change_line_type(selected, type_name)
        pd.testing.assert_frame_equal(topo.lines_df.loc[selected], once)

        equipment = topo.equipment_data[key].loc[type_name]
        for name in selected:
            line = topo.lines_df.loc[name]
            v_nom = _grid_voltage(topo, line.bus0)
            assert line.num_parallel == 1
            assert line.kind == "cable"
            assert line.type_info == type_name
            _assert_line_parameters(
                line, expected_line_parameters(equipment, line.length, 1, v_nom)
            )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: add_line(type_info=...) computes s_nom with the rated voltage "
        "of the cable type (10 kV cable in the 20 kV grid: 6.18 MVA), "
        "change_line_type with the grid voltage (12.37 MVA)",
    )
    def test_add_line_and_change_line_type_agree(self, wc_grid):
        """
        For every MV cable type, a new line with ``type_info`` and an existing
        line of the same length changed to that type get identical r, x, b and
        s_nom.
        """
        topo = wc_grid.topology
        line = topo.mv_grid.lines_df.index[0]
        bus0, length = topo.lines_df.at[line, "bus0"], topo.lines_df.at[line, "length"]
        mismatches = []
        for type_name in topo.equipment_data["mv_cables"].index.unique():
            bus1 = topo.add_bus("Bus_new", topo.mv_grid.nominal_voltage)
            new = topo.add_line(bus0, bus1, length, type_info=type_name)
            topo.change_line_type([line], type_name)
            for attr in ("r", "x", "b", "s_nom"):
                a, b = topo.lines_df.at[new, attr], topo.lines_df.at[line, attr]
                if not np.isclose(a, b, rtol=LINE_RTOL):
                    mismatches.append((type_name, attr, a, b))
        assert mismatches == []

    @GRID_SETTINGS
    @given(data=st.data(), n_new=st.integers(1, 5), n_back=st.integers(1, 5))
    def test_update_number_of_parallel_lines_keeps_per_system_values(
        self, wc_grid_factory, data, n_new, n_back
    ):
        """
        Changing the number of parallel lines keeps ``r·n``, ``x·n``, ``b/n``
        and ``s_nom/n`` of every line; going n -> m -> n restores the line; and
        after ``change_line_type`` the parameters equal the equipment formulas
        for n parallel systems.
        """
        topo = wc_grid_factory().topology
        selected = data.draw(
            st.lists(
                st.sampled_from(sorted(topo.lines_df.index)),
                min_size=1,
                max_size=6,
                unique=True,
            )
        )
        before = topo.lines_df.loc[selected].copy()
        topo.update_number_of_parallel_lines(
            pd.Series(n_new, index=selected, dtype=float)
        )
        after = topo.lines_df.loc[selected]
        n0 = before.num_parallel
        for attr, power in (("r", 1), ("x", 1), ("b", -1), ("s_nom", -1)):
            np.testing.assert_allclose(
                after[attr] * n_new**power,
                before[attr] * n0**power,
                rtol=LINE_RTOL,
                err_msg=attr,
            )
        topo.update_number_of_parallel_lines(
            pd.Series(n0.to_numpy(), index=selected, dtype=float)
        )
        pd.testing.assert_frame_equal(
            topo.lines_df.loc[selected], before, check_exact=False, rtol=LINE_RTOL
        )

        line = selected[0]
        level = "mv" if line in topo.mv_grid.lines_df.index else "lv"
        key = f"{level}_cables"
        type_name = topo.equipment_data[key].index[0]
        topo.change_line_type([line], type_name)
        topo.update_number_of_parallel_lines(
            pd.Series(n_back, index=[line], dtype=float)
        )
        row = topo.lines_df.loc[line]
        expected = expected_line_parameters(
            topo.equipment_data[key].loc[type_name],
            row.length,
            n_back,
            _grid_voltage(topo, row.bus0),
        )
        _assert_line_parameters(row, expected)

    @pytest.mark.parametrize(
        "path, legacy",
        [("ding0_test_network_2_path", True), ("ding0_test_network_3_path", False)],
    )
    def test_imported_line_parameters_match_equipment_data(self, path, legacy):
        """
        Every line of an imported ding0 grid has a type from the equipment data,
        and its r, x and s_nom follow the equipment formulas at the grid
        voltage. (Test network 1 is an old ding0 export with other cable data
        for NAYY 4x1x35 and 10 kV ratings in the 20 kV grid and is not used.)
        """
        topo = EDisGo(
            ding0_grid=getattr(pytest, path), legacy_ding0_grids=legacy
        ).topology
        bad = []
        for name, line in topo.lines_df.iterrows():
            v_nom = _grid_voltage(topo, line.bus0)
            level = "lv" if v_nom < 1 else "mv"
            keys = [k for k, n in _line_types(topo, level) if n == line.type_info]
            assert keys, f"{name}: unknown type {line.type_info}"
            expected = expected_line_parameters(
                _equipment(topo, keys[0], line.type_info, v_nom),
                line.length,
                line.num_parallel,
                v_nom,
            )
            for attr in ("r", "x", "s_nom"):
                if not np.isclose(line[attr], expected[attr], rtol=1e-6):
                    bad.append((name, attr))
        # ding0 rounds the stored values to ~7 significant digits, hence 1e-6
        assert bad == []

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: ding0 grids store b = 0 for all MV lines and the import "
        "does not derive b from C' of the equipment data, so line charging of MV "
        "cables is missing in the power flow",
    )
    def test_imported_mv_lines_have_shunt_susceptance(self, grid_3):
        """MV lines of an imported grid carry ``b = 2πf·C'·l·n`` (C' > 0)."""
        topo = grid_3.topology
        lines = topo.mv_grid.lines_df
        v_nom = topo.mv_grid.nominal_voltage
        expected = [
            expected_line_parameters(
                _equipment(
                    topo,
                    "mv_cables"
                    if line.type_info in topo.equipment_data["mv_cables"].index
                    else "mv_overhead_lines",
                    line.type_info,
                    v_nom,
                ),
                line.length,
                line.num_parallel,
                v_nom,
            )["b"]
            for _, line in lines.iterrows()
        ]
        np.testing.assert_allclose(lines.b, expected, rtol=1e-6)

    @GRID_SETTINGS
    @given(data=st.data())
    @pytest.mark.xfail(
        strict=True,
        reason="Bug: add_line only detects an existing line in the same direction "
        "(and only if no reverse line exists); add_line(bus1, bus0) creates a "
        "parallel duplicate",
    )
    def test_add_line_detects_existing_line_in_both_directions(
        self, wc_grid_factory, data
    ):
        """
        Adding a line between two buses that are already connected returns the
        existing line and adds nothing, whatever the order of the buses.
        """
        topo = wc_grid_factory().topology
        name = data.draw(st.sampled_from(sorted(topo.lines_df.index)))
        bus0, bus1 = topo.lines_df.loc[name, ["bus0", "bus1"]]
        n_lines = len(topo.lines_df)
        for a, b in ((bus0, bus1), (bus1, bus0)):
            assert topo.add_line(a, b, 0.1, x=0.1, r=0.1, s_nom=1.0) == name
        assert len(topo.lines_df) == n_lines

    def test_equipment_data_is_physical(self, wc_grid):
        """
        Equipment data are physical: positive R', L', I_th and U_n, C' >= 0, and
        MV/LV transformer impedances satisfy ``r_pu, x_pu >= 0`` and
        ``r_pu² + x_pu² = (u_kr/100)²``.
        """
        eq = wc_grid.topology.equipment_data
        for key in ("mv_cables", "mv_overhead_lines", "lv_cables"):
            df = eq[key]
            assert (df[["R_per_km", "L_per_km", "I_max_th", "U_n"]] > 0).all(axis=None)
            assert (df.C_per_km >= 0).all()
        trafo = eq["lv_transformers"]
        assert (trafo.r_pu >= 0).all() and (trafo.x_pu >= 0).all()
        np.testing.assert_allclose(
            np.hypot(trafo.r_pu, trafo.x_pu), trafo.u_kr / 100, rtol=1e-12
        )


# ======================================================================================
# topology queries
# ======================================================================================


class TestTopologyQueries:
    def test_connected_lines_and_neighbours_are_consistent(self, wc_grid):
        """
        Handshake lemma: summed over all buses, ``get_connected_lines_from_bus``
        returns every line twice, each time at one of its ends; neighbours are
        symmetric and equal the line graph's neighbours; the connecting line of
        two buses does not depend on their order and is None for non-neighbours.
        """
        topo = wc_grid.topology
        graph = _line_graph(topo)
        total = 0
        for bus in topo.buses_df.index:
            lines = topo.get_connected_lines_from_bus(bus)
            total += len(lines)
            assert ((lines.bus0 == bus) | (lines.bus1 == bus)).all()
            if len(lines) == 0:
                continue
            neighbours = topo.get_neighbours(bus)
            assert neighbours == set(graph.neighbors(bus))
            for other in neighbours:
                assert bus in topo.get_neighbours(other)
                a = topo.get_line_connecting_buses(bus, other)
                b = topo.get_line_connecting_buses(other, bus)
                assert a is not None and set(a.index) == set(b.index)
        assert total == 2 * len(topo.lines_df)
        bus0 = topo.mv_grid.station.index[0]
        far = [b for b in topo.buses_df.index if not graph.has_edge(bus0, b)]
        assert topo.get_line_connecting_buses(bus0, far[-1]) is None

    def test_connected_components_partition_nodal_components(self, wc_grid):
        """
        ``get_connected_components_from_bus`` returns every load, generator and
        storage unit at exactly its own bus (the per-bus results partition the
        component tables).
        """
        topo = wc_grid.topology
        found = {k: [] for k in ("loads", "generators", "storage_units")}
        for bus in topo.buses_df.index:
            comps = topo.get_connected_components_from_bus(bus)
            for key in found:
                assert (comps[key].bus == bus).all()
                found[key] += list(comps[key].index)
        for key, names in found.items():
            assert sorted(names) == sorted(getattr(topo, f"{key}_df").index)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: get_connected_components_from_bus only looks at "
        "switches_df.bus_closed, so _check_bus_for_removal treats the bus_open of a "
        "switch as removable and remove_line(branch) deletes it (dangling switch)",
    )
    def test_switch_buses_are_never_removed(self, wc_grid_factory):
        """
        Both buses of every switch are reported as connected to the switch, are
        not removable, and removing the switch's branch line never leaves the
        switch with an undefined bus or branch.
        """
        topo = wc_grid_factory().topology
        assert len(topo.switches_df) > 0
        for name, sw in topo.switches_df.iterrows():
            for bus in (sw.bus_open, sw.bus_closed):
                comps = topo.get_connected_components_from_bus(bus)
                assert name in comps["switches"].index, (name, bus)
                assert not topo._check_bus_for_removal(bus), (name, bus)
            copy_topo = wc_grid_factory().topology
            copy_topo.remove_line(sw.branch)
            assert undefined_references(copy_topo) == []

    @settings(
        deadline=None,
        max_examples=examples(4),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data(), add_chord=st.booleans())
    def test_check_line_for_removal_matches_graph(
        self, wc_grid_factory, data, add_chord
    ):
        """
        A line may be removed iff it is not a bridge of the line graph (it lies
        on a loop) or one of its buses is a leaf without any other element. An
        optional extra line between two non-adjacent MV buses creates a loop.
        Switch branches are excluded (see the switch test above).
        """
        topo = wc_grid_factory().topology
        if add_chord:
            mv = sorted(topo.mv_grid.buses_df.index)
            a = data.draw(st.sampled_from(mv))
            b = data.draw(st.sampled_from(mv))
            assume(a != b and b not in set(_line_graph(topo).neighbors(a)))
            _add_chord(topo, a, b)
        graph = _line_graph(topo)
        bridges = {frozenset(e) for e in nx.bridges(graph)}
        for name, line in topo.lines_df.iterrows():
            if name in topo.switches_df.branch.values:
                continue
            expected = (
                frozenset((line.bus0, line.bus1)) not in bridges
                or _free_leaf(topo, line.bus0)
                or _free_leaf(topo, line.bus1)
            )
            assert topo._check_line_for_removal(name) == expected, name

    def test_remove_line_never_isolates_components(self, wc_grid_factory):
        """
        Removing any single line (or refusing to) leaves a connected grid in
        which every component refers to existing buses; with an extra loop line,
        the lines on the loop are actually removed. Switch branches are excluded
        (see the switch test above).
        """
        base = wc_grid_factory().topology
        mv = base.mv_grid.buses_df.index
        chord = _add_chord(base, mv[3], mv[-3])
        loop = nx.find_cycle(_line_graph(base))
        loop_lines = {
            n
            for n, line in base.lines_df.iterrows()
            if (line.bus0, line.bus1) in loop or (line.bus1, line.bus0) in loop
        }
        assert chord in loop_lines
        for name in base.lines_df.index:
            if name in base.switches_df.branch.values:
                continue
            topo = copy.deepcopy(base)
            topo.remove_line(name)
            assert nx.is_connected(topo.to_graph()), name
            assert undefined_references(topo) == [], name
            if name in loop_lines:
                assert name not in topo.lines_df.index

    def test_rings_and_switch_operations(self, wc_grid):
        """
        ``rings`` does not change the line table and finds one loop per closed
        switch, each through the MV station; ``Switch.open``/``close`` are
        idempotent, the open/closed bus is on the branch accordingly, closing
        adds one mesh per switch, and open -> close -> open restores the lines.
        """
        topo = wc_grid.topology
        lines_before = topo.lines_df.copy()
        rings = topo.rings
        pd.testing.assert_frame_equal(topo.lines_df, lines_before)
        assert len(rings) == len(topo.switches_df)
        station = topo.mv_grid.station.index[0]
        assert all(station in ring for ring in rings)

        switches = [Switch(id=s, topology=topo) for s in topo.switches_df.index]
        assert _n_meshes(wc_grid) == 0
        for k, sw in enumerate(switches, start=1):
            assert sw.state == "open"
            assert sw.bus_open in topo.lines_df.loc[sw.branch, ["bus0", "bus1"]].values
            sw.close()
            sw.close()
            assert sw.state == "closed"
            line_buses = topo.lines_df.loc[sw.branch, ["bus0", "bus1"]].values
            assert sw.bus_closed in line_buses and sw.bus_open not in line_buses
            assert _n_meshes(wc_grid) == k
        for sw in switches:
            sw.open()
            sw.open()
        pd.testing.assert_frame_equal(topo.lines_df, lines_before)
        assert _n_meshes(wc_grid) == 0

    @pytest.mark.xfail(
        strict=True,
        reason="Bug: Topology.rings is cached in _rings and never invalidated, so "
        "it is stale after the topology changed",
    )
    def test_rings_reflect_current_topology(self, wc_grid):
        """After adding a loop line, ``rings`` contains one more ring."""
        topo = wc_grid.topology
        n_rings = len(topo.rings)
        mv = topo.mv_grid.buses_df.index
        _add_chord(topo, mv[3], mv[-3])
        assert len(topo.rings) == n_rings + 1

    @GRID_SETTINGS
    @given(data=st.data(), n_chords=st.integers(0, 3))
    def test_find_meshes_counts_independent_loops(
        self, wc_grid_factory, data, n_chords
    ):
        """
        ``find_meshes`` returns None for the radial grid and exactly k meshes
        after adding k lines between distinct, non-adjacent bus pairs of the MV
        grid (each extra line raises the cyclomatic number by one).
        """
        edisgo = wc_grid_factory()
        topo = edisgo.topology
        mv = sorted(topo.mv_grid.buses_df.index)
        pairs = data.draw(
            st.lists(
                st.tuples(st.sampled_from(mv), st.sampled_from(mv)).map(
                    lambda p: tuple(sorted(p))
                ),
                min_size=n_chords,
                max_size=n_chords,
                unique=True,
            )
        )
        graph = _line_graph(topo)
        assume(all(a != b and not graph.has_edge(a, b) for a, b in pairs))
        for a, b in pairs:
            _add_chord(topo, a, b)
        assert _n_meshes(edisgo) == n_chords

    @GRID_SETTINGS
    @given(flip_seed=st.integers(0, 2**32 - 1))
    def test_sort_buses_orients_lines_away_from_station(
        self, wc_grid_factory, flip_seed
    ):
        """
        After ``sort_buses`` every line points away from the MV station (bus0
        is closer in the BFS tree), the bus pair of every line is unchanged, a
        second call changes nothing, and the result does not depend on the
        original orientation of the lines.
        """
        reference = wc_grid_factory().topology
        reference.sort_buses()
        topo = wc_grid_factory().topology
        rng = np.random.default_rng(flip_seed)
        flip = rng.random(len(topo.lines_df)) < 0.5
        df = topo.lines_df.copy()
        df.loc[flip, ["bus0", "bus1"]] = df.loc[flip, ["bus1", "bus0"]].to_numpy()
        topo.lines_df = df
        pairs = {n: frozenset(p) for n, p in df[["bus0", "bus1"]].iterrows()}
        topo.sort_buses()
        pd.testing.assert_frame_equal(topo.lines_df, reference.lines_df)
        depth = nx.single_source_shortest_path_length(
            topo.to_graph(), topo.mv_grid.station.index[0]
        )
        for name, line in topo.lines_df.iterrows():
            assert depth[line.bus0] < depth[line.bus1], name
            assert frozenset((line.bus0, line.bus1)) == pairs[name]
        once = topo.lines_df.copy()
        topo.sort_buses()
        pd.testing.assert_frame_equal(topo.lines_df, once)

    def test_sort_buses_keeps_power_flow_results(self, analyzed_grid):
        """
        Line orientation is a convention only: voltages and current magnitudes
        are the same before and after ``sort_buses`` (tolerance: two power flows
        of the same physical state, see V_ATOL_SAME_STATE in the facade tests).
        """
        before = analyzed_grid.results
        v, i = before.v_res.copy(), before.i_res.copy()
        analyzed_grid.topology.sort_buses()
        analyzed_grid.analyze()
        after = analyzed_grid.results
        pd.testing.assert_frame_equal(after.v_res[v.columns], v, atol=1e-8)
        pd.testing.assert_frame_equal(after.i_res[i.columns].abs(), i.abs(), atol=1e-8)


def _n_meshes(edisgo):
    """Number of meshes reported by ``Topology.find_meshes`` (0 for None)."""
    # find_meshes takes the object with to_graph() as its only argument
    meshes = edisgo.topology.find_meshes()
    return 0 if meshes is None else len(meshes)


# ======================================================================================
# topology modifications
# ======================================================================================


class TestTopologyModifications:
    @GRID_SETTINGS
    @given(
        comp_type=st.sampled_from(["load", "generator", "storage_unit"]),
        bus_idx=st.integers(0, 141),
        p=st.floats(0.001, 2.0),
        repeat=st.integers(1, 3),
    )
    def test_added_components_are_unique_and_assigned_to_their_grid(
        self, wc_grid_factory, comp_type, bus_idx, p, repeat
    ):
        """
        Adding the same component k times (same id) yields k new unique names,
        raises the summed rating by exactly k·p, and every new component appears
        in the component table of the grid its bus belongs to; removing them
        again restores the component table.
        """
        topo = wc_grid_factory().topology
        bus = topo.buses_df.index[bus_idx]
        assume(not topo._check_bus_for_removal(bus))
        attr, col = {
            "load": ("loads_df", "p_set"),
            "generator": ("generators_df", "p_nom"),
            "storage_unit": ("storage_units_df", "p_nom"),
        }[comp_type]
        before = getattr(topo, attr).copy()
        names = []
        for _ in range(repeat):
            if comp_type == "load":
                names.append(topo.add_load(bus, p, sector="cts", load_id=7))
            elif comp_type == "generator":
                names.append(topo.add_generator(bus, p, "solar", generator_id=7))
            else:
                names.append(topo.add_storage_unit(bus, p))
        after = getattr(topo, attr)
        assert len(set(names)) == repeat
        assert after.index.is_unique and len(after) == len(before) + repeat
        np.testing.assert_allclose(
            after[col].sum(), before[col].sum() + repeat * p, rtol=1e-12
        )
        lv_id = topo.buses_df.at[bus, "lv_grid_id"]
        grid = topo.mv_grid if np.isnan(lv_id) else topo.get_lv_grid(int(lv_id))
        assert set(names) <= set(getattr(grid, attr).index)
        remove = getattr(topo, f"remove_{comp_type}")
        for name in names:
            remove(name)
        pd.testing.assert_frame_equal(getattr(topo, attr), before, check_names=False)

    @pytest.mark.parametrize("lv_grid_id", range(10))
    def test_aggregate_lv_grid_at_station_conserves_components(
        self, wc_grid, caplog, lv_grid_id
    ):
        """
        Aggregating an LV grid keeps all components with their ratings, moves
        every component of the grid to its station bus, leaves the station as
        the grid's only bus, does not touch any other grid, and keeps the
        topology connected and referentially consistent.
        """
        topo = wc_grid.topology
        assert lv_grid_id in topo._lv_grid_ids
        lv = topo.get_lv_grid(lv_grid_id)
        station = lv.station.index[0]
        lv_buses = set(lv.buses_df.index)
        comps_before = {
            k: getattr(topo, f"{k}_df").copy()
            for k in ("loads", "generators", "storage_units")
        }
        other_buses = topo.buses_df.drop(list(lv_buses))
        other_lines = topo.lines_df.drop(lv.lines_df.index)
        warnings_before = _new_warnings(caplog, topo)

        topo.aggregate_lv_grid_at_station(lv_grid_id)

        for key, before in comps_before.items():
            after = getattr(topo, f"{key}_df")
            assert sorted(after.index) == sorted(before.index)
            pd.testing.assert_frame_equal(
                after.drop(columns="bus").loc[before.index], before.drop(columns="bus")
            )
            moved = before.index[before.bus.isin(lv_buses)]
            assert (after.loc[moved, "bus"] == station).all()
            unchanged = before.index[~before.bus.isin(lv_buses)]
            pd.testing.assert_series_equal(
                after.loc[unchanged, "bus"], before.loc[unchanged, "bus"]
            )
        assert list(topo.get_lv_grid(lv_grid_id).buses_df.index) == [station]
        pd.testing.assert_frame_equal(topo.buses_df.drop(station), other_buses)
        pd.testing.assert_frame_equal(topo.lines_df, other_lines)
        assert nx.is_connected(topo.to_graph())
        assert undefined_references(topo) == []
        assert _new_warnings(caplog, topo, warnings_before) == []

    def test_check_integrity_reports_each_mutation(self, wc_grid_factory, caplog):
        """
        The imported test grid passes ``check_integrity`` (apart from its known
        very short line), and each single corruption of the topology is reported
        with the matching warning.
        """
        topo = wc_grid_factory().topology
        assert _new_warnings(caplog, topo) == []

        def mutate_missing_bus(t):
            t.loads_df.loc[t.loads_df.index[0], "bus"] = "unknown_bus"

        def mutate_line_bus(t):
            t.lines_df.loc[t.lines_df.index[0], "bus1"] = "unknown_bus"

        def mutate_impedance(t):
            t.lines_df.loc[t.lines_df.index[0], ["r", "x"]] = 0.0

        def mutate_isolated(t):
            t.add_bus("Bus_isolated", t.mv_grid.nominal_voltage)

        def mutate_duplicate(t):
            t.loads_df = pd.concat([t.loads_df, t.loads_df.iloc[:1]])

        def mutate_long_line(t):
            t.lines_df.loc[t.lines_df.index[0], "length"] = 11.0

        cases = [
            (mutate_missing_bus, "have buses which are not defined"),
            (mutate_line_bus, "have bus1 which are not defined"),
            (mutate_impedance, "Very small values for impedance"),
            (mutate_isolated, "buses are isolated"),
            (mutate_duplicate, "duplicate entry"),
            (mutate_long_line, "very large line lengths"),
        ]
        for mutate, expected in cases:
            topo = wc_grid_factory().topology
            mutate(topo)
            warnings = _new_warnings(caplog, topo)
            assert any(expected in w for w in warnings), (mutate.__name__, warnings)

    def test_topology_csv_roundtrip(self, wc_grid, tmp_path):
        """``Topology.to_csv`` -> ``from_csv`` restores all component tables."""
        wc_grid.topology.to_csv(tmp_path)
        loaded = EDisGo()
        loaded.topology.from_csv(tmp_path, loaded)
        for attr in (
            "buses_df",
            "lines_df",
            "transformers_df",
            "transformers_hvmv_df",
            "generators_df",
            "loads_df",
            "storage_units_df",
            "switches_df",
        ):
            pd.testing.assert_frame_equal(
                getattr(loaded.topology, attr),
                getattr(wc_grid.topology, attr),
                check_dtype=False,
                check_names=False,
                obj=attr,
            )
        assert (
            loaded.topology.grid_district["srid"]
            == (wc_grid.topology.grid_district["srid"])
        )


# ======================================================================================
# grids and components
# ======================================================================================


class TestGridsAndComponents:
    @pytest.mark.parametrize(
        "path", ["ding0_test_network_path", "ding0_test_network_2_path"]
    )
    def test_grids_partition_topology(self, path):
        """
        The MV grid and the LV grids partition buses, lines, loads, generators
        and storage units of the topology; LV grids have a nominal voltage below
        1 kV and the MV grid at least 1 kV; every grid's station is one of its
        buses and the secondary side of its transformers; LV grid graphs are
        trees on exactly the grid's buses.
        """
        topo = EDisGo(ding0_grid=getattr(pytest, path)).topology
        grids = list(topo.grids)
        for attr in ("buses_df", "lines_df", "loads_df", "generators_df"):
            names = np.concatenate([getattr(g, attr).index.to_numpy() for g in grids])
            assert len(names) == len(set(names)), attr
            assert set(names) == set(getattr(topo, attr).index), attr
        storages = np.concatenate([g.storage_units_df.index for g in grids])
        assert sorted(storages) == sorted(topo.storage_units_df.index)
        assert topo.mv_grid.nominal_voltage >= 1
        for grid in grids:
            station = grid.station.index[0]
            assert station in grid.buses_df.index
            assert (grid.transformers_df.bus1 == station).all()
            assert (grid.buses_df.v_nom == grid.nominal_voltage).all()
        for grid in topo.lv_grids:
            assert grid.nominal_voltage < 1
            graph = grid.graph
            assert set(graph.nodes) == set(grid.buses_df.index)
            assert_radial(graph, str(grid))

    def test_capacity_and_demand_sums(self, wc_grid):
        """
        Per grid, generation capacity per technology and demand per sector sum
        to the grid totals, and the grid totals sum to the topology totals.
        """
        topo = wc_grid.topology
        grids = list(topo.grids)
        for grid in grids:
            np.testing.assert_allclose(
                grid.peak_generation_capacity_per_technology.sum(),
                grid.peak_generation_capacity,
                rtol=1e-12,
            )
            if grid.loads_df.sector.notna().all():
                np.testing.assert_allclose(
                    grid.p_set_per_sector.sum(), grid.p_set, rtol=1e-12
                )
        np.testing.assert_allclose(
            sum(g.peak_generation_capacity for g in grids),
            topo.generators_df.p_nom.sum(),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            sum(g.p_set for g in grids), topo.loads_df.p_set.sum(), rtol=1e-12
        )

    def test_feeders_and_lengths_to_station(self, wc_grid):
        """
        Per grid: the feeders partition the non-station buses and their number
        equals the station's degree; both buses of a line lie in the line's
        feeder; the length to the station is 0 at the station and differs by
        the line length along every line; a feeder's length is at least each of
        its lines and at most their sum. With MV feeders, all buses of an LV grid
        lie in one feeder, the MV feeder of its station's MV bus.
        """
        topo = wc_grid.topology
        topo.assign_feeders(mode="grid_feeder")
        for grid in topo.grids:
            station = grid.station.index[0]
            buses = topo.buses_df.loc[grid.buses_df.index]
            lines = topo.lines_df.loc[grid.lines_df.index]
            graph = grid.graph
            assert buses.at[station, "grid_feeder"] == "station_node"
            feeders = buses.grid_feeder.drop(station)
            assert feeders.notna().all()
            assert set(feeders) == set(graph.neighbors(station))
            assert lines.grid_feeder.notna().all()
            for _, line in lines.iterrows():
                for bus in (line.bus0, line.bus1):
                    if bus != station:
                        assert buses.at[bus, "grid_feeder"] == line.grid_feeder

            stats = grid.get_feeder_stats()
            length = topo.buses_df.loc[grid.buses_df.index, "length_to_grid_station"]
            assert length[station] == 0
            np.testing.assert_allclose(
                np.abs(
                    length.loc[lines.bus1].to_numpy()
                    - length.loc[lines.bus0].to_numpy()
                ),
                lines.length,
                rtol=1e-9,
            )
            assert len(stats) == graph.degree(station)
            for feeder, row in stats.iterrows():
                feeder_lines = lines[lines.grid_feeder == feeder]
                assert row.length >= feeder_lines.length.max() * (1 - 1e-12)
                assert row.length <= feeder_lines.length.sum() * (1 + 1e-12)

        topo.assign_feeders(mode="mv_feeder")
        mv_station = topo.mv_grid.station.index[0]
        for grid in topo.lv_grids:
            feeders = topo.buses_df.loc[grid.buses_df.index, "mv_feeder"]
            assert feeders.nunique() == 1, str(grid)
            mv_bus = grid.transformers_df.bus0.iloc[0]
            # an LV grid at the HV/MV station itself is a feeder of its own
            if mv_bus != mv_station:
                assert feeders.iloc[0] == topo.buses_df.at[mv_bus, "mv_feeder"]

    def test_component_objects_match_topology_and_time_series(self, wc_grid):
        """
        For every load, generator and storage unit, the component object's
        voltage level follows its bus voltage, it is listed in its grid's table,
        its time series equal the TimeSeries columns, and moving it to another
        bus and back restores the topology.
        """
        topo, ts = wc_grid.topology, wc_grid.timeseries
        for cls, attr, ts_attr in (
            (Load, "loads_df", "loads"),
            (Generator, "generators_df", "generators"),
            (Storage, "storage_units_df", "storage_units"),
        ):
            for name in getattr(topo, attr).index:
                comp = cls(id=name, edisgo_obj=wc_grid)
                v_nom = topo.buses_df.at[comp.bus, "v_nom"]
                assert comp.voltage_level == ("lv" if v_nom < 1 else "mv")
                assert name in getattr(comp.grid, attr).index
                pd.testing.assert_series_equal(
                    comp.active_power_timeseries,
                    getattr(ts, f"{ts_attr}_active_power")[name],
                )
                pd.testing.assert_series_equal(
                    comp.reactive_power_timeseries,
                    getattr(ts, f"{ts_attr}_reactive_power")[name],
                )
        before = topo.loads_df.copy()
        load = Load(id=before.index[0], edisgo_obj=wc_grid)
        old_bus = load.bus
        load.bus = topo.lines_df.bus1.iloc[5]
        assert topo.loads_df.at[load.id, "bus"] != old_bus
        load.bus = old_bus
        pd.testing.assert_frame_equal(topo.loads_df, before)


# ======================================================================================
# grid connection of new components
# ======================================================================================


def _mv_point(topo, fx, fy):
    mv = topo.mv_grid.buses_df
    return Point(
        mv.x.min() + fx * (mv.x.max() - mv.x.min()),
        mv.y.min() + fy * (mv.y.max() - mv.y.min()),
    )


def _lv_station_point(topo, lv_grid_id, dx=0.0, dy=0.0):
    station = topo.get_lv_grid(lv_grid_id).station.iloc[0]
    return Point(station.x + dx, station.y + dy)


class TestGridConnection:
    @GRID_SETTINGS
    @given(
        comp_type=st.sampled_from(
            ["generator", "charging_point", "heat_pump", "storage_unit"]
        ),
        voltage_level=st.sampled_from([4, 5]),
        p=st.floats(0.05, 5.0),
        fx=st.floats(0, 1),
        fy=st.floats(0, 1),
    )
    def test_connect_to_mv_keeps_grid_valid(
        self, wc_grid_factory, caplog, comp_type, voltage_level, p, fx, fy
    ):
        """
        After ``connect_to_mv`` the grid is connected and radial, the component
        sits at a new MV bus with the MV nominal voltage, new lines are at least
        1 m long, split line segments keep the type of the split line, switch
        branches still exist and contain one of the switch's buses, and no new
        integrity warning appears.
        """
        edisgo = wc_grid_factory()
        topo = edisgo.topology
        warnings_before = _new_warnings(caplog, topo)
        lines_before = topo.lines_df.copy()
        buses_before = topo.buses_df.index
        data = _comp_data(comp_type, p, _mv_point(topo, fx, fy), voltage_level, 4711)
        name = topo.connect_to_mv(edisgo, data, comp_type=comp_type)

        df = _comp_df(topo, comp_type)
        assert _comp_power(df, name) == p
        bus = df.at[name, "bus"]
        assert bus not in buses_before
        assert topo.buses_df.at[bus, "v_nom"] == topo.mv_grid.nominal_voltage
        assert bus in topo.mv_grid.buses_df.index
        assert_radial(topo.to_graph(), "grid after connect_to_mv")
        new_lines = topo.lines_df.drop(lines_before.index, errors="ignore")
        assert (new_lines.length >= 0.001).all()
        removed = lines_before.index.difference(topo.lines_df.index)
        assert len(removed) <= 1
        for old in removed:
            segments = new_lines[new_lines.index.str.contains(old)]
            segments = segments[~segments.isin([bus]).any(axis=1)]
            assert len(segments) == 2
            for col in ("type_info", "num_parallel", "kind"):
                assert (segments[col] == lines_before.at[old, col]).all()
        for _, sw in topo.switches_df.iterrows():
            ends = topo.lines_df.loc[sw.branch, ["bus0", "bus1"]].values
            assert sw.bus_open in ends or sw.bus_closed in ends
        assert undefined_references(topo) == []
        # random points in the bounding box are up to 12 km from the station;
        # the plausibility warning for lines > 10 km is expected there
        long_lines = ("very large line lengths",)
        assert _new_warnings(caplog, topo, warnings_before, long_lines) == []

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: when connect_to_mv splits an existing line, both "
        "segments of the old line are recorded as 'added' in equipment_changes "
        "and are therefore costed like newly built cables",
    )
    def test_connect_to_mv_records_only_new_cable(self, wc_grid):
        """
        A line split only adds a branch tee: ``equipment_changes`` lists only
        the cable to the new component as added equipment.
        """
        topo = wc_grid.topology
        # just beside the middle of MV line Line_10014, so that it is split
        ends = topo.buses_df.loc[topo.lines_df.loc["Line_10014", ["bus0", "bus1"]]]
        x, y = ends.x.mean() + 1e-4, ends.y.mean() + 1e-4
        lines_before = topo.lines_df.index
        name = topo.connect_to_mv(
            wc_grid, _comp_data("generator", 2.5, Point(x, y), 5, 123456)
        )
        removed = lines_before.difference(topo.lines_df.index)
        assert list(removed) == ["Line_10014"], "precondition: line is split"
        bus = topo.generators_df.at[name, "bus"]
        new_cable = topo.get_connected_lines_from_bus(bus).index
        changes = wc_grid.results.equipment_changes
        assert sorted(changes.index) == sorted(new_cable)

    @pytest.mark.parametrize("level", ["mv", "lv"])
    @pytest.mark.xfail(
        strict=True,
        reason="Bug: connect_to_mv/_connect_to_lv_bus pass the active power as "
        "apparent_power to select_cable, so the cable is sized for S = P and "
        "overloaded at the configured power factor",
    )
    def test_connection_cable_carries_apparent_power(self, wc_grid, level):
        """
        The cable of a newly connected generator can carry its apparent power
        at the configured power factor, ``s_nom >= P / cos φ``. P is chosen just
        below the rating of the smallest suitable cable, the location next to
        the station so that the voltage criterion does not apply.
        """
        topo = wc_grid.topology
        pf = wc_grid.config["reactive_power_factor"][f"{level}_generator"]
        lines_before = topo.lines_df.index
        if level == "mv":
            # smallest 20 kV cable: 11.05 MVA
            p = 10.5
            station = topo.mv_grid.station.iloc[0]
            geom = Point(station.x + 0.0005, station.y)
            name = topo.connect_to_mv(wc_grid, _comp_data("generator", p, geom, 4, 1))
        else:
            # smallest LV cable NAYY 4x1x35: 0.0852 MVA
            p = 0.084
            geom = _lv_station_point(topo, 3, dx=0.0001)
            data = _comp_data("generator", p, geom, 6, 1, mvlv_subst_id=3)
            name = topo.connect_to_lv(wc_grid, data)
        bus = topo.generators_df.at[name, "bus"]
        new = topo.lines_df.drop(lines_before)
        cable = new[(new.bus0 == bus) | (new.bus1 == bus)].iloc[0]
        assert cable.s_nom >= p / pf

    @pytest.mark.parametrize(
        "level, voltage_level", [("mv", 4), ("mv", 5), ("lv", 6), ("lv", 7)]
    )
    def test_connected_generator_shifts_slack_power(
        self, wc_grid, analyzed_grid, level, voltage_level
    ):
        """
        A connected generator with worst-case time series takes part in the
        power flow: the power flow converges and the slack power changes by
        exactly ``-P_new`` plus the change in grid losses.
        """
        topo = wc_grid.topology
        p = 0.5 if level == "mv" else 0.02
        if level == "mv":
            data = _comp_data(
                "generator", p, _mv_point(topo, 0.6, 0.4), voltage_level, 9
            )
            name = topo.connect_to_mv(wc_grid, data)
        else:
            geom = _lv_station_point(topo, 3, dx=0.0002)
            data = _comp_data("generator", p, geom, voltage_level, 9, mvlv_subst_id=3)
            name = topo.connect_to_lv(wc_grid, data)
        wc_grid.timeseries.set_worst_case(
            wc_grid, cases=["feed-in_case", "load_case"], generators_names=[name]
        )
        wc_grid.analyze()
        p_new = wc_grid.timeseries.generators_active_power[name]
        before, after = analyzed_grid.results, wc_grid.results
        d_slack = after.pfa_slack.p - before.pfa_slack.p
        d_losses = after.grid_losses.p - before.grid_losses.p
        # each power balance holds to POWER_ATOL, their difference to twice that
        np.testing.assert_allclose(d_slack, -p_new + d_losses, atol=2 * POWER_ATOL)
        assert topo.generators_df.at[name, "bus"] in after.v_res.columns

    @GRID_SETTINGS
    @given(
        comp=st.sampled_from(
            [
                ("generator", None),
                ("storage_unit", None),
                ("charging_point", "home"),
                ("charging_point", "work"),
                ("charging_point", "public"),
                ("heat_pump", "individual_heating"),
                ("heat_pump", "district_heating"),
            ]
        ),
        voltage_level=st.sampled_from([6, 7]),
        lv_grid_id=st.sampled_from([1, 2, 3, 5, 6, 7, 8]),
        p=st.floats(0.001, 0.08),
        dx=st.floats(-0.0003, 0.0003),
    )
    def test_connect_to_lv_keeps_grid_valid(
        self, wc_grid_factory, caplog, comp, voltage_level, lv_grid_id, p, dx
    ):
        """
        ``connect_to_lv`` with a known LV grid connects the component to an LV
        bus of exactly that grid, keeps the grid connected and radial, causes no
        new integrity warning and is deterministic (same input, same bus).
        """
        comp_type, sector = comp
        buses = []
        for _ in range(2):
            edisgo = wc_grid_factory()
            topo = edisgo.topology
            warnings_before = _new_warnings(caplog, topo)
            geom = _lv_station_point(topo, lv_grid_id, dx=dx, dy=0.0001)
            extra = {"mvlv_subst_id": lv_grid_id}
            if sector is not None:
                extra["sector"] = sector
            data = _comp_data(comp_type, p, geom, voltage_level, 4712, **extra)
            name = topo.connect_to_lv(edisgo, data, comp_type=comp_type)
            bus = _comp_df(topo, comp_type).at[name, "bus"]
            assert topo.buses_df.at[bus, "v_nom"] < 1
            assert topo.buses_df.at[bus, "lv_grid_id"] == lv_grid_id
            assert_radial(topo.to_graph(), "grid after connect_to_lv")
            assert undefined_references(topo) == []
            assert _new_warnings(caplog, topo, warnings_before) == []
            buses.append(bus)
        assert buses[0] == buses[1]

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: connect_to_lv lets a bus that already has "
        "allowed_number_of_comp_per_bus components receive one more (<=), while "
        "its docstring says 'at most allowed'; TestConnectToLVBasedOnGeolocation "
        "pins the <= behaviour",
    )
    def test_connect_to_lv_respects_allowed_number_of_components(self, wc_grid):
        """
        No bus with a matching load receives more than
        ``allowed_number_of_comp_per_bus`` new generators of the same type; once
        all are full, further generators get their own bus at the station.
        """
        topo = wc_grid.topology
        allowed = 1
        geom = _lv_station_point(topo, 3, dx=0.0001)
        names = [
            topo.connect_to_lv(
                wc_grid,
                _comp_data("generator", 0.01, geom, 7, 600 + i, mvlv_subst_id=3),
                allowed_number_of_comp_per_bus=allowed,
            )
            for i in range(8)
        ]
        lv = topo.get_lv_grid(3)
        targets = set(lv.loads_df[lv.loads_df.sector == "residential"].bus)
        counts = topo.generators_df.loc[names].bus.value_counts()
        assert (counts[counts.index.isin(targets)] <= allowed).all(), counts

    @pytest.mark.xfail(
        strict=True,
        reason="Klaerung: connect_to_lv connects a component with an unknown "
        "mvlv_subst_id (voltage level 6/7) directly to the HV/MV station bus "
        "(documented interim solution, ToDo in code), i.e. to the MV level",
    )
    def test_connect_to_lv_with_unknown_substation_stays_in_lv(self, wc_grid):
        """An LV component always ends up at a bus with ``v_nom < 1``."""
        topo = wc_grid.topology
        data = _comp_data("generator", 0.01, None, 7, 1, mvlv_subst_id=12345)
        data.pop("geom")
        name = topo.connect_to_lv(wc_grid, data)
        assert topo.buses_df.at[topo.generators_df.at[name, "bus"], "v_nom"] < 1

    @pytest.mark.xfail(
        strict=True,
        raises=KeyError,
        reason="Bug: connect_to_lv (voltage level 7) falls back to a new bus at "
        "the station once all suitable buses are full and then needs comp_data"
        "['geom'], which is documented as optional -> KeyError",
    )
    def test_connect_to_lv_without_geometry(self, wc_grid):
        """
        Components without geometry can always be connected in voltage level 7,
        also when all buses with a matching load are full.
        """
        topo = wc_grid.topology
        for i in range(8):
            data = _comp_data("generator", 0.01, None, 7, 700 + i, mvlv_subst_id=3)
            data.pop("geom")
            topo.connect_to_lv(wc_grid, data, allowed_number_of_comp_per_bus=0)

    @settings(
        deadline=None,
        max_examples=examples(10),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        voltage_level=st.sampled_from([6, 7]),
        fx=st.floats(0, 1),
        fy=st.floats(0, 1),
        max_distance=st.sampled_from([0.0, 0.02, 0.1, 0.5]),
        comp=st.sampled_from([("generator", None), ("charging_point", "public")]),
    )
    def test_connect_to_lv_based_on_geolocation(
        self, grid_3, voltage_level, fx, fy, max_distance, comp
    ):
        """
        The component is connected to the nearest MV/LV station (level 6) or LV
        bus (level 7): directly if it is at most ``max_distance`` away, otherwise
        via exactly one new bus and one new line to that nearest bus. The grid
        stays connected and radial. Nearest is checked against exact geodesic
        distances with 0.5 % + 1 m slack for the documented approximation.
        """
        edisgo = copy.deepcopy(grid_3)
        topo = edisgo.topology
        lv_buses = topo.buses_df.drop(topo.mv_grid.buses_df.index)
        point = Point(
            lv_buses.x.min() + fx * (lv_buses.x.max() - lv_buses.x.min()),
            lv_buses.y.min() + fy * (lv_buses.y.max() - lv_buses.y.min()),
        )
        if voltage_level == 6:
            candidates = topo.buses_df.loc[topo.transformers_df.bus1.unique()]
        else:
            candidates = lv_buses
        dist = pd.Series(
            [
                geodesic((point.y, point.x), (y, x)).km
                for x, y in candidates[["x", "y"]].values
            ],
            index=candidates.index,
        )
        d_min = dist.min()
        assume(abs(d_min - max_distance) > 0.005 * d_min + 0.001)
        buses_before = topo.buses_df.index
        lines_before = topo.lines_df.index
        comp_type, sector = comp
        extra = {} if sector is None else {"sector": sector}
        data = _comp_data(comp_type, 0.01, point, voltage_level, 4713, **extra)
        name = topo.connect_to_lv_based_on_geolocation(
            edisgo, data, comp_type, max_distance_from_target_bus=max_distance
        )
        bus = _comp_df(topo, comp_type).at[name, "bus"]
        new_buses = topo.buses_df.index.difference(buses_before)
        new_lines = topo.lines_df.drop(lines_before)
        if d_min <= max_distance:
            assert len(new_buses) == 0 and len(new_lines) == 0
            target = bus
        else:
            assert list(new_buses) == [bus] and len(new_lines) == 1
            line = new_lines.iloc[0]
            target = line.bus0 if line.bus1 == bus else line.bus1
        assert dist[target] <= d_min * 1.005 + 0.001
        assert topo.buses_df.at[bus, "v_nom"] < 1
        assert_radial(topo.to_graph(), "grid after connect_to_lv_based_on_geolocation")


# silence the very verbose import warnings of the test grids in this module
logging.getLogger("pypsa").setLevel(logging.ERROR)
