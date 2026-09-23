import networkx as nx
import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.tools.pseudo_coordinates import (
    _make_coordinates,
    make_pseudo_coordinates,
)


class TestPseudoCoordinates:
    @classmethod
    def setup_class(cls):
        cls.edisgo_root = EDisGo(ding0_grid=pytest.ding0_test_network_path)

    def test_make_pseudo_coordinates(self):
        # test coordinates before
        coordinates = self.edisgo_root.topology.buses_df.loc[
            "Bus_BranchTee_LVGrid_1_9", ["x", "y"]
        ]
        assert round(coordinates[0], 5) != round(7.943307, 5)
        assert round(coordinates[1], 5) != round(48.080396, 5)

        # make pseudo coordinates
        make_pseudo_coordinates(self.edisgo_root, mv_coordinates=True)

        # test if the right coordinates are set for one node
        coordinates = self.edisgo_root.topology.buses_df.loc[
            "Bus_BranchTee_LVGrid_1_9", ["x", "y"]
        ]
        assert round(coordinates[0], 5) == round(7.943307, 5)
        assert round(coordinates[1], 5) == round(48.080396, 5)

        assert not self.edisgo_root.topology.buses_df.x.isin([np.NaN]).any()

    def test__make_coordinates_meshed_graph(self):
        """
        A ring in the graph must not abort the coordinate generation.

        The traversal assumes a tree, where each node is reached from exactly
        one neighbour. On a ring a node is discovered from both sides and was
        appended to the queue twice; by the time the second entry was processed
        the node had been removed from the working copy, and ``nx.neighbors``
        raised ``NetworkXError: The node ... is not in the graph``
        (openego/eGo#227, seen on MV grid 32064, whose ``LVGrid_8829100000``
        holds 41 buses on 41 lines).

        The graph below has the same shape: a ring (``a``-``b``-``c``) with a
        tail behind it. The tail matters — without buses left to lay out after
        the duplicate surfaces, the loop ends before it is reached and the bug
        stays hidden.
        """
        graph = nx.Graph()
        for node in ("station", "a", "b", "c", "d", "e"):
            graph.add_node(node, pos=(0, 0))
        for bus0, bus1 in (
            ("station", "a"),
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),  # closes the ring
            ("b", "d"),  # tail behind the ring
            ("d", "e"),
        ):
            graph.add_edge(bus0, bus1, length=0.1)
        assert not nx.is_tree(graph)

        # used to raise NetworkXError: The node c is not in the graph.
        result = _make_coordinates(graph, branch_detour_factor=1.3)

        # every bus got a usable position
        assert set(result.nodes) == {"station", "a", "b", "c", "d", "e"}
        for node in result.nodes:
            assert "pos" in result.nodes[node]
            assert not any(np.isnan(result.nodes[node]["pos"]))

    def test__make_coordinates_tree_graph_is_unchanged(self):
        """
        The fix for the meshed case must not move a tree-shaped grid.

        Guards the skip introduced for rings: in a tree no node is ever
        discovered twice, so nothing may be skipped and the coordinates have to
        come out exactly as before.
        """
        graph = nx.Graph()
        for node in ("station", "a", "b", "c"):
            graph.add_node(node, pos=(0, 0))
        graph.add_edge("station", "a", length=0.1)
        graph.add_edge("a", "b", length=0.2)
        graph.add_edge("a", "c", length=0.3)
        assert nx.is_tree(graph)

        result = _make_coordinates(graph, branch_detour_factor=1.3)

        # Pinned from the implementation before the ring fix, so a change in the
        # traversal that moves a tree-shaped grid fails here.
        expected = {
            "station": (0, 0),
            "a": (76.92307692307693, -1.8840719986882358e-14),
            "b": (230.7692307692308, -5.652215996064707e-14),
            "c": (76.92307692307689, -230.76923076923077),
        }
        for node, (x, y) in expected.items():
            assert result.nodes[node]["pos"] == pytest.approx((x, y), abs=1e-9)
