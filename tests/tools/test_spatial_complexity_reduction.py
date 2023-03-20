import copy

from contextlib import nullcontext as does_not_raise
from hashlib import md5

import pytest

from edisgo import EDisGo
from edisgo.tools.pseudo_coordinates import make_pseudo_coordinates
from edisgo.tools.spatial_complexity_reduction import make_busmap


def hash_df(df):
    s = df.to_json()
    return md5(s.encode()).hexdigest()


class TestSpatialComplexityReduction:
    @pytest.fixture(scope="class")
    def test_grid(self):
        """EDisGo-object is set up only once."""
        edisgo_root = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        edisgo_root.set_time_series_worst_case_analysis()
        edisgo_root = make_pseudo_coordinates(edisgo_root)
        return edisgo_root

    @pytest.fixture
    def test_grid_copy(self, test_grid):
        return copy.deepcopy(test_grid)

    @pytest.mark.parametrize(
        "mode,cluster_area,reduction_factor,reduction_factor_not_focused,"
        "test_exception,expected_remaining_nodes",
        [
            # Cluster area: 'grid'
            (
                "kmeans",
                "grid",
                0.1,
                False,
                does_not_raise(),
                "a840aec08914448c907a482834094d34",
            ),
            (
                "kmeansdijkstra",
                "grid",
                0.1,
                False,
                does_not_raise(),
                "cd0a4ce9ca72e55bfe7353ed32d5af52",
            ),
            (
                "kmeans",
                "grid",
                0.5,
                0,
                does_not_raise(),
                "3f4b25a25f5ca1c12620e92d855dae0d",
            ),
            (
                "kmeans",
                "grid",
                0.5,
                0.1,
                does_not_raise(),
                "3f4b25a25f5ca1c12620e92d855dae0d",
            ),
            # Cluster area: 'feeder'
            (
                "kmeans",
                "feeder",
                0.1,
                False,
                does_not_raise(),
                "f0126014e807b2ad6776eee6d458cdc1",
            ),
            (
                "kmeansdijkstra",
                "feeder",
                0.1,
                False,
                does_not_raise(),
                "9bcb23df6884cd2b6828676e7d67c525",
            ),
            (
                "kmeans",
                "feeder",
                0.5,
                0,
                does_not_raise(),
                "02b909b963330d31a8aeb14a23af4291",
            ),
            (
                "kmeans",
                "feeder",
                0.5,
                0.1,
                does_not_raise(),
                "193713ca9137f68e8eb93f0e369a21dc",
            ),
            # Cluster area: 'main_feeder'
            (
                "kmeans",
                "main_feeder",
                0.1,
                False,
                does_not_raise(),
                "fadcdd5531d6f846c3ece76669fecedf",
            ),
            (
                "kmeansdijkstra",
                "main_feeder",
                0.1,
                False,
                does_not_raise(),
                "2f45064275a601a5f4489104521b3d58",
            ),
            (
                "aggregate_to_main_feeder",
                "main_feeder",
                0.1,
                False,
                does_not_raise(),
                "2dd5b8b7ac6f18765fc5e4ccbf330681",
            ),
            (
                "equidistant_nodes",
                "main_feeder",
                0.1,
                False,
                does_not_raise(),
                "3b2c4de8fabe724d551b11b86bffee90",
            ),
            (
                "kmeans",
                "main_feeder",
                0.5,
                0,
                does_not_raise(),
                "c1e684b0cb671cf2d69de2e765fe5117",
            ),
            (
                "kmeans",
                "main_feeder",
                0.5,
                0.1,
                does_not_raise(),
                "3d0e7afcb3b9c8e5c9d0838d68113bf3",
            ),
            # Test raising exceptions
            ("kmeans", "grid", 0, False, pytest.raises(ValueError), None),
            ("kmeans", "grid", 1, False, pytest.raises(ValueError), None),
            ("kmeans", "grid", 0.1, 1, pytest.raises(ValueError), None),
            ("MODE", "grid", 0.1, False, pytest.raises(ValueError), None),
            ("kmeans", "CLUSTER_AREA", 0.1, False, pytest.raises(ValueError), None),
        ],
    )
    def test_make_busmap(
        self,
        test_grid_copy,
        mode,
        cluster_area,
        reduction_factor,
        reduction_factor_not_focused,
        test_exception,
        expected_remaining_nodes,
    ):
        edisgo_root = test_grid_copy
        with test_exception:
            busmap_df = make_busmap(
                edisgo_root,
                mode=mode,
                cluster_area=cluster_area,
                reduction_factor=reduction_factor,
                reduction_factor_not_focused=reduction_factor_not_focused,
            )
            # Check that results stay always the same, deterministic behaviour
            assert hash_df(busmap_df) == expected_remaining_nodes
        print("THE END")
