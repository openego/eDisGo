import copy

from contextlib import nullcontext as does_not_raise

import pytest

from edisgo import EDisGo
from edisgo.tools.pseudo_coordinates import make_pseudo_coordinates
from edisgo.tools.spatial_complexity_reduction import (
    hash_df,
    make_busmap,
    make_grid_list,
    reduce_edisgo,
    spatial_complexity_reduction,
)


class TestSpatialComplexityReduction:
    @pytest.fixture(scope="class")
    def test_edisgo_obj(self):
        """EDisGo-object is set up only once, during class lifetime."""
        edisgo_root = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        edisgo_root.set_time_series_worst_case_analysis()
        edisgo_root = make_pseudo_coordinates(edisgo_root)
        return edisgo_root

    @pytest.fixture(scope="class")
    def test_busmap_df(self, test_edisgo_obj):
        busmap_df = make_busmap(
            test_edisgo_obj,
            mode="kmeansdijkstra",
            cluster_area="main_feeder",
            reduction_factor=0.25,
            reduction_factor_not_focused=False,
        )
        return busmap_df

    @pytest.mark.parametrize(
        "mode,cluster_area,"
        "reduction_factor,"
        "reduction_factor_not_focused,"
        "test_exception,"
        "n_new_buses",
        [
            # Cluster area: 'grid'
            (
                "kmeans",
                "grid",
                0.1,
                False,
                does_not_raise(),
                19,
            ),
            (
                "kmeansdijkstra",
                "grid",
                0.1,
                False,
                does_not_raise(),
                19,
            ),
            (
                "kmeans",
                "grid",
                0.5,
                0,
                does_not_raise(),
                76,
            ),
            (
                "kmeans",
                "grid",
                0.5,
                0.1,
                does_not_raise(),
                76,
            ),
            # Cluster area: 'feeder'
            (
                "kmeans",
                "feeder",
                0.1,
                False,
                does_not_raise(),
                40,
            ),
            (
                "kmeansdijkstra",
                "feeder",
                0.1,
                False,
                does_not_raise(),
                39,
            ),
            (
                "kmeans",
                "feeder",
                0.5,
                0,
                does_not_raise(),
                23,
            ),
            (
                "kmeans",
                "feeder",
                0.5,
                0.1,
                does_not_raise(),
                46,
            ),
            # Cluster area: 'main_feeder'
            (
                "kmeans",
                "main_feeder",
                0.1,
                False,
                does_not_raise(),
                36,
            ),
            (
                "kmeansdijkstra",
                "main_feeder",
                0.1,
                False,
                does_not_raise(),
                36,
            ),
            (
                "aggregate_to_main_feeder",
                "main_feeder",
                None,
                False,
                does_not_raise(),
                105,
            ),
            (
                "equidistant_nodes",
                "main_feeder",
                0.1,
                False,
                does_not_raise(),
                36,
            ),
            (
                "kmeans",
                "main_feeder",
                0.5,
                0,
                does_not_raise(),
                20,
            ),
            (
                "kmeans",
                "main_feeder",
                0.5,
                0.1,
                does_not_raise(),
                41,
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
        test_edisgo_obj,
        mode,
        cluster_area,
        reduction_factor,
        reduction_factor_not_focused,
        test_exception,
        n_new_buses,
    ):
        edisgo_root = copy.deepcopy(test_edisgo_obj)

        with test_exception:
            busmap_df = make_busmap(
                edisgo_root,
                mode=mode,
                cluster_area=cluster_area,
                reduction_factor=reduction_factor,
                reduction_factor_not_focused=reduction_factor_not_focused,
            )
            # Check for deterministic behaviour.
            assert len(set(busmap_df["new_bus"].to_list())) == n_new_buses

    @pytest.mark.parametrize(
        "cluster_area,grid,expected_hash",
        [
            # Cluster area: 'grid'
            ("grid", "MVGrid", "f7c55d5a0933816ce2ab5f439c8193fe"),
            ("grid", "LVGrid", "fe5dd55a9bb4ed151c06841347cbc869"),
            # Cluster area: 'feeder'
            ("feeder", "MVGrid", "d23665844d28241cca314f5d4045157d"),
            ("feeder", "LVGrid", "f84068fe78e5ffeb2ffdce42e9f8762b"),
            # Cluster area: 'main_feeder'
            ("main_feeder", "MVGrid", "56913cc22a534f5f8b150b42f389957e"),
            ("main_feeder", "LVGrid", "9ce503e790b71ded6dbd30691580b646"),
        ],
    )
    def test_make_busmap_for_only_one_grid(
        self,
        test_edisgo_obj,
        cluster_area,
        grid,
        expected_hash,
    ):
        edisgo_root = copy.deepcopy(test_edisgo_obj)

        if grid == "MVGrid":
            grid = make_grid_list(edisgo_root, grid="MVGrid_1")[0]
        elif grid == "LVGrid":
            grid = make_grid_list(edisgo_root, grid="LVGrid_9")[0]

        busmap_df = make_busmap(
            edisgo_root,
            mode="kmeans",
            grid=grid,
            cluster_area=cluster_area,
            reduction_factor=0.2,
        )
        # Check for deterministic behaviour.
        assert hash_df(busmap_df) == expected_hash

    @pytest.mark.parametrize(
        "line_naming_convention,"
        "aggregation_mode,"
        "load_aggregation_mode, "
        "generator_aggregation_mode, "
        "n_loads, "
        "n_generators,",
        [
            ("standard_lines", True, "bus", "bus", 27, 17),
            ("standard_lines", True, "sector", "type", 28, 18),
            ("combined_name", False, None, None, 50, 28),
        ],
    )
    def test_reduce_edisgo(
        self,
        test_edisgo_obj,
        test_busmap_df,
        line_naming_convention,
        aggregation_mode,
        load_aggregation_mode,
        generator_aggregation_mode,
        n_loads,
        n_generators,
    ):
        edisgo_root = copy.deepcopy(test_edisgo_obj)
        busmap_df = copy.deepcopy(test_busmap_df)

        # Add second line to test line reduction
        edisgo_root.topology.lines_df.loc[
            "Line_10003_2"
        ] = edisgo_root.topology.lines_df.loc["Line_10003"]

        assert edisgo_root.topology.buses_df.shape[0] == 142
        assert edisgo_root.topology.lines_df.shape[0] == 132
        assert edisgo_root.topology.loads_df.shape[0] == 50
        assert edisgo_root.topology.generators_df.shape[0] == 28
        assert edisgo_root.topology.storage_units_df.shape[0] == 1
        assert edisgo_root.topology.transformers_df.shape[0] == 14
        assert edisgo_root.topology.switches_df.shape[0] == 2

        edisgo_reduced, linemap_df = reduce_edisgo(
            edisgo_root,
            busmap_df,
            line_naming_convention=line_naming_convention,
            aggregation_mode=aggregation_mode,
            load_aggregation_mode=load_aggregation_mode,
            generator_aggregation_mode=generator_aggregation_mode,
        )

        assert edisgo_reduced.topology.buses_df.shape[0] == 43
        assert edisgo_reduced.topology.lines_df.shape[0] == 34
        assert edisgo_reduced.topology.loads_df.shape[0] == n_loads
        assert edisgo_reduced.topology.generators_df.shape[0] == n_generators
        assert edisgo_reduced.topology.storage_units_df.shape[0] == 1
        assert edisgo_reduced.topology.transformers_df.shape[0] == 14
        assert edisgo_reduced.topology.switches_df.shape[0] == 2

        if line_naming_convention == "standard_lines":
            assert (
                edisgo_reduced.topology.lines_df.loc[
                    "Line_Bus_MVStation_1_to_Bus_mvgd_1_F0_B2", "type_info"
                ]
                == "NA2XS2Y 3x1x240"
            )
        elif line_naming_convention == "combined_name":
            assert (
                edisgo_reduced.topology.lines_df.loc[
                    "Line_Bus_MVStation_1_to_Bus_mvgd_1_F0_B2", "type_info"
                ]
                == "Merged: 48-AL1/8-ST1A 48-AL1/8-ST1A "
            )
        timeseries = edisgo_reduced.timeseries
        assert timeseries.loads_active_power.shape[1] == n_loads
        assert timeseries.loads_reactive_power.shape[1] == n_loads
        assert timeseries.generators_active_power.shape[1] == n_generators
        assert timeseries.generators_reactive_power.shape[1] == n_generators

        # Check for deterministic behaviour.
        assert hash_df(linemap_df) == "e6e50f9042722398e27488b22c9848df"

    def test_spatial_complexity_reduction(self, test_edisgo_obj):
        edisgo_root = copy.deepcopy(test_edisgo_obj)

        edisgo_reduced, busmap_df, linemap_df = spatial_complexity_reduction(
            edisgo_root,
            mode="kmeans",
            cluster_area="grid",
            reduction_factor=0.2,
            reduction_factor_not_focused=False,
        )
        # Check for deterministic behaviour.
        assert hash_df(busmap_df) == "ce1cf807409fe5e0e9abe3123a18791a"

        # Check that edisgo_object can run power flow and reinforce
        edisgo_reduced.analyze()
        edisgo_reduced.reinforce()
