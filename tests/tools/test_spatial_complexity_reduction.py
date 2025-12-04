import copy

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.tools import spatial_complexity_reduction
from edisgo.tools.pseudo_coordinates import make_pseudo_coordinates


class TestSpatialComplexityReduction:
    @pytest.fixture(autouse=True)
    def test_edisgo_obj(self):
        edisgo_root = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        edisgo_root.set_time_series_worst_case_analysis()
        make_pseudo_coordinates(edisgo_root)
        return edisgo_root

    def setup_busmap_df(self, edisgo_obj):
        busmap_df = spatial_complexity_reduction.make_busmap(
            edisgo_obj,
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
        with test_exception:
            busmap_df = spatial_complexity_reduction.make_busmap(
                test_edisgo_obj,
                mode=mode,
                cluster_area=cluster_area,
                reduction_factor=reduction_factor,
                reduction_factor_not_focused=reduction_factor_not_focused,
            )
            # Check for deterministic behaviour.
            assert len(set(busmap_df["new_bus"].to_list())) == n_new_buses

    @pytest.mark.parametrize(
        "cluster_area,grid,n_buses",
        [
            # Cluster area: 'grid'
            ("grid", "MVGrid", 7),
            ("grid", "LVGrid", 6),
            # Cluster area: 'feeder'
            ("feeder", "MVGrid", 9),
            ("feeder", "LVGrid", 8),
            # Cluster area: 'main_feeder'
            ("main_feeder", "MVGrid", 7),
            ("main_feeder", "LVGrid", 6),
        ],
    )
    def test_make_busmap_for_only_one_grid(
        self,
        test_edisgo_obj,
        cluster_area,
        grid,
        n_buses,
    ):
        if grid == "MVGrid":
            grid = spatial_complexity_reduction._make_grid_list(
                test_edisgo_obj, grid="MVGrid_1"
            )[0]
        elif grid == "LVGrid":
            grid = spatial_complexity_reduction._make_grid_list(
                test_edisgo_obj, grid="LVGrid_9"
            )[0]

        busmap_df = spatial_complexity_reduction.make_busmap(
            test_edisgo_obj,
            mode="kmeans",
            grid=grid,
            cluster_area=cluster_area,
            reduction_factor=0.2,
        )
        # Check for deterministic behaviour.
        assert len(set(busmap_df["new_bus"].to_list())) == n_buses

    @pytest.mark.parametrize(
        "line_naming_convention,"
        "aggregation_mode,"
        "load_aggregation_mode, "
        "generator_aggregation_mode, "
        "n_loads, "
        "n_generators",
        [
            ("standard_lines", True, "bus", "bus", 27, 17),
            ("standard_lines", True, "sector", "type", 28, 18),
            ("combined_name", False, None, None, 50, 28),
        ],
    )
    def test_apply_busmap(
        self,
        test_edisgo_obj,
        line_naming_convention,
        aggregation_mode,
        load_aggregation_mode,
        generator_aggregation_mode,
        n_loads,
        n_generators,
    ):
        busmap_df = self.setup_busmap_df(test_edisgo_obj)

        # Add second line to test line reduction
        test_edisgo_obj.topology.lines_df.loc[
            "Line_10003_2"
        ] = test_edisgo_obj.topology.lines_df.loc["Line_10003"]

        assert test_edisgo_obj.topology.buses_df.shape[0] == 142
        assert test_edisgo_obj.topology.lines_df.shape[0] == 132
        assert test_edisgo_obj.topology.loads_df.shape[0] == 50
        assert test_edisgo_obj.topology.generators_df.shape[0] == 28
        assert test_edisgo_obj.topology.storage_units_df.shape[0] == 1
        assert test_edisgo_obj.topology.transformers_df.shape[0] == 14
        assert test_edisgo_obj.topology.switches_df.shape[0] == 2

        linemap_df = spatial_complexity_reduction.apply_busmap(
            test_edisgo_obj,
            busmap_df,
            line_naming_convention=line_naming_convention,
            aggregation_mode=aggregation_mode,
            load_aggregation_mode=load_aggregation_mode,
            generator_aggregation_mode=generator_aggregation_mode,
        )

        assert test_edisgo_obj.topology.buses_df.shape[0] == 43
        assert test_edisgo_obj.topology.lines_df.shape[0] == 34
        assert test_edisgo_obj.topology.loads_df.shape[0] == n_loads
        assert test_edisgo_obj.topology.generators_df.shape[0] == n_generators
        assert test_edisgo_obj.topology.storage_units_df.shape[0] == 1
        assert test_edisgo_obj.topology.transformers_df.shape[0] == 14
        assert test_edisgo_obj.topology.switches_df.shape[0] == 2

        if line_naming_convention == "standard_lines":
            assert (
                test_edisgo_obj.topology.lines_df.loc[
                    "Line_Bus_MVStation_1_to_Bus_mvgd_1_F0_B2", "type_info"
                ]
                == "NA2XS2Y 3x1x240"
            )
        elif line_naming_convention == "combined_name":
            assert (
                test_edisgo_obj.topology.lines_df.loc[
                    "Line_Bus_MVStation_1_to_Bus_mvgd_1_F0_B2", "type_info"
                ]
                == "Merged: 48-AL1/8-ST1A 48-AL1/8-ST1A "
            )
        timeseries = test_edisgo_obj.timeseries
        assert timeseries.loads_active_power.shape[1] == n_loads
        assert timeseries.loads_reactive_power.shape[1] == n_loads
        assert timeseries.generators_active_power.shape[1] == n_generators
        assert timeseries.generators_reactive_power.shape[1] == n_generators
        assert len(set(linemap_df["new_line_name"].to_list())) == 34

    def test_spatial_complexity_reduction(self, test_edisgo_obj):
        (
            busmap_df,
            linemap_df,
        ) = spatial_complexity_reduction.spatial_complexity_reduction(
            test_edisgo_obj,
            mode="kmeans",
            cluster_area="grid",
            reduction_factor=0.2,
            reduction_factor_not_focused=False,
        )
        # Check for deterministic behaviour.
        assert len(set(busmap_df["new_bus"].to_list())) == 32
        assert len(set(linemap_df["new_line_name"].to_list())) == 23

        # Check that edisgo_object can run power flow and reinforce
        test_edisgo_obj.analyze()
        test_edisgo_obj.reinforce()

    def test_compare_voltage(self, test_edisgo_obj):
        edisgo_reduced = copy.deepcopy(test_edisgo_obj)
        (
            busmap_df,
            linemap_df,
        ) = spatial_complexity_reduction.spatial_complexity_reduction(
            edisgo_reduced,
            mode="kmeans",
            cluster_area="grid",
            reduction_factor=0.2,
            reduction_factor_not_focused=False,
        )
        test_edisgo_obj.analyze()
        edisgo_reduced.analyze()
        _, rms = spatial_complexity_reduction.compare_voltage(
            test_edisgo_obj, edisgo_reduced, busmap_df, "max"
        )
        assert np.isclose(rms, 0.00766, atol=1e-5)

    def test_compare_apparent_power(self, test_edisgo_obj):
        edisgo_reduced = copy.deepcopy(test_edisgo_obj)

        (
            busmap_df,
            linemap_df,
        ) = spatial_complexity_reduction.spatial_complexity_reduction(
            edisgo_reduced,
            mode="kmeans",
            cluster_area="grid",
            reduction_factor=0.2,
            reduction_factor_not_focused=False,
        )
        test_edisgo_obj.analyze()
        edisgo_reduced.analyze()
        _, rms = spatial_complexity_reduction.compare_apparent_power(
            test_edisgo_obj, edisgo_reduced, linemap_df, "max"
        )
        assert np.isclose(rms, 2.873394, atol=1e-5)

    def test_remove_short_end_lines(self, test_edisgo_obj):
        edisgo_root = copy.deepcopy(test_edisgo_obj)

        # change line length of line to switch to under 1 meter to check that it
        # is not deleted
        edisgo_root.topology.lines_df.at["Line_10016", "length"] = 0.0006

        spatial_complexity_reduction.remove_short_end_lines(edisgo_root)

        # Check that the generator changed the bus
        df_old = test_edisgo_obj.topology.generators_df
        df_new = edisgo_root.topology.generators_df
        assert (
            df_old.loc[df_old["bus"] == "Bus_GeneratorFluctuating_19", "bus"].index
            == df_new.loc[df_new["bus"] == "Bus_BranchTee_LVGrid_5_6", "bus"].index
        )
        # Check that the load changed the bus
        df_old = test_edisgo_obj.topology.loads_df
        df_new = edisgo_root.topology.loads_df
        assert (
            df_old.loc[df_old["bus"] == "Bus_Load_residential_LVGrid_5_3", "bus"].index
            == df_new.loc[df_new["bus"] == "Bus_BranchTee_LVGrid_5_6", "bus"].index
        )
        # Check that 2 lines were removed
        assert len(test_edisgo_obj.topology.lines_df) - 2 == len(
            edisgo_root.topology.lines_df
        )

    # def test_remove_lines_under_one_meter(self, test_edisgo_obj, caplog):
    #     edisgo_root = copy.deepcopy(test_edisgo_obj)
    #     edisgo_root.topology.lines_df.at["Line_50000002", "length"] = 0.0006
    #     edisgo_root.topology.lines_df.at["Line_90000009", "length"] = 0.0007
    #     edisgo_root.topology.lines_df.at["Line_90000013", "length"] = 0.0008
    #     edisgo_clean = spatial_complexity_reduction.remove_lines_under_one_meter(
    #         edisgo_root
    #     )
    #     with caplog.at_level(logging.WARNING):
    #         edisgo_clean.check_integrity()
    #     assert "isolated nodes" not in caplog.text
    #     # Check that 1 line was removed
    #     assert len(edisgo_root.topology.lines_df) - 1 == len(
    #         edisgo_clean.topology.lines_df
    #     )
