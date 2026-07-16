import copy

from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
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
        test_edisgo_obj.topology.lines_df.loc["Line_10003_2"] = (
            test_edisgo_obj.topology.lines_df.loc["Line_10003"]
        )

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


class TestApplyReducedResultsToFullGrid:
    """
    Tests for
    :func:`~.tools.spatial_complexity_reduction.apply_reduced_results_to_full_grid`.

    Uses stub OPF results (directly writing to
    ``reduced_grid.timeseries._loads_active_power`` /
    ``_storage_units_active_power``) rather than running a real OPF, since
    what is under test is the map-back/disaggregation logic, not
    ``pm_optimize`` itself.
    """

    @pytest.fixture(autouse=True)
    def test_edisgo_obj(self):
        edisgo_root = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        edisgo_root.set_time_series_worst_case_analysis()
        make_pseudo_coordinates(edisgo_root)
        return edisgo_root

    @pytest.fixture
    def full_and_reduced(self, test_edisgo_obj):
        full_grid = copy.deepcopy(test_edisgo_obj)
        reduced_grid, _, _ = full_grid.spatial_complexity_reduction(
            copy_edisgo=True,
            mode="kmeansdijkstra",
            cluster_area="feeder",
            reduction_factor=0.1,
            aggregation_mode=True,
            load_aggregation_mode="bus",
        )
        return full_grid, reduced_grid

    def _first_representative_with(self, reduced_grid, n_members):
        loads_df = reduced_grid.topology.loads_df
        candidates = loads_df[
            loads_df["old_name"].apply(
                lambda v: isinstance(v, list) and len(v) == n_members
            )
        ]
        assert not candidates.empty, (
            f"fixture grid has no aggregated load representative with "
            f"exactly {n_members} old_name member(s); adjust the fixture "
            f"or reduction_factor."
        )
        return candidates.index[0]

    def test_by_name_write_back_aggregation_mode_false(self, test_edisgo_obj):
        # aggregation_mode=False: no merging, so restore is a plain by-name
        # write-back for every flexibility type.
        full_grid = copy.deepcopy(test_edisgo_obj)
        reduced_grid, _, _ = full_grid.spatial_complexity_reduction(
            copy_edisgo=True,
            mode="kmeansdijkstra",
            cluster_area="feeder",
            reduction_factor=0.1,
            aggregation_mode=False,
        )
        ti = full_grid.timeseries.timeindex
        load_name = reduced_grid.topology.loads_df.index[0]
        storage_name = reduced_grid.topology.storage_units_df.index[0]
        full_grid.dsm.p_max = pd.DataFrame(1.0, index=ti, columns=[load_name])

        reduced_grid.timeseries._loads_active_power.loc[ti, load_name] = [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
        reduced_grid.timeseries._storage_units_active_power.loc[ti, storage_name] = [
            5.0,
            6.0,
            7.0,
            8.0,
        ]

        result = spatial_complexity_reduction.apply_reduced_results_to_full_grid(
            full_grid=full_grid,
            reduced_grid=reduced_grid,
            flexible_loads=[load_name],
            flexible_storage_units=[storage_name],
        )

        assert result.timeseries.loads_active_power.loc[ti, load_name].tolist() == [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
        assert result.timeseries.storage_units_active_power.loc[
            ti, storage_name
        ].tolist() == [5.0, 6.0, 7.0, 8.0]

    def test_accepts_numpy_array_flexible_component_lists(self, test_edisgo_obj):
        # Regression test: task_optimize derives flexible_loads as
        # edisgo.dsm.p_min.columns.values (a numpy array), unlike the other
        # three flexible_* lists which are built with .tolist(). "x or []"
        # raises ValueError ("truth value of an array... is ambiguous") for
        # any such array with more than one element - a real crash hit on
        # the first end-to-end pipeline run using aggregation_mode=False.
        full_grid = copy.deepcopy(test_edisgo_obj)
        reduced_grid, _, _ = full_grid.spatial_complexity_reduction(
            copy_edisgo=True,
            mode="kmeansdijkstra",
            cluster_area="feeder",
            reduction_factor=0.1,
            aggregation_mode=False,
        )
        ti = full_grid.timeseries.timeindex
        load_name = reduced_grid.topology.loads_df.index[0]
        full_grid.dsm.p_max = pd.DataFrame(1.0, index=ti, columns=[load_name])
        reduced_grid.timeseries._loads_active_power.loc[ti, load_name] = [
            1.0,
            2.0,
            3.0,
            4.0,
        ]

        result = spatial_complexity_reduction.apply_reduced_results_to_full_grid(
            full_grid=full_grid,
            reduced_grid=reduced_grid,
            flexible_loads=np.array([load_name]),
        )

        assert result.timeseries.loads_active_power.loc[ti, load_name].tolist() == [
            1.0,
            2.0,
            3.0,
            4.0,
        ]

    def test_disaggregation_multi_member_sums_to_representative(self, full_and_reduced):
        full_grid, reduced_grid = full_and_reduced
        ti = full_grid.timeseries.timeindex
        rep = self._first_representative_with(reduced_grid, n_members=2)
        members = reduced_grid.topology.loads_df.at[rep, "old_name"]

        p_max = pd.DataFrame(0.0, index=ti, columns=members)
        for i, member in enumerate(members):
            p_max[member] = [0.1 * (i + 1), 0.0, 0.2 * (i + 1), 0.05 * (i + 1)]
        full_grid.dsm.p_max = p_max

        rep_power = pd.Series([1.0, 2.0, 0.0, 3.0], index=ti)
        reduced_grid.timeseries._loads_active_power.loc[ti, rep] = rep_power.values

        result = spatial_complexity_reduction.apply_reduced_results_to_full_grid(
            full_grid=full_grid, reduced_grid=reduced_grid, flexible_loads=[rep]
        )

        total = result.timeseries.loads_active_power.loc[ti, members].sum(axis=1)
        assert np.allclose(total.values, rep_power.values)
        # Member with zero envelope at t0/t2/t3 gets none of the dispatch;
        # both members zero at t1 falls back to an equal split.
        assert result.timeseries.loads_active_power.at[
            ti[1], members[0]
        ] == pytest.approx(rep_power.iloc[1] / 2)

    def test_disaggregation_singleton_renamed_representative(self, full_and_reduced):
        # Regression test: under aggregation_mode=True, spatial_complexity_
        # reduction renames every group's representative, including
        # singleton groups (a bus with exactly one flexible load of a given
        # type/sector) - so the representative's name can differ from its
        # one old_name member's name. A by-name write-back using the
        # representative's name would silently miss the real target column
        # on full_grid (which only has the original, un-renamed name) and
        # create a phantom column instead - this must not happen.
        full_grid, reduced_grid = full_and_reduced
        ti = full_grid.timeseries.timeindex
        rep = self._first_representative_with(reduced_grid, n_members=1)
        member = reduced_grid.topology.loads_df.at[rep, "old_name"][0]
        assert rep != member, "fixture assumption: representative was renamed"

        full_grid.dsm.p_max = pd.DataFrame(1.0, index=ti, columns=[member])
        rep_power = pd.Series([7.0, 8.0, 9.0, 10.0], index=ti)
        reduced_grid.timeseries._loads_active_power.loc[ti, rep] = rep_power.values

        result = spatial_complexity_reduction.apply_reduced_results_to_full_grid(
            full_grid=full_grid, reduced_grid=reduced_grid, flexible_loads=[rep]
        )

        assert result.timeseries.loads_active_power.loc[ti, member].tolist() == (
            rep_power.tolist()
        )
        assert rep not in result.timeseries.loads_active_power.columns

    def test_raises_clear_error_on_time_index_mismatch(self, full_and_reduced):
        full_grid, reduced_grid = full_and_reduced
        ti = full_grid.timeseries.timeindex
        rep = self._first_representative_with(reduced_grid, n_members=1)
        member = reduced_grid.topology.loads_df.at[rep, "old_name"][0]

        # dsm.p_max missing the last time step of full_grid's active index.
        full_grid.dsm.p_max = pd.DataFrame(1.0, index=ti[:-1], columns=[member])
        reduced_grid.timeseries._loads_active_power.loc[ti, rep] = [
            1.0,
            2.0,
            3.0,
            4.0,
        ]

        with pytest.raises(ValueError, match="does not cover"):
            spatial_complexity_reduction.apply_reduced_results_to_full_grid(
                full_grid=full_grid, reduced_grid=reduced_grid, flexible_loads=[rep]
            )

    def test_storage_units_never_aggregated_always_by_name(self, full_and_reduced):
        full_grid, reduced_grid = full_and_reduced
        ti = full_grid.timeseries.timeindex
        storage_name = full_grid.topology.storage_units_df.index[0]
        assert storage_name in reduced_grid.topology.storage_units_df.index
        assert "old_name" not in reduced_grid.topology.storage_units_df.columns

        reduced_grid.timeseries._storage_units_active_power.loc[ti, storage_name] = [
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        result = spatial_complexity_reduction.apply_reduced_results_to_full_grid(
            full_grid=full_grid,
            reduced_grid=reduced_grid,
            flexible_storage_units=[storage_name],
        )
        assert result.timeseries.storage_units_active_power.loc[
            ti, storage_name
        ].tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_reactive_power_recomputed_after_restore(self, full_and_reduced):
        full_grid, reduced_grid = full_and_reduced
        ti = full_grid.timeseries.timeindex
        rep = self._first_representative_with(reduced_grid, n_members=1)
        member = reduced_grid.topology.loads_df.at[rep, "old_name"][0]
        full_grid.dsm.p_max = pd.DataFrame(1.0, index=ti, columns=[member])

        reactive_before = full_grid.timeseries.loads_reactive_power.loc[
            ti, member
        ].copy()
        reduced_grid.timeseries._loads_active_power.loc[ti, rep] = [
            50.0,
            50.0,
            50.0,
            50.0,
        ]

        result = spatial_complexity_reduction.apply_reduced_results_to_full_grid(
            full_grid=full_grid, reduced_grid=reduced_grid, flexible_loads=[rep]
        )

        reactive_after = result.timeseries.loads_reactive_power.loc[ti, member]
        assert not reactive_after.equals(reactive_before)
