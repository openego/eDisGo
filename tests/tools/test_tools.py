import copy

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_allclose, assert_array_equal

from edisgo import EDisGo
from edisgo.tools import tools


class TestTools:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        self.timesteps = self.edisgo.timeseries.timeindex
        self.edisgo.analyze()

    def test_calculate_line_reactance(self):
        # test single line
        data = tools.calculate_line_reactance(2, 3, 1)
        assert np.isclose(data, 1.88496)
        data = tools.calculate_line_reactance(np.array([2, 3]), 3, 1)
        assert_allclose(data, np.array([1.88496, 2.82743]), rtol=1e-5)
        # test parallel line
        data = tools.calculate_line_reactance(2, 3, 2)
        assert np.isclose(data, 1.88496 / 2)
        data = tools.calculate_line_reactance(np.array([2, 3]), 3, 2)
        assert_allclose(data, np.array([1.88496 / 2, 2.82743 / 2]), rtol=1e-5)

    def test_calculate_line_resistance(self):
        # test single line
        data = tools.calculate_line_resistance(2, 3, 1)
        assert data == 6
        data = tools.calculate_line_resistance(np.array([2, 3]), 3, 1)
        assert_array_equal(data, np.array([6, 9]))
        # test parallel line
        data = tools.calculate_line_resistance(2, 3, 2)
        assert data == 3
        data = tools.calculate_line_resistance(np.array([2, 3]), 3, 2)
        assert_array_equal(data, np.array([3, 4.5]))

    def test_calculate_line_susceptance(self):
        # test single line
        assert np.isclose(tools.calculate_line_susceptance(2, 3, 1), 0.00188495559)
        # test parallel line
        assert np.isclose(tools.calculate_line_susceptance(2, 3, 2), 2 * 0.00188495559)
        # test line with c = 0
        assert np.isclose(tools.calculate_line_susceptance(0, 3, 1), 0)

    def test_calculate_apparent_power(self):
        # test single line
        data = tools.calculate_apparent_power(20, 30, 1)
        assert np.isclose(data, 1039.23)
        data = tools.calculate_apparent_power(30, np.array([20, 30]), 1)
        assert_allclose(data, np.array([1039.23, 1558.84]), rtol=1e-5)
        data = tools.calculate_apparent_power(np.array([30, 30]), np.array([20, 30]), 1)
        assert_allclose(data, np.array([1039.23, 1558.84]), rtol=1e-5)
        # test parallel line
        data = tools.calculate_apparent_power(20, 30, 2)
        assert np.isclose(data, 1039.23 * 2)
        data = tools.calculate_apparent_power(30, np.array([20, 30]), 3)
        assert_allclose(data, np.array([1039.23 * 3, 1558.84 * 3]), rtol=1e-5)
        data = tools.calculate_apparent_power(np.array([30, 30]), np.array([20, 30]), 2)
        assert_allclose(data, np.array([1039.23 * 2, 1558.84 * 2]), rtol=1e-5)
        data = tools.calculate_apparent_power(
            np.array([30, 30]), np.array([20, 30]), np.array([2, 3])
        )
        assert_allclose(data, np.array([1039.23 * 2, 1558.84 * 3]), rtol=1e-5)

    def test_drop_duplicated_indices(self):
        test_df = pd.DataFrame(
            data={
                "a": [1, 2, 3],
                "b": [3, 4, 5],
                "c": [4, 5, 6],
            },
            index=[0, 1, 0],
        )
        check_df = tools.drop_duplicated_indices(test_df)
        assert len(check_df.index) == 2
        assert (check_df.loc[0, :] == [3, 5, 6]).all()

    def test_drop_duplicated_columns(self):
        test_df = pd.DataFrame(
            data={
                "a": [1, 2, 3],  # noqa: F601
                "b": [3, 4, 5],
                "a": [4, 5, 6],  # noqa: F601
            },
            index=[0, 1, 2],
        )
        check_df = tools.drop_duplicated_columns(test_df)
        assert len(check_df.columns) == 2
        assert (check_df.loc[:, "a"] == [4, 5, 6]).all()

    def test_select_cable(self):
        cable_data, num_parallel_cables = tools.select_cable(self.edisgo, "mv", 5.1)
        assert cable_data.name == "NA2XS2Y 3x1x150 RE/25"
        assert num_parallel_cables == 1

        cable_data, num_parallel_cables = tools.select_cable(self.edisgo, "mv", 40)
        assert cable_data.name == "NA2XS(FL)2Y 3x1x500 RM/35"
        assert num_parallel_cables == 2

        cable_data, num_parallel_cables = tools.select_cable(self.edisgo, "lv", 0.18)
        assert cable_data.name == "NAYY 4x1x150"
        assert num_parallel_cables == 1

    def test_assign_feeder(self):

        # ######## test MV feeder mode ########
        topo = self.edisgo.topology
        topo.assign_feeders(mode="mv_feeder")

        # check that all lines and all buses (except MV station bus and buses
        # in aggregated load areas) have an MV feeder assigned
        assert not topo.lines_df.mv_feeder.isna().any()
        mv_station = topo.mv_grid.station.index[0]
        buses_aggr_la = list(
            topo.transformers_df[topo.transformers_df.bus0 == mv_station].bus1.unique()
        )
        buses_aggr_la.append(mv_station)
        assert (
            not topo.buses_df[~topo.buses_df.index.isin(buses_aggr_la)]
            .mv_feeder.isna()
            .any()
        )

        # check specific buses
        # MV and LV bus in feeder 1
        assert (
            topo.buses_df.at["Bus_GeneratorFluctuating_7", "mv_feeder"]
            == "Bus_BranchTee_MVGrid_1_6"
        )
        assert (
            topo.buses_df.at["Bus_BranchTee_LVGrid_4_1", "mv_feeder"]
            == "Bus_BranchTee_MVGrid_1_5"
        )
        # MV bus in feeder 2
        assert (
            topo.buses_df.at["Bus_GeneratorFluctuating_3", "mv_feeder"]
            == "Bus_BranchTee_MVGrid_1_5"
        )

        # check specific lines
        assert topo.lines_df.at["Line_10003", "mv_feeder"] == "Bus_BranchTee_MVGrid_1_1"

        # ######## test LV feeder mode ########
        topo = self.edisgo.topology
        topo.assign_feeders(mode="grid_feeder")

        # check that all buses and lines have a grid feeder assigned
        assert not topo.lines_df.grid_feeder.isna().any()
        assert not topo.buses_df.grid_feeder.isna().any()

        # check specific buses
        assert (
            topo.buses_df.at["Bus_BranchTee_LVGrid_1_8", "grid_feeder"]
            == "Bus_BranchTee_LVGrid_1_7"
        )
        assert (
            topo.buses_df.at["Bus_BranchTee_LVGrid_2_4", "grid_feeder"]
            == "Bus_BranchTee_LVGrid_2_1"
        )

        # check specific lines
        assert (
            topo.lines_df.at["Line_30000005", "grid_feeder"]
            == "Bus_BranchTee_LVGrid_3_3"
        )
        assert (
            topo.lines_df.at["Line_40000001", "grid_feeder"]
            == "Bus_GeneratorFluctuating_16"
        )

        # ######## test real ding0 network ########
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path,
            worst_case_analysis="worst-case",
        )
        topo = self.edisgo.topology
        topo.assign_feeders(mode="mv_feeder")

        # check that all lines and all buses have an MV feeder assigned
        assert not topo.lines_df.mv_feeder.isna().any()
        mv_station = topo.mv_grid.station.index[0]
        buses_aggr_la = list(
            topo.transformers_df[topo.transformers_df.bus0 == mv_station].bus1.unique()
        )
        buses_aggr_la.append(mv_station)
        assert not topo.buses_df.mv_feeder.isna().any()

        topo.assign_feeders(mode="grid_feeder")
        # check that all buses and lines grid feeder assigned
        assert not topo.lines_df.grid_feeder.isna().any()
        assert not topo.buses_df.grid_feeder.isna().any()

    def test_get_downstream_buses(self):

        # ######## test with LV bus ########
        buses_downstream = tools.get_downstream_buses(
            self.edisgo, "BusBar_MVGrid_1_LVGrid_1_LV"
        )

        lv_grid = self.edisgo.topology.get_lv_grid(1)
        assert len(buses_downstream) == len(lv_grid.buses_df)
        assert all([_ in buses_downstream for _ in lv_grid.buses_df.index])

        # ######## test with MV line ########
        buses_downstream = tools.get_downstream_buses(
            self.edisgo, "Line_10010", comp_type="line"
        )

        lv_grid = self.edisgo.topology.get_lv_grid(5)
        assert len(buses_downstream) == len(lv_grid.buses_df) + 4
        assert all([_ in buses_downstream for _ in lv_grid.buses_df.index])

    def test_get_path_length_to_station(self):
        # ToDo implement
        pass

    def test_assign_voltage_level_to_component(self):
        # ToDo implement
        pass

    def test_determine_grid_integration_voltage_level(self):
        assert tools.determine_grid_integration_voltage_level(self.edisgo, 0.05) == 7
        assert tools.determine_grid_integration_voltage_level(self.edisgo, 0.2) == 6
        assert tools.determine_grid_integration_voltage_level(self.edisgo, 1.5) == 5
        assert tools.determine_grid_integration_voltage_level(self.edisgo, 16) == 4

    def test_determine_bus_voltage_level(self):
        bus_mv_station = "Bus_MVStation_1"
        bus_mv = "Bus_GeneratorFluctuating_7"
        bus_lv_station = "BusBar_MVGrid_1_LVGrid_1_LV"
        bus_lv = "Bus_BranchTee_LVGrid_1_10"
        assert tools.determine_bus_voltage_level(self.edisgo, bus_mv_station) == 4
        assert tools.determine_bus_voltage_level(self.edisgo, bus_mv) == 5
        assert tools.determine_bus_voltage_level(self.edisgo, bus_lv_station) == 6
        assert tools.determine_bus_voltage_level(self.edisgo, bus_lv) == 7

        # test if buses directly connected to station are identified as voltage level
        # 4 or 6, if they are not part of a larger feeder
        # set up bus that is directly connected to HV/MV station
        bus_voltage_level_4 = self.edisgo.topology.add_bus("dummy_bus", 20.0)
        self.edisgo.topology.add_line(
            bus_mv_station, bus_voltage_level_4, 10.0, type_info="NA2XS2Y 3x1x185 RM/25"
        )
        bus_voltage_level_5 = "Bus_BranchTee_MVGrid_1_1"
        bus_voltage_level_6 = "Bus_GeneratorFluctuating_16"
        bus_voltage_level_7 = "Bus_BranchTee_LVGrid_4_1"

        assert tools.determine_bus_voltage_level(self.edisgo, bus_voltage_level_4) == 4
        assert tools.determine_bus_voltage_level(self.edisgo, bus_voltage_level_5) == 5
        assert tools.determine_bus_voltage_level(self.edisgo, bus_voltage_level_6) == 6
        assert tools.determine_bus_voltage_level(self.edisgo, bus_voltage_level_7) == 7

    def test_get_weather_cells_intersecting_with_grid_district(self):
        weather_cells = tools.get_weather_cells_intersecting_with_grid_district(
            self.edisgo
        )
        assert len(weather_cells) == 4
        assert 1123075 in weather_cells
        assert 1122075 in weather_cells
        assert 1122076 in weather_cells
        # the following weather cell does not intersect with the grid district
        # but there are generators in the grid that have that weather cell
        # for some reason..
        assert 1122074 in weather_cells

    @pytest.mark.local
    def test_get_weather_cells_intersecting_with_grid_district_egon(self):
        edisgo_obj = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        weather_cells = tools.get_weather_cells_intersecting_with_grid_district(
            edisgo_obj, pytest.engine
        )
        assert len(weather_cells) == 2
        assert 11051 in weather_cells
        assert 11052 in weather_cells

    def test_add_line_susceptance(self):
        assert self.edisgo.topology.lines_df.loc["Line_10006", "b"] == 0
        assert self.edisgo.topology.lines_df.loc["Line_50000002", "b"] == 0

        # test mode no_b
        edisgo_root = copy.deepcopy(self.edisgo)
        edisgo_root.topology.lines_df.loc["Line_10006", "b"] = 1
        edisgo_root.topology.lines_df.loc["Line_50000002", "b"] = 1
        edisgo_root = tools.add_line_susceptance(edisgo_root, mode="no_b")
        assert edisgo_root.topology.lines_df.loc["Line_10006", "b"] == 0
        assert edisgo_root.topology.lines_df.loc["Line_50000002", "b"] == 0

        # test mode mv_b
        edisgo_root = copy.deepcopy(self.edisgo)
        edisgo_root.topology.lines_df.loc["Line_10006", "b"] = 1
        edisgo_root.topology.lines_df.loc["Line_50000002", "b"] = 1
        edisgo_root = tools.add_line_susceptance(edisgo_root, mode="mv_b")
        assert edisgo_root.topology.lines_df.loc[
            "Line_10006", "b"
        ] == tools.calculate_line_susceptance(0.304, 0.297650465459542, 1)
        assert edisgo_root.topology.lines_df.loc["Line_50000002", "b"] == 0

        # test mode all_b
        edisgo_root = copy.deepcopy(self.edisgo)
        edisgo_root = tools.add_line_susceptance(edisgo_root, mode="all_b")
        assert edisgo_root.topology.lines_df.loc[
            "Line_10006", "b"
        ] == tools.calculate_line_susceptance(0.304, 0.297650465459542, 1)
        assert edisgo_root.topology.lines_df.loc[
            "Line_50000002", "b"
        ] == tools.calculate_line_susceptance(0.25, 0.03, 1)

    def test_reduce_memory_usage(self):
        # ToDo implement
        pass
