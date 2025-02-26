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

    def test_calculate_voltage_diff_pu_per_line(self):
        correct_value_positive_sign = 0.03261946832784687
        correct_value_negative_sign = 0.06008053167215312
        r_total = 0.412
        x_total = 0.252

        # test generator, float
        data = tools.calculate_voltage_diff_pu_per_line(
            s_max=50,
            r_total=r_total,
            x_total=x_total,
            v_nom=20,
            q_sign=-1,
            power_factor=0.9,
        )
        assert np.isclose(data, correct_value_positive_sign)
        # test generator, array
        data = tools.calculate_voltage_diff_pu_per_line(
            s_max=np.array([50, 50]),
            r_total=np.array([r_total, r_total]),
            x_total=np.array([x_total, x_total]),
            v_nom=20,
            q_sign=-1,
            power_factor=0.9,
        )
        assert_allclose(
            data,
            np.array([correct_value_positive_sign, correct_value_positive_sign]),
            rtol=1e-5,
        )
        # test generator, float, higher voltage
        data = tools.calculate_voltage_diff_pu_per_line(
            s_max=50,
            r_total=r_total,
            x_total=x_total,
            v_nom=40,
            q_sign=-1,
            power_factor=0.9,
        )
        assert np.isclose(data, correct_value_positive_sign / 4)
        # test generator, array, larger cable
        data = tools.calculate_voltage_diff_pu_per_line(
            s_max=np.array([100, 100]),
            r_total=np.array([r_total, r_total]),
            x_total=np.array([x_total, x_total]),
            v_nom=np.array([20, 20]),
            q_sign=-1,
            power_factor=0.9,
        )
        assert_allclose(
            data,
            np.array(
                [correct_value_positive_sign * 2, correct_value_positive_sign * 2]
            ),
            rtol=1e-5,
        )
        # test generator, capacitive
        data = tools.calculate_voltage_diff_pu_per_line(
            s_max=100,
            r_total=r_total,
            x_total=x_total,
            v_nom=20,
            q_sign=1,
            power_factor=0.9,
        )
        assert np.isclose(data, correct_value_negative_sign * 2)
        # test load, capacitive
        data = tools.calculate_voltage_diff_pu_per_line(
            s_max=100,
            r_total=r_total,
            x_total=x_total,
            v_nom=20,
            q_sign=-1,
            power_factor=0.9,
        )
        assert np.isclose(data, correct_value_positive_sign * 2)

        # test the examples from  VDE-AR-N 4105 attachment D
        data = tools.calculate_voltage_diff_pu_per_line(
            s_max=0.02,
            r_total=0.2001,
            x_total=0.1258,
            v_nom=0.4,
            q_sign=-1,
            power_factor=1,
        )
        assert np.isclose(data, 0.025, rtol=1e-2)

        data = tools.calculate_voltage_diff_pu_per_line(
            s_max=0.022,
            r_total=0.2001,
            x_total=0.1258,
            v_nom=0.4,
            q_sign=-1,
            power_factor=0.9,
        )
        assert np.isclose(data, 0.0173, rtol=1e-2)

    def test_calculate_voltage_diff_pu_per_line_from_type(self):
        correct_value_negative_sign = 0.4916578234319946 * 1e-2
        correct_value_positive_sign = 0.017583421765680056
        data = tools.calculate_voltage_diff_pu_per_line_from_type(
            edisgo_obj=self.edisgo,
            cable_names="NA2XS(FL)2Y 3x1x300 RM/25",
            length=1,
            num_parallel=1,
            v_nom=20,
            s_max=50,
            component_type="generator",
        )
        assert np.isclose(data, correct_value_negative_sign)
        data = tools.calculate_voltage_diff_pu_per_line_from_type(
            edisgo_obj=self.edisgo,
            cable_names=np.array(
                ["NA2XS(FL)2Y 3x1x300 RM/25", "NA2XS(FL)2Y 3x1x300 RM/25"]
            ),
            length=1,
            num_parallel=1,
            v_nom=20,
            s_max=50,
            component_type="generator",
        )
        assert_allclose(
            data,
            np.array([correct_value_negative_sign, correct_value_negative_sign]),
            rtol=1e-5,
        )
        data = tools.calculate_voltage_diff_pu_per_line_from_type(
            edisgo_obj=self.edisgo,
            cable_names="NA2XS(FL)2Y 3x1x300 RM/25",
            length=2,
            num_parallel=1,
            v_nom=20,
            s_max=50,
            component_type="generator",
        )
        assert np.isclose(data, 2 * correct_value_negative_sign)
        data = tools.calculate_voltage_diff_pu_per_line_from_type(
            edisgo_obj=self.edisgo,
            cable_names=np.array(
                ["NA2XS(FL)2Y 3x1x300 RM/25", "NA2XS(FL)2Y 3x1x300 RM/25"]
            ),
            length=2,
            num_parallel=1,
            v_nom=20,
            s_max=50,
            component_type="generator",
        )
        assert_allclose(
            data,
            np.array(
                [2 * correct_value_negative_sign, 2 * correct_value_negative_sign]
            ),
            rtol=1e-5,
        )

        data = tools.calculate_voltage_diff_pu_per_line_from_type(
            edisgo_obj=self.edisgo,
            cable_names="NA2XS(FL)2Y 3x1x300 RM/25",
            length=1,
            num_parallel=2,
            v_nom=20,
            s_max=50,
            component_type="generator",
        )
        assert np.isclose(data, correct_value_negative_sign / 2)

        data = tools.calculate_voltage_diff_pu_per_line_from_type(
            edisgo_obj=self.edisgo,
            cable_names="NA2XS(FL)2Y 3x1x300 RM/25",
            length=1,
            num_parallel=2,
            v_nom=20,
            s_max=50,
            component_type="conventional_load",
        )
        assert np.isclose(data, correct_value_positive_sign / 2)

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
        # no length given
        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo,
            "mv",
            5.1,
        )
        assert cable_data.name == "NA2XS2Y 3x1x150 RE/25"
        assert num_parallel_cables == 1

        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo,
            "mv",
            40,
        )
        assert cable_data.name == "NA2XS(FL)2Y 3x1x500 RM/35"
        assert num_parallel_cables == 2

        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo,
            "lv",
            0.18,
        )
        assert cable_data.name == "NAYY 4x1x150"
        assert num_parallel_cables == 1

        # length given
        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo,
            "mv",
            5.1,
            length=2,
            component_type="conventional_load",
        )
        assert cable_data.name == "NA2XS2Y 3x1x150 RE/25"
        assert num_parallel_cables == 1

        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo,
            "mv",
            40,
            length=1,
            component_type="conventional_load",
        )
        assert cable_data.name == "NA2XS(FL)2Y 3x1x500 RM/35"
        assert num_parallel_cables == 2

        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo,
            "lv",
            0.18,
            length=1,
            component_type="conventional_load",
        )
        assert cable_data.name == "NAYY 4x1x300"
        assert num_parallel_cables == 5

        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo,
            "lv",
            0.18,
            length=1,
            max_voltage_diff=0.01,
            max_cables=100,
            component_type="conventional_load",
        )
        assert cable_data.name == "NAYY 4x1x300"
        assert num_parallel_cables == 14

        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo,
            "lv",
            0.18,
            length=1,
            max_voltage_diff=0.01,
            max_cables=100,
            component_type="generator",
        )
        assert cable_data.name == "NAYY 4x1x300"
        assert num_parallel_cables == 8

        try:
            tools.select_cable(
                self.edisgo,
                "lv",
                0.18,
                length=1,
                max_voltage_diff=0.01,
                max_cables=100,
                component_type="fail",
            )
        except ValueError as e:
            assert (
                str(e) == "Specified component type is not valid. "
                "Must either be 'generator', 'conventional_load', 'charging_point', "
                "'heat_pump' or 'storage_unit'."
            )

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
        edisgo_root = self.edisgo.copy()
        edisgo_root.topology.lines_df.loc["Line_10006", "b"] = 1
        edisgo_root.topology.lines_df.loc["Line_50000002", "b"] = 1
        edisgo_root = tools.add_line_susceptance(edisgo_root, mode="no_b")
        assert edisgo_root.topology.lines_df.loc["Line_10006", "b"] == 0
        assert edisgo_root.topology.lines_df.loc["Line_50000002", "b"] == 0

        # test mode mv_b
        edisgo_root = self.edisgo.copy()
        edisgo_root.topology.lines_df.loc["Line_10006", "b"] = 1
        edisgo_root.topology.lines_df.loc["Line_50000002", "b"] = 1
        edisgo_root = tools.add_line_susceptance(edisgo_root, mode="mv_b")
        assert edisgo_root.topology.lines_df.loc[
            "Line_10006", "b"
        ] == tools.calculate_line_susceptance(0.304, 0.297650465459542, 1)
        assert edisgo_root.topology.lines_df.loc["Line_50000002", "b"] == 0

        # test mode all_b
        edisgo_root = self.edisgo.copy()
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
