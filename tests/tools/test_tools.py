import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from math import sqrt

from edisgo import EDisGo
from edisgo.tools import tools
from edisgo.network.topology import Topology
from edisgo.io import ding0_import


class TestTools:

    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_path,
            worst_case_analysis="worst-case"
        )
        self.timesteps = self.edisgo.timeseries.timeindex
        self.edisgo.analyze()

    def test_calculate_relative_line_load(self):
        # test without providing lines and time steps
        rel_line_load = tools.calculate_relative_line_load(
            self.edisgo)
        assert rel_line_load.shape == (2, 129)

        # test with providing lines
        rel_line_load = tools.calculate_relative_line_load(
            self.edisgo,
            lines=["Line_10005", "Line_50000002", "Line_90000021"])
        assert rel_line_load.shape == (2, 3)
        assert np.isclose(
            rel_line_load.at[self.timesteps[0], "Line_10005"],
            self.edisgo.results.i_res.at[self.timesteps[0], "Line_10005"]
            / (7.274613391789284 / 20 / sqrt(3))
        )
        assert np.isclose(
            rel_line_load.at[self.timesteps[1], "Line_50000002"],
            self.edisgo.results.i_res.at[self.timesteps[1], "Line_50000002"]
            / (0.08521689973238901 / 0.4 / sqrt(3)),
        )

        # test with providing lines and timesteps
        rel_line_load = tools.calculate_relative_line_load(
            self.edisgo,
            lines=["Line_10005"],
            timesteps=self.timesteps[0])
        assert rel_line_load.shape == (1, 1)
        assert np.isclose(
            rel_line_load.at[self.timesteps[0], "Line_10005"],
            self.edisgo.results.i_res.at[self.timesteps[0], "Line_10005"]
            / (7.274613391789284 / 20 / sqrt(3))
        )

    def test_calculate_line_reactance(self):
        data = tools.calculate_line_reactance(2, 3)
        assert np.isclose(data, 1.88496)
        data = tools.calculate_line_reactance(np.array([2, 3]), 3)
        assert_allclose(data, np.array([1.88496, 2.82743]), rtol=1e-5)

    def test_calculate_line_resistance(self):
        data = tools.calculate_line_resistance(2, 3)
        assert data == 6
        data = tools.calculate_line_resistance(np.array([2, 3]), 3)
        assert_array_equal(data, np.array([6, 9]))

    def test_calculate_apparent_power(self):
        data = tools.calculate_apparent_power(20, 30)
        assert np.isclose(data, 1039.23)
        data = tools.calculate_apparent_power(30, np.array([20, 30]))
        assert_allclose(data, np.array([1039.23, 1558.84]), rtol=1e-5)
        data = tools.calculate_apparent_power(np.array([30, 30]),
                                              np.array([20, 30]))
        assert_allclose(data, np.array([1039.23, 1558.84]), rtol=1e-5)

    def test_select_cable(self):
        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo, 'mv', 5.1)
        assert cable_data.name == "NA2XS2Y 3x1x150 RE/25"
        assert num_parallel_cables == 1

        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo, 'mv', 40)
        assert cable_data.name == "NA2XS(FL)2Y 3x1x500 RM/35"
        assert num_parallel_cables == 2

        cable_data, num_parallel_cables = tools.select_cable(
            self.edisgo, 'lv', 0.18)
        assert cable_data.name == "NAYY 4x1x150"
        assert num_parallel_cables == 1

    def test_assign_feeder(self):

        # ######## test MV feeder mode ########
        tools.assign_feeder(self.edisgo, mode="mv_feeder")

        topo = self.edisgo.topology

        # check that all lines and all buses (except MV station bus and buses
        # in aggregated load areas) have an MV feeder assigned
        assert not topo.lines_df.mv_feeder.isna().any()
        mv_station = topo.mv_grid.station.index[0]
        buses_aggr_la = list(topo.transformers_df[
            topo.transformers_df.bus0 == mv_station].bus1.unique())
        buses_aggr_la.append(mv_station)
        assert not topo.buses_df[
            ~topo.buses_df.index.isin(
                buses_aggr_la)].mv_feeder.isna().any()

        # check specific buses
        # MV and LV bus in feeder 1
        assert (topo.buses_df.at[
                   "Bus_GeneratorFluctuating_7", "mv_feeder"] ==
                "Bus_BranchTee_MVGrid_1_6")
        assert (topo.buses_df.at[
                   "Bus_BranchTee_LVGrid_4_1", "mv_feeder"] ==
                "Bus_BranchTee_MVGrid_1_5")
        # MV bus in feeder 2
        assert (topo.buses_df.at[
                   "Bus_GeneratorFluctuating_3", "mv_feeder"] ==
                "Bus_BranchTee_MVGrid_1_5")

        # check specific lines
        assert topo.lines_df.at[
                   "Line_10003", "mv_feeder"] == "Bus_BranchTee_MVGrid_1_1"

        # ######## test LV feeder mode ########
        tools.assign_feeder(self.edisgo, mode="lv_feeder")

        topo = self.edisgo.topology

        # check that all buses (except LV station buses) and lines in LV have
        # an LV feeder assigned
        mv_lines = topo.mv_grid.lines_df.index
        assert not topo.lines_df[
                   ~topo.lines_df.index.isin(
                       mv_lines)].lv_feeder.isna().any()
        mv_buses = list(topo.mv_grid.buses_df.index)
        mv_buses.extend(topo.transformers_df.bus1)
        assert not topo.buses_df[
                   ~topo.buses_df.index.isin(
                       mv_buses)].lv_feeder.isna().any()

        # check specific buses
        assert (topo.buses_df.at[
            "Bus_BranchTee_LVGrid_1_8", "lv_feeder"] ==
                "Bus_BranchTee_LVGrid_1_7")
        assert (topo.buses_df.at[
            "Bus_BranchTee_LVGrid_2_4", "lv_feeder"] ==
                "Bus_BranchTee_LVGrid_2_1")

        # check specific lines
        assert topo.lines_df.at[
            "Line_30000005", "lv_feeder"] == "Bus_BranchTee_LVGrid_3_3"
        assert topo.lines_df.at[
            "Line_40000001", "lv_feeder"] == "Bus_GeneratorFluctuating_16"

        # ######## test real ding0 network ########
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path,
            worst_case_analysis="worst-case"
        )
        topo = self.edisgo.topology

        tools.assign_feeder(self.edisgo, mode="mv_feeder")

        # check that all lines and all buses (except MV station bus and buses
        # in aggregated load areas) have an MV feeder assigned
        assert not topo.lines_df.mv_feeder.isna().any()
        mv_station = topo.mv_grid.station.index[0]
        buses_aggr_la = list(
            topo.transformers_df[
                topo.transformers_df.bus0 == mv_station].bus1.unique())
        buses_aggr_la.append(mv_station)
        assert not topo.buses_df[
            ~topo.buses_df.index.isin(
                buses_aggr_la)].mv_feeder.isna().any()

        tools.assign_feeder(self.edisgo, mode="lv_feeder")
        # check that all buses (except LV station buses) and lines in LV have
        # an LV feeder assigned
        mv_lines = topo.mv_grid.lines_df.index
        assert not topo.lines_df[
            ~topo.lines_df.index.isin(
                mv_lines)].lv_feeder.isna().any()
        mv_buses = list(topo.mv_grid.buses_df.index)
        mv_buses.extend(topo.transformers_df.bus1)
        assert not topo.buses_df[
            ~topo.buses_df.index.isin(
                mv_buses)].lv_feeder.isna().any()

    def test_get_weather_cells_intersecting_with_grid_district(self):
        weather_cells = \
            tools.get_weather_cells_intersecting_with_grid_district(
                self.edisgo)
        assert len(weather_cells) == 4
        assert 1123075 in weather_cells
        assert 1122075 in weather_cells
        assert 1122076 in weather_cells
        # the following weather cell does not intersect with the grid district
        # but there are generators in the grid that have that weather cell
        # for some reason..
        assert 1122074 in weather_cells
