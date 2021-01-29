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
        assert rel_line_load.shape == (2, 198)

        # test with providing lines
        rel_line_load = tools.calculate_relative_line_load(
            self.edisgo,
            lines=["Line_10005", "Line_50000002", "Line_90000025"])
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

    def test_check_bus_for_removal(self):
        self.topology = Topology()
        ding0_import.import_ding0_grid(pytest.ding0_test_network_path, self)

        # check for assertion
        msg = "Bus of name Test_bus_to_remove not in Topology. " \
              "Cannot be checked to be removed."
        with pytest.raises(ValueError, match=msg):
            tools.check_bus_for_removal(self.topology, 'Test_bus_to_remove')

        # check buses with connected elements
        assert not \
            tools.check_bus_for_removal(self.topology, 'Bus_Generator_1')
        assert not \
            tools.check_bus_for_removal(self.topology,
                                        'Bus_Load_agricultural_LVGrid_1_1')
        assert not \
            tools.check_bus_for_removal(self.topology,
                                        'BusBar_MVGrid_1_LVGrid_7_MV')
        assert not \
            tools.check_bus_for_removal(self.topology,
                                        'Bus_BranchTee_MVGrid_1_3')

        # add bus and line that could be removed
        self.topology.add_bus(bus_name='Test_bus_to_remove', v_nom=20)
        self.topology.add_line(bus0='Bus_MVStation_1',
                               bus1='Test_bus_to_remove',
                               length=1.0)
        assert self.topology.lines_df.at[
                   'Line_Bus_MVStation_1_Test_bus_to_remove', 'length'] == 1
        assert tools.check_bus_for_removal(self.topology, 'Test_bus_to_remove')

    def test_check_line_for_removal(self):
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(
            parent_dirname, 'ding0_test_network_1')
        self.topology = Topology()
        ding0_import.import_ding0_grid(test_network_directory, self)

        # check for assertion
        msg = "Line of name Test_line_to_remove not in Topology. " \
              "Cannot be checked to be removed."
        with pytest.raises(ValueError, match=msg):
            tools.check_line_for_removal(self.topology, 'Test_line_to_remove')

        # check lines with connected elements
        # transformer
        assert not tools.check_line_for_removal(self.topology, 'Line_10024')
        # generator
        assert not tools.check_line_for_removal(self.topology, 'Line_10032')
        # load
        assert not tools.check_line_for_removal(self.topology, 'Line_10000021')
        # check for lines that could be removed
        # Todo: this case would create subnetworks, still has to be implemented
        assert tools.check_line_for_removal(self.topology, 'Line_10014')

        # create line that can be removed safely
        self.topology.add_bus(bus_name='testbus', v_nom=20)
        self.topology.add_bus(bus_name='testbus2', v_nom=20)
        line_name = self.topology.add_line(bus0='testbus', bus1='testbus2', length=2.3)
        assert tools.check_line_for_removal(self.topology, line_name)
        self.topology.remove_line(line_name)

    def test_assign_feeder(self):

        # ######## test MV feeder mode ########
        tools.assign_feeder(self.edisgo, mode="mv_feeder")

        topo = self.edisgo.topology

        # check that all lines and all buses (except MV station bus) have an
        # MV feeder assigned
        assert not topo.lines_df.mv_feeder.isna().any()
        assert not topo.buses_df[
            ~topo.buses_df.index.isin(
                topo.mv_grid.station.index)].mv_feeder.isna().any()

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
            "Bus_Load_residential_LVGrid_2_2", "lv_feeder"] ==
                "Bus_BranchTee_LVGrid_2_1")

        # check specific lines
        assert topo.lines_df.at[
            "Line_30000005", "lv_feeder"] == "Bus_BranchTee_LVGrid_3_3"
        assert topo.lines_df.at[
            "Line_40000001", "lv_feeder"] == "Bus_GeneratorFluctuating_16"
