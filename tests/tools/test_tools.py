import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from edisgo.tools import tools
from edisgo.network.topology import Topology
from edisgo.io import ding0_import

class TestTools:

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
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(
            parent_dirname, 'ding0_test_network')
        self.topology = Topology()
        ding0_import.import_ding0_grid(test_network_directory, self)

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
                                        'Bus_primary_LVStation_7')
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
            parent_dirname, 'ding0_test_network')
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

