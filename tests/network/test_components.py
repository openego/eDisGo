import pytest
import os
import math

from edisgo.grid.network import Network
from edisgo.grid.components import Load, Generator, Storage, Switch
from edisgo.data import import_data


class TestImportFromDing0:
    #ToDo add tests for switches_df and storages_df

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.network = Network()
        import_data.import_ding0_grid(test_network_directory, self.network)

    def test_load_class(self):
        """Test Load class getter, setter, methods"""

        load = Load(id='Load_agricultural_LVGrid_1_1', network=self.network)

        # test getter
        assert load.id == 'Load_agricultural_LVGrid_1_1'
        assert load.peak_load == 0.0523
        assert math.isnan(load.annual_consumption)
        assert load.sector == 'agricultural'
        assert load.bus == 'Bus_Load_agricultural_LVGrid_1_1'
        assert load.grid == self.network._grids['LVGrid_1.0']
        assert load.voltage_level == 'lv'
        assert load.geom == None
        #ToDo add test for active_power_timeseries and reactive_power_timeseries once implemented

        # test setter
        load.peak_load = 0.06
        assert load.peak_load == 0.06
        load.annual_consumption = 4
        assert load.annual_consumption == 4
        load.sector = 'residential'
        assert load.sector == 'residential'
        load.bus = 'Bus_BranchTee_MVGrid_1_1'
        assert load.bus == 'Bus_BranchTee_MVGrid_1_1'
        assert load.grid == self.network.mv_grid
        assert load.voltage_level == 'mv'
        msg = "Given bus ID does not exist."
        with pytest.raises(AttributeError, match=msg):
            load.bus = 'None'
        # ToDo add test for active_power_timeseries and reactive_power_timeseries once implemented

    def test_switch_class(self):
        """Test Switch class"""

        switch = Switch(id='circuit_breaker_1', network=self.network)

        # test getter
        assert switch.id == 'circuit_breaker_1'
        assert switch.bus_closed == 'Bus_primary_LVStation_5'
        assert switch.bus_open == 'virtual_Bus_primary_LVStation_5'
        assert switch.branch == 'Line_10012'
        assert switch.type == 'Switch Disconnector'
        assert switch.state == 'closed'
        assert switch.grid == self.network.mv_grid

        # test setter
        # test methods
        switch.open()
        switch._state = None
        assert switch.state == 'open'
        switch.close()
        switch._state = None
        assert switch.state == 'closed'



