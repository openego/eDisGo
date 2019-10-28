import pytest
import os
import math

from edisgo.grid.network import Network
from edisgo.grid.components import Load, Generator, Storage, Switch
from edisgo.data import import_data


class TestComponents:
    #ToDo add tests for storages_df

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
        assert load.annual_consumption == 238
        assert load.sector == 'agricultural'
        assert load.bus == 'Bus_Load_agricultural_LVGrid_1_1'
        assert load.grid == self.network._grids['LVGrid_1']
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

    def test_generator_class(self):
        """Test Generator class getter, setter, methods"""

        gen = Generator(id='GeneratorFluctuating_7', network=self.network)
        #GeneratorFluctuating_7,Bus_GeneratorFluctuating_7,PQ,3,wind,1122075,wind_wind_onshore
        # test getter
        assert gen.id == 'GeneratorFluctuating_7'
        assert gen.bus == 'Bus_GeneratorFluctuating_7'
        assert gen.grid == self.network.mv_grid
        assert gen.voltage_level == 'mv'
        assert pytest.approx(gen.geom.x, abs=1e-10) == 7.97127568152858
        assert pytest.approx(gen.geom.y, abs=1e-10) == 48.0666552118727
        assert gen.nominal_power == 3
        assert gen.type == 'wind'
        assert gen.subtype == 'wind_wind_onshore'
        assert gen.weather_cell_id == 1122075
        #ToDo add test for active_power_timeseries and reactive_power_timeseries once implemented

        # test setter
        gen.nominal_power = 4
        assert gen.nominal_power == 4
        gen.type = 'solar'
        assert gen.type == 'solar'
        gen.subtype = 'rooftop'
        assert gen.subtype == 'rooftop'
        gen.weather_cell_id = 2
        assert gen.weather_cell_id == 2
        gen.bus = 'Bus_GeneratorFluctuating_9'
        assert gen.bus == 'Bus_GeneratorFluctuating_9'
        assert gen.grid == self.network._grids['LVGrid_1']
        assert gen.voltage_level == 'lv'
        msg = "Given bus ID does not exist."
        with pytest.raises(AttributeError, match=msg):
            gen.bus = 'None'
        # ToDo add test for active_power_timeseries and reactive_power_timeseries once implemented

    def test_switch_class(self):
        """Test Switch class"""

        switch = Switch(id='circuit_breaker_1', network=self.network)

        # test getter
        assert switch.id == 'circuit_breaker_1'
        assert switch.bus_closed == 'Bus_primary_LVStation_4'
        assert switch.bus_open == 'virtual_Bus_primary_LVStation_4'
        assert switch.branch == 'Line_10031'
        assert switch.type == 'Switch Disconnector'
        assert switch.state == 'open'
        assert switch.grid == self.network.mv_grid
        assert switch.voltage_level == 'mv'

        # test setter
        switch.type = 'test'
        assert switch.type == 'test'

        # test methods
        switch.close()
        switch._state = None
        assert switch.state == 'closed'
        switch.open()
        switch._state = None
        assert switch.state == 'open'




