import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.network.components import Generator, Load, Storage, Switch


class TestComponents:
    # ToDo add tests for PotentialChargingParks

    @classmethod
    def setup_class(self):
        self.edisgo_obj = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo_obj.set_time_series_worst_case_analysis()

    def test_load_class(self):
        """Test Load class getter, setter, methods"""

        load = Load(id="Load_agricultural_LVGrid_1_1", edisgo_obj=self.edisgo_obj)

        # test getter
        assert load.id == "Load_agricultural_LVGrid_1_1"
        assert load.p_set == 0.0523
        assert load.annual_consumption == 238
        assert load.sector == "agricultural"
        assert load.bus == "Bus_BranchTee_LVGrid_1_2"
        assert str(load.grid) == "LVGrid_1"
        assert load.voltage_level == "lv"
        assert load.geom is None
        assert isinstance(load.active_power_timeseries, pd.Series)
        assert isinstance(load.reactive_power_timeseries, pd.Series)

        # test setter
        load.p_set = 0.06
        assert load.p_set == 0.06
        load.annual_consumption = 4
        assert load.annual_consumption == 4
        load.sector = "residential"
        assert load.sector == "residential"
        load.bus = "Bus_BranchTee_MVGrid_1_1"
        assert load.bus == "Bus_BranchTee_MVGrid_1_1"
        assert load.grid == self.edisgo_obj.topology.mv_grid
        assert load.voltage_level == "mv"
        msg = "Given bus ID does not exist."
        with pytest.raises(AttributeError, match=msg):
            load.bus = "None"
        # TODO: add test for active_power_timeseries and reactive_power_timeseries once
        #  implemented

    def test_generator_class(self):
        """Test Generator class getter, setter, methods"""

        gen = Generator(id="GeneratorFluctuating_7", edisgo_obj=self.edisgo_obj)
        # GeneratorFluctuating_7,Bus_GeneratorFluctuating_7,PQ,3,wind,1122075,wind_wind_onshore
        # test getter
        assert gen.id == "GeneratorFluctuating_7"
        assert gen.bus == "Bus_GeneratorFluctuating_7"
        assert gen.grid == self.edisgo_obj.topology.mv_grid
        assert gen.voltage_level == "mv"
        assert pytest.approx(gen.geom.x, abs=1e-10) == 7.97127568152858
        assert pytest.approx(gen.geom.y, abs=1e-10) == 48.0666552118727
        assert gen.nominal_power == 3
        assert gen.type == "wind"
        assert gen.subtype == "wind_wind_onshore"
        assert gen.weather_cell_id == 1122075
        assert isinstance(gen.active_power_timeseries, pd.Series)
        assert isinstance(gen.reactive_power_timeseries, pd.Series)

        # test setter
        gen.nominal_power = 4
        assert gen.nominal_power == 4
        gen.type = "solar"
        assert gen.type == "solar"
        gen.subtype = "rooftop"
        assert gen.subtype == "rooftop"
        gen.weather_cell_id = 2
        assert gen.weather_cell_id == 2
        gen.bus = "Bus_BranchTee_LVGrid_1_5"
        assert gen.bus == "Bus_BranchTee_LVGrid_1_5"
        assert str(gen.grid) == "LVGrid_1"
        assert gen.voltage_level == "lv"
        msg = "Given bus ID does not exist."
        with pytest.raises(AttributeError, match=msg):
            gen.bus = "None"

    def test_storage_class(self):
        """Test Storage class getter, setter, methods"""

        gen = Storage(id="Storage_1", edisgo_obj=self.edisgo_obj)
        assert gen.id == "Storage_1"
        assert gen.bus == "Bus_MVStation_1"
        assert gen.grid == self.edisgo_obj.topology.mv_grid
        assert gen.voltage_level == "mv"
        assert pytest.approx(gen.geom.x, abs=1e-10) == 7.94859122759009
        assert pytest.approx(gen.geom.y, abs=1e-10) == 48.0844553685898
        assert gen.nominal_power == 0.4
        assert isinstance(gen.active_power_timeseries, pd.Series)
        assert isinstance(gen.reactive_power_timeseries, pd.Series)

        # test setter
        gen.nominal_power = 4
        assert gen.nominal_power == 4
        gen.bus = "Bus_BranchTee_LVGrid_1_5"
        assert gen.bus == "Bus_BranchTee_LVGrid_1_5"
        assert str(gen.grid) == "LVGrid_1"
        assert gen.voltage_level == "lv"
        msg = "Given bus ID does not exist."
        with pytest.raises(AttributeError, match=msg):
            gen.bus = "None"

    def test_switch_class(self):
        """Test Switch class"""

        switch = Switch(id="circuit_breaker_1", edisgo_obj=self.edisgo_obj)

        # test getter
        assert switch.id == "circuit_breaker_1"
        assert switch.bus_closed == "BusBar_MVGrid_1_LVGrid_4_MV"
        assert switch.bus_open == "virtual_BusBar_MVGrid_1_LVGrid_4_MV"
        assert switch.branch == "Line_10031"
        assert switch.type == "Switch Disconnector"
        assert switch.state == "open"
        assert switch.grid == self.edisgo_obj.topology.mv_grid
        assert switch.voltage_level == "mv"

        # test setter
        switch.type = "test"
        assert switch.type == "test"

        # test methods
        switch.close()
        switch._state = None
        assert switch.state == "closed"
        switch.open()
        switch._state = None
        assert switch.state == "open"
