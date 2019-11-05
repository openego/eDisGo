import os
import pandas as pd

from edisgo.network.topology import Topology
from edisgo.network.timeseries import TimeSeriesControl, TimeSeries
from edisgo.data import import_data
from edisgo.network.components import Generator, Load, Switch
from edisgo.network.grids import LVGrid
from edisgo.tools.config import Config


class TestGrids:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.topology = Topology()
        import_data.import_ding0_grid(test_network_directory, self)

    def test_mv_grid(self):
        """Test MVGrid class getter, setter, methods"""

        mv_grid = self.topology.mv_grid

        # test getter
        assert mv_grid.id == 1
        assert mv_grid.nominal_voltage == 20
        assert len(list(mv_grid.lv_grids)) == 9
        assert isinstance(list(mv_grid.lv_grids)[0], LVGrid)

        assert len(mv_grid.buses_df.index) == 33
        assert 'Bus_BranchTee_MVGrid_1_7' in mv_grid.buses_df.index

        assert len(mv_grid.generators_df.index) == 9
        assert 'Generator_slack' not in mv_grid.generators_df.index
        assert 'Generator_1' in mv_grid.generators_df.index
        gen_list = list(mv_grid.generators)
        assert isinstance(gen_list[0], Generator)
        assert len(gen_list) == 9

        assert len(mv_grid.loads_df.index) == 1
        assert 'Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1' in \
               mv_grid.loads_df.index
        load_list = list(mv_grid.loads)
        assert isinstance(load_list[0], Load)
        assert len(load_list) == 1

        assert len(mv_grid.switch_disconnectors_df.index) == 2
        assert 'circuit_breaker_1' in mv_grid.switch_disconnectors_df.index
        switch_list = list(mv_grid.switch_disconnectors)
        assert isinstance(switch_list[0], Switch)
        assert len(switch_list) == 2

        assert sorted(mv_grid.weather_cells) == [1122074, 1122075]
        assert mv_grid.peak_generation_capacity == 22.075
        assert mv_grid.peak_generation_capacity_per_technology['solar'] == 4.6
        assert mv_grid.peak_load == 0.31
        assert mv_grid.peak_load_per_sector['retail'] == 0.31

    def test_lv_grid(self):
        """Test LVGrid class getter, setter, methods"""
        lv_grid = [_ for _ in self.topology.mv_grid.lv_grids if _.id == 3][0]

        assert isinstance(lv_grid, LVGrid)
        assert lv_grid.id == 3
        assert lv_grid.nominal_voltage == 0.4

        assert len(lv_grid.buses_df) == 13
        assert 'Bus_BranchTee_LVGrid_3_2' in lv_grid.buses_df.index

        assert len(lv_grid.generators_df.index) == 0
        gen_list = list(lv_grid.generators)
        assert len(gen_list) == 0

        assert len(lv_grid.loads_df.index) == 4
        assert 'Load_residential_LVGrid_3_2' in lv_grid.loads_df.index
        load_list = list(lv_grid.loads)
        assert isinstance(load_list[0], Load)
        assert len(load_list) == 4

        assert len(lv_grid.switch_disconnectors_df.index) == 0
        switch_list = list(lv_grid.switch_disconnectors)
        assert len(switch_list) == 0

        assert sorted(lv_grid.weather_cells) == []
        assert lv_grid.peak_generation_capacity == 0
        assert lv_grid.peak_generation_capacity_per_technology.empty
        assert lv_grid.peak_load == 0.054627
        assert lv_grid.peak_load_per_sector['agricultural'] == 0.051


