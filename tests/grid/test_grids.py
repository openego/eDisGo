import os
import numpy as np
import pandas as pd

from edisgo.grid.network import Network, TimeSeriesControl
from edisgo.data import import_data
from edisgo.grid.components import Generator, Load, Switch
from edisgo.grid.grids import LVGrid


class TestGrids:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.network = Network()
        import_data.import_ding0_grid(test_network_directory, self.network)

    def test_mv_grid(self):
        """Test MVGrid class getter, setter, methods"""

        mv_grid = self.network.mv_grid

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

    def test_mv_grid_to_pypsa(self):
        TimeSeriesControl(network=self.network, mode='worst-case')
        # ToDo: Remove and convert into csv table
        omega = 2 * np.pi * 50
        valid_lines = self.network.equipment_data['mv_lines'][
            self.network.equipment_data['mv_lines'].U_n ==
            self.network.buses_df.v_nom.iloc[0]]
        std_line = valid_lines.loc[valid_lines.I_max_th.idxmin()]
        self.network.lines_df[np.isnan(self.network.lines_df.x)] = \
            self.network.lines_df[
                np.isnan(self.network.lines_df.x)].assign(
                num_parallel=1,
                r=lambda _: _.length * std_line.loc['R_per_km'],
                x=lambda _: _.length * std_line.loc['L_per_km'] * omega / 1e3,
                s_nom=np.sqrt(3) * std_line.loc['I_max_th'] *
                      std_line.loc['U_n'] / 1e3,
                type_info=std_line.name)
        # run powerflow and check results
        timesteps = pd.date_range('1/1/1970', periods=1, freq='H')
        pypsa_network = self.network.mv_grid.to_pypsa()
        pf_results = pypsa_network.pf(timesteps)

        if all(pf_results['converged']['0'].tolist()):
            print('converged mv')
        else:
            raise ValueError("Power flow analysis mv did not converge.")

        pypsa_network = self.network.mv_grid.to_pypsa(mode='mvlv')
        pf_results = pypsa_network.pf(timesteps)

        if all(pf_results['converged']['0'].tolist()):
            print('converged mvlv')
        else:
            raise ValueError("Power flow analysis mvlv did not converge.")


    def test_lv_grid(self):
        """Test LVGrid class getter, setter, methods"""
        lv_grid = [_ for _ in self.network.mv_grid.lv_grids if _.id == 3][0]

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





