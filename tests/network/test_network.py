import pytest
import shapely
import os
import numpy as np
import pandas as pd

from edisgo import EDisGo
from edisgo.grid.network import Network
from edisgo.grid.grids import MVGrid, LVGrid
from edisgo.data import import_data
from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.grid.network import TimeSeriesControl


class TestNetwork:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(parent_dirname, 'test_network')
        self.network = Network()
        import_data.import_ding0_grid(test_network_directory, self.network)
        self.network.timeseries = TimeSeriesControl(
            network=self.network,
            mode='worst-case').timeseries

    def test_reinforce(self):

        # ToDo: Remove and convert into csv table
        omega = 2 * np.pi * 50
        valid_lines = self.network.equipment_data['mv_lines'][
            self.network.equipment_data['mv_lines'].U_n ==
            self.network.buses_df.v_nom.iloc[0]]
        std_line = valid_lines.loc[valid_lines.I_max_th.idxmin()]
        self.network.lines_df[np.isnan(self.network.lines_df.x)] = self.network.lines_df[
            np.isnan(self.network.lines_df.x)].assign(num_parallel=1,
                  r=lambda _: _.length*std_line.loc['R_per_km'],
                  x=lambda _: _.length*std_line.loc['L_per_km']*omega/1e3,
                  s_nom=np.sqrt(3)*std_line.loc['I_max_th']*std_line.loc['U_n']/1e3,
                  type_info=std_line.name)

        timesteps = pd.date_range('1/1/1970', periods=1, freq='H')
        path = 'C:/Users/Anya.Heider/open_BEA/eDisGo/tests/test_network'
        edisgo = EDisGo(ding0_grid=path, worst_case_analysis='worst-case')
        pypsa_network = edisgo.network.to_pypsa(timesteps=timesteps)
