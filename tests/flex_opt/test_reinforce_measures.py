import os
import pandas as pd
import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.flex_opt.reinforce_measures import \
    extend_distribution_substation_overloading, extend_substation_overloading, \
    reinforce_branches_overloading, reinforce_branches_overvoltage


def change_line_to_standard_line(test_class, line_name, std_line):
    omega = 2 * np.pi * 50
    std_line.X_per_km = std_line.L_per_km / 1e3 * omega
    std_line.S_nom = np.sqrt(3) * std_line.U_n * std_line.I_max_th
    test_class.edisgo.topology.lines_df.loc[line_name, 'type_info'] = \
        std_line.name
    test_class.edisgo.topology.lines_df.loc[line_name, 'r'] = \
        std_line.R_per_km * test_class.edisgo.topology.lines_df.loc[
            line_name, 'length']
    test_class.edisgo.topology.lines_df.loc[line_name, 'x'] = \
        std_line.X_per_km * test_class.edisgo.topology.lines_df.loc[
            line_name, 'length']
    test_class.edisgo.topology.lines_df.loc[line_name, 's_nom'] = \
        std_line.S_nom

class TestReinforceMeasures:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        dirname = os.path.realpath(os.path.dirname(__file__)+'/..')
        test_network_directory = os.path.join(dirname, 'ding0_test_network')
        self.edisgo = EDisGo(ding0_grid=test_network_directory,
                             worst_case_analysis='worst-case')
        self.edisgo.analyze()
        self.timesteps = pd.date_range('1/1/1970', periods=2, freq='H')

    def test_extend_distribution_substation_overloading(self):
        crit_lv_stations = pd.DataFrame({'s_pfa': [0.1792, 0.3321],
                                         'time_index': [self.timesteps[1]]*2},
                                        index=['LVGrid_1', 'LVGrid_5'])
        transformer_changes = \
            extend_distribution_substation_overloading(self.edisgo,
                                                       crit_lv_stations)
        assert transformer_changes['added']['LVGrid_1'][0] == \
            'LVStation_1_transformer_reinforced_2'
        assert transformer_changes['added']['LVGrid_5'][0] == \
            'LVStation_5_transformer_reinforced_1'
        assert transformer_changes['removed']['LVGrid_5'][0] == \
            'LVStation_5_transformer_1'
        assert ('LVStation_5_transformer_1' not in \
             self.edisgo.topology.transformers_df.index)
        trafo1 = self.edisgo.topology.transformers_df.loc[
            'LVStation_1_transformer_reinforced_2']
        assert trafo1.bus0 == 'Bus_primary_LVStation_1'
        assert trafo1.bus1 == 'Bus_secondary_LVStation_1'
        assert trafo1.r_pu == 0.00588
        assert trafo1.x_pu == 0.016
        assert trafo1.s_nom == 0.16
        assert trafo1.type_info == '160 kVA'
        trafo2 = self.edisgo.topology.transformers_df.loc[
            'LVStation_5_transformer_reinforced_1']
        assert trafo2.bus0 == 'Bus_primary_LVStation_5'
        assert trafo2.bus1 == 'Bus_secondary_LVStation_5'
        assert trafo2.r_pu == 0.010317460317460317
        assert trafo2.x_pu == 0.03864647477581405
        assert trafo2.s_nom == 0.63
        assert trafo2.type_info == '630 kVA'
        print('Reinforcement of LV Stations successful.')

    def test_extend_substation_overloading(self):
        self.edisgo.topology.transformers_hvmv_df.loc[
            'MVStation_1_transformer_1', 's_nom'] = 20
        self.edisgo.topology.transformers_hvmv_df.loc[
            'MVStation_1_transformer_1', 'type_info'] = 'dummy'
        crit_mv_station = pd.DataFrame({'s_pfa': [23.8241],
                                        'time_index': [self.timesteps[1]]},
                                       index=['MVGrid_1'])
        transformer_changes = \
            extend_substation_overloading(self.edisgo, crit_mv_station)
        assert transformer_changes['added']['MVGrid_1'][0] == \
            'MVStation_1_transformer_reinforced_2'
        trafo = self.edisgo.topology.transformers_hvmv_df.loc[
            'MVStation_1_transformer_reinforced_2']
        assert trafo.bus0 == 'Bus_primary_MVStation_1'
        assert trafo.bus1 == 'Bus_MVStation_1'
        assert np.isnan(trafo.x_pu)
        assert np.isnan(trafo.r_pu)
        assert trafo.s_nom == 20
        assert trafo.type_info == 'dummy'
        print('Extend substation successful.')
        # check if transformer have to be removed
        crit_mv_station.loc['MVGrid_1', 's_pfa'] = 35.45
        crit_mv_station.loc['MVGrid_1', 'time_index'] = self.timesteps[1]
        msg = 'Missing load is negative. Something went wrong. Please report.'
        with pytest.raises(ValueError, match=msg):
            extend_substation_overloading(self.edisgo, crit_mv_station)
        crit_mv_station.loc['MVGrid_1', 'time_index'] = self.timesteps[0]
        transformer_changes = \
            extend_substation_overloading(self.edisgo, crit_mv_station)
        assert len(transformer_changes['added']['MVGrid_1']) == 2
        assert len(transformer_changes['removed']['MVGrid_1']) == 2
        trafos = self.edisgo.topology.transformers_hvmv_df
        assert (trafos.bus0 == 'Bus_primary_MVStation_1').all()
        assert (trafos.bus1 == 'Bus_MVStation_1').all()
        assert (trafos.s_nom == 40).all()
        assert (trafos.type_info == '40 MVA').all()
        print('Extend substation by replacing trafos successful.')

    def test_reinforce_branches_overloading(self):
        std_line_mv = self.edisgo.equipment_data['mv_cables'].loc[
            self.edisgo.config['grid_expansion_standard_equipment'][
                'mv_line']]
        std_line_mv.U_n = self.edisgo.topology.mv_grid.nominal_voltage

        std_line_lv = self.edisgo.equipment_data['lv_cables'].loc[
            self.edisgo.config['grid_expansion_standard_equipment'][
                'lv_line']]
        # manipulate lines to sdt_line_type
        change_line_to_standard_line(self, 'Line_10017', std_line_mv)
        change_line_to_standard_line(self, 'Line_50000002', std_line_lv)

        # create crit_lines dataframe
        crit_lines = pd.DataFrame({
            'max_rel_overload': [2.3261, 2.3261, 1.1792, 2.02, 1.02, 3.32],
            'grid_level': ['mv'] * 3 + ['lv'] * 3},
            index=['Line_10003', 'Line_10017', 'Line_10019',
                   'Line_50000002', 'Line_50000004', 'Line_60000001'])

        lines_changes = reinforce_branches_overloading(self.edisgo, crit_lines)

        # assert values
        assert len(lines_changes) == 6
        assert lines_changes['Line_10003'] == 2
        assert lines_changes['Line_10017'] == 2
        assert lines_changes['Line_10019'] == 1
        assert lines_changes['Line_50000002'] == 2
        assert lines_changes['Line_50000004'] == 1
        assert lines_changes['Line_60000001'] == 2
        # check line default
        line_mv_1 = self.edisgo.topology.lines_df.loc['Line_10003']
        assert line_mv_1.type_info == std_line_mv.name
        assert line_mv_1.r == std_line_mv.R_per_km * line_mv_1.length / 2
        assert line_mv_1.x == std_line_mv.X_per_km * line_mv_1.length / 2
        assert line_mv_1.s_nom == std_line_mv.S_nom
        assert line_mv_1.num_parallel == 2
        # check line standard
        line_mv_2 = self.edisgo.topology.lines_df.loc['Line_10017']
        assert line_mv_2.type_info == std_line_mv.name
        assert line_mv_2.r == std_line_mv.R_per_km * line_mv_2.length / 3
        assert line_mv_2.x == std_line_mv.X_per_km * line_mv_2.length / 3
        assert line_mv_2.s_nom == std_line_mv.S_nom
        assert line_mv_2.num_parallel == 3
        # check line single
        line_mv_3 = self.edisgo.topology.lines_df.loc['Line_10019']
        assert line_mv_3.type_info == '48-AL1/8-ST1A'
        assert line_mv_3.r == 0.074712625365
        assert line_mv_3.x == 0.07485557547918019
        assert line_mv_3.s_nom == 7.274613391789284
        assert line_mv_3.num_parallel == 2
        # check lv_lines
        line_lv_1 = self.edisgo.topology.lines_df.loc['Line_50000002']
        assert line_lv_1.type_info == std_line_lv.name
        assert line_lv_1.r == std_line_lv.R_per_km * line_lv_1.length / 3
        assert line_lv_1.x == std_line_lv.X_per_km * line_lv_1.length / 3
        assert line_lv_1.s_nom == std_line_lv.S_nom
        assert line_lv_1.num_parallel == 3
        # check single
        line_lv_2 = self.edisgo.topology.lines_df.loc['Line_50000004']
        assert line_lv_2.type_info == 'NAYY 4x1x35'
        assert line_lv_2.r == 0.000434
        assert line_lv_2.x == 4.2568580456141706e-05
        assert line_lv_2.s_nom == 0.08521689973238901
        assert line_lv_2.num_parallel == 2
        # check default
        line_lv_3 = self.edisgo.topology.lines_df.loc['Line_60000001']
        assert line_lv_3.type_info == std_line_lv.name
        assert line_lv_3.r == std_line_lv.R_per_km * line_lv_3.length / 2
        assert line_lv_3.x == std_line_lv.X_per_km * line_lv_3.length / 2
        assert line_lv_3.s_nom == std_line_lv.S_nom
        assert line_lv_3.num_parallel == 2
        print('Check line overload reinforcement successful.')

    def test_reinforce_branches_overvoltage(self):
        crit_nodes = pd.DataFrame({
            'v_mag_pu': [0.08, 0.06, 0.05, 0.04, 0.01],
            'time_index': [self.timesteps[1], self.timesteps[0],
                           self.timesteps[0], self.timesteps[0],
                           self.timesteps[1]]},
            index=['Bus_GeneratorFluctuating_6', 'Bus_GeneratorFluctuating_5',
                   'Bus_GeneratorFluctuating_3', 'Bus_GeneratorFluctuating_2',
                   'Bus_BranchTee_MVGrid_1_4'])

        lines_changes = reinforce_branches_overvoltage(self.edisgo,
                                    self.edisgo.topology.mv_grid, crit_nodes)

        assert len(lines_changes) == 4
        #Todo: erweitern (values and LV)
        print()
