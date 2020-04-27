import os
import pandas as pd
import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.flex_opt.reinforce_measures import \
    extend_distribution_substation_overloading, extend_substation_overloading, \
    reinforce_branches_overloading, reinforce_branches_overvoltage, \
    extend_distribution_substation_overvoltage


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

    def test_reinforce_mv_lv_station_overloading(self):
        # create problems such that in LVGrid_1 existing transformer is
        # exchanged with standard transformer and in LVGrid_4 a third
        # transformer is added
        crit_lv_stations = pd.DataFrame(
            {
                "s_missing": [0.17, 0.04],
                "time_index": [self.timesteps[1]] * 2,
            },
            index=["LVGrid_1", "LVGrid_4"],
        )
        transformer_changes = reinforce_measures.reinforce_mv_lv_station_overloading(
            self.edisgo, crit_lv_stations
        )

        # check transformer changes
        assert len(transformer_changes["added"].keys()) == 2
        assert len(transformer_changes["removed"].keys()) == 1

        assert (
            transformer_changes["added"]["LVGrid_1"][0]
            == "LVStation_1_transformer_reinforced_1"
        )
        assert (
            transformer_changes["removed"]["LVGrid_1"][0]
            == "LVStation_1_transformer_1"
        )
        assert (
            transformer_changes["added"]["LVGrid_4"][0]
            == "LVStation_4_transformer_reinforced_3"
        )

        # check that removed transformer is removed from topology
        assert (
            "LVStation_1_transformer_1"
            not in self.edisgo.topology.transformers_df.index
        )

        # check values of transformers
        trafo_new = self.edisgo.topology.transformers_df.loc[
            "LVStation_1_transformer_reinforced_1"
        ]
        trafo_copy = self.edisgo.topology.equipment_data[
            "lv_transformers"].loc["630 kVA"]
        assert trafo_new.bus0 == "Bus_primary_LVStation_1"
        assert trafo_new.bus1 == "Bus_secondary_LVStation_1"
        assert trafo_new.r_pu == trafo_copy.r_pu
        assert trafo_new.x_pu == trafo_copy.x_pu
        assert trafo_new.s_nom == trafo_copy.S_nom
        assert trafo_new.type_info == "630 kVA"

        trafo_new = self.edisgo.topology.transformers_df.loc[
            "LVStation_4_transformer_reinforced_3"
        ]
        trafo_copy = self.edisgo.topology.transformers_df.loc[
            "LVStation_4_transformer_1"
        ]
        assert trafo_new.bus0 == "Bus_primary_LVStation_4"
        assert trafo_new.bus1 == "Bus_secondary_LVStation_4"
        assert trafo_new.r_pu == trafo_copy.r_pu
        assert trafo_new.x_pu == trafo_copy.x_pu
        assert trafo_new.s_nom == trafo_copy.s_nom
        assert trafo_new.type_info == "40 kVA"

    def test_extend_substation_overloading(self):
        # manipulate HV/MV transformer so that it is over-loaded (s_nom before
        # is 40 MVA)
        self.edisgo.topology.transformers_hvmv_df.loc[
            'MVStation_1_transformer_1', 's_nom'] = 20
        self.edisgo.topology.transformers_hvmv_df.loc[
            'MVStation_1_transformer_1', 'type_info'] = 'dummy'

        # check adding transformer of same MVA
        # set up made up critical station df such that adding a transformer
        # of the same size is sufficient
        crit_mv_station = pd.DataFrame({'s_pfa': [19],
                                        'time_index': [self.timesteps[1]]},
                                       index=['MVGrid_1'])
        transformer_changes = \
            extend_substation_overloading(self.edisgo, crit_mv_station)
        assert len(transformer_changes['added']['MVGrid_1']) == 1
        assert len(transformer_changes['removed']) == 0
        trafos = self.edisgo.topology.transformers_hvmv_df
        assert (trafos.bus0 == 'Bus_primary_MVStation_1').all()
        assert (trafos.bus1 == 'Bus_MVStation_1').all()
        assert (trafos.s_nom == 20).all()
        assert (trafos.type_info == 'dummy').all()
        assert 'MVStation_1_transformer_reinforced_2' in \
               transformer_changes['added']['MVGrid_1']
        print('Extend substation successful.')

        # check error message in case current transformers are sufficient
        msg = 'Missing load is negative. Something went wrong. Please report.'
        with pytest.raises(ValueError, match=msg):
            extend_substation_overloading(self.edisgo, crit_mv_station)

        # check replacing current transformers and replacing them with standard
        # transformers
        # set up made up critical station df
        crit_mv_station = pd.DataFrame({'s_pfa': [39],
                                        'time_index': [self.timesteps[1]]},
                                       index=['MVGrid_1'])
        transformer_changes = \
            extend_substation_overloading(self.edisgo, crit_mv_station)
        assert len(transformer_changes['added']['MVGrid_1']) == 2
        assert len(transformer_changes['removed']['MVGrid_1']) == 2
        trafos = self.edisgo.topology.transformers_hvmv_df
        assert (trafos.bus0 == 'Bus_primary_MVStation_1').all()
        assert (trafos.bus1 == 'Bus_MVStation_1').all()
        assert (trafos.s_nom == 40).all()
        assert (trafos.type_info == '40 MVA').all()
        assert 'MVStation_1_transformer_reinforced_2' in \
               transformer_changes['added']['MVGrid_1']
        print('Extend substation successful.')

    def test_reinforce_branches_overloading(self):
        std_line_mv = self.edisgo.topology.equipment_data['mv_cables'].loc[
            self.edisgo.config['grid_expansion_standard_equipment'][
                'mv_line']]
        std_line_mv.U_n = self.edisgo.topology.mv_grid.nominal_voltage

        std_line_lv = self.edisgo.topology.equipment_data['lv_cables'].loc[
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

    def test_extend_distribution_substation_overvoltage(self):
        crit_stations = {}
        station_9 = pd.DataFrame({'v_mag_pu': [0.03], 'time_index':
            [self.timesteps[0]]}, index=['Bus_secondary_LVStation_9'])
        crit_stations['LVGrid_9'] = station_9
        trafos_pre = self.edisgo.topology.transformers_df

        trafo_changes = \
            extend_distribution_substation_overvoltage(self.edisgo,
                                                       crit_stations)
        assert len(trafo_changes) == 1
        assert len(trafo_changes['added']['LVGrid_9']) == 1
        assert trafo_changes['added']['LVGrid_9'][0] == \
            'LVStation_9_transformer_reinforced_2'
        # check changes in transformers_df
        assert len(self.edisgo.topology.transformers_df) == \
            (len(trafos_pre) + 1)
        assert self.edisgo.topology.transformers_df.iloc[-1].name == \
            'LVStation_9_transformer_reinforced_2'
        assert self.edisgo.topology.transformers_df.iloc[-1].s_nom == 0.63
        assert self.edisgo.topology.transformers_df.iloc[-1].r_pu == \
            0.010317460317460317
        assert self.edisgo.topology.transformers_df.iloc[-1].x_pu == \
            0.03864647477581405
        assert self.edisgo.topology.transformers_df.iloc[-1].type_info == \
            '630 kVA'
