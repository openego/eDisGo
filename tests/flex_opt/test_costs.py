import os
import pandas as pd
import numpy as np

from edisgo import EDisGo
from edisgo.flex_opt.costs import grid_expansion_costs, line_expansion_costs


class TestCosts:
    @classmethod
    def setup_class(self):
        """Setup default values"""
        dirname = os.path.realpath(os.path.dirname(__file__) + '/..')
        test_network_directory = os.path.join(dirname, 'ding0_test_network')
        self.edisgo = EDisGo(ding0_grid=test_network_directory,
                             worst_case_analysis='worst-case')
        self.edisgo.analyze()

    def test_costs(self):
        hv_mv_trafo = self.edisgo.topology.transformers_hvmv_df.loc[
            'MVStation_1_transformer_1']
        hv_mv_trafo.name = 'MVStation_1_transformer_reinforced_2'
        self.edisgo.topology.transformers_hvmv_df = \
            self.edisgo.topology.transformers_hvmv_df.append(hv_mv_trafo)
        mv_lv_trafo = self.edisgo.topology.transformers_df.loc[
            'LVStation_1_transformer_1']
        mv_lv_trafo.name = 'LVStation_1_transformer_reinforced_1'
        self.edisgo.topology.transformers_df.drop('LVStation_1_transformer_1',
                                                  inplace=True)
        self.edisgo.topology.transformers_df = \
            self.edisgo.topology.transformers_df.append(mv_lv_trafo)

        self.edisgo.results.equipment_changes = pd.DataFrame({
            'iteration_step': [1, 1, 1, 1, 1, 2, 2, 4, 0],
            'change': ['added', 'added', 'removed',
                       'changed', 'changed', 'changed',
                       'changed', 'changed', 'added'],
            'equipment': ['MVStation_1_transformer_reinforced_2',
                          'LVStation_1_transformer_reinforced_1',
                          'LVStation_1_transformer_1',
                          'NA2XS2Y 3x1x185 RM/25', '48-AL1/8-ST1A',
                          'NA2XS2Y 3x1x185 RM/25',
                          'NAYY 4x1x35', 'NAYY 4x1x35', 'dummy_gen'],
            'quantity': [1, 1, 1, 2, 1, 1, 1, 3, 1]
        }, index=['MVGrid_1', 'LVGrid_1', 'LVGrid_1', 'Line_10006',
                  'Line_10019', 'Line_10019', 'Line_50000002', 'Line_50000004',
                  'dummy_gen'])

        costs = grid_expansion_costs(self.edisgo)

        assert len(costs) == 6
        assert costs.loc[
                   'MVStation_1_transformer_reinforced_2',
                   'voltage_level'] == 'mv/lv'
        assert costs.loc[
                   'MVStation_1_transformer_reinforced_2', 'quantity'] == 1
        assert costs.loc[
                   'MVStation_1_transformer_reinforced_2', 'total_costs'] == \
               1000
        assert costs.loc[
                   'LVStation_1_transformer_reinforced_1',
                   'voltage_level'] == 'mv/lv'
        assert costs.loc[
                   'LVStation_1_transformer_reinforced_1', 'quantity'] == 1
        assert costs.loc[
                   'LVStation_1_transformer_reinforced_1', 'total_costs'] == 10
        assert np.isclose(costs.loc['Line_10006', 'total_costs'], 29.765)
        assert np.isclose(costs.loc['Line_10006', 'length'], (0.29765*2))
        assert costs.loc['Line_10006', 'quantity'] == 2
        assert costs.loc['Line_10006', 'type'] == 'NA2XS2Y 3x1x185 RM/25'
        assert costs.loc['Line_10006', 'voltage_level'] == 'mv'
        assert np.isclose(costs.loc['Line_10019', 'total_costs'], 32.3082)
        assert np.isclose(costs.loc['Line_10019', 'length'], 0.40385)
        assert costs.loc['Line_10019', 'quantity'] == 1
        assert costs.loc['Line_10019', 'type'] == '48-AL1/8-ST1A'
        assert costs.loc['Line_10019', 'voltage_level'] == 'mv'
        assert np.isclose(costs.loc['Line_50000002', 'total_costs'], 1.8)
        assert np.isclose(costs.loc['Line_50000002', 'length'], 0.03)
        assert costs.loc['Line_50000002', 'quantity'] == 1
        assert costs.loc['Line_50000002', 'type'] == 'NAYY 4x1x35'
        assert costs.loc['Line_50000002', 'voltage_level'] == 'lv'
        assert np.isclose(costs.loc['Line_50000004', 'total_costs'], 0.078)
        assert np.isclose(costs.loc['Line_50000004', 'length'], 0.003)
        assert costs.loc['Line_50000004', 'quantity'] == 3
        assert costs.loc['Line_50000004', 'type'] == 'NAYY 4x1x35'
        assert costs.loc['Line_50000004', 'voltage_level'] == 'lv'

    def test_line_expansion_costs(self):
        costs = line_expansion_costs(self.edisgo,
                                     self.edisgo.topology.lines_df.index)
        assert len(costs) == len(self.edisgo.topology.lines_df)
        assert (costs.index == self.edisgo.topology.lines_df.index).all()
        assert len(costs[costs.voltage_level == 'mv']) == \
               len(self.edisgo.topology.mv_grid.lines_df)
        assert np.isclose(costs.at['Line_10001', 'costs_earthworks'], 0.06)
        assert np.isclose(costs.at['Line_10001', 'costs_cable'], 0.02)
        assert costs.at['Line_10001', 'voltage_level'] == 'mv'
        assert np.isclose(costs.at['Line_10000001', 'costs_earthworks'], 0.051)
        assert np.isclose(costs.at['Line_10000001', 'costs_cable'], 0.009)
        assert costs.at['Line_10000001', 'voltage_level'] == 'lv'
        assert np.isclose(costs.at['Line_10000015', 'costs_earthworks'], 1.53)
        assert np.isclose(costs.at['Line_10000015', 'costs_cable'], 0.27)
        assert costs.at['Line_10000015', 'voltage_level'] == 'lv'

        costs = line_expansion_costs(self.edisgo,
                            ['Line_10001', 'Line_10000001', 'Line_10000015'])
        assert len(costs) == 3
        assert (costs.index.values == ['Line_10001', 'Line_10000001',
                                       'Line_10000015']).all()
        assert np.isclose(costs.at['Line_10001', 'costs_earthworks'], 0.06)
        assert np.isclose(costs.at['Line_10001', 'costs_cable'], 0.02)
        assert costs.at['Line_10001', 'voltage_level'] == 'mv'
        assert np.isclose(costs.at['Line_10000001', 'costs_earthworks'], 0.051)
        assert np.isclose(costs.at['Line_10000001', 'costs_cable'], 0.009)
        assert costs.at['Line_10000001', 'voltage_level'] == 'lv'
        assert np.isclose(costs.at['Line_10000015', 'costs_earthworks'], 1.53)
        assert np.isclose(costs.at['Line_10000015', 'costs_cable'], 0.27)
        assert costs.at['Line_10000015', 'voltage_level'] == 'lv'