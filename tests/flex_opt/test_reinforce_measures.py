import os
import pandas as pd
import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.flex_opt.reinforce_measures import \
    extend_distribution_substation_overloading, extend_substation_overloading, \
    reinforce_branches_overloading

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

