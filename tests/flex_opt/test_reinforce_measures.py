import os
import pandas as pd

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
