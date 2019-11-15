import os
import pandas as pd
import numpy as np
from edisgo import EDisGo
from edisgo.flex_opt.check_tech_constraints import mv_voltage_deviation


class TestCheckTechConstraints:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        dirname = os.path.realpath(os.path.dirname(__file__) + '/..')
        test_network_directory = os.path.join(dirname, 'ding0_test_network')
        self.edisgo = EDisGo(ding0_grid=test_network_directory,
                             worst_case_analysis='worst-case')
        self.edisgo.analyze()
        self.timesteps = pd.date_range('1/1/1970', periods=2, freq='H')

    def test_mv_voltage_deviation(self):
        # create power flow issues
        self.edisgo.results.pfa_v_mag_pu['mv'].at[self.timesteps[0],
            'Bus_GeneratorFluctuating_2'] = 1.14
        self.edisgo.results.pfa_v_mag_pu['mv'].at[self.timesteps[1],
            'Bus_GeneratorFluctuating_2'] = 0.89
        self.edisgo.results.pfa_v_mag_pu['mv'].at[self.timesteps[0],
            'Bus_GeneratorFluctuating_3'] = 1.15
        self.edisgo.results.pfa_v_mag_pu['mv'].at[self.timesteps[1],
            'Bus_GeneratorFluctuating_4'] = 0.89
        self.edisgo.results.pfa_v_mag_pu['mv'].at[self.timesteps[0],
            'Bus_GeneratorFluctuating_5'] = 1.16
        self.edisgo.results.pfa_v_mag_pu['mv'].at[self.timesteps[1],
            'Bus_GeneratorFluctuating_6'] = 0.82

        voltage_issues = mv_voltage_deviation(self.edisgo)

        assert len(voltage_issues['MVGrid_1']) == 5
        assert np.isclose(voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_2', 'v_mag_pu'], 0.04)
        assert np.isclose(voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_3', 'v_mag_pu'], 0.05)
        assert np.isclose(voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_4', 'v_mag_pu'], 0.01)
        assert np.isclose(voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_5', 'v_mag_pu'], 0.06)
        assert np.isclose(voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_6', 'v_mag_pu'], 0.08)
        assert voltage_issues['MVGrid_1'].loc[
               'Bus_GeneratorFluctuating_2', 'time_index'] == self.timesteps[0]
        assert voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_3', 'time_index'] == \
               self.timesteps[0]
        assert voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_4', 'time_index'] == \
               self.timesteps[1]
        assert voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_5', 'time_index'] == \
               self.timesteps[0]
        assert voltage_issues['MVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_6', 'time_index'] == \
               self.timesteps[1]