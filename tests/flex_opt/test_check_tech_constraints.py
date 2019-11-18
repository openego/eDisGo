import os
import pandas as pd
import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.flex_opt.check_tech_constraints import mv_voltage_deviation, \
    lv_voltage_deviation, check_ten_percent_voltage_deviation


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

    def test_lv_voltage_deviation(self):
        # create power flow issues
        self.edisgo.results.pfa_v_mag_pu['mv'].at[self.timesteps[0],
            'Bus_primary_LVStation_9'] = 1.14
        self.edisgo.results.pfa_v_mag_pu['lv'].at[self.timesteps[0],
            'Bus_secondary_LVStation_9'] = 1.14
        self.edisgo.results.pfa_v_mag_pu['lv'].at[self.timesteps[1],
            'Bus_secondary_LVStation_9'] = 0.89
        self.edisgo.results.pfa_v_mag_pu['lv'].at[self.timesteps[0],
            'Bus_BranchTee_LVGrid_1_4'] = 1.15
        self.edisgo.results.pfa_v_mag_pu['lv'].at[self.timesteps[1],
            'Bus_BranchTee_LVGrid_1_5'] = 0.89
        self.edisgo.results.pfa_v_mag_pu['lv'].at[self.timesteps[0],
            'Bus_Load_residential_LVGrid_1_7'] = 1.16
        self.edisgo.results.pfa_v_mag_pu['lv'].at[self.timesteps[1],
            'Bus_GeneratorFluctuating_13'] = 0.82

        voltage_issues = lv_voltage_deviation(self.edisgo, mode='stations')
        assert len(voltage_issues) == 1
        assert len(voltage_issues['LVGrid_9']) == 1
        assert np.isclose(voltage_issues['LVGrid_9'].loc[
                              'Bus_secondary_LVStation_9', 'v_mag_pu'], 0.04)
        assert voltage_issues['LVGrid_9'].loc[
                   'Bus_secondary_LVStation_9', 'time_index'] == \
               self.timesteps[0]

        voltage_issues = lv_voltage_deviation(self.edisgo)

        assert len(voltage_issues['LVGrid_1']) == 4
        assert len(voltage_issues['LVGrid_9']) == 1
        assert np.isclose(voltage_issues['LVGrid_9'].loc[
            'Bus_secondary_LVStation_9', 'v_mag_pu'], 0.04)
        assert np.isclose(voltage_issues['LVGrid_1'].loc[
            'Bus_BranchTee_LVGrid_1_4', 'v_mag_pu'], 0.05)
        assert np.isclose(voltage_issues['LVGrid_1'].loc[
            'Bus_BranchTee_LVGrid_1_5', 'v_mag_pu'], 0.01)
        assert np.isclose(voltage_issues['LVGrid_1'].loc[
            'Bus_Load_residential_LVGrid_1_7', 'v_mag_pu'], 0.06)
        assert np.isclose(voltage_issues['LVGrid_1'].loc[
            'Bus_GeneratorFluctuating_13', 'v_mag_pu'], 0.08)
        assert voltage_issues['LVGrid_9'].loc[
                   'Bus_secondary_LVStation_9', 'time_index'] == \
               self.timesteps[0]
        assert voltage_issues['LVGrid_1'].loc[
                   'Bus_BranchTee_LVGrid_1_4', 'time_index'] == \
               self.timesteps[0]
        assert voltage_issues['LVGrid_1'].loc[
                   'Bus_BranchTee_LVGrid_1_5', 'time_index'] == \
               self.timesteps[1]
        assert voltage_issues['LVGrid_1'].loc[
                   'Bus_Load_residential_LVGrid_1_7', 'time_index'] == \
               self.timesteps[0]
        assert voltage_issues['LVGrid_1'].loc[
                   'Bus_GeneratorFluctuating_13', 'time_index'] == \
               self.timesteps[1]

    def test_check_ten_percent_voltage_deviation(self):
        # reset values
        if self.edisgo.results.pfa_v_mag_pu['mv'].at[
                self.timesteps[0], 'Bus_primary_LVStation_9'] == 1.14:
            self.edisgo.analyze()
        check_ten_percent_voltage_deviation(self.edisgo)
        self.edisgo.results.pfa_v_mag_pu['mv'].at[
                self.timesteps[0], 'Bus_primary_LVStation_9'] = 1.14
        msg = "Maximum allowed voltage deviation of 10% exceeded."
        with pytest.raises(ValueError, match=msg):
            check_ten_percent_voltage_deviation(self.edisgo)