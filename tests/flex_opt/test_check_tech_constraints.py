import os
import pandas as pd
import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints


class TestCheckTechConstraints:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        dirname = os.path.realpath(os.path.dirname(__file__) + '/..')
        test_network_directory = os.path.join(dirname, 'ding0_test_network')
        self.edisgo = EDisGo(ding0_grid=test_network_directory,
                             worst_case_analysis='worst-case')
        self.timesteps = pd.date_range('1/1/1970', periods=2, freq='H')

    @pytest.fixture(autouse=True)
    def run_power_flow(self):
        self.edisgo.analyze()

    def test_voltage_diff(self):

        bus0 = 'Bus_Generator_1'
        bus1 = 'Bus_GeneratorFluctuating_2'
        bus2 = 'Bus_GeneratorFluctuating_3'

        # create over- and undervoltage at bus0, with higher undervoltage
        # deviation
        self.edisgo.results._v_res.loc[self.timesteps[0], bus0] = 1.11
        self.edisgo.results._v_res.loc[self.timesteps[1], bus0] = 0.88
        # create overvoltage at bus1
        self.edisgo.results._v_res.loc[self.timesteps[0], bus1] = 1.11
        # create undervoltage at bus0
        self.edisgo.results._v_res.loc[self.timesteps[0], bus2] = 0.89

        uv_violations, ov_violations = check_tech_constraints.voltage_diff(
            self.edisgo,
            self.edisgo.topology.mv_grid.buses_df.index,
            pd.Series(data=1.1, index=self.timesteps),
            pd.Series(data=0.9, index=self.timesteps)
        )

        # check shapes of under- and overvoltage dataframes
        assert uv_violations.shape == (2, 2)
        assert ov_violations.shape == (1, 2)
        # check under- and overvoltage deviation values
        assert np.isclose(uv_violations.at[bus0, self.timesteps[1]], 0.02)
        assert np.isclose(uv_violations.at[bus2, self.timesteps[0]], 0.01)
        assert np.isclose(ov_violations.at[bus1, self.timesteps[0]], 0.01)
        assert np.isclose(uv_violations.at[bus0, self.timesteps[0]], -0.21)

    def test__voltage_deviation(self):

        bus0 = 'Bus_Generator_1'
        bus1 = 'Bus_GeneratorFluctuating_2'
        bus2 = 'Bus_GeneratorFluctuating_3'

        # create over- and undervoltage at bus0, with higher undervoltage
        # deviation
        self.edisgo.results._v_res.loc[self.timesteps[0], bus0] = 1.11
        self.edisgo.results._v_res.loc[self.timesteps[1], bus0] = 0.88
        # create overvoltage at bus1
        self.edisgo.results._v_res.loc[self.timesteps[0], bus1] = 1.11
        # create undervoltage at bus0
        self.edisgo.results._v_res.loc[self.timesteps[0], bus2] = 0.895

        v_violations = check_tech_constraints._voltage_deviation(
            self.edisgo,
            self.edisgo.topology.mv_grid.buses_df.index,
            pd.Series(data=1.1, index=self.timesteps),
            pd.Series(data=0.9, index=self.timesteps)
        )

        # check shape of dataframe
        assert v_violations.shape == (3, 2)
        # check under- and overvoltage deviation values
        assert list(v_violations.index.values) == [bus0, bus1, bus2]
        assert np.isclose(v_violations.at[bus1, 'v_mag_pu'], 0.01)
        assert v_violations.at[bus0, 'time_index'] == self.timesteps[1]

    def test_mv_voltage_deviation(self):
        # create power flow issues
        self.edisgo.results.v_res.at[self.timesteps[0],
            'Bus_GeneratorFluctuating_2'] = 1.14
        self.edisgo.results.v_res.at[self.timesteps[1],
            'Bus_GeneratorFluctuating_2'] = 0.89
        self.edisgo.results.v_res.at[self.timesteps[0],
            'Bus_GeneratorFluctuating_3'] = 1.15
        self.edisgo.results.v_res.at[self.timesteps[1],
            'Bus_GeneratorFluctuating_4'] = 0.89
        self.edisgo.results.v_res.at[self.timesteps[0],
            'Bus_GeneratorFluctuating_5'] = 1.16
        self.edisgo.results.v_res.at[self.timesteps[1],
            'Bus_GeneratorFluctuating_6'] = 0.82

        voltage_issues = check_tech_constraints.mv_voltage_deviation(
            self.edisgo)

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
        self.edisgo.results.v_res.at[self.timesteps[0],
            'Bus_primary_LVStation_9'] = 1.14
        self.edisgo.results.v_res.at[self.timesteps[0],
            'Bus_secondary_LVStation_9'] = 1.14
        self.edisgo.results.v_res.at[self.timesteps[1],
            'Bus_secondary_LVStation_9'] = 0.89
        self.edisgo.results.v_res.at[self.timesteps[0],
            'Bus_BranchTee_LVGrid_1_4'] = 1.15
        self.edisgo.results.v_res.at[self.timesteps[1],
            'Bus_BranchTee_LVGrid_1_5'] = 0.89
        self.edisgo.results.v_res.at[self.timesteps[0],
            'Bus_Load_residential_LVGrid_1_7'] = 1.16
        self.edisgo.results.v_res.at[self.timesteps[1],
            'Bus_GeneratorFluctuating_13'] = 0.82

        lvgrid_1 = self.edisgo.topology._grids['LVGrid_1']
        lvgrid_9 = self.edisgo.topology._grids['LVGrid_9']

        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo, mode='stations')
        assert len(voltage_issues) == 1
        assert len(voltage_issues[lvgrid_9]) == 1
        assert np.isclose(voltage_issues[lvgrid_9].loc[
                              'Bus_secondary_LVStation_9', 'v_mag_pu'], 0.04)
        assert voltage_issues[lvgrid_9].loc[
                   'Bus_secondary_LVStation_9', 'time_index'] == \
               self.timesteps[0]

        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo)

        assert len(voltage_issues[lvgrid_1]) == 4
        assert len(voltage_issues[lvgrid_9]) == 1
        assert np.isclose(voltage_issues[lvgrid_9].loc[
            'Bus_secondary_LVStation_9', 'v_mag_pu'], 0.04)
        assert np.isclose(voltage_issues[lvgrid_1].loc[
            'Bus_BranchTee_LVGrid_1_4', 'v_mag_pu'], 0.05)
        assert np.isclose(voltage_issues[lvgrid_1].loc[
            'Bus_BranchTee_LVGrid_1_5', 'v_mag_pu'], 0.01)
        assert np.isclose(voltage_issues[lvgrid_1].loc[
            'Bus_Load_residential_LVGrid_1_7', 'v_mag_pu'], 0.06)
        assert np.isclose(voltage_issues[lvgrid_1].loc[
            'Bus_GeneratorFluctuating_13', 'v_mag_pu'], 0.08)
        assert voltage_issues[lvgrid_9].loc[
                   'Bus_secondary_LVStation_9', 'time_index'] == \
               self.timesteps[0]
        assert voltage_issues[lvgrid_1].loc[
                   'Bus_BranchTee_LVGrid_1_4', 'time_index'] == \
               self.timesteps[0]
        assert voltage_issues[lvgrid_1].loc[
                   'Bus_BranchTee_LVGrid_1_5', 'time_index'] == \
               self.timesteps[1]
        assert voltage_issues[lvgrid_1].loc[
                   'Bus_Load_residential_LVGrid_1_7', 'time_index'] == \
               self.timesteps[0]
        assert voltage_issues[lvgrid_1].loc[
                   'Bus_GeneratorFluctuating_13', 'time_index'] == \
               self.timesteps[1]

    def test_check_ten_percent_voltage_deviation(self):
        # reset values
        if self.edisgo.results.v_res.at[
                self.timesteps[0], 'Bus_primary_LVStation_9'] == 1.14:
            self.edisgo.analyze()
            check_tech_constraints.check_ten_percent_voltage_deviation(
                self.edisgo)
        self.edisgo.results.v_res.at[
                self.timesteps[0], 'Bus_primary_LVStation_9'] = 1.14
        msg = "Maximum allowed voltage deviation of 10% exceeded."
        with pytest.raises(ValueError, match=msg):
            check_tech_constraints.check_ten_percent_voltage_deviation(
                self.edisgo)
