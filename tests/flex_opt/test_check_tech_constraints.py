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
        dirname = os.path.realpath(os.path.dirname(__file__) + "/..")
        test_network_directory = os.path.join(dirname, "ding0_test_network")
        self.edisgo = EDisGo(
            ding0_grid=test_network_directory, worst_case_analysis="worst-case"
        )
        self.timesteps = pd.date_range("1/1/1970", periods=2, freq="H")

    @pytest.fixture(autouse=True)
    def run_power_flow(self):
        """
        Fixture to run new power flow before each test.

        """
        self.edisgo.analyze()

    def mv_voltage_issues(self):
        """
        Fixture to create voltage issues in MV grid.

        """
        bus0 = "Bus_Generator_1"
        bus1 = "Bus_GeneratorFluctuating_2"
        bus2 = "Bus_GeneratorFluctuating_3"

        # create over- and undervoltage at bus0, with higher undervoltage
        # deviation
        self.edisgo.results._v_res.loc[self.timesteps[0], bus0] = 1.11
        self.edisgo.results._v_res.loc[self.timesteps[1], bus0] = 0.88
        # create overvoltage at bus1
        self.edisgo.results._v_res.loc[self.timesteps[0], bus1] = 1.11
        # create undervoltage at bus0
        self.edisgo.results._v_res.loc[self.timesteps[0], bus2] = 0.895

    def test_mv_voltage_deviation(self):

        # check with no voltage issues
        voltage_issues = check_tech_constraints.mv_voltage_deviation(
            self.edisgo
        )
        assert {} == voltage_issues

        # create voltage issues
        self.mv_voltage_issues()

        # check with voltage issues and voltage_levels="mv_lv" (default)
        voltage_issues = check_tech_constraints.mv_voltage_deviation(
            self.edisgo
        )
        # check shape of dataframe
        assert (3, 2) == voltage_issues["MVGrid_1"].shape
        # check under- and overvoltage deviation values
        assert list(voltage_issues["MVGrid_1"].index.values) == [
            "Bus_Generator_1",
            "Bus_GeneratorFluctuating_2",
            "Bus_GeneratorFluctuating_3",
        ]
        assert np.isclose(
            voltage_issues["MVGrid_1"].at[
                "Bus_GeneratorFluctuating_2", "v_mag_pu"
            ],
            0.01,
        )
        assert (
            voltage_issues["MVGrid_1"].at["Bus_Generator_1", "time_index"]
            == self.timesteps[1]
        )

        # check with voltage issues and voltage_levels="mv"
        voltage_issues = check_tech_constraints.mv_voltage_deviation(
            self.edisgo, "mv"
        )
        # check shape of dataframe
        assert (3, 2) == voltage_issues["MVGrid_1"].shape
        # check under- and overvoltage deviation values
        assert list(voltage_issues["MVGrid_1"].index.values) == [
            "Bus_Generator_1",
            "Bus_GeneratorFluctuating_2",
            "Bus_GeneratorFluctuating_3",
        ]
        assert np.isclose(
            voltage_issues["MVGrid_1"].at[
                "Bus_GeneratorFluctuating_2", "v_mag_pu"
            ],
            0.06,
        )
        assert (
            voltage_issues["MVGrid_1"].at["Bus_Generator_1", "time_index"]
            == self.timesteps[1]
        )

    def test_lv_voltage_deviation(self):

        # check with default values that there are no voltage issues
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo
        )
        assert {} == voltage_issues

        # check with mode "stations" and default value for voltage_level that
        # there are no voltage issues
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo, mode="stations"
        )
        assert {} == voltage_issues

        # check that voltage issue in station of LVGrid_6 is detected when
        # voltage_levels="lv"
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo, voltage_levels="lv", mode="stations"
        )
        lvgrid_6 = self.edisgo.topology._grids["LVGrid_6"]
        assert len(voltage_issues) == 1
        assert len(voltage_issues[lvgrid_6]) == 1
        assert np.isclose(
            voltage_issues[lvgrid_6].loc[
                "Bus_secondary_LVStation_6", "v_mag_pu"
            ],
            0.010635,
        )

        # check with voltage_levels="lv" and mode=None
        # create one voltage issue in LVGrid_6
        self.edisgo.results.v_res.at[
            self.timesteps[0], "Bus_secondary_LVStation_6"
        ] = 1.14
        self.edisgo.results.v_res.at[
            self.timesteps[0], "Bus_BranchTee_LVGrid_6_1"
        ] = 1.18
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo, voltage_levels="lv"
        )
        assert len(voltage_issues) == 1
        assert len(voltage_issues[lvgrid_6]) == 1
        assert np.isclose(
            voltage_issues[lvgrid_6].loc[
                "Bus_BranchTee_LVGrid_6_1", "v_mag_pu"
            ],
            0.005,
        )
        # create second voltage issue in LVGrid_6, greater than first issue
        self.edisgo.results.v_res.at[
            self.timesteps[0], "Bus_BranchTee_LVGrid_6_2"
        ] = 1.19
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo, voltage_levels="lv"
        )
        assert len(voltage_issues) == 1
        assert len(voltage_issues[lvgrid_6]) == 2
        assert voltage_issues[lvgrid_6].index[0] == "Bus_BranchTee_LVGrid_6_2"
        assert np.isclose(
            voltage_issues[lvgrid_6].loc[
                "Bus_BranchTee_LVGrid_6_2", "v_mag_pu"
            ],
            0.015,
        )

        # check with voltage_levels="mv_lv" and mode=None
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo
        )
        assert len(voltage_issues) == 1
        assert len(voltage_issues[lvgrid_6]) == 3
        assert voltage_issues[lvgrid_6].index[0] == "Bus_BranchTee_LVGrid_6_2"
        assert np.isclose(
            voltage_issues[lvgrid_6].loc[
                "Bus_BranchTee_LVGrid_6_2", "v_mag_pu"
            ],
            0.09,
        )

        # check with voltage_levels="mv_lv" and mode="stations"
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo, mode="stations"
        )
        assert len(voltage_issues) == 1
        assert len(voltage_issues[lvgrid_6]) == 1
        assert np.isclose(
            voltage_issues[lvgrid_6].loc[
                "Bus_secondary_LVStation_6", "v_mag_pu"
            ],
            0.04,
        )

    def test__mv_allowed_voltage_limits(self):
        # run function with voltage_levels="mv"
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._mv_allowed_voltage_limits(
            self.edisgo, "mv"
        )

        assert 1.05 == v_limits_upper.loc[self.timesteps[0]]
        assert 1.10 == v_limits_upper.loc[self.timesteps[1]]
        assert 0.90 == v_limits_lower.loc[self.timesteps[0]]
        assert 0.985 == v_limits_lower.loc[self.timesteps[1]]

        # run function with voltage_levels="mv_lv"
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._mv_allowed_voltage_limits(
            self.edisgo, "mv_lv"
        )

        assert 1.10 == v_limits_upper.loc[self.timesteps[0]]
        assert 1.10 == v_limits_upper.loc[self.timesteps[1]]
        assert 0.90 == v_limits_lower.loc[self.timesteps[0]]
        assert 0.90 == v_limits_lower.loc[self.timesteps[1]]

    def test__lv_allowed_voltage_limits(self):

        # get LVGrid_1 object
        lv_grid = self.edisgo.topology._grids["LVGrid_1"]
        # set voltage at stations' secondary side to known value
        self.edisgo.results._v_res.loc[
            self.timesteps[0], "Bus_secondary_LVStation_1"
        ] = 1.05
        self.edisgo.results._v_res.loc[
            self.timesteps[1], "Bus_secondary_LVStation_1"
        ] = 0.98

        # run function with mode=None
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._lv_allowed_voltage_limits(
            self.edisgo, lv_grid, mode=None
        )

        assert 1.085 == v_limits_upper.loc[self.timesteps[0]]
        assert 1.10 == v_limits_upper.loc[self.timesteps[1]]
        assert 0.90 == v_limits_lower.loc[self.timesteps[0]]
        assert 0.915 == v_limits_lower.loc[self.timesteps[1]]

        # set voltage at stations' primary side to known value
        self.edisgo.results._v_res.loc[
            self.timesteps[0], "Bus_primary_LVStation_1"
        ] = 1.03
        self.edisgo.results._v_res.loc[
            self.timesteps[1], "Bus_primary_LVStation_1"
        ] = 0.99

        # run function with mode='stations'
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._lv_allowed_voltage_limits(
            self.edisgo, lv_grid, mode="stations"
        )

        assert 1.045 == v_limits_upper.loc[self.timesteps[0]]
        assert 1.10 == v_limits_upper.loc[self.timesteps[1]]
        assert 0.90 == v_limits_lower.loc[self.timesteps[0]]
        assert 0.97 == v_limits_lower.loc[self.timesteps[1]]

    def test_voltage_diff(self):

        # create voltage issues
        self.mv_voltage_issues()

        uv_violations, ov_violations = check_tech_constraints.voltage_diff(
            self.edisgo,
            self.edisgo.topology.mv_grid.buses_df.index,
            pd.Series(data=1.1, index=self.timesteps),
            pd.Series(data=0.9, index=self.timesteps),
        )

        # check shapes of under- and overvoltage dataframes
        assert (2, 2) == uv_violations.shape
        assert (1, 2) == ov_violations.shape
        # check under- and overvoltage deviation values
        assert np.isclose(
            uv_violations.at["Bus_Generator_1", self.timesteps[1]], 0.02
        )
        assert np.isclose(
            uv_violations.at["Bus_GeneratorFluctuating_3", self.timesteps[0]],
            0.005,
        )
        assert np.isclose(
            ov_violations.at["Bus_GeneratorFluctuating_2", self.timesteps[0]],
            0.01,
        )
        assert np.isclose(
            uv_violations.at["Bus_Generator_1", self.timesteps[0]], -0.21
        )

    def test__voltage_deviation(self):

        # create voltage issues
        self.mv_voltage_issues()

        v_violations = check_tech_constraints._voltage_deviation(
            self.edisgo,
            self.edisgo.topology.mv_grid.buses_df.index,
            pd.Series(data=1.1, index=self.timesteps),
            pd.Series(data=0.9, index=self.timesteps),
        )

        # check shape of dataframe
        assert (3, 2) == v_violations.shape
        # check under- and overvoltage deviation values
        assert list(v_violations.index.values) == [
            "Bus_Generator_1",
            "Bus_GeneratorFluctuating_2",
            "Bus_GeneratorFluctuating_3",
        ]
        assert np.isclose(
            v_violations.at["Bus_GeneratorFluctuating_2", "v_mag_pu"], 0.01
        )
        assert (
            v_violations.at["Bus_Generator_1", "time_index"]
            == self.timesteps[1]
        )

    def test_check_ten_percent_voltage_deviation(self):
        # check without voltage issues greater than 10%
        check_tech_constraints.check_ten_percent_voltage_deviation(self.edisgo)
        # create voltage issues greater 10% and check again
        self.edisgo.results.v_res.at[
            self.timesteps[0], "Bus_primary_LVStation_9"
        ] = 1.14
        msg = "Maximum allowed voltage deviation of 10% exceeded."
        with pytest.raises(ValueError, match=msg):
            check_tech_constraints.check_ten_percent_voltage_deviation(
                self.edisgo
            )
