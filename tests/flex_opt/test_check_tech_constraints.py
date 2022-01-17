import pandas as pd
import numpy as np
import pytest
from math import sqrt

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints


class TestCheckTechConstraints:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_path,
            worst_case_analysis="worst-case",
        )
        self.timesteps = self.edisgo.timeseries.timeindex

    @pytest.fixture(autouse=True)
    def run_power_flow(self):
        """
        Fixture to run new power flow before each test.

        """
        self.edisgo.analyze()

    def test_mv_line_load(self):
        # implicitly checks function _line_load

        df = check_tech_constraints.mv_line_load(self.edisgo)
        # check shape of dataframe
        assert (4, 3) == df.shape
        # check relative overload of one line
        assert np.isclose(
            df.at["Line_10005", "max_rel_overload"],
            self.edisgo.results.i_res.at[self.timesteps[0], "Line_10005"]
            / (7.274613391789284 / 20 / sqrt(3)),
        )
        assert df.at["Line_10005", "time_index"] == self.timesteps[0]

    def test_lv_line_load(self):
        # implicitly checks function _line_load

        df = check_tech_constraints.lv_line_load(self.edisgo)
        # check shape of dataframe
        assert (2, 3) == df.shape
        # check relative overload of one line
        assert np.isclose(
            df.at["Line_50000002", "max_rel_overload"],
            self.edisgo.results.i_res.at[self.timesteps[1], "Line_50000002"]
            / (0.08521689973238901 / 0.4 / sqrt(3)),
        )
        assert df.at["Line_50000002", "time_index"] == self.timesteps[1]

    def test_hv_mv_station_load(self):
        # implicitly checks function _station_load

        # create over-load problem in both time steps with higher over-load
        # in load case
        self.edisgo.results.pfa_slack = pd.DataFrame(
            data={"p": [30, 25], "q": [30, 25]}, index=self.timesteps
        )

        df = check_tech_constraints.hv_mv_station_load(self.edisgo)
        # check shape of dataframe
        assert (1, 2) == df.shape
        # check missing transformer capacity
        assert np.isclose(
            df.at["MVGrid_1", "s_missing"],
            (sqrt(1250) - 20) / 0.5,
        )
        assert df.at["MVGrid_1", "time_index"] == self.timesteps[1]

    def test_mv_lv_station_load(self):
        # implicitly checks function _station_load

        df = check_tech_constraints.mv_lv_station_load(self.edisgo)
        # check shape of dataframe
        assert (4, 2) == df.shape
        # check missing transformer capacity of one grid
        assert np.isclose(
            df.at["LVGrid_1", "s_missing"],
            self.edisgo.results.s_res.at[self.timesteps[1], "LVStation_1_transformer_1"]
            - 0.16,
        )
        assert df.at["LVGrid_1", "time_index"] == self.timesteps[1]

    def test_lines_allowed_load(self):

        # check for MV
        df = check_tech_constraints.lines_allowed_load(self.edisgo, "mv")
        # check shape of dataframe
        assert (2, 30) == df.shape
        # check in feed-in case
        assert np.isclose(
            df.at[self.timesteps[0], "Line_10005"],
            7.274613391789284 / 20 / sqrt(3),
        )
        # check in load case (line in cycle as well as stub)
        assert np.isclose(
            df.at[self.timesteps[1], "Line_10005"],
            7.274613391789284 / 20 / sqrt(3) * 0.5,
        )
        assert np.isclose(
            df.at[self.timesteps[1], "Line_10024"],
            7.27461339178928 / 20 / sqrt(3),
        )

        # check for LV
        df = check_tech_constraints.lines_allowed_load(self.edisgo, "lv")
        # check shape of dataframe
        assert (2, 99) == df.shape
        # check in feed-in case
        assert np.isclose(
            df.at[self.timesteps[0], "Line_50000002"],
            0.08521689973238901 / 0.4 / sqrt(3),
        )
        # check in load case
        assert np.isclose(
            df.at[self.timesteps[1], "Line_50000002"],
            0.08521689973238901 / 0.4 / sqrt(3),
        )

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
        voltage_issues = check_tech_constraints.mv_voltage_deviation(self.edisgo)
        assert {} == voltage_issues

        # create voltage issues
        self.mv_voltage_issues()

        # check with voltage issues and voltage_levels="mv_lv" (default)
        voltage_issues = check_tech_constraints.mv_voltage_deviation(self.edisgo)
        # check shape of dataframe
        assert (3, 2) == voltage_issues["MVGrid_1"].shape
        # check under- and overvoltage deviation values
        assert list(voltage_issues["MVGrid_1"].index.values) == [
            "Bus_Generator_1",
            "Bus_GeneratorFluctuating_2",
            "Bus_GeneratorFluctuating_3",
        ]
        assert np.isclose(
            voltage_issues["MVGrid_1"].at["Bus_GeneratorFluctuating_2", "v_diff_max"],
            0.01,
        )
        assert (
            voltage_issues["MVGrid_1"].at["Bus_Generator_1", "time_index"]
            == self.timesteps[1]
        )

        # check with voltage issues and voltage_levels="mv"
        voltage_issues = check_tech_constraints.mv_voltage_deviation(self.edisgo, "mv")
        # check shape of dataframe
        assert (3, 2) == voltage_issues["MVGrid_1"].shape
        # check under- and overvoltage deviation values
        assert list(voltage_issues["MVGrid_1"].index.values) == [
            "Bus_Generator_1",
            "Bus_GeneratorFluctuating_2",
            "Bus_GeneratorFluctuating_3",
        ]
        assert np.isclose(
            voltage_issues["MVGrid_1"].at["Bus_GeneratorFluctuating_2", "v_diff_max"],
            0.06,
        )
        assert (
            voltage_issues["MVGrid_1"].at["Bus_Generator_1", "time_index"]
            == self.timesteps[1]
        )

    def test_lv_voltage_deviation(self):

        # check with default values that there are no voltage issues
        voltage_issues = check_tech_constraints.lv_voltage_deviation(self.edisgo)
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
        assert len(voltage_issues) == 1
        assert len(voltage_issues["LVGrid_6"]) == 1
        assert np.isclose(
            voltage_issues["LVGrid_6"].loc["BusBar_MVGrid_1_LVGrid_6_LV", "v_diff_max"],
            0.0106225,
        )

        # check with voltage_levels="lv" and mode=None
        # create one voltage issue in LVGrid_6
        self.edisgo.results.v_res.at[
            self.timesteps[0], "BusBar_MVGrid_1_LVGrid_6_LV"
        ] = 1.14
        self.edisgo.results.v_res.at[
            self.timesteps[0], "Bus_BranchTee_LVGrid_6_1"
        ] = 1.18
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo, voltage_levels="lv"
        )
        assert len(voltage_issues) == 1
        assert len(voltage_issues["LVGrid_6"]) == 1
        assert np.isclose(
            voltage_issues["LVGrid_6"].loc["Bus_BranchTee_LVGrid_6_1", "v_diff_max"],
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
        assert len(voltage_issues["LVGrid_6"]) == 2
        assert voltage_issues["LVGrid_6"].index[0] == "Bus_BranchTee_LVGrid_6_2"
        assert np.isclose(
            voltage_issues["LVGrid_6"].loc["Bus_BranchTee_LVGrid_6_2", "v_diff_max"],
            0.015,
        )

        # check with voltage_levels="mv_lv" and mode=None
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.lv_voltage_deviation(self.edisgo)
        assert len(voltage_issues) == 1
        assert len(voltage_issues["LVGrid_6"]) == 3
        assert voltage_issues["LVGrid_6"].index[0] == "Bus_BranchTee_LVGrid_6_2"
        assert np.isclose(
            voltage_issues["LVGrid_6"].loc["Bus_BranchTee_LVGrid_6_2", "v_diff_max"],
            0.09,
        )

        # check with voltage_levels="mv_lv" and mode="stations"
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.lv_voltage_deviation(
            self.edisgo, mode="stations"
        )
        assert len(voltage_issues) == 1
        assert len(voltage_issues["LVGrid_6"]) == 1
        assert np.isclose(
            voltage_issues["LVGrid_6"].loc["BusBar_MVGrid_1_LVGrid_6_LV", "v_diff_max"],
            0.04,
        )

    def test__mv_allowed_voltage_limits(self):
        # run function with voltage_levels="mv"
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._mv_allowed_voltage_limits(self.edisgo, "mv")

        assert 1.05 == v_limits_upper.loc[self.timesteps[0]]
        assert 1.10 == v_limits_upper.loc[self.timesteps[1]]
        assert 0.90 == v_limits_lower.loc[self.timesteps[0]]
        assert 0.985 == v_limits_lower.loc[self.timesteps[1]]

        # run function with voltage_levels="mv_lv"
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._mv_allowed_voltage_limits(self.edisgo, "mv_lv")

        assert 1.10 == v_limits_upper.loc[self.timesteps[0]]
        assert 1.10 == v_limits_upper.loc[self.timesteps[1]]
        assert 0.90 == v_limits_lower.loc[self.timesteps[0]]
        assert 0.90 == v_limits_lower.loc[self.timesteps[1]]

    def test__lv_allowed_voltage_limits(self):

        # get LVGrid_1 object
        lv_grid = self.edisgo.topology._grids["LVGrid_1"]
        # set voltage at stations' secondary side to known value
        self.edisgo.results._v_res.loc[
            self.timesteps[0], "BusBar_MVGrid_1_LVGrid_1_LV"
        ] = 1.05
        self.edisgo.results._v_res.loc[
            self.timesteps[1], "BusBar_MVGrid_1_LVGrid_1_LV"
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
            self.timesteps[0], "BusBar_MVGrid_1_LVGrid_1_MV"
        ] = 1.03
        self.edisgo.results._v_res.loc[
            self.timesteps[1], "BusBar_MVGrid_1_LVGrid_1_MV"
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
        assert np.isclose(uv_violations.at["Bus_Generator_1", self.timesteps[1]], 0.02)
        assert np.isclose(
            uv_violations.at["Bus_GeneratorFluctuating_3", self.timesteps[0]],
            0.005,
        )
        assert np.isclose(
            ov_violations.at["Bus_GeneratorFluctuating_2", self.timesteps[0]],
            0.01,
        )
        assert np.isclose(uv_violations.at["Bus_Generator_1", self.timesteps[0]], -0.21)

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
            v_violations.at["Bus_GeneratorFluctuating_2", "v_diff_max"], 0.01
        )
        assert v_violations.at["Bus_Generator_1", "time_index"] == self.timesteps[1]

    def test_check_ten_percent_voltage_deviation(self):
        # check without voltage issues greater than 10%
        check_tech_constraints.check_ten_percent_voltage_deviation(self.edisgo)
        # create voltage issues greater 10% and check again
        self.edisgo.results.v_res.at[
            self.timesteps[0], "BusBar_MVGrid_1_LVGrid_9_MV"
        ] = 1.14
        msg = "Maximum allowed voltage deviation of 10% exceeded."
        with pytest.raises(ValueError, match=msg):
            check_tech_constraints.check_ten_percent_voltage_deviation(self.edisgo)
