import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints


class TestCheckTechConstraints:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        self.timesteps = self.edisgo.timeseries.timeindex

    @pytest.fixture(autouse=True)
    def run_power_flow(self):
        """
        Fixture to run new power flow before each test.

        """
        self.edisgo.analyze()

    def test_mv_line_max_relative_overload(self):
        # implicitly checks function _line_overload

        df = check_tech_constraints.mv_line_max_relative_overload(self.edisgo)
        # check shape of dataframe
        assert (4, 3) == df.shape
        # check relative overload of one line
        assert np.isclose(
            df.at["Line_10005", "max_rel_overload"],
            self.edisgo.results.s_res.at[self.timesteps[3], "Line_10005"]
            / 7.274613391789284,
        )
        assert df.at["Line_10005", "time_index"] == self.timesteps[3]

    def test_lv_line_max_relative_overload(self):
        # implicitly checks function _line_overload

        df = check_tech_constraints.lv_line_max_relative_overload(self.edisgo)
        # check shape of dataframe
        assert (2, 3) == df.shape
        # check relative overload of one line
        assert np.isclose(
            df.at["Line_50000002", "max_rel_overload"],
            self.edisgo.results.s_res.at[self.timesteps[0], "Line_50000002"]
            / 0.08521689973238901,
        )
        assert df.at["Line_50000002", "time_index"] == self.timesteps[0]

        # test for single LV grid
        lv_grid_id = 5
        self.edisgo.analyze(mode="lv", lv_grid_id=lv_grid_id)
        df = check_tech_constraints.lv_line_max_relative_overload(
            self.edisgo, lv_grid_id=lv_grid_id
        )
        # check shape of dataframe
        assert (1, 3) == df.shape
        # check relative overload of line
        assert np.isclose(
            df.at["Line_50000002", "max_rel_overload"],
            self.edisgo.results.s_res.at[self.timesteps[0], "Line_50000002"]
            / 0.08521689973238901,
        )
        assert df.at["Line_50000002", "time_index"] == self.timesteps[0]

    def test_lines_allowed_load(self):
        # check with default value (all lines)
        df = check_tech_constraints.lines_allowed_load(self.edisgo)
        # check shape of dataframe
        assert (4, 131) == df.shape
        # check values (feed-in case)
        assert np.isclose(
            df.at[self.timesteps[2], "Line_10005"],
            7.27461339178928,
        )
        assert np.isclose(
            df.at[self.timesteps[2], "Line_50000002"],
            0.08521689973238901,
        )
        # check values (load case)
        assert np.isclose(
            df.at[self.timesteps[0], "Line_10005"],
            7.274613391789284,
        )
        assert np.isclose(
            df.at[self.timesteps[0], "Line_50000002"],
            0.08521689973238901,
        )

        # check with specifying lines
        df = check_tech_constraints.lines_allowed_load(
            self.edisgo, lines=["Line_10005", "Line_50000002"]
        )
        # check shape of dataframe
        assert (4, 2) == df.shape

    def test__lines_allowed_load_voltage_level(self):
        # check for MV
        df = check_tech_constraints._lines_allowed_load_voltage_level(self.edisgo, "mv")
        # check shape of dataframe
        assert (4, 30) == df.shape
        # check in feed-in case
        assert np.isclose(
            df.at[self.timesteps[2], "Line_10005"],
            7.27461339178928,
        )
        # check in load case
        assert np.isclose(
            df.at[self.timesteps[0], "Line_10005"],
            7.274613391789284,
        )
        assert np.isclose(
            df.at[self.timesteps[0], "Line_10024"],
            7.27461339178928,
        )

        # check for LV
        df = check_tech_constraints._lines_allowed_load_voltage_level(self.edisgo, "lv")
        # check shape of dataframe
        assert (4, 101) == df.shape
        # check in feed-in case
        assert np.isclose(
            df.at[self.timesteps[2], "Line_50000002"],
            0.08521689973238901,
        )
        # check in load case
        assert np.isclose(
            df.at[self.timesteps[0], "Line_50000002"],
            0.08521689973238901,
        )

    def test_lines_relative_load(self):
        # check with default value (all lines)
        df = check_tech_constraints.lines_relative_load(self.edisgo)
        # check shape of dataframe
        assert (4, 131) == df.shape
        # check values (feed-in case)
        assert np.isclose(
            df.at[self.timesteps[2], "Line_10005"], 7.74132 / 7.27461, atol=1e-5
        )
        assert np.isclose(
            df.at[self.timesteps[2], "Line_50000002"], 0.012644 / 0.085216, atol=1e-5
        )
        # check values (load case)
        assert np.isclose(
            df.at[self.timesteps[0], "Line_10005"], 0.00142 / 7.27461, atol=1e-5
        )

        # check with specifying lines
        df = check_tech_constraints.lines_relative_load(
            self.edisgo, lines=["Line_10005", "Line_50000002"]
        )
        # check shape of dataframe
        assert (4, 2) == df.shape

    def test_hv_mv_station_max_overload(self):
        # implicitly checks function _station_overload

        # create over-load problem with highest over-load in first time step (as it is
        # a load case)
        self.edisgo.results.pfa_slack = pd.DataFrame(
            data={"p": [30, 25, 30, 20], "q": [30, 25, 30, 20]}, index=self.timesteps
        )

        df = check_tech_constraints.hv_mv_station_max_overload(self.edisgo)
        # check shape of dataframe
        assert (1, 3) == df.shape
        # check missing transformer capacity
        assert np.isclose(
            df.at["MVGrid_1_station", "s_missing"],
            (np.hypot(30, 30) - 40),
        )
        assert df.at["MVGrid_1_station", "time_index"] == self.timesteps[0]

    def test_mv_lv_station_max_overload(self):
        # implicitly checks function _station_overload

        df = check_tech_constraints.mv_lv_station_max_overload(self.edisgo)
        # check shape of dataframe
        assert (4, 3) == df.shape
        # check missing transformer capacity of one grid
        assert np.isclose(
            df.at["LVGrid_1_station", "s_missing"],
            self.edisgo.results.s_res.at[self.timesteps[1], "LVStation_1_transformer_1"]
            - 0.16,
        )
        assert df.at["LVGrid_1_station", "time_index"] == self.timesteps[0]

        # test for single LV grid
        lv_grid_id = 1
        df = check_tech_constraints.mv_lv_station_max_overload(
            self.edisgo, lv_grid_id=lv_grid_id
        )
        assert (1, 3) == df.shape
        assert df.at["LVGrid_1_station", "time_index"] == self.timesteps[0]

    def test__station_load(self):
        # check LV grid
        grid = self.edisgo.topology.get_lv_grid(4)
        df = check_tech_constraints._station_load(self.edisgo, grid)
        # check shape and column of dataframe
        assert (4, 1) == df.shape
        assert grid.station_name in df.columns

        # check MV grid
        grid = self.edisgo.topology.mv_grid
        df = check_tech_constraints._station_load(self.edisgo, grid)
        # check shape and column of dataframe
        assert (4, 1) == df.shape
        assert grid.station_name in df.columns

        # check ValueErrors
        msg = "Inserted grid is invalid."
        with pytest.raises(ValueError, match=msg):
            check_tech_constraints._station_load(self.edisgo, str(grid))

        self.edisgo.analyze(mode="lv", lv_grid_id=1)
        msg = "MV was not included in power flow analysis"
        with pytest.raises(ValueError, match=msg):
            check_tech_constraints._station_load(self.edisgo, grid)

        # check KeyError in case grid was not included in power flow
        self.edisgo.analyze(mode="mv")
        grid = self.edisgo.topology.get_lv_grid(4)
        msg = "LVStation_4_transformer_1"
        with pytest.raises(KeyError, match=msg):
            check_tech_constraints._station_load(self.edisgo, grid)

    def test__station_allowed_load(self):
        # check LV grid
        grid = self.edisgo.topology.get_lv_grid(4)
        df = check_tech_constraints._station_allowed_load(self.edisgo, grid)
        # check shape of dataframe
        assert (4, 1) == df.shape
        # check values
        exp = pd.DataFrame(
            {grid.station_name: [0.05] * len(self.edisgo.timeseries.timeindex)},
            index=self.edisgo.timeseries.timeindex,
        )
        assert_frame_equal(df, exp)

        # check MV grid
        grid = self.edisgo.topology.mv_grid
        df = check_tech_constraints._station_allowed_load(self.edisgo, grid)
        # check shape of dataframe
        assert (4, 1) == df.shape
        # check values
        load_cases = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("load")
        ]
        assert np.isclose(40.0, df.loc[load_cases.values].values).all()
        feed_in_cases = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("feed")
        ]
        assert np.isclose(40.0, df.loc[feed_in_cases.values].values).all()

    def test_stations_allowed_load(self):
        # check without specifying a grid
        df = check_tech_constraints.stations_allowed_load(self.edisgo)
        # check shape of dataframe
        assert (4, 11) == df.shape
        # check values
        exp = pd.DataFrame(
            {"LVGrid_4_station": [0.05] * len(self.edisgo.timeseries.timeindex)},
            index=self.edisgo.timeseries.timeindex,
        )
        assert_frame_equal(df.loc[:, ["LVGrid_4_station"]], exp)
        load_cases = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("load")
        ]
        assert np.isclose(
            40.0, df.loc[load_cases.values, "MVGrid_1_station"].values
        ).all()

        # check with specifying grids
        grids = [self.edisgo.topology.mv_grid, self.edisgo.topology.get_lv_grid(1)]
        df = check_tech_constraints.stations_allowed_load(self.edisgo, grids)
        # check shape of dataframe
        assert (4, 2) == df.shape
        # check values
        exp = pd.DataFrame(
            {"LVGrid_1_station": [0.16] * len(self.edisgo.timeseries.timeindex)},
            index=self.edisgo.timeseries.timeindex,
        )
        assert_frame_equal(df.loc[:, ["LVGrid_1_station"]], exp)
        assert np.isclose(
            40.0, df.loc[load_cases.values, "MVGrid_1_station"].values
        ).all()
        feed_in_cases = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("feed")
        ]
        assert np.isclose(
            40.0, df.loc[feed_in_cases.values, "MVGrid_1_station"].values
        ).all()

    def test_stations_relative_load(self):
        # check without specifying grids
        df = check_tech_constraints.stations_relative_load(self.edisgo)
        # check shape of dataframe
        assert (4, 11) == df.shape
        # check values
        load_cases = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("load")
        ]
        assert np.isclose(
            0.02853, df.loc[load_cases.values, "LVGrid_4_station"].values, atol=1e-5
        ).all()

        # check with specifying grids
        grids = [self.edisgo.topology.mv_grid, self.edisgo.topology.get_lv_grid(4)]
        df = check_tech_constraints.stations_relative_load(self.edisgo, grids)
        # check shape of dataframe
        assert (4, 2) == df.shape
        # check values
        assert np.isclose(
            0.02853, df.loc[load_cases.values, "LVGrid_4_station"].values, atol=1e-5
        ).all()

        # check with missing grids
        self.edisgo.analyze(mode="mv")
        df = check_tech_constraints.stations_relative_load(self.edisgo)
        # check shape of dataframe
        assert (4, 1) == df.shape
        # check values
        load_cases = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("load")
        ]
        assert np.isclose(
            0.033765, df.loc[load_cases.values, "MVGrid_1_station"].values, atol=1e-5
        ).all()

    def test_components_relative_load(self):
        # check with power flow results available for all components
        df = check_tech_constraints.components_relative_load(self.edisgo)
        # check shape of dataframe
        assert (4, 142) == df.shape
        # check values
        load_cases = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("load")
        ]
        assert np.isclose(
            0.02853, df.loc[load_cases.values, "LVGrid_4_station"].values, atol=1e-5
        ).all()
        assert np.isclose(
            df.at[self.timesteps[0], "Line_10005"], 0.00142 / 7.27461, atol=1e-5
        )

        # check with power flow results not available for all components
        self.edisgo.analyze(mode="mvlv")
        df = check_tech_constraints.components_relative_load(self.edisgo)
        # check shape of dataframe
        assert (4, 41) == df.shape
        # check values
        assert np.isclose(
            0.02852, df.loc[load_cases.values, "LVGrid_4_station"].values, atol=1e-5
        ).all()

        # check with missing grids
        self.edisgo.analyze(mode="mv")
        df = check_tech_constraints.components_relative_load(self.edisgo)
        # check shape of dataframe
        assert (4, 31) == df.shape
        # check values
        load_cases = self.edisgo.timeseries.timeindex_worst_cases[
            self.edisgo.timeseries.timeindex_worst_cases.index.str.contains("load")
        ]
        assert np.isclose(
            0.033765, df.loc[load_cases.values, "MVGrid_1_station"].values, atol=1e-5
        ).all()

        # check single LV grid
        self.edisgo.analyze(mode="lv", lv_grid_id=1)
        df = check_tech_constraints.components_relative_load(self.edisgo)
        # check shape of dataframe
        assert (4, 14) == df.shape

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
        # create undervoltage at bus2
        self.edisgo.results._v_res.loc[self.timesteps[0], bus2] = 0.895

    def test_voltage_issues(self):
        # ########################## check mv mode #################################

        # ################### check with no voltage issues
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, voltage_level="mv"
        )
        assert voltage_issues.empty

        # create voltage issues
        self.mv_voltage_issues()

        # ################ check with voltage issues and default values
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, voltage_level="mv"
        )
        # check shape of dataframe
        assert (3, 3) == voltage_issues.shape
        # check under- and overvoltage deviation values
        assert list(voltage_issues.index.values) == [
            "Bus_Generator_1",
            "Bus_GeneratorFluctuating_3",
            "Bus_GeneratorFluctuating_2",
        ]
        assert np.isclose(
            voltage_issues.at["Bus_GeneratorFluctuating_2", "abs_max_voltage_dev"],
            0.06,
        )
        assert voltage_issues.at["Bus_Generator_1", "time_index"] == self.timesteps[1]

        # ############# check with voltage issues and split_voltage_band = False
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, split_voltage_band=False, voltage_level="mv"
        )
        # check shape of dataframe
        assert (3, 3) == voltage_issues.shape
        # check under- and overvoltage deviation values
        assert list(voltage_issues.index.values) == [
            "Bus_Generator_1",
            "Bus_GeneratorFluctuating_2",
            "Bus_GeneratorFluctuating_3",
        ]
        assert np.isclose(
            voltage_issues.at["Bus_GeneratorFluctuating_2", "abs_max_voltage_dev"],
            0.01,
        )
        assert voltage_issues.at["Bus_Generator_1", "time_index"] == self.timesteps[1]

        # ########################## check lv mode #################################

        # ######## check with default values that there are no voltage issues
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, voltage_level="lv"
        )
        assert voltage_issues.empty

        # ############### check with default
        # create voltage issue in LVGrid_6
        self.edisgo.results.v_res.at[
            self.timesteps[2], "BusBar_MVGrid_1_LVGrid_6_LV"
        ] = 1.14
        self.edisgo.results.v_res.at[
            self.timesteps[2], "Bus_BranchTee_LVGrid_6_1"
        ] = 1.18
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, voltage_level="lv"
        )
        assert len(voltage_issues) == 2
        assert np.isclose(
            voltage_issues.at["Bus_BranchTee_LVGrid_6_1", "abs_max_voltage_dev"],
            0.005,
        )
        assert voltage_issues.index[0] == "Bus_BranchTee_LVGrid_6_2"

        # ################ check with split_voltage_band = False
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, voltage_level="lv", split_voltage_band=False
        )
        assert len(voltage_issues) == 1
        assert voltage_issues.index[0] == "Bus_BranchTee_LVGrid_6_1"
        assert np.isclose(
            voltage_issues.at["Bus_BranchTee_LVGrid_6_1", "abs_max_voltage_dev"],
            0.08,
        )

        # ################ check with single LV grid
        lv_grid_id = 1
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo,
            voltage_level="lv",
            split_voltage_band=False,
            lv_grid_id=lv_grid_id,
        )
        assert len(voltage_issues) == 0

        # ########################## check mv_lv mode ###############################
        # ############## check that voltage issue in station of LVGrid_6 is detected
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, voltage_level="mv_lv"
        )
        assert len(voltage_issues) == 1
        assert np.isclose(
            voltage_issues.at["BusBar_MVGrid_1_LVGrid_6_LV", "abs_max_voltage_dev"],
            0.125027,
        )

        # ######### check with split_voltage_band = False and mode="stations"
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, voltage_level="mv_lv", split_voltage_band=False
        )
        assert len(voltage_issues) == 1
        assert np.isclose(
            voltage_issues.at["BusBar_MVGrid_1_LVGrid_6_LV", "abs_max_voltage_dev"],
            0.04,
        )

        # ################ check with single LV grid
        lv_grid_id = 1
        # uses same voltage issues as created above
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo,
            voltage_level="mv_lv",
            split_voltage_band=False,
            lv_grid_id=lv_grid_id,
        )
        assert len(voltage_issues) == 0

        # ########################## check mode None ###############################

        # ################ check with voltage issues and default values
        voltage_issues = check_tech_constraints.voltage_issues(
            self.edisgo, voltage_level=None
        )
        # check shape of dataframe
        assert (6, 3) == voltage_issues.shape

    def test__voltage_issues_helper(self):
        # create voltage issues
        self.mv_voltage_issues()

        v_violations = check_tech_constraints._voltage_issues_helper(
            self.edisgo,
            self.edisgo.topology.mv_grid.buses_df.index,
            split_voltage_band=False,
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
            v_violations.at["Bus_GeneratorFluctuating_2", "abs_max_voltage_dev"], 0.01
        )
        assert v_violations.at["Bus_Generator_1", "time_index"] == self.timesteps[1]

    def test_allowed_voltage_limits(self):
        lv_grid_1 = self.edisgo.topology.get_lv_grid(1)
        lv_grid_3 = self.edisgo.topology.get_lv_grid(3)

        # ############## test with default values
        # set voltage at stations' primary side to known value
        self.edisgo.results._v_res.loc[
            self.timesteps[3], "BusBar_MVGrid_1_LVGrid_1_MV"
        ] = 1.03
        self.edisgo.results._v_res.loc[
            self.timesteps[1], "BusBar_MVGrid_1_LVGrid_3_MV"
        ] = 0.99
        # set voltage at stations' secondary side to known value
        self.edisgo.results._v_res.loc[
            self.timesteps[2], "BusBar_MVGrid_1_LVGrid_1_LV"
        ] = 1.05
        self.edisgo.results._v_res.loc[
            self.timesteps[1], "BusBar_MVGrid_1_LVGrid_3_LV"
        ] = 0.97

        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints.allowed_voltage_limits(self.edisgo)

        # check shape
        assert v_limits_lower.shape == (4, len(self.edisgo.topology.buses_df))
        assert v_limits_upper.shape == (4, len(self.edisgo.topology.buses_df))

        # check values
        # mv
        assert (
            (
                1.05
                == v_limits_upper.loc[
                    :, ["Bus_GeneratorFluctuating_8", "BusBar_MVGrid_1_LVGrid_1_MV"]
                ]
            )
            .all()
            .all()
        )
        assert (
            (
                0.985
                == v_limits_lower.loc[
                    :, ["Bus_GeneratorFluctuating_8", "BusBar_MVGrid_1_LVGrid_1_MV"]
                ]
            )
            .all()
            .all()
        )
        # stations
        assert (
            1.03 + 0.015
            == v_limits_upper.at[self.timesteps[3], "BusBar_MVGrid_1_LVGrid_1_LV"]
        )
        assert (
            1.03 - 0.02
            == v_limits_lower.at[self.timesteps[3], "BusBar_MVGrid_1_LVGrid_1_LV"]
        )
        assert (
            0.99 - 0.02
            == v_limits_lower.at[self.timesteps[1], "BusBar_MVGrid_1_LVGrid_3_LV"]
        )
        # lv
        assert (
            1.05 + 0.035
            == v_limits_upper.loc[
                self.timesteps[2],
                lv_grid_1.buses_df.index.drop(lv_grid_1.station.index[0]),
            ]
        ).all()
        assert (
            1.05 - 0.065
            == v_limits_lower.loc[
                self.timesteps[2],
                lv_grid_1.buses_df.index.drop(lv_grid_1.station.index[0]),
            ]
        ).all()
        assert (
            0.97 - 0.065
            == v_limits_lower.loc[
                self.timesteps[1],
                lv_grid_3.buses_df.index.drop(lv_grid_3.station.index[0]),
            ]
        ).all()

        # ############## test with specifying buses
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints.allowed_voltage_limits(
            self.edisgo,
            buses=["BusBar_MVGrid_1_LVGrid_3_LV", "Bus_GeneratorFluctuating_8"],
        )

        # check shape
        assert v_limits_lower.shape == (4, 2)
        assert v_limits_upper.shape == (4, 2)

        # ############## test 10 percent
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints.allowed_voltage_limits(
            self.edisgo,
            buses=["BusBar_MVGrid_1_LVGrid_3_LV", "Bus_GeneratorFluctuating_8"],
            split_voltage_band=False,
        )

        # check shape
        assert v_limits_lower.shape == (4, 2)
        assert v_limits_upper.shape == (4, 2)

        # ############### test with missing power flow results
        self.edisgo.analyze(mode="lv", lv_grid_id=1)
        upper, lower = check_tech_constraints.allowed_voltage_limits(
            self.edisgo, buses=self.edisgo.topology.buses_df.index
        )
        assert upper.shape == (4, 45)

        self.edisgo.analyze(mode="lv", lv_grid_id=1)
        upper, lower = check_tech_constraints.allowed_voltage_limits(
            self.edisgo,
            buses=self.edisgo.topology.buses_df.index.drop(
                self.edisgo.topology.mv_grid.buses_df.index
            ),
        )
        assert upper.shape == (4, 14)

        self.edisgo.analyze(mode="mv")
        upper, lower = check_tech_constraints.allowed_voltage_limits(
            self.edisgo, buses=self.edisgo.topology.buses_df.index
        )
        assert upper.shape == (4, 41)

    def test__mv_allowed_voltage_limits(self):
        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._mv_allowed_voltage_limits(self.edisgo)

        assert 1.05 == v_limits_upper
        assert 0.985 == v_limits_lower

    def test__lv_allowed_voltage_limits(self):
        lv_grid_1 = self.edisgo.topology.get_lv_grid(1)
        lv_grid_3 = self.edisgo.topology.get_lv_grid(3)

        # ############## test with default values

        # set voltage at stations' secondary side to known value
        self.edisgo.results._v_res.loc[
            self.timesteps[2], "BusBar_MVGrid_1_LVGrid_1_LV"
        ] = 1.05
        self.edisgo.results._v_res.loc[
            self.timesteps[0], "BusBar_MVGrid_1_LVGrid_1_LV"
        ] = 0.98
        self.edisgo.results._v_res.loc[
            self.timesteps[1], "BusBar_MVGrid_1_LVGrid_3_LV"
        ] = 0.97

        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._lv_allowed_voltage_limits(self.edisgo, mode=None)

        # check shape
        assert v_limits_lower.shape == (4, 101)
        assert v_limits_upper.shape == (4, 101)

        # check values

        assert (
            1.05 + 0.035
            == v_limits_upper.loc[
                self.timesteps[2],
                lv_grid_1.buses_df.index.drop(lv_grid_1.station.index[0]),
            ]
        ).all()
        assert (
            0.98 + 0.035
            == v_limits_upper.loc[
                self.timesteps[0],
                lv_grid_1.buses_df.index.drop(lv_grid_1.station.index[0]),
            ]
        ).all()
        assert (
            1.05 - 0.065
            == v_limits_lower.loc[
                self.timesteps[2],
                lv_grid_1.buses_df.index.drop(lv_grid_1.station.index[0]),
            ]
        ).all()
        assert (
            0.98 - 0.065
            == v_limits_lower.loc[
                self.timesteps[0],
                lv_grid_1.buses_df.index.drop(lv_grid_1.station.index[0]),
            ]
        ).all()
        assert (
            0.97 - 0.065
            == v_limits_lower.loc[
                self.timesteps[1],
                lv_grid_3.buses_df.index.drop(lv_grid_3.station.index[0]),
            ]
        ).all()

        # ############## test with mode stations and providing grids

        # set voltage at stations' primary side to known value
        self.edisgo.results._v_res.loc[
            self.timesteps[3], "BusBar_MVGrid_1_LVGrid_1_MV"
        ] = 1.03
        self.edisgo.results._v_res.loc[
            self.timesteps[1], "BusBar_MVGrid_1_LVGrid_1_MV"
        ] = 0.99

        (
            v_limits_upper,
            v_limits_lower,
        ) = check_tech_constraints._lv_allowed_voltage_limits(
            self.edisgo, mode="stations", lv_grids=[lv_grid_1]
        )

        # check shape
        assert v_limits_lower.shape == (4, 1)
        assert v_limits_upper.shape == (4, 1)

        # check values
        assert (
            1.03 + 0.015
            == v_limits_upper.at[self.timesteps[3], "BusBar_MVGrid_1_LVGrid_1_LV"]
        )
        assert (
            0.99 + 0.015
            == v_limits_upper.at[self.timesteps[1], "BusBar_MVGrid_1_LVGrid_1_LV"]
        )
        assert (
            1.03 - 0.02
            == v_limits_lower.at[self.timesteps[3], "BusBar_MVGrid_1_LVGrid_1_LV"]
        )
        assert (
            0.99 - 0.02
            == v_limits_lower.at[self.timesteps[1], "BusBar_MVGrid_1_LVGrid_1_LV"]
        )

    def test_voltage_deviation_from_allowed_voltage_limits(self):
        # create MV voltage issues
        self.mv_voltage_issues()

        # ############## test with default values
        voltage_dev = (
            check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(
                self.edisgo
            )
        )

        assert voltage_dev.shape == (4, 142)
        # check that there are voltage issues created through mv_voltage_issues() and
        # at "BusBar_MVGrid_1_LVGrid_6_LV" detected
        comps_with_v_issues = (
            voltage_dev[voltage_dev != 0].dropna(how="all", axis=1).columns
        )
        comps_with_v_issues_expected = [
            "BusBar_MVGrid_1_LVGrid_6_LV",
            "Bus_Generator_1",
            "Bus_GeneratorFluctuating_2",
            "Bus_GeneratorFluctuating_3",
        ]
        assert len(comps_with_v_issues) == 4
        assert (
            len([_ for _ in comps_with_v_issues_expected if _ in comps_with_v_issues])
            == 4
        )
        assert np.isclose(
            voltage_dev.at[self.timesteps[0], "BusBar_MVGrid_1_LVGrid_6_LV"],
            -0.0106225,
        )
        assert np.isclose(
            voltage_dev.at[self.timesteps[0], "Bus_Generator_1"],
            1.11 - 1.05,
        )
        assert np.isclose(
            voltage_dev.at[self.timesteps[1], "Bus_Generator_1"],
            0.88 - 0.985,
        )

        # ############## test with split_voltage_band False
        voltage_dev = (
            check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(
                self.edisgo, split_voltage_band=False
            )
        )

        assert voltage_dev.shape == (4, 142)
        # check that "BusBar_MVGrid_1_LVGrid_6_LV" does now not have any voltage issues
        comps_with_v_issues = (
            voltage_dev[voltage_dev != 0].dropna(how="all", axis=1).columns
        )
        assert len(comps_with_v_issues) == 3
        assert "BusBar_MVGrid_1_LVGrid_6_LV" not in comps_with_v_issues
        assert np.isclose(
            voltage_dev.at[self.timesteps[0], "Bus_GeneratorFluctuating_2"],
            1.11 - 1.1,
        )

        # ############## test with specifying buses
        voltage_dev = (
            check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(
                self.edisgo,
                split_voltage_band=False,
                buses=["Bus_GeneratorFluctuating_3", "Bus_BranchTee_LVGrid_1_2"],
            )
        )

        assert voltage_dev.shape == (4, 2)
        assert (voltage_dev.loc[:, "Bus_BranchTee_LVGrid_1_2"] == 0.0).all()
        assert np.isclose(
            voltage_dev.at[self.timesteps[0], "Bus_GeneratorFluctuating_3"],
            0.895 - 0.9,
        )
