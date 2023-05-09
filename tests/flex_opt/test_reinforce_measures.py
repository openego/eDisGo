import copy

import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints, reinforce_measures


class TestReinforceMeasures:
    @classmethod
    def setup_class(cls):
        cls.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

        cls.edisgo.set_time_series_worst_case_analysis()
        cls.edisgo.analyze()
        cls.edisgo_root = copy.deepcopy(cls.edisgo)
        cls.timesteps = pd.date_range("1/1/1970", periods=2, freq="H")

    def test_reinforce_mv_lv_station_overloading(self):
        # implicitly checks function _station_overloading

        # create problems such that in LVGrid_1 existing transformer is
        # exchanged with standard transformer and in LVGrid_4 a third
        # transformer is added
        self.edisgo = copy.deepcopy(self.edisgo_root)
        lv_grid_1 = self.edisgo.topology.get_lv_grid(1)
        lv_grid_4 = self.edisgo.topology.get_lv_grid(4)

        crit_lv_stations = pd.DataFrame(
            {
                "s_missing": [0.17, 0.04],
                "time_index": [self.timesteps[1]] * 2,
                "grid": [lv_grid_1, lv_grid_4],
            },
            index=[lv_grid_1.station_name, lv_grid_4.station_name],
        )
        transformer_changes = reinforce_measures.reinforce_mv_lv_station_overloading(
            self.edisgo, crit_lv_stations
        )

        # check transformer changes
        assert len(transformer_changes["added"].keys()) == 2
        assert len(transformer_changes["removed"].keys()) == 1

        assert (
            transformer_changes["added"]["LVGrid_1_station"][0]
            == "LVStation_1_transformer_reinforced_1"
        )
        assert (
            transformer_changes["removed"]["LVGrid_1_station"][0]
            == "LVStation_1_transformer_1"
        )
        assert (
            transformer_changes["added"]["LVGrid_4_station"][0]
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
        trafo_copy = self.edisgo.topology.equipment_data["lv_transformers"].loc[
            "630 kVA"
        ]
        assert trafo_new.bus0 == "BusBar_MVGrid_1_LVGrid_1_MV"
        assert trafo_new.bus1 == "BusBar_MVGrid_1_LVGrid_1_LV"
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
        assert trafo_new.bus0 == "BusBar_MVGrid_1_LVGrid_4_MV"
        assert trafo_new.bus1 == "BusBar_MVGrid_1_LVGrid_4_LV"
        assert trafo_new.r_pu == trafo_copy.r_pu
        assert trafo_new.x_pu == trafo_copy.x_pu
        assert trafo_new.s_nom == trafo_copy.s_nom
        assert trafo_new.type_info == "40 kVA"

    def test_reinforce_hv_mv_station_overloading(self):
        # implicitly checks function _station_overloading

        # check adding transformer of same MVA
        self.edisgo = copy.deepcopy(self.edisgo_root)

        crit_mv_station = pd.DataFrame(
            {
                "s_missing": [19],
                "time_index": [self.timesteps[1]],
                "grid": self.edisgo.topology.mv_grid,
            },
            index=["MVGrid_1_station"],
        )
        transformer_changes = reinforce_measures.reinforce_hv_mv_station_overloading(
            self.edisgo, crit_mv_station
        )
        assert len(transformer_changes["added"]["MVGrid_1_station"]) == 1
        assert len(transformer_changes["removed"]) == 0

        trafo_new = self.edisgo.topology.transformers_hvmv_df.loc[
            "MVStation_1_transformer_reinforced_2"
        ]
        trafo_copy = self.edisgo.topology.transformers_hvmv_df.loc[
            "MVStation_1_transformer_1"
        ]
        assert trafo_new.bus0 == "Bus_primary_MVStation_1"
        assert trafo_new.bus1 == "Bus_MVStation_1"
        assert trafo_new.s_nom == trafo_copy.s_nom
        assert trafo_new.type_info == "40 MVA 110/20 kV"

        # delete added transformer from topology
        self.edisgo.topology.transformers_hvmv_df.drop(
            "MVStation_1_transformer_reinforced_2", inplace=True
        )

        # check replacing current transformers and replacing them with three
        # standard transformers
        crit_mv_station = pd.DataFrame(
            {
                "s_missing": [50],
                "time_index": [self.timesteps[1]],
                "grid": self.edisgo.topology.mv_grid,
            },
            index=["MVGrid_1_station"],
        )
        transformer_changes = reinforce_measures.reinforce_hv_mv_station_overloading(
            self.edisgo, crit_mv_station
        )
        assert len(transformer_changes["added"]["MVGrid_1_station"]) == 3
        assert len(transformer_changes["removed"]["MVGrid_1_station"]) == 1

        trafos = self.edisgo.topology.transformers_hvmv_df
        assert (trafos.bus0 == "Bus_primary_MVStation_1").all()
        assert (trafos.bus1 == "Bus_MVStation_1").all()
        assert (trafos.s_nom == 40).all()
        assert (trafos.type_info == "40 MVA").all()
        assert (
            "MVStation_1_transformer_reinforced_2"
            in transformer_changes["added"]["MVGrid_1_station"]
        )
        # check that removed transformer is removed from topology
        assert (
            "MVStation_1_transformer_1"
            not in self.edisgo.topology.transformers_hvmv_df.index
        )

    def test_reinforce_mv_lv_station_voltage_issues(self):
        self.edisgo = copy.deepcopy(self.edisgo_root)

        crit_stations = pd.DataFrame(
            {
                "abs_max_voltage_dev": [0.03],
                "time_index": [self.timesteps[0]],
                "lv_grid_id": 9.0,
            },
            index=["Bus_secondary_LVStation_9"],
        )

        trafos_pre = self.edisgo.topology.transformers_df

        trafo_changes = reinforce_measures.reinforce_mv_lv_station_voltage_issues(
            self.edisgo, crit_stations
        )

        assert len(trafo_changes) == 1
        assert len(trafo_changes["added"]["LVGrid_9"]) == 1
        assert (
            trafo_changes["added"]["LVGrid_9"][0]
            == "LVStation_9_transformer_reinforced_2"
        )
        # check changes in transformers_df
        assert len(self.edisgo.topology.transformers_df) == (len(trafos_pre) + 1)
        # check values
        trafo_new = self.edisgo.topology.transformers_df.loc[
            "LVStation_9_transformer_reinforced_2"
        ]
        trafo_copy = self.edisgo.topology.equipment_data["lv_transformers"].loc[
            "630 kVA"
        ]
        assert trafo_new.bus0 == "BusBar_MVGrid_1_LVGrid_9_MV"
        assert trafo_new.bus1 == "BusBar_MVGrid_1_LVGrid_9_LV"
        assert trafo_new.r_pu == trafo_copy.r_pu
        assert trafo_new.x_pu == trafo_copy.x_pu
        assert trafo_new.s_nom == trafo_copy.S_nom
        assert trafo_new.type_info == "630 kVA"

    def test_reinforce_lines_voltage_issues(self):
        # MV:
        # * check where node_2_3 is an LV station => problem at
        #   Bus_BranchTee_MVGrid_1_2, leads to disconnection at
        #   Bus_primary_LVStation_1 (Line_10007)
        # * check where node_2_3 is not an LV station but LV station is found
        #   in path => problem at Bus_primary_LVStation_7, leads to
        #   disconnection at Bus_primary_LVStation_7 (Line_10023)
        # * check where node_2_3 is not an LV station and there is also no
        #   LV station in path => problem at Bus_BranchTee_MVGrid_1_11 leads
        #   to disconnection at Bus_BranchTee_MVGrid_1_11 (Line_10028)
        # * check problem in same feeder => Bus_BranchTee_MVGrid_1_10 (node
        #   has higher voltage issue than Bus_BranchTee_MVGrid_1_11, but
        #   Bus_BranchTee_MVGrid_1_10 is farther away from station)
        self.edisgo = copy.deepcopy(self.edisgo_root)

        crit_nodes = pd.DataFrame(
            {
                "abs_max_voltage_dev": [0.08, 0.06, 0.05, 0.04],
                "time_index": [
                    self.timesteps[0],
                    self.timesteps[0],
                    self.timesteps[0],
                    self.timesteps[0],
                ],
            },
            index=[
                "Bus_BranchTee_MVGrid_1_10",
                "Bus_BranchTee_MVGrid_1_11",
                "Bus_BranchTee_MVGrid_1_2",
                "BusBar_MVGrid_1_LVGrid_7_MV",
            ],
        )

        grid = self.edisgo.topology.mv_grid
        lines_changes = reinforce_measures.reinforce_lines_voltage_issues(
            self.edisgo, grid, crit_nodes
        )

        reinforced_lines = lines_changes.keys()
        assert len(lines_changes) == 3
        assert "Line_10028" in reinforced_lines
        assert "Line_10023" in reinforced_lines
        assert "Line_10007" in reinforced_lines
        # check that MV station is one of the buses
        assert (
            "Bus_MVStation_1"
            in self.edisgo.topology.lines_df.loc["Line_10028", ["bus0", "bus1"]].values
        )
        assert (
            "Bus_MVStation_1"
            in self.edisgo.topology.lines_df.loc["Line_10023", ["bus0", "bus1"]].values
        )
        assert (
            "Bus_MVStation_1"
            in self.edisgo.topology.lines_df.loc["Line_10007", ["bus0", "bus1"]].values
        )
        # check other bus
        assert (
            "Bus_BranchTee_MVGrid_1_11"
            in self.edisgo.topology.lines_df.loc["Line_10028", ["bus0", "bus1"]].values
        )
        assert (
            "BusBar_MVGrid_1_LVGrid_3_MV"
            in self.edisgo.topology.lines_df.loc["Line_10023", ["bus0", "bus1"]].values
        )
        assert (
            "BusBar_MVGrid_1_LVGrid_1_MV"
            in self.edisgo.topology.lines_df.loc["Line_10007", ["bus0", "bus1"]].values
        )
        # check line parameters
        std_line_mv = self.edisgo.topology.equipment_data["mv_cables"].loc[
            self.edisgo.config["grid_expansion_standard_equipment"]["mv_line_20kv"]
        ]
        line = self.edisgo.topology.lines_df.loc["Line_10028"]
        assert line.type_info == std_line_mv.name
        assert np.isclose(line.r, std_line_mv.R_per_km * line.length)
        assert np.isclose(
            line.x, std_line_mv.L_per_km * line.length * 2 * np.pi * 50 / 1e3
        )
        assert np.isclose(
            line.s_nom,
            np.sqrt(3) * grid.nominal_voltage * std_line_mv.I_max_th,
        )
        assert line.num_parallel == 1
        line = self.edisgo.topology.lines_df.loc["Line_10023"]
        assert line.type_info == std_line_mv.name
        assert line.num_parallel == 1
        line = self.edisgo.topology.lines_df.loc["Line_10007"]
        assert line.type_info == std_line_mv.name
        assert line.num_parallel == 1

        # check line length of one line
        assert np.isclose(
            self.edisgo.topology.lines_df.loc[
                ["Line_10005", "Line_10026"], "length"
            ].sum()
            + 0.502639122266729,
            self.edisgo.topology.lines_df.at["Line_10028", "length"],
        )

        # LV:
        # * check where node_2_3 is in_building => problem at
        #   Bus_BranchTee_LVGrid_5_2, leads to reinforcement of line
        #   Line_50000003 (which is first line in feeder and not a
        #   standard line)
        # * check where node_2_3 is not in_building => problem at
        #   Bus_BranchTee_LVGrid_5_5, leads to reinforcement of line
        #   Line_50000009 (which is first line in feeder and a standard line)

        crit_nodes = pd.DataFrame(
            {
                "abs_max_voltage_dev": [0.08, 0.05],
                "time_index": [self.timesteps[0], self.timesteps[0]],
            },
            index=["Bus_BranchTee_LVGrid_5_2", "Bus_BranchTee_LVGrid_5_5"],
        )

        grid = self.edisgo.topology.get_lv_grid("LVGrid_5")
        lines_changes = reinforce_measures.reinforce_lines_voltage_issues(
            self.edisgo, grid, crit_nodes
        )

        reinforced_lines = lines_changes.keys()
        assert len(lines_changes) == 2
        assert "Line_50000003" in reinforced_lines
        assert "Line_50000009" in reinforced_lines
        # check that LV station is one of the buses
        assert (
            "BusBar_MVGrid_1_LVGrid_5_LV"
            in self.edisgo.topology.lines_df.loc[
                "Line_50000003", ["bus0", "bus1"]
            ].values
        )
        assert (
            "BusBar_MVGrid_1_LVGrid_5_LV"
            in self.edisgo.topology.lines_df.loc[
                "Line_50000009", ["bus0", "bus1"]
            ].values
        )
        # check other bus
        assert (
            "Bus_BranchTee_LVGrid_5_5"
            in self.edisgo.topology.lines_df.loc[
                "Line_50000009", ["bus0", "bus1"]
            ].values
        )
        assert (
            "Bus_BranchTee_LVGrid_5_1"
            in self.edisgo.topology.lines_df.loc[
                "Line_50000003", ["bus0", "bus1"]
            ].values
        )
        # check line parameters
        std_line = self.edisgo.topology.equipment_data["lv_cables"].loc[
            self.edisgo.config["grid_expansion_standard_equipment"]["lv_line"]
        ]
        line = self.edisgo.topology.lines_df.loc["Line_50000003"]
        assert line.type_info == std_line.name
        assert np.isclose(line.r, std_line.R_per_km * line.length)
        assert np.isclose(
            line.x, std_line.L_per_km * line.length * 2 * np.pi * 50 / 1e3
        )
        assert np.isclose(
            line.s_nom, np.sqrt(3) * grid.nominal_voltage * std_line.I_max_th
        )
        assert line.num_parallel == 1
        line = self.edisgo.topology.lines_df.loc["Line_50000009"]
        assert line.type_info == std_line.name
        assert line.num_parallel == 2
        assert np.isclose(line.r, 0.02781 / 2)
        assert np.isclose(line.x, 0.010857344210806 / 2)
        assert np.isclose(line.s_nom, 0.190525588832576 * 2)

    def test_reinforce_lines_overloading(self):
        # * check for needed parallel standard lines (MV and LV) => problems at
        #   Line_10007 and Line_70000006
        # * check for parallel line of same type => problems at Line_10019
        #   and Line_50000002
        # * check for replacement by parallel standard lines (MV and LV) =>
        #   problems at Line_10003 and Line_60000001
        self.edisgo = copy.deepcopy(self.edisgo_root)

        # create crit_lines dataframe
        crit_lines = pd.DataFrame(
            {
                "max_rel_overload": [2.3261, 1.1792, 2.3261, 2.02, 1.02, 2.1],
                "voltage_level": ["mv"] * 3 + ["lv"] * 3,
            },
            index=[
                "Line_10007",  # parallel standard line
                "Line_10019",  # second parallel line of same type
                "Line_10003",  # exchange by standard line
                "Line_70000006",  # parallel standard line
                "Line_50000002",  # second parallel line of same type
                "Line_60000001",  # exchange by standard line
            ],
        )

        lines_changes = reinforce_measures.reinforce_lines_overloading(
            self.edisgo, crit_lines
        )

        assert len(lines_changes) == 6
        assert lines_changes["Line_10003"] == 2
        assert lines_changes["Line_10007"] == 2
        assert lines_changes["Line_10019"] == 1
        assert lines_changes["Line_70000006"] == 2
        assert lines_changes["Line_50000002"] == 1
        assert lines_changes["Line_60000001"] == 1

        # check lines that were already standard lines and where parallel
        # standard lines were added
        line = self.edisgo.topology.lines_df.loc["Line_10007"]
        assert line.type_info == "NA2XS2Y 3x1x240"
        assert np.isclose(line.r, 0.13 * line.length / 3)
        assert np.isclose(line.x, 0.3597 * 2 * np.pi * 50 / 1e3 * line.length / 3)
        assert np.isclose(line.s_nom, 7.27461339178928 * 3)
        assert line.num_parallel == 3

        line = self.edisgo.topology.lines_df.loc["Line_70000006"]
        assert line.type_info == "NAYY 4x1x150"
        assert np.isclose(line.r, 0.206 * line.length / 3)
        assert np.isclose(line.x, 0.256 * 2 * np.pi * 50 / 1e3 * line.length / 3)
        assert np.isclose(line.s_nom, 0.275 * 0.4 * np.sqrt(3) * 3)
        assert line.num_parallel == 3

        # check lines where a second parallel line of the same type is added
        line = self.edisgo.topology.lines_df.loc["Line_10019"]
        assert line.type_info == "48-AL1/8-ST1A"
        assert np.isclose(line.r, 0.14942525073 / 2)
        assert np.isclose(line.x, 0.14971115095836038 / 2)
        assert np.isclose(line.s_nom, 7.274613391789284 * 2)
        assert line.num_parallel == 2

        line = self.edisgo.topology.lines_df.loc["Line_50000002"]
        assert line.type_info == "NAYY 4x1x35"
        assert np.isclose(line.r, 0.02604 / 2)
        assert np.isclose(line.x, 0.002554114827369 / 2)
        assert np.isclose(line.s_nom, 0.085216899732389 * 2)
        assert line.num_parallel == 2

        # check lines that were exchanged by parallel standard lines
        line = self.edisgo.topology.lines_df.loc["Line_10003"]
        assert line.type_info == "NA2XS2Y 3x1x240"
        assert np.isclose(line.r, 0.13 * line.length / 2)
        assert np.isclose(line.x, 0.3597 * 2 * np.pi * 50 / 1e3 * line.length / 2)
        assert np.isclose(line.s_nom, 0.417 * 20 * np.sqrt(3) * 2)
        assert line.num_parallel == 2

        line = self.edisgo.topology.lines_df.loc["Line_60000001"]
        assert line.type_info == "NAYY 4x1x150"
        assert np.isclose(line.r, 0.206 * line.length)
        assert np.isclose(line.x, 0.256 * 2 * np.pi * 50 / 1e3 * line.length)
        assert np.isclose(line.s_nom, 0.275 * 0.4 * np.sqrt(3))
        assert line.num_parallel == 1

    def test_separate_lv_grid(self):
        self.edisgo = copy.deepcopy(self.edisgo_root)

        crit_lines_lv = check_tech_constraints.lv_line_max_relative_overload(
            self.edisgo
        )

        all_lv_grid_ids = [
            lv_grid.id for lv_grid in self.edisgo.topology.mv_grid.lv_grids
        ]

        lv_grid_ids = (
            self.edisgo.topology.buses_df.loc[
                self.edisgo.topology.lines_df.loc[crit_lines_lv.index].bus0
            ]
            .lv_grid_id.unique()
            .astype(int)
        )

        lv_grids = [
            lv_grid
            for lv_grid in self.edisgo.topology.mv_grid.lv_grids
            if lv_grid.id in lv_grid_ids
        ]

        for lv_grid in lv_grids:
            orig_g = copy.deepcopy(lv_grid)
            grid_id = orig_g.id

            reinforce_measures.separate_lv_grid(self.edisgo, lv_grid)

            new_g_0 = [
                g for g in self.edisgo.topology.mv_grid.lv_grids if g.id == grid_id
            ][0]

            try:
                new_g_1 = [
                    g
                    for g in self.edisgo.topology.mv_grid.lv_grids
                    if g.id not in all_lv_grid_ids
                ][0]

                all_lv_grid_ids.append(new_g_1.id)
            except IndexError:
                continue

            assert np.isclose(
                orig_g.charging_points_df.p_set.sum(),
                new_g_0.charging_points_df.p_set.sum()
                + new_g_1.charging_points_df.p_set.sum(),
            )

            assert np.isclose(
                orig_g.generators_df.p_nom.sum(),
                new_g_0.generators_df.p_nom.sum() + new_g_1.generators_df.p_nom.sum(),
            )

            assert np.isclose(
                orig_g.loads_df.p_set.sum(),
                new_g_0.loads_df.p_set.sum() + new_g_1.loads_df.p_set.sum(),
            )

            assert len(orig_g.lines_df) == len(new_g_0.lines_df) + len(new_g_1.lines_df)
