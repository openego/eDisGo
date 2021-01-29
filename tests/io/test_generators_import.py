import pandas as pd
import numpy as np
import pytest
from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.network.grids import LVGrid
from edisgo.io import generators_import as generators_import


class TestGeneratorsImport:
    """
    Tests all functions in generators_import.py except where test grid
    can be used. oedb function is tested separately as a real ding0 grid
    needs to be used.

    """

    @pytest.yield_fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_path,
            worst_case_analysis="worst-case"
        )

    def test_connect_to_mv(self):
        # ToDo add tests for charging points

        # ######### Generator #############
        # test voltage level 4
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_2", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_2", "y"]
        geom = Point((x, y))
        test_gen = {
            "generator_id": 12345,
            "p_nom": 2.5,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"],
            "voltage_level": 4
        }

        comp_name = generators_import.connect_to_mv(
            self.edisgo, test_gen)

        # check if number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(
            new_bus)
        assert len(new_line_df) == 1
        # check that other bus of new line is the station
        assert (self.edisgo.topology.mv_grid.station.index[0] ==
                new_line_df.bus0.values[0])
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == test_gen["p_nom"]

        # test voltage level 5 (line split)
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_2", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = {
            "generator_id": 123456,
            "p_nom": 2.5,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"],
            "voltage_level": 5
        }

        comp_name = generators_import.connect_to_mv(
            self.edisgo, test_gen)

        # check if number of buses increased (by two because closest connection
        # object is a line)
        assert len(buses_before) + 2 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 2 == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(
            new_bus)
        assert len(new_line_df) == 1
        assert "Bus_Generator_123456" in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]])
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == test_gen["p_nom"]

        # test voltage level 5 (connected to bus)
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = {
            "generator_id": 123456,
            "p_nom": 2.5,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"],
            "voltage_level": 5
        }

        comp_name = generators_import.connect_to_mv(
            self.edisgo, test_gen)

        # check if number of buses increased (by one because closest connection
        # object is a bus)
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) +1  == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == test_gen["p_nom"]

        # ######### Charging Point #############
        # method not different from generators, wherefore only one voltage
        # level is tested
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        charging_points_before = self.edisgo.topology.charging_points_df

        # add charging point
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_2", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_2", "y"]
        geom = Point((x, y))
        test_gen = {
            "geom": geom,
            "p_nom": 2.5,
            "use_case": "fast",
            "number": 10,
            "voltage_level": 4
        }

        comp_name = generators_import.connect_to_mv(
            self.edisgo, test_gen, comp_type="ChargingPoint")

        # check if number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check if number of charging points increased
        assert len(charging_points_before) + 1 == len(
            self.edisgo.topology.charging_points_df)

        # check new bus
        new_bus = self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(
            new_bus)
        assert len(new_line_df) == 1
        # check that other bus of new line is the station
        assert (self.edisgo.topology.mv_grid.station.index[0] ==
                new_line_df.bus0.values[0])
        # check new generator
        assert self.edisgo.topology.charging_points_df.at[
                   comp_name, "number"] == test_gen["number"]

    def test_connect_to_lv(self):

        # ######### Generator #############
        # ToDo test other options when connected to voltage level 7

        # test substation ID that does not exist in the grid

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        test_gen = {
            "generator_id": 23456,
            "p_nom": 0.3,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"],
            "voltage_level": 6,
            "mvlv_subst_id": 10
        }

        comp_name = generators_import.connect_to_lv(
            self.edisgo, test_gen)

        # check if number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check if number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check that new generator is connected to HV/MV station
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "bus"] == "Bus_MVStation_1"

        # test missing substation ID

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        test_gen = {
            "generator_id": 23456,
            "p_nom": 0.3,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"],
            "voltage_level": 6,
            "mvlv_subst_id": None
        }

        comp_name = generators_import.connect_to_lv(
            self.edisgo, test_gen)

        # check if number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check if number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check that new generator is connected to random substation
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 0.4
        lv_grid_id = self.edisgo.topology.buses_df.at[new_bus, "lv_grid_id"]
        lv_grid = LVGrid(id=lv_grid_id, edisgo_obj=self.edisgo)
        assert new_bus == lv_grid.station.index[0]
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 0.3

        # test existing substation ID (voltage level 6)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = {
            "generator_id": 3456,
            "p_nom": 0.3,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"],
            "voltage_level": 6,
            "mvlv_subst_id": 6
        }

        comp_name = generators_import.connect_to_lv(
            self.edisgo, test_gen)

        # check that number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check that number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check that number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 0.4
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(
            new_bus)
        assert len(new_line_df) == 1
        assert "Bus_Generator_3456" in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]])
        lv_grid = LVGrid(id=6, edisgo_obj=self.edisgo)
        assert lv_grid.station.index[0] in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]])
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 0.3
        assert comp_name in lv_grid.generators_df.index

        # test existing substation ID (voltage level 7)
        # generator can be connected to residential load

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        test_gen = {
            "generator_id": 3456,
            "p_nom": 0.03,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"],
            "voltage_level": 7,
            "mvlv_subst_id": 1
        }

        comp_name = generators_import.connect_to_lv(
            self.edisgo, test_gen)

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check bus
        gen_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert gen_bus == "Bus_BranchTee_LVGrid_1_8"
        assert self.edisgo.topology.buses_df.at[
                   gen_bus, "lv_grid_id"] == 1
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 0.03

        # test existing substation ID (voltage level 7)
        # there is no valid load wherefore generator is connected to random bus

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        test_gen = {
            "generator_id": 3456,
            "p_nom": 0.04,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"],
            "voltage_level": 7,
            "mvlv_subst_id": 2
        }

        comp_name = generators_import.connect_to_lv(
            self.edisgo, test_gen)

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of generators increased
        assert len(generators_before) + 1 == len(
            self.edisgo.topology.generators_df)

        # check bus
        gen_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert gen_bus == "Bus_Load_residential_LVGrid_2_1"
        assert self.edisgo.topology.buses_df.at[
                   gen_bus, "lv_grid_id"] == 2
        # check new generator
        assert self.edisgo.topology.generators_df.at[
                   comp_name, "p_nom"] == 0.04

        # ######### Charging Point #############

        # test voltage level 7 - use case home (and there are residential
        # loads to add charging point to)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        cp_before = self.edisgo.topology.charging_points_df

        # add charging point
        test_cp = {
            "p_nom": 0.01,
            "geom": geom,
            "use_case": "home",
            "voltage_level": 7,
            "mvlv_subst_id": 3
        }

        comp_name = generators_import.connect_to_lv(
            self.edisgo, test_cp, comp_type="ChargingPoint")

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of charging points increased
        assert len(cp_before) + 1 == len(
            self.edisgo.topology.charging_points_df)

        # check bus
        bus = self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
        assert bus == "Bus_BranchTee_LVGrid_3_8"
        assert self.edisgo.topology.buses_df.at[
                   bus, "lv_grid_id"] == 3
        # check new charging point
        assert self.edisgo.topology.charging_points_df.at[
                   comp_name, "p_nom"] == 0.01

        # test voltage level 7 - use case work (connected to agricultural load)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        cp_before = self.edisgo.topology.charging_points_df

        # add charging point
        test_cp = {
            "p_nom": 0.02,
            "number": 2,
            "geom": geom,
            "use_case": "work",
            "voltage_level": 7,
            "mvlv_subst_id": 3
        }

        comp_name = generators_import.connect_to_lv(
            self.edisgo, test_cp, comp_type="ChargingPoint")

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of charging points increased
        assert len(cp_before) + 1 == len(
            self.edisgo.topology.charging_points_df)

        # check bus
        bus = self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
        assert bus == "Bus_BranchTee_LVGrid_3_2"
        assert self.edisgo.topology.buses_df.at[
                   bus, "lv_grid_id"] == 3
        # check new charging point
        assert self.edisgo.topology.charging_points_df.at[
                   comp_name, "number"] == 2

        # test voltage level 7 - use case public (connected somewhere in the
        # LV grid (to bus not in_building))

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        cp_before = self.edisgo.topology.charging_points_df

        # add charging point
        test_cp = {
            "p_nom": 0.02,
            "number": 2,
            "geom": geom,
            "use_case": "public",
            "voltage_level": 7,
            "mvlv_subst_id": 3
        }

        comp_name = generators_import.connect_to_lv(
            self.edisgo, test_cp, comp_type="ChargingPoint")

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of charging points increased
        assert len(cp_before) + 1 == len(
            self.edisgo.topology.charging_points_df)

        # check bus
        bus = self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
        assert bus == "Bus_Load_residential_LVGrid_3_4"
        assert self.edisgo.topology.buses_df.at[
                   bus, "lv_grid_id"] == 3
        # check new charging point
        assert self.edisgo.topology.charging_points_df.at[
                   comp_name, "number"] == 2

    def test_update_grids(self):

        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom_gen_new = Point((x, y))
        generators_mv = pd.DataFrame(
            data={
                "generator_id": [2, 3, 345],
                "geom": [None, None, str(geom_gen_new)],
                "p_nom": [3.0, 2.67, 2.5],
                "generator_type": ["wind", "solar", "solar"],
                "subtype": ["wind", "solar", "solar"],
                "weather_cell_id": [1122074, 1122075, 1122074],
                "voltage_level": [4, 4, 4]
            },
            index=[2, 3, 345]
        )
        generators_lv = pd.DataFrame(
            data={
                "generator_id": [13, 14, 456],
                "geom": [None, None, str(geom_gen_new)],
                "p_nom": [0.027, 0.005, 0.3],
                "generator_type": ["solar", "solar", "solar"],
                "subtype": ["solar", "solar", "roof"],
                "weather_cell_id": [1122075, 1122075, 1122074],
                "voltage_level": [6, 6, 6],
                "mvlv_subst_id": [None, None, 6]
            },
            index=[13, 14, 456]
        )

        generators_import.update_grids(
            self.edisgo, generators_mv, generators_lv)

        # check number of generators
        assert len(self.edisgo.topology.generators_df) == 6
        assert len(self.edisgo.topology.mv_grid.generators_df) == 3

        # check removed generators
        assert "Generator_1" not in self.edisgo.topology.generators_df.index
        assert "GeneratorFluctuating_12" not in self.edisgo.topology.generators_df.index

        # check updated generators
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_2", "p_nom"] == 3
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_2", "subtype"] == "wind_wind_onshore"
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_13", "p_nom"] == 0.027
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_13", "subtype"] == "solar_solar_roof_mounted"

        # check generators that stayed the same
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_3", "p_nom"] == 2.67
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_3", "subtype"] == "solar_solar_ground_mounted"
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_14", "p_nom"] == 0.005
        assert self.edisgo.topology.generators_df.at[
                   "GeneratorFluctuating_14", "subtype"] == "solar_solar_roof_mounted"

        # check new generators
        assert self.edisgo.topology.generators_df.at[
                   "Generator_solar_MVGrid_1_345", "p_nom"] == 2.5
        assert self.edisgo.topology.generators_df.at[
                   "Generator_solar_MVGrid_1_345", "type"] == "solar"
        assert self.edisgo.topology.generators_df.at[
                   "Generator_solar_LVGrid_6_456", "p_nom"] == 0.3
        assert self.edisgo.topology.generators_df.at[
                   "Generator_solar_LVGrid_6_456", "type"] == "solar"

    def test_update_grids_target_capacity(self):

        x = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at[
            "Bus_GeneratorFluctuating_6", "y"]
        geom_gen_new = Point((x, y))
        generators_mv = pd.DataFrame(
            data={
                "generator_id": [321, 3456, 345],
                "geom": [str(geom_gen_new), str(geom_gen_new), str(geom_gen_new)],
                "p_nom": [3.0, 2.67, 2.5],
                "generator_type": ["wind", "solar", "solar"],
                "subtype": ["wind", "solar", "solar"],
                "weather_cell_id": [1122074, 1122075, 1122074],
                "voltage_level": [4, 4, 4]
            },
            index=[321, 3456, 345]
        )
        generators_lv = pd.DataFrame(
            data={
                "generator_id": [13, 145, 456, 654],
                "geom": [None, None, str(geom_gen_new), None],
                "p_nom": [0.027, 0.005, 0.3, 0.3],
                "generator_type": ["solar", "solar", "run_of_river", "wind"],
                "subtype": ["solar", "solar", "hydro", "wind"],
                "weather_cell_id": [1122075, 1122075, 1122074, 1122074],
                "voltage_level": [6, 6, 6, 7],
                "mvlv_subst_id": [None, None, 6, 2]
            },
            index=[13, 145, 456, 654]
        )

        gens_before = self.edisgo.topology.generators_df
        p_wind_before = gens_before[gens_before["type"] == "wind"].p_nom.sum()
        p_pv_before = gens_before[gens_before["type"] == "solar"].p_nom.sum()
        p_gas_before = gens_before[gens_before["type"] == "gas"].p_nom.sum()
        p_target = {
            "wind": p_wind_before + 6,
            "solar": p_pv_before + 3,
            "gas": p_gas_before + 1.5
        }

        generators_import.update_grids(
            self.edisgo, generators_mv, generators_lv,
            p_target=p_target, remove_missing=False, update_existing=False)

        # check that all old generators still exist
        assert gens_before.index.isin(
            self.edisgo.topology.generators_df.index).all()

        # check that types for which no target capacity is specified are
        # not expanded
        assert ("run_of_river" not in
                self.edisgo.topology.generators_df["type"].unique())

        # check that target capacity for specified types is met
        # wind - target capacity higher than existing capacity plus new
        # capacity (all new generators are integrated and capacity is scaled
        # up)
        assert (self.edisgo.topology.generators_df[
                    self.edisgo.topology.generators_df[
                        'type'] == 'wind'].p_nom.sum() ==
                p_wind_before + 6)
        assert (len(self.edisgo.topology.generators_df[
                        self.edisgo.topology.generators_df[
                            'type'] == 'wind']) ==
                len(gens_before[gens_before["type"] == "wind"]) + 2)
        assert self.edisgo.topology.generators_df.at[
                   "Generator_wind_MVGrid_1_321", "p_nom"] >= 3.0

        # solar - target capacity lower than existing capacity plus new
        # capacity (not all new generators are integrated)
        assert np.isclose(self.edisgo.topology.generators_df[
                    self.edisgo.topology.generators_df[
                        'type'] == 'solar'].p_nom.sum(),
                p_pv_before + 3)
        assert (len(self.edisgo.topology.generators_df[
                        self.edisgo.topology.generators_df[
                            'type'] == 'solar']) <=
                len(gens_before[gens_before["type"] == "solar"]) + 4)

        # gas - no new generator, existing one is scaled
        assert (self.edisgo.topology.generators_df[
                    self.edisgo.topology.generators_df[
                        'type'] == 'gas'].p_nom.sum() ==
                p_gas_before + 1.5)
        assert (len(self.edisgo.topology.generators_df[
                        self.edisgo.topology.generators_df[
                            'type'] == 'gas']) ==
                len(gens_before[gens_before["type"] == "gas"]))
        assert self.edisgo.topology.generators_df.at[
                   "Generator_1", "p_nom"] == 0.775 + 1.5


class TestGeneratorsImportOEDB:
    """
    Tests in here are marked as slow, as the used test grid is quite large
    and should at some point be changed.

    """

    @pytest.mark.slow
    def test_oedb_without_timeseries(self):

        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path,
            worst_case_analysis="worst-case",
            generator_scenario="nep2035"
        )

        # check number of generators
        assert len(edisgo.topology.generators_df) == 18+1618
        # check total installed capacity
        assert np.isclose(edisgo.topology.generators_df.p_nom.sum(),
                          54.52844+19.8241)

    @pytest.mark.slow
    def test_oedb_with_worst_case_timeseries(self):

        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path,
            worst_case_analysis="worst-case"
        )

        gens_before = edisgo.topology.generators_df.copy()
        gens_ts_active_before = edisgo.timeseries.generators_active_power.copy()
        gens_ts_reactive_before = edisgo.timeseries.generators_reactive_power.copy()

        edisgo.import_generators("nep2035")

        # check number of generators
        assert len(edisgo.topology.generators_df) == 18+1618
        # check total installed capacity
        assert np.isclose(edisgo.topology.generators_df.p_nom.sum(),
                          54.52844+19.8241)

        gens_new = edisgo.topology.generators_df[
            ~edisgo.topology.generators_df.index.isin(gens_before.index)]
        # check solar generator (same weather cell ID and in same voltage
        # level, wherefore p_nom is set to be below 300 kW)
        old_solar_gen = gens_before[
                        (gens_before.type=="solar") &
                        (gens_before.p_nom<=0.3)].iloc[0, :]
        new_solar_gen = gens_new[
            (gens_new.type=="solar") &
            (gens_new.weather_cell_id==old_solar_gen.weather_cell_id) &
            (gens_new.p_nom<=0.3)].iloc[0, :]
        # check if time series of old gen is the same as before
        assert np.isclose(gens_ts_active_before.loc[:, old_solar_gen.name],
                          edisgo.timeseries.generators_active_power.loc[
                          :, old_solar_gen.name]).all()
        assert np.isclose(gens_ts_reactive_before.loc[:, old_solar_gen.name],
                          edisgo.timeseries.generators_reactive_power.loc[
                          :, old_solar_gen.name]).all()
        # check if normalized time series of new gen is the same as normalized
        # time series of old gen
        assert np.isclose(
            gens_ts_active_before.loc[:,
            old_solar_gen.name] / old_solar_gen.p_nom,
            edisgo.timeseries.generators_active_power.loc[
            :, new_solar_gen.name] / new_solar_gen.p_nom).all()
        assert np.isclose(
            edisgo.timeseries.generators_reactive_power.loc[
            :, new_solar_gen.name],
            edisgo.timeseries.generators_active_power.loc[
            :, new_solar_gen.name]  * -np.tan(np.arccos(0.95))
        ).all()
        #ToDo following test currently does fail sometimes as lv generators
        # connected to MV bus bar are handled as MV generators and therefore
        # assigned other cosphi
        # assert np.isclose(
        #     gens_ts_reactive_before.loc[:,
        #     old_solar_gen.name] / old_solar_gen.p_nom,
        #     edisgo.timeseries.generators_reactive_power.loc[
        #     :, new_solar_gen.name] / new_solar_gen.p_nom).all()

        # check wind generator
        old_wind_gen = gens_before[gens_before.type == "wind"].iloc[0, :]
        new_wind_gen = gens_new[
                            (gens_new.type == "wind") &
                            (gens_new.weather_cell_id == old_wind_gen.weather_cell_id)].iloc[
                        0, :]
        # check if time series of old gen is the same as before
        assert np.isclose(gens_ts_active_before.loc[:, old_wind_gen.name],
                          edisgo.timeseries.generators_active_power.loc[
                          :, old_wind_gen.name]).all()
        assert np.isclose(gens_ts_reactive_before.loc[:, old_wind_gen.name],
                          edisgo.timeseries.generators_reactive_power.loc[
                          :, old_wind_gen.name]).all()
        # check if normalized time series of new gen is the same as normalized
        # time series of old gen
        assert np.isclose(
            gens_ts_active_before.loc[:,
            old_wind_gen.name] / old_wind_gen.p_nom,
            edisgo.timeseries.generators_active_power.loc[
            :, new_wind_gen.name] / new_wind_gen.p_nom).all()
        assert np.isclose(
            gens_ts_reactive_before.loc[:,
            old_wind_gen.name] / old_wind_gen.p_nom,
            edisgo.timeseries.generators_reactive_power.loc[
            :, new_wind_gen.name] / new_wind_gen.p_nom).all()

        # check other generator
        new_gen = gens_new[gens_new.type == "gas"].iloc[0, :]

        # check if normalized time series of new gen is the same as normalized
        # time series of old gen
        assert np.isclose(
            edisgo.timeseries.generators_active_power.loc[
            :, new_gen.name] / new_gen.p_nom,
            [1, 0]).all()
        assert np.isclose(
            edisgo.timeseries.generators_reactive_power.loc[
            :, new_gen.name] / new_gen.p_nom,
            [-np.tan(np.arccos(0.95)), 0]).all()

    @pytest.mark.slow
    def test_oedb_with_timeseries_by_technology(self):

        timeindex = pd.date_range('1/1/2012', periods=3, freq='H')
        ts_gen_dispatchable = pd.DataFrame(
            {'other': [0.775] * 3,
             'gas': [0.9] * 3},
            index=timeindex)
        ts_gen_fluctuating = pd.DataFrame(
            {'wind': [0.1, 0.2, 0.15],
             'solar': [0.4, 0.5, 0.45]},
            index=timeindex)

        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path,
            timeseries_generation_dispatchable=ts_gen_dispatchable,
            timeseries_generation_fluctuating=ts_gen_fluctuating,
            timeseries_load="demandlib"
        )

        gens_before = edisgo.topology.generators_df.copy()
        gens_ts_active_before = edisgo.timeseries.generators_active_power.copy()
        gens_ts_reactive_before = edisgo.timeseries.generators_reactive_power.copy()

        edisgo.import_generators("nep2035")

        # check number of generators
        assert len(edisgo.topology.generators_df) == 18+1618
        # check total installed capacity
        assert np.isclose(edisgo.topology.generators_df.p_nom.sum(),
                          54.52844+19.8241)

        gens_new = edisgo.topology.generators_df[
            ~edisgo.topology.generators_df.index.isin(gens_before.index)]
        # check solar generator (same voltage level, wherefore p_nom is set
        # to be below 300 kW)
        old_solar_gen = gens_before[
                        (gens_before.type=="solar") &
                        (gens_before.p_nom<=0.3)].iloc[0, :]
        new_solar_gen = gens_new[
            (gens_new.type=="solar") &
            (gens_new.p_nom<=0.3)].iloc[0, :]
        # check if time series of old gen is the same as before
        assert np.isclose(gens_ts_active_before.loc[:, old_solar_gen.name],
                          edisgo.timeseries.generators_active_power.loc[
                          :, old_solar_gen.name]).all()
        assert np.isclose(gens_ts_reactive_before.loc[:, old_solar_gen.name],
                          edisgo.timeseries.generators_reactive_power.loc[
                          :, old_solar_gen.name]).all()
        # check if normalized time series of new gen is the same as normalized
        # time series of old gen
        assert np.isclose(
            gens_ts_active_before.loc[:,
            old_solar_gen.name] / old_solar_gen.p_nom,
            edisgo.timeseries.generators_active_power.loc[
            :, new_solar_gen.name] / new_solar_gen.p_nom).all()
        assert np.isclose(
            edisgo.timeseries.generators_reactive_power.loc[
            :, new_solar_gen.name],
            edisgo.timeseries.generators_active_power.loc[
            :, new_solar_gen.name]  * -np.tan(np.arccos(0.95))
        ).all()
        #ToDo following test currently does fail sometimes as lv generators
        # connected to MV bus bar are handled as MV generators and therefore
        # assigned other cosphi
        # assert np.isclose(
        #     gens_ts_reactive_before.loc[:,
        #     old_solar_gen.name] / old_solar_gen.p_nom,
        #     edisgo.timeseries.generators_reactive_power.loc[
        #     :, new_solar_gen.name] / new_solar_gen.p_nom).all()

        # check wind generator
        old_wind_gen = gens_before[gens_before.type == "wind"].iloc[0, :]
        new_wind_gen = gens_new[gens_new.type == "wind"].iloc[0, :]
        # check if time series of old gen is the same as before
        assert np.isclose(gens_ts_active_before.loc[:, old_wind_gen.name],
                          edisgo.timeseries.generators_active_power.loc[
                          :, old_wind_gen.name]).all()
        assert np.isclose(gens_ts_reactive_before.loc[:, old_wind_gen.name],
                          edisgo.timeseries.generators_reactive_power.loc[
                          :, old_wind_gen.name]).all()
        # check if normalized time series of new gen is the same as normalized
        # time series of old gen
        assert np.isclose(
            gens_ts_active_before.loc[:,
            old_wind_gen.name] / old_wind_gen.p_nom,
            edisgo.timeseries.generators_active_power.loc[
            :, new_wind_gen.name] / new_wind_gen.p_nom).all()
        assert np.isclose(
            gens_ts_reactive_before.loc[:,
            old_wind_gen.name] / old_wind_gen.p_nom,
            edisgo.timeseries.generators_reactive_power.loc[
            :, new_wind_gen.name] / new_wind_gen.p_nom).all()

        # check other generator
        new_gen = gens_new[
                      (gens_new.type == "gas") &
                      (gens_new.bus != edisgo.topology.mv_grid.station.index[0])].iloc[0, :]

        # check if normalized time series of new gen is the same as normalized
        # time series of old gen
        assert np.isclose(
            edisgo.timeseries.generators_active_power.loc[
            :, new_gen.name] / new_gen.p_nom,
            [0.9] * 3).all()
        assert np.isclose(
            edisgo.timeseries.generators_reactive_power.loc[
            :, new_gen.name] / (new_gen.p_nom * 0.9),
            [-np.tan(np.arccos(0.95))] * 3).all()

    @pytest.mark.slow
    def test_target_capacity(self):

        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_2_path,
            worst_case_analysis="worst-case"
        )

        gens_before = edisgo.topology.generators_df.copy()
        p_wind_before = edisgo.topology.generators_df[
            edisgo.topology.generators_df['type']=='wind'].p_nom.sum()
        p_biomass_before = edisgo.topology.generators_df[
            edisgo.topology.generators_df['type'] == 'biomass'].p_nom.sum()

        p_target = {'wind': p_wind_before * 1.6,
                    'biomass': p_biomass_before * 1.0}

        edisgo.import_generators(
            generator_scenario="nep2035",
            p_target=p_target,
            remove_missing=False,
            update_existing=False
        )

        # check that all old generators still exist
        assert gens_before.index.isin(
            edisgo.topology.generators_df.index).all()

        # check that installed capacity of types, for which no target capacity
        # was specified, remained the same
        assert (gens_before[gens_before["type"] == "solar"].p_nom.sum() ==
                edisgo.topology.generators_df[
                    edisgo.topology.generators_df["type"] == "solar"].p_nom.sum())
        assert (gens_before[gens_before["type"] == "run_of_river"].p_nom.sum() ==
                edisgo.topology.generators_df[
                    edisgo.topology.generators_df[
                        "type"] == "run_of_river"].p_nom.sum())

        # check that installed capacity of types, for which a target capacity
        # was specified, is met
        assert (edisgo.topology.generators_df[
            edisgo.topology.generators_df['type']=='wind'].p_nom.sum() ==
                p_wind_before * 1.6)
        assert (edisgo.topology.generators_df[
                    edisgo.topology.generators_df[
                        'type'] == 'biomass'].p_nom.sum() ==
                p_biomass_before * 1.0)