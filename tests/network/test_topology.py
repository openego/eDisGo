import pytest
import shutil
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from shapely.geometry import Point

from edisgo.network.topology import Topology
from edisgo.io import ding0_import
from edisgo import EDisGo
from edisgo.network.grids import LVGrid


class TestTopology:
    """
    Tests Topology class, except methods that require an edisgo object.

    """
    @classmethod
    def setup_class(self):
        self.topology = Topology()
        ding0_import.import_ding0_grid(pytest.ding0_test_network_path, self)

    def test_to_csv(self):
        """Test for method to_csv"""
        self.topology.to_csv("topology")
        edisgo = pd.DataFrame()
        edisgo.topology = Topology()
        ding0_import.import_ding0_grid("topology", edisgo)
        assert_frame_equal(self.topology.buses_df, edisgo.topology.buses_df)
        assert_frame_equal(
            self.topology.generators_df, edisgo.topology.generators_df
        )
        assert_frame_equal(self.topology.lines_df, edisgo.topology.lines_df)
        assert_frame_equal(self.topology.loads_df, edisgo.topology.loads_df)
        assert_frame_equal(self.topology.slack_df, edisgo.topology.slack_df)
        assert_frame_equal(self.topology.storage_units_df,
                           edisgo.topology.storage_units_df)
        assert_frame_equal(
            self.topology.switches_df, edisgo.topology.switches_df
        )
        assert_frame_equal(
            self.topology.transformers_df, edisgo.topology.transformers_df
        )
        assert_frame_equal(
            self.topology.transformers_hvmv_df,
            edisgo.topology.transformers_hvmv_df,
        )
        assert_frame_equal(
            pd.DataFrame([self.topology.grid_district]),
            pd.DataFrame([edisgo.topology.grid_district]),
        )
        # Todo: check files before rmtree?
        shutil.rmtree("topology", ignore_errors=True)

    def test_add_line(self):
        """Test add_line method"""

        len_df_before = len(self.topology.lines_df)

        bus0 = "Bus_BranchTee_MVGrid_1_8"
        bus1 = "Bus_GeneratorFluctuating_7"
        name = self.topology.add_line(
            bus0=bus0, bus1=bus1, length=1, x=1, r=1, s_nom=1, kind="cable"
        )

        assert len_df_before + 1 == len(self.topology.lines_df)
        assert (
            name == "Line_Bus_BranchTee_MVGrid_1_8_Bus_GeneratorFluctuating_7"
        )
        assert self.topology.lines_df.loc[name, "bus0"] == bus0
        assert self.topology.lines_df.loc[name, "s_nom"] == 1

        bus0 = "Bus_BranchTee_MVGrid_1_8"
        bus1 = "Bus_GeneratorFluctuating_9"
        msg = (
            "When line 'type_info' is provided when creating a new "
            "line, x, r and s_nom are calculated and provided "
            "parameters are overwritten."
        )
        with pytest.warns(UserWarning, match=msg):
            name = self.topology.add_line(
                bus0=bus0,
                bus1=bus1,
                length=1,
                kind="cable",
                type_info="NA2XS2Y 3x1x185 RM/25",
                x=2,
            )
        assert len_df_before + 2 == len(self.topology.lines_df)
        assert (
            name == "Line_Bus_BranchTee_MVGrid_1_8_Bus_GeneratorFluctuating_9"
        )
        assert self.topology.lines_df.loc[name, "s_nom"] == 6.1834213830208915
        assert self.topology.lines_df.loc[name, "x"] == 0.38

        msg = (
            "Specified bus Testbus is not valid as it is not defined in "
            "buses_df."
        )
        with pytest.raises(ValueError, match=msg):
            name = self.topology.add_line(
                bus0="Testbus",
                bus1=bus1,
                length=1,
                kind="cable",
                type_info="NA2XS2Y 3x1x185 RM/25",
                x=2,
            )

        msg = (
            "Specified bus Testbus1 is not valid as it is not defined in "
            "buses_df."
        )
        with pytest.raises(ValueError, match=msg):
            name = self.topology.add_line(
                bus0=bus0,
                bus1="Testbus1",
                length=1,
                kind="cable",
                type_info="NA2XS2Y 3x1x185 RM/25",
                x=2,
            )

    def test_add_generator(self):
        """Test add_generator method"""

        len_df_before = len(self.topology.generators_df)

        name = self.topology.add_generator(
            generator_id=2,
            bus="Bus_BranchTee_MVGrid_1_8",
            p_nom=1,
            generator_type="solar",
            subtype="roof",
            weather_cell_id=1000,
        )

        assert len_df_before + 1 == len(self.topology.generators_df)
        assert name == "Generator_solar_MVGrid_1_2"
        assert self.topology.generators_df.loc[name, "p_nom"] == 1

    def test_add_load(self):
        """Test add_load method"""

        msg = (
            "Specified bus Unknown_bus is not valid as it is not defined in "
            "buses_df."
        )
        with pytest.raises(ValueError, match=msg):
            self.topology.add_load(
                load_id=8,
                bus="Unknown_bus",
                peak_load=1,
                annual_consumption=1,
                sector="retail",
            )

        len_df_before = len(self.topology.loads_df)

        # check if name of load does not exist yet
        name = self.topology.add_load(
            load_id=10,
            bus="Bus_BranchTee_LVGrid_1_4",
            peak_load=1,
            annual_consumption=2,
            sector="residential",
        )
        assert len_df_before + 1 == len(self.topology.loads_df)
        assert name == "Load_residential_LVGrid_1_10"
        assert self.topology.loads_df.loc[name, "peak_load"] == 1
        assert self.topology.loads_df.loc[name, "annual_consumption"] == 2
        assert self.topology.loads_df.loc[name, "sector"] == "residential"

        # check auto creation of name when load name with load_id already
        # exists
        name = self.topology.add_load(
            load_id=1,
            bus="Bus_BranchTee_LVGrid_1_4",
            peak_load=2,
            annual_consumption=1,
            sector="agricultural",
        )
        assert len_df_before + 2 == len(self.topology.loads_df)
        assert name == "Load_agricultural_LVGrid_1_9"
        assert self.topology.loads_df.loc[name, "peak_load"] == 2
        assert self.topology.loads_df.loc[name, "annual_consumption"] == 1
        assert self.topology.loads_df.loc[name, "sector"] == "agricultural"

        # check auto creation of name if auto created name already exists
        name = self.topology.add_load(
            load_id=4,
            bus="Bus_BranchTee_LVGrid_1_4",
            peak_load=5,
            annual_consumption=4,
            sector="residential",
        )

        assert len_df_before + 3 == len(self.topology.loads_df)
        assert name != "Load_residential_LVGrid_1_10"
        assert len(name) == (len("Load_residential_LVGrid_1_") + 9)
        assert self.topology.loads_df.loc[name, "peak_load"] == 5
        assert self.topology.loads_df.loc[name, "annual_consumption"] == 4
        assert self.topology.loads_df.loc[name, "sector"] == "residential"

    def test_add_storage_unit(self):
        """Test add_storage_unit method"""

        msg = (
            "Specified bus Unknown_bus is not valid as it is not "
            "defined in buses_df."
        )
        with pytest.raises(ValueError, match=msg):
            self.topology.add_storage_unit(
                bus="Unknown_bus", p_nom=1, control="PQ"
            )

        len_df_before = len(self.topology.storage_units_df)

        # check if name of load does not exist yet
        name = self.topology.add_storage_unit(
            bus="Bus_BranchTee_LVGrid_1_5", p_nom=1, control="Test"
        )
        assert len_df_before + 1 == len(self.topology.storage_units_df)
        assert name == "StorageUnit_LVGrid_1_0"
        assert self.topology.storage_units_df.loc[name, "p_nom"] == 1
        assert self.topology.storage_units_df.loc[name, "control"] == "Test"

        # check auto creation of name when load name with load_id already
        # exists
        name = self.topology.add_storage_unit(
            bus="Bus_BranchTee_LVGrid_1_4", p_nom=2
        )
        assert len_df_before + 2 == len(self.topology.storage_units_df)
        assert name == "StorageUnit_LVGrid_1_1"
        assert self.topology.storage_units_df.loc[name, "p_nom"] == 2
        assert self.topology.storage_units_df.loc[name, "control"] == "PQ"

        # check auto creation of name if auto created name already exists
        name = self.topology.add_storage_unit(
            bus="Bus_BranchTee_LVGrid_1_4", p_nom=5
        )

        assert len_df_before + 3 == len(self.topology.storage_units_df)
        assert name == "StorageUnit_LVGrid_1_2"
        assert self.topology.storage_units_df.loc[name, "p_nom"] == 5
        assert self.topology.storage_units_df.loc[name, "control"] == "PQ"

    def test_add_bus(self):
        """Test add_bus method"""
        len_df_pre = len(self.topology.buses_df)

        # check adding MV bus
        self.topology.add_bus(bus_name="Test_bus", v_nom=20)
        assert len_df_pre + 1 == len(self.topology.buses_df)
        assert self.topology.buses_df.at["Test_bus", "v_nom"] == 20
        assert self.topology.buses_df.at["Test_bus", "mv_grid_id"] == 1
        self.topology.remove_bus("Test_bus")

        # check LV assertion
        msg = "You need to specify an lv_grid_id for low-voltage buses."
        with pytest.raises(ValueError, match=msg):
            self.topology.add_bus("Test_bus_LV", v_nom=0.4)

        # check adding LV bus
        self.topology.add_bus("Test_bus_LV", v_nom=0.4, lv_grid_id=1)
        assert len_df_pre + 1 == len(self.topology.buses_df)
        assert self.topology.buses_df.at["Test_bus_LV", "v_nom"]
        assert self.topology.buses_df.at["Test_bus_LV", "lv_grid_id"] == 1
        assert self.topology.buses_df.at["Test_bus_LV", "mv_grid_id"] == 1
        self.topology.remove_bus("Test_bus_LV")

    def test_remove_bus(self):
        """Test remove_bus method"""
        # create isolated bus to check
        self.topology.add_bus(bus_name="Test_bus_to_remove", v_nom=20)
        assert self.topology.buses_df.at["Test_bus_to_remove", "v_nom"] == 20
        # check removing bus
        self.topology.remove_bus("Test_bus_to_remove")
        with pytest.raises(KeyError):
            self.topology.buses_df.loc["Test_bus_to_remove"]
        # check assertion when bus is connected to element
        # check connected Generator
        msg = (
            "Bus Bus_Generator_1 is not isolated. Remove all connected "
            "elements first to remove bus."
        )
        with pytest.raises(AssertionError, match=msg):
            self.topology.remove_bus("Bus_Generator_1")
        # check connected Load
        msg = (
            "Bus Bus_Load_agricultural_LVGrid_1_1 is not isolated. "
            "Remove all connected elements first to remove bus."
        )
        with pytest.raises(AssertionError, match=msg):
            self.topology.remove_bus("Bus_Load_agricultural_LVGrid_1_1")
        # check connected line
        msg = (
            "Bus Bus_BranchTee_MVGrid_1_1 is not isolated. "
            "Remove all connected elements first to remove bus."
        )
        with pytest.raises(AssertionError, match=msg):
            self.topology.remove_bus("Bus_BranchTee_MVGrid_1_1")

    def test_remove_generator(self):
        """Test remove_load method"""
        name_generator = "GeneratorFluctuating_5"
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            "Bus_" + name_generator
        )
        # check if elements are part of topology:
        assert name_generator in self.topology.generators_df.index
        assert "Bus_" + name_generator in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()
        # check case where only load is connected to line,
        # line and bus are therefore removed as well
        self.topology.remove_generator(name_generator)
        assert name_generator not in self.topology.generators_df.index
        assert "Bus_" + name_generator not in self.topology.buses_df.index
        assert ~(
            connected_lines.index.isin(self.topology.lines_df.index)
        ).any()

        # check case where load is not the only connected element
        name_generator = "GeneratorFluctuating_7"
        self.topology.add_load(
            100, "Bus_" + name_generator, 2, 3, "agricultural"
        )
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            "Bus_" + name_generator
        )
        # check if elements are part of topology:
        assert name_generator in self.topology.generators_df.index
        assert "Bus_" + name_generator in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()
        # check case where other elements are connected to line as well,
        # line and bus are therefore not removed
        self.topology.remove_generator(name_generator)
        assert name_generator not in self.topology.generators_df.index
        assert "Bus_" + name_generator in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()

    def test_remove_load(self):
        """Test remove_load method"""
        name_load = "Load_residential_LVGrid_1_4"
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            "Bus_" + name_load
        )
        # check if elements are part of topology:
        assert name_load in self.topology.loads_df.index
        assert "Bus_" + name_load in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()
        # check case where only load is connected to line,
        # line and bus are therefore removed as well
        self.topology.remove_load(name_load)
        assert name_load not in self.topology.loads_df.index
        assert "Bus_" + name_load not in self.topology.buses_df.index
        assert ~(
            connected_lines.index.isin(self.topology.lines_df.index)
        ).any()

        # check case where load is not the only connected element
        name_load = "Load_residential_LVGrid_1_6"
        self.topology.add_load(100, "Bus_" + name_load, 2, 3, "agricultural")
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            "Bus_" + name_load
        )
        # check if elements are part of topology:
        assert name_load in self.topology.loads_df.index
        assert "Bus_" + name_load in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()
        # check case where other elements are connected to line as well,
        # line and bus are therefore not removed
        self.topology.remove_load(name_load)
        assert name_load not in self.topology.loads_df.index
        assert "Bus_" + name_load in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()

    def test_remove_line(self):
        """Test remove_line method"""

        # test only remove a line
        line_name = "Line_10012"
        bus0 = self.topology.lines_df.at[line_name, "bus0"]
        bus1 = self.topology.lines_df.at[line_name, "bus1"]
        self.topology.remove_line(line_name)
        assert line_name not in self.topology.lines_df.index
        assert bus0 in self.topology.buses_df.index
        assert bus1 in self.topology.buses_df.index

        # test remove line and bordering node
        self.topology.add_bus(bus_name="Test_bus", v_nom=20)
        self.topology.add_line(bus0=bus0, bus1="Test_bus", length=2)
        assert "Test_bus" in self.topology.buses_df.index
        assert (
            "Line_Bus_BranchTee_MVGrid_1_3_Test_bus"
            in self.topology.lines_df.index
        )
        self.topology.remove_line("Line_Bus_BranchTee_MVGrid_1_3_Test_bus")
        assert bus0 in self.topology.buses_df.index
        assert "Test_bus" not in self.topology.buses_df.index
        assert (
            "Line_Bus_BranchTee_MVGrid_1_3_Test_bus"
            not in self.topology.lines_df.index
        )

    def test_update_number_of_parallel_lines(self):

        line_1 = "Line_10026"
        line_2 = "Line_90000010"
        # manipulate number of parallel lines of line_2
        self.topology.lines_df.at[line_2, "num_parallel"] = 3
        # save values before update
        lines_attributes_pre = self.topology.lines_df.loc[
            [line_1, line_2], :
        ].copy()

        lines = pd.Series(index=[line_1, line_2], data=[2, 5])
        self.topology.update_number_of_parallel_lines(lines)

        assert self.topology.lines_df.at[line_1, "num_parallel"] == 2
        assert (
            self.topology.lines_df.at[line_1, "x"]
            == lines_attributes_pre.at[line_1, "x"] / 2
        )
        assert (
            self.topology.lines_df.at[line_1, "r"]
            == lines_attributes_pre.at[line_1, "r"] / 2
        )
        assert (
            self.topology.lines_df.at[line_1, "s_nom"]
            == lines_attributes_pre.at[line_1, "s_nom"] * 2
        )

        assert self.topology.lines_df.at[line_2, "num_parallel"] == 5
        assert (
            self.topology.lines_df.at[line_2, "x"]
            == lines_attributes_pre.at[line_2, "x"] * 3 / 5
        )
        assert (
            self.topology.lines_df.at[line_2, "r"]
            == lines_attributes_pre.at[line_2, "r"] * 3 / 5
        )
        assert (
            self.topology.lines_df.at[line_2, "s_nom"]
            == lines_attributes_pre.at[line_2, "s_nom"] * 5 / 3
        )

    def test_change_line_type(self):

        # test line type not in equipment data
        line_1 = "Line_10027"
        msg = ("Given new line type is not in equipment data. Please "
               "make sure to use line type with technical data provided "
               "in equipment_data 'mv_cables' or 'lv_cables'.")
        with pytest.raises(Exception, match=msg):
            self.topology.change_line_type([line_1], "NAYY")

        # test for single MV line and line type with different nominal voltage
        self.topology.change_line_type([line_1], "NA2XS2Y 3x1x185 RM/25")

        assert (self.topology.lines_df.at[line_1, "type_info"] ==
                "NA2XS2Y 3x1x185 RM/25")
        assert self.topology.lines_df.at[line_1, "num_parallel"] == 1
        assert self.topology.lines_df.at[line_1, "kind"] == "cable"
        assert np.isclose(
            self.topology.lines_df.at[line_1, "r"],
            0.32265687717586305 * 0.164
        )
        assert np.isclose(
            self.topology.lines_df.at[line_1, "x"],
            0.38 * 2 * np.pi * 50 / 1e3 * 0.32265687717586305
        )
        assert np.isclose(
            self.topology.lines_df.at[line_1, "s_nom"],
            0.357 * 20 * np.sqrt(3)
        )

        # test for multiple LV lines
        line_1 = "Line_50000006"
        line_2 = "Line_90000010"
        self.topology.change_line_type([line_1, line_2], "NAYY 4x1x300")

        assert (self.topology.lines_df.loc[[line_1, line_2], "type_info"] ==
                "NAYY 4x1x300").all()
        assert np.isclose(
            self.topology.lines_df.at[line_1, "r"],
            0.1 * self.topology.lines_df.at[line_1, "length"]
        )
        assert np.isclose(
            self.topology.lines_df.at[line_2, "r"],
            0.1 * self.topology.lines_df.at[line_2, "length"]
        )
        assert (
            self.topology.lines_df.loc[[line_1, line_2], "s_nom"] ==
            np.sqrt(3) * 0.4 * 0.419
        ).all()


class TestTopologyConnectMethods:
    """
    Tests connect methods in Topology that require edisgo object.

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

        comp_name = self.edisgo.topology.connect_to_mv(
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

        comp_name = self.edisgo.topology.connect_to_mv(
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

        comp_name = self.edisgo.topology.connect_to_mv(
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

        comp_name = self.edisgo.topology.connect_to_mv(
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

        comp_name = self.edisgo.topology.connect_to_lv(
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

        comp_name = self.edisgo.topology.connect_to_lv(
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

        comp_name = self.edisgo.topology.connect_to_lv(
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

        comp_name = self.edisgo.topology.connect_to_lv(
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

        comp_name = self.edisgo.topology.connect_to_lv(
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

        comp_name = self.edisgo.topology.connect_to_lv(
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

        comp_name = self.edisgo.topology.connect_to_lv(
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

        comp_name = self.edisgo.topology.connect_to_lv(
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
