import copy
import logging
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from geopandas import GeoDataFrame
from pandas.testing import assert_frame_equal
from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.io import ding0_import
from edisgo.network.components import Switch
from edisgo.network.grids import LVGrid, MVGrid
from edisgo.network.topology import Topology
from edisgo.tools.geopandas_helper import GeoPandasGridContainer

logger = logging.getLogger(__name__)


class TestTopology:
    """
    Tests Topology class, except methods that require an edisgo object.

    """

    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        self.topology = Topology()
        ding0_import.import_ding0_grid(pytest.ding0_test_network_path, self)

    def test_grids(self):
        grids = list(self.topology.grids)
        assert len(grids) == 11
        assert isinstance(grids[0], MVGrid)
        assert isinstance(grids[1], LVGrid)

    def test_lv_grids(self):
        lv_grids = list(self.topology.lv_grids)
        assert len(lv_grids) == 10
        assert isinstance(lv_grids[0], LVGrid)

    def test__lv_grid_ids(self):
        lv_grid_ids = self.topology._lv_grid_ids
        assert len(lv_grid_ids) == 10
        assert isinstance(lv_grid_ids[0], int)
        assert 2 in lv_grid_ids

    def test__grids_repr(self):
        grids_repr = self.topology._grids_repr
        assert len(grids_repr) == 11
        assert isinstance(grids_repr[0], str)
        assert "LVGrid_1" in grids_repr

    def test_get_lv_grid(self, caplog):
        # test integer input
        name = 1
        lv_grid = self.topology.get_lv_grid(name)
        assert isinstance(lv_grid, LVGrid)
        assert lv_grid.id == name

        # test string input
        name = "LVGrid_2"
        lv_grid = self.topology.get_lv_grid(name)
        assert isinstance(lv_grid, LVGrid)
        assert str(lv_grid) == name

        # test invalid input
        name = 1.0
        lv_grid = self.topology.get_lv_grid(name)
        assert lv_grid is None
        assert "`name` must be integer or string." in caplog.text

    def test_rings(self):
        """Test rings getter."""

        # getting rings
        ring = self.topology.rings
        # sorting ring elements, as they have a different order on each pass
        ring[0].sort()
        ring[1].sort()
        ring.sort()

        rings_ding0_test_network_1 = [
            [
                "BusBar_MVGrid_1_LVGrid_1_MV",
                "BusBar_MVGrid_1_LVGrid_5_MV",
                "BusBar_MVGrid_1_LVGrid_6_MV",
                "BusBar_MVGrid_1_LVGrid_9_MV",
                "Bus_BranchTee_MVGrid_1_1",
                "Bus_BranchTee_MVGrid_1_2",
                "Bus_BranchTee_MVGrid_1_3",
                "Bus_BranchTee_MVGrid_1_4",
                "Bus_MVStation_1",
            ],
            [
                "BusBar_MVGrid_1_LVGrid_4_MV",
                "BusBar_MVGrid_1_LVGrid_8_MV",
                "Bus_BranchTee_MVGrid_1_10",
                "Bus_BranchTee_MVGrid_1_11",
                "Bus_BranchTee_MVGrid_1_5",
                "Bus_BranchTee_MVGrid_1_6",
                "Bus_BranchTee_MVGrid_1_7",
                "Bus_BranchTee_MVGrid_1_8",
                "Bus_BranchTee_MVGrid_1_9",
                "Bus_MVStation_1",
            ],
        ]
        # test if rings have expected elements
        assert ring == rings_ding0_test_network_1

    def test_get_connected_lines_from_bus(self):
        """Test get_connected_lines_from_bus method."""

        # get connected lines
        connected_lines = self.topology.get_connected_lines_from_bus(
            "Bus_BranchTee_MVGrid_1_8"
        )
        # test if the expected lines are connected
        assert "Line_10019" in connected_lines.index
        assert "Line_10020" in connected_lines.index
        assert "Line_10021" in connected_lines.index
        # test if the selected bus is connected to the found lines
        assert (
            "Bus_BranchTee_MVGrid_1_8"
            in connected_lines.loc["Line_10019"].values.tolist()
        )
        assert (
            "Bus_BranchTee_MVGrid_1_8"
            in connected_lines.loc["Line_10020"].values.tolist()
        )
        assert (
            "Bus_BranchTee_MVGrid_1_8"
            in connected_lines.loc["Line_10021"].values.tolist()
        )

    def test_get_connected_components_from_bus(self):
        """Test get_connected_components_from_bus method."""

        # test if loads and lines are found at the bus
        components = self.topology.get_connected_components_from_bus(
            "Bus_BranchTee_LVGrid_3_6"
        )
        assert "Load_residential_LVGrid_3_3" in components["loads"].index
        assert "Line_30000007" in components["lines"].index

        assert components["generators"].empty
        assert components["storage_units"].empty
        assert components["transformers"].empty
        assert components["transformers_hvmv"].empty
        assert components["switches"].empty

        # test if generators and lines are found at the bus
        components = self.topology.get_connected_components_from_bus(
            "Bus_BranchTee_LVGrid_1_10"
        )
        assert "GeneratorFluctuating_9" in components["generators"].index
        assert "GeneratorFluctuating_10" in components["generators"].index
        assert "Line_10000007" in components["lines"].index
        assert "Load_residential_LVGrid_1_5" in components["loads"].index

        assert components["storage_units"].empty
        assert components["transformers"].empty
        assert components["transformers_hvmv"].empty
        assert components["switches"].empty

        # test if lines, storage unit and transformers are found at bus
        components = self.topology.get_connected_components_from_bus("Bus_MVStation_1")
        assert "Storage_1" in components["storage_units"].index
        assert "Line_10003" in components["lines"].index
        assert "Line_10004" in components["lines"].index
        assert "Line_10005" in components["lines"].index
        assert "Line_10006" in components["lines"].index
        assert "MVStation_1_transformer_1" in components["transformers_hvmv"].index
        assert len(components["transformers"]) == 4
        assert "Transformer_lv_load_area_1_1" in components["transformers"].index

        assert components["generators"].empty
        assert components["loads"].empty
        assert components["switches"].empty

        # test if lines, transformers and switches are found at the bus for a
        # closed switch
        switch = Switch(id="circuit_breaker_1", topology=self.topology)
        switch.close()
        components = self.topology.get_connected_components_from_bus(
            "BusBar_MVGrid_1_LVGrid_4_MV"
        )
        assert "Line_10030" in components["lines"].index
        # "Line_10031" only is connected if switch is closed
        assert "Line_10031" in components["lines"].index
        assert "Line_10032" in components["lines"].index
        assert "LVStation_4_transformer_1" in components["transformers"].index
        assert "LVStation_4_transformer_2" in components["transformers"].index
        assert "circuit_breaker_1" in components["switches"].index

        assert components["generators"].empty
        assert components["loads"].empty
        assert components["storage_units"].empty
        assert components["transformers_hvmv"].empty

        # test if lines, transformers and switches are found at the bus for an
        # open switch
        switch.open()
        components = self.topology.get_connected_components_from_bus(
            "BusBar_MVGrid_1_LVGrid_4_MV"
        )
        assert "Line_10030" in components["lines"].index
        assert "Line_10032" in components["lines"].index
        assert "LVStation_4_transformer_1" in components["transformers"].index
        assert "LVStation_4_transformer_2" in components["transformers"].index
        assert "circuit_breaker_1" in components["switches"].index

        assert components["generators"].empty
        assert components["loads"].empty
        assert components["storage_units"].empty
        assert components["transformers_hvmv"].empty

    def test_get_neighbours(self):
        """Test get_neighbours method."""

        # test for bus without a switch
        neighbours = self.topology.get_neighbours("Bus_BranchTee_MVGrid_1_8")
        assert "Bus_BranchTee_MVGrid_1_7" in neighbours
        assert "Bus_GeneratorFluctuating_5" in neighbours
        assert "BusBar_MVGrid_1_LVGrid_8_MV" in neighbours

        # test for bus with a switch
        # closed switch
        switch = Switch(id="circuit_breaker_1", topology=self.topology)
        switch.close()
        neighbours = self.topology.get_neighbours("BusBar_MVGrid_1_LVGrid_4_MV")
        assert "Bus_GeneratorFluctuating_8" in neighbours
        assert "Bus_BranchTee_MVGrid_1_11" in neighbours
        # "Bus_BranchTee_MVGrid_1_9" is connected through a switch
        assert "Bus_BranchTee_MVGrid_1_9" in neighbours

        # open switch
        switch.open()
        neighbours = self.topology.get_neighbours("BusBar_MVGrid_1_LVGrid_4_MV")
        assert "Bus_GeneratorFluctuating_8" in neighbours
        assert "Bus_BranchTee_MVGrid_1_11" in neighbours

    def test_add_load(self):
        """Test add_load method"""

        # test adding conventional load

        len_df_before = len(self.topology.loads_df)

        # test with kwargs
        name = self.topology.add_load(
            load_id=10,
            bus="Bus_BranchTee_LVGrid_1_4",
            p_set=1,
            annual_consumption=2,
            sector="residential",
            test_info="test",
        )
        assert len_df_before + 1 == len(self.topology.loads_df)
        assert name == "Conventional_Load_LVGrid_1_residential_10"
        assert self.topology.loads_df.at[name, "p_set"] == 1
        assert self.topology.loads_df.at[name, "test_info"] == "test"

        # test without kwargs
        name = self.topology.add_load(
            bus="Bus_BranchTee_LVGrid_1_4", p_set=2, annual_consumption=1
        )
        assert len_df_before + 2 == len(self.topology.loads_df)
        assert name == "Conventional_Load_LVGrid_1_9"
        assert self.topology.loads_df.loc[name, "p_set"] == 2
        assert self.topology.loads_df.loc[name, "sector"] is np.nan

        # test without kwargs (name created using number of loads in grid)
        name = self.topology.add_load(
            bus="Bus_BranchTee_LVGrid_1_4", p_set=3, annual_consumption=1
        )
        assert len_df_before + 3 == len(self.topology.loads_df)
        assert name == "Conventional_Load_LVGrid_1_10"
        assert self.topology.loads_df.loc[name, "p_set"] == 3

        # test error raising if bus is not valid
        msg = (
            "Specified bus Unknown_bus is not valid as it is not defined in "
            "buses_df."
        )
        with pytest.raises(ValueError, match=msg):
            self.topology.add_load(
                load_id=8,
                bus="Unknown_bus",
                p_set=1,
                annual_consumption=1,
                sector="retail",
            )

        # test adding charging point

        len_df_before = len(self.topology.charging_points_df)

        # test with kwargs
        name = self.topology.add_load(
            bus="Bus_BranchTee_MVGrid_1_8",
            p_set=1,
            type="charging_point",
            sector="home",
            number=2,
            test_info="test",
        )
        assert len_df_before + 1 == len(self.topology.charging_points_df)
        assert name == "Charging_Point_MVGrid_1_home_1"
        assert self.topology.charging_points_df.at[name, "sector"] == "home"
        assert self.topology.charging_points_df.at[name, "test_info"] == "test"

        # test without kwargs
        name = self.topology.add_load(
            bus="Bus_BranchTee_LVGrid_1_2",
            type="charging_point",
            p_set=0.5,
            sector="work",
        )
        assert len_df_before + 2 == len(self.topology.charging_points_df)
        assert name == "Charging_Point_LVGrid_1_work_1"
        assert self.topology.charging_points_df.at[name, "p_set"] == 0.5

        # test error raising if bus is not valid
        msg = (
            "Specified bus Unknown_bus is not valid as it is not defined in "
            "buses_df."
        )
        with pytest.raises(ValueError, match=msg):
            self.topology.add_load(bus="Unknown_bus", p_set=0.5, sector="work")

    def test_add_generator(self):
        """Test add_generator method"""

        len_df_before = len(self.topology.generators_df)

        # test with kwargs
        name = self.topology.add_generator(
            bus="Bus_BranchTee_MVGrid_1_8",
            p_nom=1,
            generator_type="solar",
            subtype="roof",
            weather_cell_id=1000,
            generator_id=2,
            test_info="test",
        )

        assert len_df_before + 1 == len(self.topology.generators_df)
        assert name == "Generator_MVGrid_1_solar_2"
        assert self.topology.generators_df.at[name, "weather_cell_id"] == 1000
        assert self.topology.generators_df.at[name, "test_info"] == "test"

        # test without kwargs
        name = self.topology.add_generator(
            bus="Bus_BranchTee_LVGrid_1_4", p_nom=0.5, generator_type="solar"
        )

        assert len_df_before + 2 == len(self.topology.generators_df)
        assert name == "Generator_LVGrid_1_solar"
        assert self.topology.generators_df.at[name, "p_nom"] == 0.5

        # test error raising if bus is not valid
        msg = (
            "Specified bus Unknown_bus is not valid as it is not defined in "
            "buses_df."
        )
        with pytest.raises(ValueError, match=msg):
            self.topology.add_generator(
                bus="Unknown_bus", p_nom=0.5, generator_type="solar"
            )

    def test_add_storage_unit(self):
        """Test add_storage_unit method"""

        len_df_before = len(self.topology.storage_units_df)

        # test with kwargs
        name = self.topology.add_storage_unit(
            bus="Bus_BranchTee_LVGrid_1_3",
            p_nom=1,
            control="Test",
            test_info="test",
        )
        assert len_df_before + 1 == len(self.topology.storage_units_df)
        assert name == "StorageUnit_LVGrid_1_1"
        assert self.topology.storage_units_df.at[name, "p_nom"] == 1
        assert self.topology.storage_units_df.loc[name, "test_info"] == "test"

        # test without kwargs
        name = self.topology.add_storage_unit(bus="Bus_BranchTee_LVGrid_1_6", p_nom=2)
        assert len_df_before + 2 == len(self.topology.storage_units_df)
        assert name == "StorageUnit_LVGrid_1_2"
        assert self.topology.storage_units_df.at[name, "p_nom"] == 2
        assert self.topology.storage_units_df.at[name, "control"] == "PQ"

        # test error raising if bus is not valid
        msg = (
            "Specified bus Unknown_bus is not valid as it is not "
            "defined in buses_df."
        )
        with pytest.raises(ValueError, match=msg):
            self.topology.add_storage_unit(bus="Unknown_bus", p_nom=1, control="PQ")

    def test_add_line(self, caplog):
        """Test add_line method"""

        len_df_before = len(self.topology.lines_df)

        # test with all values provided
        bus0 = "Bus_BranchTee_MVGrid_1_8"
        bus1 = "Bus_GeneratorFluctuating_7"
        name = self.topology.add_line(
            bus0=bus0, bus1=bus1, length=1, x=1, r=1, s_nom=1, kind="cable"
        )

        assert len_df_before + 1 == len(self.topology.lines_df)
        assert name == "Line_Bus_BranchTee_MVGrid_1_8_Bus_GeneratorFluctuating_7"
        assert self.topology.lines_df.at[name, "bus0"] == bus0
        assert self.topology.lines_df.at[name, "s_nom"] == 1

        # test with line type provided
        bus1 = "Bus_BranchTee_LVGrid_1_10"
        msg = (
            "When line 'type_info' is provided when creating a new "
            "line, x, r, b and s_nom are calculated and provided "
            "parameters are overwritten."
        )
        with caplog.at_level(logging.WARNING):
            name = self.topology.add_line(
                bus0=bus0,
                bus1=bus1,
                length=1,
                kind="cable",
                type_info="NA2XS2Y 3x1x185 RM/25",
                x=2,
            )
        assert msg in caplog.text
        assert len_df_before + 2 == len(self.topology.lines_df)
        assert name == "Line_Bus_BranchTee_MVGrid_1_8_Bus_BranchTee_LVGrid_1_10"
        assert np.isclose(self.topology.lines_df.at[name, "s_nom"], 6.18342)
        assert np.isclose(self.topology.lines_df.at[name, "r"], 0.164)

        # test no creation of new line when line between buses already exists
        line = name
        name = self.topology.add_line(
            bus0=bus0,
            bus1=bus1,
            length=1,
            kind="cable",
            type_info="NA2XS2Y 3x1x185 RM/25",
            x=3,
        )
        assert len_df_before + 2 == len(self.topology.lines_df)
        assert name == line
        assert (
            self.topology.lines_df.at[name, "x"] == self.topology.lines_df.at[line, "x"]
        )

        # test error raising when given buses are not valid
        msg = "Specified bus Testbus is not valid as it is not defined in buses_df."
        with pytest.raises(ValueError, match=msg):
            self.topology.add_line(
                bus0="Testbus",
                bus1=bus1,
                length=1,
                kind="cable",
                type_info="NA2XS2Y 3x1x185 RM/25",
                x=2,
            )
        msg = "Specified bus Testbus1 is not valid as it is not defined in buses_df."
        with pytest.raises(ValueError, match=msg):
            self.topology.add_line(
                bus0=bus0,
                bus1="Testbus1",
                length=1,
                kind="cable",
                type_info="NA2XS2Y 3x1x185 RM/25",
                x=2,
            )

        msg = "Newly added line has no line resistance and/or reactance."
        with pytest.raises(AttributeError, match=msg):
            self.topology.add_line(bus0=bus0, bus1="Bus_BranchTee_LVGrid_2_1", length=1)

    def test_add_bus(self):
        """Test add_bus method"""
        len_df_before = len(self.topology.buses_df)

        # check adding MV bus
        name = self.topology.add_bus(bus_name="Test_bus", v_nom=20)
        assert len_df_before + 1 == len(self.topology.buses_df)
        assert name == "Test_bus"
        assert self.topology.buses_df.at["Test_bus", "v_nom"] == 20
        assert self.topology.buses_df.at["Test_bus", "mv_grid_id"] == 1

        # check LV assertion
        msg = "You need to specify an lv_grid_id for low-voltage buses."
        with pytest.raises(ValueError, match=msg):
            self.topology.add_bus("Test_bus_LV", v_nom=0.4)

        # check adding LV bus (where bus name already exists)
        name = self.topology.add_bus(bus_name="Test_bus", v_nom=0.4, lv_grid_id=1)
        assert len_df_before + 2 == len(self.topology.buses_df)
        assert name != "Test_bus"
        assert self.topology.buses_df.at[name, "v_nom"]
        assert self.topology.buses_df.at[name, "lv_grid_id"] == 1
        assert self.topology.buses_df.at[name, "mv_grid_id"] == 1

    def test_check_bus_for_removal(self, caplog):
        # test warning if line does not exist
        msg = "Bus of name TestBus not in Topology. Cannot be removed."
        with caplog.at_level(logging.WARNING):
            self.topology._check_bus_for_removal("TestBus")
        assert msg in caplog.text
        return_value = self.topology._check_bus_for_removal("TestBus")
        assert not return_value

        # test bus cannot be removed because it is no end bus
        return_value = self.topology._check_bus_for_removal("Bus_BranchTee_LVGrid_4_2")
        assert not return_value
        return_value = self.topology._check_bus_for_removal(
            "BusBar_MVGrid_1_LVGrid_1_MV"
        )
        assert not return_value

        # test bus that is end bus, but has connected components
        return_value = self.topology._check_bus_for_removal(
            "Bus_GeneratorFluctuating_16"
        )
        assert not return_value

        # test bus that is end bus, and has no connected components
        # delete connected generator
        self.topology._generators_df.drop("GeneratorFluctuating_16", inplace=True)
        return_value = self.topology._check_bus_for_removal(
            "Bus_GeneratorFluctuating_16"
        )
        assert return_value

    def test_check_line_for_removal(self, caplog):
        # test warning if line does not exist
        msg = "Line of name TestLine not in Topology. Cannot be removed."
        with caplog.at_level(logging.WARNING):
            self.topology._check_line_for_removal("TestLine")
        assert msg in caplog.text
        return_value = self.topology._check_line_for_removal("TestLine")
        assert not return_value

        # test line cannot be removed because both buses cannot be removed
        return_value = self.topology._check_line_for_removal("Line_10024")
        assert not return_value
        return_value = self.topology._check_line_for_removal("Line_20000002")
        assert not return_value

        # test line where one bus can be removed
        # delete connected load
        self.topology._loads_df.drop("Load_residential_LVGrid_2_1", inplace=True)
        return_value = self.topology._check_line_for_removal("Line_20000002")
        assert return_value

        # test line in ring
        # add line to create ring
        self.topology.lines_df = pd.concat(
            [
                self.topology.lines_df,
                pd.DataFrame(
                    data={
                        "bus0": "Bus_BranchTee_LVGrid_2_2",
                        "bus1": "Bus_BranchTee_LVGrid_2_3",
                    },
                    index=["TestLine"],
                ),
            ]
        )
        return_value = self.topology._check_line_for_removal("TestLine")
        assert return_value

    def test_remove_load(self):
        """Test remove_load method"""

        # test removing conventional load

        # check case where only load is connected to line,
        # line and bus are therefore removed as well
        name = "Load_residential_LVGrid_1_4"
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus("Bus_" + name)
        self.topology.remove_load(name)
        assert name not in self.topology.loads_df.index
        assert "Bus_" + name not in self.topology.buses_df.index
        assert ~(connected_lines.index.isin(self.topology.lines_df.index)).any()

        # check case where load is not the only connected element
        name = "Load_residential_LVGrid_1_6"
        self.topology.add_load("Bus_BranchTee_LVGrid_1_12", 2, annual_consumption=3)
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            "Bus_BranchTee_LVGrid_1_12"
        )
        self.topology.remove_load(name)
        assert name not in self.topology.loads_df.index
        assert "Bus_BranchTee_LVGrid_1_12" in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()

        # test removing charging point

        # check case where only charging point is connected to line,
        # line and bus are therefore removed as well
        name = "ChargingPoint_LVGrid_1_work_1"
        bus = "Bus_Load_agricultural_LVGrid_1_1"
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(bus)
        # remove load
        self.topology.remove_load("Load_agricultural_LVGrid_1_1")
        self.topology.remove_load(name)
        assert name not in self.topology.charging_points_df.index
        assert bus not in self.topology.buses_df.index
        assert ~(connected_lines.index.isin(self.topology.lines_df.index)).any()

        # check case where charging point is not the only connected element
        name = "ChargingPoint_MVGrid_1_home_1"
        bus = "Bus_BranchTee_MVGrid_1_8"
        # get connected lines
        connected_lines = self.topology.get_connected_lines_from_bus(bus)
        self.topology.remove_load(name)
        assert name not in self.topology.charging_points_df.index
        assert bus in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()

    def test_remove_generator(self):
        """Test remove_generator method"""

        # check case where only generator is connected to line,
        # wherefore line and bus are removed as well
        name = "GeneratorFluctuating_17"
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus("Bus_" + name)
        self.topology.remove_generator(name)
        assert name not in self.topology.generators_df.index
        assert "Bus_" + name not in self.topology.buses_df.index
        assert ~(connected_lines.index.isin(self.topology.lines_df.index)).any()

        # check case where generator is not the only connected element
        name = "GeneratorFluctuating_18"
        self.topology.add_load("Bus_BranchTee_LVGrid_4_2", 2, annual_consumption=3)
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            "Bus_BranchTee_LVGrid_4_2"
        )
        self.topology.remove_generator(name)
        assert name not in self.topology.generators_df.index
        assert "Bus_BranchTee_LVGrid_4_2" in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()

    def test_remove_storage_unit(self):
        """Test remove_storage_unit method"""

        # check case where only storage unit is connected to line,
        # line and bus are therefore removed as well
        name = "StorageUnit_LVGrid_1_2"
        bus = "Bus_BranchTee_LVGrid_1_6"
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(bus)
        # remove load
        self.topology.remove_load("Load_agricultural_LVGrid_1_3")
        self.topology.remove_storage_unit(name)
        assert name not in self.topology.storage_units_df.index
        assert bus not in self.topology.buses_df.index
        assert ~(connected_lines.index.isin(self.topology.lines_df.index)).any()

        # check case where storage is not the only connected element
        name = "StorageUnit_LVGrid_1_1"
        bus = "Bus_BranchTee_LVGrid_1_3"
        # get connected lines
        connected_lines = self.topology.get_connected_lines_from_bus(bus)
        self.topology.remove_storage_unit(name)
        assert name not in self.topology.storage_units_df.index
        assert bus in self.topology.buses_df.index
        assert (connected_lines.index.isin(self.topology.lines_df.index)).all()

    def test_remove_line(self, caplog):
        """Test remove_line method"""

        # test try removing line that cannot be removed
        msg = "Removal of line Line_30000010 would create isolated node."
        with caplog.at_level(logging.WARNING):
            self.topology.remove_line("Line_30000010")
        assert msg in caplog.text

        # test remove line in cycle (no bus is removed)
        # add line to create ring
        line_name = "TestLine_LVGrid_3"
        self.topology.lines_df = pd.concat(
            [
                self.topology.lines_df,
                pd.DataFrame(
                    data={
                        "bus0": "Bus_BranchTee_LVGrid_3_2",
                        "bus1": "Bus_BranchTee_LVGrid_3_5",
                    },
                    index=[line_name],
                ),
            ]
        )

        len_df_before = len(self.topology.lines_df)

        self.topology.remove_line(line_name)
        assert len_df_before - 1 == len(self.topology.lines_df)
        assert line_name not in self.topology.lines_df.index
        assert "Bus_BranchTee_LVGrid_3_2" in self.topology.buses_df.index
        assert "Bus_BranchTee_LVGrid_3_5" in self.topology.buses_df.index

        # test remove line and bordering node
        # drop connected load
        self.topology._loads_df.drop("Load_residential_LVGrid_3_3", inplace=True)
        line_name = "Line_30000007"
        self.topology.remove_line(line_name)
        assert len_df_before - 2 == len(self.topology.lines_df)
        assert line_name not in self.topology.lines_df.index
        assert "Bus_BranchTee_LVGrid_3_6" not in self.topology.buses_df.index
        assert "Bus_BranchTee_LVGrid_3_5" in self.topology.buses_df.index

    def test_remove_bus(self, caplog):
        """Test remove_bus method"""

        # test bus cannot be removed
        msg = (
            "Bus Bus_BranchTee_LVGrid_4_2 is not isolated and "
            "therefore not removed. Remove all connected elements "
        )
        with caplog.at_level(logging.WARNING):
            self.topology.remove_bus("Bus_BranchTee_LVGrid_4_2")
        assert msg in caplog.text

        # test bus can be removed
        # create isolated bus
        bus_name = "TestBusIsolated"
        self.topology.buses_df = pd.concat(
            [
                self.topology.buses_df,
                pd.DataFrame(
                    data={"v_nom": 20},
                    index=[bus_name],
                ),
            ]
        )
        len_df_before = len(self.topology.buses_df)
        self.topology.remove_bus(bus_name)
        assert len_df_before - 1 == len(self.topology.buses_df)
        assert bus_name not in self.topology.buses_df.index

    def test_update_number_of_parallel_lines(self):
        line_1 = "Line_10026"
        line_2 = "Line_90000010"
        # manipulate number of parallel lines of line_2
        self.topology.lines_df.at[line_2, "num_parallel"] = 3
        # save values before update
        lines_attributes_pre = self.topology.lines_df.loc[[line_1, line_2], :].copy()

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
        msg = (
            "Given new line type is not in equipment data. Please "
            "make sure to use line type with technical data provided "
            "in equipment_data 'mv_cables' or 'lv_cables'."
        )
        with pytest.raises(Exception, match=msg):
            self.topology.change_line_type([line_1], "NAYY")

        # test for single MV line and line type with different nominal voltage
        self.topology.change_line_type([line_1], "NA2XS2Y 3x1x185 RM/25")

        assert self.topology.lines_df.at[line_1, "type_info"] == "NA2XS2Y 3x1x185 RM/25"
        assert self.topology.lines_df.at[line_1, "num_parallel"] == 1
        assert self.topology.lines_df.at[line_1, "kind"] == "cable"
        assert np.isclose(
            self.topology.lines_df.at[line_1, "r"], 0.32265687717586305 * 0.164
        )
        assert np.isclose(
            self.topology.lines_df.at[line_1, "x"],
            0.38 * 2 * np.pi * 50 / 1e3 * 0.32265687717586305,
        )
        assert np.isclose(
            self.topology.lines_df.at[line_1, "s_nom"], 0.357 * 20 * np.sqrt(3)
        )

        # test for multiple LV lines
        line_1 = "Line_50000006"
        line_2 = "Line_90000010"
        self.topology.change_line_type([line_1, line_2], "NAYY 4x1x300")

        assert (
            self.topology.lines_df.loc[[line_1, line_2], "type_info"] == "NAYY 4x1x300"
        ).all()
        assert np.isclose(
            self.topology.lines_df.at[line_1, "r"],
            0.1 * self.topology.lines_df.at[line_1, "length"],
        )
        assert np.isclose(
            self.topology.lines_df.at[line_2, "r"],
            0.1 * self.topology.lines_df.at[line_2, "length"],
        )
        assert (
            self.topology.lines_df.loc[[line_1, line_2], "s_nom"]
            == np.sqrt(3) * 0.4 * 0.419
        ).all()

    def test_sort_buses(self):
        lines_df_before = self.topology.lines_df.copy()

        self.topology.sort_buses()

        # check that buses were exchanged
        line = "Line_10008"
        assert (
            lines_df_before.at[line, "bus0"] == self.topology.lines_df.at[line, "bus1"]
        )
        assert (
            lines_df_before.at[line, "bus1"] == self.topology.lines_df.at[line, "bus0"]
        )

        # check number of lines where buses were exchanged
        assert (lines_df_before.bus0 == self.topology.lines_df.bus0).value_counts().loc[
            False
        ] == 11

    def test_to_csv(self):
        """Test for method to_csv."""
        dir = os.path.join(os.getcwd(), "topology")
        self.topology.to_csv(dir)

        saved_files = os.listdir(dir)
        assert len(saved_files) == 9
        assert "generators.csv" in saved_files

        shutil.rmtree(dir)

    def test_assign_feeders(self):
        # Test mode 'grid_feeder'
        self.topology.assign_feeders(mode="grid_feeder")
        assert self.topology.buses_df.loc[
            ["Bus_MVStation_1", "Bus_Generator_1"], "grid_feeder"
        ].to_list() == [
            "station_node",
            "Bus_BranchTee_MVGrid_1_1",
        ]
        assert self.topology.lines_df.loc[
            ["Line_10003", "Line_10004"], "grid_feeder"
        ].to_list() == [
            "Bus_BranchTee_MVGrid_1_1",
            "Bus_BranchTee_MVGrid_1_4",
        ]

        # test mode 'mv_feeder'
        self.topology.assign_feeders(mode="mv_feeder")
        assert self.topology.buses_df.loc[
            ["Bus_MVStation_1", "Bus_Generator_1"], "mv_feeder"
        ].to_list() == [
            "station_node",
            "Bus_BranchTee_MVGrid_1_1",
        ]
        assert self.topology.lines_df.loc[
            ["Line_10003", "Line_10004"], "mv_feeder"
        ].to_list() == [
            "Bus_BranchTee_MVGrid_1_1",
            "Bus_BranchTee_MVGrid_1_4",
        ]
        lv_grids_mv_bus = self.topology.grids[2].transformers_df["bus0"][0]
        feeder_of_lv_grids_mv_bus = self.topology.buses_df.loc[
            lv_grids_mv_bus, "mv_feeder"
        ]
        list_of_feeders = self.topology.grids[2].buses_df["mv_feeder"].to_list()
        assert len(list_of_feeders) == 15
        assert len(set(list_of_feeders)) == 1
        assert list_of_feeders[0] == feeder_of_lv_grids_mv_bus
        list_of_feeders = self.topology.grids[2].lines_df["mv_feeder"].to_list()
        assert len(list_of_feeders) == 14
        assert len(set(list_of_feeders)) == 1
        assert list_of_feeders[0] == feeder_of_lv_grids_mv_bus

    def test_aggregate_lv_grid_at_station(self, caplog):
        """Test method aggregate_lv_grid_at_station"""

        lv_grid_id = 1
        topology_obj = copy.deepcopy(self.topology)
        lv_grid_orig = self.topology.get_lv_grid(lv_grid_id)
        topology_obj.aggregate_lv_grid_at_station(lv_grid_id=lv_grid_id)
        lv_grid = topology_obj.get_lv_grid(lv_grid_id)

        assert lv_grid_orig.buses_df.shape[0] == 15
        assert lv_grid.buses_df.shape[0] == 1

        with caplog.at_level(logging.WARNING):
            topology_obj.check_integrity()
        assert "which are not defined" not in caplog.text
        assert "The following buses are isolated" not in caplog.text
        assert "The network has isolated nodes or edges." not in caplog.text


class TestTopologyWithEdisgoObject:
    """
    Tests methods in Topology that require edisgo object.

    """

    @pytest.yield_fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()

    def test_to_geopandas(self):
        geopandas_container = self.edisgo.topology.to_geopandas()

        assert isinstance(geopandas_container, GeoPandasGridContainer)

        attrs = [
            "buses_gdf",
            "generators_gdf",
            "lines_gdf",
            "loads_gdf",
            "storage_units_gdf",
            "transformers_gdf",
        ]

        for attr_str in attrs:
            attr = getattr(geopandas_container, attr_str)
            grid_attr = getattr(
                self.edisgo.topology.mv_grid, attr_str.replace("_gdf", "_df")
            )

            assert isinstance(attr, GeoDataFrame)

            common_cols = list(set(attr.columns).intersection(grid_attr.columns))

            assert_frame_equal(
                attr[common_cols], grid_attr[common_cols], check_names=False
            )

    def test_from_csv(self):
        """
        Test for method from_csv.

        """
        dir = os.path.join(os.getcwd(), "topology")
        self.edisgo.topology.to_csv(dir)

        # reset self.topology
        self.edisgo.topology = Topology()

        self.edisgo.topology.from_csv(dir, self.edisgo)

        assert len(self.edisgo.topology.loads_df) == 50
        assert len(self.edisgo.topology.generators_df) == 28
        assert self.edisgo.topology.charging_points_df.empty
        assert len(self.edisgo.topology.storage_units_df) == 1
        assert len(self.edisgo.topology.lines_df) == 131
        assert len(self.edisgo.topology.buses_df) == 142
        assert len(self.edisgo.topology.switches_df) == 2
        assert self.edisgo.topology.grid_district["population"] == 23358

        # check if analyze works
        self.edisgo.analyze()

        shutil.rmtree(dir)

    def test_connect_to_mv(self):
        """
        Tests connect_to_mv method and implicitly
        _connect_mv_bus_to_target_object method.

        """

        # ######### Generator #############
        # test voltage level 4
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_2", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_2", "y"]
        geom = Point((x, y))
        test_gen = {
            "generator_id": 12345,
            "p_nom": 2.5,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 4,
        }

        comp_name = self.edisgo.topology.connect_to_mv(self.edisgo, test_gen)

        # check if number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(new_bus)
        assert len(new_line_df) == 1
        # check that other bus of new line is the station
        assert (
            self.edisgo.topology.mv_grid.station.index[0] == new_line_df.bus0.values[0]
        )
        # check new generator
        assert (
            self.edisgo.topology.generators_df.at[comp_name, "p_nom"]
            == test_gen["p_nom"]
        )

        # test voltage level 5 (line split)
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_2", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = {
            "generator_id": 123456,
            "p_nom": 2.5,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 5,
        }

        comp_name = self.edisgo.topology.connect_to_mv(self.edisgo, test_gen)

        # check if number of buses increased (by two because closest connection
        # object is a line)
        assert len(buses_before) + 2 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 2 == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(new_bus)
        assert len(new_line_df) == 1
        assert "Bus_Generator_123456" in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]]
        )
        # check new generator
        assert (
            self.edisgo.topology.generators_df.at[comp_name, "p_nom"]
            == test_gen["p_nom"]
        )

        # test voltage level 5 (connected to bus)
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = {
            "generator_id": 123456,
            "p_nom": 2.5,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 5,
        }

        comp_name = self.edisgo.topology.connect_to_mv(self.edisgo, test_gen)

        # check if number of buses increased (by one because closest connection
        # object is a bus)
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check new generator
        assert (
            self.edisgo.topology.generators_df.at[comp_name, "p_nom"]
            == test_gen["p_nom"]
        )

        # ######### Charging Point #############
        # method not different from generators, wherefore only one voltage
        # level is tested
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        charging_points_before = self.edisgo.topology.charging_points_df

        # add charging point
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_2", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_2", "y"]
        geom = Point((x, y))
        test_cp = {
            "geom": geom,
            "p_set": 2.5,
            "sector": "hpc",
            "number": 10,
            "voltage_level": 4,
        }

        comp_name = self.edisgo.topology.connect_to_mv(
            self.edisgo, test_cp, comp_type="charging_point"
        )

        # check if number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check if number of charging points increased
        assert len(charging_points_before) + 1 == len(
            self.edisgo.topology.charging_points_df
        )

        # check new bus
        new_bus = self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(new_bus)
        assert len(new_line_df) == 1
        # check that other bus of new line is the station
        assert (
            self.edisgo.topology.mv_grid.station.index[0] == new_line_df.bus0.values[0]
        )
        # check new generator
        assert (
            self.edisgo.topology.charging_points_df.at[comp_name, "number"]
            == test_cp["number"]
        )

        # ######### Heat Pump #############
        # method not different from generators, wherefore only one voltage
        # level is tested
        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        loads_before = self.edisgo.topology.loads_df

        # add heat pump
        test_hp = {
            "geom": geom,
            "p_set": 2.5,
            "sector": "district_heating",
            "voltage_level": 4,
        }

        comp_name = self.edisgo.topology.connect_to_mv(
            self.edisgo, test_hp, comp_type="heat_pump"
        )

        # check if number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check if number of charging points increased
        assert len(loads_before) + 1 == len(self.edisgo.topology.loads_df)

        # check new bus
        new_bus = self.edisgo.topology.loads_df.at[comp_name, "bus"]
        assert "HeatPump" in new_bus
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 20
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(new_bus)
        assert len(new_line_df) == 1
        # check that other bus of new line is the station
        assert (
            self.edisgo.topology.mv_grid.station.index[0] == new_line_df.bus0.values[0]
        )
        # check new heat pump
        assert (
            self.edisgo.topology.loads_df.at[comp_name, "sector"] == "district_heating"
        )
        assert self.edisgo.topology.loads_df.at[comp_name, "type"] == "heat_pump"

        # ######### Storage unit #############
        # add generator
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_stor = {
            "p_nom": 2.5,
            "geom": geom,
            "voltage_level": 5,
        }
        num_storage_units_before = len(self.edisgo.topology.storage_units_df)
        num_buses_before = len(self.edisgo.topology.buses_df)
        num_lines_before = len(self.edisgo.topology.lines_df)
        comp_name = self.edisgo.topology.connect_to_mv(
            self.edisgo, test_stor, comp_type="storage_unit"
        )

        # check if number of buses increased (by one because closest connection
        # object is a bus)
        assert num_buses_before + 1 == len(self.edisgo.topology.buses_df)
        # check if number of lines increased
        assert num_lines_before + 1 == len(self.edisgo.topology.lines_df)
        # check if number of storage units increased
        assert num_storage_units_before + 1 == len(
            self.edisgo.topology.storage_units_df
        )

        # check new storage
        assert (
            self.edisgo.topology.storage_units_df.at[comp_name, "p_nom"]
            == test_stor["p_nom"]
        )
        assert self.edisgo.topology.storage_units_df.at[comp_name, "control"] == "PQ"
        assert "Storage" in self.edisgo.topology.storage_units_df.at[comp_name, "bus"]

    def test_connect_to_lv(self):
        # ######### Generator #############

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
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 6,
            "mvlv_subst_id": 10.0,
        }

        comp_name = self.edisgo.topology.connect_to_lv(self.edisgo, test_gen)

        # check if number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check if number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check that new generator is connected to random substation
        # assert self.edisgo.topology.generators_df.at[
        #            comp_name, "bus"] == 'BusBar_MVGrid_1_LVGrid_7_LV'
        assert (
            self.edisgo.topology.generators_df.at[comp_name, "bus"] == "Bus_MVStation_1"
        )

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
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 6,
            "mvlv_subst_id": None,
        }

        comp_name = self.edisgo.topology.connect_to_lv(self.edisgo, test_gen)

        # check if number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check if number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check that new generator is connected to random substation
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 0.4
        lv_grid_id = self.edisgo.topology.buses_df.at[new_bus, "lv_grid_id"]
        lv_grid = self.edisgo.topology.get_lv_grid(int(lv_grid_id))
        assert new_bus == lv_grid.station.index[0]
        # check new generator
        assert self.edisgo.topology.generators_df.at[comp_name, "p_nom"] == 0.3

        # test missing geom in voltage level 6

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
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 6,
            "mvlv_subst_id": None,
            "geom": None,
        }

        comp_name = self.edisgo.topology.connect_to_lv(self.edisgo, test_gen)

        # check if number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check if number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check if number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check that new generator is connected to random substation
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 0.4
        lv_grid_id = self.edisgo.topology.buses_df.at[new_bus, "lv_grid_id"]
        lv_grid = self.edisgo.topology.get_lv_grid(int(lv_grid_id))
        assert new_bus == lv_grid.station.index[0]
        # check new generator
        assert self.edisgo.topology.generators_df.at[comp_name, "p_nom"] == 0.3

        # test existing substation ID and geom (voltage level 6)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        generators_before = self.edisgo.topology.generators_df

        # add generator
        x = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "x"]
        y = self.edisgo.topology.buses_df.at["Bus_GeneratorFluctuating_6", "y"]
        geom = Point((x, y))
        test_gen = {
            "generator_id": 3456,
            "p_nom": 0.3,
            "geom": geom,
            "generator_type": "solar",
            "subtype": "roof",
            "weather_cell_id": self.edisgo.topology.generators_df.at[
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 6,
            "mvlv_subst_id": 6,
        }

        comp_name = self.edisgo.topology.connect_to_lv(self.edisgo, test_gen)

        # check that number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check that number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check that number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check new bus
        new_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 0.4
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(new_bus)
        assert len(new_line_df) == 1
        assert "Bus_Generator_3456" in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]]
        )
        lv_grid = self.edisgo.topology.get_lv_grid(6)
        assert lv_grid.station.index[0] in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]]
        )
        # check new generator
        assert self.edisgo.topology.generators_df.at[comp_name, "p_nom"] == 0.3
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
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 7,
            "mvlv_subst_id": 1,
        }

        comp_name = self.edisgo.topology.connect_to_lv(self.edisgo, test_gen)

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check bus
        gen_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert gen_bus == "Bus_BranchTee_LVGrid_1_10"
        assert self.edisgo.topology.buses_df.at[gen_bus, "lv_grid_id"] == 1
        # check new generator
        assert self.edisgo.topology.generators_df.at[comp_name, "p_nom"] == 0.03

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
                "GeneratorFluctuating_2", "weather_cell_id"
            ],
            "voltage_level": 7,
            "mvlv_subst_id": 2,
        }

        comp_name = self.edisgo.topology.connect_to_lv(self.edisgo, test_gen)

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of generators increased
        assert len(generators_before) + 1 == len(self.edisgo.topology.generators_df)

        # check bus
        gen_bus = self.edisgo.topology.generators_df.at[comp_name, "bus"]
        assert gen_bus == "Bus_BranchTee_LVGrid_2_1"
        assert self.edisgo.topology.buses_df.at[gen_bus, "lv_grid_id"] == 2
        # check new generator
        assert self.edisgo.topology.generators_df.at[comp_name, "p_nom"] == 0.04

        # ######### Charging Point #############

        # test voltage level 7 - use case home (and there are residential
        # loads to add charging point to)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        cp_before = self.edisgo.topology.charging_points_df

        # add charging point
        test_cp = {
            "p_set": 0.01,
            "geom": geom,
            "sector": "home",
            "voltage_level": 7,
            "mvlv_subst_id": 3.0,
        }

        comp_name = self.edisgo.topology.connect_to_lv(
            self.edisgo, test_cp, comp_type="charging_point"
        )

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of charging points increased
        assert len(cp_before) + 1 == len(self.edisgo.topology.charging_points_df)

        # check bus
        bus = self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
        assert bus == "Bus_BranchTee_LVGrid_3_6"
        assert self.edisgo.topology.buses_df.at[bus, "lv_grid_id"] == 3
        # check new charging point
        assert self.edisgo.topology.charging_points_df.at[comp_name, "p_set"] == 0.01

        # test voltage level 7 - use case work (connected to agricultural load)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        cp_before = self.edisgo.topology.charging_points_df

        # add charging point
        test_cp = {
            "p_set": 0.02,
            "number": 2,
            "geom": geom,
            "sector": "work",
            "voltage_level": 7,
            "mvlv_subst_id": 3,
        }

        comp_name = self.edisgo.topology.connect_to_lv(
            self.edisgo, test_cp, comp_type="charging_point"
        )

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of charging points increased
        assert len(cp_before) + 1 == len(self.edisgo.topology.charging_points_df)

        # check bus
        bus = self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
        assert bus == "Bus_BranchTee_LVGrid_3_2"
        assert self.edisgo.topology.buses_df.at[bus, "lv_grid_id"] == 3
        # check new charging point
        assert self.edisgo.topology.charging_points_df.at[comp_name, "number"] == 2

        # test voltage level 7 - use case public (connected somewhere in the
        # LV grid (to bus not in_building))

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        cp_before = self.edisgo.topology.charging_points_df

        # add charging point
        test_cp = {
            "p_set": 0.02,
            "number": 2,
            "geom": geom,
            "sector": "public",
            "voltage_level": 7,
            "mvlv_subst_id": 3,
        }

        comp_name = self.edisgo.topology.connect_to_lv(
            self.edisgo, test_cp, comp_type="charging_point"
        )

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of charging points increased
        assert len(cp_before) + 1 == len(self.edisgo.topology.charging_points_df)

        # check bus
        bus = self.edisgo.topology.charging_points_df.at[comp_name, "bus"]
        assert bus == "BusBar_MVGrid_1_LVGrid_3_LV"
        assert self.edisgo.topology.buses_df.at[bus, "lv_grid_id"] == 3
        # check new charging point
        assert self.edisgo.topology.charging_points_df.at[comp_name, "number"] == 2

        # ######### Heat Pump #############

        # test voltage level 7 - sector individual heating

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        loads_before = self.edisgo.topology.loads_df

        # add heat pump
        test_hp = {
            "p_set": 0.01,
            "geom": geom,
            "sector": "individual_heating",
            "voltage_level": 7,
            "mvlv_subst_id": 3.0,
        }

        comp_name = self.edisgo.topology.connect_to_lv(
            self.edisgo, test_hp, comp_type="heat_pump"
        )

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of loads increased
        assert len(loads_before) + 1 == len(self.edisgo.topology.loads_df)

        # check bus
        bus = self.edisgo.topology.loads_df.at[comp_name, "bus"]
        assert bus == "Bus_BranchTee_LVGrid_3_8"
        assert self.edisgo.topology.buses_df.at[bus, "lv_grid_id"] == 3
        # check new heat pump
        assert self.edisgo.topology.loads_df.at[comp_name, "p_set"] == 0.01

        # test voltage level 7 - sector district_heating

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        loads_before = self.edisgo.topology.loads_df

        # add heat pump
        test_hp = {
            "p_set": 0.02,
            "number": 2,
            "geom": geom,
            "sector": "district_heating",
            "voltage_level": 7,
            "mvlv_subst_id": 3,
        }

        comp_name = self.edisgo.topology.connect_to_lv(
            self.edisgo, test_hp, comp_type="heat_pump"
        )

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of loads increased
        assert len(loads_before) + 1 == len(self.edisgo.topology.loads_df)

        # check bus
        bus = self.edisgo.topology.loads_df.at[comp_name, "bus"]
        assert bus == "Bus_BranchTee_LVGrid_3_3"
        assert self.edisgo.topology.buses_df.at[bus, "lv_grid_id"] == 3
        # check new heat pump
        assert self.edisgo.topology.loads_df.at[comp_name, "number"] == 2

        # test voltage level 7 - other sector

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        loads_before = self.edisgo.topology.loads_df

        # add charging point
        test_hp = {
            "p_set": 0.02,
            "number": 2,
            "geom": geom,
            "sector": None,
            "voltage_level": 7,
            "mvlv_subst_id": 3,
        }

        comp_name = self.edisgo.topology.connect_to_lv(
            self.edisgo, test_hp, comp_type="heat_pump"
        )

        # check that number of buses stayed the same
        assert len(buses_before) == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert len(lines_before) == len(self.edisgo.topology.lines_df)
        # check that number of loads increased
        assert len(loads_before) + 1 == len(self.edisgo.topology.loads_df)

        # check bus
        bus = self.edisgo.topology.loads_df.at[comp_name, "bus"]
        assert bus == "Bus_BranchTee_LVGrid_3_8"
        assert self.edisgo.topology.buses_df.at[bus, "lv_grid_id"] == 3
        # check new heat pump
        assert self.edisgo.topology.loads_df.at[comp_name, "type"] == "heat_pump"

        # test voltage level 6
        # test existing substation ID and geom (voltage level 6)

        lines_before = self.edisgo.topology.lines_df
        buses_before = self.edisgo.topology.buses_df
        loads_before = self.edisgo.topology.loads_df

        test_hp = {
            "p_set": 0.3,
            "geom": geom,
            "voltage_level": 6,
            "mvlv_subst_id": 6,
        }

        comp_name = self.edisgo.topology.connect_to_lv(
            self.edisgo, test_hp, comp_type="heat_pump"
        )

        # check that number of buses increased
        assert len(buses_before) + 1 == len(self.edisgo.topology.buses_df)
        # check that number of lines increased
        assert len(lines_before) + 1 == len(self.edisgo.topology.lines_df)
        # check that number of loads increased
        assert len(loads_before) + 1 == len(self.edisgo.topology.loads_df)

        # check new bus
        new_bus = self.edisgo.topology.loads_df.at[comp_name, "bus"]
        assert self.edisgo.topology.buses_df.at[new_bus, "v_nom"] == 0.4
        # check new line
        new_line_df = self.edisgo.topology.get_connected_lines_from_bus(new_bus)
        assert len(new_line_df) == 1
        assert "Bus_HeatPump_56" in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]]
        )
        lv_grid = self.edisgo.topology.get_lv_grid(6)
        assert lv_grid.station.index[0] in list(
            new_line_df.loc[new_line_df.index[0], ["bus0", "bus1"]]
        )
        # check new heat pump
        assert self.edisgo.topology.loads_df.at[comp_name, "p_set"] == 0.3

        # ############# storage unit #################
        # test existing substation ID (voltage level 7)
        # storage can be connected to residential load

        num_lines_before = len(self.edisgo.topology.lines_df)
        num_buses_before = len(self.edisgo.topology.buses_df)
        num_stores_before = len(self.edisgo.topology.storage_units_df)

        # add generator
        test_stor = {
            "p_nom": 0.03,
            "geom": geom,
            "voltage_level": 7,
            "mvlv_subst_id": 1,
        }

        comp_name = self.edisgo.topology.connect_to_lv(
            self.edisgo, test_stor, comp_type="storage_unit"
        )

        # check that number of buses stayed the same
        assert num_buses_before == len(self.edisgo.topology.buses_df)
        # check that number of lines stayed the same
        assert num_lines_before == len(self.edisgo.topology.lines_df)
        # check that number of storage units increased
        assert num_stores_before + 1 == len(self.edisgo.topology.storage_units_df)

        # check bus
        bus = self.edisgo.topology.storage_units_df.at[comp_name, "bus"]
        assert bus == "Bus_BranchTee_LVGrid_1_12"
        assert self.edisgo.topology.buses_df.at[bus, "lv_grid_id"] == 1
        # check new storage
        assert self.edisgo.topology.storage_units_df.at[comp_name, "p_nom"] == 0.03

    def test_check_integrity(self, caplog):
        """Test of validation of grids."""
        comps_dict = {
            "buses": "BusBar_MVGrid_1_LVGrid_2_MV",
            "generators": "GeneratorFluctuating_14",
            "loads": "Load_residential_LVGrid_3_2",
            "transformers": "LVStation_5_transformer_1",
            "lines": "Line_10014",
            "switches": "circuit_breaker_1",
        }
        # check duplicate node
        for comp, name in comps_dict.items():
            new_comp = getattr(self.edisgo.topology, "_{}_df".format(comp)).loc[name]
            comps = getattr(self.edisgo.topology, "_{}_df".format(comp))
            setattr(
                self.edisgo.topology,
                "_{}_df".format(comp),
                pd.concat([comps, new_comp.to_frame().T]),
            )  # comps.append(new_comp))
            self.edisgo.topology.check_integrity()
            assert (
                f"{name} have duplicate entry in one of the following components' "
                f"dataframes: {comp}." in caplog.text
            )
            caplog.clear()

            # reset dataframe
            setattr(self.edisgo.topology, "_{}_df".format(comp), comps)
            self.edisgo.topology.check_integrity()

        # check not connected generator and load
        for nodal_component in ["loads", "generators"]:
            comps = getattr(self.edisgo.topology, "_{}_df".format(nodal_component))
            new_comp = comps.loc[comps_dict[nodal_component]]
            new_comp.name = "new_nodal_component"
            new_comp.bus = "Non_existent_bus_" + nodal_component
            setattr(
                self.edisgo.topology,
                "_{}_df".format(nodal_component),
                pd.concat([comps, new_comp.to_frame().T]),
            )
            self.edisgo.topology.check_integrity()
            assert (
                "The following {} have buses which are not defined: {}.".format(
                    nodal_component, new_comp.name
                )
                in caplog.text
            )
            caplog.clear()
            # reset dataframe
            setattr(self.edisgo.topology, "_{}_df".format(nodal_component), comps)
            self.edisgo.topology.check_integrity()

        # check branch components
        i = 0
        for branch_component in ["lines", "transformers"]:
            comps = getattr(self.edisgo.topology, "_{}_df".format(branch_component))
            new_comp = comps.loc[comps_dict[branch_component]]
            new_comp.name = "new_branch_component"
            setattr(
                new_comp,
                "bus" + str(i),
                "Non_existent_bus_" + branch_component,
            )
            setattr(
                self.edisgo.topology,
                "_{}_df".format(branch_component),
                pd.concat([comps, new_comp.to_frame().T]),
            )
            self.edisgo.topology.check_integrity()
            assert (
                "The following {} have bus{} which are not defined: {}.".format(
                    branch_component, i, new_comp.name
                )
                in caplog.text
            )
            caplog.clear()
            # reset dataframe
            setattr(self.edisgo.topology, "_{}_df".format(branch_component), comps)
            self.edisgo.topology.check_integrity()
            i += 1

        # check switches
        comps = self.edisgo.topology.switches_df
        for attr in ["bus_open", "bus_closed"]:
            new_comp = comps.loc[comps_dict["switches"]]
            new_comp.name = "new_switch"
            new_comps = pd.concat([comps, new_comp.to_frame().T])
            new_comps.at[new_comp.name, attr] = "Non_existent_" + attr
            self.edisgo.topology.switches_df = new_comps
            self.edisgo.topology.check_integrity()
            assert (
                "The following switches have {} which are not defined: {}.".format(
                    attr, new_comp.name
                )
                in caplog.text
            )
            caplog.clear()
            self.edisgo.topology.switches_df = comps
            self.edisgo.topology.check_integrity()

        # check isolated node
        bus = self.edisgo.topology.buses_df.loc[comps_dict["buses"]]
        bus.name = "New_bus"
        self.edisgo.topology.buses_df = pd.concat(
            [self.edisgo.topology.buses_df, bus.to_frame().T]
        )
        self.edisgo.topology.check_integrity()
        assert "The following buses are isolated: {}.".format(bus.name) in caplog.text
        assert "The network has isolated nodes or edges." in caplog.text
        caplog.clear()

        # check small impedance and large/short line length
        line = "Line_10017"
        self.edisgo.topology.lines_df.at[line, "length"] = 12.0
        self.edisgo.topology.lines_df.at[line, "x"] = 1e-7
        self.edisgo.topology.lines_df.at[line, "r"] = 1e-7
        self.edisgo.topology.check_integrity()
        assert "There are lines with very large line lengths" in caplog.text
        assert "There are lines with very short line lengths" in caplog.text
        assert "Very small values for impedance of lines" and line in caplog.text
        caplog.clear()
