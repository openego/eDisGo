import os
import pytest
import warnings
import shutil
import pandas as pd
from pandas.util.testing import assert_frame_equal

from edisgo.network.topology import Topology
from edisgo.io import ding0_import


class TestTopology:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(
            parent_dirname, 'ding0_test_network')
        self.topology = Topology()
        ding0_import.import_ding0_grid(test_network_directory, self)

    def test_add_line(self):
        """Test add_line method"""

        len_df_before = len(self.topology.lines_df)

        bus0 = 'Bus_BranchTee_MVGrid_1_8'
        bus1 = 'Bus_GeneratorFluctuating_7'
        name = self.topology.add_line(
            bus0=bus0,
            bus1=bus1,
            length=1,
            x=1,
            r=1,
            s_nom=1,
            kind='cable'
        )

        assert len_df_before + 1 == len(self.topology.lines_df)
        assert name == \
               'Line_Bus_BranchTee_MVGrid_1_8_Bus_GeneratorFluctuating_7'
        assert self.topology.lines_df.loc[name, 'bus0'] == bus0

        bus0 = 'Bus_BranchTee_MVGrid_1_8'
        bus1 = 'Bus_GeneratorFluctuating_7'
        name = self.topology.add_line(
            bus0=bus0,
            bus1=bus1,
            length=1,
            kind='cable',
            type_info='NA2XS2Y 3x1x185 RM/25',
            x=2
        )
        with pytest.warns(Warning):
            warnings.warn(
                "When line 'type_info' is provided when creating a new "
                "line, x, r and s_nom are calculated and provided "
                "parameters are overwritten.", Warning)
        assert self.topology.lines_df.loc[name, 'x'] == 1

    def test_add_generator(self):
        """Test add_generator method"""

        len_df_before = len(self.topology.generators_df)

        name = self.topology.add_generator(
            generator_id=2,
            bus='Bus_BranchTee_MVGrid_1_8',
            p_nom=1,
            generator_type='solar',
            subtype='roof',
            weather_cell_id=1000)

        assert len_df_before + 1 == len(self.topology.generators_df)
        assert name == 'Generator_solar_2'
        assert self.topology.generators_df.loc[name, 'p_nom'] == 1

    def test_add_load(self):
        """Test add_load method"""

        msg = "Specified bus Unknown_bus is not valid as it is not defined in " \
              "buses_df."
        with pytest.raises(ValueError, match=msg):
            self.topology.add_load(
                load_id=8,
                bus="Unknown_bus",
                peak_load=1,
                annual_consumption=1,
                sector='retail'
            )

        len_df_before = len(self.topology.loads_df)

        # check if name of load does not exist yet
        name = self.topology.add_load(
                load_id=10,
                bus="Bus_BranchTee_LVGrid_1_4",
                peak_load=1,
                annual_consumption=2,
                sector='residential'
            )
        assert len_df_before+1 == len(self.topology.loads_df)
        assert name == "Load_residential_LVGrid_1_10"
        assert self.topology.loads_df.loc[name, 'peak_load'] == 1
        assert self.topology.loads_df.loc[name, 'annual_consumption'] == 2
        assert self.topology.loads_df.loc[name, 'sector'] == "residential"

        # check auto creation of name when load name with load_id already
        # exists
        name = self.topology.add_load(
            load_id=1,
            bus="Bus_BranchTee_LVGrid_1_4",
            peak_load=2,
            annual_consumption=1,
            sector='agricultural'
        )
        assert len_df_before + 2 == len(self.topology.loads_df)
        assert name == "Load_agricultural_LVGrid_1_9"
        assert self.topology.loads_df.loc[name, 'peak_load'] == 2
        assert self.topology.loads_df.loc[name, 'annual_consumption'] == 1
        assert self.topology.loads_df.loc[name, 'sector'] == "agricultural"

        # check auto creation of name if auto created name already exists
        name = self.topology.add_load(
            load_id=4,
            bus="Bus_BranchTee_LVGrid_1_4",
            peak_load=5,
            annual_consumption=4,
            sector='residential'
        )

        assert len_df_before + 3 == len(self.topology.loads_df)
        assert name != "Load_residential_LVGrid_1_10"
        assert len(name) == (len("Load_residential_LVGrid_1_") + 9)
        assert self.topology.loads_df.loc[name, 'peak_load'] == 5
        assert self.topology.loads_df.loc[name, 'annual_consumption'] == 4
        assert self.topology.loads_df.loc[name, 'sector'] == "residential"

    def test_add_storage_unit(self):
        """Test add_storage_unit method"""

        msg = "Specified bus Unknown_bus is not valid as it is not defined in " \
              "buses_df."
        with pytest.raises(ValueError, match=msg):
            self.topology.add_storage_unit(
                storage_id=8,
                bus="Unknown_bus",
                p_nom=1,
                control='PQ'
            )

        len_df_before = len(self.topology.storage_units_df)

        # check if name of load does not exist yet
        name = self.topology.add_storage_unit(
            storage_id=3,
            bus="Bus_BranchTee_LVGrid_1_5",
            p_nom=1,
            control='Test'
            )
        assert len_df_before+1 == len(self.topology.storage_units_df)
        assert name == "StorageUnit_LVGrid_1_3"
        assert self.topology.storage_units_df.loc[name, 'p_nom'] == 1
        assert self.topology.storage_units_df.loc[name, 'control'] == "Test"

        # check auto creation of name when load name with load_id already
        # exists
        name = self.topology.add_storage_unit(
            storage_id=3,
            bus="Bus_BranchTee_LVGrid_1_4",
            p_nom=2
        )
        assert len_df_before + 2 == len(self.topology.storage_units_df)
        assert name == "StorageUnit_LVGrid_1_2"
        assert self.topology.storage_units_df.loc[name, 'p_nom'] == 2
        assert self.topology.storage_units_df.loc[name, 'control'] == 'PQ'

        # check auto creation of name if auto created name already exists
        name = self.topology.add_storage_unit(
            storage_id=3,
            bus="Bus_BranchTee_LVGrid_1_4",
            p_nom=5
        )

        assert len_df_before + 3 == len(self.topology.storage_units_df)
        assert name != "StorageUnit_LVGrid_1_3"
        assert len(name) == 30
        assert self.topology.storage_units_df.loc[name, 'p_nom'] == 5
        assert self.topology.storage_units_df.loc[name, 'control'] == 'PQ'

    def test_add_bus(self):
        """Test add_bus method"""
        len_df_pre = len(self.topology.buses_df)

        # check adding MV bus
        self.topology.add_bus(bus_name='Test_bus', v_nom=20)
        assert len_df_pre+1 == len(self.topology.buses_df)
        assert self.topology.buses_df.at['Test_bus', 'v_nom'] == 20
        assert self.topology.buses_df.at['Test_bus', 'mv_grid_id'] == 1

        # check LV assertion
        msg = "You need to specify an lv_grid_id for low-voltage buses."
        with pytest.raises(ValueError, match=msg):
            self.topology.add_bus('Test_bus_LV', v_nom=0.4)

        # check adding LV bus
        self.topology.add_bus('Test_bus_LV', v_nom=0.4, lv_grid_id=1)
        assert len_df_pre+2 == len(self.topology.buses_df)
        assert self.topology.buses_df.at['Test_bus_LV', 'v_nom']
        assert self.topology.buses_df.at['Test_bus_LV', 'lv_grid_id'] == 1
        assert self.topology.buses_df.at['Test_bus_LV', 'mv_grid_id'] == 1

    def test_remove_bus(self):
        """Test remove_bus method"""
        # create isolated bus to check
        self.topology.add_bus(bus_name='Test_bus_to_remove', v_nom=20)
        assert self.topology.buses_df.at['Test_bus_to_remove', 'v_nom'] == 20
        # check removing bus
        self.topology.remove_bus('Test_bus_to_remove')
        with pytest.raises(KeyError):
            self.topology.buses_df.loc['Test_bus_to_remove']
        # check assertion when bus is connected to element
        # check connected Generator
        msg = "Bus Bus_Generator_1 is not isolated. Remove all connected " \
              "elements first to remove bus."
        with pytest.raises(AssertionError, match=msg):
            self.topology.remove_bus('Bus_Generator_1')
        # check connected Load
        msg = "Bus Bus_Load_agricultural_LVGrid_1_1 is not isolated. " \
              "Remove all connected elements first to remove bus."
        with pytest.raises(AssertionError, match=msg):
            self.topology.remove_bus('Bus_Load_agricultural_LVGrid_1_1')
        # check connected line
        msg = "Bus Bus_BranchTee_MVGrid_1_1 is not isolated. " \
              "Remove all connected elements first to remove bus."
        with pytest.raises(AssertionError, match=msg):
            self.topology.remove_bus('Bus_BranchTee_MVGrid_1_1')

    def test_remove_generator(self):
        """Test remove_load method"""
        name_generator = 'GeneratorFluctuating_5'
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            'Bus_' + name_generator)
        # check if elements are part of topology:
        assert name_generator in self.topology.generators_df.index
        assert 'Bus_' + name_generator in self.topology.buses_df.index
        assert (connected_lines.index.isin(
                self.topology.lines_df.index)).all()
        # check case where only load is connected to line,
        # line and bus are therefore removed as well
        self.topology.remove_generator(name_generator)
        assert name_generator not in self.topology.generators_df.index
        assert 'Bus_' + name_generator not in self.topology.buses_df.index
        assert ~(connected_lines.index.isin(
            self.topology.lines_df.index)).any()

        # check case where load is not the only connected element
        name_generator = 'GeneratorFluctuating_7'
        self.topology.add_load(100, 'Bus_' + name_generator, 2, 3, 'agricultural')
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            'Bus_' + name_generator)
        # check if elements are part of topology:
        assert name_generator in self.topology.generators_df.index
        assert 'Bus_' + name_generator in self.topology.buses_df.index
        assert (connected_lines.index.isin(
            self.topology.lines_df.index)).all()
        # check case where other elements are connected to line as well,
        # line and bus are therefore not removed
        self.topology.remove_generator(name_generator)
        assert name_generator not in self.topology.generators_df.index
        assert 'Bus_' + name_generator in self.topology.buses_df.index
        assert (connected_lines.index.isin(
            self.topology.lines_df.index)).all()

    def test_remove_load(self):
        """Test remove_load method"""
        name_load = 'Load_residential_LVGrid_1_4'
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            'Bus_' + name_load)
        # check if elements are part of topology:
        assert name_load in self.topology.loads_df.index
        assert 'Bus_' + name_load in self.topology.buses_df.index
        assert (connected_lines.index.isin(
                self.topology.lines_df.index)).all()
        # check case where only load is connected to line,
        # line and bus are therefore removed as well
        self.topology.remove_load(name_load)
        assert name_load not in self.topology.loads_df.index
        assert 'Bus_' + name_load not in self.topology.buses_df.index
        assert ~(connected_lines.index.isin(
            self.topology.lines_df.index)).any()

        # check case where load is not the only connected element
        name_load = 'Load_residential_LVGrid_1_6'
        self.topology.add_load(100, 'Bus_' + name_load, 2, 3, 'agricultural')
        # get connected line
        connected_lines = self.topology.get_connected_lines_from_bus(
            'Bus_' + name_load)
        # check if elements are part of topology:
        assert name_load in self.topology.loads_df.index
        assert 'Bus_' + name_load in self.topology.buses_df.index
        assert (connected_lines.index.isin(
            self.topology.lines_df.index)).all()
        # check case where other elements are connected to line as well,
        # line and bus are therefore not removed
        self.topology.remove_load(name_load)
        assert name_load not in self.topology.loads_df.index
        assert 'Bus_' + name_load in self.topology.buses_df.index
        assert (connected_lines.index.isin(
            self.topology.lines_df.index)).all()

    def test_remove_line(self):
        """Test remove_line method"""

        # test only remove a line
        line_name = "Line_10012"
        bus0 = self.topology.lines_df.at[line_name, 'bus0']
        bus1 = self.topology.lines_df.at[line_name, 'bus1']
        self.topology.remove_line(line_name)
        assert line_name not in self.topology.lines_df.index
        assert bus0 in self.topology.buses_df.index
        assert bus1 in self.topology.buses_df.index

        # test remove line and bordering node
        self.topology.add_bus('Test_bus', 20)
        self.topology.add_line(bus0, 'Test_bus', 2)
        assert 'Test_bus' in self.topology.buses_df.index
        assert 'Line_Bus_BranchTee_MVGrid_1_3_Test_bus' in \
               self.topology.lines_df.index
        self.topology.remove_line('Line_Bus_BranchTee_MVGrid_1_3_Test_bus')
        assert bus0 in self.topology.buses_df.index
        assert 'Test_bus' not in self.topology.buses_df.index
        assert 'Line_Bus_BranchTee_MVGrid_1_3_Test_bus' not in \
            self.topology.lines_df.index

    def test_to_csv(self):
        """Test for method to_csv"""
        dir = os.getcwd()
        self.topology.to_csv(dir)
        edisgo = pd.DataFrame()
        edisgo.topology = Topology()
        ding0_import.import_ding0_grid(os.path.join(dir,'topology'), edisgo)
        assert_frame_equal(self.topology.buses_df, edisgo.topology.buses_df)
        assert_frame_equal(self.topology.generators_df,
                           edisgo.topology.generators_df)
        assert_frame_equal(self.topology.lines_df, edisgo.topology.lines_df)
        assert_frame_equal(self.topology.loads_df, edisgo.topology.loads_df)
        assert_frame_equal(self.topology.slack_df, edisgo.topology.slack_df)
        assert edisgo.topology.storage_units_df.empty
        assert_frame_equal(self.topology.switches_df,
                           edisgo.topology.switches_df)
        assert_frame_equal(self.topology.transformers_df,
                           edisgo.topology.transformers_df)
        assert_frame_equal(self.topology.transformers_hvmv_df,
                           edisgo.topology.transformers_hvmv_df)
        assert_frame_equal(pd.DataFrame([self.topology.grid_district]),
                           pd.DataFrame([edisgo.topology.grid_district]))
        shutil.rmtree(os.path.join(dir,'topology'), ignore_errors=True)
