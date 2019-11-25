import os
import pytest
import warnings

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
