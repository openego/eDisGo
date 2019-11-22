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
