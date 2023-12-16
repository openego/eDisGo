import pytest
import shapely

from edisgo.io import ding0_import
from edisgo.network.grids import MVGrid
from edisgo.network.topology import Topology


class TestImportFromDing0:
    @classmethod
    def setup_class(self):
        self.topology = Topology()

    def test_import_ding0_grid(self):
        """Test successful import of ding0 network."""

        ding0_import.import_ding0_grid(pytest.ding0_test_network_path, self)

        # buses, generators, loads, lines, transformers dataframes
        # check number of imported components
        assert self.topology.buses_df.shape[0] == 142
        assert self.topology.generators_df.shape[0] == 28
        assert self.topology.loads_df.shape[0] == 50
        assert self.topology.lines_df.shape[0] == 131
        assert self.topology.transformers_df.shape[0] == 14
        assert self.topology.transformers_hvmv_df.shape[0] == 1
        assert self.topology.switches_df.shape[0] == 2
        assert self.topology.storage_units_df.shape[0] == 1

        # grid district
        assert self.topology.grid_district["population"] == 23358
        assert isinstance(self.topology.grid_district["geom"], shapely.geometry.Polygon)

        # grids
        assert isinstance(self.topology.mv_grid, MVGrid)
        lv_grid = self.topology.get_lv_grid(3)
        assert len(lv_grid.buses_df) == 9

    def test_path_error(self):
        """Test catching error when path to network does not exist."""
        msg = "Directory wrong_directory does not exist."
        with pytest.raises(AssertionError, match=msg):
            ding0_import.import_ding0_grid("wrong_directory", self.topology)

    def test_transformer_buses(self):
        ding0_import.import_ding0_grid(pytest.ding0_test_network_path, self)
        assert (
            self.topology.buses_df.loc[self.topology.transformers_df.bus1].v_nom.values
            < self.topology.buses_df.loc[
                self.topology.transformers_df.bus0
            ].v_nom.values
        ).all()
        self.topology.transformers_df.loc[
            "LVStation_7_transformer_1", "bus0"
        ] = "Bus_secondary_LVStation_7"
        self.topology.transformers_df.loc[
            "LVStation_7_transformer_1", "bus1"
        ] = "Bus_primary_LVStation_7"
        with pytest.raises(AssertionError):
            assert (
                self.topology.buses_df.reindex(
                    index=self.topology.transformers_df.bus1
                ).v_nom.values
                < self.topology.buses_df.reindex(
                    index=self.topology.transformers_df.bus0
                ).v_nom.values
            ).all()
