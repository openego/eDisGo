import pytest

from edisgo import EDisGo
from edisgo.tools import geopandas_helper


class TestGeopandasHelper:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)

    def test_to_geopandas(self):
        # further tests of this function are conducted in test_topology.py
        # test MV grid
        data = geopandas_helper.to_geopandas(self.edisgo.topology.mv_grid, 4326)
        assert data.buses_gdf.shape[0] == self.edisgo.topology.mv_grid.buses_df.shape[0]
        assert (
            data.buses_gdf.shape[1]
            == self.edisgo.topology.mv_grid.buses_df.shape[1] + 1 - 2
        )
        assert "geometry" in data.buses_gdf.columns

        assert data.lines_gdf.shape[0] == self.edisgo.topology.mv_grid.lines_df.shape[0]
        assert (
            data.lines_gdf.shape[1]
            == self.edisgo.topology.mv_grid.lines_df.shape[1] + 2
        )
        assert "geometry" in data.lines_gdf.columns

        assert data.loads_gdf.shape[0] == self.edisgo.topology.mv_grid.loads_df.shape[0]
        assert (
            data.loads_gdf.shape[1]
            == self.edisgo.topology.mv_grid.loads_df.shape[1] + 2
        )
        assert "geometry" in data.loads_gdf.columns

        assert (
            data.generators_gdf.shape[0]
            == self.edisgo.topology.mv_grid.generators_df.shape[0]
        )
        assert (
            data.generators_gdf.shape[1]
            == self.edisgo.topology.mv_grid.generators_df.shape[1] + 2
        )
        assert "geometry" in data.generators_gdf.columns

        assert (
            data.storage_units_gdf.shape[0]
            == self.edisgo.topology.mv_grid.storage_units_df.shape[0]
        )
        assert (
            data.storage_units_gdf.shape[1]
            == self.edisgo.topology.mv_grid.storage_units_df.shape[1] + 2
        )
        assert "geometry" in data.storage_units_gdf.columns

        assert (
            data.transformers_gdf.shape[0]
            == self.edisgo.topology.mv_grid.transformers_df.shape[0]
        )
        assert (
            data.transformers_gdf.shape[1]
            == self.edisgo.topology.mv_grid.transformers_df.shape[1] + 2
        )
        assert "geometry" in data.transformers_gdf.columns
