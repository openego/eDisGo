import numpy as np
import pytest

from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.tools import geo


class TestTools:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )

    def test_find_nearest_bus(self):
        # test with coordinates of existing bus
        bus = self.edisgo.topology.buses_df.index[5]
        point = Point(
            (
                self.edisgo.topology.buses_df.at[bus, "x"],
                self.edisgo.topology.buses_df.at[bus, "y"],
            )
        )
        nearest_bus, dist = geo.find_nearest_bus(point, self.edisgo.topology.buses_df)
        assert nearest_bus == bus
        assert dist == 0.0

        # test with random coordinates
        point = Point((10.002736, 47.5426))
        nearest_bus, dist = geo.find_nearest_bus(point, self.edisgo.topology.buses_df)
        assert nearest_bus == "BranchTee_mvgd_33535_lvgd_1163360000_building_431698"
        assert np.isclose(dist, 0.000806993475812168, atol=1e-6)
