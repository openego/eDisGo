import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.tools.pseudo_coordinates import make_pseudo_coordinates


class TestPseudoCoordinates:
    @classmethod
    def setup_class(cls):
        cls.edisgo_root = EDisGo(ding0_grid=pytest.ding0_test_network_path)

    def test_make_pseudo_coordinates(self):
        # test coordinates before
        coordinates = self.edisgo_root.topology.buses_df.loc[
            "Bus_BranchTee_LVGrid_1_9", ["x", "y"]
        ]
        assert round(coordinates[0], 5) != round(7.943307, 5)
        assert round(coordinates[1], 5) != round(48.080396, 5)

        # make pseudo coordinates
        make_pseudo_coordinates(self.edisgo_root, mv_coordinates=True)

        # test if the right coordinates are set for one node
        coordinates = self.edisgo_root.topology.buses_df.loc[
            "Bus_BranchTee_LVGrid_1_9", ["x", "y"]
        ]
        assert round(coordinates[0], 5) == round(7.943307, 5)
        assert round(coordinates[1], 5) == round(48.080396, 5)

        assert not self.edisgo_root.topology.buses_df.x.isin([np.NaN]).any()
