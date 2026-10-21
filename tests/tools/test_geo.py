import numpy as np
import pandas as pd
import pytest

from geopy.distance import geodesic
from shapely.geometry import Point

from edisgo import EDisGo
from edisgo.tools import geo
from edisgo.tools.geo import find_nearest_bus


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


class TestFindNearestBus:
    @classmethod
    def setup_class(cls):
        # three candidate buses around a German grid district
        cls.bus_target = pd.DataFrame(
            {"x": [9.00, 9.05, 9.20], "y": [51.00, 51.02, 51.10]},
            index=["Bus_far", "Bus_near", "Bus_farther"],
        )
        cls.point = Point(9.04, 51.015)

    def test_picks_nearest_and_returns_label(self):
        name, dist = find_nearest_bus(self.point, self.bus_target)

        assert name == "Bus_near"
        assert isinstance(dist, float)

    def test_returned_distance_is_exact_geodesic(self):
        name, dist = find_nearest_bus(self.point, self.bus_target)

        expected = geodesic(
            (self.point.y, self.point.x),
            (self.bus_target.loc[name, "y"], self.bus_target.loc[name, "x"]),
        ).km
        assert dist == pytest.approx(expected)

    def test_single_candidate(self):
        bus_target = pd.DataFrame({"x": [9.1], "y": [51.1]}, index=["Bus_only"])

        name, dist = find_nearest_bus(Point(9.0, 51.0), bus_target)

        assert name == "Bus_only"
        assert dist > 0

    def test_does_not_mutate_input(self):
        bus_target = pd.DataFrame(
            {"x": [9.0, 9.1], "y": [51.0, 51.1]}, index=["a", "b"]
        )
        columns_before = list(bus_target.columns)

        find_nearest_bus(Point(9.05, 51.05), bus_target)

        # the old implementation added a "dist" side-effect column; the
        # vectorised version must leave the input untouched
        assert list(bus_target.columns) == columns_before
