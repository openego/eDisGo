import pandas as pd
import pytest

from geopy.distance import geodesic
from shapely.geometry import Point

from edisgo.tools.geo import find_nearest_bus


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
