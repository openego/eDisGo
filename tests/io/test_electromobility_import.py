import os

import pandas as pd
import pytest

from edisgo.edisgo import EDisGo
from edisgo.io.electromobility_import import (
    distribute_charging_demand,
    import_electromobility,
    integrate_charging_parks,
)


class TestElectromobilityImport:
    """
    Tests all functions in electromobility_import.py.

    """

    @classmethod
    def setup_class(cls):
        cls.ding0_path = pytest.ding0_test_network_2_path
        cls.simbev_path = pytest.simbev_example_scenario_path
        cls.tracbev_path = pytest.tracbev_example_scenario_path
        cls.standing_times_path = cls.simbev_path
        cls.charging_strategies = ["dumb", "reduced", "residual"]

        cls.edisgo_obj = EDisGo(ding0_grid=cls.ding0_path)
        timeindex = pd.date_range("1/1/2011", periods=24 * 7, freq="H")
        cls.edisgo_obj.set_timeindex(timeindex)

        cls.edisgo_obj.resample_timeseries()

    def test_import_electromobility(self):

        import_electromobility(self.edisgo_obj, self.simbev_path, self.tracbev_path)

        electromobility = self.edisgo_obj.electromobility

        # The number of files should be the same as the maximum car id + 1 (starts with
        # zero)
        files = 0

        for dirpath, dirnames, filenames in os.walk(self.standing_times_path):
            files += len([f for f in filenames if f.endswith(".csv")])

        assert electromobility.charging_processes_df.car_id.max() == files - 1
        assert isinstance(electromobility.eta_charging_points, float)
        assert isinstance(electromobility.simulated_days, int)
        assert isinstance(electromobility.stepsize, int)
        assert len(electromobility.potential_charging_parks_gdf.columns) == 4
        # There should be as many potential charging parks in the DataFrame as in the
        # generator object
        assert len(electromobility.potential_charging_parks_gdf) == len(
            list(electromobility.potential_charging_parks)
        )

    def test_distribute_charging_demand(self):

        # test user friendly
        distribute_charging_demand(self.edisgo_obj)

        electromobility = self.edisgo_obj.electromobility

        total_charging_demand_at_charging_parks = sum(
            cp.charging_processes_df.chargingdemand_kWh.sum()
            for cp in list(electromobility.potential_charging_parks)
            if cp.designated_charging_point_capacity > 0
        )

        total_charging_demand = (
            electromobility.charging_processes_df.chargingdemand_kWh.sum()
        )

        assert round(total_charging_demand_at_charging_parks, 0) == round(
            total_charging_demand, 0
        )

        # test grid friendly
        self.edisgo_obj = EDisGo(ding0_grid=self.ding0_path)
        timeindex = pd.date_range("1/1/2011", periods=24 * 7, freq="H")
        self.edisgo_obj.set_timeindex(timeindex)
        self.edisgo_obj.resample_timeseries()

        import_electromobility(self.edisgo_obj, self.simbev_path, self.tracbev_path)
        distribute_charging_demand(self.edisgo_obj, mode="grid_friendly")

        electromobility = self.edisgo_obj.electromobility

        total_charging_demand_at_charging_parks = sum(
            cp.charging_processes_df.chargingdemand_kWh.sum()
            for cp in list(electromobility.potential_charging_parks)
            if cp.designated_charging_point_capacity > 0
        )

        total_charging_demand = (
            electromobility.charging_processes_df.chargingdemand_kWh.sum()
        )

        assert round(total_charging_demand_at_charging_parks, 0) == round(
            total_charging_demand, 0
        )

        # test weight factors
        self.edisgo_obj = EDisGo(ding0_grid=self.ding0_path)
        timeindex = pd.date_range("1/1/2011", periods=24 * 7, freq="H")
        self.edisgo_obj.set_timeindex(timeindex)
        self.edisgo_obj.resample_timeseries()

        import_electromobility(self.edisgo_obj, self.simbev_path, self.tracbev_path)
        distribute_charging_demand(
            self.edisgo_obj,
            generators_weight_factor=1 / 3,
            distance_weight=0.5,
            user_friendly_weight=1 / 3,
        )

        electromobility = self.edisgo_obj.electromobility

        total_charging_demand_at_charging_parks = sum(
            cp.charging_processes_df.chargingdemand_kWh.sum()
            for cp in list(electromobility.potential_charging_parks)
            if cp.designated_charging_point_capacity > 0
        )

        total_charging_demand = (
            electromobility.charging_processes_df.chargingdemand_kWh.sum()
        )

        assert round(total_charging_demand_at_charging_parks, 0) == round(
            total_charging_demand, 0
        )

    def test_integrate_charging_parks(self):

        integrate_charging_parks(self.edisgo_obj)

        electromobility = self.edisgo_obj.electromobility

        topology = self.edisgo_obj.topology

        designated_charging_parks_with_charging_points = len(
            [
                cp
                for cp in list(electromobility.potential_charging_parks)
                if cp.designated_charging_point_capacity > 0 and cp.within_grid
            ]
        )

        integrated_charging_parks = [
            cp
            for cp in list(electromobility.potential_charging_parks)
            if cp.grid is not None
        ]

        assert (
            designated_charging_parks_with_charging_points
            == len(integrated_charging_parks)
            == len(electromobility.integrated_charging_parks_df)
        )

        edisgo_ids_cp = sorted(cp.edisgo_id for cp in integrated_charging_parks)
        edisgo_ids_topology = sorted(topology.charging_points_df.index.tolist())

        assert edisgo_ids_cp == edisgo_ids_topology
