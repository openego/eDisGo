import os

import pytest

from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.charging_strategies import charging_strategy
from edisgo.io.electromobility_import import (
    distribute_charging_demand,
    import_electromobility,
    integrate_charging_parks,
)


class TestElectromobility:
    @classmethod
    def setup_class(self):
        self.ding0_path = pytest.ding0_test_network_3_path
        self.simbev_path = pytest.simbev_example_scenario_path
        self.tracbev_path = pytest.tracbev_example_scenario_path
        self.standing_times_path = os.path.join(self.simbev_path, "simbev_run")
        self.charging_strategies = ["dumb", "reduced", "residual"]

        self.edisgo_obj = import_edisgo_from_files(
            self.ding0_path, import_topology=True, import_timeseries=True
        )

    def test_import_simbev_electromobility(self):

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
        self.edisgo_obj = import_edisgo_from_files(
            self.ding0_path, import_topology=True, import_timeseries=True
        )
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

    def test_integrate_charging_parks(self):

        integrate_charging_parks(self.edisgo_obj)

        electromobility = self.edisgo_obj.electromobility

        ts = self.edisgo_obj.timeseries

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

    def test_charging_strategy(self):
        charging_demand_lst = []

        for count, strategy in enumerate(self.charging_strategies):
            charging_strategy(self.edisgo_obj, strategy=strategy)

            ts = self.edisgo_obj.timeseries

            # Check if all charging points have a valid chargingdemand_kWh > 0
            df = ts.charging_points_active_power(self.edisgo_obj).loc[
                :, (ts.charging_points_active_power(self.edisgo_obj) <= 0).any(axis=0)
            ]

            assert df.shape == ts.charging_points_active_power(self.edisgo_obj).shape

            charging_demand_lst.append(
                ts.charging_points_active_power(self.edisgo_obj).sum()
            )

        # the chargingdemand_kWh per charging point and therefore in total should
        # always be the same
        assert all(
            (_.round(4) == charging_demand_lst[0].round(4)).all()
            for _ in charging_demand_lst
        )
