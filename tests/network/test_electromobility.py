import os
import pytest
import pandas as pd

from edisgo.edisgo import import_edisgo_from_files
from edisgo.io.electromobility_import import import_simbev_electromobility, distribute_charging_demand
from edisgo.flex_opt.charging_strategies import integrate_charging_points, charging_strategy


class TestElectromobility:

    @classmethod
    def setup_class(self):
        self.ding0_path = pytest.ding0_test_network_3_path
        self.simbev_path = pytest.simbev_test_results_path
        self.standing_times_path = os.path.join(self.simbev_path, "simbev_run")
        self.charging_strategies = ["dumb", "reduced", "residual"]

        self.edisgo_obj = import_edisgo_from_files(
            self.ding0_path, import_topology=True, import_timeseries=True)

    def test_import_simbev_electromobility(self):

        import_simbev_electromobility(pytest.simbev_test_results_path, self.edisgo_obj)

        electromobility = self.edisgo_obj.electromobility

        # The number of files should be the same as the maximum car id + 1 (starts with zero)
        files = 0

        for dirpath, dirnames, filenames in os.walk(self.standing_times_path):
                files += len([f for f in filenames if f.endswith(".csv")])

        assert electromobility.charging_processes_df.car_id.max() == files - 1
        assert isinstance(electromobility.eta_charging_points, float)
        assert isinstance(electromobility.simulated_days, int)
        assert isinstance(electromobility.stepsize, int)
        assert len(electromobility.grid_connections_gdf.columns) == 4
        # There should be as many grid connections as potential charging parks
        assert len(electromobility.grid_connections_gdf) == len(list(electromobility.potential_charging_parks))

    def test_distribute_charging_demand(self):

        distribute_charging_demand(self.edisgo_obj)

        electromobility = self.edisgo_obj.electromobility

        # all charging point use cases have designated charging points
        assert all(
            ~electromobility.designated_charging_points_dfs[val].empty for val in ["home", "work", "public", "hpc"])

        designated_charging_parks_with_charging_points = len(
            [cp for cp in list(
                electromobility.potential_charging_parks) if cp.designated_charging_point_capacity > 0])

        designated_charging_parks_with_charging_points_in_dfs = 0

        for key, val in electromobility.designated_charging_points_dfs.items():
            designated_charging_parks_with_charging_points_in_dfs += len(val.grid_connection_point_id.unique())

        assert designated_charging_parks_with_charging_points == designated_charging_parks_with_charging_points_in_dfs

        total_charging_demand_at_charging_parks = sum([cp.charging_processes_df.chargingdemand.sum() for cp in list(
            electromobility.potential_charging_parks) if cp.designated_charging_point_capacity > 0])

        total_charging_demand = electromobility.charging_processes_df.chargingdemand.sum()

        assert round(total_charging_demand_at_charging_parks, 0) == round(total_charging_demand, 0)

    def test_integrate_charging_points(self):

        integrate_charging_points(self.edisgo_obj)

        electromobility = self.edisgo_obj.electromobility

        ts = self.edisgo_obj.timeseries

        topology = self.edisgo_obj.topology

        designated_charging_parks_with_charging_points = len(
            [cp for cp in list(
                electromobility.potential_charging_parks) if cp.designated_charging_point_capacity > 0])

        integrated_charging_parks = [cp for cp in list(
            electromobility.potential_charging_parks) if cp.grid is not None]

        assert designated_charging_parks_with_charging_points == len(integrated_charging_parks)
        assert len(integrated_charging_parks) == len(ts.charging_points_active_power.columns)
        assert len(integrated_charging_parks) == len(ts.charging_points_reactive_power.columns)

        edisgo_ids_cp = sorted(cp.edisgo_id for cp in integrated_charging_parks)
        edisgo_ids_ts = sorted(ts.charging_points_active_power.columns.tolist())
        edisgo_ids_topology = sorted(topology.charging_points_df.index.tolist())

        assert edisgo_ids_cp == edisgo_ids_ts == edisgo_ids_topology

    def test_charging_strategy(self):
        charging_demand_lst = []

        for count, strategy in enumerate(self.charging_strategies):
            charging_strategy(self.edisgo_obj, strategy=strategy)

            electromobility = self.edisgo_obj.electromobility

            ts = self.edisgo_obj.timeseries

            # Check if all charging points have a valid chargingdemand > 0
            df = ts.charging_points_active_power.loc[:, (ts.charging_points_active_power <= 0).any(axis=0)]

            assert df.shape == ts.charging_points_active_power.shape

            charging_demand_lst.append(ts.charging_points_active_power.sum())

        # the chargingdemand per charging point and therefore in total should always be the same
        assert all((_.round(4) == charging_demand_lst[0].round(4)).all() for _ in charging_demand_lst)
