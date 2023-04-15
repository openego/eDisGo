import os

import numpy as np
import pandas as pd
import pytest

from edisgo.edisgo import EDisGo
from edisgo.io import electromobility_import
from edisgo.tools.geo import mv_grid_gdf


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

        electromobility_import.import_electromobility_from_dir(
            self.edisgo_obj, self.simbev_path, self.tracbev_path
        )

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

    def test_assure_minimum_potential_charging_parks(self):
        self.edisgo_obj.electromobility.charging_processes_df = (
            electromobility_import.read_csvs_charging_processes(self.simbev_path)
        )
        pot_cp_gdf_raw = electromobility_import.read_gpkg_potential_charging_parks(
            self.tracbev_path, self.edisgo_obj
        )

        # manipulate data in order to catch every case handled in assure_minimum...
        # drop hpc charging point to have no hpc points available
        hpc_points = pot_cp_gdf_raw[pot_cp_gdf_raw.use_case == "hpc"].index
        pot_cp_gdf_raw = pot_cp_gdf_raw.drop(hpc_points)
        # drop all but one work charging point
        work_points = pot_cp_gdf_raw[pot_cp_gdf_raw.use_case == "work"].index
        pot_cp_gdf_raw = pot_cp_gdf_raw.drop(work_points[1:])

        pot_cp_gdf = electromobility_import.assure_minimum_potential_charging_parks(
            self.edisgo_obj, pot_cp_gdf_raw, gc_to_car_rate_work=0.3
        )

        assert len(pot_cp_gdf_raw) < len(pot_cp_gdf)
        assert len(pot_cp_gdf[pot_cp_gdf.use_case == "hpc"]) == 32
        assert len(pot_cp_gdf[pot_cp_gdf.use_case == "work"]) == 4

    def test_distribute_charging_demand(self):

        # test user friendly
        electromobility_import.distribute_charging_demand(self.edisgo_obj)

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

        electromobility_import.import_electromobility_from_dir(
            self.edisgo_obj, self.simbev_path, self.tracbev_path
        )
        electromobility_import.distribute_charging_demand(
            self.edisgo_obj, mode="grid_friendly"
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

        # test weight factors
        self.edisgo_obj = EDisGo(ding0_grid=self.ding0_path)
        timeindex = pd.date_range("1/1/2011", periods=24 * 7, freq="H")
        self.edisgo_obj.set_timeindex(timeindex)
        self.edisgo_obj.resample_timeseries()

        electromobility_import.import_electromobility_from_dir(
            self.edisgo_obj, self.simbev_path, self.tracbev_path
        )
        electromobility_import.distribute_charging_demand(
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

        electromobility_import.integrate_charging_parks(self.edisgo_obj)

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

    @pytest.mark.local
    def test_simbev_config_from_oedb(self):
        config_df = electromobility_import.simbev_config_from_oedb(
            engine=pytest.engine, scenario="eGon2035"
        )
        assert len(config_df) == 1
        assert config_df["eta_cp"][0] == 0.9
        assert config_df["stepsize"][0] == 15
        assert config_df["days"][0] == 365

    @pytest.mark.local
    def test_potential_charging_parks_from_oedb(self):
        edisgo_obj = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        potential_parks_df = electromobility_import.potential_charging_parks_from_oedb(
            edisgo_obj=edisgo_obj, engine=pytest.engine
        )
        assert len(potential_parks_df) == 1083
        # check for random charging points if they are within MV grid district
        grid_gdf = mv_grid_gdf(edisgo_obj)
        assert all(potential_parks_df.geom.iloc[10].within(grid_gdf.geometry))
        assert all(potential_parks_df.geom.iloc[100].within(grid_gdf.geometry))

    @pytest.mark.local
    def test_charging_processes_from_oedb(self):
        edisgo_obj = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        charging_processes_df = electromobility_import.charging_processes_from_oedb(
            edisgo_obj=edisgo_obj, engine=pytest.engine, scenario="eGon2035"
        )
        assert len(charging_processes_df.car_id.unique()) == 1604
        assert len(charging_processes_df) == 324117
        assert charging_processes_df[
            charging_processes_df.chargingdemand_kWh == 0
        ].empty
        assert np.isclose(
            charging_processes_df.chargingdemand_kWh.sum() / 1604, 2414.55, atol=1e-3
        )
