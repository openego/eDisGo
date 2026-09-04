import os

from collections import Counter
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from edisgo.edisgo import EDisGo
from edisgo.io import electromobility_import
from edisgo.tools.geo import mv_grid_gdf


@pytest.fixture
def mock_oep_emobility_query(monkeypatch):
    """Replace one OEP electromobility query with synthetic return data."""

    def install_mock(function_name, return_value):
        query_mock = Mock(return_value=return_value)
        monkeypatch.setattr(electromobility_import, function_name, query_mock)
        return query_mock

    return install_mock


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

    def test_simbev_config_from_oedb_offline(self, mock_oep_emobility_query):
        database_data = pd.DataFrame(
            {
                "scenario": ["eGon2035"],
                "eta_cp": [0.9],
                "stepsize": [15],
                "start_date": [pd.Timestamp("2035-01-01")],
                "end_date": [pd.Timestamp("2035-12-31")],
            }
        )
        query_mock = mock_oep_emobility_query(
            "_query_simbev_config_from_oedb",
            database_data,
        )

        config_df = electromobility_import.simbev_config_from_oedb(
            engine=None,
            scenario="eGon2035",
        )

        assert len(config_df) == 1
        assert config_df["eta_cp"][0] == 0.9
        assert config_df["stepsize"][0] == 15
        assert config_df["days"][0] == 365
        query_mock.assert_called_once_with(scenario="eGon2035", engine=None)

    @pytest.mark.oep
    def test_query_simbev_config_from_oedb_live(self, oep_engine):
        database_data = electromobility_import._query_simbev_config_from_oedb(
            engine=oep_engine,
            scenario="eGon2035",
        )

        assert len(database_data) == 1
        assert {
            "scenario",
            "eta_cp",
            "stepsize",
            "start_date",
            "end_date",
        }.issubset(database_data.columns)

    def test_potential_charging_parks_from_oedb_offline(
        self,
        mock_oep_emobility_query,
    ):
        edisgo_obj = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        grid_gdf = mv_grid_gdf(edisgo_obj)
        point_in_grid = grid_gdf.geometry.iat[0].representative_point()
        database_data = gpd.GeoDataFrame(
            {
                "use_case": ["home", "work"],
                "user_centric_weight": [0.25, 0.75],
                "geom": [point_in_grid, point_in_grid],
            },
            geometry="geom",
            crs=grid_gdf.crs,
            index=pd.Index([1, 2], name="cp_id"),
        )
        query_mock = mock_oep_emobility_query(
            "_query_potential_charging_parks_from_oedb",
            database_data,
        )

        potential_parks_df = electromobility_import.potential_charging_parks_from_oedb(
            edisgo_obj=edisgo_obj,
            engine=None,
        )

        assert len(potential_parks_df) == 2
        assert potential_parks_df.crs == grid_gdf.crs
        assert (potential_parks_df.ags == 0).all()
        assert all(potential_parks_df.geom.iloc[0].within(grid_gdf.geometry))
        assert all(potential_parks_df.geom.iloc[1].within(grid_gdf.geometry))
        query_mock.assert_called_once_with(edisgo_obj=edisgo_obj, engine=None)

    @pytest.mark.oep
    def test_query_potential_charging_parks_from_oedb_live(self, oep_engine):
        edisgo_obj = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        database_data = (
            electromobility_import._query_potential_charging_parks_from_oedb(
                edisgo_obj=edisgo_obj,
                engine=oep_engine,
                limit=2,
            )
        )

        assert 0 < len(database_data) <= 2
        assert {"use_case", "user_centric_weight", "geom"}.issubset(
            database_data.columns
        )
        assert database_data.geom.notna().all()

    def test_charging_processes_from_oedb_offline(
        self,
        mock_oep_emobility_query,
    ):
        edisgo_obj = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        pool = Counter({10: 2, 20: 1})
        database_data = pd.DataFrame(
            {
                "car_id": [10, 10, 20],
                "use_case": ["home", "work", "public"],
                "destination": ["6_home", "0_work", "public"],
                "nominal_charging_capacity_kW": [11.0, 11.0, 22.0],
                "grid_charging_capacity_kW": [11.0, 11.0, 22.0],
                "chargingdemand_kWh": [10.0, 20.0, 30.0],
                "park_start_timesteps": [1, 5, 2],
                "park_end_timesteps": [4, 8, 3],
            }
        )
        query_mock = mock_oep_emobility_query(
            "_query_charging_processes_from_oedb",
            (pool, database_data),
        )

        charging_processes_df = electromobility_import.charging_processes_from_oedb(
            edisgo_obj=edisgo_obj,
            engine=None,
            scenario="eGon2035",
        )

        assert len(charging_processes_df.car_id.unique()) == 3
        assert len(charging_processes_df) == 5
        assert charging_processes_df.car_id.value_counts().to_dict() == {
            0: 2,
            1: 2,
            2: 1,
        }
        assert charging_processes_df[
            charging_processes_df.chargingdemand_kWh == 0
        ].empty
        assert np.isclose(charging_processes_df.chargingdemand_kWh.sum(), 90.0)
        assert charging_processes_df.park_start_timesteps.min() == 0
        assert (
            charging_processes_df.park_time_timesteps
            == charging_processes_df.park_end_timesteps
            - charging_processes_df.park_start_timesteps
            + 1
        ).all()
        query_mock.assert_called_once_with(
            edisgo_obj=edisgo_obj,
            engine=None,
            scenario="eGon2035",
            mode_parking_times="frugal",
        )

    @pytest.mark.oep
    def test_query_charging_processes_from_oedb_live(self, oep_engine):
        edisgo_obj = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )
        pool, database_data = (
            electromobility_import._query_charging_processes_from_oedb(
                edisgo_obj=edisgo_obj,
                engine=oep_engine,
                scenario="eGon2035",
                mode_parking_times="frugal",
                pool_limit=1,
                trip_limit=2,
            )
        )

        assert len(pool) == 1
        assert 0 < len(database_data) <= 2
        assert {
            "car_id",
            "use_case",
            "destination",
            "nominal_charging_capacity_kW",
            "grid_charging_capacity_kW",
            "chargingdemand_kWh",
            "park_start_timesteps",
            "park_end_timesteps",
        }.issubset(database_data.columns)
        assert (database_data.chargingdemand_kWh > 0).all()
