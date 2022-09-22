import os
import shutil

import geopandas as gpd
import pandas as pd
import pytest

from edisgo.edisgo import EDisGo
from edisgo.io.electromobility_import import (
    import_electromobility,
    integrate_charging_parks,
)
from edisgo.network.electromobility import Electromobility


class TestElectromobility:
    @classmethod
    def setup_class(self):
        self.edisgo_obj = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        self.simbev_path = pytest.simbev_example_scenario_path
        self.tracbev_path = pytest.tracbev_example_scenario_path
        import_electromobility(self.edisgo_obj, self.simbev_path, self.tracbev_path)
        integrate_charging_parks(self.edisgo_obj)

    def test_charging_processes_df(self):
        charging_processes_df = self.edisgo_obj.electromobility.charging_processes_df
        assert len(charging_processes_df) == 45
        assert isinstance(charging_processes_df, pd.DataFrame)

    def test_potential_charging_parks_gdf(self):
        potential_charging_parks_gdf = (
            self.edisgo_obj.electromobility.potential_charging_parks_gdf
        )
        assert len(potential_charging_parks_gdf) == 452
        assert isinstance(potential_charging_parks_gdf, gpd.GeoDataFrame)

    def test_simbev_config_df(self):
        simbev_config_df = self.edisgo_obj.electromobility.simbev_config_df
        assert len(simbev_config_df) == 1
        assert isinstance(simbev_config_df, pd.DataFrame)

    def test_integrated_charging_parks_df(self):
        integrated_charging_parks_df = (
            self.edisgo_obj.electromobility.integrated_charging_parks_df
        )
        assert integrated_charging_parks_df.empty
        assert isinstance(integrated_charging_parks_df, pd.DataFrame)

    def test_stepsize(self):
        stepsize = self.edisgo_obj.electromobility.stepsize
        assert stepsize == 15

    def test_simulated_days(self):
        simulated_days = self.edisgo_obj.electromobility.simulated_days
        assert simulated_days == 7

    def test_eta_charging_points(self):
        eta_charging_points = self.edisgo_obj.electromobility.eta_charging_points
        assert eta_charging_points == 0.9

    def test_to_csv(self):
        """Test for method to_csv."""
        dir = os.path.join(os.getcwd(), "electromobility")
        self.edisgo_obj.electromobility.to_csv(dir)

        saved_files = os.listdir(dir)
        assert len(saved_files) == 3
        assert "charging_processes.csv" in saved_files

        shutil.rmtree(dir)

    def test_from_csv(self):
        """
        Test for method from_csv.

        """
        dir = os.path.join(os.getcwd(), "electromobility")
        self.edisgo_obj.electromobility.to_csv(dir)

        # reset self.topology
        self.edisgo_obj.electromobility = Electromobility()

        self.edisgo_obj.electromobility.from_csv(dir, self.edisgo_obj)

        assert len(self.edisgo_obj.electromobility.charging_processes_df) == 45
        assert len(self.edisgo_obj.electromobility.potential_charging_parks_gdf) == 452
        assert self.edisgo_obj.electromobility.integrated_charging_parks_df.empty

        shutil.rmtree(dir)
