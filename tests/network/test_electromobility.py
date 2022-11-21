import os
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from pandas.util.testing import assert_frame_equal

from edisgo.edisgo import EDisGo
from edisgo.io.electromobility_import import (
    distribute_charging_demand,
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
        distribute_charging_demand(self.edisgo_obj)
        integrate_charging_parks(self.edisgo_obj)

    def test_charging_processes_df(self):
        charging_processes_df = self.edisgo_obj.electromobility.charging_processes_df
        assert len(charging_processes_df) == 48
        assert isinstance(charging_processes_df, pd.DataFrame)
        assert charging_processes_df.at[45, "park_end_timesteps"] == 232
        assert charging_processes_df.at[216, "charging_park_id"] == 1466

    def test_potential_charging_parks_gdf(self):
        potential_charging_parks_gdf = (
            self.edisgo_obj.electromobility.potential_charging_parks_gdf
        )
        assert len(potential_charging_parks_gdf) == 1621
        assert isinstance(potential_charging_parks_gdf, gpd.GeoDataFrame)

    def test_simbev_config_df(self):
        simbev_config_df = self.edisgo_obj.electromobility.simbev_config_df
        assert len(simbev_config_df) == 1
        assert isinstance(simbev_config_df, pd.DataFrame)

    def test_integrated_charging_parks_df(self):
        integrated_charging_parks_df = (
            self.edisgo_obj.electromobility.integrated_charging_parks_df
        )
        assert len(integrated_charging_parks_df) == 3
        assert isinstance(integrated_charging_parks_df, pd.DataFrame)
        assert (
            integrated_charging_parks_df.at[1391, "edisgo_id"]
            == "Charging_Point_LVGrid_131957_public_1"
        )
        assert (
            integrated_charging_parks_df.at[1466, "edisgo_id"]
            == "Charging_Point_LVGrid_362451_public_1"
        )
        assert (
            integrated_charging_parks_df.at[1602, "edisgo_id"]
            == "Charging_Point_LVGrid_136124_work_1"
        )

    def test_stepsize(self):
        stepsize = self.edisgo_obj.electromobility.stepsize
        assert stepsize == 15

    def test_simulated_days(self):
        simulated_days = self.edisgo_obj.electromobility.simulated_days
        assert simulated_days == 7

    def test_eta_charging_points(self):
        eta_charging_points = self.edisgo_obj.electromobility.eta_charging_points
        assert eta_charging_points == 0.9

    def test_get_flexibility_bands(self):
        self.edisgo_obj.electromobility.get_flexibility_bands(
            self.edisgo_obj, ["work", "public"]
        )

        integrated_charging_parks = (
            self.edisgo_obj.electromobility.integrated_charging_parks_df
        )
        charging_processes = self.edisgo_obj.electromobility.charging_processes_df
        flex_bands = self.edisgo_obj.electromobility.flexibility_bands

        # check concrete values
        cp = "Charging_Point_LVGrid_131957_public_1"
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[76:108, 0].values, 0.0122222
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[0:76, 0].values, 0.0
        ).all()

        tmp = flex_bands["upper_energy"].loc[:, [cp]]
        assert np.isclose(tmp.iloc[648, 0] + 11 / 4 / 1000, tmp.iloc[649, 0])
        assert np.isclose(tmp.iloc[654, 0], tmp.iloc[655, 0])

        tmp = flex_bands["lower_energy"].loc[:, [cp]]
        assert np.isclose(tmp.iloc[648, 0], tmp.iloc[649, 0])
        assert np.isclose(tmp.iloc[654, 0] + 11 / 4 / 1000, tmp.iloc[655, 0])

        # check charging demand
        for cp in [
            "Charging_Point_LVGrid_131957_public_1",
            "Charging_Point_LVGrid_362451_public_1",
            "Charging_Point_LVGrid_136124_work_1",
        ]:

            charging_park_id = integrated_charging_parks.loc[
                integrated_charging_parks.edisgo_id == cp
            ].index
            charging_processes_cp = charging_processes.loc[
                charging_processes.charging_park_id.isin(charging_park_id)
            ]
            assert np.isclose(
                charging_processes_cp.loc[:, "chargingdemand_kWh"].sum() / 1e3,
                flex_bands["upper_energy"].loc[:, [cp]].iloc[-1, 0],
            )
            assert np.isclose(
                charging_processes_cp.loc[:, "chargingdemand_kWh"].sum() / 1e3,
                flex_bands["lower_energy"].loc[:, [cp]].iloc[-1, 0],
            )

    def test_resample(self):
        """
        Checks resampling function with flexibility bands determined using standing
        times.

        """
        self.edisgo_obj.electromobility.get_flexibility_bands(
            self.edisgo_obj, ["work", "public"]
        )
        flex_bands_orig = self.edisgo_obj.electromobility.flexibility_bands.copy()

        # ##################### check down-sampling #####################
        self.edisgo_obj.electromobility.resample(freq="1h")

        # check that integrity check does not fail
        self.edisgo_obj.electromobility.check_integrity()

        # check shape
        flex_bands = self.edisgo_obj.electromobility.flexibility_bands
        assert (
            len(flex_bands["upper_energy"].index)
            == len(flex_bands_orig["upper_energy"].index) / 4
        )
        assert len(flex_bands["upper_energy"].columns) == len(
            flex_bands_orig["upper_energy"].columns
        )

        # check concrete values
        cp = "Charging_Point_LVGrid_131957_public_1"

        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[19:27, 0].values, 0.0122222
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[0:19, 0].values, 0.0
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[63, 0], 0.0122222 * 3 / 4
        )

        assert np.isclose(
            flex_bands["upper_energy"].loc[:, [cp]].iloc[19, 0],
            flex_bands_orig["upper_energy"].loc[:, [cp]].iloc[76:80, 0].max(),
        )
        assert np.isclose(
            flex_bands["lower_energy"].loc[:, [cp]].iloc[26, 0],
            flex_bands_orig["lower_energy"].loc[:, [cp]].iloc[104:108, 0].max(),
        )

        # ##################### check up-sampling #####################
        self.edisgo_obj.electromobility.resample(freq="15min")

        # check that integrity check does not fail
        self.edisgo_obj.electromobility.check_integrity()

        # check index and columns
        flex_bands = self.edisgo_obj.electromobility.flexibility_bands
        assert (
            flex_bands["upper_energy"].index == flex_bands_orig["upper_energy"].index
        ).all()
        assert (
            flex_bands["upper_energy"].columns
            == flex_bands_orig["upper_energy"].columns
        ).all()

        # check concrete values
        cp = "Charging_Point_LVGrid_131957_public_1"
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[76:108, 0].values, 0.0122222
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[0:76, 0].values, 0.0
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[252:260, 0], 0.0122222 * 3 / 4
        ).all()

        assert (
            flex_bands["upper_energy"].loc[:, [cp]].iloc[3::4, 0]
            == flex_bands_orig["upper_energy"].loc[:, [cp]].iloc[3::4, 0]
        ).all()
        assert (
            flex_bands["lower_energy"].loc[:, [cp]].iloc[3::4, 0]
            == flex_bands_orig["lower_energy"].loc[:, [cp]].iloc[3::4, 0]
        ).all()

    def test_resample_2(self):
        """
        Checks resampling function with set up flexibility bands.

        """
        # CP1 - charge 12 kWh between time steps [1, 4]
        # CP2 - charge 2 kWh between time steps [0, 1] and 2 kWh between
        #       time steps [4, 5]

        # set charging efficiency to 1 to make things easier
        self.edisgo_obj.electromobility.simbev_config_df.at[0, "eta_cp"] = 1.0

        # set up flexibility bands
        timeindex = pd.date_range("1/1/1970", periods=6, freq="30min")
        flex_bands = {}
        flex_bands["upper_power"] = pd.DataFrame(
            data={
                "CP1": [0.0, 12.0, 12.0, 12.0, 12.0, 0.0],
                "CP2": [3.0, 3.0, 0.0, 0.0, 3.0, 3.0],
            },
            index=timeindex,
        )
        flex_bands["upper_energy"] = pd.DataFrame(
            data={
                "CP1": [0.0, 6.0, 12.0, 12.0, 12.0, 12.0],
                "CP2": [1.5, 2.0, 2.0, 2.0, 3.5, 4.0],
            },
            index=timeindex,
        )
        flex_bands["lower_energy"] = pd.DataFrame(
            data={
                "CP1": [0.0, 0.0, 0.0, 6.0, 12.0, 12.0],
                "CP2": [0.5, 2.0, 2.0, 2.0, 2.5, 4.0],
            },
            index=timeindex,
        )

        self.edisgo_obj.electromobility.flexibility_bands = flex_bands
        self.edisgo_obj.electromobility.check_integrity()

        # ##################### check up-sampling ####################

        self.edisgo_obj.electromobility.resample(freq="15min")

        # check that integrity check does not fail
        self.edisgo_obj.electromobility.check_integrity()

        # check concrete values
        timeindex = pd.date_range("1/1/1970", periods=12, freq="15min")
        flex_bands_checking = {}
        flex_bands_checking["upper_power"] = pd.DataFrame(
            data={
                "CP1": [
                    0.0,
                    0.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    0.0,
                    0.0,
                ],
                "CP2": [3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0],
            },
            index=timeindex,
        )
        flex_bands_checking["upper_energy"] = pd.DataFrame(
            data={
                "CP1": [
                    0.0,
                    0.0,
                    3.0,
                    6.0,
                    9.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                ],
                "CP2": [0.75, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.75, 3.5, 3.75, 4.0],
            },
            index=timeindex,
        )
        flex_bands_checking["lower_energy"] = pd.DataFrame(
            data={
                "CP1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 12.0, 12.0, 12.0],
                "CP2": [0.25, 0.5, 1.25, 2.0, 2.0, 2.0, 2.0, 2.0, 2.25, 2.5, 3.25, 4.0],
            },
            index=timeindex,
        )

        flex_bands_new = self.edisgo_obj.electromobility.flexibility_bands
        for band in flex_bands_checking.keys():
            assert_frame_equal(flex_bands_checking[band], flex_bands_new[band])

        # check resampling to 2 hours
        self.edisgo_obj.electromobility.resample(freq="2H")
        # check that integrity check does not fail
        self.edisgo_obj.electromobility.check_integrity()
        # check shape and no NaN values
        flex_bands_new = self.edisgo_obj.electromobility.flexibility_bands
        for band in flex_bands_new.keys():
            assert len(flex_bands_new[band]) == 2
            assert (flex_bands[band].columns == flex_bands_new[band].columns).all()
            assert not flex_bands_new[band].isna().any().any()

        # check resampling to uneven amount of times new index fits into old index
        self.edisgo_obj.electromobility.resample(freq="30min")
        # check that integrity check does not fail
        self.edisgo_obj.electromobility.check_integrity()
        # check shape and no NaN values
        flex_bands_new = self.edisgo_obj.electromobility.flexibility_bands
        for band in flex_bands_new.keys():
            assert len(flex_bands_new[band]) == 8
            assert (flex_bands[band].columns == flex_bands_new[band].columns).all()
            assert not flex_bands_new[band].isna().any().any()

    def test_to_csv(self):
        """Test for method to_csv."""
        dir = os.path.join(os.getcwd(), "electromobility")
        timeindex = pd.date_range("1/1/1970", periods=2, freq="H")
        flex_bands = {
            "upper_energy": pd.DataFrame({"cp_1": [1, 2]}, index=timeindex),
            "upper_power": pd.DataFrame({"cp_1": [1, 2]}, index=timeindex),
            "lower_energy": pd.DataFrame({"cp_1": [1, 2]}, index=timeindex),
        }
        self.edisgo_obj.electromobility.flexibility_bands = flex_bands

        # ############ test with default values #####################
        self.edisgo_obj.electromobility.to_csv(dir)

        saved_files = os.listdir(dir)
        assert len(saved_files) == 7
        assert "charging_processes.csv" in saved_files
        assert "flexibility_band_upper_power.csv" in saved_files

        shutil.rmtree(dir)

        # ############ test specifying attributes #####################
        self.edisgo_obj.electromobility.to_csv(
            dir, attributes=["potential_charging_parks_gdf"]
        )

        saved_files = os.listdir(dir)
        assert len(saved_files) == 1
        assert "potential_charging_parks.csv" in saved_files

        shutil.rmtree(dir)

    def test_from_csv(self):
        """
        Test for method from_csv.

        """
        dir = os.path.join(os.getcwd(), "electromobility")
        timeindex = pd.date_range("1/1/1970", periods=2, freq="H")
        flex_bands = {
            "upper_energy": pd.DataFrame({"cp_1": [1, 2]}, index=timeindex),
            "upper_power": pd.DataFrame({"cp_1": [1, 2]}, index=timeindex),
            "lower_energy": pd.DataFrame({"cp_1": [1, 2]}, index=timeindex),
        }
        self.edisgo_obj.electromobility.flexibility_bands = flex_bands
        self.edisgo_obj.electromobility.to_csv(dir)

        # reset Electromobility
        self.edisgo_obj.electromobility = Electromobility()

        self.edisgo_obj.electromobility.from_csv(dir, self.edisgo_obj)

        assert len(self.edisgo_obj.electromobility.charging_processes_df) == 48
        assert len(self.edisgo_obj.electromobility.potential_charging_parks_gdf) == 1621
        assert len(self.edisgo_obj.electromobility.integrated_charging_parks_df) == 3

        assert_frame_equal(
            self.edisgo_obj.electromobility.flexibility_bands["upper_energy"],
            flex_bands["upper_energy"],
            check_freq=False,
        )
        assert_frame_equal(
            self.edisgo_obj.electromobility.flexibility_bands["lower_energy"],
            flex_bands["lower_energy"],
            check_freq=False,
        )
        assert_frame_equal(
            self.edisgo_obj.electromobility.flexibility_bands["upper_power"],
            flex_bands["upper_power"],
            check_freq=False,
        )

        shutil.rmtree(dir)
