import logging
import os
import re
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from edisgo.edisgo import EDisGo
from edisgo.io import electromobility_import
from edisgo.network.electromobility import Electromobility
from edisgo.network.timeseries import TimeSeries


class TestElectromobility:
    @classmethod
    def setup_class(self):
        self.edisgo_obj = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        self.simbev_path = pytest.simbev_example_scenario_path
        self.tracbev_path = pytest.tracbev_example_scenario_path
        electromobility_import.import_electromobility_from_dir(
            self.edisgo_obj, self.simbev_path, self.tracbev_path
        )
        electromobility_import.distribute_charging_demand(self.edisgo_obj)
        electromobility_import.integrate_charging_parks(self.edisgo_obj)

    def setup_simple_flex_band_data(self):
        """
        Sets up flex bands for testing.

        """
        # CP1 - charge 12 kWh between time steps [1, 4]
        # CP2 - charge 2 kWh between time steps [0, 1] and 2 kWh between
        #       time steps [4, 5]
        # CP3 - charge 12 kWh between time steps [1, 4] with offsets (energy bands
        #       starting at 12 kWh)
        # CP4 - charge 2 kWh between time steps [0, 1] and 2 kWh between
        #       time steps [4, 5] with different offsets for lower and upper energy
        #       band
        timeindex = pd.date_range("1/1/1970", periods=6, freq="30min")
        flex_bands = {}
        flex_bands["upper_power"] = pd.DataFrame(
            data={
                "CP1": [0.0, 12.0, 12.0, 12.0, 12.0, 0.0],
                "CP2": [3.0, 3.0, 0.0, 0.0, 3.0, 3.0],
                "CP3": [0.0, 12.0, 12.0, 12.0, 12.0, 0.0],
                "CP4": [3.0, 3.0, 0.0, 0.0, 3.0, 3.0],
            },
            index=timeindex,
        )
        flex_bands["upper_energy"] = pd.DataFrame(
            data={
                "CP1": [0.0, 6.0, 12.0, 12.0, 12.0, 12.0],
                "CP2": [1.5, 2.0, 2.0, 2.0, 3.5, 4.0],
                "CP3": [12.0, 18.0, 24.0, 24.0, 24.0, 24.0],
                "CP4": [4.0, 4.0, 4.0, 4.0, 5.5, 6.0],
            },
            index=timeindex,
        )
        flex_bands["lower_energy"] = pd.DataFrame(
            data={
                "CP1": [0.0, 0.0, 0.0, 6.0, 12.0, 12.0],
                "CP2": [0.5, 2.0, 2.0, 2.0, 2.5, 4.0],
                "CP3": [12.0, 12.0, 12.0, 18.0, 24.0, 24.0],
                "CP4": [2.5, 4.0, 4.0, 4.0, 4.5, 6.0],
            },
            index=timeindex,
        )
        return flex_bands

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
            flex_bands["upper_power"].loc[:, [cp]].iloc[76:108, 0].values, 0.0122232
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[0:76, 0].values, 1e-6
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

        # ############# check with automatic resampling of flex bands ###############
        self.edisgo_obj.set_time_series_worst_case_analysis()
        self.edisgo_obj.electromobility.get_flexibility_bands(
            self.edisgo_obj, ["work", "public"]
        )
        # check that integrity check does not fail
        self.edisgo_obj.electromobility.check_integrity()

        # check frequency
        flex_bands_index = self.edisgo_obj.electromobility.flexibility_bands[
            "upper_energy"
        ].index
        assert (flex_bands_index[1] - flex_bands_index[0]) == pd.Timedelta("1H")

    def test_fix_flexibility_bands_rounding_errors(self, caplog):
        # set up test data
        # set charging efficiency to 1 to make things easier
        self.edisgo_obj.electromobility.simbev_config_df.at[0, "eta_cp"] = 1.0

        # set up flexibility bands
        timeindex = pd.date_range("1/1/1970", periods=6, freq="30min")
        flex_bands = self.setup_simple_flex_band_data()
        self.edisgo_obj.electromobility.flexibility_bands = flex_bands

        # test upper power too low to reach upper energy
        flex_bands["upper_power"].at[timeindex[4], "CP2"] = 3.0 - 1e-6
        with caplog.at_level(logging.DEBUG):
            self.edisgo_obj.electromobility.fix_flexibility_bands_rounding_errors()
        assert len(caplog.messages) == 1
        assert (
            "There are cases when upper power is not sufficient to meet charged "
            "upper energy." in caplog.text
        )
        assert np.isclose(
            flex_bands["upper_power"].at[timeindex[4], "CP2"], 3.0, atol=1e-7
        )

        # test that debug message is not raised again
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            self.edisgo_obj.electromobility.fix_flexibility_bands_rounding_errors()
        assert len(caplog.messages) == 0

        # test reduce lower energy band when it is above upper energy band -> this
        # also results in upper power being too low to meet charged lower energy
        flex_bands["lower_energy"].at[timeindex[4], "CP1"] = 12.0 + 5e-7
        with caplog.at_level(logging.DEBUG):
            self.edisgo_obj.electromobility.fix_flexibility_bands_rounding_errors()
        assert len(caplog.messages) == 2
        assert (
            "There are cases when lower energy band is larger than upper energy "
            "band." in caplog.text
        )
        assert (
            "There are cases when upper power is not sufficient to meet charged "
            "lower energy." in caplog.text
        )
        assert np.isclose(
            flex_bands["lower_energy"].at[timeindex[4], "CP1"],
            12.0 + 5e-7 - 1e-6,
            atol=1e-7,
        )
        assert np.isclose(
            flex_bands["upper_power"].at[timeindex[4], "CP1"], 12.0 + 2e-6, atol=1e-7
        )

        # test that debug message is not raised again
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            self.edisgo_obj.electromobility.fix_flexibility_bands_rounding_errors()
        assert len(caplog.messages) == 0

        # reset charging efficiency
        self.edisgo_obj.electromobility.simbev_config_df.at[0, "eta_cp"] = 0.9

    def test_resample(self):
        """
        Checks resampling function with flexibility bands determined using standing
        times.

        """
        # reset Timeseries object to avoid automatic resampling of flex bands
        self.edisgo_obj.timeseries = TimeSeries()
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
            flex_bands["upper_power"].loc[:, [cp]].iloc[19:27, 0].values, 0.0122232
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[0:19, 0].values, 1e-6
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[63, 0],
            0.0122232 * 3 / 4,
            atol=1e-6,
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
            flex_bands["upper_power"].loc[:, [cp]].iloc[76:108, 0].values, 0.0122232
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[0:76, 0].values, 1e-6
        ).all()
        assert np.isclose(
            flex_bands["upper_power"].loc[:, [cp]].iloc[252:260, 0],
            0.0122222 * 3 / 4,
            atol=1e-6,
        ).all()

        assert (
            flex_bands["upper_energy"].loc[:, [cp]].iloc[3::4, 0]
            == flex_bands_orig["upper_energy"].loc[:, [cp]].iloc[3::4, 0]
        ).all()
        assert (
            flex_bands["lower_energy"].loc[:, [cp]].iloc[3::4, 0]
            == flex_bands_orig["lower_energy"].loc[:, [cp]].iloc[3::4, 0]
        ).all()

    def test_resample_2(self, caplog):
        """
        Checks resampling function with set up flexibility bands.

        """
        # set charging efficiency to 1 to make things easier
        self.edisgo_obj.electromobility.simbev_config_df.at[0, "eta_cp"] = 1.0

        # set up flexibility bands
        flex_bands = self.setup_simple_flex_band_data()
        self.edisgo_obj.electromobility.flexibility_bands = flex_bands

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
                "CP3": [
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
                "CP4": [3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0],
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
                "CP3": [
                    12.0,
                    12.0,
                    15.0,
                    18.0,
                    21.0,
                    24.0,
                    24.0,
                    24.0,
                    24.0,
                    24.0,
                    24.0,
                    24.0,
                ],
                "CP4": [3.25, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.75, 5.5, 5.75, 6.0],
            },
            index=timeindex,
        )
        flex_bands_checking["lower_energy"] = pd.DataFrame(
            data={
                "CP1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 12.0, 12.0, 12.0],
                "CP2": [0.25, 0.5, 1.25, 2.0, 2.0, 2.0, 2.0, 2.0, 2.25, 2.5, 3.25, 4.0],
                "CP3": [
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    15.0,
                    18.0,
                    21.0,
                    24.0,
                    24.0,
                    24.0,
                ],
                "CP4": [1.75, 2.5, 3.25, 4.0, 4.0, 4.0, 4.0, 4.0, 4.25, 4.5, 5.25, 6.0],
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

        # check resampling back to 30 minutes
        self.edisgo_obj.electromobility.resample(freq="30min")
        # check that integrity check does not fail
        self.edisgo_obj.electromobility.check_integrity()
        # check shape and no NaN values
        flex_bands_new = self.edisgo_obj.electromobility.flexibility_bands
        for band in flex_bands_new.keys():
            assert len(flex_bands_new[band]) == 8
            assert (flex_bands[band].columns == flex_bands_new[band].columns).all()
            assert not flex_bands_new[band].isna().any().any()

        # check that resampling to an uneven number of times new index fits into old
        # index leads to error raising
        self.edisgo_obj.electromobility.resample(freq="8min")
        assert (
            "Up-sampling to an uneven number of times the new index fits into "
            "the old index is not possible." in caplog.text
        )

    def test_integrity_check(self):
        """
        Checks resampling function with set up flexibility bands.

        """
        # set charging efficiency to 1 to make things easier
        self.edisgo_obj.electromobility.simbev_config_df.at[0, "eta_cp"] = 1.0

        # set up valid flexibility bands
        flex_bands = self.setup_simple_flex_band_data()
        timeindex = pd.date_range("1/1/1970", periods=6, freq="30min")

        # ######### check upper energy band lower than lower energy band ############
        # modify flex band such that error is raised
        flex_bands["upper_energy"].at[timeindex[1], "CP2"] = 1.0
        self.edisgo_obj.electromobility.flexibility_bands = flex_bands
        msg = re.escape(
            "Lower energy band is higher than upper energy band for the following "
            "charging points: ['CP2']. The maximum exceedance is 1.0. Please check."
        )
        with pytest.raises(ValueError, match=msg):
            self.edisgo_obj.electromobility.check_integrity()

        # ######### check upper energy higher than charging power ############
        # reset previously modified value
        flex_bands["upper_energy"].at[timeindex[1], "CP2"] = 2.0
        # modify flex band such that error is raised
        flex_bands["upper_energy"].at[timeindex[1], "CP1"] = 7.0
        self.edisgo_obj.electromobility.flexibility_bands = flex_bands
        msg = re.escape(
            "Upper energy band has power values higher than nominal power for the "
            "following charging points: ['CP1']. The maximum exceedance is 1.0. "
            "Please check."
        )
        with pytest.raises(ValueError, match=msg):
            self.edisgo_obj.electromobility.check_integrity()

        # ######### check lower energy higher than charging power ############
        # reset previously modified value
        flex_bands["upper_energy"].at[timeindex[1], "CP1"] = 6.0
        # modify flex band such that error is raised
        flex_bands["lower_energy"].at[timeindex[3], "CP1"] = 7.0
        self.edisgo_obj.electromobility.flexibility_bands = flex_bands
        msg = re.escape(
            "Lower energy band has power values higher than nominal power for the "
            "following charging points: ['CP1']. The maximum exceedance is 1.0. "
            "Please check."
        )
        with pytest.raises(ValueError, match=msg):
            self.edisgo_obj.electromobility.check_integrity()

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
        config_df = self.edisgo_obj.electromobility.simbev_config_df.copy()
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
        assert_frame_equal(
            self.edisgo_obj.electromobility.simbev_config_df,
            config_df,
            check_dtype=False,
        )
        shutil.rmtree(dir)
