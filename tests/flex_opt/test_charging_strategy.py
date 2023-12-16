import pandas as pd
import pytest

from edisgo.edisgo import EDisGo
from edisgo.flex_opt.charging_strategies import charging_strategy


class TestChargingStrategy:
    """
    Tests all charging strategies implemented in charging_strategies.py.

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

        cls.edisgo_obj.import_electromobility(
            data_source="directory",
            charging_processes_dir=cls.simbev_path,
            potential_charging_points_dir=cls.tracbev_path,
        )

    def test_charging_strategy(self, caplog):
        charging_demand_lst = []

        ts = self.edisgo_obj.timeseries

        for strategy in self.charging_strategies:
            charging_strategy(self.edisgo_obj, strategy=strategy)

            # Check if all charging points have a valid chargingdemand_kWh > 0
            df = ts.charging_points_active_power(self.edisgo_obj).loc[
                :, (ts.charging_points_active_power(self.edisgo_obj) <= 0).any(axis=0)
            ]

            assert df.shape == ts.charging_points_active_power(self.edisgo_obj).shape

            charging_demand_lst.append(
                ts.charging_points_active_power(self.edisgo_obj).sum()
            )

        # Check charging strategy for different timestamp_share_threshold value
        charging_strategy(
            self.edisgo_obj, strategy="dumb", timestamp_share_threshold=0.5
        )

        # Check if resampling warning is raised
        assert (
            "The frequency of the time series data of the edisgo object differs"
            in caplog.text
        )

        # Check if all charging points have a valid chargingdemand_kWh > 0
        df = ts.charging_points_active_power(self.edisgo_obj).loc[
            :, (ts.charging_points_active_power(self.edisgo_obj) <= 0).any(axis=0)
        ]

        assert df.shape == ts.charging_points_active_power(self.edisgo_obj).shape

        # Check charging strategy for different minimum_charging_capacity_factor
        charging_strategy(
            self.edisgo_obj, strategy="reduced", minimum_charging_capacity_factor=0.5
        )

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

        # ##################### check time index #####################
        assert ts._loads_active_power.index.freqstr == "H"
        # change time index to quarter-hourly
        timeindex = pd.date_range("1/1/2011", periods=24 * 7, freq="0.25H")
        self.edisgo_obj.set_timeindex(timeindex)
        charging_strategy(self.edisgo_obj, strategy="dumb")
        assert ts._loads_active_power.index.freqstr == "15T"
