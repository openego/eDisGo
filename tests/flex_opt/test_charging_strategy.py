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

    @pytest.mark.parametrize("strategy", ["dumb", "reduced", "residual"])
    def test_charging_strategy_trims_to_short_timeindex(self, strategy):
        """
        Regression test for eDisGo#703: charging_strategy used to write the
        full SimBEV-simulation-length series into loads_active_power/
        loads_reactive_power regardless of a shorter active timeindex. When
        the edisgo/SimBEV frequencies already match (no internal resample
        round-trip), the written series must be trimmed to exactly
        edisgo.timeseries.timeindex - no extra rows, no missing rows.
        """
        edisgo = EDisGo(ding0_grid=self.ding0_path)
        # 15-min frequency matches the SimBEV fixture's stepsize (see
        # metadata_simbev_run.json), so no internal resample round-trip is
        # triggered - one day instead of the fixture's full simulated week.
        short_timeindex = pd.date_range("1/1/2011", periods=96, freq="15min")
        edisgo.set_timeindex(short_timeindex)
        edisgo.import_electromobility(
            data_source="directory",
            charging_processes_dir=self.simbev_path,
            potential_charging_points_dir=self.tracbev_path,
        )

        charging_strategy(edisgo, strategy=strategy)

        pd.testing.assert_index_equal(
            edisgo.timeseries._loads_active_power.index, short_timeindex
        )
        pd.testing.assert_index_equal(
            edisgo.timeseries._loads_reactive_power.index, short_timeindex
        )
        assert not edisgo.timeseries.loads_active_power.isna().any().any()

    def test_charging_strategy_trims_to_gapped_timeindex(self):
        """
        Regression test for eDisGo#703: a gapped timeindex (as produced by
        select_timesteps in auto mode) must survive charging_strategy
        unchanged when no internal frequency resample round-trip is
        triggered - the written series must match the gapped index exactly,
        not a contiguous range spanning it.
        """
        edisgo = EDisGo(ding0_grid=self.ding0_path)
        gapped_timeindex = pd.date_range("1/1/2011", periods=24, freq="15min").union(
            pd.date_range("1/6/2011 18:00", periods=24, freq="15min")
        )
        edisgo.set_timeindex(gapped_timeindex)
        edisgo.import_electromobility(
            data_source="directory",
            charging_processes_dir=self.simbev_path,
            potential_charging_points_dir=self.tracbev_path,
        )

        charging_strategy(edisgo, strategy="dumb")

        pd.testing.assert_index_equal(
            edisgo.timeseries._loads_active_power.index, gapped_timeindex
        )

    def test_charging_strategy_with_subset_of_parks(self):
        """
        Charging strategies can be applied to different subsets of charging parks
        without overwriting each other's results.
        """
        # edisgo = self.edisgo_obj
        timeindex = pd.date_range("1/1/2011", periods=24 * 7, freq="H")
        edisgo = self.edisgo_obj
        edisgo.set_timeindex(timeindex)

        ts = edisgo.timeseries
        # Baseline: apply a strategy to all integrated charging parks
        # so that we have non-zero time series for all of them.
        edisgo.apply_charging_strategy(strategy="dumb")

        integrated = edisgo.electromobility.integrated_charging_parks_df
        all_park_ids = list(integrated.index)

        # If the test network had less than two charging parks, this test would
        # not make sense.
        if len(all_park_ids) < 2:
            pytest.skip("Not enough charging parks for subset test.")

        park_a = all_park_ids[0]
        park_b = all_park_ids[1]

        edisgo_id_a = integrated.loc[park_a, "edisgo_id"]
        edisgo_id_b = integrated.loc[park_b, "edisgo_id"]

        # store baseline time series for both parks
        loads_before = ts._loads_active_power.copy()
        ts_b_before = loads_before[edisgo_id_b].copy()

        # 1) apply a strategy only to park A
        edisgo.apply_charging_strategy(
            strategy="reduced",
            charging_park_ids=[park_a],
        )

        loads_after_first = ts._loads_active_power
        ts_a_after_first = loads_after_first[edisgo_id_a].copy()
        ts_b_after_first = loads_after_first[edisgo_id_b].copy()

        # park B should be unchanged by a call that only targets park A
        pd.testing.assert_series_equal(ts_b_before, ts_b_after_first, check_names=True)

        # 2) apply a different strategy only to park B
        edisgo.apply_charging_strategy(
            strategy="residual",
            charging_park_ids=[park_b],
        )

        loads_after_second = ts._loads_active_power
        ts_a_after_second = loads_after_second[edisgo_id_a].copy()

        # park A must not be changed by the second call that targets only park B
        pd.testing.assert_series_equal(
            ts_a_after_first, ts_a_after_second, check_names=True
        )
