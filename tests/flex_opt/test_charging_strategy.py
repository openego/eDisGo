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

    def _setup_edisgo_with_single_synthetic_event(
        self, timeindex, park_start_timesteps, park_end_timesteps, chargingdemand_kWh
    ):
        """
        Helper for the ADR 0001 regression tests below: imports the real
        SimBEV/TracBEV fixture (so charging park integration/topology wiring
        is realistic), then overwrites charging_processes_df with a single,
        fully controlled synthetic event on one of the fixture's own
        integrated charging parks, reusing that park's own use_case/capacity
        so `harmonize_charging_processes_df` sees realistic values.
        """
        edisgo = EDisGo(ding0_grid=self.ding0_path)
        edisgo.set_timeindex(timeindex)
        edisgo.import_electromobility(
            data_source="directory",
            charging_processes_dir=self.simbev_path,
            potential_charging_points_dir=self.tracbev_path,
        )
        integrated = edisgo.electromobility.integrated_charging_parks_df
        park_id = integrated.index[0]
        template = edisgo.electromobility.charging_processes_df[
            edisgo.electromobility.charging_processes_df.charging_park_id == park_id
        ].iloc[0]

        park_time_timesteps = park_end_timesteps - park_start_timesteps + 1
        edisgo.electromobility.charging_processes_df = pd.DataFrame(
            {
                "ags": [template.ags],
                "car_id": [0],
                "destination": [template.destination],
                # avoid public/hpc (always dumb-charged even under
                # "residual") so these events exercise the flex-ranking path
                "use_case": ["work"],
                "nominal_charging_capacity_kW": [template.nominal_charging_capacity_kW],
                "grid_charging_capacity_kW": [template.grid_charging_capacity_kW],
                "chargingdemand_kWh": [chargingdemand_kWh],
                "park_time_timesteps": [park_time_timesteps],
                "park_start_timesteps": [park_start_timesteps],
                "park_end_timesteps": [park_end_timesteps],
                "charging_park_id": [park_id],
                "charging_point_id": [template.charging_point_id],
            }
        )
        return edisgo, park_id

    def test_residual_drops_fully_out_of_window_events(self):
        """
        Regression test for ADR 0001: a charging event whose parking window
        has zero overlap with the active timeindex must contribute nothing -
        not be tiled/fabricated against repeated residual_load data.
        """
        timeindex = pd.date_range("1/1/2011", periods=96, freq="15min")  # 1 day
        # park window entirely on day 3, well beyond the 1-day active window
        edisgo, park_id = self._setup_edisgo_with_single_synthetic_event(
            timeindex,
            park_start_timesteps=300,
            park_end_timesteps=320,
            chargingdemand_kWh=10.0,
        )

        charging_strategy(edisgo, strategy="residual")

        edisgo_id = edisgo.electromobility.integrated_charging_parks_df.at[
            park_id, "edisgo_id"
        ]
        written = edisgo.timeseries.loads_active_power[edisgo_id]
        assert (written == 0).all()

    def test_residual_prorates_boundary_straddling_event(self):
        """
        Regression test for ADR 0001: a charging event whose parking window
        straddles the active timeindex boundary must have its charging
        demand prorated by the in-window fraction of its parking time, not
        fully charged (which would require tiled/fabricated residual_load
        data beyond the active timeindex) and not dropped.
        """
        timeindex = pd.date_range("1/1/2011", periods=96, freq="15min")  # steps 0-95
        # park window [90, 149] (60 steps), only steps 90-95 (6 steps) are
        # in-window -> in_window_fraction = 6/60 = 0.1
        edisgo, park_id = self._setup_edisgo_with_single_synthetic_event(
            timeindex,
            park_start_timesteps=90,
            park_end_timesteps=149,
            chargingdemand_kWh=12.0,
        )
        edisgo_id = edisgo.electromobility.integrated_charging_parks_df.at[
            park_id, "edisgo_id"
        ]

        charging_strategy(edisgo, strategy="residual")

        written = edisgo.timeseries.loads_active_power[edisgo_id]

        # only in-window steps (90-95) may carry any charging
        assert (written.iloc[:90] == 0).all()
        assert written.iloc[90:].sum() > 0

        # compare against the same event fully inside a timeindex covering
        # its whole parking window - the straddling case must deliver
        # strictly less energy than the fully-observable case, since only
        # 1/10th of its parking time is actually in-window here
        full_timeindex = pd.date_range("1/1/2011", periods=150, freq="15min")
        edisgo_full, park_id_full = self._setup_edisgo_with_single_synthetic_event(
            full_timeindex,
            park_start_timesteps=90,
            park_end_timesteps=149,
            chargingdemand_kWh=12.0,
        )
        charging_strategy(edisgo_full, strategy="residual")
        edisgo_id_full = edisgo_full.electromobility.integrated_charging_parks_df.at[
            park_id_full, "edisgo_id"
        ]
        full_energy = edisgo_full.timeseries.loads_active_power[edisgo_id_full].sum()

        straddling_energy = written.sum()
        assert 0 < straddling_energy < full_energy

    def test_residual_fully_inside_event_unaffected(self):
        """
        Regression test for ADR 0001: an event whose parking window is
        entirely inside the active timeindex must be scheduled exactly the
        same regardless of how much longer the active timeindex extends
        beyond the event's own window - proration must not affect
        fully-observable events at all.
        """
        edisgo_short, park_id_short = self._setup_edisgo_with_single_synthetic_event(
            pd.date_range("1/1/2011", periods=60, freq="15min"),
            park_start_timesteps=10,
            park_end_timesteps=50,
            chargingdemand_kWh=8.0,
        )
        edisgo_long, park_id_long = self._setup_edisgo_with_single_synthetic_event(
            pd.date_range("1/1/2011", periods=200, freq="15min"),
            park_start_timesteps=10,
            park_end_timesteps=50,
            chargingdemand_kWh=8.0,
        )

        charging_strategy(edisgo_short, strategy="residual")
        charging_strategy(edisgo_long, strategy="residual")

        edisgo_id_short = edisgo_short.electromobility.integrated_charging_parks_df.at[
            park_id_short, "edisgo_id"
        ]
        edisgo_id_long = edisgo_long.electromobility.integrated_charging_parks_df.at[
            park_id_long, "edisgo_id"
        ]
        energy_short = edisgo_short.timeseries.loads_active_power[edisgo_id_short].sum()
        energy_long = edisgo_long.timeseries.loads_active_power[edisgo_id_long].sum()

        assert energy_short > 0
        assert energy_short == pytest.approx(energy_long)

    def test_residual_no_tiling_across_gapped_timeindex(self):
        """
        Regression test for ADR 0001: with a gapped active timeindex, an
        event that overlaps both disjoint runs must only ever be scheduled
        into steps actually present in the active timeindex - never into the
        gap, and never against fabricated/tiled residual_load data.
        """
        # two disjoint 15-min runs: steps 0-23 and steps 100-123 (gap of 76
        # steps in between, well beyond any real residual_load coverage)
        run_1 = pd.date_range("1/1/2011", periods=24, freq="15min")
        run_2 = pd.date_range("1/1/2011", periods=24, freq="15min") + pd.Timedelta(
            minutes=15 * 100
        )
        gapped_timeindex = run_1.union(run_2)

        edisgo, park_id = self._setup_edisgo_with_single_synthetic_event(
            gapped_timeindex,
            park_start_timesteps=10,
            park_end_timesteps=110,
            chargingdemand_kWh=6.0,
        )

        charging_strategy(edisgo, strategy="residual")

        edisgo_id = edisgo.electromobility.integrated_charging_parks_df.at[
            park_id, "edisgo_id"
        ]
        written = edisgo.timeseries.loads_active_power[edisgo_id]

        # written series must exactly match the gapped index - nothing
        # fabricated to bridge the gap
        pd.testing.assert_index_equal(written.index, gapped_timeindex)
        assert written.sum() > 0

    def test_residual_dumb_subbucket_respects_internal_gap(self):
        """
        Regression test for ADR 0001: a "dumb-charged" event within the
        residual strategy (use_case in {public, hpc} or flex_time == 0)
        whose deterministic charging interval spans a gap in the active
        timeindex must only ever write to in-window positions - never a
        blind contiguous slice bridging the gap.
        """
        run_1 = pd.date_range("1/1/2011", periods=10, freq="15min")
        run_2 = pd.date_range("1/1/2011", periods=10, freq="15min") + pd.Timedelta(
            minutes=15 * 20
        )
        gapped_timeindex = run_1.union(run_2)

        edisgo = EDisGo(ding0_grid=self.ding0_path)
        edisgo.set_timeindex(gapped_timeindex)
        edisgo.import_electromobility(
            data_source="directory",
            charging_processes_dir=self.simbev_path,
            potential_charging_points_dir=self.tracbev_path,
        )
        integrated = edisgo.electromobility.integrated_charging_parks_df
        park_id = integrated.index[0]
        template = edisgo.electromobility.charging_processes_df[
            edisgo.electromobility.charging_processes_df.charging_park_id == park_id
        ].iloc[0]

        # public use_case -> always dumb-charged even under "residual";
        # park window [5, 24] straddles the gap (steps 10-19)
        edisgo.electromobility.charging_processes_df = pd.DataFrame(
            {
                "ags": [template.ags],
                "car_id": [0],
                "destination": [template.destination],
                "use_case": ["public"],
                "nominal_charging_capacity_kW": [template.nominal_charging_capacity_kW],
                "grid_charging_capacity_kW": [template.grid_charging_capacity_kW],
                "chargingdemand_kWh": [2.0],
                "park_time_timesteps": [20],
                "park_start_timesteps": [5],
                "park_end_timesteps": [24],
                "charging_park_id": [park_id],
                "charging_point_id": [template.charging_point_id],
            }
        )

        charging_strategy(edisgo, strategy="residual")

        edisgo_id = integrated.at[park_id, "edisgo_id"]
        written = edisgo.timeseries.loads_active_power[edisgo_id]

        pd.testing.assert_index_equal(written.index, gapped_timeindex)
        # no NaNs, no crash from writing into a position that doesn't exist
        assert not written.isna().any()

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
