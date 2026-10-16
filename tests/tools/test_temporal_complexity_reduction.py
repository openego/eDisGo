import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.tools import temporal_complexity_reduction as temp_red


class TestTemporalComplexityReduction:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()

        # Resample timeseries and reindex to hourly timedelta
        self.edisgo.resample_timeseries(freq="1min")
        self.timesteps = pd.date_range(start="01/01/2018", periods=240, freq="h")
        attributes = self.edisgo.timeseries._attributes
        for attr in attributes:
            if not getattr(self.edisgo.timeseries, attr).empty:
                df = pd.DataFrame(
                    index=self.timesteps,
                    columns=getattr(self.edisgo.timeseries, attr).columns,
                    data=getattr(self.edisgo.timeseries, attr).values,
                )
                setattr(
                    self.edisgo.timeseries,
                    attr,
                    df,
                )
        self.edisgo.timeseries.timeindex = self.timesteps
        self.edisgo.analyze()

    def test__scored_most_critical_loading(self):
        ts_crit = temp_red._scored_most_critical_loading(
            self.edisgo, weight_by_costs=False
        )
        assert len(ts_crit) == 180
        assert np.isclose(ts_crit.iloc[0], 1.45613)
        assert np.isclose(ts_crit.iloc[-1], 1.14647)

        ts_crit = temp_red._scored_most_critical_loading(self.edisgo)

        assert len(ts_crit) == 180
        assert np.isclose(ts_crit.iloc[0], 190.63611)
        assert np.isclose(ts_crit.iloc[-1], 48.13501)

    def test__scored_most_critical_voltage_issues(self):
        ts_crit = temp_red._scored_most_critical_voltage_issues(
            self.edisgo, weight_by_costs=False
        )
        assert len(ts_crit) == 120
        assert np.isclose(ts_crit.iloc[0], 0.01062258)
        assert np.isclose(ts_crit.iloc[-1], 0.01062258)

        ts_crit = temp_red._scored_most_critical_voltage_issues(self.edisgo)
        assert len(ts_crit) == 120
        assert np.isclose(ts_crit.iloc[0], 0.1062258)
        assert np.isclose(ts_crit.iloc[-1], 0.1062258)

    def test_get_most_critical_time_steps(self):
        ts_crit = temp_red.get_most_critical_time_steps(
            self.edisgo,
            num_steps_loading=2,
            num_steps_voltage=2,
            weight_by_costs=False,
            run_initial_analyze=False,
        )
        assert len(ts_crit) == 2

        ts_crit = temp_red.get_most_critical_time_steps(
            self.edisgo,
            num_steps_loading=2,
            num_steps_voltage=2,
            timesteps=self.edisgo.timeseries.timeindex[:24],
        )
        assert len(ts_crit) == 2

        ts_crit = temp_red.get_most_critical_time_steps(
            self.edisgo,
            mode="lv",
            lv_grid_id=2,
            percentage=0.5,
            num_steps_voltage=2,
        )
        assert len(ts_crit) == 0

        ts_crit = temp_red.get_most_critical_time_steps(
            self.edisgo,
            mode="lv",
            lv_grid_id=6,
            percentage=0.5,
            num_steps_voltage=2,
        )
        assert len(ts_crit) == 60

    def test__scored_most_critical_loading_time_interval(self):
        # test with default values
        ts_crit = temp_red._scored_most_critical_loading_time_interval(self.edisgo, 24)
        assert len(ts_crit) == 10
        assert (
            ts_crit.loc[0, "time_steps"]
            == pd.date_range("1/8/2018", periods=24, freq="H")
        ).all()
        assert np.isclose(
            ts_crit.loc[0, "percentage_max_overloaded_components"], 0.96479
        )
        assert np.isclose(
            ts_crit.loc[1, "percentage_max_overloaded_components"], 0.96479
        )

        # test with non-default values
        ts_crit = temp_red._scored_most_critical_loading_time_interval(
            self.edisgo, 24, time_step_day_start=4, overloading_factor=0.9
        )
        assert len(ts_crit) == 9
        assert (
            ts_crit.loc[0, "time_steps"]
            == pd.date_range("1/5/2018 4:00", periods=24, freq="H")
        ).all()
        assert ts_crit.loc[0, "percentage_max_overloaded_components"] == 1

        # test without weighting by costs
        ts_crit = temp_red._scored_most_critical_loading_time_interval(
            self.edisgo,
            48,
            weight_by_costs=False,
        )
        assert len(ts_crit) == 9
        assert (
            ts_crit.loc[0, "time_steps"]
            == pd.date_range("1/5/2018 0:00", periods=48, freq="H")
        ).all()

    def test__scored_most_critical_voltage_issues_time_interval(self):
        # test with default values
        ts_crit = temp_red._scored_most_critical_voltage_issues_time_interval(
            self.edisgo, 24
        )
        assert len(ts_crit) == 5
        assert (
            ts_crit.loc[0, "time_steps"]
            == pd.date_range("1/1/2018", periods=24, freq="H")
        ).all()
        assert (
            ts_crit.loc[:, "percentage_buses_max_voltage_deviation"].values == 1.0
        ).all()

        # test with non-default values
        ts_crit = temp_red._scored_most_critical_voltage_issues_time_interval(
            self.edisgo, 72, time_step_day_start=4, weight_by_costs=False
        )
        assert len(ts_crit) == 5
        assert (
            ts_crit.loc[0, "time_steps"]
            == pd.date_range("1/1/2018 4:00", periods=72, freq="H")
        ).all()

    def test__costs_per_line_and_transformer(self):
        costs = temp_red._costs_per_line_and_transformer(self.edisgo)
        assert len(costs) == 131 + 11
        assert np.isclose(costs["Line_10007"], 0.722445826838636 * 80)
        assert np.isclose(costs["LVGrid_1_station"], 10)

    def test__costs_per_feeder(self):
        costs = temp_red._costs_per_feeder(self.edisgo)
        assert len(costs) == 37
        assert np.isclose(costs["Bus_BranchTee_MVGrid_1_1"], 295.34795)
        assert np.isclose(costs["BusBar_MVGrid_1_LVGrid_1_LV"], 10)

    def test_get_most_critical_time_intervals(self):
        self.edisgo.timeseries.scale_timeseries(p_scaling_factor=2, q_scaling_factor=2)
        steps = temp_red.get_most_critical_time_intervals(
            self.edisgo, time_steps_per_time_interval=24, percentage=0.5
        )

        assert len(steps) == 5
        assert (
            steps.loc[0, "time_steps_overloading"]
            == pd.date_range("1/8/2018", periods=24, freq="H")
        ).all()
        assert (
            steps.loc[0, "time_steps_voltage_issues"]
            == pd.date_range("1/1/2018", periods=24, freq="H")
        ).all()


class TestIntervalHelpers:
    """Relocated pure helpers + residual_load selection (no DB / no power flow)."""

    @staticmethod
    def _week(start):
        return pd.date_range(start=start, periods=168, freq="h")

    def test_intervals_overlap(self):
        a = self._week("2035-01-01")
        assert temp_red.intervals_overlap(a, self._week("2035-01-04"))
        assert not temp_red.intervals_overlap(a, self._week("2035-06-01"))

    def test_select_two_intervals_disjoint(self):
        result = temp_red.select_two_intervals(
            [self._week("2035-01-01")], [self._week("2035-06-01")]
        )
        assert len(result) == 2
        assert not temp_red.intervals_overlap(result[0], result[1])

    def test_select_two_intervals_next_ranked(self):
        load = [self._week("2035-01-01")]
        volt = [self._week("2035-01-03"), self._week("2035-09-01")]
        result = temp_red.select_two_intervals(load, volt)
        assert len(result) == 2 and result[1].equals(volt[1])

    def test_select_two_intervals_concatenate(self):
        result = temp_red.select_two_intervals(
            [self._week("2035-01-01")], [self._week("2035-01-03")]
        )
        assert len(result) == 1
        assert (result[0][1:] - result[0][:-1]).nunique() == 1  # contiguous

    def test_select_two_intervals_single_and_empty(self):
        week = self._week("2035-01-01")
        assert temp_red.select_two_intervals([week], [])[0].equals(week)
        assert temp_red.select_two_intervals([], [week])[0].equals(week)
        assert temp_red.select_two_intervals([], []) == []

    def test_build_centered_interval(self):
        idx = pd.date_range("2035-01-01 00:00", periods=8760, freq="h")
        t = pd.Timestamp("2035-02-10 12:00")
        iv = temp_red._build_centered_interval(t, idx, 168, 4)
        assert len(iv) == 168
        assert iv[0].hour == 4  # starts on the day-start hour
        assert t in iv  # critical step contained
        assert iv[-1] != t  # centered -> not the last step

    def test_residual_load_steps_and_intervals(self, monkeypatch):
        import types

        idx = pd.date_range("2035-01-01 00:00", periods=8760, freq="h")
        residual = pd.Series(range(8760), index=idx, dtype=float)
        fake = types.SimpleNamespace(
            timeseries=types.SimpleNamespace(residual_load=residual)
        )
        monkeypatch.setattr(
            "edisgo.network.overlying_grid.distribute_overlying_grid_requirements",
            lambda e: fake,
        )
        # steps: top-3 highest + bottom-2 lowest residual
        steps = temp_red.get_most_critical_time_steps(
            object(), by="residual_load", num_steps_loading=3, num_steps_voltage=2
        )
        assert idx[-1] in steps and idx[0] in steps and len(steps) == 5

        # intervals: per-case columns, centered on the residual max/min steps
        e = types.SimpleNamespace(
            timeseries=types.SimpleNamespace(timeindex=idx),
            topology=types.SimpleNamespace(id="g"),
        )
        df = temp_red.get_most_critical_time_intervals(
            e,
            by="residual_load",
            num_time_intervals=2,
            time_steps_per_time_interval=168,
            time_step_day_start=4,
        )
        assert list(df.columns) == ["time_steps_load_case", "time_steps_feedin_case"]
        assert len(df) == 2
        # top load-case interval is centered on the global max residual step
        assert idx[-1] in df.loc[0, "time_steps_load_case"]

    def test_bad_by_raises(self):
        with pytest.raises(ValueError, match="power_flow.*residual_load"):
            temp_red.get_most_critical_time_steps(object(), by="bogus")
        with pytest.raises(ValueError, match="power_flow.*residual_load"):
            temp_red.get_most_critical_time_intervals(object(), by="bogus")
