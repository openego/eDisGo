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
        ts_crit = temp_red._scored_most_critical_loading(self.edisgo)

        assert len(ts_crit) == 180
        assert np.isclose(ts_crit.iloc[0], 1.45613)
        assert np.isclose(ts_crit.iloc[-1], 1.14647)

    def test__scored_most_critical_voltage_issues(self):
        ts_crit = temp_red._scored_most_critical_voltage_issues(self.edisgo)

        assert len(ts_crit) == 120
        assert np.isclose(ts_crit.iloc[0], 0.01062258)
        assert np.isclose(ts_crit.iloc[-1], 0.01062258)

    def test_get_most_critical_time_steps(self):
        ts_crit = temp_red.get_most_critical_time_steps(
            self.edisgo, num_steps_loading=2, num_steps_voltage=2
        )
        assert len(ts_crit) == 3

        ts_crit = temp_red.get_most_critical_time_steps(
            self.edisgo,
            num_steps_loading=2,
            num_steps_voltage=2,
            timesteps=self.edisgo.timeseries.timeindex[:24],
        )
        assert len(ts_crit) == 2

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
