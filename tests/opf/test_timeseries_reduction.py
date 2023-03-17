import numpy as np
import pytest

from edisgo import EDisGo
from edisgo.opf.timeseries_reduction import (
    _scored_most_critical_loading,
    _scored_most_critical_voltage_issues,
    get_steps_reinforcement,
)


class TestTimeseriesReduction:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        self.timesteps = self.edisgo.timeseries.timeindex

    @pytest.fixture(autouse=True)
    def run_power_flow(self):
        """
        Fixture to run new power flow before each test.

        """
        self.edisgo.analyze()

    def test__scored_most_critical_loading(self):

        ts_crit = _scored_most_critical_loading(self.edisgo)

        assert len(ts_crit) == 3

        assert (ts_crit.index == self.timesteps[[0, 1, 3]]).all()

        assert (
            np.isclose(ts_crit[self.timesteps[[0, 1, 3]]], [1.45613, 1.45613, 1.14647])
        ).all()

    def test__scored_most_critical_voltage_issues(self):

        ts_crit = _scored_most_critical_voltage_issues(self.edisgo)

        assert len(ts_crit) == 2

        assert (ts_crit.index == self.timesteps[[0, 1]]).all()

        assert (
            np.isclose(ts_crit[self.timesteps[[0, 1]]], [0.01062258, 0.01062258])
        ).all()

    def test_get_steps_reinforcement(self):

        ts_crit = get_steps_reinforcement(self.edisgo)

        assert len(ts_crit) == 3

        assert (ts_crit == self.timesteps[[0, 1, 3]]).all()
