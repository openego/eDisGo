import numpy as np

# import pandas as pd
import pytest

# from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files
from edisgo.opf.timeseries_reduction import (
    _scored_most_critical_loading,
    _scored_most_critical_loading_time_interval,
    _scored_most_critical_voltage_issues,
    _scored_most_critical_voltage_issues_time_interval,
    distribute_overlying_grid_timeseries,
    get_steps_flex_opf,
    get_steps_reinforcement,
)


class TestTimeseriesReduction:
    @classmethod
    def setup_class(self):
        # self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        # self.edisgo.set_time_series_worst_case_analysis()
        # self.timesteps = self.edisgo.timeseries.timeindex
        self.edisgo = import_edisgo_from_files(
            pytest.ding0_test_network_4_path,
            import_timeseries=True,
            import_electromobility=True,
            import_heat_pump=True,
            import_dsm=True,
            import_overlying_grid=True,
            from_zip_archive=True,
        )
        # timeindex = pd.date_range("1/1/2018", periods=168, freq="H")
        # gens_ts = pd.DataFrame(
        #     data={
        #         "GeneratorFluctuating_15": pd.concat(
        #             [pd.Series([2.0, 5.0, 6.0])] * 56).values,
        #         "GeneratorFluctuating_24": pd.concat(
        #             [pd.Series([4.0, 7.0, 8.0])] * 56).values,
        #     },
        #     index=timeindex,
        # )
        # loads_ts = pd.DataFrame(
        #     data={
        #         "Load_residential_LVGrid_5_3": pd.concat(
        #         [pd.Series([2.0, 5.0, 6.0])] * 56, axis=1
        #     ).transpose()
        #     .values,
        #     },
        #     index=timeindex,
        # )
        # storage_units_ts = pd.DataFrame(
        #     data={
        #         "Storage_1": pd.concat(
        #         [pd.Series([4.0, 7.0, 8.0])] * 56, axis=1
        #     ).transpose()
        #     .values,
        #     },
        #     index=timeindex,
        # )
        #
        # # ToDo: add DSM, heatpump, emob, OG
        #
        # self.edisgo.set_time_series_manual(
        #     generators_p=gens_ts,
        #     generators_q=gens_ts,
        #     loads_p=loads_ts,
        #     storage_units_q=storage_units_ts,
        # )

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

    def test__scored_most_critical_loading_time_interval(self):

        ts_crit = _scored_most_critical_loading_time_interval(self.edisgo, 1)

        assert len(ts_crit) == 3

        assert (ts_crit.index == self.timesteps[[0, 1, 3]]).all()

        assert (
            np.isclose(ts_crit[self.timesteps[[0, 1, 3]]], [1.45613, 1.45613, 1.14647])
        ).all()

    def test__scored_most_critical_voltage_issues_time_interval(self):

        ts_crit = _scored_most_critical_voltage_issues_time_interval(self.edisgo)

        assert len(ts_crit) == 2

        assert (ts_crit.index == self.timesteps[[0, 1]]).all()

        assert (
            np.isclose(ts_crit[self.timesteps[[0, 1]]], [0.01062258, 0.01062258])
        ).all()

    def test_get_steps_flex_opf(self):

        ts_crit = get_steps_flex_opf(self.edisgo)

        assert len(ts_crit) == 3

        assert (ts_crit == self.timesteps[[0, 1, 3]]).all()

    def test_distribute_overlying_grid_timeseries(self):

        ts_crit = distribute_overlying_grid_timeseries(self.edisgo)

        assert len(ts_crit) == 3

        assert (ts_crit == self.timesteps[[0, 1, 3]]).all()
