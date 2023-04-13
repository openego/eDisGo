import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
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
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        self.timesteps = self.edisgo.timeseries.timeindex

    def setup_flexibility_data(self):
        self.edisgo.add_component(
            comp_type="load",
            type="heat_pump",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex,
                data=[1.0 / 5, 2.0 / 6, 2.0 / 5, 1.0 / 6],
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[26],
            p_set=2,
        )
        self.edisgo.add_component(
            comp_type="load",
            type="heat_pump",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex,
                data=[2.0 / 7.0, 4.0 / 8.0, 3.0 / 7.0, 3.0 / 8.0],
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[30],
            p_set=3,
        )

        # add heat pump, electromobility, overlying grid dummy data
        self.edisgo.heat_pump.cop_df = pd.DataFrame(
            data={
                "Heat_Pump_LVGrid_3_1": [5.0, 6.0, 5.0, 6.0],
                "Heat_Pump_LVGrid_5_1": [7.0, 8.0, 7.0, 8.0],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.heat_pump.heat_demand_df = pd.DataFrame(
            data={
                "Heat_Pump_LVGrid_3_1": [1.0, 2.0, 2.0, 1.0],
                "Heat_Pump_LVGrid_5_1": [2.0, 4.0, 3.0, 3.0],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.heat_pump.thermal_storage_units_df = pd.DataFrame(
            data={
                "capacity": [4, 8],
                "efficiency": [1, 1],
            },
            index=self.edisgo.heat_pump.heat_demand_df.columns,
        )

        self.edisgo.add_component(
            comp_type="load",
            type="charging_point",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex, data=[0.5, 0.5, 0.5, 0.5]
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[32],
            p_set=3,
        )

        flex_bands = {
            "lower_energy": pd.DataFrame(
                {"Charging_Point_LVGrid_6_1": [0, 0, 1, 2]},
                index=self.edisgo.timeseries.timeindex,
            ),
            "upper_energy": pd.DataFrame(
                {"Charging_Point_LVGrid_6_1": [1, 2, 2, 3]},
                index=self.edisgo.timeseries.timeindex,
            ),
            "upper_power": pd.DataFrame(
                {"Charging_Point_LVGrid_6_1": [1, 1, 2, 1]},
                index=self.edisgo.timeseries.timeindex,
            ),
        }
        self.edisgo.electromobility.flexibility_bands = flex_bands
        # ToDo: add DSM attribute to EDisGo object
        # self.edisgo.dsm.p_min = pd.DataFrame(
        #     data={
        #         "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
        #             -0.3,
        #             -0.3,
        #             -0.3,
        #             -0.3,
        #         ],
        #         "Load_industrial_LVGrid_5_1": [-0.07, -0.07, -0.07, -0.07],
        #     },
        #     index=self.edisgo.timeseries.timeindex,
        # )
        # self.edisgo.dsm.p_max = pd.DataFrame(
        #     data={
        #         "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
        #             0.3,
        #             0.3,
        #             0.3,
        #             0.3,
        #         ],
        #         "Load_industrial_LVGrid_5_1": [0.07, 0.07, 0.07, 0.07],
        #     },
        #     index=self.edisgo.timeseries.timeindex,
        # )
        # self.edisgo.dsm.e_min = pd.DataFrame(
        #     data={
        #         "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
        #             -0.3,
        #             -0.4,
        #             -0.5,
        #             -0.4,
        #         ],
        #         "Load_industrial_LVGrid_5_1": [-0.07, -0.07, -0.07, -0.07],
        #     },
        #     index=self.edisgo.timeseries.timeindex,
        # )
        # self.edisgo.dsm.e_max = pd.DataFrame(
        #     data={
        #         "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
        #             0.3,
        #             0.5,
        #             0.5,
        #             0.4,
        #         ],
        #         "Load_industrial_LVGrid_5_1": [0.07, 0.1, 0.09, 0.07],
        #     },
        #     index=self.edisgo.timeseries.timeindex,
        # )

        # ToDo: Add OG Data

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
        self.setup_flexibility_data()
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
