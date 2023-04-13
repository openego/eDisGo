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
        # add heat pump dummy data
        self.edisgo.add_component(
            comp_type="load",
            type="heat_pump",
            sector="individual_heating",
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
            sector="individual_heating",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex,
                data=[2.0 / 7.0, 4.0 / 8.0, 3.0 / 7.0, 3.0 / 8.0],
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[30],
            p_set=3,
        )

        self.edisgo.heat_pump.cop_df = pd.DataFrame(
            data={
                "Heat_Pump_LVGrid_3_individual_heating_1": [5.0, 6.0, 5.0, 6.0],
                "Heat_Pump_LVGrid_5_individual_heating_1": [7.0, 8.0, 7.0, 8.0],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.heat_pump.heat_demand_df = pd.DataFrame(
            data={
                "Heat_Pump_LVGrid_3_individual_heating_1": [1.0, 2.0, 2.0, 1.0],
                "Heat_Pump_LVGrid_5_individual_heating_1": [2.0, 4.0, 3.0, 3.0],
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
        # add electromobility dummy data
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
        # add DSM dummy data
        self.edisgo.dsm.p_min = pd.DataFrame(
            data={
                "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
                    -0.3,
                    -0.3,
                    -0.3,
                    -0.3,
                ],
                "Load_industrial_LVGrid_5_1": [-0.07, -0.07, -0.07, -0.07],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.dsm.p_max = pd.DataFrame(
            data={
                "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
                    0.3,
                    0.3,
                    0.3,
                    0.3,
                ],
                "Load_industrial_LVGrid_5_1": [0.07, 0.07, 0.07, 0.07],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.dsm.e_min = pd.DataFrame(
            data={
                "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
                    -0.3,
                    -0.4,
                    -0.5,
                    -0.4,
                ],
                "Load_industrial_LVGrid_5_1": [-0.07, -0.07, -0.07, -0.07],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.dsm.e_max = pd.DataFrame(
            data={
                "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
                    0.3,
                    0.5,
                    0.5,
                    0.4,
                ],
                "Load_industrial_LVGrid_5_1": [0.07, 0.1, 0.09, 0.07],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        # add overlying grid dummy data
        for attr in [
            "dsm_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "renewables_curtailment",
            "storage_units_active_power",
        ]:
            if attr == "dsm_active_power":
                data = [0.1, -0.1, -0.1, 0.1]
            elif attr == "electromobility_active_power":
                data = [0.4, 0.5, 0.5, 0.6]
            elif attr == "heat_pump_decentral_active_power":
                data = [0.5, 0.85, 0.85, 0.55]
            elif attr == "storage_units_active_power":
                data = [-0.35, -0.35, 0.35, 0.35]
            if attr == "renewables_curtailment":
                df = pd.DataFrame(
                    index=self.timesteps,
                    columns=["solar", "wind"],
                    data=0.1,
                )
            else:
                df = pd.Series(
                    index=self.timesteps,
                    data=data,
                )
            setattr(
                self.edisgo.overlying_grid,
                attr,
                df,
            )

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
        # Battery electric vehicle timeseries
        for key, df in self.edisgo.electromobility.flexibility_bands.items():
            if not df.empty:
                df.index = self.timesteps
                self.edisgo.electromobility.flexibility_bands.update({key: df})
        # Heat pumps timeseries
        for attr in ["cop_df", "heat_demand_df"]:
            if not getattr(self.edisgo.heat_pump, attr).empty:
                df = pd.DataFrame(
                    index=self.timesteps,
                    columns=getattr(self.edisgo.heat_pump, attr).columns,
                    data=getattr(self.edisgo.heat_pump, attr).values,
                )
                setattr(
                    self.edisgo.heat_pump,
                    attr,
                    df,
                )
        # Demand Side Management timeseries
        for attr in ["e_min", "e_max", "p_min", "p_max"]:
            if not getattr(self.edisgo.dsm, attr).empty:
                df = pd.DataFrame(
                    index=self.timesteps,
                    columns=getattr(self.edisgo.dsm, attr).columns,
                    data=getattr(self.edisgo.dsm, attr).values,
                )
                setattr(
                    self.edisgo.dsm,
                    attr,
                    df,
                )
        # overlying grid timeseries
        for attr in [
            "dsm_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "renewables_curtailment",
            "storage_units_active_power",
        ]:
            if not getattr(self.edisgo.overlying_grid, attr).empty:
                if attr == "renewables_curtailment":
                    df = pd.DataFrame(
                        index=self.timesteps,
                        columns=getattr(self.edisgo.overlying_grid, attr).columns,
                        data=getattr(self.edisgo.overlying_grid, attr).values,
                    )
                else:
                    df = pd.Series(
                        index=self.timesteps,
                        data=getattr(self.edisgo.overlying_grid, attr).values,
                    )
                setattr(
                    self.edisgo.overlying_grid,
                    attr,
                    df,
                )

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
        self.edisgo.analyze()
        ts_crit = _scored_most_critical_loading_time_interval(self.edisgo, 1)

        assert len(ts_crit) == 9
        for ts in ts_crit.OL_t1:
            assert ts in self.timesteps

    def test__scored_most_critical_voltage_issues_time_interval(self):
        self.setup_flexibility_data()
        self.edisgo.analyze()
        ts_crit = _scored_most_critical_voltage_issues_time_interval(self.edisgo, 1)

        assert len(ts_crit) == 9
        for ts in ts_crit.V_t1:
            assert ts in self.timesteps

    def test_get_steps_flex_opf(self):
        self.setup_flexibility_data()
        self.edisgo.analyze()
        steps = get_steps_flex_opf(self.edisgo, num_ti=6, window_days=1)

        assert len(steps) == 6
        assert len(steps.columns) == 6

    def test_distribute_overlying_grid_timeseries(self):
        self.setup_flexibility_data()
        # self.edisgo.analyze()
        edisgo_copy = distribute_overlying_grid_timeseries(self.edisgo)
        dsm = self.edisgo.dsm.e_max.columns.values
        hps = self.edisgo.heat_pump.cop_df.columns.values
        # res = self.edisgo.topology.generators_df.loc[
        #     (self.edisgo.topology.generators_df.type == "solar")
        #     | (self.edisgo.topology.generators_df.type == "wind")
        # ].index.values
        assert {
            np.isclose(
                edisgo_copy.timeseries.loads_active_power[hps].sum(axis=1)[i],
                self.edisgo.overlying_grid.heat_pump_decentral_active_power[i],
                atol=1e-5,
            )
            for i in range(len(self.timesteps))
        } == {True}
        assert (
            edisgo_copy.timeseries.loads_active_power["Charging_Point_LVGrid_6_1"]
            == self.edisgo.overlying_grid.electromobility_active_power.values
        ).all()
        assert (
            edisgo_copy.timeseries.storage_units_active_power["Storage_1"]
            == self.edisgo.overlying_grid.storage_units_active_power.values
        ).all()
        assert {
            np.isclose(
                edisgo_copy.timeseries.loads_active_power[dsm].sum(axis=1)[i],
                self.edisgo.timeseries.loads_active_power[dsm].sum(axis=1)[i]
                + self.edisgo.overlying_grid.dsm_active_power.values[i],
                atol=1e-5,
            )
            for i in range(len(self.timesteps))
        } == {True}
        # ToDo: Curtailment funktioniert nicht
        # assert set([np.isclose(edisgo_copy.timeseries.generators_active_power[
        #             res].sum(axis=1)[i],
        #             self.edisgo.timeseries.generators_active_power[
        #             res].sum(
        #     axis=1)[i] -
        #     self.edisgo.overlying_grid.renewables_curtailment.sum(axis=1).values[i],
        #     atol=1e-5) for i in range(len(self.timesteps))]) == {True}
        print(" ")
