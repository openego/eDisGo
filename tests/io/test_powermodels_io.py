import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.io import powermodels_io


class TestPowermodelsIO:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
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

    def test_to_powermodels(self):
        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(self.edisgo)

        assert len(powermodels_network["gen"].keys()) == 1 + 1
        assert len(powermodels_network["gen_slack"].keys()) == 1
        assert len(powermodels_network["gen_nd"].keys()) == 27
        assert len(powermodels_network["bus"].keys()) == 142
        assert len(powermodels_network["branch"].keys()) == 141
        assert len(powermodels_network["load"].keys()) == 50 + 1 + 3
        assert len(powermodels_network["storage"].keys()) == 0
        assert len(powermodels_network["electromobility"].keys()) == 0
        assert len(powermodels_network["heatpumps"].keys()) == 0
        assert len(powermodels_network["heat_storage"].keys()) == 0
        assert len(powermodels_network["dsm"].keys()) == 0

        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(
            self.edisgo,
            flexible_cps=["Charging_Point_LVGrid_6_1"],
            flexible_hps=self.edisgo.heat_pump.cop_df.columns.values,
            flexible_loads=self.edisgo.dsm.e_min.columns.values,
            flexible_storage_units=self.edisgo.topology.storage_units_df.index.values,
        )
        assert len(powermodels_network["gen"].keys()) == 1
        assert len(powermodels_network["gen_slack"].keys()) == 1
        assert len(powermodels_network["gen_nd"].keys()) == 27
        assert len(powermodels_network["bus"].keys()) == 143
        assert len(powermodels_network["branch"].keys()) == 142
        assert len(powermodels_network["load"].keys()) == 50
        assert len(powermodels_network["storage"].keys()) == 1
        assert len(powermodels_network["electromobility"].keys()) == 1
        assert len(powermodels_network["heatpumps"].keys()) == 2
        assert len(powermodels_network["heat_storage"].keys()) == 2
        assert len(powermodels_network["dsm"].keys()) == 2

        # ToDo: test more options with test network including overlying grid

    def test__get_pf(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()

        # test mode None
        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(self.edisgo)
        for component in ["gen", "storage"]:
            pf, sign = powermodels_io._get_pf(
                self.edisgo, powermodels_network, 1, component
            )
            assert pf == 0.9
            assert sign == -1
            pf, sign = powermodels_io._get_pf(
                self.edisgo, powermodels_network, 29, component
            )
            assert pf == 0.95
            assert sign == -1

        for component in ["hp", "cp"]:
            for bus in [1, 29]:
                pf, sign = powermodels_io._get_pf(
                    self.edisgo, powermodels_network, 1, component
                )
                assert pf == 1
                assert sign == 1
