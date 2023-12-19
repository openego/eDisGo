import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.io import powermodels_io
from edisgo.tools.tools import aggregate_district_heating_components


class TestPowermodelsIO:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
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
        self.edisgo.add_component(
            comp_type="load",
            type="heat_pump",
            sector="district_heating_resistive_heater",
            district_heating_id="grid1",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex,
                data=[2.0, 8.0, 3.0, 3.0],
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[27],
            p_set=2,
        )
        self.edisgo.add_component(
            comp_type="load",
            type="heat_pump",
            sector="district_heating",
            district_heating_id="grid1",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex,
                data=[2.0 / 7.0, 8.0 / 2.0, 3.0 / 7.0, 3.0 / 8.0],
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[27],
            p_set=3,
        )

        # add heat pump, electromobility, overlying grid dummy data
        self.edisgo.heat_pump.cop_df = pd.DataFrame(
            data={
                "Heat_Pump_LVGrid_3_individual_heating_1": [5.0, 6.0, 5.0, 6.0],
                "Heat_Pump_LVGrid_5_individual_heating_1": [7.0, 8.0, 7.0, 8.0],
                "Heat_Pump_MVGrid_1_district_heating_resistive_heater_1": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                "Heat_Pump_MVGrid_1_district_heating_2": [7.0, 2.0, 7.0, 8.0],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.heat_pump.heat_demand_df = pd.DataFrame(
            data={
                "Heat_Pump_LVGrid_3_individual_heating_1": [1.0, 2.0, 2.0, 1.0],
                "Heat_Pump_LVGrid_5_individual_heating_1": [2.0, 4.0, 3.0, 3.0],
                "Heat_Pump_MVGrid_1_district_heating_2": [2.0, 8.0, 3.0, 3.0],
                "Heat_Pump_MVGrid_1_district_heating_resistive_heater_1": [
                    2.0,
                    8.0,
                    3.0,
                    3.0,
                ],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.heat_pump.thermal_storage_units_df = pd.DataFrame(
            data={
                "capacity": [4.0, 8.0, 8.0],
                "efficiency": [1.0, 1.0, 1.0],
            },
            index=self.edisgo.heat_pump.heat_demand_df.columns[:-1],
        )
        aggregate_district_heating_components(self.edisgo)
        self.edisgo.apply_heat_pump_operating_strategy()

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
                {"Charging_Point_LVGrid_6_1": [0.0, 0.0, 1.0, 2.0]},
                index=self.edisgo.timeseries.timeindex,
            ),
            "upper_energy": pd.DataFrame(
                {"Charging_Point_LVGrid_6_1": [1.0, 2.0, 2.0, 3.0]},
                index=self.edisgo.timeseries.timeindex,
            ),
            "upper_power": pd.DataFrame(
                {"Charging_Point_LVGrid_6_1": [1.0, 1.0, 2.0, 1.0]},
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

        # add overlying grid dummy data
        for attr in [
            "dsm_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "heat_pump_central_active_power",
            "renewables_curtailment",
            "storage_units_active_power",
            "feedin_district_heating",
        ]:
            if attr == "dsm_active_power":
                data = [0.1, -0.1, -0.1, 0.1]
            elif attr == "electromobility_active_power":
                data = [0.4, 0.5, 0.5, 0.6]
            elif attr in [
                "heat_pump_decentral_active_power",
                "heat_pump_central_active_power",
            ]:
                data = [0.5, 0.85, 0.85, 0.55]
            elif attr == "storage_units_active_power":
                data = [-0.35, -0.35, 0.35, 0.35]
            elif attr == "renewables_curtailment":
                data = [0, 0, 0.1, 0.1]

            if attr == "feedin_district_heating":
                df = pd.DataFrame(
                    index=self.edisgo.timeseries.timeindex,
                    columns=["grid1"],
                    data=[1.0, 2.0, 1.0, 2.0],
                )
            else:
                df = pd.Series(
                    index=self.edisgo.timeseries.timeindex,
                    data=data,
                )
            setattr(
                self.edisgo.overlying_grid,
                attr,
                df,
            )

    def test_to_powermodels(self):
        # test without flexibilities
        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(self.edisgo)

        assert len(powermodels_network["gen"].keys()) == 1 + 1
        assert len(powermodels_network["gen_slack"].keys()) == 1
        assert len(powermodels_network["gen_nd"].keys()) == 27
        assert len(powermodels_network["bus"].keys()) == 142
        assert len(powermodels_network["branch"].keys()) == 141
        assert len(powermodels_network["load"].keys()) == 50 + 1 + 3 + 1
        assert len(powermodels_network["storage"].keys()) == 0
        assert len(powermodels_network["electromobility"].keys()) == 0
        assert len(powermodels_network["heatpumps"].keys()) == 0
        assert len(powermodels_network["heat_storage"].keys()) == 0
        assert len(powermodels_network["dsm"].keys()) == 0
        assert powermodels_network["load"]["55"]["pd"] == 0.4
        assert powermodels_network["time_series"]["load"]["55"]["pd"] == [
            0.4,
            0.4,
            0.0,
            0.0,
        ]
        assert powermodels_network["time_series"]["gen"]["2"]["pg"] == [
            0.0,
            0.0,
            0.4,
            0.4,
        ]
        assert min(
            np.unique(
                np.isclose(
                    np.array(powermodels_network["time_series"]["load"]["36"]["pd"]),
                    self.edisgo.timeseries.loads_active_power[
                        powermodels_network["load"]["36"]["name"]
                    ].values,
                    atol=1e-3,
                )
            )
        )
        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(
            self.edisgo,
            opf_version=4,
            flexible_cps=["Charging_Point_LVGrid_6_1"],
            flexible_hps=self.edisgo.heat_pump.thermal_storage_units_df.index.values,
            flexible_loads=np.array(
                ["Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1"]
            ),
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
        assert len(powermodels_network["heatpumps"].keys()) == 2 + 1
        assert len(powermodels_network["heat_storage"].keys()) == 2 + 1
        assert len(powermodels_network["dsm"].keys()) == 1
        assert len(powermodels_network["HV_requirements"].keys()) == 5
        assert min(
            np.unique(
                np.isclose(
                    powermodels_network["time_series"]["heatpumps"]["3"]["pd"],
                    self.edisgo.heat_pump.heat_demand_df[
                        "Heat_Pump_MVGrid_1_district_heating_2"
                    ],
                    atol=1e-3,
                )
            )
        )
        assert len(powermodels_network["dsm"].keys()) == 1
        assert min(
            np.unique(
                np.isclose(
                    hv_flex_dict["dsm"],
                    self.edisgo.overlying_grid.dsm_active_power
                    - self.edisgo.timeseries.loads_active_power[
                        "Load_industrial_LVGrid_5_1"
                    ],
                    atol=1e-3,
                )
            )
        )

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
