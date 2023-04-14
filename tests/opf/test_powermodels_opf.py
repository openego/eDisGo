import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.opf.powermodels_opf import pm_optimize


class TestPowerModelsOPF:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
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
            # ToDo: add dummy Data
            # "solarthermal_energy_feedin_district_heating",
            # "geothermal_energy_feedin_district_heating"
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
                    index=self.edisgo.timeseries.timeindex,
                    columns=["solar", "wind"],
                    data=[[0, 0], [0, 0], [0.1, 0.1], [0.1, 0.1]],
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

    def test_pm_optimize(self):
        # OPF with all flexibilities
        pm_optimize(
            self.edisgo,
            opf_version=2,
            silence_moi=True,
            method="nc",
            flexible_cps=["Charging_Point_LVGrid_6_1"],
            flexible_hps=self.edisgo.heat_pump.cop_df.columns.values,
            flexible_loads=self.edisgo.dsm.e_min.columns.values,
            flexible_storage_units=self.edisgo.topology.storage_units_df.index.values,
        )

        assert np.isclose(
            np.round(self.edisgo.opf_results.slack_generator_t.pg[-1], 3),
            -20.75,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.opf_results.heat_storage_t.p.Heat_Pump_LVGrid_3_1[-1], 3
            ),
            -0.666,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.timeseries.loads_active_power.Charging_Point_LVGrid_6_1[-1],
                3,
            ),
            0.761,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.timeseries.loads_active_power.Heat_Pump_LVGrid_5_1[-1], 3
            ),
            0.525,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.timeseries.storage_units_active_power.Storage_1[-1], 3
            ),
            0.157,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.timeseries.loads_active_power[
                    "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1"
                ][-1],
                3,
            ),
            0.031 + 0.208,
            atol=1e-3,
        )
        assert self.edisgo.opf_results.status == "LOCALLY_SOLVED"

        # ToDo: add assertions for overlying grid OPF
