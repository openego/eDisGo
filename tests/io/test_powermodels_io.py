import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.io import powermodels_io
from edisgo.io.powermodels_io import _get_time_elapsed_in_hours
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

    def test_to_powermodels_aligns_storage_units_soc_to_different_year(self, caplog):
        """
        OverlyingGrid.storage_units_soc may be indexed in a different
        calendar year than the EDisGo object's own timeindex (e.g. imported
        independently, see _build_battery_storage). It must be year-aligned
        (with a warning) onto edisgo.timeseries.timeindex plus one trailing
        step - not left in its original year, and not raise a KeyError.
        """
        edisgo_obj = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        edisgo_obj.set_time_series_worst_case_analysis()
        assert edisgo_obj.timeseries.timeindex[0].year == 1970

        soc_year = 2018
        edisgo_obj.overlying_grid.storage_units_soc = pd.Series(
            data=[0.5, 0.6, 0.7, 0.8],
            index=pd.date_range(f"{soc_year}-01-01", periods=4, freq="h"),
        )

        with caplog.at_level("WARNING"):
            powermodels_network, _ = powermodels_io.to_powermodels(
                edisgo_obj,
                flexible_storage_units=(
                    edisgo_obj.topology.storage_units_df.index.values
                ),
            )

        assert "OverlyingGrid.storage_units_soc" in caplog.text
        assert str(soc_year) in caplog.text
        assert "1970" in caplog.text

        # storage_units_soc now lives on edisgo_obj's own (1970) timeindex
        # plus one trailing step, not the SOC series' original 2018 calendar
        assert edisgo_obj.overlying_grid.storage_units_soc.index[0].year == 1970
        assert len(edisgo_obj.overlying_grid.storage_units_soc) == (
            len(edisgo_obj.timeseries.timeindex) + 1
        )

        storage = powermodels_network["storage"]["1"]
        assert not pd.isna(storage["soc_initial"])
        assert not pd.isna(storage["soc_end"])

    def test__get_pf(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()

        # test mode None
        powermodels_network, hv_flex_dict = powermodels_io.to_powermodels(self.edisgo)
        for component in ["generator", "storage_unit"]:
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

        for component in ["heat_pump", "charging_point"]:
            for bus in [1, 29]:
                pf, sign = powermodels_io._get_pf(
                    self.edisgo, powermodels_network, bus, component
                )
                assert pf == 1
                assert sign == 1


class TestOverlyingGridTimeindexAlignment:
    """to_powermodels must read the overlying-grid requirements for the ACTIVE
    time index, not positionally off a wider one.

    Regression tests for openego/eDisGo#762.
    """

    @pytest.fixture
    def edisgo_obj(self):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        edisgo.set_time_series_worst_case_analysis()
        return edisgo

    @staticmethod
    def _set_overlying_grid(edisgo, index, values):
        for attr in [
            "renewables_curtailment",
            "storage_units_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "heat_pump_central_active_power",
            "dsm_active_power",
        ]:
            setattr(edisgo.overlying_grid, attr, pd.Series(values, index=index))

    def test_hv_requirements_use_active_timeindex(self, edisgo_obj):
        """With the overlying grid narrowed to the active index, the scalar
        target and the per-time-step series belong to that interval."""
        a = edisgo_obj.timeseries.timeindex[:2]
        b = edisgo_obj.timeseries.timeindex[2:]
        full = a.union(b)
        # 1.0 over the first interval, 9.0 over the second
        self._set_overlying_grid(edisgo_obj, full, [1.0] * len(a) + [9.0] * len(b))
        # narrow BOTH the active index and the overlying grid to interval 2,
        # as _narrow_flex_inputs does per interval
        edisgo_obj.set_timeindex(b)
        self._set_overlying_grid(edisgo_obj, b, [9.0] * len(b))

        pm, hv_flex_dict = powermodels_io.to_powermodels(edisgo_obj, opf_version=3)

        # the scalar target for network 1 is interval 2's first value, not
        # interval 1's
        assert pm["HV_requirements"]["1"]["P"] == pytest.approx(9.0)
        # and the per-time-step series covers only interval 2
        assert len(pm["time_series"]["HV_requirements"]["1"]["P"]) == len(b)
        assert pm["time_series"]["HV_requirements"]["1"]["P"] == [9.0] * len(b)

    def test_raises_when_overlying_grid_does_not_cover_timeindex(self, edisgo_obj):
        """A requirement series that misses time steps of the active index
        raises instead of being read positionally."""
        full = edisgo_obj.timeseries.timeindex
        # overlying grid covers only the first two steps ...
        self._set_overlying_grid(edisgo_obj, full[:2], [1.0, 1.0])
        # ... while the optimization runs on all four
        with pytest.raises(ValueError, match="do not cover the time index"):
            powermodels_io.to_powermodels(edisgo_obj, opf_version=3)

    def test_error_names_attributes_and_missing_steps(self, edisgo_obj):
        """The error message names the offending attribute and step count."""
        full = edisgo_obj.timeseries.timeindex
        self._set_overlying_grid(edisgo_obj, full, [1.0] * len(full))
        # shorten a single attribute
        edisgo_obj.overlying_grid.electromobility_active_power = pd.Series(
            [1.0], index=full[:1]
        )
        with pytest.raises(ValueError) as exc:
            powermodels_io.to_powermodels(edisgo_obj, opf_version=3)
        assert "electromobility_active_power" in str(exc.value)
        # (missing, additional) for that attribute
        assert f"({len(full) - 1}, 0)" in str(exc.value)

    def test_raises_when_overlying_grid_is_wider_than_timeindex(self, edisgo_obj):
        """The reported failure mode: the active index is narrowed to one
        interval but the overlying grid still spans all of them, so the
        positional reads take the wrong interval's values."""
        full = edisgo_obj.timeseries.timeindex
        a, b = full[:2], full[2:]
        self._set_overlying_grid(edisgo_obj, full, [1.0] * len(a) + [9.0] * len(b))
        # narrow the active index only -- the overlying grid stays wide
        edisgo_obj.set_timeindex(b)
        with pytest.raises(ValueError, match="do not cover the time index"):
            powermodels_io.to_powermodels(edisgo_obj, opf_version=3)

    def test_empty_overlying_grid_still_falls_back(self, edisgo_obj):
        """An absent overlying grid is not a coverage error — it keeps the
        existing fallback to opf_version 2."""
        pm, hv_flex_dict = powermodels_io.to_powermodels(edisgo_obj, opf_version=3)
        assert pm["opf_version"] == 2

    def test_no_check_for_opf_version_below_three(self, edisgo_obj):
        """opf_version 1 and 2 do not use HV requirements, so a short
        overlying-grid series is irrelevant and must not raise."""
        full = edisgo_obj.timeseries.timeindex
        self._set_overlying_grid(edisgo_obj, full[:1], [1.0])
        pm, hv_flex_dict = powermodels_io.to_powermodels(edisgo_obj, opf_version=2)
        assert pm["opf_version"] == 2


# test _get_time_elapsed_in_hours for inter-timestep couplings in
# Julia to simulate
# 1) timesteps <1h, 1h, 2h, >24h and
# 2) whether too few, non-equidistant or
#    negative timestamps raise an error
def _snapshots(*timestamps):
    return pd.to_datetime(timestamps)


# 1) test various snapshot intervals and expected time elapsed in hours
@pytest.mark.parametrize(
    ("snapshots", "expected"),
    [
        (pd.date_range("2035-01-01", periods=3, freq="15min"), 0.25),
        (pd.date_range("2035-01-01", periods=3, freq="h"), 1.0),
        (pd.date_range("2035-01-01", periods=3, freq="2h"), 2.0),
        (_snapshots("2035-01-01 00:00", "2035-01-02 01:00"), 25.0),
    ],
)
# test expected time elapsed in hours for above defined snapshot intervals
def test_get_time_elapsed_in_hours(snapshots, expected):
    assert _get_time_elapsed_in_hours(snapshots) == pytest.approx(expected)


# 2) test three other cases:
# 1. not enough snapshots (1 snapshot)
# 2. non-equidistant snapshots (15min, 1h, 1h 15min)
# 3. negative time elapsed (snapshots in reverse order)
@pytest.mark.parametrize(
    ("snapshots", "error_message"),
    [
        (
            _snapshots("2035-01-01 00:00"),
            "At least two snapshots",
        ),
        (
            _snapshots(
                "2035-01-01 00:00",
                "2035-01-01 00:15",
                "2035-01-01 01:15",
            ),
            "equidistant",
        ),
        (
            _snapshots(
                "2035-01-01 01:00",
                "2035-01-01 00:00",
            ),
            "positive",
        ),
    ],
)
# test that ValueError is raised for the above three cases
def test_get_time_elapsed_in_hours_raises(snapshots, error_message):
    with pytest.raises(ValueError, match=error_message):
        _get_time_elapsed_in_hours(snapshots)
