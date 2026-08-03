import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.opf.powermodels_opf import pm_optimize
from edisgo.tools.tools import aggregate_district_heating_components


class TestPowerModelsOPF:
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
                "capacity": [4, 8, 8],
                "efficiency": [1, 1, 1],
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

    @pytest.mark.runonlinux
    def test_pm_optimize(self):
        # OPF with all flexibilities but without overlying grid constraints
        pm_optimize(
            self.edisgo,
            opf_version=2,
            silence_moi=True,
            method="nc",
            flexible_cps=np.array(["Charging_Point_LVGrid_6_1"]),
            flexible_hps=self.edisgo.heat_pump.thermal_storage_units_df.index.values,
            flexible_loads=self.edisgo.dsm.e_min.columns.values,
            flexible_storage_units=self.edisgo.topology.storage_units_df.index.values,
        )

        assert np.isclose(
            np.round(self.edisgo.opf_results.slack_generator_t.pg.iloc[-1], 3),
            -20.683,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.opf_results.heat_storage_t.p[
                    "Heat_Pump_LVGrid_3_individual_heating_1"
                ].iloc[-1],
                3,
            ),
            0,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.timeseries.loads_active_power.Charging_Point_LVGrid_6_1.iloc[
                    -1
                ],
                3,
            ),
            0.761,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.timeseries.loads_active_power[
                    "Heat_Pump_LVGrid_5_individual_heating_1"
                ].iloc[-1],
                3,
            ),
            0.375,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.timeseries.storage_units_active_power.Storage_1.iloc[-1], 3
            ),
            0.16,
            atol=1e-3,
        )
        assert np.isclose(
            np.round(
                self.edisgo.timeseries.loads_active_power[
                    "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1"
                ].iloc[-1],
                3,
            ),
            0.031 + 0.193,
            atol=1e-3,
        )
        assert self.edisgo.opf_results.status == "LOCALLY_SOLVED"

        # OPF with all flexibilities and including overlying grid constraints
        self.setup_class()
        pm_optimize(
            self.edisgo,
            opf_version=4,
            silence_moi=True,
            method="nc",
            flexible_cps=np.array(["Charging_Point_LVGrid_6_1"]),
            flexible_hps=self.edisgo.heat_pump.thermal_storage_units_df.index.values,
            flexible_loads=self.edisgo.dsm.e_min.columns.values,
            flexible_storage_units=self.edisgo.topology.storage_units_df.index.values,
        )

        assert min(
            np.unique(
                np.isclose(
                    self.edisgo.overlying_grid.heat_pump_central_active_power.values
                    + (
                        self.edisgo.overlying_grid.heat_pump_decentral_active_power
                    ).values,
                    self.edisgo.timeseries.loads_active_power[
                        self.edisgo.heat_pump.cop_df.columns.values
                    ]
                    .sum(axis=1)
                    .values
                    + self.edisgo.opf_results.hv_requirement_slacks_t.hp.values,
                    atol=1e-3,
                )
            )
        )

        assert min(
            np.unique(
                np.isclose(
                    self.edisgo.overlying_grid.electromobility_active_power.values,
                    self.edisgo.timeseries.loads_active_power[
                        "Charging_Point_LVGrid_6_1"
                    ].values
                    + self.edisgo.opf_results.hv_requirement_slacks_t.cp.values,
                    atol=1e-3,
                )
            )
        )

        assert min(
            np.unique(
                np.isclose(
                    self.edisgo.overlying_grid.storage_units_active_power.values,
                    self.edisgo.timeseries.storage_units_active_power.sum(axis=1).values
                    + self.edisgo.opf_results.hv_requirement_slacks_t.storage.values,
                    atol=1e-3,
                )
            )
        )


class TestContiguousIntervals:
    def test_contiguous_index_is_one_interval(self):
        from edisgo.opf.powermodels_opf import _contiguous_intervals

        ti = pd.date_range("2035-01-01", periods=48, freq="h")
        result = _contiguous_intervals(ti)
        assert len(result) == 1 and result[0].equals(ti)

    def test_two_disconnected_intervals(self):
        from edisgo.opf.powermodels_opf import _contiguous_intervals

        a = pd.date_range("2035-01-01", periods=24, freq="h")
        b = pd.date_range("2035-07-01", periods=24, freq="h")
        result = _contiguous_intervals(a.union(b))
        assert len(result) == 2
        assert result[0].equals(a) and result[1].equals(b)

    def test_freq_restored_on_freqless_index(self):
        from edisgo.opf.powermodels_opf import _contiguous_intervals

        idx = pd.DatetimeIndex(pd.date_range("2035-01-01", periods=48, freq="h").values)
        assert idx.freq is None
        result = _contiguous_intervals(idx)
        assert len(result) == 1 and result[0].freq is not None

    def test_single_and_empty(self):
        from edisgo.opf.powermodels_opf import _contiguous_intervals

        assert len(_contiguous_intervals(pd.date_range("2035-01-01", periods=1))) == 1
        assert _contiguous_intervals(pd.DatetimeIndex([])) == []


class TestMergeOpfTimeFrames:
    @staticmethod
    def _empty_snapshot(opf, slack_generator_t):
        return {
            "slack_generator_t": slack_generator_t,
            "hv_requirement_slacks_t": pd.DataFrame(),
            "lines_t": {k: pd.DataFrame() for k in opf.lines_t._attributes()},
            "heat_storage_t": {
                k: pd.DataFrame() for k in opf.heat_storage_t._attributes()
            },
            "grid_slacks_t": {
                k: pd.DataFrame() for k in opf.grid_slacks_t._attributes()
            },
            "battery_storage_t": {
                k: pd.DataFrame() for k in opf.battery_storage_t._attributes()
            },
        }

    def test_flat_frame_concatenated_and_sorted(self):
        from edisgo.opf.powermodels_opf import _merge_opf_time_frames
        from edisgo.opf.results.opf_result_class import OPFResults

        opf = OPFResults()
        a = pd.DataFrame({"x": [1.0]}, index=pd.date_range("2035-01-01", periods=1))
        b = pd.DataFrame({"x": [2.0]}, index=pd.date_range("2035-07-01", periods=1))
        _merge_opf_time_frames(
            opf, [self._empty_snapshot(opf, a), self._empty_snapshot(opf, b)]
        )
        assert len(opf.slack_generator_t) == 2
        assert list(opf.slack_generator_t["x"]) == [1.0, 2.0]


class TestPmOptimizeIntervalSplit:
    """pm_optimize's multi-interval split, with the single-interval OPF stubbed
    (no Julia/DB). Patches powermodels_opf._pm_optimize_single."""

    @pytest.fixture
    def edisgo_obj(self):
        e = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        e.set_timeindex(pd.date_range("2035-01-01", periods=24, freq="h"))
        return e

    def test_single_interval_calls_once_with_freq(self, edisgo_obj, monkeypatch):
        import edisgo.opf.powermodels_opf as pmo

        ti = pd.DatetimeIndex(pd.date_range("2035-01-01", periods=24, freq="h").values)
        assert ti.freq is None
        edisgo_obj.set_timeindex(ti)
        seen = []
        monkeypatch.setattr(
            pmo,
            "_pm_optimize_single",
            lambda e, **kw: seen.append(e.timeseries.timeindex.freq),
        )
        pmo.pm_optimize(edisgo_obj)
        assert seen == [pd.tseries.frequencies.to_offset("h")]

    def test_two_intervals_run_separately_and_restore(self, edisgo_obj, monkeypatch):
        import edisgo.opf.powermodels_opf as pmo

        a = pd.date_range("2035-01-01", periods=24, freq="h")
        b = pd.date_range("2035-07-01", periods=24, freq="h")
        full = a.union(b)
        edisgo_obj.set_timeindex(full)
        seen = []

        def fake_single(e, **kw):
            seen.append(e.timeseries.timeindex)
            e.opf_results.status = "OPTIMAL"
            e.opf_results.solver = "Gurobi"
            e.opf_results.solution_time = 1.0

        monkeypatch.setattr(pmo, "_pm_optimize_single", fake_single)
        pmo.pm_optimize(edisgo_obj)
        assert len(seen) == 2 and seen[0].equals(a) and seen[1].equals(b)
        assert edisgo_obj.timeseries.timeindex.equals(full)
        assert len(edisgo_obj.opf_results.interval_results) == 2
        assert edisgo_obj.opf_results.solution_time == 2.0
        assert edisgo_obj.opf_results.status == "OPTIMAL"

    def test_overlying_grid_state_restored(self, edisgo_obj, monkeypatch):
        import edisgo.opf.powermodels_opf as pmo

        a = pd.date_range("2035-01-01", periods=24, freq="h")
        b = pd.date_range("2035-07-01", periods=24, freq="h")
        full = a.union(b)
        edisgo_obj.set_timeindex(full)
        edisgo_obj.overlying_grid.storage_units_soc = pd.Series(1.0, index=full)
        seen_types = []

        def fake_single(e, **kw):
            og = e.overlying_grid
            seen_types.append(type(og.storage_units_soc).__name__)
            og.storage_units_soc = pd.DataFrame(
                0.0, index=e.timeseries.timeindex, columns=["s1"]
            )
            e.opf_results.status = "OPTIMAL"

        monkeypatch.setattr(pmo, "_pm_optimize_single", fake_single)
        pmo.pm_optimize(edisgo_obj)
        assert seen_types == ["Series", "Series"]
        assert isinstance(edisgo_obj.overlying_grid.storage_units_soc, pd.Series)

    def test_reactive_power_restored(self, edisgo_obj, monkeypatch):
        import edisgo.opf.powermodels_opf as pmo

        a = pd.date_range("2035-01-01", periods=24, freq="h")
        b = pd.date_range("2035-07-01", periods=24, freq="h")
        full = a.union(b)
        edisgo_obj.set_timeindex(full)
        gen = edisgo_obj.topology.generators_df.index[0]
        edisgo_obj.timeseries._generators_reactive_power = pd.DataFrame(
            0.0, index=full, columns=[gen]
        )
        seen_ok = []

        def fake_single(e, **kw):
            ti = e.timeseries.timeindex
            q = e.timeseries.generators_reactive_power
            seen_ok.append(not q.empty and q.index.equals(ti))
            e.timeseries._generators_reactive_power = pd.DataFrame(
                0.0, index=ti, columns=[gen]
            )
            e.opf_results.status = "OPTIMAL"

        monkeypatch.setattr(pmo, "_pm_optimize_single", fake_single)
        pmo.pm_optimize(edisgo_obj)
        assert seen_ok == [True, True]
        assert edisgo_obj.timeseries._generators_reactive_power.index.equals(full)

    def test_infeasible_interval_stores_report_and_raises(
        self, edisgo_obj, monkeypatch
    ):
        import edisgo.opf.powermodels_opf as pmo

        from edisgo.flex_opt.exceptions import InfeasibleModelError

        a = pd.date_range("2035-01-01", periods=24, freq="h")
        b = pd.date_range("2035-07-01", periods=24, freq="h")
        full = a.union(b)
        edisgo_obj.set_timeindex(full)

        def fake_single(e, **kw):
            if e.timeseries.timeindex[0] == a[0]:
                e.opf_results.status = "OPTIMAL"
                e.opf_results.solution_time = 1.0
            else:
                raise InfeasibleModelError("stub")

        monkeypatch.setattr(pmo, "_pm_optimize_single", fake_single)
        with pytest.raises(InfeasibleModelError):
            pmo.pm_optimize(edisgo_obj)
        report = edisgo_obj.opf_results.interval_results
        assert len(report) == 2
        assert report[0]["status"] == "OPTIMAL"
        assert report[1]["status"] == "infeasible"
        assert edisgo_obj.timeseries.timeindex.equals(full)
