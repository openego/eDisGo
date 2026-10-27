import copy

import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.flex_opt import check_tech_constraints
from edisgo.flex_opt.hosting_capacity import (
    TEST_COMPONENT,
    _search_hosting_capacity,
    hosting_capacity,
)


def _max_violations(edisgo_obj):
    """Maximum relative loading and absolute voltage deviation per component."""
    loading = check_tech_constraints.components_relative_load(edisgo_obj).max()
    voltage = (
        check_tech_constraints.voltage_deviation_from_allowed_voltage_limits(edisgo_obj)
        .abs()
        .max()
    )
    return loading, voltage


def _worsens(edisgo_obj, reference):
    """Whether any constraint is violated more than in `reference`."""
    loading, voltage = _max_violations(edisgo_obj)
    ref_loading, ref_voltage = reference
    return bool(
        (loading > ref_loading.clip(lower=1.0) + 1e-6).any()
        or (voltage > ref_voltage + 1e-6).any()
    )


class TestSearchHostingCapacity:
    @staticmethod
    def _check_with_limit(limit, calls):
        def check(p):
            calls.append(p)
            return None if p <= limit else ("overloading", "Line_1", None)

        return check

    def test_bisects_to_tolerance(self):
        calls = []
        capacity, violation = _search_hosting_capacity(
            self._check_with_limit(3.3, calls), 0.1, 100.0, 0.001, 0.01
        )
        assert 3.3 - 0.01 * 3.4 <= capacity <= 3.3
        assert violation == ("overloading", "Line_1", None)
        # doubling 0.1, 0.2, ..., 6.4 and then bisecting
        assert calls[:7] == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])

    def test_capped_at_p_max(self):
        calls = []
        capacity, violation = _search_hosting_capacity(
            self._check_with_limit(1000.0, calls), 0.1, 5.0, 0.001, 0.01
        )
        assert capacity == 5.0
        assert violation is None
        assert calls[-1] == 5.0

    def test_violation_at_p_start(self):
        calls = []
        capacity, violation = _search_hosting_capacity(
            self._check_with_limit(0.0, calls), 0.1, 100.0, 0.001, 0.0
        )
        assert capacity == 0.0
        assert violation is not None
        assert max(calls) == 0.1
        assert min(calls) <= 0.001


class TestHostingCapacity:
    @classmethod
    def setup_class(cls):
        cls.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        cls.edisgo.set_time_series_worst_case_analysis()
        cls.mv_bus = "Bus_Generator_1"
        cls.lv_bus = "Bus_BranchTee_LVGrid_1_14"

    def test_generator(self):
        edisgo_before = copy.deepcopy(self.edisgo)
        df = hosting_capacity(self.edisgo, [self.mv_bus, self.lv_bus], rtol=0.05)

        assert list(df.index) == [self.mv_bus, self.lv_bus]
        assert list(df.columns) == [
            "hosting_capacity",
            "limiting_constraint",
            "limiting_component",
            "limiting_timestep",
            "voltage_level",
        ]
        assert list(df.voltage_level) == ["mv", "lv"]
        assert (df.hosting_capacity > 0).all()
        # more power fits at the MV bus than in the LV grid
        assert (
            df.at[self.mv_bus, "hosting_capacity"]
            > (df.at[self.lv_bus, "hosting_capacity"])
        )
        assert df.limiting_constraint.isin(
            ["overloading", "overvoltage", "undervoltage"]
        ).all()
        assert df.limiting_timestep.isin(self.edisgo.timeseries.timeindex).all()

        # the EDisGo object is not changed
        assert TEST_COMPONENT not in self.edisgo.timeseries.generators_active_power
        assert TEST_COMPONENT not in self.edisgo.topology.generators_df.index
        assert self.edisgo.results.v_res.empty
        pd.testing.assert_frame_equal(
            self.edisgo.timeseries.generators_active_power,
            edisgo_before.timeseries.generators_active_power,
        )

    def test_result_matches_connected_component(self):
        # connecting a generator with the hosting capacity with the regular API must
        # not worsen any constraint, connecting a larger one must
        df = hosting_capacity(self.edisgo, self.lv_bus, rtol=0.02)
        capacity = df.at[self.lv_bus, "hosting_capacity"]

        reference_edisgo = copy.deepcopy(self.edisgo)
        reference_edisgo.analyze()
        reference = _max_violations(reference_edisgo)

        for factor, worsens in [(1.0, False), (1.1, True)]:
            edisgo_obj = copy.deepcopy(self.edisgo)
            p = capacity * factor
            edisgo_obj.add_component(
                "generator",
                bus=self.lv_bus,
                p_nom=p,
                generator_type="solar",
                ts_active_power=pd.Series(p, index=edisgo_obj.timeseries.timeindex),
                ts_reactive_power="default",
            )
            edisgo_obj.analyze()
            assert _worsens(edisgo_obj, reference) is worsens

    def test_load(self):
        df = hosting_capacity(
            self.edisgo, self.mv_bus, component_type="heat_pump", rtol=0.05
        )
        assert df.at[self.mv_bus, "hosting_capacity"] >= 0
        assert df.at[self.mv_bus, "limiting_constraint"] is not None

    def test_profile_and_p_max(self):
        timeindex = self.edisgo.timeseries.timeindex
        # without power in any time step, nothing can be violated
        df = hosting_capacity(
            self.edisgo,
            self.lv_bus,
            profile=pd.Series(0.0, index=timeindex),
            p_start=1.0,
            p_max=2.0,
        )
        assert df.at[self.lv_bus, "hosting_capacity"] == 2.0
        assert df.at[self.lv_bus, "limiting_constraint"] is None

        # with power in only one time step, only that time step can be limiting
        profile = pd.Series(0.0, index=timeindex)
        profile.iloc[1] = 1.0
        df = hosting_capacity(self.edisgo, self.lv_bus, profile=profile, rtol=0.1)
        assert df.at[self.lv_bus, "limiting_timestep"] == timeindex[1]

    def test_method_on_edisgo(self):
        df = self.edisgo.hosting_capacity(
            self.lv_bus, timesteps=self.edisgo.timeseries.timeindex[:2], rtol=0.1
        )
        assert df.at[self.lv_bus, "hosting_capacity"] > 0
        assert (
            df.at[self.lv_bus, "limiting_timestep"]
            in (self.edisgo.timeseries.timeindex[:2])
        )

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="component_type"):
            hosting_capacity(self.edisgo, self.lv_bus, component_type="storage_unit")
        with pytest.raises(ValueError, match="not in the grid"):
            hosting_capacity(self.edisgo, ["Bus_does_not_exist"])
        with pytest.raises(ValueError, match="p_start"):
            hosting_capacity(self.edisgo, self.lv_bus, p_start=2.0, p_max=1.0)
        with pytest.raises(ValueError, match="profile"):
            hosting_capacity(
                self.edisgo,
                self.lv_bus,
                profile=pd.Series(
                    np.ones(1), index=self.edisgo.timeseries.timeindex[:1]
                ),
            )
