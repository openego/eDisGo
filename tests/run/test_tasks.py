"""
Unit tests for individual pipeline tasks in :mod:`edisgo.run.tasks`.

These cover the task control flow without a database or SSH tunnel: a
small self-constructed ding0 grid is enough, and the DB-free branches
of ``import_overlying_grid_data`` are exercised directly.
"""

import glob
import os

import pandas as pd
import pytest

import edisgo.run as edisgo_run

from edisgo.edisgo import EDisGo
from edisgo.io import electromobility_import
from edisgo.run.config import load_config
from edisgo.run.context import RunContext
from edisgo.run.tasks import flex as flex_tasks
from edisgo.run.tasks.analysis import task_optimize
from edisgo.run.tasks.flex import task_build_flexibility_bands, task_import_flex
from edisgo.run.tasks.io import task_import_overlying_grid_data
from edisgo.run.tasks.timeseries import (
    task_manual_ts,
    task_select_critical_timesteps,
    task_set_timeindex,
)
from edisgo.run.validator import validate
from edisgo.tools.temporal_complexity_reduction import (
    intervals_overlap,
    select_two_intervals,
)


@pytest.fixture
def edisgo_obj():
    """Small ding0 grid with a 3-step time index, no DB access."""
    edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
    edisgo.set_timeindex(pd.date_range("2011-01-01", periods=3, freq="h"))
    return edisgo


class TestManualTs:
    def test_manual_ts_applies_active_power(self, edisgo_obj):
        """
        task_manual_ts must forward the ``*_active_power`` args to
        EDisGo.set_time_series_manual's real parameter names (regression: the
        task used to pass unsupported kwargs and always raised TypeError).
        """
        ti = edisgo_obj.timeseries.timeindex
        gen = edisgo_obj.topology.generators_df.index[0]
        df = pd.DataFrame({gen: [0.1, 0.2, 0.3]}, index=ti)

        ctx = RunContext()
        result = task_manual_ts(edisgo_obj, ctx, generators_active_power=df)

        assert gen in result.timeseries.generators_active_power.columns
        assert ctx.flags["timeseries_set"] is True


class TestImportFlex:
    """import_flex reads the flexibilities: list and dispatches per carrier."""

    def _fake_carrier_tasks(self, monkeypatch):
        calls = []

        def make_fake(carrier):
            def fake(edisgo, ctx, **kwargs):
                calls.append((carrier, kwargs))
                return edisgo

            return fake

        for carrier in flex_tasks.FLEX_CARRIERS:
            monkeypatch.setitem(flex_tasks._CARRIER_TASKS, carrier, make_fake(carrier))
        return calls

    def test_imports_configured_carriers(self, monkeypatch):
        calls = self._fake_carrier_tasks(monkeypatch)
        ctx = RunContext(raw_config={"flexibilities": ["dsm", "heat_pumps"]})
        sentinel = object()
        assert task_import_flex(sentinel, ctx) is sentinel
        # dispatched in FLEX_CARRIERS order, regardless of list order
        assert [c for c, _ in calls] == ["heat_pumps", "dsm"]

    def test_carrier_kwargs_forwarded(self, monkeypatch):
        calls = self._fake_carrier_tasks(monkeypatch)
        ctx = RunContext(raw_config={"flexibilities": ["electromobility"]})
        task_import_flex(object(), ctx, electromobility={"charging_strategy": None})
        assert calls == [("electromobility", {"charging_strategy": None})]

    def test_carriers_param_overrides_config(self, monkeypatch):
        calls = self._fake_carrier_tasks(monkeypatch)
        ctx = RunContext(raw_config={"flexibilities": ["dsm"]})
        task_import_flex(object(), ctx, carriers=["home_batteries"])
        assert [c for c, _ in calls] == ["home_batteries"]

    def test_no_carriers_raises(self):
        with pytest.raises(ValueError, match="nothing to import"):
            task_import_flex(object(), RunContext(raw_config={}))

    def test_unknown_carrier_raises(self):
        ctx = RunContext(raw_config={"flexibilities": ["fusion_reactors"]})
        with pytest.raises(ValueError, match="Unknown flexibilities"):
            task_import_flex(object(), ctx)


class TestBuildFlexibilityBands:
    def test_build_flexibility_bands_scopes_to_timeindex(self):
        """
        Regression test for eDisGo#703: task_build_flexibility_bands used to
        need an explicit reduce_timeseries_data_to_given_timeindex call after
        get_flexibility_bands to trim/year-align the bands to the active
        timeindex; get_flexibility_bands now does this itself, so the task
        (which no longer makes that call) must still produce bands matching
        edisgo.timeseries.timeindex exactly - including across the year
        mismatch between SimBEV's own calendar (2011 in this fixture) and
        the scenario timeindex (2035 here).
        """
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_2_path)
        electromobility_import.import_electromobility_from_dir(
            edisgo,
            pytest.simbev_example_scenario_path,
            pytest.tracbev_example_scenario_path,
        )
        electromobility_import.distribute_charging_demand(edisgo)
        electromobility_import.integrate_charging_parks(edisgo)

        short_timeindex = pd.date_range("2035-01-15", periods=24, freq="h")
        edisgo.set_timeindex(short_timeindex)

        ctx = RunContext(flags={"has_electromobility": True})
        result = task_build_flexibility_bands(edisgo, ctx)

        for key in ("upper_power", "lower_energy", "upper_energy"):
            pd.testing.assert_index_equal(
                result.electromobility.flexibility_bands[key].index,
                short_timeindex,
            )


class TestImportOverlyingGridData:
    def _ctx(self, og_cfg, overlying_grid_data=None):
        return RunContext(
            raw_config={"overlying_grid": og_cfg},
            overlying_grid_data=overlying_grid_data,
        )

    def test_disabled_returns_unchanged(self):
        """enabled: false short-circuits before the grid is even touched."""
        sentinel = object()
        ctx = self._ctx({"enabled": False})
        assert task_import_overlying_grid_data(sentinel, ctx) is sentinel

    def test_unknown_source_warns(self, edisgo_obj, caplog):
        ctx = self._ctx({"enabled": True, "source": "bogus"})
        result = task_import_overlying_grid_data(edisgo_obj, ctx)
        assert result is edisgo_obj
        assert "unknown source" in caplog.text

    def test_etrago_without_data_warns(self, edisgo_obj, caplog):
        ctx = self._ctx({"enabled": True, "source": "etrago"}, overlying_grid_data=None)
        result = task_import_overlying_grid_data(edisgo_obj, ctx)
        assert result is edisgo_obj
        assert "no" in caplog.text.lower()

    def test_etrago_empty_data_does_not_crash(self, edisgo_obj):
        """
        A partial/empty etrago dict must not raise (regression: the task used
        to call .empty on dict.get() results that were None).
        """
        ctx = self._ctx({"enabled": True, "source": "etrago"}, overlying_grid_data={})
        # must simply return without AttributeError
        assert task_import_overlying_grid_data(edisgo_obj, ctx) is edisgo_obj

    def test_csv_without_path_warns(self, edisgo_obj, caplog):
        ctx = self._ctx({"enabled": True, "source": "csv"})
        result = task_import_overlying_grid_data(edisgo_obj, ctx)
        assert result is edisgo_obj
        assert "path" in caplog.text.lower()


class TestIntervalSelectionHelpers:
    """Pure helpers for auto interval selection — no DB, no power flow."""

    @staticmethod
    def _week(start):
        return pd.date_range(start=start, periods=168, freq="h")

    def test_overlap_detection(self):
        a = self._week("2035-01-01")
        assert intervals_overlap(a, self._week("2035-01-04"))  # overlaps
        assert not intervals_overlap(a, self._week("2035-06-01"))  # disjoint

    def test_disjoint_top_intervals_kept_as_two(self):
        load = [self._week("2035-01-01")]
        volt = [self._week("2035-06-01")]
        result = select_two_intervals(load, volt)
        assert len(result) == 2
        assert not intervals_overlap(result[0], result[1])

    def test_overlap_falls_to_next_ranked_voltage(self):
        load = [self._week("2035-01-01")]
        # first voltage candidate overlaps the top loading interval, second does not
        volt = [self._week("2035-01-03"), self._week("2035-09-01")]
        result = select_two_intervals(load, volt)
        assert len(result) == 2
        assert result[1].equals(volt[1])

    def test_all_overlap_concatenates_to_one(self):
        load = [self._week("2035-01-01")]
        volt = [self._week("2035-01-03")]  # only candidate, overlaps
        result = select_two_intervals(load, volt)
        assert len(result) == 1
        merged = result[0]
        # merged interval is contiguous and spans both inputs
        assert merged.min() == load[0].min()
        assert merged.max() == volt[0].max()
        assert (merged[1:] - merged[:-1]).nunique() == 1  # regular spacing

    def test_single_side_only(self):
        week = self._week("2035-01-01")
        for result in (
            select_two_intervals([week], []),
            select_two_intervals([], [week]),
        ):
            assert len(result) == 1
            assert result[0].equals(week)
        assert select_two_intervals([], []) == []


class TestSetTimeindex:
    def test_explicit_timestamps_on_empty_index(self, edisgo_obj):
        """
        With no time index set yet (early pipeline position), set_timeindex
        sets the index and stashes it for the data imports.
        """
        edisgo_obj.set_timeindex(pd.DatetimeIndex([]))
        ts = ["2011-01-01 00:00", "2011-01-01 02:00"]
        ctx = RunContext()
        result = task_set_timeindex(edisgo_obj, ctx, timestamps=ts)
        assert list(result.timeseries.timeindex) == list(pd.to_datetime(ts))
        assert list(ctx.flags["selected_timeindex"]) == list(pd.to_datetime(ts))

    def test_range_reduces_existing_timeseries(self, edisgo_obj):
        """A start/periods range with an existing 3-step index reduces to it."""
        ctx = RunContext()
        result = task_set_timeindex(
            edisgo_obj, ctx, start="2011-01-01 00:00", periods=2
        )
        assert len(result.timeseries.timeindex) == 2

    def test_missing_args_raises(self, edisgo_obj):
        with pytest.raises(ValueError, match="timestamps.*start"):
            task_set_timeindex(edisgo_obj, RunContext())

    def test_shifts_user_timestamps_to_timeseries_year(self, edisgo_obj):
        """
        Reducing an existing (differently-yeared) time series shifts the user
        timestamps to the time-series year so slicing matches.
        """
        # existing time series in 2011
        edisgo_obj.set_timeindex(pd.date_range("2011-06-01", periods=5, freq="h"))
        # user selects timestamps written in the scenario year 2035
        ctx = RunContext()
        task_set_timeindex(
            edisgo_obj, ctx, timestamps=["2035-06-01 01:00", "2035-06-01 03:00"]
        )
        assert list(edisgo_obj.timeseries.timeindex) == [
            pd.Timestamp("2011-06-01 01:00"),
            pd.Timestamp("2011-06-01 03:00"),
        ]


class TestSelectCriticalTimesteps:
    def test_without_active_power_raises(self, edisgo_obj):
        with pytest.raises(ValueError, match="active-power time series"):
            task_select_critical_timesteps(edisgo_obj, RunContext())

    def test_unknown_method_raises(self, edisgo_obj):
        ctx = RunContext()
        ctx.flags["timeseries_set"] = True
        with pytest.raises(ValueError, match="power_flow.*residual_load"):
            task_select_critical_timesteps(edisgo_obj, ctx, method="bogus")

    def test_residual_load_requires_overlying_grid(self, edisgo_obj):
        """residual_load method must raise when no overlying-grid data is set."""
        ctx = RunContext()
        ctx.flags["timeseries_set"] = True
        with pytest.raises(ValueError, match="overlying-grid data"):
            task_select_critical_timesteps(edisgo_obj, ctx, method="residual_load")


class TestOptimizeTaskDelegation:
    """task_optimize is thin: expand the flexibility selection and call
    edisgo.pm_optimize. The multi-interval split lives in pm_optimize and is
    tested in tests/opf/test_powermodels_opf.py."""

    def _capture(self, edisgo_obj):
        captured = {}

        def fake_pm_optimize(**kw):
            captured.update(kw)

        edisgo_obj.pm_optimize = fake_pm_optimize
        return captured

    def test_expands_flexible_shortcut_and_calls_pm_optimize(self, edisgo_obj):
        edisgo_obj.set_timeindex(pd.date_range("2035-01-01", periods=24, freq="h"))
        captured = self._capture(edisgo_obj)
        task_optimize(edisgo_obj, RunContext(), flexible=["heat_pumps", "storage"])
        # shortcut expanded to explicit name lists (empty ok if grid lacks type)
        assert "flexible_hps" in captured and "flexible_storage_units" in captured
        assert isinstance(captured["flexible_hps"], list)

    def test_flexible_defaults_to_config_flexibilities(self, edisgo_obj):
        """
        Without an explicit `flexible` param, the top-level flexibilities:
        list drives the selection — carrier names are mapped to the OPF-level
        selectors (electromobility→charging_points, home_batteries→storage).
        """
        captured = self._capture(edisgo_obj)
        ctx = RunContext(
            raw_config={"flexibilities": ["electromobility", "home_batteries"]}
        )
        task_optimize(edisgo_obj, ctx)
        assert captured["flexible_cps"] == list(
            edisgo_obj.topology.loads_df.loc[
                edisgo_obj.topology.loads_df.type == "charging_point"
            ].index
        )
        assert captured["flexible_storage_units"] == list(
            edisgo_obj.topology.storage_units_df.index
        )
        # carriers not listed stay empty
        assert captured["flexible_hps"] == []


def test_all_bundled_presets_validate():
    """
    Every bundled preset must pass the (metadata-driven) validator — this
    keeps the task requires/provides declarations in sync with real configs.
    """
    presets_dir = os.path.join(os.path.dirname(edisgo_run.__file__), "presets")
    presets = sorted(glob.glob(os.path.join(presets_dir, "*.yaml")))
    assert presets, "no bundled presets found"
    for path in presets:
        validate(load_config(path))


class TestOverlyingGridKeyAliases:
    """
    The two state-of-charge frames arrive from eGo under a name without the
    ``units`` infix. They used to match no attribute and were dropped without a
    trace, which left every thermal storage starting and ending the optimisation
    horizon empty.
    """

    def _grid(self):
        edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        edisgo.set_timeindex(pd.date_range("2011-01-01", periods=3, freq="h"))
        return edisgo

    def _ctx(self, edisgo):
        ctx = RunContext()
        ctx.raw_config = {"overlying_grid": {"enabled": True, "source": "etrago"}}
        ctx.overlying_grid_data = {}
        return ctx

    def test_deprecated_key_is_read_and_reported(self, caplog):
        import logging

        edisgo = self._grid()
        ctx = self._ctx(edisgo)
        soc = pd.DataFrame({"130": [0.4, 0.5, 0.6]}, index=edisgo.timeseries.timeindex)
        ctx.overlying_grid_data = {"thermal_storage_central_soc": soc}

        with caplog.at_level(logging.WARNING, logger="edisgo.run"):
            task_import_overlying_grid_data(edisgo, ctx)

        # the data arrived under the canonical attribute name
        assert not edisgo.overlying_grid.thermal_storage_units_central_soc.empty
        assert "deprecated name" in caplog.text

    def test_attributes_that_did_not_arrive_are_named(self, caplog):
        import logging

        edisgo = self._grid()
        ctx = self._ctx(edisgo)
        ctx.overlying_grid_data = {}

        with caplog.at_level(logging.INFO, logger="edisgo.run"):
            task_import_overlying_grid_data(edisgo, ctx)

        assert "set 0 of 13 overlying-grid attributes" in caplog.text
        assert "thermal_storage_units_central_soc" in caplog.text

    def test_csv_export_written_under_the_deprecated_name_is_read(self, tmp_path):
        from edisgo.network.overlying_grid import OverlyingGrid

        ti = pd.date_range("2011-01-01", periods=3, freq="h")
        pd.DataFrame({"130": [0.4, 0.5, 0.6]}, index=ti).to_csv(
            tmp_path / "thermal_storage_central_soc.csv"
        )
        og = OverlyingGrid()
        og.from_csv(str(tmp_path))
        assert not og.thermal_storage_units_central_soc.empty
        assert list(og.thermal_storage_units_central_soc.columns) == ["130"]
