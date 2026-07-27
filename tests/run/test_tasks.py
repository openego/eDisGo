"""
Unit tests for individual pipeline tasks in :mod:`edisgo.run.tasks`.

These cover the task control flow that unit tests previously missed — the
task modules were the source of every bug found in the review. They run
without a database or SSH tunnel: a small self-constructed ding0 grid is
enough, and the DB-free branches of ``import_overlying_grid_data`` are
exercised directly.
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
from edisgo.run.tasks.analysis import task_optimize
from edisgo.run.tasks.flex import task_build_flexibility_bands
from edisgo.run.tasks.io import task_import_overlying_grid_data
from edisgo.run.tasks.timeseries import (
    task_manual_ts,
    task_select_timesteps,
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
        task_manual_ts must forward the eGo-style ``*_active_power`` args to
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

        ctx = RunContext()
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


class TestSelectTimestepsHelpers:
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


class TestSelectTimestepsManual:
    def test_manual_explicit_timestamps_before_imports(self, edisgo_obj):
        """
        Manual mode with an empty timeseries (positioned before imports) sets
        the index and stashes it for HP/DSM imports.
        """
        # start from an empty time index to mimic the pre-import position
        edisgo_obj.set_timeindex(pd.DatetimeIndex([]))
        ts = ["2011-01-01 00:00", "2011-01-01 02:00"]
        ctx = RunContext(
            raw_config={"timeseries_selection": {"mode": "manual", "timestamps": ts}}
        )
        result = task_select_timesteps(edisgo_obj, ctx)
        assert list(result.timeseries.timeindex) == list(pd.to_datetime(ts))
        assert list(ctx.flags["selected_timeindex"]) == list(pd.to_datetime(ts))
        assert ctx.flags["timesteps_selected"] is True

    def test_manual_range_reduces_existing_timeseries(self, edisgo_obj):
        """Manual range with an existing 3-step index reduces to the range."""
        ctx = RunContext(
            raw_config={
                "timeseries_selection": {
                    "mode": "manual",
                    "start": "2011-01-01 00:00",
                    "periods": 2,
                    "freq": "h",
                }
            }
        )
        result = task_select_timesteps(edisgo_obj, ctx)
        assert len(result.timeseries.timeindex) == 2

    def test_missing_mode_raises(self, edisgo_obj):
        with pytest.raises(ValueError, match="mode 'manual' or 'auto'"):
            task_select_timesteps(edisgo_obj, RunContext(raw_config={}))

    def test_auto_without_active_power_raises(self, edisgo_obj):
        ctx = RunContext(raw_config={"timeseries_selection": {"mode": "auto"}})
        with pytest.raises(ValueError, match="active-power time series"):
            task_select_timesteps(edisgo_obj, ctx)

    def test_auto_unknown_method_raises(self, edisgo_obj):
        ctx = RunContext(
            raw_config={
                "timeseries_selection": {
                    "mode": "auto",
                    "method": "bogus",
                }
            }
        )
        ctx.flags["timeseries_set"] = True
        with pytest.raises(ValueError, match="power_flow.*residual_load"):
            task_select_timesteps(edisgo_obj, ctx)

    def test_residual_load_requires_overlying_grid(self, edisgo_obj):
        """residual_load method must raise when no overlying-grid data is set."""
        ctx = RunContext(
            raw_config={
                "timeseries_selection": {
                    "mode": "auto",
                    "method": "residual_load",
                }
            }
        )
        ctx.flags["timeseries_set"] = True
        with pytest.raises(ValueError, match="overlying-grid data"):
            task_select_timesteps(edisgo_obj, ctx)


class TestSelectTimestepsPosition:
    """The `position` param lets one pipeline carry both a pre-import and a
    post-grid select_timesteps step; each no-ops off its mode."""

    def test_pre_import_noops_in_auto_mode(self, edisgo_obj):
        """
        A pre_import step in auto mode must be a no-op — crucially it must NOT
        hit the auto guard (no active power set yet), it just returns.
        """
        before = edisgo_obj.timeseries.timeindex
        ctx = RunContext(raw_config={"timeseries_selection": {"mode": "auto"}})
        result = task_select_timesteps(edisgo_obj, ctx, position="pre_import")
        assert result is edisgo_obj
        assert result.timeseries.timeindex.equals(before)
        assert "timesteps_selected" not in ctx.flags

    def test_post_grid_noops_in_manual_mode(self, edisgo_obj):
        """A post_grid step in manual mode must be a no-op (manual already ran
        earlier at pre_import)."""
        before = edisgo_obj.timeseries.timeindex
        ctx = RunContext(
            raw_config={
                "timeseries_selection": {
                    "mode": "manual",
                    "timestamps": ["2011-01-01 00:00"],
                }
            }
        )
        result = task_select_timesteps(edisgo_obj, ctx, position="post_grid")
        assert result is edisgo_obj
        assert result.timeseries.timeindex.equals(before)

    def test_pre_import_acts_in_manual_mode(self, edisgo_obj):
        edisgo_obj.set_timeindex(pd.DatetimeIndex([]))
        ctx = RunContext(
            raw_config={
                "timeseries_selection": {
                    "mode": "manual",
                    "timestamps": ["2011-01-01 00:00"],
                }
            }
        )
        task_select_timesteps(edisgo_obj, ctx, position="pre_import")
        assert len(edisgo_obj.timeseries.timeindex) == 1
        assert ctx.flags["timesteps_selected"] is True

    def test_bad_position_raises(self, edisgo_obj):
        ctx = RunContext(raw_config={"timeseries_selection": {"mode": "manual"}})
        with pytest.raises(ValueError, match="position"):
            task_select_timesteps(edisgo_obj, ctx, position="bogus")

    def test_pre_import_sets_default_index_when_empty_in_auto(self, edisgo_obj):
        """
        A pre_import step in auto mode with no time index set must establish a
        full-year default (so later imports build hourly full-year data), even
        though it otherwise no-ops.
        """
        edisgo_obj.set_timeindex(pd.DatetimeIndex([]))
        ctx = RunContext(
            scenario="eGon2035",
            raw_config={"timeseries_selection": {"mode": "auto"}},
        )
        task_select_timesteps(edisgo_obj, ctx, position="pre_import")
        ti = edisgo_obj.timeseries.timeindex
        assert len(ti) == 8760
        assert ti[0].year == 2035
        # still a no-op for actual selection
        assert "timesteps_selected" not in ctx.flags

    def test_manual_shifts_user_timestamps_to_timeseries_year(self, edisgo_obj):
        """
        Manual mode reducing an existing (differently-yeared) time series shifts
        the user timestamps to the time-series year so slicing matches.
        """
        # existing time series in 2011
        edisgo_obj.set_timeindex(pd.date_range("2011-06-01", periods=5, freq="h"))
        # user selects timestamps written in the scenario year 2035
        ctx = RunContext(
            raw_config={
                "timeseries_selection": {
                    "mode": "manual",
                    "timestamps": ["2035-06-01 01:00", "2035-06-01 03:00"],
                }
            }
        )
        task_select_timesteps(edisgo_obj, ctx)
        ti = edisgo_obj.timeseries.timeindex
        assert list(ti) == [
            pd.Timestamp("2011-06-01 01:00"),
            pd.Timestamp("2011-06-01 03:00"),
        ]


class TestOptimizeTaskDelegation:
    """task_optimize is thin: expand the `flexible` shortcut and call
    edisgo.pm_optimize. The multi-interval split lives in pm_optimize and is
    tested in tests/opf/test_powermodels_opf.py."""

    def test_expands_flexible_shortcut_and_calls_pm_optimize(self, edisgo_obj):
        edisgo_obj.set_timeindex(pd.date_range("2035-01-01", periods=24, freq="h"))
        captured = {}

        def fake_pm_optimize(**kw):
            captured.update(kw)

        edisgo_obj.pm_optimize = fake_pm_optimize
        task_optimize(edisgo_obj, RunContext(), flexible=["heat_pumps", "storage"])
        # shortcut expanded to explicit name lists (empty ok if grid lacks type)
        assert "flexible_hps" in captured and "flexible_storage_units" in captured
        assert isinstance(captured["flexible_hps"], list)


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
