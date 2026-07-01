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
from edisgo.run.config import load_config
from edisgo.run.context import RunContext
from edisgo.run.tasks.io import task_import_overlying_grid_data
from edisgo.run.tasks.timeseries import task_manual_ts
from edisgo.run.validator import validate


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
        ctx = self._ctx({"enabled": True, "source": "etrago"},
                        overlying_grid_data=None)
        result = task_import_overlying_grid_data(edisgo_obj, ctx)
        assert result is edisgo_obj
        assert "no" in caplog.text.lower()

    def test_etrago_empty_data_does_not_crash(self, edisgo_obj):
        """
        A partial/empty etrago dict must not raise (regression: the task used
        to call .empty on dict.get() results that were None).
        """
        ctx = self._ctx({"enabled": True, "source": "etrago"},
                        overlying_grid_data={})
        # must simply return without AttributeError
        assert task_import_overlying_grid_data(edisgo_obj, ctx) is edisgo_obj

    def test_csv_without_path_warns(self, edisgo_obj, caplog):
        ctx = self._ctx({"enabled": True, "source": "csv"})
        result = task_import_overlying_grid_data(edisgo_obj, ctx)
        assert result is edisgo_obj
        assert "path" in caplog.text.lower()


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
