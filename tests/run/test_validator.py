"""
Unit tests for :mod:`edisgo.run.validator`.

Each test pins one ordering rule: reactive-before-TS, reinforce
without TS, optimize without flex, flex import without grid.
"""

import pytest

from edisgo.run.validator import validate


def _wrap(pipeline):
    """
    Wrap a pipeline list into a config dict.

    Parameters
    ----------
    pipeline : list
        Ordered list of task names / single-key mappings.

    Returns
    -------
    dict
        Minimal config in the shape expected by :func:`validate`.

    """
    return {"pipeline": pipeline}


def test_valid_pipeline():
    """A well-formed pipeline must pass validation without raising."""
    validate(
        _wrap(["setup_grid", "worst_case_ts", "reactive_power", "reinforce", "save"])
    )


def test_empty_pipeline_rejected():
    """An empty pipeline must be rejected."""
    with pytest.raises(ValueError, match="no pipeline"):
        validate(_wrap([]))


def test_unknown_task_rejected():
    """Typo'd task names must be rejected."""
    with pytest.raises(ValueError, match="Unknown task"):
        validate(_wrap(["setup_grid", "nonexistent_task"]))


def test_reactive_before_ts_rejected():
    """reactive_power before a TS task violates the ordering rule."""
    with pytest.raises(ValueError, match="reactive_power"):
        validate(_wrap(["setup_grid", "reactive_power", "worst_case_ts"]))


def test_select_critical_timesteps_after_reactive_rejected():
    """
    select_critical_timesteps is ts_altering (it reduces the time index),
    so it must not appear after reactive_power.
    """
    with pytest.raises(ValueError, match="reactive_power"):
        validate(
            _wrap(
                [
                    "setup_grid",
                    "worst_case_ts",
                    "reactive_power",
                    "select_critical_timesteps",
                ]
            )
        )


def test_select_critical_timesteps_before_reactive_ok():
    """select_critical_timesteps before reactive_power is the intended order."""
    validate(
        _wrap(
            [
                "setup_grid",
                "worst_case_ts",
                "select_critical_timesteps",
                "reactive_power",
                "reinforce",
                "save",
            ]
        )
    )


def test_select_critical_timesteps_without_ts_rejected():
    """select_critical_timesteps needs a time-series task before it."""
    with pytest.raises(ValueError, match="time series"):
        validate(_wrap(["setup_grid", "select_critical_timesteps"]))


def test_reinforce_without_ts_rejected():
    """reinforce without any prior time-series step must fail."""
    with pytest.raises(ValueError, match="time series"):
        validate(_wrap(["setup_grid", "reinforce"]))


def test_optimize_without_flex_rejected():
    """optimize requires at least one flex asset to be imported."""
    with pytest.raises(ValueError, match="flex asset"):
        validate(_wrap(["setup_grid", "worst_case_ts", "optimize"]))


def test_import_flex_satisfies_optimize():
    """import_flex provides the flex capability optimize requires."""
    validate(
        _wrap(
            [
                "setup_grid",
                "import_flex",
                "worst_case_ts",
                "reactive_power",
                "optimize",
            ]
        )
    )


def test_flex_import_before_grid_rejected():
    """Flex imports require a loaded grid — pre-loading is not enough."""
    with pytest.raises(ValueError, match="loaded grid"):
        validate(_wrap(["import_flex", "worst_case_ts", "reinforce"]))
