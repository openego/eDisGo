"""
Unit tests for :mod:`edisgo.run.validator`.

Each test pins one ordering rule: reactive-before-TS, reinforce
without TS, optimize without flex, flex import without grid, and the
stage-level ``load_from`` constraints.
"""
import pytest

from edisgo.run.validator import validate


def _wrap(pipeline):
    """
    Wrap a flat pipeline into a single-stage config dict.

    Parameters
    ----------
    pipeline : list
        Ordered list of task names / single-key mappings.

    Returns
    -------
    dict
        Minimal config in the shape expected by :func:`validate`.

    """
    return {"stages": [{"name": "main", "pipeline": pipeline}]}


def test_valid_pipeline():
    """A well-formed pipeline must pass validation without raising."""
    validate(_wrap(["setup_grid", "worst_case_ts", "reactive_power",
                    "reinforce", "save"]))


def test_unknown_task_rejected():
    """Typo'd task names must be rejected."""
    with pytest.raises(ValueError, match="Unknown task"):
        validate(_wrap(["setup_grid", "nonexistent_task"]))


def test_reactive_before_ts_rejected():
    """reactive_power before a TS task violates the ordering rule."""
    with pytest.raises(ValueError, match="reactive_power"):
        validate(_wrap(["setup_grid", "reactive_power", "worst_case_ts"]))


def test_reinforce_without_ts_rejected():
    """reinforce without any prior time-series step must fail."""
    with pytest.raises(ValueError, match="time series"):
        validate(_wrap(["setup_grid", "reinforce"]))


def test_optimize_without_flex_rejected():
    """optimize requires at least one flex asset to be imported."""
    with pytest.raises(ValueError, match="flex asset"):
        validate(_wrap(["setup_grid", "worst_case_ts", "optimize"]))


def test_flex_import_before_grid_rejected():
    """Flex imports require a loaded grid — pre-loading is not enough."""
    with pytest.raises(ValueError, match="loaded grid"):
        validate(_wrap(["import_heat_pumps", "worst_case_ts", "reinforce"]))


def test_stage_load_from_missing_rejected():
    """``load_from: X`` where X has not run must fail."""
    cfg = {"stages": [
        {"name": "a", "pipeline": ["setup_grid", "worst_case_ts",
                                    "reinforce"]},
        {"name": "b", "load_from": "nonexistent",
         "pipeline": ["reinforce"]},
    ]}
    with pytest.raises(ValueError, match="load_from"):
        validate(cfg)


def test_stage_load_from_requires_save_in_source():
    """A stage consumed by ``load_from`` must itself end with ``save``."""
    cfg = {"stages": [
        {"name": "a", "pipeline": ["setup_grid", "worst_case_ts",
                                    "reinforce"]},  # no save
        {"name": "b", "load_from": "a", "pipeline": ["reinforce"]},
    ]}
    with pytest.raises(ValueError, match="load_from"):
        validate(cfg)


def test_stage_load_from_with_save_ok():
    """Stage chain with a save in the source must validate successfully."""
    cfg = {"stages": [
        {"name": "a", "pipeline": ["setup_grid", "worst_case_ts",
                                    "reinforce", "save"]},
        {"name": "b", "load_from": "a", "pipeline": ["reinforce", "save"]},
    ]}
    validate(cfg)
