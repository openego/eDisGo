"""
End-to-end tests for the eDisGo pipeline runner.

Uses the small test grid under ``tests/data/ding0_test_network_2``
(exposed by :mod:`tests.conftest` as
``pytest.ding0_test_network_2_path``) to run full pipelines without
touching the database. Covers:

* the standalone ``run_edisgo`` entry point with a flat pipeline,
* the instance method ``EDisGo.run_pipeline``,
* the stage mechanism with ``save`` + ``load_from``.
"""
import os

import pytest

from edisgo.run import run_edisgo


@pytest.fixture
def basic_cfg(tmp_path):
    """
    Minimal end-to-end config fixture.

    Produces a config that loads the small ding0 test grid, sets
    worst-case time series, fixes reactive power, checks integrity,
    runs reinforcement, and saves — no database needed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temp directory for the run's artifacts.

    Returns
    -------
    dict
        The config dict.

    """
    return {
        "scenario": "eGon2035",
        "grid": {
            "ding0_path": pytest.ding0_test_network_2_path,
            "legacy_ding0_grids": True,
        },
        "results": {"directory": str(tmp_path)},
        "pipeline": [
            "setup_grid",
            "worst_case_ts",
            "reactive_power",
            "check_integrity",
            "reinforce",
            "save",
        ],
    }


def test_runner_basic_end_to_end(basic_cfg):
    """A flat-pipeline run must execute and persist the expected artifact."""
    edisgo = run_edisgo(basic_cfg)
    assert edisgo is not None
    assert edisgo.topology is not None
    assert os.path.isdir(os.path.join(basic_cfg["results"]["directory"],
                                       "main"))


def test_runner_method_on_edisgo(basic_cfg):
    """``EDisGo.run_pipeline`` must operate on the existing instance."""
    from edisgo import EDisGo

    basic_cfg["pipeline"] = basic_cfg["pipeline"][1:]  # skip setup_grid
    edisgo = EDisGo(
        ding0_grid=basic_cfg["grid"]["ding0_path"],
        legacy_ding0_grids=True,
    )
    edisgo = edisgo.run_pipeline(basic_cfg)
    assert edisgo.topology is not None


def test_runner_two_stages_with_load_from(tmp_path):
    """
    A two-stage run must save the first stage and reload it via
    ``load_from`` in the second stage, producing both artifacts.
    """
    cfg = {
        "scenario": "eGon2035",
        "grid": {
            "ding0_path": pytest.ding0_test_network_2_path,
            "legacy_ding0_grids": True,
        },
        "results": {"directory": str(tmp_path)},
        "stages": [
            {
                "name": "base",
                "pipeline": [
                    "setup_grid",
                    "worst_case_ts",
                    "reactive_power",
                    "reinforce",
                    {"save": {"archive": True}},
                ],
            },
            {
                "name": "scenario",
                "load_from": "base",
                "pipeline": [
                    "worst_case_ts",
                    "reactive_power",
                    "reinforce",
                    "save",
                ],
            },
        ],
    }
    edisgo = run_edisgo(cfg)
    assert edisgo.topology is not None
    assert os.path.exists(os.path.join(str(tmp_path), "base.zip"))
    assert os.path.isdir(os.path.join(str(tmp_path), "scenario"))
