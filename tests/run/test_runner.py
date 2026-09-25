"""
End-to-end tests for the eDisGo pipeline runner.

Uses the small test grid under ``tests/data/ding0_test_network_2``
(exposed by :mod:`tests.conftest` as
``pytest.ding0_test_network_2_path``) to run full pipelines without
touching the database.
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
        Pytest-provided temp directory for the run's outputs.

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
    """A flat-pipeline run must execute and persist to results.directory."""
    edisgo = run_edisgo(basic_cfg)
    assert edisgo is not None
    assert edisgo.topology is not None
    assert os.path.isdir(os.path.join(basic_cfg["results"]["directory"], "topology"))


def test_runner_extends_preset(basic_cfg, tmp_path):
    """Extending the bundled worst_case preset must run end-to-end."""
    cfg = {
        "extends": "worst_case",
        "grid": basic_cfg["grid"],
        "results": {"directory": str(tmp_path / "preset_run")},
    }
    edisgo = run_edisgo(cfg)
    assert edisgo.topology is not None
    assert not edisgo.results.equipment_changes.empty
