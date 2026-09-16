"""
Unit tests for :mod:`edisgo.run.config` — loader and merger.

Covers YAML/JSON parity, ``extends`` resolution (preset-by-name and
relative paths), deep-merge semantics, and the rejection of removed
schemas (``stages``, ``external_config``, eGo-legacy).
"""

import json

import pytest
import yaml

from edisgo.run.config import _deep_merge, load_config


def _write(tmp_path, name, data):
    """
    Helper: write ``data`` to ``tmp_path/name`` as YAML or JSON.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    name : str
        File name with extension (``.yaml``/``.yml``/``.json``).
    data : dict
        Payload.

    Returns
    -------
    pathlib.Path
        Path to the written file.

    """
    path = tmp_path / name
    if name.endswith(".json"):
        path.write_text(json.dumps(data))
    else:
        path.write_text(yaml.safe_dump(data))
    return path


def test_load_flat_pipeline(tmp_path):
    """A flat ``pipeline:`` config must load unchanged."""
    p = _write(
        tmp_path,
        "cfg.yaml",
        {
            "scenario": "eGon2035",
            "pipeline": ["setup_grid", "worst_case_ts", "reinforce"],
        },
    )
    cfg = load_config(str(p))
    assert cfg["pipeline"] == ["setup_grid", "worst_case_ts", "reinforce"]
    assert cfg["scenario"] == "eGon2035"


def test_yaml_and_json_equivalent(tmp_path):
    """YAML and JSON payloads with identical content must load equal."""
    data = {
        "scenario": "eGon2035",
        "pipeline": ["setup_grid", "worst_case_ts", "reinforce"],
    }
    yaml_path = _write(tmp_path, "cfg.yaml", data)
    json_path = _write(tmp_path, "cfg.json", data)
    assert load_config(str(yaml_path)) == load_config(str(json_path))


def test_extends_merges_parent(tmp_path):
    """Child config must deep-merge with its ``extends:`` parent."""
    parent = _write(
        tmp_path,
        "parent.yaml",
        {
            "scenario": "eGon2035",
            "grid": {"legacy_ding0_grids": False},
            "pipeline": ["setup_grid", "reinforce"],
        },
    )
    child = _write(
        tmp_path,
        "child.yaml",
        {
            "extends": str(parent),
            "grid": {"ding0_path": "/tmp/xyz"},
        },
    )
    cfg = load_config(str(child))
    assert cfg["scenario"] == "eGon2035"
    assert cfg["grid"] == {"legacy_ding0_grids": False, "ding0_path": "/tmp/xyz"}
    assert cfg["pipeline"] == ["setup_grid", "reinforce"]


def test_extends_preset_by_name(tmp_path):
    """``extends: worst_case`` must resolve to the bundled preset."""
    child = _write(
        tmp_path,
        "child.yaml",
        {
            "extends": "worst_case",
            "grid": {"ding0_path": "/tmp/xyz"},
        },
    )
    cfg = load_config(str(child))
    assert "pipeline" in cfg
    assert cfg["grid"]["ding0_path"] == "/tmp/xyz"


def test_extends_child_replaces_pipeline(tmp_path):
    """A child ``pipeline:`` replaces the preset's list wholesale."""
    child = _write(
        tmp_path,
        "child.yaml",
        {
            "extends": "worst_case",
            "pipeline": ["setup_grid", "worst_case_ts", "reactive_power", "analyze"],
        },
    )
    cfg = load_config(str(child))
    assert cfg["pipeline"][-1] == "analyze"
    assert "reinforce" not in cfg["pipeline"]


def test_deep_merge_nested():
    """Nested dicts must be merged key-by-key, child wins on conflict."""
    base = {"a": {"b": 1, "c": 2}, "d": 4}
    over = {"a": {"b": 99, "e": 5}}
    merged = _deep_merge(base, over)
    assert merged == {"a": {"b": 99, "c": 2, "e": 5}, "d": 4}


def test_missing_pipeline_rejected(tmp_path):
    """A config without a ``pipeline`` list must be rejected."""
    p = _write(tmp_path, "cfg.yaml", {"scenario": "eGon2035"})
    with pytest.raises(ValueError, match="pipeline"):
        load_config(str(p))


def test_stages_rejected(tmp_path):
    """The removed ``stages`` schema must fail with a pointed message."""
    p = _write(
        tmp_path,
        "cfg.yaml",
        {
            "stages": [{"name": "x", "pipeline": ["setup_grid"]}],
        },
    )
    with pytest.raises(ValueError, match="stages.*removed"):
        load_config(str(p))


def test_external_config_rejected(tmp_path):
    """The removed ``external_config`` key must fail with a pointed message."""
    p = _write(
        tmp_path,
        "cfg.yaml",
        {
            "pipeline": ["setup_grid"],
            "external_config": "~/.edisgo/secrets.json",
        },
    )
    with pytest.raises(ValueError, match="external_config.*removed"):
        load_config(str(p))


def test_ego_legacy_rejected(tmp_path):
    """A legacy eGo scenario_setting JSON must fail, not silently adapt."""
    ego_cfg = {
        "eGo": {"eDisGo": True},
        "eTraGo": {"scn_name": "eGon2035"},
        "eDisGo": {
            "grid_path": "/some/path",
            "tasks": ["1_setup_grid", "worst_case_ts"],
        },
    }
    p = _write(tmp_path, "legacy.json", ego_cfg)
    with pytest.raises(ValueError, match="no longer supported"):
        load_config(str(p))


def test_unknown_top_level_key_warns(tmp_path, caplog):
    """An unread top-level key must be warned about, not silently dropped."""
    p = _write(
        tmp_path,
        "cfg.yaml",
        {
            "pipeline": ["setup_grid"],
            "timeseries_selection": {"mode": "auto"},
        },
    )
    with caplog.at_level("WARNING", logger="edisgo.run.config"):
        cfg = load_config(str(p))
    # the key is kept (the loader does not strip it) but flagged
    assert "timeseries_selection" in cfg
    assert "timeseries_selection" in caplog.text


def test_known_top_level_keys_do_not_warn(tmp_path, caplog):
    """Recognised sections must not produce a warning."""
    p = _write(
        tmp_path,
        "cfg.yaml",
        {
            "pipeline": ["setup_grid"],
            "scenario": "eGon2035",
            "grid": {"ding0_path": "/tmp/xyz"},
            "database": {"source": "oep"},
            "spatial_reduction": {"enabled": False},
            "_comment": "documentation only",
        },
    )
    with caplog.at_level("WARNING", logger="edisgo.run.config"):
        load_config(str(p))
    assert "unknown top-level" not in caplog.text.lower()


def test_caller_overrides_preset_database_and_timeindex():
    """
    A caller must be able to override a preset's ``database.source`` and the
    params of its ``set_timeindex`` step (regression for openego/eGo#207/#222).
    """
    preset = load_config({"extends": "spatial_reduction_opf"})
    # the preset's own defaults, which the caller overrides below
    assert preset["database"]["source"] == "egon-data"
    pipeline = [
        {"set_timeindex": {"start": "2035-06-01 00:00", "periods": 48}}
        if isinstance(step, dict) and "set_timeindex" in step
        else step
        for step in preset["pipeline"]
    ]

    cfg = load_config(
        {
            "extends": "spatial_reduction_opf",
            "database": {"source": "oep"},
            "pipeline": pipeline,
        }
    )
    assert cfg["database"]["source"] == "oep"
    step = next(
        s for s in cfg["pipeline"] if isinstance(s, dict) and "set_timeindex" in s
    )
    assert step["set_timeindex"] == {"start": "2035-06-01 00:00", "periods": 48}
