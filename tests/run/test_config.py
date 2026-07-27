"""
Unit tests for :mod:`edisgo.run.config` — loader, merger, adapter.

Covers YAML/JSON parity, ``extends`` resolution (preset-by-name and
relative paths), deep-merge semantics, stage normalization, and the
eGo-legacy adapter.
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


def test_load_flat_pipeline_normalized_to_stages(tmp_path):
    """A flat ``pipeline:`` must normalize to a single 'main' stage."""
    p = _write(tmp_path, "cfg.yaml", {
        "scenario": "eGon2035",
        "pipeline": ["setup_grid", "worst_case_ts", "reinforce"],
    })
    cfg = load_config(str(p))
    assert "pipeline" not in cfg
    assert cfg["stages"] == [
        {"name": "main",
         "pipeline": ["setup_grid", "worst_case_ts", "reinforce"]}
    ]


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
    parent = _write(tmp_path, "parent.yaml", {
        "scenario": "eGon2035",
        "grid": {"legacy_ding0_grids": False},
        "pipeline": ["setup_grid", "reinforce"],
    })
    child = _write(tmp_path, "child.yaml", {
        "extends": str(parent),
        "grid": {"ding0_path": "/tmp/xyz"},
    })
    cfg = load_config(str(child))
    assert cfg["scenario"] == "eGon2035"
    assert cfg["grid"] == {
        "legacy_ding0_grids": False, "ding0_path": "/tmp/xyz"
    }
    assert cfg["stages"][0]["pipeline"] == ["setup_grid", "reinforce"]


def test_extends_preset_by_name(tmp_path):
    """``extends: basic`` must resolve to the bundled basic preset."""
    child = _write(tmp_path, "child.yaml", {
        "extends": "basic",
        "grid": {"ding0_path": "/tmp/xyz"},
    })
    cfg = load_config(str(child))
    assert "stages" in cfg
    assert cfg["grid"]["ding0_path"] == "/tmp/xyz"


def test_deep_merge_nested():
    """Nested dicts must be merged key-by-key, child wins on conflict."""
    base = {"a": {"b": 1, "c": 2}, "d": 4}
    over = {"a": {"b": 99, "e": 5}}
    merged = _deep_merge(base, over)
    assert merged == {"a": {"b": 99, "c": 2, "e": 5}, "d": 4}


def test_both_pipeline_and_stages_rejected(tmp_path):
    """Top-level ``pipeline`` and ``stages`` are mutually exclusive."""
    p = _write(tmp_path, "cfg.yaml", {
        "pipeline": ["setup_grid"],
        "stages": [{"name": "x", "pipeline": ["setup_grid"]}],
    })
    with pytest.raises(ValueError, match="both"):
        load_config(str(p))


def test_duplicate_stage_names_rejected(tmp_path):
    """Stage names must be unique; duplicates raise ValueError."""
    p = _write(tmp_path, "cfg.yaml", {
        "stages": [
            {"name": "x", "pipeline": ["setup_grid"]},
            {"name": "x", "pipeline": ["reinforce"]},
        ],
    })
    with pytest.raises(ValueError, match="Duplicate stage"):
        load_config(str(p))


def test_ego_legacy_adapter(tmp_path):
    """An eGo ``scenario_setting_*.json`` must adapt to the new schema."""
    ego_cfg = {
        "eGo": {"eDisGo": True},
        "eTraGo": {"scn_name": "eGon2035"},
        "eDisGo": {
            "grid_path": "/some/path",
            "results": "/tmp/results",
            "tasks": [
                "1_setup_grid",
                "base_reinforce",
                "import_heat_pumps_from_db",
                "worst_case_ts",
                "5_grid_reinforcement",
            ],
        },
        "database": {"host": "localhost"},
    }
    p = _write(tmp_path, "legacy.json", ego_cfg)
    cfg = load_config(str(p))
    assert cfg["scenario"] == "eGon2035"
    assert cfg["grid"]["ding0_path"] == "/some/path"
    assert cfg["stages"][0]["pipeline"] == [
        "setup_grid", "base_reinforce", "import_heat_pumps",
        "worst_case_ts", "reinforce",
    ]
    assert cfg["database"]["host"] == "localhost"
