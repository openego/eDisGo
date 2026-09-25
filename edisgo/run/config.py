# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Config loader for the eDisGo pipeline runner.

The loader turns a YAML file, JSON file, or Python dict into the
canonical internal schema consumed by :mod:`edisgo.run.runner`. It
handles two concerns in a fixed order:

1. **Read** — parse YAML/JSON (auto-detected by extension; unknown
   extensions are tried as JSON first, then YAML).
2. **extends** — resolve an ``extends:`` key recursively into the
   parent config and deep-merge; the child overrides parent keys. The
   ``extends:`` value may be a path (relative to the including file)
   or a bare preset name (resolved against
   :mod:`edisgo.run.presets`).

A config is a flat, single ``pipeline:`` list plus supplementary
top-level sections (``scenario``, ``grid``, ``database``,
``flexibilities``, ``overlying_grid``, ``results``). Multi-phase
workflows are separate runner calls; a run can pick up a previously
saved grid via the ``load_from_base`` task.

Only :func:`load_config` is public. Everything else is implementation
detail.
"""

from __future__ import annotations

import copy
import json
import logging

from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("edisgo.run.config")


def load_config(cfg_or_path) -> dict[str, Any]:
    """
    Load, merge, and check a pipeline config.

    Accepts a path to a YAML/JSON file or a dict. The returned dict
    always has the shape expected by the runner:

    * top-level ``pipeline`` (list of task steps)
    * optional ``scenario``, ``grid``, ``database``, ``flexibilities``,
      ``overlying_grid``, ``results`` sections
    * no ``extends`` key (it has been consumed)

    Parameters
    ----------
    cfg_or_path : str, pathlib.Path, or dict
        Either a path to a YAML/JSON config file, or a dict already
        holding the config. A dict is deep-copied so the caller's
        dict is not mutated.

    Returns
    -------
    dict
        The fully resolved config.

    Raises
    ------
    FileNotFoundError
        If the given path (or an ``extends`` reference) does not
        exist.
    ValueError
        If the config has no ``pipeline`` list, or uses a removed
        schema (``stages``, ``external_config``, eGo
        ``scenario_setting_*.json``).

    """
    if isinstance(cfg_or_path, (dict,)):
        cfg = copy.deepcopy(cfg_or_path)
        base_dir = Path.cwd()
    else:
        path = Path(cfg_or_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        cfg = _read_file(path)
        base_dir = path.parent

    cfg = _resolve_extends(cfg, base_dir)
    _check_schema(cfg)
    return cfg


def _read_file(path: Path) -> dict[str, Any]:
    """
    Parse a YAML or JSON file into a dict.

    Parameters
    ----------
    path : pathlib.Path
        File path. Extension (``.json``, ``.yaml``, ``.yml``) selects
        the parser. Unknown extensions fall back to JSON first, then
        YAML.

    Returns
    -------
    dict
        Parsed config contents.

    """
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _resolve_extends(cfg: dict, base_dir: Path) -> dict:
    """
    Resolve an ``extends:`` reference and deep-merge parent into child.

    The parent is loaded recursively, so a chain of ``extends:`` works.
    A relative reference is looked up as (1) a path relative to
    ``base_dir``, (2) a bundled preset name under
    :mod:`edisgo.run.presets`. The child's keys override the parent's on
    conflicts.

    Parameters
    ----------
    cfg : dict
        Child config (may contain ``extends:``).
    base_dir : pathlib.Path
        Directory against which relative ``extends`` paths are
        resolved (usually the directory of the child config).

    Returns
    -------
    dict
        Merged config with ``extends`` consumed.

    Raises
    ------
    FileNotFoundError
        If the referenced parent config file does not exist.

    """
    ext = cfg.pop("extends", None)
    if ext is None:
        return cfg
    ext_path = Path(ext).expanduser()
    if not ext_path.is_absolute():
        # Resolve relative to the including file first (least surprise: a
        # local file next to the config wins), then fall back to a bundled
        # preset of that name.
        local_path = (base_dir / ext_path).resolve()
        if local_path.is_file():
            ext_path = local_path
        else:
            preset_path = _preset_path(str(ext_path))
            ext_path = preset_path if preset_path is not None else local_path
    if not ext_path.is_file():
        raise FileNotFoundError(f"extends: file not found: {ext_path}")
    parent = _read_file(ext_path)
    parent = _resolve_extends(parent, ext_path.parent)
    return _deep_merge(parent, cfg)


def _preset_path(name: str) -> Path | None:
    """
    Look up a preset YAML/JSON by bare name.

    Searches the ``edisgo/run/presets/`` directory for a file matching
    ``name``, ``name.yaml``, ``name.yml``, or ``name.json`` (in that
    order).

    Parameters
    ----------
    name : str
        Preset identifier, e.g. ``"flex_opf"`` or
        ``"presets/flex_opf.yaml"``.

    Returns
    -------
    pathlib.Path or None
        The resolved preset path, or ``None`` if no match is found.

    """
    presets_dir = Path(__file__).parent / "presets"
    candidates = [
        presets_dir / name,
        presets_dir / f"{name}.yaml",
        presets_dir / f"{name}.yml",
        presets_dir / f"{name}.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dicts, with ``override`` winning on conflicts.

    Nested dicts are merged key-by-key. Non-dict values (including
    lists) are replaced wholesale — lists are NOT concatenated, to
    keep the merge semantics predictable (otherwise a preset could
    silently extend the child's pipeline).

    Parameters
    ----------
    base : dict
        Parent / lower-priority dict.
    override : dict
        Child / higher-priority dict.

    Returns
    -------
    dict
        A new dict holding the merge result. Inputs are not mutated.

    """
    out = copy.deepcopy(base) if base else {}
    for key, val in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _check_schema(cfg: dict) -> None:
    """
    Reject removed config schemas with a pointed error message.

    Parameters
    ----------
    cfg : dict
        Fully merged config.

    Raises
    ------
    ValueError
        If the config uses ``stages`` or ``external_config`` (both
        removed), looks like an eGo ``scenario_setting_*.json``
        (top-level ``eDisGo`` section — no longer auto-adapted), or
        is missing the ``pipeline`` list.

    """
    if "stages" in cfg:
        raise ValueError(
            "Config uses 'stages', which has been removed. Use a single "
            "flat 'pipeline' list; multi-phase workflows are separate "
            "run_edisgo calls (reload a saved grid with the "
            "'load_from_base' task)."
        )
    if "external_config" in cfg:
        raise ValueError(
            "Config uses 'external_config', which has been removed. "
            "Database credentials live in the egon-data configuration "
            "file / OEP token file; machine-specific paths belong in the "
            "caller's config or an 'extends' override."
        )
    if "eDisGo" in cfg and "pipeline" not in cfg:
        raise ValueError(
            "Config looks like a legacy eGo scenario_setting JSON "
            "(top-level 'eDisGo' section). This schema is no longer "
            "supported — provide a config with a top-level 'pipeline' "
            "list instead (see edisgo/run/presets/ for examples)."
        )
    pipeline = cfg.get("pipeline")
    if not isinstance(pipeline, list) or not pipeline:
        raise ValueError("Config must define a non-empty 'pipeline' list.")
