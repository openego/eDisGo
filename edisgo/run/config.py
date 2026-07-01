"""
Config loader and schema normalizer for the eDisGo pipeline runner.

The loader turns a YAML file, JSON file, or Python dict into the
canonical internal schema consumed by :mod:`edisgo.run.runner`. It
handles four concerns in a fixed order:

1. **Read** — parse YAML/JSON (auto-detected by extension; unknown
   extensions are tried as JSON first, then YAML).
2. **extends** — resolve a ``extends:`` key recursively into the
   parent config and deep-merge; the child overrides parent keys. The
   ``extends:`` value may be a path (relative to the including file)
   or a bare preset name (resolved against
   :mod:`edisgo.run.presets`).
3. **external_config** — merge machine-specific overrides from an
   ``external_config:`` path (typically ``~/.edisgo/secrets.json``
   with DB credentials). Keys in the external file override keys in
   the main config.
4. **eGo-legacy adaptation** — if the config looks like an eGo
   ``scenario_setting_*.json`` (has top-level ``eDisGo.tasks``), map
   it onto the new schema so old eGo configs run unchanged.
5. **Stage normalization** — collapse a flat ``pipeline:`` into a
   single-stage ``stages: [{name: main, pipeline: [...]}]`` so the
   runner only ever deals with the stage form.

Only :func:`load_config` is public. Everything else is implementation
detail.
"""
from __future__ import annotations

import copy
import json
import logging
import os

from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("edisgo.run.config")


def load_config(cfg_or_path) -> dict[str, Any]:
    """
    Load, merge, adapt, and normalize a pipeline config.

    Accepts a path to a YAML/JSON file or a dict. The returned dict
    always has the normalized shape expected by the runner:

    * top-level ``stages`` (list of ``{name, pipeline, ...}``)
    * ``scenario`` (may be ``None``)
    * optional ``grid``, ``database``, ``results`` sections
    * no ``pipeline``, ``extends``, or ``external_config`` keys
      (they have been consumed)

    Parameters
    ----------
    cfg_or_path : str, pathlib.Path, or dict
        Either a path to a YAML/JSON config file, or a dict already
        holding the config. A dict is deep-copied so the caller's
        dict is not mutated.

    Returns
    -------
    dict
        The fully resolved, normalized config.

    Raises
    ------
    FileNotFoundError
        If the given path (or an ``extends`` reference) does not
        exist.
    ValueError
        If the config has both ``pipeline`` and ``stages``, missing
        ``pipeline``/``stages``, duplicate stage names, or a stage
        without ``name``/``pipeline``.

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
    cfg = _apply_external_config(cfg)
    cfg = _adapt_ego_legacy(cfg)
    cfg = _normalize_stages(cfg)
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
        Preset identifier, e.g. ``"uc2_flex_opf"`` or
        ``"presets/uc2_flex_opf.yaml"``.

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


def _apply_external_config(cfg: dict) -> dict:
    """
    Merge an ``external_config:`` file on top of the current config.

    Used to keep machine-specific secrets (DB credentials, result
    directories) out of versioned scenario configs. If the referenced
    file does not exist, a warning is logged but the config is used
    as-is.

    Parameters
    ----------
    cfg : dict
        Config possibly containing an ``external_config:`` key.

    Returns
    -------
    dict
        Merged config with ``external_config`` consumed.

    """
    ext = cfg.pop("external_config", None)
    if ext is None:
        return cfg
    path = Path(os.path.expanduser(ext))
    if not path.is_file():
        logger.warning(f"external_config file not found, skipping: {path}")
        return cfg
    override = _read_file(path)
    return _deep_merge(cfg, override)


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
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _normalize_stages(cfg: dict) -> dict:
    """
    Collapse a flat ``pipeline:`` into the canonical ``stages`` shape.

    After this step the runner only has to iterate ``cfg["stages"]``;
    flat configs become a single stage named ``main``.

    Parameters
    ----------
    cfg : dict
        Config with either ``pipeline`` or ``stages`` at the top
        level.

    Returns
    -------
    dict
        Config with ``stages`` guaranteed to be present and
        ``pipeline`` removed.

    Raises
    ------
    ValueError
        If both ``pipeline`` and ``stages`` are present, if neither
        is present, if any stage is missing ``name``/``pipeline``, or
        if stage names are not unique.

    """
    if "stages" in cfg and "pipeline" in cfg:
        raise ValueError(
            "Config has both top-level 'pipeline' and 'stages'. "
            "Use only one."
        )
    if "stages" not in cfg:
        pipeline = cfg.pop("pipeline", None)
        if pipeline is None:
            raise ValueError(
                "Config must define either 'pipeline' or 'stages'."
            )
        cfg["stages"] = [{"name": "main", "pipeline": pipeline}]

    seen = set()
    for stage in cfg["stages"]:
        if "name" not in stage:
            raise ValueError("Every stage needs a 'name' key.")
        if stage["name"] in seen:
            raise ValueError(
                f"Duplicate stage name: {stage['name']}"
            )
        seen.add(stage["name"])
        if "pipeline" not in stage:
            raise ValueError(
                f"Stage '{stage['name']}' is missing 'pipeline'."
            )
    return cfg


_EGO_TASK_MAP = {
    "1_setup_grid": "setup_grid",
    "5_grid_reinforcement": "reinforce",
    "4_optimisation": "optimize",
    "worst_case_ts": "worst_case_ts",
    "base_reinforce": "base_reinforce",
    "oedb_ts": "oedb_ts",
    "import_heat_pumps_from_db": "import_heat_pumps",
    "import_home_batteries_from_db": "import_home_batteries",
    "import_dsm_from_db": "import_dsm",
    "import_electromobility_from_db": "import_electromobility",
    "load_charging_from_files": "load_charging_from_files",
    "load_from_base": "load_from_base",
}
"""Mapping from eGo task names to edisgo.run task names. eGo-specific
tasks with no eDisGo equivalent (e.g. ``2_specs_overlying_grid``,
``3_temporal_complexity_reduction``) are intentionally missing — they
require eTraGo and are logged as "skipped" when adapted."""


def _adapt_ego_legacy(cfg: dict) -> dict:
    """
    Map an eGo-style ``scenario_setting_*.json`` onto the new schema.

    Recognizes an eGo config by the presence of an ``eDisGo.tasks``
    key at the top level together with the absence of
    ``pipeline``/``stages``. Translates:

    * ``eDisGo.grid_path`` → ``grid.ding0_path``
    * ``eDisGo.results`` → ``results.directory``
    * ``eTraGo.scn_name`` → ``scenario``
    * ``eDisGo.tasks`` → ``pipeline`` (via :data:`_EGO_TASK_MAP`)
    * top-level ``database``/``ssh`` kept under ``database``

    eGo-only tasks (overlying grid / temporal reduction) are
    dropped with a warning. Cosmetic keys (``eGo``, ``eTraGo``,
    ``_comment``, ``_workflow``) are stripped.

    Parameters
    ----------
    cfg : dict
        Possibly-legacy config.

    Returns
    -------
    dict
        Adapted config. If the input is not an eGo-legacy config, it
        is returned unchanged.

    """
    if "eDisGo" not in cfg or "pipeline" in cfg or "stages" in cfg:
        return cfg

    edisgo_cfg = cfg["eDisGo"]
    tasks = edisgo_cfg.get("tasks")
    if tasks is None:
        return cfg

    logger.info(
        "Detected legacy eGo config schema — adapting to edisgo.run."
    )
    mapped = []
    for t in tasks:
        if t not in _EGO_TASK_MAP:
            logger.warning(
                f"eGo task '{t}' has no eDisGo equivalent — skipping "
                "(likely eTraGo-specific)."
            )
            continue
        mapped.append(_EGO_TASK_MAP[t])

    adapted: dict[str, Any] = {
        "scenario": cfg.get("eTraGo", {}).get("scn_name", "eGon2035"),
        "grid": {"ding0_path": edisgo_cfg.get("grid_path")},
        "results": {"directory": edisgo_cfg.get("results")},
        "pipeline": mapped,
        "overlying_grid": {
            "path": edisgo_cfg.get("overlying_grid_source"),
            "selection": edisgo_cfg.get("overlying_grid"),
        },
    }
    if "database" in cfg:
        # Deep-copy so injecting ssh below does not mutate the caller's
        # cfg["database"] (which is merged again in _deep_merge afterwards).
        adapted["database"] = copy.deepcopy(cfg["database"])
        if "ssh" in cfg:
            adapted["database"]["ssh"] = copy.deepcopy(cfg["ssh"])
    for side_key in ("eGo", "eTraGo", "ssh", "_comment", "_workflow"):
        cfg.pop(side_key, None)
    cfg.pop("eDisGo", None)
    return _deep_merge(adapted, cfg)
