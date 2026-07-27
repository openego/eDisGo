"""
Static validator for pipeline configs.

The validator enforces structural and ordering rules that the runner
would otherwise hit at execution time — often after 20 minutes of work.
Running these checks up-front turns "cryptic AttributeError after half
the pipeline" into a clear ``ValueError`` at startup.

Checked rules:

* every step maps to a known, registered task name;
* ``reactive_power`` comes after every time-series task in a stage,
  never before — ``set_time_series_reactive_power_control`` overwrites
  reactive power on the currently set active-power time series;
* ``analyze`` and ``reinforce`` require a time-series task earlier in
  the stage (or a ``load_from:`` that brings a prepared grid);
* ``optimize`` requires both a time-series task and at least one flex
  import earlier in the stage — OPF without flexibility is meaningless;
* flex imports (``import_heat_pumps``, …) require a loaded grid, i.e.
  an earlier ``setup_grid`` / ``load_from_base`` / a stage-level
  ``load_from:``;
* ``base_reinforce`` likewise requires a loaded grid;
* a stage that declares ``load_from: X`` can only run if stage ``X``
  ran earlier AND contains a ``save`` step.
"""
from __future__ import annotations

from typing import Any

from edisgo.run.registry import get_task_meta, known_tasks

# Human-readable message per required capability. The wording keeps the
# substrings the validator tests assert on ("loaded grid", "time series",
# "flex asset").
_REQUIREMENT_MESSAGES = {
    "grid": "requires a loaded grid (setup_grid or load_from_base) before it",
    "timeseries": (
        "requires time series to be set (e.g. worst_case_ts or oedb_ts) "
        "before it"
    ),
    "flex": "requires at least one flex asset to be imported",
}
# Order in which a missing capability is reported when several are missing.
_REQUIREMENT_PRIORITY = ("grid", "timeseries", "flex")


def validate(cfg: dict) -> None:
    """
    Validate a normalized pipeline config against the ordering rules.

    This function does not return a value. On success it simply
    returns; on any rule violation it raises :class:`ValueError` with
    a message identifying the offending stage and task.

    Parameters
    ----------
    cfg : dict
        Normalized config as returned by
        :func:`edisgo.run.config.load_config`. Must have a ``stages``
        list at the top level.

    Raises
    ------
    ValueError
        If the config has no stages, an unknown task name, a
        structural problem (reactive before TS, reinforce without TS,
        optimize without flex, flex import without grid, …), or a
        stage references a ``load_from`` source that doesn't exist or
        has no ``save`` step.

    """
    stages = cfg.get("stages") or []
    if not stages:
        raise ValueError("Config has no stages to run.")

    known = set(known_tasks())
    available_artifacts: set[str] = set()

    for stage in stages:
        name = stage["name"]
        pipeline = stage.get("pipeline") or []
        load_from = stage.get("load_from")

        if load_from is not None and load_from not in available_artifacts:
            raise ValueError(
                f"Stage '{name}' requires 'load_from: {load_from}' but "
                f"that stage has not run or did not save. Available: "
                f"{sorted(available_artifacts)}"
            )

        # Capabilities established so far in this stage. A stage-level
        # load_from reloads the grid topology only — _load_artifact drops
        # time series and flex data (import_timeseries=False) — so it
        # provides "grid" but NOT "timeseries"/"flex". A task's requirements
        # must therefore be satisfied by tasks run in this stage itself.
        satisfied: set[str] = {"grid"} if load_from is not None else set()
        reactive_set = False
        has_save = False

        for step in pipeline:
            task_name, _params = _split_step(step)
            if task_name not in known:
                raise ValueError(
                    f"Unknown task '{task_name}' in stage '{name}'. "
                    f"Known: {sorted(known)}"
                )

            meta = get_task_meta(task_name)

            # reactive_power must be the last time-series-altering step.
            if meta.ts_altering and reactive_set:
                raise ValueError(
                    f"Stage '{name}': time-series task '{task_name}' comes "
                    f"after 'reactive_power' — reactive_power must be the "
                    f"last time-series-altering step."
                )

            # Check declared requirements against what the stage provides.
            missing = meta.requires - satisfied
            if missing:
                cap = next(
                    (c for c in _REQUIREMENT_PRIORITY if c in missing),
                    sorted(missing)[0],
                )
                detail = _REQUIREMENT_MESSAGES.get(
                    cap, f"requires '{cap}' to be established before it"
                )
                raise ValueError(
                    f"Stage '{name}': task '{task_name}' {detail}."
                )

            satisfied |= meta.provides
            if task_name == "reactive_power":
                reactive_set = True
            if task_name == "save":
                has_save = True

        if has_save:
            available_artifacts.add(name)


def _split_step(step: Any) -> tuple[str, dict]:
    """
    Normalize a pipeline step into ``(task_name, params)``.

    Steps are allowed in two forms in YAML/JSON:

    * bare string — ``worst_case_ts`` → ``("worst_case_ts", {})``
    * single-key mapping —
      ``import_electromobility: {charging_strategy: dumb}``
      → ``("import_electromobility", {"charging_strategy": "dumb"})``

    ``None`` as the parameter value is treated as an empty dict so
    that YAML's ``task:`` (with nothing after the colon) works.

    Parameters
    ----------
    step : str or dict
        Raw step as it appears in the pipeline list.

    Returns
    -------
    tuple of (str, dict)
        The task name and its keyword arguments.

    Raises
    ------
    ValueError
        If ``step`` is not a string or a single-key mapping, or if
        the parameter value is not a mapping.

    """
    if isinstance(step, str):
        return step, {}
    if isinstance(step, dict):
        if len(step) != 1:
            raise ValueError(
                f"Task step must be a string or single-key mapping, "
                f"got: {step}"
            )
        (name, params), = step.items()
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(
                f"Parameters for task '{name}' must be a mapping, "
                f"got: {type(params).__name__}"
            )
        return name, params
    raise ValueError(
        f"Task step must be string or mapping, got: {step!r}"
    )
