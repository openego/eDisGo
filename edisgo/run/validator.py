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

from edisgo.run.registry import known_tasks

_TS_TASKS = {"worst_case_ts", "oedb_ts", "manual_ts", "set_timeindex"}
_GRID_CREATING_TASKS = {"setup_grid", "load_from_base"}
_FLEX_IMPORTS = {
    "import_heat_pumps",
    "import_home_batteries",
    "import_dsm",
    "import_electromobility",
}


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

        grid_available = load_from is not None
        ts_set = False
        reactive_set = False
        flex_imported = False
        has_save = False

        for step in pipeline:
            task_name, _params = _split_step(step)
            if task_name not in known_tasks():
                raise ValueError(
                    f"Unknown task '{task_name}' in stage '{name}'. "
                    f"Known: {known_tasks()}"
                )

            if task_name in _GRID_CREATING_TASKS:
                grid_available = True
            if task_name in _TS_TASKS:
                if reactive_set:
                    raise ValueError(
                        f"Stage '{name}': time-series task "
                        f"'{task_name}' comes after 'reactive_power' "
                        f"— reactive_power must be the last "
                        f"time-series-altering step."
                    )
                ts_set = True
            if task_name == "reactive_power":
                reactive_set = True
            if task_name in _FLEX_IMPORTS:
                flex_imported = True
                if not grid_available:
                    raise ValueError(
                        f"Stage '{name}': task '{task_name}' requires "
                        f"a loaded grid (setup_grid or "
                        f"load_from_base) before it."
                    )
            if task_name in {"analyze", "reinforce"} and not (
                ts_set or load_from
            ):
                raise ValueError(
                    f"Stage '{name}': task '{task_name}' requires time "
                    f"series to be set (e.g. worst_case_ts or "
                    f"oedb_ts) before it."
                )
            if task_name == "optimize":
                if not ts_set and not load_from:
                    raise ValueError(
                        f"Stage '{name}': 'optimize' requires time "
                        f"series."
                    )
                if not flex_imported and not load_from:
                    raise ValueError(
                        f"Stage '{name}': 'optimize' requires at least "
                        f"one flex asset to be imported."
                    )
            if task_name == "base_reinforce" and not grid_available:
                raise ValueError(
                    f"Stage '{name}': 'base_reinforce' requires a "
                    f"loaded grid before it."
                )
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
