"""
Unit tests for :mod:`edisgo.run.validator`.

Each test pins one rule: reactive-before-TS, reinforce without TS,
optimize without flex, flex import without grid, and the checks of the
time-step selection and spatial_reduction settings and unknown step params.
"""

import pytest

from edisgo.run.config import load_config
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


_SELECTION_PIPELINE = [
    "setup_grid",
    "set_timeindex",
    "worst_case_ts",
    "select_critical_timesteps",
    "reactive_power",
]


_CRITICAL = {"task": "select_critical_timesteps"}


@pytest.mark.parametrize(
    "selection, match",
    [
        (["select_critical_timesteps"], "must be a mapping"),
        ({"task": "auto"}, "'none', got 'auto'"),
        ({"task": False}, "'none', got False"),
        ({**_CRITICAL, "method": "power_flow"}, "timestep_selection.<task>"),
        ({**_CRITICAL, "select_critical_timesteps": ["power_flow"]}, "mapping"),
        (
            {**_CRITICAL, "select_critical_timesteps": {"metod": "power_flow"}},
            r"\['metod'\]",
        ),
        # unknown names are rejected in the sub-block of the skipped task, too
        ({**_CRITICAL, "set_timeindex": {"strat": "2011-01-01"}}, r"\['strat'\]"),
    ],
)
def test_invalid_timestep_selection_rejected(selection, match):
    """A wrong timestep_selection block is rejected before the run."""
    with pytest.raises(ValueError, match=match):
        validate({**_wrap(_SELECTION_PIPELINE), "timestep_selection": selection})


def test_timestep_selection_without_its_task_rejected():
    """A picked task that is not in the pipeline would be ignored."""
    cfg = _wrap(["setup_grid", "worst_case_ts", "reactive_power"])
    with pytest.raises(ValueError, match="no 'select_critical_timesteps' step"):
        validate({**cfg, "timestep_selection": _CRITICAL})
    # task none needs no selection step
    validate({**cfg, "timestep_selection": {"task": "none"}})


@pytest.mark.parametrize("index", [1, 3])
def test_inline_params_with_timestep_selection_rejected(index):
    """With the block set, inline params of either step would be ignored or
    mixed with the block, also if the step appears twice."""
    pipeline = [*_SELECTION_PIPELINE, "set_timeindex"]
    pipeline[index] = {pipeline[index]: {"periods": 2}}
    with pytest.raises(ValueError, match="Move them into timestep_selection"):
        validate({**_wrap(pipeline), "timestep_selection": _CRITICAL})


@pytest.mark.parametrize(
    "set_timeindex, match",
    [
        ({"timestamps": ["2035-01-15 00:00"]}, "'timestamps' or 'start'"),
        ({"end": "2035-01-16 00:00"}, "'periods' or 'end'"),
        ({"start": None}, "needs 'timestamps'"),
    ],
)
def test_effective_set_timeindex_params_checked(set_timeindex, match):
    """Params merged with the preset's (start + periods) are checked before
    the run, not after the data imports."""
    cfg = load_config(
        {
            "extends": "spatial_reduction_opf",
            "timestep_selection": {"set_timeindex": set_timeindex},
        }
    )
    with pytest.raises(ValueError, match=match):
        validate(cfg)


def test_inline_params_checked_without_timestep_selection():
    """Without the block, inline params are checked before the run."""
    with pytest.raises(ValueError, match=r"\['strat'\]"):
        validate(_wrap(["setup_grid", {"set_timeindex": {"strat": "2011-01-01"}}]))
    with pytest.raises(ValueError, match="needs 'timestamps'"):
        validate(_wrap(["setup_grid", "set_timeindex"]))


def test_null_timestep_selection_restores_inline_params():
    """timestep_selection: null drops a preset's block, so a custom pipeline
    can configure the selection inline again."""
    cfg = load_config(
        {
            "extends": "overlying_grid_opf",
            "timestep_selection": None,
            "pipeline": [
                "setup_grid",
                {"set_timeindex": {"start": "2035-01-15", "periods": 24}},
                "worst_case_ts",
                "reactive_power",
            ],
        }
    )
    validate(cfg)


def test_skipped_selection_step_ignored_by_ordering_rules():
    """A skipped step neither provides time series nor alters them."""
    pipeline = ["setup_grid", "set_timeindex", "reinforce"]
    with pytest.raises(ValueError, match="requires time series"):
        validate({**_wrap(pipeline), "timestep_selection": {"task": "none"}})
    # a skipped select_critical_timesteps after reactive_power is harmless
    pipeline = [
        "setup_grid",
        "set_timeindex",
        "worst_case_ts",
        "reactive_power",
        "select_critical_timesteps",
    ]
    validate(
        {
            **_wrap(pipeline),
            "timestep_selection": {
                "task": "set_timeindex",
                "set_timeindex": {"start": "2011-01-01", "periods": 2},
            },
        }
    )


def test_unknown_task_params_rejected():
    """A typo in a step param is reported before the run."""
    with pytest.raises(ValueError, match=r"\['strategi'\] for task"):
        validate(
            _wrap(
                [
                    "setup_grid",
                    "worst_case_ts",
                    {"apply_charging_strategy": {"strategi": "dumb"}},
                ]
            )
        )


def test_enabled_spatial_reduction_without_its_task_rejected():
    """An enabled spatial_reduction block needs a spatial_reduce step."""
    cfg = _wrap(["setup_grid", "worst_case_ts", "reactive_power"])
    with pytest.raises(ValueError, match="no 'spatial_reduce' step"):
        validate({**cfg, "spatial_reduction": {"enabled": True}})
    validate({**cfg, "spatial_reduction": {"enabled": False}})
    with pytest.raises(ValueError, match="must be a mapping"):
        validate({**cfg, "spatial_reduction": True})
