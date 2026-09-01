"""
Behavioural guard for openego/eGo#202.

Deliberately in its own module and importing nothing from
:mod:`edisgo.run.tasks.flex`: a test that imports the new task by name fails
on a tree without it with an ``ImportError``, which proves nothing about the
behaviour. This one fails with the actual finding -- no pipeline step consumes
``feedin_district_heating`` -- which is the regression itself.
"""


def test_every_overlying_grid_preset_consumes_the_feedin():
    """
    Behavioural guard against the regression itself, independent of the
    task's name: for each preset that imports overlying-grid data, some
    pipeline step must consume ``feedin_district_heating``. On dev no step
    does, which is openego/eGo#202 -- and a test that only imports the new
    task by name would fail there with an ImportError, proving nothing
    about the behaviour.
    """
    from edisgo.run.config import load_config
    from edisgo.run.registry import get_task_meta

    consumers = {"aggregate_district_heating"}
    for preset in (
        "overlying_grid_opf",
        "overlying_grid_opf_spatial",
        "spatial_reduction_opf",
    ):
        cfg = load_config({"extends": preset, "grid": {"ding0_path": "/x"}})
        steps = [
            step if isinstance(step, str) else next(iter(step))
            for step in cfg["pipeline"]
        ]
        assert "import_overlying_grid_data" in steps, preset
        assert consumers & set(steps), (
            f"{preset}: imports overlying-grid data but no step consumes "
            f"feedin_district_heating -- openego/eGo#202"
        )
        # and the consumer must come after the import
        assert steps.index("aggregate_district_heating") > steps.index(
            "import_overlying_grid_data"
        ), preset
        # the dependency is declared, not just ordered by luck
        assert "overlying_grid" in get_task_meta(
            "aggregate_district_heating"
        ).requires
