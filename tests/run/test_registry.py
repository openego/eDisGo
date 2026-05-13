"""
Unit tests for :mod:`edisgo.run.registry`.

Verifies that core tasks are discoverable, that ``get_task`` raises a
useful error on typos, and that duplicate registrations are rejected.
"""
import pytest

from edisgo.run.registry import get_task, known_tasks, register_task


def test_known_tasks_contains_core():
    """All core task names must be registered on import."""
    tasks = known_tasks()
    for core in ["setup_grid", "worst_case_ts", "reactive_power",
                 "reinforce", "analyze", "save"]:
        assert core in tasks


def test_get_task_unknown_raises():
    """Unknown task names must surface as a descriptive KeyError."""
    with pytest.raises(KeyError, match="Unknown task"):
        get_task("does_not_exist")


def test_register_task_duplicate_raises():
    """Registering the same task name twice is a bug — must raise."""
    @register_task("_test_task_for_dup_check")
    def _a(edisgo, ctx):
        """Marker task #1 — test fixture only."""

    with pytest.raises(ValueError, match="already registered"):
        @register_task("_test_task_for_dup_check")
        def _b(edisgo, ctx):
            """Marker task #2 — test fixture only, must not register."""
