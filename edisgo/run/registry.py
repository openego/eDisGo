"""
Task registry for the eDisGo pipeline runner.

This module holds the global, process-wide mapping of task names to task
functions. Tasks are registered via the :func:`register_task` decorator
and looked up by name at pipeline execution time by the runner. Keeping
the registry separate from both the runner and the task implementations
lets external projects add their own tasks without patching eDisGo —
just import ``register_task`` and decorate a function.

Registered tasks all share the signature ``(edisgo, ctx, **params)``
where ``edisgo`` is the current :class:`~edisgo.EDisGo` instance (or
``None`` before it has been created by the first task), ``ctx`` is a
:class:`~edisgo.run.context.RunContext`, and ``**params`` are the
parameters passed from the YAML/JSON step definition. A task may return
an updated ``edisgo`` object (e.g. ``setup_grid`` creates it, ``load_*``
replaces it); otherwise the runner keeps using the same instance.
"""
from __future__ import annotations

from typing import Callable

_TASKS: dict[str, Callable] = {}


def register_task(name: str) -> Callable[[Callable], Callable]:
    """
    Decorator to register a task function under the given name.

    The decorated function becomes addressable from YAML/JSON pipelines
    as either a plain string ``name`` or a single-key mapping
    ``name: {param: value, ...}``. The name must be unique globally —
    re-registering raises :class:`ValueError` to prevent silent
    overrides across plugins.

    Parameters
    ----------
    name : str
        Unique task name used in pipeline definitions.

    Returns
    -------
    Callable
        A decorator that registers ``fn`` and returns it unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered.

    Examples
    --------
    >>> @register_task("set_timeindex_weekly")
    ... def task_weekly(edisgo, ctx, *, start):
    ...     import pandas as pd
    ...     edisgo.set_timeindex(pd.date_range(start, periods=168, freq="h"))

    """
    def deco(fn: Callable) -> Callable:
        if name in _TASKS:
            raise ValueError(
                f"Task '{name}' is already registered "
                f"(existing={_TASKS[name].__qualname__}, "
                f"new={fn.__qualname__})."
            )
        _TASKS[name] = fn
        return fn

    return deco


def get_task(name: str) -> Callable:
    """
    Look up a registered task function by name.

    Parameters
    ----------
    name : str
        Task name as used in pipeline definitions.

    Returns
    -------
    Callable
        The task function registered under ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not registered. The error message lists all
        known task names to aid typo debugging.

    """
    if name not in _TASKS:
        raise KeyError(
            f"Unknown task: '{name}'. Known tasks: {sorted(_TASKS)}"
        )
    return _TASKS[name]


def known_tasks() -> list[str]:
    """
    Return a sorted list of all registered task names.

    Useful for error messages, CLI completion, and tests that assert
    core tasks exist.

    Returns
    -------
    list of str
        All registered task names in alphabetical order.

    """
    return sorted(_TASKS)
