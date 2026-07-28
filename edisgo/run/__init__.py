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
YAML/JSON-driven pipeline runner for eDisGo.

Single entry point::

    from edisgo.run import run_edisgo
    edisgo = run_edisgo({"extends": "flex_opf", "grid": {"ding0_path": ...}})

A config is a flat ``pipeline:`` — a list of named tasks from
:mod:`edisgo.run.tasks`. Each step is either a string
(``worst_case_ts``) or a single-key mapping with parameters
(``import_electromobility: {charging_strategy: dumb}``). Configs can
build on a bundled preset (``edisgo/run/presets/``) or another file
via ``extends:``.
"""

from edisgo.run.context import RunContext
from edisgo.run.registry import known_tasks, register_task
from edisgo.run.runner import run_edisgo

__all__ = [
    "RunContext",
    "known_tasks",
    "register_task",
    "run_edisgo",
]
