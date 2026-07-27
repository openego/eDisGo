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
Task implementations for the eDisGo pipeline runner.

Importing this package as a side effect registers every task defined
in its submodules with :func:`edisgo.run.registry.register_task`, so
that the runner sees them at execution time. The submodules are:

* :mod:`.grid` — ``setup_grid``, ``load_from_base``
* :mod:`.timeseries` — ``worst_case_ts``, ``oedb_ts``, ``manual_ts``,
  ``set_timeindex``, ``reactive_power``
* :mod:`.flex` — flex imports
  (``import_heat_pumps``, ``import_home_batteries``, ``import_dsm``,
  ``import_electromobility``, ``import_generators``),
  ``build_flexibility_bands``, and operating strategies
  (``apply_charging_strategy``, ``apply_heat_pump_strategy``)
* :mod:`.analysis` — ``check_integrity``, ``analyze``, ``reinforce``,
  ``base_reinforce``, ``optimize``
* :mod:`.io` — ``save``, ``load_charging_from_files``
* :mod:`.spatial` — ``spatial_reduce``, ``spatial_restore``

Task signature convention: ``(edisgo, ctx, **params)``. A task may
mutate ``edisgo`` in place and/or return a new EDisGo instance (the
returned value, if non-None, replaces the current one in the runner's
loop).
"""

from edisgo.run.tasks import (  # noqa: F401
    analysis,
    flex,
    grid,
    io,
    spatial,
    timeseries,
)
