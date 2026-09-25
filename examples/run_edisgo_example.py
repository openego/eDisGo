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
Example for the eDisGo pipeline runner (:mod:`edisgo.run`).

A pipeline run is driven by a single config (YAML/JSON file or dict):
a flat ``pipeline:`` list of tasks plus supplementary sections such as
``grid``, ``scenario``, ``flexibilities``, ``database`` and
``results``. Typically you extend one of the bundled presets
(``edisgo/run/presets/``) and only override what differs on your
machine:

* ``worst_case`` — database-free: worst-case time series + reinforce.
* ``flex_opf`` — import the flexibilities selected in
  ``flexibilities:``, real egon_data time series, powermodels OPF,
  final reinforce.
* ``overlying_grid_opf`` — like ``flex_opf`` but with overlying-grid
  requirements (e.g. eTraGo results) and automatic critical-timestep
  selection; used by ego.

Database access (``flex_opf`` / ``overlying_grid_opf``) is
auto-detected: if an egon-data configuration file is found
(``~/.ssh/egon-data.configuration.yaml`` or ``EGON_DATA_CONFIG``), the
egon-data database it describes is used, otherwise the OEP. Pin it
explicitly via ``"database": {"source": "oep"}`` or
``{"source": "egon-data", "config_path": ...}``.
"""

import os

from edisgo.run import run_edisgo

grid_path = os.path.join(
    os.path.dirname(__file__), "ding0_example_grid"
)  # adjust to your ding0 grid directory
results_path = os.path.join(os.path.dirname(__file__), "results")


# --- 1) simplest case: extend a bundled preset -------------------------------
edisgo = run_edisgo(
    {
        "extends": "worst_case",
        "grid": {"ding0_path": grid_path},
        "results": {"directory": results_path},
    }
)
print(edisgo.results.grid_expansion_costs)


# --- 2) OPF over selected flexibilities --------------------------------------
# `flexibilities:` controls both which carriers import_flex imports and
# which assets optimize treats as flexible. Requires database access and
# a georeferenced (non-legacy) ding0 grid.
#
# edisgo = run_edisgo(
#     {
#         "extends": "flex_opf",
#         "grid": {"ding0_path": grid_path},
#         "results": {"directory": results_path},
#         "flexibilities": ["heat_pumps", "home_batteries"],
#     }
# )


# --- 3) fully custom pipeline -------------------------------------------------
# Instead of extending a preset, write the pipeline yourself — any
# registered task (see edisgo.run.known_tasks()) can be used. The same
# config can also live in a YAML file: run_edisgo("my_run.yaml").
#
# edisgo = run_edisgo(
#     {
#         "scenario": "eGon2035",
#         "grid": {"ding0_path": grid_path},
#         "results": {"directory": results_path},
#         "pipeline": [
#             "setup_grid",
#             {"set_timeindex": {"start": "2035-01-01", "periods": 24}},
#             {"oedb_ts": {"dispatchable": {"other": 0.7}}},
#             "reactive_power",
#             "analyze",
#             "save",
#         ],
#     }
# )
