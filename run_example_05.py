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
"""Runner für uc5_select_timesteps.yaml — einfach ``python run_example_05.py``."""

import logging

from edisgo.run.runner import run_edisgo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

# edisgo = run_edisgo("/storage/JoDa/ego/edisgo_run_edisgo/eDisGo/edisgo/run/presets/uc4_example_MS.yaml")  # noqa: E501
edisgo = run_edisgo(
    {
        "extends": "uc5_select_timesteps.yaml",
        # "grid": {"ding0_path": "/home/gurobi/.ding0/run_hetzner_59763_2023_04_06/ding0_grids/32355"}  # noqa: E501
        "grid": {
            "ding0_path": "/home/gurobi/.ding0/2024-07-25T17:38:34_new_planning_new_edisgo/ding0_grids/32377"  # noqa: E501
        },
        # OG path must be the leaf dir for THIS grid (like ding0_path), not the parent.
        "overlying_grid": {
            "path": "/storage/JoDa/edisgo_playground/overlying_grid_data/32377"
        },
    }
)

print("\n=== Fertig ===")
print("Ausbaukosten:\n", edisgo.results.grid_expansion_costs)
print("\nUngelöste Probleme:\n", edisgo.results.unresolved_issues)
