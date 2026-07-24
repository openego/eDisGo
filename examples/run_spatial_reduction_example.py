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
"""Runner for spatial_reduction_opf.yaml — run via ``python run_spatial_reduction_example.py``."""

import logging

from edisgo.run.runner import run_edisgo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

edisgo = run_edisgo(
    {
        "extends": "spatial_reduction_opf",
        "grid": {
            "ding0_path": "/home/gurobi/.ding0/2024-07-25T17:38:34_new_planning_new_edisgo/ding0_grids/32377"  # noqa: E501
        },
        # OG path must be the leaf dir for THIS grid (like ding0_path), not the parent.
        "overlying_grid": {"path": "/home/gurobi/.edisgo_input/overlying_grid/32377"},
    }
)

print("\n=== Fertig ===")
print("Ausbaukosten:\n", edisgo.results.grid_expansion_costs)
print("\nUngelöste Probleme:\n", edisgo.results.unresolved_issues)
