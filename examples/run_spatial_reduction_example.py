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

# Adjust the paths below to your local ding0 grid and overlying-grid CSV data.
# Both point at the leaf directory for THIS MV grid (not the parent), e.g.
# ".../ding0_grids/<mv_grid_id>" and ".../overlying_grid_csvs/<mv_grid_id>".
MV_GRID_ID = 32377

edisgo = run_edisgo(
    {
        "extends": "spatial_reduction_opf",
        "grid": {"ding0_path": f"/path/to/ding0_grids/{MV_GRID_ID}"},
        "overlying_grid": {"path": f"/path/to/overlying_grid_csvs/{MV_GRID_ID}"},
    }
)

print("\n=== Done ===")
print("Grid expansion costs:\n", edisgo.results.grid_expansion_costs)
print("\nUnresolved issues:\n", edisgo.results.unresolved_issues)
