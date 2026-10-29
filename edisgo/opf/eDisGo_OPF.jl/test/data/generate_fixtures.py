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

"""Generate the PowerModels JSON fixtures used by the Julia test suite.

The Julia OPF package receives its input as a single JSON document produced on the
Python side by :func:`edisgo.io.powermodels_io.to_powermodels`. Rather than
hand-writing that (large and highly interdependent) structure in the tests, the
fixtures are generated once from eDisGo's own ding0 test grid, using the same
flexibility set-up as ``tests/opf/test_powermodels_opf.py``, and checked in.

Run from the repository root, with eDisGo importable::

    python edisgo/opf/eDisGo_OPF.jl/test/data/generate_fixtures.py

This writes ``pm_test_network.json`` (no overlying grid) and
``pm_test_network_og.json`` (with overlying-grid requirements) next to this
script. Regenerate the fixtures whenever ``to_powermodels`` changes the input
schema, and commit the result together with the code change so the Julia tests
keep matching Python.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from edisgo import EDisGo
from edisgo.network.topology import Topology
from edisgo.tools.tools import aggregate_district_heating_components

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
DING0_GRID = os.path.join(REPO_ROOT, "tests", "data", "ding0_test_network_1")


def _convert(o):
    """json.dump helper: numpy integers are not JSON serialisable."""
    for f in (np.int8, np.int16, np.int32, np.int64):
        if isinstance(o, f):
            return int(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def build_edisgo_object():
    """Build the eDisGo test object with all flexibilities used by the OPF tests.

    Mirrors ``TestPowerModelsOPF.setup_class`` in ``tests/opf/test_powermodels_opf.py``
    so the Julia fixtures stay comparable to the Python integration test.
    """
    edisgo = EDisGo(ding0_grid=DING0_GRID)
    edisgo.set_time_series_worst_case_analysis()
    timeindex = edisgo.timeseries.timeindex

    edisgo.add_component(
        comp_type="load",
        type="heat_pump",
        sector="individual_heating",
        ts_active_power=pd.Series(
            index=timeindex, data=[1.0 / 5, 2.0 / 6, 2.0 / 5, 1.0 / 6]
        ),
        ts_reactive_power="default",
        bus=edisgo.topology.buses_df.index[26],
        p_set=2,
    )
    edisgo.add_component(
        comp_type="load",
        type="heat_pump",
        sector="individual_heating",
        ts_active_power=pd.Series(
            index=timeindex, data=[2.0 / 7.0, 4.0 / 8.0, 3.0 / 7.0, 3.0 / 8.0]
        ),
        ts_reactive_power="default",
        bus=edisgo.topology.buses_df.index[30],
        p_set=3,
    )
    edisgo.add_component(
        comp_type="load",
        type="heat_pump",
        sector="district_heating_resistive_heater",
        district_heating_id="grid1",
        ts_active_power=pd.Series(index=timeindex, data=[2.0, 8.0, 3.0, 3.0]),
        ts_reactive_power="default",
        bus=edisgo.topology.buses_df.index[27],
        p_set=2,
    )
    edisgo.add_component(
        comp_type="load",
        type="heat_pump",
        sector="district_heating",
        district_heating_id="grid1",
        ts_active_power=pd.Series(
            index=timeindex, data=[2.0 / 7.0, 8.0 / 2.0, 3.0 / 7.0, 3.0 / 8.0]
        ),
        ts_reactive_power="default",
        bus=edisgo.topology.buses_df.index[27],
        p_set=3,
    )

    edisgo.heat_pump.cop_df = pd.DataFrame(
        data={
            "Heat_Pump_LVGrid_3_individual_heating_1": [5.0, 6.0, 5.0, 6.0],
            "Heat_Pump_LVGrid_5_individual_heating_1": [7.0, 8.0, 7.0, 8.0],
            "Heat_Pump_MVGrid_1_district_heating_resistive_heater_1": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "Heat_Pump_MVGrid_1_district_heating_2": [7.0, 2.0, 7.0, 8.0],
        },
        index=timeindex,
    )
    edisgo.heat_pump.heat_demand_df = pd.DataFrame(
        data={
            "Heat_Pump_LVGrid_3_individual_heating_1": [1.0, 2.0, 2.0, 1.0],
            "Heat_Pump_LVGrid_5_individual_heating_1": [2.0, 4.0, 3.0, 3.0],
            "Heat_Pump_MVGrid_1_district_heating_2": [2.0, 8.0, 3.0, 3.0],
            "Heat_Pump_MVGrid_1_district_heating_resistive_heater_1": [
                2.0,
                8.0,
                3.0,
                3.0,
            ],
        },
        index=timeindex,
    )
    edisgo.heat_pump.thermal_storage_units_df = pd.DataFrame(
        data={"capacity": [4, 8, 8], "efficiency": [1, 1, 1]},
        index=edisgo.heat_pump.heat_demand_df.columns[:-1],
    )
    aggregate_district_heating_components(edisgo)
    edisgo.apply_heat_pump_operating_strategy()

    edisgo.add_component(
        comp_type="load",
        type="charging_point",
        ts_active_power=pd.Series(index=timeindex, data=[0.5, 0.5, 0.5, 0.5]),
        ts_reactive_power="default",
        bus=edisgo.topology.buses_df.index[32],
        p_set=3,
    )
    edisgo.electromobility.flexibility_bands = {
        "lower_energy": pd.DataFrame(
            {"Charging_Point_LVGrid_6_1": [0, 0, 1, 2]}, index=timeindex
        ),
        "upper_energy": pd.DataFrame(
            {"Charging_Point_LVGrid_6_1": [1, 2, 2, 3]}, index=timeindex
        ),
        "upper_power": pd.DataFrame(
            {"Charging_Point_LVGrid_6_1": [1, 1, 2, 1]}, index=timeindex
        ),
    }

    dsm_cols = [
        "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1",
        "Load_industrial_LVGrid_5_1",
    ]
    edisgo.dsm.p_min = pd.DataFrame(
        data={dsm_cols[0]: [-0.3] * 4, dsm_cols[1]: [-0.07] * 4}, index=timeindex
    )
    edisgo.dsm.p_max = pd.DataFrame(
        data={dsm_cols[0]: [0.3] * 4, dsm_cols[1]: [0.07] * 4}, index=timeindex
    )
    edisgo.dsm.e_min = pd.DataFrame(
        data={dsm_cols[0]: [-0.3, -0.4, -0.5, -0.4], dsm_cols[1]: [-0.07] * 4},
        index=timeindex,
    )
    edisgo.dsm.e_max = pd.DataFrame(
        data={
            dsm_cols[0]: [0.3, 0.5, 0.5, 0.4],
            dsm_cols[1]: [0.07, 0.1, 0.09, 0.07],
        },
        index=timeindex,
    )

    overlying_grid_data = {
        "dsm_active_power": [0.1, -0.1, -0.1, 0.1],
        "electromobility_active_power": [0.4, 0.5, 0.5, 0.6],
        "heat_pump_decentral_active_power": [0.5, 0.85, 0.85, 0.55],
        "heat_pump_central_active_power": [0.5, 0.85, 0.85, 0.55],
        "renewables_curtailment": [0, 0, 0.1, 0.1],
        "storage_units_active_power": [-0.35, -0.35, 0.35, 0.35],
    }
    for attr, data in overlying_grid_data.items():
        setattr(edisgo.overlying_grid, attr, pd.Series(index=timeindex, data=data))
    edisgo.overlying_grid.feedin_district_heating = pd.DataFrame(
        index=timeindex, columns=["grid1"], data=[1.0, 2.0, 1.0, 2.0]
    )

    # to_powermodels expects the mesh information to be present on the topology
    Topology.find_meshes(edisgo)
    return edisgo


def main():
    edisgo = build_edisgo_object()
    flexible_cps = np.array(["Charging_Point_LVGrid_6_1"])
    flexible_hps = edisgo.heat_pump.thermal_storage_units_df.index.values
    flexible_loads = edisgo.dsm.e_min.columns.values
    flexible_storage_units = edisgo.topology.storage_units_df.index.values

    # Only two fixtures are checked in: across the four opf_versions the generated
    # network differs solely in "opf_version", the "curt" entry in "flexibilities"
    # and the presence of the overlying-grid requirements. The Julia tests patch
    # those few fields themselves (see TestUtils.pm_network), so one fixture
    # without and one with overlying-grid requirements covers all four versions.
    for opf_version, filename in (
        (2, "pm_test_network.json"),
        (4, "pm_test_network_og.json"),
    ):
        pm, _ = edisgo.to_powermodels(
            s_base=1,
            flexible_cps=flexible_cps,
            flexible_hps=flexible_hps,
            flexible_loads=flexible_loads,
            flexible_storage_units=flexible_storage_units,
            opf_version=opf_version,
        )
        path = os.path.join(HERE, filename)
        with open(path, "w") as f:
            json.dump(pm, f, default=_convert, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
