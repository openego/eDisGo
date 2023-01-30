import os

from copy import deepcopy

import pandas as pd

from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.costs import grid_expansion_costs

ev = True
hp = True
dsm = True
storage = True

grid = "5bus_testcase"
opf_version = 3
i = 0
path = os.path.join(os.getcwd(), "edisgo_scenario_data")
directory = os.path.join(path, str(grid))
results = pd.DataFrame(
    index=["EV", "HP", "Storage", "DSM", "NE_cost", "Sum_active_loads", "solve_time"]
)

edisgo1 = import_edisgo_from_files(
    directory,
    import_topology=True,
    import_timeseries=True,
    import_electromobility=True,
    import_heat_pump=True,
    import_dsm=True,
)


edisgo = deepcopy(edisgo1)
edisgo1.reinforce()
cost = grid_expansion_costs(edisgo1)
results[i] = [
    False,
    False,
    False,
    False,
    cost.sum().total_costs,
    edisgo1.timeseries.loads_active_power.sum().sum(),
    "-",
]
i += 1
psa_net = edisgo.to_pypsa()
if ev:
    flexible_cps = psa_net.loads.loc[
        psa_net.loads.index.str.contains("home")
        | (psa_net.loads.index.str.contains("work"))
    ].index.values
else:
    flexible_cps = None
if hp:
    flexible_hps = edisgo.heat_pump.thermal_storage_units_df.index.values
else:
    flexible_hps = None
if dsm:
    flexible_loads = edisgo.dsm.p_max.columns
else:
    flexible_loads = None
if storage:
    flexible_storages = edisgo.topology.storage_units_df.index.values
else:
    flexible_storages = None

edisgo.pm_optimize(
    flexible_cps=flexible_cps,
    flexible_hps=flexible_hps,
    flexible_loads=flexible_loads,
    flexible_storages=flexible_storages,
    opf_version=opf_version,
    silence_moi=False,
    method="soc",
)

edisgo.reinforce()
cost = grid_expansion_costs(edisgo)
edisgo.save(
    os.path.join("opf_solutions", str(grid)),
    save_results=True,
    save_timeseries=True,
    save_electromobility=True,
    save_heatpump=True,
    save_dsm=True,
)

results[i] = [
    ev,
    hp,
    storage,
    dsm,
    cost.sum().total_costs,
    edisgo.timeseries.loads_active_power.sum().sum(),
    edisgo.opf_results.solution_time,
]

results.transpose().to_csv(os.path.join("opf_solutions", str(grid) + ".csv"))
print(results.transpose())
