import os

from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.costs import grid_expansion_costs

opf_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(opf_dir, "edisgo_scenario_data", "5_bus_ex")
edisgo1 = import_edisgo_from_files(
    directory,
    import_topology=True,
    import_timeseries=True,
    import_electromobility=True,
    import_heat_pump=True,
    import_dsm=True,
)

edisgo1.reinforce()
cost = grid_expansion_costs(edisgo1)
print(cost)

edisgo = import_edisgo_from_files(
    directory,
    import_topology=True,
    import_timeseries=True,
    import_electromobility=True,
    import_heat_pump=True,
    import_dsm=True,
)

psa_net = edisgo.to_pypsa()
flexible_cps = psa_net.loads.loc[
    psa_net.loads.index.str.contains("home")
    | (psa_net.loads.index.str.contains("work"))
].index.values
flexible_hps = edisgo.heat_pump.thermal_storage_units_df.index.values

edisgo.pm_optimize(flexible_cps=flexible_cps, flexible_hps=flexible_hps, opt_version=3)
edisgo.set_time_series_reactive_power_control()
edisgo.reinforce()
cost_opt = grid_expansion_costs(edisgo)
print(cost_opt)
