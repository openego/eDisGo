import os

import matplotlib.pyplot as plt
import pandas as pd

import edisgo.opf.lopf as opt
from edisgo.edisgo import import_edisgo_from_files
from edisgo.network.electromobility import (Electromobility,
                                            get_energy_bands_for_optimization)
from edisgo.network.timeseries import get_component_timeseries
from Script_prepare_grids_for_optimization import \
    get_downstream_nodes_matrix_iterative

grid_dir = "minimum_working"
opt_ev = False
opt_stor = False
save_res = False
opt_hp = True

if os.path.isfile("x_charge_ev_pre.csv"):
    ts_pre = pd.read_csv("x_charge_ev_pre.csv", index_col=0, parse_dates=True)
else:
    ts_pre = pd.DataFrame()

timeindex = pd.date_range("2011-01-01", periods=8760, freq="h")
storage_ts = pd.DataFrame({"Storage 1": 8760 * [0]}, index=timeindex)

edisgo = import_edisgo_from_files(grid_dir)
get_component_timeseries(
    edisgo,
    timeseries_load="demandlib",
    timeseries_generation_fluctuating="oedb",
    timeseries_storage_units=storage_ts,
)
timesteps = edisgo.timeseries.timeindex[7 * 24 : 2 * 24 * 7]

if opt_ev:
    cp_id = 1
    ev_data = pd.read_csv(
        os.path.join(grid_dir, "BEV_standing_times_minimum_working.csv"), index_col=0
    )
    charging_events = ev_data.loc[ev_data.chargingdemand > 0]
    charging_events["charging_park_id"] = cp_id
    Electromobility(edisgo_obj=edisgo)
    edisgo.electromobility.charging_processes_df = charging_events
    cp = edisgo.add_component(
        "ChargingPoint", bus="Bus 2", p_nom=0.011, use_case="home", add_ts=False
    )
    edisgo.electromobility.integrated_charging_parks_df = pd.DataFrame(
        index=[cp_id], columns=["edisgo_id"], data=cp
    )
    edisgo.electromobility.simbev_config_df = pd.DataFrame(
        index=["eta_CP", "stepsize"], columns=["value"], data=[0.9, 60]
    )
    energy_bands = get_energy_bands_for_optimization(edisgo_obj=edisgo, use_case="home")
else:
    energy_bands = {}

if opt_hp:
    hp_name = "HP 1"
    heat_demand = (
        pd.read_csv("minimum_working/hp_heat_2011.csv", index_col=0)
        .set_index(timeindex)
        .rename(columns={"0": hp_name})
    )
    cop = pd.read_csv("minimum_working/COP_2011.csv").set_index(timeindex)
    heat_pump_df = pd.DataFrame(
        index=[hp_name],
        columns=["bus", "p_nom", "capacity_tes"],
        data={"bus": "Bus 3", "p_nom": 0.003, "capacity_tes": 0.05},
    )
    edisgo.topology.heat_pumps_df = heat_pump_df
else:
    cop = None
    heat_demand = None


downstream_node_matrix = get_downstream_nodes_matrix_iterative(edisgo.topology)
parameters = opt.prepare_time_invariant_parameters(
    edisgo,
    downstream_node_matrix,
    pu=False,
    optimize_storage=False,
    optimize_ev_charging=opt_ev,
    optimize_hp=opt_hp,
    ev_flex_bands=energy_bands,
)
model = opt.setup_model(
    parameters,
    timesteps=timesteps,
    objective="residual_load",
    optimize_storage=False,
    optimize_ev_charging=opt_ev,
    optimize_hp=opt_hp,
    heat_demand=heat_demand,
    cop=cop,
)

results = opt.optimize(model, "gurobi")
if opt_ev:
    results["x_charge_ev"].plot()
    plt.show()
    if not ts_pre.empty:
        ts_pre.plot()
        plt.show()
        pd.testing.assert_frame_equal(ts_pre, results["x_charge_ev"])
    if save_res:
        results["x_charge_ev"].to_csv("x_charge_ev_pre.csv")
print("SUCCESS")
