import os

import matplotlib.pyplot as plt
import pandas as pd

import edisgo.opf.lopf as opt

from edisgo.edisgo import import_edisgo_from_files
from edisgo.network.electromobility import Electromobility
from Script_prepare_grids_for_optimization import get_downstream_nodes_matrix_iterative

par_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
grid_dir = os.path.join(par_dir, "tests", "optimisation_minimum_working")
opt_ev = True
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
edisgo.timeseries.timeindex = timeindex
edisgo.set_time_series_active_power_predefined(
    fluctuating_generators_ts="oedb", conventional_loads_ts="demandlib"
)
edisgo.set_time_series_manual(storage_units_p=storage_ts)
edisgo.set_time_series_reactive_power_control()
timesteps = edisgo.timeseries.timeindex[7 * 24 : 2 * 24 * 7]

if opt_ev:
    cp_id = 1
    ev_data = pd.read_csv(
        os.path.join(grid_dir, "BEV_standing_times_minimum_working.csv"), index_col=0
    )
    charging_events = ev_data.loc[ev_data.chargingdemand_kWh > 0]
    charging_events["charging_park_id"] = cp_id
    Electromobility(edisgo_obj=edisgo)
    edisgo.electromobility.charging_processes_df = charging_events
    cp = edisgo.add_component(
        "load",
        bus="Bus 2",
        type="charging_point",
        p_set=0.011,
        sector="home",
        add_ts=False,
    )
    edisgo.electromobility.integrated_charging_parks_df = pd.DataFrame(
        index=[cp_id], columns=["edisgo_id"], data=cp
    )
    edisgo.electromobility.simbev_config_df = pd.DataFrame(
        index=["eta_cp", "stepsize"], columns=[0], data=[0.9, 60]
    ).T
    energy_bands = edisgo.electromobility.get_flexibility_bands(
        edisgo_obj=edisgo, use_case="home"
    )
else:
    energy_bands = {}

if opt_hp:
    hp_name = edisgo.add_component(
        "load",
        bus="Bus 3",
        type="heat_pump",
        p_set=0.003,
        sector="flexible",
        add_ts=False,
    )

    edisgo.heat_pump.heat_demand_df = (
        pd.read_csv(os.path.join(grid_dir, "hp_heat_2011.csv"), index_col=0)
        .set_index(timeindex)
        .rename(columns={"0": hp_name})
    )
    edisgo.heat_pump.cop_df = (
        pd.read_csv(os.path.join(grid_dir, "COP_2011.csv"))
        .set_index(timeindex)
        .rename(columns={"COP 2011": hp_name})
    )
    edisgo.heat_pump.thermal_storage_units_df = pd.DataFrame(
        data={
            "capacity": [0.05],
            "efficiency": [1.0],
            "state_of_charge_initial": [0.5],
        },
        index=[hp_name],
    )


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
)

results = opt.optimize(model, "gurobi")
if opt_ev and not opt_hp:
    results["x_charge_ev"].plot()
    plt.show()
    if not ts_pre.empty:
        ts_pre.plot()
        plt.show()
        pd.testing.assert_frame_equal(ts_pre, results["x_charge_ev"])
    if save_res:
        results["x_charge_ev"].to_csv("x_charge_ev_pre.csv")

print("SUCCESS")
