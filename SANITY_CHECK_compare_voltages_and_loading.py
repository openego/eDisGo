# Script to compare the line loading and bus voltage from optimisation and ACPF

import os
import pandas as pd
from edisgo.edisgo import import_edisgo_from_files
from edisgo.opf.lopf import combine_results_for_grid

# Load eDisGo object of feeder
base_dir = r"H:\Grids"
grid_id = 176
feeder_id = 0
grid_dir = os.path.join(base_dir, str(grid_id), "feeder", str(feeder_id))

edisgo_obj = import_edisgo_from_files(grid_dir, import_timeseries=True)

# Load results from optimisation
results_base_dir = r"U:\Software\SEST\eDisGo\results\minimize_loading_SEST_v1"
ev_charging = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "x_charge_ev")
curtailment_ev = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "curtailment_ev")
curtailment_load = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "curtailment_load")
curtailment_feedin = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "curtailment_feedin")
curtailment_reactive_load = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "curtailment_reactive_load")
curtailment_reactive_feedin = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "curtailment_reactive_feedin")
bus_voltage = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "v_bus")/20
branch_active_power = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "p_line")
branch_reactive_power = combine_results_for_grid(
    [feeder_id], grid_id, results_base_dir, "q_line")

# Add optimised ev charging
edisgo_obj.timeseries._charging_points_active_power.loc[:, ev_charging.columns] = ev_charging

# Add curtailment as new loads and feedin to grid
curtailment_load = curtailment_load + curtailment_ev
curtailment_load = curtailment_load[curtailment_load.columns[curtailment_load.sum()>0]]
curtailment_reactive_load = curtailment_reactive_load[curtailment_load.columns]
curtailment_feedin = curtailment_feedin[curtailment_feedin.columns[curtailment_feedin.sum()>0]]
curtailment_reactive_feedin = curtailment_reactive_feedin[curtailment_feedin.columns]

edisgo_obj.timeseries.mode = 'manual'
edisgo_obj.add_components('Load', ts_active_power=curtailment_feedin,
                          ts_reactive_power=curtailment_reactive_feedin, buses=curtailment_feedin.columns,
                          load_ids=curtailment_feedin.columns, peak_loads=curtailment_feedin.max().values,
                          annual_consumptions=curtailment_feedin.sum().values,
                          sectors=['feedin_curtailment']*len(curtailment_feedin.columns))
print('Load added for curtailment at buses {}'.format(curtailment_feedin.columns))
edisgo_obj.add_components('Generator', ts_active_power=curtailment_load,
                          ts_reactive_power=curtailment_reactive_load, buses=curtailment_load.columns,
                          generator_ids=curtailment_load.columns, p_noms=curtailment_load.max().values,
                          generator_types=['load_curtailment']*len(curtailment_load.columns))
print('Generator added for curtailment at buses {}'.format(curtailment_load.columns))

# Calculate ACPF and compare with optimised results
pypsa_grid = edisgo_obj.to_pypsa()
pypsa_grid.pf()
losses = pd.concat([pypsa_grid.lines_t.p0 + pypsa_grid.lines_t.p1,
                    pypsa_grid.transformers_t.p0 + pypsa_grid.transformers_t.p1], axis=1)
losses_q = pd.concat([pypsa_grid.lines_t.q0 + pypsa_grid.lines_t.q1,
                    pypsa_grid.transformers_t.q0 + pypsa_grid.transformers_t.q1], axis=1)

branch_active_power_acpf = pd.concat([(pypsa_grid.lines_t.p0 - pypsa_grid.lines_t.p1)/2,
                                      (pypsa_grid.transformers_t.p0 - pypsa_grid.transformers_t.p1)/2], axis=1)
branch_reactive_power_acpf = pd.concat([(pypsa_grid.lines_t.q0 - pypsa_grid.lines_t.q1)/2,
                                      (pypsa_grid.transformers_t.q0 - pypsa_grid.transformers_t.q1)/2], axis=1)
bus_voltage_acpf = pypsa_grid.buses_t.v_mag_pu

# Calculate differences
diff_p = branch_active_power.abs() - branch_active_power_acpf.abs()
diff_q = branch_reactive_power.abs() - branch_reactive_power_acpf.abs()
diff_v = bus_voltage - bus_voltage_acpf

print("SUCCESS")
