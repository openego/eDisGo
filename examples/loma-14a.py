from datetime import datetime

import geopandas as gpd
import pandas as pd

from edisgo import EDisGo


def run_optimization_14a(edisgo):
    """
    Run optimization with §14a curtailment enabled.

    Uses opf_version=5 which uses §14a curtailment as the only flexibility tool.
    Minimizes line losses and §14a usage. Grid restrictions (voltage 0.9-1.1 p.u.,
    current limits) are enforced as hard constraints. Feasibility slacks exist but
    are penalized at 1e8 to ensure the model remains feasible.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object with time series

    Returns
    -------
    EDisGo
        EDisGo object with optimization results
    """
    print(f"\n{'='*80}")
    print("⚡ Running OPF with §14a Curtailment")
    print(f"{'='*80}")
    print("\nUsing OPF version 5:")
    print("  - §14a curtailment as only flexibility tool")
    print("  - Minimize line losses + §14a usage")
    print("  - Grid restrictions enforced (voltage 0.9-1.1, current limits)")
    print("  - Feasibility slacks penalized at 1e8")

    start_time = datetime.now()

    # Run optimization
    edisgo.pm_optimize(opf_version=5, curtailment_14a=True)

    duration = (datetime.now() - start_time).total_seconds()

    print("\n✓ Optimization complete!")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    return edisgo


# grid_path = "/home/carlos/LoMa/exec_folder/results/MGB_quo_model_pypsa"

# Whole husum paths
grid_path = "/home/carlos/LoMa/exec_folder/results/Husum_SLP_CP_pypsa"
path_husum_district_shp = (
    "/home/carlos/LoMa/exec_folder/data/Input_files/MV_grid_district/husum_district.shp"
)

# MGB paths
# grid_path = "/home/paul/LoMa/MGB_2035_model_pypsa"
# path_husum_district_shp = "/home/paul/LoMa/loma-repo/data/Input_files/MGB_district"

edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(0, 23))

mv_grid_geom = gpd.read_file(path_husum_district_shp).to_crs(4326)
edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
edisgo.topology.grid_district["srid"] = 4326

edisgo.topology.check_integrity()
pypsa_n = edisgo.to_pypsa()
edisgo.analyze()

############################# MANUAL FIXES ####################################
edisgo.topology.generators_df = edisgo.topology.generators_df[
    edisgo.topology.generators_df.index != "HV_dummy_gen_slack"
]
edisgo.topology.buses_df = edisgo.topology.buses_df[
    edisgo.topology.buses_df.v_nom <= 20
]

edisgo.topology.buses_df = edisgo.topology.buses_df[
    edisgo.topology.buses_df.index != "HV_dummy_bus"
]

############################ EV INTEGRATION PART ##############################
from edisgo.tools.loma_tools import (
    buses_with_existing_loads,
    create_network_gif,
    get_curtailment_data,
    plot_load_before_after,
    plot_network,
    set_charging_points_to_target,
    set_heat_pumps_to_target,
    transfer_ts_from_new_to_existing_cp,
)

# Temporary Check: Amount of CPs before importing eDisGo CPs
names = edisgo.topology.loads_df.query("type == 'charging_point'").index.astype(str)
print(
    {
        "existing": names.str.contains("Existing", case=False).sum(),
        "additional": names.str.contains("Additional", case=False).sum(),
        "rest": (
            ~(
                names.str.contains("Existing", case=False)
                | names.str.contains("Additional", case=False)
            )
        ).sum(),
        "total": len(names),
    }
)

# -------------------------
# Import + distribute + integrate EV data (creates new charging points)
# -------------------------
"""
After this function there are no time series yet. Only charging points and 
a overall demand which is then transferred into a time series in 
apply_charging_strategy.

Note: Afterwards there should be the Existing CP (411) and Additional CP (589) 
from the LoMa side for the 2035 scenario and all new eDisGo CP (for whole Husum 
there should be 2337). So the total should be 3337. 
"""
edisgo.import_electromobility_14a(
    scenario="eGon2035",
    import_electromobility_data_kwds={"shapefile_path": path_husum_district_shp},
)

# Temporary Check: Amount of CPs after importing eDisGo CPs
names = edisgo.topology.loads_df.query("type == 'charging_point'").index.astype(str)
print(
    {
        "existing": names.str.contains("Existing", case=False).sum(),
        "additional": names.str.contains("Additional", case=False).sum(),
        "rest": (
            ~(
                names.str.contains("Existing", case=False)
                | names.str.contains("Additional", case=False)
            )
        ).sum(),
        "total": len(names),
    }
)

# -------------------------
# Apply charging strategy
# -------------------------
"""
This step created the time series for the new eDisGo charging points.
Without the preparation of Q before charging strategy I got an error while 
apply_charging_strategy which was caused by a deviating time index.

Note: After this step ONLY the charging points from eDisGo have a time series.
"""
# Prepare Q before charging strategy
ti = edisgo.timeseries.timeindex
lap_cols = edisgo.timeseries.loads_active_power.columns

edisgo.timeseries.loads_reactive_power = pd.DataFrame(
    0.0,
    index=ti,
    columns=lap_cols,
)

edisgo.apply_charging_strategy(strategy="dumb")

# -------------------------
# Transfer time series from new eDisGo CPs to existing CPs
# -------------------------
"""
This step then finally transfers the time series from suitable eDisGo 
charging_points to Existing_ und Additional_ charging points which are 
created on the LoMa side.

Note: After this step there should be 411 Existing CP and 589 Additional for
the 2035 scenario and 1337 eDisGo CP as 1000 of those were used for matching
and transferring the time series and deleted afterwards.
"""
ev_match_results = transfer_ts_from_new_to_existing_cp(
    edisgo,
    existing_markers=("Existing", "Additional"),
    radius_1=2000.0,
    tol_1=0.15,
    radius_2=2000.0,
    tol_2=0.9,
)

# Temporary Check: Amount of CPs fter transferring time series
names = edisgo.topology.loads_df.query("type == 'charging_point'").index.astype(str)
print(
    {
        "existing": names.str.contains("Existing", case=False).sum(),
        "additional": names.str.contains("Additional", case=False).sum(),
        "rest": (
            ~(
                names.str.contains("Existing", case=False)
                | names.str.contains("Additional", case=False)
            )
        ).sum(),
        "total": len(names),
    }
)

# ============================================================
# Optional Utilities for sensitivity analysis/chaning the amount of cp/hp
# - target by absolute value or relative percentage
# - Only use one option at a time (traget_total, percentage)
# ============================================================
"""
In this step the total amount of charging points or heat pumps can be adjusted.
Either by percentage or by a total amount including the infrastructure from
LoMa. When deleting CP/HP there is an option to export the deleted ones.
New CP/HP will have 'dup' in their name.

Note: for the 2035 scenario the target total would need to be set to 1000.
CPs with the marker Additional and Existing in their name will be removed last.
This way only the remaining 1337 eDisGo CP would be deleted.
"""
output_dir = "/home/paul/LoMa/test/shapes"

cp_eligible_buses = buses_with_existing_loads(edisgo)
hp_eligible_buses = buses_with_existing_loads(edisgo)

change_cp_amount = set_charging_points_to_target(
    edisgo,
    target_total=100,  # sets total amount of CP to 1000
    # percentage=0.10, # increases total amount of CP by 10%
    # percentage=-0.10, # decreases total amount of CP by 10%
    eligible_buses=cp_eligible_buses,
    removal_priority=["Additional", "Existing"],
    add_tracking_columns=False,
    export_removed=True,  # only applies when there are deleted CP
    export_dir=output_dir,  # only applies when there are deleted CP
)

change_hp_amount = set_heat_pumps_to_target(
    edisgo,
    target_total=100,  # sets total amount of HP to 50
    # percentage=0.10, # increases total amount of CP by 10%
    # percentage=-0.10, # decreases total amount of CP by 10%
    eligible_buses=hp_eligible_buses,
    add_tracking_columns=False,
    export_removed=True,  # only applies when there are deleted HP
    export_dir=output_dir,  # only applies when there are deleted HP
)

# Temporary Check: Amount of CPs after total amount changed
names = edisgo.topology.loads_df.query("type == 'charging_point'").index.astype(str)
print(
    {
        "existing": names.str.contains("Existing", case=False).sum(),
        "additional": names.str.contains("Additional", case=False).sum(),
        "rest": (
            ~(
                names.str.contains("Existing", case=False)
                | names.str.contains("Additional", case=False)
            )
        ).sum(),
        "total": len(names),
    }
)

############################ EV INTEGRATION PART ##############################

# Set Zero active power for batteries (let the OPF optimize dispatch freely)
storage_names = edisgo.topology.storage_units_df.index
timeindex = edisgo.timeseries.timeindex
edisgo.timeseries.storage_units_active_power = pd.DataFrame(
    0.0,
    index=timeindex,
    columns=storage_names,
)

### Add synthetic data to edisgo.heat_pump.cop_df and edisgo.heat_pump.heat_demand_df
# So the electrical analysis is not affected and no error appears
# edisgo.timeseries._loads_active_power.loc[:,edisgo.timeseries._loads_active_power.columns.str.contains("heat_")] = 0.5
hp_names = list(
    edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"]
        == "heat_pump"  # adjust filter to match your data
    ].index
)
timeindex = edisgo.timeseries.timeindex
cop = 3.0  # flat synthetic COP
# COP dataframe
edisgo.heat_pump.cop_df = pd.DataFrame(
    cop,
    index=timeindex,
    columns=hp_names,
)
# Heat demand = electrical active power * COP
# This means the thermal constraint is always just-met,
# so it won't add any extra restriction beyond what the electrical side already imposes
edisgo.heat_pump.heat_demand_df = edisgo.timeseries.loads_active_power[hp_names] * cop
###

# set reactive power time series
edisgo.set_time_series_reactive_power_control()
############################ END MANUAL FIXES #################################


edisgo = run_optimization_14a(edisgo)
edisgo.analyze()

# ── Slack diagnosis ──────────────────────────────────────────────────────────
slacks = edisgo.opf_results.grid_slacks_t
print("\n=== OPF Slack Diagnosis (v5) ===")
for name, df in [
    ("gen_nd_crt  (renewable curtailment)", slacks.gen_nd_crt),
    ("gen_d_crt   (disp. gen curtailment)", slacks.gen_d_crt),
    ("load_shed   (load shedding)", slacks.load_shedding),
    ("hp_shed     (HP load shedding)", slacks.hp_load_shedding),
]:
    total = df.abs().sum(axis=1)
    if (total > 5e-3).any():
        print(f"  {name}: {total.sum():.4f} MW  ← NON-ZERO")
    else:
        print(f"  {name}: 0 (not used)")

print("\n=== Voltage after OPF (edisgo.results.v_res) ===")
v = edisgo.results.v_res
print(f"  Min:  {v.min().min():.4f} p.u.")
print(f"  Max:  {v.max().max():.4f} p.u.")
viol = (v < 0.9) | (v > 1.1)
if viol.any().any():
    print(f"  Violations:{viol.sum().sum()}")
    print()
else:
    print("  No voltage violations.")
# ── End diagnosis ────────────────────────────────────────────────────────────
print("\n=== 14a analysis ===")
gen = edisgo.topology.generators_df
gen_t = edisgo.timeseries.generators_active_power
gen_14a = gen[gen.index.str.contains("14a")]
gen_t_14a = gen_t.loc[:, gen_14a.index]
print(f"Total use of 14a:{gen_t_14a.sum().sum()}")
print("\n=== end 14a analysis ===")


# # Create gif
# for ts in edisgo.timeseries.timeindex:
#     plot_network(edisgo, show=False, snapshot=str(ts))
# create_network_gif(duration=500)


# ── Presentation plots ───────────────────────────────────────────────────────
# Select days that have non-trivial §14a curtailment (threshold: 1 kW total)
curt_daily = (
    get_curtailment_data(edisgo)
    .groupby(edisgo.timeseries.timeindex.normalize())
    .sum()
    .sum(axis=1)
)
active_days = curt_daily[curt_daily > 1e-3].index.strftime("%Y-%m-%d").tolist()

print(f"\n=== Days with §14a curtailment: {active_days} ===")

for day in active_days:
    print(f"  Plotting {day}...")
    plot_load_before_after(edisgo, day=day, show=False, save=True)

print(f"Saved plots to ./plots/")
