import os

from datetime import datetime
import contextily as ctx
import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from edisgo import EDisGo


def get_curtailment_data(edisgo):
    """
    Return the §14a virtual generator curtailment time series.

    Returns a DataFrame of generators_active_power columns corresponding to
    hp_14a_support and cp_14a_support virtual generators, transposed so that
    the index is the generator name (ready for bus mapping).
    """
    gen_cols = [
        col
        for col in edisgo.timeseries.generators_active_power.columns
        if "hp_14a_support" in col
        or "cp_14a_support" in col
        or "charging_point_14a_support" in col
    ]
    return edisgo.timeseries.generators_active_power[gen_cols]


def create_network_gif(
    folder_path="./plots", output_name="network_evolution.gif", duration=1
):
    """
    Creates a GIF from PNG files in a folder.
    duration: time in seconds between frames
    """
    images = []

    # Get all png files that start with 'grid_analysis_'
    files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".png") and f.startswith("grid_analysis_")
    ]

    # IMPORTANT: Sort files by time.
    # Since your files are named 'grid_analysis_YYYY-MM-DD HH:MM:SS.png',
    # standard string sorting works perfectly for chronological order.
    files.sort()

    print(f"Found {len(files)} frames. Processing...")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(file_path))
        print(f"Added: {filename}")

    # Save the GIF
    # loop=0 means it will loop forever
    imageio.mimsave(output_name, images, duration=duration, loop=0)
    print(f"Success! GIF saved as {output_name}")


def plot_network(
    edisgo,
    snapshot: str = "2035-01-15 09:00:00",
    show: bool = True,
    save: bool = True,
    base_bus_size=0.000000002,
):
    results = edisgo.results

    n = edisgo.to_pypsa()

    coords = edisgo.topology.buses_df[["x", "y"]]
    coords = coords.reindex(n.buses.index)  # secure that index is matching
    n.buses["x"] = coords["x"].values
    n.buses["y"] = coords["y"].values

    line_columns = n.lines.index
    lines_t = results.s_res.loc[:, line_columns]

    # 1. Define limits for line loading
    loading_relative = (
        results.s_res.loc[snapshot, line_columns] / n.lines.s_nom
    )

    # 1. Limits für Farbskala (jetzt auf 0% - 100% bezogen)
    v_min, v_max = 0.0, 1.0
    norm_lines = mcolors.Normalize(vmin=v_min, vmax=v_max)

    # 2. Prepare bus data
    # Actual voltage in p.u.; diverging norm centered at nominal 1.0 p.u.
    bus_colors = edisgo.results.v_res.T[snapshot]

    # TwoSlopeNorm: purple = undervoltage (<1), blue = nominal (1), red = overvoltage (>1)
    norm_buses = mcolors.TwoSlopeNorm(vmin=0.9, vcenter=1.0, vmax=1.1)
    voltage_cmap = mcolors.LinearSegmentedColormap.from_list(
        "voltage", ["purple", "blue", "red"]
    )

    # --- (Curtailment logic and bus_sizes calculation) ---
    curt_14a = get_curtailment_data(edisgo).T

    # Clean up index names to match load names
    curt_14a["load"] = curt_14a.index
    curt_14a["load"] = curt_14a["load"].apply(
        lambda x: x.replace("cp_14a_support_", "").replace(
            "hp_14a_support_", ""
        )
    )

    # Map loads to their respective buses and aggregate curtailment per bus
    curt_14a["bus"] = curt_14a["load"].map(edisgo.topology.loads_df["bus"])
    grouped_14a = curt_14a.groupby("bus").sum()
    grouped_14a.columns = grouped_14a.columns.map(str)

    # Calculate bus sizes based on curtailment; reindex to include all buses in the network
    bus_sizes = base_bus_size + (grouped_14a[snapshot] * 0.000001)
    bus_sizes = bus_sizes.reindex(bus_colors.index, fill_value=base_bus_size)
    # -------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the grid
    n.plot(
        margin=0.05,
        ax=ax,
        geomap=False,
        bus_colors=bus_colors,
        bus_alpha=1,
        bus_sizes=bus_sizes,
        bus_cmap=voltage_cmap,
        bus_norm=norm_buses,
        line_colors=loading_relative,
        line_widths=1.6,
        line_cmap="jet",
        line_norm=norm_lines,
        title=f"Grid Analysis: {snapshot}",
        geometry=False,
    )

    # Add background basemap
    ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)

    # --- COLORBAR 1: LINE LOADING (LEFT SIDE) ---
    sm_lines = plt.cm.ScalarMappable(cmap="jet", norm=norm_lines)
    # Use location='left' and a slightly larger pad to avoid overlap with axis labels
    cb_lines = fig.colorbar(
        sm_lines,
        ax=ax,
        orientation="vertical",
        location="left",
        pad=0.08,
        aspect=20,
    )
    cb_lines.set_label("Line Loading [relative]", fontsize=8)

    # --- COLORBAR 2: BUS VOLTAGE (RIGHT SIDE) ---
    sm_buses = plt.cm.ScalarMappable(cmap=voltage_cmap, norm=norm_buses)
    # Default location is right
    cb_buses = fig.colorbar(
        sm_buses,
        ax=ax,
        orientation="vertical",
        location="right",
        pad=0.02,
        aspect=20,
    )
    cb_buses.set_label("Bus Voltage [p.u.]  — blue: under, yellow: nominal, red: over", fontsize=8)

    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(
            f"plots/grid_analysis_{snapshot}.png", dpi=300, bbox_inches="tight"
        )

    if show:
        plt.show()


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


#grid_path = "/home/carlos/LoMa/exec_folder/results/MGB_quo_model_pypsa"

# Whole husum paths
grid_path = "/home/paul/LoMa/loma-repo/results/Whole_Husum_model_pypsa"
path_husum_district_shp = "/home/paul/LoMa/loma-repo/data/Input_files/MV_grid_district/husum_district.shp"

# MGB paths
#grid_path = "/home/paul/LoMa/MGB_2035_model_pypsa"
#path_husum_district_shp = "/home/paul/LoMa/loma-repo/data/Input_files/MGB_district"

edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(0, 167))

mv_grid_geom = gpd.read_file(path_husum_district_shp).to_crs(4326)
edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
edisgo.topology.grid_district["srid"] = 4326

# edisgo.topology.check_integrity()
# pypsa_n = edisgo.to_pypsa()
# edisgo.analyze()

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
    transfer_ts_from_new_to_existing_cp,
    set_charging_points_to_target,
    set_heat_pumps_to_target,
    buses_with_existing_loads,
)

# -------------------------
# Import + distribute + integrate EV data (creates new charging points) 
# -------------------------
'''
After this function there are no time series yet. Only charging points and 
a overall demand which is then transferred into a time series in 
apply_charging_strategy.
'''
edisgo.import_electromobility_14a( 
    scenario="eGon2035",
    import_electromobility_data_kwds={
        "shapefile_path": path_husum_district_shp
    },
)

# -------------------------
# Apply charging strategy (writes time series for the new integrated charging points from eDisGo)
# -------------------------
'''
Without the preparation of Q before charging strategy I got an error while 
apply_charging_strategy which was caused by deviating time index.
After this step only the charging point from eDisGo have a time series.
'''
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
'''
This step then finally transfers the time series from suitable eDisGo 
charging_points to Existing_ und Additional_ charging points which are 
created on the LoMa side.
'''
ev_match_results = transfer_ts_from_new_to_existing_cp(
    edisgo,
    existing_markers=("Existing", "Additional"),
    radius_1=2000.0,
    tol_1=0.15,
    radius_2=2000.0,
    tol_2=0.9,   
)

# ============================================================
# Optional Utilities for sensitivity analysis/chaning the amount of cp/hp
# - target by absolute value or relative percentage
# - Only use one option at a time (traget_total, percentage)
# ============================================================
'''
In this step the total amount of charging points or heat pumps can be adjusted.
Either by percentage or by a total amount including the infrastructure from
LoMa. When deleting CP/HP there is an option to export the deleted ones.
New CP/HP will have 'dup' in their name.
'''
output_dir = "/home/paul/LoMa/test/shapes"

cp_eligible_buses = buses_with_existing_loads(edisgo)
hp_eligible_buses = buses_with_existing_loads(edisgo)

change_cp_amount = set_charging_points_to_target(
    edisgo,
    target_total=500, # sets total amount of CP to 1000
    #percentage=0.10, # increases total amount of CP by 10%
    #percentage=-0.10, # decreases total amount of CP by 10%
    eligible_buses=cp_eligible_buses,
    removal_priority=["Additional", "Existing"],
    add_tracking_columns=False,
    export_removed=True, # only applies when negative percentage for debugging
    export_dir=output_dir, # only applies when negative percentage for debugging
)

# change_hp_amount = set_heat_pumps_to_target(
#     edisgo,
#     target_total=50, # sets total amount of HP to 50
#     #percentage=0.10, # increases total amount of CP by 10%
#     #percentage=-0.10, # decreases total amount of CP by 10%
#     eligible_buses=hp_eligible_buses,
#     add_tracking_columns=False,
#     export_removed=True, # only applies when negative percentage for debugging
#     export_dir=output_dir, # only applies when negative percentage for debugging
# )


# ==========================
# Graphs
# ==========================

# ============================================================
# Common CP selection + classification
# ============================================================
cp_ids = edisgo.topology.loads_df.index[
    edisgo.topology.loads_df["type"] == "charging_point"
]
cp_ids = cp_ids.intersection(edisgo.timeseries.loads_active_power.columns)

cp_ts = edisgo.timeseries.loads_active_power.loc[:, cp_ids].copy()

def classify_cp(col):
    s = str(col)
    if s.startswith("Existing_Charging_Point"):
        return "existing"
    elif s.startswith("Additional_Charging_Point"):
        return "additional"
    elif s.startswith("cp_dup_"):
        return "duplicated"
    else:
        return "new"

cp_groups = pd.Series({c: classify_cp(c) for c in cp_ts.columns}, name="group")

existing_cols = cp_groups[cp_groups == "existing"].index.tolist()
additional_cols = cp_groups[cp_groups == "additional"].index.tolist()
duplicated_cols = cp_groups[cp_groups == "duplicated"].index.tolist()
new_cols = cp_groups[cp_groups == "new"].index.tolist()

print(f"Existing CPs:   {len(existing_cols)}")
print(f"Additional CPs: {len(additional_cols)}")
print(f"New CPs:        {len(new_cols)}")
print(f"Duplicated CPs: {len(duplicated_cols)}")


# ============================================================
# Graph 1: 5 individual EVs
# ============================================================
five = cp_ts.columns[0:5]

cp_ts[five].plot(figsize=(12, 5))
plt.title("Charging demand – 5 individual CPs")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.tight_layout()
plt.show()


# ============================================================
# Graph 2: Aggregated EVs
# ============================================================
cp_sum = cp_ts.sum(axis=1)

cp_sum.plot(figsize=(12, 5))
plt.title("Aggregated EV charging demand")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.tight_layout()
plt.show()


# ============================================================
# Graph 3: EV load vs grid load
# ============================================================
total_load = edisgo.timeseries.loads_active_power.sum(axis=1)

plt.figure(figsize=(12, 5))
plt.plot(total_load, label="Total load")
plt.plot(cp_sum, label="CP load")
plt.legend()
plt.title("CP load vs total grid load")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.tight_layout()
plt.show()

# ============================================================
# Graph 4: Compare CP timeseries by type (energy-normalized)
# existing vs additional vs new
# ============================================================
group_existing = cp_ts[existing_cols].sum(axis=1) if existing_cols else None
group_additional = cp_ts[additional_cols].sum(axis=1) if additional_cols else None
group_new = cp_ts[new_cols].sum(axis=1) if new_cols else None

# infer dt from time index
if len(cp_ts.index) > 1:
    dt = (cp_ts.index[1] - cp_ts.index[0]).total_seconds() / 3600.0
else:
    dt = 1.0

def normalize_to_energy(series, target_mwh=100.0):
    if series is None:
        return None
    energy = series.sum() * dt
    if energy == 0:
        return None
    return series * (target_mwh / energy)

group_existing_norm = normalize_to_energy(group_existing, target_mwh=100.0)
group_additional_norm = normalize_to_energy(group_additional, target_mwh=100.0)
group_new_norm = normalize_to_energy(group_new, target_mwh=100.0)

plt.figure(figsize=(12, 5))

if group_existing_norm is not None:
    plt.plot(group_existing_norm, label=f"Existing CPs ({len(existing_cols)})")

if group_additional_norm is not None:
    plt.plot(group_additional_norm, label=f"Additional CPs ({len(additional_cols)})")

if group_new_norm is not None:
    plt.plot(group_new_norm, label=f"New CPs ({len(new_cols)})")

plt.title("CP timeseries by type (Energy-normalized)")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Graph 5: Plot EV loads per use case
# ============================================================
use_cases = ["home", "work", "public", "hpc"]

aggregated = {}
counts = {}

for uc in use_cases:
    cols = [c for c in cp_ts.columns if f"_{uc}_" in c]
    counts[uc] = len(cols)
    aggregated[uc] = cp_ts[cols].sum(axis=1) if len(cols) > 0 else None

plt.figure(figsize=(12, 6))

for uc in use_cases:
    if aggregated[uc] is not None:
        plt.plot(aggregated[uc], label=f"{uc} ({counts[uc]} CPs)")

plt.title("Aggregated Charging Demand per Use Case")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Graph 6: Differentiate between LV and MV grid EV loads by v_nom
# ============================================================
loads_df = edisgo.topology.loads_df
cp_loads = loads_df[loads_df["type"] == "charging_point"].copy()

buses = edisgo.topology.buses_df

cp_with_voltage = cp_loads.copy()
cp_with_voltage["v_nom"] = cp_with_voltage["bus"].map(buses["v_nom"])

cp_with_voltage = cp_with_voltage.loc[
    cp_with_voltage.index.intersection(edisgo.timeseries.loads_active_power.columns)
].copy()

cp_with_voltage["voltage_level"] = cp_with_voltage["v_nom"].apply(
    lambda x: "LV" if x <= 1 else "MV"
)

cp_ts_vm = edisgo.timeseries.loads_active_power.loc[:, cp_with_voltage.index]

lv_cols = cp_with_voltage.loc[
    cp_with_voltage["voltage_level"] == "LV"
].index

mv_cols = cp_with_voltage.loc[
    cp_with_voltage["voltage_level"] == "MV"
].index

lv_sum = cp_ts_vm[lv_cols].sum(axis=1) if len(lv_cols) > 0 else None
mv_sum = cp_ts_vm[mv_cols].sum(axis=1) if len(mv_cols) > 0 else None

plt.figure(figsize=(12, 6))
if lv_sum is not None:
    plt.plot(lv_sum, label=f"LV ({len(lv_cols)} CPs)")
if mv_sum is not None:
    plt.plot(mv_sum, label=f"MV ({len(mv_cols)} CPs)")
plt.title("EV Charging Demand: LV vs MV")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Graph 7: Comparing CP timeseries
# existing vs additional vs new vs duplicated
# ============================================================
group_dup = cp_ts[duplicated_cols].sum(axis=1) if duplicated_cols else None

def energy_mwh(series):
    if series is None:
        return 0.0
    return series.sum() * dt

E_existing = energy_mwh(group_existing)
E_additional = energy_mwh(group_additional)
E_new = energy_mwh(group_new)
E_dup = energy_mwh(group_dup)

E_target = 100.0  # MWh

def normalize_to_energy_explicit(series, energy_mwh, target_mwh=100.0):
    if series is None or energy_mwh == 0:
        return None
    return series * (target_mwh / energy_mwh)

group_existing_norm = normalize_to_energy_explicit(group_existing, E_existing, E_target)
group_additional_norm = normalize_to_energy_explicit(group_additional, E_additional, E_target)
group_new_norm = normalize_to_energy_explicit(group_new, E_new, E_target)
group_dup_norm = normalize_to_energy_explicit(group_dup, E_dup, E_target)

# ---- Graph 7 normalized ----
plt.figure(figsize=(12, 5))

if group_existing_norm is not None:
    plt.plot(group_existing_norm, label=f"Existing CPs ({len(existing_cols)})")

if group_additional_norm is not None:
    plt.plot(group_additional_norm, label=f"Additional CPs ({len(additional_cols)})")

if group_new_norm is not None:
    plt.plot(group_new_norm, label=f"New CPs ({len(new_cols)})")

if group_dup_norm is not None:
    plt.plot(group_dup_norm, label=f"Duplicated CPs ({len(duplicated_cols)})")

plt.title("CP timeseries by type (Energy-normalized)")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Graph 7.1 absolute ----
plt.figure(figsize=(12, 5))

if group_existing is not None:
    plt.plot(group_existing, label=f"Existing CPs ({len(existing_cols)})")

if group_additional is not None:
    plt.plot(group_additional, label=f"Additional CPs ({len(additional_cols)})")

if group_new is not None:
    plt.plot(group_new, label=f"New CPs ({len(new_cols)})")

if group_dup is not None:
    plt.plot(group_dup, label=f"Duplicated CPs ({len(duplicated_cols)})")

plt.title("CP timeseries by type (Aggregated absolute load)")
plt.ylabel("Power [MW]")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()

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
edisgo.heat_pump.heat_demand_df = (
    edisgo.timeseries.loads_active_power[hp_names] * cop
)
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
    ("load_shed   (load shedding)",         slacks.load_shedding),
    ("hp_shed     (HP load shedding)",      slacks.hp_load_shedding),
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
gen_14a  = gen[gen.index.str.contains("14a")]
gen_t_14a = gen_t.loc[:,gen_14a.index]
print(f"Total use of 14a:{gen_t_14a.sum().sum()}")
print("\n=== end 14a analysis ===")

# Plot
for ts in edisgo.timeseries.timeindex:
    plot_network(edisgo, show=False, snapshot=str(ts))
create_network_gif(duration=500)

# edisgo.results.v_res
# edisgo.results.s_res

# load_t = edisgo.timeseries.loads_active_power
# g = edisgo.topology.generators_df
# gt = edisgo.timeseries.generators_active_power
# lines = edisgo.topology.lines_df

# demand_q = edisgo.timeseries.loads_reactive_power
