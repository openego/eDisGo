import os

from datetime import datetime
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from edisgo import EDisGo


def analyze_curtailment_results(edisgo, output_dir="results_14a"):
    """
    Analyze §14a curtailment results and generate statistics.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object with optimization results
    output_dir : str
        Directory to save results

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print("📊 Analyzing §14a Curtailment Results")
    print(f"{'='*80}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get curtailment data for both heat pumps and charging points
    hp_gen_cols = [
        col
        for col in edisgo.timeseries.generators_active_power.columns
        if "hp_14a_support" in col
    ]
    cp_gen_cols = [
        col
        for col in edisgo.timeseries.generators_active_power.columns
        if "cp_14a_support" in col or "charging_point_14a_support" in col
    ]

    all_gen_cols = hp_gen_cols + cp_gen_cols

    if len(all_gen_cols) == 0:
        print("⚠ WARNING: No §14a virtual generators found in results!")
        return {}

    curtailment = edisgo.timeseries.generators_active_power[all_gen_cols]

    # Get heat pump and charging point load data
    heat_pumps = edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"] == "heat_pump"
    ]
    charging_points = edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"] == "charging_point"
    ]

    all_flexible_loads = pd.concat([heat_pumps, charging_points])
    flexible_loads = edisgo.timeseries.loads_active_power[
        all_flexible_loads.index
    ]

    # Separate for detailed analysis
    hp_loads = (
        edisgo.timeseries.loads_active_power[heat_pumps.index]
        if len(heat_pumps) > 0
        else pd.DataFrame()
    )
    cp_loads = (
        edisgo.timeseries.loads_active_power[charging_points.index]
        if len(charging_points) > 0
        else pd.DataFrame()
    )

    # Calculate statistics
    total_curtailment = curtailment.sum().sum()
    total_flexible_load = flexible_loads.sum().sum()
    total_hp_load = hp_loads.sum().sum() if len(hp_loads) > 0 else 0
    total_cp_load = cp_loads.sum().sum() if len(cp_loads) > 0 else 0
    curtailment_percentage = (
        (total_curtailment / total_flexible_load * 100)
        if total_flexible_load > 0
        else 0
    )

    flexible_curtailment_total = curtailment.sum()
    curtailed_units = flexible_curtailment_total[
        flexible_curtailment_total > 0
    ]

    # Separate HP and CP curtailment
    hp_curtailment_total = (
        curtailment[hp_gen_cols].sum() if len(hp_gen_cols) > 0 else pd.Series()
    )
    cp_curtailment_total = (
        curtailment[cp_gen_cols].sum() if len(cp_gen_cols) > 0 else pd.Series()
    )
    curtailed_hps = (
        hp_curtailment_total[hp_curtailment_total > 0]
        if len(hp_curtailment_total) > 0
        else pd.Series()
    )
    curtailed_cps = (
        cp_curtailment_total[cp_curtailment_total > 0]
        if len(cp_curtailment_total) > 0
        else pd.Series()
    )

    # Time series statistics
    curtailment_per_timestep = curtailment.sum(axis=1)
    max_curtailment_timestep = curtailment_per_timestep.idxmax()
    max_curtailment_value = curtailment_per_timestep.max()

    # Daily statistics
    curtailment_daily = curtailment_per_timestep.resample("D").sum()

    # Monthly statistics
    curtailment_monthly = curtailment_per_timestep.resample("M").sum()

    results = {
        "total_curtailment_MWh": total_curtailment,
        "total_flexible_load_MWh": total_flexible_load,
        "total_hp_load_MWh": total_hp_load,
        "total_cp_load_MWh": total_cp_load,
        "curtailment_percentage": curtailment_percentage,
        "num_virtual_gens": len(all_gen_cols),
        "num_hp_gens": len(hp_gen_cols),
        "num_cp_gens": len(cp_gen_cols),
        "num_curtailed_hps": len(curtailed_hps),
        "num_curtailed_cps": len(curtailed_cps),
        "max_curtailment_MW": curtailment.max().max(),
        "max_curtailment_timestep": max_curtailment_timestep,
        "max_curtailment_value_MW": max_curtailment_value,
        "curtailment_per_timestep": curtailment_per_timestep,
        "curtailment_daily": curtailment_daily,
        "curtailment_monthly": curtailment_monthly,
        "curtailment_data": curtailment,
        "hp_curtailment_data": (
            curtailment[hp_gen_cols]
            if len(hp_gen_cols) > 0
            else pd.DataFrame()
        ),
        "cp_curtailment_data": (
            curtailment[cp_gen_cols]
            if len(cp_gen_cols) > 0
            else pd.DataFrame()
        ),
        "hp_loads": hp_loads,
        "cp_loads": cp_loads,
        "flexible_loads": flexible_loads,
        "hp_curtailment_total": hp_curtailment_total,
        "cp_curtailment_total": cp_curtailment_total,
        "curtailed_hps": curtailed_hps,
        "curtailed_cps": curtailed_cps,
    }

    # Print summary
    print("\n📈 Summary Statistics:")
    print(
        f"  Virtual generators: {len(all_gen_cols)} (HPs: {len(hp_gen_cols)}, CPs: {len(cp_gen_cols)})"
    )
    print(f"  Heat pumps curtailed: {len(curtailed_hps)} / {len(heat_pumps)}")
    print(
        f"  Charging points curtailed: {len(curtailed_cps)} / {len(charging_points)}"
    )
    print(f"  Total curtailment: {total_curtailment:.2f} MWh")
    print(
        f"  Total flexible load: {total_flexible_load:.2f} MWh (HP: {total_hp_load:.2f}, CP: {total_cp_load:.2f})"
    )
    print(f"  Curtailment ratio: {curtailment_percentage:.2f}%")
    print(f"  Max curtailment: {curtailment.max().max():.4f} MW")
    print(
        f"  Max total curtailment (timestep): {max_curtailment_value:.4f} MW at {max_curtailment_timestep}"
    )

    if len(curtailed_hps) > 0:
        print("\n  Top 5 curtailed heat pumps:")
        for i, (hp, value) in enumerate(
            curtailed_hps.sort_values(ascending=False).head().items(), 1
        ):
            hp_name = hp.replace("hp_14a_support_", "")
            print(f"    {i}. {hp_name}: {value:.4f} MWh")

    if len(curtailed_cps) > 0:
        print("\n  Top 5 curtailed charging points:")
        for i, (cp, value) in enumerate(
            curtailed_cps.sort_values(ascending=False).head().items(), 1
        ):
            cp_name = cp.replace("cp_14a_support_", "").replace(
                "charging_point_14a_support_", ""
            )
            print(f"    {i}. {cp_name}: {value:.4f} MWh")

    # Save statistics to CSV
    stats_df = pd.DataFrame(
        {
            "Metric": [
                "Total Curtailment (MWh)",
                "Total Flexible Load (MWh)",
                "Total HP Load (MWh)",
                "Total CP Load (MWh)",
                "Curtailment Percentage (%)",
                "Virtual Generators (Total)",
                "Virtual Generators (HPs)",
                "Virtual Generators (CPs)",
                "Curtailed HPs",
                "Curtailed CPs",
                "Max Curtailment (MW)",
                "Max Total Curtailment (MW)",
            ],
            "Value": [
                f"{total_curtailment:.2f}",
                f"{total_flexible_load:.2f}",
                f"{total_hp_load:.2f}",
                f"{total_cp_load:.2f}",
                f"{curtailment_percentage:.2f}",
                len(all_gen_cols),
                len(hp_gen_cols),
                len(cp_gen_cols),
                len(curtailed_hps),
                len(curtailed_cps),
                f"{curtailment.max().max():.4f}",
                f"{max_curtailment_value:.4f}",
            ],
        }
    )
    stats_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    print(
        f"\n✓ Summary statistics saved to {output_dir}/summary_statistics.csv"
    )

    # Save detailed curtailment data
    curtailment.to_csv(f"{output_dir}/curtailment_timeseries.csv")
    curtailment_daily.to_csv(f"{output_dir}/curtailment_daily.csv")
    curtailment_monthly.to_csv(f"{output_dir}/curtailment_monthly.csv")
    if len(hp_curtailment_total) > 0:
        hp_curtailment_total.to_csv(f"{output_dir}/hp_curtailment_total.csv")
    if len(cp_curtailment_total) > 0:
        cp_curtailment_total.to_csv(f"{output_dir}/cp_curtailment_total.csv")

    print(f"✓ Detailed data saved to {output_dir}/")

    return results


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
    # Calculating voltage deviation from nominal (1.0 p.u.)
    bus_colors = (1 - edisgo.results.v_res.T[snapshot]).apply(abs)

    # Voltage limits (adjust vmin/vmax based on your bus_colors results)
    norm_buses = mcolors.Normalize(vmin=0.0, vmax=0.3)

    # --- (Curtailment logic and bus_sizes calculation) ---
    curt_14a = analyze_curtailment_results(edisgo, output_dir="results_14a")[
        "curtailment_data"
    ].T

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
        bus_cmap="jet",
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
    sm_buses = plt.cm.ScalarMappable(cmap="jet", norm=norm_buses)
    # Default location is right
    cb_buses = fig.colorbar(
        sm_buses,
        ax=ax,
        orientation="vertical",
        location="right",
        pad=0.02,
        aspect=20,
    )
    cb_buses.set_label("Voltage Deviation |1 - V| [p.u.]", fontsize=8)

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


#grid_path = "/home/carlos/LoMa/exec_folder/results/MGB_model_pypsa"
grid_path = "/home/paul/LoMa/loma-repo/results/Whole_Husum_model_pypsa"

edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(0, 167))

# mv_grid_geom = gpd.read_file(
#     "/home/carlos/LoMa/exec_folder/data/Input_files/MV_grid_district/husum_district.shp"
# )
# mv_grid_geom = mv_grid_geom.to_crs(4326)

# edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
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

path_husum_district_shp = "/home/paul/LoMa/loma-repo/data/Input_files/MV_grid_district/husum_district.shp"
output_dir = "/home/paul/LoMa/test/shapes"

mv_grid_geom = gpd.read_file(path_husum_district_shp).to_crs(4326)
edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
edisgo.topology.grid_district["srid"] = 4326

# Import + distribute + integrate EV data (creates new charging points) 
edisgo.import_electromobility_14a( 
    scenario="eGon2035",
    import_electromobility_data_kwds={
        "shapefile_path": path_husum_district_shp
    },
)

# -------------------------
# Fix loads_reactive_power:
# Make it explicitly consistent with the current timeindex before charging strategy
# -------------------------
ti = edisgo.timeseries.timeindex
edisgo.timeseries.loads_reactive_power = pd.DataFrame(index=ti)

# -------------------------
# Apply charging strategy (writes time series for the new integrated charging points from eDisGo)
# -------------------------
edisgo.apply_charging_strategy(strategy="dumb")

#temp
print("\n--- CHECK AFTER CHARGING STRATEGY ---")
cp_topology = edisgo.topology.loads_df.index[
    edisgo.topology.loads_df["type"] == "charging_point"
]
cp_missing = cp_topology.difference(edisgo.timeseries.loads_active_power.columns)

print("CPs in topology:", len(cp_topology))
print("CPs missing in active power:", len(cp_missing))
if len(cp_missing):
    print("first missing:", cp_missing[:20].tolist())
#temp

# Transfer time series from new eDisGo CPs to existing CPs
# Existing cp will get 'matched' with new cp by nearest bus location and p_set within a tolerance
ev_match_results = transfer_ts_from_new_to_existing_cp(
    edisgo,
    existing_markers=("Existing", "Additional"),
    radius_1=2000.0,
    tol_1=0.15,
    radius_2=2000.0,
    tol_2=0.9,   
)

#temp
print("\n--- CHECK AFTER MATCHING / BEFORE REACTIVE CONTROL ---")
cp_topology = edisgo.topology.loads_df.index[
    edisgo.topology.loads_df["type"] == "charging_point"
]
cp_missing = cp_topology.difference(edisgo.timeseries.loads_active_power.columns)

print("CPs in topology:", len(cp_topology))
print("CPs missing in active power:", len(cp_missing))
if len(cp_missing):
    print("first missing:", cp_missing[:20].tolist())
#temp

edisgo.set_time_series_reactive_power_control()

# ============================================================
# Utilities for sensitivity analysis/chaning the amount of cp/hp
# - supports charging points and heat pumps
# - target by absolute value or relative percentage
# - Only use one option at a time (traget_total, percentage)
# - for charging points: existing ones are removed last
# - duplicates/removes topology rows + power-flow time series
# ============================================================
cp_eligible_buses = buses_with_existing_loads(edisgo)
hp_eligible_buses = buses_with_existing_loads(edisgo)

# change_cp_amount = set_charging_points_to_target(
#     edisgo,
#     #target_total=1000, # sets total amount of CP to 1000
#     #percentage=0.10, # increases total amount of CP by 10%
#     #percentage=-0.10, # decreases total amount of CP by 10%
#     eligible_buses=cp_eligible_buses,
#     existing_marker="Existing",
#     add_tracking_columns=False,
#     export_removed=True, # only applies when negative percentage for debugging
#     export_dir=output_dir, # only applies when negative percentage for debugging
# )

# change_hp_amount = set_heat_pumps_to_target(
#     edisgo,
#     #target_total=500, # sets total amount of HP to 500
#     #percentage=0.10, # increases total amount of CP by 10%
#     #percentage=-0.10, # decreases total amount of CP by 10%
#     eligible_buses=hp_eligible_buses,
#     add_tracking_columns=False,
#     export_removed=True, # only applies when negative percentage for debugging
#     export_dir=output_dir, # only applies when negative percentage for debugging
# )

# edisgo.topology.loads_df = edisgo.topology.loads_df[
#     edisgo.topology.loads_df.type != "charging_point"
# ]


# ==========================
# Graphs
# ==========================
import pandas as pd

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

# ==========================
# EV Exports
# ==========================
from shapely.geometry import Point
output_dir="/home/paul/LoMa/test/shapes"

def export_emob_debug_data(edisgo_obj, output_dir=output_dir):
    """
    Exports all relevant electromobility DataFrames from an EDisGo object
    to CSV and GeoPackage (WKT for geometries) for debugging.

    Parameters
    ----------
    edisgo_obj : EDisGo
    output_dir : str
        Directory where the exported files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ---- Charging processes ----
    df = edisgo_obj.electromobility.charging_processes_df
    if df is not None and not df.empty:

        df.to_csv(os.path.join(output_dir, "charging_processes.csv"), index=False)
        print("[EXPORT] Exported charging_processes.csv")
    else:
        print("[EXPORT][ERROR] charging_processes_df is empty, skipping export.")


    # ---- SimBeV config ----
    df = edisgo_obj.electromobility.simbev_config_df
    if df is not None and not df.empty:
        df.to_csv(os.path.join(output_dir, "simbev_config.csv"), index=False)
        print("[EXPORT] Exported simbev_config.csv")
    else:
        print("[EXPORT][ERROR] simbev_config_df is empty, skipping export.")

    # ---- Potential charging parks ----
    gdf = edisgo_obj.electromobility.potential_charging_parks_gdf
    if gdf is not None and not gdf.empty:
        # Konvertiere alle Geometriespalten in WKT
        geom_cols = gdf.select_dtypes(include="geometry").columns
        for col in geom_cols:
            gdf[col] = gdf[col].apply(lambda x: x.wkt if x is not None else None)

        # Speichere als CSV (GPKG geht nur mit einer Geometriespalte)
        gdf.to_csv(os.path.join(output_dir, "potential_charging_parks.csv"), index=False)
        print("[EXPORT] Exported potential_charging_parks.csv with WKT geometries")
    else:
        print("[EXPORT][ERROR] potential_charging_parks_gdf is empty, skipping export.")

    print(f"[EXPORT] All exports saved to {output_dir}")

def clean_point_geometry_gdf(edisgo):
    import geopandas as gpd
    import pandas as pd
    from shapely import wkt

    gdf = edisgo.electromobility.potential_charging_parks_gdf.copy()

    # WKT -> echte Geometrie
    gdf["geometry"] = gdf["geometry"].apply(wkt.loads)

    # Alles neu und sauber aufsetzen
    df = pd.DataFrame(gdf.drop(columns="geometry"))

    clean_gdf = gpd.GeoDataFrame(
        df,
        geometry=gdf["geometry"],
        crs="EPSG:4326"
    )

    return clean_gdf

def export_parks_points_shapefile(edisgo, output_dir=output_dir):
    gdf = clean_point_geometry_gdf(edisgo)

    shp_path = f"{output_dir}/potential_charging_parks_points.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile")

    print("[EXPORT] Shapefile written to:", shp_path)

def export_integrated_charging_points_shapefile(edisgo, output_dir=output_dir):
    """
    Export all actually integrated charging points as a shapefile
    using the REAL bus coordinates they were connected to.

    This shows the true grid connection location chosen by eDisGo.
    """

    import os
    import geopandas as gpd

    os.makedirs(output_dir, exist_ok=True)

    loads = edisgo.topology.loads_df.copy()

    # Nur Charging Points
    cp_df = loads[loads["type"] == "charging_point"].copy()

    if cp_df.empty:
        print("[EXPORT][ERROR] No integrated charging points found.")
        return

    # Bus-Koordinaten anhängen
    buses = edisgo.topology.buses_df[["x", "y"]]
    cp_df = cp_df.join(buses, on="bus")

    # Geometrie aus Bus-Koordinaten
    cp_df["geometry"] = cp_df.apply(
        lambda row: Point(row["x"], row["y"]), axis=1
    )

    gdf = gpd.GeoDataFrame(cp_df, geometry="geometry", crs="EPSG:4326")

    shp_path = os.path.join(output_dir, "integrated_charging_points_grid_connection.shp")
    gdf.to_file(shp_path, driver="ESRI Shapefile")

    print(f"[EXPORT] Integrated charging points exported: {len(gdf)}")
    print(f"[EXPORT] Shapefile written to: {shp_path}")

def export_ev_profiles_one_week(
    edisgo,
    out_dir: str,
    *,
    fname_prefix: str = "ev_profiles_week",
):
    """
    Export weekly CP time series (P, optional Q) for ALL charging points in the current run.
    Exports:
      - {prefix}_active_power_MW.csv   (index=time, columns=cp_id)
      - {prefix}_meta.csv             (per-CP metadata)
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- select charging_point loads from topology ---
    loads_df = edisgo.topology.loads_df
    cp_topology = loads_df[loads_df["type"] == "charging_point"].copy()
    cp_ids_topology = cp_topology.index

    # --- intersect with TS columns (robust) ---
    tsP = edisgo.timeseries.loads_active_power
    cp_ids = cp_ids_topology.intersection(tsP.columns)

    missing = cp_ids_topology.difference(tsP.columns)

    # --- slice exactly the currently configured timeindex (your week @ 15min after resample) ---
    # (This is already the week if you ran set_timeindex(TIMEINDEX_ONE_WEEK) and resample_timeseries())
    evP_week = tsP.loc[edisgo.timeseries.timeindex, cp_ids].copy()

    # --- export P in MW (native) ---
    path_mw = os.path.join(out_dir, f"{fname_prefix}_active_power_MW.csv")
    evP_week.to_csv(path_mw, index=True)
    print(f"[EXPORT] Wrote: {path_mw}")

    # --- metadata export (per CP) ---
    buses = edisgo.topology.buses_df[["x", "y", "v_nom"]].copy()

    meta = cp_topology.loc[cp_ids, ["bus", "p_set"]].copy()
    meta = meta.join(buses, on="bus", how="left")

    # Use case from name convention (Charging_Point_*_{home/work/public/hpc}_*)
    def _use_case_from_id(cp_id: str) -> str:
        s = str(cp_id).lower()
        for uc in ["home", "work", "public", "hpc"]:
            if f"_{uc}_" in s:
                return uc
        return "unknown"

    meta["use_case"] = [ _use_case_from_id(i) for i in meta.index ]
    meta["voltage_level"] = meta["v_nom"].apply(lambda x: "LV" if pd.notna(x) and float(x) <= 1 else "MV")

    path_meta = os.path.join(out_dir, f"{fname_prefix}_meta.csv")
    meta.to_csv(path_meta, index=True)
    print(f"[EXPORT] Wrote: {path_meta}")

    return {
        "cp_ids_exported": cp_ids.tolist(),
        "cp_ids_missing_ts": missing.tolist(),
        "paths": {
            "P_MW": path_mw,
            "meta": path_meta,
        },
    }

# Choose an output folder (reuse your existing output_dir or make a subfolder)
ev_export_dir = os.path.join(output_dir, "ev_profiles")
export_ev_profiles_one_week(
    edisgo,
    out_dir=ev_export_dir,
    fname_prefix="ev_profiles_week_2023_01_01",
)

export_emob_debug_data(edisgo) #CSV (simbev_config, potential_charging_parks, charging_processes)

export_parks_points_shapefile(edisgo) # only point geometries of potential charging parks

export_integrated_charging_points_shapefile(edisgo)

# ==========================
# Heat pump exports + plots
# ==========================
def export_hp_profiles_one_week(
    edisgo,
    out_dir: str,
    *,
    fname_prefix: str = "hp_profiles_week",
):
    """
    Export weekly HP time series (P, optional Q) for ALL heat pumps in the current run.

    Exports:
      - {prefix}_active_power_MW.csv
      - {prefix}_meta.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- select heat pump loads from topology ---
    loads_df = edisgo.topology.loads_df
    hp_topology = loads_df[loads_df["type"] == "heat_pump"].copy()
    hp_ids_topology = hp_topology.index

    # --- intersect with TS columns ---
    tsP = edisgo.timeseries.loads_active_power
    hp_ids = hp_ids_topology.intersection(tsP.columns)

    missing = hp_ids_topology.difference(tsP.columns)

    # --- active power export ---
    hpP_week = tsP.loc[edisgo.timeseries.timeindex, hp_ids].copy()
    path_p_mw = os.path.join(out_dir, f"{fname_prefix}_active_power_MW.csv")
    hpP_week.to_csv(path_p_mw, index=True)
    print(f"[EXPORT][HP] Wrote: {path_p_mw}")

    # --- metadata export ---
    buses = edisgo.topology.buses_df[["x", "y", "v_nom"]].copy()

    meta = hp_topology.loc[hp_ids].copy()
    keep_cols = [c for c in ["bus", "p_set"] if c in meta.columns]
    meta = meta[keep_cols].join(buses, on="bus", how="left")

    meta["is_duplicate"] = meta.index.astype(str).str.startswith("hp_dup_")
    meta["voltage_level"] = meta["v_nom"].apply(
        lambda x: "LV" if pd.notna(x) and float(x) <= 1 else "MV"
    )

    # optional provenance columns if you used add_tracking_columns=True
    for col in ["source_load_id", "is_duplicate"]:
        if col in hp_topology.columns and col not in meta.columns:
            meta[col] = hp_topology.loc[hp_ids, col]

    path_meta = os.path.join(out_dir, f"{fname_prefix}_meta.csv")
    meta.to_csv(path_meta, index=True)
    print(f"[EXPORT][HP] Wrote: {path_meta}")

    return {
        "hp_ids_exported": hp_ids.tolist(),
        "hp_ids_missing_ts": missing.tolist(),
        "paths": {
            "P_MW": path_p_mw,
            "meta": path_meta,
        },
    }


def export_integrated_heat_pumps_shapefile(edisgo, output_dir):
    """
    Export all heat pumps as a shapefile using their connected bus coordinates.
    Includes duplicated heat pumps as long as type == 'heat_pump'.
    """
    os.makedirs(output_dir, exist_ok=True)

    loads = edisgo.topology.loads_df.copy()
    hp_df = loads[loads["type"] == "heat_pump"].copy()

    if hp_df.empty:
        print("[EXPORT][ERROR][HP] No heat pumps found.")
        return

    buses = edisgo.topology.buses_df[["x", "y"]]
    hp_df = hp_df.join(buses, on="bus", how="left")

    hp_df["geometry"] = hp_df.apply(
        lambda row: Point(row["x"], row["y"]), axis=1
    )

    gdf = gpd.GeoDataFrame(hp_df, geometry="geometry", crs="EPSG:4326")

    shp_path = os.path.join(output_dir, "integrated_heat_pumps_grid_connection.shp")
    gdf.to_file(shp_path, driver="ESRI Shapefile")

    print(f"[EXPORT][HP] Integrated heat pumps exported: {len(gdf)}")
    print(f"[EXPORT][HP] Shapefile written to: {shp_path}")


def plot_heat_pump_profiles_comparison(
    edisgo,
    *,
    original_prefixes=None,
    duplicate_prefix: str = "hp_dup_",
    title_suffix: str = "",
):
    """
    Plot aggregated HP profiles split into:
      - original/status-quo HPs (identified by prefixes)
      - duplicated HPs (identified by duplicate_prefix)
      - all remaining HPs

    Produces:
      1) energy-normalized plot
      2) absolute aggregated load plot
    """
    if original_prefixes is None:
        # adapt this if your original HP naming scheme is known
        original_prefixes = ["Heat_Pump", "Existing_Heat_Pump"]

    hp_ids = edisgo.topology.loads_df.index[edisgo.topology.loads_df["type"] == "heat_pump"]
    hp_ts = edisgo.timeseries.loads_active_power.loc[
        :, hp_ids.intersection(edisgo.timeseries.loads_active_power.columns)
    ].copy()

    # --- grouping ---
    original_cols = [
        c for c in hp_ts.columns
        if any(str(c).startswith(pref) for pref in original_prefixes)
    ]
    duplicated_cols = [c for c in hp_ts.columns if str(c).startswith(duplicate_prefix)]
    other_cols = [c for c in hp_ts.columns if c not in original_cols and c not in duplicated_cols]

    print(f"Original HPs:   {len(original_cols)}")
    print(f"Other HPs:      {len(other_cols)}")
    print(f"Duplicated HPs: {len(duplicated_cols)}")

    # --- aggregate ---
    group_original = hp_ts[original_cols].sum(axis=1) if original_cols else None
    group_other = hp_ts[other_cols].sum(axis=1) if other_cols else None
    group_dup = hp_ts[duplicated_cols].sum(axis=1) if duplicated_cols else None

    dt = 1.0  # 60 min resolution

    if group_original is not None:
        E_original = group_original.sum() * dt
        print(f"Energy original HPs:   {E_original:.1f} MWh")

    if group_other is not None:
        E_other = group_other.sum() * dt
        print(f"Energy other HPs:      {E_other:.1f} MWh")

    if group_dup is not None:
        E_dup = group_dup.sum() * dt
        print(f"Energy duplicated HPs: {E_dup:.1f} MWh")

    def normalize_to_energy(series, energy_mwh, target_mwh=100.0):
        if series is None or energy_mwh == 0:
            return None
        return series * (target_mwh / energy_mwh)

    E_target = 100.0
    group_original_norm = normalize_to_energy(group_original, E_original, E_target) if group_original is not None else None
    group_other_norm = normalize_to_energy(group_other, E_other, E_target) if group_other is not None else None
    group_dup_norm = normalize_to_energy(group_dup, E_dup, E_target) if group_dup is not None else None

    # --- plot normalized ---
    plt.figure(figsize=(12, 5))

    if group_original_norm is not None:
        plt.plot(group_original_norm, label=f"Original HPs ({len(original_cols)})")
    if group_other_norm is not None:
        plt.plot(group_other_norm, label=f"Other HPs ({len(other_cols)})")
    if group_dup_norm is not None:
        plt.plot(group_dup_norm, label=f"Duplicated HPs ({len(duplicated_cols)})")

    plt.title(f"Heat pump timeseries (Energy-normalized){title_suffix}")
    plt.ylabel("Power [MW]")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- plot absolute ---
    plt.figure(figsize=(12, 5))

    if group_original is not None:
        plt.plot(group_original, label=f"Original HPs ({len(original_cols)})")
    if group_other is not None:
        plt.plot(group_other, label=f"Other HPs ({len(other_cols)})")
    if group_dup is not None:
        plt.plot(group_dup, label=f"Duplicated HPs ({len(duplicated_cols)})")

    plt.title(f"Heat pump timeseries (Aggregated absolute load){title_suffix}")
    plt.ylabel("Power [MW]")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "original_ids": original_cols,
        "other_ids": other_cols,
        "duplicated_ids": duplicated_cols,
    }

hp_export_dir = os.path.join(output_dir, "hp_profiles")

hp_export_info = export_hp_profiles_one_week(
    edisgo,
    out_dir=hp_export_dir,
    fname_prefix="hp_profiles_week_2023_01_01",
)

export_integrated_heat_pumps_shapefile(
    edisgo,
    output_dir=output_dir,
)

hp_groups = plot_heat_pump_profiles_comparison(
    edisgo,
    # adapt prefixes if your original HP IDs follow another naming scheme
    original_prefixes=["heat_load"],
    duplicate_prefix="hp_dup_",
    title_suffix="",
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
edisgo.heat_pump.heat_demand_df = (
    edisgo.timeseries.loads_active_power[hp_names] * cop
)
###

# set reactive power time series
edisgo.set_time_series_reactive_power_control()
############################ END MANUAL FIXES #################################


edisgo = run_optimization_14a(edisgo)
edisgo.analyze()

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
