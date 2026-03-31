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
    print(f"📊 Analyzing §14a Curtailment Results")
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
    print(f"\n📈 Summary Statistics:")
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
        print(f"\n  Top 5 curtailed heat pumps:")
        for i, (hp, value) in enumerate(
            curtailed_hps.sort_values(ascending=False).head().items(), 1
        ):
            hp_name = hp.replace("hp_14a_support_", "")
            print(f"    {i}. {hp_name}: {value:.4f} MWh")

    if len(curtailed_cps) > 0:
        print(f"\n  Top 5 curtailed charging points:")
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
    print(f"⚡ Running OPF with §14a Curtailment")
    print(f"{'='*80}")
    print(f"\nUsing OPF version 5:")
    print(f"  - §14a curtailment as only flexibility tool")
    print(f"  - Minimize line losses + §14a usage")
    print(f"  - Grid restrictions enforced (voltage 0.9-1.1, current limits)")
    print(f"  - Feasibility slacks penalized at 1e8")

    start_time = datetime.now()

    # Run optimization
    edisgo.pm_optimize(opf_version=5, curtailment_14a=True, flexible_hps=[])

    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✓ Optimization complete!")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    return edisgo


grid_path = "/home/carlos/LoMa/exec_folder/results/MGB_model_pypsa"

edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(0, 3))
mv_grid_geom = gpd.read_file(
    "/home/carlos/LoMa/exec_folder/data/Input_files/MV_grid_district/husum_district.shp"
)
mv_grid_geom = mv_grid_geom.to_crs(4326)
edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
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

# Remove remporarily the charging points
edisgo.topology.loads_df = edisgo.topology.loads_df[
    edisgo.topology.loads_df.type != "charging_point"
]

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
