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
    print(f"⚡ Running OPF with §14a Curtailment")
    print(f"{'='*80}")
    print(f"\nUsing OPF version 5:")
    print(f"  - §14a curtailment as only flexibility tool")
    print(f"  - Minimize line losses + §14a usage")
    print(f"  - Grid restrictions enforced (voltage 0.9-1.1, current limits)")
    print(f"  - Feasibility slacks penalized at 1e8")

    start_time = datetime.now()

    # Run optimization
    edisgo.pm_optimize(opf_version=5, curtailment_14a=True)

    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✓ Optimization complete!")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    return edisgo


grid_path = "/home/carlos/LoMa/exec_folder/results/MGB_quo_model_pypsa"

edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(0, 2))
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

######### This will be removed once Paul include the CP timeseries ############
# Remove remporarily the charging points
edisgo.topology.loads_df = edisgo.topology.loads_df[
    edisgo.topology.loads_df.type != "charging_point"
]
######### This will be removed once Paul include the CP timeseries ############

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
