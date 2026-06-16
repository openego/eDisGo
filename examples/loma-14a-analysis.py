"""
Analyse and plot §14a OPF results from loma-14a.py runs.

Reads per-seed files:
    <RESULTS_ROOT>/<seed>/curtailment_14a.csv
    <RESULTS_ROOT>/<seed>/line_usage

Reads per-month edisgo saves for the network map:
    <RESULTS_ROOT>/<seed>/<month>/edisgo/topology/
    <RESULTS_ROOT>/<seed>/<month>/edisgo/timeseries/generators_active_power.csv

Works for both test2 (7 days per month) and full-year runs, and for any
number of seeds (box plots summarise the distribution rather than one
element per seed).
"""
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd

try:
    import contextily as ctx
    _HAS_CTX = True
except ImportError:
    _HAS_CTX = False

# ── configuration ─────────────────────────────────────────────────────────────
RESULTS_ROOT = "/home/carlos/LoMa/output_edisgo"
PLOTS_DIR    = f"{RESULTS_ROOT}/presentation_plots"
CURT_THRESHOLD_MW = 1e-3   # solver noise floor
LINE_STRESS_PCT   = 90.0   # threshold for "stressed" line [%]
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_results(results_root):
    """Return {seed: {"curtailment": df, "line_usage": df}}."""
    data = {}
    for seed_dir in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(seed_dir):
            continue
        curt_path = os.path.join(seed_dir, "curtailment_14a.csv")
        lu_path   = os.path.join(seed_dir, "line_usage")
        if not (os.path.isfile(curt_path) and os.path.isfile(lu_path)):
            continue
        seed = os.path.basename(seed_dir)

        curt = pd.read_csv(curt_path, index_col=0, parse_dates=True)
        curt = curt.clip(lower=0)
        curt[curt < CURT_THRESHOLD_MW] = 0.0

        lu = pd.read_csv(lu_path, index_col=0, parse_dates=True)

        data[seed] = {"curtailment": curt, "line_usage": lu}
        print(f"  seed={seed}: {len(curt)} hours, {lu.shape[1]} lines")
    return data


def load_topology(results_root):
    """
    Load network topology from the first available saved edisgo month directory.
    All seeds share the same network topology.

    Returns buses, lines, loads, transformers DataFrames.
    """
    for seed_dir in sorted(glob.glob(os.path.join(results_root, "*"))):
        for edisgo_dir in sorted(glob.glob(os.path.join(seed_dir, "*/edisgo"))):
            topo = os.path.join(edisgo_dir, "topology")
            if not os.path.isdir(topo):
                continue
            buses = pd.read_csv(os.path.join(topo, "buses.csv"), index_col=0)
            lines = pd.read_csv(os.path.join(topo, "lines.csv"), index_col=0)
            loads = pd.read_csv(os.path.join(topo, "loads.csv"), index_col=0)
            trafos_path = os.path.join(topo, "transformers.csv")
            transformers = (pd.read_csv(trafos_path, index_col=0)
                            if os.path.isfile(trafos_path) else pd.DataFrame())
            print(f"  Topology from {edisgo_dir}")
            return buses, lines, loads, transformers
    raise FileNotFoundError("No saved edisgo topology found under " + results_root)


def load_per_bus_curtailment(results_root, loads_df):
    """
    Read generators_active_power.csv for every seed × month, extract
    per-load §14a curtailment, map to buses, and sum across seeds.

    Returns a DataFrame indexed by bus_name with columns:
        hp_mwh, cp_mwh  (sum over seeds and months)
    """
    seed_bus_curt = {}

    for seed_dir in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(seed_dir):
            continue
        seed = os.path.basename(seed_dir)
        bus_hp, bus_cp = {}, {}

        for edisgo_dir in sorted(glob.glob(os.path.join(seed_dir, "*/edisgo"))):
            gen_path = os.path.join(edisgo_dir, "timeseries", "generators_active_power.csv")
            if not os.path.isfile(gen_path):
                continue

            gen_ts = pd.read_csv(
                gen_path, index_col=0,
                usecols=lambda c: c == "snapshot" or "14a_support" in c,
            )
            if gen_ts.empty:
                continue

            gen_ts = gen_ts.clip(lower=0)
            gen_ts[gen_ts < CURT_THRESHOLD_MW] = 0.0

            for col in gen_ts.columns:
                total = gen_ts[col].sum()
                if total <= 0:
                    continue
                if "hp_14a_support" in col:
                    load_name = col.replace("hp_14a_support_", "")
                    store = bus_hp
                else:
                    load_name = (col.replace("cp_14a_support_", "")
                                    .replace("charging_point_14a_support_", ""))
                    store = bus_cp

                if load_name not in loads_df.index:
                    continue
                bus = loads_df.at[load_name, "bus"]
                store[bus] = store.get(bus, 0.0) + total

        if bus_hp or bus_cp:
            seed_bus_curt[seed] = {"hp": bus_hp, "cp": bus_cp}

    if not seed_bus_curt:
        return pd.DataFrame(columns=["hp_mwh", "cp_mwh"])

    all_buses = set(
        b for d in seed_bus_curt.values()
        for store in d.values()
        for b in store
    )
    rows = []
    for bus in sorted(all_buses):
        hp = np.sum([d["hp"].get(bus, 0.0) for d in seed_bus_curt.values()])
        cp = np.sum([d["cp"].get(bus, 0.0) for d in seed_bus_curt.values()])
        rows.append({"bus": bus, "hp_mwh": hp, "cp_mwh": cp})

    return pd.DataFrame(rows).set_index("bus")


# ══════════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ══════════════════════════════════════════════════════════════════════════════

def _daily_by_month(data, col):
    """
    Return a dict {month_int: np.array of daily values across all seeds}.

    col: "hp_curtailment_mw" | "cp_curtailment_mw" | "total_curtailment"
    """
    from collections import defaultdict
    month_vals = defaultdict(list)
    for d in data.values():
        curt = d["curtailment"]
        if col == "total_curtailment":
            series = curt["hp_curtailment_mw"] + curt["cp_curtailment_mw"]
        else:
            series = curt[col]
        daily = series.resample("D").sum()
        for m, grp in daily.groupby(daily.index.month):
            month_vals[m].extend(grp.values.tolist())
    return dict(month_vals)


def _line_hours_over_threshold(data, pct=LINE_STRESS_PCT):
    """
    Return a Series indexed by line name with the mean (across seeds) number
    of hours where loading exceeded `pct` %.
    """
    counts = {}
    for d in data.values():
        lu = d["line_usage"]
        over = (lu > pct).sum(axis=0)
        for line, n in over.items():
            counts.setdefault(line, []).append(n)
    return pd.Series({line: np.mean(v) for line, v in counts.items()})


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_monthly_curtailment_boxplot(data, plots_dir):
    """
    Box plots of daily §14a curtailment [MWh/day] per month.
    Distribution comes from all seeds combined — one box per month.
    Two panels: HP (top) and CP (bottom).
    """
    months = sorted(
        set(m for d in data.values() for m in d["curtailment"].index.month.unique())
    )
    mlabels = [pd.Timestamp(2035, m, 1).strftime("%b") for m in months]

    hp_by_month    = _daily_by_month(data, "hp_curtailment_mw")
    cp_by_month    = _daily_by_month(data, "cp_curtailment_mw")
    total_by_month = _daily_by_month(data, "total_curtailment")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, by_month, ylabel, color in [
        (axes[0], hp_by_month,    "HP curtailment\n[MWh/day]", "#d62728"),
        (axes[1], cp_by_month,    "CP curtailment\n[MWh/day]", "#1f77b4"),
        (axes[2], total_by_month, "Total curtailment\n[MWh/day]", "#7f7f7f"),
    ]:
        boxes = [by_month.get(m, [0]) for m in months]
        bp = ax.boxplot(boxes, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        n_seeds = len(data)
        ax.text(0.98, 0.97, f"n={n_seeds} seeds", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")

    axes[2].set_xticks(range(1, len(months) + 1))
    axes[2].set_xticklabels(mlabels)
    axes[2].set_xlabel("Month (2035)")
    fig.suptitle("Monthly §14a Curtailment — Distribution Across Seeds", fontsize=13)
    plt.tight_layout()
    _save(fig, plots_dir, "curtailment_monthly_boxplot.png")


def plot_monthly_curtailment_mean_stacked(data, plots_dir):
    """
    Stacked bar: mean HP + CP curtailment per month with std error bars.
    Shows the seasonal pattern and HP/CP split at a glance.
    """
    months = sorted(
        set(m for d in data.values() for m in d["curtailment"].index.month.unique())
    )
    mlabels = [pd.Timestamp(2035, m, 1).strftime("%b") for m in months]

    hp_means, hp_stds, cp_means, cp_stds = [], [], [], []
    for m in months:
        hp_vals = _daily_by_month(data, "hp_curtailment_mw").get(m, [0])
        cp_vals = _daily_by_month(data, "cp_curtailment_mw").get(m, [0])
        hp_means.append(np.mean(hp_vals))
        hp_stds.append(np.std(hp_vals))
        cp_means.append(np.mean(cp_vals))
        cp_stds.append(np.std(cp_vals))

    hp_means = np.array(hp_means)
    cp_means = np.array(cp_means)
    hp_stds  = np.array(hp_stds)

    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, hp_means, color="#d62728", alpha=0.85, label="HP (mean)")
    ax.bar(x, cp_means, bottom=hp_means, color="#1f77b4", alpha=0.85, label="CP (mean)")
    ax.errorbar(x, hp_means + cp_means,
                yerr=np.sqrt(hp_stds**2 + np.array(cp_stds)**2),
                fmt="none", color="black", capsize=3, linewidth=1, label="±1 std")

    ax.set_xticks(x)
    ax.set_xticklabels(mlabels)
    ax.set_ylabel("Mean daily §14a curtailment [MWh/day]")
    ax.set_xlabel("Month (2035)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Mean Daily §14a Curtailment per Month — HP vs CP", fontsize=13)
    plt.tight_layout()
    _save(fig, plots_dir, "curtailment_monthly_mean_stacked.png")


def plot_monthly_peak_line_loading_boxplot(data, plots_dir):
    """
    Box plots of the daily maximum line loading [%] per month.
    Distribution comes from all lines × all seeds.
    """
    months = sorted(
        set(m for d in data.values() for m in d["line_usage"].index.month.unique())
    )
    mlabels = [pd.Timestamp(2035, m, 1).strftime("%b") for m in months]

    from collections import defaultdict
    month_vals = defaultdict(list)
    for d in data.values():
        lu = d["line_usage"]
        daily_max = lu.max(axis=1).resample("D").max()
        for m, grp in daily_max.groupby(daily_max.index.month):
            month_vals[m].extend(grp.values.tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    boxes = [month_vals.get(m, [0]) for m in months]
    bp = ax.boxplot(boxes, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#2ca02c")
        patch.set_alpha(0.65)

    ax.axhline(100, color="red", linestyle="--", linewidth=1, label="100 % thermal limit")
    ax.axhline(LINE_STRESS_PCT, color="orange", linestyle=":", linewidth=1,
               label=f"{LINE_STRESS_PCT:.0f} % stress threshold")
    ax.set_xticks(range(1, len(months) + 1))
    ax.set_xticklabels(mlabels)
    ax.set_ylabel("Daily peak line loading [%]")
    ax.set_xlabel("Month (2035)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    n_seeds = len(data)
    ax.text(0.98, 0.97, f"n={n_seeds} seeds", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray")
    ax.set_title("Monthly Peak Line Loading — Distribution Across Seeds and Lines", fontsize=13)
    plt.tight_layout()
    _save(fig, plots_dir, "line_loading_monthly_boxplot.png")


def plot_line_loading_cdf(data, plots_dir):
    """
    CDF of line loading [%] across all lines and hours.
    Shows envelope (min/max) and mean across seeds.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    cdfs = []
    x_common = np.linspace(0, 120, 500)
    for d in data.values():
        vals = d["line_usage"].values.ravel()
        vals = np.sort(vals[np.isfinite(vals)])
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        cdfs.append(np.interp(x_common, vals, cdf, left=0, right=1))

    cdfs = np.array(cdfs)
    ax.fill_between(x_common, cdfs.min(axis=0), cdfs.max(axis=0),
                    alpha=0.25, color="#2ca02c", label="Min–max range across seeds")
    ax.plot(x_common, cdfs.mean(axis=0),
            color="#2ca02c", linewidth=2, label="Mean across seeds")

    ax.axvline(100, color="red", linestyle="--", linewidth=1, label="100 % thermal limit")
    ax.axvline(LINE_STRESS_PCT, color="orange", linestyle=":", linewidth=1,
               label=f"{LINE_STRESS_PCT:.0f} % stress threshold")
    ax.set_xlabel("Line Loading [%]")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 120)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"CDF of Line Loading — all lines, all hours ({len(data)} seeds)", fontsize=13)
    plt.tight_layout()
    _save(fig, plots_dir, "line_loading_cdf.png")


def plot_curtailment_vs_line_loading(data, plots_dir):
    """
    Scatter of hourly total §14a curtailment [MW] vs. peak line loading [%].
    All seeds pooled; colour encodes month to show seasonal pattern.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    cmap  = plt.get_cmap("tab10")
    month_done = set()

    for d in data.values():
        curt  = d["curtailment"]
        total = curt["hp_curtailment_mw"] + curt["cp_curtailment_mw"]
        peak  = d["line_usage"].max(axis=1).reindex(total.index)
        for m, grp in total.groupby(total.index.month):
            color = cmap(m - 1)
            label = pd.Timestamp(2035, m, 1).strftime("%b") if m not in month_done else "_"
            month_done.add(m)
            ax.scatter(grp.values, peak.loc[grp.index].values,
                       s=6, alpha=0.4, color=color, label=label)

    ax.axhline(100, color="red", linestyle="--", linewidth=1, label="100 % limit")
    ax.set_xlabel("Total §14a Curtailment [MW]")
    ax.set_ylabel("Peak Line Loading [%]")
    ax.legend(fontsize=8, ncol=3, markerscale=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Hourly Curtailment vs. Peak Line Loading (all seeds, colour = month)",
                 fontsize=12)
    plt.tight_layout()
    _save(fig, plots_dir, "curtailment_vs_line_loading.png")


def plot_network_map(data, buses, lines, loads, bus_curt, plots_dir):
    """
    Network map showing:
      - Lines coloured by mean hours with loading > LINE_STRESS_PCT %
      - Bus markers sized by total curtailment (HP red, CP blue pie-style)
    """
    hours_over = _line_hours_over_threshold(data, LINE_STRESS_PCT)

    # ── figure setup ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10))

    # ── draw lines ──────────────────────────────────────────────────────────
    max_hours = max(hours_over.max(), 1)
    norm  = mcolors.Normalize(vmin=0, vmax=max_hours)
    cmap_lines = cm.get_cmap("YlOrRd")

    for line_name, row in lines.iterrows():
        b0 = row.get("bus0") or row.get("bus_0")
        b1 = row.get("bus1") or row.get("bus_1")
        if b0 not in buses.index or b1 not in buses.index:
            continue
        x0, y0 = buses.at[b0, "x"], buses.at[b0, "y"]
        x1, y1 = buses.at[b1, "x"], buses.at[b1, "y"]
        h = hours_over.get(line_name, 0)
        if h > 0:
            color = cmap_lines(norm(h))
            lw    = 1.0 + 2.5 * (h / max_hours)
            zorder = 3
        else:
            color  = "#cccccc"
            lw     = 0.7
            zorder = 2
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=zorder, solid_capstyle="round")

    # ── draw bus curtailment markers ─────────────────────────────────────────
    if not bus_curt.empty:
        bus_xy = buses[["x", "y"]].reindex(bus_curt.index).dropna()
        curt_align = bus_curt.reindex(bus_xy.index).fillna(0)
        total_curt = curt_align["hp_mwh"] + curt_align["cp_mwh"]
        max_total  = total_curt.max() or 1.0

        x_range = buses["x"].dropna()
        grid_extent = x_range.max() - x_range.min()
        MIN_R = grid_extent * 0.0015
        MAX_R = grid_extent * 0.010

        import matplotlib.patches as mpatches
        for bus in bus_xy.index:
            if total_curt[bus] <= 0:
                continue
            bx, by = bus_xy.at[bus, "x"], bus_xy.at[bus, "y"]
            r = MIN_R + (MAX_R - MIN_R) * np.sqrt(total_curt[bus] / max_total)
            hp = curt_align.at[bus, "hp_mwh"]
            cp = curt_align.at[bus, "cp_mwh"]
            total = hp + cp
            start = 90.0
            for val, color in [(hp, "#d62728"), (cp, "#1f77b4")]:
                if val <= 0:
                    continue
                angle = 360.0 * val / total
                ax.add_patch(mpatches.Wedge(
                    (bx, by), r, start, start + angle,
                    facecolor=color, edgecolor="white", linewidth=0.3,
                    alpha=0.9, zorder=5,
                ))
                start += angle

    # ── basemap ──────────────────────────────────────────────────────────────
    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    # ── colourbar for line stress ─────────────────────────────────────────────
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.018, pos.height])
    sm  = cm.ScalarMappable(cmap=cmap_lines, norm=norm)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label(f"Mean hours loading > {LINE_STRESS_PCT:.0f} %", fontsize=9)

    # ── legend ────────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(facecolor="#d62728", alpha=0.9, label="HP curtailment"),
        mpatches.Patch(facecolor="#1f77b4", alpha=0.9, label="CP curtailment"),
        plt.Line2D([0], [0], color="#cccccc", linewidth=1.5, label="Line (no stress)"),
        plt.Line2D([0], [0], color=cmap_lines(0.99), linewidth=3, label=f"Line (>{LINE_STRESS_PCT:.0f} %, max hours)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title(
        f"Network Map — Line Stress (hours >{LINE_STRESS_PCT:.0f} %) and §14a Curtailment\n"
        f"(mean across {len(data)} seeds)",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, plots_dir, "network_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# NetworkX — curtailment reach along the feeder
# ══════════════════════════════════════════════════════════════════════════════

def build_feeder_graph(lines_df):
    """
    Build an undirected NetworkX graph from the line topology.
    Each edge carries its line name as the 'line' attribute.
    """
    G = nx.Graph()
    for line_name, row in lines_df.iterrows():
        G.add_edge(row["bus0"], row["bus1"], line=line_name)
    return G


def find_root_bus(lines_df, transformers_df):
    """
    Return the LV substation bus (feeder root) — the secondary (bus1) of the
    LV transformer that is connected to at least one line in the LV network.
    Falls back to the highest-degree node if no transformer info is available.
    """
    line_buses = set(lines_df["bus0"]) | set(lines_df["bus1"])
    if not transformers_df.empty and "bus1" in transformers_df.columns:
        for bus in transformers_df["bus1"]:
            if bus in line_buses:
                return bus
    G = nx.Graph()
    for _, row in lines_df.iterrows():
        G.add_edge(row["bus0"], row["bus1"])
    return max(dict(G.degree()), key=lambda b: G.degree(b))


def compute_bus_upstream_lines(G, root_bus):
    """
    For every bus reachable from root_bus, return the ordered list of line
    names on the shortest path from that bus back to root_bus.

    These are the lines that 'carry' that bus's load toward the feeder —
    i.e., the lines affected when curtailment occurs at that bus.
    """
    bus_to_lines = {root_bus: []}
    for bus in nx.bfs_tree(G, root_bus).nodes():
        if bus == root_bus:
            continue
        try:
            path_nodes = nx.shortest_path(G, bus, root_bus)
        except nx.NetworkXNoPath:
            continue
        bus_to_lines[bus] = [
            G[path_nodes[i]][path_nodes[i + 1]]["line"]
            for i in range(len(path_nodes) - 1)
        ]
    return bus_to_lines


def compute_line_curtailment_reach(results_root, loads_df, bus_upstream_lines):
    """
    For each seed: load per-load §14a timeseries, map loads to buses, then
    propagate curtailment events upstream through the feeder graph.

    For each line, a timestep is 'affected' when at least one bus in its
    downstream subtree had active §14a curtailment that hour.

    Returns a pd.Series (index = line name, values = mean affected hours
    across seeds).
    """
    def _extract_load_name(col):
        return (col.replace("hp_14a_support_", "")
                   .replace("cp_14a_support_", "")
                   .replace("charging_point_14a_support_", ""))

    seed_line_counts = {}

    for seed_dir in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(seed_dir):
            continue
        seed = os.path.basename(seed_dir)
        month_dfs = []

        for edisgo_dir in sorted(glob.glob(os.path.join(seed_dir, "*/edisgo"))):
            gen_path = os.path.join(edisgo_dir, "timeseries", "generators_active_power.csv")
            if not os.path.isfile(gen_path):
                continue
            gen_ts = pd.read_csv(
                gen_path, index_col=0, parse_dates=True,
                usecols=lambda c: c == "snapshot" or "14a_support" in c,
            )
            month_dfs.append(gen_ts)

        if not month_dfs:
            continue

        all_gen = pd.concat(month_dfs, axis=0).clip(lower=0)
        all_gen[all_gen < CURT_THRESHOLD_MW] = 0.0

        # Build bus → boolean Series (True = curtailment active that hour)
        bus_active = {}
        for col in all_gen.columns:
            load_name = _extract_load_name(col)
            if load_name not in loads_df.index:
                continue
            bus = loads_df.at[load_name, "bus"]
            col_active = all_gen[col] > 0
            if bus in bus_active:
                bus_active[bus] = bus_active[bus] | col_active
            else:
                bus_active[bus] = col_active

        if not bus_active:
            continue

        bus_active_df = pd.DataFrame(bus_active)  # (timesteps × curtailed buses)

        # Propagate upstream: for each bus, OR its activity into every upstream line
        line_affected = {}
        for bus, upstream in bus_upstream_lines.items():
            if bus not in bus_active_df.columns or not upstream:
                continue
            for line in upstream:
                if line not in line_affected:
                    line_affected[line] = bus_active_df[bus].copy()
                else:
                    line_affected[line] = line_affected[line] | bus_active_df[bus]

        seed_line_counts[seed] = {
            line: int(series.sum()) for line, series in line_affected.items()
        }
        print(f"    seed={seed}: {len(seed_line_counts[seed])} lines reached")

    if not seed_line_counts:
        return pd.Series(dtype=float)

    all_lines = set(l for d in seed_line_counts.values() for l in d)
    return pd.Series(
        {line: np.sum([d.get(line, 0) for d in seed_line_counts.values()])
         for line in all_lines},
        name="total_affected_hours",
    )


def plot_curtailment_reach_map(buses, lines, line_reach_hours, bus_curt, n_seeds, plots_dir):
    """
    Network map combining two layers:

    Lines — coloured by mean hours per simulated period in which §14a
    curtailment occurred somewhere downstream.  All lines are visible;
    unaffected ones are drawn in a muted grey.

    Buses — every bus is shown as a small dot.  Buses with §14a activity
    get a larger pie marker (red = HP, blue = CP) whose area scales with
    the total curtailment energy (sum of HP + CP MWh, mean across seeds).
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(13, 10))

    # ── lines ─────────────────────────────────────────────────────────────────
    affected = line_reach_hours[line_reach_hours > 0]
    h_min      = affected.min() if not affected.empty else 0
    h_max      = affected.max() if not affected.empty else 1
    norm_lines = mcolors.Normalize(vmin=h_min, vmax=h_max)
    cmap_lines = cm.get_cmap("jet")

    for line_name, row in lines.iterrows():
        b0, b1 = row["bus0"], row["bus1"]
        if b0 not in buses.index or b1 not in buses.index:
            continue
        if pd.isna(buses.at[b0, "x"]) or pd.isna(buses.at[b1, "x"]):
            continue
        x0, y0 = buses.at[b0, "x"], buses.at[b0, "y"]
        x1, y1 = buses.at[b1, "x"], buses.at[b1, "y"]
        h = line_reach_hours.get(line_name, 0)
        if h > 0:
            color  = cmap_lines(norm_lines(h))
            lw     = 0.9 + 2.6 * (h / h_max)
            zorder = 3
        else:
            color  = "black"
            lw     = 0.8
            zorder = 2
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                zorder=zorder, solid_capstyle="round")

    # ── buses ─────────────────────────────────────────────────────────────────
    bus_xy = buses[["x", "y"]].dropna()

    x_extent = bus_xy["x"].max() - bus_xy["x"].min()
    MIN_R = x_extent * 0.0012
    MAX_R = x_extent * 0.009

    # visible dot for every bus (no §14a)
    ax.scatter(bus_xy["x"], bus_xy["y"], s=18, color="#888888",
               zorder=4, linewidths=0.4, edgecolors="white")

    # pie marker for buses with curtailment
    if not bus_curt.empty:
        curt_align = bus_curt.reindex(bus_xy.index).fillna(0)
        total_curt = curt_align["hp_mwh"] + curt_align["cp_mwh"]
        max_total  = total_curt.max() or 1.0

        for bus in bus_xy.index:
            tot = total_curt.get(bus, 0.0)
            if tot <= 0:
                continue
            bx, by = bus_xy.at[bus, "x"], bus_xy.at[bus, "y"]
            r      = MIN_R + (MAX_R - MIN_R) * np.sqrt(tot / max_total)
            hp     = curt_align.at[bus, "hp_mwh"]
            cp     = curt_align.at[bus, "cp_mwh"]
            start  = 90.0
            for val, color in [(hp, "#d62728"), (cp, "#1f77b4")]:
                if val <= 0:
                    continue
                angle = 360.0 * val / tot
                ax.add_patch(mpatches.Wedge(
                    (bx, by), r, start, start + angle,
                    facecolor=color, edgecolor="white", linewidth=0.3,
                    alpha=0.9, zorder=5,
                ))
                start += angle

    # ── basemap ───────────────────────────────────────────────────────────────
    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    ax.set_axis_off()

    # ── colourbar for line reach ───────────────────────────────────────────────
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.018, pos.height])
    sm  = cm.ScalarMappable(cmap=cmap_lines, norm=norm_lines)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Total hours affected by downstream §14a curtailment\n(sum across seeds)", fontsize=9)

    # ── type legend (upper left) ──────────────────────────────────────────────
    n_affected = int((line_reach_hours > 0).sum())
    type_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.2,
                   label="Line — not reached by §14a"),
        plt.Line2D([0], [0], color=cmap_lines(0.99), linewidth=3.5,
                   label="Line — most affected hours"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=7, label="Bus (no §14a)"),
        mpatches.Patch(facecolor="#d62728", alpha=0.9, label="Bus — HP curtailment"),
        mpatches.Patch(facecolor="#1f77b4", alpha=0.9, label="Bus — CP curtailment"),
    ]
    leg1 = ax.legend(handles=type_handles, loc="upper left", fontsize=9)
    ax.add_artist(leg1)

    # ── size reference legend (lower left) ───────────────────────────────────
    if not bus_curt.empty:
        curt_align = bus_curt.reindex(bus_xy.index).fillna(0)
        total_curt = curt_align["hp_mwh"] + curt_align["cp_mwh"]
        max_total  = total_curt.max() or 1.0

        # pick three round reference values spanning the data range
        ref_vals = []
        for frac in [0.1, 0.5, 1.0]:
            ref_mwh = frac * max_total
            # round to 1 significant figure for readability
            magnitude = 10 ** np.floor(np.log10(max(ref_mwh, 1e-9)))
            ref_vals.append(round(ref_mwh / magnitude) * magnitude)
        ref_vals = sorted(set(ref_vals))

        size_handles = []
        for ref_mwh in ref_vals:
            r_deg   = MIN_R + (MAX_R - MIN_R) * np.sqrt(ref_mwh / max_total)
            # convert radius in degrees to approximate points for the legend marker
            fig.canvas.draw()
            deg_per_pt = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_position().width * fig.get_size_inches()[0] * 72)
            ms = max(3, 2 * r_deg / deg_per_pt)
            size_handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#9467bd", markersize=ms,
                           label=f"{ref_mwh:.1f} MWh")
            )
        ax.legend(handles=size_handles, loc="lower left", fontsize=9,
                  title="Bus size = total §14a [MWh]", title_fontsize=8)

    ax.set_title(
        f"§14a Curtailment Reach Along the Feeder\n"
        f"{n_affected} of {len(lines)} lines reached  —  "
        f"bus size ∝ total §14a use  —  sum across {n_seeds} seeds",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, plots_dir, "network_curtailment_reach.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def save_curtailment_summary(data, plots_dir):
    rows = []
    for seed, d in data.items():
        curt = d["curtailment"]
        hp   = curt["hp_curtailment_mw"]
        cp   = curt["cp_curtailment_mw"]
        rows.append({
            "seed": seed,
            "HP curtailment [MWh]":   round(hp.sum(), 4),
            "CP curtailment [MWh]":   round(cp.sum(), 4),
            "Total curtailment [MWh]": round((hp + cp).sum(), 4),
            "Hours HP curtailed":     int((hp > 0).sum()),
            "Hours CP curtailed":     int((cp > 0).sum()),
            "Max hourly HP [MW]":     round(hp.max(), 4),
            "Max hourly CP [MW]":     round(cp.max(), 4),
        })
    df = pd.DataFrame(rows).set_index("seed")
    print("\n=== §14a Curtailment Summary ===")
    print(df.to_string())
    path = os.path.join(plots_dir, "curtailment_summary.csv")
    df.to_csv(path)
    print(f"  Saved: {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, plots_dir, name):
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading results from {RESULTS_ROOT} …")
    data = load_results(RESULTS_ROOT)
    if not data:
        raise FileNotFoundError(
            f"No seed directories with curtailment_14a.csv + line_usage in {RESULTS_ROOT}"
        )
    print(f"\nSeeds found: {list(data.keys())}")

    save_curtailment_summary(data, PLOTS_DIR)

    print("\nGenerating statistical plots …")
    plot_monthly_curtailment_boxplot(data, PLOTS_DIR)
    plot_monthly_curtailment_mean_stacked(data, PLOTS_DIR)
    plot_monthly_peak_line_loading_boxplot(data, PLOTS_DIR)
    plot_line_loading_cdf(data, PLOTS_DIR)
    plot_curtailment_vs_line_loading(data, PLOTS_DIR)

    print("\nLoading topology for network maps …")
    buses, lines, loads, transformers = load_topology(RESULTS_ROOT)
    bus_curt = load_per_bus_curtailment(RESULTS_ROOT, loads)
    print(f"  Per-bus curtailment: {len(bus_curt)} buses with §14a activity")
    plot_network_map(data, buses, lines, loads, bus_curt, PLOTS_DIR)

    print("\nComputing §14a curtailment reach along feeders …")
    G         = build_feeder_graph(lines)
    root_bus  = find_root_bus(lines, transformers)
    print(f"  Feeder root: {root_bus}")
    bus_upstream = compute_bus_upstream_lines(G, root_bus)
    print(f"  Buses in feeder tree: {len(bus_upstream)}")
    line_reach = compute_line_curtailment_reach(RESULTS_ROOT, loads, bus_upstream)
    print(f"  Lines reached by §14a: {(line_reach > 0).sum()} / {len(lines)}")
    plot_curtailment_reach_map(buses, lines, line_reach, bus_curt, len(data), PLOTS_DIR)

    print(f"\nDone. All plots saved to {PLOTS_DIR}")
