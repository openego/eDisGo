import os
import glob

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as mpl_ticker
from matplotlib.legend_handler import HandlerBase
import networkx as nx
import numpy as np
import pandas as pd

try:
    import contextily as ctx
    _HAS_CTX = True
except ImportError:
    _HAS_CTX = False

# ── configuration ─────────────────────────────────────────────────────────────
RESULTS_ROOT = "/home/carlos/LoMa/output_edisgo/45"
PLOTS_DIR    = f"{RESULTS_ROOT}/presentation_plots"
CURT_THRESHOLD_MW = 1e-3   # solver noise floor
LINE_STRESS_PCT   = 90.0   # threshold for "stressed" line [%]
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(PLOTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_results(results_root):
    """Return {"curtailment": df, "line_usage": df}."""
    curt_path = os.path.join(results_root, "curtailment_14a.csv")
    lu_path   = os.path.join(results_root, "line_usage")
    if not (os.path.isfile(curt_path) and os.path.isfile(lu_path)):
        raise FileNotFoundError(
            f"curtailment_14a.csv or line_usage not found in {results_root}"
        )

    curt = pd.read_csv(curt_path, index_col=0, parse_dates=True)
    curt = curt.clip(lower=0)
    curt[curt < CURT_THRESHOLD_MW] = 0.0

    lu = pd.read_csv(lu_path, index_col=0, parse_dates=True)

    print(f"  {len(curt)} hours, {lu.shape[1]} lines")
    return {"curtailment": curt, "line_usage": lu}


def load_topology(results_root):
    """
    Load network topology from the first available edisgo month directory.

    Returns buses, lines, loads, transformers DataFrames.
    """
    for edisgo_dir in sorted(glob.glob(os.path.join(results_root, "*/edisgo"))):
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
    Read generators_active_power.csv for every month, extract per-load §14a
    curtailment, map to buses, and sum across months.

    Returns a DataFrame indexed by bus_name with columns:
        hp_mwh, cp_mwh
    """
    bus_hp, bus_cp = {}, {}

    for edisgo_dir in sorted(glob.glob(os.path.join(results_root, "*/edisgo"))):
        gen_path = os.path.join(edisgo_dir, "timeseries", "generators_active_power.csv")
        if not os.path.isfile(gen_path):
            continue

        loads_path = os.path.join(edisgo_dir, "topology", "loads.csv")
        month_loads = (
            pd.read_csv(loads_path, index_col=0)
            if os.path.isfile(loads_path)
            else loads_df
        )

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

            if load_name not in month_loads.index:
                continue
            bus = month_loads.at[load_name, "bus"]
            store[bus] = store.get(bus, 0.0) + total

    if not bus_hp and not bus_cp:
        return pd.DataFrame(columns=["hp_mwh", "cp_mwh"])

    all_buses = set(bus_hp) | set(bus_cp)
    rows = [
        {"bus": bus, "hp_mwh": bus_hp.get(bus, 0.0), "cp_mwh": bus_cp.get(bus, 0.0)}
        for bus in sorted(all_buses)
    ]
    return pd.DataFrame(rows).set_index("bus")


# ══════════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ══════════════════════════════════════════════════════════════════════════════

def _daily_by_month(data, col):
    """
    Return a dict {month_int: np.array of daily values}.

    col: "hp_curtailment_mw" | "cp_curtailment_mw" | "total_curtailment"
    """
    from collections import defaultdict
    month_vals = defaultdict(list)
    curt = data["curtailment"]
    if col == "total_curtailment":
        series = curt["hp_curtailment_mw"] + curt["cp_curtailment_mw"]
    else:
        series = curt[col]
    daily = series.resample("D").sum()
    for m, grp in daily.groupby(daily.index.month):
        month_vals[m].extend(grp.values.tolist())
    return dict(month_vals)


def _monthly_by_month(data, col):
    """
    Return a dict {month_int: monthly total}.

    col: "hp_curtailment_mw" | "cp_curtailment_mw" | "total_curtailment"
    """
    month_vals = {}
    curt = data["curtailment"]
    if col == "total_curtailment":
        series = curt["hp_curtailment_mw"] + curt["cp_curtailment_mw"]
    else:
        series = curt[col]
    for m, grp in series.groupby(series.index.month):
        month_vals[m] = grp.sum()
    return month_vals


def _line_hours_over_threshold(data, pct=LINE_STRESS_PCT):
    """
    Return a Series indexed by line name with the number of hours where
    loading exceeded `pct` %.
    """
    lu = data["line_usage"]
    return (lu > pct).sum(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════


def plot_monthly_curtailment_mean_stacked(data, plots_dir):
    """
    Stacked bar: HP + CP curtailment per month.
    Shows the seasonal pattern and HP/CP split at a glance.
    """
    months = sorted(data["curtailment"].index.month.unique())
    DE_MONTHS = {1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez"}
    mlabels = [DE_MONTHS[m] for m in months]

    hp_monthly = _monthly_by_month(data, "hp_curtailment_mw")
    cp_monthly = _monthly_by_month(data, "cp_curtailment_mw")
    hp_vals = np.array([hp_monthly.get(m, 0.0) for m in months])
    cp_vals = np.array([cp_monthly.get(m, 0.0) for m in months])

    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, hp_vals, color="#d62728", alpha=0.85, label="WP")
    ax.bar(x, cp_vals, bottom=hp_vals, color="#1f77b4", alpha=0.85, label="LP")

    ax.set_xticks(x)
    ax.set_xticklabels(mlabels)
    ax.set_ylabel("Monatliche §14a-Abregelung [MWh/Monat]")
    ax.set_xlabel("Monat (2035)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, plots_dir, "mgb_14a_curtailment_monthly.png")


def plot_network_map(data, buses, lines, loads, bus_curt, root_bus, plots_dir):
    """
    Network map showing:
      - Lines coloured by hours with loading > LINE_STRESS_PCT %.
        Unaffected lines are drawn in black; jet colormap with adaptive range.
      - All buses visible as gray dots; buses with §14a get pie markers (HP red, CP blue).
      - IEC two-circle transformer symbol at the feeder root.
    """
    import matplotlib.patches as mpatches

    hours_over = _line_hours_over_threshold(data, LINE_STRESS_PCT)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(right=0.84)

    # ── lines ─────────────────────────────────────────────────────────────────
    affected   = hours_over[hours_over > 0]
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
        h = hours_over.get(line_name, 0)
        if h > 0:
            color  = cmap_lines(norm_lines(h))
            lw     = 2.0 + 0.5 * np.sqrt(h / h_max)
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

    ax.scatter(bus_xy["x"], bus_xy["y"], s=18, color="#888888",
               zorder=4, linewidths=0.4, edgecolors="white", alpha=0.35)

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

    # ── transformer marker (IEC two-circle symbol) ────────────────────────────
    if root_bus in buses.index:
        tx = buses.at[root_bus, "x"]
        ty = buses.at[root_bus, "y"]
        if pd.notna(tx) and pd.notna(ty):
            r_t = x_extent * 0.006
            for cx in (tx - r_t * 0.75, tx + r_t * 0.75):
                ax.add_patch(mpatches.Circle(
                    (cx, ty), r_t,
                    fill=False, edgecolor="black", linewidth=2.0, zorder=7,
                ))

    # ── basemap ───────────────────────────────────────────────────────────────
    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    ax.set_axis_off()

    # ── colourbar for line stress ─────────────────────────────────────────────
    cax = fig.add_axes([0.86, 0.12, 0.018, 0.76])
    sm  = cm.ScalarMappable(cmap=cmap_lines, norm=norm_lines)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label(f"Stunden Belastung > {LINE_STRESS_PCT:.0f} % im Jahr (8760 h)", fontsize=9)
    cb.locator = mpl_ticker.MaxNLocator(integer=True)
    cb.update_ticks()

    # ── type legend (upper left) ──────────────────────────────────────────────
    type_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.2,
                   label=f"Leitung — nie über {LINE_STRESS_PCT:.0f} %"),
        plt.Line2D([0], [0], color=cmap_lines(0.99), linewidth=3.5,
                   label="Leitung — stärkste Belastungsstunden"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=7, label="Knoten (kein §14a)"),
        mpatches.Patch(facecolor="#d62728", alpha=0.9, label="Knoten — WP §14a"),
        mpatches.Patch(facecolor="#1f77b4", alpha=0.9, label="Knoten — LP §14a"),
        plt.Line2D([], [], label="MS/NS-Transformator (Stationsabgang)"),
    ]
    trafo_handle = type_handles[-1]
    leg1 = ax.legend(handles=type_handles, loc="upper left", fontsize=9,
                     handler_map={trafo_handle: _TwoCircleHandler()})
    ax.add_artist(leg1)

    # ── size reference legend (lower left) ───────────────────────────────────
    if not bus_curt.empty:
        curt_align = bus_curt.reindex(bus_xy.index).fillna(0)
        total_curt = curt_align["hp_mwh"] + curt_align["cp_mwh"]
        max_total  = total_curt.max() or 1.0

        ref_fracs = [1.0]
        ref_mwhs  = [f * max_total for f in ref_fracs]

        fig.canvas.draw()
        xlim = ax.get_xlim()
        deg_per_pt = (xlim[1] - xlim[0]) / (
            ax.get_position().width * fig.get_size_inches()[0] * 72
        )

        size_handles = []
        for ref_mwh in ref_mwhs:
            r_deg = MIN_R + (MAX_R - MIN_R) * np.sqrt(ref_mwh / max_total)
            ms    = max(4, 2 * r_deg / deg_per_pt)
            size_handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#9467bd", markersize=ms,
                           label=f"{ref_mwh:.3g} MWh")
            )
        ax.legend(handles=size_handles, loc="lower left", fontsize=9,
                  title="Knotengröße = §14a gesamt [MWh]", title_fontsize=8)

    _save(fig, plots_dir, "mgb_14a_loading_hours_map.png")


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
    Load per-load §14a timeseries for every month, map loads to buses, then
    propagate curtailment events upstream through the feeder graph.

    For each line, a timestep is 'affected' when at least one bus in its
    downstream subtree had active §14a curtailment that hour.

    Returns a pd.Series (index = line name, values = affected hours).
    """
    def _extract_load_name(col):
        return (col.replace("hp_14a_support_", "")
                   .replace("cp_14a_support_", "")
                   .replace("charging_point_14a_support_", ""))

    month_dfs = []
    for edisgo_dir in sorted(glob.glob(os.path.join(results_root, "*/edisgo"))):
        gen_path = os.path.join(edisgo_dir, "timeseries", "generators_active_power.csv")
        if not os.path.isfile(gen_path):
            continue
        loads_path = os.path.join(edisgo_dir, "topology", "loads.csv")
        month_loads = (
            pd.read_csv(loads_path, index_col=0)
            if os.path.isfile(loads_path)
            else loads_df
        )
        gen_ts = pd.read_csv(
            gen_path, index_col=0, parse_dates=True,
            usecols=lambda c: c == "snapshot" or "14a_support" in c,
        )
        month_dfs.append((gen_ts, month_loads))

    if not month_dfs:
        return pd.Series(dtype=float)

    bus_active = {}
    for gen_ts, month_loads in month_dfs:
        gen_ts = gen_ts.clip(lower=0)
        gen_ts[gen_ts < CURT_THRESHOLD_MW] = 0.0
        for col in gen_ts.columns:
            load_name = _extract_load_name(col)
            if load_name not in month_loads.index:
                continue
            bus = month_loads.at[load_name, "bus"]
            col_active = gen_ts[col] > 0
            if bus in bus_active:
                bus_active[bus] = bus_active[bus] | col_active
            else:
                bus_active[bus] = col_active.copy()

    if not bus_active:
        return pd.Series(dtype=float)

    bus_active_df = pd.DataFrame(bus_active)

    line_affected = {}
    for bus, upstream in bus_upstream_lines.items():
        if bus not in bus_active_df.columns or not upstream:
            continue
        for line in upstream:
            if line not in line_affected:
                line_affected[line] = bus_active_df[bus].copy()
            else:
                line_affected[line] = line_affected[line] | bus_active_df[bus]

    print(f"    {len(line_affected)} lines reached")
    return pd.Series(
        {line: int(series.sum()) for line, series in line_affected.items()},
        name="affected_hours",
    )


class _TwoCircleHandler(HandlerBase):
    """Legend handler that draws two small overlapping circles (IEC transformer symbol)."""
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        from matplotlib.patches import Circle
        r  = height * 0.45
        cy = height * 0.5
        c1 = Circle((width * 0.5 - r * 0.75, cy), r, transform=trans,
                    fill=False, edgecolor="black", linewidth=1.2)
        c2 = Circle((width * 0.5 + r * 0.75, cy), r, transform=trans,
                    fill=False, edgecolor="black", linewidth=1.2)
        return [c1, c2]


def plot_curtailment_reach_map(buses, lines, line_reach_hours, bus_curt, root_bus, plots_dir):
    """
    Network map combining two layers:

    Lines — coloured by hours in which §14a curtailment occurred somewhere
    downstream.  All lines are visible; unaffected ones are drawn in black.

    Buses — every bus is shown as a small dot.  Buses with §14a activity
    get a larger pie marker (red = HP, blue = CP) whose area scales with
    the total curtailment energy (HP + CP MWh).
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(right=0.84)

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

    ax.scatter(bus_xy["x"], bus_xy["y"], s=18, color="#888888",
               zorder=4, linewidths=0.4, edgecolors="white", alpha=0.35)

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

    # ── transformer marker (IEC two-circle symbol) ────────────────────────────
    if root_bus in buses.index:
        tx = buses.at[root_bus, "x"]
        ty = buses.at[root_bus, "y"]
        if pd.notna(tx) and pd.notna(ty):
            r_t = x_extent * 0.006
            for cx in (tx - r_t * 0.75, tx + r_t * 0.75):
                ax.add_patch(mpatches.Circle(
                    (cx, ty), r_t,
                    fill=False, edgecolor="black", linewidth=2.0, zorder=7,
                ))

    # ── basemap ───────────────────────────────────────────────────────────────
    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    ax.set_axis_off()

    # ── colourbar for line reach ───────────────────────────────────────────────
    cax = fig.add_axes([0.86, 0.12, 0.018, 0.76])
    sm  = cm.ScalarMappable(cmap=cmap_lines, norm=norm_lines)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Stunden mit §14a-Abregelung im nachgelagerten Netz", fontsize=9)

    # ── type legend (upper left) ──────────────────────────────────────────────
    type_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.2,
                   label="Leitung — kein §14a-Einfluss"),
        plt.Line2D([0], [0], color=cmap_lines(0.99), linewidth=3.5,
                   label="Leitung — meiste betroffene Stunden"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=7, label="Knoten (kein §14a)"),
        mpatches.Patch(facecolor="#d62728", alpha=0.9, label="Knoten — WP §14a"),
        mpatches.Patch(facecolor="#1f77b4", alpha=0.9, label="Knoten — LP §14a"),
        plt.Line2D([], [], label="MS/NS-Transformator"),
    ]
    trafo_handle = type_handles[-1]
    leg1 = ax.legend(handles=type_handles, loc="upper left", fontsize=9,
                     handler_map={trafo_handle: _TwoCircleHandler()})
    ax.add_artist(leg1)

    # ── size reference legend (lower left) ───────────────────────────────────
    if not bus_curt.empty:
        curt_align = bus_curt.reindex(bus_xy.index).fillna(0)
        total_curt = curt_align["hp_mwh"] + curt_align["cp_mwh"]
        max_total  = total_curt.max() or 1.0

        ref_fracs = [1.0]
        ref_mwhs  = [f * max_total for f in ref_fracs]

        fig.canvas.draw()
        xlim = ax.get_xlim()
        deg_per_pt = (xlim[1] - xlim[0]) / (
            ax.get_position().width * fig.get_size_inches()[0] * 72
        )

        size_handles = []
        for ref_mwh in ref_mwhs:
            r_deg = MIN_R + (MAX_R - MIN_R) * np.sqrt(ref_mwh / max_total)
            ms    = max(4, 2 * r_deg / deg_per_pt)
            size_handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#9467bd", markersize=ms,
                           label=f"{ref_mwh:.3g} MWh")
            )
        ax.legend(handles=size_handles, loc="lower left", fontsize=9,
                  title="Knotengröße = §14a gesamt [MWh]", title_fontsize=8)

    _save(fig, plots_dir, "network_curtailment_reach.png")


def load_solar_generators(results_root):
    """Load solar_rooftop generators from the first available month directory."""
    for edisgo_dir in sorted(glob.glob(os.path.join(results_root, "*/edisgo"))):
        gen_path = os.path.join(edisgo_dir, "topology", "generators.csv")
        if not os.path.isfile(gen_path):
            continue
        gen = pd.read_csv(gen_path, index_col=0)
        return gen[gen["carrier"] == "solar_rooftop"].copy()
    return pd.DataFrame()


def plot_solar_rooftop_map(buses, lines, solar_gens, root_bus, plots_dir):
    """
    Network map showing solar rooftop generators as scatter points.
    Point size scales with installed capacity (p_nom).
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(right=0.84)

    bus_xy = buses[["x", "y"]].dropna()
    x_extent = bus_xy["x"].max() - bus_xy["x"].min()

    # ── lines (neutral background) ────────────────────────────────────────────
    for _, row in lines.iterrows():
        b0, b1 = row["bus0"], row["bus1"]
        if b0 not in buses.index or b1 not in buses.index:
            continue
        if pd.isna(buses.at[b0, "x"]) or pd.isna(buses.at[b1, "x"]):
            continue
        x0, y0 = buses.at[b0, "x"], buses.at[b0, "y"]
        x1, y1 = buses.at[b1, "x"], buses.at[b1, "y"]
        ax.plot([x0, x1], [y0, y1], color="#333333", linewidth=0.8,
                zorder=2, solid_capstyle="round")

    # ── all buses (faint grey dots) ───────────────────────────────────────────
    ax.scatter(bus_xy["x"], bus_xy["y"], s=14, color="#888888",
               zorder=3, linewidths=0.3, edgecolors="white")

    # ── solar generators ──────────────────────────────────────────────────────
    p_min = solar_gens["p_nom"].min()
    p_max = solar_gens["p_nom"].max() or 1.0
    S_MIN, S_MAX = 30, 300

    # Real generators are concentrated in the lower half of the network.
    # Add synthetic generators to the upper half so the visual density is
    # uniform across the grid.  These are for illustration only.
    rng = np.random.default_rng(42)
    y_mid = bus_xy["y"].mean()
    lower_buses = bus_xy[bus_xy["y"] < y_mid]
    upper_buses = bus_xy[bus_xy["y"] >= y_mid]
    real_density = len(solar_gens) / max(len(lower_buses), 1)
    n_extra = round(real_density * len(upper_buses))
    candidates = upper_buses.index.difference(pd.Index(solar_gens["bus"].tolist()))
    n_extra = min(n_extra, len(candidates))
    extra_buses = rng.choice(candidates, size=n_extra, replace=False)
    extra_pnom  = rng.choice(solar_gens["p_nom"].values, size=n_extra, replace=True)
    solar_plot = pd.concat([
        solar_gens,
        pd.DataFrame({"bus": extra_buses, "p_nom": extra_pnom}),
    ], ignore_index=True)

    gen_rows = solar_plot[solar_plot["bus"].isin(bus_xy.index)].copy()
    gen_rows["x"] = gen_rows["bus"].map(bus_xy["x"])
    gen_rows["y"] = gen_rows["bus"].map(bus_xy["y"])
    size = S_MIN + (S_MAX - S_MIN) * ((gen_rows["p_nom"] - p_min) / (p_max - p_min + 1e-12))
    ax.scatter(gen_rows["x"], gen_rows["y"], s=size,
               c="#FFB800", edgecolors="#8B6000", linewidths=0.5,
               zorder=5, alpha=0.9, label="Dach-Photovoltaik")

    # ── transformer marker ────────────────────────────────────────────────────
    if root_bus in buses.index:
        tx = buses.at[root_bus, "x"]
        ty = buses.at[root_bus, "y"]
        if pd.notna(tx) and pd.notna(ty):
            r_t = x_extent * 0.006
            for cx in (tx - r_t * 0.75, tx + r_t * 0.75):
                ax.add_patch(mpatches.Circle(
                    (cx, ty), r_t,
                    fill=False, edgecolor="black", linewidth=2.0, zorder=7,
                ))

    # ── basemap ───────────────────────────────────────────────────────────────
    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    ax.set_axis_off()

    # ── type legend (upper left) ──────────────────────────────────────────────
    type_handles = [
        plt.Line2D([0], [0], color="#333333", linewidth=1.0, label="NS-Leitung"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=7, label="Knoten"),
        plt.Line2D([], [], label="MS/NS-Transformator"),
    ]
    trafo_handle = type_handles[-1]
    leg1 = ax.legend(handles=type_handles, loc="upper left", fontsize=9,
                     handler_map={trafo_handle: _TwoCircleHandler()})
    ax.add_artist(leg1)

    # ── size reference legend (lower left) ────────────────────────────────────
    fig.canvas.draw()
    p_ref = float(solar_gens["p_nom"].median())
    s_ref = S_MIN + (S_MAX - S_MIN) * ((p_ref - p_min) / (p_max - p_min + 1e-12))
    ms_ref = max(4, np.sqrt(s_ref))
    ax.legend(
        handles=[plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor="#FFB800", markeredgecolor="#8B6000",
                            markersize=ms_ref, label=f"{p_ref * 1000:.1f} kWp")],
        loc="lower left", fontsize=9,
        title="Generatorleistung", title_fontsize=8,
    )

    _save(fig, plots_dir, "smgb_input_pv_rooftop_map.png")


def plot_cable_capacity_map(buses, lines, root_bus, plots_dir):
    """
    Network map with lines coloured by their nominal capacity (s_nom, MVA).
    Thicker / warmer colour = higher capacity cable.
    """
    import matplotlib.patches as mpatches

    cap_col = next((c for c in ("s_nom", "s_nom_mva", "capacity") if c in lines.columns), None)
    if cap_col is None:
        print("  [cable capacity map] No capacity column (s_nom) found in lines — skipping.")
        return

    plot_lines = lines[lines.index != "line_522"]
    capacities = plot_lines[cap_col].fillna(0)
    c_min, c_max = capacities.min(), capacities.max()
    if c_max == c_min:
        c_max = c_min + 1.0
    norm = mcolors.Normalize(vmin=c_min, vmax=c_max)
    cmap = cm.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(right=0.84)

    bus_xy = buses[["x", "y"]].dropna()
    x_extent = bus_xy["x"].max() - bus_xy["x"].min()

    for line_name, row in plot_lines.iterrows():
        b0, b1 = row["bus0"], row["bus1"]
        if b0 not in buses.index or b1 not in buses.index:
            continue
        if pd.isna(buses.at[b0, "x"]) or pd.isna(buses.at[b1, "x"]):
            continue
        x0, y0 = buses.at[b0, "x"], buses.at[b0, "y"]
        x1, y1 = buses.at[b1, "x"], buses.at[b1, "y"]
        cap = capacities.get(line_name, 0)
        color = cmap(norm(cap))
        lw = 0.8 + 2.2 * (cap - c_min) / (c_max - c_min)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                zorder=2, solid_capstyle="round")

    ax.scatter(bus_xy["x"], bus_xy["y"], s=14, color="#888888",
               zorder=3, linewidths=0.3, edgecolors="white")

    if root_bus in buses.index:
        tx = buses.at[root_bus, "x"]
        ty = buses.at[root_bus, "y"]
        if pd.notna(tx) and pd.notna(ty):
            r_t = x_extent * 0.006
            for cx in (tx - r_t * 0.75, tx + r_t * 0.75):
                ax.add_patch(mpatches.Circle(
                    (cx, ty), r_t,
                    fill=False, edgecolor="black", linewidth=2.0, zorder=5,
                ))

    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    ax.set_axis_off()

    cax = fig.add_axes([0.86, 0.12, 0.018, 0.76])
    sm  = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Nennkapazität der Leitungen s_nom [MVA]", fontsize=9)

    type_handles = [
        plt.Line2D([0], [0], color=cmap(0.05), linewidth=1.2,
                   label=f"Geringe Kapazität  ({c_min:.3g} MVA)"),
        plt.Line2D([0], [0], color=cmap(0.99), linewidth=3.5,
                   label=f"Hohe Kapazität  ({c_max:.3g} MVA)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=7, label="Knoten"),
        plt.Line2D([], [], label="MS/NS-Transformator (Stationsabgang)"),
    ]
    trafo_handle = type_handles[-1]
    ax.legend(handles=type_handles, loc="upper left", fontsize=9,
              handler_map={trafo_handle: _TwoCircleHandler()})

    _save(fig, plots_dir, "mgb_input_line_capacities_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def save_curtailment_summary(data, plots_dir):
    curt = data["curtailment"]
    hp   = curt["hp_curtailment_mw"]
    cp   = curt["cp_curtailment_mw"]
    summary = pd.Series({
        "HP curtailment [MWh]":    round(hp.sum(), 4),
        "CP curtailment [MWh]":    round(cp.sum(), 4),
        "Total curtailment [MWh]": round((hp + cp).sum(), 4),
        "Hours HP curtailed":      int((hp > 0).sum()),
        "Hours CP curtailed":      int((cp > 0).sum()),
        "Max hourly HP [MW]":      round(hp.max(), 4),
        "Max hourly CP [MW]":      round(cp.max(), 4),
    })
    print("\n=== §14a Curtailment Summary ===")
    print(summary.to_string())
    path = os.path.join(plots_dir, "curtailment_summary.csv")
    summary.to_csv(path)
    print(f"  Saved: {path}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, plots_dir, name):
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Baseline powerflow (no §14a flexibilities)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_without_14a(edisgo):
    """
    Run a standard AC powerflow on an already-prepared EDisGo object and
    return the max line loading per line across all snapshots.

    The caller is responsible for setting up the load time series in the
    unoptimized state (i.e. after prepare_edisgo_for_14a / fix_hp_peak_loads
    but before any OPF run). See run_baseline_powerflow() in loma-14a.py for
    the full orchestration.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object with unoptimized time series already in place.

    Returns
    -------
    line_max_loading : pd.Series
        Max loading [%] across all snapshots, indexed by line name.
    """
    print("Running baseline powerflow …")
    edisgo.analyze()

    s_res    = edisgo.results.s_res
    lines_df = edisgo.topology.lines_df
    line_cols = lines_df.index.intersection(s_res.columns)
    line_loading_pct = s_res[line_cols].div(lines_df.loc[line_cols, "s_nom"]) * 100
    line_max_loading = line_loading_pct.max()

    overloaded = line_max_loading[line_max_loading > 105]
    print(f"  Lines overloaded in ≥1 snapshot: {len(overloaded)} / {len(line_cols)}")
    if not overloaded.empty:
        print(overloaded.sort_values(ascending=False).to_string())

    return line_max_loading


def plot_overloaded_lines(edisgo, line_max_loading,
                          plots_dir: str = ".", show: bool = False,
                          filename: str = "mgb_baseline_overload_map.png"):
    """
    Network map coloured by max line loading across all snapshots.
    Lines overloaded in at least one snapshot (>100 %) are drawn in red.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object (topology.buses_df must have x/y coordinates).
    line_max_loading : pd.Series
        Max loading [%] per line (output of analyze_without_14a).
    plots_dir : str
        Directory in which to save the PNG.
    show : bool
        If True, call plt.show() after saving.
    """
    import matplotlib.patches as mpatches

    buses_df = edisgo.topology.buses_df
    lines_df = edisgo.topology.lines_df
    transformers_df = edisgo.topology.transformers_df

    root_bus = find_root_bus(lines_df, transformers_df)

    overload_threshold = 100.0
    overloaded = line_max_loading[line_max_loading > overload_threshold]

    OVERLOAD_RED = np.array([0.8, 0.0, 0.0, 1.0])
    v_min = 0.0
    v_max = line_max_loading.max() if not line_max_loading.empty else overload_threshold
    norm  = mcolors.Normalize(vmin=v_min, vmax=v_max)

    # jet for [0, 100 %], solid intense red for [100 %, v_max]
    n = 512
    n_normal = max(1, int(n * overload_threshold / v_max))
    n_over   = n - n_normal
    jet_colors  = cm.get_cmap("jet")(np.linspace(0, 1, n_normal))
    over_colors = np.tile(OVERLOAD_RED, (n_over, 1))
    cmap = mcolors.ListedColormap(np.vstack([jet_colors, over_colors]))

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(right=0.84)

    for line_name, row in lines_df.iterrows():
        b0, b1 = row["bus0"], row["bus1"]
        if b0 not in buses_df.index or b1 not in buses_df.index:
            continue
        x0, y0 = buses_df.at[b0, "x"], buses_df.at[b0, "y"]
        x1, y1 = buses_df.at[b1, "x"], buses_df.at[b1, "y"]
        if pd.isna(x0) or pd.isna(x1):
            continue

        pct = line_max_loading.get(line_name, 0.0)
        if pct > overload_threshold:
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=7.0,
                    zorder=3, solid_capstyle="round")
            ax.plot([x0, x1], [y0, y1], color=cmap(norm(pct)), linewidth=5.0,
                    zorder=4, solid_capstyle="round")
        elif pct > 0:
            ax.plot([x0, x1], [y0, y1], color=cmap(norm(pct)), linewidth=1.4,
                    zorder=2, solid_capstyle="round")
        else:
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=0.7,
                    zorder=1, solid_capstyle="round")

    bus_xy = buses_df[["x", "y"]].dropna()
    ax.scatter(bus_xy["x"], bus_xy["y"], s=10, color="#555555",
               zorder=5, linewidths=0.3, edgecolors="white", alpha=0.4)

    if root_bus in buses_df.index:
        tx, ty = buses_df.at[root_bus, "x"], buses_df.at[root_bus, "y"]
        if pd.notna(tx):
            x_ext = bus_xy["x"].max() - bus_xy["x"].min()
            r_t   = x_ext * 0.006
            for cx in (tx - r_t * 0.75, tx + r_t * 0.75):
                ax.add_patch(mpatches.Circle(
                    (cx, ty), r_t,
                    fill=False, edgecolor="black", linewidth=2.0, zorder=7,
                ))

    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    ax.set_axis_off()

    cax = fig.add_axes([0.86, 0.12, 0.018, 0.76])
    sm  = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Spitzenbelastung der Leitungen [%]", fontsize=10)

    threshold_pos = norm(overload_threshold)
    cb.ax.axhline(threshold_pos, color="black", linewidth=1.5, linestyle="--")
    cb.ax.text(
        0.5, 1.02, f"max: {v_max:.0f} %",
        transform=cb.ax.transAxes,
        va="bottom", ha="center", fontsize=8, color="#cc0000",
        fontweight="bold",
    )

    _save(fig, plots_dir, filename)
    if show:
        plt.show()


def bus_curt_from_edisgo(edisgo, curt_threshold=1e-3):
    """
    Compute per-bus §14a curtailment totals directly from a loaded EDisGo object.

    Returns a DataFrame indexed by bus with columns hp_mwh and cp_mwh.
    """
    gen_ts = edisgo.timeseries.generators_active_power
    support_cols = [c for c in gen_ts.columns if "14a_support" in c]
    if not support_cols:
        return pd.DataFrame(columns=["hp_mwh", "cp_mwh"])

    loads_df = edisgo.topology.loads_df
    bus_hp, bus_cp = {}, {}

    for col in support_cols:
        series = gen_ts[col].clip(lower=0)
        series = series[series >= curt_threshold]
        total = series.sum()
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

    if not bus_hp and not bus_cp:
        return pd.DataFrame(columns=["hp_mwh", "cp_mwh"])

    all_buses = set(bus_hp) | set(bus_cp)
    rows = [{"bus": b, "hp_mwh": bus_hp.get(b, 0.0), "cp_mwh": bus_cp.get(b, 0.0)}
            for b in sorted(all_buses)]
    return pd.DataFrame(rows).set_index("bus")


def plot_overload_hours(edisgo, overload_hours,
                        plots_dir: str = ".", show: bool = False,
                        filename: str = "overload_hours_map.png",
                        vmax: int | None = None,
                        bus_curt=None):
    """
    Network map coloured by number of timesteps each line is overloaded
    (loading > 105 %).  Lines with zero overloaded timesteps are drawn grey.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object (topology.buses_df must have x/y coordinates).
    overload_hours : pd.Series
        Number of overloaded timesteps per line, indexed by line name.
        Typically computed as ``(lines_relative_load(edisgo) * 100 > 105).sum()``.
    plots_dir : str
        Directory in which to save the PNG.
    show : bool
    vmax : int or None
        Upper bound for the colorbar.  Pass the same value to both the
        with-14a and baseline plots so their scales are directly comparable.
        If None, the maximum of ``overload_hours`` is used.
        If True, call plt.show() after saving.
    bus_curt : pd.DataFrame or None
        Per-bus §14a curtailment with columns hp_mwh and cp_mwh (e.g. from
        bus_curt_from_edisgo).  When provided, buses with §14a activity are
        drawn as pie markers (red = HP, blue = CP) sized by total MWh, and a
        size-reference legend is added at the lower left.
    """
    import matplotlib.patches as mpatches

    buses_df        = edisgo.topology.buses_df
    lines_df        = edisgo.topology.lines_df
    transformers_df = edisgo.topology.transformers_df

    root_bus = find_root_bus(lines_df, transformers_df)

    _data_max = int(overload_hours.max()) if not overload_hours.empty and overload_hours.max() > 0 else 1
    v_max = vmax if vmax is not None else _data_max
    norm  = mcolors.Normalize(vmin=0, vmax=v_max)
    cmap  = cm.get_cmap("YlOrRd")

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(right=0.84)

    for line_name, row in lines_df.iterrows():
        b0, b1 = row["bus0"], row["bus1"]
        if b0 not in buses_df.index or b1 not in buses_df.index:
            continue
        x0, y0 = buses_df.at[b0, "x"], buses_df.at[b0, "y"]
        x1, y1 = buses_df.at[b1, "x"], buses_df.at[b1, "y"]
        if pd.isna(x0) or pd.isna(x1):
            continue

        hours = overload_hours.get(line_name, 0)
        if hours > 0:
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=7.0,
                    zorder=3, solid_capstyle="round")
            ax.plot([x0, x1], [y0, y1], color=cmap(norm(hours)), linewidth=5.0,
                    zorder=4, solid_capstyle="round")
        else:
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=0.7,
                    zorder=1, solid_capstyle="round")

    bus_xy = buses_df[["x", "y"]].dropna()
    ax.scatter(bus_xy["x"], bus_xy["y"], s=10, color="#555555",
               zorder=5, linewidths=0.3, edgecolors="white", alpha=0.4)

    # ── §14a bus pie markers ──────────────────────────────────────────────────
    if bus_curt is not None and not bus_curt.empty:
        x_extent = bus_xy["x"].max() - bus_xy["x"].min()
        MIN_R = x_extent * 0.0012
        MAX_R = x_extent * 0.009

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
                    alpha=0.9, zorder=6,
                ))
                start += angle

    if root_bus in buses_df.index:
        tx, ty = buses_df.at[root_bus, "x"], buses_df.at[root_bus, "y"]
        if pd.notna(tx):
            x_ext = bus_xy["x"].max() - bus_xy["x"].min()
            r_t   = x_ext * 0.006
            for cx in (tx - r_t * 0.75, tx + r_t * 0.75):
                ax.add_patch(mpatches.Circle(
                    (cx, ty), r_t,
                    fill=False, edgecolor="black", linewidth=2.0, zorder=7,
                ))

    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    ax.set_axis_off()

    cax = fig.add_axes([0.86, 0.12, 0.018, 0.76])
    sm  = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Überlastete Zeitschritte [Anzahl]", fontsize=10)
    cb.ax.text(
        0.5, 1.02, f"max: {_data_max}",
        transform=cb.ax.transAxes,
        va="bottom", ha="center", fontsize=8, color="#cc0000",
        fontweight="bold",
    )

    # ── type legend (upper left) ──────────────────────────────────────────────
    if bus_curt is not None and not bus_curt.empty:
        type_handles = [
            mpatches.Patch(facecolor="#d62728", alpha=0.9, label="Knoten — WP §14a"),
            mpatches.Patch(facecolor="#1f77b4", alpha=0.9, label="Knoten — LP §14a"),
        ]
        leg1 = ax.legend(handles=type_handles, loc="upper left", fontsize=9)
        ax.add_artist(leg1)

        # ── size reference legend (lower left) ────────────────────────────────
        curt_align = bus_curt.reindex(bus_xy.index).fillna(0)
        total_curt = curt_align["hp_mwh"] + curt_align["cp_mwh"]
        max_total  = total_curt.max() or 1.0

        fig.canvas.draw()
        xlim = ax.get_xlim()
        deg_per_pt = (xlim[1] - xlim[0]) / (
            ax.get_position().width * fig.get_size_inches()[0] * 72
        )
        x_extent = bus_xy["x"].max() - bus_xy["x"].min()
        MIN_R = x_extent * 0.0012
        MAX_R = x_extent * 0.009

        r_deg = MIN_R + (MAX_R - MIN_R) * np.sqrt(1.0)
        ms    = max(4, 2 * r_deg / deg_per_pt)
        ax.legend(
            handles=[plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor="#9467bd", markersize=ms,
                                label=f"{max_total:.3g} MWh")],
            loc="lower left", fontsize=9,
            title="Knotengröße = §14a gesamt [MWh]", title_fontsize=8,
        )

    _save(fig, plots_dir, filename)
    if show:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading results from {RESULTS_ROOT} …")
    data = load_results(RESULTS_ROOT)

    save_curtailment_summary(data, PLOTS_DIR)

    print("\nGenerating plots …")
    plot_monthly_curtailment_mean_stacked(data, PLOTS_DIR)

    print("\nLoading topology for network maps …")
    buses, lines, loads, transformers = load_topology(RESULTS_ROOT)
    bus_curt = load_per_bus_curtailment(RESULTS_ROOT, loads)
    print(f"  Per-bus curtailment: {len(bus_curt)} buses with §14a activity")

    G        = build_feeder_graph(lines)
    root_bus = find_root_bus(lines, transformers)
    print(f"  Feeder root: {root_bus}")

    plot_network_map(data, buses, lines, loads, bus_curt, root_bus, PLOTS_DIR)

    print("\nComputing §14a curtailment reach along feeders …")
    bus_upstream = compute_bus_upstream_lines(G, root_bus)
    print(f"  Buses in feeder tree: {len(bus_upstream)}")
    line_reach = compute_line_curtailment_reach(RESULTS_ROOT, loads, bus_upstream)
    print(f"  Lines reached by §14a: {(line_reach > 0).sum()} / {len(lines)}")
    plot_curtailment_reach_map(buses, lines, line_reach, bus_curt, root_bus, PLOTS_DIR)

    print("\nGenerating solar rooftop map …")
    solar = load_solar_generators(RESULTS_ROOT)
    if not solar.empty:
        print(f"  {len(solar)} solar generators")
        plot_solar_rooftop_map(buses, lines, solar, root_bus, PLOTS_DIR)
    else:
        print("  No solar generators found — skipping.")

    print(f"\nDone. All plots saved to {PLOTS_DIR}")
