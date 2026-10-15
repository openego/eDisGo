# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the explanatory figures used in the Methodology documentation.

The figures are intentionally schematic / illustrative (synthetic data) so that
they stay clean and didactic. Run this script to (re)generate the PNGs in this
directory::

    python doc/images/methodology_figures.py

It only needs matplotlib and numpy.
"""

from __future__ import annotations

import heapq
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# colour palette (colour-blind friendly)
BLUE = "#1f6feb"
ORANGE = "#fb8c00"
GREEN = "#2e7d32"
GREY = "#9aa0a6"
DARK = "#202124"
PURPLE = "#6f42c1"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _save(fig, name: str) -> None:
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", name)


def _box(ax, xy, w, h, text, fc, ec=DARK, tc=DARK, fs=10):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, p0, p1, color=DARK, text=None, rad=0.0, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.06, text, ha="center", va="bottom", fontsize=9, color=color)


# --------------------------------------------------------------------------- #
# 1. Generic flexibility bands (power + energy band)
# --------------------------------------------------------------------------- #
def fig_flexibility_bands_generic():
    t = np.arange(0, 24)
    t0, t1, e_req, rate = 4, 20, 6.0, 0.5
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)

    # energy corridor: charge-as-soon-as-possible (upper) vs as-late-as-possible
    # (lower); both must reach the required energy e_req by t1.
    avail = (t >= t0) & (t <= t1)
    e_upper = np.where(t < t0, 0.0, np.clip((t - t0) * rate, 0, e_req))
    e_lower = np.clip(e_req - (t1 - t) * rate, 0, e_req)
    e_lower = np.where(t > t1, e_req, e_lower)
    # an example operation: the corridor midline (guaranteed inside the band)
    e_op = 0.5 * (e_lower + e_upper)
    p_op = np.diff(e_op, prepend=0.0)

    # power band
    p_max = np.where(avail, 1.0, 0.0)
    ax1.fill_between(
        t, 0, p_max, step="mid", color=BLUE, alpha=0.18, label="power band (allowed)"
    )
    ax1.step(t, p_op, where="mid", color=ORANGE, lw=2, label="example operation $P(t)$")
    ax1.plot(t, p_max, drawstyle="steps-mid", color=BLUE, lw=1.2)
    ax1.set_ylabel("power")
    ax1.set_title("Power band")
    ax1.set_ylim(0, 1.25)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # energy band
    ax2.fill_between(
        t, e_lower, e_upper, color=GREEN, alpha=0.18, label="energy band (corridor)"
    )
    ax2.plot(t, e_upper, color=GREEN, lw=1.2)
    ax2.plot(t, e_lower, color=GREEN, lw=1.2)
    ax2.plot(t, e_op, color=ORANGE, lw=2, label="cumulative energy $E(t)$")
    ax2.set_ylabel("cumulative energy")
    ax2.set_xlabel("time step")
    ax2.set_title("Energy band")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "A flexibility = a power band + an energy band", fontsize=12, fontweight="bold"
    )
    _save(fig, "flexibility_bands.png")


# --------------------------------------------------------------------------- #
# 2. EV flexibility bands (with parking window)
# --------------------------------------------------------------------------- #
def fig_ev_flexibility_bands():
    t = np.arange(0, 24)
    arrive, leave = 17, 23
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)

    plugged = (t >= arrive) & (t <= leave)
    p_max = np.where(plugged, 1.0, 0.0)
    ax1.fill_between(
        t,
        0,
        p_max,
        step="mid",
        color=BLUE,
        alpha=0.18,
        label="$P_\\mathrm{max}$ (plugged in)",
    )
    p_op = np.zeros_like(t, dtype=float)
    p_op[[20, 21, 22]] = [0.8, 1.0, 0.7]
    ax1.step(t, p_op, where="mid", color=ORANGE, lw=2, label="OPF-chosen $P(t)$")
    ax1.axvspan(arrive - 0.5, leave + 0.5, color=GREY, alpha=0.08)
    ax1.set_ylabel("charging power")
    ax1.set_title("Power band — only while the car is plugged in")
    ax1.set_ylim(0, 1.25)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    demand = 2.5
    e_upper = np.where(plugged, np.clip(np.cumsum(p_max), 0, demand), 0)
    e_upper = np.where(t > leave, demand, e_upper)
    e_lower = np.zeros_like(t, dtype=float)
    e_lower[t >= leave] = demand
    e_op = np.cumsum(p_op)
    ax2.fill_between(
        t, e_lower, e_upper, color=GREEN, alpha=0.18, label="energy corridor"
    )
    ax2.plot(t, e_upper, color=GREEN, lw=1.2)
    ax2.plot(t, e_lower, color=GREEN, lw=1.2)
    ax2.plot(t, e_op, color=ORANGE, lw=2, label="charged energy $E(t)$")
    ax2.axhline(demand, color=DARK, ls=":", lw=1)
    ax2.annotate(
        "required energy\nby departure",
        xy=(leave, demand),
        xytext=(9.5, 1.35),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=DARK),
    )
    ax2.set_ylabel("cumulative energy")
    ax2.set_xlabel("time step (hour of day)")
    ax2.set_title("Energy band — must be full by departure")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle("Electric-vehicle flexibility bands", fontsize=12, fontweight="bold")
    _save(fig, "ev_flexibility_bands.png")


# --------------------------------------------------------------------------- #
# 3. SOC relaxation
# --------------------------------------------------------------------------- #
def fig_soc_relaxation():
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    x = np.linspace(0, 1.0, 200)
    ax.fill_between(
        x,
        0,
        x,
        color=BLUE,
        alpha=0.15,
        label="SOC-relaxed feasible set\n$P^2+Q^2 \\leq V^2 I^2$",
    )
    ax.plot(
        x, x, color=DARK, lw=2, label="exact AC (cone surface)\n$P^2+Q^2 = V^2 I^2$"
    )
    ax.scatter([0.72], [0.72], color=ORANGE, zorder=5, s=70)
    ax.annotate(
        "optimum lies on the\nsurface ⇒ relaxation\nexact (radial grids)",
        xy=(0.72, 0.72),
        xytext=(0.18, 0.82),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=ORANGE),
    )
    ax.set_xlabel("$V \\cdot I$")
    ax.set_ylabel("$\\sqrt{P^2 + Q^2}$")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.set_title("Second-order cone relaxation")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    _save(fig, "soc_relaxation.png")


# --------------------------------------------------------------------------- #
# 4. Optimise-then-reinforce loop
# --------------------------------------------------------------------------- #
def fig_optimise_reinforce_loop():
    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    steps = [
        "Set up grid,\ntime series &\nflexibilities",
        "Compute\nflexibility\nbands",
        "Optimal\npower flow\n(pm_optimize)",
        "Write\nschedules\nback",
        "Reinforce\nresidual\nproblems",
    ]
    colours = [GREY, GREEN, BLUE, GREEN, ORANGE]
    w, h, gap = 1.55, 1.2, 0.35
    x = 0.15
    centers = []
    for s, c in zip(steps, colours):
        _box(ax, (x, 0.4), w, h, s, fc=c + "22", ec=c, fs=8.5)
        centers.append(x + w / 2)
        x += w + gap
    for i in range(len(centers) - 1):
        _arrow(ax, (centers[i] + w / 2, 1.0), (centers[i + 1] - w / 2, 1.0))
    ax.set_title("The optimise-then-reinforce workflow", fontsize=12)
    _save(fig, "optimise_reinforce_loop.png")


# --------------------------------------------------------------------------- #
# 5. Reinforcement cost comparison
# --------------------------------------------------------------------------- #
def fig_cost_comparison():
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    levels = ["LV", "MV/LV", "MV"]
    without = [142, 88, 210]
    withopt = [121, 75, 178]
    cols = [GREEN, ORANGE, BLUE]
    for vals, xpos in ((without, 0), (withopt, 1)):
        bottom = 0
        for v, c, lv in zip(vals, cols, levels):
            ax.bar(
                xpos,
                v,
                bottom=bottom,
                width=0.55,
                color=c,
                edgecolor="white",
                label=lv if xpos == 0 else None,
            )
            bottom += v
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["without\noptimisation", "with\noptimisation"])
    ax.set_ylabel("grid reinforcement cost (k€)")
    ax.set_title("Flexibility reduces grid expansion")
    ax.legend(title="voltage level", fontsize=9)
    _save(fig, "reinforcement_cost_comparison.png")


# --------------------------------------------------------------------------- #
# 6. Heat pump + thermal storage
# --------------------------------------------------------------------------- #
def fig_heat_pump():
    cop = 3.0
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1, 1.35]}
    )

    # left: energy-flow schematic
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 6)
    axl.axis("off")
    _box(
        axl,
        (0.1, 2.5),
        1.9,
        1.1,
        "Grid\n(electricity)",
        fc=BLUE + "22",
        ec=BLUE,
        fs=8.5,
    )
    _box(
        axl,
        (3.1, 2.5),
        2.0,
        1.1,
        "Heat pump\n(COP ≈ 3)",
        fc=GREEN + "22",
        ec=GREEN,
        fs=8.5,
    )
    _box(
        axl,
        (6.2, 4.2),
        2.0,
        1.1,
        "Thermal\nstore $E_{th}$",
        fc=PURPLE + "22",
        ec=PURPLE,
        fs=8.5,
    )
    _box(
        axl,
        (6.2, 0.6),
        2.0,
        1.1,
        "Building\n(heat demand)",
        fc=GREY + "22",
        ec=GREY,
        fs=8.5,
    )
    _arrow(axl, (2.0, 3.05), (3.1, 3.05), color=BLUE, text="$P_{el}$")
    _arrow(axl, (5.2, 3.4), (6.3, 4.2), color=GREEN)
    axl.text(
        5.0, 4.15, "$P_{heat}=\\mathrm{COP}\\cdot P_{el}$", color=GREEN, fontsize=8
    )
    _arrow(axl, (7.2, 4.2), (7.2, 1.7), color=GREY)
    axl.text(7.4, 2.9, "$\\dot Q_{heat}$", color=GREY, fontsize=9, va="center")
    axl.set_title("Energy flow", fontsize=11)

    # right: time profile (energy-balanced over the day)
    t = np.arange(0, 24)
    q = 0.4 + 0.5 * np.exp(-((t - 7) ** 2) / 6) + 0.6 * np.exp(-((t - 19) ** 2) / 6)
    p_el = np.zeros_like(t, dtype=float)
    p_el[2:6] = 1.0  # pre-heat at night
    p_el[11:15] = 1.0  # run during midday PV
    # scale electrical use so produced heat matches the daily demand
    p_el *= q.sum() / (cop * p_el.sum())
    p_heat = cop * p_el
    e_th = 1.5 + np.cumsum(p_heat - q)

    axr.plot(t, q, color=GREY, lw=2.2, label="$\\dot Q_{heat}$ (demand, served)")
    axr.plot(t, p_el, color=BLUE, lw=2, label="$P_{el}$ (from grid)")
    axr.plot(
        t,
        p_heat,
        color=GREEN,
        lw=2,
        ls="--",
        label="$P_{heat}=\\mathrm{COP}\\cdot P_{el}$ (produced)",
    )
    axr.set_xlabel("time step (hour)")
    axr.set_ylabel("power")
    axr.set_ylim(0, max(p_heat.max(), q.max()) * 1.25)

    axe = axr.twinx()  # separate axis: E_th is an energy, not a power
    axe.fill_between(t, 0, e_th, color=PURPLE, alpha=0.12)
    axe.plot(t, e_th, color=PURPLE, lw=2.4, label="$E_{th}$ (energy in store)")
    axe.set_ylabel("energy in store $E_{th}$", color=PURPLE)
    axe.tick_params(axis="y", labelcolor=PURPLE)
    axe.set_ylim(0, e_th.max() * 1.3)

    lines = axr.get_lines() + axe.get_lines()
    axr.legend(lines, [ln.get_label() for ln in lines], fontsize=7.5, framealpha=0.9)
    axr.set_title("Storage decouples $P_{el}$ from $\\dot Q$", fontsize=11)

    fig.tight_layout()
    _save(fig, "heat_pump_thermal_storage.png")


# --------------------------------------------------------------------------- #
# 7. Load case vs feed-in case
# --------------------------------------------------------------------------- #
def _mini_grid(ax, title, direction, load_big, gen_big):
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    _box(ax, (1.8, 6.6), 2.4, 1.0, "HV grid", fc=DARK + "15", ec=DARK, fs=9)
    _box(ax, (2.2, 4.6), 1.6, 0.9, "HV/MV\nstation", fc=GREY + "22", ec=GREY, fs=8)
    ax.plot([3.0, 3.0], [4.6, 2.2], color=DARK, lw=1.6)  # feeder
    ax.scatter([3.0], [2.2], s=40, color=DARK, zorder=5)
    # load
    lc = ORANGE if load_big else GREY
    _box(ax, (0.4, 1.6), 1.5, 0.9, "load", fc=lc + "22", ec=lc, fs=8)
    ax.plot([1.9, 3.0], [2.05, 2.2], color=GREY, lw=1.2)
    # generator (PV)
    gc = GREEN if gen_big else GREY
    _box(ax, (4.1, 1.6), 1.5, 0.9, "PV", fc=gc + "22", ec=gc, fs=8)
    ax.plot([4.1, 3.0], [2.05, 2.2], color=GREY, lw=1.2)
    # power-direction arrow on the feeder
    if direction == "down":
        _arrow(ax, (3.55, 4.4), (3.55, 2.6), color=ORANGE, lw=2.4)
        ax.text(
            3.75, 3.5, "HV → grid", color=ORANGE, fontsize=9, rotation=90, va="center"
        )
    else:
        _arrow(ax, (3.55, 2.6), (3.55, 4.4), color=GREEN, lw=2.4)
        ax.text(
            3.75, 3.5, "grid → HV", color=GREEN, fontsize=9, rotation=90, va="center"
        )


def fig_load_feedin_case():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.4))
    _mini_grid(ax1, "Load case\n(high demand, low generation)", "down", True, False)
    _mini_grid(ax2, "Feed-in case\n(high generation, low demand)", "up", False, True)
    _save(fig, "load_feedin_case.png")


# --------------------------------------------------------------------------- #
# 8. Radial branch flow model
# --------------------------------------------------------------------------- #
def fig_branch_flow():
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.scatter([2, 8], [2, 2], s=260, color=BLUE, zorder=5)
    ax.text(2, 2, "$i$", color="white", ha="center", va="center", fontsize=12)
    ax.text(8, 2, "$j$", color="white", ha="center", va="center", fontsize=12)
    ax.text(2, 1.2, "$V_i$", ha="center", fontsize=11)
    ax.text(7.5, 1.5, "$V_j$", ha="center", fontsize=11)
    ax.plot([2.4, 7.6], [2, 2], color=DARK, lw=2)
    ax.text(5, 2.55, "branch $Z = R + \\mathrm{j}X$", ha="center", fontsize=10)
    _arrow(ax, (3.0, 2.0), (5.2, 2.0), color=ORANGE, lw=2.2)
    ax.text(
        4.1, 1.55, "$P_{ij}, Q_{ij}, I_{ij}$", ha="center", color=ORANGE, fontsize=10
    )
    _arrow(ax, (8.0, 1.6), (8.0, 0.5), color=GREEN, lw=2.0)
    ax.text(8.25, 1.0, "injection at $j$", color=GREEN, fontsize=9, va="center")
    ax.set_title("Radial branch-flow model", fontsize=12)
    _save(fig, "branch_flow_model.png")


# --------------------------------------------------------------------------- #
# 9. Spatial complexity reduction
# --------------------------------------------------------------------------- #
def fig_spatial_reduction():
    rng = np.random.default_rng(7)
    # build a radial tree of buses; node 0 is the HV/MV station (the root)
    n = 45
    xs = rng.uniform(0, 10, n)
    ys = rng.uniform(0, 10, n)
    xs[0], ys[0] = 5, 0
    edges = []
    for k in range(1, n):
        d = (xs[:k] - xs[k]) ** 2 + (ys[:k] - ys[k]) ** 2
        edges.append((k, int(np.argmin(d))))

    # adjacency with line lengths
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        w = float(np.hypot(xs[a] - xs[b], ys[a] - ys[b]))
        adj[a].append((b, w))
        adj[b].append((a, w))

    def dijkstra(sources):
        dist = np.full(n, np.inf)
        src = np.full(n, -1)
        heap = []
        for s in sources:
            dist[s] = 0.0
            src[s] = s
            heapq.heappush(heap, (0.0, s))
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if d + w < dist[v]:
                    dist[v] = d + w
                    src[v] = src[u]
                    heapq.heappush(heap, (d + w, v))
        return dist, src

    # representatives: the station (kept!) + farthest-point sampling along the grid
    k = 8
    reps = [0]
    mindist, _ = dijkstra([0])
    for _ in range(k - 1):
        reps.append(int(np.argmax(mindist)))
        mindist = np.minimum(mindist, dijkstra([reps[-1]])[0])

    # assign each bus to the nearest representative ALONG THE GRID (graph Voronoi).
    # On a tree this gives connected clusters, so contracting them keeps the grid
    # radial: a reduced line appears only where a real line crossed two clusters.
    _, src = dijkstra(reps)
    rep_index = {r: i for i, r in enumerate(reps)}
    assign = np.array([rep_index[int(s)] for s in src])
    cedges = {
        (min(assign[a], assign[b]), max(assign[a], assign[b]))
        for a, b in edges
        if assign[a] != assign[b]
    }
    rx, ry = xs[reps], ys[reps]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax in (axl, axr):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # original grid
    for a, b in edges:
        axl.plot([xs[a], xs[b]], [ys[a], ys[b]], color=GREY, lw=0.8, zorder=1)
    axl.scatter(xs[1:], ys[1:], s=18, color=BLUE, zorder=2)
    axl.scatter([xs[0]], [ys[0]], s=150, color=DARK, marker="s", zorder=5)
    axl.set_title(f"original grid ({n} buses)", fontsize=11)

    # reduced grid: representatives + aggregated lines, station preserved
    axr.scatter(xs, ys, s=10, color=BLUE, alpha=0.2, zorder=1)
    for ca, cb in cedges:
        axr.plot([rx[ca], rx[cb]], [ry[ca], ry[cb]], color=GREY, lw=1.5, zorder=2)
    axr.scatter(rx[1:], ry[1:], s=120, color=ORANGE, edgecolor=DARK, zorder=3)
    axr.scatter([rx[0]], [ry[0]], s=150, color=DARK, marker="s", zorder=5)
    axr.set_title(f"reduced grid ({k} clustered buses)", fontsize=11)

    # shared legend
    handles = [
        plt.Line2D([], [], marker="s", color=DARK, ls="", label="HV/MV station"),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=ORANGE,
            mec=DARK,
            ls="",
            label="representative bus",
        ),
    ]
    axr.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle("Spatial complexity reduction", fontsize=12, fontweight="bold")
    _save(fig, "spatial_complexity_reduction.png")


# --------------------------------------------------------------------------- #
# 10. DSM flexibility bands
# --------------------------------------------------------------------------- #
def fig_dsm_bands():
    t = np.arange(0, 24)
    p0 = 1.0 + 0.3 * np.sin((t - 6) / 24 * 2 * np.pi)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)

    dec, inc = 0.4, 0.4
    ax1.fill_between(
        t, p0 - dec, p0 + inc, color=BLUE, alpha=0.16, label="allowed load (band)"
    )
    ax1.plot(t, p0, color=GREY, lw=2, ls="--", label="baseline $P_0(t)$")
    p_shift = p0.copy()
    p_shift[9:13] += 0.35
    p_shift[17:21] -= 0.35
    ax1.plot(t, p_shift, color=ORANGE, lw=2, label="shifted load")
    ax1.set_ylabel("load")
    ax1.set_title("Power band — shift around the baseline")
    ax1.legend(fontsize=8, framealpha=0.9, loc="upper right")

    cum = np.cumsum(p_shift - p0)
    ax2.fill_between(t, -2.0, 2.0, color=GREEN, alpha=0.12, label="shift corridor")
    ax2.plot(t, cum, color=ORANGE, lw=2, label="cumulative shift")
    ax2.axhline(0, color=DARK, ls=":", lw=1)
    ax2.annotate(
        "returns to 0:\nenergy conserved",
        xy=(23, cum[-1]),
        xytext=(13, 1.2),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=DARK),
    )
    ax2.set_ylabel("cumulative shift")
    ax2.set_xlabel("time step")
    ax2.set_title("Energy band — only moved in time")
    ax2.set_ylim(-2.2, 2.2)
    ax2.legend(fontsize=8, framealpha=0.9, loc="upper left")

    fig.suptitle(
        "Demand-side-management flexibility bands", fontsize=12, fontweight="bold"
    )
    _save(fig, "dsm_flexibility_bands.png")


# --------------------------------------------------------------------------- #
# 11. Feeder split due to voltage issues (figure and data of the interactive
#     step-through in doc/methodology/feeder_split.html)
# --------------------------------------------------------------------------- #
# LV cables from equipment-parameters_LV_cables.csv: R in Ohm/km, L in mH/km
FS_CABLES = {
    "NAYY 4x1x300": (0.100, 0.279),
    "NAYY 4x1x150": (0.206, 0.256),
    "NAYY 4x1x95": (0.320, 0.261),
    "NAYY 4x1x50": (0.449, 0.270),
}
FS_STANDARD = "NAYY 4x1x150"
FS_KM = [0.40, 0.20, 0.18, 0.12, 0.12, 0.10]
FS_LIMIT = 0.065  # lv_max_v_drop
FS_TAN_PHI = 0.33  # cos phi of about 0.95
FS_SCENARIOS = {
    "thin": {
        "title": "Older, thinner cables",
        "types": ["NAYY 4x1x95"] * 3 + ["NAYY 4x1x50"] * 3,
        "n": [1] * 6,
        "loads": [8] * 6,
    },
    "thick": {
        "title": "Thick first section (as in grid 34632)",
        "types": ["NAYY 4x1x150"] + ["NAYY 4x1x300"] * 2 + ["NAYY 4x1x150"] * 3,
        "n": [5, 2, 2, 1, 1, 1],
        "loads": [10, 10, 40, 50, 60, 64],
    },
}


def _fs_impedance(cable, km, n):
    r_km, l_km = FS_CABLES[cable]
    return r_km * km / n, 2 * np.pi * 50 * l_km * 1e-3 * km / n


def _fs_power_flow(lines, p_kw, v_start=None):
    """
    Newton-Raphson power flow of a 0.4 kV grid with the slack at bus 0.

    `lines` are tuples (bus0, bus1, r, x) in Ohm, `p_kw` the active power of the
    loads at each bus in kW. Returns the complex bus voltages in p.u., or None if
    the power flow does not converge within 100 iterations, as in PyPSA.
    """
    z_base = 0.4**2  # Ohm, for 1 MVA
    nb = len(p_kw)
    y_bus = np.zeros((nb, nb), complex)
    for a, b, r, x in lines:
        y = 1 / complex(r / z_base, x / z_base)
        y_bus[a, a] += y
        y_bus[b, b] += y
        y_bus[a, b] -= y
        y_bus[b, a] -= y
    p = np.asarray(p_kw, float) / 1e3
    s_set = -(p + 1j * p * FS_TAN_PHI)
    v = np.ones(nb, complex) if v_start is None else v_start.copy()
    pq = np.arange(1, nb)
    for _ in range(100):
        i = y_bus @ v
        mis = v * i.conj() - s_set
        f = np.r_[mis.real[pq], mis.imag[pq]]
        if np.max(np.abs(f)) < 1e-8:
            return v
        vn = v / np.abs(v)
        ds_dva = (
            1j * np.diag(v) @ (np.diag(i.conj()) - y_bus.conj() @ np.diag(v.conj()))
        )
        ds_dvm = np.diag(v) @ (y_bus @ np.diag(vn)).conj() + np.diag(i.conj() * vn)
        jac = np.block(
            [
                [ds_dva.real[np.ix_(pq, pq)], ds_dvm.real[np.ix_(pq, pq)]],
                [ds_dva.imag[np.ix_(pq, pq)], ds_dvm.imag[np.ix_(pq, pq)]],
            ]
        )
        try:
            dx = np.linalg.solve(jac, -f)
        except np.linalg.LinAlgError:
            return None
        va, vm = np.angle(v), np.abs(v)
        va[pq] += dx[: len(pq)]
        vm[pq] += dx[len(pq) :]
        v = vm * np.exp(1j * va)
        if not np.all(np.isfinite(v)) or vm.min() < 0.05:
            return None
    return None


def _fs_lines(sc, split_n=None):
    """Lines of a scenario before the split or after it with `split_n` systems."""
    lines = []
    for k, (cable, km, n) in enumerate(zip(sc["types"], FS_KM, sc["n"])):
        if split_n and k == sc["split"] - 1:
            # the line ending at the split node is connected to the station
            km_new = sum(FS_KM[: sc["split"]])
            lines.append((0, k + 1, *_fs_impedance(FS_STANDARD, km_new, split_n)))
        else:
            lines.append((k, k + 1, *_fs_impedance(cable, km, n)))
    return lines


def _fs_drop(v):
    return None if v is None else [round(float(1 - abs(x)), 5) for x in v]


def _fs_transfer_curve(sc, split_n):
    """Voltage drop at the end of the feeder over the load behind the split node."""
    lines = _fs_lines(sc, split_n)
    p = np.r_[0.0, sc["loads"]]
    behind = np.arange(len(p)) >= sc["split"]
    p_behind = p[behind].sum()

    def solve(scale, v_start):
        return _fs_power_flow(lines, np.where(behind, p * scale, p), v_start)

    curve, v, scale, step = [], None, 0.0, 0.02
    while step > 1e-4:
        v_new = solve(scale + step, v)
        if v_new is None:
            step /= 2
            continue
        scale += step
        v = v_new
        curve.append([round(scale * p_behind, 1), round(float(1 - abs(v[-1])), 5)])
    # thin out the points, which get dense where the step size was reduced
    points = [curve[0]]
    for point in curve[1:-1]:
        if point[0] - points[-1][0] >= p_behind / 25:
            points.append(point)
    return {"points": points + curve[-1:], "p_max": round(scale * p_behind, 1)}


def feeder_split_example():
    """Compute the example feeders before and after the split."""
    nodes_km = np.r_[0.0, np.cumsum(FS_KM)]
    data = {"km": FS_KM, "limit": FS_LIMIT, "std": FS_STANDARD, "scenarios": {}}
    for key, sc in FS_SCENARIOS.items():
        sc = dict(sc)
        before = _fs_drop(_fs_power_flow(_fs_lines(sc), np.r_[0.0, sc["loads"]]))
        crit = [b for b in range(1, len(before)) if before[b] > FS_LIMIT]
        far = max(crit, key=lambda b: nodes_km[b])
        two_thirds = 2 / 3 * nodes_km[far]
        sc["split"] = int(np.argmax(nodes_km >= two_thirds))
        r_path = sum(
            _fs_impedance(c, km, n)[0]
            for c, km, n in list(zip(sc["types"], FS_KM, sc["n"]))[: sc["split"]]
        )
        km_new = nodes_km[sc["split"]]
        after = {}
        n_new = 1
        while True:
            v = _fs_power_flow(_fs_lines(sc, n_new), np.r_[0.0, sc["loads"]])
            after[n_new] = _fs_drop(v)
            if v is not None or n_new == 5:
                break
            n_new += 1
        data["scenarios"][key] = {
            "title": sc["title"],
            "types": sc["types"],
            "n": sc["n"],
            "loads": sc["loads"],
            "critical": crit,
            "far": far,
            "two_thirds": round(two_thirds, 4),
            "split": sc["split"],
            "km_new": round(km_new, 4),
            "r_path": round(r_path, 4),
            "r_new": round(_fs_impedance(FS_STANDARD, km_new, 1)[0], 4),
            "p_behind": float(sum(sc["loads"][sc["split"] - 1 :])),
            "before": before,
            "after": {str(n): d for n, d in after.items()},
            "curves": {
                "before": _fs_transfer_curve(sc, None),
                **{str(n): _fs_transfer_curve(sc, n) for n in after},
            },
        }
    return data


def write_feeder_split_widget(data):
    """Write the example data into the interactive step-through."""
    import json
    import re

    path = os.path.join(HERE, "..", "methodology", "feeder_split.html")
    with open(path) as f:
        html = f.read()
    html = re.sub(
        r"var D = .*?;\n",
        lambda _: "var D = " + json.dumps(data, separators=(",", ":")) + ";\n",
        html,
        count=1,
    )
    with open(path, "w") as f:
        f.write(html)
    print("wrote feeder_split.html")


def fig_feeder_split(data):
    sc = data["scenarios"]["thick"]
    nodes_km = np.r_[0.0, np.cumsum(data["km"])]
    split, n_new = sc["split"], max(int(n) for n in sc["after"])
    colors = {
        "NAYY 4x1x300": PURPLE,
        "NAYY 4x1x150": BLUE,
        "NAYY 4x1x95": ORANGE,
        "NAYY 4x1x50": ORANGE,
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    def feeder(ax, split_n, title):
        ax.set_xlim(-0.12, nodes_km[-1] + 0.05)
        ax.set_ylim(-0.75, 0.42)
        ax.axis("off")
        for k, (cable, n) in enumerate(zip(sc["types"], sc["n"])):
            xa, xb = nodes_km[k] + 0.012, nodes_km[k + 1] - 0.012
            if split_n and k == split - 1:
                ys = [(j - (split_n - 1) / 2) * 0.035 - 0.45 for j in range(split_n)]
                for y in ys:
                    dx = y + 0.45
                    ax.plot(
                        [dx, dx, xb + 0.012 + dx, xb + 0.012 + dx],
                        [-0.1, y, y, -0.04],
                        color=BLUE,
                        lw=1.6,
                    )
                ax.text(
                    nodes_km[split] / 2,
                    min(ys) - 0.1,
                    f"new line: {split_n}× {FS_STANDARD}, {sc['km_new']:.2f} km",
                    ha="center",
                    fontsize=8.5,
                    color=BLUE,
                )
                continue
            for j in range(min(n, 6)):
                y = (j - (min(n, 6) - 1) / 2) * 0.035
                ax.plot([xa, xb], [y, y], color=colors[cable], lw=1.6)
            ax.text(
                (xa + xb) / 2,
                0.2,
                f"L{k + 1}" + (f" ×{n}" if n > 1 else ""),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
            ax.text(
                (xa + xb) / 2, 0.32, cable.split("x")[-1], ha="center", fontsize=7.5
            )
        ax.add_patch(plt.Rectangle((-0.05, -0.1), 0.04, 0.2, color=DARK))
        for b in range(1, len(nodes_km)):
            crit = b in sc["critical"] and not split_n
            ax.plot(
                nodes_km[b],
                0,
                "o",
                ms=7,
                mfc="#c2362b" if crit else "white",
                mec="#c2362b" if crit else DARK,
                mew=1.4,
            )
            ax.text(nodes_km[b], 0.07, f"B{b}", ha="center", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10)

    feeder(ax_a, None, "(a) Feeder before the split, critical buses in red")
    ax_a.axvline(sc["two_thirds"], ymin=0.25, ymax=0.62, color=GREY, ls="--", lw=1.2)
    ax_a.text(
        sc["two_thirds"],
        -0.42,
        "2/3 of the path\nto B6",
        ha="center",
        fontsize=8,
        color=DARK,
    )
    feeder(
        ax_b,
        n_new,
        f"(b) After the split: L{split} connects B{split} to the station",
    )

    for key, label, color, ls in [
        ("before", "before the split", GREY, "--"),
        ("1", f"split, 1× {FS_STANDARD}", ORANGE, "-"),
        (str(n_new), f"split, {n_new}× {FS_STANDARD}", BLUE, "-"),
    ]:
        pts = np.array(sc["curves"][key]["points"])
        ax_d.plot(pts[:, 0], pts[:, 1] * 100, color=color, ls=ls, lw=2, label=label)
        ax_d.plot(pts[-1, 0], pts[-1, 1] * 100, "x", color=color, ms=8, mew=2)
        ax_d.text(
            pts[-1, 0] + 8,
            pts[-1, 1] * 100 + 1.5,
            f"max. {pts[-1, 0]:.0f} kW",
            fontsize=8,
            color=color,
        )
    ax_d.axvline(sc["p_behind"], color=DARK, lw=1)
    ax_d.text(
        sc["p_behind"] + 6,
        16,
        f"load behind\nthe split: {sc['p_behind']:.0f} kW",
        fontsize=8.5,
    )
    ax_d.axhline(data["limit"] * 100, color="#c2362b", lw=1.2, ls=(0, (6, 4)))
    ax_d.set_xlabel("load behind the split node in kW")
    ax_d.set_ylabel("voltage drop at B6 in %")
    ax_d.set_ylim(0, 60)
    ax_d.set_xlim(0, max(c["p_max"] for c in sc["curves"].values()) * 1.25)
    ax_d.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax_d.set_title(
        "(d) Power flow solutions end at ×: no solution for more load",
        loc="left",
        fontsize=10,
    )

    for key, label, color, ls in [
        ("before", "before the split", GREY, "--"),
        (str(n_new), f"split, {n_new}× {FS_STANDARD}", BLUE, "-"),
    ]:
        drop = np.array(sc["before"] if key == "before" else sc["after"][key]) * 100
        if key == "before":
            ax_c.plot(nodes_km, drop, color=color, ls=ls, lw=2, marker="o", ms=4)
        else:
            front = list(range(split))
            rear = [0] + list(range(split, len(nodes_km)))
            ax_c.plot(nodes_km[front], drop[front], color=color, lw=2, marker="o", ms=4)
            ax_c.plot(nodes_km[rear], drop[rear], color=color, lw=2, marker="o", ms=4)
        ax_c.plot([], [], color=color, ls=ls, lw=2, label=label)
    ax_c.axhline(data["limit"] * 100, color="#c2362b", lw=1.2, ls=(0, (6, 4)))
    ax_c.text(0.01, data["limit"] * 100 + 0.6, "allowed 6.5 %", color="#c2362b")
    ax_c.set_xticks(nodes_km)
    ax_c.set_xticklabels(["station"] + [f"B{b}" for b in range(1, len(nodes_km))])
    ax_c.set_ylabel("voltage drop in %")
    ax_c.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax_c.set_title(
        f"(c) Voltage drop after the split with {n_new} lines (1 line: no solution)",
        loc="left",
        fontsize=10,
    )
    for ax in (ax_c, ax_d):
        ax.grid(axis="y", color="#e3e8eb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "feeder_split.png")


def main():
    fig_flexibility_bands_generic()
    fig_ev_flexibility_bands()
    fig_soc_relaxation()
    fig_optimise_reinforce_loop()
    fig_cost_comparison()
    fig_heat_pump()
    fig_load_feedin_case()
    fig_branch_flow()
    fig_spatial_reduction()
    fig_dsm_bands()
    feeder_split = feeder_split_example()
    fig_feeder_split(feeder_split)
    write_feeder_split_widget(feeder_split)


if __name__ == "__main__":
    main()
