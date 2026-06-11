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
    withopt = [40, 22, 70]
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
    axr.set_ylim(0, max(p_heat.max(), q.max()) * 1.15)

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


if __name__ == "__main__":
    main()
