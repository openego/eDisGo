"""
For every saved seed × month EDisGo object (post-OPF), run two AC powerflows:

  1. With §14a generators active  →  line loading as seen after §14a optimisation
  2. §14a generators zeroed out   →  counterfactual baseline without §14a

The baseline uses the same seed/CP/HP placement as the OPF run, so the
comparison is apples-to-apples.  The resulting line_usage_baseline CSV is
written to each month directory AND to the seed directory (concatenated), so
load_baseline_results() in loma_14a_analysis.py can pick it up for the
before/after plot.

At the end a summary table shows the improvement and whether §14a alone was
sufficient (i.e. whether residual load shedding was still needed).
"""
import glob
import os

import pandas as pd
from edisgo.edisgo import import_edisgo_from_files
from edisgo.flex_opt.check_tech_constraints import lines_relative_load

from loma_14a_analysis import analyze_without_14a, bus_curt_from_edisgo, plot_overload_hours

RESULTS_ROOT = "/home/carlos/LoMa/output_edisgo"
PLOTS_DIR    = f"{RESULTS_ROOT}/presentation_plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

summary_rows = []

for seed_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*"))):
    if not os.path.isdir(seed_dir):
        continue
    seed = os.path.basename(seed_dir)

    bl_parts = []   # per-month baseline DataFrames, concatenated at seed level

    for edisgo_dir in sorted(glob.glob(os.path.join(seed_dir, "*/edisgo"))):
        month_dir = os.path.dirname(edisgo_dir)
        month     = os.path.basename(month_dir)

        print(f"\n{'='*62}")
        print(f"  seed={seed}  month={month}")
        print(f"{'='*62}")

        edisgo = import_edisgo_from_files(
            edisgo_dir,
            import_topology=True,
            import_timeseries=True,
        )

        # ── 1. With §14a (post-OPF state) ────────────────────────────────────
        # The stored eDisGo was saved after the OPF, so 14a_support generators
        # carry the optimised curtailment schedule; analyze() reflects that.
        analyze_without_14a(edisgo)
        lu_with_14a = lines_relative_load(edisgo) * 100

        # ── 2. Without §14a (zero out support generators, then restore) ───────
        # Must write directly to _generators_active_power; the public property
        # returns a .loc slice (copy), so chained assignment is silently dropped.
        _gen_df      = edisgo.timeseries._generators_active_power
        support_cols = [c for c in _gen_df.columns if "14a_support" in c]
        saved        = _gen_df[support_cols].copy()
        _gen_df.loc[:, support_cols] = 0.0

        analyze_without_14a(edisgo)
        lu_baseline = lines_relative_load(edisgo) * 100

        _gen_df.loc[:, support_cols] = saved

        # ── Plot both with a shared colorbar scale ────────────────────────────
        hours_with_14a = (lu_with_14a > 105).sum()
        hours_baseline = (lu_baseline > 105).sum()
        shared_vmax    = int(max(hours_with_14a.max(), hours_baseline.max(), 1))

        bus_curt = bus_curt_from_edisgo(edisgo)

        plot_overload_hours(
            edisgo, hours_with_14a,
            plots_dir=month_dir,
            filename=f"overload_hours_with_14a_{month}.png",
            vmax=shared_vmax,
            bus_curt=bus_curt,
        )
        plot_overload_hours(
            edisgo, hours_baseline,
            plots_dir=month_dir,
            filename=f"overload_hours_baseline_{month}.png",
            vmax=shared_vmax,
        )

        # ── Save per-month baseline for load_baseline_results() ───────────────
        lu_baseline.to_csv(os.path.join(month_dir, "line_usage_baseline"))
        bl_parts.append(lu_baseline)

        # ── Load residual shedding written by loma-14a.py ─────────────────────
        ls_total = 0.0
        for fname in ("load_shedding.csv", "hp_load_shedding.csv"):
            p = os.path.join(month_dir, fname)
            if os.path.isfile(p):
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                ls_total += df.abs().sum().sum()

        # ── Per-month comparison table ────────────────────────────────────────
        ol_before = int((lu_baseline > 105).sum().sum())
        ol_after  = int((lu_with_14a > 105).sum().sum())
        reduction = (ol_before - ol_after) / ol_before * 100 if ol_before > 0 else 0.0
        feasible  = ls_total <= 1e-3

        print(f"\n  {'Metric':<35} {'Before §14a':>12}  {'After §14a':>10}")
        print(f"  {'─'*60}")
        print(f"  {'Overloaded line-hours':<35} {ol_before:>12}  {ol_after:>10}")
        print(f"  {'Reduction':<35} {'':>12}  {reduction:>9.0f}%")
        print(f"  {'Residual load shedding [MWh]':<35} {'—':>12}  {ls_total:>10.4f}")
        if not feasible:
            print(f"  ⚠  §14a insufficient: {ls_total:.4f} MWh of extra shedding required")

        summary_rows.append({
            "seed":          seed,
            "month":         month,
            "ol_before":     ol_before,
            "ol_after":      ol_after,
            "reduction_pct": round(reduction, 1),
            "shed_mwh":      round(ls_total, 4),
            "feasible":      feasible,
        })

    # ── Save seed-level baseline (all months concatenated) ────────────────────
    if bl_parts:
        lu_bl_seed = pd.concat(bl_parts, sort=False)
        lu_bl_seed.to_csv(os.path.join(seed_dir, "line_usage_baseline"))
        print(f"\n[seed={seed}] line_usage_baseline saved "
              f"({len(lu_bl_seed)} timesteps, {lu_bl_seed.shape[1]} lines)")

# ── Final cross-seed summary ──────────────────────────────────────────────────
if summary_rows:
    summary = pd.DataFrame(summary_rows)
    print(f"\n{'='*80}")
    print("  Before / After §14a — Summary")
    print(f"{'='*80}")
    print(summary.to_string(index=False))

    n_infeas = (~summary["feasible"]).sum()
    total    = len(summary)
    print(f"\n  Months where §14a was insufficient : {n_infeas} / {total}")
    print(f"  Mean overload reduction            : {summary['reduction_pct'].mean():.0f}%")
    print(f"  Total residual shedding            : {summary['shed_mwh'].sum():.4f} MWh")

print(f"\nDone. Per-month plots in each month directory; "
      f"seed-level line_usage_baseline saved alongside line_usage.")
