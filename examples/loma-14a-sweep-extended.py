"""
§14a Penetration Sweep – Extended
==================================
Extended version of loma-14a-sweep.py.

New features vs. original:
  - Reinforce costs measured before/after §14a optimisation → CSV per run
  - Load-shedding time series (general, HP, CP) saved as individual CSVs
  - Extended §14a activation report:
      * relative line loading  (s_res / s_nom)  instead of absolute MVA
      * count of lines exceeding capacity per timestep
      * top overloaded lines ranked by relative loading
      * top injecting generators per timestep
  - Full eDisGo object saved to CSV after optimisation
  - Focus plots for top-3 §14a buses (only buses where peak curtailment > 0.5 kW)
"""

import copy
import os
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from edisgo import EDisGo
from edisgo.tools.loma_tools import (
    buses_with_existing_loads,
    create_network_gif,
    get_curtailment_data,
    plot_14a_focus_bus,
    plot_14a_overview_full_period,
    plot_network,
    set_charging_points_to_target,
    set_heat_pumps_to_target,
    set_storage_timeseries_bus_level,
    transfer_ts_from_new_to_existing_cp,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# SWEEP CONFIGURATION  –  adapt here
# ===========================================================================
SWEEP_STEPS = [
    (412, 575)
    #(1000, 2600),   # baseline
    #(1200, 3120),   # 20 %
    #(1400, 3640),   # 40 %
    #(1600, 4160),   # 60 %
    #(1800, 4680),   # 80 %
    #(2000, 5200),   # 100 %
]

# ===========================================================================
# PATHS  –  adapt as needed
# ===========================================================================
GRID_PATH      = "/home/student/Execution/LoMa_exe/results/Whole_Husum_final_statusQuo_LV_ids"
SHP_PATH       = "/home/student/Execution/LoMa_exe/data/Input_files/MV_grid_district/husum_district.shp"
BASE_OUT_DIR   = "/home/student/Execution/eDisGo_exe/sweep_14a_Husum_StatusQuo_extended"
EMOB_CACHE_DIR = "/home/student/Execution/eDisGo_exe/emob_cache_Whole_Husum_final_statusQuo"
SNAPSHOT_RANGE = (7944, 7968)


# ===========================================================================
# Helper – emob cache  (unchanged)
# ===========================================================================
_EMOB_CACHE_FILES = [
    "charging_processes_df.pkl",
    "potential_charging_parks_gdf.pkl",
    "simbev_config_df.pkl",
    "integrated_charging_parks_df.pkl",
    "new_cp_loads_df.pkl",
]


def _emob_cache_exists(cache_dir):
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in _EMOB_CACHE_FILES)


def _save_emob_cache(edisgo, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    emob = edisgo.electromobility
    emob.charging_processes_df.to_pickle(os.path.join(cache_dir, "charging_processes_df.pkl"))
    emob.potential_charging_parks_gdf.to_pickle(os.path.join(cache_dir, "potential_charging_parks_gdf.pkl"))
    emob.simbev_config_df.to_pickle(os.path.join(cache_dir, "simbev_config_df.pkl"))
    emob.integrated_charging_parks_df.to_pickle(os.path.join(cache_dir, "integrated_charging_parks_df.pkl"))

    edisgo_cp_ids = set(emob.integrated_charging_parks_df["edisgo_id"].dropna())
    new_cp_rows = edisgo.topology.loads_df[edisgo.topology.loads_df.index.isin(edisgo_cp_ids)]
    new_cp_rows.to_pickle(os.path.join(cache_dir, "new_cp_loads_df.pkl"))
    logger.info("[emob cache] Saved to %s", cache_dir)


def _load_emob_cache(edisgo, cache_dir):
    emob = edisgo.electromobility
    emob.charging_processes_df          = pd.read_pickle(os.path.join(cache_dir, "charging_processes_df.pkl"))
    emob.potential_charging_parks_gdf   = pd.read_pickle(os.path.join(cache_dir, "potential_charging_parks_gdf.pkl"))
    emob.simbev_config_df               = pd.read_pickle(os.path.join(cache_dir, "simbev_config_df.pkl"))
    emob.integrated_charging_parks_df   = pd.read_pickle(os.path.join(cache_dir, "integrated_charging_parks_df.pkl"))

    new_cp_rows = pd.read_pickle(os.path.join(cache_dir, "new_cp_loads_df.pkl"))
    missing = new_cp_rows.index.difference(edisgo.topology.loads_df.index)
    if len(missing) > 0:
        edisgo.topology.loads_df = pd.concat([edisgo.topology.loads_df, new_cp_rows.loc[missing]])
    logger.info("[emob cache] Loaded from %s (%d eDisGo CPs restored)", cache_dir, len(new_cp_rows))


# ===========================================================================
# Preparation pipeline (unchanged)
# ===========================================================================

def _integrate_ev_and_hp(edisgo, *, shapefile_path, output_dir, cp_target, hp_target,
                          setup_days=None, cache_dir=None):
    """Import EV data, apply charging strategy, then scale CP/HP to targets."""

    if cache_dir is not None and _emob_cache_exists(cache_dir):
        _load_emob_cache(edisgo, cache_dir)
    else:
        edisgo.import_electromobility_14a(
            scenario="eGon2035",
            import_electromobility_data_kwds={"shapefile_path": shapefile_path},
        )
        if cache_dir is not None:
            _save_emob_cache(edisgo, cache_dir)

    _orig_days = None
    if setup_days is not None:
        _orig_days = int(edisgo.electromobility.simbev_config_df.at[0, "days"])
        _capped = min(setup_days, _orig_days)
        edisgo.electromobility.simbev_config_df.at[0, "days"] = _capped

    ti = edisgo.timeseries.timeindex
    lap_cols = edisgo.timeseries.loads_active_power.columns
    edisgo.timeseries.loads_reactive_power = pd.DataFrame(0.0, index=ti, columns=lap_cols)
    edisgo.apply_charging_strategy(strategy="dumb")

    if _orig_days is not None:
        edisgo.electromobility.simbev_config_df.at[0, "days"] = _orig_days

    transfer_ts_from_new_to_existing_cp(
        edisgo,
        existing_markers=("Existing", "Additional"),
        radius_1=2000.0, tol_1=0.15,
        radius_2=2000.0, tol_2=0.9,
    )

    valid_buses = set(edisgo.topology.buses_df.index)
    cp_eligible = [b for b in buses_with_existing_loads(edisgo) if b in valid_buses]
    set_charging_points_to_target(
        edisgo,
        target_total=cp_target,
        eligible_buses=cp_eligible,
        removal_priority=["Additional", "Existing"],
        add_tracking_columns=False,
        export_removed=False,
        export_dir=output_dir,
    )

    valid_buses = set(edisgo.topology.buses_df.index)
    hp_eligible = [b for b in buses_with_existing_loads(edisgo) if b in valid_buses]
    set_heat_pumps_to_target(
        edisgo,
        target_total=hp_target,
        eligible_buses=hp_eligible,
        add_tracking_columns=False,
        export_removed=False,
        export_dir=output_dir,
    )


def prepare_edisgo(edisgo, *, shapefile_path, output_dir, cp_target, hp_target,
                   cache_dir=None, setup_days=None):
    """Topology fixes, EV/HP integration, storage ts, reactive power setup."""

    edisgo.topology.generators_df = edisgo.topology.generators_df[
        edisgo.topology.generators_df.index != "HV_dummy_gen_slack"
    ]
    edisgo.topology.buses_df = edisgo.topology.buses_df[edisgo.topology.buses_df.v_nom <= 20]
    edisgo.topology.buses_df = edisgo.topology.buses_df[
        ~edisgo.topology.buses_df.index.isin(["HV_dummy_bus", "bus_20111_HV"])
    ]

    _integrate_ev_and_hp(
        edisgo,
        shapefile_path=shapefile_path,
        output_dir=output_dir,
        cp_target=cp_target,
        hp_target=hp_target,
        cache_dir=cache_dir,
        setup_days=setup_days,
    )

    set_storage_timeseries_bus_level(edisgo)

    hp_names = list(edisgo.topology.loads_df[edisgo.topology.loads_df["type"] == "heat_pump"].index)
    timeindex = edisgo.timeseries.timeindex
    edisgo.heat_pump.cop_df = pd.DataFrame(3.0, index=timeindex, columns=hp_names)
    edisgo.heat_pump.heat_demand_df = edisgo.timeseries.loads_active_power[hp_names] * 3.0

    edisgo.set_time_series_reactive_power_control()


def run_optimization_14a(edisgo):
    """Run OPF with §14a curtailment (opf_version=5)."""
    logger.info("Running OPF with §14a curtailment …")
    t0 = datetime.now()
    edisgo.pm_optimize(opf_version=5, curtailment_14a=True)
    dt = (datetime.now() - t0).total_seconds()
    logger.info("Optimization done in %.1f s (%.1f min)", dt, dt / 60)
    return edisgo


# ===========================================================================
# NEW: Reinforce cost measurement
# ===========================================================================

def find_convergent_snapshots(edisgo, snapshots, _depth=0):
    """
    Robustly split *snapshots* into ones the power-flow solver can analyze
    cleanly and ones that must be excluded.

    edisgo.analyze(raise_not_converged=False) only ever reports soft
    non-convergence (Newton-Raphson not reaching tolerance within the
    iteration limit). Some snapshots crash the solver outright with a hard
    linear-algebra error (e.g. singular Jacobian → "failed to factorize
    matrix"). That exception aborts analyze() for the entire batch before it
    can finish building its non-convergence list.

    To isolate the offending snapshot(s) this function recursively bisects
    *snapshots* whenever analyze() raises: a crash on a sub-range means at
    least one "poison" snapshot is in there, so the range is split in half
    and each half is probed independently until the individual culprits are
    pinned down. This needs only O(log n) probes per culprit instead of
    checking every snapshot individually.

    Each probe runs analyze() on a disposable copy of *edisgo* so a crash
    cannot corrupt the solver's internal state for subsequent probes.

    Returns
    -------
    tuple[pd.DatetimeIndex, pd.DatetimeIndex]
        (convergent_snapshots, excluded_snapshots)
    """
    if len(snapshots) == 0:
        return snapshots, snapshots

    indent = "    " * _depth

    try:
        _, soft_not_converged = edisgo.copy().analyze(
            timesteps=snapshots, raise_not_converged=False
        )
    except Exception as e:
        if len(snapshots) == 1:
            logger.warning(
                "%s✗ Hard solver crash at %s: %s",
                indent, snapshots[0], str(e)[:120],
            )
            return snapshots[:0], snapshots

        mid = len(snapshots) // 2
        logger.info(
            "%sanalyze() crashed for %d timesteps (%s … %s) – "
            "bisecting to isolate the culprit(s) …",
            indent, len(snapshots), snapshots[0], snapshots[-1],
        )
        ok_left,  bad_left  = find_convergent_snapshots(edisgo, snapshots[:mid],  _depth + 1)
        ok_right, bad_right = find_convergent_snapshots(edisgo, snapshots[mid:], _depth + 1)
        return ok_left.append(ok_right), bad_left.append(bad_right)

    if len(soft_not_converged):
        logger.info(
            "%s%d/%d timesteps with soft non-convergence (NR tolerance not reached).",
            indent, len(soft_not_converged), len(snapshots),
        )

    convergent = snapshots.difference(soft_not_converged)
    excluded   = snapshots.intersection(soft_not_converged)
    return convergent, excluded


def _diagnose_generators(edisgo, label=""):
    """Log generators with missing or invalid bus entries."""
    gen_df = edisgo.topology.generators_df
    buses = edisgo.topology.buses_df.index

    null_bus = gen_df[gen_df["bus"].isna()]
    if not null_bus.empty:
        logger.warning(
            "[%s] %d generator(s) with bus=NaN: %s",
            label, len(null_bus), null_bus.index.tolist(),
        )

    missing_bus = gen_df[~gen_df["bus"].isna() & ~gen_df["bus"].isin(buses)]
    if not missing_bus.empty:
        logger.warning(
            "[%s] %d generator(s) whose bus is not in buses_df: %s",
            label, len(missing_bus),
            missing_bus[["bus"]].to_dict(orient="index"),
        )

    if null_bus.empty and missing_bus.empty:
        logger.info("[%s] All %d generators have valid bus entries.", label, len(gen_df))


def measure_reinforce_costs(edisgo, label=""):
    """
    Run reinforce on a deep copy to measure grid-expansion costs without
    modifying the original grid topology.

    find_convergent_snapshots() filters out timesteps that crash the solver
    (hard factorisation errors) or fail to converge (soft NR non-convergence,
    including NaN results which PyPSA logs as "solved" with error=nan but
    marks as non-converged internally).  Reinforce is then run only on the
    remaining clean timesteps.  If reinforce fails despite the filtering
    (e.g. the modified grid develops a new singularity during the loop),
    an empty DataFrame is returned – this is acceptable.
    """
    logger.info("Measuring reinforce costs (%s) …", label)
    _diagnose_generators(edisgo, label=label)
    t0 = datetime.now()

    logger.info("[%s] Finding convergent snapshots (bisective search) …", label)
    edisgo_copy = edisgo.copy()
    ts_ok, ts_not_converged = find_convergent_snapshots(
        edisgo_copy, edisgo_copy.timeseries.timeindex
    )

    if len(ts_not_converged) > 0:
        logger.warning(
            "[%s] %d/%d timestep(s) excluded (non-convergent or singular).",
            label, len(ts_not_converged), len(edisgo_copy.timeseries.timeindex),
        )

    if len(ts_ok) == 0:
        logger.warning("[%s] No convergent snapshots – skipping reinforce.", label)
        return pd.DataFrame()

    # Populate PF results on the copy before reinforce so reinforce_grid()
    # has results to work with from the start.
    try:
        edisgo_copy.analyze(timesteps=ts_ok, raise_not_converged=False)
    except Exception as exc_a:
        logger.warning("[%s] Pre-analyze failed: %s", label, exc_a)

    try:
        edisgo_copy.reinforce(
            timesteps_pfa=ts_ok if len(ts_not_converged) > 0 else None,
            reduced_analysis=False,
            mode=None,
            catch_convergence_problems=True,
            max_while_iterations=50,
            copy_grid=False,
        )
        costs = edisgo_copy.results.grid_expansion_costs
        result = costs.copy() if costs is not None else pd.DataFrame()
        dt = (datetime.now() - t0).total_seconds()
        logger.info(
            "[%s] Reinforce done in %.1f s → %d cost rows (method: %s)",
            label, dt, len(result),
            "filtered" if len(ts_not_converged) > 0 else "full",
        )
        return result
    except Exception as exc:
        import traceback
        logger.warning(
            "[%s] Reinforce failed (topology modified by reinforce loop may have "
            "become singular): %s\n%s",
            label, exc, traceback.format_exc(),
        )
        return pd.DataFrame()


def save_reinforce_costs(costs_before, costs_after, run_dir):
    """Persist reinforce cost DataFrames and a scalar summary CSV."""
    os.makedirs(run_dir, exist_ok=True)

    if not costs_before.empty:
        costs_before.to_csv(os.path.join(run_dir, "reinforce_costs_before_14a.csv"))
    if not costs_after.empty:
        costs_after.to_csv(os.path.join(run_dir, "reinforce_costs_after_14a.csv"))

    def _total(df):
        if df.empty:
            return float("nan")
        num = df.select_dtypes(include="number")
        return float(num.values.sum()) if not num.empty else float("nan")

    summary = {
        "reinforce_total_before_14a": _total(costs_before),
        "reinforce_total_after_14a":  _total(costs_after),
    }
    pd.Series(summary).to_csv(
        os.path.join(run_dir, "reinforce_costs_summary.csv"), header=["value"]
    )
    logger.info(
        "Reinforce costs  |  before=%.2f  after=%.2f",
        summary["reinforce_total_before_14a"],
        summary["reinforce_total_after_14a"],
    )


# ===========================================================================
# NEW: Load-shedding time series
# ===========================================================================

def save_loadshedding_timeseries(edisgo, run_dir):
    """Save load-shedding time series (general, HP, CP) to individual CSVs."""
    os.makedirs(run_dir, exist_ok=True)
    slacks = edisgo.opf_results.grid_slacks_t

    for filename, ts in [
        ("ts_loadshedding_general.csv", slacks.load_shedding),
        ("ts_loadshedding_hp.csv",      slacks.hp_load_shedding),
        ("ts_loadshedding_cp.csv",      slacks.cp_load_shedding),
    ]:
        if ts is not None and not ts.empty:
            ts.to_csv(os.path.join(run_dir, filename))
            logger.info("  Saved shedding: %s", filename)
        else:
            logger.info("  Empty / not available: %s (skipped)", filename)


# ===========================================================================
# NEW: Relative line loading helper + extended activation report
# ===========================================================================

def _build_relative_line_loading(edisgo):
    """Return s_rel DataFrame (s_res / s_nom) for lines and transformers."""
    s_res = edisgo.results.s_res
    s_nom = pd.concat([
        edisgo.topology.lines_df["s_nom"],
        edisgo.topology.transformers_df["s_nom"],
    ])
    common = s_res.columns.intersection(s_nom.index)
    return s_res[common].div(s_nom[common], axis=1)


def create_extended_activation_report(edisgo, pre_opt_line_loading, threshold_kw=0.5):
    """
    Build an extended §14a activation report per timestep.

    For every timestep where §14a curtailment occurs above threshold_kw the
    report includes:
      - total_14a_curtailment_mw : aggregate §14a power reduced
      - n_lines_overloaded        : number of branches with s_res > s_nom
      - top_overloaded_lines      : up to 5 branches sorted by relative loading
      - top_injecting_generators  : up to 3 generators by active power injection

    Also returns:
      s_rel              : full relative line loading DataFrame (time × branch)
      top_lines_overall  : Series – max relative loading per branch (all timesteps)
    """
    s_rel = _build_relative_line_loading(edisgo)
    n_overloaded_ts = (s_rel > 1.0).sum(axis=1)
    top_lines_overall = s_rel.max().nlargest(10)

    curt_t = get_curtailment_data(edisgo)   # time × 14a-generators only

    # Aggregate §14a curtailment per timestep
    ts_14a_total = curt_t.sum(axis=1)
    threshold_mw = threshold_kw / 1000.0

    active_ts = ts_14a_total[ts_14a_total > threshold_mw].index

    def _top_lines(ts, n=5):
        if ts not in s_rel.index:
            return []
        row = s_rel.loc[ts]
        over = row[row > 1.0].nlargest(n)
        candidates = over if not over.empty else row.nlargest(n)
        return [f"{ln}={v:.3f}x" for ln, v in candidates.items()]

    def _top_gens(ts, n=3):
        if ts not in curt_t.index:
            return []
        top = curt_t.loc[ts].nlargest(n)
        return [f"{g}={v:.4f} MW" for g, v in top.items()]

    rows = []
    for ts in active_ts:
        rows.append({
            "timestamp":                  ts,
            "total_14a_curtailment_mw":   float(ts_14a_total.loc[ts]),
            "n_lines_overloaded":         int(n_overloaded_ts.get(ts, 0)),
            "top_overloaded_lines":       _top_lines(ts),
            "top_injecting_generators":   _top_gens(ts),
        })

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("total_14a_curtailment_mw", ascending=False).reset_index(drop=True)

    return report, s_rel, top_lines_overall


# ===========================================================================
# Metrics collection (extended with n_lines_overloaded_max)
# ===========================================================================

def collect_metrics(edisgo, cp_target, hp_target):
    """Return a dict with key scalar metrics for this run."""
    gen    = edisgo.topology.generators_df
    gen_t  = edisgo.timeseries.generators_active_power
    gen_14a = gen[gen.index.str.contains("14a")]

    if gen_14a.empty or not set(gen_14a.index).issubset(gen_t.columns):
        total_14a_mwh  = 0.0
        ts_14a         = pd.Series(0.0, index=edisgo.timeseries.timeindex)
        ts_14a_per_gen = pd.DataFrame(index=edisgo.timeseries.timeindex)
    else:
        gen_t_14a      = gen_t.loc[:, gen_14a.index]
        total_14a_mwh  = float(gen_t_14a.sum().sum())
        ts_14a         = gen_t_14a.sum(axis=1)
        ts_14a_per_gen = gen_t_14a

    slacks    = edisgo.opf_results.grid_slacks_t
    THRESHOLD = 1e-5
    load_shed_mwh = float(slacks.load_shedding[slacks.load_shedding > THRESHOLD].clip(lower=0).sum().sum())
    hp_shed_mwh   = float(slacks.hp_load_shedding[slacks.hp_load_shedding > THRESHOLD].clip(lower=0).sum().sum())
    cp_shed_mwh   = float(slacks.cp_load_shedding[slacks.cp_load_shedding > THRESHOLD].clip(lower=0).sum().sum())

    v     = edisgo.results.v_res
    s     = edisgo.results.s_res
    s_rel = _build_relative_line_loading(edisgo)

    n_cp_actual = int((edisgo.topology.loads_df["type"] == "charging_point").sum())
    n_hp_actual = int((edisgo.topology.loads_df["type"] == "heat_pump").sum())

    s_nom  = pd.concat([
        edisgo.topology.lines_df["s_nom"],
        edisgo.topology.transformers_df["s_nom"],
    ])
    common = s.columns.intersection(s_nom.index)
    max_rel_line_loading = (
        float((s[common] / s_nom[common]).max().max()) if not s.empty and len(common) > 0
        else float("nan")
    )

    return {
        "cp_target":              cp_target,
        "hp_target":              hp_target,
        "cp_actual":              n_cp_actual,
        "hp_actual":              n_hp_actual,
        "total_14a_mwh":          total_14a_mwh,
        "load_shed_mwh":          load_shed_mwh,
        "hp_shed_mwh":            hp_shed_mwh,
        "cp_shed_mwh":            cp_shed_mwh,
        "min_voltage_pu":         float(v.min().min()) if not v.empty else float("nan"),
        "max_voltage_pu":         float(v.max().max()) if not v.empty else float("nan"),
        "max_line_loading":       float(s.max().max()) if not s.empty else float("nan"),
        "max_rel_line_loading":   max_rel_line_loading,
        "n_lines_overloaded_max": int((s_rel > 1.0).sum(axis=1).max()) if not s_rel.empty else 0,
        "_ts_14a":                ts_14a,
        "_ts_14a_per_gen":        ts_14a_per_gen,
    }


# ===========================================================================
# Per-run saving (extended)
# ===========================================================================

def save_run_results(edisgo, metrics, run_dir, pre_opt_line_loading=None):
    """Persist CSV outputs for one sweep step."""
    os.makedirs(run_dir, exist_ok=True)

    # Scalar summary
    scalar = {k: v for k, v in metrics.items() if not k.startswith("_")}
    pd.Series(scalar).to_csv(os.path.join(run_dir, "metrics.csv"), header=["value"])

    # §14a time series (aggregated + per generator)
    metrics["_ts_14a"].to_csv(os.path.join(run_dir, "ts_14a_mw.csv"), header=["14a_mw"])
    if not metrics["_ts_14a_per_gen"].empty:
        metrics["_ts_14a_per_gen"].to_csv(os.path.join(run_dir, "ts_14a_per_gen_mw.csv"))

    # Voltage + absolute line loading
    if not edisgo.results.v_res.empty:
        edisgo.results.v_res.to_csv(os.path.join(run_dir, "v_res.csv"))
    if not edisgo.results.s_res.empty:
        edisgo.results.s_res.to_csv(os.path.join(run_dir, "s_res.csv"))

    # Relative line loading (NEW)
    s_rel = _build_relative_line_loading(edisgo)
    if not s_rel.empty:
        s_rel.to_csv(os.path.join(run_dir, "s_res_relative.csv"))
        s_rel.max().to_csv(
            os.path.join(run_dir, "s_res_relative_max_per_branch.csv"),
            header=["max_rel_loading"],
        )

    # Load-shedding time series (NEW)
    save_loadshedding_timeseries(edisgo, run_dir)

    # Extended §14a activation report (NEW)
    if pre_opt_line_loading is not None:
        activation_report, _, top_lines_overall = create_extended_activation_report(
            edisgo, pre_opt_line_loading, threshold_kw=0.5
        )
        if not activation_report.empty:
            activation_report.to_csv(
                os.path.join(run_dir, "14a_activation_report.csv"), index=False
            )
        if not top_lines_overall.empty:
            top_lines_overall.to_csv(
                os.path.join(run_dir, "top_lines_max_rel_loading.csv"),
                header=["max_rel_loading"],
            )

    logger.info("Run results saved to %s", run_dir)


# ===========================================================================
# Focus-bus helper (updated: top-3, filter by peak > 0.5 kW)
# ===========================================================================

def _top_14a_buses(edisgo, n=3, min_max_kw=0.5):
    """
    Return up to n buses with highest cumulative §14a curtailment, restricted
    to buses where peak curtailment exceeds min_max_kw at least once.
    """
    curt = get_curtailment_data(edisgo).T
    curt["load"] = curt.index
    curt["load"] = curt["load"].apply(
        lambda x: x.replace("cp_14a_support_", "").replace("hp_14a_support_", "")
    )
    curt["bus"] = curt["load"].map(edisgo.topology.loads_df["bus"])
    totals = curt.drop(columns=["load", "bus"]).groupby(curt["bus"]).sum().sum(axis=1)
    peak   = curt.drop(columns=["load", "bus"]).groupby(curt["bus"]).sum().max(axis=1)
    valid  = peak[peak > min_max_kw / 1000.0].index
    return totals.loc[totals.index.isin(valid)].nlargest(n).index.tolist()


# ===========================================================================
# Plots (updated: n=3 focus buses, threshold filter)
# ===========================================================================

def produce_run_plots(edisgo, run_dir):
    """Create per-timestep network plots, a GIF, and §14a day plots."""
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Network snapshot plots + GIF
    plot_network(edisgo, show=False, snapshots=edisgo.timeseries.timeindex, output_folder=plots_dir)

    gif_path = os.path.join(plots_dir, "network_evolution.gif")
    try:
        create_network_gif(folder_path=plots_dir, output_name=gif_path, duration=500)
        logger.info("Network GIF saved to %s", gif_path)
    except Exception as exc:
        logger.warning("Network GIF creation failed (no frames?): %s", exc)

    # §14a overview plot for the full simulation period
    overview_path = os.path.join(plots_dir, "14a_overview_full_period.png")
    plot_14a_overview_full_period(edisgo, output_path=overview_path, show=False, save=True)
    logger.info("Overview plot saved to %s", overview_path)

    # Focus GIFs + analysis for top-3 §14a buses where peak curtailment > 0.5 kW (updated)
    top_buses = _top_14a_buses(edisgo, n=3, min_max_kw=0.5)
    logger.info("Top §14a buses for focus plots: %s", top_buses)

    for bus in top_buses:
        safe_name = bus.replace("/", "_").replace(" ", "_")
        focus_dir = os.path.join(plots_dir, f"focus_{safe_name}")
        os.makedirs(focus_dir, exist_ok=True)
        plot_network(edisgo, show=False, snapshots=edisgo.timeseries.timeindex, focus_bus=bus, output_folder=focus_dir)
        focus_gif = os.path.join(focus_dir, "network_evolution.gif")
        try:
            create_network_gif(folder_path=focus_dir, output_name=focus_gif, duration=500)
            logger.info("Focus GIF saved to %s", focus_gif)
        except Exception as exc:
            logger.warning("Focus GIF creation failed for bus %s (no frames?): %s", bus, exc)
        focus_analysis_path = os.path.join(focus_dir, "14a_focus_analysis.png")
        plot_14a_focus_bus(edisgo, bus=bus, output_path=focus_analysis_path, show=False, save=True)
        logger.info("Focus analysis plot saved to %s", focus_analysis_path)


# ===========================================================================
# Combined summary plot (unchanged)
# ===========================================================================

def plot_sweep_summary(all_metrics, out_dir):
    """
    Two separate figures for the penetration sweep.

    Figure 1 – Bar chart: total §14a energy (MWh) per scenario.
    Figure 2 – Bar chart: peak §14a power (MW) per scenario.
    Figure 3 – Line chart: §14a power averaged by time-of-day per scenario.
    """
    import matplotlib.cm as cm

    n_runs = len(all_metrics)
    labels     = [f"CP={m['cp_target']} / HP={m['hp_target']}" for m in all_metrics]
    totals_mwh = [m["total_14a_mwh"] for m in all_metrics]
    totals_mw  = [m["_ts_14a"].max() for m in all_metrics]

    cmap   = cm.get_cmap("YlOrRd", n_runs)
    colors = [cmap(i / max(n_runs - 1, 1)) for i in range(n_runs)]

    def _bar_plot(values, ylabel, title, filename, unit):
        fig, ax = plt.subplots(figsize=(max(8, n_runs * 1.8), 5))
        bars = ax.bar(range(n_runs), values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(n_runs))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.2f} {unit}",
                ha="center", va="bottom", fontsize=8,
            )
        fig.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    path_mwh = _bar_plot(
        totals_mwh,
        ylabel   = "§14a Gesamtenergie (MWh)",
        title    = "§14a Abgeregelte Energie pro Szenario",
        filename = "sweep_bar_energy.png",
        unit     = "MWh",
    )

    _bar_plot(
        totals_mw,
        ylabel   = "§14a Spitzenleistung (MW)",
        title    = "§14a Max. Abregelungsleistung pro Szenario",
        filename = "sweep_bar_power.png",
        unit     = "MW",
    )

    # Time-of-day profile
    fig3, ax_line = plt.subplots(figsize=(12, 5))
    for m, label, color in zip(all_metrics, labels, colors):
        ts  = m["_ts_14a"]
        tod = ts.groupby(ts.index.time).mean()
        x   = [t.hour + t.minute / 60 for t in tod.index]
        ax_line.plot(x, tod.values, color=color, linewidth=1.8, label=label, alpha=0.85)

    ts_last  = all_metrics[-1]["_ts_14a"]
    tod_last = ts_last.groupby(ts_last.index.time).mean()
    x_last   = [t.hour + t.minute / 60 for t in tod_last.index]
    ax_line.fill_between(x_last, tod_last.values, alpha=0.08, color=colors[-1])

    ax_line.set_xlim(0, 24)
    ax_line.set_xticks(range(0, 25, 2))
    ax_line.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=9)
    ax_line.set_xlabel("Uhrzeit", fontsize=10)
    ax_line.set_ylabel("Mittlere §14a-Leistung (MW)", fontsize=10)
    ax_line.set_title(
        "§14a Tageszeitprofil – Mittlere Abregelungsleistung nach Uhrzeit",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_line.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_line.set_axisbelow(True)
    ax_line.legend(fontsize=8, loc="upper left", framealpha=0.85)

    fig3.tight_layout()
    path_tod = os.path.join(out_dir, "sweep_timeofday.png")
    fig3.savefig(path_tod, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    logger.info("Saved: %s", path_tod)

    return path_mwh


# ===========================================================================
# Work-CP line reinforcement fix
# ===========================================================================

def reinforce_work_cp_lines(edisgo, s_nom_target=0.19):
    """
    Strengthen all hc_line segments reachable from buses with 'work' charging points.

    The hc_line connecting a work-CP to the main feeder can be split into
    multiple segments. This function does a BFS from each work-CP bus,
    traversing only edges with comp_type == 'hc_line', and reinforces every
    segment found along the way.

    Steps:
      1. Find all loads whose index contains "work" → their buses.
      2. BFS from each such bus, following hc_line edges only.
      3. Set s_nom of every collected hc_line segment to s_nom_target.
    """
    loads = edisgo.topology.loads_df
    work_cp_buses = set(loads.loc[loads.index.str.contains("work", case=False), "bus"].dropna())

    if not work_cp_buses:
        logger.info("[work-CP fix] No 'work' charging point buses found – skipping.")
        return 0

    lines = edisgo.topology.lines_df
    if "comp_type" not in lines.columns:
        logger.warning("[work-CP fix] 'comp_type' column missing in lines_df – skipping.")
        return 0

    hc_lines = lines[lines["comp_type"] == "hc_line"]

    lines_to_reinforce = set()
    for start_bus in work_cp_buses:
        visited = {start_bus}
        queue = [start_bus]
        while queue:
            bus = queue.pop(0)
            connected = hc_lines[(hc_lines["bus0"] == bus) | (hc_lines["bus1"] == bus)]
            for line_id, row in connected.iterrows():
                lines_to_reinforce.add(line_id)
                other = row["bus1"] if row["bus0"] == bus else row["bus0"]
                if other not in visited:
                    visited.add(other)
                    queue.append(other)

    if not lines_to_reinforce:
        logger.info(
            "[work-CP fix] %d work-CP bus(es) found but no hc_lines reachable – skipping.",
            len(work_cp_buses),
        )
        return 0

    edisgo.topology.lines_df.loc[list(lines_to_reinforce), "s_nom"] = s_nom_target
    logger.info(
        "[work-CP fix] Reinforced %d hc_line segment(s) reachable from %d work-CP bus(es) "
        "to s_nom=%.3f MVA.",
        len(lines_to_reinforce), len(work_cp_buses), s_nom_target,
    )
    return len(lines_to_reinforce)


# ===========================================================================
# Main sweep loop (extended)
# ===========================================================================

def main():
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    all_metrics = []

    for step_idx, (cp_target, hp_target) in enumerate(SWEEP_STEPS):
        run_label = f"step{step_idx:02d}_CP{cp_target}_HP{hp_target}"
        run_dir   = os.path.join(BASE_OUT_DIR, run_label)

        logger.info("=" * 70)
        logger.info(
            "SWEEP STEP %d/%d  –  CP=%d  HP=%d",
            step_idx + 1, len(SWEEP_STEPS), cp_target, hp_target,
        )
        logger.info("=" * 70)

        # ── Load fresh grid ─────────────────────────────────────────────────
        edisgo = EDisGo(pypsa_csv_dir=GRID_PATH, snapshot_range=SNAPSHOT_RANGE)
        
        # Set HV/MV transformer secondary voltage to 1.025 pu to reflect the common
        # DSO practice of boosting LV bus voltage to compensate for feeder voltage drops.
        edisgo.config["grid_expansion_allowed_voltage_deviations"]["hv_mv_trafo_offset"] = 0.025
        
        mv_grid_geom = gpd.read_file(SHP_PATH).to_crs(4326)
        edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
        edisgo.topology.grid_district["srid"] = 4326   

        edisgo.topology.check_integrity()
        edisgo.to_pypsa()
        edisgo.analyze()

        # ── Prepare (with this step's CP/HP targets) ─────────────────────
        prepare_edisgo(
            edisgo,
            shapefile_path=SHP_PATH,
            output_dir=run_dir,
            cp_target=cp_target,
            hp_target=hp_target,
            cache_dir=EMOB_CACHE_DIR,
            setup_days=None,
        )
        
        # ── Remove problematic CTS loads (hardcoded) ─────────────────────
        _LOADS_TO_REMOVE = [
            "CTS_Load_185_bus_20762",
            "CTS_Load_236_bus_21403",
            "CTS_Load_87_bus_27868",
            "CTS_Load_184_bus_23225",
            "CTS_Load_3_bus_24359",
            "CTS_Load_303_bus_26637",
        ]
        for _load in _LOADS_TO_REMOVE:
            edisgo.topology.remove_load(_load)
            lap = edisgo.timeseries.loads_active_power
            if lap is not None and _load in lap.columns:
                edisgo.timeseries.loads_active_power = lap.drop(columns=[_load])
            logger.info("Removed load: %s", _load)
            
        # ── Reinforce hc_lines at work charging-point buses ──────────────
        reinforce_work_cp_lines(edisgo, s_nom_target=0.19)

        edisgo.analyze()
        pre_opt_line_loading = edisgo.results.s_res.copy()

        # ── Save pre-OPF power flow results ──────────────────────────────
        os.makedirs(run_dir, exist_ok=True)
        if not edisgo.results.s_res.empty:
            edisgo.results.s_res.to_csv(os.path.join(run_dir, "pre_opt_s_res.csv"))
            s_rel_pre = _build_relative_line_loading(edisgo)
            s_rel_pre.to_csv(os.path.join(run_dir, "pre_opt_s_res_relative.csv"))
            s_rel_pre.max().to_csv(
                os.path.join(run_dir, "pre_opt_s_res_relative_max_per_branch.csv"),
                header=["max_rel_loading"],
            )
            logger.info("Saved pre-OPF s_res and s_res_relative to %s", run_dir)
        if not edisgo.results.v_res.empty:
            edisgo.results.v_res.to_csv(os.path.join(run_dir, "pre_opt_v_res.csv"))
            logger.info("Saved pre-OPF v_res to %s", run_dir)

        # ── Reinforce BEFORE §14a optimisation (NEW) ─────────────────────
        costs_before_14a = measure_reinforce_costs(edisgo, label="before_14a")

        # ── Optimize ─────────────────────────────────────────────────────
        edisgo = run_optimization_14a(edisgo)
        edisgo.analyze()

        # ── Reinforce AFTER §14a optimisation (NEW) ──────────────────────
        costs_after_14a = measure_reinforce_costs(edisgo, label="after_14a")

        # ── Collect & save ────────────────────────────────────────────────
        metrics = collect_metrics(edisgo, cp_target, hp_target)
        save_run_results(edisgo, metrics, run_dir, pre_opt_line_loading=pre_opt_line_loading)
        save_reinforce_costs(costs_before_14a, costs_after_14a, run_dir)
        produce_run_plots(edisgo, run_dir)
        all_metrics.append(metrics)

        # ── Save topology + timeseries as CSV (NEW) ──────────────────────
        edisgo_csv_dir = os.path.join(run_dir, "edisgo_csv")
        logger.info("Saving topology + timeseries to CSV: %s", edisgo_csv_dir)
        try:
            edisgo.topology.to_csv(edisgo_csv_dir)
            logger.info(
                "Topology saved (loads, generators, lines, buses, transformers, …)"
            )
        except Exception as exc:
            logger.warning("topology.to_csv() failed: %s", exc)
        try:
            edisgo.timeseries.to_csv(edisgo_csv_dir)
            logger.info(
                "Timeseries saved (loads_active_power, generators_active_power, "
                "loads_reactive_power, generators_reactive_power, storage_units, …)"
            )
        except Exception as exc:
            logger.warning("timeseries.to_csv() failed: %s", exc)

        logger.info(
            "  §14a total: %.4f MWh | Load shed: %.4f MWh | Min V: %.4f p.u.",
            metrics["total_14a_mwh"], metrics["load_shed_mwh"], metrics["min_voltage_pu"],
        )

    # ── Save scalar summary CSV ───────────────────────────────────────────
    summary_rows = [{k: v for k, v in m.items() if not k.startswith("_")} for m in all_metrics]
    summary_df   = pd.DataFrame(summary_rows)
    summary_csv  = os.path.join(BASE_OUT_DIR, "sweep_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Summary CSV saved to %s", summary_csv)
    logger.info("\n%s", summary_df.to_string(index=False))

    # ── Combined plot ──────────────────────────────────────────────────────
    plot_path = plot_sweep_summary(all_metrics, BASE_OUT_DIR)

    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE  –  %d steps finished", len(all_metrics))
    logger.info("Results  : %s", BASE_OUT_DIR)
    logger.info("Summary  : %s", summary_csv)
    logger.info("Plot     : %s", plot_path)
    logger.info("=" * 70)

    return summary_df, all_metrics


#if __name__ == "__main__":
    #summary_df, all_metrics = main()
