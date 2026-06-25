import os
from datetime import datetime
from venv import logger

import geopandas as gpd
import pandas as pd

from edisgo import EDisGo
from edisgo.flex_opt.check_tech_constraints import lines_relative_load
from edisgo.tools.loma_tools import (
    buses_with_existing_loads,
    get_curtailment_data,
    plot_cp_hp_locations,
    plot_load_before_after,
    set_charging_points_to_target,
    set_heat_pumps_to_target,
    set_storage_timeseries_bus_level,
    transfer_ts_from_new_to_existing_cp,
)

# Define global variables
emob_cache_dir = "/home/student/Execution/eDisGo_exe/emob_cache_Whole_Husum_final_statusQuo"
grid_path = "/home/student/Execution/LoMa_exe/results/Whole_Husum_final_statusQuo_LV_ids"
path_husum_district_shp = "/home/student/Execution/LoMa_exe/data/Input_files/MV_grid_district/husum_district.shp"
BASE_OUT_DIR = "/home/carlos/LoMa/output_edisgo"

# ===========================================================================
# SWEEP CONFIGURATION – adapt here. Each (cp_target, hp_target) pair is run
# as an independent daily-chunk week, in its own sub-directory of
# BASE_OUT_DIR.
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
    start_time = datetime.now()

    edisgo.pm_optimize(opf_version=5, curtailment_14a=True, hours_limit_14a=24)

    duration = (datetime.now() - start_time).total_seconds()
    print(f"  ✓ Optimization complete ({duration:.1f}s)")

    return edisgo


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

    emob.charging_processes_df.to_pickle(
        os.path.join(cache_dir, "charging_processes_df.pkl")
    )
    emob.potential_charging_parks_gdf.to_pickle(
        os.path.join(cache_dir, "potential_charging_parks_gdf.pkl")
    )
    emob.simbev_config_df.to_pickle(
        os.path.join(cache_dir, "simbev_config_df.pkl")
    )
    emob.integrated_charging_parks_df.to_pickle(
        os.path.join(cache_dir, "integrated_charging_parks_df.pkl")
    )

    edisgo_cp_ids = set(emob.integrated_charging_parks_df["edisgo_id"].dropna())
    new_cp_rows = edisgo.topology.loads_df[
        edisgo.topology.loads_df.index.isin(edisgo_cp_ids)
    ]
    new_cp_rows.to_pickle(os.path.join(cache_dir, "new_cp_loads_df.pkl"))

    print(f"[emob cache] Saved to {cache_dir}")


def _load_emob_cache(edisgo, cache_dir):
    emob = edisgo.electromobility

    emob.charging_processes_df = pd.read_pickle(
        os.path.join(cache_dir, "charging_processes_df.pkl")
    )
    emob.potential_charging_parks_gdf = pd.read_pickle(
        os.path.join(cache_dir, "potential_charging_parks_gdf.pkl")
    )
    emob.simbev_config_df = pd.read_pickle(
        os.path.join(cache_dir, "simbev_config_df.pkl")
    )
    emob.integrated_charging_parks_df = pd.read_pickle(
        os.path.join(cache_dir, "integrated_charging_parks_df.pkl")
    )

    new_cp_rows = pd.read_pickle(os.path.join(cache_dir, "new_cp_loads_df.pkl"))
    missing = new_cp_rows.index.difference(edisgo.topology.loads_df.index)
    if len(missing) > 0:
        edisgo.topology.loads_df = pd.concat(
            [edisgo.topology.loads_df, new_cp_rows.loc[missing]]
        )

    print(f"[emob cache] Loaded from {cache_dir} ({len(new_cp_rows)} eDisGo CPs restored)")


def integrate_ev_and_hp_for_14a(
    edisgo, *, shapefile_path, output_dir, cp_target=50, hp_target=130,
    setup_days=None, cache_dir=None, seed=42,
):
    """Import EV charging points, apply charging strategy, and adjust CP/HP counts."""
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
        print(
            f"[integrate_ev_and_hp] setup_days={setup_days}: "
            f"using {_capped} of {_orig_days} simulated days for charging strategy"
        )

    # Prepare Q before charging strategy
    ti = edisgo.timeseries.timeindex
    lap_cols = edisgo.timeseries.loads_active_power.columns
    edisgo.timeseries.loads_reactive_power = pd.DataFrame(
        0.0,
        index=ti,
        columns=lap_cols,
    )

    edisgo.apply_charging_strategy(strategy="dumb")

    # Apply EV charging efficiency correction
    ev_charging_efficiency = 0.9

    lap = edisgo.timeseries.loads_active_power.copy()

    cp_names_before_transfer = edisgo.topology.loads_df.query(
        "type == 'charging_point'"
    ).index.intersection(lap.columns)

    lap.loc[:, cp_names_before_transfer] = (
        lap.loc[:, cp_names_before_transfer] * ev_charging_efficiency
    )
    edisgo.timeseries.loads_active_power = lap

    if _orig_days is not None:
        edisgo.electromobility.simbev_config_df.at[0, "days"] = _orig_days

    transfer_ts_from_new_to_existing_cp(
        edisgo,
        existing_markers=("Existing", "Additional"),
        radius_1=2000.0,
        tol_1=0.15,
        radius_2=2000.0,
        tol_2=0.9,
    )

    valid_buses = set(edisgo.topology.buses_df.index)

    base_eligible_buses = [
        bus for bus in buses_with_existing_loads(edisgo)
        if bus in valid_buses
    ]

    cp_eligible_buses = base_eligible_buses.copy()
    hp_eligible_buses = base_eligible_buses.copy()

    set_charging_points_to_target(
        edisgo,
        target_total=cp_target,
        eligible_buses=cp_eligible_buses,
        removal_priority=["Additional", "Existing"],
        add_tracking_columns=False,
        export_removed=False,
        export_dir=output_dir,
        seed=seed,
    )

    set_heat_pumps_to_target(
        edisgo,
        target_total=hp_target,
        eligible_buses=hp_eligible_buses,
        add_tracking_columns=False,
        export_removed=False,
        export_dir=output_dir,
        seed=seed,
    )


def prepare_edisgo_for_14a(
    edisgo, *, shapefile_path, output_dir, cp_target=50, hp_target=130,
    cache_dir=None, setup_days=None, seed=42,
):
    """Apply topology fixes, EV integration, and pre-optimization setup.

    Run ONCE for the whole loaded period (e.g. a full week). The resulting
    object is later split into daily chunks for optimization, see
    `restrict_edisgo_to_window`.
    """
    edisgo.topology.generators_df = edisgo.topology.generators_df[
        edisgo.topology.generators_df.index != "HV_dummy_gen_slack"
    ]
    edisgo.topology.buses_df = edisgo.topology.buses_df[
        edisgo.topology.buses_df.v_nom <= 20
    ]

    dummy_buses = ["HV_dummy_bus", "bus_20111_HV"]
    loads_dummy_buses = edisgo.topology.loads_df[
        edisgo.topology.loads_df.bus.isin(dummy_buses)
    ]

    edisgo.topology.loads_df = edisgo.topology.loads_df.drop(
        loads_dummy_buses.index
    )

    edisgo.timeseries.loads_active_power = (
        edisgo.timeseries.loads_active_power.drop(
            columns=loads_dummy_buses, errors="ignore"
        )
    )
    edisgo.timeseries.loads_reactive_power = (
        edisgo.timeseries.loads_reactive_power.drop(
            columns=loads_dummy_buses, errors="ignore"
        )
    )
    edisgo.topology.buses_df = edisgo.topology.buses_df.drop(
        dummy_buses, errors="ignore"
    )

    ref = edisgo.topology.lines_df[
        edisgo.topology.lines_df["type_info"] == "NAYY 4x95"
    ].iloc[0]
    r_per_km = ref["r"] / ref["length"]
    x_per_km = ref["x"] / ref["length"]
    b_per_km = ref["b"] / ref["length"]
    hc_mask = edisgo.topology.lines_df["comp_type"] == "hc_line"
    hc_lengths = edisgo.topology.lines_df.loc[hc_mask, "length"]
    edisgo.topology.lines_df.loc[hc_mask, "r"] = r_per_km * hc_lengths
    edisgo.topology.lines_df.loc[hc_mask, "x"] = x_per_km * hc_lengths
    edisgo.topology.lines_df.loc[hc_mask, "b"] = b_per_km * hc_lengths
    edisgo.topology.lines_df.loc[hc_mask, "s_nom"] = ref["s_nom"]
    edisgo.topology.lines_df.loc[hc_mask, "type_info"] = ref["type_info"]

    integrate_ev_and_hp_for_14a(
        edisgo,
        shapefile_path=shapefile_path,
        output_dir=output_dir,
        cp_target=cp_target,
        hp_target=hp_target,
        cache_dir=cache_dir,
        setup_days=setup_days,
        seed=seed,
    )

    set_storage_timeseries_bus_level(edisgo)

    hp_names = list(
        edisgo.topology.loads_df[edisgo.topology.loads_df["type"] == "heat_pump"].index
    )
    timeindex = edisgo.timeseries.timeindex
    cop = 3.0  # flat synthetic COP
    edisgo.heat_pump.cop_df = pd.DataFrame(
        cop,
        index=timeindex,
        columns=hp_names,
    )
    edisgo.heat_pump.heat_demand_df = (
        edisgo.timeseries.loads_active_power[hp_names] * cop
    )

    edisgo.set_time_series_reactive_power_control()


def get_daily_windows(timeindex):
    """Split a DatetimeIndex into daily (calendar day) chunks.

    Returns
    -------
    list of (str, pandas.DatetimeIndex)
        One entry per day, labelled "YYYY-MM-DD", in chronological order.
    """
    days = timeindex.normalize().unique().sort_values()
    return [
        (day.strftime("%Y-%m-%d"), timeindex[timeindex.normalize() == day])
        for day in days
    ]


def restrict_edisgo_to_window(edisgo_obj, window_index):
    """Restrict an already-loaded EDisGo object to a subset of timesteps.

    Setting `timeseries.timeindex` alone is NOT enough: properties like
    `generators_active_power` are filtered through it automatically when
    READ (TimeSeries._internal_getter does `.loc[self.timeindex]`), but the
    underlying private DataFrames (`_generators_active_power`,
    `_loads_active_power`, `_storage_units_active_power`, ...) still hold all
    of the original (e.g. weekly) rows. `from_powermodels` writes OPF results
    back into exactly those private DataFrames with `.loc[:, names] = ...`
    (no row filtering), so if they still have e.g. 168 rows while the OPF
    result array has only 24, this raises
    "ValueError: shape mismatch: value array of shape (24, n) could not be
    broadcast to indexing result of shape (168, n)".

    Reading each property through its getter (already window-filtered) and
    writing it back through its setter (plain full replace, see
    `edisgo/network/timeseries.py`) shrinks the private DataFrame itself to
    just this window.

    `heat_pump.cop_df`/`heat_pump.heat_demand_df` are read directly by their
    own row index inside `to_powermodels` (no `.loc[timeindex]` filtering
    there either), so they are trimmed explicitly as well.
    """
    ts = edisgo_obj.timeseries
    ts.timeindex = window_index

    for attr in [
        "generators_active_power",
        "generators_reactive_power",
        "loads_active_power",
        "loads_reactive_power",
        "storage_units_active_power",
        "storage_units_reactive_power",
    ]:
        setattr(ts, attr, getattr(ts, attr))

    edisgo_obj.heat_pump.cop_df = edisgo_obj.heat_pump.cop_df.loc[window_index]
    edisgo_obj.heat_pump.heat_demand_df = edisgo_obj.heat_pump.heat_demand_df.loc[
        window_index
    ]
    return edisgo_obj


def run_day_chunk(edisgo_full, day_label, window_index):
    """Run analyze + §14a OPF + analyze for a single day on an isolated copy.

    `edisgo_full` (already loaded and prepared for the whole week) is left
    untouched; a deep copy is restricted to `window_index` and optimized.
    """
    print(f"\n{'='*80}\nDay {day_label} ({len(window_index)} timesteps)\n{'='*80}")

    edisgo_day = edisgo_full.copy(deep=True)
    restrict_edisgo_to_window(edisgo_day, window_index)

    edisgo_day.analyze()
    edisgo_day = run_optimization_14a(edisgo_day)
    edisgo_day.analyze()

    return edisgo_day


def get_curtailment_14a_summary(edisgo):
    """
    Return hourly §14a curtailment split by HP and CP.

    Returns
    -------
    pd.DataFrame
        Columns: hp_curtailment_mw, cp_curtailment_mw; index is timeindex.
    """
    curt = get_curtailment_data(edisgo)
    hp_cols = [c for c in curt.columns if "hp_14a_support" in c]
    cp_cols = [c for c in curt.columns if "cp_14a_support" in c or "charging_point_14a_support" in c]
    return pd.DataFrame(
        {
            "hp_curtailment_mw": curt[hp_cols].sum(axis=1) if hp_cols else 0.0,
            "cp_curtailment_mw": curt[cp_cols].sum(axis=1) if cp_cols else 0.0,
        },
        index=edisgo.timeseries.timeindex,
    )


def main(output_dir, snapshot_range, seed=42, cp_target=50, hp_target=130):
    """Load an EDisGo object once for the given period, then optimize it day
    by day in independent 24h chunks and merge the results back together.

    Parameters
    ----------
    output_dir : str
    snapshot_range : tuple(int, int)
        Inclusive (start, end) row range to load, e.g. (0, 7*24-1) for the
        first week of the year. Should span whole calendar days.
    seed : int
    cp_target : int
        Target total number of charging points to scale to.
    hp_target : int
        Target total number of heat pumps to scale to.
    """
    t0 = datetime.now()
    print(f"=== main() started at {t0:%Y-%m-%d %H:%M:%S} ===")

    edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=snapshot_range)

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

    # Set HV/MV transformer secondary voltage to 1.025 pu to reflect the common
    # DSO practice of boosting LV bus voltage to compensate for feeder voltage drops.
    edisgo.config["grid_expansion_allowed_voltage_deviations"]["hv_mv_trafo_offset"] = 0.05

    mv_grid_geom = gpd.read_file(path_husum_district_shp).to_crs(4326)
    edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
    edisgo.topology.grid_district["srid"] = 4326

    edisgo.topology.check_integrity()
    edisgo.analyze()

    # Prepare ONCE for the whole period (EV import/charging strategy, HP setup).
    prepare_edisgo_for_14a(
        edisgo,
        shapefile_path=path_husum_district_shp,
        output_dir=output_dir,
        cp_target=cp_target,
        hp_target=hp_target,
        cache_dir=emob_cache_dir,
        setup_days=None,
        seed=seed,
    )

    # ── Save full-week topology + timeseries (after EV/HP/storage setup, ──
    # ── before splitting into daily chunks) ─────────────────────────────
    edisgo_csv_dir = f"{output_dir}/edisgo_csv_week"
    os.makedirs(edisgo_csv_dir, exist_ok=True)
    edisgo.topology.to_csv(edisgo_csv_dir)
    edisgo.timeseries.to_csv(edisgo_csv_dir)
    print(f"Saved full-week topology + timeseries to {edisgo_csv_dir}")

    # ── Pre-optimization power flow (once for the whole week, before §14a) ──
    edisgo.analyze()
    os.makedirs(output_dir, exist_ok=True)
    if not edisgo.results.s_res.empty:
        edisgo.results.s_res.to_csv(f"{output_dir}/pre_opt_s_res.csv")
    if not edisgo.results.v_res.empty:
        edisgo.results.v_res.to_csv(f"{output_dir}/pre_opt_v_res.csv")
    print(f"Saved pre-optimization s_res/v_res (whole week) to {output_dir}")

    # ──────────────────────── Optimize day by day ─────────────────────────
    daily_windows = get_daily_windows(edisgo.timeseries.timeindex)
    print(
        f"\nSplitting {len(edisgo.timeseries.timeindex)} timesteps into "
        f"{len(daily_windows)} daily chunks: {[d for d, _ in daily_windows]}"
    )

    edisgo_days = []
    slack_parts = {"gen_nd_crt": [], "gen_d_crt": [], "load_shedding": [], "hp_load_shedding": []}
    v_res_parts = []
    s_res_parts = []
    gen_p_parts = []
    gen_q_parts = []
    loads_p_parts = []
    line_usage_parts = []
    curtailment_parts = []

    for day_label, window_index in daily_windows:
        edisgo_day = run_day_chunk(edisgo, day_label, window_index)
        edisgo_days.append((day_label, edisgo_day))

        slacks = edisgo_day.opf_results.grid_slacks_t
        slack_parts["gen_nd_crt"].append(slacks.gen_nd_crt)
        slack_parts["gen_d_crt"].append(slacks.gen_d_crt)
        slack_parts["load_shedding"].append(slacks.load_shedding)
        slack_parts["hp_load_shedding"].append(slacks.hp_load_shedding)

        v_res_parts.append(edisgo_day.results.v_res)
        s_res_parts.append(edisgo_day.results.s_res)
        gen_p_parts.append(edisgo_day.timeseries.generators_active_power)
        gen_q_parts.append(edisgo_day.timeseries.generators_reactive_power)
        loads_p_parts.append(edisgo_day.timeseries.loads_active_power)
        line_usage_parts.append(lines_relative_load(edisgo_day) * 100)
        curtailment_parts.append(get_curtailment_14a_summary(edisgo_day))

    # ─────────────────────── Merge daily results into one week ────────────
    slacks_week = {
        name: pd.concat(parts, axis=0).sort_index() for name, parts in slack_parts.items()
    }
    v_res_week = pd.concat(v_res_parts, axis=0).sort_index()
    s_res_week = pd.concat(s_res_parts, axis=0).sort_index()
    generators_active_power_week = pd.concat(gen_p_parts, axis=0).sort_index()
    generators_reactive_power_week = pd.concat(gen_q_parts, axis=0).sort_index()
    loads_active_power_week = pd.concat(loads_p_parts, axis=0).sort_index()
    line_usage_week = pd.concat(line_usage_parts, axis=0).sort_index()
    curtailment_week = pd.concat(curtailment_parts, axis=0).sort_index()

    # ────────────────────────── Slack diagnosis ──────────────────────────
    print("\n=== OPF Slack Diagnosis (v5, merged over week) ===")
    for name, df in slacks_week.items():
        total = df.abs().sum(axis=1)
        if (total > 5e-3).any():
            print(f"  {name}: {total.sum():.4f} MW  ← NON-ZERO")
        else:
            print(f"  {name}: 0 (not used)")

    print("\n=== Voltage after OPF (merged over week) ===")
    print(f"  Min:  {v_res_week.min().min():.4f} p.u.")
    print(f"  Max:  {v_res_week.max().max():.4f} p.u.")
    viol = (v_res_week < 0.9) | (v_res_week > 1.1)
    if viol.any().any():
        print(f"  Violations:{viol.sum().sum()}")
    else:
        print("  No voltage violations.")
    # ────────────────────────── End diagnosis ────────────────────────────

    print("\n=== 14a analysis (merged over week) ===")
    print(f"Total use of 14a: {curtailment_week.sum().sum():.4f}")
    print("\n=== end 14a analysis ===")

    # ── Presentation plots ──────────────────────────────────────────────
    # Select days that have non-trivial §14a curtailment (threshold: 1 kW total)
    curt_daily = curtailment_week.groupby(curtailment_week.index.normalize()).sum().sum(axis=1)
    active_days = curt_daily[curt_daily > 1e-3].index.strftime("%Y-%m-%d").tolist()
    print(f"\n=== Days with §14a curtailment: {active_days} ===")

    for day_label, edisgo_day in edisgo_days:
        if day_label in active_days:
            print(f"  Plotting {day_label}...")
            plot_load_before_after(
                edisgo_day, day=day_label, show=False, save=True,
                folder_path=f"{output_dir}/load_plots/",
            )

    os.makedirs(output_dir, exist_ok=True)
    line_usage_week.to_csv(f"{output_dir}/line_usage.csv")
    curtailment_week.to_csv(f"{output_dir}/curtailment_14a.csv")
    v_res_week.to_csv(f"{output_dir}/v_res.csv")
    s_res_week.to_csv(f"{output_dir}/s_res.csv")
    generators_active_power_week.to_csv(f"{output_dir}/generators_active_power.csv")
    generators_reactive_power_week.to_csv(f"{output_dir}/generators_reactive_power.csv")
    loads_active_power_week.to_csv(f"{output_dir}/loads_active_power.csv")
    for name, df in slacks_week.items():
        df.to_csv(f"{output_dir}/slack_{name}.csv")

    print(f"Saved plots to {output_dir}/load_plots/")
    t_end = datetime.now()
    print(f"=== main() finished at {t_end:%Y-%m-%d %H:%M:%S} (duration: {t_end - t0}) ===")

    return (
        edisgo,
        edisgo_days,
        line_usage_week,
        curtailment_week,
        v_res_week,
        s_res_week,
        generators_active_power_week,
        generators_reactive_power_week,
        loads_active_power_week,
    )


def run_sweep_step(step_idx, cp_target, hp_target, seed, snapshot_range):
    """Run one full daily-chunk week for a given (cp_target, hp_target) pair.

    Mirrors the per-step body of `loma-14a-sweep-extended.py`'s sweep loop,
    but delegates the actual week to `main()` (load + prepare + daily OPF
    chunks + merge).
    """
    run_label = f"step{step_idx:02d}_CP{cp_target}_HP{hp_target}"
    output_dir = f"{BASE_OUT_DIR}/{seed}/{run_label}/week"

    print(f"\n{'='*80}\nSWEEP STEP {step_idx} – CP={cp_target}  HP={hp_target}\n{'='*80}")

    (
        edisgo,
        edisgo_days,
        line_usage_week,
        curtailment_week,
        v_res_week,
        s_res_week,
        generators_active_power_week,
        generators_reactive_power_week,
        loads_active_power_week,
    ) = main(
        output_dir,
        snapshot_range=snapshot_range,
        seed=seed,
        cp_target=cp_target,
        hp_target=hp_target,
    )

    print(f"\n=== §14a Curtailment (seed={seed}, CP={cp_target}, HP={hp_target}) ===")
    print(f"  HP: {curtailment_week['hp_curtailment_mw'].sum():.4f} MWh")
    print(f"  CP: {curtailment_week['cp_curtailment_mw'].sum():.4f} MWh")

    print("\n=== Line loading (%) ===")
    print(line_usage_week.describe())

    plot_cp_hp_locations(edisgo, show=False, save=True, path=output_dir)

    return {
        "step": step_idx,
        "cp_target": cp_target,
        "hp_target": hp_target,
        "hp_curtailment_mwh": curtailment_week["hp_curtailment_mw"].sum(),
        "cp_curtailment_mwh": curtailment_week["cp_curtailment_mw"].sum(),
        "max_line_loading_pct": line_usage_week.max().max(),
        "min_voltage_pu": v_res_week.min().min(),
        "max_voltage_pu": v_res_week.max().max(),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    script_t0 = datetime.now()
    print(f"=== Script started at {script_t0:%Y-%m-%d %H:%M:%S} ===")

    seed = 42
    # One example week, first 7 days of the loaded year (snapshot_range is an
    # inclusive row range and must span whole calendar days).
    snapshot_range = (0, 7 * 24 - 1)

    all_metrics = []
    for step_idx, (cp_target, hp_target) in enumerate(SWEEP_STEPS):
        metrics = run_sweep_step(step_idx, cp_target, hp_target, seed, snapshot_range)
        all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)
    summary_dir = f"{BASE_OUT_DIR}/{seed}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_csv = f"{summary_dir}/sweep_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\n=== Sweep summary saved to {summary_csv} ===")
    print(summary)

    script_t_end = datetime.now()
    print(
        f"=== Script finished at {script_t_end:%Y-%m-%d %H:%M:%S} "
        f"(total duration: {script_t_end - script_t0}) ==="
    )
