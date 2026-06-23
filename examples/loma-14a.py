import calendar
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd

from edisgo import EDisGo
from edisgo.flex_opt.check_tech_constraints import lines_relative_load
from edisgo.tools.loma_tools import (
    buses_with_existing_loads,
    create_network_gif,
    get_curtailment_data,
    plot_cp_hp_locations,
    plot_load_before_after,
    plot_network,
    plot_storage_dispatch,
    set_charging_points_to_target,
    set_heat_pumps_to_target,
    set_storage_timeseries_bus_level,
    transfer_ts_from_new_to_existing_cp,
)

# Define global variables
emob_cache_dir = "/home/carlos/LoMa/emob_cache/husum_eGon2035_MGB"
grid_path = "/home/carlos/LoMa/exec_folder/results/MGB_010626"
path_husum_district_shp = "/home/carlos/LoMa/exec_folder/MGB_district"

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
    edisgo.pm_optimize(opf_version=5, curtailment_14a=True, hours_limit_14a=24)

    duration = (datetime.now() - start_time).total_seconds()

    print("\n✓ Optimization complete!")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

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

def integrate_ev_and_hp_for_14a(edisgo, *, shapefile_path, output_dir, setup_days=None, cache_dir=None, seed=42):
    """Import EV charging points, apply charging strategy, and adjust CP/HP counts."""

    """
    After this function there are no time series yet. Only charging points and
    a overall demand which is then transferred into a time series in
    apply_charging_strategy.

    Note: Afterwards there should be the Existing CP (411) and Additional CP (589)
    from the LoMa side for the 2035 scenario and all new eDisGo CP (for whole Husum
    there should be 2337). So the total should be 3337.
    """
    if cache_dir is not None and _emob_cache_exists(cache_dir):
        _load_emob_cache(edisgo, cache_dir)
    else:
        edisgo.import_electromobility_14a(
            scenario="eGon2035",
            import_electromobility_data_kwds={"shapefile_path": shapefile_path},
        )
        if cache_dir is not None:
            _save_emob_cache(edisgo, cache_dir)

    """
    This step created the time series for the new eDisGo charging points.
    Without the preparation of Q before charging strategy I got an error while
    apply_charging_strategy which was caused by a deviating time index.

    Note: After this step ONLY the charging points from eDisGo have a time series.
    """
    # Optionally limit simulated days so apply_charging_strategy is faster.
    # charging_strategies.py filters events via park_start_timesteps <= len_ts,
    # where len_ts = simulated_days * 24*60 / stepsize, so a shorter day count
    # proportionally reduces both the event set and the dummy_ts array size.
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

    """
    This step then finally transfers the time series from suitable eDisGo
    charging_points to Existing_ und Additional_ charging points which are
    created on the LoMa side.

    Note: After this step there should be 411 Existing CP and 589 Additional for
    the 2035 scenario and 1337 eDisGo CP as 1000 of those were used for matching
    and transferring the time series and deleted afterwards.
    """
    transfer_ts_from_new_to_existing_cp(
        edisgo,
        existing_markers=("Existing", "Additional"),
        radius_1=2000.0,
        tol_1=0.15,
        radius_2=2000.0,
        tol_2=0.9,
    )

    # ------------------------------------------------------------
    # Optional Utilities for sensitivity analysis/changing the amount of cp/hp
    # - target by absolute value or relative percentage
    # - Only use one option at a time (target_total, percentage)
    # ------------------------------------------------------------
    """
    In this step the total amount of charging points or heat pumps can be adjusted.
    Either by percentage or by a total amount including the infrastructure from
    LoMa. When deleting CP/HP there is an option to export the deleted ones.
    New CP/HP will have 'dup' in their name.

    Note: for the 2035 scenario the target total would need to be set to 1000.
    CPs with the marker Additional and Existing in their name will be removed last.
    This way only the remaining 1337 eDisGo CP would be deleted.
    """
    # Compute CP eligible buses for CP scaling
    valid_buses = set(edisgo.topology.buses_df.index)

    base_eligible_buses = [
        bus for bus in buses_with_existing_loads(edisgo)
        if bus in valid_buses
    ]
    
    cp_eligible_buses = base_eligible_buses.copy()
    hp_eligible_buses = base_eligible_buses.copy()
    
    set_charging_points_to_target(
        edisgo,
        target_total=50, # sets total amount of CP #412 SQ, 1000 2035
        # percentage=0.10, # increases total amount of CP by 10%
        # percentage=-0.10, # decreases total amount of CP by 10%
        eligible_buses=cp_eligible_buses,
        removal_priority=["Additional", "Existing"],
        add_tracking_columns=False,
        export_removed=False,
        export_dir=output_dir,
        seed=seed,
        max_p_set_mw=0.1,
    )

    set_heat_pumps_to_target(
        edisgo,
        target_total=130,  # sets total amount of HP #575 SQ, 2600 2035
        # percentage=0.10, # increases total amount of HP by 10%
        # percentage=-0.10, # decreases total amount of HP by 10%
        eligible_buses=hp_eligible_buses,
        add_tracking_columns=False,
        export_removed=False,  # only applies when there are deleted HP
        export_dir=output_dir,  # only applies when there are deleted HP
        seed=seed,
        max_p_set_mw=0.1,
    )


def prepare_edisgo_for_14a(edisgo, *, shapefile_path, output_dir, cache_dir=None, setup_days=None, seed=42):
    """Apply topology fixes, EV integration, and pre-optimization setup."""

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

    sector_loads = edisgo.topology.loads_df[
        edisgo.topology.loads_df["sector"].isin(["industrial", "cts"])
    ]
    edisgo.topology.loads_df = edisgo.topology.loads_df.drop(sector_loads.index)
    edisgo.timeseries.loads_active_power = (
        edisgo.timeseries.loads_active_power.drop(
            columns=sector_loads.index, errors="ignore"
        )
    )
    edisgo.timeseries.loads_reactive_power = (
        edisgo.timeseries.loads_reactive_power.drop(
            columns=sector_loads.index, errors="ignore"
        )
    )
    print(f"[prepare] Removed {len(sector_loads)} industrial/cts loads.")

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


def fix_hp_peak_loads(edisgo, seed=None):
    """
    Enforce minimum HP p_set constraints and ensure adequate representation
    in higher power bands. Timeseries are scaled proportionally when p_set changes.

    Rules:
      1. p_set >= 0.003 MW for every heat pump.
      2. At least 10 % of HPs have p_set in [0.01, 0.02] MW.
      3. At least 10 % of HPs have p_set in [0.02, 0.03] MW.
    """
    rng = np.random.default_rng(seed)
    loads_df = edisgo.topology.loads_df
    hp_idx = loads_df[loads_df["type"] == "heat_pump"].index.tolist()
    n_hp = len(hp_idx)
    if n_hp == 0:
        return

    lap = edisgo.timeseries.loads_active_power

    def _rescale(hp, new_p):
        old_p = loads_df.at[hp, "p_set"]
        if old_p > 0 and hp in lap.columns:
            lap[hp] = lap[hp] * (new_p / old_p)
        loads_df.at[hp, "p_set"] = new_p

    # Rule 1: floor all HPs at 0.003 MW
    for hp in hp_idx:
        if loads_df.at[hp, "p_set"] < 0.003:
            _rescale(hp, 0.003)

    # Rules 2–3: ensure at least 10 % of HPs fall in each target band
    target_bands = [(0.01, 0.02), (0.02, 0.03)]
    min_count = max(1, int(np.ceil(n_hp * 0.10)))

    assigned = set()
    for lo, hi in target_bands:
        already = [hp for hp in hp_idx if lo <= loads_df.at[hp, "p_set"] <= hi]
        assigned |= set(already)
        need = min_count - len(already)
        if need <= 0:
            continue
        # prefer HPs not yet assigned to any target band; fall back to all others
        candidates = [hp for hp in hp_idx if hp not in assigned]
        if len(candidates) < need:
            candidates = [hp for hp in hp_idx if hp not in set(already)]
        chosen = rng.choice(candidates, size=min(need, len(candidates)), replace=False).tolist()
        for hp in chosen:
            _rescale(hp, float(rng.uniform(lo, hi)))
            assigned.add(hp)

    # Keep heat_demand_df consistent with updated timeseries
    hp_in_lap = [h for h in hp_idx if h in lap.columns]
    if not edisgo.heat_pump.cop_df.empty and hp_in_lap:
        edisgo.heat_pump.heat_demand_df = (
            lap[hp_in_lap] * edisgo.heat_pump.cop_df[hp_in_lap]
        )
    print(f"[fix_hp_peak_loads] Adjusted {n_hp} heat pumps.")


def get_monthly_snapshot_ranges(year=2025, test=False):
    """Return list of (month_label, start_idx, end_idx) for each month of year.

    test=False  : full month windows for all 12 months
    test="test1": 3-day windows for January and February only
    test="test2": first 7-day window for each of the 12 months
    """
    if test == "test1":
        jan_start = 0
        feb_start = calendar.monthrange(year, 1)[1] * 24
        return [
            (f"{year}-01", jan_start, jan_start + 3 * 24 - 1),
        ]
    idx, months = 0, []
    for m in range(1, 13):
        hours = calendar.monthrange(year, m)[1] * 24
        if test == "test2":
            months.append((f"{year}-{m:02d}", idx, idx + 7 * 24 - 1))
        else:
            months.append((f"{year}-{m:02d}", idx, idx + hours - 1))
        idx += hours
    return months


def main(output_dir, snapshot_range, seed=42):
    t0 = datetime.now()

    edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=snapshot_range)

    # Set HV/MV transformer secondary voltage to 1.025 pu to reflect the common
    # DSO practice of boosting LV bus voltage to compensate for feeder voltage drops.
    edisgo.config["grid_expansion_allowed_voltage_deviations"]["hv_mv_trafo_offset"] = 0.05

    mv_grid_geom = gpd.read_file(path_husum_district_shp).to_crs(4326)
    edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
    edisgo.topology.grid_district["srid"] = 4326

    edisgo.topology.check_integrity()

    edisgo.analyze()

    # Plot original cable capacities before hc_line s_nom is overwritten
    from loma_14a_analysis import plot_cable_capacity_map, find_root_bus

    _root_bus = find_root_bus(edisgo.topology.lines_df, edisgo.topology.transformers_df)
    plot_cable_capacity_map(edisgo.topology.buses_df, edisgo.topology.lines_df,
                            _root_bus, os.path.dirname(output_dir))

    prepare_edisgo_for_14a(
        edisgo,
        shapefile_path=path_husum_district_shp,
        output_dir=output_dir,
        cache_dir=emob_cache_dir,
        setup_days=None,
        seed=seed,
    )

    fix_hp_peak_loads(edisgo, seed=seed)

    mv_buses = set(edisgo.topology.buses_df[edisgo.topology.buses_df.v_nom > 0.4].index)
    mv_loads = edisgo.topology.loads_df[edisgo.topology.loads_df["bus"].isin(mv_buses)]
    edisgo.topology.loads_df = edisgo.topology.loads_df.drop(mv_loads.index)
    edisgo.timeseries.loads_active_power = edisgo.timeseries.loads_active_power.drop(
        columns=mv_loads.index, errors="ignore"
    )
    edisgo.timeseries.loads_reactive_power = edisgo.timeseries.loads_reactive_power.drop(
        columns=mv_loads.index, errors="ignore"
    )
    print(f"[main] Removed {len(mv_loads)} loads connected to MV buses (v_nom > 0.4 kV).")

    known_loads = edisgo.topology.loads_df.index
    orphan_ap = edisgo.timeseries.loads_active_power.columns.difference(known_loads)
    orphan_rp = edisgo.timeseries.loads_reactive_power.columns.difference(known_loads)
    orphans = orphan_ap.union(orphan_rp)
    edisgo.timeseries.loads_active_power = edisgo.timeseries.loads_active_power.drop(
        columns=orphan_ap, errors="ignore"
    )
    edisgo.timeseries.loads_reactive_power = edisgo.timeseries.loads_reactive_power.drop(
        columns=orphan_rp, errors="ignore"
    )
    print(f"[main] Removed {len(orphans)} orphan timeseries loads not in topology.loads_df.")

    edisgo = run_optimization_14a(edisgo)
    edisgo.analyze()

    # ── OPF Results Summary ──────────────────────────────────────────────────
    slacks = edisgo.opf_results.grid_slacks_t
    v      = edisgo.results.v_res
    s_res  = edisgo.results.s_res

    curt_sum = get_curtailment_14a_summary(edisgo)
    hp_mwh   = curt_sum["hp_curtailment_mw"].sum()
    cp_mwh   = curt_sum["cp_curtailment_mw"].sum()

    load_shed = slacks.load_shedding.abs().sum(axis=1).sum()
    hp_shed   = slacks.hp_load_shedding.abs().sum(axis=1).sum()

    v_min, v_max = v.min().min(), v.max().max()
    viol = ((v < 0.9) | (v > 1.1)).sum().sum()

    trafos      = edisgo.topology.transformers_df
    trafo_cols  = trafos.index.intersection(s_res.columns)
    if len(trafo_cols):
        trafo_load  = s_res[trafo_cols] / trafos.loc[trafo_cols, "s_nom"] * 100
        peak_pct    = trafo_load.max().max()
        peak_trafo  = trafo_load.max().idxmax()
        peak_ts     = trafo_load[peak_trafo].idxmax()
        trafo_line  = f"{peak_pct:.1f}% ({peak_trafo}, {peak_ts})"
    else:
        trafo_line  = "n/a"

    lines      = edisgo.topology.lines_df
    line_cols  = lines.index.intersection(s_res.columns)
    line_load  = s_res[line_cols] / lines.loc[line_cols, "s_nom"] * 100
    ol_slots   = int((line_load > 100).sum().sum())
    ol_lines   = int((line_load > 100).any().sum())

    slack_line = (
        f"load_shed={load_shed:.4f} MW, hp_shed={hp_shed:.4f} MW  ⚠"
        if (load_shed > 5e-3 or hp_shed > 5e-3)
        else "none (feasible)"
    )

    print(f"\n{'─'*62}")
    print("  OPF Results Summary")
    print(f"{'─'*62}")
    print(f"  Voltage      : {v_min:.4f} – {v_max:.4f} p.u.   violations: {viol}")
    print(f"  §14a         : HP {hp_mwh:.4f} MWh | CP {cp_mwh:.4f} MWh | Total {hp_mwh+cp_mwh:.4f} MWh")
    print(f"  Slacks       : {slack_line}")
    print(f"  Trafo peak   : {trafo_line}")
    print(f"  Lines >100%  : {ol_slots} hour-line slots across {ol_lines} lines")
    print(f"{'─'*62}")

    # ── Load-shedding diagnostics ────────────────────────────────────────────
    # Non-zero pds means the linearized BF model couldn't meet voltage/current
    # limits with §14a alone. Print which buses/timesteps are affected so we
    # can tell whether this is a real grid issue or a Q-modelling artefact.
    if load_shed > 5e-3:
        ls = slacks.load_shedding
        nonzero = ls[ls.abs() > 1e-4].stack().rename_axis(["time", "load"])
        nonzero.name = "shed_MW"
        nonzero = nonzero.reset_index().sort_values("shed_MW", ascending=False)

        load_buses = edisgo.topology.loads_df["bus"]
        nonzero["bus"] = nonzero["load"].map(load_buses)

        print(f"\n{'─'*62}")
        print("  Load-shedding diagnostic (pds > 0.1 kW)")
        print(f"{'─'*62}")
        print(f"  Affected timesteps : {nonzero['time'].nunique()}")
        print(f"  Affected loads     : {nonzero['load'].nunique()}")
        print(f"  Affected buses     : {nonzero['bus'].nunique()}")
        print(f"\n  Top 10 events:")
        print(nonzero[["time", "bus", "load", "shed_MW"]].head(10).to_string(index=False))

        # Per-bus voltage at the affected timesteps (from full AC power flow)
        shed_times  = nonzero["time"].unique()
        shed_buses  = nonzero["bus"].dropna().unique()
        v_affected  = v.loc[v.index.isin(shed_times), v.columns.isin(shed_buses)]
        if not v_affected.empty:
            print(f"\n  AC voltage at affected buses/times (p.u.):")
            print(f"    min={v_affected.min().min():.4f}  max={v_affected.max().max():.4f}")
            worst_bus = v_affected.min().idxmin()
            worst_ts  = v_affected[worst_bus].idxmin()
            print(f"    Worst: bus={worst_bus}  time={worst_ts}  v={v_affected.loc[worst_ts, worst_bus]:.4f}")

            # Also show OPF voltage at same bus/time if available
            opf_v = edisgo.opf_results.v_mag_pu if hasattr(edisgo.opf_results, "v_mag_pu") else None
            if opf_v is not None and worst_bus in opf_v.columns:
                print(f"    OPF voltage at worst bus/time: {opf_v.loc[worst_ts, worst_bus]:.4f}")
        print(f"{'─'*62}")

    # Create plots for grid results per hour
    # plot_network(edisgo, show=False, snapshots=edisgo.timeseries.timeindex,
    #              folder_path=f"{output_dir}/plot")
    #create_network_gif(duration=500)

    # ── Presentation plots ───────────────────────────────────────────────────
    # Select days that have non-trivial §14a curtailment (threshold: 1 kW total)
    curt_data = get_curtailment_data(edisgo)
    curt_daily = (
        curt_data
        .groupby(edisgo.timeseries.timeindex.normalize())
        .sum()
        .sum(axis=1)
    )
    active_days = curt_daily[curt_daily > 1e-3].index.strftime("%Y-%m-%d").tolist()

    print(f"\n=== Days with §14a curtailment: {active_days} ===")

    for day in active_days:
        print(f"  Plotting {day}...")
        plot_load_before_after(edisgo, day=day, show=False, save=True,
                               folder_path=f"{output_dir}/load_plots/")

    curt_hourly = curt_data.sum(axis=1)
    active_hours = curt_hourly[curt_hourly > 1e-3].index

    print(f"\n=== Hours with §14a curtailment: {len(active_hours)} hours ===")

    if len(active_hours) > 0:
        plot_network(edisgo, show=False, snapshots=active_hours,
                     folder_path=f"{output_dir}/network_plots/")

    print(f"Saved plots to {output_dir}/plots/")
    print(f"Time: {datetime.now() - t0}")

    os.makedirs(output_dir, exist_ok=True)
    edisgo.save(f"{output_dir}/edisgo")
    return edisgo


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


if __name__ == "__main__":
    for rnd_seed in range(42,44):
        line_usage_parts = []
        curtailment_parts = []
        for month_name, snap_start, snap_end in get_monthly_snapshot_ranges(2035, test="test1"):
            output_dir = f"/home/carlos/LoMa/output_edisgo/{rnd_seed}"
            edisgo = main(f"{output_dir}/{month_name}", snapshot_range=(snap_start, snap_end), seed=rnd_seed)
            line_usage_parts.append(lines_relative_load(edisgo) * 100)
            curtailment_parts.append(get_curtailment_14a_summary(edisgo))

        line_usage = pd.concat(line_usage_parts, axis=0)
        line_usage.to_csv(f"{output_dir}/line_usage")

        curtailment_14a = pd.concat(curtailment_parts, axis=0)
        curtailment_14a.to_csv(f"{output_dir}/curtailment_14a.csv")
        print(f"\n=== §14a Curtailment (seed={rnd_seed}) ===")
        print(f"  HP: {curtailment_14a['hp_curtailment_mw'].sum():.4f} MWh")
        print(f"  CP: {curtailment_14a['cp_curtailment_mw'].sum():.4f} MWh")

        plot_cp_hp_locations(edisgo, show=False, save=True, path=output_dir)
        print("\n=== Line loading (%) ===")

