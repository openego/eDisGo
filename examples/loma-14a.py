import os
from datetime import datetime

import geopandas as gpd
import pandas as pd

from edisgo import EDisGo
from edisgo.tools.loma_tools import (
    analyze_14a_activations,
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

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


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

def integrate_ev_and_hp_for_14a(edisgo, *, shapefile_path, output_dir, setup_days=None, cache_dir=None):
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
        target_total=1000, # sets total amount of CP    #1000 for Husum2035 default, 412 for statusQuo default
        # percentage=0.10, # increases total amount of CP by 10%
        # percentage=-0.10, # decreases total amount of CP by 10%
        eligible_buses=cp_eligible_buses,
        removal_priority=["Additional", "Existing"],
        add_tracking_columns=False,
        export_removed=False,
        export_dir=output_dir,
    )

    set_heat_pumps_to_target(
        edisgo,
        target_total=2600,  # sets total amount of HP      #2600 for Husum2035 default, 575 for statusQuo default
        # percentage=0.10, # increases total amount of HP by 10%
        # percentage=-0.10, # decreases total amount of HP by 10%
        eligible_buses=hp_eligible_buses,
        add_tracking_columns=False,
        export_removed=False,  # only applies when there are deleted HP
        export_dir=output_dir,  # only applies when there are deleted HP
    )


def prepare_edisgo_for_14a(edisgo, *, shapefile_path, output_dir, cache_dir=None, setup_days=None):
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

    integrate_ev_and_hp_for_14a(
        edisgo,
        shapefile_path=shapefile_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        setup_days=setup_days,
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

def main():
    # General Paths
    output_dir = "/home/student/Execution/eDisGo_exe/eDisGo_exe_results_Whole_Husum_final_2035"
    emob_cache_dir = "/home/student/Execution/eDisGo_exe/emob_cache_Whole_Husum_final_2035"
    #"/home/student/Execution/eDisGo_exe/emob_cache_MGB_final_statusQuo"
    #"/home/student/Execution/eDisGo_exe/emob_cache_Whole_Husum_final_statusQuo"
    
    # Whole Husum paths
    grid_path = "/home/student/Execution/LoMa_exe/results/Whole_Husum_final_2035_LV_ids" # Husum
    #/home/student/Execution/LoMa_exe/results/Whole_Husum_final_statusQuo_LV_ids
    #/home/student/Execution/LoMa_exe/results/MGB_final_statusQuo_LV_ids
    
    path_husum_district_shp = (
        "/home/student/Execution/LoMa_exe/data/Input_files/MV_grid_district/husum_district.shp"
    )
   #/home/student/Execution/LoMa_exe/data/Input_files/MV_grid_district/husum_district.shp
    #/home/student/Execution/eDisGo_exe/MGB_district/MGB_district.shp
    

    edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(0, 6))

    mv_grid_geom = gpd.read_file(path_husum_district_shp).to_crs(4326)
    edisgo.topology.grid_district["geom"] = mv_grid_geom.loc[0, "geometry"]
    edisgo.topology.grid_district["srid"] = 4326
    
    edisgo.topology.check_integrity()
    pypsa_n = edisgo.to_pypsa()
    edisgo.analyze()
    
    prepare_edisgo_for_14a(
        edisgo,
        shapefile_path=path_husum_district_shp,
        output_dir=output_dir,
        cache_dir=emob_cache_dir,
        setup_days=None,
    )
    
    plot_cp_hp_locations(edisgo, show=False, save=True)

    # Save pre-optimization line loading for §14a activation diagnosis
    edisgo.analyze()
    pre_edisgo = edisgo.copy()
    pre_opt_line_loading = edisgo.results.s_res.copy()

    edisgo = run_optimization_14a(edisgo)
    edisgo.analyze()

    # ────────────────────────── Slack diagnosis ──────────────────────────────
    load_shed = edisgo.opf_results.grid_slacks_t.load_shedding

    # Positiv = echter Last-Abwurf, Negativ = Solver-Artefakt trennen
    pos = load_shed.clip(lower=0)   # nur echtes Shedding
    neg = load_shed.clip(upper=0)   # nur Artefakte
      
    print("=== OPF Slack Diagnosis ===")
    print(f"Echtes Load Shedding (p > 0):      {pos.sum().sum():.6f} MW")
    print(f"Negative Artefakte (p < 0):        {neg.sum().sum():.6f} MW")
    print(f"Rohe Summe (irreführend):           {load_shed.sum().sum():.6f} MW")
      
      # Nur Knoten mit echtem Shedding
    threshold = 1e-4  # MW, unter diesem Wert = Rauschen
    col_sums_pos = pos.sum()
    active_cols = col_sums_pos[col_sums_pos > threshold]
    
    if active_cols.empty:
        print("\n✓ Kein relevantes Load Shedding – nur Solver-Rauschen")
    else:
        print(f"\nKnoten mit echtem Shedding (>{threshold} MW gesamt):")
        for col, val in active_cols.items():
              peak = pos[col].max()
              n_ts  = (pos[col] > 1e-6).sum()
              print(f"  {col}: {val:.4f} MW gesamt | Peak: {peak:.4f} MW | {n_ts} Zeitschritte aktiv")
      
    # Alle drei Slack-Typen korrekt
    for name, df in [
          ("Load Shedding",    edisgo.opf_results.grid_slacks_t.load_shedding),
          ("HP Load Shedding", edisgo.opf_results.grid_slacks_t.hp_load_shedding),
          ("CP Load Shedding", edisgo.opf_results.grid_slacks_t.cp_load_shedding),
      ]:
        p = df.clip(lower=0).sum().sum()
        n = df.clip(upper=0).sum().sum()
        print(f"  {name:<20}: {p:.6f} MW  (Artefakte: {n:.2e} MW)")
        
        
        
    slacks = edisgo.opf_results.grid_slacks_t
    print("\n=== OPF Slack Diagnosis (v5) ===")
    load_shed = edisgo.opf_results.grid_slacks_t.load_shedding
    print("Load Shedding aktiv:")
    print(load_shed[(load_shed > 1e-6).any(axis=1)]) # Nur wenn > 0.00001
    
    for col in load_shed.columns:
          if (load_shed[col] > 1e-6).any():
                print(f" Knoten {col}: {load_shed[col].sum():.4f} MW abgeworfen")
                
    print("\nGesamtes Slack-Volume pro Komponente:")
    print(f" Load Shedding: {load_shed.sum().sum():.4f} MW")
    print(f" HP Load Shedding: {edisgo.opf_results.grid_slacks_t.hp_load_shedding.sum().sum():.4f} MW")
    print(f" CP Load Shedding: {edisgo.opf_results.grid_slacks_t.cp_load_shedding.sum().sum():.4f} MW")


    for name, df in [
        ("gen_nd_crt  (renewable curtailment)", slacks.gen_nd_crt),
        ("gen_d_crt   (disp. gen curtailment)", slacks.gen_d_crt),
        ("load_shed   (load shedding)", slacks.load_shedding),
        ("hp_shed     (HP load shedding)", slacks.hp_load_shedding),
    ]:
        total = df.abs().sum(axis=1)
        if (total > 5e-3).any():
            print(f"  {name}: {total.sum():.4f} MW  ← NON-ZERO")
        else:
            print(f"  {name}: 0 (not used)")

    print("\n=== Voltage after OPF (edisgo.results.v_res) ===")
    v = edisgo.results.v_res
    print(f"  Min:  {v.min().min():.4f} p.u.")
    print(f"  Max:  {v.max().max():.4f} p.u.")
    viol = (v < 0.9) | (v > 1.1)
    if viol.any().any():
        print(f"  Violations:{viol.sum().sum()}")
        print()
    else:
        print("  No voltage violations.")
    # ────────────────────────── End diagnosis ────────────────────────────────

    print("\n=== 14a analysis ===")
    gen = edisgo.topology.generators_df
    gen_t = edisgo.timeseries.generators_active_power
    gen_14a = gen[gen.index.str.contains("14a")]
    gen_t_14a = gen_t.loc[:, gen_14a.index]
    print(f"Total use of 14a:{gen_t_14a.sum().sum()}")
    print("\n=== end 14a analysis ===")

    # ── §14a Activation Diagnosis ────────────────────────────────────────────
    # Cross-correlates §14a activations with pre-optimization line loading.
    # "has_pre_overload=False" rows are candidates for spurious activation.
    activation_report = analyze_14a_activations(
        edisgo,
        pre_opt_line_loading,
        threshold_kw=0.5,
    )
    if not activation_report.empty:
        report_csv = os.path.join(output_dir, "14a_activation_report.csv")
        activation_report.drop(columns=["top_generators"]).to_csv(report_csv)
        print(f"\n§14a activation report saved to: {report_csv}")
        spurious = activation_report[~activation_report["has_pre_overload"]]
        if not spurious.empty:
            print("\nTimesteps with §14a activation but no pre-opt overload:")
            print(spurious[["14a_total_mw", "n_active_generators",
                             "max_line_loading_pre"]].to_string())
    # ── End §14a Diagnosis ───────────────────────────────────────────────────



    # Create gif
    output_folder = "plots/2035_before_analyze"   # <-- hier einmal definieren

    for ts in edisgo.timeseries.timeindex:
          plot_network(edisgo, show=False, snapshot=str(ts), output_folder=output_folder, focus_bus=None)
      
    create_network_gif(
          folder_path=output_folder,
          output_name=f"{output_folder}/network_evolution.gif",
          duration=500,
    )

    # ── Presentation plots ───────────────────────────────────────────────────────
    # Select days that have non-trivial §14a curtailment (threshold: 1 kW total)
    curt_daily = (
        get_curtailment_data(edisgo)
        .groupby(edisgo.timeseries.timeindex.normalize())
        .sum()
        .sum(axis=1)
    )
    active_days = curt_daily[curt_daily > 1e-3].index.strftime("%Y-%m-%d").tolist()

    print(f"\n=== Days with §14a curtailment: {active_days} ===")

    for day in active_days:
        print(f"  Plotting {day}...")
        plot_load_before_after(edisgo, day=day, show=False, save=True)

    print("Saved plots to ./plots/")

    return edisgo


if __name__ == "__main__":
    edisgo = main()

    
