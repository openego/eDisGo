import os
from datetime import datetime

import geopandas as gpd
import pandas as pd

from edisgo import EDisGo
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

#temp1
def export_cp_hp_locations_and_timeseries(
    edisgo,
    *,
    output_dir,
    prefix="cp_hp_export",
):
    """
    Export final charging point and heat pump locations as shapefiles
    and their time series as CSV files.

    Exports:
    - <prefix>_charging_points.shp
    - <prefix>_heat_pumps.shp
    - <prefix>_charging_points_active_power.csv
    - <prefix>_heat_pumps_active_power.csv
    - <prefix>_charging_points_reactive_power.csv, if available
    - <prefix>_heat_pumps_reactive_power.csv, if available
    - <prefix>_heat_pumps_heat_demand.csv, if available
    """
    os.makedirs(output_dir, exist_ok=True)

    srid = int(edisgo.topology.grid_district.get("srid", 4326))
    crs = f"EPSG:{srid}"

    def _export_load_type(load_type, export_name):
        loads_df = edisgo.topology.loads_df
        buses_df = edisgo.topology.buses_df

        load_ids = loads_df.index[loads_df["type"] == load_type].tolist()

        if len(load_ids) == 0:
            print(f"[EXPORT] No loads of type '{load_type}' found.")
            return

        # --- Export locations ---
        export_df = loads_df.loc[load_ids].copy()

        export_df = export_df.join(
            buses_df[["x", "y"]],
            on="bus",
            how="left",
        )

        missing_xy = export_df["x"].isna() | export_df["y"].isna()
        if missing_xy.any():
            missing_examples = export_df.index[missing_xy].tolist()[:10]
            raise ValueError(
                f"{missing_xy.sum()} {load_type} loads have missing bus coordinates. "
                f"Examples: {missing_examples}"
            )

        # Shapefile column names are limited to 10 characters.
        # Therefore keep/rename only robust columns.
        shp_df = export_df.reset_index().rename(
            columns={
                "index": "load_id",
                "building_id": "bld_id",
                "source_load_id": "src_id",
                "is_duplicate": "is_dup",
            }
        )

        gdf = gpd.GeoDataFrame(
            shp_df,
            geometry=gpd.points_from_xy(shp_df["x"], shp_df["y"]),
            crs=crs,
        )

        shp_path = os.path.join(output_dir, f"{prefix}_{export_name}.shp")
        gdf.to_file(shp_path, driver="ESRI Shapefile")

        print(f"[EXPORT] Wrote {len(gdf)} {export_name} locations to:")
        print(f"         {shp_path}")

        # --- Export active power time series ---
        p_cols = [
            load_id for load_id in load_ids
            if load_id in edisgo.timeseries.loads_active_power.columns
        ]

        if p_cols:
            p_path = os.path.join(
                output_dir,
                f"{prefix}_{export_name}_active_power.csv",
            )
            edisgo.timeseries.loads_active_power.loc[:, p_cols].to_csv(p_path)
            print(f"[EXPORT] Wrote {export_name} active power time series to:")
            print(f"         {p_path}")
        else:
            print(f"[EXPORT] No active power time series found for {export_name}.")

        # --- Export reactive power time series ---
        if hasattr(edisgo.timeseries, "loads_reactive_power"):
            q_df = edisgo.timeseries.loads_reactive_power

            q_cols = [
                load_id for load_id in load_ids
                if load_id in q_df.columns
            ]

            if q_cols:
                q_path = os.path.join(
                    output_dir,
                    f"{prefix}_{export_name}_reactive_power.csv",
                )
                q_df.loc[:, q_cols].to_csv(q_path)
                print(f"[EXPORT] Wrote {export_name} reactive power time series to:")
                print(f"         {q_path}")

        # --- Optional: export heat demand for heat pumps ---
        if load_type == "heat_pump" and hasattr(edisgo, "heat_pump"):
            heat_demand_df = getattr(edisgo.heat_pump, "heat_demand_df", None)

            if heat_demand_df is not None and not heat_demand_df.empty:
                heat_cols = [
                    load_id for load_id in load_ids
                    if load_id in heat_demand_df.columns
                ]

                if heat_cols:
                    heat_path = os.path.join(
                        output_dir,
                        f"{prefix}_{export_name}_heat_demand.csv",
                    )
                    heat_demand_df.loc[:, heat_cols].to_csv(heat_path)
                    print(f"[EXPORT] Wrote {export_name} heat demand time series to:")
                    print(f"         {heat_path}")

    _export_load_type("charging_point", "charging_points")
    _export_load_type("heat_pump", "heat_pumps")
#temp2

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

    #temp1
    base_eligible_buses = [
        bus for bus in buses_with_existing_loads(edisgo)
        if bus in valid_buses
    ]
    
    cp_eligible_buses = base_eligible_buses.copy()
    hp_eligible_buses = base_eligible_buses.copy()
    #temp2
    
    set_charging_points_to_target(
        edisgo,
        target_total=3500, # sets total amount of CP #412 SQ, 1000 2035
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
        target_total=2000,  # sets total amount of HP #575 SQ, 2600 2035
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
    output_dir = "/home/paul/LoMa/test/edisgo_output"
    emob_cache_dir = "/home/paul/LoMa/loma-repo/emob_cache/husum_eGon2035"

    # Whole Husum paths
    grid_path = "/home/paul/LoMa/loma-repo/results/Whole_Husum_final_statusQuo_LV_ids" # Status-Quo
    # grid_path = "/home/paul/LoMa/loma-repo/results/Whole_Husum_model_pypsa_2035" # 2035
    path_husum_district_shp = (
        "/home/paul/LoMa/loma-repo/data/Input_files/MV_grid_district/husum_district.shp"
    )

    # MGB paths
    # grid_path = ""
    # path_husum_district_shp = "/home/paul/LoMa/loma-repo/data/Input_files/MGB_district"

    edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(0, 12)) 
    #edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(0, 167)) #first week january 2025
    #edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(2159, 2327)) #first week april 2025
    #edisgo = EDisGo(pypsa_csv_dir=grid_path, snapshot_range=(5088, 5256)) #first week august 2025

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

    #temp3
    export_cp_hp_locations_and_timeseries(
        edisgo,
        output_dir=output_dir,
        prefix="status_quo_before_opf",
    )
    #temp4
    
    plot_cp_hp_locations(edisgo, show=False, save=True)

    #edisgo = run_optimization_14a(edisgo)
    edisgo.analyze()


    # ────────────────────────── Slack diagnosis ──────────────────────────────
    slacks = edisgo.opf_results.grid_slacks_t
    print("\n=== OPF Slack Diagnosis (v5) ===")
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

    # # Create gif
    # for ts in edisgo.timeseries.timeindex:
    #     plot_network(edisgo, show=False, snapshot=str(ts))
    # create_network_gif(duration=500)

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
    