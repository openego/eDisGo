import logging
import os

import contextily as ctx
import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as _cKDTree
from shapely.geometry import Point

from sqlalchemy.engine.base import Engine

from edisgo.flex_opt.battery_storage_operation import _reference_operation
from edisgo.io.db import get_srid_of_db_table, session_scope_egon_data
from edisgo.tools.config import Config

logger = logging.getLogger(__name__)

# ==========================
# Transferring timeseries from new edisgo cp to matched existing/additional cp
# ==========================
def transfer_ts_from_new_to_existing_cp(
    edisgo,
    *,
    existing_markers=("Existing", "Additional"),
    radius_1=2000.0,
    tol_1=0.15,
    radius_2=2000.0,
    tol_2=0.9,
    metric_epsg=32632,
    keep_existing_bus_and_pset=True,
    drop_new=True,
):
    buses = edisgo.topology.buses_df[["x", "y"]].copy()

    loads = edisgo.topology.loads_df.copy()
    charging_points = loads[loads["type"] == "charging_point"].copy()

    existing_mask = charging_points.index.astype(str).str.contains(
        "|".join(existing_markers),
        regex=True,
    )

    existing_cp = charging_points[existing_mask].copy()
    new_cp = charging_points[~existing_mask].copy()

    grid_srid = int(edisgo.topology.grid_district.get("srid", 4326))

    existing_gdf = _cp_points_gdf(existing_cp, buses=buses, crs_epsg=grid_srid)
    new_gdf = _cp_points_gdf(new_cp, buses=buses, crs_epsg=grid_srid)

    existing_m = existing_gdf.to_crs(epsg=metric_epsg)
    new_m = new_gdf.to_crs(epsg=metric_epsg)

    matches, unmatched_existing, unused_new = _match_two_step_nearest(
        existing_m,
        new_m,
        radius_1=radius_1,
        tol_1=tol_1,
        radius_2=radius_2,
        tol_2=tol_2,
    )

    _transfer_ts_and_replace_new_with_existing(
        edisgo,
        matches,
        keep_existing_bus_and_pset=keep_existing_bus_and_pset,
        drop_new=drop_new,
    )

    return {
        "matches": matches,
        "unmatched_existing": unmatched_existing,
        "unused_new": unused_new,
        "num_matches": len(matches),
        "num_unmatched_existing": len(unmatched_existing),
        "num_unused_new": len(unused_new),
    }

def _cp_points_gdf(
    loads_df_subset: pd.DataFrame,
    *,
    buses: pd.DataFrame,
    crs_epsg: int = 4326,
    ) -> gpd.GeoDataFrame:    
    """
    Build a GeoDataFrame of CPs with geometry at their connected bus coordinates.
    Assumes buses_df x/y are in EPSG:4326 if edisgo topology says srid=4326.
    """
    if not {"x", "y"}.issubset(buses.columns):
        raise KeyError("buses must contain 'x' and 'y' columns")
        
    df = loads_df_subset[["bus", "p_set"]].copy()
    df = df.join(buses, on="bus", how="left")

    missing_xy = df["x"].isna() | df["y"].isna()
    if missing_xy.any():
        missing_ids = df.index[missing_xy].tolist()[:10]
        raise ValueError(
            f"Missing bus coordinates for {missing_xy.sum()} loads. Example IDs: {missing_ids}"
        )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["x"].values, df["y"].values)],
        crs=f"EPSG:{crs_epsg}",
    )
    return gdf

# --------------------------
# Matching EXISTING -> NEW by nearest bus location + p_set tolerance
# --------------------------
def _match_two_step_nearest(
    existing_m: gpd.GeoDataFrame,
    new_m: gpd.GeoDataFrame,
    *,
    radius_1=2000.0,
    tol_1=0.15,
    radius_2=2000.0,
    tol_2=0.90,
    w_dist_2=0.3,
    w_pset_2=0.7,
):
    from scipy.optimize import linear_sum_assignment

    new_ids = np.array(new_m.index.to_list(), dtype=object)

    # Projected (metric) coordinate arrays — Euclidean distance is exact here.
    new_xy = np.column_stack([new_m.geometry.x.values, new_m.geometry.y.values])
    ex_xy  = np.column_stack([existing_m.geometry.x.values, existing_m.geometry.y.values])
    new_pset_arr = new_m["p_set"].values.astype(float)
    ex_pset_arr  = existing_m["p_set"].values.astype(float)
    _ex_gdf_pos = {label: i for i, label in enumerate(existing_m.index)}
    _new_gdf_pos = {label: i for i, label in enumerate(new_m.index)}
    kd_tree = _cKDTree(new_xy)

    used_new = set()

    phase1_matches = []
    phase2_matches = []

    def _candidate_pairs(existing_ids, max_radius_m, pset_tol):
        """Build all admissible candidate pairs for the given existing IDs."""
        pairs = []
        for ex_id in existing_ids:
            i = _ex_gdf_pos[ex_id]
            ex_coord = ex_xy[i]
            p_ex = ex_pset_arr[i]

            cand_js = np.array(kd_tree.query_ball_point(ex_coord, r=max_radius_m), dtype=int)
            if len(cand_js) == 0:
                continue

            p_lo, p_hi = p_ex * (1 - pset_tol), p_ex * (1 + pset_tol)
            pset_ok = (new_pset_arr[cand_js] >= p_lo) & (new_pset_arr[cand_js] <= p_hi)
            not_used = np.array([new_ids[j] not in used_new for j in cand_js])
            valid = cand_js[pset_ok & not_used]
            if len(valid) == 0:
                continue

            dists = np.linalg.norm(new_xy[valid] - ex_coord, axis=1)
            for k, j in enumerate(valid):
                new_id = new_ids[j]
                p_new = float(new_pset_arr[j])
                rel_dev = abs(p_new / p_ex - 1.0) if p_ex != 0 else 0.0
                pairs.append((ex_id, new_id, float(dists[k]), rel_dev))

        return pairs

    def _greedy_phase1(existing_ids, max_radius_m, pset_tol):
        """
        Phase 1:
        Greedy nearest matching with tight p_set tolerance.
        Primary criterion: distance
        Secondary criterion: relative p_set deviation
        """
        pass_matches = []
        pass_unmatched = []

        for ex_id in existing_ids:
            i = _ex_gdf_pos[ex_id]
            ex_coord = ex_xy[i]
            p_ex = ex_pset_arr[i]

            cand_js = np.array(kd_tree.query_ball_point(ex_coord, r=max_radius_m), dtype=int)
            if len(cand_js) == 0:
                pass_unmatched.append(ex_id)
                continue

            p_lo, p_hi = p_ex * (1 - pset_tol), p_ex * (1 + pset_tol)
            pset_ok = (new_pset_arr[cand_js] >= p_lo) & (new_pset_arr[cand_js] <= p_hi)
            not_used = np.array([new_ids[j] not in used_new for j in cand_js])
            valid = cand_js[pset_ok & not_used]

            if len(valid) == 0:
                pass_unmatched.append(ex_id)
                continue

            dists = np.linalg.norm(new_xy[valid] - ex_coord, axis=1)
            best_local = int(np.argmin(dists))
            best_j = int(valid[best_local])
            best_new_id = new_ids[best_j]

            used_new.add(best_new_id)
            pass_matches.append((ex_id, best_new_id, float(dists[best_local])))

        return pass_matches, pass_unmatched

    def _global_phase2(existing_ids, max_radius_m, pset_tol, w_dist, w_pset):
        """
        Phase 2:
        Global bipartite assignment for all remaining unmatched existing CPs.

        Objective:
        weighted score of normalized distance and normalized p_set deviation.
        """
        pairs = _candidate_pairs(existing_ids, max_radius_m, pset_tol)

        if len(existing_ids) == 0:
            return [], []

        if len(pairs) == 0:
            return [], list(existing_ids)

        ex_list = list(existing_ids)
        new_list = sorted({new_id for _, new_id, _, _ in pairs})

        ex_pos = {ex_id: i for i, ex_id in enumerate(ex_list)}
        new_pos = {new_id: j for j, new_id in enumerate(new_list)}

        BIG = 1e12
        cost = np.full((len(ex_list), len(new_list)), BIG, dtype=float)

        for ex_id, new_id, dist, rel_dev in pairs:
            i = ex_pos[ex_id]
            j = new_pos[new_id]

            dist_norm = dist / max_radius_m if max_radius_m > 0 else dist
            pset_norm = rel_dev / pset_tol if pset_tol > 0 else rel_dev

            score = w_dist * dist_norm + w_pset * pset_norm

            if score < cost[i, j]:
                cost[i, j] = score

        row_ind, col_ind = linear_sum_assignment(cost)

        pass_matches = []
        matched_existing = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= BIG:
                continue

            ex_id = ex_list[i]
            new_id = new_list[j]

            dist = float(np.linalg.norm(ex_xy[_ex_gdf_pos[ex_id]] - new_xy[_new_gdf_pos[new_id]]))

            used_new.add(new_id)
            matched_existing.add(ex_id)
            pass_matches.append((ex_id, new_id, dist))

        pass_unmatched = [ex_id for ex_id in ex_list if ex_id not in matched_existing]
        return pass_matches, pass_unmatched

    def _log_phase(phase_name, phase_matches, tol):
        print(f"[EV] {phase_name} matches (p_set tol={tol}): {len(phase_matches)}")
        for i, (ex_id, new_id, dist) in enumerate(phase_matches, 1):
            bus = existing_m.loc[ex_id, "bus"]
            p_ex = float(existing_m.loc[ex_id, "p_set"])
            p_new = float(new_m.loc[new_id, "p_set"])
            rel_dev = (p_new / p_ex - 1.0) if p_ex != 0 else float("nan")
            print(
                f"[EV] [{phase_name.upper()}][{i}] "
                f"{ex_id} (bus={bus}, p_set_old={p_ex:.6f}) "
                f"-> {new_id} (p_set_new={p_new:.6f})  "
                f"dist={dist:.1f} m  rel_dev={rel_dev:+.2%}"
            )

    # ---- Phase 1: greedy, distance-dominated ----
    m1, u1 = _greedy_phase1(list(existing_m.index), radius_1, tol_1)
    phase1_matches.extend(m1)

    # ---- Phase 2: one global pass for all remaining ----
    m2, u2 = _global_phase2(u1, radius_2, tol_2, w_dist_2, w_pset_2)
    phase2_matches.extend(m2)

    matches = m1 + m2
    unused_new = [i for i in new_ids.tolist() if i not in used_new]

    _log_phase("Phase 2", phase2_matches, tol_2)

    print(f"[EV] Unmatched existing CPs after Phase 2: {len(u2)}")
    for i, ex_id in enumerate(u2, 1):
        bus = existing_m.loc[ex_id, "bus"]
        p_ex = float(existing_m.loc[ex_id, "p_set"])
        print(f"[EV] [UNMATCHED][{i}] {ex_id} (bus={bus}, p_set_old={p_ex:.6f})")

    return matches, u2, unused_new

# --------------------------
# Transfer time series + electromobility mapping NEW -> EXISTING, then remove NEW loads
# --------------------------
def _transfer_ts_and_replace_new_with_existing(
    edisgo,
    matches,
    keep_existing_bus_and_pset=True,
    drop_new=True,
):
    tindex = edisgo.timeseries.timeindex

    icp = edisgo.electromobility.integrated_charging_parks_df
    if icp is None or icp.empty:
        raise ValueError("integrated_charging_parks_df missing/empty.")
    if "edisgo_id" not in icp.columns:
        raise KeyError(f"Missing 'edisgo_id' in integrated_charging_parks_df. Columns: {list(icp.columns)}")

    for existing_id, new_id, dist in matches:
        # --- P ---
        if new_id in edisgo.timeseries.loads_active_power.columns:
            sP = edisgo.timeseries.loads_active_power[new_id].reindex(tindex)

            edisgo.timeseries.drop_component_time_series("loads_active_power", [existing_id])
            edisgo.timeseries.add_component_time_series(
                "loads_active_power",
                pd.DataFrame({existing_id: sP.values}, index=tindex),
            )
        else:
            print(f"[EV] new_id has no active TS: {new_id}")

        # --- Q (optional) ---
        if new_id in edisgo.timeseries.loads_reactive_power.columns:
            sQ = edisgo.timeseries.loads_reactive_power[new_id].reindex(tindex)

            edisgo.timeseries.drop_component_time_series("loads_reactive_power", [existing_id])
            edisgo.timeseries.add_component_time_series(
                "loads_reactive_power",
                pd.DataFrame({existing_id: sQ.values}, index=tindex),
            )

        # --- mapping: new -> existing ---
        mask = icp["edisgo_id"] == new_id
        if mask.any():
            icp.loc[mask, "edisgo_id"] = existing_id

        # keep existing bus+p_set: do NOT touch those columns
        if keep_existing_bus_and_pset and existing_id in edisgo.topology.loads_df.index:
            edisgo.topology.loads_df.at[existing_id, "type"] = "charging_point"
            
        # --- remove NEW load + TS to keep CP count constant ---
        if drop_new and (new_id in edisgo.topology.loads_df.index):
            if new_id in edisgo.timeseries.loads_active_power.columns:
                edisgo.timeseries.drop_component_time_series("loads_active_power", [new_id])
        
            if new_id in edisgo.timeseries.loads_reactive_power.columns:
                edisgo.timeseries.drop_component_time_series("loads_reactive_power", [new_id])
        
            # Remove only the load row, keep the connected bus in buses_df
            edisgo.topology.loads_df = edisgo.topology.loads_df.drop(index=new_id)

    edisgo.electromobility.integrated_charging_parks_df = icp

# ============================================================
# Utilities for sensitivity analysis
# - supports charging points and heat pumps
# - target by absolute value or relative percentage
# - for charging points: existing ones can be removed last
# - duplicates/removes topology rows + power-flow time series
# ============================================================

# ============================================================
# Basic helpers
# ============================================================
def _make_unique_load_id(existing_index, base_name):
    """
    Create a unique load ID not yet present in loads_df.index.
    """
    if base_name not in existing_index:
        return base_name

    i = 1
    while f"{base_name}_{i}" in existing_index:
        i += 1
    return f"{base_name}_{i}"

def buses_with_existing_loads(edisgo):
    """
    Return buses that already host at least one load.
    """
    return edisgo.topology.loads_df["bus"].dropna().unique()

def buses_with_same_load_type(edisgo, load_type):
    """
    Return buses that already host at least one load of the given type.
    """
    df = edisgo.topology.loads_df
    return df.loc[df["type"] == load_type, "bus"].dropna().unique()

def get_load_ids_by_type(edisgo, load_type):
    """
    Return all load IDs of one load type.
    """
    loads_df = edisgo.topology.loads_df
    return loads_df.index[loads_df["type"] == load_type].tolist()

def split_ids_by_marker(load_ids, marker="Existing"):
    """
    Split IDs into two groups based on a substring marker.
    Helps with removin existing loads last. 
    Currently only used for charging points.
    
    Parameters
    ----------
    load_ids : list-like
    marker : str | None

    Returns
    -------
    tuple[list[str], list[str]]
        (marked_ids, unmarked_ids)

    Notes
    -----
    If marker is None, everything is returned as unmarked.
    """
    load_ids = list(load_ids)

    if marker is None:
        return [], load_ids

    marked_ids = [i for i in load_ids if marker in str(i)]
    unmarked_ids = [i for i in load_ids if marker not in str(i)]
    return marked_ids, unmarked_ids

def _resolve_target_total(current_count, *, target_total=None, percentage=None):
    """
    Resolve desired final count.

    Exactly one of target_total or percentage must be provided.

    Parameters
    ----------
    current_count : int
        Current number of loads
    target_total : int | None
        Desired final absolute number
    percentage : float | None
        Relative change of current count:
        +0.10 -> increase by 10%
        -0.10 -> decrease by 10%

    Returns
    -------
    int
        Desired final load count
    """
    if (target_total is None) == (percentage is None):
        raise ValueError("Provide exactly one of 'target_total' or 'percentage'.")

    if target_total is not None:
        if target_total < 0:
            raise ValueError("target_total must be >= 0")
        return int(target_total)

    desired_total = int(round(current_count * (1 + percentage)))
    return max(0, desired_total)

# ============================================================
# Selection logic
# ============================================================
def _select_keep_ids(
    candidate_ids,
    *,
    target_total,
    seed=42,
):
    """
    Randomly select IDs to keep.
    """
    rng = np.random.default_rng(seed)
    candidate_ids = list(candidate_ids)

    if target_total < 0:
        raise ValueError("target_total must be >= 0")

    if target_total >= len(candidate_ids):
        return candidate_ids.copy()

    if target_total == 0:
        return []

    return rng.choice(np.array(candidate_ids), size=target_total, replace=False).tolist()

def _select_keep_ids_by_removal_priority(
    candidate_ids,
    *,
    target_total,
    seed=42,
    removal_priority=None,
):
    """
    Select IDs to keep such that removal happens in priority stages.

    Removal order:
    1. IDs without any marker
    2. IDs matching earlier markers in `removal_priority`
    3. IDs matching later markers in `removal_priority`

    Example
    -------
    removal_priority = ["Additional", "Existing"]

    -> removal order:
       - no marker
       - "Additional"
       - "Existing"

    Parameters
    ----------
    candidate_ids : list[str]
    target_total : int
    seed : int
    removal_priority : list[str] | None
        Markers ordered from earlier removable to later removable.
        Later entries are protected longer.

    Returns
    -------
    list[str]
        IDs to keep
    """
    rng = np.random.default_rng(seed)
    candidate_ids = list(candidate_ids)

    if target_total < 0:
        raise ValueError("target_total must be >= 0")

    if target_total >= len(candidate_ids):
        return candidate_ids.copy()

    if target_total == 0:
        return []

    if not removal_priority:
        return rng.choice(np.array(candidate_ids), size=target_total, replace=False).tolist()

    # Buckets:
    # bucket 0 = IDs without any marker
    # bucket 1 = first marker in removal_priority
    # bucket 2 = second marker in removal_priority
    # ...
    buckets = {i: [] for i in range(len(removal_priority) + 1)}

    for load_id in candidate_ids:
        load_id_str = str(load_id)

        matched_bucket = 0
        for i, marker in enumerate(removal_priority, start=1):
            if marker in load_id_str:
                matched_bucket = i
        buckets[matched_bucket].append(load_id)

    # Keep from highest protection bucket first
    keep_ids = []
    remaining = target_total

    for bucket_idx in range(len(removal_priority), -1, -1):
        bucket_ids = buckets[bucket_idx]

        if remaining <= 0:
            break

        if remaining >= len(bucket_ids):
            keep_ids.extend(bucket_ids)
            remaining -= len(bucket_ids)
        else:
            keep_ids.extend(
                rng.choice(np.array(bucket_ids), size=remaining, replace=False).tolist()
            )
            remaining = 0

    return keep_ids

def _select_source_ids_for_duplication(
    candidate_ids,
    *,
    n_add,
    seed=42,
):
    """
    Select source IDs for duplication.

    Sampling is with replacement to allow arbitrary growth.
    No special priority logic is applied.
    """
    rng = np.random.default_rng(seed)
    candidate_ids = list(candidate_ids)

    if len(candidate_ids) == 0:
        raise ValueError("No candidate source IDs available for duplication.")

    return rng.choice(np.array(candidate_ids), size=n_add, replace=True)

# ============================================================
# Topology / time-series write helpers
# ============================================================
def _duplicate_loads_from_source_ids(
    edisgo,
    *,
    source_ids,
    load_type,
    eligible_buses=None,
    seed=42,
    copy_reactive_power=True,
    name_prefix=None,
    avoid_source_bus=False,
    add_tracking_columns=False,
):
    """
    Duplicate given source loads to eligible buses.
    """
    rng = np.random.default_rng(seed)

    loads_df = edisgo.topology.loads_df
    tindex = edisgo.timeseries.timeindex

    if name_prefix is None:
        name_prefix = f"{load_type}_dup"

    if eligible_buses is None:
        eligible_buses = buses_with_same_load_type(edisgo, load_type)

    eligible_buses = np.array(pd.Index(eligible_buses).dropna().unique())

    if len(eligible_buses) == 0:
        raise ValueError(f"No eligible buses available for load type '{load_type}'.")

    new_rows = []
    new_p_ts = {}
    new_q_ts = {}
    new_ids = []

    for k, src_id in enumerate(source_ids, start=1):
        if src_id not in loads_df.index:
            raise KeyError(f"Source load '{src_id}' not found in topology.loads_df.")

        src_row = loads_df.loc[src_id].copy()
        src_bus = src_row["bus"]

        if avoid_source_bus:
            bus_pool = eligible_buses[eligible_buses != src_bus]
            if len(bus_pool) == 0:
                raise ValueError(
                    f"No eligible target buses left after excluding source bus '{src_bus}'."
                )
        else:
            bus_pool = eligible_buses

        tgt_bus = rng.choice(bus_pool)

        new_id_base = f"{name_prefix}_{k}"
        new_id = _make_unique_load_id(loads_df.index.union(pd.Index(new_ids)), new_id_base)

        new_row = src_row.copy()
        new_row.name = new_id
        new_row["bus"] = tgt_bus
        new_row["type"] = load_type

        if add_tracking_columns:
            new_row["source_load_id"] = src_id
            new_row["is_duplicate"] = True

        new_rows.append(new_row)
        new_ids.append(new_id)

        if src_id in edisgo.timeseries.loads_active_power.columns:
            new_p_ts[new_id] = (
                edisgo.timeseries.loads_active_power[src_id]
                .reindex(tindex)
                .values
            )
        else:
            raise KeyError(f"Source load '{src_id}' has no active power time series.")

        if copy_reactive_power and src_id in edisgo.timeseries.loads_reactive_power.columns:
            new_q_ts[new_id] = (
                edisgo.timeseries.loads_reactive_power[src_id]
                .reindex(tindex)
                .values
            )

    new_rows_df = pd.DataFrame(new_rows)
    edisgo.topology.loads_df = pd.concat([edisgo.topology.loads_df, new_rows_df], axis=0)

    new_p_df = pd.DataFrame(new_p_ts, index=tindex)
    edisgo.timeseries.add_component_time_series("loads_active_power", new_p_df)

    if new_q_ts:
        new_q_df = pd.DataFrame(new_q_ts, index=tindex)
        edisgo.timeseries.add_component_time_series("loads_reactive_power", new_q_df)

    return new_ids

def export_removed_loads_report(
    edisgo,
    removed_ids,
    *,
    output_dir,
    file_prefix,
):
    """
    Export removed loads to CSV and SHP using the bus coordinates they were
    connected to before removal.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not removed_ids:
        print("[EXPORT] No removed IDs to export.")
        return None

    loads = edisgo.topology.loads_df.loc[removed_ids].copy()

    buses = edisgo.topology.buses_df[["x", "y"]].copy()
    loads = loads.join(buses, on="bus", how="left")

    csv_path = os.path.join(output_dir, f"{file_prefix}.csv")
    loads.to_csv(csv_path, index=True)
    print(f"[EXPORT] Wrote CSV: {csv_path}")

    shp_path = None
    if {"x", "y"}.issubset(loads.columns):
        shp_df = loads.copy()
        shp_df["geometry"] = shp_df.apply(lambda r: Point(r["x"], r["y"]), axis=1)
        gdf = gpd.GeoDataFrame(shp_df, geometry="geometry", crs="EPSG:4326")

        shp_path = os.path.join(output_dir, f"{file_prefix}.shp")
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        print(f"[EXPORT] Wrote SHP: {shp_path}")

    return {
        "csv": csv_path,
        "shp": shp_path,
        "removed_ids": removed_ids,
    }

def _remove_load_ids(
    edisgo,
    remove_ids,
    *,
    export_removed=False,
    export_dir=None,
    export_prefix=None,
):
    """
    Remove specified loads from topology and time series, but keep buses.

    This intentionally does not call edisgo.remove_component("load", ...)
    because remove_component may also remove now-empty buses.
    """
    if not remove_ids:
        return []

    if export_removed and export_dir is None:
        raise ValueError("export_dir must be provided when export_removed=True")

    if export_removed:
        export_removed_loads_report(
            edisgo,
            remove_ids,
            output_dir=export_dir,
            file_prefix=export_prefix or "removed_loads",
        )

    p_cols = [
        i for i in remove_ids
        if i in edisgo.timeseries.loads_active_power.columns
    ]
    q_cols = [
        i for i in remove_ids
        if i in edisgo.timeseries.loads_reactive_power.columns
    ]

    if p_cols:
        edisgo.timeseries.drop_component_time_series(
            "loads_active_power",
            p_cols,
        )

    if q_cols:
        edisgo.timeseries.drop_component_time_series(
            "loads_reactive_power",
            q_cols,
        )

    # Important: remove only the load rows, keep buses untouched
    existing_remove_ids = [
        i for i in remove_ids
        if i in edisgo.topology.loads_df.index
    ]

    if existing_remove_ids:
        edisgo.topology.loads_df = edisgo.topology.loads_df.drop(
            index=existing_remove_ids
        )

    return remove_ids


# ============================================================
# Main generic function
# ============================================================
def set_loads_to_target(
    edisgo,
    *,
    load_type,
    target_total=None,
    percentage=None,
    seed=42,
    eligible_buses=None,
    avoid_source_bus=False,
    add_tracking_columns=False,
    copy_reactive_power=True,
    export_removed=False,
    export_dir=None,
    export_prefix=None,
    name_prefix=None,
    remove_marked_last=True,
    removal_priority=None,
):
    """
    Set final number of loads of one type to a target value.

    Supports:
    - absolute control via target_total
    - relative control via percentage

    Increase logic:
    - duplicates are created from the full current population
    - no existing/new priority is applied

    Reduction logic:
    - remove_marked_last=False: fully random removal
    - remove_marked_last=True: staged removal according to `removal_priority`

    Parameters
    ----------
    edisgo : object
        eDisGo object
    load_type : str
        e.g. "charging_point" or "heat_pump"
    target_total : int | None
        Final desired absolute number of loads
    percentage : float | None
        Relative change:
        +0.10 -> final count = round(current * 1.10)
        -0.10 -> final count = round(current * 0.90)
    seed : int
        RNG seed
    eligible_buses : array-like | None
        Eligible target buses for added duplicates
    avoid_source_bus : bool
        If True, duplicates will not be placed on their source bus
    add_tracking_columns : bool
        If True, write source_load_id and is_duplicate into new rows
    copy_reactive_power : bool
    export_removed : bool
    export_dir : str | None
    export_prefix : str | None
    name_prefix : str | None
    remove_marked_last : bool
        If True, removal follows staged priority.
    removal_priority : list[str] | None
        Markers ordered from earlier removable to later removable.
        Example:
        ["Additional", "Existing"]
        means:
        no marker -> Additional -> Existing
    """
    current_ids = get_load_ids_by_type(edisgo, load_type)
    current_count = len(current_ids)

    if current_count == 0:
        raise ValueError(f"No loads of type '{load_type}' found.")

    desired_total = _resolve_target_total(
        current_count,
        target_total=target_total,
        percentage=percentage,
    )

    if name_prefix is None:
        name_prefix = f"{load_type}_dup"

    print(f"[{load_type}] Current count: {current_count}")
    print(f"[{load_type}] Target count:  {desired_total}")

    # no change
    if desired_total == current_count:
        print(f"[{load_type}] Nothing to do.")
        return {
            "kept_ids": current_ids.copy(),
            "removed_ids": [],
            "new_ids": [],
            "final_count": current_count,
        }

    # reduction
    if desired_total < current_count:
        if remove_marked_last:
            keep_ids = _select_keep_ids_by_removal_priority(
                current_ids,
                target_total=desired_total,
                seed=seed,
                removal_priority=removal_priority,
            )
        else:
            keep_ids = _select_keep_ids(
                current_ids,
                target_total=desired_total,
                seed=seed,
            )

        keep_set = set(keep_ids)
        remove_ids = [i for i in current_ids if i not in keep_set]

        _remove_load_ids(
            edisgo,
            remove_ids,
            export_removed=export_removed,
            export_dir=export_dir,
            export_prefix=export_prefix or f"removed_{load_type}",
        )

        final_count = int((edisgo.topology.loads_df["type"] == load_type).sum())

        print(f"[{load_type}] Removed count: {len(remove_ids)}")
        print(f"[{load_type}] Final count:   {final_count}")

        return {
            "kept_ids": keep_ids,
            "removed_ids": remove_ids,
            "new_ids": [],
            "final_count": final_count,
        }

    # increase
    n_add = desired_total - current_count
    source_ids = _select_source_ids_for_duplication(
        current_ids,
        n_add=n_add,
        seed=seed,
    )

    new_ids = _duplicate_loads_from_source_ids(
        edisgo,
        source_ids=source_ids,
        load_type=load_type,
        eligible_buses=eligible_buses,
        seed=seed,
        copy_reactive_power=copy_reactive_power,
        name_prefix=name_prefix,
        avoid_source_bus=avoid_source_bus,
        add_tracking_columns=add_tracking_columns,
    )

    final_count = int((edisgo.topology.loads_df["type"] == load_type).sum())

    print(f"[{load_type}] Added count: {len(new_ids)}")
    print(f"[{load_type}] Final count: {final_count}")

    return {
        "kept_ids": current_ids.copy(),
        "removed_ids": [],
        "new_ids": new_ids,
        "final_count": final_count,
    }

# ============================================================
# Convenience wrappers: charging points
# ============================================================
def set_charging_points_to_target(
    edisgo,
    *,
    target_total=None,
    percentage=None,
    seed=42,
    eligible_buses=None,
    removal_priority=None,
    remove_existing_last=True,
    avoid_source_bus=False,
    add_tracking_columns=False,
    export_removed=False,
    export_dir=None,
):
    """
    Set charging points to a final target count.

    Important
    ---------
    - when increasing: no existing/new priority is applied
    - when reducing: removal follows staged priority

    Default removal order:
    no marker -> Additional -> Existing
    """
    if removal_priority is None:
        removal_priority = ["Additional", "Existing"]

    return set_loads_to_target(
        edisgo,
        load_type="charging_point",
        target_total=target_total,
        percentage=percentage,
        seed=seed,
        eligible_buses=eligible_buses,
        avoid_source_bus=avoid_source_bus,
        add_tracking_columns=add_tracking_columns,
        copy_reactive_power=True,
        export_removed=export_removed,
        export_dir=export_dir,
        export_prefix="removed_charging_points",
        name_prefix="cp_dup",
        remove_marked_last=remove_existing_last,
        removal_priority=removal_priority,
    )

# ============================================================
# Convenience wrappers: heat pumps
# ============================================================
def set_heat_pumps_to_target(
    edisgo,
    *,
    target_total=None,
    percentage=None,
    seed=42,
    eligible_buses=None,
    hp_type="heat_pump",
    avoid_source_bus=False,
    add_tracking_columns=False,
    export_removed=False,
    export_dir=None,
):
    """
    Set heat pumps to a final target count.

    No existing/new prioritization is applied.
    """
    return set_loads_to_target(
        edisgo,
        load_type=hp_type,
        target_total=target_total,
        percentage=percentage,
        seed=seed,
        eligible_buses=eligible_buses,
        avoid_source_bus=avoid_source_bus,
        add_tracking_columns=add_tracking_columns,
        copy_reactive_power=True,
        export_removed=export_removed,
        export_dir=export_dir,
        export_prefix=f"removed_{hp_type}",
        name_prefix="hp_dup",
        remove_marked_last=False,
        removal_priority=None,
    )


# Helper function for 14a functions
def _get_intersecting_mv_grid_ids_from_shapefile(
    engine: Engine,
    shapefile_path: str,
):
    """
    Determine all mv_grid_ids whose charging infrastructure intersects the given shapefile.
    This is used as a common spatial selector for both cars and charging parks.
    """
    from sqlalchemy import func

    if shapefile_path is None:
        raise ValueError("shapefile_path must be provided.")

    config = Config()
    (egon_emob_charging_infrastructure,) = config.import_tables_from_oep(
        engine, ["egon_emob_charging_infrastructure"], "grid"
    )

    local_shape = gpd.read_file(shapefile_path)

    with session_scope_egon_data(engine) as session:
        srid = get_srid_of_db_table(session, egon_emob_charging_infrastructure.geometry)

        local_shape = local_shape.to_crs(f"EPSG:{srid}")
        shape_union = local_shape.unary_union
        shape_wkt = shape_union.wkt

        query = (
            session.query(egon_emob_charging_infrastructure.mv_grid_id)
            .filter(
                func.ST_Intersects(
                    egon_emob_charging_infrastructure.geometry,
                    func.ST_GeomFromText(shape_wkt, srid),
                )
            )
            .distinct()
        )

        mv_grid_ids = pd.read_sql(sql=query.statement, con=session.bind)[
            "mv_grid_id"
        ].tolist()

    mv_grid_ids = sorted(set(mv_grid_ids))

    if len(mv_grid_ids) == 0:
        print(
            "No intersecting mv_grid_ids found for shapefile %s.", shapefile_path
        )

    return mv_grid_ids


# ==========================
# Storage timeseries
# ==========================


def set_storage_timeseries_bus_level(edisgo, soe_init=0.0, freq=None):
    """
    Set storage unit active power time series based on bus-level net balance.

    Charges when total generation at the bus exceeds total demand (surplus),
    discharges when demand exceeds total generation (deficit). The battery
    never exchanges energy with the grid — it only covers local surpluses and
    deficits at its own bus.

    When multiple storage units share a bus the net balance signal is split
    proportionally by p_nom so that their combined power never exceeds the
    available surplus or deficit.

    Parameters
    ----------
    edisgo : EDisGo
    soe_init : float
        Initial state of energy in MWh for all storage units. Default: 0.
    freq : float or None
        Timestep length in hours (e.g. 1.0 for hourly, 0.25 for 15-min).
        Inferred from timeindex when None.
    """
    stor_df = edisgo.topology.storage_units_df.copy()
    if stor_df.empty:
        return

    timeindex = edisgo.timeseries.timeindex

    if freq is None:
        freq = (
            (timeindex[1] - timeindex[0]).total_seconds() / 3600
            if len(timeindex) > 1
            else 1.0
        )

    for col in ("efficiency_store", "efficiency_dispatch"):
        if col not in stor_df.columns:
            stor_df[col] = float("nan")
        mask = stor_df[col].isna()
        if mask.any():
            logger.warning(
                f"'{col}' not set for storage units {stor_df.index[mask].tolist()}. "
                "Defaulting to 0.95."
            )
            stor_df.loc[mask, col] = 0.95

    gen_df = edisgo.topology.generators_df
    load_df = edisgo.topology.loads_df
    gen_ts = edisgo.timeseries.generators_active_power
    load_ts = edisgo.timeseries.loads_active_power

    def _bus_sum(component_df, ts):
        result = {}
        for bus, group in component_df.groupby("bus"):
            cols = [c for c in group.index if c in ts.columns]
            result[bus] = ts[cols].sum(axis=1) if cols else pd.Series(0.0, index=timeindex)
        return result

    all_gen_at_bus = _bus_sum(gen_df, gen_ts)
    demand_at_bus = _bus_sum(load_df, load_ts)

    storage_ts_dict = {}
    for bus, bus_stor in stor_df.groupby("bus"):
        net_signal = all_gen_at_bus.get(
            bus, pd.Series(0.0, index=timeindex)
        ) - demand_at_bus.get(bus, pd.Series(0.0, index=timeindex))

        if all_gen_at_bus.get(bus, pd.Series(0.0, index=timeindex)).abs().sum() == 0:
            logger.warning(
                f"Bus {bus} has no generation. "
                f"Storage units {bus_stor.index.tolist()} will remain at zero dispatch."
            )

        total_p_nom = bus_stor["p_nom"].sum()
        for stor_name, stor_data in bus_stor.iterrows():
            if pd.isna(stor_data["max_hours"]):
                raise ValueError(
                    f"'max_hours' not set for storage unit {stor_name}."
                )
            scale = stor_data["p_nom"] / total_p_nom
            result = _reference_operation(
                df=pd.DataFrame({"feedin_minus_demand": net_signal * scale}),
                soe_init=soe_init,
                soe_max=stor_data["p_nom"] * stor_data["max_hours"],
                storage_p_nom=stor_data["p_nom"],
                freq=freq,
                efficiency_store=stor_data["efficiency_store"],
                efficiency_dispatch=stor_data["efficiency_dispatch"],
            )
            storage_ts_dict[stor_name] = result["storage_power"]

    edisgo.timeseries.storage_units_active_power = pd.DataFrame(
        storage_ts_dict, index=timeindex
    )

    edisgo.set_time_series_reactive_power_control(
        generators_parametrisation=None,
        loads_parametrisation=None,
        storage_units_parametrisation=pd.DataFrame(
            {
                "components": [stor_df.index.tolist()],
                "mode": ["default"],
                "power_factor": ["default"],
            },
            index=[1],
        ),
    )


# ==========================
# §14a visualization and curtailment utilities
# ==========================


def get_curtailment_data(edisgo):
    """
    Return the §14a virtual generator curtailment time series.

    Returns a DataFrame of generators_active_power columns corresponding to
    hp_14a_support and cp_14a_support virtual generators, transposed so that
    the index is the generator name (ready for bus mapping).
    """
    gen_cols = [
        col
        for col in edisgo.timeseries.generators_active_power.columns
        if "hp_14a_support" in col
        or "cp_14a_support" in col
        or "charging_point_14a_support" in col
    ]
    return edisgo.timeseries.generators_active_power[gen_cols]


def create_network_gif(
    folder_path="./plots", output_name="network_evolution.gif", duration=1
):
    """
    Creates a GIF from PNG files in a folder.
    duration: time in seconds between frames
    """
    images = []

    files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".png") and f.startswith("grid_analysis_")
    ]
    files.sort()

    print(f"Found {len(files)} frames. Processing...")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(file_path))
        print(f"Added: {filename}")

    imageio.mimsave(output_name, images, duration=duration, loop=0)
    print(f"Success! GIF saved as {output_name}")


def plot_network(
    edisgo,
    snapshot: str = "2035-01-15 09:00:00",
    show: bool = True,
    save: bool = True,
    base_bus_size=0.000000002,
):
    results = edisgo.results

    n = edisgo.to_pypsa()

    coords = edisgo.topology.buses_df[["x", "y"]]
    coords = coords.reindex(n.buses.index)
    n.buses["x"] = coords["x"].values
    n.buses["y"] = coords["y"].values

    line_columns = n.lines.index
    loading_relative = results.s_res.loc[snapshot, line_columns] / n.lines.s_nom

    v_min, v_max = 0.0, 1.0
    norm_lines = mcolors.Normalize(vmin=v_min, vmax=v_max)

    bus_colors = edisgo.results.v_res.T[snapshot]

    norm_buses = mcolors.TwoSlopeNorm(vmin=0.9, vcenter=1.0, vmax=1.1)
    voltage_cmap = mcolors.LinearSegmentedColormap.from_list(
        "voltage",
        [(0.0, "navy"), (0.35, "dodgerblue"), (0.5, "limegreen"), (0.65, "orangered"), (1.0, "darkred")],
    )

    curt_14a = get_curtailment_data(edisgo).T
    curt_14a["load"] = curt_14a.index
    curt_14a["load"] = curt_14a["load"].apply(
        lambda x: x.replace("cp_14a_support_", "").replace("hp_14a_support_", "")
    )
    curt_14a["bus"] = curt_14a["load"].map(edisgo.topology.loads_df["bus"])
    grouped_14a = curt_14a.groupby("bus").sum()
    grouped_14a.columns = grouped_14a.columns.map(str)

    bus_sizes = base_bus_size + (grouped_14a[snapshot] * 0.000001)
    bus_sizes = bus_sizes.reindex(bus_colors.index, fill_value=base_bus_size)

    fig, ax = plt.subplots(figsize=(12, 8))

    n.plot(
        margin=0.05,
        ax=ax,
        geomap=False,
        bus_colors=bus_colors,
        bus_alpha=1,
        bus_sizes=bus_sizes,
        bus_cmap=voltage_cmap,
        bus_norm=norm_buses,
        line_colors=loading_relative,
        line_widths=1.6,
        line_cmap="jet",
        line_norm=norm_lines,
        title=f"Grid Analysis: {snapshot}",
        geometry=False,
    )

    ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)

    sm_lines = plt.cm.ScalarMappable(cmap="jet", norm=norm_lines)
    cb_lines = fig.colorbar(
        sm_lines, ax=ax, orientation="vertical", location="left", pad=0.08, aspect=20
    )
    cb_lines.set_label("Line Loading [relative]", fontsize=8)

    sm_buses = plt.cm.ScalarMappable(cmap=voltage_cmap, norm=norm_buses)
    cb_buses = fig.colorbar(
        sm_buses, ax=ax, orientation="vertical", location="right", pad=0.02, aspect=20
    )
    cb_buses.set_label(
        "Bus Voltage [p.u.]  — navy: under, green: nominal, red: over", fontsize=8
    )

    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/grid_analysis_{snapshot}.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def plot_storage_dispatch(
    edisgo, day: str = None, show: bool = True, save: bool = True
):
    """
    Plot total solar + wind generation against total storage charge/discharge.

    Two subplots sharing the x-axis:
      - Top: stacked solar and wind generation.
      - Bottom: storage discharge (positive) and charging (negative, shown below zero).

    Parameters
    ----------
    edisgo : EDisGo
    day : str or None
        Date string e.g. "2035-01-15". If None, all snapshots are shown.
    show : bool
    save : bool
    """
    ti = edisgo.timeseries.timeindex
    if day is not None:
        ti = ti[ti.normalize() == pd.Timestamp(day)]

    gen_df = edisgo.topology.generators_df
    gen_ts = edisgo.timeseries.generators_active_power
    stor_ts = edisgo.timeseries.storage_units_active_power

    solar_cols = [
        g
        for g in gen_df[gen_df["type"] == "solar"].index
        if g in gen_ts.columns
    ]
    wind_cols = [
        g
        for g in gen_df[gen_df["type"] == "wind"].index
        if g in gen_ts.columns
    ]
    solar = gen_ts[solar_cols].loc[ti].sum(axis=1)
    wind = gen_ts[wind_cols].loc[ti].sum(axis=1)

    if stor_ts is not None and not stor_ts.empty:
        stor = stor_ts.loc[ti].sum(axis=1)
    else:
        stor = pd.Series(0.0, index=ti)
    discharge = stor.clip(lower=0)
    charge = stor.clip(upper=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.fill_between(ti, 0, solar.values, alpha=0.75, color="gold", label="Solar")
    ax1.fill_between(
        ti,
        solar.values,
        (solar + wind).values,
        alpha=0.75,
        color="steelblue",
        label="Wind",
    )
    ax1.set_ylabel("Generation [MW]")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(
        ti, 0, discharge.values, alpha=0.75, color="darkorange", label="Discharge"
    )
    ax2.fill_between(
        ti, 0, charge.values, alpha=0.75, color="mediumpurple", label="Charge"
    )
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Storage Power [MW]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    if day is not None:
        ax2.set_xlabel("Hour of day")
        fig.autofmt_xdate(rotation=0)
        ax2.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%H:%M")
        )
    else:
        ax2.set_xlabel("Timestamp")
        fig.autofmt_xdate(rotation=45)

    title = f"Generation vs Storage Dispatch — {day if day else 'all snapshots'}"
    fig.suptitle(title)
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        label = day if day else "all"
        plt.savefig(
            f"plots/storage_dispatch_{label}.png", dpi=300, bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close()


def plot_load_before_after(edisgo, day: str, show: bool = True, save: bool = True):
    """
    Plot aggregate CP + HP load before and after §14a curtailment for a 24h day.

    Parameters
    ----------
    day : str
        Date string, e.g. "2035-01-15".
    """
    ti = edisgo.timeseries.timeindex
    day_ti = ti[ti.normalize() == pd.Timestamp(day)]

    loads_df = edisgo.topology.loads_df
    cp_loads = loads_df[loads_df["type"] == "charging_point"].index
    hp_loads = loads_df[loads_df["type"] == "heat_pump"].index
    conv_loads = loads_df[loads_df["type"] == "conventional_load"].index

    lap = edisgo.timeseries.loads_active_power
    cp_ts = lap[[c for c in cp_loads if c in lap.columns]].loc[day_ti].sum(axis=1)
    hp_ts = lap[[c for c in hp_loads if c in lap.columns]].loc[day_ti].sum(axis=1)
    conv_ts = lap[[c for c in conv_loads if c in lap.columns]].loc[day_ti].sum(axis=1)

    curt = get_curtailment_data(edisgo).loc[day_ti]
    cp_curt = curt[
        [
            c
            for c in curt.columns
            if "cp_14a_support" in c or "charging_point_14a_support" in c
        ]
    ].sum(axis=1)
    hp_curt = curt[[c for c in curt.columns if "hp_14a_support" in c]].sum(axis=1)

    cp_opt = cp_ts - cp_curt
    hp_opt = hp_ts - hp_curt

    stack_conv = conv_ts.values
    stack_conv_hp = (conv_ts + hp_opt).values
    stack_conv_hp_cp = (conv_ts + hp_opt + cp_opt).values
    stack_with_hp_curt = (conv_ts + hp_ts + cp_opt).values
    original_total = (conv_ts + hp_ts + cp_ts).values

    hours = [t.hour for t in day_ti]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(
        hours, 0, stack_conv, alpha=0.6, color="gray", label="Conventional load"
    )
    ax.fill_between(
        hours,
        stack_conv,
        stack_conv_hp,
        alpha=0.6,
        color="mediumseagreen",
        label="Heat pumps (optimized)",
    )
    ax.fill_between(
        hours,
        stack_conv_hp,
        stack_conv_hp_cp,
        alpha=0.6,
        color="steelblue",
        label="Charging points (optimized)",
    )
    ax.fill_between(
        hours,
        stack_conv_hp_cp,
        stack_with_hp_curt,
        alpha=0.55,
        color="mediumseagreen",
        label="§14a HP curtailment",
        hatch="////",
        edgecolor="darkgreen",
    )
    ax.fill_between(
        hours,
        stack_with_hp_curt,
        original_total,
        alpha=0.55,
        color="steelblue",
        label="§14a CP curtailment",
        hatch="////",
        edgecolor="darkblue",
    )
    ax.plot(
        hours,
        original_total,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Original total (unoptimized)",
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Active Power [MW]")
    ax.set_title(f"§14a Impact on Load by Type — {day}")
    ax.set_xticks(range(24))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/load_before_after_{day}.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
