import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

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
    #tol_2=0.25,
    tol_2=0.9,
    # radius_3=2000.0,
    # tol_3=0.50,
    # radius_4=2000.0,
    # tol_4=0.9,
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
        # radius_3=radius_3,
        # tol_3=tol_3,
        # radius_4=radius_4,
        # tol_4=tol_4,
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
# def _match_four_step_nearest(
#     existing_m: gpd.GeoDataFrame,
#     new_m: gpd.GeoDataFrame,
#     *,
#     radius_1=2000.0,
#     tol_1=0.15,
#     radius_2=2000.0,
#     tol_2=0.25,
#     radius_3=2000.0,
#     tol_3=0.50,
#     radius_4=2000.0,
#     tol_4=0.90,
# ):

#     new_ids = np.array(new_m.index.to_list())
#     new_geoms = np.array(new_m.geometry.values, dtype=object)

#     tree = STRtree(new_geoms)

#     new_pset = new_m["p_set"].astype(float)
#     ex_pset = existing_m["p_set"].astype(float)

#     used_new = set()
#     phase2_matches = []
#     phase3_matches = []
#     phase4_matches = []

#     def _pass(existing_ids, max_radius_m, pset_tol, phase):
#         pass_matches = []
#         pass_unmatched = []

#         for ex_id in existing_ids:
#             ex_geom = existing_m.loc[ex_id].geometry
#             p_ex = float(ex_pset.loc[ex_id])

#             idxs = tree.query(ex_geom.buffer(max_radius_m))
#             if idxs is None or len(idxs) == 0:
#                 pass_unmatched.append(ex_id)
#                 continue

#             p_lo, p_hi = p_ex * (1 - pset_tol), p_ex * (1 + pset_tol)

#             best = None
#             for j in idxs:
#                 new_id = new_ids[j]

#                 if new_id in used_new:
#                     continue

#                 p_new = float(new_pset.loc[new_id])
#                 if not (p_lo <= p_new <= p_hi):
#                     continue

#                 dist = float(ex_geom.distance(new_geoms[j]))

#                 if (best is None) or (dist < best[0]):
#                     best = (dist, new_id)

#             if best is None:
#                 pass_unmatched.append(ex_id)
#             else:
#                 dist_best, new_best = best
#                 used_new.add(new_best)

#                 tup = (ex_id, new_best, dist_best)
#                 pass_matches.append(tup)

#                 if phase == 2:
#                     phase2_matches.append(tup)
#                 elif phase == 3:
#                     phase3_matches.append(tup)
#                 elif phase == 4:
#                     phase4_matches.append(tup)

#         return pass_matches, pass_unmatched

#     # ---- Phase 1 ----
#     m1, u1 = _pass(list(existing_m.index), radius_1, tol_1, phase=1)

#     # ---- Phase 2 ----
#     m2, u2 = _pass(u1, radius_2, tol_2, phase=2)

#     # ---- Phase 3 ----
#     m3, u3 = _pass(u2, radius_3, tol_3, phase=3)

#     # ---- Phase 4 ----
#     m4, u4 = _pass(u3, radius_4, tol_4, phase=4)

#     matches = m1 + m2 + m3 + m4
#     unused_new = [i for i in new_ids.tolist() if i not in used_new]

#     def _log_phase(phase_name, phase_matches, tol):
#         print(f"[EV] {phase_name} matches (p_set tol={tol}): {len(phase_matches)}")
#         for i, (ex_id, new_id, dist) in enumerate(phase_matches, 1):
#             bus = existing_m.loc[ex_id, "bus"]
#             p_ex = float(existing_m.loc[ex_id, "p_set"])
#             p_new = float(new_m.loc[new_id, "p_set"])
#             rel_dev = (p_new / p_ex - 1.0) if p_ex != 0 else float('nan')
#             print(
#                 f"[EV] [{phase_name.upper()}][{i}] "
#                 f"{ex_id} (bus={bus}, p_set_old={p_ex:.6f}) "
#                 f"-> {new_id} (p_set_new={p_new:.6f})  "
#                 f"dist={dist:.1f} m  rel_dev={rel_dev:+.2%}"
#             )
            
#     _log_phase("Phase 2", phase2_matches, tol_2)
#     _log_phase("Phase 3", phase3_matches, tol_3)
#     _log_phase("Phase 4", phase4_matches, tol_4)
    
#     # # Phase 2 logging
#     # print(f"[EV] Phase 2 matches (p_set tol={tol_2}): {len(phase2_matches)}")
#     # for i, (ex_id, new_id, dist) in enumerate(phase2_matches, 1):
#     #     bus = existing_m.loc[ex_id, "bus"]
#     #     print(f"[EV] [PHASE 2][{i}] {ex_id} (bus={bus}) -> {new_id}  dist={dist:.1f} m")

#     # # Phase 3 logging
#     # print(f"[EV] Phase 3 matches (p_set tol={tol_3}): {len(phase3_matches)}")
#     # for i, (ex_id, new_id, dist) in enumerate(phase3_matches, 1):
#     #     bus = existing_m.loc[ex_id, "bus"]
#     #     print(f"[EV] [PHASE 3][{i}] {ex_id} (bus={bus}) -> {new_id}  dist={dist:.1f} m")

#     # # Phase 4 logging
#     # print(f"[EV] Phase 4 matches (p_set tol={tol_4}): {len(phase4_matches)}")
#     # for i, (ex_id, new_id, dist) in enumerate(phase4_matches, 1):
#     #     bus = existing_m.loc[ex_id, "bus"]
#     #     print(f"[EV] [PHASE 4][{i}] {ex_id} (bus={bus}) -> {new_id}  dist={dist:.1f} m")

#     # Unmatched logging
#     print(f"[EV] Unmatched existing CPs after Phase 4: {len(u4)}")
#     for i, ex_id in enumerate(u4, 1):
#         bus = existing_m.loc[ex_id, "bus"]
#         print(f"[EV] [UNMATCHED][{i}] {ex_id} (bus={bus})")

#     return matches, u4, unused_new


# def _match_four_step_nearest(
#     existing_m: gpd.GeoDataFrame,
#     new_m: gpd.GeoDataFrame,
#     *,
#     radius_1=2000.0,
#     tol_1=0.15,
#     radius_2=2000.0,
#     tol_2=0.25,
#     radius_3=2000.0,
#     tol_3=0.50,
#     radius_4=2000.0,
#     tol_4=0.90,
# ):
#     new_ids = np.array(new_m.index.to_list())
#     new_geoms = np.array(new_m.geometry.values, dtype=object)

#     tree = STRtree(new_geoms)

#     new_pset = new_m["p_set"].astype(float)
#     ex_pset = existing_m["p_set"].astype(float)

#     used_new = set()
#     phase2_matches = []
#     phase3_matches = []
#     phase4_matches = []

#     def _pass(existing_ids, max_radius_m, pset_tol, phase):
#         pass_matches = []
#         pass_unmatched = []

#         for ex_id in existing_ids:
#             ex_geom = existing_m.loc[ex_id].geometry
#             p_ex = float(ex_pset.loc[ex_id])

#             idxs = tree.query(ex_geom.buffer(max_radius_m))
#             if idxs is None or len(idxs) == 0:
#                 pass_unmatched.append(ex_id)
#                 continue

#             p_lo, p_hi = p_ex * (1 - pset_tol), p_ex * (1 + pset_tol)

#             best = None
#             for j in idxs:
#                 new_id = new_ids[j]

#                 if new_id in used_new:
#                     continue

#                 p_new = float(new_pset.loc[new_id])
#                 if not (p_lo <= p_new <= p_hi):
#                     continue

#                 dist = float(ex_geom.distance(new_geoms[j]))

#                 if dist > max_radius_m:
#                     continue

#                 if (best is None) or (dist < best[0]):
#                     best = (dist, new_id)

#             if best is None:
#                 pass_unmatched.append(ex_id)
#             else:
#                 dist_best, new_best = best
#                 used_new.add(new_best)

#                 tup = (ex_id, new_best, dist_best)
#                 pass_matches.append(tup)

#                 if phase == 2:
#                     phase2_matches.append(tup)
#                 elif phase == 3:
#                     phase3_matches.append(tup)
#                 elif phase == 4:
#                     phase4_matches.append(tup)

#         return pass_matches, pass_unmatched

#     m1, u1 = _pass(list(existing_m.index), radius_1, tol_1, phase=1)
#     m2, u2 = _pass(u1, radius_2, tol_2, phase=2)
#     m3, u3 = _pass(u2, radius_3, tol_3, phase=3)
#     m4, u4 = _pass(u3, radius_4, tol_4, phase=4)

#     matches = m1 + m2 + m3 + m4
#     unused_new = [i for i in new_ids.tolist() if i not in used_new]

#     def _log_phase(phase_name, phase_matches, tol):
#         print(f"[EV] {phase_name} matches (p_set tol={tol}): {len(phase_matches)}")
#         for i, (ex_id, new_id, dist) in enumerate(phase_matches, 1):
#             bus = existing_m.loc[ex_id, "bus"]
#             p_ex = float(existing_m.loc[ex_id, "p_set"])
#             p_new = float(new_m.loc[new_id, "p_set"])
#             rel_dev = (p_new / p_ex - 1.0) if p_ex != 0 else float("nan")

#             print(
#                 f"[EV] [{phase_name.upper()}][{i}] "
#                 f"{ex_id} (bus={bus}, p_set_old={p_ex:.6f}) "
#                 f"-> {new_id} (p_set_new={p_new:.6f})  "
#                 f"dist={dist:.1f} m  rel_dev={rel_dev:+.2%}"
#             )

#     _log_phase("Phase 2", phase2_matches, tol_2)
#     _log_phase("Phase 3", phase3_matches, tol_3)
#     _log_phase("Phase 4", phase4_matches, tol_4)

#     print(f"[EV] Unmatched existing CPs after Phase 4: {len(u4)}")
#     for i, ex_id in enumerate(u4, 1):
#         bus = existing_m.loc[ex_id, "bus"]
#         p_ex = float(existing_m.loc[ex_id, "p_set"])
#         print(f"[EV] [UNMATCHED][{i}] {ex_id} (bus={bus}, p_set_old={p_ex:.6f})")

#     return matches, u4, unused_new

# # --------------------------
# # Matching EXISTING -> NEW by nearest bus location + p_set tolerance
# # Phase 1: greedy
# # Phase 2-4: global assignment on remaining candidates
# # --------------------------
# def _match_four_step_nearest(
#     existing_m: gpd.GeoDataFrame,
#     new_m: gpd.GeoDataFrame,
#     *,
#     radius_1=2000.0,
#     tol_1=0.15,
#     radius_2=2000.0,
#     tol_2=0.25,
#     radius_3=2000.0,
#     tol_3=0.50,
#     radius_4=2000.0,
#     tol_4=0.90,
# ):
#     from scipy.optimize import linear_sum_assignment

#     new_ids = np.array(new_m.index.to_list(), dtype=object)
#     new_geoms = np.array(new_m.geometry.values, dtype=object)

#     tree = STRtree(new_geoms)

#     new_pset = new_m["p_set"].astype(float)
#     ex_pset = existing_m["p_set"].astype(float)

#     used_new = set()

#     phase1_matches = []
#     phase2_matches = []
#     phase3_matches = []
#     phase4_matches = []

#     def _candidate_pairs(existing_ids, max_radius_m, pset_tol):
#         """
#         Build all admissible candidate pairs:
#         existing_id -> new_id within radius and p_set tolerance, excluding already used new_ids.
#         """
#         pairs = []

#         for ex_id in existing_ids:
#             ex_geom = existing_m.loc[ex_id].geometry
#             p_ex = float(ex_pset.loc[ex_id])

#             idxs = tree.query(ex_geom.buffer(max_radius_m))
#             if idxs is None or len(idxs) == 0:
#                 continue

#             p_lo, p_hi = p_ex * (1 - pset_tol), p_ex * (1 + pset_tol)

#             for j in idxs:
#                 new_id = new_ids[j]

#                 if new_id in used_new:
#                     continue

#                 p_new = float(new_pset.loc[new_id])
#                 if not (p_lo <= p_new <= p_hi):
#                     continue

#                 dist = float(ex_geom.distance(new_geoms[j]))
#                 rel_dev = abs(p_new / p_ex - 1.0) if p_ex != 0 else 0.0

#                 pairs.append((ex_id, new_id, dist, rel_dev))

#         return pairs

#     def _greedy_pass(existing_ids, max_radius_m, pset_tol, phase_name):
#         """
#         Greedy nearest match, used only for Phase 1.
#         """
#         pass_matches = []
#         pass_unmatched = []

#         for ex_id in existing_ids:
#             ex_geom = existing_m.loc[ex_id].geometry
#             p_ex = float(ex_pset.loc[ex_id])

#             idxs = tree.query(ex_geom.buffer(max_radius_m))
#             if idxs is None or len(idxs) == 0:
#                 pass_unmatched.append(ex_id)
#                 continue

#             p_lo, p_hi = p_ex * (1 - pset_tol), p_ex * (1 + pset_tol)

#             best = None
#             for j in idxs:
#                 new_id = new_ids[j]

#                 if new_id in used_new:
#                     continue

#                 p_new = float(new_pset.loc[new_id])
#                 if not (p_lo <= p_new <= p_hi):
#                     continue

#                 dist = float(ex_geom.distance(new_geoms[j]))
#                 rel_dev = abs(p_new / p_ex - 1.0) if p_ex != 0 else 0.0

#                 # primary weight: distance, secondary: p_set deviation
#                 dist_norm = dist / max_radius_m
#                 pset_norm = rel_dev / pset_tol if pset_tol > 0 else rel_dev
#                 score = 0.8 * dist_norm + 0.2 * pset_norm
#                 cand = (score, dist, new_id)

#                 if best is None or cand < best:
#                     best = cand

#             if best is None:
#                 pass_unmatched.append(ex_id)
#             else:
#                 dist_best, rel_dev_best, new_best = best
#                 used_new.add(new_best)
#                 tup = (ex_id, new_best, dist_best)
#                 pass_matches.append(tup)

#         return pass_matches, pass_unmatched

#     def _global_pass(existing_ids, max_radius_m, pset_tol, phase_name):
#         """
#         Global bipartite assignment for remaining unmatched existing CPs.
#         Objective: primarily minimize p_set deviation, secondarily distance.
#         """
#         pairs = _candidate_pairs(existing_ids, max_radius_m, pset_tol)

#         if len(existing_ids) == 0:
#             return [], []

#         if len(pairs) == 0:
#             return [], list(existing_ids)

#         ex_list = list(existing_ids)
#         new_list = sorted({new_id for _, new_id, _, _ in pairs})

#         ex_pos = {ex_id: i for i, ex_id in enumerate(ex_list)}
#         new_pos = {new_id: j for j, new_id in enumerate(new_list)}

#         BIG = 1e12
#         # weight distance strongly, rel_dev only as tie-breaker
#         cost = np.full((len(ex_list), len(new_list)), BIG, dtype=float)

#         for ex_id, new_id, dist, rel_dev in pairs:
#             i = ex_pos[ex_id]
#             j = new_pos[new_id]
#             # primary weight: p_set deviation, secondary: distance
#             dist_norm = dist / max_radius_m
#             pset_norm = rel_dev / pset_tol if pset_tol > 0 else rel_dev
#             score = 0.3 * dist_norm + 0.7 * pset_norm
#             if score < cost[i, j]:
#                 cost[i, j] = score

#         row_ind, col_ind = linear_sum_assignment(cost)

#         pass_matches = []
#         matched_existing = set()

#         for i, j in zip(row_ind, col_ind):
#             if cost[i, j] >= BIG:
#                 continue

#             ex_id = ex_list[i]
#             new_id = new_list[j]

#             ex_geom = existing_m.loc[ex_id].geometry
#             new_geom = new_m.loc[new_id].geometry
#             dist = float(ex_geom.distance(new_geom))

#             used_new.add(new_id)
#             matched_existing.add(ex_id)
#             pass_matches.append((ex_id, new_id, dist))

#         pass_unmatched = [ex_id for ex_id in ex_list if ex_id not in matched_existing]
#         return pass_matches, pass_unmatched

#     def _log_phase(phase_name, phase_matches, tol):
#         print(f"[EV] {phase_name} matches (p_set tol={tol}): {len(phase_matches)}")
#         for i, (ex_id, new_id, dist) in enumerate(phase_matches, 1):
#             bus = existing_m.loc[ex_id, "bus"]
#             p_ex = float(existing_m.loc[ex_id, "p_set"])
#             p_new = float(new_m.loc[new_id, "p_set"])
#             rel_dev = (p_new / p_ex - 1.0) if p_ex != 0 else float("nan")
#             print(
#                 f"[EV] [{phase_name.upper()}][{i}] "
#                 f"{ex_id} (bus={bus}, p_set_old={p_ex:.6f}) "
#                 f"-> {new_id} (p_set_new={p_new:.6f})  "
#                 f"dist={dist:.1f} m  rel_dev={rel_dev:+.2%}"
#             )

#     # ---- Phase 1: greedy ----
#     m1, u1 = _greedy_pass(list(existing_m.index), radius_1, tol_1, "Phase 1")
#     phase1_matches.extend(m1)

#     # ---- Phase 2: global ----
#     m2, u2 = _global_pass(u1, radius_2, tol_2, "Phase 2")
#     phase2_matches.extend(m2)

#     # ---- Phase 3: global ----
#     m3, u3 = _global_pass(u2, radius_3, tol_3, "Phase 3")
#     phase3_matches.extend(m3)

#     # ---- Phase 4: global ----
#     m4, u4 = _global_pass(u3, radius_4, tol_4, "Phase 4")
#     phase4_matches.extend(m4)

#     matches = m1 + m2 + m3 + m4
#     unused_new = [i for i in new_ids.tolist() if i not in used_new]

#     _log_phase("Phase 2", phase2_matches, tol_2)
#     _log_phase("Phase 3", phase3_matches, tol_3)
#     _log_phase("Phase 4", phase4_matches, tol_4)

#     print(f"[EV] Unmatched existing CPs after Phase 4: {len(u4)}")
#     for i, ex_id in enumerate(u4, 1):
#         bus = existing_m.loc[ex_id, "bus"]
#         p_ex = float(existing_m.loc[ex_id, "p_set"])
#         print(f"[EV] [UNMATCHED][{i}] {ex_id} (bus={bus}, p_set_old={p_ex:.6f})")

#     return matches, u4, unused_new

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
    new_geoms = np.array(new_m.geometry.values, dtype=object)

    tree = STRtree(new_geoms)

    new_pset = new_m["p_set"].astype(float)
    ex_pset = existing_m["p_set"].astype(float)

    used_new = set()

    phase1_matches = []
    phase2_matches = []

    def _candidate_pairs(existing_ids, max_radius_m, pset_tol):
        """
        Build all admissible candidate pairs for the current phase.

        Note:
        tree.query(ex_geom.buffer(max_radius_m)) is used only as a coarse
        spatial preselection via bounding-box intersection.
        We intentionally do NOT apply a hard dist <= max_radius_m cutoff here,
        because slightly more distant candidates are allowed as fallback.
        """
        pairs = []

        for ex_id in existing_ids:
            ex_geom = existing_m.loc[ex_id].geometry
            p_ex = float(ex_pset.loc[ex_id])

            idxs = tree.query(ex_geom.buffer(max_radius_m))
            if idxs is None or len(idxs) == 0:
                continue

            p_lo, p_hi = p_ex * (1 - pset_tol), p_ex * (1 + pset_tol)

            for j in idxs:
                new_id = new_ids[j]

                if new_id in used_new:
                    continue

                p_new = float(new_pset.loc[new_id])
                if not (p_lo <= p_new <= p_hi):
                    continue

                dist = float(ex_geom.distance(new_geoms[j]))
                rel_dev = abs(p_new / p_ex - 1.0) if p_ex != 0 else 0.0

                pairs.append((ex_id, new_id, dist, rel_dev))

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
            ex_geom = existing_m.loc[ex_id].geometry
            p_ex = float(ex_pset.loc[ex_id])

            idxs = tree.query(ex_geom.buffer(max_radius_m))
            if idxs is None or len(idxs) == 0:
                pass_unmatched.append(ex_id)
                continue

            p_lo, p_hi = p_ex * (1 - pset_tol), p_ex * (1 + pset_tol)

            best = None
            for j in idxs:
                new_id = new_ids[j]

                if new_id in used_new:
                    continue

                p_new = float(new_pset.loc[new_id])
                if not (p_lo <= p_new <= p_hi):
                    continue

                dist = float(ex_geom.distance(new_geoms[j]))
                rel_dev = abs(p_new / p_ex - 1.0) if p_ex != 0 else 0.0

                # distance-dominated in phase 1
                cand = (dist, rel_dev, new_id)

                if best is None or cand < best:
                    best = cand

            if best is None:
                pass_unmatched.append(ex_id)
            else:
                dist_best, rel_dev_best, new_best = best
                used_new.add(new_best)
                pass_matches.append((ex_id, new_best, dist_best))

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

            ex_geom = existing_m.loc[ex_id].geometry
            new_geom = new_m.loc[new_id].geometry
            dist = float(ex_geom.distance(new_geom))

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

            edisgo.remove_component("load", new_id)

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


def _select_keep_ids_remove_marked_last(
    candidate_ids,
    *,
    target_total,
    seed=42,
    marker="Existing",
):
    """
    Select IDs to keep such that IDs containing `marker` are removed last.

    Logic:
    - first keep all marked IDs if possible (e.g. existing charging points)
    - remaining slots are filled randomly from unmarked IDs
    - if target_total is smaller than number of marked IDs,
      keep a random subset of marked IDs only

    Parameters
    ----------
    candidate_ids : list[str]
    target_total : int
    seed : int
    marker : str | None

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

    marked_ids, unmarked_ids = split_ids_by_marker(candidate_ids, marker=marker)

    # No marker logic available -> fully random
    if marker is None or len(marked_ids) == 0:
        return rng.choice(np.array(candidate_ids), size=target_total, replace=False).tolist()

    # If target smaller than number of marked IDs, only marked IDs survive
    if target_total <= len(marked_ids):
        return rng.choice(np.array(marked_ids), size=target_total, replace=False).tolist()

    keep_ids = list(marked_ids)
    remaining = target_total - len(marked_ids)

    if remaining >= len(unmarked_ids):
        keep_ids.extend(unmarked_ids)
    elif remaining > 0:
        keep_ids.extend(
            rng.choice(np.array(unmarked_ids), size=remaining, replace=False).tolist()
        )

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
    Remove specified loads from topology and time series.
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

    p_cols = [i for i in remove_ids if i in edisgo.timeseries.loads_active_power.columns]
    q_cols = [i for i in remove_ids if i in edisgo.timeseries.loads_reactive_power.columns]

    if p_cols:
        edisgo.timeseries.drop_component_time_series("loads_active_power", p_cols)

    if q_cols:
        edisgo.timeseries.drop_component_time_series("loads_reactive_power", q_cols)

    for i in remove_ids:
        if i in edisgo.topology.loads_df.index:
            edisgo.remove_component("load", i)

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
    removal_marker=None,
):
    """
    Set final number of loads of one type to a target value.

    Supports:
    - absolute control via target_total
    - relative control via percentage

    Increase logic:
    - duplicates are created from the full current population
    - no existing/new priority is applied

    Reduction logic: remove_marked_last
    - default (False): fully random removal
    - optional (True): IDs containing `removal_marker` are removed last

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
        If True, IDs containing `removal_marker` are kept as long as possible
        during reduction
    removal_marker : str | None
        Marker used only for reduction prioritization
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
            keep_ids = _select_keep_ids_remove_marked_last(
                current_ids,
                target_total=desired_total,
                seed=seed,
                marker=removal_marker,
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
    existing_marker="Existing",
    remove_existing_last=True,
    avoid_source_bus=False,
    add_tracking_columns=False,
    export_removed=False,
    export_dir=None,
):
    """
    Set charging points to a final target count.

    Important:
    - when increasing: no existing/new priority is applied
    - when reducing: CPs whose IDs contain `existing_marker`
      are removed last
    """
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
        removal_marker=existing_marker,
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
        removal_marker=None,
    )
