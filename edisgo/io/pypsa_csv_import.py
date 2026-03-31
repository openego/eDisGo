"""
pypsa_csv_import.py
===================
Populate an eDisGo object from a PyPSA network exported via
``network.export_to_csv_folder()``.

This module is called internally by ``EDisGo.import_pypsa_csv()``, which is
triggered from ``EDisGo.__init__()`` when the ``pypsa_csv_dir`` kwarg is
passed:

    edisgo_obj = EDisGo(pypsa_csv_dir="/path/to/csv_folder")

Column mappings and component transformations follow the structure validated
in ``adjust_network_shape()`` (see project history).

Tested against:
  - PyPSA  >= 0.26  (CSV export version 1.1.2)
  - eDisGo >= 0.3   (topology + timeseries attribute layout)
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer as CrsTransformer

from edisgo.network.grids import MVGrid
from edisgo.network.timeseries import TimeSeries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-name mappings: PyPSA name → eDisGo name
#
# Only columns consumed by eDisGo setters are listed; everything else is
# handled explicitly in the body of populate_edisgo_from_pypsa_csv().
# ---------------------------------------------------------------------------

# Buses: coordinate reprojection and extra fields are added separately.
BUSES_RENAME: dict[str, str] = {
    "v_nom":   "v_nom",
    "carrier": "carrier",
}

LINES_RENAME: dict[str, str] = {
    "bus0":         "bus0",
    "bus1":         "bus1",
    "x":            "x",
    "r":            "r",
    "b":            "b",
    "s_nom":        "s_nom",
    "length":       "length",
    "num_parallel": "num_parallel",
    "carrier":      "carrier",
    "type":         "type",
    # cable_type → type_info handled explicitly below
}

# PyPSA exports x/r in physical units; eDisGo stores them as x_pu/r_pu.
TRANSFORMERS_RENAME: dict[str, str] = {
    "bus0":  "bus0",
    "bus1":  "bus1",
    "x":     "x_pu",
    "r":     "r_pu",
    "s_nom": "s_nom",
}

GENERATORS_RENAME: dict[str, str] = {
    "bus":           "bus",
    "carrier":       "carrier",
    "p_nom":         "p_nom",
    "control":       "control",
    "marginal_cost": "marginal_cost",
    "efficiency":    "efficiency",
}

LOADS_RENAME: dict[str, str] = {
    "bus":     "bus",
    "carrier": "type",
    "p_set":   "p_set",
}

STORAGE_RENAME: dict[str, str] = {
    "bus":       "bus",
    "carrier":   "carrier",
    "p_nom":     "p_nom",
    "max_hours": "max_hours",
    "control":   "control",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(
    folder: Path, filename: str, index_col: int = 0, **kwargs
) -> pd.DataFrame:
    """Read a CSV from the export folder; return an empty DataFrame if missing."""
    p = folder / filename
    if not p.exists():
        logger.warning("File not found, skipping: %s", p)
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=index_col, **kwargs)
    logger.debug("Loaded %s  ->  %d rows, %d cols", filename, len(df), len(df.columns))
    return df


def _read_ts(
    folder: Path,
    filename: str,
    ts_slice: slice,
) -> pd.DataFrame:
    """
    Read a timeseries CSV and immediately apply *ts_slice* to rows.

    Uses ``skiprows`` / ``nrows`` so that only the requested rows are loaded
    into memory — avoids reading all 8760 rows when a subset is sufficient.
    Returns an empty DataFrame if the file does not exist.
    """
    p = folder / filename
    if not p.exists():
        logger.warning("File not found, skipping: %s", p)
        return pd.DataFrame()

    start = ts_slice.start or 0
    stop  = ts_slice.stop          # None means read to end

    # skiprows skips data rows (header is never skipped by pandas here).
    # Range(1, start+1) skips the first `start` data rows.
    skip  = list(range(1, start + 1)) if start > 0 else None
    nrows = (stop - start) if stop is not None else None

    df = pd.read_csv(p, index_col=0, skiprows=skip, nrows=nrows)
    logger.debug(
        "Loaded %s [rows %s:%s]  ->  %d rows, %d cols",
        filename, start, stop, len(df), len(df.columns),
    )
    return df


def _keep_and_rename(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    """
    Keep only columns present in *rename_map* AND in *df*, then rename them.
    Extra PyPSA-internal columns (solver artefacts, etc.) are silently dropped.
    """
    available = {k: v for k, v in rename_map.items() if k in df.columns}
    return df[list(available.keys())].rename(columns=available)


def _assign_timeseries(
    ts_df: pd.DataFrame, timeindex: pd.DatetimeIndex | None
) -> pd.DataFrame:
    """Align a raw time-series DataFrame to *timeindex* when lengths match."""
    if timeindex is not None and len(ts_df) == len(timeindex):
        ts_df.index = timeindex
    return ts_df


def _parse_timestamps(series: pd.Series) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(series.values), name="snapshot")
    return idx.freq and idx or pd.DatetimeIndex(idx, freq=pd.infer_freq(idx), name="snapshot")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def populate_edisgo_from_pypsa_csv(
    edisgo_obj,
    csv_folder: str | os.PathLike,
    mv_grid_id: int = 1,
    source_crs: str = "EPSG:32632",
    target_crs: str = "EPSG:4326",
    snapshot_range: tuple[int, int] | None = None,
) -> None:
    """
    Populate *edisgo_obj* in-place from a PyPSA CSV export folder.

    Called by ``EDisGo.import_pypsa_csv()`` during ``__init__``.

    Parameters
    ----------
    edisgo_obj : edisgo.EDisGo
        Already-constructed (but empty) EDisGo instance.
    csv_folder : str or Path
        Directory produced by ``pypsa.Network.export_to_csv_folder()``.
    mv_grid_id : int
        MV grid ID to embed across all component tables and the MVGrid object.
        Defaults to 1.
    source_crs : str
        EPSG string of the coordinate system used in the PyPSA export
        (buses x/y columns). Defaults to ``"EPSG:32632"`` (UTM zone 32N).
    target_crs : str
        EPSG string that eDisGo expects for bus coordinates.
        Defaults to ``"EPSG:4326"`` (WGS-84 lon/lat).
    snapshot_range : tuple[int, int] or None
        Inclusive ``(start, end)`` row indices into the snapshot list to load.
        Useful for quick testing without reading all 8760 rows of large
        timeseries CSVs. For example, ``snapshot_range=(120, 148)`` loads
        29 timesteps (indices 120 through 148 inclusive).
        ``None`` (default) loads all available snapshots.

    Examples
    --------
    Load the full year (default)::

        EDisGo(pypsa_csv_dir=path)

    Load a quick 29-step test window::

        EDisGo(pypsa_csv_dir=path, snapshot_range=(120, 148))
    """
    folder = Path(csv_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"PyPSA CSV folder not found: {folder}")
    # Build the row slice used for all timeseries reads.
    # snapshot_range is inclusive on both ends: (120, 148) -> slice(120, 149).
    if snapshot_range is not None:
        start, end = snapshot_range
        if start < 0 or end < start:
            raise ValueError(
                f"snapshot_range must satisfy 0 <= start <= end, got {snapshot_range}"
            )
        ts_slice = slice(start, end + 1)
        logger.info(
            "Loading snapshot subset: rows %d - %d (%d steps)",
            start, end, end - start + 1,
        )
    else:
        ts_slice = slice(None)  # all rows

    topo = edisgo_obj.topology

    # ------------------------------------------------------------------ #
    # 1. TOPOLOGY
    # ------------------------------------------------------------------ #

    # --- Buses --------------------------------------------------------- #
    buses_raw = _read(folder, "buses.csv")
    buses_raw = buses_raw[buses_raw["v_nom"] <= 20]
    buses = _keep_and_rename(buses_raw, BUSES_RENAME)

    # Reproject coordinates from source CRS to WGS-84
    crs_transformer = CrsTransformer.from_crs(source_crs, target_crs, always_xy=True)
    buses["x"], buses["y"] = crs_transformer.transform(
        buses_raw["x"].values, buses_raw["y"].values
    )

    # Parse geometry strings into Shapely objects when present
    if "geom" in buses_raw.columns:
        try:
            from shapely import wkt as shapely_wkt
            buses["geom"] = buses_raw["geom"].apply(
                lambda g: shapely_wkt.loads(g) if pd.notna(g) else None
            )
        except ImportError:
            logger.warning("shapely not installed - geom kept as plain strings.")
            buses["geom"] = buses_raw["geom"]

    # Grid ID columns (required by eDisGo internals)
    buses["mv_grid_id"] = mv_grid_id
    if "lv_grid_id" in buses_raw.columns:
        buses["lv_grid_id"] = buses_raw["lv_grid_id"]
    else:
        # MV and HV buses (identified by "MS" or "MV" in the name) get NaN;
        # all LV buses get lv_grid_id = 1.
        buses["lv_grid_id"] = buses.index.to_series().apply(
            lambda name: np.nan if ("MS" in str(name) or "MV" in str(name)) else 1
        )

    # eDisGo's buses_df setter unconditionally accesses this column
    buses["in_building"] = False

    # Pass-through metadata columns that eDisGo tolerates as extras
    for col in ("comp_type", "household_count", "trafo_cap", "location", "HP"):
        if col in buses_raw.columns:
            buses[col] = buses_raw[col]

    topo.buses_df = buses

    # --- Lines --------------------------------------------------------- #
    lines_raw = _read(folder, "lines.csv")
    lines = _keep_and_rename(lines_raw, LINES_RENAME)

    # b and num_parallel are required by edisgo's to_pypsa() but are not
    # always present in PyPSA CSV exports — provide safe defaults.
    if "b" not in lines.columns:
        lines["b"] = 0.0
        logger.warning("lines.csv has no 'b' column - defaulting to 0.0.")
    if "num_parallel" not in lines.columns:
        lines["num_parallel"] = 1
        logger.warning("lines.csv has no 'num_parallel' column - defaulting to 1.")

    # cable_type -> type_info (proven naming used by eDisGo)
    if "cable_type" in lines_raw.columns:
        lines["type_info"] = lines_raw["cable_type"]
    lines["kind"] = "cable"

    # Pass-through columns
    for col in ("comp_type", "geom", "x_pu", "r_pu"):
        if col in lines_raw.columns:
            lines[col] = lines_raw[col]

    lines["mv_grid_id"] = mv_grid_id
    topo.lines_df = lines

    # --- Transformers -------------------------------------------------- #
    # eDisGo splits LV/MV trafos (transformers_df) from the HV/MV station
    # transformer (transformers_hvmv_df), distinguished by comp_type.
    trafos_raw = _read(folder, "transformers.csv")
    trafos_base = _keep_and_rename(trafos_raw, TRANSFORMERS_RENAME)

    def _build_lv_trafos(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        df = base.copy()
        df["type_info"]  = (raw["s_nom"] * 1e3).astype(int).astype(str) + " kVA"
        df["type"]       = df["type_info"]
        df["mv_grid_id"] = mv_grid_id
        return df

    def _build_hv_trafos(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        df = base.copy()
        # HV/MV dummy trafo: impedance values are not meaningful
        df["x_pu"]       = np.nan
        df["r_pu"]       = np.nan
        df["type_info"]  = raw["s_nom"].astype(int).astype(str) + " MVA"
        df["type"]       = df["type_info"]
        df["mv_grid_id"] = mv_grid_id
        return df

    if "comp_type" in trafos_raw.columns:
        mask_hv = trafos_raw["comp_type"] == "trafo_HV"
        topo.transformers_df      = _build_lv_trafos(trafos_raw[~mask_hv], trafos_base[~mask_hv])
        topo.transformers_hvmv_df = _build_hv_trafos(trafos_raw[mask_hv],  trafos_base[mask_hv])
    else:
        logger.warning(
            "transformers.csv has no 'comp_type' column; all transformers "
            "placed in transformers_df. Verify HV/MV trafos manually."
        )
        topo.transformers_df      = _build_lv_trafos(trafos_raw, trafos_base)
        topo.transformers_hvmv_df = pd.DataFrame(columns=trafos_base.columns)

    # --- Generators ---------------------------------------------------- #
    gens_raw = _read(folder, "generators.csv")
    gens = _keep_and_rename(gens_raw, GENERATORS_RENAME)

    if "carrier" in gens_raw.columns:
        gens["type"] = gens_raw["carrier"].apply(
            lambda c: (
                "solar"         if "solar_rooftop" in c else
                "station"       if c == "AC"        else
                "conventional"
            )
        )
        gens["subtype"] = gens_raw["carrier"].apply(
            lambda c: (
                "pv_rooftop"    if "solar_rooftop" in c else
                "mv_substation" if c == "AC"        else
                "unknown"
            )
        )
    else:
        gens["type"]    = "conventional"
        gens["subtype"] = "unknown"

    gens["voltage_level"]   = gens.index.to_series().apply(
        lambda name: "mv" if "MS" in str(name) else "lv"
    )
    gens["weather_cell_id"] = None
    gens["source_id"]       = None
    gens["mv_grid_id"]      = mv_grid_id
    topo.generators_df = gens

    # --- Loads --------------------------------------------------------- #
    loads_raw = _read(folder, "loads.csv")

    loads = _keep_and_rename(loads_raw, LOADS_RENAME)

    # p_set: static baseline, overwritten with time-series max where available.
    # The full loads-p_set.csv is read here (topology phase) regardless of
    # snapshot_range — we need the column-wise max across all timesteps.
    loads["p_set"] = loads_raw["p_set"].astype(float) if "p_set" in loads_raw.columns else 0.0
    load_ts_full = _read(folder, "loads-p_set.csv", index_col=0)
    if not load_ts_full.empty:
        common = loads.index.intersection(load_ts_full.columns)
        if not common.empty:
            loads.loc[common, "p_set"] = load_ts_full[common].max(axis=0)

    loads["sector"] = loads.index.to_series().apply(
        lambda name: (
            "cts"        if "cts" in str(name).lower() else
            "industrial" if "ind" in str(name).lower() else
            "residential"
        )
    )
    loads["voltage_level"]     = "lv"   # adjust here if MV loads are present
    loads["annual_consumption"] = None
    loads["number_households"]  = 1
    loads["building_id"]        = None
    loads["mv_grid_id"]         = mv_grid_id
    topo.loads_df = loads

    # --- Storage units ------------------------------------------------- #
    stor_raw = _read(folder, "storage_units.csv")
    stor = _keep_and_rename(stor_raw, STORAGE_RENAME)
    stor["mv_grid_id"] = mv_grid_id
    topo.storage_units_df = stor

    # ------------------------------------------------------------------ #
    # 2. TIME SERIES
    # ------------------------------------------------------------------ #
    edisgo_obj.timeseries = TimeSeries()
    ts = edisgo_obj.timeseries

    # Resolve the timeindex from snapshots.csv, applying snapshot_range.
    snap = _read(folder, "snapshots.csv", index_col=None)
    if not snap.empty:
        timeindex = _parse_timestamps(snap.loc[:, "snapshot"].iloc[ts_slice])
        ts.timeindex = pd.DatetimeIndex(timeindex, name="snapshot",
                                        freq=pd.infer_freq(timeindex))
    else:
        timeindex = None

    # All timeseries files go through _read_ts() which applies ts_slice at
    # the CSV reader level — only the requested rows are loaded into memory.

    # Active power - generators
    gen_p = _read_ts(folder, "generators-p_max_pu.csv", ts_slice)
    if not gen_p.empty:
        ts.generators_active_power = _assign_timeseries(gen_p, timeindex)

    # Reactive power - generators (optional)
    gen_q = _read_ts(folder, "generators-q_set.csv", ts_slice)
    if not gen_q.empty:
        ts.generators_reactive_power = _assign_timeseries(gen_q, timeindex)

    # Active power - loads
    load_p = _read_ts(folder, "loads-p_set.csv", ts_slice)
    #load_p.loc[:,load_p.columns.str.contains("heat_")] = 0.1 # to force usage of 14a
    #edisgo_obj.topology.loads_df.loc[edisgo_obj.topology.loads_df.index.str.contains("heat_"),"p_set"] = 0.1 # to force usage of 14a

    if not load_p.empty:
        ts.loads_active_power = _assign_timeseries(load_p, timeindex)

    # Reactive power - loads (optional)
    load_q = _read_ts(folder, "loads-q_set.csv", ts_slice)
    if not load_q.empty:
        ts.loads_reactive_power = _assign_timeseries(load_q, timeindex)

    # Storage (optional)
    stor_p = _read_ts(folder, "storage_units-p_set.csv", ts_slice)
    if not stor_p.empty:
        ts.storage_units_active_power = _assign_timeseries(stor_p, timeindex)

    # ------------------------------------------------------------------ #
    # 3. GRID METADATA
    # ------------------------------------------------------------------ #
    net_meta = _read(folder, "network.csv")
    srid = (
        int(net_meta["srid"].iloc[0])
        if not net_meta.empty and "srid" in net_meta.columns
        else int(target_crs.split(":")[1])
    )
    topo.grid_district = {"srid": srid, "mv_grid_id": mv_grid_id, "population":22227}

    # ------------------------------------------------------------------ #
    # 4. STRUCTURAL WIRING
    # Wire the MVGrid object and all back-references that eDisGo internals
    # rely on (mirrors what import_ding0_grid() sets up).
    # ------------------------------------------------------------------ #
    try:
        topo._mv_grid = MVGrid(id=mv_grid_id, topology=topo)
    except TypeError:
        # Older eDisGo versions do not accept the topology kwarg
        topo._mv_grid = MVGrid(id=mv_grid_id)

    topo._edisgo_obj          = edisgo_obj
    topo._mv_grid._topology   = topo
    topo._mv_grid._edisgo_obj = edisgo_obj
    topo._lv_grids            = []

    # ------------------------------------------------------------------ #
    # 5. SUMMARY LOG
    # ------------------------------------------------------------------ #
    logger.info(
        "PyPSA CSV import complete | "
        "%d buses | %d lines | %d trafos LV/MV | %d trafos HV/MV | "
        "%d generators | %d loads | %d storage units | %d timesteps",
        len(topo.buses_df),
        len(topo.lines_df),
        len(topo.transformers_df),
        len(topo.transformers_hvmv_df),
        len(topo.generators_df),
        len(topo.loads_df),
        len(topo.storage_units_df),
        len(timeindex) if timeindex is not None else 0,
    )