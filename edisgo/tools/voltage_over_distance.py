# edisgo/tools/voltage_over_distance.py

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _get_v_res_df(edisgo_obj) -> pd.DataFrame:
    """
    Support both v_res attribute and v_res() function (version differences).
    Expects index = timesteps, columns = buses, values = voltage in p.u.
    """
    if not hasattr(edisgo_obj, "results") or edisgo_obj.results is None:
        raise RuntimeError("No results found. Run edisgo.analyze() first.")

    v_res = getattr(edisgo_obj.results, "v_res", None)
    if v_res is None:
        raise RuntimeError(
            "No voltage results (results.v_res) found. Run edisgo.analyze() first."
        )

    return v_res() if callable(v_res) else v_res


def _infer_load_and_feedin_timesteps(
    edisgo_obj, v_df: pd.DataFrame | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Robust worst-case timestep inference.

    Preferred (always works after analyze):
      - load case: timestep with minimum voltage across all buses (worst undervoltage)
      - feed-in case: timestep with maximum voltage across all buses (worst overvoltage)

    Fallback (if voltage results not suitable):
      - residual load = sum(loads) - sum(generation)
    """
    # 1) Prefer voltage results (available after analyze)
    if v_df is None:
        try:
            v_df = _get_v_res_df(edisgo_obj)
        except Exception:
            v_df = None

    if isinstance(v_df, pd.DataFrame) and len(v_df.index) > 0 and len(v_df.columns) > 0:
        # row-wise min/max across buses
        row_min = v_df.min(axis=1)
        row_max = v_df.max(axis=1)

        # If all-NaN, fall back
        if not row_min.dropna().empty and not row_max.dropna().empty:
            t_load = pd.Timestamp(row_min.idxmin())  # worst undervoltage
            t_feed = pd.Timestamp(row_max.idxmax())  # worst overvoltage
            return t_load, t_feed

    # 2) Fallback: residual load from timeseries
    ts = getattr(edisgo_obj, "timeseries", None)
    if ts is None:
        raise RuntimeError(
            "Cannot infer worst-case timesteps: no voltage results and no timeseries."
        )

    loads_p = getattr(ts, "loads_active_power", None)
    gens_p = getattr(ts, "generators_active_power", None)

    if (
        loads_p is None
        or gens_p is None
        or len(loads_p.index) == 0
        or len(gens_p.index) == 0
    ):
        raise RuntimeError(
            "Cannot infer worst-case timesteps: voltage results are empty and "
            "timeseries.loads_active_power / generators_active_power are missing or empty."
        )

    loads_sum = loads_p.sum(axis=1)
    gens_sum = gens_p.sum(axis=1)
    residual = (loads_sum - gens_sum).dropna()

    if residual.empty:
        raise RuntimeError(
            "Cannot infer worst-case timesteps: residual load series is empty."
        )

    t_load = pd.Timestamp(residual.idxmax())
    t_feed = pd.Timestamp(residual.idxmin())
    return t_load, t_feed


def _series_at_t(v_df: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """
    Return bus-voltage series at timestep t. If exact t not found, use nearest.
    """
    if t not in v_df.index:
        # use nearest timestep if index is datetime-like
        try:
            nearest_pos = v_df.index.get_indexer([t], method="nearest")[0]
            t = v_df.index[nearest_pos]
        except Exception as e:
            raise KeyError(f"Timestamp {t} not found in voltage results index.") from e

    s = v_df.loc[t]
    # Ensure 1D
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    s = s.astype(float)
    # normalize bus names to str
    s.index = s.index.map(str)
    return s


def build_voltage_over_distance_df(
    *,
    buses: pd.Index,
    dist_a: pd.Series,
    v_a_load: pd.Series,
    v_a_feed: pd.Series,
    dist_b: pd.Series | None = None,
    v_b_load: pd.Series | None = None,
    v_b_feed: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build the DataFrame used for voltage-over-distance plots.

    Returns columns:
      - bus (str)
      - x (float)   (path length / distance)
      - v_pu (float)
      - label (str) (scenario label)
    """
    buses = pd.Index([str(b) for b in buses])

    def _mk_df(dist: pd.Series, vv: pd.Series, label: str) -> pd.DataFrame:
        df = pd.DataFrame({"bus": buses})
        df["x"] = df["bus"].map(dist)
        df["v_pu"] = df["bus"].map(vv)
        df["label"] = label
        return df.dropna(subset=["x", "v_pu"])

    parts = [
        _mk_df(dist_a, v_a_load, "base — load case"),
        _mk_df(dist_a, v_a_feed, "base — feed-in case"),
    ]

    if dist_b is not None and v_b_load is not None and v_b_feed is not None:
        parts.extend(
            [
                _mk_df(dist_b, v_b_load, "other — load case"),
                _mk_df(dist_b, v_b_feed, "other — feed-in case"),
            ]
        )

    return pd.concat(parts, ignore_index=True)


def make_voltage_over_distance_figure(
    *,
    title: str,
    buses: pd.Index,
    dist_a: pd.Series,
    v_a_load: pd.Series,
    v_a_feed: pd.Series,
    dist_b: pd.Series | None = None,
    v_b_load: pd.Series | None = None,
    v_b_feed: pd.Series | None = None,
    band_low: float,
    band_high: float,
    x_label: str = "Path length to station [-]",
    df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Builds a plotly scatter figure with 4 traces:
      base-load, base-feed-in, other-load, other-feed-in
    """

    buses = pd.Index([str(b) for b in buses])

    if df is None:
        df = build_voltage_over_distance_df(
            buses=buses,
            dist_a=dist_a,
            v_a_load=v_a_load,
            v_a_feed=v_a_feed,
            dist_b=dist_b,
            v_b_load=v_b_load,
            v_b_feed=v_b_feed,
        )

    fig = go.Figure()

    # tolerance band
    fig.add_hrect(
        y0=band_low,
        y1=band_high,
        line_width=0,
        fillcolor="rgba(0,0,0,0.06)",
    )

    for label in df["label"].unique():
        sub = df[df["label"] == label]
        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["v_pu"],
                mode="markers",
                name=label,
                customdata=np.stack([sub["bus"]], axis=-1),
                hovertemplate="bus=%{customdata[0]}<br>path=%{x}<br>v=%{y:.4f} p.u.<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Voltage [p.u.]",
        legend_title="Scenario",
    )
    return fig
