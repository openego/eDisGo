# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging

from collections.abc import Iterable
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from edisgo import EDisGo

RELEVANT_CHARGING_STRATEGIES_COLUMNS = {
    "dumb": [
        "park_start_timesteps",
        "minimum_charging_time",
        "nominal_charging_capacity_mva",
    ],
    "reduced": [
        "use_case",
        "park_start_timesteps",
        "minimum_charging_time",
        "nominal_charging_capacity_mva",
        "reduced_charging_time",
        "reduced_charging_capacity_mva",
    ],
    "residual_dumb": [
        "charging_park_id",
        "park_start_timesteps",
        "minimum_charging_time",
        "nominal_charging_capacity_mva",
    ],
    "residual": [
        "park_start_timesteps",
        "park_end_timesteps",
        "minimum_charging_time",
        "charging_park_id",
        "nominal_charging_capacity_mva",
    ],
}

logger = logging.getLogger(__name__)


# TODO: the dummy timeseries should be as long as the simulated days and not
#  the timeindex of the edisgo object. At the moment this would result into
#  wrong results if the timeindex of the edisgo object is not continuously
#  (e.g. 2 weeks of the year)
def charging_strategy(
    edisgo_obj: EDisGo,
    strategy: str = "dumb",
    timestamp_share_threshold: Number = 0.2,
    minimum_charging_capacity_factor: Number = 0.1,
    charging_park_ids: Iterable[int] | None = None,
) -> None:
    """
    Applies charging strategy to set EV charging time series at charging parks.

    See :attr:`~.edisgo.EDisGo.apply_charging_strategy` for more information.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    strategy : str
        Defines the charging strategy to apply. See `strategy` parameter
        :attr:`~.edisgo.EDisGo.apply_charging_strategy` for more information.
        Default: 'dumb'.
    timestamp_share_threshold : float
        Percental threshold of the time required at a time step for charging
        the vehicle. See `timestamp_share_threshold` parameter
        :attr:`~.edisgo.EDisGo.apply_charging_strategy` for more information.
        Default: 0.2.
    minimum_charging_capacity_factor : float
        Technical minimum charging power of charging points in p.u. used in case of
        charging strategy 'reduced'. See `minimum_charging_capacity_factor` parameter
        :attr:`~.edisgo.EDisGo.apply_charging_strategy` for more information.
        Default: 0.1.

    Notes
    -----
    The written ``loads_active_power``/``loads_reactive_power`` are trimmed to
    ``edisgo_obj.timeseries.timeindex`` when its frequency already matches the
    SimBEV charging-process data's ``stepsize`` (the common case). When it
    doesn't, this function internally resamples ``edisgo_obj.timeseries`` to
    SimBEV's frequency and back (see the frequency-mismatch warning below);
    that round-trip currently fabricates a contiguous timeindex, which can
    reopen a gap left by ``select_timesteps`` (auto mode). The trim is
    skipped in that case rather than risk operating on the wrong window -
    tracked as a known limitation in
    ``docs_notes/issue_temporal_reduction_flexibility_bands.md``.

    """
    # Capture the target time index before any internal frequency resampling
    # (below) can mutate it. `TimeSeries.resample` fabricates a contiguous
    # index spanning first-to-last timestamp, which would silently reopen any
    # gap `select_timesteps` (auto mode) deliberately left in the timeindex -
    # trimming against this entry-time snapshot instead ensures only the
    # steps actually selected by the caller are written. Only used when no
    # internal resample round-trip happens (see Notes above).
    target_timeindex = edisgo_obj.timeseries.timeindex

    # get integrated charging parks
    integrated_parks = edisgo_obj.electromobility.integrated_charging_parks_df

    # Bestimmen, welche Ladeparks überhaupt angesprochen werden
    if charging_park_ids is None:
        target_park_ids = integrated_parks.index
    else:
        charging_park_ids = list(charging_park_ids)
        target_park_ids = integrated_parks.index.intersection(charging_park_ids)

        missing_ids = sorted(set(charging_park_ids) - set(target_park_ids))
        if missing_ids:
            logger.warning(
                "The following charging park IDs are not integrated and will be "
                "ignored in charging_strategy: %s",
                missing_ids,
            )

    # PotentialChargingParks-Objekte auf diese Submenge filtern
    charging_parks = [
        cp
        for cp in edisgo_obj.electromobility.potential_charging_parks
        if cp.grid is not None and cp.id in target_park_ids
    ]

    if len(charging_parks) == 0:
        logger.info(
            "No charging parks selected for charging strategy '%s'. Nothing to do.",
            strategy,
        )
        return

    # EDisGo-IDs der betroffenen Ladeparks
    edisgo_ids_to_update = integrated_parks.loc[target_park_ids, "edisgo_id"].values

    # Nur für diese Ladeparks alte Zeitreihen löschen
    edisgo_obj.timeseries.drop_component_time_series(
        "loads_active_power",
        edisgo_ids_to_update,
    )
    edisgo_obj.timeseries.drop_component_time_series(
        "loads_reactive_power",
        edisgo_ids_to_update,
    )

    eta_cp = edisgo_obj.electromobility.eta_charging_points

    len_ts = int(
        edisgo_obj.electromobility.simulated_days
        * 24
        * 60
        / edisgo_obj.electromobility.stepsize
    )

    timeindex = pd.date_range(
        edisgo_obj.timeseries.timeindex[0],
        periods=len_ts,
        freq=f"{edisgo_obj.electromobility.stepsize}min",
    )

    edisgo_timedelta = (
        edisgo_obj.timeseries.timeindex[1] - edisgo_obj.timeseries.timeindex[0]
    )
    simbev_timedelta = timeindex[1] - timeindex[0]

    resample = edisgo_timedelta != simbev_timedelta

    if resample:
        logger.warning(
            f"The frequency of the time series data of the edisgo object differs from "
            f"the simbev time series frequency. The edisgo frequency is "
            f"{edisgo_timedelta}, while the simbev frequency is {simbev_timedelta}. "
            f"The edisgo time series data "
            f"will be resampled accordingly before applying the charging strategy. "
            f"After applying the charging strategy all time series will be resampled "
            f"to the original frequency of the edisgo time series data."
        )

        edisgo_obj.timeseries.resample(freq=simbev_timedelta)

    # Map each SimBEV step position (0 .. len_ts - 1, the same positional
    # space as park_start_timesteps/the placement slices below) to whether it
    # is present in the active timeindex (`target_timeindex`). `dumb` and
    # `reduced` place each event's demand deterministically at
    # [start, start+stop) - rather than building the full-SimBEV-length
    # series unconditionally and cropping the *output* down to the active
    # timeindex afterwards (as before), the placement itself is now clipped
    # to whatever of that interval is actually in-window, so an event's
    # reported energy is a direct consequence of which positions get
    # written, not a separate proration calculation (see ADR 0002).
    # `resample=True` means `target_timeindex` predates an internal
    # frequency round-trip and is no longer in the same step space as
    # `park_start_timesteps` - the crop-after-build step already skips
    # trimming in that case (see the module docstring), so this reduction is
    # skipped here too and every step is treated as in-window, preserving
    # today's (build-full) behavior only for that known limitation.
    if resample:
        step_in_window = np.ones(len_ts, dtype=bool)
    else:
        step_in_window = np.isin(timeindex, target_timeindex)

    if strategy == "dumb":
        # "dumb" charging
        # Collect each charging park's series and add them to the time series in a
        # single call after the loop. Adding them one at a time concatenates onto
        # the growing loads_active_power frame on every iteration (O(parks^2)),
        # which dominated the runtime on large grids.
        cp_ts = {}
        for cp in charging_parks:
            dummy_ts = np.zeros(len_ts)

            charging_processes_df = harmonize_charging_processes_df(
                cp.charging_processes_df,
                edisgo_obj,
                len_ts,
                timestamp_share_threshold,
                strategy=strategy,
                eta_cp=eta_cp,
            )

            for _, start, stop, cap in charging_processes_df[
                RELEVANT_CHARGING_STRATEGIES_COLUMNS["dumb"]
            ].itertuples():
                # Write only to in-window positions of the deterministic
                # charging interval [start, start+stop) - if the active
                # timeindex has a gap inside this interval, every in-window
                # sub-slice still gets the event's full, unscaled power (see
                # ADR 0002); out-of-window positions are simply not written.
                in_window_idx = (
                    np.flatnonzero(step_in_window[start : start + stop]) + start
                )
                dummy_ts[in_window_idx] += cap

            cp_ts[cp.edisgo_id] = dummy_ts

        if cp_ts:
            edisgo_obj.timeseries.add_component_time_series(
                "loads_active_power",
                pd.DataFrame(data=cp_ts, index=timeindex),
            )

    elif strategy == "reduced":
        # "reduced" charging
        # See the "dumb" branch above: accumulate all park columns and add them
        # once to avoid the O(parks^2) per-park concatenation.
        cp_ts = {}
        for cp in charging_parks:
            dummy_ts = np.zeros(len_ts)

            charging_processes_df = harmonize_charging_processes_df(
                cp.charging_processes_df,
                edisgo_obj,
                len_ts,
                timestamp_share_threshold,
                strategy=strategy,
                minimum_charging_capacity_factor=minimum_charging_capacity_factor,
                eta_cp=eta_cp,
            )

            for (
                _,
                use_case,
                start,
                stop_dumb,
                cap_dumb,
                stop_reduced,
                cap_reduced,
            ) in charging_processes_df[
                RELEVANT_CHARGING_STRATEGIES_COLUMNS["reduced"]
            ].itertuples():
                # See the "dumb" branch above for why the placement slice
                # itself (not a separate energy calculation) is clipped to
                # in-window positions.
                if use_case == "public" or use_case == "hpc":
                    # if the charging process takes place in a "public" setting
                    # the charging is "dumb"
                    start_, stop_, cap = start, stop_dumb, cap_dumb
                else:
                    start_, stop_, cap = start, stop_reduced, cap_reduced

                in_window_idx = (
                    np.flatnonzero(step_in_window[start_ : start_ + stop_]) + start_
                )
                dummy_ts[in_window_idx] += cap

            cp_ts[cp.edisgo_id] = dummy_ts

        if cp_ts:
            edisgo_obj.timeseries.add_component_time_series(
                "loads_active_power",
                pd.DataFrame(data=cp_ts, index=timeindex),
            )

    elif strategy == "residual":
        # "residual" charging
        # only use charging processes from integrated charging parks
        charging_processes_df = edisgo_obj.electromobility.charging_processes_df[
            edisgo_obj.electromobility.charging_processes_df.charging_park_id.isin(
                target_park_ids
            )
        ]

        charging_processes_df = harmonize_charging_processes_df(
            charging_processes_df,
            edisgo_obj,
            len_ts,
            timestamp_share_threshold,
            strategy=strategy,
            eta_cp=eta_cp,
        )

        len_residual_load = int(charging_processes_df.park_end_timesteps.max())

        if not resample:
            # The active timeindex can extend past the last charging event
            # (e.g. a trailing gapped run with no events in it at all) - the
            # step-space array built below must cover at least as far as
            # target_timeindex itself, or the crop-after-build step later
            # would reindex into positions that were never built, producing
            # NaN rather than a legitimate zero.
            target_span_steps = int(
                (target_timeindex[-1] - target_timeindex[0])
                / pd.Timedelta(f"{edisgo_obj.electromobility.stepsize}min")
            )
            len_residual_load = max(len_residual_load, target_span_steps)

        # Map each SimBEV step position (0 .. len_residual_load, the same
        # positional space as park_start_timesteps/park_end_timesteps) to
        # whether it is present in the active timeindex (`target_timeindex`).
        # Real residual_load only exists for `target_timeindex`. Rather than
        # tiling (cyclically repeating) it to cover steps beyond the active
        # timeindex - which would rank timesteps against a fabricated,
        # non-periodic-in-reality signal (see ADR 0001) - steps outside the
        # active timeindex are simply marked as having no usable data.
        # `resample=True` means `target_timeindex` predates an internal
        # frequency round-trip and is no longer in the same step space as
        # `park_start_timesteps` - the crop-after-build step already skips
        # trimming in that case (see the module docstring), so this reduction
        # is skipped here too and every step is treated as in-window,
        # preserving today's (tiling) behavior only for that known
        # limitation.
        if resample:
            step_in_window = np.ones(len_residual_load + 1, dtype=bool)
        else:
            step_in_window = np.isin(
                pd.date_range(
                    target_timeindex[0],
                    periods=len_residual_load + 1,
                    freq=f"{edisgo_obj.electromobility.stepsize}min",
                ),
                target_timeindex,
            )
        in_window_steps_cumsum = np.concatenate(([0], np.cumsum(step_in_window)))

        if not resample:
            # Events are reduced to the active timeindex before being
            # scheduled: fully in-window events are untouched, fully
            # out-of-window events are dropped, and boundary-straddling
            # events have their charging demand prorated by how much of
            # their parking time is actually observable. This mirrors
            # `harmonize_charging_processes_df`'s own derivation of
            # `minimum_charging_time` from demand and nominal power.
            parking_time = (
                charging_processes_df.park_end_timesteps
                - charging_processes_df.park_start_timesteps
                + 1
            )
            overlap_steps = (
                in_window_steps_cumsum[
                    charging_processes_df.park_end_timesteps.to_numpy() + 1
                ]
                - in_window_steps_cumsum[
                    charging_processes_df.park_start_timesteps.to_numpy()
                ]
            )

            # drop events with zero overlap - nothing to schedule, no
            # residual_load data exists for them at all
            in_window = overlap_steps > 0
            charging_processes_df = charging_processes_df.loc[in_window]
            in_window_fraction = (
                (overlap_steps[in_window]) / (parking_time.to_numpy()[in_window])
            )

            scaled_demand_kWh = (
                charging_processes_df.harmonized_chargingdemand * in_window_fraction
            )
            scaled_minimum_charging_time = (
                scaled_demand_kWh
                / charging_processes_df.nominal_charging_capacity_kW
                * 60
                / edisgo_obj.electromobility.stepsize
            )
            scaled_minimum_charging_time = np.ceil(scaled_minimum_charging_time).astype(
                np.uint16
            )

            # defensive clamp: proration preserves
            # minimum_charging_time <= parking_time, so this should only ever
            # bind on pre-existing anomalous input (an event whose full,
            # unscaled demand already didn't fit its own parking time)
            scaled_minimum_charging_time = np.minimum(
                scaled_minimum_charging_time, overlap_steps[in_window]
            )

            charging_processes_df = charging_processes_df.assign(
                minimum_charging_time=scaled_minimum_charging_time,
                flex_time=charging_processes_df.park_time_timesteps
                - scaled_minimum_charging_time,
            )

        # get residual load; steps outside the active timeindex carry no
        # real data (see above) and are set to NaN so they can never be
        # selected as charging candidates below
        init_residual_load = edisgo_obj.timeseries.residual_load

        timeindex_residual = pd.date_range(
            edisgo_obj.timeseries.timeindex[0],
            periods=len_residual_load + 1,
            freq=f"{edisgo_obj.electromobility.stepsize}min",
        )
        init_residual_load = init_residual_load.reindex(timeindex_residual).to_numpy()
        init_residual_load[~step_in_window] = np.nan

        dummy_ts = pd.DataFrame(
            data=0.0, columns=[_.id for _ in charging_parks], index=timeindex_residual
        )

        # determine which charging processes can be flexibilized
        dumb_charging_processes_df = charging_processes_df.loc[
            charging_processes_df.use_case.isin(["public", "hpc"])
            | (charging_processes_df.flex_time == 0)
        ]

        flex_charging_processes_df = charging_processes_df.loc[
            ~charging_processes_df.index.isin(dumb_charging_processes_df.index)
        ]

        # perform dumb charging processes and respect them in the residual load
        for _, cp_id, start, stop, cap in dumb_charging_processes_df[
            RELEVANT_CHARGING_STRATEGIES_COLUMNS["residual_dumb"]
        ].itertuples():
            try:
                # Write only to in-window positions of the deterministic
                # charging interval [start, start+stop) - if the active
                # timeindex has a gap inside this interval, every in-window
                # sub-slice still gets the event's full, unscaled power (see
                # ADR 0002); out-of-window positions are simply not written.
                in_window_idx = (
                    np.flatnonzero(step_in_window[start : start + stop]) + start
                )
                dummy_ts.loc[:, cp_id].iloc[in_window_idx] += cap

            except Exception:
                maximum_ts = len(dummy_ts)
                logger.warning(
                    f"Charging process with index {_} could not be respected. The park "
                    f"start is at time step {start} and the park end is at time step "
                    f"{start + stop}, while the time series consists of {maximum_ts} "
                    f"time steps."
                )

        residual_load = init_residual_load + dummy_ts.sum(axis=1).to_numpy()

        for _, start, end, k, cp_id, cap in flex_charging_processes_df[
            RELEVANT_CHARGING_STRATEGIES_COLUMNS["residual"]
        ].itertuples():
            # Restrict ranking candidates to timesteps that are both within
            # the parking window and present in the active timeindex -
            # `residual_load` is NaN outside the active timeindex (no real
            # data exists there, see above), so those positions must never
            # be selected, even if the parking window itself spans a gap.
            candidates = np.flatnonzero(step_in_window[start : end + 1]) + start

            if k >= len(candidates):
                # k charging demand may (after proration/clamping) exactly
                # saturate the available in-window candidates - nothing left
                # to rank, every candidate is used.
                idx = candidates
            else:
                flex_band = residual_load[candidates]
                # get k time steps with the lowest residual load in the
                # parking time, among the valid (in-window) candidates only
                idx = candidates[np.argpartition(flex_band, k)[:k]]

            try:
                dummy_ts[cp_id].iloc[idx] += cap

                residual_load[idx] += cap

            except Exception:
                logger.warning(
                    f"Charging process with index {_} could not be "
                    f"respected. The charging takes place within the "
                    f"time steps {idx}, while the time series consists of "
                    f"{maximum_ts} time steps."
                )
        edisgo_obj.timeseries.add_component_time_series(
            "loads_active_power",
            dummy_ts.rename(
                columns={
                    cp_id: edisgo_obj.electromobility.integrated_charging_parks_df.at[
                        cp_id, "edisgo_id"
                    ]
                    for cp_id in dummy_ts.columns
                }
            ),
        )

    else:
        raise ValueError(f"Strategy {strategy} has not yet been implemented.")

    if resample:
        edisgo_obj.timeseries.resample(freq=edisgo_timedelta)
        # `TimeSeries.resample` fabricates a contiguous index spanning
        # first-to-last timestamp, which would reopen any gap
        # `select_timesteps` (auto mode) left in `target_timeindex`. The trim
        # below only removes *extra trailing* rows past `target_timeindex`'s
        # own span - it does not (and, given the above, safely cannot)
        # reintroduce a gap `resample` already closed. Fixing that root cause
        # in `TimeSeries.resample` itself is tracked separately (see
        # docs_notes/issue_temporal_reduction_flexibility_bands.md); until
        # then, a `select_timesteps`-produced gap combined with a
        # SimBEV/edisgo frequency mismatch is a known limitation here.
    else:
        # Trim the newly written columns down to the target time index. The
        # writes above (all three strategies) span the full SimBEV
        # simulation length rather than the active timeindex.
        # `TimeSeries.loads_active_power` itself already scopes reads to
        # `self.timeindex`, but the private `_loads_active_power` can still
        # carry the untrimmed rows (visible to anything reading the private
        # attribute directly, e.g. `reduce_timeseries_data_to_given_timeindex`)
        # - rebuild just the touched columns via drop+add so only the extra
        # rows for `edisgo_ids_to_update` are removed, leaving other
        # components untouched.
        trimmed_active_power = edisgo_obj.timeseries._loads_active_power.loc[
            :, edisgo_ids_to_update
        ].reindex(target_timeindex)
        edisgo_obj.timeseries.drop_component_time_series(
            "loads_active_power", edisgo_ids_to_update
        )
        edisgo_obj.timeseries.add_component_time_series(
            "loads_active_power", trimmed_active_power
        )

    # set reactive power time series to 0 Mvar. Use `target_timeindex` only
    # when it still matches `edisgo_obj.timeseries.timeindex` (i.e. no
    # internal resample round-trip happened above) - see the comment on the
    # active-power trim above for why a resampled, gap-closed timeindex isn't
    # safely reconcilable with `target_timeindex` here yet.
    # fmt: off
    edisgo_obj.timeseries.add_component_time_series(
        "loads_reactive_power",
        pd.DataFrame(
            data=0.0,
            index=target_timeindex if not resample else edisgo_obj.timeseries.timeindex,
            columns=edisgo_ids_to_update,
        ),
    )
    # fmt: on

    logger.info(f"Charging strategy {strategy} completed.")


def harmonize_charging_processes_df(
    df,
    edisgo_obj,
    len_ts,
    timestamp_share_threshold,
    strategy=None,
    minimum_charging_capacity_factor=0.1,
    eta_cp=1.0,
):
    """
    Harmonizes the charging processes to prevent differences in the energy
    demand per charging strategy.

    Parameters
    ----------
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Charging processes DataFrame.
    len_ts : int
        Length of the timeseries.
    timestamp_share_threshold : float
        See description in :func:`~.flex_opt.charging_strategies.charging_strategy`.
    strategy : str
        See description in :func:`~.flex_opt.charging_strategies.charging_strategy`.
    minimum_charging_capacity_factor : float
        See description in :func:`~.flex_opt.charging_strategies.charging_strategy`.
        Default: 0.1.
    eta_cp : float
        Charging point efficiency. Default: 1.0.

    """
    # FIXME: This should become obsolete in the future when SimBEV is bugfixed
    # drop rows that have a park start higher than simulated days
    df = df.loc[df.park_start_timesteps <= len_ts]

    # calculate the minimum time taken to fulfill the charging demand
    minimum_charging_time = (
        df.chargingdemand_kWh
        / df.nominal_charging_capacity_kW
        * 60
        / edisgo_obj.electromobility.stepsize
    )

    # calculate in which time steps the last time step needed to fulfill
    # the charging demand is considered in the time series
    mask = (minimum_charging_time % 1) >= timestamp_share_threshold

    minimum_charging_time.loc[mask] = minimum_charging_time.apply(np.ceil)

    minimum_charging_time.loc[~mask] = minimum_charging_time.apply(np.floor)

    # recalculate the charging demand from the charging capacity
    # and the minimum charging time
    # Calculate the grid sided charging capacity in MVA
    df = df.assign(
        minimum_charging_time=minimum_charging_time.astype(np.uint16),
        harmonized_chargingdemand=minimum_charging_time
        * df.nominal_charging_capacity_kW
        * edisgo_obj.electromobility.stepsize
        / 60,
        nominal_charging_capacity_mva=df.nominal_charging_capacity_kW.divide(
            10**3 * eta_cp
        ),  # kW --> MW
    )

    if strategy == "reduced":
        parking_time = df.park_end_timesteps - df.park_start_timesteps

        # calculate the maximum needed charging time with the minimum
        # charging capacity
        maximum_needed_charging_time = (
            df.harmonized_chargingdemand
            / (minimum_charging_capacity_factor * df.nominal_charging_capacity_kW)
            * 60
            / edisgo_obj.electromobility.stepsize
        )

        maximum_needed_charging_time = maximum_needed_charging_time.apply(
            np.floor
        ).astype(np.uint16)

        # when the parking time is less than the maximum needed charging
        # time, the total charging time equates the parking time and the
        # charging capacity is recalculated accordingly
        mask = parking_time <= maximum_needed_charging_time

        df = df.assign(
            reduced_charging_time=0,
            reduced_charging_capacity=0,
        )

        df.loc[mask, "reduced_charging_time"] = parking_time.loc[mask]

        df.loc[~mask, "reduced_charging_time"] = maximum_needed_charging_time.loc[~mask]

        df.reduced_charging_capacity = (
            df.harmonized_chargingdemand
            / df.reduced_charging_time
            * 60
            / edisgo_obj.electromobility.stepsize
        )

        df = df.assign(
            reduced_charging_capacity_mva=df.reduced_charging_capacity.divide(
                10**3 * eta_cp
            )
        )

    elif strategy == "residual":
        # the flex time/band is defined as the amount of time steps not
        # needed to fulfill the charging demand in a parking process
        df = df.assign(flex_time=df.park_time_timesteps - df.minimum_charging_time)

        df = df.sort_values(
            by=["flex_time", "park_start_timesteps", "park_end_timesteps"],
            ascending=[True, True, True],
        )

    return df
