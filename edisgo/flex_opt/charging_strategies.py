from __future__ import annotations

import logging

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
):
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

    """
    # get integrated charging parks
    charging_parks = [
        cp
        for cp in list(edisgo_obj.electromobility.potential_charging_parks)
        if cp.grid is not None
    ]

    # Delete possible old time series as these influence "residual" charging
    edisgo_obj.timeseries.drop_component_time_series(
        "loads_active_power",
        edisgo_obj.electromobility.integrated_charging_parks_df.edisgo_id.values,
    )

    edisgo_obj.timeseries.drop_component_time_series(
        "loads_reactive_power",
        edisgo_obj.electromobility.integrated_charging_parks_df.edisgo_id.values,
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

    if strategy == "dumb":
        # "dumb" charging
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
                dummy_ts[start : start + stop] += cap

            edisgo_obj.timeseries.add_component_time_series(
                "loads_active_power",
                pd.DataFrame(data={cp.edisgo_id: dummy_ts}, index=timeindex),
            )

    elif strategy == "reduced":
        # "reduced" charging
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
                if use_case == "public" or use_case == "hpc":
                    # if the charging process takes place in a "public" setting
                    # the charging is "dumb"
                    dummy_ts[start : start + stop_dumb] += cap_dumb
                else:
                    dummy_ts[start : start + stop_reduced] += cap_reduced

            edisgo_obj.timeseries.add_component_time_series(
                "loads_active_power",
                pd.DataFrame(data={cp.edisgo_id: dummy_ts}, index=timeindex),
            )

    elif strategy == "residual":
        # "residual" charging
        # only use charging processes from integrated charging parks
        charging_processes_df = edisgo_obj.electromobility.charging_processes_df[
            edisgo_obj.electromobility.charging_processes_df.charging_park_id.isin(
                edisgo_obj.electromobility.integrated_charging_parks_df.index
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

        # get residual load
        init_residual_load = edisgo_obj.timeseries.residual_load

        len_residual_load = int(charging_processes_df.park_end_timesteps.max())

        if len(init_residual_load) >= len_residual_load:
            init_residual_load = init_residual_load.loc[timeindex]
        else:
            while len(init_residual_load) < len_residual_load:
                len_rl = len(init_residual_load)
                len_append = min(len_rl, len_residual_load - len_rl)

                s_append = init_residual_load.iloc[:len_append]

                init_residual_load = pd.concat(
                    [
                        init_residual_load,
                        s_append,
                    ],
                    ignore_index=True,
                )

        init_residual_load = init_residual_load.to_numpy()

        timeindex_residual = pd.date_range(
            edisgo_obj.timeseries.timeindex[0],
            periods=len(init_residual_load),
            freq=f"{edisgo_obj.electromobility.stepsize}min",
        )

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
                dummy_ts.loc[:, cp_id].iloc[start : start + stop] += cap

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
            flex_band = residual_load[start : end + 1]

            # get k time steps with the lowest residual load in the parking
            # time
            idx = np.argpartition(flex_band, k)[:k] + start

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

    # set reactive power time series to 0 Mvar
    # fmt: off
    edisgo_obj.timeseries.add_component_time_series(
        "loads_reactive_power",
        pd.DataFrame(
            data=0.0,
            index=edisgo_obj.timeseries.timeindex,
            columns=edisgo_obj.electromobility.integrated_charging_parks_df
            .edisgo_id.values,
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
