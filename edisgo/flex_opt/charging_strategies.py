import logging

import numpy as np
import pandas as pd

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

logger = logging.getLogger("edisgo")


# TODO: the dummy timeseries should be as long as the simulated days and not
#  the timeindex of the edisgo object. At the moment this would result into
#  wrong results if the timeindex of the edisgo object is not continuously
#  (e.g. 2 weeks of the year)
def charging_strategy(
    edisgo_obj,
    strategy="dumb",
    timestamp_share_threshold=0.2,
    minimum_charging_capacity_factor=0.1,
):
    """
    Calculates the timeseries per charging park for a given charging strategy.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    strategy : str
        The charging strategy. Default 'dumb'. Only 'private' charging
        processes at 'home' or at 'work' can be flexibilized. 'public' charging
        processes will always be 'dumb'. For now the following charging
        strategies are valid:
        * 'dumb': The cars are charged directly after arrival with the
        maximum possible charging capacity.
        * 'reduced': The cars are charged directly after arrival with the
        minimum possible charging capacity. The minimum possible charging
        capacity is determined by the parking time and the
        minimum_charging_capacity_factor.
        * 'residual': The cars are charged when the residual load in the MV
        grid is at it's lowest (high generation and low consumption).
        Charging processes with a low flexibility band are given priority.
    timestamp_share_threshold : float
        Percental threshold of the time required at a time step for charging
        the vehicle. If the time requirement is below this limit, then the
        charging process is not mapped into the time series. If, however, it is
        above this limit, the time step is mapped to 100% into the time series.
        This prevents differences between the charging strategies and creates a
        compromise between the simultaneity of charging processes and an
        artificial increase in the charging demand. Default 0.2
    minimum_charging_capacity_factor : float
        Technical percental minimum charging capacity per charging point.
        Default 0.1

    """
    # get integrated charging parks
    charging_parks = [
        cp
        for cp in list(edisgo_obj.electromobility.potential_charging_parks)
        if cp.grid is not None
    ]

    # Reset possible old timeseries as these influence "residual" charging
    ts = pd.Series(data=0, index=edisgo_obj.timeseries.timeindex)

    for cp in charging_parks:
        _overwrite_timeseries(edisgo_obj, cp.edisgo_id, ts)

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

    assert edisgo_timedelta == simbev_timedelta, (
        "The stepsize of the timeseries of the edisgo object differs from the"
        f"simbev stepsize. The edisgo timedelta is {edisgo_timedelta}, while"
        f" the simbev timedelta is {simbev_timedelta}. Make sure to use a "
        f"matching stepsize."
    )

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

            _overwrite_timeseries(
                edisgo_obj, cp.edisgo_id, pd.Series(data=dummy_ts, index=timeindex)
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

            _overwrite_timeseries(
                edisgo_obj, cp.edisgo_id, pd.Series(data=dummy_ts, index=timeindex)
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
                    f"start is at timestep {start} and the park end is at timestep "
                    f"{start + stop}, while the timeseries consists of {maximum_ts} "
                    f"timesteps."
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
                    f"timesteps {idx}, while the timeseries consists of "
                    f"{maximum_ts} timesteps."
                )

        for count, col in enumerate(dummy_ts.columns):
            _overwrite_timeseries(
                edisgo_obj, charging_parks[count].edisgo_id, dummy_ts[col]
            )

    else:
        raise ValueError(f"Strategy {strategy} has not yet been implemented.")

    logging.info(f"Charging strategy {strategy} completed.")


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
        Charging processes DataFrame
    len_ts : int
        Length of the timeseries
    timestamp_share_threshold : float
        See description in the main function. Default 0.2
    strategy : str
        See description in the main function. Default 'dumb'
    minimum_charging_capacity_factor : float
        See description in the main function. Default 0.1
    eta_cp : float
        Charging Point efficiency. Default 1.0

    """
    # FIXME: This should become obsolete in the future when SimBEV is
    #  bugfixed drop rows that have a park start higher than simulated days
    df = df.loc[df.park_start_timesteps <= len_ts]

    # calculate the minimum time taken the fulfill the charging demand
    minimum_charging_time = (
        df.chargingdemand_kWh
        / df.nominal_charging_capacity_kW
        * 60
        / edisgo_obj.electromobility.stepsize
    )

    # calculate in which time steps the last time step needed to fulfill
    # the charging demand is considered in the timeseries
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


def _overwrite_timeseries(edisgo_obj, edisgo_id, ts):
    """
    Overwrites the dummy timeseries for the Charging Point

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    edisgo_id : str
        eDisGo ID of the Charging Point
    ts : :pandas:`pandas.Series<Series>`
        New timeseries

    """
    edisgo_obj.timeseries._loads_active_power.loc[:, edisgo_id] = ts.loc[
        edisgo_obj.timeseries.timeindex
    ]
