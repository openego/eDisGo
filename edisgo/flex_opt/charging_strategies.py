import logging
import numpy as np
import pandas as pd

from edisgo import EDisGo

COLUMNS = {
    "integrated_charging_parks_df": ["edisgo_id"],
}

logger = logging.getLogger("edisgo")


def integrate_charging_points(edisgo_obj, comp_type="ChargingPoint"):
    charging_parks = list(edisgo_obj.electromobility.potential_charging_parks)

    designated_charging_parks = [
        _ for _ in charging_parks if _.designated_charging_point_capacity > 0]

    charging_park_ids = [_.id for _ in designated_charging_parks]

    dummy_timeseries = pd.Series([0] * len(edisgo_obj.timeseries.timeindex), index=edisgo_obj.timeseries.timeindex)

    edisgo_id = [
        EDisGo.integrate_component(
            edisgo_obj,
            comp_type=comp_type,
            geolocation=cp.geometry,
            use_case=cp.use_case,
            add_ts=True,
            ts_active_power=dummy_timeseries,
            ts_reactive_power=dummy_timeseries,
            p_nom=cp.grid_connection_capacity,
        ) for cp in designated_charging_parks
    ]

    edisgo_obj.electromobility.integrated_charging_parks_df = pd.DataFrame(
        columns=COLUMNS["integrated_charging_parks_df"], data=edisgo_id, index=charging_park_ids)


def charging_strategy(
        edisgo_obj, strategy="dumb", timestamp_share_threshold=0.2, minimum_charging_capacity_factor=0.1):
    def harmonize_charging_processes_df(
            df, timestamp_share_threshold, strategy=None, minimum_charging_capacity_factor=0.1):
        minimum_charging_time = df.chargingdemand/df.netto_charging_capacity \
                                * 60/edisgo_obj.electromobility.stepsize

        mask = (minimum_charging_time % 1) >= timestamp_share_threshold

        minimum_charging_time.loc[mask] = minimum_charging_time.apply(np.ceil)

        minimum_charging_time.loc[~mask] = minimum_charging_time.apply(np.floor)

        df = df.assign(
            minimum_charging_time=minimum_charging_time.astype(np.uint16),
            harmonized_chargingdemand=minimum_charging_time * df.netto_charging_capacity * edisgo_obj.electromobility.stepsize / 60,
            netto_charging_capacity_mva=df.netto_charging_capacity.divide(10**3),  # kW --> MW
        )

        if strategy == "reduced":
            parking_time = df.charge_end - df.charge_start + 1

            maximum_needed_charging_time = df.harmonized_chargingdemand /\
                                           (minimum_charging_capacity_factor * df.netto_charging_capacity)\
                                           * 60/edisgo_obj.electromobility.stepsize

            maximum_needed_charging_time = maximum_needed_charging_time.apply(np.floor).astype(np.uint16)

            mask = parking_time <= maximum_needed_charging_time

            df = df.assign(
                reduced_charging_time=0,
                reduced_charging_capacity=0,
            )

            df.loc[mask, "reduced_charging_time"] = parking_time.loc[mask]

            df.loc[~mask, "reduced_charging_time"] = maximum_needed_charging_time.loc[~mask]

            df.reduced_charging_capacity = df.harmonized_chargingdemand / df.reduced_charging_time \
                                           * 60/edisgo_obj.electromobility.stepsize

            df = df.assign(
                reduced_charging_capacity_mva=df.reduced_charging_capacity.divide(10**3),
            )

        return df

    def overwrite_timeseries(edisgo_obj, edisgo_id, ts):
        edisgo_obj.timeseries._charging_points_active_power[edisgo_id].values[:] = ts

    charging_parks = [
        cp for cp in list(edisgo_obj.electromobility.potential_charging_parks)
        if cp.grid is not None
    ]

    if strategy == "dumb":
        for cp in charging_parks:
            dummy_ts = np.zeros(len(edisgo_obj.timeseries.timeindex))

            charging_processes_df = harmonize_charging_processes_df(
                cp.charging_processes_df, timestamp_share_threshold, strategy)

            eta_cp = edisgo_obj.electromobility.eta_charging_points

            for _, row in charging_processes_df.iterrows():
                dummy_ts[row["charge_start"]:row["charge_start"] + row["minimum_charging_time"]] += \
                    row["netto_charging_capacity_mva"] / eta_cp

            print(sum(dummy_ts))

            overwrite_timeseries(edisgo_obj, cp.edisgo_id, dummy_ts)

    elif strategy == "reduced":
        for cp in charging_parks:
            dummy_ts = np.zeros(len(edisgo_obj.timeseries.timeindex))

            charging_processes_df = harmonize_charging_processes_df(
                cp.charging_processes_df, timestamp_share_threshold, strategy, minimum_charging_capacity_factor)

            eta_cp = edisgo_obj.electromobility.eta_charging_points

            for _, row in charging_processes_df.iterrows():
                dummy_ts[row["charge_start"]:row["charge_start"] + row["reduced_charging_time"]] += \
                    row["reduced_charging_capacity_mva"] / eta_cp

            print(sum(dummy_ts))

            overwrite_timeseries(edisgo_obj, cp.edisgo_id, dummy_ts)

    else:
        raise ValueError(f"Strategy {strategy} has not yet been implemented.")

    logging.info(f"Charging strategy {strategy} completed.")
