import logging
import numpy as np
import pandas as pd

from edisgo import EDisGo


COLUMNS = {
    "integrated_charging_parks_df": ["edisgo_id"],
}


def integrate_charging_points(edisgo_obj, comp_type="ChargingPoint"):
    charging_parks = list(edisgo_obj.electromobility.potential_charging_parks)

    designated_charging_parks = [
        _ for _ in charging_parks if _.designated_charging_point_capacity > 0]

    charging_park_ids = [_.id for _ in designated_charging_parks]

    dummy_timeseries = pd.Series([0]*len(edisgo_obj.timeseries.timeindex), index=edisgo_obj.timeseries.timeindex)

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
        edisgo_obj, strategy="dumb", time_share_threshold=0.2):
    def harmonize_charging_processes_df(df, time_share_threshold):
        total_charging_time = df.chargingdemand/df.netto_charging_capacity * 60/edisgo_obj.electromobility.stepsize

        mask = (total_charging_time % 1) >= time_share_threshold

        total_charging_time.loc[mask] = total_charging_time.apply(np.ceil)

        total_charging_time.loc[~mask] = total_charging_time.apply(np.floor)

        df = df.assign(
            total_charging_time=total_charging_time.astype(np.uint16),
            chargingdemand=total_charging_time * df.netto_charging_capacity * edisgo_obj.electromobility.stepsize/60,
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
                cp.charging_processes_df, time_share_threshold)

            eta_cp = edisgo_obj.electromobility.eta_charging_points

            for _, row in charging_processes_df.iterrows():
                dummy_ts[row["charge_start"]:row["charge_start"]+row["total_charging_time"]] += \
                    row["netto_charging_capacity"]/eta_cp

            overwrite_timeseries(edisgo_obj, cp.edisgo_id, dummy_ts)

    else:
        raise ValueError("Strategy {} has not yet been implemented.")

