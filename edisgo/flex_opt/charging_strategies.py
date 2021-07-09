import logging
import numpy as np
import pandas as pd

from edisgo import EDisGo

COLUMNS = {
    "integrated_charging_parks_df": ["edisgo_id"],
}

logger = logging.getLogger("edisgo")


def integrate_charging_points(edisgo_obj, comp_type="ChargingPoint"):
    """
    Integrates all designated charging parks into the grid. The charging demand is
    not integrated here, but an empty dummy timeseries is generated.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    comp_type : str
        Component Type. Default "ChargingPoint"

    """
    charging_parks = list(edisgo_obj.electromobility.potential_charging_parks)

    # Only integrate charging parks with designated charging points
    designated_charging_parks = [
        _ for _ in charging_parks if _.designated_charging_point_capacity > 0]

    charging_park_ids = [_.id for _ in designated_charging_parks]

    dummy_timeseries = pd.Series(
        [0] * len(edisgo_obj.timeseries.timeindex), index=edisgo_obj.timeseries.timeindex)

    # integrate ChargingPoints and save the names of the eDisGo ID
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
    """
    Calculates the timeseries per charging park if parking times are given.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    strategy : str
        The charging strategy. Default "dumb". Only "private" charging processes at "home"
        or at "work" can be flexibilized. "Public" charging processes will always be "dumb".
        For now the following charging strategies are valid:
            "dumb": The cars are charged directly after arrival with the maximum possible
            charging capacity.
            "reduced": The cars are charged directly after arrival with the minimum possible
            charging capacity. The minimum possible charging capacity is determined by the
            parking time and the minimum_charging_capacity_factor.
            "residual": The cars are charged when the residual load in the MV grid is at it's
            lowest (high generation and low consumption). Charging processes with a low
            flexibility band are given priority.
    timestamp_share_threshold : float
        Percental threshold of the time required at a time step for charging the vehicle.
        If the time requirement is below this limit, then the charging process is not mapped
        into the time series. If, however, it is above this limit, the time step is mapped
        to 100% into the time series. This prevents differences between the charging strategies
        and creates a compromise between the simultaneity of charging processes and an
        artificial increase in the charging demand. Default 0.2
    minimum_charging_capacity_factor : float
        Technical percental minimum charging capacity per charging point. Default 0.1

    """
    def harmonize_charging_processes_df(
            df, timestamp_share_threshold, strategy=None, minimum_charging_capacity_factor=0.1, eta_cp=1.0):
        """
        Harmonizes the charging processes to prevent differences in the energy
        demand per charging strategy.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Charging processes DataFrame
        timestamp_share_threshold : float
            See description in the main function. Default 0.2
        strategy : str
            See description in the main function. Default "dumb"
        minimum_charging_capacity_factor : float
            See description in the main function. Default 0.1
        eta_cp : float
            Charging Point efficiency. Default 1.0

        """
        # SimBEV has a MATLAB legacy and at the moment 1 is eDisGos 0
        df.charge_start -= 1

        # calculate the minimum time taken the fulfill the charging demand
        minimum_charging_time = df.chargingdemand/df.netto_charging_capacity \
                                * 60/edisgo_obj.electromobility.stepsize

        # calculate in which time steps the last time step needed to fulfill the
        # charging demand is considered in the timeseries
        mask = (minimum_charging_time % 1) >= timestamp_share_threshold

        minimum_charging_time.loc[mask] = minimum_charging_time.apply(np.ceil)

        minimum_charging_time.loc[~mask] = minimum_charging_time.apply(np.floor)

        # recalculate the charging demand from the charging capacity
        # and the minimum charging time
        # Calculate the grid sided charging capacity in MVA
        df = df.assign(
            minimum_charging_time=minimum_charging_time.astype(np.uint16),
            harmonized_chargingdemand=minimum_charging_time * df.netto_charging_capacity * edisgo_obj.electromobility.stepsize / 60,
            netto_charging_capacity_mva=df.netto_charging_capacity.divide(10**3).divide(eta_cp),  # kW --> MW
        )

        if strategy == "reduced":
            parking_time = df.charge_end - df.charge_start

            # calculate the maximum needed charging time with the minimum charging capacity
            maximum_needed_charging_time = df.harmonized_chargingdemand /\
                                           (minimum_charging_capacity_factor * df.netto_charging_capacity)\
                                           * 60/edisgo_obj.electromobility.stepsize

            maximum_needed_charging_time = maximum_needed_charging_time.apply(np.floor).astype(np.uint16)

            # when the parking time is less than the maximum needed charging time, the total charging time
            # equates the parking time and the charging capacity is recalculated accordingly
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

        elif strategy == "residual":
            # the flex time/band is defined as the amount of time steps not needed to fulfill
            # the charging demand in a parking process
            parking_time = df.charge_end - df.charge_start

            df = df.assign(
                flex_time=parking_time - df.minimum_charging_time
            )

            df = df.sort_values(by=["flex_time", "charge_start"], ascending=[True, True])

        return df

    def overwrite_timeseries(edisgo_obj, edisgo_id, ts):
        """
        Overwrites the dummy timeseries for the Charging Point

        Parameters
        ----------
        edisgo_obj : :class:`~.EDisGo`
        edisgo_id : str
            eDisGo ID of the Charging Point
        ts : array_like or float
            New timeseries

        """
        edisgo_obj.timeseries._charging_points_active_power[edisgo_id] = ts

    # get integrated charging parks
    charging_parks = [
        cp for cp in list(edisgo_obj.electromobility.potential_charging_parks)
        if cp.grid is not None
    ]

    # Reset possible old timeseries as these influence "residual" charging
    for cp in charging_parks:
        overwrite_timeseries(edisgo_obj, cp.edisgo_id, 0)

    eta_cp = edisgo_obj.electromobility.eta_charging_points

    if strategy == "dumb":
        # "dumb" charging
        for cp in charging_parks:
            dummy_ts = np.zeros(len(edisgo_obj.timeseries.timeindex))

            charging_processes_df = harmonize_charging_processes_df(
                cp.charging_processes_df, timestamp_share_threshold, strategy=strategy, eta_cp=eta_cp)

            for _, row in charging_processes_df.iterrows():
                dummy_ts[row["charge_start"]:row["charge_start"] + row["minimum_charging_time"]] += \
                    row["netto_charging_capacity_mva"]

            overwrite_timeseries(edisgo_obj, cp.edisgo_id, dummy_ts)

    elif strategy == "reduced":
        # "reduced" charging
        for cp in charging_parks:
            dummy_ts = np.zeros(len(edisgo_obj.timeseries.timeindex))

            charging_processes_df = harmonize_charging_processes_df(
                cp.charging_processes_df, timestamp_share_threshold, strategy=strategy,
                minimum_charging_capacity_factor=minimum_charging_capacity_factor, eta_cp=eta_cp)

            for _, row in charging_processes_df.iterrows():
                if row["use_case"] == "public":
                    # if the charging process takes place in a "public" setting
                    # the charging is "dumb"
                    dummy_ts[row["charge_start"]:row["charge_start"] + row["minimum_charging_time"]] += \
                        row["netto_charging_capacity_mva"]
                else:
                    dummy_ts[row["charge_start"]:row["charge_start"] + row["reduced_charging_time"]] += \
                        row["reduced_charging_capacity_mva"]

            overwrite_timeseries(edisgo_obj, cp.edisgo_id, dummy_ts)

    elif strategy == "residual":
        # "residual" charging
        charging_processes_df = harmonize_charging_processes_df(
                edisgo_obj.electromobility.charging_processes_df, timestamp_share_threshold,
                strategy=strategy, eta_cp=eta_cp)

        # get residual load
        init_residual_load = edisgo_obj.timeseries.residual_load.multiply(-1).to_numpy()

        dummy_ts = pd.DataFrame(
            data=0, columns=[_.id for _ in charging_parks], index=edisgo_obj.timeseries.timeindex)

        # determine which charging processes can be flexibilized
        dumb_charging_processes_df = charging_processes_df.loc[
            (charging_processes_df.use_case == "public") |
            (charging_processes_df.flex_time == 0)
        ]

        flex_charging_processes_df = charging_processes_df.loc[
            ~charging_processes_df.index.isin(dumb_charging_processes_df.index)]

        # perform dumb charging processes and respect them in the residual load
        for _, row in dumb_charging_processes_df.iterrows():
            dummy_ts[row["grid_connection_point_id"]].iloc[
                row["charge_start"]:row["charge_start"] + row["minimum_charging_time"]
            ] += row["netto_charging_capacity_mva"]

        residual_load = init_residual_load + dummy_ts.sum(axis=1).values

        for _, row in flex_charging_processes_df.iterrows():
            flex_band = residual_load[row["charge_start"]:row["charge_end"]]

            k = row["minimum_charging_time"]

            # get k time steps with the lowest residual load in the parking time
            idx = np.argpartition(flex_band, k)[:k] + row["charge_start"]

            dummy_ts[row["grid_connection_point_id"]].iloc[idx] += \
                row["netto_charging_capacity_mva"]

            residual_load[idx] += row["netto_charging_capacity_mva"]

        for count, col in enumerate(dummy_ts.columns):
            overwrite_timeseries(
                edisgo_obj, charging_parks[count].edisgo_id, dummy_ts[col].to_numpy())

    else:
        raise ValueError(f"Strategy {strategy} has not yet been implemented.")

    logging.info(f"Charging strategy {strategy} completed.")
