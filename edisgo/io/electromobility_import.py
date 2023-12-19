from __future__ import annotations

import json
import logging
import os

from collections import Counter
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import saio

from numpy.random import default_rng
from sklearn import preprocessing
from sqlalchemy.engine.base import Engine

from edisgo.io.db import get_srid_of_db_table, session_scope_egon_data

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)

min_max_scaler = preprocessing.MinMaxScaler()

COLUMNS = {
    "integrated_charging_parks_df": ["edisgo_id"],
    "charging_processes_df": [
        "ags",
        "car_id",
        "destination",
        "use_case",
        "nominal_charging_capacity_kW",
        "grid_charging_capacity_kW",
        "chargingdemand_kWh",
        "park_time_timesteps",
        "park_start_timesteps",
        "park_end_timesteps",
    ],
    "simbev_config_df": [
        "eta_cp",
        "stepsize",
        "start_date",
        "end_date",
        "soc_min",
        "grid_timeseries",
        "grid_timeseries_by_usecase",
        "days",
    ],
    "matching_demand_and_location": ["charging_park_id", "charging_point_id"],
    "potential_charging_parks_gdf": [
        "ags",
        "use_case",
        "user_centric_weight",
        "geometry",
    ],
    "available_charging_points_df": [
        "park_end_timesteps",
        "nominal_charging_capacity_kW",
        "charging_park_id",
        "use_case",
    ],
}

DTYPES = {
    "charging_processes_df": {
        "ags": np.uint32,
        "car_id": np.uint32,
        "destination": str,
        "use_case": str,
        "nominal_charging_capacity_kW": np.float64,
        "grid_charging_capacity_kW": np.float64,
        "chargingdemand_kWh": np.float64,
        "park_time_timesteps": np.uint16,
        "park_start_timesteps": np.uint16,
        "park_end_timesteps": np.uint16,
    },
    "simbev_config_df": {
        "eta_cp": float,
        "stepsize": int,
        "soc_min": float,
        "grid_timeseries": bool,
        "grid_timeseries_by_usecase": bool,
    },
    "potential_charging_parks_gdf": {
        "ags": np.uint32,
        "use_case": str,
        "user_centric_weight": np.float64,
    },
}

KEEP_COLS = {"potential_charging_parks_gdf": ["user_centric_weight", "geometry"]}

USECASES = ["hpc", "public", "home", "work"]

PRIVATE_DESTINATIONS = {
    "0_work": "work",
    "6_home": "home",
}


def import_electromobility_from_dir(
    edisgo_obj: EDisGo,
    simbev_directory: PurePath | str,
    tracbev_directory: PurePath | str,
    **kwargs,
):
    """
    Import electromobility data from
    `SimBEV <https://github.com/rl-institut/simbev>`_ and
    `TracBEV <https://github.com/rl-institut/tracbev>`_ from directory.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    simbev_directory : str or pathlib.PurePath
        SimBEV directory holding SimBEV data.
    tracbev_directory : str or pathlib.PurePath
        TracBEV directory holding TracBEV data.
    kwargs :
        Kwargs may contain any further attributes you want to specify.

        gc_to_car_rate_home : float
            Specifies the minimum rate between potential charging parks
            points for the use case "home" and the total number of cars.
            Default 0.5 .
        gc_to_car_rate_work : float
            Specifies the minimum rate between potential charging parks
            points for the use case "work" and the total number of cars.
            Default 0.25 .
        gc_to_car_rate_public : float
            Specifies the minimum rate between potential charging parks
            points for the use case "public" and the total number of cars.
            Default 0.1 .
        gc_to_car_rate_hpc : float
            Specifies the minimum rate between potential charging parks
            points for the use case "hpc" and the total number of cars.
            Default 0.005 .
        mode_parking_times : str
            If the mode_parking_times is set to "frugal" only parking times
            with any charging demand are imported. Default "frugal".
        charging_processes_dir : str
            Charging processes sub-directory. Default None.
        simbev_config_file : str
            Name of the simbev config file. Default "metadata_simbev_run.json".

    """
    # TODO: SimBEV is in development and this import will need constant
    #  updating for now
    edisgo_obj.electromobility.charging_processes_df = read_csvs_charging_processes(
        simbev_directory,
        mode=kwargs.pop("mode_parking_times", "frugal"),
        csv_dir=kwargs.pop("charging_processes_dir", None),
    )

    edisgo_obj.electromobility.simbev_config_df = read_simbev_config_df(
        simbev_directory,
        edisgo_obj,
        simbev_config_file=kwargs.pop("simbev_config_file", "metadata_simbev_run.json"),
    )

    potential_charging_parks_gdf = read_gpkg_potential_charging_parks(
        tracbev_directory,
        edisgo_obj,
    )
    edisgo_obj.electromobility.potential_charging_parks_gdf = (
        assure_minimum_potential_charging_parks(
            edisgo_obj=edisgo_obj,
            potential_charging_parks_gdf=potential_charging_parks_gdf,
            **kwargs,
        )
    )


def read_csvs_charging_processes(csv_path, mode="frugal", csv_dir=None):
    """
    Reads all CSVs in a given path and returns a DataFrame with all
    `SimBEV <https://github.com/rl-institut/simbev>`_ charging processes.

    Parameters
    ----------
    csv_path : str
        Main path holding SimBEV output data
    mode : str
        Returns all information if None. Returns only rows with charging
        demand greater than 0 if "frugal". Default: "frugal".
    csv_dir : str
        Optional sub-directory holding charging processes CSVs under path.
        Default: None.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with AGS, car ID, trip destination, charging use case
        (private or public), netto charging capacity, charging demand,
        charge start, charge end, potential charging park ID and charging point
        ID.

    """
    if csv_dir is not None:
        csv_path = os.path.join(csv_path, csv_dir)

    files = []

    for dirpath, dirnames, filenames in os.walk(csv_path):
        files.extend(
            Path(os.path.join(dirpath, f)) for f in filenames if f.endswith(".csv")
        )

    if not files:
        raise ValueError(f"Couldn't find any CSVs in path {csv_path}.")

    files.sort()

    # wrapper function for csv files read in with map_except function
    def rd_csv(file):
        ags = int(file[1].parts[-2])
        car_id = file[0]
        try:
            return pd.read_csv(file[1]).assign(ags=ags, car_id=car_id)
        except Exception:
            logger.warning(f"File '{file[1]}' couldn't be read and is skipped.")

            return pd.DataFrame()

    df = pd.concat(map(rd_csv, list(enumerate(files))), ignore_index=True)

    if mode == "frugal":
        df = df.loc[df.chargingdemand_kWh > 0]

    df = df.rename(columns={"location": "destination"})

    df = df[COLUMNS["charging_processes_df"]].astype(DTYPES["charging_processes_df"])

    return pd.merge(
        df,
        pd.DataFrame(columns=COLUMNS["matching_demand_and_location"]),
        how="outer",
        left_index=True,
        right_index=True,
    )


def read_simbev_config_df(
    path, edisgo_obj, simbev_config_file="metadata_simbev_run.json"
):
    """
    Get `SimBEV <https://github.com/rl-institut/simbev>`_ config data.

    Parameters
    ----------
    path : str
        Main path holding SimBEV output data.
    edisgo_obj : :class:`~.EDisGo`
    simbev_config_file : str
        SimBEV config file name. Default: "metadata_simbev_run.json".

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with used random seed, used threads, stepsize in minutes,
        year, scenarette, simulated days, maximum number of cars per AGS,
        completed standing times and time series per AGS and used ramp up
        data CSV.

    """
    try:
        if simbev_config_file is not None:
            with open(os.path.join(path, simbev_config_file)) as f:
                data = json.load(f)

            df = pd.DataFrame.from_dict(
                data["config"]["basic"], orient="index"
            ).T.astype(DTYPES["simbev_config_df"])

            for col in ["start_date", "end_date"]:
                df[col] = pd.to_datetime(df[col])

            return df.assign(days=(df.end_date - df.start_date).iat[0].days + 1)

    except Exception:
        logger.warning(
            "SimBEV config file could not be imported. Charging point "
            "efficiency is set to 100%, the stepsize is set to 15 minutes "
            "and the simulated days are estimated from the charging "
            "processes."
        )

        mx_t = edisgo_obj.electromobility.charging_processes_df.park_end_timesteps.max()
        data = {
            "eta_cp": [1.0],
            "stepsize": [15],
            "days": [np.ceil(mx_t / (4 * 24))],
        }
        return pd.DataFrame(data=data, index=[0])


def read_gpkg_potential_charging_parks(path, edisgo_obj):
    """
    Get GeoDataFrame with all
    `TracBEV <https://github.com/rl-institut/tracbev>`_ potential charging parks.

    Parameters
    ----------
    path : str
        Main path holding TracBEV data.
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :geopandas:`GeoDataFrame`
        GeoDataFrame with AGS, charging use case (home, work, public or
        hpc), user-centric weight and geometry.

    """
    files = [f for f in os.listdir(path) if f.endswith(".gpkg")]

    potential_charging_parks_gdf_list = []

    if isinstance(path, str):
        path = Path(path)

    for f in files:
        gdf = gpd.read_file(path / f)

        if "undefined" in gdf.crs.name.lower():
            gdf = gdf.set_crs(epsg=3035, allow_override=True).to_crs(
                epsg=edisgo_obj.topology.grid_district["srid"]
            )
        else:
            gdf = gdf.to_crs(epsg=edisgo_obj.topology.grid_district["srid"])

        gdf = gdf.rename(
            columns={
                "charge_spots": "user_centric_weight",
                "potential": "user_centric_weight",
            }
        )

        # drop unnecessary columns
        gdf = gdf[KEEP_COLS["potential_charging_parks_gdf"]]

        # add ags and use case info as well as normalize weights 0..1
        gdf = gdf.assign(
            user_centric_weight=min_max_scaler.fit_transform(
                gdf.user_centric_weight.values.reshape(-1, 1)
            ),
            ags=int(f.split(".")[0].split("_")[-1]),
            use_case=f.split(".")[0].split("_")[-2],
        )

        potential_charging_parks_gdf_list.append(gdf)

    potential_charging_parks_gdf = gpd.GeoDataFrame(
        pd.concat(
            potential_charging_parks_gdf_list,
            ignore_index=True,
        ),
        crs=potential_charging_parks_gdf_list[0].crs,
    )

    return potential_charging_parks_gdf


def assure_minimum_potential_charging_parks(
    edisgo_obj: EDisGo,
    potential_charging_parks_gdf: gpd.GeoDataFrame,
    **kwargs,
):
    # ensure minimum number of potential charging parks per car
    num_cars = len(edisgo_obj.electromobility.charging_processes_df.car_id.unique())

    for use_case in USECASES:
        if use_case == "home":
            gc_to_car_rate = kwargs.get("gc_to_car_rate_home", 0.5)
        elif use_case == "work":
            gc_to_car_rate = kwargs.get("gc_to_car_rate_work", 0.25)
        elif use_case == "public":
            gc_to_car_rate = kwargs.get("gc_to_car_rate_public", 0.1)
        elif use_case == "hpc":
            gc_to_car_rate = kwargs.get("gc_to_car_rate_hpc", 0.005)

        use_case_gdf = potential_charging_parks_gdf.loc[
            potential_charging_parks_gdf.use_case == use_case
        ]

        num_gcs = len(use_case_gdf)

        # if tracbev doesn't provide possible grid connections choose a
        # random public potential charging park and duplicate
        if num_gcs == 0:
            logger.warning(
                f"There are no potential charging parks for use case {use_case}. "
                f"Therefore 10 % of public potential charging parks are duplicated "
                f"randomly and assigned to use case {use_case}."
            )

            public_gcs = potential_charging_parks_gdf.loc[
                potential_charging_parks_gdf.use_case == "public"
            ]

            random_gcs = public_gcs.sample(
                int(np.ceil(len(public_gcs) / 10)),
                random_state=edisgo_obj.topology.mv_grid.id,
            ).assign(use_case=use_case)

            potential_charging_parks_gdf = pd.concat(
                [
                    potential_charging_parks_gdf,
                    random_gcs,
                ],
                ignore_index=True,
            )
            use_case_gdf = potential_charging_parks_gdf.loc[
                potential_charging_parks_gdf.use_case == use_case
            ]
            num_gcs = len(use_case_gdf)

        # escape zero division
        actual_gc_to_car_rate = np.Infinity if num_cars == 0 else num_gcs / num_cars

        # duplicate potential charging parks until desired quantity is ensured
        max_it = 50
        n = 0

        while actual_gc_to_car_rate < gc_to_car_rate and n < max_it:
            logger.info(
                f"Duplicating potential charging parks to meet the desired grid "
                f"connections to cars rate of {gc_to_car_rate*100:.2f} % for use case "
                f"{use_case}. Iteration: {n+1}."
            )

            if actual_gc_to_car_rate * 2 < gc_to_car_rate:
                potential_charging_parks_gdf = pd.concat(
                    [
                        potential_charging_parks_gdf,
                        use_case_gdf,
                    ],
                    ignore_index=True,
                )

            else:
                extra_gcs = (
                    int(np.ceil(num_gcs * gc_to_car_rate / actual_gc_to_car_rate))
                    - num_gcs
                )

                extra_gdf = use_case_gdf.sample(
                    n=extra_gcs, random_state=edisgo_obj.topology.mv_grid.id
                )

                potential_charging_parks_gdf = pd.concat(
                    [
                        potential_charging_parks_gdf,
                        extra_gdf,
                    ],
                    ignore_index=True,
                )

            use_case_gdf = potential_charging_parks_gdf.loc[
                potential_charging_parks_gdf.use_case == use_case
            ]

            num_gcs = len(use_case_gdf)

            actual_gc_to_car_rate = num_gcs / num_cars

            n += 1

    # sort GeoDataFrame
    potential_charging_parks_gdf = potential_charging_parks_gdf.sort_values(
        by=["use_case", "ags", "user_centric_weight"], ascending=[True, True, False]
    ).reset_index(drop=True)

    # in case of polygons use the centroid as potential charging parks point
    # and set crs to match edisgo object
    return (
        potential_charging_parks_gdf.assign(
            geometry=potential_charging_parks_gdf.geometry.representative_point()
        )
        .to_crs(epsg=edisgo_obj.topology.grid_district["srid"])
        .astype(DTYPES["potential_charging_parks_gdf"])
    )


def distribute_charging_demand(edisgo_obj, **kwargs):
    """
    Distribute charging demand from SimBEV onto potential charging parks from TracBEV.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    kwargs :
        Kwargs may contain any further attributes you want to specify.

        mode : str
            Distribution mode. If the mode is set to "user_friendly" only the
            simbev weights are used for the distribution. If the mode is
            "grid_friendly" also grid conditions are respected.
            Default "user_friendly".
        generators_weight_factor : float
            Weighting factor of the generators weight within an LV grid in
            comparison to the loads weight. Default 0.5.
        distance_weight : float
            Weighting factor for the distance between a potential charging park
            and its nearest substation in comparison to the combination of
            the generators and load factors of the LV grids.
            Default 1 / 3.
        user_friendly_weight : float
            Weighting factor of the user friendly weight in comparison to the
            grid friendly weight. Default 0.5.

    """
    distribute_private_charging_demand(edisgo_obj)

    distribute_public_charging_demand(edisgo_obj, **kwargs)


def get_weights_df(edisgo_obj, potential_charging_park_indices, **kwargs):
    """
    Get weights per potential charging point for a given set of grid connection indices.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    potential_charging_park_indices : list
        List of potential charging parks indices

    Other Parameters
    -----------------
    mode : str
        Only use user friendly weights ("user_friendly") or combine with
        grid friendly weights ("grid_friendly"). Default: "user_friendly".
    user_friendly_weight : float
        Weight of user friendly weight if mode "grid_friendly". Default: 0.5.
    distance_weight: float
        Grid friendly weight is a combination of the installed capacity of
        generators and loads within a LV grid and the distance towards the
        nearest substation. This parameter sets the weight for the distance
        parameter. Default: 1/3.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with numeric weights

    """

    def _get_lv_grid_weights():
        """
        DataFrame containing technical data of LV grids.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Columns of the DataFrame are:
                peak_generation_capacity : float
                    Cumulative peak generation capacity of generators in the network in
                    MW.

                p_set : float
                    Cumulative peak load of loads in the network in MW.

                substation_capacity : float
                    Cumulative capacity of transformers to overlaying network.

                generators_weight : float
                    Weighting used in grid friendly siting of public charging points.
                    In the case of generators the weight is defined by dividing the
                    peak_generation_capacity by substation_capacity and norming the
                    results from 0 .. 1. A higher weight is more attractive.

                loads_weight : float
                    Weighting used in grid friendly siting of public charging points.
                    In the case of loads the weight is defined by dividing the
                    p_set by substation_capacity and norming the results from 0 .. 1.
                    The result is then substracted from 1 as the higher the p_set is
                    in relation to the substation_capacity the less attractive this LV
                    grid is for new loads from a grid perspective. A higher weight is
                    more attractive.

        """
        lv_grids = list(edisgo_obj.topology.mv_grid.lv_grids)

        lv_grids_df = pd.DataFrame(
            index=[_._id for _ in lv_grids],
            columns=[
                "peak_generation_capacity",
                "substation_capacity",
                "generators_weight",
                "p_set",
                "loads_weight",
            ],
        )

        lv_grids_df.peak_generation_capacity = [
            _.peak_generation_capacity for _ in lv_grids
        ]

        lv_grids_df.substation_capacity = [
            _.transformers_df.s_nom.sum() for _ in lv_grids
        ]

        min_max_scaler = preprocessing.MinMaxScaler()
        lv_grids_df.generators_weight = lv_grids_df.peak_generation_capacity.divide(
            lv_grids_df.substation_capacity
        )
        lv_grids_df.generators_weight = min_max_scaler.fit_transform(
            lv_grids_df.generators_weight.values.reshape(-1, 1)
        )

        lv_grids_df.p_set = [_.p_set for _ in lv_grids]

        lv_grids_df.loads_weight = lv_grids_df.p_set.divide(
            lv_grids_df.substation_capacity
        )
        lv_grids_df.loads_weight = 1 - min_max_scaler.fit_transform(
            lv_grids_df.loads_weight.values.reshape(-1, 1)
        )
        return lv_grids_df

    mode = kwargs.get("mode", "user_friendly")

    if mode == "user_friendly":
        weights = [
            _.user_centric_weight
            for _ in edisgo_obj.electromobility.potential_charging_parks
            if _.id in potential_charging_park_indices
        ]
    elif mode == "grid_friendly":
        potential_charging_parks = list(
            edisgo_obj.electromobility.potential_charging_parks
        )

        user_friendly_weights = [
            _.user_centric_weight
            for _ in potential_charging_parks
            if _.id in potential_charging_park_indices
        ]

        lv_grids_df = _get_lv_grid_weights()

        generators_weight_factor = kwargs.get("generators_weight_factor", 0.5)
        loads_weight_factor = 1 - generators_weight_factor

        combined_weights = (
            generators_weight_factor * lv_grids_df["generators_weight"]
            + loads_weight_factor * lv_grids_df["loads_weight"]
        )

        lv_grid_ids = [
            _.nearest_substation["lv_grid_id"] for _ in potential_charging_parks
        ]

        load_and_generator_capacity_weights = [
            combined_weights.at[lv_grid_id] for lv_grid_id in lv_grid_ids
        ]

        # fmt: off
        distance_weights = (
            edisgo_obj.electromobility._potential_charging_parks_df.distance_weight
            .tolist()
        )
        # fmt: on

        distance_weight = kwargs.get("distance_weight", 1 / 3)

        grid_friendly_weights = [
            (1 - distance_weight) * load_and_generator_capacity_weights[i]
            + distance_weight * distance_weights[i]
            for i in range(len(distance_weights))
        ]

        user_friendly_weight = kwargs.get("user_friendly_weight", 0.5)

        weights = [
            (1 - user_friendly_weight) * grid_friendly_weights[i]
            + user_friendly_weight * user_friendly_weights[i]
            for i in range(len(grid_friendly_weights))
        ]

    else:
        raise ValueError(
            "Provided mode is not valid, needs to be 'user_friendly' or "
            "'grid_friendly'."
        )
    return pd.DataFrame(weights)


def normalize(weights_df):
    """
    Normalize a given DataFrame so that its sum equals 1 and return a
    flattened Array.

    Parameters
    ----------
    weights_df : :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with single numeric column

    Returns
    -------
    Numpy 1-D array
        Array with normalized weights

    """
    if weights_df.sum().sum() == 0:
        return np.array([1 / len(weights_df) for _ in range(len(weights_df))])
    else:
        return weights_df.divide(weights_df.sum().sum()).T.to_numpy().flatten()


def combine_weights(
    potential_charging_park_indices, designated_charging_point_capacity_df, weights_df
):
    """
    Add designated charging capacity weights into the initial weights and
    normalize weights

    Parameters
    ----------
    potential_charging_park_indices : list
        List of potential charging parks indices
    designated_charging_point_capacity_df :
        :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with designated charging point capacity per potential
        charging park
    weights_df : :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with initial user or combined weights

    Returns
    -------
    Numpy 1-D array
        Array with normalized weights

    """
    capacity_df = designated_charging_point_capacity_df.loc[
        potential_charging_park_indices
    ]

    capacity_weights = (
        1
        - min_max_scaler.fit_transform(
            capacity_df.designated_charging_point_capacity.values.reshape(-1, 1)
        )
    ).flatten()

    user_df = weights_df.loc[potential_charging_park_indices]

    user_df[0] += capacity_weights

    return normalize(user_df)


def weighted_random_choice(
    edisgo_obj,
    potential_charging_park_indices,
    car_id,
    destination,
    charging_point_id,
    normalized_weights,
    rng=None,
):
    """
    Weighted random choice of a potential charging park. Setting the chosen
    values into :obj:`~.network.electromobility.Electromobility.charging_processes_df`

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    potential_charging_park_indices : list
        List of potential charging parks indices
    car_id : int
        Car ID
    destination : str
        Trip destination
    charging_point_id : int
        Charging Point ID
    normalized_weights : Numpy 1-D array
        Array with normalized weights
    rng : Numpy random generator
        If None a random generator with seed=charging_point_id is
        initialized

    Returns
    -------
    :obj:`int`
        Chosen Charging Park ID

    """
    if rng is None:
        rng = default_rng(seed=charging_point_id)

    charging_park_id = rng.choice(
        a=potential_charging_park_indices,
        p=normalized_weights,
    )

    edisgo_obj.electromobility.charging_processes_df.loc[
        (edisgo_obj.electromobility.charging_processes_df.car_id == car_id)
        & (edisgo_obj.electromobility.charging_processes_df.destination == destination)
    ] = edisgo_obj.electromobility.charging_processes_df.loc[
        (edisgo_obj.electromobility.charging_processes_df.car_id == car_id)
        & (edisgo_obj.electromobility.charging_processes_df.destination == destination)
    ].assign(
        charging_park_id=charging_park_id,
        charging_point_id=charging_point_id,
    )

    return charging_park_id


def distribute_private_charging_demand(edisgo_obj):
    """
    Distributes all private charging processes. Each car gets its own
    private charging point if a charging process takes place.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    """
    try:
        rng = default_rng(seed=edisgo_obj.topology.id)
    except Exception:
        rng = None

    private_charging_df = edisgo_obj.electromobility.charging_processes_df.loc[
        (edisgo_obj.electromobility.charging_processes_df.chargingdemand_kWh > 0)
        & edisgo_obj.electromobility.charging_processes_df.use_case.isin(
            ["home", "work"]
        )
    ]

    charging_point_id = 0

    user_centric_weights_df = get_weights_df(
        edisgo_obj, edisgo_obj.electromobility.potential_charging_parks_gdf.index
    )

    designated_charging_point_capacity_df = pd.DataFrame(
        index=user_centric_weights_df.index,
        columns=["designated_charging_point_capacity"],
        data=0,
    )

    for destination in private_charging_df.destination.sort_values().unique():
        private_charging_destination_df = private_charging_df.loc[
            private_charging_df.destination == destination
        ]

        use_case = PRIVATE_DESTINATIONS[destination]

        if use_case == "work":
            potential_charging_park_indices = (
                edisgo_obj.electromobility.potential_charging_parks_gdf.loc[
                    edisgo_obj.electromobility.potential_charging_parks_gdf.use_case
                    == use_case
                ].index
            )

            for car_id in private_charging_destination_df.car_id.sort_values().unique():
                weights = combine_weights(
                    potential_charging_park_indices,
                    designated_charging_point_capacity_df,
                    user_centric_weights_df,
                )

                charging_park_id = weighted_random_choice(
                    edisgo_obj,
                    potential_charging_park_indices,
                    car_id,
                    destination,
                    charging_point_id,
                    weights,
                    rng=rng,
                )

                charging_capacity = (
                    private_charging_destination_df.loc[
                        (private_charging_destination_df.car_id == car_id)
                        & (private_charging_destination_df.destination == "0_work")
                    ].nominal_charging_capacity_kW.iat[0]
                    / edisgo_obj.electromobility.eta_charging_points
                )

                designated_charging_point_capacity_df.at[
                    charging_park_id, "designated_charging_point_capacity"
                ] += charging_capacity

                charging_point_id += 1

        elif use_case == "home":
            for ags in private_charging_destination_df.ags.sort_values().unique():
                private_charging_ags_df = private_charging_destination_df.loc[
                    private_charging_destination_df.ags == ags
                ]

                # fmt: off
                potential_charging_park_indices = edisgo_obj.electromobility.\
                    potential_charging_parks_gdf.loc[
                        (
                            edisgo_obj.electromobility.potential_charging_parks_gdf.ags
                            == ags
                        )
                        & (
                            edisgo_obj.electromobility.potential_charging_parks_gdf.
                            use_case == use_case
                        )
                    ].index
                # fmt: on

                for car_id in private_charging_ags_df.car_id.sort_values().unique():
                    weights = combine_weights(
                        potential_charging_park_indices,
                        designated_charging_point_capacity_df,
                        user_centric_weights_df,
                    )

                    weighted_random_choice(
                        edisgo_obj,
                        potential_charging_park_indices,
                        car_id,
                        destination,
                        charging_point_id,
                        weights,
                        rng=rng,
                    )

                    charging_capacity = private_charging_destination_df.loc[
                        (private_charging_destination_df.car_id == car_id)
                        & (private_charging_destination_df.destination == "6_home")
                    ].nominal_charging_capacity_kW.iat[0]

                    designated_charging_point_capacity_df.at[
                        charging_park_id, "designated_charging_point_capacity"
                    ] += charging_capacity

                    charging_point_id += 1

        else:
            raise ValueError(f"Destination {destination} is unknown.")


def distribute_public_charging_demand(edisgo_obj, **kwargs):
    """
    Distributes all public charging processes. For each process it is
    checked if a matching charging point exists to minimize the
    number of charging points.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    """
    public_charging_df = edisgo_obj.electromobility.charging_processes_df.loc[
        (edisgo_obj.electromobility.charging_processes_df.chargingdemand_kWh > 0)
        & edisgo_obj.electromobility.charging_processes_df.use_case.isin(
            ["public", "hpc"]
        )
    ].sort_values(
        by=["park_start_timesteps", "park_end_timesteps"],
        ascending=[True, True],
    )

    try:
        rng = default_rng(seed=edisgo_obj.topology.id)
    except Exception:
        rng = default_rng(seed=1)

    available_charging_points_df = pd.DataFrame(
        columns=COLUMNS["available_charging_points_df"]
    )

    grid_and_user_centric_weights_df = get_weights_df(
        edisgo_obj,
        edisgo_obj.electromobility.potential_charging_parks_gdf.index,
        **kwargs,
    )

    designated_charging_point_capacity_df = pd.DataFrame(
        index=grid_and_user_centric_weights_df.index,
        columns=["designated_charging_point_capacity"],
        data=0,
    )

    columns = [
        "destination",
        "use_case",
        "park_start_timesteps",
        "park_end_timesteps",
        "nominal_charging_capacity_kW",
    ]

    for (
        idx,
        destination,
        use_case,
        park_start_timesteps,
        park_end_timesteps,
        nominal_charging_capacity_kW,
    ) in public_charging_df[columns].itertuples():
        matching_charging_points_df = available_charging_points_df.loc[
            (available_charging_points_df.park_end_timesteps < park_start_timesteps)
            & (
                available_charging_points_df.nominal_charging_capacity_kW.round(1)
                == round(nominal_charging_capacity_kW, 1)
            )
        ]

        if len(matching_charging_points_df) > 0:
            potential_charging_park_indices = matching_charging_points_df.index

            weights = normalize(
                grid_and_user_centric_weights_df.loc[
                    matching_charging_points_df.charging_park_id
                ]
            )

            charging_point_s = matching_charging_points_df.loc[
                rng.choice(a=potential_charging_park_indices, p=weights)
            ]

            edisgo_obj.electromobility.charging_processes_df.at[
                idx, "charging_park_id"
            ] = charging_point_s["charging_park_id"]

            edisgo_obj.electromobility.charging_processes_df.at[
                idx, "charging_point_id"
            ] = charging_point_s.name

            available_charging_points_df.at[
                charging_point_s.name, "park_end_timesteps"
            ] = park_end_timesteps

        else:
            potential_charging_park_indices = (
                edisgo_obj.electromobility.potential_charging_parks_gdf.loc[
                    (
                        edisgo_obj.electromobility.potential_charging_parks_gdf.use_case
                        == use_case
                    )
                ].index
            )

            weights = combine_weights(
                potential_charging_park_indices,
                designated_charging_point_capacity_df,
                grid_and_user_centric_weights_df,
            )

            charging_park_id = rng.choice(
                a=potential_charging_park_indices,
                p=weights,
            )

            # fmt: off
            charging_point_id = (
                edisgo_obj.electromobility.charging_processes_df.charging_point_id
                .max()
                + 1
            )
            # fmt: on

            if charging_point_id != charging_point_id:
                charging_point_id = 0

            edisgo_obj.electromobility.charging_processes_df.at[
                idx, "charging_park_id"
            ] = charging_park_id

            edisgo_obj.electromobility.charging_processes_df.at[
                idx, "charging_point_id"
            ] = charging_point_id

            available_charging_points_df.loc[
                charging_point_id
            ] = edisgo_obj.electromobility.charging_processes_df.loc[
                idx, available_charging_points_df.columns
            ].tolist()

            designated_charging_point_capacity_df.at[
                charging_park_id, "designated_charging_point_capacity"
            ] += nominal_charging_capacity_kW


def determine_grid_connection_capacity(
    total_charging_point_capacity, lower_limit=0.3, upper_limit=1.0, minimum_factor=0.45
):
    if total_charging_point_capacity <= lower_limit:
        return total_charging_point_capacity
    elif total_charging_point_capacity >= upper_limit:
        return minimum_factor * total_charging_point_capacity
    else:
        return (
            ((minimum_factor - 1) / (upper_limit - lower_limit))
            * (total_charging_point_capacity - lower_limit)
            + 1
        ) * total_charging_point_capacity


def integrate_charging_parks(edisgo_obj):
    """
    Integrates all designated charging parks into the grid.

    The charging time series at each charging park are not set in this function.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    """
    charging_parks = list(edisgo_obj.electromobility.potential_charging_parks)

    # Only integrate charging parks with designated charging points
    designated_charging_parks = [
        cp
        for cp in charging_parks
        if (cp.designated_charging_point_capacity > 0) and cp.within_grid
    ]

    charging_park_ids = [_.id for _ in designated_charging_parks]

    comp_type = "charging_point"

    # integrate ChargingPoints and save the names of the eDisGo ID
    edisgo_ids = [
        edisgo_obj.integrate_component_based_on_geolocation(
            comp_type=comp_type,
            geolocation=cp.geometry,
            sector=cp.use_case,
            add_ts=False,
            p_set=cp.grid_connection_capacity,
        )
        for cp in designated_charging_parks
    ]

    edisgo_obj.electromobility.integrated_charging_parks_df = pd.DataFrame(
        columns=COLUMNS["integrated_charging_parks_df"],
        data=edisgo_ids,
        index=charging_park_ids,
    )


def import_electromobility_from_oedb(
    edisgo_obj: EDisGo,
    scenario: str,
    engine: Engine,
    **kwargs,
):
    """
    Gets electromobility data for specified scenario from oedb.

    Electromobility data includes data on standing times, charging demand,
    etc. per vehicle, as well as information on potential charging point locations.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve electromobility data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Other Parameters
    ----------------
    kwargs :
        Possible options are `gc_to_car_rate_home`, `gc_to_car_rate_work`,
        `gc_to_car_rate_public`, `gc_to_car_rate_hpc`, and `mode_parking_times`. See
        parameter documentation of `import_electromobility_data_kwds` parameter in
        :attr:`~.EDisGo.import_electromobility` for more information.

    """
    edisgo_obj.electromobility.charging_processes_df = charging_processes_from_oedb(
        edisgo_obj=edisgo_obj, engine=engine, scenario=scenario, **kwargs
    )
    edisgo_obj.electromobility.simbev_config_df = simbev_config_from_oedb(
        scenario=scenario, engine=engine
    )
    potential_charging_parks_gdf = potential_charging_parks_from_oedb(
        edisgo_obj=edisgo_obj, engine=engine, **kwargs
    )
    edisgo_obj.electromobility.potential_charging_parks_gdf = (
        assure_minimum_potential_charging_parks(
            edisgo_obj=edisgo_obj,
            potential_charging_parks_gdf=potential_charging_parks_gdf,
            **kwargs,
        )
    )


def simbev_config_from_oedb(
    scenario: str,
    engine: Engine,
):
    """
    Gets :attr:`~.network.electromobility.Electromobility.simbev_config_df`
    for specified scenario from oedb.

    Parameters
    ----------
    scenario : str
        Scenario for which to retrieve electromobility data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        See :attr:`~.network.electromobility.Electromobility.simbev_config_df` for
        more information.

    """
    saio.register_schema("demand", engine)
    from saio.demand import egon_ev_metadata

    with session_scope_egon_data(engine) as session:
        query = session.query(egon_ev_metadata).filter(
            egon_ev_metadata.scenario == scenario
        )

        df = pd.read_sql(sql=query.statement, con=query.session.bind)

    return df.assign(days=(df.end_date - df.start_date).iat[0].days + 1)


def potential_charging_parks_from_oedb(
    edisgo_obj: EDisGo,
    engine: Engine,
):
    """
    Gets :attr:`~.network.electromobility.Electromobility.potential_charging_parks_gdf`
    data from oedb.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    --------
    :geopandas:`geopandas.GeoDataFrame<GeoDataFrame>`
        See
        :attr:`~.network.electromobility.Electromobility.potential_charging_parks_gdf`
        for more information.

    """
    saio.register_schema("grid", engine)
    from saio.grid import egon_emob_charging_infrastructure

    crs = edisgo_obj.topology.grid_district["srid"]

    with session_scope_egon_data(engine) as session:
        srid = get_srid_of_db_table(session, egon_emob_charging_infrastructure.geometry)

        query = session.query(
            egon_emob_charging_infrastructure.cp_id,
            egon_emob_charging_infrastructure.use_case,
            egon_emob_charging_infrastructure.weight.label("user_centric_weight"),
            egon_emob_charging_infrastructure.geometry.label("geom"),
        ).filter(egon_emob_charging_infrastructure.mv_grid_id == edisgo_obj.topology.id)

        gdf = gpd.read_postgis(
            sql=query.statement,
            con=query.session.bind,
            geom_col="geom",
            crs=f"EPSG:{srid}",
            index_col="cp_id",
        ).to_crs(crs)

    return gdf.assign(ags=0)


def charging_processes_from_oedb(
    edisgo_obj: EDisGo, engine: Engine, scenario: str, **kwargs
):
    """
    Gets :attr:`~.network.electromobility.Electromobility.charging_processes_df` data
    for specified scenario from oedb.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    scenario : str
        Scenario for which to retrieve data. Possible options are 'eGon2035' and
        'eGon100RE'.

    Other Parameters
    ----------------
    kwargs :
        Possible option is `mode_parking_times`. See parameter documentation of
        `import_electromobility_data_kwds` parameter in
        :attr:`~.EDisGo.import_electromobility` for more information.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        See :attr:`~.network.electromobility.Electromobility.charging_processes_df` for
        more information.

    """

    saio.register_schema("demand", engine)
    from saio.demand import egon_ev_mv_grid_district, egon_ev_trip

    # get EV pool in grid
    scenario_variation = {"eGon2035": "NEP C 2035", "eGon100RE": "Reference 2050"}
    with session_scope_egon_data(engine) as session:
        query = session.query(egon_ev_mv_grid_district.egon_ev_pool_ev_id).filter(
            egon_ev_mv_grid_district.scenario == scenario,
            egon_ev_mv_grid_district.scenario_variation == scenario_variation[scenario],
            egon_ev_mv_grid_district.bus_id == edisgo_obj.topology.id,
        )

        pool = Counter(pd.read_sql(sql=query.statement, con=engine).egon_ev_pool_ev_id)

    # get charging processes for each EV ID
    with session_scope_egon_data(engine) as session:
        query = session.query(
            egon_ev_trip.egon_ev_pool_ev_id.label("car_id"),
            egon_ev_trip.use_case,
            egon_ev_trip.location.label("destination"),
            egon_ev_trip.charging_capacity_nominal.label(
                "nominal_charging_capacity_kW"
            ),
            egon_ev_trip.charging_capacity_grid.label("grid_charging_capacity_kW"),
            egon_ev_trip.charging_demand.label("chargingdemand_kWh"),
            egon_ev_trip.park_start.label("park_start_timesteps"),
            egon_ev_trip.park_end.label("park_end_timesteps"),
        ).filter(
            egon_ev_trip.scenario == scenario,
            egon_ev_trip.egon_ev_pool_ev_id.in_(pool.keys()),
        )
        if kwargs.get("mode_parking_times", "frugal") == "frugal":
            query = query.filter(egon_ev_trip.charging_demand > 0)
        ev_trips_df = pd.read_sql(sql=query.statement, con=engine)

    # duplicate EVs that were chosen more than once from EV pool
    df_list = []
    last_id = 0
    n_max = max(pool.values())
    for i in range(n_max, 0, -1):
        evs = sorted([ev_id for ev_id, count in pool.items() if count >= i])
        df = ev_trips_df.loc[ev_trips_df.car_id.isin(evs)]
        mapping = {ev: count + last_id for count, ev in enumerate(evs)}
        df.car_id = df.car_id.map(mapping)
        last_id = max(mapping.values()) + 1
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    # make sure count starts at 0
    if df.park_start_timesteps.min() == 1:
        df.loc[:, ["park_start_timesteps", "park_end_timesteps"]] -= 1

    return df.assign(
        ags=0,
        park_time_timesteps=df.park_end_timesteps - df.park_start_timesteps + 1,
        charging_park_id=np.nan,
        charging_point_id=np.nan,
    ).astype(DTYPES["charging_processes_df"])
