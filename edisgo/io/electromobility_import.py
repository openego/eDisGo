import logging
import os
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn import preprocessing
from pathlib import Path
from numpy.random import default_rng

logger = logging.getLogger("edisgo")

min_max_scaler = preprocessing.MinMaxScaler()

COLUMNS = {
    "charging_processes_df": [
        "ags", "car_id", "destination", "use_case", "netto_charging_capacity",
        "chargingdemand", "park_start", "park_end"
    ],
    "matching_demand_and_location": ["grid_connection_point_id", "charging_point_id"],
    "grid_connections_gdf": ["ags", "use_case", "user_centric_weight", "geometry"],
    "simbev_config_df": ["value"],
    "available_charging_points_df": ["park_end", "netto_charging_capacity", "grid_connection_point_id", "use_case"],
}

DTYPES = {
    "charging_processes_df": {
        "ags": np.uint32,
        "car_id": np.uint32,
        "destination": str,
        "use_case": str,
        "netto_charging_capacity": np.float64,
        "chargingdemand": np.float64,
        "park_start": np.uint16,
        "park_end": np.uint16,
    },
    "grid_connections_gdf": {
        "ags": np.uint32,
        "use_case": str,
        "user_centric_weight": np.float64,
    },
}

USECASES = {
    "uc1": "hpc",
    "uc2": "public",
    "uc3": "home",
    "uc4": "work",
}

PRIVATE_DESTINATIONS = {
    "0_work": "work",
    "6_home": "home",
}

PUBLIC_DESTINATIONS = {
    "0_work": "public",
    "1_business": "public",
    "2_school": "public",
    "3_shopping": "public",
    "4_private/ridesharing": "public",
    "5_leisure": "public",
    "6_home": "public",
    "7_charging_hub": "hpc",
}


def import_simbev_electromobility(path, edisgo_obj, **kwargs):
    # TODO: SimBEV is in development and this import will need constant updating for now
    def read_csvs_charging_processes(path, mode=None, dir=None):
        """
        Reads all CSVs in a given path and returns a DataFrame with all
        `SimBEV <https://github.com/rl-institut/simbev>`_ charging processes.

        Parameters
        ----------
        path : str
            Main path holding SimBEV output data
        mode : str
            Returns all information if None. Returns only rows with charging demand
            greater than 0 if 'frugal'. Default is None.
        dir : str
            Optional sub-directory holding charging processes CSVs under path

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with AGS, car ID, trip destination, charging use case (private or public),
            netto charging capacity, charging demand, charge start, charge end, grid connection point and
            charging point ID.

        """
        if dir is not None:
            path = os.path.join(path, dir)

        files = []

        for dirpath, dirnames, filenames in os.walk(path):
            files.extend(Path(os.path.join(dirpath, f)) for f in filenames if f.endswith(".csv"))

        if len(files) == 0:
            raise ValueError(
                "Couldn't find any CSVs in path {}.".format(path)
            )

        files.sort()

        charging_processes_df = pd.DataFrame(columns=COLUMNS["charging_processes_df"])

        charging_processes_df = charging_processes_df.astype(DTYPES["charging_processes_df"])

        for car_id, f in enumerate(files):
            df = pd.read_csv(f, index_col=[0])

            if mode == "frugal":
                df = df.loc[df.chargingdemand > 0]
            else:
                pass

            df = df.rename(columns={"location": "destination"})

            df = df.assign(ags=int(f.parts[-2]), car_id=car_id)

            df = df[COLUMNS["charging_processes_df"]].astype(DTYPES["charging_processes_df"])

            charging_processes_df = charging_processes_df.append(
                df, ignore_index=True,
            )

        charging_processes_df = pd.merge(
            charging_processes_df,
            pd.DataFrame(columns=COLUMNS["matching_demand_and_location"]),
            how="outer",
            left_index=True,
            right_index=True
        )

        return charging_processes_df

    def read_csv_simbev_config(path, simbev_config_file=None):
        """
        Get `SimBEV <https://github.com/rl-institut/simbev>`_ config data.

        Parameters
        ----------
        path : str
            Main path holding SimBEV output data
        simbev_config_file : str
            SimBEV config file name

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with used random seed, used threads, stepsize in minutes, year, scenarette,
            simulated days, maximum number of cars per AGS, completed standing times and timeseries per AGS
            and used ramp up data CSV.

        """
        try:
            if simbev_config_file is not None:
                return pd.read_csv(
                    os.path.join(path, simbev_config_file), index_col=[0], header=0, names=COLUMNS["simbev_config_df"])
        except:
            return pd.DataFrame(columns=COLUMNS["simbev_config_df"])

    def read_geojsons_grid_connections(path, dir=None):
        """
        Get GeoDataFrame with all `SimBEV <https://github.com/rl-institut/simbev>`_ grid connections.

        Parameters
        ----------
        path : str
            Main path holding SimBEV output data
        dir : str
            Optional sub-directory holding potential grid connection GEOJSONs under path

        Returns
        -------
        :geopandas:`geodataframe`
            GeoDataFrame with AGS, charging use case (home, work, public or hpc),
            user centric weight and geometry.

        """
        if dir is not None:
            path = os.path.join(path, dir)

        files = [f for f in os.listdir(path) if f.endswith(".geojson")]

        epsg = edisgo_obj.topology.grid_district["srid"]

        grid_connections_gdf = gpd.GeoDataFrame(
            pd.DataFrame(columns=COLUMNS["grid_connections_gdf"]), crs={"init": f"epsg:{epsg}"}
        ).astype(DTYPES["grid_connections_gdf"])

        for f in files:
            gdf = gpd.read_file(os.path.join(path, f))

            # drop unnecessary columns
            if len(gdf) > 0:
                if "name" in gdf.columns:
                    gdf = gdf.drop(["name"], axis="columns")

                if "landuse" in gdf.columns:
                    gdf = gdf.drop(["landuse"], axis="columns")

                if "area" in gdf.columns:
                    gdf = gdf.drop(["area"], axis="columns")

                # levelize all GeoDataFrames to the same format
                if len(gdf.columns) == 2:
                    for col in [col for col in gdf.columns if col != "geometry"]:
                        gdf = gdf.rename(columns={col: "user_centric_weight"})

                elif len(gdf.columns) == 1:
                    gdf = gdf.assign(user_centric_weight=0)

                else:
                    raise ValueError(
                        f"GEOJSON {f} contains unknown properties."
                    )

                gdf = gdf.assign(use_case=USECASES[f[:3]], ags=int(f.split("_")[-2]))

                gdf = gdf[COLUMNS["grid_connections_gdf"]].astype(DTYPES["grid_connections_gdf"])

                grid_connections_gdf = grid_connections_gdf.append(
                    gdf, ignore_index=True,
                )

        # sort GeoDataFrame and normalize weights 0 .. 1
        grid_connections_gdf = grid_connections_gdf.sort_values(
            by=["use_case", "ags", "user_centric_weight"], ascending=[True, True, False]).reset_index(drop=True)

        normalized_weight = []

        for use_case in grid_connections_gdf.use_case.unique():
            use_case_weights = grid_connections_gdf.loc[
                grid_connections_gdf.use_case == use_case].user_centric_weight.values.reshape(-1, 1)

            normalized_weight.extend(
                min_max_scaler.fit_transform(use_case_weights).reshape(1, -1).tolist()[0])

        grid_connections_gdf = grid_connections_gdf.assign(
            user_centric_weight=normalized_weight)

        return grid_connections_gdf

    edisgo_obj.electromobility.charging_processes_df = read_csvs_charging_processes(
        path, mode=kwargs.get("mode_standing_times", "frugal"),
        dir=kwargs.get("charging_processes_dir", "simbev_run")
    )

    edisgo_obj.electromobility.simbev_config_df = read_csv_simbev_config(
        path, simbev_config_file=kwargs.get("simbev_config_file", "config_data.csv"))

    edisgo_obj.electromobility.grid_connections_gdf = read_geojsons_grid_connections(
        path, dir=kwargs.get("grid_connections_dir", "grid_connections"))


def distribute_charging_demand(edisgo_obj, **kwargs):
    def get_weights_df(
            grid_connections_indices, **kwargs):
        """
        Get weights per potential charging point for a given set of grid connection indices.

        Parameters
        ----------
        grid_connections_indices : list
            List of grid connection indices
        mode : str
            Only use user friendly weights ('user_friendly') or combine with grid friendly weights ('grid_friendly').
            Default 'user_friendly'
        user_friendly_weight : float
            Weight of user friendly weight if mode 'grid_friendly'. Default 0.5
        distance_weight: float
            Grid friendly weight is a combination of the installed capacity of generators and loads within
            a LV grid and the distance towards the nearest substation. This parameter sets the weight for
            the distance parameter. Default 1/3

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with numeric weights

        """
        mode = kwargs.get("mode", "user_friendly")

        if mode == "user_friendly":
            weights = [
                _.user_centric_weight for _ in edisgo_obj.electromobility.potential_charging_parks
                if _.id in grid_connections_indices
            ]
        elif mode == "grid_friendly":
            potential_charging_parks = list(edisgo_obj.electromobility.potential_charging_parks)

            user_friendly_weights = [
                _.user_centric_weight for _ in potential_charging_parks
                if _.id in grid_connections_indices
            ]

            lv_grids_df = edisgo_obj.topology.lv_grids_df

            generators_weight_factor = kwargs.get("generators_weight_factor", 0.5)
            loads_weight_factor = kwargs.get("loads_weight_factor", 0.5)

            combined_weights = generators_weight_factor*lv_grids_df["generators_weight"] +\
                               loads_weight_factor*lv_grids_df["loads_weight"]

            lv_grid_ids = [_.nearest_substation["lv_grid_id"] for _ in potential_charging_parks]

            load_and_generator_capacity_weights = [
                combined_weights.at[lv_grid_id] for lv_grid_id in lv_grid_ids
            ]

            distance_weights = edisgo_obj.electromobility._potential_charging_parks_df.distance_weight.tolist()

            distance_weight = kwargs.get("distance_weight", 1 / 3)

            grid_friendly_weights = [
                (1 - distance_weight) * load_and_generator_capacity_weights[i] + distance_weight * distance_weights[i]
                for i in range(len(distance_weights))
            ]

            user_friendly_weight = kwargs.get("user_friendly_weight", 0.5)

            weights = [
                (1 - user_friendly_weight) * grid_friendly_weights[i] + user_friendly_weight * user_friendly_weights[i]
                for i in range(len(grid_friendly_weights))
            ]

        return pd.DataFrame(weights)

    def normalize(weights_df):
        """
        Normalize a given DataFrame so that it's sum equals 1 and return a flattened Array.

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
            return np.array(1/len(weights_df) for _ in range(len(weights_df)))
        else:
            return weights_df.divide(weights_df.sum().sum()).T.to_numpy().flatten()

    def combine_weights(
            grid_connections_indices, designated_charging_point_capacity_df, weights_df):
        """
        Add designated charging capacity weights into the initial weights and normalize weights

        Parameters
        ----------
        grid_connections_indices : list
            List of grid connection indices
        designated_charging_point_capacity_df : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with designated charging point capacity per potential charging park
        weights_df : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with initial user or combined weights

        Returns
        -------
        Numpy 1-D array
            Array with normalized weights

        """
        capacity_df = designated_charging_point_capacity_df.loc[grid_connections_indices]

        capacity_weights = (1 - min_max_scaler.fit_transform(
            capacity_df.designated_charging_point_capacity.values.reshape(-1, 1))).flatten()

        user_df = weights_df.loc[grid_connections_indices]

        user_df[0] += capacity_weights

        return normalize(user_df)

    def weighted_random_choice(
            edisgo_obj, grid_connections_indices, car_id, destination, charging_point_id, normalized_weights, rng=None,
    ):
        """
        Weighted random choice of a potential charging park. Setting the chosen values into
        :obj:`~.network.electromobility.charging_processes_df`

        Parameters
        ----------
        edisgo_obj : :class:`~.EDisGo`
        grid_connections_indices : list
            List of grid connection indices
        car_id : int
            Car ID
        destination : str
            Trip destination
        charging_point_id : int
            Charging Point ID
        normalized_weights : Numpy 1-D array
            Array with normalized weights
        rng : Numpy random generator
            If None a random generator with seed=charging_point_id is initialized

        Returns
        -------
        :obj:`int`
            Chosen Charging Park ID

        """
        if rng is None:
            rng = default_rng(seed=charging_point_id)

        grid_connection_point_id = rng.choice(
            a=grid_connections_indices,
            p=normalized_weights,
        )

        edisgo_obj.electromobility.charging_processes_df.loc[
            (edisgo_obj.electromobility.charging_processes_df.car_id == car_id) &
            (edisgo_obj.electromobility.charging_processes_df.destination == destination)
            ] = edisgo_obj.electromobility.charging_processes_df.loc[
            (edisgo_obj.electromobility.charging_processes_df.car_id == car_id) &
            (edisgo_obj.electromobility.charging_processes_df.destination == destination)
            ].assign(
            grid_connection_point_id=grid_connection_point_id,
            charging_point_id=charging_point_id,
        )

        return grid_connection_point_id

    def distribute_private_charging_demand(edisgo_obj):
        """
        Distributes all private charging processes. Each car gets it's own private charging
        point if a charging process takes place.

        Parameters
        ----------
        edisgo_obj : :class:`~.EDisGo`

        """
        try:
            rng = default_rng(seed=edisgo_obj.topology.id)
        except:
            rng = None

        private_charging_df = edisgo_obj.electromobility.charging_processes_df.loc[
            edisgo_obj.electromobility.charging_processes_df.use_case == "private"
        ]

        charging_point_id = 0

        user_centric_weights_df = get_weights_df(edisgo_obj.electromobility.grid_connections_gdf.index)

        designated_charging_point_capacity_df = pd.DataFrame(
            index=user_centric_weights_df.index, columns=["designated_charging_point_capacity"], data=0)

        for destination in private_charging_df.destination.sort_values().unique():
            private_charging_destination_df = private_charging_df.loc[
                private_charging_df.destination == destination
            ]

            use_case = PRIVATE_DESTINATIONS[destination]

            if use_case == "work":
                grid_connections_indices = edisgo_obj.electromobility.grid_connections_gdf.loc[
                    edisgo_obj.electromobility.grid_connections_gdf.use_case == use_case
                ].index

                for car_id in private_charging_destination_df.car_id.sort_values().unique():
                    weights = combine_weights(
                        grid_connections_indices, designated_charging_point_capacity_df, user_centric_weights_df)

                    grid_connection_point_id = weighted_random_choice(
                        edisgo_obj, grid_connections_indices, car_id, destination, charging_point_id, weights, rng=rng)

                    charging_capacity = private_charging_destination_df.loc[
                        (private_charging_destination_df.car_id == car_id) &
                        (private_charging_destination_df.destination == "0_work")
                    ].netto_charging_capacity.iat[0] / edisgo_obj.electromobility.eta_charging_points

                    designated_charging_point_capacity_df.at[
                        grid_connection_point_id, "designated_charging_point_capacity"
                    ] += charging_capacity

                    charging_point_id += 1

            elif use_case == "home":
                for ags in private_charging_destination_df.ags.sort_values().unique():
                    private_charging_ags_df = private_charging_destination_df.loc[
                        private_charging_destination_df.ags == ags
                    ]

                    grid_connections_indices = edisgo_obj.electromobility.grid_connections_gdf.loc[
                        (edisgo_obj.electromobility.grid_connections_gdf.ags == ags) &
                        (edisgo_obj.electromobility.grid_connections_gdf.use_case == use_case)
                    ].index

                    for car_id in private_charging_ags_df.car_id.sort_values().unique():
                        weights = combine_weights(
                            grid_connections_indices, designated_charging_point_capacity_df, user_centric_weights_df)

                        weighted_random_choice(
                            edisgo_obj, grid_connections_indices, car_id, destination,
                            charging_point_id, weights, rng=rng)

                        charging_capacity = private_charging_destination_df.loc[
                            (private_charging_destination_df.car_id == car_id) &
                            (private_charging_destination_df.destination == "6_home")
                        ].netto_charging_capacity.iat[0]

                        designated_charging_point_capacity_df.at[
                            grid_connection_point_id, "designated_charging_point_capacity"
                        ] += charging_capacity

                        charging_point_id += 1

            else:
                raise ValueError(
                    "Destination {} is unknown.".format(destination)
                )

    def distribute_public_charging_demand(edisgo_obj, **kwargs):
        """
        Distributes all public charging processes. For each process it is checked if a matching
        charging point is existing to minimize the number of charging points.

        Parameters
        ----------
        edisgo_obj : :class:`~.EDisGo`

        """
        public_charging_df = edisgo_obj.electromobility.charging_processes_df.loc[
            edisgo_obj.electromobility.charging_processes_df.use_case == "public"
        ].sort_values(by=["park_start", "park_end"], ascending=[True, True])

        try:
            rng = default_rng(seed=edisgo_obj.topology.id)
        except:
            rng = default_rng(seed=1)

        available_charging_points_df = pd.DataFrame(columns=COLUMNS["available_charging_points_df"])

        grid_and_user_centric_weights_df = get_weights_df(
            edisgo_obj.electromobility.grid_connections_gdf.index, **kwargs)

        designated_charging_point_capacity_df = pd.DataFrame(
            index=grid_and_user_centric_weights_df.index, columns=["designated_charging_point_capacity"], data=0)

        for idx, row in public_charging_df.iterrows():
            use_case = PUBLIC_DESTINATIONS[row["destination"]]

            matching_charging_points_df = available_charging_points_df.loc[
                (available_charging_points_df.park_end < row["park_end"]) &
                (available_charging_points_df.netto_charging_capacity.round(1) ==
                 round(row["netto_charging_capacity"], 1))
            ]

            if len(matching_charging_points_df) > 0:
                grid_connections_indices = matching_charging_points_df.index

                weights = normalize(grid_and_user_centric_weights_df.loc[grid_connections_indices])

                charging_point_s = matching_charging_points_df.loc[rng.choice(
                    a=grid_connections_indices, p=weights)]

                edisgo_obj.electromobility.charging_processes_df.at[idx, "grid_connection_point_id"] = \
                    charging_point_s["grid_connection_point_id"]

                edisgo_obj.electromobility.charging_processes_df.at[idx, "charging_point_id"] = charging_point_s.name

                available_charging_points_df.at[idx, "park_end"] = row["park_end"]

            else:
                grid_connections_indices = edisgo_obj.electromobility.grid_connections_gdf.loc[
                    edisgo_obj.electromobility.grid_connections_gdf.use_case == use_case
                    ].index

                weights = combine_weights(
                    grid_connections_indices, designated_charging_point_capacity_df, grid_and_user_centric_weights_df)

                grid_connection_point_id = rng.choice(
                    a=grid_connections_indices,
                    p=weights,
                )

                charging_point_id = edisgo_obj.electromobility.charging_processes_df.charging_point_id.max() + 1

                if charging_point_id != charging_point_id:
                    charging_point_id = 0

                edisgo_obj.electromobility.charging_processes_df.at[idx, "grid_connection_point_id"] = \
                    grid_connection_point_id

                edisgo_obj.electromobility.charging_processes_df.at[idx, "charging_point_id"] = charging_point_id

                available_charging_points_df.loc[charging_point_id] = edisgo_obj.electromobility.charging_processes_df[
                    available_charging_points_df.columns].loc[idx].tolist()

                designated_charging_point_capacity_df.at[
                    grid_connection_point_id, "designated_charging_point_capacity"] += row["netto_charging_capacity"]


    distribute_private_charging_demand(edisgo_obj)

    distribute_public_charging_demand(edisgo_obj, **kwargs)


def determine_grid_connection_capacity(
        total_charging_point_capacity, lower_limit=0.3, upper_limit=1.0, minimum_factor=0.45):
    if total_charging_point_capacity <= lower_limit:
        return total_charging_point_capacity
    elif total_charging_point_capacity >= upper_limit:
        return minimum_factor * total_charging_point_capacity
    else:
        return (((minimum_factor-1) / (upper_limit-lower_limit))
                * (total_charging_point_capacity-lower_limit) + 1) * total_charging_point_capacity


