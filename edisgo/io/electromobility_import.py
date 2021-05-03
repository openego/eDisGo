import logging
import os
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn import preprocessing
from pathlib import Path


logger = logging.getLogger("edisgo")

COLUMNS = {
    "charging_processes_df": [
        "ags", "car_id", "destination", "use_case", "netto_charging_capacity",
        "chargingdemand", "charge_start", "charge_end"
    ],
    "matching_demand_and_location": ["grid_connection_point_id", "charging_point_id"],
    "grid_connections_gdf": ["ags", "use_case", "user_centric_weight", "geometry"],
    "simbev_config_df": ["value"],
}

DTYPES = {
    "charging_processes_df": {
        "ags": np.uint32,
        "car_id": np.uint32,
        "destination": str,
        "use_case": str,
        "netto_charging_capacity": np.float64,
        "chargingdemand": np.float64,
        "charge_start": np.uint16,
        "charge_end": np.uint16,
    },
    "grid_connections_gdf": {
        "ags": np.uint32,
        "use_case": str,
        "user_centric_weight": np.float64,
    },
}

USECASES = {
    "uc1": "fast",
    "uc2": "public",
    "uc3": "home",
    "uc4": "work",
}


def import_simbev_electromobility(path, edisgo_obj, **kwargs):
    # TODO: SimBEV is in development and this import will need constant updating for now
    def read_csvs_charging_processes(path, mode=None, dir=None):
        if dir is not None:
            path = os.path.join(path, dir)

        files = []

        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(".csv")]:
                files.append(Path(os.path.join(dirpath, filename)))

        if len(files) == 0:
            raise ValueError(
                "Couldn't find any CSVs in path {}.".format(path)
            )

        charging_processes_df = pd.DataFrame(columns=COLUMNS["charging_processes_df"])

        charging_processes_df = charging_processes_df.astype(DTYPES["charging_processes_df"])

        for car_id, f in enumerate(files):
            df = pd.read_csv(f, index_col=[0])

            df = df.rename(columns={"location": "destination"})

            df = df.assign(ags=int(f.parts[-2]), car_id=car_id)

            df = df[COLUMNS["charging_processes_df"]].astype(DTYPES["charging_processes_df"])

            if mode == "frugal":
                df = df.loc[df.chargingdemand > 0]
            else:
                pass

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
        try:
            if simbev_config_file is not None:
                return pd.read_csv(
                    os.path.join(path, simbev_config_file), index_col=[0], header=0, names=COLUMNS["simbev_config_df"])
        except:
            return pd.DataFrame(columns=COLUMNS["simbev_config_df"])

    def read_geojsons_grid_connections(path, dir=None):

        if dir is not None:
            path = os.path.join(path, dir)

        files = [f for f in os.listdir(path) if f.endswith(".geojson")]

        epsg = edisgo_obj.topology.grid_district["srid"]

        grid_connections_gdf = gpd.GeoDataFrame(
            pd.DataFrame(columns=COLUMNS["grid_connections_gdf"]), crs={"init": f"epsg:{epsg}"}
        ).astype(DTYPES["grid_connections_gdf"])

        for f in files:
            gdf = gpd.read_file(os.path.join(path, f))

            if len(gdf) > 0:
                if "name" in gdf.columns:
                    gdf = gdf.drop(["name"], axis="columns")

                if "landuse" in gdf.columns:
                    gdf = gdf.drop(["landuse"], axis="columns")

                if "area" in gdf.columns:
                    gdf = gdf.drop(["area"], axis="columns")

                if len(gdf.columns) == 2:
                    for col in [col for col in gdf.columns if col is not "geometry"]:
                        gdf = gdf.rename(columns={col: "user_centric_weight"})

                elif len(gdf.columns) == 1:
                    gdf = gdf.assign(user_centric_weight=0)

                else:
                    raise ValueError(
                        "GEOJSON {} contains unknown properties.".format(f)
                    )

                gdf = gdf.assign(use_case=USECASES[f[:3]], ags=int(f.split("_")[-2]))

                gdf = gdf[COLUMNS["grid_connections_gdf"]].astype(DTYPES["grid_connections_gdf"])

                grid_connections_gdf = grid_connections_gdf.append(
                    gdf, ignore_index=True,
                )

        grid_connections_gdf = grid_connections_gdf.sort_values(
            by=["use_case", "ags", "user_centric_weight"], ascending=[True, True, False]).reset_index(drop=True)

        min_max_scaler = preprocessing.MinMaxScaler()

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
        dir=kwargs.get("charging_processes_dir", "standing_times_looped")
    )

    edisgo_obj.electromobility.simbev_config_df = read_csv_simbev_config(
        path, simbev_config_file=kwargs.get("simbev_config_file", "config_data.csv"))

    edisgo_obj.electromobility.grid_connections_gdf = read_geojsons_grid_connections(
        path, dir=kwargs.get("grid_connections_dir", "grid_connections"))
