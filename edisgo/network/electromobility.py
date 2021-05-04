import logging
import pandas as pd

from sklearn import preprocessing

from edisgo.network.components import PotentialChargingParkGridConnection

logger = logging.getLogger("edisgo")

COLUMNS = {
    "charging_processes_df": [
        "location", "use_case", "netto_charging_capacity", "chargingdemand", "charge_start",
        "charge_end", "grid_connection_point_id", "charging_point_id"
    ],
    "grid_connections_gdf": ["id", "use_case", "user_centric_weight", "geometry"],
    "simbev_config_df": ["value"],
    "potential_charging_points_df": ["lv_grid_id", "distance_to_nearest_substation", "distance_weight",
                                     "charging_point_capacity", "charging_point_weight"]
}


class Electromobility:

    def __init__(self, **kwargs):
        self._edisgo_obj = kwargs.get("edisgo_obj", None)

    @property
    def charging_processes_df(self):
        try:
            return self._charging_processes_df
        except:
            return pd.DataFrame(columns=COLUMNS["charging_processes_df"])

    @charging_processes_df.setter
    def charging_processes_df(self, df):
        self._charging_processes_df = df

    @property
    def grid_connections_gdf(self):
        try:
            return self._grid_connections_gdf
        except:
            return pd.DataFrame(columns=COLUMNS["grid_connections_gdf"])

    @grid_connections_gdf.setter
    def grid_connections_gdf(self, df):
        self._grid_connections_gdf = df

    @property
    def potential_charging_points_df(self):
        try:
            potential_charging_points_df = pd.DataFrame(columns=COLUMNS["potential_charging_points_df"])

            potential_charging_points = list(self.potential_charging_points)

            potential_charging_points_df.lv_grid_id = [
                _.nearest_substation["lv_grid_id"]
                for _ in potential_charging_points
            ]

            potential_charging_points_df.distance_to_nearest_substation = [
                _.nearest_substation["distance"]
                for _ in potential_charging_points
            ]

            min_max_scaler = preprocessing.MinMaxScaler()

            potential_charging_points_df.distance_weight = 1 - min_max_scaler.fit_transform(
                potential_charging_points_df.distance_to_nearest_substation.values.reshape(-1, 1))

            potential_charging_points_df.charging_point_capacity = [
                _.designated_charging_point_capacity
                for _ in potential_charging_points
            ]

            potential_charging_points_df.charging_point_weight = 1 - min_max_scaler.fit_transform(
                potential_charging_points_df.charging_point_capacity.values.reshape(-1, 1))

            return potential_charging_points_df
        except:
            return pd.DataFrame(columns=COLUMNS["potential_charging_points_df"])

    @property
    def potential_charging_points(self):
        for cp_id in self.grid_connections_gdf.index:
            yield PotentialChargingParkGridConnection(id=cp_id, edisgo_obj=self._edisgo_obj)

    @property
    def simbev_config_df(self):
        try:
            return self._simbev_config_df
        except:
            return pd.DataFrame(columns=COLUMNS["simbev_config_df"])

    @simbev_config_df.setter
    def simbev_config_df(self, df):
        self._simbev_config_df = df

    @property
    def stepsize(self):
        try:
            return int(self.simbev_config_df.at["stepsize", "value"])
        except:
            return None

    @property
    def simulated_days(self):
        try:
            return int(self.simbev_config_df.at["days", "value"])
        except:
            return None

    @property
    def eta_charging_points(self):
        try:
            return float(self.simbev_config_df.at["eta_CP", "value"])
        except:
            return None


