import logging
import pandas as pd

from edisgo.network.components import ChargingParkGridConnection

logger = logging.getLogger("edisgo")

COLUMNS = {
    "charging_processes_df": [
        "location", "use_case", "netto_charging_capacity", "chargingdemand", "charge_start",
        "charge_end", "grid_connection_point_id", "charging_point_id"
    ],
    "grid_connections_gdf": ["id", "use_case", "user_centric_weight", "geometry"],
    "simbev_config_df": ["value"]
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
    def charging_park_grid_connections(self):
        return {use_case: {cp_id: ChargingParkGridConnection(id=cp_id, edisgo_obj=self._edisgo_obj)
                for cp_id in self.grid_connections_gdf.loc[
                    self.grid_connections_gdf.use_case == use_case
                    ].index} for use_case in self.grid_connections_gdf.use_case.sort_values().unique()}

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


