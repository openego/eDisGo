import logging
import os
import numpy as np
import pandas as pd

import edisgo


logger = logging.getLogger("edisgo")

COLUMNS = {
    "charging_processes_df": [
        "location", "use_case", "netto_charging_capacity", "chargingdemand", "charge_start",
        "charge_end", "grid_connection_point_id", "charging_point_id"
    ],
    "simbev_config": ["value"]
}


class Electromobility:

    def __init__(self, **kwargs):
        pass

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
    def simbev_config(self):
        try:
            return self._simbev_config
        except:
            return pd.DataFrame(columns=COLUMNS["simbev_config"])

    @simbev_config.setter
    def simbev_config(self, df):
        self._simbev_config = df

    @property
    def stepsize(self):
        try:
            return int(self.simbev_config.at["stepsize", "value"])
        except:
            return np.nan

    @property
    def simulated_days(self):
        try:
            return int(self.simbev_config.at["days", "value"])
        except:
            return np.nan

    @property
    def eta_charging_points(self):
        try:
            return float(self.simbev_config.at["eta_CP", "value"])
        except:
            return np.nan


