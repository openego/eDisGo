import logging
import os
import numpy as np
import pandas as pd

from pathlib import Path

logger = logging.getLogger("edisgo")

COLUMNS = {
    "charging_processes_df": [
        "location", "use_case", "netto_charging_capacity", "chargingdemand", "charge_start", "charge_end"
    ],
    "matching_demand_and_location": ["grid_connection_point_id", "charging_point_id"],
    "simbev_config": ["value"],
}

DTYPES = {
    "charging_processes_df": {
        "location": str,
        "use_case": str,
        "netto_charging_capacity": np.float64,
        "chargingdemand": np.float64,
        "charge_start": np.int32,
        "charge_end": np.int32,
    },
    "matching_demand_and_location": {
        "grid_connection_point_id": np.int32,
        "charging_point_id": np.int32,
    }
}


def import_simbev_electromobility(path, edisgo_obj, **kwargs):
    # TODO: SimBEV is in development and this import will need constant updating for now
    def read_csvs_charging_processes(path, mode, dir="standing_times_looped"):

        path = os.path.join(path, dir)

        files = []

        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(".csv")]:
                files.append(os.path.join(dirpath, filename))

        if len(files) == 0:
            raise ValueError(
                "Couldn't find any CSVs in path {}.".format(path)
            )

        charging_processes_df = pd.DataFrame(columns=COLUMNS["charging_processes_df"])

        charging_processes_df = charging_processes_df.astype(DTYPES["charging_processes_df"])

        for f in files:
            df = pd.read_csv(f, index_col=[0])

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
            pd.DataFrame(columns=COLUMNS["matching_demand_and_location"]).astype(
                DTYPES["matching_demand_and_location"]),
            how="outer",
            left_index=True,
            right_index=True
            )

        return charging_processes_df

    def read_csv_simbev_config(path, config_file):
        try:
            return pd.read_csv(
                os.path.join(path, config_file), index_col=[0], header=0, names=COLUMNS["simbev_config"])
        except:
            return pd.DataFrame(columns=COLUMNS["simbev_config"])


    edisgo_obj.electromobility.charging_processes_df = read_csvs_charging_processes(
        path, mode=kwargs.get("mode_standing_times", "frugal"))

    edisgo_obj.electromobility.simbev_config = read_csv_simbev_config(
        path, config_file=kwargs.get("config_file", "config_data.csv"))
