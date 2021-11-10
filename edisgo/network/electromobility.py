import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn import preprocessing

from edisgo.network.components import PotentialChargingParks

logger = logging.getLogger("edisgo")

COLUMNS = {
    "charging_processes_df": [
        "ags", "car_id", "destination", "use_case", "netto_charging_capacity",
        "chargingdemand", "park_start", "park_end"
    ],
    "grid_connections_gdf": ["id", "use_case", "user_centric_weight", "geometry"],
    "simbev_config_df": ["value"],
    "potential_charging_parks_df": ["lv_grid_id", "distance_to_nearest_substation", "distance_weight",
                                    "charging_point_capacity", "charging_point_weight"],
    "designated_charging_points_df": ["park_end", "netto_charging_capacity", "charging_park_id", "use_case"],
    "integrated_charging_parks_df": ["edisgo_id"],
}

USECASES = ["hpc", "public", "home", "work"]


class Electromobility:
    """
    Electromobility base class

    """

    def __init__(self, **kwargs):
        self._edisgo_obj = kwargs.get("edisgo_obj", None)

    @property
    def charging_processes_df(self):
        """
        DataFrame with all `SimBEV <https://github.com/rl-institut/simbev>`_ charging processes.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with AGS, car ID, trip destination, charging use case (private or public),
            netto charging capacity, charging demand, charge start, charge end, grid connection point and
            charging point ID.

        """
        try:
            return self._charging_processes_df
        except:
            return pd.DataFrame(columns=COLUMNS["charging_processes_df"])

    @charging_processes_df.setter
    def charging_processes_df(self, df):
        self._charging_processes_df = df

    @property
    def grid_connections_gdf(self):
        """
        GeoDataFrame with all `SimBEV <https://github.com/rl-institut/simbev>`_ grid connections.

        Returns
        -------
        :geopandas:`geodataframe`
            GeoDataFrame with AGS, charging use case (home, work, public or hpc),
            user centric weight and geometry.

        """
        try:
            return self._grid_connections_gdf
        except:
            return gpd.GeoDataFrame(columns=COLUMNS["grid_connections_gdf"])

    @grid_connections_gdf.setter
    def grid_connections_gdf(self, gdf):
        self._grid_connections_gdf = gdf

    @property
    def potential_charging_parks(self):
        """
        Potential Charging Parks within the AGS.

        Returns
        -------
        list(:class:`~.network.components.PotentialChargingParks`)
            List of Potential Charging Parks within the AGS.

        """
        for cp_id in self.grid_connections_gdf.index:
            yield PotentialChargingParks(id=cp_id, edisgo_obj=self._edisgo_obj)

    @property
    def simbev_config_df(self):
        """
        DataFrame with all `SimBEV <https://github.com/rl-institut/simbev>`_ config data.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with used random seed, used threads, stepsize in minutes, year, scenarette,
            simulated days, maximum number of cars per AGS, completed standing times and timeseries per AGS
            and used ramp up data CSV.

        """
        try:
            return self._simbev_config_df
        except:
            return pd.DataFrame(columns=COLUMNS["simbev_config_df"])

    @simbev_config_df.setter
    def simbev_config_df(self, df):
        self._simbev_config_df = df

    @property
    def integrated_charging_parks_df(self):
        try:
            return self._integrated_charging_parks_df
        except:
            return pd.DataFrame(columns=COLUMNS["integrated_charging_parks_df"])

    @integrated_charging_parks_df.setter
    def integrated_charging_parks_df(self, df):
        self._integrated_charging_parks_df = df

    @property
    def stepsize(self):
        """
        Stepsize in minutes used in `SimBEV <https://github.com/rl-institut/simbev>`_.

        Returns
        -------
        :obj:`int`
            Stepsize in minutes

        """
        try:
            return int(self.simbev_config_df.at["stepsize", "value"])
        except:
            return None

    @property
    def simulated_days(self):
        """
        Number of simulated days in `SimBEV <https://github.com/rl-institut/simbev>`_.

        Returns
        -------
        :obj:`int`
            Number of simulated days

        """
        try:
            return int(self.simbev_config_df.at["days", "value"])
        except:
            return None

    @property
    def eta_charging_points(self):
        """
        `SimBEV <https://github.com/rl-institut/simbev>`_ charging point efficiency.

        Returns
        -------
        :obj:`float`
            Charging point efficiency

        """
        try:
            return float(self.simbev_config_df.at["eta_CP", "value"])
        except:
            return None

    def to_csv(self, directory):
        """
        Exports electromobility to csv files.

        The following attributes are exported:

        * 'charging_processes_df' : Attribute :py:attr:`~charging_processes_df` is saved to
          `charging_processes.csv`.
        * 'grid_connections_gdf' : Attribute :py:attr:`~grid_connections_gdf` is saved to
          `grid_connections.csv`.
        * 'integrated_charging_parks_df' : Attribute :py:attr:`~integrated_charging_parks_df` is
          saved to `integrated_charging_parks.csv`.
        * 'simbev_config_df' : Attribute :py:attr:`~simbev_config_df` is
          saved to `simbev_config.csv`.

        Parameters
        ----------
        directory : str
            Path to save electromobility to.

        """
        os.makedirs(directory, exist_ok=True)

        if not self.charging_processes_df.empty:
            self.charging_processes_df.to_csv(
                os.path.join(directory, "charging_processes.csv"))

        if not self.grid_connections_gdf.empty:
            self.grid_connections_gdf.to_csv(
                os.path.join(directory, "grid_connections.csv"))

        if not self.integrated_charging_parks_df.empty:
            self.integrated_charging_parks_df.to_csv(
                os.path.join(directory, "integrated_charging_parks.csv"))

        if not self.simbev_config_df.empty:
            self.simbev_config_df.to_csv(
                os.path.join(directory, "simbev_config.csv"))

    def from_csv(self, directory, edisgo_obj):
        """
        Restores electromobility from csv files.

        Parameters
        ----------
        edisgo_obj : :class:`~.EDisGo`
        directory : str
            Path to electromobility csv files.

        """
        if os.path.exists(os.path.join(directory, "charging_processes.csv")):
            self.charging_processes_df = pd.read_csv(
                os.path.join(directory, "charging_processes.csv"), index_col=0)

        if os.path.exists(os.path.join(directory, "grid_connections.csv")):
            epsg = edisgo_obj.topology.grid_district["srid"]

            grid_connections_df = pd.read_csv(
                    os.path.join(directory, "grid_connections.csv"), index_col=0)

            grid_connections_df = grid_connections_df.assign(
                geometry=gpd.GeoSeries.from_wkt(grid_connections_df["geometry"]))

            try:
                self.grid_connections_gdf = gpd.GeoDataFrame(
                    grid_connections_df, geometry="geometry", crs={"init": f"epsg:{epsg}"})
            except:
                logging.warning(
                    f"""Grid connections could not be loaded with EPSG {epsg}.
                    Trying with EPSG 4326 as fallback.""")

                self.grid_connections_gdf = gpd.GeoDataFrame(
                    grid_connections_df, geometry="geometry", crs={"init": "epsg:4326"})

        if os.path.exists(os.path.join(directory, "integrated_charging_parks.csv")):
            self.integrated_charging_parks_df = pd.read_csv(
                os.path.join(directory, "integrated_charging_parks.csv"), index_col=0)

        if os.path.exists(os.path.join(directory, "simbev_config.csv")):
            self.simbev_config_df = pd.read_csv(
                os.path.join(directory, "simbev_config.csv"), index_col=0)


    @property
    def _potential_charging_parks_df(self):
        """
        Overview over `SimBEVs <https://github.com/rl-institut/simbev>`_ potential charging parks from
        :class:`~.network.components.PotentialChargingParks`.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with LV Grid ID, distance to nearest substation, distance weight, charging point
            capacity and charging point weight.

        """
        try:
            potential_charging_parks_df = pd.DataFrame(columns=COLUMNS["potential_charging_parks_df"])

            potential_charging_parks = list(self.potential_charging_parks)

            potential_charging_parks_df.lv_grid_id = [
                _.nearest_substation["lv_grid_id"] for _ in potential_charging_parks
            ]

            potential_charging_parks_df.distance_to_nearest_substation = [
                _.nearest_substation["distance"] for _ in potential_charging_parks
            ]

            min_max_scaler = preprocessing.MinMaxScaler()

            potential_charging_parks_df.distance_weight = 1 - min_max_scaler.fit_transform(
                potential_charging_parks_df.distance_to_nearest_substation.values.reshape(-1, 1))

            potential_charging_parks_df.charging_point_capacity = [
                _.designated_charging_point_capacity for _ in potential_charging_parks
            ]

            potential_charging_parks_df.charging_point_weight = 1 - min_max_scaler.fit_transform(
                potential_charging_parks_df.charging_point_capacity.values.reshape(-1, 1))

            return potential_charging_parks_df
        except:
            return pd.DataFrame(columns=COLUMNS["potential_charging_parks_df"])
