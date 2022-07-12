import logging
import os

from zipfile import ZipFile

import geopandas as gpd
import pandas as pd

from sklearn import preprocessing

from edisgo.network.components import PotentialChargingParks

logger = logging.getLogger("edisgo")

COLUMNS = {
    "charging_processes_df": [
        "ags",
        "car_id",
        "destination",
        "use_case",
        "nominal_charging_capacity_kW",
        "chargingdemand_kWh",
        "park_start_timesteps",
        "park_end_timesteps",
    ],
    "grid_connections_gdf": ["id", "use_case", "user_centric_weight", "geometry"],
    "potential_charging_parks_df": [
        "lv_grid_id",
        "distance_to_nearest_substation",
        "distance_weight",
        "charging_point_capacity",
        "charging_point_weight",
    ],
    "designated_charging_points_df": [
        "park_end_timesteps",
        "nominal_charging_capacity_kW",
        "charging_park_id",
        "use_case",
    ],
    "integrated_charging_parks_df": ["edisgo_id"],
}


class Electromobility:
    """
    Electromobility base class

    """

    def __init__(self, **kwargs):
        self._edisgo_obj = kwargs.get("edisgo_obj", None)

    @property
    def charging_processes_df(self):
        """
        DataFrame with all `SimBEV <https://github.com/rl-institut/simbev>`_
        charging processes.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with AGS, car ID, trip destination, charging use case
            (private or public), netto charging capacity, charging demand,
            charge start, charge end, grid connection point and charging point
            ID.

        """
        try:
            return self._charging_processes_df
        except Exception:
            return pd.DataFrame(columns=COLUMNS["charging_processes_df"])

    @charging_processes_df.setter
    def charging_processes_df(self, df):
        self._charging_processes_df = df

    @property
    def grid_connections_gdf(self):
        """
        GeoDataFrame with all `SimBEV <https://github.com/rl-institut/simbev>`_
        grid connections.

        Returns
        -------
        :geopandas:`GeoDataFrame`
            GeoDataFrame with AGS, charging use case (home, work, public or
            hpc), user centric weight and geometry.

        """
        try:
            return self._grid_connections_gdf
        except Exception:
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
        Dict with all `SimBEV <https://github.com/rl-institut/simbev>`_. config data.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with used random seed, used threads, stepsize in minutes,
            year, scenarette, simulated days, maximum number of cars per AGS,
            completed standing times and timeseries per AGS and used ramp up
            data CSV.

        """
        try:
            return self._simbev_config_df
        except Exception:
            return pd.DataFrame()

    @simbev_config_df.setter
    def simbev_config_df(self, df):
        self._simbev_config_df = df

    @property
    def integrated_charging_parks_df(self):
        try:
            return self._integrated_charging_parks_df
        except Exception:
            return pd.DataFrame(columns=COLUMNS["integrated_charging_parks_df"])

    @integrated_charging_parks_df.setter
    def integrated_charging_parks_df(self, df):
        self._integrated_charging_parks_df = df

    @property
    def stepsize(self):
        """
        Stepsize in minutes used in
        `SimBEV <https://github.com/rl-institut/simbev>`_.

        Returns
        -------
        :obj:`int`
            Stepsize in minutes

        """
        try:
            return int(self.simbev_config_df.at[0, "stepsize"])
        except Exception:
            return None

    @property
    def simulated_days(self):
        """
        Number of simulated days in
        `SimBEV <https://github.com/rl-institut/simbev>`_.

        Returns
        -------
        :obj:`int`
            Number of simulated days

        """
        try:
            return int(self.simbev_config_df.at[0, "days"])
        except Exception:
            return None

    @property
    def eta_charging_points(self):
        """
        `SimBEV <https://github.com/rl-institut/simbev>`_ charging point
        efficiency.

        Returns
        -------
        :obj:`float`
            Charging point efficiency

        """
        try:
            return float(self.simbev_config_df.at[0, "eta_cp"])
        except Exception:
            return None

    def to_csv(self, directory):
        """
        Exports electromobility to csv files.

        The following attributes are exported:

        * 'charging_processes_df' : Attribute :py:attr:`~charging_processes_df`
          is saved to `charging_processes.csv`.
        * 'grid_connections_gdf' : Attribute :py:attr:`~grid_connections_gdf`
          is saved to `grid_connections.csv`.
        * 'integrated_charging_parks_df' : Attribute
          :py:attr:`~integrated_charging_parks_df` is saved to
          `integrated_charging_parks.csv`.
        * 'simbev_config_df' : Attribute :py:attr:`~simbev_config_df` is
          saved to `simbev_config.csv`.

        Parameters
        ----------
        directory : str
            Path to save electromobility to.

        """
        os.makedirs(directory, exist_ok=True)

        attrs = _get_matching_dict_of_attributes_and_file_names()

        for attr, file in attrs.items():
            df = getattr(self, attr)

            if not df.empty:
                path = os.path.join(directory, file)
                df.to_csv(path)

    def from_csv(self, data_path, edisgo_obj, from_zip_archive=False):
        """
        Restores electromobility from csv files.

        Parameters
        ----------
        data_path : str
            Path to electromobility csv files.
        edisgo_obj : :class:`~.EDisGo`
        from_zip_archive : bool, optional
            Set True if data is archived in a zip archive. Default: False

        """
        attrs = _get_matching_dict_of_attributes_and_file_names()

        if from_zip_archive:
            # read from zip archive
            # setup ZipFile Class
            zip = ZipFile(data_path)

            # get all directories and files within zip archive
            files = zip.namelist()

            # add directory and .csv to files to match zip archive
            attrs = {k: f"electromobility/{v}" for k, v in attrs.items()}

        else:
            # read from directory
            # check files within the directory
            files = os.listdir(data_path)

        attrs_to_read = {k: v for k, v in attrs.items() if v in files}

        for attr, file in attrs_to_read.items():
            if from_zip_archive:
                # open zip file to make it readable for pandas
                with zip.open(file) as f:
                    df = pd.read_csv(f, index_col=0)
            else:
                path = os.path.join(data_path, file)
                df = pd.read_csv(path, index_col=0)

            if attr == "grid_connections_gdf":
                epsg = edisgo_obj.topology.grid_district["srid"]

                df = df.assign(geometry=gpd.GeoSeries.from_wkt(df["geometry"]))

                try:
                    df = gpd.GeoDataFrame(
                        df, geometry="geometry", crs={"init": f"epsg:{epsg}"}
                    )

                except Exception:
                    logging.warning(
                        f"Grid connections could not be loaded with "
                        f"EPSG {epsg}. Trying with EPSG 4326 as fallback."
                    )

                    df = gpd.GeoDataFrame(
                        df, geometry="geometry", crs={"init": "epsg:4326"}
                    )

            setattr(self, attr, df)

        if from_zip_archive:
            # make sure to destroy ZipFile Class to close any open connections
            zip.close()

    @property
    def _potential_charging_parks_df(self):
        """
        Overview over `SimBEVs <https://github.com/rl-institut/simbev>`_
        potential charging parks from
        :class:`~.network.components.PotentialChargingParks`.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with LV Grid ID, distance to nearest substation, distance
            weight, charging point capacity and charging point weight.

        """
        try:
            potential_charging_parks_df = pd.DataFrame(
                columns=COLUMNS["potential_charging_parks_df"]
            )

            potential_charging_parks = list(self.potential_charging_parks)

            potential_charging_parks_df.lv_grid_id = [
                _.nearest_substation["lv_grid_id"] for _ in potential_charging_parks
            ]

            potential_charging_parks_df.distance_to_nearest_substation = [
                _.nearest_substation["distance"] for _ in potential_charging_parks
            ]

            min_max_scaler = preprocessing.MinMaxScaler()

            # fmt: off
            potential_charging_parks_df.distance_weight = (
                1 - min_max_scaler.fit_transform(
                    potential_charging_parks_df.distance_to_nearest_substation.values
                        .reshape(-1, 1)  # noqa: E131
                )
            )
            # fmt: on

            potential_charging_parks_df.charging_point_capacity = [
                _.designated_charging_point_capacity for _ in potential_charging_parks
            ]

            potential_charging_parks_df.charging_point_weight = (
                1
                - min_max_scaler.fit_transform(
                    potential_charging_parks_df.charging_point_capacity.values.reshape(
                        -1, 1
                    )
                )
            )

            return potential_charging_parks_df
        except Exception:
            return pd.DataFrame(columns=COLUMNS["potential_charging_parks_df"])


def _get_matching_dict_of_attributes_and_file_names():
    """
    Helper function to specify which Electromobility attributes to save and
    restore and maps them to the file name.

    Is used in functions
    :attr:`~.network.electromobility.Electromobility.from_csv`.

    Returns
    -------
    dict
        Dict of Electromobility attributes to save and restore as keys and
        and matching files as values.

    """
    emob_dict = {
        "charging_processes_df": "charging_processes.csv",
        "grid_connections_gdf": "grid_connections.csv",
        "integrated_charging_parks_df": "integrated_charging_parks.csv",
        "simbev_config_df": "metadata_simbev_run.csv",
    }

    return emob_dict
