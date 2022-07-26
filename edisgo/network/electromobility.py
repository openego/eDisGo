import logging
import os

from zipfile import ZipFile

import pandas as pd

from sklearn import preprocessing

from edisgo.network.components import PotentialChargingParks

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

logger = logging.getLogger("edisgo")

COLUMNS = {
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
        "charging_park_id",
        "charging_point_id",
    ],
    "potential_charging_parks_gdf": [
        "id",
        "use_case",
        "user_centric_weight",
        "geometry",
    ],
    "simbev_config_df": [
        "regio_type",
        "eta_cp",
        "stepsize",
        "start_date",
        "end_date",
        "soc_min",
        "grid_timeseries",
        "grid_timeseries_by_usecase",
        "days",
    ],
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
    Data container for all electromobility data.

    This class holds data on charging processes (how long cars are parking at a
    charging station, how much they need to charge, etc.) necessary to apply different
    charging strategies, as well as information on potential charging sites and
    integrated charging parks.

    """

    def __init__(self, **kwargs):
        self._edisgo_obj = kwargs.get("edisgo_obj", None)

    @property
    def charging_processes_df(self):
        """
        DataFrame with all
        `SimBEV <https://github.com/rl-institut/simbev>`_
        charging processes.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with AGS, car ID, trip destination, charging use case,
            netto charging capacity, charging demand, charge start, charge end, grid
            connection point and charging point ID. The columns are:

                ags : int
                    8-digit AGS (Amtlicher Gemeindeschlüssel, eng. Community
                    Identification Number). Leading zeros are missing.

                car_id : int
                    Car ID to differntiate charging processes from different cars.

                destination : str
                    SimBEV driving destination.

                use_case : str
                    SimBEV use case. Can be "hpc", "home", "public" or "work".

                nominal_charging_capacity_kW : float
                    Vehicle charging capacity in kW.

                grid_charging_capacity_kW : float
                    Grid-sided charging capacity including charging infrastructure
                    losses in kW.

                chargingdemand_kWh : float
                    Charging demand in kWh.

                park_time_timesteps : int
                    Number of parking time steps.

                park_start_timesteps : int
                    Time step the parking event starts.

                park_end_timesteps : int
                    Time step the parking event ends.

                charging_park_id : int
                    Designated charging park ID from potential_charging_parks_gdf. Is
                    NaN if the charging demand is not yet distributed.

                charging_point_id : int
                    Designated charging point ID. Is used to differentiate between
                    multiple charging points at one charging park.

        """
        try:
            return self._charging_processes_df
        except Exception:
            return pd.DataFrame(columns=COLUMNS["charging_processes_df"])

    @charging_processes_df.setter
    def charging_processes_df(self, df):
        self._charging_processes_df = df

    @property
    def potential_charging_parks_gdf(self):
        """
        GeoDataFrame with all
        `TracBEV <https://github.com/rl-institut/tracbev>`_
        potential charging parks.

        Returns
        -------
        :geopandas:`GeoDataFrame`
            GeoDataFrame with ID as index, AGS, charging use case (home, work, public or
            hpc), user centric weight and geometry. Columns are:

                index : int
                    Charging park ID.

                use_case : str
                    TracBEV use case. Can be "hpc", "home", "public" or "work".

                user_centric_weight : flaot
                    User centric weight used in distribution of charging demand. Weight
                    is determined by TracBEV but normalized from 0 .. 1.

                geometry : GeoSeries
                    Geolocation of charging parks.

        """
        try:
            return self._potential_charging_parks_gdf
        except Exception:
            return gpd.GeoDataFrame(columns=COLUMNS["potential_charging_parks_gdf"])

    @potential_charging_parks_gdf.setter
    def potential_charging_parks_gdf(self, gdf):
        self._potential_charging_parks_gdf = gdf

    @property
    def potential_charging_parks(self):
        """
        Potential charging parks within the AGS.

        Returns
        -------
        list(:class:`~.network.components.PotentialChargingParks`)
            List of potential charging parks within the AGS.

        """
        for cp_id in self.potential_charging_parks_gdf.index:
            yield PotentialChargingParks(id=cp_id, edisgo_obj=self._edisgo_obj)

    @property
    def simbev_config_df(self):
        """
        Dict with all
        `SimBEV <https://github.com/rl-institut/simbev>`_
        config data.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with used regio type, charging point efficiency, stepsize in
            minutes, start date, end date, minimum SoC for hpc, grid timeseries setting,
            grid timeseries by use case setting and the number of simulated days.
            Columns are:

                regio_type : str
                    RegioStaR 7 ID used in SimBEV.

                eta_cp : float or int
                    Charging point efficiency used in SimBEV.

                stepsize : int
                    Stepsize in minutes the driving profile is simulated for in SimBEV.

                start_date : datetime64
                    Start date of the SimBEV simulation.

                end_date : datetime64
                    End date of the SimBEV simulation.

                soc_min : float
                    Minimum SoC when a HPC event is initialized in SimBEV.

                grid_timeseries : bool
                    Setting whether a grid timeseries is generated within the SimBEV
                    simulation.

                grid_timeseries_by_usecase : bool
                    Setting whether a grid timeseries by use case is generated within
                    the SimBEV simulation.

                days : int
                    Timedelta between the end_date and start_date in days.

        """
        try:
            return self._simbev_config_df
        except Exception:
            return pd.DataFrame(columns=COLUMNS["simbev_config_df"])

    @simbev_config_df.setter
    def simbev_config_df(self, df):
        self._simbev_config_df = df

    @property
    def integrated_charging_parks_df(self):
        """
        Mapping DataFrame to map the charging park ID to the internal eDisGo ID.

        The eDisGo ID is determined when integrating components using
        :func:`~.EDisGo.add_component` or
        :func:`~.EDisGo.integrate_component_based_on_geolocation` method.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Mapping DataFrame to map the charging park ID to the internal eDisGo ID.

        """
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
        int
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
        int
            Number of simulated days

        """
        try:
            return int(self.simbev_config_df.at[0, "days"])
        except Exception:
            return None

    @property
    def eta_charging_points(self):
        """
        Charging point efficiency.

        Returns
        -------
        float
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
        * 'potential_charging_parks_gdf' : Attribute
          :py:attr:`~potential_charging_parks_gdf` is saved to
          `potential_charging_parks.csv`.
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

            if attr == "potential_charging_parks_gdf":
                epsg = edisgo_obj.topology.grid_district["srid"]

                df = df.assign(geometry=gpd.GeoSeries.from_wkt(df["geometry"]))

                try:
                    df = gpd.GeoDataFrame(
                        df, geometry="geometry", crs={"init": f"epsg:{epsg}"}
                    )

                except Exception:
                    logging.warning(
                        f"Potential charging parks could not be loaded with "
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
            weight, charging point capacity and charging point weight. Columns are:

                lv_grid_id : int
                    ID of nearest lv grid.

                distance_to_nearest_substation : float
                    Distance to nearest lv grid substation.

                distance_weight : float
                    Weighting used in grid friendly siting of public charging points.
                    In the case of distance to nearest substation the weight is higher
                    the closer the substation is to the charging park. The weight is
                    normalized between 0 .. 1. A higher weight is more attractive.

                charging_point_capacity : float
                    Total gross designated charging park capacity in kW.

                charging_point_weight : float
                    Weighting used in grid friendly siting of public charging points.
                    In the case of charging points the weight is higher the lower the
                    designated charging point capacity is. The weight is normalized
                    between 0 .. 1. A higher weight is more attractive.

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
        "potential_charging_parks_gdf": "potential_charging_parks.csv",
        "integrated_charging_parks_df": "integrated_charging_parks.csv",
        "simbev_config_df": "metadata_simbev_run.csv",
    }

    return emob_dict
