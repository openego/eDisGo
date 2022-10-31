import logging
import os

from zipfile import ZipFile

import pandas as pd

from sklearn import preprocessing

from edisgo.network.components import PotentialChargingParks

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

logger = logging.getLogger(__name__)

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
                    Car ID to differentiate charging processes from different cars.

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
            Charging point efficiency in p.u..

        """
        try:
            return float(self.simbev_config_df.at[0, "eta_cp"])
        except Exception:
            return None

    @property
    def flexibility_bands(self):
        """
        Dictionary with flexibility bands (lower and upper energy band as well as
        upper power band).

        Parameters
        -----------
        flex_dict : dict(str, :pandas:`pandas.DataFrame<DataFrame>`)
            Keys are 'upper_power', 'lower_energy' and 'upper_energy'.
            Values are dataframes containing the corresponding band per each charging
            point. Columns of the dataframe are the charging point names as in
            :attr:`~.network.topology.Topology.loads_df`. Index is a time index.

        Returns
        -------
        dict(str, :pandas:`pandas.DataFrame<DataFrame>`)
            See input parameter `flex_dict` for more information on the dictionary.

        """
        try:
            return self._flexibility_bands
        except Exception:
            return {
                "upper_power": pd.DataFrame(),
                "lower_energy": pd.DataFrame(),
                "upper_energy": pd.DataFrame(),
            }

    @flexibility_bands.setter
    def flexibility_bands(self, flex_dict):
        self._flexibility_bands = flex_dict

    def get_flexibility_bands(self, edisgo_obj, use_case):
        """
        Method to determine flexibility bands (lower and upper energy band as well as
        upper power band).

        Besides being returned by this function, flexibility bands are written to
        :attr:`flexibility_bands`.

        Parameters
        -----------
        edisgo_obj : :class:`~.EDisGo`
        use_case : str or list(str)
            Charging point use case(s) to determine flexibility bands for.

        Returns
        --------
        dict(str, :pandas:`pandas.DataFrame<DataFrame>`)
            Keys are 'upper_power', 'lower_energy' and 'upper_energy'.
            Values are dataframes containing the corresponding band for each charging
            point of the specified use case. Columns of the dataframe are the
            charging point names as in :attr:`~.network.topology.Topology.loads_df`.
            Index is a time index.

        """

        def _shorten_and_set_index(band):
            """
            Method to adjust bands to time index of EDisGo object.
            #Todo: change such that first day is replaced by (365+1)th day
            """
            band = band.iloc[: len(edisgo_obj.timeseries.timeindex)]
            band.index = edisgo_obj.timeseries.timeindex
            return band

        if isinstance(use_case, str):
            use_case = [use_case]

        # get all relevant charging points
        cp_df = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df.type == "charging_point"
        ]
        cps = cp_df[cp_df.sector.isin(use_case)]

        # set up bands
        t_max = 372 * 4 * 24
        tmp_idx = range(t_max)
        upper_power = pd.DataFrame(index=tmp_idx, columns=cps.index, data=0)
        upper_energy = pd.DataFrame(index=tmp_idx, columns=cps.index, data=0)
        lower_energy = pd.DataFrame(index=tmp_idx, columns=cps.index, data=0)
        hourly_steps = 60 / self.stepsize
        for cp in cps.index:
            # get index of charging park used in charging processes
            charging_park_id = self.integrated_charging_parks_df.loc[
                self.integrated_charging_parks_df.edisgo_id == cp
            ].index
            # get relevant charging processes
            charging_processes = self.charging_processes_df.loc[
                self.charging_processes_df.charging_park_id.isin(charging_park_id)
            ]
            # iterate through charging processes and fill matrices
            for idx, charging_process in charging_processes.iterrows():
                # Last time steps can lead to problems --> skip
                if charging_process.park_end_timesteps == t_max:
                    continue

                start = charging_process.park_start_timesteps
                end = charging_process.park_end_timesteps
                power = charging_process.nominal_charging_capacity_kW

                # charging power
                upper_power.loc[start:end, cp] += (
                    power / edisgo_obj.electromobility.eta_charging_points
                )
                # energy bands
                charging_time = (
                    charging_process.chargingdemand_kWh / power * hourly_steps
                )
                if charging_time - (end - start + 1) > 1e-6:
                    raise ValueError(
                        "Charging demand cannot be fulfilled for charging process {}. "
                        "Please check.".format(idx)
                    )
                full_charging_steps = int(charging_time)
                part_time_step = charging_time - full_charging_steps
                # lower band
                lower_energy.loc[end - full_charging_steps + 1 : end, cp] += power
                if part_time_step != 0.0:
                    lower_energy.loc[end - full_charging_steps, cp] += (
                        part_time_step * power
                    )
                # upper band
                upper_energy.loc[start : start + full_charging_steps - 1, cp] += power
                upper_energy.loc[start + full_charging_steps, cp] += (
                    part_time_step * power
                )
        # sanity check
        if (
            (
                (
                    lower_energy
                    - upper_power * edisgo_obj.electromobility.eta_charging_points
                )
                > 1e-6
            )
            .any()
            .any()
        ):
            raise ValueError(
                "Lower energy has power values higher than nominal power. Please check."
            )
        if ((upper_energy - upper_power * self.eta_charging_points) > 1e-6).any().any():
            raise ValueError(
                "Upper energy has power values higher than nominal power. Please check."
            )
        if ((upper_energy.cumsum() - lower_energy.cumsum()) < -1e-6).any().any():
            raise ValueError(
                "Lower energy is higher than upper energy bound. Please check."
            )
        # Convert to MW and cumulate energy
        upper_power = upper_power / 1e3
        lower_energy = lower_energy.cumsum() / hourly_steps / 1e3
        upper_energy = upper_energy.cumsum() / hourly_steps / 1e3
        # Set time_index
        upper_power = _shorten_and_set_index(upper_power)
        lower_energy = _shorten_and_set_index(lower_energy)
        upper_energy = _shorten_and_set_index(upper_energy)

        flex_band_dict = {
            "upper_power": upper_power,
            "lower_energy": lower_energy,
            "upper_energy": upper_energy,
        }
        self.flexibility_bands = flex_band_dict
        return flex_band_dict

    def to_csv(self, directory, attributes=None):
        """
        Exports electromobility data to csv files.

        The following attributes can be exported:

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
        * 'flexibility_bands' : The three flexibility bands in attribute
          :py:attr:`~flexibility_bands` are saved to
          `flexibility_band_upper_power.csv`, `flexibility_band_lower_energy.csv`, and
          `flexibility_band_upper_energy.csv`.

        Parameters
        ----------
        directory : str
            Path to save electromobility data to.
        attributes : list(str) or None
            List of attributes to export. See above for attributes that can be exported.
            If None, all specified attributes are exported. Default: None.

        """
        os.makedirs(directory, exist_ok=True)

        attrs_file_names = _get_matching_dict_of_attributes_and_file_names()

        if attributes is None:
            attributes = list(attrs_file_names.keys())

        for attr in attributes:
            file = attrs_file_names[attr]
            df = getattr(self, attr)
            if attr == "flexibility_bands":
                for band in file.keys():
                    if band in df.keys() and not df[band].empty:
                        path = os.path.join(directory, file[band])
                        df[band].to_csv(path)
            else:
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
            attrs = {
                k: (
                    f"electromobility/{v}"
                    if isinstance(v, str)
                    else {k2: f"electromobility/{v2}" for k2, v2 in v.items()}
                )
                for k, v in attrs.items()
            }

        else:
            # read from directory
            # check files within the directory
            files = os.listdir(data_path)

        attrs_to_read = {
            k: v
            for k, v in attrs.items()
            if (isinstance(v, str) and v in files)
            or (isinstance(v, dict) and any([_ in files for _ in v.values()]))
        }

        for attr, file in attrs_to_read.items():
            if attr == "flexibility_bands":
                df = {}
                for band, file_name in file.items():
                    if file_name in files:
                        if from_zip_archive:
                            # open zip file to make it readable for pandas
                            with zip.open(file_name) as f:
                                df[band] = pd.read_csv(f, index_col=0, parse_dates=True)
                        else:
                            path = os.path.join(data_path, file_name)
                            df[band] = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
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
    :attr:`~.network.electromobility.Electromobility.from_csv` and
    attr:`~.network.electromobility.Electromobility.to_csv`.

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
        "flexibility_bands": {
            "upper_power": "flexibility_band_upper_power.csv",
            "lower_energy": "flexibility_band_lower_energy.csv",
            "upper_energy": "flexibility_band_upper_energy.csv",
        },
    }

    return emob_dict
