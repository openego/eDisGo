import logging
import os
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
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
        "netto_charging_capacity",
        "chargingdemand",
        "park_start",
        "park_end",
    ],
    "grid_connections_gdf": ["id", "use_case", "user_centric_weight", "geometry"],
    "simbev_config_df": ["value"],
    "potential_charging_parks_df": [
        "lv_grid_id",
        "distance_to_nearest_substation",
        "distance_weight",
        "charging_point_capacity",
        "charging_point_weight",
    ],
    "designated_charging_points_df": [
        "park_end",
        "netto_charging_capacity",
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
        :geopandas:`geodataframe`
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
        DataFrame with all `SimBEV <https://github.com/rl-institut/simbev>`_
        config data.

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
            return pd.DataFrame(columns=COLUMNS["simbev_config_df"])

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
            return int(self.simbev_config_df.at["stepsize", "value"])
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
            return int(self.simbev_config_df.at["days", "value"])
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
            return float(self.simbev_config_df.at["eta_CP", "value"])
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

                except Exception as _:
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

            potential_charging_parks_df.distance_weight = 1 - min_max_scaler.fit_transform(
                potential_charging_parks_df.distance_to_nearest_substation.values.reshape(
                    -1, 1
                )
            )

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
        "simbev_config_df": "simbev_config.csv",
    }

    return emob_dict


def get_energy_bands_for_optimization(edisgo_obj, use_case, t_max=372 * 4 * 24):
    """
    Method to extract flexibility bands for linear optimisation.
    """

    def _shorten_and_set_index(band):
        """
        Method to adjust bands to timeindex of edisgo_obj
        #Todo: change such that first day is replaced by (365+1)th day
        """
        band = band.iloc[: len(edisgo_obj.timeseries.timeindex)]
        band.index = edisgo_obj.timeseries.timeindex
        return band

    # get all relevant charging points
    cps = edisgo_obj.topology.charging_points_df.loc[
        edisgo_obj.topology.charging_points_df.use_case == use_case
    ]
    # set up bands
    tmp_idx = range(t_max)
    upper_power = pd.DataFrame(index=tmp_idx, columns=cps.index, data=0)
    upper_energy = pd.DataFrame(index=tmp_idx, columns=cps.index, data=0)
    lower_energy = pd.DataFrame(index=tmp_idx, columns=cps.index, data=0)
    hourly_steps = 60 / edisgo_obj.electromobility.stepsize
    for cp in cps.index:
        # get index of charging park used in charging processes
        charging_park_id = edisgo_obj.electromobility.integrated_charging_parks_df.loc[
            edisgo_obj.electromobility.integrated_charging_parks_df.edisgo_id == cp
        ].index
        # get relevant charging processes
        charging_processes = edisgo_obj.electromobility.charging_processes_df.loc[
            edisgo_obj.electromobility.charging_processes_df.charging_park_id.isin(
                charging_park_id
            )
        ]
        # iterate through charging processes and fill matrices
        for idx, charging_process in charging_processes.iterrows():
            # Last timesteps can lead to problems --> skip
            if charging_process.park_end == t_max:
                continue
            # get brutto charging capacity
            brutto_charging_capacity = (
                charging_process.netto_charging_capacity
                / edisgo_obj.electromobility.eta_charging_points
            )
            # charging power
            upper_power.loc[
                charging_process.park_start - 1 : charging_process.park_end - 1, cp
            ] += brutto_charging_capacity
            # energy bands
            charging_time = (
                charging_process.chargingdemand
                / charging_process.netto_charging_capacity
                * hourly_steps
            )
            if (
                charging_time
                - (charging_process.park_end - charging_process.park_start + 1)
                > 1e-6
            ):
                raise ValueError(
                    "Charging demand cannot be fulfilled for charging process {}. "
                    "Please check.".format(idx)
                )
            full_charging_steps = int(charging_time)
            part_time_step = charging_time - full_charging_steps
            # lower band
            lower_energy.loc[
                charging_process.park_end
                - full_charging_steps : charging_process.park_end
                - 1,
                cp,
            ] += charging_process.netto_charging_capacity
            lower_energy.loc[
                charging_process.park_end - full_charging_steps - 1, cp
            ] += (part_time_step * charging_process.netto_charging_capacity)
            # upper band
            upper_energy.loc[
                charging_process.park_start
                - 1 : charging_process.park_start
                + full_charging_steps
                - 2,
                cp,
            ] += charging_process.netto_charging_capacity
            upper_energy.loc[
                charging_process.park_start + full_charging_steps - 1, cp
            ] += (part_time_step * charging_process.netto_charging_capacity)
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
    if (
        (
            (
                upper_energy
                - upper_power * edisgo_obj.electromobility.eta_charging_points
            )
            > 1e-6
        )
        .any()
        .any()
    ):
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
    return {
        "upper_power": upper_power,
        "lower_energy": lower_energy,
        "upper_energy": upper_energy,
    }
