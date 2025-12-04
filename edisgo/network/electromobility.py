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
        self._edisgo_obj = kwargs.get("edisgo_obj")

    @property
    def charging_processes_df(self):
        """
        DataFrame with all charging processes.

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
                    losses (nominal_charging_capacity_kW / eta_cp) in kW.

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
        GeoDataFrame with all potential charging parks.

        Returns
        -------
        :geopandas:`geopandas.GeoDataFrame<GeoDataFrame>`
            GeoDataFrame with ID as index, AGS, charging use case (home, work, public or
            hpc), user-centric weight and geometry. Columns are:

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
        Dictionary containing configuration data.

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
                    Minimum SoC when an HPC event is initialized in SimBEV.

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

    def get_flexibility_bands(self, edisgo_obj, use_case, resample=True, tol=1e-6):
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
        resample : bool (optional)
            If True, flexibility bands are resampled to the same frequency as time
            series data in :class:`~.network.timeseries.TimeSeries` object. If False,
            original frequency is kept.
            Default: True.
        tol : float
            Tolerance to reduce or increase flexibility band values by to fix
            possible rounding errors that may lead to failing integrity checks
            and infeasibility when used to optimise charging.
            See :py:attr:`~fix_flexibility_bands_rounding_errors`
            for more information. To avoid this behaviour, set `tol` to 0.0.
            Default: 1e-6.

        Returns
        --------
        dict(str, :pandas:`pandas.DataFrame<DataFrame>`)
            Keys are 'upper_power', 'lower_energy' and 'upper_energy'.
            Values are dataframes containing the corresponding band for each charging
            point of the specified use case. Columns of the dataframe are the
            charging point names as in :attr:`~.network.topology.Topology.loads_df`.
            Index is a time index.

        """

        if isinstance(use_case, str):
            use_case = [use_case]

        # get all relevant charging points
        cp_df = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df.type == "charging_point"
        ]
        cps = cp_df[cp_df.sector.isin(use_case)]

        # set up time index
        start_date = self.simbev_config_df.start_date.values[0]
        # end date from SimBEV includes to the specified day, wherefore 1 day needs
        # to be added to have the day included in the time index
        end_date = self.simbev_config_df.end_date.values[0] + pd.Timedelta(1, "day")
        stepsize = self.stepsize
        flex_band_index = pd.date_range(
            start=start_date, end=end_date, freq=f"{stepsize}min", inclusive="left"
        )
        # check if maximum end time step in charging data is larger than length of
        # time index and if so, expand time index and raise warning
        t_max = self.charging_processes_df.park_end_timesteps.max()
        if len(flex_band_index) < t_max:
            logger.warning(
                "Time steps in charging processes exceed time steps specified in "
                "SimBEV config data."
            )
            flex_band_index = pd.date_range(
                start=start_date, periods=t_max + 1, freq=f"{stepsize}min"
            )

        # set up bands
        tmp_idx = range(len(flex_band_index))
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
                if charging_process.park_end_timesteps == max(tmp_idx):
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

        # convert to MW and cumulate energy
        upper_power = upper_power / 1e3
        lower_energy = lower_energy.cumsum() / hourly_steps / 1e3
        upper_energy = upper_energy.cumsum() / hourly_steps / 1e3

        # set time index
        upper_power.index = flex_band_index
        lower_energy.index = flex_band_index
        upper_energy.index = flex_band_index

        # write to self.flexibility_bands
        flex_band_dict = {
            "upper_power": upper_power,
            "lower_energy": lower_energy,
            "upper_energy": upper_energy,
        }
        self.flexibility_bands = flex_band_dict

        # fix rounding errors
        self.fix_flexibility_bands_rounding_errors(tol=tol)

        edisgo_timeindex = edisgo_obj.timeseries.timeindex
        if resample:
            # check if time index matches Timeseries.timeindex and if not resample flex
            # bands
            if len(edisgo_timeindex) > 1:
                # check if frequencies match
                freq_edisgo = edisgo_timeindex[1] - edisgo_timeindex[0]
                if freq_edisgo != pd.Timedelta(f"{stepsize}min"):
                    # resample
                    self.resample(freq=freq_edisgo)

        # sanity check
        self.check_integrity()
        # check time index
        if len(edisgo_timeindex) > 0:
            missing_indices = [_ for _ in edisgo_timeindex if _ not in flex_band_index]
            if len(missing_indices) > 0:
                logger.warning(
                    "There are time steps in timeindex of TimeSeries object that "
                    "are not in the index of the flexibility bands. This may lead "
                    "to problems."
                )
        return self.flexibility_bands

    def fix_flexibility_bands_rounding_errors(self, tol=1e-6):
        """
        Fixes possible rounding errors that may lead to failing integrity checks.

        Due to rounding errors it may occur, that e.g. the upper energy band is lower
        than the lower energy band. This does in some cases lead to infeasibilities
        when used to optimise charging processes.

        This function increases or reduces a flexibility band by the specified tolerance
        in case an integrity check fails as follows:

        * If there are cases where the upper power band is not sufficient to meet
          the charged upper energy, the upper power band is increased for all
          charging points and all time steps.
        * If there are cases where the lower energy band is larger than the upper
          energy band, the lower energy band is reduced for all charging points and
          all time steps.
        * If there are cases where upper power band is not sufficient
          to meet charged lower energy, the upper power band is increased for all
          charging points and all time steps.

        Parameters
        -----------
        tol : float
            Tolerance to reduce or increase values by to fix rounding errors.
            Default: 1e-6.

        """

        flex_band = list(self.flexibility_bands.values())[0]
        # if there are no flex bands, skip
        if flex_band.empty:
            return

        efficiency = self.eta_charging_points
        freq_orig = flex_band.index[1] - flex_band.index[0]
        hourly_steps = int(60 / (freq_orig.total_seconds() / 60))

        # increase upper power, if there are cases where upper power is not sufficient
        # to meet charged upper energy
        if (
            (
                (
                    self.flexibility_bands["upper_energy"].diff()
                    - self.flexibility_bands["upper_power"] * efficiency / hourly_steps
                )
                > 0.0
            )
            .any()
            .any()
        ):
            logger.debug(
                "There are cases when upper power is not sufficient to meet charged "
                "upper energy. Upper power band is therefore increased to avoid "
                "infeasibilities arising from rounding errors."
            )
            self.flexibility_bands["upper_power"] += tol

        # reduce lower energy band if there are cases where it is larger than upper
        # energy band
        if (
            (
                (
                    self.flexibility_bands["upper_energy"]
                    - self.flexibility_bands["lower_energy"]
                )
                < 0.0
            )
            .any()
            .any()
        ):
            logger.debug(
                "There are cases when lower energy band is larger than upper energy "
                "band. Lower energy band is therefore reduced to avoid infeasibilities "
                "arising from rounding errors."
            )
            self.flexibility_bands["lower_energy"] -= tol

        # increase upper power, if there are cases where upper power is not sufficient
        # to meet charged lower energy
        if (
            (
                (
                    self.flexibility_bands["lower_energy"].diff()
                    - self.flexibility_bands["upper_power"] * efficiency / hourly_steps
                )
                > 0.0
            )
            .any()
            .any()
        ):
            logger.debug(
                "There are cases when upper power is not sufficient to meet charged "
                "lower energy. Upper power band is therefore increased to avoid "
                "infeasibilities arising from rounding errors."
            )
            self.flexibility_bands["upper_power"] += tol

    def resample(self, freq: str = "15min"):
        """
        Resamples flexibility bands.

        Parameters
        ----------
        freq : str or :pandas:`pandas.Timedelta<Timedelta>`, optional
            Frequency that time series is resampled to. Offset aliases can be found
            here:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Default: '15min'.

        """

        flex_band = list(self.flexibility_bands.values())[0]
        if flex_band.empty or len(flex_band.index) < 2:
            return

        # check if frequency is always the same (can only be checked for more than two
        # time steps as pd.infer_freq needs more than two time steps)
        if len(flex_band.index) > 2:
            freq_inferred = pd.infer_freq(flex_band.index)
            if freq_inferred is None:
                logger.warning(
                    "Index of flexibility bands does not have a discernible frequency. "
                    "The flexibility bands can therefore not be resampled."
                )
                return

        # determine frequency of flexibility bands
        # pd.infer_freq is not used to determine frequency as it is not always
        # compatible with pd.Timedelta() needed to check whether to sample down or up
        freq_orig = flex_band.index[1] - flex_band.index[0]

        if not isinstance(freq, pd.Timedelta):
            freq = pd.Timedelta(freq)
        # in case of up-sampling, check if index is continuous and if new index fits
        # into old index a discrete number of times
        if freq < freq_orig:
            check_index = pd.date_range(
                start=flex_band.index.min(), end=flex_band.index.max(), freq=freq_orig
            )
            if not len(check_index) == len(flex_band.index):
                logger.warning(
                    "Index of flexibility bands is not continuous. This might lead "
                    "to problems."
                )
            num_times = int(freq_orig.total_seconds()) / int(freq.total_seconds())
            if not int(num_times) == num_times:
                logger.error(
                    "Up-sampling to an uneven number of times the new index fits into "
                    "the old index is not possible."
                )
                return

        # add time step at the end of the time series in case of up-sampling so that
        # last time interval in the original time series is still included
        df_dict = {}
        for band in self.flexibility_bands.keys():
            df_dict[band] = getattr(self, "flexibility_bands")[band]
            if freq < freq_orig:  # up-sampling
                end_date = pd.DatetimeIndex([df_dict[band].index[-1] + freq_orig])
            else:  # down-sampling (nothing happens)
                end_date = pd.DatetimeIndex([df_dict[band].index[-1]])
            df_dict[band] = (
                df_dict[band]
                .reindex(df_dict[band].index.union(end_date).unique().sort_values())
                .ffill()
            )

        # resample time series
        if freq < freq_orig:  # up-sampling
            for band in self.flexibility_bands.keys():
                if band == "upper_power":
                    df_dict[band] = df_dict[band].resample(freq, closed="left").ffill()
                    # drop last time step, as closed left does somehow still include the
                    # last time step
                    df_dict[band] = df_dict[band].iloc[:-1, :]
                else:
                    df_dict[band].sort_index(inplace=True)
                    index_pre = df_dict[band].index[0] - freq
                    # check how often the new index fits into the old index
                    num_times = int(freq_orig.total_seconds()) / int(
                        freq.total_seconds()
                    )
                    # shift index and re-append first time step
                    df_dict[band].index = df_dict[band].index.shift(
                        int(freq.total_seconds()) * (num_times - 1), "s"
                    )
                    # values of first time step are energy values minus possible change
                    # in energy negative values are set to zero
                    index_pre_values = (
                        df_dict[band].iloc[0]
                        - df_dict["upper_power"].iloc[0] / num_times
                    )
                    index_pre_values[index_pre_values < 0.0] = 0.0
                    df_dict[band] = pd.concat(
                        [
                            pd.DataFrame(
                                index=[index_pre],
                                columns=df_dict[band].columns,
                                data=index_pre_values.to_dict(),
                            ),
                            df_dict[band],
                        ]
                    )

                    # resample by interpolating
                    df_dict[band] = (
                        df_dict[band].resample(freq, closed="left").interpolate()
                    )

                    # drop time steps - time step that was added in the beginning
                    # and time steps that were added due to the shift
                    df_dict[band] = df_dict[band].loc[: end_date[0], :]
                    df_dict[band] = df_dict[band].iloc[1:-1, :]
        else:  # down-sampling
            for band in self.flexibility_bands.keys():
                if band == "upper_power":
                    df_dict[band] = df_dict[band].resample(freq).mean()
                else:
                    df_dict[band] = df_dict[band].resample(freq).max()
        self.flexibility_bands = df_dict

    def check_integrity(self):
        """
        Method to check the integrity of the Electromobility object.

        Raises an error in case any of the checks fails.

        Currently only checks integrity of flexibility bands.

        """
        # pick random flex band for some pre-checks
        flex_band = list(self.flexibility_bands.values())[0]

        # if there are no flex bands, skip integrity check
        if flex_band.empty:
            return

        efficiency = self.eta_charging_points
        freq_orig = flex_band.index[1] - flex_band.index[0]
        hourly_steps = int(60 / (freq_orig.total_seconds() / 60))

        diff = (
            self.flexibility_bands["upper_energy"]
            - self.flexibility_bands["lower_energy"]
        )
        tmp = (diff < 0.0).any()
        if tmp.any():
            max_exceedance = abs(diff.min().min())
            raise ValueError(
                f"Lower energy band is higher than upper energy band for the "
                f"following charging points: {list(tmp[tmp].index)}. The maximum "
                f"exceedance is {max_exceedance}. Please check."
            )

        diff = (
            self.flexibility_bands["upper_energy"].diff()
            - self.flexibility_bands["upper_power"] * efficiency / hourly_steps
        )
        tmp = (diff > 0.0).any()
        if tmp.any():
            max_exceedance = diff.max().max()
            raise ValueError(
                f"Upper energy band has power values higher than nominal power for the "
                f"following charging points: {list(tmp[tmp].index)}. The maximum "
                f"exceedance is {max_exceedance}. Please check."
            )

        diff = (
            self.flexibility_bands["lower_energy"].diff()
            - self.flexibility_bands["upper_power"] * efficiency / hourly_steps
        )
        tmp = (diff > 0.0).any()
        if tmp.any():
            max_exceedance = diff.max().max()
            raise ValueError(
                f"Lower energy band has power values higher than nominal power for the "
                f"following charging points: {list(tmp[tmp].index)}. The maximum "
                f"exceedance is {max_exceedance}. Please check."
            )

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
                    logger.warning(
                        f"Potential charging parks could not be loaded with "
                        f"EPSG {epsg}. Trying with EPSG 4326 as fallback."
                    )

                    df = gpd.GeoDataFrame(
                        df, geometry="geometry", crs={"init": "epsg:4326"}
                    )

            if attr == "simbev_config_df":
                for col in ["start_date", "end_date"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])

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
    :attr:`~.network.electromobility.Electromobility.to_csv`.

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
