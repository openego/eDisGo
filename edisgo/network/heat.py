from __future__ import annotations

import logging
import os

from zipfile import ZipFile

import numpy as np
import pandas as pd

from edisgo.io import heat_pump_import, timeseries_import
from edisgo.tools import tools

logger = logging.getLogger(__name__)


class HeatPump:
    """
    Data container for all heat pump data.

    This class holds data on heat pump COP, heat demand time series, and thermal storage
    data.

    """

    def __init__(self, **kwargs):
        pass

    @property
    def cop_df(self):
        """
        DataFrame with COP time series of heat pumps.

        Parameters
        -----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with COP time series of heat pumps in p.u.. Index of the dataframe
            is a time index and should contain all time steps given in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Column names are names of heat pumps as in
            :attr:`~.network.topology.Topology.loads_df`.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with COP time series of heat pumps in p.u..
            For more information on the dataframe see input parameter `df`.

        """
        try:
            return self._cop_df
        except Exception:
            return pd.DataFrame()

    @cop_df.setter
    def cop_df(self, df):
        self._cop_df = df

    @property
    def heat_demand_df(self):
        """
        DataFrame with heat demand time series of heat pumps.

        Parameters
        -----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with heat demand time series of heat pumps in MW.
            Index of the dataframe is a time index and should contain all time steps
            given in :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Column names are names of heat pumps as in
            :attr:`~.network.topology.Topology.loads_df`.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with heat demand time series of heat pumps in MW.
            For more information on the dataframe see input parameter `df`.

        """
        try:
            return self._heat_demand_df
        except Exception:
            return pd.DataFrame()

    @heat_demand_df.setter
    def heat_demand_df(self, df):
        self._heat_demand_df = df

    @property
    def thermal_storage_units_df(self):
        """
        DataFrame with heat pump's thermal storage information.

        Parameters
        -----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with thermal storage information.
            Index of the dataframe are names of heat pumps as in
            :attr:`~.network.topology.Topology.loads_df`.
            Columns of the dataframe are:

            capacity : float
                Thermal storage capacity in MWh.

            efficiency : float
                Charging and discharging efficiency in p.u..

            state_of_charge_initial : float
                Initial state of charge in p.u..

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with thermal storage information.
            For more information on the dataframe see input parameter `df`.

        """
        try:
            return self._thermal_storage_units_df
        except Exception:
            return pd.DataFrame(
                columns=["capacity", "efficiency", "state_of_charge_initial"]
            )

    @thermal_storage_units_df.setter
    def thermal_storage_units_df(self, df):
        self._thermal_storage_units_df = df

    def set_cop(self, edisgo_object, ts_cop, **kwargs):
        """
        Write COP time series for heat pumps to py:attr:`~cop_df`.

        COP time series can either be given to this function or be obtained from the
        `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.
        In case they are obtained from the OpenEnergy DataBase the heat pumps need to
        already be integrated into the grid, i.e. given in
        :attr:`~.network.topology.Topology.loads_df`.

        In case COP time series are set for heat pumps that were already
        assigned a COP time series, their existing COP time series is
        overwritten by this function.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_cop : str or :pandas:`pandas.DataFrame<DataFrame>`
            Defines option used to set COP time series.
            Possible options are:

            * 'oedb'

                COP / efficiency data are obtained from the `OpenEnergy DataBase
                <https://openenergy-platform.org/dataedit/schemas>`_.
                In case of heat pumps weather cell specific hourly COP time series
                are obtained (see :func:`edisgo.io.timeseries_import.cop_oedb` for more
                information). Using information on which weather cell each heat pump
                is in, the weather cell specific time series are mapped to each heat
                pump.
                In case of resistive heaters a constant efficiency is set (see
                :func:`edisgo.io.heat_pump_import.efficiency_resistive_heaters_oedb`).

                Weather cell information of heat pumps is obtained from column
                'weather_cell_id' in :attr:`~.network.topology.Topology.loads_df`. In
                case no heat pump has weather cell information, this function will throw
                an error. In case only some heat pumps are missing weather cell
                information, a random existing weather cell is used to fill missing
                information.

                This option requires that the parameter `engine` is provided as keyword
                argument. For further settings, the parameters `timeindex` and
                `heat_pump_names` can also be provided as keyword arguments.

            * :pandas:`pandas.DataFrame<DataFrame>`

                DataFrame with self-provided COP time series per heat pump.
                See :py:attr:`~cop_df` on information on the required dataframe format.

        Other Parameters
        ------------------
        engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine. This parameter is required in case `ts_cop` is 'oedb'.
        heat_pump_names : list(str) or None
            Defines for which heat pumps to set COP time series in case `ts_cop` is
            'oedb'. If None, all heat pumps in
            :attr:`~.network.topology.Topology.loads_df` (type is 'heat_pump') are
            used. Default: None.
        timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
            Specifies time steps for which to set data in case `ts_cop` is
            'oedb'. Leap years can currently not be handled. In case the given
            timeindex contains a leap year, the data will be indexed using the default
            year 2011 and returned for the whole year.
            If no timeindex is provided, the timeindex set in
            :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
            If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
            is indexed using the default year 2011 and returned for the whole year.

        """
        if isinstance(ts_cop, str) and ts_cop == "oedb":
            heat_pump_names = kwargs.get("heat_pump_names", None)
            # get heat_pump_names in case they are not specified
            if heat_pump_names is None:
                heat_pump_names = edisgo_object.topology.loads_df[
                    edisgo_object.topology.loads_df.type == "heat_pump"
                ].index

            pth_df = edisgo_object.topology.loads_df.loc[heat_pump_names, :]
            hp_df = pth_df[
                ~pth_df.sector.isin(
                    [
                        "individual_heating_resistive_heater",
                        "district_heating_resistive_heater",
                    ]
                )
            ]

            # set COP of heat pumps
            if len(hp_df) > 0:
                # check weather cell information of heat pumps
                # if no heat pump has weather cell information, throw an error
                if (
                    "weather_cell_id" not in hp_df.columns
                    or hp_df.weather_cell_id.isna().all()
                ):
                    raise ValueError(
                        "In order to obtain COP time series data from database  "
                        "information on weather cells (expected in column "
                        "'weather_cell_id' in Topology.loads_df) is needed, but none "
                        "is given."
                    )
                # in case only some heat pumps have missing weather cell information,
                # give a warning and use random weather cell ID to fill missing
                # information
                if hp_df.weather_cell_id.isna().any():
                    logger.warning(
                        "There are heat pumps with no weather cell ID. They are "
                        "assigned a weather cell ID from another heat pump."
                    )
                    random_weather_cell_id = hp_df.weather_cell_id.dropna().unique()[0]
                    hp_without_weather_cell = hp_df[hp_df.weather_cell_id.isna()].index
                    # random weather cell ID is not written to loads_df!
                    hp_df.loc[
                        hp_without_weather_cell, "weather_cell_id"
                    ] = random_weather_cell_id
                weather_cells = hp_df.weather_cell_id.dropna().unique()

                # get COP per weather cell
                ts_cop_per_weather_cell = timeseries_import.cop_oedb(
                    edisgo_object=edisgo_object,
                    engine=kwargs.get("engine", None),
                    weather_cell_ids=weather_cells,
                    timeindex=kwargs.get("timeindex", None),
                )
                # assign COP time series to each heat pump
                cop_df = pd.DataFrame(
                    data={
                        _: ts_cop_per_weather_cell.loc[
                            :, hp_df.at[_, "weather_cell_id"]
                        ]
                        for _ in hp_df.index
                    }
                )
            else:
                cop_df = pd.DataFrame()

            # set efficiency of resistive heaters
            rh_df = pth_df[
                pth_df.sector.isin(
                    [
                        "individual_heating_resistive_heater",
                        "district_heating_resistive_heater",
                    ]
                )
            ]
            if len(rh_df) > 0:
                # get efficiencies of resistive heaters
                eta_dict = heat_pump_import.efficiency_resistive_heaters_oedb(
                    scenario="eGon2035",  # currently only possible scenario
                    engine=kwargs.get("engine", None),
                )
                # determine timeindex to use
                if not cop_df.empty:
                    timeindex = cop_df.index
                else:
                    timeindex, _ = timeseries_import._timeindex_helper_func(
                        edisgo_object,
                        kwargs.get("timeindex", None),
                        default_year=2011,
                        allow_leap_year=False,
                    )
                # assign efficiency time series to each heat pump
                eta_df = pd.DataFrame(
                    data={
                        _: (
                            eta_dict["central_resistive_heater"]
                            if rh_df.at[_, "sector"]
                            == "district_heating_resistive_heater"
                            else eta_dict["rural_resistive_heater"]
                        )
                        for _ in rh_df.index
                    },
                    index=timeindex,
                )
            else:
                eta_df = pd.DataFrame()
            cop_df = pd.concat([cop_df, eta_df], axis=1)
        elif isinstance(ts_cop, pd.DataFrame):
            cop_df = ts_cop
        else:
            raise ValueError("'ts_cop' must either be a pandas DataFrame or 'oedb'.")
        # concat new COP time series with existing ones and drop any duplicate entries
        self.cop_df = tools.drop_duplicated_columns(
            pd.concat([self.cop_df, cop_df], axis=1)
        )

    def set_heat_demand(self, edisgo_object, ts_heat_demand, **kwargs):
        """
        Write heat demand time series of heat pumps to py:attr:`~heat_demand_df`.

        Heat demand time series can either be given to this function or be obtained from
        the `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.
        In case they are obtained from the OpenEnergy DataBase the heat pumps need to
        already be integrated into the grid, i.e. given in
        :attr:`~.network.topology.Topology.loads_df`.

        In case heat demand time series are set for heat pumps that were already
        assigned a heat demand time series, their existing heat demand time series is
        overwritten by this function.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_heat_demand : str or :pandas:`pandas.DataFrame<DataFrame>`
            Defines option used to set heat demand time series.
            Possible options are:

            * 'oedb'

                Heat demand time series are obtained from the `OpenEnergy DataBase
                <https://openenergy-platform.org/dataedit/schemas>`_ (see
                :func:`edisgo.io.timeseries_import.heat_demand_oedb` for more
                information).
                Time series are only obtained for heat pumps that are already integrated
                into the grid.
                This option requires that the parameters `engine` and `scenario` are
                provided as keyword arguments. For further settings, the parameters
                `timeindex` and `heat_pump_names` can also be provided as keyword
                arguments.

            * :pandas:`pandas.DataFrame<DataFrame>`

                DataFrame with self-provided heat demand time series per heat pump.
                See :py:attr:`~heat_demand_df` for information on the required
                dataframe format.

        Other Parameters
        ------------------
        scenario : str
            Scenario for which to retrieve heat demand data. This parameter is required
            in case `ts_heat_demand` is 'oedb'.  Possible options are 'eGon2035' and
            'eGon100RE'.
        engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine. This parameter is required in case `ts_heat_demand` is
            'oedb'.
        heat_pump_names : list(str) or None
            Defines for which heat pumps to get heat demand time series for in
            case `ts_heat_demand` is 'oedb'. If None, all heat pumps in
            :attr:`~.network.topology.Topology.loads_df` (type is 'heat_pump') are
            used. Default: None.
        timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
            Specifies time steps for which to set data in case `ts_heat_demand` is
            'oedb'. Leap years can currently not be handled. In case the given
            timeindex contains a leap year, the data will be indexed using the default
            year (2035 in case of the 'eGon2035' and to 2045 in case of the
            'eGon100RE' scenario) and returned for the whole year.
            If no timeindex is provided, the timeindex set in
            :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
            If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
            is indexed using the default year and returned for the whole year.

        """
        # in case time series from oedb are used, retrieve oedb time series
        if isinstance(ts_heat_demand, str) and ts_heat_demand == "oedb":
            heat_pump_names = kwargs.get("heat_pump_names", None)
            # get heat_pump_names in case they are not specified
            if heat_pump_names is None:
                heat_pump_names = edisgo_object.topology.loads_df[
                    edisgo_object.topology.loads_df.type == "heat_pump"
                ].index

            if len(heat_pump_names) > 0:
                # get heat demand per heat pump
                heat_demand_df = timeseries_import.heat_demand_oedb(
                    edisgo_object,
                    scenario=kwargs.get("scenario", ""),
                    engine=kwargs.get("engine", None),
                    timeindex=kwargs.get("timeindex", None),
                )
                heat_pump_names_select = [
                    _ for _ in heat_demand_df.columns if _ in heat_pump_names
                ]
                heat_demand_df = heat_demand_df.loc[:, heat_pump_names_select]
            else:
                heat_demand_df = pd.DataFrame()

        elif isinstance(ts_heat_demand, pd.DataFrame):
            heat_demand_df = ts_heat_demand
        else:
            raise ValueError(
                "'ts_heat_demand' must either be a pandas DataFrame or 'oedb'."
            )
        # concat new COP time series with existing ones and drop any duplicate entries
        self.heat_demand_df = tools.drop_duplicated_columns(
            pd.concat([self.heat_demand_df, heat_demand_df], axis=1)
        )

    def reduce_memory(self, attr_to_reduce=None, to_type="float32"):
        """
        Reduces size of dataframes to save memory.

        See :attr:`~.edisgo.EDisGo.reduce_memory` for more information.

        Parameters
        -----------
        attr_to_reduce : list(str), optional
            List of attributes to reduce size for. Per default, the following attributes
            are reduced if they exist: cop_df, heat_demand_df.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        """
        if attr_to_reduce is None:
            attr_to_reduce = self._timeseries_attributes
        for attr in attr_to_reduce:
            setattr(
                self,
                attr,
                getattr(self, attr).apply(lambda _: _.astype(to_type)),
            )

    def _get_matching_dict_of_attributes_and_file_names(self):
        """
        Helper function that matches attribute names to file names.

        Is used in functions :py:attr:`~to_csv` and :py:attr:`~from_csv` to set
        which attribute of :class:`~.network.heat.HeatPump` is saved under
        which file name.

        Returns
        -------
        dict
            Dictionary matching attribute names and file names with attribute
            names as keys and corresponding file names as values.

        """
        return {
            "cop_df": "cop.csv",
            "heat_demand_df": "heat_demand.csv",
            "thermal_storage_units_df": "thermal_storage_units.csv",
        }

    @property
    def _timeseries_attributes(self):
        return ["heat_demand_df", "cop_df"]

    def to_csv(self, directory, reduce_memory=False, **kwargs):
        """
        Exports heat pump data to csv files.

        The following attributes are exported:

        * 'cop_df'

            Attribute :py:attr:`~cop_df` is saved to `cop.csv`.
        * 'heat_demand_df'

            Attribute :py:attr:`~heat_demand_df` is saved to `heat_demand.csv`.
        * 'thermal_storage_units_df'

            Attribute :py:attr:`~thermal_storage_units_df` is saved to
            `thermal_storage_units.csv`.

        Parameters
        ----------
        directory : str
            Path to save data to.
        reduce_memory : bool, optional
            If True, size of dataframes is reduced using
            :attr:`~.network.heat.HeatPump.reduce_memory`.
            Optional parameters of :attr:`~.network.heat.HeatPump.reduce_memory`
            can be passed as kwargs to this function. Default: False.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain arguments of
            :attr:`~.network.heat.HeatPump.reduce_memory`.

        """
        if reduce_memory is True:
            self.reduce_memory(**kwargs)

        os.makedirs(directory, exist_ok=True)

        attrs = self._get_matching_dict_of_attributes_and_file_names()

        for attr, file in attrs.items():
            df = getattr(self, attr)

            if not df.empty:
                path = os.path.join(directory, file)
                df.to_csv(path)

    def from_csv(self, data_path, from_zip_archive=False):
        """
        Restores heat pump data from csv files.

        Parameters
        ----------
        data_path : str
            Path to heat pump csv files.
        from_zip_archive : bool, optional
            Set True if data is archived in a zip archive. Default: False

        """
        attrs = self._get_matching_dict_of_attributes_and_file_names()

        if from_zip_archive:
            # read from zip archive
            # setup ZipFile Class
            zip = ZipFile(data_path)

            # get all directories and files within zip archive
            files = zip.namelist()

            # add directory and .csv to files to match zip archive
            attrs = {k: f"heat_pump/{v}" for k, v in attrs.items()}

        else:
            # read from directory
            # check files within the directory
            files = os.listdir(data_path)

        attrs_to_read = {k: v for k, v in attrs.items() if v in files}

        for attr, file in attrs_to_read.items():
            if from_zip_archive:
                # open zip file to make it readable for pandas
                with zip.open(file) as f:
                    df = pd.read_csv(f, index_col=0, parse_dates=True)
            else:
                path = os.path.join(data_path, file)
                df = pd.read_csv(path, index_col=0, parse_dates=True)

            setattr(self, attr, df)

        if from_zip_archive:
            # make sure to destroy ZipFile Class to close any open connections
            zip.close()

    def resample_timeseries(
        self, method: str = "ffill", freq: str | pd.Timedelta = "15min"
    ):
        """
        Resamples COP and heat demand time series to a desired resolution.

        Both up- and down-sampling methods are possible.

        Parameters
        ----------
        method : str, optional
            See :attr:`~.EDisGo.resample_timeseries` for more information.

        freq : str, optional
            See :attr:`~.EDisGo.resample_timeseries` for more information.

        """
        for attr in self._timeseries_attributes:
            attr_index = getattr(self, attr).index
            if len(attr_index) < 2:
                logger.debug(
                    f"{attr} cannot be resampled as it contains less than two "
                    f"time steps."
                )
            else:
                freq_orig = attr_index[1] - attr_index[0]
                tools.resample(self, freq_orig, method, freq, attr_to_resample=[attr])

    def check_integrity(self):
        """
        Check data integrity.

        Checks for duplicated and missing labels as well as implausible values.

        """
        check_dfs = ["heat_demand_df", "cop_df"]
        # check for duplicate columns
        for ts in check_dfs:
            df = getattr(self, ts)
            duplicated_labels = df.columns[df.columns.duplicated()].values

            if len(duplicated_labels) > 0:
                logger.warning(
                    f"HeatPump timeseries {ts} contains the following duplicates: "
                    f"{set(duplicated_labels)}."
                )

        # check that all profiles exist for the same heat pumps
        columns = set(np.concatenate([getattr(self, _).columns for _ in check_dfs]))
        for ts in check_dfs:
            df = getattr(self, ts)
            missing_entries = [_ for _ in columns if _ not in df.columns]
            if len(missing_entries) > 0:
                logger.warning(
                    f"HeatPump timeseries {ts} is missing the following "
                    f"entries: {missing_entries}."
                )
