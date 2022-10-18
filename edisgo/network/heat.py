import ast
import os

from zipfile import ZipFile

import pandas as pd

# from edisgo.io import timeseries_import


class HeatPump:
    """
    Data container for all heat pump data.

    This class holds data on heat pump COP, heat demand time series, thermal storage
    data...

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
                Initial state of charge in MWh.

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

    @property
    def building_ids_df(self):
        """
        DataFrame with buildings served by each heat pump.

        Parameters
        -----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with building IDs of buildings served by each heat pump.
            Index of the dataframe are names of heat pumps as in
            :attr:`~.network.topology.Topology.loads_df`.
            Columns of the dataframe are:

            building_ids : list(int)
                List of building IDs.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with building IDs of buildings served by each heat pump.
            For more information on the dataframe see input parameter `df`.

        """
        try:
            return self._building_ids_df
        except Exception:
            return pd.DataFrame(columns=["building_ids"])

    @building_ids_df.setter
    def building_ids_df(self, df):
        # convert data in building_ids to list (when read from csv it is read as string)
        if not df.empty and isinstance(df.building_ids[0], str):
            df["building_ids"] = df["building_ids"].apply(lambda x: ast.literal_eval(x))
        self._building_ids_df = df

    def set_cop(self, edisgo_object, ts_cop, heat_pump_names=None):
        """
        Get COP time series for heat pumps.

        Heat pumps need to already be integrated into the grid.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_cop : str or :pandas:`pandas.DataFrame<dataframe>`
            Defines option used to set COP time series.
            Possible options are:

            * 'oedb'

                Not yet implemented!
                Weather cell specific hourly COP time series are obtained from the
                `OpenEnergy DataBase
                <https://openenergy-platform.org/dataedit/schemas>`_
                for the weather year 2011. See
                :func:`edisgo.io.timeseries_import.cop_oedb` for more information.
                Using information on which weather cell each heat pump is in, the
                weather cell specific time series are mapped to each heat pump.

            * :pandas:`pandas.DataFrame<dataframe>`

                DataFrame with self-provided COP time series per heat pump.
                See :py:attr:`~cop_df` on information on the required dataframe format.

        heat_pump_names : list(str) or None
            Defines for which heat pumps to get COP time series for in case `ts_cop` is
            'oedb'. If None, all heat pumps in
            :attr:`~.network.topology.Topology.loads_df` (type is 'heat_pump') are
            used. Default: None.

        """

        # in case time series from oedb are used, retrieve oedb time series
        if isinstance(ts_cop, str) and ts_cop == "oedb":
            raise NotImplementedError
            # # get COP per weather cell
            # ts_cop_per_weather_cell = timeseries_import.cop_oedb(
            #     edisgo_object.config, weather_cell_ids,
            #     edisgo_object.timeseries.timeindex
            # )
            # # get weather cells per heat pump and assign COP to each heat pump
            # if heat_pump_names is None:
            #     heat_pump_names = edisgo_object.topology.loads_df[
            #         edisgo_object.topology.loads_df.type == "heat_pump"]

        elif isinstance(ts_cop, pd.DataFrame):
            self.cop_df = ts_cop
        else:
            raise ValueError("'ts_cop' must either be a pandas DataFrame or 'oedb'.")

    def set_heat_demand(self, edisgo_object, ts_heat_demand, heat_pump_names=None):
        """
        Get heat demand time series for buildings with heat pumps.

        Heat pumps need to already be integrated into the grid.

        Parameters
        ----------
        edisgo_object : :class:`~.EDisGo`
        ts_heat_demand : str or :pandas:`pandas.DataFrame<dataframe>`
            Defines option used to set heat demand time series.
            Possible options are:

            * 'oedb'

                Not yet implemented!
                Heat demand time series are obtained from the `OpenEnergy DataBase
                <https://openenergy-platform.org/dataedit/schemas>`_
                for the weather year 2011. See
                :func:`edisgo.io.timeseries_import.heat_demand_oedb` for more
                information.

            * :pandas:`pandas.DataFrame<dataframe>`

                DataFrame with self-provided heat demand time series per heat pump.
                See :py:attr:`~heat_demand_df` on information on the required
                dataframe format.

        heat_pump_names : list(str) or None
            Defines for which heat pumps to get heat demand time series for in
            case `ts_heat_demand` is 'oedb'. If None, all heat pumps in
            :attr:`~.network.topology.Topology.loads_df` (type is 'heat_pump') are
            used. Default: None.

        """

        # in case time series from oedb are used, retrieve oedb time series
        if isinstance(ts_heat_demand, str) and ts_heat_demand == "oedb":
            raise NotImplementedError
            # ToDo Also include large heat pumps for district heating that don't have
            #  a building ID
            #
            # if heat_pump_names is None:
            #     heat_pump_names = edisgo_object.topology.loads_df[
            #         edisgo_object.topology.loads_df.type == "heat_pump"
            #         ]
            # # get building ID each heat pump is in
            #
            # # get heat demand per building
            # heat_demand_buildings_df = timeseries_import.heat_demand_oedb(
            #     edisgo_object.config,
            #     building_ids,
            #     edisgo_object.timeseries.timeindex,
            # )
            #
            # # map building ID back to heat pump
            # self.heat_demand_df = heat_demand_buildings_df.rename(
            #     columns={}
            # )

        elif isinstance(ts_heat_demand, pd.DataFrame):
            self.heat_demand_df = ts_heat_demand
        else:
            raise ValueError(
                "'ts_heat_demand' must either be a pandas DataFrame or 'oedb'."
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
            attr_to_reduce = list(
                self._get_matching_dict_of_attributes_and_file_names().keys()
            )
            attr_to_reduce.remove("thermal_storage_units_df")
            attr_to_reduce.remove("building_ids_df")
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
            "building_ids_df": "building_ids.csv",
        }

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
        * 'building_ids_df'

            Attribute :py:attr:`~building_ids_df` is saved to
            `building_ids.csv`.

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
