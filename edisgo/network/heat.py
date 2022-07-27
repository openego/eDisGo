import logging
import os
import pandas as pd

from zipfile import ZipFile


class HeatPump:
    """
    Data container for all heat pump data.

    This class holds data on heat pump COP, heat demand time series, heat storage
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
    def heat_storage_units_df(self):
        """
        DataFrame with heat pump's heat storage information.

        Parameters
        -----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with heat storage information.
            Index of the dataframe are names of heat pumps as in
            :attr:`~.network.topology.Topology.loads_df`.
            Columns of the dataframe are:

            capacity : float
                Heat storage capacity in MWh.

            efficiency : float
                Charging and discharging efficiency in p.u..

            state_of_charge_initial : float
                Initial state of charge in MWh.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            DataFrame with heat storage information.
            For more information on the dataframe see input parameter `df`.

        """
        try:
            return self._heat_storage_units_df
        except Exception:
            return pd.DataFrame(
                columns=["capacity", "efficiency", "state_of_charge_initial"]
            )

    @heat_storage_units_df.setter
    def heat_storage_units_df(self, df):
        self._heat_storage_units_df = df

    def reduce_memory(
        self, attr_to_reduce=None, to_type="float32"
    ):
        """
        Reduces size of dataframes to save memory.

        See :attr:`~.edisgo.EDisGo.reduce_memory` for more information.

        Parameters
        -----------
        attr_to_reduce : list(str), optional
            List of attributes to reduce size for. Per default, the following attributes
            are reduced if they exist: cop_df, heat_demand_df, heat_storage_units_df.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        """
        if attr_to_reduce is None:
            attr_to_reduce = self._get_matching_dict_of_attributes_and_file_names.keys()
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
            "heat_storage_units_df": "heat_storage_units.csv",
        }

    def to_csv(self, directory, reduce_memory=False, **kwargs):
        """
        Exports heat pump data to csv files.

        The following attributes are exported:

        * 'cop_df'

            Attribute :py:attr:`~cop_df` is saved to `cop.csv`.
        * 'heat_demand_df'

            Attribute :py:attr:`~heat_demand_df` is saved to `heat_demand.csv`.
        * 'heat_storage_units_df'

            Attribute :py:attr:`~heat_storage_units_df` is saved to
            `heat_storage_units.csv`.

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
                    df = pd.read_csv(f, index_col=0)
            else:
                path = os.path.join(data_path, file)
                df = pd.read_csv(path, index_col=0)

            setattr(self, attr, df)

        if from_zip_archive:
            # make sure to destroy ZipFile Class to close any open connections
            zip.close()
