import os

from zipfile import ZipFile

import pandas as pd


class OverlyingGrid:
    """
    Data container for requirements from the overlying grid.

    The requirements from the overlying grid are used as constraints for flexibilities.

    Attributes
    -----------
    renewables_curtailment : :pandas:`pandas.Series<Series>`
        Curtailment of fluctuating renewables per time step in MW.
    storage_units_active_power : :pandas:`pandas.Series<Series>`
        Aggregated dispatch of storage units per time step in MW.
    dsm_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand side management utilisation per time step in MW.
    electromobility_active_power : :pandas:`pandas.Series<Series>`
        Aggregated charging demand at flexible charging sites per time step in MW.
    heat_pump_decentral_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand of flexible decentral heat pumps per time step in MW.
    heat_pump_central_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand of flexible central heat pumps per time step in MW.
    geothermal_energy_feedin_district_heating : :pandas:`pandas.DataFrame<DataFrame>`
        Geothermal feed-in into district heating per district heating area (in columns)
        and time step (in index) in MW.
    solarthermal_energy_feedin_district_heating : :pandas:`pandas.DataFrame<DataFrame>`
        Solarthermal feed-in into district heating per district heating area (in
        columns) and time step (in index) in MW.

    """

    def __init__(self, **kwargs):
        self.renewables_curtailment = kwargs.get(
            "renewables_curtailment", pd.Series(dtype="float64")
        )

        self.storage_units_active_power = kwargs.get(
            "storage_units_active_power", pd.Series(dtype="float64")
        )
        self.dsm_active_power = kwargs.get(
            "dsm_active_power", pd.Series(dtype="float64")
        )
        self.electromobility_active_power = kwargs.get(
            "electromobility_active_power", pd.Series(dtype="float64")
        )
        self.heat_pump_decentral_active_power = kwargs.get(
            "heat_pump_decentral_active_power", pd.Series(dtype="float64")
        )
        self.heat_pump_central_active_power = kwargs.get(
            "heat_pump_central_active_power", pd.Series(dtype="float64")
        )

        self.geothermal_energy_feedin_district_heating = kwargs.get(
            "geothermal_energy_feedin_district_heating", pd.DataFrame(dtype="float64")
        )
        self.solarthermal_energy_feedin_district_heating = kwargs.get(
            "solarthermal_energy_feedin_district_heating", pd.DataFrame(dtype="float64")
        )

    @property
    def _attributes(self):
        return [
            "renewables_curtailment",
            "storage_units_active_power",
            "dsm_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "heat_pump_central_active_power",
            "geothermal_energy_feedin_district_heating",
            "solarthermal_energy_feedin_district_heating",
        ]

    def reduce_memory(self, attr_to_reduce=None, to_type="float32"):
        """
        Reduces size of time series data to save memory.

        Parameters
        -----------
        attr_to_reduce : list(str), optional
            List of attributes to reduce size for. Per default, all time series data
            are reduced.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        """
        if attr_to_reduce is None:
            attr_to_reduce = self._attributes
        for attr in attr_to_reduce:
            if isinstance(getattr(self, attr), pd.Series):
                setattr(
                    self,
                    attr,
                    getattr(self, attr).astype(to_type),
                )
            else:
                setattr(
                    self,
                    attr,
                    getattr(self, attr).apply(lambda _: _.astype(to_type)),
                )

    def to_csv(self, directory, reduce_memory=False, **kwargs):
        """
        Saves data in object to csv.

        Parameters
        ----------
        directory : str
            Directory to save data in.
        reduce_memory : bool, optional
            If True, size of time series data is reduced using
            :attr:`~.network.overlying_grid.OverlyingGrid.reduce_memory`.
            Optional parameters of
            :attr:`~.network.overlying_grid.OverlyingGrid.reduce_memory`
            can be passed as kwargs to this function. Default: False.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain arguments of
            :attr:`~.network.overlying_grid.OverlyingGrid.reduce_memory`.

        """
        if reduce_memory is True:
            self.reduce_memory(**kwargs)

        os.makedirs(directory, exist_ok=True)

        for attr in self._attributes:
            if not getattr(self, attr).empty:
                if isinstance(getattr(self, attr), pd.Series):
                    getattr(self, attr).to_frame(name=attr).to_csv(
                        os.path.join(directory, f"{attr}.csv")
                    )
                else:
                    getattr(self, attr).to_csv(os.path.join(directory, f"{attr}.csv"))

    def from_csv(
        self,
        data_path,
        dtype=None,
        from_zip_archive=False,
        **kwargs,
    ):
        """
        Restores data in object from csv files.

        Parameters
        ----------
        data_path : str
            Path to directory to obtain data from. Must be a directory or zip
            archive.
        dtype : str, optional
            Numerical data type for data to be loaded from csv. E.g. "float32".
            Default: None.
        from_zip_archive : bool, optional
            Set True if data is archived in a zip archive. Default: False.

        """

        # get all attributes
        attrs = self._attributes

        if from_zip_archive:
            # read from zip archive
            # setup ZipFile Class
            zip = ZipFile(data_path)

            # get all directories and files within zip archive
            files = zip.namelist()

            # add directory and .csv to files to match zip archive
            attrs = {v: f"overlying_grid/{v}.csv" for v in attrs}

        else:
            # read from directory
            # check files within the directory
            files = os.listdir(data_path)

            # add .csv to files to match directory structure
            attrs = {v: f"{v}.csv" for v in attrs}

        attrs_to_read = {k: v for k, v in attrs.items() if v in files}

        for attr, file in attrs_to_read.items():
            if from_zip_archive:
                # open zip file to make it readable for pandas
                with zip.open(file) as f:
                    df = pd.read_csv(f, index_col=0, parse_dates=True, dtype=dtype)
            else:
                path = os.path.join(data_path, file)
                df = pd.read_csv(path, index_col=0, parse_dates=True, dtype=dtype)

            # convert to Series if Series is expected
            if isinstance(getattr(self, attr), pd.Series):
                df = df.squeeze("columns")

            setattr(self, attr, df)

        if from_zip_archive:
            # make sure to destroy ZipFile Class to close any open connections
            zip.close()
