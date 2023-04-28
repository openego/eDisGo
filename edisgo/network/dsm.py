from __future__ import annotations

import logging

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DSM:
    """
    Data container for demand side management potential data.

    """

    def __init__(self, **kwargs):
        pass

    @property
    def p_min(self):
        """
        Maximum load decrease in MW.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Maximum load decrease in MW. Index of the dataframe is a time index and
            column names are names of DSM loads as in
            :attr:`~.network.topology.Topology.loads_df`.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Maximum load decrease in MW. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._p_min
        except Exception:
            return pd.DataFrame()

    @p_min.setter
    def p_min(self, df: pd.DataFrame):
        self._p_min = df

    @property
    def p_max(self):
        """
        Maximum load increase in MW.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Maximum load increase in MW. Index of the dataframe is a time index and
            column names are names of DSM loads as in
            :attr:`~.network.topology.Topology.loads_df`.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Maximum load decrease in MW. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._p_max
        except Exception:
            return pd.DataFrame()

    @p_max.setter
    def p_max(self, df: pd.DataFrame):
        self._p_max = df

    @property
    def e_min(self):
        """
        Maximum energy preponing in MWh.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Maximum energy preponing in MWh. Index of the dataframe is a time index and
            column names are names of DSM loads as in
            :attr:`~.network.topology.Topology.loads_df`.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Maximum energy preponing in MWh. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._e_min
        except Exception:
            return pd.DataFrame()

    @e_min.setter
    def e_min(self, df: pd.DataFrame):
        self._e_min = df

    @property
    def e_max(self):
        """
        Maximum energy postponing in MWh.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Maximum energy postponing in MWh. Index of the dataframe is a time index and
            column names are names of DSM loads as in
            :attr:`~.network.topology.Topology.loads_df`.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Maximum energy postponing in MWh. For more information on the dataframe see
            input parameter `df`.

        """
        try:
            return self._e_max
        except Exception:
            return pd.DataFrame()

    @e_max.setter
    def e_max(self, df: pd.DataFrame):
        self._e_max = df

    @property
    def _attributes(self):
        return [
            "p_min",
            "p_max",
            "e_min",
            "e_max",
        ]

    def reduce_memory(
        self,
        attr_to_reduce=None,
        to_type="float32",
    ):
        """
        Reduces size of dataframes to save memory.

        See :attr:`~.edisgo.EDisGo.reduce_memory` for more information.

        Parameters
        -----------
        attr_to_reduce : list(str), optional
            List of attributes to reduce size for. Per default, all active
            and reactive power time series of generators, loads, and storage units
            are reduced.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        """
        if attr_to_reduce is None:
            attr_to_reduce = self._attributes
        for attr in attr_to_reduce:
            setattr(
                self,
                attr,
                getattr(self, attr).apply(lambda _: _.astype(to_type)),
            )

    def to_csv(self, directory: str | Path, reduce_memory=False, **kwargs):
        """
        Exports DSM data to csv files.

        The following attributes are exported:

        * 'p_min' : Attribute :py:attr:`~p_min` is saved to `p_min.csv`.
        * 'p_max' : Attribute :py:attr:`~p_max` is saved to `p_max.csv`.
        * 'e_min' : Attribute :py:attr:`~e_min` is saved to `e_min.csv`.
        * 'e_max' : Attribute :py:attr:`~e_max` is saved to `e_max.csv`.

        Parameters
        ----------
        directory : str
            Path to save DSM data to.
        reduce_memory : bool, optional
            If True, size of dataframes is reduced using
            :attr:`~.network.dsm.DSM.reduce_memory`.
            Optional parameters of :attr:`~.network.dsm.DSM.reduce_memory`
            can be passed as kwargs to this function. Default: False.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain arguments of
            :attr:`~.network.dsm.DSM.reduce_memory`.

        """
        if reduce_memory is True:
            self.reduce_memory(**kwargs)

        if not isinstance(directory, Path):
            directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)

        for attr in self._attributes:
            if not getattr(self, attr).empty:
                getattr(self, attr).to_csv(directory / f"{attr}.csv")

    def from_csv(self, data_path: str | Path, from_zip_archive: bool = False):
        """
        Restores DSM data from csv files.

        Parameters
        ----------
        data_path : str
            Path to DSM csv files or zip archive.
        from_zip_archive : bool
            Set to True if data is archived in a zip archive. Default: False.

        """
        if not isinstance(data_path, Path):
            data_path = Path(data_path)

        attrs = self._attributes

        if from_zip_archive:
            # read from zip archive
            # setup ZipFile Class
            zip = ZipFile(data_path)

            # get all directories and files within zip archive
            files = zip.namelist()

            # add directory and .csv to files to match zip archive
            attrs = {v: f"dsm/{v}.csv" for v in attrs}

        else:
            # read from directory
            # check files within the directory
            files = [f.parts[-1] for f in data_path.iterdir()]

            # add .csv to files to match directory structure
            attrs = {v: f"{v}.csv" for v in attrs}

        attrs_to_read = {k: v for k, v in attrs.items() if v in files}

        for attr, file in attrs_to_read.items():
            if from_zip_archive:
                # open zip file to make it readable for pandas
                with zip.open(file) as f:
                    df = pd.read_csv(f, index_col=0, parse_dates=True)
            else:
                path = data_path / file
                df = pd.read_csv(path, index_col=0, parse_dates=True)

            setattr(self, attr, df)

        if from_zip_archive:
            # make sure to destroy ZipFile Class to close any open connections
            zip.close()

    def check_integrity(self):
        """
        Check data integrity.

        Checks for duplicated and missing labels as well as implausible values.

        """
        # check for duplicate columns
        duplicated_labels = []
        for ts in self._attributes:
            df = getattr(self, ts)
            if any(df.columns.duplicated()):
                duplicated_labels.append(df.columns[df.columns.duplicated()].values)

        if len(duplicated_labels) > 0:
            duplicates = set(
                np.concatenate([list.tolist() for list in duplicated_labels])
            )
            logger.warning(
                f"DSM timeseries contain the following duplicates: {duplicates}."
            )

        # check that all profiles exist for the same loads
        columns = set(
            np.concatenate([getattr(self, _).columns for _ in self._attributes])
        )
        for ts in self._attributes:
            df = getattr(self, ts)
            missing_entries = [_ for _ in columns if _ not in df.columns]
            if len(missing_entries) > 0:
                logger.warning(
                    f"DSM timeseries {ts} is missing the following "
                    f"entries: {missing_entries}."
                )

        # check for implausible values
        if not (self.p_min <= 0.0).all().all():
            logger.warning(
                "DSM timeseries p_min contains values larger than zero, which is "
                "not allowed."
            )
        if not (self.e_min <= 0.0).all().all():
            logger.warning(
                "DSM timeseries e_min contains values larger than zero, which is "
                "not allowed."
            )
        if not (self.p_max >= 0.0).all().all():
            logger.warning(
                "DSM timeseries p_max contains values smaller than zero, which is "
                "not allowed."
            )
        if not (self.e_max >= 0.0).all().all():
            logger.warning(
                "DSM timeseries e_max contains values smaller than zero, which is "
                "not allowed."
            )
