from __future__ import annotations

import logging

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

logger = logging.getLogger(__name__)


class DSM:
    def __init__(self, **kwargs):
        self._edisgo_obj = kwargs.get("edisgo_obj")

    @property
    def egon_etrago_link(self):
        try:
            return self._egon_etrago_link
        except Exception:
            return pd.DataFrame()

    @egon_etrago_link.setter
    def egon_etrago_link(self, df: pd.DataFrame):
        self._egon_etrago_link = df

    @property
    def egon_etrago_link_timeseries(self):
        try:
            return self._egon_etrago_link_timeseries
        except Exception:
            return pd.DataFrame()

    @egon_etrago_link_timeseries.setter
    def egon_etrago_link_timeseries(self, df: pd.DataFrame):
        self._egon_etrago_link_timeseries = df

    @property
    def egon_etrago_store(self):
        try:
            return self._egon_etrago_store
        except Exception:
            return pd.DataFrame()

    @egon_etrago_store.setter
    def egon_etrago_store(self, df: pd.DataFrame):
        self._egon_etrago_store = df

    @property
    def egon_etrago_store_timeseries(self):
        try:
            return self._egon_etrago_store_timeseries
        except Exception:
            return pd.DataFrame()

    @egon_etrago_store_timeseries.setter
    def egon_etrago_store_timeseries(self, df: pd.DataFrame):
        self._egon_etrago_store_timeseries = df

    @property
    def p_min(self):
        try:
            return self._p_min
        except Exception:
            return pd.DataFrame()

    @p_min.setter
    def p_min(self, df: pd.DataFrame):
        self._p_min = df

    @property
    def p_max(self):
        try:
            return self._p_max
        except Exception:
            return pd.DataFrame()

    @p_max.setter
    def p_max(self, df: pd.DataFrame):
        self._p_max = df

    @property
    def e_min(self):
        try:
            return self._e_min
        except Exception:
            return pd.DataFrame()

    @e_min.setter
    def e_min(self, df: pd.DataFrame):
        self._e_min = df

    @property
    def e_max(self):
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
            "egon_etrago_link",
            "egon_etrago_link_timeseries",
            "egon_etrago_store",
            "egon_etrago_store_timeseries",
            "p_min",
            "p_max",
            "e_min",
            "e_max",
        ]

    def to_csv(self, directory: str | Path):
        if not isinstance(directory, Path):
            directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)

        for attr in self._attributes:
            if not getattr(self, attr).empty:
                getattr(self, attr).to_csv(directory / f"{attr}.csv")

    def from_csv(self, data_path: str | Path, from_zip_archive: bool = False):
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
