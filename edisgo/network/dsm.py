from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DSM:
    def __init__(self, **kwargs):
        self._edisgo_obj = kwargs.get("edisgo_obj")

    @property
    def egon_etrago_link(self):
        return self._egon_etrago_link

    @egon_etrago_link.setter
    def egon_etrago_link(self, df: pd.DataFrame):
        self._egon_etrago_link = df

    @property
    def egon_etrago_link_timeseries(self):
        return self._egon_etrago_link_timeseries

    @egon_etrago_link_timeseries.setter
    def egon_etrago_link_timeseries(self, df: pd.DataFrame):
        self._egon_etrago_link_timeseries = df

    @property
    def dsm_time_series(self):
        return self._dsm_time_series

    @dsm_time_series.setter
    def dsm_time_series(self, df: pd.DataFrame):
        self._dsm_time_series = df

    @property
    def egon_etrago_store(self):
        return self._egon_etrago_store

    @egon_etrago_store.setter
    def egon_etrago_store(self, df: pd.DataFrame):
        self._egon_etrago_store = df

    @property
    def egon_etrago_store_timeseries(self):
        return self._egon_etrago_store_timeseries

    @egon_etrago_store_timeseries.setter
    def egon_etrago_store_timeseries(self, df: pd.DataFrame):
        self._egon_etrago_store_timeseries = df

    @property
    def store_time_series(self):
        return self._store_time_series

    @store_time_series.setter
    def store_time_series(self, df: pd.DataFrame):
        self._store_time_series = df
