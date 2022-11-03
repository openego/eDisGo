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
    def p_min(self):
        return self._p_min

    @p_min.setter
    def p_min(self, df: pd.DataFrame):
        self._p_min = df

    @property
    def p_max(self):
        return self._p_max

    @p_max.setter
    def p_max(self, df: pd.DataFrame):
        self._p_max = df

    @property
    def e_min(self):
        return self._e_min

    @e_min.setter
    def e_min(self, df: pd.DataFrame):
        self._e_min = df

    @property
    def e_max(self):
        return self._e_max

    @e_max.setter
    def e_max(self, df: pd.DataFrame):
        self._e_max = df
