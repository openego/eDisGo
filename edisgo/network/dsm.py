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
    def grid_time_series(self):
        return self._grid_time_series

    @grid_time_series.setter
    def grid_time_series(self, df: pd.DataFrame):
        self._grid_time_series = df
