"""
This "test" runs storage for different curtailment requirements and
methods `voltage-based` and `feedin-proportional`.
It requires a ding0 network called ding0_grid_example.pkl in the same directory.

"""

import pandas as pd
from edisgo import EDisGo

timeindex = pd.date_range('2011-05-01 00:00', periods=24, freq='H')
ts_gens_dispatchable = pd.DataFrame({'other': 1}, index=timeindex)

edisgo = EDisGo(ding0_grid='ding0_grid_example.pkl',
                generator_scenario='ego100',
                timeseries_generation_fluctuating='oedb',
                timeseries_generation_dispatchable=ts_gens_dispatchable,
                timeseries_load='demandlib',
                timeindex=timeindex)

storage_ts = pd.read_csv('storage.csv', index_col=[0], parse_dates=True)
edisgo.integrate_storage(timeseries=storage_ts.p,
                         position='distribute_storages_mv',
                         timeseries_reactive_power=storage_ts.q)
storages = edisgo.network.results.storages
