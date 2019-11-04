"""
This "test" runs curtailment for different curtailment requirements and
methods `voltage-based` and `feedin-proportional`.
It requires a ding0 network called ding0_grid_example.pkl in the same directory.

"""
import pandas as pd
import numpy as np

from edisgo import EDisGo


def get_generator_feedins(edisgo_grid):
    generator_feedins = {}
    for i in edisgo_grid.network.mv_grid.graph.nodes_by_attribute(
            'generator'):
        generator_feedins[i] = i.timeseries['p']
    for i in edisgo_grid.network.mv_grid.graph.nodes_by_attribute(
            'generator_agg'):
        generator_feedins[i] = i.timeseries['p']

    for lvgd in edisgo_grid.network.mv_grid.lv_grids:
        for i in lvgd.graph.nodes_by_attribute('generator'):
            generator_feedins[i] = i.timeseries['p']
        for i in lvgd.graph.nodes_by_attribute('generator_agg'):
            generator_feedins[i] = i.timeseries['p']

    return pd.DataFrame(generator_feedins)

timeindex = pd.date_range('2011-01-01 00:00', periods=8, freq='H')
feedin_pu = pd.DataFrame(
    {'solar': np.array([0.0, 0.0, 0.5, 0.5, 0.8, 0.8, 1.0, 1.0]),
     'wind': np.array([0.0, 1.0, 0.5, 1.0, 0.6, 1.0, 0.0, 1.0])},
    index=timeindex)
gen_dispatchable_df = pd.DataFrame(
    {'other': [0.0] * len(timeindex)},
    index=timeindex)
edisgo = EDisGo(
    ding0_grid="ding0_grid_example.pkl",
    generator_scenario='ego100',
    timeseries_generation_fluctuating=feedin_pu,
    timeseries_generation_dispatchable=gen_dispatchable_df,
    timeseries_load='demandlib',
    timeindex=timeindex)

for cm in ['voltage-based', 'feedin-proportional']:
    for curtailment_percent in [0, 50, 100]:

        # curtail
        feedin_gens = get_generator_feedins(edisgo)
        curtailment = feedin_gens.sum(axis=1) * curtailment_percent / 100.0
        edisgo.curtail(curtailment_timeseries=curtailment,
                       methodology=cm, voltage_threshold=0.0)
