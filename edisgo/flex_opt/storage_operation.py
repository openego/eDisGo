import pandas as pd
from math import sqrt


def fifty_fifty(storage, feedin_threshold=0.5):
    """
    Operational mode where the storage operation depends on actual power by
    generators. If cumulative generation exceeds 50% of nominal power, the
    storage is charged. Otherwise, the storage is discharged.
    The time series for active and reactive power are written into the
    storage.

    Parameters
    -----------
    storage : :class:`~.grid.components.Storage`
        Storage instance for which to generate time series.
    feedin_threshold : :obj:`float`
        Ratio of generation to installed power specifying when to charge or
        discharge the storage. If feed-in threshold is e.g. 0.5 the storage
        will be charged when the total generation is 50% of the installed
        generator capacity and discharged when it is below.

    """
    # determine generators cumulative apparent power output
    generators = storage.grid.graph.nodes_by_attribute('generator') + \
                 [generators for lv_grid in
                  storage.grid.lv_grids for generators in
                  lv_grid.graph.nodes_by_attribute('generator')]
    generators_p = pd.concat([_.timeseries['p'] for _ in generators],
                             axis=1).sum(axis=1).rename('p')
    generators_q = pd.concat([_.timeseries['q'] for _ in generators],
                             axis=1).sum(axis=1).rename('q')
    generation = pd.concat([generators_p, generators_q], axis=1)
    generation['s'] = generation.apply(
        lambda x: sqrt(x['p'] ** 2 + x['q'] ** 2), axis=1)

    generators_nom_capacity = sum([_.nominal_capacity for _ in generators])

    feedin_bool = generation['s'] > (feedin_threshold *
                                     generators_nom_capacity)
    feedin = feedin_bool.apply(
        lambda x: storage.nominal_capacity if x
        else -storage.nominal_capacity).rename('p').to_frame()
    feedin['q'] = 0
    storage.timeseries = feedin * storage.nominal_capacity