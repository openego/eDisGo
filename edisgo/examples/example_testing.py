# -*- coding: utf-8 -*-

"""
This example shows the general usage of eDisGo. Grid expansion costs for
distribution grids generated with ding0 are calculated assuming renewable
and conventional power plant capacities as stated in the scenario framework of
the German Grid Development Plan (Netzentwicklungsplan) for the year 2035
and conducting a worst-case analysis.

As the grids generated with ding0 should represent current stable grids but
have in some cases stability issues, grid expansion is first conducted before
connecting future generators in order to obtain stable grids. Final grid
expansion costs in the DataFrame 'costs' only contain grid expansion costs
of measures conducted after future generators are connected to the stable
grids.

"""

import os
import pandas as pd

from edisgo import EDisGo

import logging
logging.basicConfig(filename='example.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)

grids = ['294']

for dingo_grid in grids:

    logging.info('Grid expansion for {}'.format(dingo_grid))

    # set up worst-case scenario
    edisgo = EDisGo(ding0_grid=os.path.join(
        'data', 'ding0_grids__{}_with_w_id.pkl'.format(dingo_grid)),
                    worst_case_analysis='worst-case-feedin')

    # timeindex = pd.date_range('1/1/1971', periods=3, freq='H')
    # conv_dispatch = pd.DataFrame({'biomass': [1] * len(timeindex),
    #                               'coal': [1] * len(timeindex),
    #                               'other': [1] * len(timeindex)},
    #                              index=timeindex)
    # ren_dispatch = pd.DataFrame({('solar', 1124078): [0.2, 0.8, 0.5],
    #                              ('solar', 1125078): [0.2, 0.8, 0.5],
    #                              ('solar', 1124077): [0.2, 0.8, 0.5],
    #                              ('wind', 1124078): [0.3, 0.8, 0.5],
    #                              ('wind', 1125078): [0.3, 0.8, 0.5],
    #                              ('wind', 1124077): [0.3, 0.8, 0.5]
    #                              },
    #                             index=timeindex)
    # load = pd.DataFrame({'residential': [0.00002] * len(timeindex),
    #                      'retail': [0.00003] * len(timeindex),
    #                      'industrial': [0.00002] * len(timeindex),
    #                      'agricultural': [0.00003] * len(timeindex)},
    #                     index=timeindex)
    #
    # edisgo = EDisGo(ding0_grid=os.path.join(
    #     'data', 'ding0_grids__{}.pkl'.format(dingo_grid)),
    #                 timeseries_generation_fluctuating=ren_dispatch,
    #                 timeseries_generation_dispatchable=conv_dispatch,
    #                 timeseries_load=load)

    from pypsa import Network as PyPSANetwork
    edisgo.network.pypsa = PyPSANetwork()
    edisgo.network.pypsa.import_from_csv_folder(
        'pypsa_network_294_feedin_case')
    edisgo.network.pypsa.edisgo_mode = None

    # edisgo.import_generators(generator_scenario='nep2035')

    from edisgo.flex_opt import storage_positioning
    storage_parameters = {
        'soc_initial': 0,
        'efficiency_in': 1.0,
        'efficiency_out': 1.0,
        'standing_loss': 0.0}
    storage_positioning.one_storage_per_feeder(
        edisgo, storage_parameters=storage_parameters)

    # edisgo.reinforce()
    #
    # edisgo.network.results.grid_expansion_costs.to_csv('costs.csv')
