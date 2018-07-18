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

    # # set up worst-case scenario
    # edisgo = EDisGo(ding0_grid=os.path.join(
    #     'data', 'ding0_grids__{}.pkl'.format(dingo_grid)),
    #                 worst_case_analysis='worst-case-feedin')
    # timeindex = pd.date_range('1/1/1970', periods=2, freq='H')
    # curtailment = pd.DataFrame({('solar', 1125096): [0.2, 0.8],
    #                              ('solar', 1126096): [0.2, 0.8],
    #                              ('solar', 1126095): [0.2, 0.8],
    #                              ('wind', 1125096): [0.3, 0.8],
    #                              ('wind', 1126096): [0.3, 0.8],
    #                              ('wind', 1126095): [0.3, 0.8]
    #                              },
    #                             index=timeindex)
    # print(edisgo.network.mv_grid.weather_cells)
    # edisgo.curtail(curtailment_methodology='curtail_all',
    #                timeseries_curtailment=curtailment)
    # edisgo.analyze()

    timeindex = pd.date_range('1/1/1971', periods=3, freq='H')
    conv_dispatch = pd.DataFrame({'biomass': [1] * len(timeindex),
                                  'coal': [1] * len(timeindex),
                                  'other': [1] * len(timeindex)},
                                 index=timeindex)
    ren_dispatch = pd.DataFrame({('solar', 1125096): [0.2, 0.8, 0.5],
                                 ('solar', 1126096): [0.2, 0.8, 0.5],
                                 ('solar', 1126095): [0.2, 0.8, 0.5],
                                 ('wind', 1125096): [0.3, 0.8, 0.5],
                                 ('wind', 1126096): [0.3, 0.8, 0.5],
                                 ('wind', 1126095): [0.3, 0.8, 0.5]
                                 },
                                index=timeindex)
    load = pd.DataFrame({'residential': [0.00021372] * len(timeindex),
                         'retail': [0.0002404] * len(timeindex),
                         'industrial': [0.000132] * len(timeindex),
                         'agricultural': [0.00024036] * len(timeindex)},
                        index=timeindex)

    edisgo = EDisGo(ding0_grid=os.path.join(
        'data', 'ding0_grids__{}.pkl'.format(dingo_grid)),
                    timeseries_generation_fluctuating=ren_dispatch,
                    timeseries_generation_dispatchable=conv_dispatch,
                    timeseries_load=load)

    # curtailment = pd.DataFrame({'solar': [100, 200, 100],
    #                             'wind': [200, 300, 400]},
    #                             index=timeindex)
    # edisgo.curtail(curtailment_methodology='curtail_all',
    #                timeseries_curtailment=curtailment,
    #                voltage_threshold=1.02)

    # # export to pypsa
    # edisgo.analyze()
    # edisgo.network.pypsa.export_to_csv_folder('before_storage_integration')

    # # import from pypsa
    # from pypsa import Network as PyPSANetwork
    # edisgo.network.pypsa = PyPSANetwork()
    # edisgo.network.pypsa.import_from_csv_folder(
    #     'pypsa_network_294_feedin_case')
    # edisgo.network.pypsa.edisgo_mode = None

    # edisgo.import_generators(generator_scenario='nep2035')

    from edisgo.flex_opt import storage_positioning
    # # storage_timeseries = pd.DataFrame({'p': [-200, -300, 400],
    # #                                    'q': [0.3, 0.8, 0.5]},
    # #                                   index=timeindex)
    lv_station = (list(edisgo.network.mv_grid.lv_grids)[0]).station
    # line = lv_station.mv_grid.graph.line_from_nodes(
    #     lv_station.mv_grid.graph.neighbors(lv_station)[0],
    #     lv_station)
    # print(lv_station)
    # print(line)
    # storage_timeseries = pd.Series([-20000, -3000, 4000], index=timeindex)
    # storage_timeseries_q = pd.Series([10, -3000, 4000], index=timeindex)
    storage_timeseries = pd.DataFrame({'p': [-2000, -3000, 4000],
                                       'q': [0, 0, 0]},
                                      index=timeindex)
    #
    # storage_power = 1000
    storage_positioning.one_storage_per_feeder(
        edisgo, storage_timeseries=storage_timeseries)

    # storage_parameters = {'nominal_power': 400}
    # edisgo.integrate_storage(timeseries='fifty-fifty',
    #                          parameters=storage_parameters,
    #                          position='hvmv_substation_busbar',#lv_station,
    #                          voltage_level='mv',
    #                          timeseries_reactive_power=storage_timeseries_q)
    # edisgo.analyze()
    # edisgo.network.pypsa.export_to_csv_folder('test_none_parameters')
    #
    # edisgo.network.results.grid_expansion_costs.to_csv('costs.csv')
