from edisgo.grid.network import Network, Scenario, TimeSeries, ETraGoSpecs
import os
import sys
import pandas as pd
from datetime import date

import logging
logging.basicConfig(filename='example.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)

# import pickle
# import_network = True
# if import_network:
#     network = Network.import_from_ding0(
#         os.path.join('data', 'ding0_grids__76.pkl'),
#         id='Test grid',
#         scenario=scenario
#     )
#     # Do non-linear power flow analysis with PyPSA
#     network.analyze()
#     #network.pypsa.export_to_csv_folder('data/pypsa_export')
#     #network.pypsa = None
#     #pickle.dump(network, open('test_network.pkl', 'wb'))
# else:
#     network = pickle.load(open('test_results_neu.pkl', 'rb'))

if __name__ == '__main__':
    grids = []
    for file in os.listdir(os.path.join(sys.path[0], "data")):
        if file.endswith(".pkl"):
            grids.append(file)

    # # worst-case scenario
    # scenario = Scenario(power_flow='worst-case')

    # time-range non-empty tuple
    power_flow = (date(2017, 10, 10), date(2017, 10, 13))
    timeindex = pd.date_range(power_flow[0], power_flow[1], freq='H')
    etrago_specs = ETraGoSpecs(
        dispatch=pd.DataFrame({'biomass': [1] * len(timeindex),
                               'solar': [1] * len(timeindex),
                               'gas': [1] * len(timeindex),
                               'wind': [1] * len(timeindex)}, index=timeindex),
        capacity=pd.DataFrame({'biomass': 1846.5,
                               'solar': 7131,
                               'gas': 1564,
                               'wind': 10}, index=['cap']),
        load=pd.DataFrame({'residential': [1] * len(timeindex),
                           'retail': [1] * len(timeindex),
                           'industrial': [1] * len(timeindex),
                           'agricultural': [1] * len(timeindex)},
                          index=timeindex),
        annual_load=pd.DataFrame({'residential': 1,
                           'retail': 1,
                           'industrial': 1,
                           'agricultural': 1},
                          index=timeindex)
    )
    scenario = Scenario(etrago_specs=etrago_specs, power_flow=())

    costs = pd.DataFrame()
    faulty_grids = []
    for dingo_grid in grids:
        logging.info('Grid expansion for {}'.format(dingo_grid))
        network = Network.import_from_ding0(
            os.path.join('data', dingo_grid),
            id='Test grid',
            scenario=scenario)
        # Do non-linear power flow analysis with PyPSA
        network.analyze()
        # Do grid reinforcement
        try:
            network.reinforce()
            costs_grouped = network.results.grid_expansion_costs.groupby(
                ['type']).sum()
            costs = costs.append(
                pd.DataFrame(costs_grouped.values,
                             columns=costs_grouped.columns,
                             index=[[network.id] * len(costs_grouped),
                             costs_grouped.index]))
            logging.info('SUCCESS!')
        except:
            faulty_grids.append(dingo_grid)
            logging.info('Something went wrong.')

    costs.to_csv('costs.csv')


gens = network.mv_grid.graph.nodes_by_attribute('generator')
for lv_grid in network.mv_grid.lv_grids:
    gens.extend(lv_grid.graph.nodes_by_attribute('generator'))
sum = 0
for gen in gens:
    sum += gen.timeseries['p']
typ = []
for gen in gens:
    typ.append(gen.type)
print(set(typ))
