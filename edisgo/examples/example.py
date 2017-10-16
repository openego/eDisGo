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

    timeindex = pd.date_range('12/4/2011', periods=1, freq='H')
    etrago_specs = ETraGoSpecs(
        dispatch={'all_other': pd.DataFrame({'p': 1}, index=timeindex)})

    # scenario = Scenario(power_flow='worst-case')
    scenario = Scenario(etrago_specs=etrago_specs,
                        power_flow=(date(2017, 10, 10), date(2017, 10, 13)))
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
