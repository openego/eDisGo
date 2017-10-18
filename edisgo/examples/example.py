from edisgo.grid.network import Network, Scenario, TimeSeries
import os
import sys
import pandas as pd

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

    timeseries = TimeSeries()
    scenario = Scenario(timeseries=timeseries,
                        power_flow='worst-case')
    costs = pd.DataFrame()
    faulty_grids = {'grid': [], 'msg': []}
    for dingo_grid in grids:
        logging.info('Grid expansion for {}'.format(dingo_grid))
        network = Network.import_from_ding0(
            os.path.join('data', dingo_grid),
            id='Test grid',
            scenario=scenario)
        try:
            # Do non-linear power flow analysis with PyPSA
            network.analyze()
            # Do grid reinforcement
            network.reinforce()
            costs_grouped = network.results.grid_expansion_costs.groupby(
                ['type']).sum()
            costs = costs.append(
                pd.DataFrame(costs_grouped.values,
                             columns=costs_grouped.columns,
                             index=[[network.id] * len(costs_grouped),
                                    costs_grouped.index]))
            logging.info('SUCCESS!')
        except Exception as e:
            faulty_grids['grid'].append(network.id)
            faulty_grids['msg'].append(e)
            logging.info('Something went wrong.')

    pd.DataFrame(faulty_grids).to_csv('faulty_grids.csv', index_label='grid')
    f = open('costs.csv', 'a')
    f.write('# units: length in km, total_costs in kEUR\n')
    costs.to_csv(f)
