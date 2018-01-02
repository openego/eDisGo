from edisgo.grid.network import Network, Scenario, TimeSeries, Results
import os
import sys
import pandas as pd
from edisgo.flex_opt.exceptions import MaximumIterationError

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
#     # Import generators
#     network.import_generators()
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

    technologies = ['wind', 'solar']
    timeseries = TimeSeries()
    scenario = Scenario(timeseries=timeseries,
                        power_flow='worst-case')
    costs_before_geno_import = pd.DataFrame()
    faulty_grids_before_geno_import = {'grid': [], 'msg': []}
    costs = pd.DataFrame()
    faulty_grids = {'grid': [], 'msg': []}
    for dingo_grid in grids:
        logging.info('Grid expansion for {}'.format(dingo_grid))
        network = Network.import_from_ding0(
            os.path.join('data', dingo_grid),
            id='Test grid',
            scenario=scenario)
        try:
            # Calculate grid expansion costs before generator import

            logging.info('Grid expansion before generator import.')
            before_geno_import = True
            # Do non-linear power flow analysis with PyPSA
            network.analyze()
            # Do grid reinforcement
            network.reinforce()
            # Get costs
            costs_grouped = network.results.grid_expansion_costs.groupby(
                ['type']).sum()
            costs_before_geno_import = costs_before_geno_import.append(
                pd.DataFrame(costs_grouped.values,
                             columns=costs_grouped.columns,
                             index=[[network.id] * len(costs_grouped),
                                    costs_grouped.index]))

            # Clear results
            network.results = Results()
            network.pypsa = None

            # Calculate grid expansion costs after generator import

            logging.info('Grid expansion after generator import.')
            before_geno_import = False
            # Import generators
            network.import_generators(types=technologies)
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
        except MaximumIterationError:
            if before_geno_import:
                faulty_grids_before_geno_import['grid'].append(network.id)
                faulty_grids_before_geno_import['msg'].append(
                    str(network.results.unresolved_issues))
            else:
                faulty_grids['grid'].append(network.id)
                faulty_grids['msg'].append(
                    str(network.results.unresolved_issues))
            logging.info('Unresolved issues left after grid expansion.')
        except Exception as e:
            if before_geno_import:
                faulty_grids_before_geno_import['grid'].append(network.id)
                faulty_grids_before_geno_import['msg'].append(e)
            else:
                faulty_grids['grid'].append(network.id)
                faulty_grids['msg'].append(
                    str(network.results.unresolved_issues))
            logging.info('Something went wrong.')

    pd.DataFrame(faulty_grids_before_geno_import).to_csv(
        'faulty_grids_before_geno_import.csv', index=False)
    f = open('costs_before_geno_import.csv', 'a')
    f.write('# units: length in km, total_costs in kEUR\n')
    costs_before_geno_import.to_csv(f)
    f.close()

    pd.DataFrame(faulty_grids).to_csv('faulty_grids.csv', index=False)
    f = open('costs.csv', 'a')
    f.write('# units: length in km, total_costs in kEUR\n')
    costs.to_csv(f)
    f.close()
