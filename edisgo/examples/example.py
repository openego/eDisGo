from edisgo.grid.network import Network, Scenario, TimeSeries, Results, \
    ETraGoSpecs
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

    costs_before_geno_import = pd.DataFrame()
    faulty_grids_before_geno_import = {'grid': [], 'msg': []}
    costs = pd.DataFrame()
    faulty_grids = {'grid': [], 'msg': []}
    for dingo_grid in grids:
        # # worst-case scenario
        # scenario = Scenario(power_flow='worst-case')

        # # time-range non-empty tuple
        # power_flow = (date(2017, 10, 10), date(2017, 10, 13))
        # timeindex = pd.date_range(power_flow[0], power_flow[1], freq='H')
        # etrago_specs = ETraGoSpecs(
        #     dispatch=pd.DataFrame({'biomass': [1] * len(timeindex),
        #                            'solar': [1] * len(timeindex),
        #                            'gas': [1] * len(timeindex),
        #                            'wind': [1] * len(timeindex)},
        # index=timeindex),
        #     capacity=pd.DataFrame({'biomass': 1846.5,
        #                            'solar': 7131,
        #                            'gas': 1564,
        #                            'wind': 10}, index=['cap']),
        #     load=pd.DataFrame({'residential': [1] * len(timeindex),
        #                        'retail': [1] * len(timeindex),
        #                        'industrial': [1] * len(timeindex),
        #                        'agricultural': [1] * len(timeindex)},
        #                       index=timeindex),
        #     annual_load=pd.DataFrame({'residential': 1,
        #                        'retail': 1,
        #                        'industrial': 1,
        #                        'agricultural': 1},
        #                       index=timeindex)
        # )
        # scenario = Scenario(etrago_specs=etrago_specs, power_flow=(),
        #                     scenario_name='NEP 2035')

        # time-range non-empty tuple
        mv_grid_id = dingo_grid.split('_')[-1].split('.')[0]
        scenario = Scenario(power_flow=(), mv_grid_id=mv_grid_id,
                            scenario_name='NEP 2035')

        logging.info('Grid expansion for {}'.format(dingo_grid))
        network = Network.import_from_ding0(
            os.path.join('data', dingo_grid),
            id='Test grid',
            scenario=scenario)

        # Do non-linear power flow analysis with PyPSA
        network.analyze()
        # Do grid reinforcement
        try:
            # Calculate grid expansion costs before generator import

            # Do non-linear power flow analysis with PyPSA
            network.analyze()
            # Do grid reinforcement
            network.reinforce()
            # Get costs
            costs_grouped = network.results.grid_expansion_costs.groupby(
                ['type']).sum()
            costs_before_geno_import = costs.append(
                pd.DataFrame(costs_grouped.values,
                             columns=costs_grouped.columns,
                             index=[[network.id] * len(costs_grouped),
                                    costs_grouped.index]))
            if network.results.unresolved_issues:
                faulty_grids_before_geno_import['grid'].append(network.id)
                faulty_grids_before_geno_import['msg'].append(
                    str(network.results.unresolved_issues))
            # Clear results
            network.results = Results()
            network.pypsa = None

            # Calculate grid expansion costs after generator import

            logging.info('Grid expansion after generator import.')
            # Import generators
            network.import_generators()
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
            if network.results.unresolved_issues:
                faulty_grids['grid'].append(network.id)
                faulty_grids['msg'].append(
                    str(network.results.unresolved_issues))
                logging.info('Unresolved issues left after grid expansion.')
            else:
                logging.info('SUCCESS!')
        except Exception as e:
            faulty_grids['grid'].append(network.id)
            faulty_grids['msg'].append(e)
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
