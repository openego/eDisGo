from edisgo.grid.network import Network, Scenario, TimeSeries, Results, \
    ETraGoSpecs
import os
import sys
import pandas as pd
from datetime import date
from edisgo.flex_opt.exceptions import MaximumIterationError

import logging
logging.basicConfig(filename='example.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':

    grids = []
    # for file in os.listdir(os.path.join(sys.path[0], "data")):
    #     if file.endswith(".pkl"):
    #         grids.append(file)

    scenario_name = 'NEP 2035'
    technologies = ['wind', 'solar']

    costs_before_geno_import = pd.DataFrame()
    faulty_grids_before_geno_import = {'grid': [], 'msg': []}
    costs = pd.DataFrame()
    faulty_grids = {'grid': [], 'msg': []}
    for dingo_grid in grids:
        mv_grid_id = dingo_grid  # dingo_grid.split('_')[-1].split('.')[0]
        # # worst-case scenario
        # scenario = Scenario(power_flow='worst-case', mv_grid_id=mv_grid_id)

        # scenario with etrago specs
        power_flow = (date(2017, 10, 10), date(2017, 10, 13))
        timeindex = pd.date_range(power_flow[0], power_flow[1], freq='H')
        etrago_specs = ETraGoSpecs(
            conv_dispatch=pd.DataFrame({'biomass': [1] * len(timeindex),
                                        'coal': [1] * len(timeindex),
                                        'gas': [1] * len(timeindex)},
                                       index=timeindex),
            ren_dispatch=pd.DataFrame({'0': [0.2] * len(timeindex),
                                       '1': [0.3] * len(timeindex),
                                       '2': [0.4] * len(timeindex),
                                       '3': [0.5] * len(timeindex)},
                                      index=timeindex),
            renewables=pd.DataFrame({
                'name': ['wind', 'wind', 'solar', 'solar'],
                'w_id': ['1', '2', '1', '2'],
                'ren_id': ['0', '1', '2', '3']}, index=[0, 1, 2, 3]),
            battery_capacity=100,
            battery_active_power=pd.Series(data=[50, 20, -10, 20])
        )
        scenario = Scenario(etrago_specs=etrago_specs, power_flow=(),
                            scenario_name=scenario_name, mv_grid_id=mv_grid_id)

        # scenario with time series
        # scenario = Scenario(
            # power_flow=(date(2011, 10, 10), date(2011, 10, 13)),
            # mv_grid_id=mv_grid_id,
            # scenario_name=['NEP 2035', 'Status Quo'])
        # scenario = Scenario(power_flow=(), mv_grid_id=mv_grid_id,
        #                     scenario_name='NEP 2035')

        logging.info('Grid expansion for {}'.format(dingo_grid))
        network = Network.import_from_ding0(
            os.path.join('data', dingo_grid),
            id='Test grid',
            scenario=scenario)

        # Do grid reinforcement
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
                faulty_grids_before_geno_import['msg'].append(repr(e))
            else:
                faulty_grids['grid'].append(network.id)
                faulty_grids['msg'].append(repr(e))
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
