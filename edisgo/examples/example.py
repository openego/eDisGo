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
import sys
import pandas as pd

from edisgo.grid.network import EDisGo, Results
from edisgo.flex_opt.exceptions import MaximumIterationError

import logging
logging.basicConfig(filename='example.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':

    # get filenames of all pickled ding0 grids in directory
    # 'edisgo/examples/data/'
    grids = []
    for file in os.listdir(os.path.join(sys.path[0], "data")):
        if file.endswith(".pkl"):
            grids.append(file)

    # set scenario to define future power plant capacities
    scenario = 'nep2035'

    # initialize containers that will hold grid expansion costs and, in the
    # case of any errors during the run, the error messages
    costs_before_geno_import = pd.DataFrame()
    faulty_grids_before_geno_import = {'grid': [], 'msg': []}
    costs = pd.DataFrame()
    faulty_grids = {'grid': [], 'msg': []}

    for dingo_grid in grids:

        logging.info('Grid expansion for {}'.format(dingo_grid))

        # set up worst-case scenario
        edisgo = EDisGo(ding0_grid=os.path.join('data', dingo_grid),
                        worst_case_analysis='worst-case-feedin')

        try:
            # Calculate grid expansion costs before generator import
            logging.info('Grid expansion before generator import.')
            before_geno_import = True
            # Do non-linear power flow analysis with PyPSA
            edisgo.analyze()
            # Do grid reinforcement
            edisgo.reinforce()
            # Get costs
            costs_grouped = \
                edisgo.network.results.grid_expansion_costs.groupby(
                    ['type']).sum()
            costs_before_geno_import = costs_before_geno_import.append(
                pd.DataFrame(costs_grouped.values,
                             columns=costs_grouped.columns,
                             index=[[edisgo.network.id] * len(costs_grouped),
                                    costs_grouped.index]))

            # Clear results
            edisgo.network.results = Results()
            edisgo.network.pypsa = None

            # Calculate grid expansion costs after generator import
            logging.info('Grid expansion after generator import.')
            before_geno_import = False
            # Import generators
            edisgo.import_generators(generator_scenario=scenario)
            # Do non-linear power flow analysis with PyPSA
            edisgo.analyze()
            # Do grid reinforcement
            edisgo.reinforce()
            costs_grouped = \
                edisgo.network.results.grid_expansion_costs.groupby(
                    ['type']).sum()
            costs = costs.append(
                pd.DataFrame(costs_grouped.values,
                             columns=costs_grouped.columns,
                             index=[[edisgo.network.id] * len(costs_grouped),
                                    costs_grouped.index]))
            logging.info('SUCCESS!')

        except MaximumIterationError:
            if before_geno_import:
                faulty_grids_before_geno_import['grid'].append(
                    edisgo.network.id)
                faulty_grids_before_geno_import['msg'].append(
                    str(edisgo.network.results.unresolved_issues))
            else:
                faulty_grids['grid'].append(edisgo.network.id)
                faulty_grids['msg'].append(
                    str(edisgo.network.results.unresolved_issues))
            logging.info('Unresolved issues left after grid expansion.')

        except Exception as e:
            if before_geno_import:
                faulty_grids_before_geno_import['grid'].append(
                    edisgo.network.id)
                faulty_grids_before_geno_import['msg'].append(repr(e))
            else:
                faulty_grids['grid'].append(edisgo.network.id)
                faulty_grids['msg'].append(repr(e))
            logging.info('Something went wrong.')

    # write costs and error messages to csv files
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
