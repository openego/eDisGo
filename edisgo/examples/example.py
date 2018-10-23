# -*- coding: utf-8 -*-

"""
This example shows the general usage of eDisGo. Grid expansion costs for
distribution grids generated with ding0 are calculated assuming renewable
and conventional power plant capacities as stated in the scenario framework of
the German Grid Development Plan 2015 (Netzentwicklungsplan) for the year 2035
(scenario B2). To determine grid expansion needs worst-case scenarios (heavy
load flow and reverse power flow) used in conventional grid expansion planning
are set up.

The example assumes you have ding0 grids in the current working directory. If
you need more information on how to get ding0 grids see the ding0 documentation
or the Quickstart section of the eDisGo documentation.

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

from edisgo import EDisGo
from edisgo.grid.network import Results
from edisgo.flex_opt.exceptions import MaximumIterationError

import logging
logging.basicConfig(filename='example.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':

    # get filenames of all pickled ding0 grids in current working directory
    grids = []
    for file in os.listdir(sys.path[0]):
        if file.endswith(".pkl"):
            grids.append(file)

    # set scenario to define future power plant capacities
    scenario = 'nep2035'

    # initialize containers that will hold any error messages
    faulty_grids_before_geno_import = {'grid': [], 'msg': []}
    faulty_grids = {'grid': [], 'msg': []}

    for dingo_grid in grids:

        logging.info('Grid expansion for {}'.format(dingo_grid))

        # set up worst-case scenario
        edisgo = EDisGo(ding0_grid=os.path.join('data', dingo_grid),
                        worst_case_analysis='worst-case')

        try:
            # Calculate grid expansion costs before generator import
            logging.info('Grid expansion before generator import.')
            before_geno_import = True

            # overwrite config parameters for allowed voltage deviations in
            # initial grid reinforcement (status quo)
            edisgo.network.config[
                'grid_expansion_allowed_voltage_deviations'] = {
                'hv_mv_trafo_offset': 0.04,
                'hv_mv_trafo_control_deviation': 0.0,
                'mv_load_case_max_v_deviation': 0.055,
                'mv_feedin_case_max_v_deviation': 0.02,
                'lv_load_case_max_v_deviation': 0.065,
                'lv_feedin_case_max_v_deviation': 0.03,
                'mv_lv_station_load_case_max_v_deviation': 0.02,
                'mv_lv_station_feedin_case_max_v_deviation': 0.01
            }
            # Do grid reinforcement
            edisgo.reinforce()
            # Save results
            edisgo.network.results.save(
                'results_grid_{}_before_generator_import'.format(
                    edisgo.network.id))

            # Clear results and reset configs
            edisgo.network.results = Results(edisgo.network)
            edisgo.network.config = None

            # Calculate grid expansion costs after generator import
            logging.info('Grid expansion after generator import.')
            before_geno_import = False

            # Import generators
            edisgo.import_generators(generator_scenario=scenario)

            # Do grid reinforcement
            edisgo.reinforce()
            # Save results
            edisgo.network.results.save('results_grid_{}'.format(
                edisgo.network.id))

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

    # write error messages to csv files
    pd.DataFrame(faulty_grids_before_geno_import).to_csv(
        'faulty_grids_before_geno_import.csv', index=False)
    pd.DataFrame(faulty_grids).to_csv('faulty_grids.csv', index=False)
