# -*- coding: utf-8 -*-

"""
This example shows the general usage of eDisGo. Grid expansion costs for
distribution grids generated with ding0 are calculated assuming renewable
and conventional power plant capacities as stated in the scenario framework of
the German Grid Development Plan 2015 (Netzentwicklungsplan) for the year 2035
(scenario B2). To determine topology expansion needs worst-case scenarios
(heavy load flow and reverse power flow) used in conventional grid expansion
planning are set up.

The example assumes you have ding0 grids in the current working directory. If
you need more information on how to get ding0 grids see the ding0 documentation
or the Quickstart section of the eDisGo documentation.

As the grids generated with ding0 should represent current stable grids but
have in some cases stability issues, topology expansion is first conducted
before connecting future generators in order to obtain stable grids. Final
grid expansion costs in the DataFrame 'costs' only contain grid expansion costs
of measures conducted after future generators are connected to the stable
grids.

"""

import os
import pandas as pd

from edisgo import EDisGo
from edisgo.network.results import Results

import logging
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':

    # specify path to directory containing ding0 grid csv files
    dingo_grid_path = os.path.join(os.path.dirname(__file__), '460')

    # set scenario to define future power plant capacities
    scenario = 'nep2035'

    # initialize containers that will hold any error messages
    faulty_grids_before_geno_import = {'topology': [], 'msg': []}
    faulty_grids = {'topology': [], 'msg': []}

    # set up worst-case scenario
    edisgo = EDisGo(ding0_grid=dingo_grid_path,
                    worst_case_analysis='worst-case')

    # Calculate topology expansion costs before generator import
    logging.info('Grid expansion before generator import.')
    before_geno_import = True

    # overwrite config parameters for allowed voltage deviations in
    # initial topology reinforcement (status quo)
    edisgo.config[
        'grid_expansion_allowed_voltage_deviations'] = {
        'feedin_case_lower': 0.9,
        'load_case_upper': 1.1,
        'hv_mv_trafo_offset': 0.04,
        'hv_mv_trafo_control_deviation': 0.0,
        'mv_load_case_max_v_deviation': 0.055,
        'mv_feedin_case_max_v_deviation': 0.02,
        'lv_load_case_max_v_deviation': 0.065,
        'lv_feedin_case_max_v_deviation': 0.03,
        'mv_lv_station_load_case_max_v_deviation': 0.02,
        'mv_lv_station_feedin_case_max_v_deviation': 0.01
    }
    # Do topology reinforcement
    edisgo.reinforce()
    # Save results
    edisgo.results.save(
        'results_grid_{}_before_generator_import'.format(
            edisgo.topology.id))

    # Clear results and reset configs
    edisgo.results = Results(edisgo)
    edisgo.config = None

    # Calculate topology expansion costs after generator import
    logging.info('Grid expansion after generator import.')
    before_geno_import = False

    # Import generators
    edisgo.import_generators(generator_scenario=scenario)

    # Do topology reinforcement
    edisgo.reinforce()
    # Save results
    edisgo.results.save('results_grid_{}'.format(
        edisgo.topology.id))

    logging.info('SUCCESS!')

    # write error messages to csv files
    pd.DataFrame(faulty_grids_before_geno_import).to_csv(
        'faulty_grids_before_geno_import.csv', index=False)
    pd.DataFrame(faulty_grids).to_csv('faulty_grids.csv', index=False)
