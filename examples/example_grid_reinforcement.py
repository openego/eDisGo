# -*- coding: utf-8 -*-

"""
This example shows how to determine grid expansion costs with eDisGo.
Grid expansion costs for an example distribution grid generated with ding0
are calculated assuming renewable and conventional power plant capacities as
stated in the scenario framework of the German Grid Development Plan 2015
(Netzentwicklungsplan) for the year 2035 (scenario B2).

To determine topology expansion needs worst-case scenarios
(heavy load flow and reverse power flow) used in conventional grid expansion
planning are set up.

The example automatically downloads an example ding0 grid to the current
working directory in case it does not yet exist. If you want more information
on how to get more ding0 grids, see the ding0 documentation or the Quickstart
section of the eDisGo documentation.

As the grids generated with ding0 should represent current stable grids but
have in some cases stability issues, topology expansion is first conducted
before connecting future generators in order to obtain stable grids. Final
grid expansion costs in the DataFrame 'costs' only contain grid expansion costs
of measures conducted after future generators are connected to the stable
grids.

"""

import os
import pandas as pd
import requests

from edisgo import EDisGo
from edisgo.network.results import Results

import logging
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)


def run_example():
    # Specify path to directory containing ding0 grid csv files
    edisgo_path = os.path.join(os.path.expanduser('~'), '.eDisGo')
    dingo_grid_path = os.path.join(edisgo_path,
                                   'ding0_example_grid')
    # Download example grid data in case it does not yet exist
    if not os.path.isdir(dingo_grid_path):
        logger.debug("Download example grid data.")
        os.makedirs(dingo_grid_path)
        file_list = ["buses.csv", "lines.csv",
                     "generators.csv", "loads.csv",
                     "transformers.csv", "transformers_hvmv.csv",
                     "switches.csv", "network.csv"]
        base_path = "https://raw.githubusercontent.com/openego/eDisGo/" \
                    "dev/tests/ding0_test_network_2/"
        for f in file_list:
            file = os.path.join(dingo_grid_path, f)
            req = requests.get(base_path + f)
            with open(file, "wb") as fout:
                fout.write(req.content)

    # Set scenario to define future power plant capacities
    scenario = 'nep2035'

    # Set up worst-case scenario
    edisgo = EDisGo(ding0_grid=dingo_grid_path,
                    worst_case_analysis='worst-case')

    # Reinforce ding0 grid to obtain a stable status quo grid
    logging.info('Conduct grid reinforcement to obtain stable '
                 'status quo grid.')
    # Overwrite config parameters for allowed voltage deviations in
    # initial topology reinforcement to better represent currently used limits
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
    # Conduct reinforcement
    edisgo.reinforce()

    # Clear results and reset configs
    edisgo.results = Results(edisgo)
    edisgo.config = None

    # Calculate expansion costs for NEP scenario
    logging.info('Determine grid expansion costs for NEP scenario.')

    # Get data on generators in NEP scenario and connect generators to the grid
    edisgo.import_generators(generator_scenario=scenario)

    # Conduct topology reinforcement
    edisgo.reinforce()

    # Get total grid expansion costs
    total_costs = edisgo.results.grid_expansion_costs.total_costs.sum()
    logging.info(
        'Grid expansion costs for NEP scenario are: {} kEUR.'.format(
            total_costs)
    )

    # Save grid expansion results
    edisgo.results.to_csv(
        directory=dingo_grid_path,
        parameters={'grid_expansion_results': None}
    )

    logging.info('SUCCESS!')

    return total_costs


if __name__ == '__main__':
    run_example()
