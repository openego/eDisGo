import os
import sys

import random
import itertools
import time
import queue
import multiprocessing as mp

import argparse
import logging


import pandas as pd
from edisgo.grid.network import EDisGo


def setup_logging(logfilename=None,
                  logfile_loglevel='debug',
                  console_loglevel='info',
                  **logging_kwargs):
    # a dict to help with log level definition
    loglevel_dict = {'info': logging.INFO,
                     'debug': logging.DEBUG,
                     'warn': logging.WARNING,
                     'warning': logging.WARNING,
                     'error': logging.ERROR,
                     'critical': logging.CRITICAL}

    if not (logfilename):
        logfilename = 'edisgo_run.log'

    logging.basicConfig(filename=logfilename,
                        format='%(asctime)s - %(name)s -' +
                               ' %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=loglevel_dict[logfile_loglevel])

    root_logger = logging.getLogger()

    console_stream = logging.StreamHandler()
    console_stream.setLevel(loglevel_dict[console_loglevel])
    console_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s -' +
                                              ' %(levelname)s - %(message)s',
                                          datefmt='%m/%d/%Y %H:%M:%S')
    console_stream.setFormatter(console_formatter)

    # add stream handler to root logger
    root_logger.addHandler(console_stream)

    return root_logger

def run_edisgo_worst_case(grid_district, data_dir,
                          filename_template="ding0_grids__{}.pkl",
                          technologies=None):
    """
    Analyze worst-case grid extension cost as reference scenario

    Parameters
    ----------
    grid_district : int
        ID of the MV grid district
    data_dir : str
        Path to directory with ding0 .pkl data
    filename_template : str
        Specify file name pattern. Defaults to
        `"ding0_grids__" + str(``grid_district``) + ".pkl"`


    Returns
    -------
    pandas.DataFrame
        Cost of grid extension
    pandas.DataFrame
        Information about grid that cannot be properly equipped to host new
        generators
    """
    logging.info('Grid expansion for MV grid district {}'.format(grid_district))
    filename = filename_template.format(grid_district)

    timeseries = TimeSeries()
    scenario = Scenario(timeseries=timeseries,
                        power_flow='worst-case',
                        mv_grid_id=grid_district)
    try:
        network = Network.import_from_ding0(
            os.path.join(data_dir, filename),
            id=grid_district,
            scenario=scenario)

        # Calculate grid expansion costs before generator import

        # Do non-linear power flow analysis with PyPSA
        network.analyze()
        # Do grid reinforcement
        network.reinforce()
        # Get costs
        costs_grouped = network.results.grid_expansion_costs.groupby(
            ['type']).sum()
        costs_before_geno_import = pd.DataFrame(costs_grouped.values,
                                                columns=costs_grouped.columns,
                                                index=[[network.id] * len(costs_grouped),
                                                       costs_grouped.index])
        if network.results.unresolved_issues:
            faulty_grids_before_geno_import = pd.DataFrame(
                {'grid': [grid_district],
                 'msg': [str(network.results.unresolved_issues)]}
            ).set_index('grid', drop=True)
            logging.info('Unresolved issues left after grid expansion '
                         '(before grid connection of generators).')
            costs = pd.DataFrame()
            faulty_grids = pd.DataFrame({
                'grid': [grid_district],
                'msg': ["Unresolved issues before "
                        "generator import"]}).set_index('grid', drop=True)
        else:
            faulty_grids_before_geno_import = pd.DataFrame()
            logging.info('SUCCESS: grid reinforce before generators!')

            # Clear results
            network.results = Results()
            network.pypsa = None

            # Calculate grid expansion costs after generator import
            logging.info('Grid expansion after generator import.')

            # Import generators
            network.import_generators(types=technologies)

            # Do non-linear power flow analysis with PyPSA
            network.analyze()

            # Do grid reinforcement
            network.reinforce()

            # Retrieve cost results data
            costs_grouped = network.results.grid_expansion_costs.groupby(
                ['type']).sum()
            costs = pd.DataFrame(costs_grouped.values,
                                 columns=costs_grouped.columns,
                                 index=[[network.id] * len(costs_grouped),
                                        costs_grouped.index])
            if network.results.unresolved_issues:
                faulty_grids = pd.DataFrame(
                    {'grid': [grid_district],
                     'msg': [str(network.results.unresolved_issues)]}
                ).set_index('grid', drop=True)
                logging.info('Unresolved issues left after grid expansion.')
            else:
                faulty_grids = pd.DataFrame()
                logging.info('SUCCESS!')
        return costs, faulty_grids, costs_before_geno_import, faulty_grids_before_geno_import
    except Exception as e:
        faulty_grids = {'grid': [grid_district], 'msg': [e]}
        faulty_grids_before_geno_import = {'grid': [grid_district], 'msg': [e]}
        logging.info('Something went wrong.')
        return pd.DataFrame(), \
               pd.DataFrame(faulty_grids).set_index('grid', drop=True), \
               pd.DataFrame(), \
               pd.DataFrame(faulty_grids_before_geno_import).set_index('grid', drop=True)


def run_edisgo_timeseries_worst_case(grid_district, data_dir,
                                     filename_template="ding0_grids__{}.pkl",
                                     technologies=None,
                                     curtailment=None):
    """
    Analyze worst-case grid extension cost as reference scenario

    Parameters
    ----------
    grid_district : int
        ID of the MV grid district
    data_dir : str
        Path to directory with ding0 .pkl data
    filename_template : str
        Specify file name pattern. Defaults to
        `"ding0_grids__" + str(``grid_district``) + ".pkl"`
    curtailment : dict
        Specify curtail power generation of technologies to an upper limit that
        is defined relatively to generators' nominal capacity.


    Returns
    -------
    pandas.DataFrame
        Cost of grid extension
    pandas.DataFrame
        Information about grid that cannot be properly equipped to host new
        generators
    """
    logging.info('Grid expansion for MV grid district {}'.format(grid_district))
    filename = filename_template.format(grid_district)

    try:
        scenario = Scenario(
            power_flow=(),
            mv_grid_id=grid_district,
            scenario_name=['NEP 2035', 'Status Quo'])
        network = Network.import_from_ding0(
            os.path.join(data_dir, filename),
            id=grid_district,
            scenario=scenario,
            curtailment=curtailment)

        ts_load, ts_gen = select_worstcase_snapshots(network)
        del network
        scenario = Scenario(
            power_flow=(ts_gen, ts_gen),
            mv_grid_id=grid_district,
            scenario_name=['NEP 2035', 'Status Quo'],
            curtailment=curtailment)
        network = Network.import_from_ding0(
            os.path.join(data_dir, filename),
            id=grid_district,
            scenario=scenario)

        # Calculate grid expansion costs before generator import

        # Do non-linear power flow analysis with PyPSA
        network.analyze()

        # Do grid reinforcement
        network.reinforce()
        # Get costs
        costs_grouped = network.results.grid_expansion_costs.groupby(
            ['type']).sum()
        costs_before_geno_import = pd.DataFrame(costs_grouped.values,
                                                columns=costs_grouped.columns,
                                                index=[[network.id] * len(costs_grouped),
                                                       costs_grouped.index])
        if network.results.unresolved_issues:
            faulty_grids_before_geno_import = pd.DataFrame(
                {'grid': [grid_district],
                 'msg': [str(network.results.unresolved_issues)]}
            ).set_index('grid', drop=True)
            logging.info('Unresolved issues left after grid expansion '
                         '(before grid connection of generators).')
            costs = pd.DataFrame()
            faulty_grids = pd.DataFrame({
                'grid': [grid_district],
                'msg': ["Unresolved issues before "
                        "generator import"]}).set_index('grid', drop=True)
        else:
            faulty_grids_before_geno_import = pd.DataFrame()
            logging.info('SUCCESS: grid reinforce before generators!')

            # Clear results
            network.results = Results()
            network.pypsa = None

            # Calculate grid expansion costs after generator import
            logging.info('Grid expansion after generator import.')

            # Import generators
            network.import_generators(types=technologies)

            # Do non-linear power flow analysis with PyPSA
            network.analyze()

            # Do grid reinforcement
            network.reinforce()

            # Retrieve cost results data
            costs_grouped = network.results.grid_expansion_costs.groupby(
                ['type']).sum()
            costs = pd.DataFrame(costs_grouped.values,
                                 columns=costs_grouped.columns,
                                 index=[[network.id] * len(costs_grouped),
                                        costs_grouped.index])
            if network.results.unresolved_issues:
                faulty_grids = pd.DataFrame(
                    {'grid': [grid_district],
                     'msg': [str(network.results.unresolved_issues)]}
                ).set_index('grid', drop=True)
                logging.info('Unresolved issues left after grid expansion.')
            else:
                faulty_grids = pd.DataFrame()
                logging.info('SUCCESS!')
        return costs, faulty_grids, costs_before_geno_import, faulty_grids_before_geno_import
    except Exception as e:
        faulty_grids = {'grid': [grid_district], 'msg': [e]}
        faulty_grids_before_geno_import = {'grid': [grid_district], 'msg': [e]}
        logging.info('Something went wrong.')
        return pd.DataFrame(), \
               pd.DataFrame(faulty_grids).set_index('grid', drop=True), \
               pd.DataFrame(), \
               pd.DataFrame(faulty_grids_before_geno_import).set_index('grid', drop=True)


def first_and_last_grid(s):
    try:
        f, l = map(int, s.split(','))
        return f, l
    except:
        raise argparse.ArgumentTypeError("Grid range must be first last")


def select_worstcase_snapshots(network):
    """
    Select two worst-case snapshot from time series

    Two time steps in a time series represent worst-case snapshots. These are

    1. Maximum residual load: refers to the point in the time series where the
        (load - generation) achieves its maximum
    2. Minimum residual load: refers to the point in the time series where the
        (load - generation) achieves its minimum

    These to points are identified based on the generation and load time series.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo overall container

    Returns
    -------
    type
        Timestamp of snapshot maximum residual load
    type
        Timestamp of snapshot minimum residual load
    """

    grids = [network.mv_grid] + list(network.mv_grid.lv_grids)

    peak_generation = pd.concat(
        [_.peak_generation_per_technology for _ in grids], axis=1).fillna(
        0).sum(axis=1)

    non_solar_wind = [_ for _ in list(peak_generation.index)
                      if _ not in ['wind', 'solar']]
    peak_generation['other'] = peak_generation[non_solar_wind].sum()
    peak_generation.drop(non_solar_wind, inplace=True)

    peak_load = pd.concat(
        [_.consumption for _ in grids], axis=1).fillna(
        0).sum(axis=1)

    residual_load = (
            (network.scenario.timeseries.load * peak_load).sum(axis=1) - (
            network.scenario.timeseries.generation * peak_generation).sum(
        axis=1))

    return residual_load.idxmax(), residual_load.idxmin()


if __name__ == '__main__':
    logger = setup_logging(logfilename='test.log',
                           logfile_loglevel='debug',
                           console_loglevel='info')

    # create the argument parser
    parser = argparse.ArgumentParser(description="Commandline running" + \
                                                 "of eDisGo")

    # add the arguments
    # verbosity arguments

    parser.add_argument('analysis',
                        # nargs='?',
                        metavar='{worst-case,timeseries}',
                        help='Choose how the power flow analysis is performed. '
                             'Either on a single snapshot \'worst-case\' or '
                             'along a time period \'timeseries\'')

    parser.add_argument('-g', '--grids',
                        nargs='*',
                        metavar='1 3608',
                        default=None,
                        type=first_and_last_grid,
                        help='Specify grid range for analysis for two '
                             'comma separated integers.')

    parser.add_argument('-gf', '--grids-file',
                        nargs='?',
                        metavar='/path/to/file',
                        default=None,
                        dest='grids_file',
                        type=str,
                        help='Specify grid range for analysis by list of grid '
                             'IDs in a simple text file.')

    parser.add_argument('-d', '--data-dir',
                        nargs='?',
                        metavar='/path/to/data/',
                        dest="data_dir",
                        type=str,
                        default="",
                        help='Absolute path to Ding0 grid data.')

    parser.add_argument('-o', '--output-dir',
                        nargs='?',
                        metavar='/path/to/output/',
                        dest="out_dir",
                        type=str,
                        default=os.path.join(sys.path[0]),
                        help='Absolute path to results data location.')

    parser.add_argument('-p', '--parallelization',
                        nargs='?',
                        metavar='{processes,pool}',
                        dest="parallelization",
                        type=str,
                        default='processes',
                        help='Method of parallelization.')

    parser.add_argument('-twc', '--timeseries-worst-case',
                        nargs='?',
                        dest='twc',
                        help='Analyze grid only in worst-case snapshots of a '
                             'time series.',
                        default=False)

    args = parser.parse_args(sys.argv[1:])
