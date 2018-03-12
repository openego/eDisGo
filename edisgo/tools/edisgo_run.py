import os
import sys
import glob
import re

import random
import itertools
import time
import queue
import multiprocessing as mp

import argparse
import logging

import pandas as pd
from edisgo.grid.network import EDisGo
from edisgo.grid.network import Results
from edisgo.flex_opt.exceptions import MaximumIterationError


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


def _get_griddistrict(ding0_filepath):
    """
    Just get the grid district number from ding0 pkl file path

    Parameters
    ----------
    ding0_filepath : str
        Path to ding0 .pkl data ending typically
        `/path/to/ding0_pkl/"ding0_grids__" + str(``grid_district``) + ".pkl"`
    Returns
    -------
    int
        grid_district number
    """
    grid_district = os.path.basename(ding0_filepath)
    grid_district_search = re.search('[_]+\d+', grid_district)
    if grid_district_search:
        grid_district = int(grid_district_search.group(0)[2:])
        return grid_district
    else:
        raise (KeyError('Grid District not found in '.format(grid_district)))


def run_edisgo_worst_case(ding0_filepath, scenario):
    """
    Analyze worst-case grid extension cost as reference scenario

    Parameters
    ----------
    dig0_filepath : str
        Path to ding0 .pkl data ending typically
        `/path/to/ding0_pkl/"ding0_grids__" + str(``grid_district``) + ".pkl"`
    scenario : None or str
        If provided defines which scenario of future generator park to use
        and invokes import of these generators. Possible options are 'nep2035'
        and 'ego100'.

    Returns
    -------
    pandas.DataFrame
        Cost of grid extension
    pandas.DataFrame
        Information about grid that cannot be properly equipped to host new
        generators
        :param ding0_filepath:
    """

    grid_district = _get_griddistrict(ding0_filepath)

    logging.info('Grid expansion for MV grid district {}'.format(grid_district))
    edisgo_grid = EDisGo(ding0_grid=ding0_filepath,
                         worst_case_analysis='worst-case-feedin')

    try:
        # Calculate grid expansion costs before generator import
        logging.info('Grid expansion before generator import.')
        before_geno_import = True

        # Do non-linear power flow analysis with PyPSA
        edisgo_grid.analyze()

        # Do grid reinforcement
        edisgo_grid.reinforce()

        # Get costs
        costs_grouped = \
            edisgo_grid.network.results.grid_expansion_costs.groupby(
                ['type']).sum()
        costs_before_geno_import = \
            pd.DataFrame(costs_grouped.values,
                         columns=costs_grouped.columns,
                         index=[[edisgo_grid.network.id] * len(costs_grouped),
                                costs_grouped.index])

        # Clear results
        edisgo_grid.network.results = Results()
        edisgo_grid.network.pypsa = None

        # Calculate grid expansion costs after generator import
        logging.info('Grid expansion after generator import.')
        before_geno_import = False

        # Import generators
        if scenario:
            edisgo_grid.import_generators(generator_scenario=scenario)

        # Do non-linear power flow analysis with PyPSA
        edisgo_grid.analyze()

        # Do grid reinforcement
        edisgo_grid.reinforce()

        costs_grouped = \
            edisgo_grid.network.results.grid_expansion_costs.groupby(
                ['type']).sum()

        costs = \
            pd.DataFrame(costs_grouped.values,
                         columns=costs_grouped.columns,
                         index=[[edisgo_grid.network.id] * len(costs_grouped),
                                costs_grouped.index])
        logging.info('SUCCESS!')

    return costs_before_geno_import, \
           costs


def run_edisgo_timeseries_worst_case(grid_district, data_dir, filename_template="ding0_grids__{}.pkl",
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


def run_pool(number_of_processes, grid_ids, data_dir, worker_lifetime,
             analysis_mode, technologies=None, curtailment=None):
    def collect_pool_results(result):
        results.append(result)

    results = []

    tech_dict = dict(technologies=technologies,
                     curtailment=curtailment)

    pool = mp.Pool(number_of_processes,
                   maxtasksperchild=worker_lifetime)

    if analysis_mode == 'worst-case':
        for grid_id in grid_ids:
            pool.apply_async(func=run_edisgo_worst_case,
                             args=(grid_id, data_dir),
                             kwds=tech_dict,
                             callback=collect_pool_results)
    elif analysis_mode == 'timeseries':
        for grid_id in grid_ids:
            pool.apply_async(func=run_edisgo_timeseries_worst_case,
                             args=(grid_id, data_dir),
                             kwds=tech_dict,
                             callback=collect_pool_results)
    else:
        logger.error("Unknown analysis mode {]".format(analysis_mode))

    pool.close()
    pool.join()

    # process results data
    costs_dfs = [r[0] for r in results]
    faulty_grids_dfs = [r[1] for r in results]
    costs_before_generators_dfs = [r[2] for r in results]
    faulty_grids_before_generators_dfs = [r[3] for r in results]

    if costs_dfs:
        costs = pd.concat(costs_dfs, axis=0)
    else:
        costs = pd.DataFrame()

    if faulty_grids_dfs:
        faulty_grids = pd.concat(faulty_grids_dfs, axis=0)
    else:
        faulty_grids = pd.DataFrame()

    if costs_before_generators_dfs:
        costs_before_generators = pd.concat(costs_before_generators_dfs, axis=0)
    else:

        costs_before_generators = pd.DataFrame()

    if faulty_grids_before_generators_dfs:
        faulty_grids_before_generators = pd.concat(
            faulty_grids_before_generators_dfs, axis=0)
    else:
        faulty_grids_before_generators = pd.DataFrame()

    return costs, faulty_grids, costs_before_generators, faulty_grids_before_generators


if __name__ == '__main__':
    # create the argument parser
    parser = argparse.ArgumentParser(description="Commandline running" + \
                                                 "of eDisGo")

    # add the arguments
    # verbosity arguments

    ding0_files_parsegroup = parser.add_mutually_exclusive_group(required=True)

    ding0_files_parsegroup.add_argument('-f', '--ding0-file-path', type=str,
                                        action='store',
                                        dest='ding0_filename',
                                        help='Path to a single ding0 pkl file')
    ding0_files_parsegroup.add_argument('-d', '--ding0-files-directory', type=str,
                                        action='store',
                                        dest='ding0_dirpath',
                                        help='Path to a directory of ding0 pkl files')

    analysis_parsegroup = parser.add_mutually_exclusive_group()

    analysis_parsegroup.add_argument('-wc', '--worst-case',
                                     help='Perfroms a worst-case simulation with ' + \
                                          'a single snapshot',
                                     default=True)

    analysis_parsegroup.add_argument('-ts', '--time-series',
                                     action='store',
                                     dest='time_series',
                                     default=None,
                                     help='Perfroms a worst-case simulation with ' + \
                                          'a timeseries')

    analysis_parsegroup.add_argument('-twc', '--timeseries-worst-case',
                                     action='store',
                                     dest='twc_time_series',
                                     default=None,
                                     help='Analyze grid only in worst-case snapshots of a '
                                          'time series.')

    parser.add_argument('-s', '--scenario',
                        type=str,
                        default=None,
                        choices=['nep2035', 'ego100'],
                        help="\'None\' or \'string\'\n" + \
                             "If provided defines which scenario " + \
                             "of future generator park to use " + \
                             "and invokes import of these generators.\n" + \
                             "Possible options are \'nep2035\'and \'ego100\'.")

    parser.add_argument('--steps',
                        nargs='?',
                        metavar='4',
                        dest="steps",
                        type=int,
                        default=1,
                        help='Number of grid district that are analyzed in one '
                             'bunch. Hence, that are saved into one CSV file.')

    parser.add_argument('-o', '--output-dir',
                        nargs='?',
                        metavar='/path/to/output/',
                        dest="out_dir",
                        type=str,
                        default=os.path.join(sys.path[0]),
                        help='Absolute path to results data location.')

    parser.add_argument('-lw', '--lifetime-workers',
                        nargs='?',
                        metavar='1..inf',
                        dest="worker_lifetime",
                        type=int,
                        default=None,
                        help='Lifetime of a worker of the cluster doing the '
                             'work. The lifetime is given is number of jobs a'
                             ' worker does before it is replaced by a freshly '
                             'new one.'
                             'The default sets the lifetime to the pools '
                             'lifetime. This can cause memory issues!')
    args = parser.parse_args(sys.argv[1:])

    logger = setup_logging(logfilename='test.log',
                           logfile_loglevel='debug',
                           console_loglevel='info')

    if args.ding0_filename:
        costs_before_geno_import, \
        costs = \
            run_edisgo_worst_case(args.ding0_filename, args.scenario)

        grid_district = _get_griddistrict(args.ding0_filename)

        output_csv_name = '{}_costs_before_geno_import.csv'.format(grid_district)
        costs_before_geno_import.to_csv(args.out_dir + output_csv_name)

        output_csv_name = '{}_costs.csv'.format(grid_district)
        costs.to_csv(args.out_dir + output_csv_name)

    elif args.ding0_dirpath:
        ding0_file_list = glob.glob(args.ding0_dirpath + '*.pkl')
        for ding0_filename in ding0_file_list:
            costs_before_geno_import, \
            costs = \
                run_edisgo_worst_case(args.ding0_filename, args.scenario)

            grid_district = _get_griddistrict(args.ding0_filename)

            output_csv_name = '{}_costs_before_geno_import.csv'.format(grid_district)
            costs_before_geno_import.to_csv(args.out_dir + output_csv_name)

            output_csv_name = '{}_costs.csv'.format(grid_district)
            costs.to_csv(args.out_dir + output_csv_name)

    # # Range of grid districts
    # if args.grids is not None and args.grids_file is not None:
    #     raise ValueError("Please specify only one of \'grids\', \'grids-file\'")
    # elif args.grids is None and args.grids_file is None:
    #     raise ValueError("Please specify at least one of \'grids\', "
    #                      "\'grids-file\'")
    # elif args.grids is not None:
    #     grid_range = args.grids[0]
    #     print(grid_range)
    # elif args.grids_file is not None:
    #     with open(args.grids_file) as f:
    #         grids_str = f.read()
    #         grid_range = [int(_) for _ in grids_str.split('\n') if _]
    #     f.close()
    #
    # # Define number of parallel process (=number of grids calculated in parallel)
    # number_of_processes = mp.cpu_count()
    #
    # technologies = ['wind', 'solar']
    # curtailment = {'wind': 0.7, 'solar': 0.7}
    #
    # # Determine bunches of grid districts analyzed in one batch
    # grid_bunches = [grid_range[x:x + args.steps]
    #                 for x in range(0, len(grid_range), args.steps)]
    #
    # # Run iteratively of bunches of grid districts
    # for grid_bunch in grid_bunches:
    #     logger.info("Running eDisGo worst-case grid extension cost estimation "
    #                 "across the following grids: \n\t {}".format(
    #         str(grid_bunch)))
    #
    #
    #
    #     # Calculate multiple grids using pool
    #
    #     # costs, faulty_grids, \
    #     # cost_before_generators, \
    #     # faulty_grids_before_generators = \
    #     #     run_pool(
    #     #         number_of_processes,
    #     #         grid_bunch,
    #     #         args.data_dir,
    #     #         args.worker_lifetime,
    #     #         args.analysis,
    #     #         technologies=technologies,
    #     #         curtailment=curtailment)
    #
    #     # Save results data
    #     f = open(os.path.join(args.out_dir, 'costs_{start}_{end}.csv'.format(start=min(grid_bunch),
    #                                                                          end=max(grid_bunch))), 'w')
    #     f.write('# units: length in km, total_costs in kEUR\n')
    #     costs.to_csv(f)
    #     f.close()
    #     faulty_grids.to_csv(
    #         os.path.join(args.out_dir,
    #                      'faulty_grids_{start}_{end}.csv'.format(
    #                          start=min(grid_bunch),
    #                          end=max(grid_bunch))),
    #         index_label='grid')
    #
    #     f = open(
    #         os.path.join(args.out_dir,
    #                      'costs_before_generators_{start}_{end}.csv'.format(
    #                          start=min(grid_bunch),
    #                          end=max(grid_bunch))),
    #         'w')
    #     f.write('# units: length in km, total_costs in kEUR\n')
    #     cost_before_generators.to_csv(f)
    #     f.close()
    #     faulty_grids_before_generators.to_csv(
    #         os.path.join(args.out_dir,
    #                      'faulty_grids_before_generators_{start}_{end}.csv'.format(
    #                          start=min(grid_bunch),
    #                          end=max(grid_bunch))),
    #         index_label='grid')
    #
    #     logger.info("Run across grid {one} to {two} finished!".format(
    #         one=min(grid_bunch), two=max(grid_bunch)))
