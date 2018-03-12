import os
import sys
import readline
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
    Just get the grid district number from ding0 data file path

    Parameters
    ----------
    ding0_filepath : str
        Path to ding0 data ending typically
        `/path/to/ding0_data/"ding0_grids__" + str(``grid_district``) + ".xxx"`
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


def run_edisgo_basic(ding0_filepath,
                     generator_scenario=None,
                     analysis='worst-case'):
    """
    Analyze edisgo grid extension cost as reference scenario

    Parameters
    ----------
    ding0_filepath : str
        Path to ding0 data ending typically
        `/path/to/ding0_data/"ding0_grids__" + str(``grid_district``) + ".xxx"`

    analysis : str
        Either 'worst-case' or 'timeseries'

    generator_scenario : None or :obj:`str`
        If provided defines which scenario of future generator park to use
        and invokes import of these generators. Possible options are 'nep2035'
        and 'ego100'.
    Returns
    -------
    pandas : Dataframe
        costs
    dict
        grid_issues

    """

    grid_district = _get_griddistrict(ding0_filepath)

    grid_issues = {'grid': [],
                   'msg': []}

    logging.info('Grid expansion for MV grid district {}'.format(grid_district))

    if 'worst-case' in analysis:
        edisgo_grid = EDisGo(ding0_grid=ding0_filepath,
                             worst_case_analysis=analysis)
    elif 'timeseries' in analysis:
        edisgo_grid = EDisGo(ding0_grid=ding0_filepath,
                             timeseries_generation_fluctuating='oedb',
                             timeseries_load='demandlib')
    # Import generators
    if generator_scenario:
        logging.info('Grid expansion for scenario \'{}\'.'.format(generator_scenario))
        edisgo_grid.import_generators(generator_scenario=generator_scenario)
    else:
        logging.info('Grid expansion with no generator imports based on scenario')

    try:
        # Do non-linear power flow analysis with PyPSA
        edisgo_grid.analyze()

        # Do grid reinforcement
        edisgo_grid.reinforce()

        # Get costs
        costs_grouped = \
            edisgo_grid.network.results.grid_expansion_costs.groupby(
                ['type']).sum()
        costs = pd.DataFrame(costs_grouped.values,
                             columns=costs_grouped.columns,
                             index=[[edisgo_grid.network.id] * len(costs_grouped),
                                    costs_grouped.index])
        costs.reset_index(inplace=True)

        logging.info('SUCCESS!')
    except MaximumIterationError:
        grid_issues['grid'].append(edisgo_grid.network.id)
        grid_issues['msg'].append(str(edisgo_grid.network.results.unresolved_issues))
        costs = pd.DataFrame()
        logging.info('Unresolved issues left after grid expansion.')
    except Exception as e:
        grid_issues['grid'].append(edisgo_grid.network.id)
        grid_issues['msg'].append(repr(e))
        costs = pd.DataFrame()
        logging.info('Inexplicable Error, Please Check error messages and logs.')

    return costs, grid_issues

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

    # add the verbosity arguments

    ding0_files_parsegroup = parser.add_mutually_exclusive_group(required=True)

    ding0_files_parsegroup.add_argument('-f', '--ding0-file-path', type=str,
                                        action='store',
                                        dest='ding0_filename',
                                        help='Path to a single ding0 file.')
    ding0_files_parsegroup.add_argument('-d', '--ding0-files-directory', type=str,
                                        action='store',
                                        dest='ding0_dirglob',
                                        help='Path to a directory of ding0 files ' + \
                                             'along with  a file name pattern for glob input.')
    ding0_files_parsegroup.add_argument('-ds', '--ding0-files-directory-selection', type=str,
                                        nargs=3,
                                        action='store',
                                        dest='ding0_dir_select',
                                        help='Path to a directory of ding0 files, ' + \
                                             'Path to file with list of grid district numbers ' + \
                                             '(one number per line), ' + \
                                             'and file name template using {} where number ' + \
                                             'is to be inserted . Convention is to use ' + \
                                             'a double underscore before grid district number ' + \
                                             ' like so \'__{}\'.')

    analysis_parsegroup = parser.add_mutually_exclusive_group()

    analysis_parsegroup.add_argument('-wc', '--worst-case',
                                     help='Performs a worst-case simulation with ' + \
                                          'a single snapshot')

    analysis_parsegroup.add_argument('-ts', '--timeseries',
                                     action='store_true',
                                     help='Performs a worst-case simulation with ' + \
                                          'a time-series')

    parser.add_argument('-s', '--scenario',
                        type=str,
                        default=None,
                        choices=[None, 'nep2035', 'ego100'],
                        help="\'None\' or \'string\'\n" + \
                             "If provided defines which scenario " + \
                             "of future generator park to use " + \
                             "and invokes import of these generators.\n" + \
                             "Possible options are \'nep2035\'and \'ego100\'.")

    parser.add_argument('-o', '--output-dir',
                        nargs='?',
                        metavar='/path/to/output/',
                        dest="out_dir",
                        type=str,
                        default=os.path.join(sys.path[0]),
                        help='Absolute path to results data location.')

    parser.add_argument('--steps',
                        nargs='?',
                        metavar='4',
                        dest="steps",
                        type=int,
                        default=1,
                        help='Number of grid district that are analyzed in one '
                             'bunch. Hence, that are saved into one CSV file.')

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

    # get the list of files to run on
    if args.ding0_filename:
        ding0_file_list = [args.ding0_filename]

    elif args.ding0_dirglob:
        ding0_file_list = glob.glob(args.ding0_dirglob)

    elif args.ding0_dirselect:
        with open(args.ding0_dirselect[1], 'r') as file_handle:
            ding0_file_list_grid_district_numbers = list(file_handle)

        ding0_file_list = map(lambda x: args.ding0_dirselect[0] +
                                        args.ding0_dirselect[2].format(x),
                              ding0_file_list_grid_district_numbers)
    else:
        raise FileNotFoundError('Some of the Arguments for input files are missing.')

    # this is the serial version of the run system

    run_func = run_edisgo_basic

    run_args_opt_no_scenario = [None]
    run_args_opt = [args.scenario]
    if args.worst_case:
        run_args_opt_no_scenario.append('worst-case')
        run_args_opt.append('worst-case')
    elif args.timeseries:
        run_args_opt_no_scenario.append('timeseries')
        run_args_opt.append('timeseries')

    all_costs_before_geno_import = []
    all_grid_issues_before_geno_import = {'grid': [], 'msg': []}
    all_costs = []
    all_grid_issues = {'grid': [], 'msg': []}

    for ding0_filename in ding0_file_list:
        grid_district = _get_griddistrict(ding0_filename)

        # base case with no generator import
        run_args = [ding0_filename]
        run_args.extend(run_args_opt_no_scenario)

        costs_before_geno_import, \
            grid_issues_before_geno_import = run_func(*run_args)

        all_costs_before_geno_import.append(costs_before_geno_import)

        all_grid_issues_before_geno_import['grid'].extend(
            grid_issues_before_geno_import['grid'])

        all_grid_issues_before_geno_import['msg'].extend(
            grid_issues_before_geno_import['msg'])

        # case after generator import
        run_args = [ding0_filename]
        run_args.extend(run_args_opt)
        costs, \
            grid_issues = run_func(*run_args)

        all_costs.append(costs)
        all_grid_issues['grid'].extend(grid_issues['grid'])
        all_grid_issues['msg'].extend(grid_issues['msg'])

    # consolidate costs for all the networks
    all_costs_before_geno_import = pd.concat(all_costs_before_geno_import,
                                             ignore_index=True)
    all_costs = pd.concat(all_costs, ignore_index=True)
    # write costs and error messages to csv files

    pd.DataFrame(all_grid_issues_before_geno_import).to_csv(
        args.out_dir + 'grid_issues_before_geno_import.csv', index=False)

    with open(args.out_dir + 'costs_before_geno_import.csv', 'a') as f:
        f.write('# units: length in km, total_costs in kEUR\n')
        all_costs_before_geno_import.to_csv(f)

    pd.DataFrame(all_grid_issues).to_csv(args.out_dir + \
                                         'grid_issues.csv', index=False)
    with open(args.out_dir + 'costs.csv', 'a') as f:
        f.write('# units: length in km, total_costs in kEUR\n')
        all_costs.to_csv(f)

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
