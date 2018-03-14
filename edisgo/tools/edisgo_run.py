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
                     analysis='worst-case',
                     *edisgo_grid):
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
    edisgo_grid : :class:`~.grid.network.EDisGo`
        eDisGo network container
    costs : :pandas:`pandas.Dataframe<dataframe>`
        Cost of grid extension
    grid_issues : dict
        Grids resulting in an error including error message

    """

    grid_district = _get_griddistrict(ding0_filepath)

    grid_issues = {}

    logging.info('Grid expansion for MV grid district {}'.format(grid_district))

    if edisgo_grid: # if an edisgo_grid is passed in arg then ignore everything else
        edisgo_grid = edisgo_grid[0]
    else:
        try:
            if 'worst-case' in analysis:
                edisgo_grid = EDisGo(ding0_grid=ding0_filepath,
                                     worst_case_analysis=analysis)
            elif 'timeseries' in analysis:
                edisgo_grid = EDisGo(ding0_grid=ding0_filepath,
                                     timeseries_generation_fluctuating='oedb',
                                     timeseries_load='demandlib')
        except FileNotFoundError as e:
            return None, pd.DataFrame(), {'grid': grid_district, 'msg': str(e)}

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
                                    costs_grouped.index]).reset_index()
        costs.rename(columns={'level_0': 'grid'}, inplace=True)

        grid_issues['grid'] = None
        grid_issues['msg'] = None

        logging.info('SUCCESS!')
    except MaximumIterationError:
        grid_issues['grid'] = edisgo_grid.network.id
        grid_issues['msg'] = str(edisgo_grid.network.results.unresolved_issues)
        costs = pd.DataFrame()
        logging.info('Unresolved issues left after grid expansion.')
    except Exception as e:
        grid_issues['grid'] = edisgo_grid.network.id
        grid_issues['msg'] = repr(e)
        costs = pd.DataFrame()
        logging.info('Inexplicable Error, Please Check error messages and logs.')

    return edisgo_grid, costs, grid_issues


def run_edisgo_twice(run_args):
    """
    Run grid analysis twice on same grid: once w/ and once w/o new generators

    First run without connection of new generators approves sufficient grid
    hosting capacity. Otherwise, grid is reinforced.
    Second run assessment grid extension needs in terms of RES integration

    Parameters
    ----------
    run_args : list
        Optional parameters for :func:`run_edisgo_basic`.

    Returns
    -------
    all_costs_before_geno_import : :pandas:`pandas.Dataframe<dataframe>`
        Grid extension cost before grid connection of new generators
    all_grid_issues_before_geno_import : dict
        Remaining overloading or over-voltage issues in grid
    all_costs : :pandas:`pandas.Dataframe<dataframe>`
        Grid extension cost due to grid connection of new generators
    all_grid_issues : dict
        Remaining overloading or over-voltage issues in grid
    """

    # base case with no generator import
    edisgo_grid, \
    costs_before_geno_import, \
    grid_issues_before_geno_import = run_edisgo_basic(*run_args)

    if edisgo_grid:
        # clear the pypsa object and results from edisgo_grid
        edisgo_grid.network.results = Results()
        edisgo_grid.network.pypsa = None

        # case after generator import
        # run_args = [ding0_filename]
        # run_args.extend(run_args_opt)
        run_args.append(edisgo_grid)

        _, costs, \
        grid_issues = run_edisgo_basic(*run_args)

        return costs_before_geno_import, grid_issues_before_geno_import, \
               costs, grid_issues
    else:
        return costs_before_geno_import, grid_issues_before_geno_import, \
               costs_before_geno_import, grid_issues_before_geno_import


def run_edisgo_pool(ding0_file_list, run_args_opt,
                    workers=mp.cpu_count(), worker_lifetime=1):
    """
    Use python multiprocessing toolbox for parallelization

    Several grids are analyzed in parallel.

    Parameters
    ----------
    ding0_file_list : list
        Ding0 grid data file names
    run_args_opt : list
        eDisGo options, see :func:`run_edisgo_basic` and
        :func:`run_edisgo_twice`
    workers: int
        Number of parallel process
    worker_lifetime : int
        Bunch of grids sequentially analyzed by a worker

    Returns
    -------
    all_costs_before_geno_import : list
        Grid extension cost before grid connection of new generators
    all_grid_issues_before_geno_import : list
        Remaining overloading or over-voltage issues in grid
    all_costs : list
        Grid extension cost due to grid connection of new generators
    all_grid_issues : list
        Remaining overloading or over-voltage issues in grid
    """
    def collect_pool_results(result):
        results.append(result)

    results = []

    pool = mp.Pool(workers,
                   maxtasksperchild=worker_lifetime)

    for file in ding0_file_list:
        edisgo_args = [file] + run_args_opt
        pool.apply_async(func=run_edisgo_twice,
                         args=(edisgo_args,),
                         callback=collect_pool_results)

    pool.close()
    pool.join()

    # process results data
    all_costs_before_geno_import = [r[0] for r in results]
    all_grid_issues_before_geno_import = [r[1] for r in results]
    all_costs = [r[2] for r in results]
    all_grid_issues = [r[3] for r in results]

    return all_costs_before_geno_import, all_grid_issues_before_geno_import, \
           all_costs, all_grid_issues


def edisgo_run():
    # create the argument parser
    example_text = '''Examples
    
    ...assumes all files located in PWD.
    
    Analyze a single grid in 'worst-case'
    
         edisgo_run -f ding0_grids__997.pkl -wc
         
         
    Analyze multiple grids in 'worst-case' using parallelization. Grid IDs are 
    specified by the grids_list.txt.
    
         edisgo_run -ds '' grids_list.txt ding0_grids__{}.pkl -wc --parallel
         '''
    parser = argparse.ArgumentParser(
        description="Commandline running" + \
                    "of eDisGo",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)

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
                                     action='store_true',
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

    parser.add_argument('-p', '--parallel',
                                     action='store_true',
                                     help='Parallel execution of multiple '
                                          'grids. Parallelization is provided '
                                          'by multiprocessing.')

    parser.add_argument('-w', '--workers',
                        nargs='?',
                        metavar='1..inf',
                        dest="workers",
                        type=int,
                        default=mp.cpu_count(),
                        help='Number of workers in parallel. In other words, '
                             'cores that are used for parallelization.')

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

    # get current time for output file names
    exec_time = pd.datetime.now().strftime('%Y-%m-%d_%H%M')

    logger = setup_logging(logfilename='test.log',
                           logfile_loglevel='debug',
                           console_loglevel='info')

    # get the list of files to run on
    if args.ding0_filename:
        ding0_file_list = [args.ding0_filename]

    elif args.ding0_dirglob:
        ding0_file_list = glob.glob(args.ding0_dirglob)

    elif args.ding0_dir_select:
        with open(args.ding0_dir_select[1], 'r') as file_handle:
            ding0_file_list_grid_district_numbers = list(file_handle)
            ding0_file_list_grid_district_numbers = [
                _.splitlines()[0] for _ in ding0_file_list_grid_district_numbers]

        ding0_file_list = map(lambda x: args.ding0_dir_select[0] +
                                        args.ding0_dir_select[2].format(x),
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

    if not args.parallel:
        for ding0_filename in ding0_file_list:
            grid_district = _get_griddistrict(ding0_filename)

            run_args = [ding0_filename]
            run_args.extend(run_args_opt_no_scenario)

            costs_before_geno_import, \
            grid_issues_before_geno_import, \
                costs, grid_issues = run_edisgo_twice(run_args)
            
            all_costs_before_geno_import.append(costs_before_geno_import)
            all_grid_issues_before_geno_import['grid'].append(grid_issues_before_geno_import['grid'])
            all_grid_issues_before_geno_import['msg'].append(grid_issues_before_geno_import['msg'])
            all_costs.append(costs)
            all_grid_issues['grid'].append(grid_issues['grid'])
            all_grid_issues['msg'].append(grid_issues['msg'])
    else:
        all_costs_before_geno_import, \
        all_grid_issues_before_geno_import, \
        all_costs, all_grid_issues = run_edisgo_pool(
            ding0_file_list,
            run_args_opt_no_scenario,
            args.workers,
            args.worker_lifetime)

    # consolidate costs for all the networks
    all_costs_before_geno_import = pd.concat(all_costs_before_geno_import,
                                             ignore_index=True)
    all_costs = pd.concat(all_costs, ignore_index=True)

    # write costs and error messages to csv files
    pd.DataFrame(all_grid_issues_before_geno_import).dropna(axis=0, how='all').to_csv(
        args.out_dir +
        exec_time + '_' +
        'grid_issues_before_geno_import.csv', index=False)

    with open(args.out_dir +
              exec_time + '_' + 'costs_before_geno_import.csv', 'a') as f:
        f.write(',,,# units: length in km,, total_costs in kEUR\n')
        all_costs_before_geno_import.to_csv(f, index=False)

    pd.DataFrame(all_grid_issues).dropna(axis=0, how='all').to_csv(args.out_dir + exec_time + '_' + \
                                         'grid_issues.csv', index=False)
    with open(args.out_dir +
              exec_time + '_' + 'costs.csv', 'a') as f:
        f.write(',,,# units: length in km,, total_costs in kEUR\n')
        all_costs.to_csv(f, index=False)


if __name__ == '__main__':
    pass