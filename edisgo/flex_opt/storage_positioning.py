import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra as \
    dijkstra_shortest_path_length
import pandas as pd
import numpy as np
from math import sqrt, ceil

from edisgo.grid import tools
from edisgo.grid.components import LVStation
from edisgo.flex_opt import check_tech_constraints, costs
from edisgo.tools import plots

import logging

logger = logging.getLogger('edisgo')


def one_storage_per_feeder(edisgo, storage_timeseries,
                           storage_nominal_power=None, **kwargs):
    """
    Allocates the given storage capacity to multiple smaller storages.

    For each feeder with load or voltage issues it is checked if integrating a
    storage will reduce peaks in the feeder, starting with the feeder with
    the highest theoretical grid expansion costs. A heuristic approach is used
    to estimate storage sizing and siting while storage operation is carried
    over from the given storage operation.

    Parameters
    -----------
    edisgo : :class:`~.grid.network.EDisGo`
    storage_timeseries : :pandas:`pandas.DataFrame<dataframe>`
        Total active and reactive power time series that will be allocated to
        the smaller storages in feeders with load or voltage issues. Columns of
        the dataframe are 'p' containing active power time series in kW and 'q'
        containing the reactive power time series in kvar. Index is a
        :pandas:`pandas.DatetimeIndex<datetimeindex>`.
    storage_nominal_power : :obj:`float` or None
        Nominal power in kW that will be allocated to the smaller storages in
        feeders with load or voltage issues. If no nominal power is provided
        the maximum active power given in `storage_timeseries` is used.
        Default: None.
    debug : :obj:`Boolean`, optional
        If dedug is True a dataframe with storage size and path to storage of
        all installed and possibly discarded storages is saved to a csv file
        and a plot with all storage positions is created and saved, both to the
        current working directory with filename `storage_results_{MVgrid_id}`.
        Default: False.
    check_costs_reduction : :obj:`Boolean` or :obj:`str`, optional
        This parameter specifies when and whether it should be checked if a
        storage reduced grid expansion costs or not. It can be used as a safety
        check but can be quite time consuming. Possible options are:

        * 'each_feeder'
          Costs reduction is checked for each feeder. If the storage did not
          reduce grid expansion costs it is discarded.
        * 'once'
          Costs reduction is checked after the total storage capacity is
          allocated to the feeders. If the storages did not reduce grid
          expansion costs they are all discarded.
        * False
          Costs reduction is never checked.

    """

    def _feeder_ranking(grid_expansion_costs):
        """
        Get feeder ranking from grid expansion costs DataFrame.

        MV feeders are ranked descending by grid expansion costs that are
        attributed to that feeder.

        Parameters
        ----------
        grid_expansion_costs : :pandas:`pandas.DataFrame<dataframe>`
            grid_expansion_costs DataFrame from :class:`~.grid.network.Results`
            of the copied edisgo object.

        Returns
        -------
        :pandas:`pandas.Series<series>`
            Series with ranked MV feeders (in the copied graph) of type
            :class:`~.grid.components.Line`. Feeders are ranked by total grid
            expansion costs of all measures conducted in the feeder. The
            feeder with the highest costs is in the first row and the feeder
            with the lowest costs in the last row.

        """
        return grid_expansion_costs.groupby(
            ['mv_feeder'], sort=False).sum().reset_index().sort_values(
            by=['total_costs'], ascending=False)['mv_feeder']

    def _shortest_path(node):
        if isinstance(node, LVStation):
            return len(nx.shortest_path(
                node.mv_grid.graph, node.mv_grid.station, node))
        else:
            return len(nx.shortest_path(
                node.grid.graph, node.grid.station, node))

    def _find_battery_node(edisgo, critical_lines_feeder,
                           critical_nodes_feeder):
        """
        Evaluates where to install the storage.

        Parameters
        -----------
        edisgo : :class:`~.grid.network.EDisGo`
            The original edisgo object.
        critical_lines_feeder : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe containing over-loaded lines in MV feeder, their maximum
            relative over-loading and the corresponding time step. See
            :func:`edisgo.flex_opt.check_tech_constraints.mv_line_load` for
            more information.
        critical_nodes_feeder : :obj:`list`
            List with all nodes in MV feeder with voltage issues.

        Returns
        -------
        :obj:`float`
            Node where storage is installed.

        """

        # if there are overloaded lines in the MV feeder the battery storage
        # will be installed at the node farthest away from the MV station
        if not critical_lines_feeder.empty:
            logger.debug("Storage positioning due to overload.")
            # dictionary with nodes and their corresponding path length to
            # MV station
            path_length_dict = {}
            for l in critical_lines_feeder.index:
                nodes = l.grid.graph.nodes_from_line(l)
                for node in nodes:
                    path_length_dict[node] = _shortest_path(node)
            # return node farthest away
            return [_ for _ in path_length_dict.keys()
                    if path_length_dict[_] == max(
                    path_length_dict.values())][0]

        # if there are voltage issues in the MV grid the battery storage will
        # be installed at the first node in path that exceeds 2/3 of the line
        # length from station to critical node with highest voltage deviation
        if critical_nodes_feeder:
            logger.debug("Storage positioning due to voltage issues.")
            node = critical_nodes_feeder[0]

            # get path length from station to critical node
            get_weight = lambda u, v, data: data['line'].length
            path_length = dijkstra_shortest_path_length(
                edisgo.network.mv_grid.graph,
                edisgo.network.mv_grid.station,
                get_weight, target=node)

            # find first node in path that exceeds 2/3 of the line length
            # from station to critical node farthest away from the station
            path = nx.shortest_path(edisgo.network.mv_grid.graph,
                                    edisgo.network.mv_grid.station,
                                    node)
            return next(j for j in path
                        if path_length[j] >= path_length[node] * 2 / 3)

        return None

    def _calc_storage_size(edisgo, feeder, max_storage_size):
        """
        Calculates storage size that reduces residual load.

        Parameters
        -----------
        edisgo : :class:`~.grid.network.EDisGo`
            The original edisgo object.
        feeder : :class:`~.grid.components.Line`
            MV feeder the storage will be connected to. The line object is an
            object from the copied graph.

        Returns
        -------
        :obj:`float`
            Storage size that reduced the residual load in the feeder.

        """
        step_size = 200
        sizes = [0] + np.arange(
            p_storage_min, max_storage_size + 0.5 * step_size, step_size)
        p_feeder = edisgo.network.results.pfa_p.loc[:, repr(feeder)]
        q_feeder = edisgo.network.results.pfa_q.loc[:, repr(feeder)]
        p_slack = edisgo.network.pypsa.generators_t.p.loc[
                  :, 'Generator_slack'] * 1e3

        # get sign of p and q
        l = edisgo.network.pypsa.lines.loc[repr(feeder), :]
        mv_station_bus = 'bus0' if l.loc['bus0'] == 'Bus_'.format(
            repr(edisgo.network.mv_grid.station)) else 'bus1'
        if mv_station_bus == 'bus0':
            diff = edisgo.network.pypsa.lines_t.p1.loc[:, repr(feeder)] - \
                   edisgo.network.pypsa.lines_t.p0.loc[:, repr(feeder)]
            diff_q = edisgo.network.pypsa.lines_t.q1.loc[:, repr(feeder)] - \
                     edisgo.network.pypsa.lines_t.q0.loc[:, repr(feeder)]
        else:
            diff = edisgo.network.pypsa.lines_t.p0.loc[:, repr(feeder)] - \
                   edisgo.network.pypsa.lines_t.p1.loc[:, repr(feeder)]
            diff_q = edisgo.network.pypsa.lines_t.q0.loc[:, repr(feeder)] - \
                     edisgo.network.pypsa.lines_t.q1.loc[:, repr(feeder)]
        p_sign = pd.Series([-1 if _ < 0 else 1 for _ in diff],
                           index=p_feeder.index)
        q_sign = pd.Series([-1 if _ < 0 else 1 for _ in diff_q],
                           index=p_feeder.index)

        # get allowed load factors per case
        lf = {'feedin_case': edisgo.network.config[
            'grid_expansion_load_factors']['mv_feedin_case_line'],
              'load_case': network.config[
            'grid_expansion_load_factors']['mv_load_case_line']}

        # calculate maximum apparent power for each storage size to find
        # storage size that minimizes apparent power in the feeder
        p_feeder = p_feeder.multiply(p_sign)
        q_feeder = q_feeder.multiply(q_sign)
        s_max = []
        for size in sizes:
            share = size / storage_nominal_power
            p_storage = storage_timeseries.p * share
            q_storage = storage_timeseries.q * share
            p_total = p_feeder + p_storage
            q_total = q_feeder + q_storage
            p_hv_mv_station = p_slack - p_storage
            lf_ts = p_hv_mv_station.apply(
                lambda _: lf['feedin_case'] if _ < 0 else lf['load_case'])
            s_max_ts = (p_total ** 2 + q_total ** 2).apply(
                sqrt).divide(lf_ts)
            s_max.append(max(s_max_ts))

        return sizes[pd.Series(s_max).idxmin()]

    def _critical_nodes_feeder(edisgo, feeder):
        """
        Returns all nodes in MV feeder with voltage issues.

        Parameters
        -----------
        edisgo : :class:`~.grid.network.EDisGo`
            The original edisgo object.
        feeder : :class:`~.grid.components.Line`
            MV feeder the storage will be connected to. The line object is an
            object from the copied graph.

        Returns
        -------
        :obj:`list`
            List with all nodes in MV feeder with voltage issues.

        """
        # get all nodes with voltage issues in MV grid
        critical_nodes = check_tech_constraints.mv_voltage_deviation(
            edisgo.network, voltage_levels='mv')
        if critical_nodes:
            critical_nodes = critical_nodes[edisgo.network.mv_grid]
        else:
            return []
        # filter nodes with voltage issues in feeder
        critical_nodes_feeder = []
        for n in critical_nodes.index:
            if repr(n.mv_feeder) == repr(feeder):
                critical_nodes_feeder.append(n)
        return critical_nodes_feeder

    def _critical_lines_feeder(edisgo, feeder):
        """
        Returns all lines in MV feeder with overload issues.

        Parameters
        -----------
        edisgo : :class:`~.grid.network.EDisGo`
            The original edisgo object.
        feeder : :class:`~.grid.components.Line`
            MV feeder the storage will be connected to. The line object is an
            object from the copied graph.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe containing over-loaded lines in MV feeder, their maximum
            relative over-loading and the corresponding time step. See
            :func:`edisgo.flex_opt.check_tech_constraints.mv_line_load` for
            more information.

        """
        # get all overloaded MV lines
        critical_lines = check_tech_constraints.mv_line_load(edisgo.network)
        # filter overloaded lines in feeder
        critical_lines_feeder = []
        for l in critical_lines.index:
            if repr(tools.get_mv_feeder_from_line(l)) == repr(feeder):
                critical_lines_feeder.append(l)
        return critical_lines.loc[critical_lines_feeder, :]

    def _estimate_new_number_of_lines(critical_lines_feeder):
        number_parallel_lines = 0
        for crit_line in critical_lines_feeder.index:
            number_parallel_lines += ceil(critical_lines_feeder.loc[
                crit_line, 'max_rel_overload'] * crit_line.quantity) - \
                                     crit_line.quantity
        return number_parallel_lines

    debug = kwargs.get('debug', False)
    check_costs_reduction = kwargs.get('check_costs_reduction', False)

    # global variables
    # minimum and maximum storage power to be connected to the MV grid
    p_storage_min = 300
    p_storage_max = 4500

    # remaining storage nominal power
    if storage_nominal_power is None:
        storage_nominal_power = max(abs(storage_timeseries.p))
    p_storage_remaining = storage_nominal_power

    if debug:
        feeder_repr = []
        storage_path = []
        storage_repr = []
        storage_size = []

    # rank MV feeders by grid expansion costs

    # conduct grid reinforcement on copied edisgo object on worst-case time
    # steps
    grid_expansion_results_init = edisgo.reinforce(
        copy_graph=True, timesteps_pfa='snapshot_analysis')

    # only analyse storage integration if there were any grid expansion needs
    if grid_expansion_results_init.equipment_changes.empty:
        logger.debug('No storage integration necessary since there are no '
                     'grid expansion needs.')
        return
    else:
        equipment_changes_reinforcement_init = \
            grid_expansion_results_init.equipment_changes.loc[
                grid_expansion_results_init.equipment_changes.iteration_step >
                0]
        total_grid_expansion_costs = \
            grid_expansion_results_init.grid_expansion_costs.total_costs.sum()
        if equipment_changes_reinforcement_init.empty:
            logger.debug('No storage integration necessary since there are no '
                         'grid expansion needs.')
            return
        else:
            network = equipment_changes_reinforcement_init.index[
                0].grid.network

    # calculate grid expansion costs without costs for new generators
    # to be used in feeder ranking
    grid_expansion_costs_feeder_ranking = costs.grid_expansion_costs(
        network, without_generator_import=True)

    ranked_feeders = _feeder_ranking(grid_expansion_costs_feeder_ranking)

    count = 1
    storage_obj_list = []
    total_grid_expansion_costs_new = 'not calculated'
    for feeder in ranked_feeders.values:
        logger.debug('Feeder: {}'.format(count))
        count += 1

        # first step: find node where storage will be installed

        critical_nodes_feeder = _critical_nodes_feeder(edisgo, feeder)
        critical_lines_feeder = _critical_lines_feeder(edisgo, feeder)

        # get node the storage will be connected to (in original graph)
        battery_node = _find_battery_node(edisgo, critical_lines_feeder,
                                          critical_nodes_feeder)

        if battery_node:

            # add to output lists
            if debug:
                feeder_repr.append(repr(feeder))
                storage_path.append(nx.shortest_path(
                    edisgo.network.mv_grid.graph,
                    edisgo.network.mv_grid.station,
                    battery_node))

            # second step: calculate storage size

            max_storage_size = min(p_storage_remaining, p_storage_max)
            p_storage = _calc_storage_size(edisgo, feeder, max_storage_size)

            # if p_storage is greater than or equal to the minimum storage
            # power required, do storage integration
            if p_storage >= p_storage_min:

                # third step: integrate storage

                share = p_storage / storage_nominal_power
                edisgo.integrate_storage(
                    timeseries=storage_timeseries.p * share,
                    position=battery_node,
                    voltage_level='mv',
                    timeseries_reactive_power=storage_timeseries.q * share)
                tools.assign_mv_feeder_to_nodes(edisgo.network.mv_grid)

                # get new storage object
                storage_obj = [_
                               for _ in
                               edisgo.network.mv_grid.graph.nodes_by_attribute(
                                   'storage') if _ in
                               edisgo.network.mv_grid.graph.neighbors(
                                   battery_node)][0]
                storage_obj_list.append(storage_obj)

                logger.debug(
                    'Storage with nominal power of {} kW connected to '
                    'node {} (path to HV/MV station {}).'.format(
                        p_storage, battery_node, nx.shortest_path(
                            battery_node.grid.graph, battery_node.grid.station,
                            battery_node)))

                # fourth step: check if storage integration reduced grid
                # reinforcement costs or number of issues

                if check_costs_reduction == 'each_feeder':

                    # calculate new grid expansion costs

                    grid_expansion_results_new = edisgo.reinforce(
                        copy_graph=True, timesteps_pfa='snapshot_analysis')

                    total_grid_expansion_costs_new = \
                        grid_expansion_results_new.grid_expansion_costs.\
                            total_costs.sum()

                    costs_diff = total_grid_expansion_costs - \
                                 total_grid_expansion_costs_new

                    if costs_diff > 0:
                        logger.debug(
                            'Storage integration in feeder {} reduced grid '
                            'expansion costs by {} kEuro.'.format(
                                feeder, costs_diff))

                        if debug:
                            storage_repr.append(repr(storage_obj))
                            storage_size.append(storage_obj.nominal_power)

                        total_grid_expansion_costs = \
                            total_grid_expansion_costs_new

                    else:
                        logger.debug(
                            'Storage integration in feeder {} did not reduce '
                            'grid expansion costs (costs increased by {} '
                            'kEuro).'.format(feeder, -costs_diff))

                        tools.disconnect_storage(edisgo.network, storage_obj)
                        p_storage = 0

                        if debug:
                            storage_repr.append(None)
                            storage_size.append(0)

                            edisgo.integrate_storage(
                                timeseries=storage_timeseries.p * 0,
                                position=battery_node,
                                voltage_level='mv',
                                timeseries_reactive_power=
                                storage_timeseries.q * 0)
                            tools.assign_mv_feeder_to_nodes(
                                edisgo.network.mv_grid)

                else:
                    number_parallel_lines_before = \
                        _estimate_new_number_of_lines(critical_lines_feeder)
                    edisgo.analyze()
                    critical_lines_feeder_new = _critical_lines_feeder(
                        edisgo, feeder)
                    critical_nodes_feeder_new = _critical_nodes_feeder(
                        edisgo, feeder)
                    number_parallel_lines = _estimate_new_number_of_lines(
                        critical_lines_feeder_new)

                    # if there are critical lines check if number of parallel
                    # lines was reduced
                    if not critical_lines_feeder.empty:
                        diff_lines = number_parallel_lines_before - \
                                     number_parallel_lines
                        # if it was not reduced check if there are critical
                        # nodes and if the number was reduced
                        if diff_lines <= 0:
                            # if there are no critical nodes remove storage
                            if not critical_nodes_feeder:
                                logger.debug(
                                    'Storage integration in feeder {} did not '
                                    'reduce number of critical lines (number '
                                    'increased by {}), storage '
                                    'is therefore removed.'.format(
                                        feeder, -diff_lines))

                                tools.disconnect_storage(edisgo.network,
                                                         storage_obj)
                                p_storage = 0

                                if debug:
                                    storage_repr.append(None)
                                    storage_size.append(0)

                                    edisgo.integrate_storage(
                                        timeseries=storage_timeseries.p * 0,
                                        position=battery_node,
                                        voltage_level='mv',
                                        timeseries_reactive_power=
                                        storage_timeseries.q * 0)
                                    tools.assign_mv_feeder_to_nodes(
                                        edisgo.network.mv_grid)
                            else:
                                logger.debug(
                                    'Critical nodes in feeder {} '
                                    'before and after storage integration: '
                                    '{} vs. {}'.format(
                                        feeder, critical_nodes_feeder,
                                        critical_nodes_feeder_new))
                                if debug:
                                    storage_repr.append(repr(storage_obj))
                                    storage_size.append(
                                        storage_obj.nominal_power)
                        else:
                            logger.debug(
                                'Storage integration in feeder {} reduced '
                                'number of critical lines.'.format(feeder))

                            if debug:
                                storage_repr.append(repr(storage_obj))
                                storage_size.append(storage_obj.nominal_power)

                    # if there are no critical lines
                    else:
                        logger.debug(
                            'Critical nodes in feeder {} '
                            'before and after storage integration: '
                            '{} vs. {}'.format(
                                feeder, critical_nodes_feeder,
                                critical_nodes_feeder_new))
                        if debug:
                            storage_repr.append(repr(storage_obj))
                            storage_size.append(storage_obj.nominal_power)

                # fifth step: if there is storage capacity left, rerun
                # the past steps for the next feeder in the ranking
                # list
                p_storage_remaining = p_storage_remaining - p_storage
                if not p_storage_remaining > p_storage_min:
                    break

            else:
                logger.debug('No storage integration in feeder {}.'.format(
                    feeder))

                if debug:
                    storage_repr.append(None)
                    storage_size.append(0)

                    edisgo.integrate_storage(
                        timeseries=storage_timeseries.p * 0,
                        position=battery_node,
                        voltage_level='mv',
                        timeseries_reactive_power=storage_timeseries.q * 0)
                    tools.assign_mv_feeder_to_nodes(edisgo.network.mv_grid)
        else:
            logger.debug('No storage integration in feeder {} because there '
                         'are neither overloading nor voltage issues.'.format(
                feeder))

            if debug:
                storage_repr.append(None)
                storage_size.append(0)
                feeder_repr.append(repr(feeder))
                storage_path.append([])

    if check_costs_reduction == 'once':
        # check costs reduction and discard all storages if costs were not
        # reduced
        grid_expansion_results_new = edisgo.reinforce(
            copy_graph=True, timesteps_pfa='snapshot_analysis')

        total_grid_expansion_costs_new = \
            grid_expansion_results_new.grid_expansion_costs. \
                total_costs.sum()

        costs_diff = total_grid_expansion_costs - \
                     total_grid_expansion_costs_new

        if costs_diff > 0:
            logger.info(
                'Storage integration in grid {} reduced grid '
                'expansion costs by {} kEuro.'.format(
                    edisgo.network.id, costs_diff))
        else:
            logger.info(
                'Storage integration in grid {} did not reduce '
                'grid expansion costs (costs increased by {} '
                'kEuro).'.format(edisgo.network.id, -costs_diff))

            for storage in storage_obj_list:
                tools.disconnect_storage(edisgo.network, storage)
    elif check_costs_reduction == 'each_feeder':
        # if costs redcution was checked after each storage only give out
        # total costs reduction
        if total_grid_expansion_costs_new == 'not calculated':
            costs_diff = 0
        else:
            total_grid_expansion_costs = grid_expansion_results_init.\
                grid_expansion_costs.total_costs.sum()
            costs_diff = total_grid_expansion_costs - \
                         total_grid_expansion_costs_new

        logger.info(
            'Storage integration in grid {} reduced grid '
            'expansion costs by {} kEuro.'.format(
                edisgo.network.id, costs_diff))

    if debug:
        plots.storage_size(edisgo.network.mv_grid, edisgo.network.pypsa,
                           filename='storage_results_{}.pdf'.format(
                               edisgo.network.id), lopf=False)
        storages_df = pd.DataFrame({'path': storage_path,
                                    'repr': storage_repr,
                                    'p_nom': storage_size},
                                   index=feeder_repr)
        storages_df.to_csv('storage_results_{}.csv'.format(edisgo.network.id))

    edisgo.network.results.storages_costs_reduction = pd.DataFrame(
        {'grid_expansion_costs_initial': total_grid_expansion_costs,
         'grid_expansion_costs_with_storages': total_grid_expansion_costs_new},
        index=[edisgo.network.id])
