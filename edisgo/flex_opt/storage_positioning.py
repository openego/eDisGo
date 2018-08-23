import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra as \
    dijkstra_shortest_path_length
import pandas as pd
import numpy as np
from math import sqrt

from edisgo.grid import tools
from edisgo.grid.components import LVStation
from edisgo.flex_opt import check_tech_constraints, costs
from edisgo.tools import plots

import logging

logger = logging.getLogger('edisgo')


def one_storage_per_feeder(edisgo, storage_timeseries,
                           storage_nominal_power=None,
                           debug=True, check_costs_reduction=False):
    """
    Parameters
    -----------
    edisgo : :class:`~.grid.network.EDisGo`
    storage_parameters : :obj:`dict`
        Dictionary with storage parameters. See
        :class:`~.grid.network.StorageControl` class definition for more
        information.
    storage_timeseries : :pandas:`pandas.DataFrame<dataframe>`
        p and q in kW and kvar of total storage
    storage_power : in kW of total storage

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

    def _find_battery_node(edisgo, feeder):
        """
        Evaluates where to install the storage.

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
            Node where storage is installed.

        """

        # get overloaded MV lines in feeder
        critical_lines = check_tech_constraints.mv_line_load(edisgo.network)

        critical_lines_feeder = []
        for l in critical_lines.index:
            if repr(tools.get_mv_feeder_from_line(l)) == repr(feeder):
                critical_lines_feeder.append(l)

        # if there are overloaded lines in the MV feeder the battery storage
        # will be installed at the node farthest away from the MV station
        if critical_lines_feeder:
            logger.debug("Storage positioning due to overload.")
            # dictionary with nodes and their corresponding path length to
            # MV station
            path_length_dict = {}
            for l in critical_lines_feeder:
                nodes = l.grid.graph.nodes_from_line(l)
                for node in nodes:
                    path_length_dict[node] = _shortest_path(node)
            # return node farthest away
            return [_ for _ in path_length_dict.keys()
                    if path_length_dict[_] == max(
                    path_length_dict.values())][0]

        # get nodes with voltage issues in MV grid
        critical_nodes = check_tech_constraints.mv_voltage_deviation(
            edisgo.network, voltage_levels='mv')
        if critical_nodes:
            critical_nodes = critical_nodes[edisgo.network.mv_grid]
        else:
            return None

        critical_nodes_feeder = []
        for n in critical_nodes.index:
            if repr(n.mv_feeder) == repr(feeder):
                critical_nodes_feeder.append(n)

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
    equipment_changes_reinforcement_init = \
        grid_expansion_results_init.equipment_changes.loc[
            grid_expansion_results_init.equipment_changes.iteration_step > 0]
    total_grid_expansion_costs = \
        grid_expansion_results_init.grid_expansion_costs.total_costs.sum()
    print('costs initial: {}'.format(total_grid_expansion_costs))
    if equipment_changes_reinforcement_init.empty:
        logger.debug('No storage integration necessary since there are no '
                     'grid expansion costs.')
        return
    else:
        network = equipment_changes_reinforcement_init.index[0].grid.network

    # calculate grid expansion costs without costs for new generators
    # to be used in feeder ranking
    grid_expansion_costs_feeder_ranking = costs.grid_expansion_costs(
        network, without_generator_import=True)

    ranked_feeders = _feeder_ranking(grid_expansion_costs_feeder_ranking)

    # analyze for all time steps
    edisgo.analyze()

    count = 1
    storage_obj_list = []
    for feeder in ranked_feeders.values:
        logger.debug('Feeder: {}'.format(count))
        count += 1

        # first step: find node where storage will be installed

        # get node the storage will be connected to (in original graph)
        battery_node = _find_battery_node(edisgo, feeder)

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
                edisgo.analyze()

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
                # reinforcement costs

                if check_costs_reduction:

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

    # if cost reduction was not checked after each storage integration check it
    # now
    if not check_costs_reduction:
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
    else:
        if total_grid_expansion_costs_new is None:
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
