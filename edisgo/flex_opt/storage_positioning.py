import networkx as nx
import pandas as pd
import numpy as np
from math import sqrt
from edisgo.grid.grids import LVGrid, MVGrid
from edisgo.grid import tools
from edisgo.grid.components import LVStation, Transformer, Line
from edisgo.flex_opt import check_tech_constraints, costs
from edisgo.tools import plots

import logging

logger = logging.getLogger('edisgo')


def one_storage_per_feeder(edisgo, storage_timeseries,
                           storage_nominal_power=None,
                           debug=False):
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

    def feeder_ranking(grid_expansion_costs):
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

    def find_battery_node(edisgo_original, feeder):
        """
        Evaluates where to install the storage

        Returns
        --------
        Node where storage is installed.

        """

        # get overloaded MV lines in feeder
        critical_lines = check_tech_constraints.mv_line_load(
            edisgo_original.network)

        critical_lines_feeder = []
        for l in critical_lines.index:
            if repr(tools.get_mv_feeder_from_line(l)) == repr(feeder):
                critical_lines_feeder.append(l)

        # if there are overloaded lines in the MV feeder the battery storage
        # will be installed at the node farthest away from the MV station
        if critical_lines_feeder:
            # dictionary with nodes and their corresponding path length to
            # MV station
            path_length_dict = {}
            for l in critical_lines_feeder:
                nodes = l.grid.graph.nodes_from_line(l)
                for node in nodes:
                    path_length_dict[node] = _shortest_path(node)
            # get node farthest away
            logger.debug("Storage positioning due to overload.")
            return [_ for _ in path_length_dict.keys()
                    if path_length_dict[_] == max(
                    path_length_dict.values())][0]

        # get nodes with voltage issues in MV grid
        critical_nodes = check_tech_constraints.mv_voltage_deviation(
            edisgo_original.network, voltage_levels='mv')
        if critical_nodes:
            critical_nodes = critical_nodes[edisgo_original.network.mv_grid]
        else:

    def calc_storage_size(edisgo, feeder):
        sizes = [0] + np.arange(300, 4500, 200)
        p_feeder = edisgo.network.results.pfa_p.loc[:, repr(feeder)]
        q_feeder = edisgo.network.results.pfa_q.loc[:, repr(feeder)]
        # get sign of p and q
        l = edisgo.network.pypsa.lines.loc[repr(feeder), :]
        mv_station_bus = 'bus0' if l.loc['bus0'] == 'Bus_'.format(
            repr(edisgo.network.mv_grid.station)) else 'bus1'
        if mv_station_bus == 'bus0':
            diff = edisgo.network.pypsa.lines_t.p1.loc[:, repr(feeder)] - \
                   edisgo.network.pypsa.lines_t.p0.loc[:, repr(feeder)]
            diff_q = edisgo.network.pypsa.lines_t.p1.loc[:, repr(feeder)] - \
                     edisgo.network.pypsa.lines_t.p0.loc[:, repr(feeder)]
        else:
            diff = edisgo.network.pypsa.lines_t.p0.loc[:, repr(feeder)] - \
                   edisgo.network.pypsa.lines_t.p1.loc[:, repr(feeder)]
            diff_q = edisgo.network.pypsa.lines_t.q0.loc[:, repr(feeder)] - \
                     edisgo.network.pypsa.lines_t.q1.loc[:, repr(feeder)]
        p_sign = pd.Series([-1 if _ < 0 else 1 for _ in diff],
                           index=p_feeder.index)
        q_sign = pd.Series([-1 if _ < 0 else 1 for _ in diff_q],
                           index=p_feeder.index)
        p_feeder = p_feeder.multiply(p_sign)
        q_feeder = q_feeder.multiply(q_sign)
        s_max = []
        for size in sizes:
            share = size / storage_nominal_power
            p_storage = storage_timeseries.p * share
            q_storage = storage_timeseries.q * share
            p_total = p_feeder + p_storage
            q_total = q_feeder + q_storage
            s_max.append(max((p_total ** 2 + q_total ** 2).apply(sqrt)))
        return sizes[pd.Series(s_max).idxmin()]

    # global variables
    # minimum and maximum storage power to be connected to the MV grid
    p_storage_min = 300
    p_storage_max = 4500

    feeder_repr = []
    storage_path = []
    storage_repr = []
    storage_size = []

    # rank MV feeders by grid expansion costs

    # conduct grid reinforcement on copied edisgo object on worst-case time
    # steps
    grid_expansion_results_init = edisgo.reinforce(
        copy_graph=True, timesteps_pfa='snapshot_analysis')

    equipment_changes_reinforcement_init = \
        grid_expansion_results_init.equipment_changes.loc[
            grid_expansion_results_init.equipment_changes.iteration_step > 0]
    if equipment_changes_reinforcement_init.empty:
        logging.info('No storage integration necessary since there are no '
                     'grid expansion costs.')
        return
    else:
        network = equipment_changes_reinforcement_init.index[0].grid.network
    # calculate grid expansion costs without costs for new generators
    grid_expansion_costs_reinforce_init = costs.grid_expansion_costs(
        network, without_generator_import=True)

    ranked_feeders = feeder_ranking(grid_expansion_costs_reinforce_init)

    # remaining storage nominal power available for the feeder
    if storage_nominal_power is None:
        storage_nominal_power = max(abs(storage_timeseries.p))
    p_storage_remaining = storage_nominal_power

    count = 1
    for feeder in ranked_feeders.values:
        logger.debug('Feeder: {}'.format(count))
        count += 1

        # first step: find node where storage will be installed

        # get all reinforced components in respective feeder
        components = grid_expansion_costs_reinforce_init.loc[
            grid_expansion_costs_reinforce_init.mv_feeder == feeder].index
        # get node the storage will be connected to (in original graph)
        battery_node = find_battery_node(components, edisgo)

        # temporary - get critical MV lines in feeder
        critical_mv_lines = [_ for _ in components if isinstance(_, Line) and
                             isinstance(_.grid, MVGrid)]
        logger.debug("Critical MV lines in feeder: {}".format(
            critical_mv_lines))

        # second step: calculate storage size

        # add to output lists
        feeder_repr.append(repr(feeder))
        storage_path.append(nx.shortest_path(
            edisgo.network.mv_grid.graph, edisgo.network.mv_grid.station,
            battery_node))

        # analyze for all time steps
        edisgo.analyze()

        logger.debug('Check MV line load before storage integration.')
        tmp_crit_lines = check_tech_constraints.mv_line_load(edisgo.network)
        tmp_crit_lines_repr = [repr(_) for _ in tmp_crit_lines.index]
        tmp_crit_lines['repr'] = tmp_crit_lines_repr
        tmp_crit_lines = tmp_crit_lines.reset_index().set_index(['repr'])

        overload = []
        for l in critical_mv_lines:
            if repr(l) in tmp_crit_lines_repr:
                overload.append(
                    tmp_crit_lines.loc[repr(l), 'max_rel_overload'])
            else:
                overload.append(0)
        overload_df = pd.DataFrame({'max_rel_overload': overload},
                                   index=critical_mv_lines)
        logger.debug('Overload: {}'.format(overload_df))
        # get node the storage will be connected to (in original graph)
        battery_node = find_battery_node(edisgo, feeder)

        p_storage = calc_storage_size(edisgo, feeder)

        # third step: integrate storage

        # if p_storage is still minimal storage power p_storage_min this means
        # that the storage integration did not reduce apparent power and should
        # not be integrated
        if p_storage > p_storage_min:

            share = p_storage / storage_nominal_power
            edisgo.integrate_storage(
                timeseries=storage_timeseries.p * share,
                position=battery_node,
                voltage_level='mv',
                timeseries_reactive_power=storage_timeseries.q * share)

            # get new storage object
            storage_obj = [_
                           for _ in
                           edisgo.network.mv_grid.graph.nodes_by_attribute(
                               'storage') if _ in
                           edisgo.network.mv_grid.graph.neighbors(
                               battery_node)][0]

            edisgo.analyze()

            logger.debug('Check MV line load after storage integration.')
            tmp_crit_lines = check_tech_constraints.mv_line_load(
                edisgo.network)
            tmp_crit_lines_repr = [repr(_) for _ in tmp_crit_lines.index]
            tmp_crit_lines['repr'] = tmp_crit_lines_repr
            tmp_crit_lines = tmp_crit_lines.reset_index().set_index(['repr'])

            overload = []
            for l in critical_mv_lines:
                if repr(l) in tmp_crit_lines_repr:
                    overload.append(
                        tmp_crit_lines.loc[repr(l), 'max_rel_overload'])
                else:
                    overload.append(0)
            overload_df_new = pd.DataFrame({'max_rel_overload': overload},
                                           index=critical_mv_lines)
            logger.debug('Overload new: {}'.format(overload_df_new))

            # check if overload was reduced
            if overload_df_new.max_rel_overload.max() < \
                    overload_df.max_rel_overload.max():
                logger.debug(
                    'Storage with nominal power of {} kW connected to '
                    'node {} (path to HV/MV station {}).'.format(
                        p_storage, battery_node, nx.shortest_path(
                            battery_node.grid.graph, battery_node.grid.station,
                            battery_node)))

                # fourth step: check if storage integration reduced grid
                # reinforcement costs
                total_grid_expansion_costs = \
                    grid_expansion_costs_reinforce_init.total_costs.sum()

                grid_expansion_results_new = edisgo.reinforce(
                    copy_graph=True, timesteps_pfa='snapshot_analysis')

                # calculate new grid expansion costs (without costs for
                # generator import)
                equipment_changes_reinforcement = \
                    grid_expansion_results_new.equipment_changes.loc[
                        grid_expansion_results_new.equipment_changes.iteration_step
                        > 0]
                if not equipment_changes_reinforcement.empty:
                    network = equipment_changes_reinforcement.index[
                        0].grid.network
                    grid_expansion_costs_reinforce_new = costs.grid_expansion_costs(
                        network, without_generator_import=True)
                    total_grid_expansion_costs_new = \
                        grid_expansion_costs_reinforce_new.total_costs.sum()
                else:
                    total_grid_expansion_costs_new = 0

                costs_diff = total_grid_expansion_costs - \
                             total_grid_expansion_costs_new

                if costs_diff > 0:
                    logging.info(
                        'Storage integration in feeder {} reduced grid '
                        'expansion costs by {} kEuro.'.format(
                            feeder, costs_diff))
                    storage_repr.append(repr(storage_obj))
                    storage_size.append(storage_obj.nominal_power)

                    # fifth step: if there is storage capacity left, rerun the
                    # past steps for the next feeder in the ranking list
                    p_storage_remaining = p_storage_remaining - p_storage
                    if not p_storage_remaining > p_storage_min:
                        break

                else:
                    logging.info(
                        'Storage integration in feeder {} did not reduce '
                        'grid expansion costs (costs increased by {} '
                        'kEuro).'.format(feeder, -costs_diff))
                    tools.disconnect_storage(edisgo.network, storage_obj)
                    storage_repr.append(None)
                    storage_size.append(0)

                    edisgo.integrate_storage(
                        timeseries=storage_timeseries.p * 0,
                        position=battery_node,
                        voltage_level='mv',
                        timeseries_reactive_power=storage_timeseries.q * 0)

            else:
                logging.info(
                    'Line overloading was not reduced, therefore no storage '
                    'integration in feeder {}.'.format(feeder))
                tools.disconnect_storage(edisgo.network, storage_obj)
                storage_repr.append(None)
                storage_size.append(0)
                edisgo.integrate_storage(
                    timeseries=storage_timeseries.p * 0,
                    position=battery_node,
                    voltage_level='mv',
                    timeseries_reactive_power=storage_timeseries.q * 0)

        else:
            logging.info('No storage integration in feeder {}.'.format(feeder))
            storage_repr.append(None)
            storage_size.append(0)

            edisgo.integrate_storage(
                timeseries=storage_timeseries.p * 0,
                position=battery_node,
                voltage_level='mv',
                timeseries_reactive_power=storage_timeseries.q * 0)

    if debug:
        plots.storage_size(edisgo.network.mv_grid, edisgo.network.pypsa,
                           filename='storage_results_{}.pdf'.format(
                               edisgo.network.id), lopf=False)
        storages_df = pd.DataFrame({'path': storage_path,
                                    'repr': storage_repr,
                                    'p_nom': storage_size},
                                   index=feeder_repr)
        storages_df.to_csv('storage_results_{}.csv'.format(edisgo.network.id))