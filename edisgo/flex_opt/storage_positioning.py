import networkx as nx
from math import sqrt
from edisgo.grid.grids import LVGrid, MVGrid
from edisgo.grid.components import LVStation, Transformer, Line
from edisgo.flex_opt import check_tech_constraints, costs

import logging
logger = logging.getLogger('edisgo')


def one_storage_per_feeder(edisgo, storage_timeseries,
                           storage_nominal_power=None,
                           storage_parameters=None):
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

    def find_battery_node(components, edisgo_original):
        """
        Evaluates where to install the storage

        Returns
        --------
        Node where storage is installed.

        """
        # nodes_mv will contain all nodes of MV lines with over-loading or
        # over-voltage issues; lv_stations will contain all stations of LV
        # grids with over-loading or over-voltage issues
        nodes_mv = []
        lv_stations = []
        for comp in components:
            if isinstance(comp, Transformer) or isinstance(comp.grid, LVGrid):
                lv_stations.append(comp.grid.station)
            else:
                nodes_mv.extend(list(comp.grid.graph.nodes_from_line(comp)))

        # if there are nodes with issues in the MV grid the battery storage
        # will be installed at the node farthest away from the MV station
        if nodes_mv:
            # dictionary with nodes and their corresponding path length to
            # MV station
            path_length_dict = {}
            for node in nodes_mv:
                if isinstance(node, LVStation):
                    path_length_dict[node] = len(
                        nx.shortest_path(node.mv_grid.graph,
                                         node.mv_grid.station, node))
                else:
                    path_length_dict[node] = len(
                        nx.shortest_path(node.grid.graph, node.grid.station,
                                         node))
            # find node farthest away
            battery_node = [_ for _ in path_length_dict.keys()
                            if path_length_dict[_] ==
                            max(path_length_dict.values())][0]
        # if there are only issues in the LV grids find LV station with issues
        # closest to MV station
        else:
            # calculate path lengths
            path_length_dict = {}
            for node in lv_stations:
                path_length_dict[node] = len(nx.shortest_path(
                    node.mv_grid.graph, node.mv_grid.station, node))
            # find closest node
            battery_node = [_ for _ in path_length_dict.keys()
                            if path_length_dict[_] ==
                            min(path_length_dict.values())][0]

        # assign battery_node node the corresponding node in the original graph
        nodes = edisgo_original.network.mv_grid.graph.nodes()
        for node in nodes:
            if repr(node) == repr(battery_node):
                battery_node = node
        # if no node from original graph could be assigned raise error
        if battery_node not in nodes:
            raise ValueError("Could not assign battery node {} from the "
                             "copied graph to a node in the original "
                             "graph.".format(battery_node))

        return battery_node

    def calculate_storage_nominal_power():
        # actual apparent power
        pfa_p = edisgo.network.results.pfa_p[repr(battery_line)]
        pfa_q = edisgo.network.results.pfa_q[repr(battery_line)]

        s_line_max = max((pfa_p ** 2 + pfa_q ** 2).apply(sqrt))
        # start iteration at minimum needed power to be connected to the MV
        # grid
        p_storage = p_storage_min
        # set very high value to go into while loop
        s_line_max_previous_iteration = 10 ** 4

        while s_line_max < s_line_max_previous_iteration and p_storage < p_storage_remaining and p_storage < p_storage_max:
            # calculate share of storage of whole storage
            share = p_storage / storage_nominal_power
            # calculate storage p and q
            p = storage_timeseries.p * share
            q = storage_timeseries.q * share
            # calculate new line p and q
            line_p = pfa_p + p
            line_q = pfa_q + q
            s_line_max_previous_iteration = s_line_max
            s_line_max = max((line_p ** 2 + line_q ** 2).apply(sqrt))
            # if apparent power of line decreased
            if s_line_max < s_line_max_previous_iteration:
                # continue iteration process
                # increase storage capacity
                p_storage += 100
                print(p_storage, s_line_max)
        return p_storage

    def calculate_storage_nominal_power_2(battery_line):
        """
        Methodology to calculate storage nominal power

        Returns
        --------

        """
        # find critical time steps (critical time steps are time steps where
        # storage discharges in feed-in case and charges in load case)
        timesteps_load_feedin_case = \
            edisgo.network.timeseries.timesteps_load_feedin_case
        critical_timesteps = [
            _ for _ in storage_timeseries.index
            if (timesteps_load_feedin_case.loc[_, 'case'] == 'feedin_case' and
                storage_timeseries.loc[_, 'p'] > 0) or
               (timesteps_load_feedin_case.loc[_, 'case'] == 'load_case' and
                storage_timeseries.loc[_, 'p'] < 0)]

        # allowed apparent power
        i_line_allowed_per_case = {}
        i_line_allowed_per_case['feedin_case'] = \
            battery_line.type['I_max_th'] * battery_line.quantity * \
            edisgo.network.config['grid_expansion_load_factors'][
                'mv_feedin_case_line']
        i_line_allowed_per_case['load_case'] = \
            battery_line.type['I_max_th'] * battery_line.quantity * \
            edisgo.network.config['grid_expansion_load_factors'][
                'mv_load_case_line']
        # maximum allowed line load in each time step
        i_allowed = edisgo.network.timeseries. \
            timesteps_load_feedin_case.case.apply(
            lambda _: i_line_allowed_per_case[_])
        u_allowed = battery_line.type['U_n']
        # ToDo: correct?
        s_allowed = sqrt(3) / 1000 * i_allowed * u_allowed

        if critical_timesteps:
            # actual apparent power
            pfa_p = edisgo.network.results.pfa_p[repr(battery_line)]
            pfa_q = edisgo.network.results.pfa_q[repr(battery_line)]
            pfa_s = (pfa_p ** 2 + pfa_q ** 2).apply(sqrt)

            # ToDo: p muss noch aus s und q berechnet werden, wenn q != 0
            p = min((s_allowed - pfa_s).loc[critical_timesteps])
            if p > p_storage_min:
                p_storage = p
                if p_storage > p_storage_max:
                    p_storage = p_storage_max
        else:
            # allow maximal allowed line load
            p = max(s_allowed)
        return p_storage



    # rank MV feeders by grid expansion costs
    # conduct grid reinforcement on copied edisgo object on worst-case time
    # steps
    # grid_expansion_results = edisgo.reinforce(
    #     copy_graph=True, timesteps_pfa='snapshot_analysis')
    # ToDo: delete once implementation is finished
    import pickle
    edisgo_reinforce = pickle.load(open('edisgo_reinforce_239.pkl', 'rb'))
    grid_expansion_results = edisgo_reinforce.network.results

    equipment_changes_reinforcement = \
        grid_expansion_results.equipment_changes.loc[
            grid_expansion_results.equipment_changes.iteration_step > 0]
    if equipment_changes_reinforcement.empty:
        logging.info('No storage integration necessary since there are no '
                     'grid expansion costs.')
        return
    else:
        network = equipment_changes_reinforcement.index[0].grid.network
    grid_expansion_costs_reinforce = costs.grid_expansion_costs(
        network, without_generator_import=True)
    ranked_feeders = feeder_ranking(grid_expansion_costs_reinforce)

    # remaining storage nominal power available for the feeder
    if storage_nominal_power is None:
        storage_nominal_power = max(abs(storage_timeseries.p))
    p_storage_remaining = storage_nominal_power
    # minimum storage power to be connected to the MV grid
    p_storage_min = 300
    p_storage_max = 4500
    count = 1
    for feeder in ranked_feeders.values:
        print('Feeder: {}'.format(count))
        count += 1

        # first step: find node where storage will be installed

        # get all reinforced components in respective feeder
        components = grid_expansion_costs_reinforce.loc[
            grid_expansion_costs_reinforce.mv_feeder == feeder].index
        # get node the storage will be connected to (in original graph)
        battery_node = find_battery_node(components, edisgo)

        # temporary - get critical MV lines in feeder
        critical_mv_lines = [_ for _ in components if isinstance(_, Line) and
                             isinstance(_.grid, MVGrid)]
        print("Critical MV lines in feeder: {}".format(critical_mv_lines))

        # second step: estimate residual line load to calculate size of battery

        # get line (in original graph) from node the storage will be connected
        # to to next node towards the MV station
        path_to_battery_node = nx.shortest_path(
            edisgo.network.mv_grid.graph, edisgo.network.mv_grid.station,
            battery_node)
        battery_line = edisgo.network.mv_grid.graph.line_from_nodes(
            path_to_battery_node[-1], path_to_battery_node[-2])

        # analyze for all time steps
        edisgo.analyze()

        # temporary - dump results
        edisgo.network.results.i_res.to_csv('i_res_{}.csv'.format(feeder))
        edisgo.network.results.s_res().to_csv('s_res_{}.csv'.format(feeder))
        edisgo.network.results.v_res(
            nodes=edisgo.network.mv_grid.graph.nodes(), level='mv').to_csv(
            'v_res_{}.csv'.format(feeder))
        print('Check MV line load.')
        print(check_tech_constraints.mv_line_load(edisgo.network))

        p_storage = calculate_storage_nominal_power()

        # third step: integrate storage

        # if p_storage is still minimal storage power p_storage_min this means
        # that the storage integration did not reduce apparent power and should
        # not be integrated
        if p_storage > p_storage_min:

            share = p_storage / storage_nominal_power
            edisgo.integrate_storage(timeseries=storage_timeseries.p * share,
                                 position=battery_node,
                                 voltage_level='mv',
                                 timeseries_reactive_power=storage_timeseries.q * share)

            logging.info('Storage with nominal power of {} kW connected to node '
                          '{} (path to HV/MV station {}).'.format(
                p_storage, battery_node, nx.shortest_path(
                    battery_node.grid.graph, battery_node.grid.station,
                    battery_node)))

            # temporary - dump results
            edisgo.analyze()
            edisgo.network.results.i_res.to_csv('i_res_{}_post.csv'.format(feeder))
            edisgo.network.results.s_res().to_csv('s_res_{}_post.csv'.format(feeder))
            edisgo.network.results.v_res(
                nodes=edisgo.network.mv_grid.graph.nodes(), level='mv').to_csv(
                'v_res_{}_post.csv'.format(feeder))
            print('Check MV line load after storage integration.')
            print(check_tech_constraints.mv_line_load(edisgo.network))

            # fourth step: check if storage integration reduced grid reinforcement
            # costs
            total_grid_expansion_costs = \
                grid_expansion_costs_reinforce.total_costs.sum()

            grid_expansion_results_new = edisgo.reinforce(
                copy_graph=True, timesteps_pfa='snapshot_analysis')

            equipment_changes_reinforcement = \
                grid_expansion_results_new.equipment_changes.loc[
                    grid_expansion_results_new.equipment_changes.iteration_step > 0]
            if not equipment_changes_reinforcement.empty:
                network = equipment_changes_reinforcement.index[0].grid.network
                grid_expansion_costs_reinforce_new = costs.grid_expansion_costs(
                    network, without_generator_import=True)
                total_grid_expansion_costs_new = \
                    grid_expansion_costs_reinforce.total_costs.sum()
            else:
                total_grid_expansion_costs_new = 0


            costs_diff = total_grid_expansion_costs - \
                         total_grid_expansion_costs_new

            if costs_diff > 0:
                logging.info('Storage integration in feeder {} reduced grid '
                             'expansion costs by {} kEuro.'.format(
                    feeder, costs_diff))
            else:
                logging.info('Storage integration in feeder {} did not reduce '
                             'grid expansion costs (costs increased by {} '
                             'kEuro).'.format(feeder, -costs_diff))

            # fifth step: if there is storage capacity left, rerun the past steps
            # for the next feeder in the ranking list
            p_storage_remaining = p_storage_remaining - p_storage
            if not p_storage_remaining > p_storage_min:
                break
        else:
            logging.info('No storage integration in feeder {}.'.format(feeder))