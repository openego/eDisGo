import pandas as pd
from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import reinforce_measures, exceptions
import logging

logger = logging.getLogger('edisgo')


def reinforce_grid(edisgo, max_while_iterations=10):
    """
    Evaluates grid reinforcement needs and performs measures.

    This function is the parent function for all grid reinforcements.

    Parameters
    ----------
    edisgo : :class:`~.grid.network.EDisGo`
        The eDisGo API object
    max_while_iterations : int
        Maximum number of times each while loop is conducted.

    Notes
    -----
    See :ref:`features-in-detail` for more information on how grid
    reinforcement is conducted.

    References
    ----------
    The methodology and parameters found on [DenaVNS]_ and [VNSRP]_.

    """

    def _add_lines_changes_to_equipment_changes():
        equipment, index, quantity = [], [], []
        for line, number_of_lines in lines_changes.items():
            equipment.append(line.type.name)
            index.append(line)
            quantity.append(number_of_lines)
        edisgo.network.results.equipment_changes = \
            edisgo.network.results.equipment_changes.append(
                pd.DataFrame(
                    {'iteration_step': [iteration_step] * len(
                        lines_changes),
                     'change': ['changed'] * len(lines_changes),
                     'equipment': equipment,
                     'quantity': quantity},
                    index=index))

    def _add_transformer_changes_to_equipment_changes(mode):
        for station, transformer_list in transformer_changes[mode].items():
            edisgo.network.results.equipment_changes = \
                edisgo.network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            transformer_list),
                         'change': [mode] * len(transformer_list),
                         'equipment': transformer_list,
                         'quantity': [1] * len(transformer_list)},
                        index=[station] * len(transformer_list)))

    # REINFORCE OVERLOADED TRANSFORMERS AND LINES
    iteration_step = 1

    logger.debug('==> Check station load.')
    overloaded_mv_station = checks.hv_mv_station_load(edisgo.network)
    overloaded_stations = checks.mv_lv_station_load(edisgo.network)
    logger.debug('==> Check line load.')
    crit_lines_lv = checks.lv_line_load(edisgo.network)
    crit_lines_mv = checks.mv_line_load(edisgo.network)
    crit_lines = {**crit_lines_lv, **crit_lines_mv}

    while_counter = 0
    while ((overloaded_mv_station or overloaded_stations or crit_lines) and
            while_counter < max_while_iterations):

        if overloaded_mv_station:
            # reinforce substations
            transformer_changes = \
                reinforce_measures.extend_substation_overloading(
                    edisgo.network, overloaded_mv_station)
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes('added')
            _add_transformer_changes_to_equipment_changes('removed')

        if overloaded_stations:
            # reinforce distribution substations
            transformer_changes = \
                reinforce_measures.extend_distribution_substation_overloading(
                    edisgo.network, overloaded_stations)
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes('added')
            _add_transformer_changes_to_equipment_changes('removed')

        if crit_lines:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_branches_overloading(
                edisgo.network, crit_lines)
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

        # run power flow analysis again and check if all over-loading
        # problems were solved
        logger.debug('==> Run power flow analysis.')
        edisgo.analyze()
        logger.debug('==> Recheck station load.')
        overloaded_mv_station = checks.hv_mv_station_load(edisgo.network)
        overloaded_stations = checks.mv_lv_station_load(edisgo.network)
        logger.debug('==> Recheck line load.')
        crit_lines_lv = checks.lv_line_load(edisgo.network)
        crit_lines_mv = checks.mv_line_load(edisgo.network)
        crit_lines = {**crit_lines_lv, **crit_lines_mv}

        iteration_step += 1
        while_counter += 1

    # check if all load problems were solved after maximum number of
    # iterations allowed
    if (while_counter == max_while_iterations and
            (crit_lines or overloaded_mv_station or overloaded_stations)):
        edisgo.network.results.unresolved_issues.update(crit_lines)
        edisgo.network.results.unresolved_issues.update(overloaded_stations)
        edisgo.network.results.unresolved_issues.update(overloaded_mv_station)
        raise exceptions.MaximumIterationError(
            "Overloading issues for the following lines could not be solved:"
            "{}".format(crit_lines))
    else:
        logger.info('==> Load issues in MV grid were solved in {} iteration '
                     'step(s).'.format(while_counter))

    # REINFORCE BRANCHES DUE TO VOLTAGE ISSUES
    iteration_step += 1

    # solve voltage problems in MV grid
    logger.debug('==> Check voltage in MV grid.')
    crit_nodes = checks.mv_voltage_deviation(edisgo.network)

    while_counter = 0
    while crit_nodes and while_counter < max_while_iterations:

        # ToDo: get crit_nodes as objects instead of string
        # for now iterate through grid to find node for repr
        crit_nodes_objects = pd.Series()
        for node in edisgo.network.mv_grid.graph.nodes():
            if repr(node) in crit_nodes[edisgo.network.mv_grid].index:
                crit_nodes_objects = pd.concat(
                    [crit_nodes_objects,
                     pd.Series(crit_nodes[edisgo.network.mv_grid].loc[
                                   repr(node)], index=[node])])
        crit_nodes_objects.sort_values(ascending=False, inplace=True)

        # reinforce lines
        lines_changes = reinforce_measures.reinforce_branches_overvoltage(
            edisgo.network, edisgo.network.mv_grid, crit_nodes_objects)
        # write changed lines to results.equipment_changes
        _add_lines_changes_to_equipment_changes()

        # run power flow analysis again and check if all over-voltage
        # problems were solved
        logger.debug('==> Run power flow analysis.')
        edisgo.analyze()
        logger.debug('==> Recheck voltage in MV grid.')
        crit_nodes = checks.mv_voltage_deviation(edisgo.network)

        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and crit_nodes:
        for k, v in crit_nodes.items():
            for i, d in v.iteritems():
                edisgo.network.results.unresolved_issues.update({repr(i): d})
        raise exceptions.MaximumIterationError(
            "Overvoltage issues for the following nodes in MV grid could "
            "not be solved: {}".format(crit_nodes))
    else:
        logger.info('==> Voltage issues in MV grid were solved in {} '
                     'iteration step(s).'.format(while_counter))

    # solve voltage problems at secondary side of LV stations
    logger.debug('==> Check voltage at secondary side of LV stations.')
    crit_stations = checks.lv_voltage_deviation(edisgo.network,
                                                mode='stations')

    while_counter = 0
    while crit_stations and while_counter < max_while_iterations:
        # reinforce distribution substations
        transformer_changes = \
            reinforce_measures.extend_distribution_substation_overvoltage(
                edisgo.network, crit_stations)
        # write added transformers to results.equipment_changes
        _add_transformer_changes_to_equipment_changes('added')

        logger.debug('==> Run power flow analysis.')
        edisgo.analyze()
        logger.debug('==> Recheck voltage at secondary side of LV stations.')
        crit_stations = checks.lv_voltage_deviation(edisgo.network,
                                                    mode='stations')

        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and crit_stations:
        for k, v in crit_stations.items():
            edisgo.network.results.unresolved_issues.update(
                {repr(k.station): v})
        raise exceptions.MaximumIterationError(
            "Overvoltage issues at busbar could not be solved for the "
            "following LV grids: {}".format(crit_stations))
    else:
        logger.info('==> Voltage issues at busbars in LV grids were solved '
                     'in {} iteration step(s).'.format(while_counter))

    # solve voltage problems in LV grids
    logger.debug('==> Check voltage in LV grids.')
    crit_nodes = checks.lv_voltage_deviation(edisgo.network)

    while_counter = 0
    while crit_nodes and while_counter < max_while_iterations:
        # for every grid in crit_nodes do reinforcement
        for grid in crit_nodes:

            # for now iterate through grid to find node for repr
            crit_nodes_objects = pd.Series()
            for node in grid.graph.nodes():
                if repr(node) in crit_nodes[grid].index:
                    crit_nodes_objects = pd.concat(
                        [crit_nodes_objects,
                         pd.Series(crit_nodes[grid].loc[repr(node)],
                                   index=[node])])
            crit_nodes_objects.sort_values(ascending=False, inplace=True)

            # reinforce lines
            lines_changes = reinforce_measures.reinforce_branches_overvoltage(
                edisgo.network, grid, crit_nodes_objects)
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

        logger.debug('==> Run power flow analysis.')
        edisgo.analyze()
        logger.debug('==> Recheck voltage in LV grids.')
        crit_nodes = checks.lv_voltage_deviation(edisgo.network)

        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and crit_nodes:
        for k, v in crit_nodes.items():
            for i, d in v.iteritems():
                edisgo.network.results.unresolved_issues.update({repr(i): d})
        raise exceptions.MaximumIterationError(
            "Overvoltage issues for the following nodes in LV grids could "
            "not be solved: {}".format(crit_nodes))
    else:
        logger.info(
            '==> Voltage issues in LV grids were solved '
            'in {} iteration step(s).'.format(while_counter))

    # RECHECK FOR OVERLOADED TRANSFORMERS AND LINES
    logger.debug('==> Recheck station load.')
    overloaded_mv_station = checks.hv_mv_station_load(edisgo.network)
    overloaded_stations = checks.mv_lv_station_load(edisgo.network)
    logger.debug('==> Recheck line load.')
    crit_lines_lv = checks.lv_line_load(edisgo.network)
    crit_lines_mv = checks.mv_line_load(edisgo.network)
    crit_lines = {**crit_lines_lv, **crit_lines_mv}

    while_counter = 0
    while ((overloaded_mv_station or overloaded_stations or crit_lines) and
            while_counter < max_while_iterations):

        if overloaded_mv_station:
            # reinforce substations
            transformer_changes = \
                reinforce_measures.extend_substation_overloading(
                    edisgo.network, overloaded_mv_station)
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes('added')
            _add_transformer_changes_to_equipment_changes('removed')

        if overloaded_stations:
            # reinforce substations
            transformer_changes = \
                reinforce_measures.extend_distribution_substation_overloading(
                    edisgo.network, overloaded_stations)
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes('added')
            _add_transformer_changes_to_equipment_changes('removed')

        if crit_lines:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_branches_overloading(
                edisgo.network, crit_lines)
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

        # run power flow analysis again and check if all over-loading
        # problems were solved
        logger.debug('==> Run power flow analysis.')
        edisgo.analyze()
        logger.debug('==> Recheck station load.')
        overloaded_mv_station = checks.hv_mv_station_load(edisgo.network)
        overloaded_stations = checks.mv_lv_station_load(edisgo.network)
        logger.debug('==> Recheck line load.')
        crit_lines_lv = checks.lv_line_load(edisgo.network)
        crit_lines_mv = checks.mv_line_load(edisgo.network)
        crit_lines = {**crit_lines_lv, **crit_lines_mv}

        iteration_step += 1
        while_counter += 1

    # check if all load problems were solved after maximum number of
    # iterations allowed
    if (while_counter == max_while_iterations and
            (crit_lines or overloaded_mv_station or overloaded_stations)):
        edisgo.network.results.unresolved_issues.update(crit_lines)
        edisgo.network.results.unresolved_issues.update(overloaded_stations)
        edisgo.network.results.unresolved_issues.update(overloaded_mv_station)
        raise exceptions.MaximumIterationError(
            "Overloading issues (after solving over-voltage issues) for the"
            "following lines could not be solved: {}".format(crit_lines))
    else:
        logger.info(
            '==> Load issues were rechecked and solved '
            'in {} iteration step(s).'.format(while_counter))
