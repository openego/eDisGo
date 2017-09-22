import sys
import pandas as pd
from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import reinforce_measures
import logging

logger = logging.getLogger('edisgo')


def reinforce_grid(network, while_counter_max=10):
    """ Evaluates grid reinforcement needs and performs measures. This function
        is the parent function for all grid reinforcements.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
    while_counter_max : int
        Maximum number of iterations when solving overvoltage problems, to
        prevent infinite grid expansion.

    Notes
    -----
    Vorgehen laut BW-Studie:
    * getrennte oder kombinierte Betrachtung von NS und MS muss noch entschieden
      werden, BW-Studie führt getrennte Betrachtung durch
    * Reihenfolge der Behebung von Grenzwertverletzungen:
    ** Trafo
    ** Spannung
    ** Leitungen
    * Thermische Belastung:
    ** vorhandene BM werden maximal durch ein weiteres gleiches BM verstärkt
    ** ist das nicht ausreichend, wird das BM durch beliebig viele Standard-
       BM ersetzt
    * Spannungsbandverletztung
    ** Strangauftrennung nach 2/3 der Distanz
    ** danach eventuell weitere Strangauftrennung wenn sinnvoll, sonst parallele
       BM

    Sonstiges:
    * nur Rückspeisefall
    ** NS: 20% Last, 85-100% Einspeisung, BM-Belastung 100%
    ** MS: 30% Last, 85-100% Einspeisung, BM-Belastung 100%
    * Spannungsbandaufteilung wie in Wuppertal Studie
    * bei Spannungsproblemen am Trafo wird nicht Trafo ausgebaut, sondern
      Leistung in der MS

    References
    ----------
    .. [1] dena VNS
    .. [2] Ackermann et al. (RP VNS)

    """

    def _add_lines_changes_to_equipment_changes():
        equipment, index, quantity = [], [], []
        for line, number_of_lines in lines_changes.items():
            equipment.append(line.type.name)
            index.append(line)
            quantity.append(number_of_lines)
        network.results.equipment_changes = \
            network.results.equipment_changes.append(
                pd.DataFrame(
                    {'iteration_step': [iteration_step] * len(
                        lines_changes),
                     'change': ['changed'] * len(lines_changes),
                     'equipment': equipment,
                     'quantity': quantity},
                    index=index))

    def _add_transformer_changes_to_equipment_changes(mode):
        for station, transformer_list in transformer_changes[mode].items():
            network.results.equipment_changes = \
                network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            transformer_list),
                         'change': [mode] * len(transformer_list),
                         'equipment': transformer_list,
                         'quantity': [1] * len(transformer_list)},
                        index=[station] * len(transformer_list)))

    # STEP 1: reinforce overloaded transformers
    iteration_step = 1

    logger.info('==> Check LV stations')

    # ToDo: check overloading of HV/MV Trafo?
    overloaded_stations = checks.mv_lv_station_load(network)

    if overloaded_stations:
        # reinforce substations
        transformer_changes = \
            reinforce_measures.extend_distribution_substation(
                network, overloaded_stations)
        # write added and removed transformers to results.equipment_changes
        _add_transformer_changes_to_equipment_changes('added')
        _add_transformer_changes_to_equipment_changes('removed')

    # STEP 2: reinforce branches due to overloading

    crit_lines_lv = checks.lv_line_load(network)
    crit_lines_mv = checks.mv_line_load(network)
    crit_lines = {**crit_lines_lv, **crit_lines_mv}

    # do reinforcement
    if crit_lines:
        lines_changes = reinforce_measures.reinforce_branches_overloading(
            network, crit_lines)
        # write changed lines to results.equipment_changes
        _add_lines_changes_to_equipment_changes()

    # run power flow analysis again and check if all overloading
    # problems were solved
    network.analyze()
    overloaded_stations = checks.mv_lv_station_load(network)
    if overloaded_stations:
        logger.error("==> Overloading issues of LV stations were not "
                     "solved in the first iteration step.")
        sys.exit()

    crit_lines_lv = checks.lv_line_load(network)
    crit_lines_mv = checks.mv_line_load(network)
    crit_lines = {**crit_lines_lv, **crit_lines_mv}
    if crit_lines:
        logger.error("==> Overloading issues of lines were not "
                     "solved in the first iteration step.")
        sys.exit()

    # STEP 3: reinforce branches due to voltage problems

    iteration_step += 1

    # solve voltage problems in MV grid
    crit_nodes = checks.mv_voltage_deviation(network)

    # as long as there are voltage issues, do reinforcement
    while_counter = 0
    while crit_nodes and while_counter < while_counter_max:
        # ToDo: get crit_nodes as objects instead of string
        # for now iterate through grid to find node for repr
        crit_nodes_objects = pd.Series()
        for node in network.mv_grid.graph.nodes():
            if repr(node) in crit_nodes[network.mv_grid].index:
                crit_nodes_objects = pd.concat(
                    [crit_nodes_objects,
                     pd.Series(crit_nodes[network.mv_grid].loc[repr(node)],
                               index=[node])])
                break

        lines_changes = reinforce_measures.reinforce_branches_overvoltage(
            network, network.mv_grid, crit_nodes_objects)
        # write changed lines to results.equipment_changes
        _add_lines_changes_to_equipment_changes()
        network.analyze()
        crit_nodes = checks.mv_voltage_deviation(network)
        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == while_counter_max and crit_nodes:
        logger.error("==> Voltage issues in MV grid were not solved.")
        sys.exit()

    logger.info('==> All voltage issues in MV grid are solved.')

    # solve voltage problems in LV grid
    crit_nodes = checks.lv_voltage_deviation(network)

    # as long as there are voltage issues, do reinforcement
    while_counter = 0
    while crit_nodes and while_counter < while_counter_max:
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
                    break

            lines_changes = reinforce_measures.reinforce_branches_overvoltage(
                network, grid, crit_nodes_objects)
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()
        network.analyze()
        crit_nodes = checks.lv_voltage_deviation(network)
        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == while_counter_max and crit_nodes:
        logger.error("==> Voltage issues in LV grids were not solved.")
        sys.exit()

    logger.info('==> All voltage issues in LV grids are solved.')

    # recheck over-loading of lines
    crit_lines_lv = checks.lv_line_load(network)
    crit_lines_mv = checks.mv_line_load(network)
    crit_lines = {**crit_lines_lv, **crit_lines_mv}

    # do reinforcement
    if crit_lines:
        lines_changes = reinforce_measures.reinforce_branches_overloading(
            network, crit_lines)
        # write changed lines to results.equipment_changes
        _add_lines_changes_to_equipment_changes()

