import sys
import pandas as pd
from edisgo.flex_opt import check_tech_constraints as checks
from .reinforce_measures import reinforce_branches_current, \
    reinforce_branches_voltage, extend_distribution_substation
import logging

logger = logging.getLogger('edisgo')


def reinforce_grid(network):
    """ Evaluates grid reinforcement needs and performs measures. This function
        is the parent function for all grid reinforcements.

    Parameters
    ----------
    network: edisgo network object
    results: edisgo results object

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

    # ToDo: Move appending added and removed comp. to function
    # STEP 1: reinforce overloaded transformers
    iteration_step = 1

    logger.info('==> Check LV stations')

    # ToDo: check overloading of HV/MV Trafo?
    overloaded_stations = checks.mv_lv_station_load(network)

    # ToDo: remove at some point
    # random overloaded station
    stations = network.mv_grid.graph.nodes_by_attribute('lv_station')
    overloaded_stations = {[_ for _ in stations
                            if len(_.transformers) >= 2][0]: 1400}
    station_2 = [_ for _ in stations if len(_.transformers) >= 1][1]
    overloaded_stations[station_2] = 3000

    if overloaded_stations:
        # reinforce substations
        transformer_changes = extend_distribution_substation(
            network, overloaded_stations)
        # write added and removed transformers to results.equipment_changes
        for station, transformer_list in transformer_changes['added'].items():
            network.results.equipment_changes = \
                network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            transformer_list),
                         'change': ['added'] * len(transformer_list),
                         'equipment': transformer_list},
                        index=[station] * len(transformer_list)))
        for station, transformer_list in \
                transformer_changes['removed'].items():
            network.results.equipment_changes = \
                network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            transformer_list),
                         'change': ['removed'] * len(transformer_list),
                         'equipment': transformer_list},
                        index=[station] * len(transformer_list)))

    # STEP 2: reinforce branches due to overloading
    iteration_step += 1

    crit_lines_lv = checks.lv_line_load(network)
    crit_lines_mv = checks.mv_line_load(network)
    crit_lines = {**crit_lines_lv, **crit_lines_mv}

    # # ToDo: remove at some point
    # # random overloaded line
    # overloaded_line = list(network.mv_grid.graph.graph_edges())[0]['line']
    # crit_lines = {overloaded_line: 2.3}

    # do reinforcement
    if crit_lines:
        lines_changes = reinforce_branches_current(network, crit_lines)
        # write added and removed lines to results.equipment_changes
        # ToDo: what makes sense in equipment?
        if lines_changes['added']:
            network.results.equipment_changes = \
                network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            lines_changes['added']),
                         'change': ['added'] * len(lines_changes['added']),
                         'equipment': lines_changes['added']},
                        index=lines_changes['added']))
        if lines_changes['removed']:
            network.results.equipment_changes = \
                network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            lines_changes['removed']),
                         'change': ['removed'] * len(lines_changes['removed']),
                         'equipment': lines_changes['removed']},
                        index=lines_changes['removed']))

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
    # ToDo: crit_nodes needs to be list or dict
    crit_nodes = checks.mv_voltage_deviation(network)

    # as long as there are voltage issues, do reinforcement
    while crit_nodes:
        lines_changes = reinforce_branches_voltage(network, crit_nodes)
        # ToDo: what makes sense in equipment?
        if lines_changes['added']:
            network.results.equipment_changes = \
                network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            lines_changes['added']),
                         'change': ['added'] * len(lines_changes['added']),
                         'equipment': lines_changes['added']},
                        index=lines_changes['added']))
        if lines_changes['removed']:
            network.results.equipment_changes = \
                network.results.equipment_changes.append(
                    pd.DataFrame(
                        {'iteration_step': [iteration_step] * len(
                            lines_changes['removed']),
                         'change': ['removed'] * len(lines_changes['removed']),
                         'equipment': lines_changes['removed']},
                        index=lines_changes['removed']))
        network.analyze()
        iteration_step += 1
        crit_nodes = checks.mv_voltage_deviation(network)

    logger.info('==> All voltage issues in MV grid are solved.')

    # solve voltage problems in LV grid
    crit_nodes = checks.lv_voltage_deviation(network)

    # as long as there are voltage issues, do reinforcement
    while crit_nodes:
        # for every grid in crit_nodes do reinforcement
        for grid in crit_nodes:
            lines_changes = reinforce_branches_voltage(
                network, crit_nodes[grid])
            # ToDo: what makes sense in equipment?
            if lines_changes['added']:
                network.results.equipment_changes = \
                    network.results.equipment_changes.append(
                        pd.DataFrame(
                            {'iteration_step': [iteration_step] * len(
                                lines_changes['added']),
                             'change': ['added'] * len(lines_changes['added']),
                             'equipment': lines_changes['added']},
                            index=lines_changes['added']))
            if lines_changes['removed']:
                network.results.equipment_changes = \
                    network.results.equipment_changes.append(
                        pd.DataFrame(
                            {'iteration_step': [iteration_step] * len(
                                lines_changes['removed']),
                             'change': ['removed'] * len(
                                 lines_changes['removed']),
                             'equipment': lines_changes['removed']},
                            index=lines_changes['removed']))
        network.analyze()
        iteration_step += 1
        crit_nodes = checks.lv_voltage_deviation(network)

    logger.info('==> All voltage issues in LV grids are solved.')
