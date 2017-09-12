import pandas as pd
from .check_tech_constraints import check_line_load, check_station_load, \
    check_voltage_lv #, get_critical_line_loading, get_critical_voltage_at_nodes
from .reinforce_measures import reinforce_branches_current, \
    reinforce_branches_voltage, extend_distribution_substation
import logging

logger = logging.getLogger('ding0')


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

    # STEP 1: reinforce overloaded transformers
    iteration_step = 1

    # ToDo: get overlaoded stations
    # ToDo: check if MV/LV Trafo, warning if HV/MV Trafo
    overloaded_stations = check_station_load(network)
    # random overloaded station
    stations = network.mv_grid.graph.nodes_by_attribute('lv_station')
    overloaded_stations = {[_ for _ in stations
                            if len(_.transformers) >= 2][0]: 1400}
    station_2 = [_ for _ in stations if len(_.transformers) >= 1][1]
    overloaded_stations[station_2] = 1000

    # reinforce substations
    transformer_changes = extend_distribution_substation(
        network, overloaded_stations)
    # write added and removed transformers to results.equipment_changes
    network.results.equipment_changes = \
        network.results.equipment_changes.append(
            pd.DataFrame(
                {'iteration_step': [iteration_step] * len(
                    transformer_changes['added']),
                 'change': ['added'] * len(transformer_changes['added'])},
                index=transformer_changes['added']))
    network.results.equipment_changes = \
        network.results.equipment_changes.append(
            pd.DataFrame(
                {'iteration_step': [iteration_step] * len(
                    transformer_changes['removed']),
                 'change': ['added'] * len(transformer_changes['removed'])},
                index=transformer_changes['removed']))

    # if stations have been reinforced: run PF again and check if all
    # overloading problems for all stations were solved
    # if overloaded_stations:
    #     grid.network.run_powerflow(conn=None, method='onthefly')
    #     # check for overloaded stations
    #     # give error message if any overloaded stations are left

    # STEP 2: reinforce branches due to overloading
    iteration_step += 1

    # ToDo: get overlaoded lines
    # random overloaded line
    overloaded_line = list(network.mv_grid.graph.graph_edges())[0]['line']
    crit_lines = {overloaded_line: 2.3}

    # do reinforcement
    lines_changes = reinforce_branches_current(network, crit_lines)
    # write added and removed transformers to results.equipment_changes
    network.results.equipment_changes = \
        network.results.equipment_changes.append(
            pd.DataFrame(
                {'iteration_step': [iteration_step] * len(
                    lines_changes['added']),
                 'change': ['added'] * len(lines_changes['added'])},
                index=lines_changes['added']))
    network.results.equipment_changes = \
        network.results.equipment_changes.append(
            pd.DataFrame(
                {'iteration_step': [iteration_step] * len(
                    lines_changes['removed']),
                 'change': ['added'] * len(lines_changes['removed'])},
                index=lines_changes['removed']))

    # if lines have been reinforced: run PF again and check if all
    # overloading problems for all lines were solved
    # if crit_lines:
    #     grid.network.run_powerflow(conn=None, method='onthefly')
    #     # check for overloaded lines
    #     # give error message if any overloaded lines are left


    # STEP 3: reinforce branches due to voltage problems

    #crit_nodes = check_voltage(network, results.pfa_nodes)
    # crit_nodes_count_prev_step = len(crit_nodes)
    # ToDo: erst Spannungsprobleme in MV lösen, dann LV
    # ToDo: get nodes with overvoltage (als dict mit grid und liste von Knoten {GridXY: [NodeA, NodeB]})
    # ToDo: Knoten nach abfallender Spannung sortieren, damit pro Netz geprüft werden kann, ob nächster krit. Knoten schon in path enthalten ist
    # random critical node
    lv_grid = list(network.mv_grid.lv_grids)[0]
    crit_nodes = {lv_grid: pd.Series(
        [1.12, 1.11],
        index=list(lv_grid.graph.nodes_by_attribute('load'))[0:2])}
    # as long as there are voltage issues, do reinforcement
    while crit_nodes:
        # for every grid in crit_nodes do reinforcement
        for grid in crit_nodes:
            reinforce_branches_voltage(network, crit_nodes[grid])
        crit_nodes = {}
        # # run PF
        # grid.network.run_powerflow(conn=None, method='onthefly')
        #
        # crit_nodes = check_voltage(grid, mode) # for MV
        # crit_nodes = get_critical_voltage_at_nodes(grid)  # for LV
        # # if there are critical nodes left but no larger cable available, stop reinforcement
        # if len(crit_nodes) == crit_nodes_count_prev_step:
        #     logger.warning('==> There are {0} branches that cannot be '
        #                    'reinforced (no appropriate cable '
        #                    'available).'.format(
        #         len(grid.find_and_union_paths(grid.station(),
        #             crit_nodes))))
        #     break
        #
        # crit_nodes_count_prev_step = len(crit_nodes)

    if not crit_nodes:
        logger.info('==> All voltage issues could be '
                    'solved using reinforcement.')