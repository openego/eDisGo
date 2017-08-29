from .check_tech_constraints import check_line_load, check_station_load
    #check_voltage, get_critical_line_loading, get_critical_voltage_at_nodes
from .reinforce_measures import reinforce_branches_current, \
    reinforce_branches_voltage, extend_distribution_substation
import logging

logger = logging.getLogger('ding0')


def reinforce_grid(network, results):
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
    # ToDo: get overlaoded stations (als dict mit maximaler Belastung {StationXY: 640kVA})
    # ToDo: check if MV/LV Trafo, warning if HV/MV Trafo
    #crit_lines = check_line_load(network, results.pfa_edges)
    #overloaded_stations = check_station_load(network, results.pfa_edges)
    # critical_branches, critical_stations = get_critical_line_loading(grid)

    # random overloaded station
    stations = network.mv_grid.graph.nodes_by_attribute('lv_station')
    overloaded_stations = {[_ for _ in stations
                            if len(_.transformers) >= 2][0]: 2000}

    # reinforce substations
    extend_distribution_substation(overloaded_stations)

    # if stations have been reinforced: run PF again and check if all
    # overloading problems for all stations were solved
    # if overloaded_stations:
    #     grid.network.run_powerflow(conn=None, method='onthefly')
    #     # check for overloaded stations
    #     # give error message if any overloaded stations are left

    # STEP 2: reinforce branches due to overloading

    # ToDo: get overlaoded lines (als dict mit relativer maximaler Belastung {LineXY: 1.2})
    # random overloaded line
    overloaded_line = list(network.mv_grid.graph.graph_edges())[0]['line']
    crit_lines = {overloaded_line: 2.3}

    # do reinforcement
    # ToDo: erst MV dann LV
    reinforce_branches_current(crit_lines)

    # if lines have been reinforced: run PF again and check if all
    # overloading problems for all lines were solved
    # if crit_lines:
    #     grid.network.run_powerflow(conn=None, method='onthefly')
    #     # check for overloaded lines
    #     # give error message if any overloaded lines are left


    # STEP 3: reinforce branches due to voltage problems

    # crit_nodes = check_voltage(grid, mode)
    # crit_nodes_count_prev_step = len(crit_nodes)

    # ToDo: get nodes with overvoltage (als dict mit grid und liste von Knoten {GridXY: [NodeA, NodeB]})
    # ToDo: Knoten nach abfallender Spannung sortieren, damit pro Netz geprüft werden kann, ob nächster krit. Knoten schon in path enthalten ist
    # random critical node
    lv_grid = list(network.mv_grid.lv_grids)[0]
    crit_nodes = {lv_grid: list(lv_grid.graph.nodes_by_attribute('load'))}

    # as long as there are voltage issues, do reinforcement
    while crit_nodes:
        # for every grid in crit_nodes do reinforcement
        for key in crit_nodes:
            reinforce_branches_voltage(key, crit_nodes[key])
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