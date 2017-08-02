from .check_tech_constraints import check_load, check_voltage, \
    get_critical_line_loading, get_critical_voltage_at_nodes
from .reinforce_measures import reinforce_branches_current, \
    reinforce_branches_voltage, reinforce_lv_branches_overloading, \
    extend_distribution_substation
import logging

logger = logging.getLogger('ding0')


def reinforce_grid(network):
    """ Evaluates grid reinforcement needs and performs measures. This function
        is the parent function for all grid reinforcements.

    Parameters
    ----------
    network: edisgo network object

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

    References
    ----------
    .. [1] dena VNS
    .. [2] Ackermann et al. (RP VNS)


    """


    # STEP 1: reinforce transformers
    # ToDo: get overlaoded stations (als dict mit maximaler Belastung? {StationXY: 640kVA})
    # crit_branches, crit_stations = check_load(grid, mode)
    # critical_branches, critical_stations = get_critical_line_loading(grid)

    # random overloaded station
    stations = network.mv_grid.graph.nodes_by_attribute('lv_station')
    overloaded_stations = {[_ for _ in stations
                            if len(_.transformers) >= 2][0]: 2000}

    # reinforce substations
    extend_distribution_substation(overloaded_stations)

    # # if branches or stations have been reinforced: run PF again to check
    # # for voltage issues
    # if crit_branches or crit_stations:
    #     grid.network.run_powerflow(conn=None, method='onthefly')
    #
    #     # reinforcement of LV stations on voltage issues
    #     crit_stations_voltage = [_ for _ in crit_nodes
    #                              if isinstance(_['node'], LVStationDing0)]
    #     if crit_stations_voltage:
    #         extend_substation_voltage(crit_stations_voltage, grid_level='LV')

    # STEP 2: reinforce branches due to voltage problems

    # bei Spannungsproblemen am Trafo wird nicht Trafo ausgebaut, sondern
    # Leistung in der MS

    # crit_nodes = check_voltage(grid, mode)
    # crit_nodes_count_prev_step = len(crit_nodes)
    #
    # # as long as there are voltage issues, do reinforcement
    # while crit_nodes:
    #     # determine all branches on the way from HV-MV substation to crit. nodes
    #     crit_branches_v = grid.find_and_union_paths(grid.station(), crit_nodes)
    #
    #     # do reinforcement
    #     reinforce_branches_voltage(grid, crit_branches_v)
    #
    #     # run PF
    #     grid.network.run_powerflow(conn=None, method='onthefly')
    #
    #     crit_nodes = check_voltage(grid, mode)
    #
    #     # if there are critical nodes left but no larger cable available, stop reinforcement
    #     if len(crit_nodes) == crit_nodes_count_prev_step:
    #         logger.warning('==> There are {0} branches that cannot be '
    #                        'reinforced (no appropriate cable '
    #                        'available).'.format(
    #             len(grid.find_and_union_paths(grid.station(),
    #                 crit_nodes))))
    #         break
    #
    #     crit_nodes_count_prev_step = len(crit_nodes)
    #
    # if not crit_nodes:
    #     logger.info('==> All voltage issues in {mode} grid could be '
    #                 'solved using reinforcement.'.format(mode=mode))

    # # get node with over-voltage
    # crit_nodes = get_critical_voltage_at_nodes(grid)  # over-voltage issues
    #
    # crit_nodes_count_prev_step = len(crit_nodes)
    #
    # logger.info('{cnt_crit_branches} in {grid} have voltage issues'.format(
    #     cnt_crit_branches=crit_nodes_count_prev_step,
    #     grid=grid))
    #
    # # as long as there are voltage issues, do reinforcement
    # while crit_nodes:
    #     # determine all branches on the way from HV-MV substation to crit. nodes
    #     crit_branches_v = grid.find_and_union_paths(
    #         grid.station(),
    #         [_['node'] for _ in crit_nodes])
    #
    #     # do reinforcement
    #     reinforce_branches_voltage(grid, crit_branches_v, mode)
    #
    #     # get node with over-voltage
    #     crit_nodes = get_critical_voltage_at_nodes(grid)
    #
    #     # if there are critical nodes left but no larger cable available, stop reinforcement
    #     if len(crit_nodes) == crit_nodes_count_prev_step:
    #         logger.warning('==> There are {0} branches that cannot be '
    #                        'reinforced (no appropriate cable '
    #                        'available).'.format(
    #             len(crit_branches_v)))
    #         break
    #
    #     crit_nodes_count_prev_step = len(crit_nodes)
    #
    # if not crit_nodes:
    #     logger.info('==> All voltage issues in {mode} grid could be '
    #                 'solved using reinforcement.'.format(mode=mode))
    #
    # STEP 3: reinforce branches due to overloading
    # random overloaded line
    # overloaded_line = list(network.mv_grid.graph.graph_edges())[0]['line']
    # s_max_th = (3 ** 0.5 * overloaded_line._type['U_n'] *
    #             overloaded_line._type['I_max_th'])
    # s_max_th = s_max_th + 100  # invoke overload
    # crit_branches = {overloaded_line: s_max_th}

    # do reinforcement
    #reinforce_branches_current(network, crit_branches)

    # # reinforce overloaded lines by increasing size
    # unresolved = reinforce_lv_branches_overloading(grid, critical_branches)
    # logger.info(
    #     "Out of {crit_branches} with overloading {unresolved} remain "
    #     "with unresolved issues due to line overloading. "
    #     "LV grid: {grid}".format(
    #         crit_branches=len(critical_branches),
    #         unresolved=len(unresolved),
    #         grid=grid))