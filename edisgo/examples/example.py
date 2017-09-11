from edisgo.grid.network import Network, Scenario, TimeSeries, Results
from edisgo.flex_opt import reinforce_grid
import os
import pickle
import pandas as pd
from ast import literal_eval
import numpy as np

timeseries = TimeSeries()
scenario = Scenario(timeseries=timeseries)

network = Network.import_from_ding0(
    os.path.join('data', 'ding0_grids__3545.pkl'),
    id='Test grid',
    scenario=scenario
)
# pickle.dump(network, open('test_network.pkl', 'wb'))
# network = pickle.load(open('test_network.pkl', 'rb'))

# Do non-linear power flow analysis with PyPSA (MV+LV)
# network.analyze()

# Print LV station secondary side voltage levels returned by PFA
# print(network.results.v_res(
#     network.mv_grid.graph.nodes_by_attribute('lv_station'), 'lv'))

# Print voltage levels for entire LV grid
# for attr in ['lv_station', 'load', 'generator', 'branch_tee']:
#     objs = []
#     for lv_grid in network.mv_grid.lv_grids:
#         objs.extend(lv_grid.graph.nodes_by_attribute(attr))
#     print("\n\n\n{}\n".format(attr))
#     print(network.results.v_res(
#         objs, 'lv'))

# Print voltage level of all nodes
# print(network.results.pfa_v_mag_pu)

# Print apparent power at lines
# print(network.results.s_res([_['line'] for _ in network.mv_grid.graph.graph_edges()]))

# Print voltage levels for all lines
# print(network.results.s_res())

# for now create results object
# ToDo: Werte in DataFrame als List oder Array?
# results = Results()
# results.pfa_edges = pd.read_csv('Exemplary_PyPSA_line_results.csv',
#                                 index_col=0,
#                                 converters={'p0': literal_eval,
#                                             'q0': literal_eval,
#                                             'p1': literal_eval,
#                                             'q1': literal_eval})
# results.pfa_edges['p0'] = results.pfa_edges['p0'].apply(lambda x: np.array(x))
# results.pfa_edges['q0'] = results.pfa_edges['q0'].apply(lambda x: np.array(x))
# results.pfa_edges['p1'] = results.pfa_edges['p1'].apply(lambda x: np.array(x))
# results.pfa_edges['q1'] = results.pfa_edges['q1'].apply(lambda x: np.array(x))
# results.pfa_nodes = pd.read_csv('Exemplary_PyPSA_bus_results.csv', index_col=0,
#                                 converters={'v_mag_pu': literal_eval})
# results.pfa_nodes['v_mag_pu'] = results.pfa_nodes['v_mag_pu'].apply(
#     lambda x: np.array(x))

# # MV generators
# gens = network.mv_grid.graph.nodes_by_attribute('generator')
# print('Generators in MV grid incl. aggregated generators from MV and LV')
# print('Type\tSubtype\tCapacity in kW')
# for gen in gens:
#     print("{type}\t{sub}\t{capacity}".format(
#         type=gen.type, sub=gen.subtype, capacity=gen.nominal_capacity))
# 
# # Load located in aggregated LAs
# print('\n\nAggregated load in LA adds up to\n')
# if network.mv_grid.graph.nodes_by_attribute('load'):
#     [print('\t{0}: {1} MWh'.format(
#         _,
#         network.mv_grid.graph.nodes_by_attribute('load')[0].consumption[_] / 1e3))
#         for _ in ['retail', 'industrial', 'agricultural', 'residential']]
# else:
#     print("O MWh")

# from

# liste aller lv grids
# [_ for _ in network.mv_grid.lv_grids]

# nx.draw_spectral(list(network.mv_grid.lv_grids)[0].graph)

# ToDo: wie halten wir fest, welche Betriebsmittel erneuert wurden, um hinterher Kosten berechnen zu können?
# ToDo: Parameter bei Komponenten einführen mit dem man feststellen kann, ob die Komponente bereits in einer ersten Maßnahme verstärkt oder ausgebaut wurde
# ToDo: config mit Standardbetriebsmitteln?
# ToDo: Abbruchkriterium einführen - Anzahl paralleler lines
