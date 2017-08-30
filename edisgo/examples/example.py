from edisgo.grid.network import Network, Results
from edisgo.grid_expansion import reinforce_grid
import os
import pickle
import pandas as pd
from ast import literal_eval
import numpy as np

network = Network.import_from_dingo('ding0_grids_example.pkl')
genos = network.import_generators()
#pickle.dump(network, open('test_network.pkl', 'wb'))

# network = pickle.load(open('test_network.pkl', 'rb'))

# for now create results object
# ToDo: Werte in DataFrame als List oder Array?
results = Results()
results.pfa_edges = pd.read_csv('Exemplary_PyPSA_line_results.csv',
                                index_col=0,
                                converters={'p0': literal_eval,
                                            'q0': literal_eval,
                                            'p1': literal_eval,
                                            'q1': literal_eval})
results.pfa_edges['p0'] = results.pfa_edges['p0'].apply(lambda x: np.array(x))
results.pfa_edges['q0'] = results.pfa_edges['q0'].apply(lambda x: np.array(x))
results.pfa_edges['p1'] = results.pfa_edges['p1'].apply(lambda x: np.array(x))
results.pfa_edges['q1'] = results.pfa_edges['q1'].apply(lambda x: np.array(x))
results.pfa_nodes = pd.read_csv('Exemplary_PyPSA_bus_results.csv', index_col=0,
                                converters={'v_mag_pu': literal_eval})
results.pfa_nodes['v_mag_pu'] = results.pfa_nodes['v_mag_pu'].apply(
    lambda x: np.array(x))

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
# [print('\t{0}: {1} MWh'.format(
#     _,
#     network.mv_grid.graph.nodes_by_attribute('load')[0].consumption[_] / 1e3))
#     for _ in ['retail', 'industrial', 'agricultural', 'residential']]
#
reinforce_grid.reinforce_grid(network, results)

# liste aller lv grids
#[_ for _ in network.mv_grid.lv_grids]

# nx.draw_spectral(list(network.mv_grid.lv_grids)[0].graph)

# ToDo: wie halten wir fest, welche Betriebsmittel erneuert wurden, um hinterher Kosten berechnen zu können?
# ToDo: Parameter bei Komponenten einführen mit dem man feststellen kann, ob die Komponente bereits in einer ersten Maßnahme verstärkt oder ausgebaut wurde
# ToDo: config mit Standardbetriebsmitteln?
# ToDo: Abbruchkriterium einführen - Anzahl paralleler lines
