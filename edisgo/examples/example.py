from edisgo.grid.network import Network
from edisgo.grid_expansion import reinforce_grid
import os
import pickle

#network = Network.import_from_dingo(os.path.join('data', 'ding0_grids_example.pkl'))
#pickle.dump(network, open('test_network.pkl', 'wb'))

network = pickle.load(open('test_network.pkl', 'rb'))

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
reinforce_grid.reinforce_grid(network)

# liste aller lv grids
#[_ for _ in network.mv_grid.lv_grids]

# nx.draw_spectral(list(network.mv_grid.lv_grids)[0].graph)

# ToDo: Anzahl als Attribut bei Lines einführen um parallele Leitungen abbilden zu können
# ToDo: wie halten wir fest, welche Betriebsmittel erneuert wurden, um hinterher Kosten berechnen zu können?
# ToDo: Parameter bei Komponenten einführen mit dem man feststellen kann, ob die Komponente bereits in einer ersten Maßnahme verstärkt oder ausgebaut wurde
# ToDo: config mit Standardbetriebsmitteln?