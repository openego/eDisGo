from edisgo.grid.network import Network
import os

network = Network.import_from_dingo(os.path.join('data', 'dingo_grids__3545.pkl'))

# MV generators
gens = network.mv_grid.graph.nodes_by_attribute('generator')
print('Generators in MV grid incl. aggregated generators from MV and LV')
print('Type\tSubtype\tCapacity in kW')
for gen in gens:
    print("{type}\t{sub}\t{capacity}".format(
        type=gen.type, sub=gen.subtype, capacity=gen.nominal_capacity))

# Load located in aggregated LAs
print('\n\nAggregated load in LA adds up to\n')
[print('\t{0}: {1} MWh'.format(
    _,
    network.mv_grid.graph.nodes_by_attribute('load')[0].consumption[_] / 1e3))
    for _ in ['retail', 'industrial', 'agricultural', 'residential']]