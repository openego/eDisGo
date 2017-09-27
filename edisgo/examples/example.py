from edisgo.grid.network import Network, Scenario, TimeSeries
from edisgo.flex_opt import reinforce_grid
import os
import pickle

import logging
logging.basicConfig(filename='example.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)

timeseries = TimeSeries()
scenario = Scenario(timeseries=timeseries,
                    power_flow='worst-case')

import_network = True

if import_network:
    network = Network.import_from_ding0(
        os.path.join('data', 'ding0_grids_example_mvgd265_new-genos-dp-v0.3.pkl'),
        id='Test grid',
        scenario=scenario
    )
    network.import_generators()
    # Do non-linear power flow analysis with PyPSA
    #network.analyze()
    #network.pypsa.export_to_csv_folder('data/pypsa_export')
    #network.pypsa = None
    #pickle.dump(network, open('test_network.pkl', 'wb'))
else:
    network = None #pickle.load(open('test_network.pkl', 'rb'))

# from pypsa import Network as PyPSANetwork
# pypsa_network = PyPSANetwork(csv_folder_name='data/pypsa_export_80_stations')
# # q unterscheidet sich
# b1 = pypsa_network.transformers_t['q0']
# b2 = network.pypsa.transformers_t['q0']
# b3 = b1 - b2
# b1 = pypsa_network.loads_t['q_set']
# b2 = network.pypsa.loads_t['q_set']

# # Print LV station secondary side voltage levels returned by PFA
# print(network.results.v_res(
#     network.mv_grid.graph.nodes_by_attribute('lv_station'), 'lv'))

# Print LV station apparent power returned by PFA
# lv_transformers = [transformer for station in
#                    network.mv_grid.graph.nodes_by_attribute('lv_station') for
#                    transformer in station.transformers]
# print(network.results.s_res(lv_transformers))

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

# Print current (line loading) at MV lines
# print(network.results.i_res([_['line'] for _ in network.mv_grid.graph.graph_edges()]))

# Print apparent power at lines
# print(network.results.s_res([_['line'] for _ in network.mv_grid.graph.graph_edges()]))

# Print number of buses, generators, load and lines to study problem size
# print('buses: ', network.pypsa.buses.shape)
# print('generators: ', network.pypsa.generators.shape)
# print('loads: ', network.pypsa.loads.shape)
# print('lines: ', network.pypsa.lines.shape)

# Print voltage levels for all lines
# print(network.results.s_res())

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

#reinforce_grid.reinforce_grid(network)
#print(network.results.grid_expansion_costs)

# liste aller lv grids
# [_ for _ in network.mv_grid.lv_grids]

# nx.draw_spectral(list(network.mv_grid.lv_grids)[0].graph)

# ToDo: Möglichkeit MV und LV getrennt zu rechnen

