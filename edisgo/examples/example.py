from edisgo.grid.network import Network, Scenario, TimeSeries
import os
import pickle

import logging
logging.basicConfig(filename='example.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger('edisgo')
logger.setLevel(logging.DEBUG)

timeseries = TimeSeries()
scenario = Scenario(timeseries=timeseries,
                    power_flow='worst-case')

import_network = True

if import_network:
    network = Network.import_from_ding0(
        os.path.join('data', 'ding0_grids__76.pkl'),
        id='Test grid',
        scenario=scenario
    )
    # Do non-linear power flow analysis with PyPSA
    network.analyze()
    #network.pypsa.export_to_csv_folder('data/pypsa_export')
    #network.pypsa = None
    #pickle.dump(network, open('test_network.pkl', 'wb'))
else:
    network = pickle.load(open('test_results_neu.pkl', 'rb'))

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

network.reinforce()
print(network.results.grid_expansion_costs.total_costs.sum())

# nx.draw_spectral(list(network.mv_grid.lv_grids)[0].graph)

# ToDo: Möglichkeit MV und LV getrennt zu rechnen

