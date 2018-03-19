from edisgo.grid.network import Network, Scenario
#network = Network.import_from_csv('/home/local/RL-INSTITUT/inga.loeser/ding0/20170922152523/ding0_grids__76/example')

scenario = Scenario(power_flow='worst-case', mv_grid_id='76')
network = Network.import_from_csv('/home/local/RL-INSTITUT/inga.loeser/ding0/20170922152523/ding0_grids__76/20180313135935',
                                  scenario=scenario)

#network = Network.import_from_ding0('/home/local/RL-INSTITUT/inga.loeser/ding0/20170922152523/ding0_grids__76.pkl',
#                                  scenario=scenario)
print(network.mv_grid._lv_grids)
print(network)