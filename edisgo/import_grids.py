from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
from edisgo.grid.components import *

run_id = '/20180228165554/';
grid_id = '76';
base_path = "/home/local/RL-INSTITUT/inga.loeser/ding0/20170922152523/ding0_grids__"
path = ''.join([base_path, grid_id, run_id])



def read_network_as_pd(path, tablename):
 # Reads csv files and loads data in a pandas dataframe
    file = '/'.join([path, tablename])
    table = pd.read_csv(file, sep = ';')
    return table

lv_grid = read_network_as_pd(path, 'lv_grid.csv')
lv_gen = read_network_as_pd(path, 'lv_gen.csv')
lv_cd = read_network_as_pd(path, 'lv_cd.csv')
lv_stations = read_network_as_pd(path, 'lv_stations.csv')
lv_trafos = read_network_as_pd(path, 'lv_trafos.csv')
lv_loads = read_network_as_pd(path, 'lv_loads.csv')
mv_grid = read_network_as_pd(path, 'mv_grid.csv')
mv_gen = read_network_as_pd(path, 'mv_gen.csv')
mv_cb = read_network_as_pd(path, 'mv_cb.csv')
mv_cd = read_network_as_pd(path, 'mv_cd.csv')
mv_stations = read_network_as_pd(path, 'mv_stations.csv')
mv_trafos = read_network_as_pd(path, 'mv_trafos.csv')
mv_loads = read_network_as_pd(path, 'mv_loads.csv')
edges = read_network_as_pd(path, 'edges.csv')
mapping = read_network_as_pd(path, 'mapping.csv')

#edges.pop(28)
#edges.drop(edges.index[[28]], inplace=True)

def add_node_to_grid(grids, items, item_type):
    for g in grids:
        dic = {}
        for i in items:
            if items[i].grid == grids[g].id:
                dic.update({i : items[i]})
        grids[g].graph.add_nodes_from(dic, type=item_type)

def add_edge_to_grid(grids, items, item_type):
    for g in grids:
        list = []
        for i in items:
            n1 , n2, l = i
            if l['line'].grid == grids[g].id:
                list.append(i)
        grids[g].graph.add_edges_from(list, type=item_type)


#LV-Grid

# Create LV grid instance
lvgrids = {}
lv_grid.apply(lambda row:
              lvgrids.update({row['id_db']: LVGrid(
                    id=row['LV_grid_id'],
                    grid_district={
                        'geom': row['id_db'],
                        'population': row['population']},
                    voltage_nom=row['voltage_nom'],#in kV
                    network=row['network'],
                    station=None,
                    )})
              ,axis = 1)


# LV-Stations

lvstations = {}

lv_stations.apply(lambda row:
               lvstations.update({row['id_db']: Station(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['LV_grid_id'],
            )})
            ,axis = 1)

#add station to corresponding grid
for g in lvgrids:
    for i in lvstations:
        if lvstations[i].grid == lvgrids[g].id:
            lvgrids[g]._station = lvstations[i]

# LV-Trafo
lvtrafos = {}
lv_trafos.apply(lambda row:
                lvtrafos.update({row['id_db']: Transformer(
                    id=row['id_db'],
                    geom=row['geom'],
                    grid=row['LV_grid_id'],
                    mv_grid=None,
                    voltage_op=row['voltage_op'],
                    type=pd.Series(data=[row['X'], row['R'], row['S_nom']],
                                   index=['X', 'R', 'S_nom']),
                )})
                , axis=1)

for s in lvstations:
    for t in lvtrafos:
        item = lvstations[i]
        if lvtrafos[t].grid == lvstations[s].grid:
            lvstations[s]._transformers = lvtrafos[t]


# LV-Generators
lvgens = {}
lv_gen.apply(lambda row:
             lvgens.update({row['id_db']: Generator(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['LV_grid_id'],
                 nominal_capacity=row['nominal_capacity'],
                 type = row['type'],
                 subtype=row['subtype'],
                 v_level=row['v_level']
                  )})
            ,axis = 1)
add_node_to_grid(lvgrids, lvgens, 'generator')

# LV-Loads

lvloads = {}

lv_loads.apply(lambda row:
            lvloads.update({row['id_db']: Load(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['LV_grid_id'],
                 consumption=row['consumption'],
                #ToDo: Timeseries
                 #type=row['type'],
                 #ToDo: peak_load
                #peak_load #=row['peak_load']
            )})
            ,axis = 1)
add_node_to_grid(lvgrids, lvloads, 'load')

#LV-Cable-Distributors

lvcds = {}
lv_cd.apply(lambda row:
            lvcds.update({row['id_db']: BranchTee(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['LV_grid_id'],
                 v_level=row['LV_grid_id'], #ToDo: v_level notwendig?
                 #ToDo: in_building
              )})
            ,axis = 1)
add_node_to_grid(lvgrids, lvcds, 'branch_tee')

# Create MV grid instance

mvgrids = {}
mv_grid.apply(lambda row:
              mvgrids.update({row['id_db']: MVGrid(
                    id=row['MV_grid_id'],
                    network= row['network'],
                    voltage_nom=row['voltage_nom'],  #TODO: check MV/kv/V
                    #ToDo: peak_load und peak_generation notwendig?
                    #peak_load =,
                    #peak_generation=,
                    grid_district={
                        'geom': row['geom'],#ding0_lv_grid.grid_district.geo_data,
                        'population': row['population']},# ding0_lv_grid.grid_district.population},
                    station=None,
                    )})
              ,axis = 1)

# MV-Stations

mvstations = {}

mv_stations.apply(lambda row:
               mvstations.update({row['id_db']: Station(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['MV_grid_id'],
            )})
            ,axis = 1)

#add station to corresponding grid
for g in mvgrids:
    for i in mvstations:
        if mvstations[i].grid == mvgrids[g].id:
            mvgrids[g]._station = mvstations[i]

# MV-Trafo
mvtrafos = {}
mv_trafos.apply(lambda row:
                mvtrafos.update({row['id_db']: Transformer(
                    id=row['id_db'],
                    geom=row['geom'],
                    grid=row['MV_grid_id'],
                    mv_grid=row['MV_grid_id'],
                    voltage_op=row['voltage_op'],
                    type=pd.Series(data=[row['X'], row['R'], row['S_nom']],
                                   index=['X', 'R', 'S_nom']),
                )})
                , axis=1)

for s in mvstations:
    for t in mvtrafos:
        item = mvstations[i]
        if mvtrafos[t].grid == mvstations[s].grid:
            mvstations[s]._transformers = mvtrafos[t]


# MV-Generators
edges_aggr = {}
edges_aggr_idx = 0
mvgens = {}
mv_gen.apply(lambda row:
             mvgens.update({row['id_db']: Generator(
                 id=row['id_db'],
                 geom=row['geom'],
                 nominal_capacity=row['nominal_capacity'],
                 type = row['type'],
                 subtype=row['subtype'],
                 grid=row['MV_grid_id'],
                 v_level=row['v_level']
              )})
            ,axis = 1)
add_node_to_grid(mvgrids, mvgens, 'generator')


# MV-Loads

mvloads = {}

mv_loads.apply(lambda row:
            mvloads.update({row['id_db']: Load(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['MV_grid_id'],
                 consumption=row['consumption'],
                 type=row['type'],
                 peak_load=row['peak_load']
            )})
            ,axis = 1)
add_node_to_grid(mvgrids, mvloads, 'load')

#MV-Cable-Distributors

mvcds = {}
mv_cd.apply(lambda row:
            mvcds.update({row['id_db']: BranchTee(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['MV_grid_id'],
                 #ToDo: in_building=
              )})
            ,axis = 1)
add_node_to_grid(mvgrids, mvcds, 'branch_tee')

# Merge node defined above to a single dict
nodes = {**lvstations,
         **lvtrafos,
         **lvgens,
         **lvloads,
         **lvcds,
         **mvstations,
         **mvtrafos,
         **mvgens,
         **mvloads,
         **mvcds}#,

#Lines
lines=[]
edges.apply(lambda row:
         lines.append((nodes[row['node1']], nodes[row['node2']], {'line': Line(
              id=row['edge_name'],
              type=pd.Series(data = [row['type_name'], row['U_n'], row['I_max_th'], row['R'], row['L'], row['C']], index = ['name', 'U_n', 'I_max_th', 'R', 'L', 'C']), #, columns = []#er row['branch'].type,
              length=row['length'] / 1e3, #ToDo: Check if all edges, that are exported from Ding0, have the same scale
              kind=row['type_kind'],
              grid=row['grid']
         )}))
        ,axis = 1)
add_edge_to_grid(lvgrids, lines , 'line')

#ToDo: Keine Circuit Breakers?!

# Import transformers
#def __init__(self, **kwargs):
#    super().__init__(**kwargs)
#    self._id = kwargs.get('id', None)
#    self._geom = kwargs.get('geom', None)
#    self._grid = kwargs.get('grid', None)
#    self._mv_grid = kwargs.get('mv_grid', None)
#    self._voltage_op = kwargs.get('voltage_op', None)
#    self._type = kwargs.get('type', None)
print(mv_trafos['S_nom'])

#for it, row in mv_gen.iterrows():
#    gens.update({row['id_db']: Generator(
#        id=row['id_db'],
#        nominal_capacity=row['nominal_capacity']
#              )})

print(source)
