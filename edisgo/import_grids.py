from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
from edisgo.grid.components import *

run_id = '/20180122152121/';
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
areacenter = read_network_as_pd(path, 'areacenter.csv')
mv_trafos = read_network_as_pd(path, 'mv_trafos.csv')
mv_loads = read_network_as_pd(path, 'mv_loads.csv')
edges = read_network_as_pd(path, 'edges.csv')
mapping = read_network_as_pd(path, 'mapping.csv')

#LV-Grid



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

# LV-Loads

lvloads = {}

lv_loads.apply(lambda row:
               lvloads.update({row['id_db']: Load(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['LV_grid_id'],
                 consumption=row['consumption'],
                 #type=row['type'],
                 #peak_load=row['peak_load']
                  )})
            ,axis = 1)

#LV-Cable-Distributors

lvcds = {}
lv_cd.apply(lambda row:
             #if row['LV_grid_id']
            lvcds.update({row['id_db']: BranchTee(
                 id=row['id_db'],
                 geom=row['geom'],
                 grid=row['LV_grid_id'],
                 v_level=row['LV_grid_id'],
                 #in_building=
              )})
            ,axis = 1)

# MV-Generators
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
#grid.graph.add_nodes_from(mvgens.values(), type='generator')

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

