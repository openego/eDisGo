import pandas as pd
from pathlib import Path
from edisgo import flex_opt
from edisgo import network
from edisgo import tools

from edisgo import EDisGo
from edisgo.network.grids import MVGrid, LVGrid
from tabulate import tabulate
from io import StringIO

from pypsa import Network as PyPSANetwork

# Some helpful debugging functions
def pd_print(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))
    
def search_str(df,col,search_str,invert=False):
    if invert == False:
        pd_print(df.loc[df[col].str.contains(search_str)])
        return df.loc[df[col].str.contains(search_str)]
    else:
        pd_print(df.loc[~df[col].str.contains(search_str)])
        return df.loc[~df[col].str.contains(search_str)]
    
def search_str_df(df,col,search_str,invert=False):
    if invert == False:
        return df.loc[df[col].str.contains(search_str)]
    else:
        return df.loc[~df[col].str.contains(search_str)]


grid_name = '1-MVLV-comm-all-0-no_sw'
# grid_name = '1-MVLV-semiurb-3.202-1-no_sw'

curr_path = Path.cwd()
parent_dir = curr_path.parents[3]
# simbench_grids_dir = parent_dir / 'simbench'
grid_dir = parent_dir / grid_name
print (grid_dir)

file_list = [i for i in grid_dir.iterdir()]
simbench_dict = {i.name[:-4]:pd.read_csv(i,delimiter=";") for i in file_list}

sb_ding0_dict = {}

def buses_df(simbench_dict):

    def label_mv_grid_id(name):
        if "LV" in name:
            return ""
        elif "MV" in name:
            return name[0:name.find(" ")]

    def label_lv_grid_id(name):
        if "MV" in name:
            return ""
        elif "LV" in name:
            return name[name.find(".")+1:]
        
    buses_cols_to_drop = [
        'type',
        'vmSetp',
        'vaSetp',
        'vmMin',
        'vmMax',
        'substation',
        'voltLvl'
    ]

    buses_col_location = [
        'name',
        'x',
        'y',
        'mv_grid_id',
        'lv_grid_id',
        'v_nom',
        'in_building'
    ]

    buses_df = simbench_dict['Node']

    buses_df = buses_df.rename(columns={'vmR':'v_nom'})
    buses_df = buses_df.rename(columns={'id':'name'})
    coord_df = simbench_dict['Coordinates']
    coord_df = coord_df.set_index('id')
    buses_df['x'] = buses_df['coordID'].apply(lambda coordID: coord_df.loc[coordID,'x'] )
    buses_df['y'] = buses_df['coordID'].apply(lambda coordID: coord_df.loc[coordID,'y'] )
    buses_df['in_building'] = 'False'

    buses_df['mv_grid_id'] =  buses_df['name'].apply(label_mv_grid_id)
    buses_df['lv_grid_id'] =  buses_df['subnet'].apply(label_lv_grid_id)

    # get the HV grid to adopt the mv grid id
    # hv_name = buses_df[buses_df['name'].str.contains('HV')]['name'].iloc[0]
    hv_index = buses_df.index[buses_df['name'].str.contains('HV')]
    print (hv_index)
    mv_grid_id = buses_df[buses_df['name'].str.contains('MV')]['mv_grid_id'].iloc[0]
    buses_df.loc[hv_index,'mv_grid_id'] = mv_grid_id 
    # experimental drop
    buses_df = buses_df.drop(hv_index)
    search_str(buses_df,'name','HV',invert=False)
    pd_print(buses_df.head())

    buses_df = buses_df.drop(buses_cols_to_drop,axis=1)
    buses_df = buses_df[buses_col_location]

    return buses_df

