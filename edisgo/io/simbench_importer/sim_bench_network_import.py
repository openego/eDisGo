import pandas as pd
from pathlib import Path
from edisgo import flex_opt
from edisgo import network
from edisgo import tools
from edisgo import EDisGo
from edisgo.network.grids import MVGrid, LVGrid
from edisgo.network.timeseries import get_component_timeseries
from edisgo.io.ding0_import import _validate_ding0_grid_import


def search_str_df(df,col,search_str,invert=False):
    if invert == False:
        return df.loc[df[col].str.contains(search_str)]
    else:
        return df.loc[~df[col].str.contains(search_str)]


def create_sb_dict():
    grid_name = '1-MVLV-comm-all-0-no_sw'
    # grid_name = '1-MVLV-semiurb-3.202-1-no_sw'

    curr_path = Path.cwd()
    print ("curr_path: "+ str(curr_path))
    parent_dir = curr_path.parents[3]
    print ("parent_dir: "+ str(parent_dir))
    # simbench_grids_dir = parent_dir / 'simbench'
    grid_dir = parent_dir / grid_name
    print ("grid_dir: " + str(grid_dir))

    # import the csv files into a dict og dataframes
    file_list = [i for i in grid_dir.iterdir()]
    simbench_dict = {i.name[:-4]:pd.read_csv(i,delimiter=";") for i in file_list}
    return simbench_dict

def import_sb_network(simbench_dict):
    
    
    # for i in simbench_dict.keys(): print (i)
    sb_ding0_dict = {}

    # import busses_df

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
    hv_index = buses_df.index[buses_df['name'].str.contains('HV')]
    mv_grid_id = buses_df[buses_df['name'].str.contains('MV')]['mv_grid_id'].iloc[0]
    buses_df.loc[hv_index,'mv_grid_id'] = mv_grid_id 
    # Dropping High Voltage Bus
    buses_df = buses_df.drop(hv_index)
    buses_df = buses_df[buses_col_location]

    sb_ding0_dict['buses_df'] = buses_df


    # creating transformer_df

    trans_rename_dict = {
        'id':'name',
        'nodeHV':'bus0',
        'nodeLV':'bus1',
    }

    trans_col_location = [
        'name',
        'bus0',
        'bus1',
        's_nom',
        'r',
        'x',
        'type_info'
    ]

    sb_trans_df = simbench_dict['Transformer']
    sb_trans_type_df = simbench_dict['TransformerType']
    sb_trans_df = sb_trans_df.rename(columns=trans_rename_dict)
    sb_trans_type_df = sb_trans_type_df.set_index('id')
    sb_trans_df['s_nom'] = sb_trans_df['type'].apply(lambda x: sb_trans_type_df.loc[x,'sR'] )
    sb_trans_df['r'] = ''
    sb_trans_df['x'] = ''
    sb_trans_df['type_info'] = sb_trans_df['type']
    sb_trans_df = sb_trans_df[trans_col_location]

    transformers_hvmv_df = sb_trans_df[sb_trans_df['bus0'].str.contains('HV')]
    transformers_hvmv_df = sb_trans_df[sb_trans_df['bus1'].str.contains('MV')]
    transformers_hvmv_df = transformers_hvmv_df.set_index('name')

    hvmv_bus_row = transformers_hvmv_df['bus1'][0]
    transformers_hvmv_df = transformers_hvmv_df.reset_index()

    transformers_df = sb_trans_df[sb_trans_df['bus0'].str.contains('MV')]
    transformers_df = sb_trans_df[sb_trans_df['bus1'].str.contains('LV')]

    sb_ding0_dict['transformers_hvmv_df'] = transformers_hvmv_df
    sb_ding0_dict['transformers_df'] = transformers_df

    # Creating generators df

    gen_col_location = [
        'name',
        'bus',
        'control',
        'p_nom',
        'q_nom',
        'type',
        'weather_cell_id',
        'subtype'
    ]

    gen_rename_dict = {
        'id':'name',
        'node':'bus',
        'pRES':'p_nom',
        'qRES':'q_nom',
        'profile':'weather_cell_id',
        'calc_type':'control'
    }

    slack_rename_dict = {
        'id':'name',
        'node':'bus',
        'calc_type':'control'
    }


    def add_slack_terms(gen_df):
        add_gen_dict = {
            'p_nom':0.0,
            'q_nom':0.0,
            'type':'station',
            'weather_cell_id':"",
            'subtype':'mv_station'
        }
        for key in add_gen_dict:
            gen_df[key] = gen_df['control'].apply(lambda x: add_gen_dict[key] if x == 'Slack' else x)
        return gen_df

    generators_df = simbench_dict['RES']
    generators_df['calc_type'] = generators_df['calc_type'].apply(lambda x: x.upper())
    generators_df = generators_df.rename(columns=gen_rename_dict)
    generators_df['type'] = generators_df['type'].apply(lambda x: 'solar' if x == 'PV' else x.lower())
    generators_df['subtype'] = generators_df['type']
    generators_df = generators_df[gen_col_location]


    # including the slack bus
    slack_df = simbench_dict['ExternalNet']
    slack_df = slack_df.rename(columns=slack_rename_dict)
    # slack_df = slack_df.drop(slack_col_to_drop,axis=1)
    slack_df['control'] = slack_df['control'].apply(lambda x: 'Slack' if x == 'vavm' else x)
    slack_df['name'] = slack_df['name'].apply(lambda x: x+'_slack')
    slack_df = add_slack_terms(slack_df)
    slack_df = slack_df[gen_col_location] 
    slack_df['bus'] = hvmv_bus_row
    # generators_df = pd.concat([slack_df,generators_df] ,ignore_index=True)

    sb_ding0_dict['generators_df'] = generators_df

    #  creating line df
    lines_col_location = [
        'name',
        'bus0',
        'bus1',
        'length',
        'r',
        'x',
        's_nom',
        'num_parallel',
        'kind',
        'type_info'
    ]

    def get_line_char(lines_df, line_type_df):
        char_dict = {
            'r':'r',
            'x': 'x',
            'iMax':'iMax',
            'kind':'type'
        }
        for key in char_dict:
            lines_df[key] = lines_df['type_info'].apply(lambda x: line_type_df.loc[x,char_dict[key]])
        return lines_df

    def cal_s_nom(row):
        i = row['iMax']
        bus0 = row['bus0']
        bus1 = row['bus1']
        bus0_v_nom = search_str_df(buses_df,'name',bus0,invert=False)['v_nom'].values[0]
        bus1_v_nom = search_str_df(buses_df,'name',bus1,invert=False)['v_nom'].values[0]
        bus_v_nom = (bus0_v_nom+bus1_v_nom)/2 
        return 3**0.5*bus_v_nom*i*10**(-3)

    lines_df = simbench_dict['Line']
    line_type_df = simbench_dict['LineType']
    line_type_df = line_type_df.set_index('id')

    lines_rename_dict = {
        'id':'name',
        'nodeA':'bus0',
        'nodeB':'bus1',
        'type': 'type_info'
    }

    lines_df = lines_df.rename(columns=lines_rename_dict)
    lines_df = get_line_char(lines_df,line_type_df)
    lines_df['kind'] = lines_df['kind'].apply(lambda x : 'line' if x =='ohl' else x)
    lines_df['num_parallel'] = 1
    lines_df['r'] = lines_df.apply(lambda name: name['r']*name['length'],axis=1)
    lines_df['x'] = lines_df.apply(lambda name: name['x']*name['length'],axis=1)
    lines_df['s_nom'] = lines_df.apply(cal_s_nom,axis=1)
    lines_df = lines_df[lines_col_location]

    sb_ding0_dict['lines_df'] = lines_df

    # creating loads df
    loads_rename_dict = {
        'id':'name',
        'node':'bus',
        'profile': 'sector'
    }

    loads_cols_location = [
        'name',
        'bus',
        'peak_load',
        'annual_consumption',
        'sector',
        'pLoad',
        'qLoad'
    ]

    def cal_annual_consumption(name):
        sR = name['sR']
        sector = name['sector']
        loads_profile[sector+'_sload'] = (loads_profile[ sector + '_pload']**2+loads_profile[sector + '_qload']**2)**0.5
        annual_consumption = loads_profile[sector+'_sload'].sum()*sR
        return annual_consumption

    def cal_peak_load(name):
        sR = name['sR']
        sector = name['sector']
        peak_load = loads_profile[sector+'_pload'].max()*sR
        return peak_load

    loads_profile = simbench_dict['LoadProfile']
    loads_df = simbench_dict['Load']
    loads_df = loads_df.rename(columns=loads_rename_dict)
    loads_df['peak_load'] = loads_df.apply(cal_peak_load,axis=1)
    loads_df['annual_consumption'] = loads_df.apply(cal_annual_consumption,axis=1)
    loads_df = loads_df[loads_cols_location]

    sb_ding0_dict['loads_df'] = loads_df

    # Creating switches df

    switch_rename_dict = {
        'id':'name',
        'nodeA':'bus_closed',
        'nodeB':'bus_open',
        'substation':'branch',
        'type':'type_info'
    }

    switch_col_location = [
        'name',
        'bus_closed',
        'bus_open',
        'branch',
        'type_info'
    ]

    switches_df = simbench_dict['Switch']
    switches_df = switches_df.rename(columns=switch_rename_dict)
    switches_df = switches_df[switch_col_location]
    # switches_df = switches_df.set_index('name')

    sb_ding0_dict['switches_df'] = switches_df

    # creating empty storage_df
    storage_col_dict = {
        'name': pd.Series([],dtype='str'),
        'bus': pd.Series([],dtype='str'),
        'control': pd.Series([],dtype='str'),
        'p_nom': pd.Series([],dtype='float'),
        'capacity': pd.Series([],dtype='float'),
        'efficiency_store': pd.Series([],dtype='float'),
        'efficiency_dispatch': pd.Series([],dtype='float'),
    }

    storage_units_df = pd.DataFrame(storage_col_dict)
    sb_ding0_dict['storage_units_df'] = storage_units_df

    #  creating the edisgo obj

    edisgo_obj = EDisGo(import_timeseries=False)
    edisgo_obj.topology.buses_df = buses_df.set_index('name')
    edisgo_obj.topology.generators_df = generators_df.set_index('name')
    edisgo_obj.topology.lines_df = lines_df.set_index('name')
    edisgo_obj.topology.loads_df = loads_df.set_index('name')
    edisgo_obj.topology.switches_df = switches_df.set_index('name')
    edisgo_obj.topology.transformers_hvmv_df = transformers_hvmv_df.set_index('name')
    edisgo_obj.topology.transformers_df = transformers_df.set_index('name')
    edisgo_obj.topology.storage_units_df = storage_units_df.set_index('name')

    edisgo_obj.topology.mv_grid = MVGrid(id=mv_grid_id, edisgo_obj=edisgo_obj)
    edisgo_obj.topology._grids = {}
    edisgo_obj.topology._grids[str(edisgo_obj.topology.mv_grid)] = edisgo_obj.topology.mv_grid

    #creating the lv grid
    lv_grid_df =  search_str_df(buses_df,'name','LV',invert=False)
    lv_grid_ids = lv_grid_df['lv_grid_id'].unique()
    for lv_grid_id in lv_grid_ids:
            lv_grid = LVGrid(id=lv_grid_id, edisgo_obj=edisgo_obj)
            edisgo_obj.topology.mv_grid._lv_grids.append(lv_grid)
            edisgo_obj.topology._grids[str(lv_grid)] = lv_grid

    _validate_ding0_grid_import(edisgo_obj.topology)

    print(edisgo_obj.topology)
    print ('Alles Gut')

    # importing timeseries
    load_profile_df = simbench_dict['LoadProfile']
    res_profile_df = simbench_dict['RESProfile']
    # Importing _timeindex
    timestamp_list = list(load_profile_df['time'][519:521])
    timeindex = pd.to_datetime(timestamp_list)
    edisgo_obj.timeseries._timeindex = timeindex
    load_profile_df = load_profile_df.set_index('time')
    res_profile_df = res_profile_df.set_index('time')
    print('imported time index')

    # getting generator time series
    def get_gen_meta_data(gen_name):
        gen_df = search_str_df(generators_df,'name',gen_name,invert=False)
        return {
            'name':gen_name,
            'p_nom':gen_df['p_nom'].values[0],
            'q_nom':gen_df['q_nom'].values[0],
            'wcid':gen_df['weather_cell_id'].values[0]
        }

    gen_list = [ get_gen_meta_data(gen_name)  for gen_name in generators_df['name']]
    generators_active_power = pd.DataFrame(index=timeindex)
    generators_reactive_power = pd.DataFrame(index=timeindex)

    for gen_dict in gen_list:
        if 'slack' not in gen_dict['name']:
            generators_active_power[gen_dict['name']] = res_profile_df.loc[timestamp_list,gen_dict['wcid']]*gen_dict['p_nom']
            generators_reactive_power[gen_dict['name']] = res_profile_df.loc[timestamp_list,gen_dict['wcid']]*gen_dict['q_nom']
    print ('created generator time series DataFrames')

    def get_load_meta_data(load_name):
        load_df = search_str_df(loads_df,'name',load_name,invert=False)
        return {
            'name':load_name,
            'pLoad':load_df['pLoad'].values[0],
            'qLoad':load_df['qLoad'].values[0],
            'sector':load_df['sector'].values[0],
        }
    load_list = [get_load_meta_data(load_name) for load_name in loads_df['name']]
    loads_active_power = pd.DataFrame(index=timeindex)
    loads_reactive_power = pd.DataFrame(index=timeindex)

    for load_dict in load_list:
        loads_active_power[load_dict['name']] = load_profile_df.loc[timestamp_list,load_dict['sector']+'_pload']*load_dict['pLoad']
        loads_reactive_power[load_dict['name']] = load_profile_df.loc[timestamp_list,load_dict['sector']+'_qload']*load_dict['qLoad']
    print ('created load time series DataFrames')

    get_component_timeseries(
        edisgo_obj=edisgo_obj,
        mode='manual',
        timeindex=timeindex,
        generators_active_power=generators_active_power,
        generators_reactive_power=generators_reactive_power,
        loads_active_power=loads_active_power,
        loads_reactive_power=loads_reactive_power
    )
    print('loaded timeseries into edisgo_obj')

    return edisgo_obj






if __name__ == "__main__":
    import time
    start_time = time.time()
    simbench_dict = create_sb_dict()
    elapsed_time = time.time() - start_time
    print ("elapsed_time-create_sb_dict: " + str(elapsed_time))
    start_time = elapsed_time
    edisgo_obj = import_sb_network(simbench_dict)
    edisgo_obj.analyze()
    print('done for now')
