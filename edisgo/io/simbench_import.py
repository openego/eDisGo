import pandas as pd
import pathlib as pl
import pickle
import copy
from edisgo import EDisGo
from edisgo.network.grids import MVGrid, LVGrid
from edisgo.network.timeseries import get_component_timeseries
from edisgo.io.ding0_import import _validate_ding0_grid_import

def pickle_obj(file_path,obj_to_pickle):
    """
    pickles an object

    Parameters
    ----------
    file_path: `path`
        Path to pickle the object
    obj_to_pickle: `obj`
        The object to pickle

    Returns
    -------
    Null
    
    """
    pickle.dump( obj_to_pickle, open( file_path, "wb" ) )

def unpickle_obj(file_path):
    """
    unpickle an object

    Parameters
    ----------
    file_path: `path`
        path to pickled object
    
    Returns
    -------
    obj

    """

    return pickle.load( open( file_path, "rb" ) )

def pickle_df(pickle_path,df_dict):
    """
    pickles a dictionary that contains a dataframe

    Parameters
    ----------
    pickle_path: `path`
        Path to pickle the object
    df_dict: `obj`
        The object to pickle

    Returns
    -------
    Null

    """
    for df_name in df_dict:
        df_path = str(pickle_path) + '/' + df_name + '.pkl'
        df_dict[df_name].to_pickle(df_path)
        print ('pickled '+df_name)

def unpickle_df(pickle_dir,df_name_list):
    """
    unpickle a dictionary that contains a dataframe

    Parameters
    ----------
    pickle_dir: `path`
        path to pickled object
    df_name_list: `list`
        list of strings that are names of the dataframes in the pickle obj

    Returns
    -------
        a dictionary of dataframes
    """
    df_dict = {}
    for df_name in df_name_list:
        pickle_path = str(pickle_dir) +'/' + df_name + '.pkl'
        if pl.Path(pickle_path).is_file():
            print ('took ' + df_name + ' from pickle')
            df_dict[df_name] = pd.read_pickle(pickle_path)
    return df_dict

def search_str_df(df,col,search_str,invert=False):
    """
    Helper function to search for specific rows in a dataframe

    Parameters
    ----------
    df: `Dataframe`
        dataframe of interest
    col: `str`
        the column of interest within the data frame
    search_str: `str`
        the specific string of interest within the row

    Returns
    -------
    `dataframe`:
        a sub dataframe from the input dataframe that fufills the criteria

    """
    if invert == False:
        return df.loc[df[col].str.contains(search_str)]
    else:
        return df.loc[~df[col].str.contains(search_str)]

def set_time_index(df,timeindex):
    """
    set the index of the input df as the time index. 
    
    *** Note that the dataframe must have the same number of rows as the timeindex

    Parameters
    ----------
    df: `dataframe`
        data frame with a need to set time index
    timeindex: `pandas timeindex object`

    Returns
    -------
    `dataframe`
        a dataframe with the index set as the time index
    """
    df['timeindex'] = list(timeindex)
    df = df.set_index('timeindex')
    return df

def create_sb_dict(grid_dir):
    """
    import the csv files of the raw data into a dictionary of dataframes

    Parameters
    ----------
    grid_dir: `str`
        the file path to the "topology" folder of the simbench grid

    Returns
    -------
    `dictionary`
        a dictionary with the following dataframe as keys
        Line
        LoadProfile
        RES
        ExternalNet
        TransformerType
        StudyCases
        Node
        Switch
        Measurement
        Coordinates
        NodePFResult
        Substation
        Transformer
        Load
        RESProfile
        LineType
    """
    file_list = [i for i in grid_dir.iterdir()]
    simbench_dict = {i.name[:-4]:pd.read_csv(i,sep=";") for i in file_list}
    def remove_daylight_saving(df_ext):
        df = copy.deepcopy(df_ext)
        start = pd.to_datetime(df.iloc[0]['time'])
        end = pd.to_datetime(df.iloc[-1]['time'])
        my_index = pd.period_range(start=start,end=end,freq='15min').to_timestamp()
        df['time'] = my_index
        return df
    simbench_dict['LoadProfile'] = remove_daylight_saving(simbench_dict['LoadProfile'])
    simbench_dict['RESProfile'] = remove_daylight_saving(simbench_dict['RESProfile'])
    return simbench_dict


def create_buses_df(simbench_dict,pickle_dir=False):
    """
    Creating the eDisGo compatible buses dataframe

    Parameters
    ----------
    simbench_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path

    Return
    ------
    `dataframe`
        eDisGo compatible buses dataframe 
    """
    df_dict = unpickle_df(pickle_dir,['buses_df'])
    if len(df_dict) is not 0:
        return df_dict['buses_df']

    def label_mv_grid_id(name):
        if "LV" in name:
            return ""
        elif "MV" in name:
            return name[0:name.find(" ")]

    def label_lv_grid_id(name):
        if "MV" in name:
            return
        elif "LV" in name:
            return name

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
    mv_grid_ids = buses_df['mv_grid_id'].unique()
    mv_grid_ids = [i for i in mv_grid_ids if i is not None and len(i) > 0]
    if len(mv_grid_ids) > 1:
        raise Exception("There is more than one MV grid present, please check.")
    else:
        buses_df['mv_grid_id'] = mv_grid_ids[0]

    # get the HV grid to adopt the mv grid id
    hv_index = buses_df.index[buses_df['name'].str.contains('HV')]
    #todo: is this not unnecessary if bus is dropped afterwards?
    mv_grid_id = buses_df[buses_df['name'].str.contains('MV')]['mv_grid_id'].iloc[0]
    buses_df.loc[hv_index,'mv_grid_id'] = mv_grid_id 
    # Dropping High Voltage Bus
    buses_df = buses_df.drop(hv_index)
    buses_df = buses_df[buses_col_location]
    
    print ('Converted buses_df')
    if pickle_dir != False:
        pickle_df(pickle_dir,{'buses_df':buses_df})


    return buses_df


def create_transformer_dfs(simbench_dict,pickle_dir=False):
    """
    Creating eDisGo compatible transformers dataframe and transformers_hvmv dataframe

    Parameters
    ----------
    simbench_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path

    Return
    ------
    `dictionary`
        contains the following dataframe
            transformers_df
            transformers_hvmv_df
    """
    df_dict = unpickle_df(pickle_dir,["transformers_df","transformers_hvmv_df"])
    if len(df_dict) is not 0:
        return df_dict
    
    trans_rename_dict = {
        'id': 'name',
        'nodeHV': 'bus0',
        'nodeLV': 'bus1'
    }

    trans_col_location = [
        'name',
        'bus0',
        'bus1',
        's_nom',
        'r_pu',
        'x_pu',
        'type_info'
    ]

    sb_trans_df = simbench_dict['Transformer']
    sb_trans_type_df = simbench_dict['TransformerType']
    sb_trans_df = sb_trans_df.rename(columns=trans_rename_dict)
    sb_trans_type_df = sb_trans_type_df.set_index('id')
    sb_trans_df['s_nom'] = sb_trans_df['type'].apply(lambda x: sb_trans_type_df.loc[x,'sR'] )
    sb_trans_df['pCu'] = sb_trans_df['type'].apply(lambda x: sb_trans_type_df.loc[x,'pCu'] )
    sb_trans_df['vmImp'] = sb_trans_df['type'].apply(lambda x: sb_trans_type_df.loc[x,'vmImp']/100 )

    sb_trans_df['r_pu'] = (sb_trans_df['pCu']/1000)/sb_trans_df['s_nom']
    sb_trans_df['x_pu'] = (sb_trans_df['vmImp']**2-sb_trans_df['r_pu']**2)**0.5
  
    sb_trans_df['type_info'] = sb_trans_df['type']
    sb_trans_df = sb_trans_df[trans_col_location]

    #Todo: first line is not used
    # transformers_hvmv_df = sb_trans_df[sb_trans_df['bus0'].str.contains('HV')]
    transformers_hvmv_df = sb_trans_df[sb_trans_df['bus1'].str.contains('MV')]
    transformers_hvmv_df = transformers_hvmv_df.set_index('name')

    # hvmv_bus_row = transformers_hvmv_df['bus1'][0]
    transformers_hvmv_df = transformers_hvmv_df.reset_index()

    # Todo: first line is not used
    # transformers_df = sb_trans_df[sb_trans_df['bus0'].str.contains('MV')]
    transformers_df = sb_trans_df[sb_trans_df['bus1'].str.contains('LV')]

    trans_df_dict = {
        "transformers_df":transformers_df,
        "transformers_hvmv_df":transformers_hvmv_df
    }

    print ('Converted both transformers_df')
    if pickle_dir != False:
        pickle_df(pickle_dir,trans_df_dict)

    return trans_df_dict


def create_generators_df(simbench_dict,pickle_dir=False):
    """
    Creating the eDisGo compatible generators dataframe

    Parameters
    ----------
    simbench_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path

    Return
    ------
    `dataframe`
        eDisGo compatible generators dataframe 
    """
    df_dict = unpickle_df(pickle_dir,["generators_df"])
    if len(df_dict) is not 0:
        return df_dict['generators_df']

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
    #Todo: change weather_cell_id?

    generators_df = simbench_dict['RES']
    generators_df['calc_type'] = generators_df['calc_type'].apply(lambda x: x.upper())
    generators_df = generators_df.rename(columns=gen_rename_dict)
    generators_df['type'] = generators_df['type'].apply(lambda x: 'solar' if x == 'PV' else x.lower())
    generators_df['subtype'] = generators_df['type']
    generators_df = generators_df[gen_col_location]

    print ('Converted the generators_df')
    if pickle_dir != False:
        pickle_df(pickle_dir,{'generators_df':generators_df})
    return generators_df


def create_lines_df(simbench_dict,buses_df,pickle_dir=False):
    """
    Creating the eDisGo compatible lines dataframe

    Parameters
    ----------
    simbench_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    buses_df: `dataframe`
        dataframe created by function "create_buses_df"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path

    Return
    ------
    `dataframe`
        eDisGo compatible lines dataframe 
    """
    df_dict = unpickle_df(pickle_dir,["lines_df"])
    if len(df_dict) is not 0:
        return df_dict['lines_df']

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

    print ('Converted lines_df')
    if pickle_dir != False:
        pickle_df(pickle_dir,{'lines_df':lines_df})
    return lines_df


def create_loads_df(simbench_dict,pickle_dir=False):
    """
    Creating the eDisGo compatible loads dataframe

    Parameters
    ----------
    simbench_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path

    Return
    ------
    `dataframe`
        eDisGo compatible loads dataframe 
    """
    df_dict = unpickle_df(pickle_dir,["loads_df"])
    if len(df_dict) is not 0:
        return df_dict['loads_df']
    
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

    print ('Converted loads_df')
    if pickle_dir != False:
        pickle_df(pickle_dir,{'loads_df':loads_df})

    return loads_df


def create_switches_df(simbench_dict,pickle_dir=False):
    """
    Creating the eDisGo compatible switches dataframe

    Parameters
    ----------
    simbench_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path

    Return
    ------
    `dataframe`
        eDisGo compatible switches dataframe 
    """
    df_dict = unpickle_df(pickle_dir,["switches_df"])
    if len(df_dict) is not 0:
        return df_dict['switches_df']
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

    print ('Converted switches_df')
    if pickle_dir != False:
        pickle_df(pickle_dir,{'switches_df':switches_df})

    return switches_df


def create_storage_units_df(simbench_dict,pickle_dir=False):
    """
    Creating the eDisGo compatible storage_units dataframe

    Parameters
    ----------
    simbench_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path

    Return
    ------
    `dataframe`
        eDisGo compatible storage_units dataframe 
    """
    df_dict = unpickle_df(pickle_dir,["storage_units_df"])
    if len(df_dict) is not 0:
        return df_dict['storage_units_df']
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

    print ('Converted storage_units_df')
    if pickle_dir != False:
        pickle_df(pickle_dir,{'storage_units_df':storage_units_df})

    return storage_units_df


# HK current
def import_sb_topology(sb_dict,pickle_file_path=False):
    """
    Creating eDisGo object with topology data

    Parameters
    ----------
    sb_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_file_path: `path` (default to False)
        if path is provided, then the function will search for the pickle file. If the pickle file exist,
        then the function will unpickle the file and return the object. If not, then the function will create
        the object and pickle it into the file path

        if path is not provided, the function will create the edisgo object. ( Not reccomended as it takes quite
        a bit of time. pickling should be done if storage space is available)

    Return
    ------
    `obj`
        eDisGo object with imported topology
    """
    if pickle_file_path is not False:
        if pickle_file_path.exists():
            edisgo_obj = unpickle_obj(pickle_file_path)
            return edisgo_obj

    buses_df = create_buses_df(sb_dict)
    transformer_df_dict = create_transformer_dfs(sb_dict)
    transformers_df = transformer_df_dict['transformers_df']
    transformers_hvmv_df = transformer_df_dict['transformers_hvmv_df']
    generators_df = create_generators_df(sb_dict)
    lines_df = create_lines_df(sb_dict,buses_df)
    loads_df = create_loads_df(sb_dict)
    switches_df = create_switches_df(sb_dict)
    storage_units_df = create_storage_units_df(sb_dict)

    # Get mv grid id
    mv_grid_id = buses_df[buses_df['name'].str.contains('MV')]['mv_grid_id'].iloc[0]

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
    if pickle_file_path is not False:
        pickle_obj(pickle_file_path,edisgo_obj)
    print('created edisgo topology')

    return edisgo_obj

# Hk Current
def create_timestamp_list(sb_dict,time_accuracy='1_hour'):
    """
    Creating a list of timestamps for the given time sereies data from the simbench grid
    according to the time_accuracy stated

    Parameters
    ----------
    sb_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    time_accuracy: `str` (default '1_hour')
        can be the following values:
            {'1_hour','15_min'}
    Return
    ------
    `list`
        A list of the timestamps according to the time_accuracy stated in the input
    """
    load_profile_df = sb_dict['LoadProfile']
    timestamp_list = list(load_profile_df['time'])
    if time_accuracy == '15_min':
        return timestamp_list
    else:
        timestamp_list_hour = []
        for i in range(len(timestamp_list)):
            if i%4==0:
                timestamp_list_hour.append(timestamp_list[i])
        return timestamp_list_hour

# Hk Current
def create_timestep_list(sb_dict,time_accuracy='1_hour'):
    """
    Creating a list of timesteps for the given time sereies data from the simbench grid
    according to the time_accuracy stated

    Parameters
    ----------
    sb_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    time_accuracy: `str` (default '1_hour')
        can be the following values:
            {'1_hour','15_min'}
    Return
    ------
    `list`
        A list of the timesteps according to the time_accuracy stated in the input
    """
    load_profile_df = sb_dict['LoadProfile']
    timestamp_list = list(load_profile_df['time'])
    if time_accuracy == '15_min':
        return timestamp_list
    else:
        timestep_list_hour = []
        for i in range(len(timestamp_list)):
            if i%4==0:
                timestep_list_hour.append(i)
        return timestep_list_hour

# Hk Current
def create_timeindex(edisgo_obj,sb_dict,time_accuracy='1_hour'):
    """
    Creating a list of timestamps for the given time sereies data from the simbench grid
    according to the time_accuracy stated

    Parameters
    ----------
    edisgo_obj: `obj`
        edsigo_obj created by function "import_sb_topology"
    sb_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    time_accuracy: `str` (default '1_hour')
        can be the following values:
            {'1_hour','15_min'}
    Return
    ------
    `obj`
        an eDisGo object with a timeseries index that reflects the time accuracy stated
        in the input
    """
    print("create_timeindex_0")
    load_profile_df = sb_dict['LoadProfile']
    time_col = load_profile_df['time']
    timestep_list = create_timestep_list(sb_dict,time_accuracy=time_accuracy)
    timestamp_list = time_col.iloc[timestep_list]
    print("timestamp_list length"+str(len(timestamp_list)))
    timeindex = pd.to_datetime(timestamp_list)
    edisgo_obj.timeseries._timeindex = timeindex.index
    edisgo_obj.timeseries._timestamp = {
        'timestamp_list': timestamp_list
    }
    print('imported time index')
    return edisgo_obj

def create_sb_gen_timeseries(edisgo_obj,sb_dict,pickle_dir=False):
    """
    Create generator timeseries dictionary that is compatible with eDisGo

    Parameters
    ----------
    edisgo_obj: `obj`
        edsigo_obj created by function "create_timeindex"
    sb_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path
    
    Return
    ------
    `dictionary`
        generator timeseries dictionary that is compatible with eDisGo
    """
    # Importing _timeindex
    timeindex = edisgo_obj.timeseries._timeindex
    res_profile_df = sb_dict['RESProfile']
    timestamp_list = edisgo_obj.timeseries._timestamp['timestamp_list']
    res_profile_df = res_profile_df.set_index('time')
    generators_df = edisgo_obj.topology.generators_df.reset_index()
    
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
    # generators_active_power_df = pd.DataFrame(index=timeindex)
    # generators_reactive_power_df = pd.DataFrame(index=timeindex)
    generators_active_power_df = pd.DataFrame()
    generators_reactive_power_df = pd.DataFrame()

    for gen_dict in gen_list:
        if 'slack' not in gen_dict['name']:
            generators_active_power_df[gen_dict['name']] = res_profile_df.loc[timestamp_list,gen_dict['wcid']]*gen_dict['p_nom']
            generators_reactive_power_df[gen_dict['name']] = res_profile_df.loc[timestamp_list,gen_dict['wcid']]*gen_dict['q_nom']
    print ('created generator time series DataFrames')
    
    # Set the time index right
    generators_active_power_df = set_time_index(generators_active_power_df,timeindex)
    generators_reactive_power_df = set_time_index(generators_reactive_power_df,timeindex)
    gen_timeseries_dict = {
        "generators_active_power_df":generators_active_power_df,
        "generators_reactive_power_df":generators_reactive_power_df
    }

    if pickle_dir != False:
        pickle_df(pickle_dir,gen_timeseries_dict)

    return gen_timeseries_dict


def create_sb_loads_timeseries(edisgo_obj,sb_dict,pickle_dir=False):
    """
    Create loads timeseries dictionary that is compatible with eDisGo

    Parameters
    ----------
    edisgo_obj: `obj`
        edsigo_obj created by function "create_timeindex"
    sb_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    pickle_dir: `path` (default to False)
        if path is provided, the function will pickle the result at the given path
    
    Return
    ------
    `dictionary`
        loads timeseries dictionary that is compatible with eDisGo
    """
    timeindex = edisgo_obj.timeseries._timeindex
    df_dict = unpickle_df(pickle_dir,["loads_active_power_df","loads_reactive_power_df"])
    if len(df_dict) is not 0:
        if timeindex.equals(df_dict['loads_active_power_df'].index) and timeindex.equals(df_dict['loads_reactive_power_df'].index):
            return df_dict
        else:
            print('new timestamp')
    def get_load_meta_data(load_name):
        load_df = search_str_df(loads_df,'name',load_name,invert=False)
        return {
            'name':load_name,
            'pLoad':load_df['pLoad'].values[0],
            'qLoad':load_df['qLoad'].values[0],
            'sector':load_df['sector'].values[0],
        }
    timestamp_list = edisgo_obj.timeseries._timestamp['timestamp_list']
    load_profile_df = sb_dict['LoadProfile']
    load_profile_df = load_profile_df.set_index('time')
    loads_df = edisgo_obj.topology.loads_df.reset_index()
    load_list = [get_load_meta_data(load_name) for load_name in loads_df['name']]
    # loads_active_power_df = pd.DataFrame(index=timeindex)
    # loads_reactive_power_df = pd.DataFrame(index=timeindex)
    loads_active_power_df = pd.DataFrame()
    loads_reactive_power_df = pd.DataFrame()

    #Todo: Is there a way not to loop over all loads?
    for load_dict in load_list:
        loads_active_power_df[load_dict['name']] = load_profile_df.loc[timestamp_list,load_dict['sector']+'_pload']*load_dict['pLoad']
        loads_reactive_power_df[load_dict['name']] = load_profile_df.loc[timestamp_list,load_dict['sector']+'_qload']*load_dict['qLoad']
    print ('created load time series DataFrames')

    loads_active_power_df = set_time_index(loads_active_power_df,timeindex)
    loads_reactive_power_df = set_time_index(loads_reactive_power_df,timeindex)

    loads_timeseries_dict = {
        "loads_active_power_df":loads_active_power_df,
        "loads_reactive_power_df":loads_reactive_power_df
    }

    if pickle_dir != False:
        pickle_df(pickle_dir,loads_timeseries_dict)

    return loads_timeseries_dict


# HK Current
def import_sb_timeseries(edisgo_obj_ext,sb_dict,time_accuracy='1_hour'):
    """
    Creating eDisGo object with timeseries of loads and generators

    Parameters
    ----------
    edisgo_obj_ext: `obj`
        edsigo_obj created by function "import_sb_topology"
    sb_dict: `dictionary`
        dictionary created by function "create_sb_dict"
    time_accuracy: `str` (default '1_hour')
        can be the following values:
            {'1_hour','15_min'}
    Return
    ------
    `obj`
        an eDisGo object with timeseries of loads and generators
    """
    edisgo_obj = copy.deepcopy(edisgo_obj_ext)
    load_profile_df = sb_dict['LoadProfile']
    timestep_list = create_timestep_list(sb_dict)
    load_profile_df_reduced = load_profile_df.iloc[timestep_list].set_index('time')
    timeindex = load_profile_df_reduced.index
    edisgo_obj.timeseries._timeindex = timeindex
    gen_timeseries_dict = create_sb_gen_timeseries(edisgo_obj,sb_dict,pickle_dir=False)
    loads_timeseries_dict = create_sb_loads_timeseries(edisgo_obj,sb_dict,pickle_dir=False)
    gen_timeseries_dict['generators_active_power_df'] = gen_timeseries_dict['generators_active_power_df'].set_index(timeindex)
    gen_timeseries_dict['generators_reactive_power_df'] = gen_timeseries_dict['generators_reactive_power_df'].set_index(timeindex)
    loads_timeseries_dict['loads_active_power_df'] = loads_timeseries_dict['loads_active_power_df'].set_index(timeindex)
    loads_timeseries_dict['loads_reactive_power_df'] = loads_timeseries_dict['loads_reactive_power_df'].set_index(timeindex)
    get_component_timeseries(
        edisgo_obj=edisgo_obj,
        mode='manual',
        timeindex=timeindex,
        generators_active_power=gen_timeseries_dict['generators_active_power_df'],
        generators_reactive_power=gen_timeseries_dict['generators_reactive_power_df'],
        loads_active_power=loads_timeseries_dict['loads_active_power_df'],
        loads_reactive_power=loads_timeseries_dict['loads_reactive_power_df']
    )
    print('loaded timeseries into edisgo_obj')
    return edisgo_obj



