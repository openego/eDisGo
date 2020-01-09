#using JSON
include("add_storage_to_network.jl")

function read_statics(network_name::String)
    filename = "$(network_name)_static.json"

    io = open(filename, "r");
    data_string=read(io,String)
    network_data = JSON.parse(data_string)
    close(io)
    
    PowerModels.check_network_data(network_data)

    return network_data
end

function read_setting_file(network_name::String)    
    setting_file = "$(network_name)_opf_setting.json"
    io = open(setting_file, "r");
    data_string=read(io,String)
    opf_settings = JSON.parse(data_string)
    close(io)
    return opf_settings
end
    
function read_load_timeseries(network_name::String)
    filename = "$(network_name)_loads.json"
    io = open(filename, "r");
    data_string=read(io,String)
    load_file = JSON.parse(data_string)
    close(io)
    # check if load_file contains a timehorizon and if it has load data
    load_data,isloads = haskey(load_file,"time_horizon") ?
                    (load_file["load_data"],!isempty(load_file["load_data"]["1"])) : (Dict(),false)
    return load_data, isloads
end
        
function read_gen_timeseries(network_name::String)
    filename = "$(network_name)_gens.json"
    io = open(filename, "r");
    data_string=read(io,String)
    gen_file = JSON.parse(data_string)
    close(io)
    gen_data,isgens = haskey(gen_file,"time_horizon") ? 
                    (gen_file["gen_data"],!isempty(gen_file["gen_data"]["1"])) : (Dict(),false)
    return gen_data, isgens
end

"Returns dictionaries to look up load bus for load and to look up load ids of of load buses in data"
function look_up_dicts(data,load_data)
    look_up_load_bus = Dict()
    for (load_i,val) in load_data["1"]
        look_up_load_bus[load_i] = val["load_bus"]
    end
    look_up_load_ids = Dict()
    for (load_ids,val) in data["nw"]["1"]["load"]
        look_up_load_ids[val["load_bus"]] = [load_ids,false]
    end
    return look_up_load_bus,look_up_load_ids
end

function add_load_timeseries(data,load_data)
    # create look up dictionaries for load buses and load ids at load buses
    look_up_load_bus,look_up_load_ids = look_up_dicts(data,load_data)
    # iterate over all timesteps nw and add set values for loads
    for (nw,loads) in load_data
        for (i,load_i) in loads
            # aggregate loads for each load bus
            load_id,is_set = look_up_load_ids[load_i["load_bus"]]
            if !is_set
                # first time load_id is called set new value in data and set is_set=true
                data["nw"][nw]["load"][load_id]["pd"] = load_i["pd"]
                data["nw"][nw]["load"][load_id]["qd"] = load_i["qd"]
                look_up_load_ids[load_i["load_bus"]][2]=true
            else
                # load_id has been called before add load to existing data
                data["nw"][nw]["load"][load_id]["pd"] += load_i["pd"]
                data["nw"][nw]["load"][load_id]["qd"] += load_i["qd"]
            end
        end
    end
    return look_up_load_bus,look_up_load_ids
end

function read_data(network_name::String)
    network_data = read_statics(network_name)
    load_data, isloads = read_load_timeseries(network_name)
    gen_data, isgens = read_gen_timeseries(network_name)
    
    # replicate static network_data if time horizon is given
    if isgens || isloads
        thorizon = isgens ? length(gen_data) : length(load_data)
    else
        println("no time horizon given")
        thorizon = 0
    end

    if thorizon == 0 
        data = network_data
    else
        data = PowerModels.replicate(network_data,thorizon)
    end
    
    # write timeseries on replicated data
    if isloads
        add_load_timeseries(data,load_data)
    else
        println("no loads given")
    end
    if isgens
        for (nw,gens) in gen_data
            for (i,gen_i) in gens
                data["nw"][nw]["gen"][i]["pmax"] = gen_i["pg"]

                if data["nw"][nw]["gen"][i]["fluctuating"] == 1
                    data["nw"][nw]["gen"][i]["pmin"] = gen_i["pg"]
                    data["nw"][nw]["gen"][i]["qmax"] = gen_i["qg"]
                    data["nw"][nw]["gen"][i]["qmin"] = gen_i["qg"]
                else
                    data["nw"][nw]["gen"][i]["qmax"] = gen_i["qg"]< 0 ? 0 : gen_i["qg"]
                    data["nw"][nw]["gen"][i]["qmin"] = gen_i["qg"]> 0 ? 0 : gen_i["qg"]
                end


            end
        end
    else
        println("no gens given")
    end
    
    return data
end

function read_data(network_data::Dict{String,Any},
    load_data::Dict{String,Any},gen_data::Dict{String,Any};
    global_keys::Set{String}=Set{String}(),
    timehorizon::Int=-1)
    if timehorizon==-1
        thorizon = length(gen_data)
    else
        thorizon = timehorizon
    end
    data = PowerModels.replicate(network_data,thorizon,global_keys=global_keys)
    # write timeseries on replicated data
    add_load_timeseries(data,load_data)
    for (nw,gens) in gen_data
        for (i,gen_i) in gens
            data["nw"][nw]["gen"][i]["pmax"] = gen_i["pg"]
            if data["nw"][nw]["gen"][i]["fluctuating"] == 1
                data["nw"][nw]["gen"][i]["pmin"] = gen_i["pg"]
                data["nw"][nw]["gen"][i]["qmax"] = gen_i["qg"]
                data["nw"][nw]["gen"][i]["qmin"] = gen_i["qg"]
            else
                data["nw"][nw]["gen"][i]["qmax"] = gen_i["qg"]< 0 ? 0 : gen_i["qg"]
                data["nw"][nw]["gen"][i]["qmin"] = gen_i["qg"]> 0 ? 0 : gen_i["qg"]
            end
        end
    end
    return data
end
"""
# Read eDisGo OPF problem
## Arguments
- `network_name::String`: "path/to/network_name" without file ending

- Read network static data `network_name`_static.json
- Read OPF settings `network_name`_opf_setting.json, add settings to network data
- Read load timeseries `network_name`_loads.json
- Read generation timeseries `network_name`_gens.json

Return dictionary containing `data` for edisgo problem
"""
function read_edisgo_problem(network_name::String;timehorizon::Int=-1)

    network_data = read_statics(network_name)
    opf_settings = read_setting_file(network_name)

    # =======================================
    # ADD OPF SETTINGS to network_data and to global keys of data, 
    # i.e.:
    #    global_keys=Set(["max_exp","curtailment_total", "time_elapsed",...])
    #    DEFAUL SETTING: Set(["time_elapsed", "baseMVA", "per_unit"])
    # ========================================#

    global_keys = String[]
    for (key,val) in opf_settings
        network_data[key] = val
        push!(global_keys,key)
    end
    # add default storage units to network_data if considered
    if network_data["storage_units"]
        storage_buses = network_data["storage_buses"]
        println("Storage buses is empty= $(isempty(storage_buses))")
        add_storages_to_network_data(network_data,storage_buses)
    end
    # read time series for demand and generation
    load_data,_ = read_load_timeseries(network_name)
    gen_data,_ = read_gen_timeseries(network_name)

    if timehorizon==0 || length(load_data)<=timehorizon
        @warn("timehorizon has to be -1 and in range of timeseries given, timehorizon is set on maximal length of timeseries")
        timehorizon= length(load_data)
    end

    # shrink timeseries
    if timehorizon!=-1
        load_data =  Dict{String,Any}(string(i)=>load_data[string(i)] for i in 1:timehorizon)
        gen_data =  Dict{String,Any}(string(i)=>gen_data[string(i)] for i in 1:timehorizon)
#         tmp_load = Dict()
#         tmp_gen = Dict()
#         for t in 1:timehorizon
#             tmp_load[string(t)] = load_data[string(t)]
#             tmp_gen[string(t)] = gen_data[string(t)]
#         end
#         load_data = tmp_load
#         gen_data = tmp_gen
    end

    # merge network_data with load and generation data
    data = read_data(network_data,load_data,gen_data,global_keys=Set(global_keys),timehorizon=timehorizon)
    return data
#     return gen_data,gen_data2
end