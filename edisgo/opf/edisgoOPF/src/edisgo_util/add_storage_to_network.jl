"""
create fields for storages in network data according to a given bus_list

### Arguments

- `network_data::Dict{String,Any}`, dictionary containing network data 
#### optional

- `bus_list::Array`, list of bus ids where a storage is possible, DEFAULT= `[]` => every bus is used
"""
function add_storages_to_network_data(network_data::Dict{String,Any},bus_list::Array=[])
    if isempty(bus_list)
        bus_list = [bus for bus in 1:length(network_data["bus"])]
    end
    #
    storage_dict = Dict(
        "charge_efficiency" => 1.0,
        "discharge_efficiency" => 1.0,
        "c_rate" =>1.0,
        "storage_bus" => [],
        "index" => [],
        "status"=>1)
    # #copy initial parameters for storages
    # initial_storage = copy(network_data["storage"]["1"])
    # #clear storage dict and add a storage for each bus in bus_list
    network_data["storage"] = Dict()
    for bus_i in bus_list#1:length(network_data["bus"])
        network_data["storage"][bus_i] = copy(storage_dict)
        network_data["storage"][(bus_i)]["storage_bus"] = bus_i
        network_data["storage"][bus_i]["index"] = bus_i
        #         network_data["storage"]["$(bus_i)"] = copy(initial_storage)
        #         network_data["storage"]["$(bus_i)"]["storage_bus"]=bus_i
    end
end