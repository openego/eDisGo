"Parses json from iostream or string"
function parse_json(io::Union{IO,String}; kwargs...)::Dict{String,Any}
    pm_data = JSON.parse(io)

    PowerModels._jsonver2juliaver!(pm_data)

    if haskey(pm_data, "conductors")
        Memento.warn(_LOGGER, "The JSON data contains the conductor parameter, but only single conductors are supported.  Consider using PowerModelsDistribution.")
    end

    if get(kwargs, :validate, true)
        eDisGo_OPF.correct_network_data!(pm_data)
    end

    return pm_data
end
