function check_conductors(data::Dict{String,<:Any})
    # Handle legacy 'conductors' entries in JSON data.
    # Currently only single conductors are supported; warn and ignore if present.
    try
        if haskey(data, "conductors")
            Memento.warn(_LOGGER, "Network data contains 'conductors'. Only single conductors are supported; ignoring 'conductors'. Consider converting to single conductor format or using PowerModelsDistribution.")
            # Optionally convert or remove the key to avoid downstream issues
            delete!(data, "conductors")
        end
    catch e
        # if _LOGGER is not available or other error, fallback to printing
        try
            @warn "check_conductors warning: $e"
        catch
        end
    end

end


function correct_network_data!(data::Dict{String,<:Any})
    check_conductors(data)
    check_connectivity(data)
    check_status(data)
    # check_reference_bus(data)
    make_per_unit!(data)

    correct_transformer_parameters!(data)
    correct_voltage_angle_differences!(data)
    correct_thermal_limits!(data)
    correct_current_limits!(data)
    correct_branch_directions!(data)

    check_branch_loops(data)
    correct_dcline_limits!(data)

    # data_ep = _IM.ismultiinfrastructure(data) ? data["it"][pm_it_name] : data

    # if length(data_ep["gen"]) > 0 && any(gen["gen_status"] != 0 for (i, gen) in data_ep["gen"])
    #     eDisGo_OPF.correct_bus_types!(data)
    # end

    check_voltage_setpoints(data)
    check_storage_parameters(data)
    check_switch_parameters(data)

    correct_cost_functions!(data)

    simplify_cost_terms!(data)
end
