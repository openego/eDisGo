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
