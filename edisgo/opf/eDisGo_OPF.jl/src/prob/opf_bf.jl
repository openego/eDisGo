"Solve multinetwork branch flow OPF with multiple flexibilities"
function solve_mn_opf_bf_flex(file, model_type::Type{T}, optimizer; kwargs...) where T <: AbstractBFModel
    return eDisGo_OPF.solve_model(file, model_type, optimizer, build_mn_opf_bf_flex; multinetwork=true, kwargs...)
end


"Build multinetwork branch flow OPF with multiple flexibilities"
function build_mn_opf_bf_flex(pm::AbstractBFModelEdisgo)
    for (n, network) in PowerModels.nws(pm)
        # VARIABLES
        if PowerModels.ref(pm, 1, :opf_version) in(1, 2, 3, 4)
            #variable_branch_power_radial(pm, nw=n)  # Eq. ():  branch power <= rate_a (s_nom)
            if PowerModels.ref(pm, 1, :opf_version) in(1, 3)
                eDisGo_OPF.variable_branch_current(pm, nw=n, bounded=false)
                variable_branch_power(pm, nw=n, bounded=false)  # ToDo: nur bounded = false falls kein Storage!!!
            else
                eDisGo_OPF.variable_branch_current(pm, nw=n)  # Eq. ()
                variable_gen_power_curt(pm, nw=n)  #  Eq. (20)
                variable_branch_power(pm, nw=n)
                variable_slack_grid_restrictions(pm, nw=n)
            end
            variable_bus_voltage(pm, nw=n)  # Eq. (29)
            variable_battery_storage_power(pm, nw=n)  # Eq. (21), (22)
            variable_heat_storage(pm, nw=n)  # Eq. (22)
            variable_cp_power(pm, nw=n)  #  Eq. (23), (24)
            variable_heat_pump_power(pm, nw=n)  # Eq. (25)
            variable_dsm_storage_power(pm, nw=n)  # Eq. (26), (27)
            variable_slack_gen(pm, nw=n)  # Eq. (28)
            variable_slack_HV_requirements(pm, nw=n)
        else
            throw(ArgumentError("OPF version $(PowerModels.ref(pm, 1, :opf_version)) is not implemented! Choose between version 1 to 4."))
        end

        # CONSTRAINTS
        for i in PowerModels.ids(pm, :bus, nw=n)
            constraint_power_balance_bf(pm, i, nw=n) # Eq. (2)-(5)
        end
        for i in PowerModels.ids(pm, :branch, nw=n)
            constraint_voltage_magnitude_difference_radial(pm, i, nw=n) # Eq. (6)
        end
        eDisGo_OPF.constraint_model_current(pm, nw=n)  # Eq. (7) as SOC
        for i in PowerModels.ids(pm, :heatpumps, nw=n)
            constraint_hp_operation(pm, i, n) # Eq. (14)
        end

        for i in PowerModels.ids(pm, :HV_requirements, nw=n)
            constraint_HV_requirements(pm, i, n) # Eq. (15)-(19)
        end

    end

    # CONSTRAINTS
    network_ids = sort(collect(PowerModels.nw_ids(pm)))
    for kind in ["storage", "heat_storage", "dsm"]
        n_1 = network_ids[1]
        for i in PowerModels.ids(pm, Symbol(kind), nw=n_1)
            constraint_store_state(pm, i, nw=n_1, kind=kind)  # Eq. (8), (10)
        end

        for n_2 in network_ids[2:end]
            for i in PowerModels.ids(pm, Symbol(kind), nw=n_2)
                constraint_store_state(pm, i, n_1, n_2, kind) # Eq. (9), (11)
            end
            n_1 = n_2
        end
    end

    n_1 = network_ids[1]

    for i in PowerModels.ids(pm, :electromobility, nw=n_1)
        eta = PowerModels.ref(pm, 1, :electromobility)[i]["eta"]
        constraint_cp_state_initial(pm, n_1, i, eta)  # Eq. (12)
    end

    for n_2 in network_ids[2:end]
        for i in PowerModels.ids(pm, :electromobility, nw=n_2)
            eta = PowerModels.ref(pm, 1, :electromobility)[i]["eta"]
            constraint_cp_state(pm, n_1, n_2, i, eta) # Eq. (13)
        end
        n_1 = n_2
    end

    # OBJECTIVE FUNCTION
    if PowerModels.ref(pm, 1, :opf_version) in(1,3)
        objective_min_losses(pm)  # Eq. (1)
        if (PowerModels.ref(pm, 1, :opf_version) == 1)
            #objective_min_hv_slacks(pm)
            # Set multiple objectives
            # https://www.gurobi.com/documentation/9.1/refman/specifying_multiple_object.html
        end
    elseif PowerModels.ref(pm, 1, :opf_version) in(2,4)
        objective_min_losses_slacks(pm)  # Eq. (1)
        if (PowerModels.ref(pm, 1, :opf_version) == 2)
            #objective_min_hv_slacks(pm)
        end
    end
end
