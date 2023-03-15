"Solve multinetwork branch flow OPF with multiple flexibilities"
function solve_mn_opf_bf_flex(file, model_type::Type{T}, optimizer; kwargs...) where T <: AbstractBFModel
    return eDisGo_OPF.solve_model(file, model_type, optimizer, build_mn_opf_bf_flex; multinetwork=true, kwargs...)
end


"Build multinetwork branch flow OPF with multiple flexibilities"
function build_mn_opf_bf_flex(pm::AbstractBFModelEdisgo)
    eDisGo_OPF.variable_max_line_loading(pm, nw=1)
    for (n, network) in PowerModels.nws(pm)
        # VARIABLES
        if PowerModels.ref(pm, 1, :opf_version) in(1, 2, 3, 4)
            if PowerModels.ref(pm, 1, :opf_version) in(1, 3)
                eDisGo_OPF.variable_branch_current(pm, nw=n, bounded=false) # Eq. 3.9i (für Version 1 bzw. 3 keine Eq. (3.9))
                eDisGo_OPF.variable_bus_voltage(pm, nw=n, bounded=false)
            else
                eDisGo_OPF.variable_branch_current(pm, nw=n)  # Eq. (3.9) und (3.9i)
                eDisGo_OPF.variable_gen_power_curt(pm, nw=n)  # Eq. (3.29) für non-dispatchable Generators
                eDisGo_OPF.variable_slack_grid_restrictions(pm, nw=n) # Eq. (3.29)-(3.32)
                eDisGo_OPF.variable_bus_voltage(pm, nw=n)  # Eq. (3.10)
            end
            eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false)
            eDisGo_OPF.variable_battery_storage_power(pm, nw=n)  # Eq. (3.13) und (3.14)
            eDisGo_OPF.variable_heat_storage(pm, nw=n)  # Eq. (3.19)
            eDisGo_OPF.variable_heat_pump_power(pm, nw=n)  # Eq. (3.16)
            eDisGo_OPF.variable_cp_power(pm, nw=n)  #  Eq. (3.22), (3.23)
            eDisGo_OPF.variable_dsm_storage_power(pm, nw=n)  # Eq. (3.26), (3.27)
            eDisGo_OPF.variable_slack_gen(pm, nw=n)  # keine Bounds für Slack Generator

            if PowerModels.ref(pm, 1, :opf_version) in(3, 4)
                eDisGo_OPF.variable_slack_HV_requirements(pm, nw=n) # Nicht Teil der MA
                if PowerModels.ref(pm, 1, :opf_version) in(3)
                    eDisGo_OPF.variable_gen_power_curt(pm, nw=n) # Nicht Teil der MA
                end
                for i in PowerModels.ids(pm, :HV_requirements, nw=n)
                    eDisGo_OPF.constraint_HV_requirements(pm, i, n) # Nicht Teil der MA
                end
            end
        else
            throw(ArgumentError("OPF version $(PowerModels.ref(pm, 1, :opf_version)) is not implemented! Choose between version 1 to 4."))
        end

        # CONSTRAINTS
        for i in PowerModels.ids(pm, :bus, nw=n)
            eDisGo_OPF.constraint_power_balance_bf(pm, i, nw=n) # Eq. (3.2ii), (3.3ii), (3.4ii), (3.5ii) für Version 1 und 3 bzw. iii für Version 2 und 4
            # zudem Eq. (3.2i) und (3.3i) für die Storages (virtuelle Leitungen)
        end
        for i in PowerModels.ids(pm, :branch, nw=n)
            eDisGo_OPF.constraint_voltage_magnitude_difference_radial(pm, i, nw=n) # Eq. (3.6)
        end
        eDisGo_OPF.constraint_model_current(pm, nw=n)  # Eq. (3.7) bzw. (3.7i) (je nachdem ob nicht-konvex oder konvex gelöst wird) und (3.7ii)
        eDisGo_OPF.constraint_max_line_loading(pm, n)  # Eq. (3.8)

        for i in PowerModels.ids(pm, :heatpumps, nw=n)
            eDisGo_OPF.constraint_hp_operation(pm, i, n) # Eq. (3.15)
        end

    end

    # CONSTRAINTS
    network_ids = sort(collect(PowerModels.nw_ids(pm)))
    for kind in ["storage", "heat_storage", "dsm"]
        n_1 = network_ids[1]
        for i in PowerModels.ids(pm, Symbol(kind), nw=n_1)
            eDisGo_OPF.constraint_store_state(pm, i, nw=n_1, kind=kind)  # Eq. (3.11)+(3.12), (3.17)+(3.18), (3.24)+(3.25)
        end

        for n_2 in network_ids[2:end]
            for i in PowerModels.ids(pm, Symbol(kind), nw=n_2)
                eDisGo_OPF.constraint_store_state(pm, i, n_1, n_2, kind) # Eq. (3.12), (3.18), (3.25)
            end
            n_1 = n_2
        end
    end

    n_1 = network_ids[1]

    for i in PowerModels.ids(pm, :electromobility, nw=n_1)
        eta = PowerModels.ref(pm, 1, :electromobility)[i]["eta"]
        eDisGo_OPF.constraint_cp_state_initial(pm, n_1, i, eta)  # Eq. (3.20)
    end

    for n_2 in network_ids[2:end]
        for i in PowerModels.ids(pm, :electromobility, nw=n_2)
            eta = PowerModels.ref(pm, 1, :electromobility)[i]["eta"]
            eDisGo_OPF.constraint_cp_state(pm, n_1, n_2, i, eta) # Eq. (3.21) (und (3.20) für letzten Zeitschritt)
        end
        n_1 = n_2
    end

    # OBJECTIVE FUNCTION
    if PowerModels.ref(pm, 1, :opf_version) in(1,3)
        #eDisGo_OPF.objective_min_losses(pm)
        eDisGo_OPF.objective_min_line_loading_max(pm) # Eq. (3.1 i)
        if (PowerModels.ref(pm, 1, :opf_version) == 3) # Nicht Teil der MA
            #objective_min_hv_slacks(pm)
        end
    elseif PowerModels.ref(pm, 1, :opf_version) in(2,4)
        eDisGo_OPF.objective_min_losses_slacks(pm)  # Eq. (3.1 ii)
        if (PowerModels.ref(pm, 1, :opf_version) == 4) # Nicht Teil der MA
            #objective_min_hv_slacks(pm)
        end
    end
end
