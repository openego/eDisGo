"Solve multinetwork branch flow OPF with multiple flexibilities"
function solve_mn_opf_bf_flex(file, model_type::Type{T}, optimizer; kwargs...) where T <: AbstractBFModel
    return eDisGo_OPF.solve_model(file, model_type, optimizer, build_mn_opf_bf_flex; multinetwork=true, kwargs...)
end


"Build multinetwork branch flow OPF with multiple flexibilities"
function build_mn_opf_bf_flex(pm::AbstractBFModelEdisgo)
    if PowerModels.ref(pm, 1, :opf_version) in(1, 3)
        eDisGo_OPF.variable_max_line_loading(pm, nw=1) # Eq. (3.41) (nur für Version 1 und 3)
    end
    for (n, network) in PowerModels.nws(pm)
        # VARIABLES
        if PowerModels.ref(pm, 1, :opf_version) in(1, 2, 3, 4)
            eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false) # keine Begrenzung für Leistung auf Leitungen/Trafos (Strombegrenzung stattdessen)
            if PowerModels.ref(pm, 1, :opf_version) in(1, 3) # nur für Version 1 und 3 (ohne Netzrestriktionen)
                eDisGo_OPF.variable_branch_current(pm, nw=n, bounded=false) # keine Eq. (3.7)!
                eDisGo_OPF.variable_bus_voltage(pm, nw=n, bounded=false) # keine Eq. (3.8)!
                eDisGo_OPF.constraint_max_line_loading(pm, n)  # Eq. (3.40)
            else # nur für Version 2 und 4 (mit Netzrestriktionen)
                eDisGo_OPF.variable_branch_current(pm, nw=n)  # Eq. (3.7) und (3.7i)
                eDisGo_OPF.variable_gen_power_curt(pm, nw=n)  # Eq. (3.44) für non-dispatchable Generators
                eDisGo_OPF.variable_slack_grid_restrictions(pm, nw=n) # Eq. (3.44)-(3.47)
                eDisGo_OPF.variable_bus_voltage(pm, nw=n)  # Eq. (3.8)
            end
            eDisGo_OPF.variable_slack_heat_pump_storage(pm, nw=n) # Eq. (3.44)-(3.47)
            eDisGo_OPF.variable_battery_storage(pm, nw=n)  # Eq. (3.11) und (3.12)
            eDisGo_OPF.variable_heat_storage(pm, nw=n)  # Eq. (3.24)
            eDisGo_OPF.variable_heat_pump_power(pm, nw=n)  # Eq. (3.20)
            eDisGo_OPF.variable_cp_power(pm, nw=n)  #  Eq. (3.27), (3.28)
            eDisGo_OPF.variable_dsm_storage_power(pm, nw=n)  # Eq. (3.34), (3.35)
            eDisGo_OPF.variable_slack_gen(pm, nw=n)  # keine Bounds für Slack Generator

            if PowerModels.ref(pm, 1, :opf_version) in(3, 4) # Nicht Teil der MA
                eDisGo_OPF.variable_slack_HV_requirements(pm, nw=n)
                if PowerModels.ref(pm, 1, :opf_version) in(3)
                    eDisGo_OPF.variable_gen_power_curt(pm, nw=n)
                end
                for i in PowerModels.ids(pm, :HV_requirements, nw=n)
                    eDisGo_OPF.constraint_HV_requirements(pm, i, n)
                end
            end
        else
            throw(ArgumentError("OPF version $(PowerModels.ref(pm, 1, :opf_version)) is not implemented! Choose between version 1 to 4."))
        end

        # CONSTRAINTS
        for i in PowerModels.ids(pm, :bus, nw=n)
            eDisGo_OPF.constraint_power_balance_bf(pm, i, nw=n) # Eq. (3.3ii), (3.4ii) für Version 1 und 3 bzw. (3.3iii), (3.4iii) für Version 2 und 4
        end
        for i in PowerModels.ids(pm, :branch, nw=n)
            eDisGo_OPF.constraint_voltage_magnitude_difference_radial(pm, i, nw=n) # Eq. (3.5)
        end
        eDisGo_OPF.constraint_model_current(pm, nw=n)  # Eq. (3.6) bzw. (3.6i) (je nachdem ob nicht-konvex oder konvex gelöst wird) und (3.6ii)


        for i in PowerModels.ids(pm, :heatpumps, nw=n)
            eDisGo_OPF.constraint_hp_operation(pm, i, n) # Eq. (3.19)
        end

    end

    # CONSTRAINTS
    network_ids = sort(collect(PowerModels.nw_ids(pm)))
    for kind in ["storage", "heat_storage", "dsm"]
        n_1 = network_ids[1]
        for i in PowerModels.ids(pm, Symbol(kind), nw=n_1)
            eDisGo_OPF.constraint_store_state(pm, i, nw=n_1, kind=kind)  # Eq. (3.9)+(3.10), (3.22)+(3.23), (3.32)+(3.33)
        end

        for n_2 in network_ids[2:end]
            for i in PowerModels.ids(pm, Symbol(kind), nw=n_2)
                eDisGo_OPF.constraint_store_state(pm, i, n_1, n_2, kind) # Eq. (3.10), (3.23), (3.33)
            end
            n_1 = n_2
        end
    end

    n_1 = network_ids[1]

    for i in PowerModels.ids(pm, :electromobility, nw=n_1)
        eta = PowerModels.ref(pm, 1, :electromobility)[i]["eta"]
        eDisGo_OPF.constraint_cp_state_initial(pm, n_1, i, eta)  # Eq. (3.25)
    end

    for n_2 in network_ids[2:end]
        for i in PowerModels.ids(pm, :electromobility, nw=n_2)
            eta = PowerModels.ref(pm, 1, :electromobility)[i]["eta"]
            eDisGo_OPF.constraint_cp_state(pm, n_1, n_2, i, eta) # Eq. (3.26) (und (3.25) für letzten Zeitschritt)
        end
        n_1 = n_2
    end

    # OBJECTIVE FUNCTION
    if PowerModels.ref(pm, 1, :opf_version) == 1
        #eDisGo_OPF.objective_min_losses(pm)
        eDisGo_OPF.objective_min_line_loading_max(pm) # Eq. (3.2 ii)
    elseif (PowerModels.ref(pm, 1, :opf_version) == 3) # Nicht Teil der MA
        eDisGo_OPF.objective_min_line_loading_max_OG(pm)
    elseif PowerModels.ref(pm, 1, :opf_version) == 2
        eDisGo_OPF.objective_min_losses_slacks(pm)  # Eq. (3.2 iii)
    elseif PowerModels.ref(pm, 1, :opf_version) == 4
        eDisGo_OPF.objective_min_losses_slacks_OG(pm)  # Nicht Teil der MA
    end
end
