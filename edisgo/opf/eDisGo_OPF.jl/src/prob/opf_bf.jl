"Solve multinetwork branch flow OPF with multiple flexibilities"
function solve_mn_opf_bf_flex(file, model_type::Type{T}, optimizer; kwargs...) where T <: AbstractBFModel
    return eDisGo_OPF.solve_model(file, model_type, optimizer, build_mn_opf_bf_flex; multinetwork=true, kwargs...)
end


"Build multinetwork branch flow OPF with multiple flexibilities"
function build_mn_opf_bf_flex(pm::AbstractBFModelEdisgo)
    # Check if line loading variable is needed (opf_version or objective_config)
    needs_line_loading = false
    if PowerModels.ref(pm, 1, :opf_version) in(1, 3)
        needs_line_loading = true
    elseif haskey(PowerModels.ref(pm, 1), :objective_config)
        obj_config = PowerModels.ref(pm, 1, :objective_config)
        needs_line_loading = get(obj_config, "minimize_line_loading", false)
    end
    
    if needs_line_loading
        eDisGo_OPF.variable_max_line_loading(pm, nw=1) # Eq. (3.41)
    end
    
    for (n, network) in PowerModels.nws(pm)
        # VARIABLES
        if PowerModels.ref(pm, 1, :opf_version) in(1, 2, 3, 4)
            eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false) # keine Begrenzung für Leistung auf Leitungen/Trafos (Strombegrenzung stattdessen)
            if PowerModels.ref(pm, 1, :opf_version) in(1, 3) || needs_line_loading # Version 1/3 oder wenn minimize_line_loading aktiv
                eDisGo_OPF.variable_branch_current(pm, nw=n, bounded=false) # keine Eq. (3.7)!
                eDisGo_OPF.variable_bus_voltage(pm, nw=n, bounded=false) # keine Eq. (3.8)!
                if needs_line_loading
                    eDisGo_OPF.constraint_max_line_loading(pm, n)  # Eq. (3.40)
                end
            elseif PowerModels.ref(pm, 1, :opf_version) in(2, 4) # Version 2 und 4 (mit Netzrestriktionen)
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
            
            # §14a EnWG virtual generators for heat pump support
            if haskey(PowerModels.ref(pm, n), :gen_hp_14a) && !isempty(PowerModels.ref(pm, n, :gen_hp_14a))
                eDisGo_OPF.variable_gen_hp_14a_power(pm, nw=n)
                eDisGo_OPF.variable_gen_hp_14a_binary(pm, nw=n)
            end

            # §14a EnWG virtual generators for charging point support
            if haskey(PowerModels.ref(pm, n), :gen_cp_14a) && !isempty(PowerModels.ref(pm, n, :gen_cp_14a))
                eDisGo_OPF.variable_gen_cp_14a_power(pm, nw=n)
                eDisGo_OPF.variable_gen_cp_14a_binary(pm, nw=n)
            end

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
        
        # §14a EnWG constraints for virtual generators
        if haskey(PowerModels.ref(pm, n), :gen_hp_14a) && !isempty(PowerModels.ref(pm, n, :gen_hp_14a))
            for i in PowerModels.ids(pm, :gen_hp_14a, nw=n)
                eDisGo_OPF.constraint_hp_14a_binary_coupling(pm, i, n)
                eDisGo_OPF.constraint_hp_14a_min_net_load(pm, i, n)
            end
        end

        # §14a EnWG constraints for charging point virtual generators
        if haskey(PowerModels.ref(pm, n), :gen_cp_14a) && !isempty(PowerModels.ref(pm, n, :gen_cp_14a))
            for i in PowerModels.ids(pm, :gen_cp_14a, nw=n)
                eDisGo_OPF.constraint_cp_14a_binary_coupling(pm, i, n)
                eDisGo_OPF.constraint_cp_14a_min_net_load(pm, i, n)
            end
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

    # §14a EnWG daily time budget constraints
    if haskey(PowerModels.ref(pm, 1), :gen_hp_14a) && !isempty(PowerModels.ref(pm, 1, :gen_hp_14a))
        println("\n" * "="^80)
        println("🔍 JULIA DEBUG: §14a Generators")
        println("="^80)
        
        gen_hp_14a_dict = PowerModels.ref(pm, 1, :gen_hp_14a)
        println("Number of gen_hp_14a entries: ", length(gen_hp_14a_dict))
        
        # Show first 5 generators
        count = 0
        for (idx, gen) in gen_hp_14a_dict
            count += 1
            if count <= 5
                println("  [$idx]: hp_name=$(get(gen, "hp_name", "N/A")), hp_index=$(get(gen, "hp_index", "N/A")), pmax=$(get(gen, "pmax", "N/A"))")
            end
        end
        println("="^80 * "\n")
        
        # Determine timesteps per day based on time_elapsed (in hours)
        n_first = network_ids[1]
        time_elapsed = PowerModels.ref(pm, n_first, :time_elapsed)
        timesteps_per_day = Int(round(24.0 / time_elapsed))
        
        # Group network_ids into days
        for day_start_idx in 1:timesteps_per_day:length(network_ids)
            day_end_idx = min(day_start_idx + timesteps_per_day - 1, length(network_ids))
            day_network_ids = network_ids[day_start_idx:day_end_idx]
            
            # Apply daily time budget constraint for each §14a generator
            for i in PowerModels.ids(pm, :gen_hp_14a, nw=network_ids[1])
                # Call with correct argument order: (pm, day_start, day_end, i)
                eDisGo_OPF.constraint_hp_14a_time_budget_daily(pm, day_network_ids[1], day_network_ids[end], i)
            end
        end
    else
        println("\n⚠ JULIA DEBUG: No gen_hp_14a found or empty!\n")
    end

    # §14a EnWG daily time budget constraints for charging points
    if haskey(PowerModels.ref(pm, 1), :gen_cp_14a) && !isempty(PowerModels.ref(pm, 1, :gen_cp_14a))
        println("\n" * "="^80)
        println("🔍 JULIA DEBUG: §14a Charging Point Generators")
        println("="^80)
        
        gen_cp_14a_dict = PowerModels.ref(pm, 1, :gen_cp_14a)
        println("Number of gen_cp_14a entries: ", length(gen_cp_14a_dict))
        
        # Show first 5 generators
        count = 0
        for (idx, gen) in gen_cp_14a_dict
            count += 1
            if count <= 5
                println("  [$idx]: cp_name=$(get(gen, "cp_name", "N/A")), cp_index=$(get(gen, "cp_index", "N/A")), pmax=$(get(gen, "pmax", "N/A"))")
            end
        end
        println("="^80 * "\n")
        
        # Determine timesteps per day based on time_elapsed (in hours)
        n_first = network_ids[1]
        time_elapsed = PowerModels.ref(pm, n_first, :time_elapsed)
        timesteps_per_day = Int(round(24.0 / time_elapsed))
        
        # Group network_ids into days
        for day_start_idx in 1:timesteps_per_day:length(network_ids)
            day_end_idx = min(day_start_idx + timesteps_per_day - 1, length(network_ids))
            day_network_ids = network_ids[day_start_idx:day_end_idx]
            
            # Apply daily time budget constraint for each §14a generator
            for i in PowerModels.ids(pm, :gen_cp_14a, nw=network_ids[1])
                # Call with correct argument order: (pm, day_start, day_end, i)
                eDisGo_OPF.constraint_cp_14a_time_budget_daily(pm, day_network_ids[1], day_network_ids[end], i)
            end
        end
    else
        println("\n⚠ JULIA DEBUG: No gen_cp_14a found or empty!\n")
    end

    # OBJECTIVE FUNCTION
    # Check if custom objective configuration is provided
    if haskey(PowerModels.ref(pm, 1), :objective_config)
        obj_config = PowerModels.ref(pm, 1, :objective_config)
        println("\n✓ JULIA DEBUG: Using custom objective_config: ", obj_config)
        
        # Extract configuration with defaults
        minimize_losses = get(obj_config, "minimize_losses", true)
        minimize_line_loading = get(obj_config, "minimize_line_loading", false)
        minimize_slacks = get(obj_config, "minimize_slacks", false)
        minimize_hv_slacks = get(obj_config, "minimize_hv_slacks", false)
        minimize_14a_curtailment = get(obj_config, "minimize_14a_curtailment", false)
        
        # Extract weights (will be auto-calculated in objective_edisgo if not provided)
        weight_losses = get(obj_config, "weight_losses", nothing)
        weight_line_loading = get(obj_config, "weight_line_loading", nothing)
        weight_slacks = get(obj_config, "weight_slacks", 0.6)
        weight_hv_slacks = get(obj_config, "weight_hv_slacks", nothing)
        weight_14a = get(obj_config, "weight_14a", 0.5)
        
        # Call unified objective with custom configuration
        eDisGo_OPF.objective_edisgo(pm,
            minimize_losses=minimize_losses,
            minimize_line_loading=minimize_line_loading,
            minimize_slacks=minimize_slacks,
            minimize_hv_slacks=minimize_hv_slacks,
            minimize_14a_curtailment=minimize_14a_curtailment,
            weight_losses=weight_losses,
            weight_line_loading=weight_line_loading,
            weight_slacks=weight_slacks,
            weight_hv_slacks=weight_hv_slacks,
            weight_14a=weight_14a
        )
    else
        # Use backward compatible version mapping
        opf_version = PowerModels.ref(pm, 1, :opf_version)
        println("\n✓ JULIA DEBUG: Using opf_version: ", opf_version)
        eDisGo_OPF.objective_by_version(pm, opf_version)
    end
end
