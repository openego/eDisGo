"Solve multinetwork branch flow OPF with multiple flexibilities"
function solve_mn_opf_bf_flex(file, model_type::Type{T}, optimizer; kwargs...) where T <: AbstractBFModel
    return eDisGo_OPF.solve_model(file, model_type, optimizer, build_mn_opf_bf_flex; multinetwork=true, kwargs...)
end


"Build multinetwork branch flow OPF with multiple flexibilities"
function build_mn_opf_bf_flex(pm::AbstractBFModelEdisgo)
    opf_version = PowerModels.ref(pm, 1, :opf_version)

    if opf_version in(1, 3)
        eDisGo_OPF.variable_max_line_loading(pm, nw=1) # Eq. (3.41) (nur für Version 1 und 3)
    end

    for (n, network) in PowerModels.nws(pm)
        # VARIABLES
        if opf_version in(1, 2, 3, 4, 5)
            eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false) # keine Begrenzung für Leistung auf Leitungen/Trafos (Strombegrenzung stattdessen)

            if opf_version in(1, 3) # nur für Version 1 und 3 (ohne Netzrestriktionen)
                eDisGo_OPF.variable_branch_current(pm, nw=n, bounded=false) # keine Eq. (3.7)!
                eDisGo_OPF.variable_bus_voltage(pm, nw=n, bounded=false) # keine Eq. (3.8)!
                eDisGo_OPF.constraint_max_line_loading(pm, n)  # Eq. (3.40)
            else # Version 2, 4, 5 (mit Netzrestriktionen)
                eDisGo_OPF.variable_branch_current(pm, nw=n)  # Eq. (3.7) und (3.7i)
                eDisGo_OPF.variable_gen_power_curt(pm, nw=n)  # Eq. (3.44) für non-dispatchable Generators
                eDisGo_OPF.variable_slack_grid_restrictions(pm, nw=n) # Eq. (3.44)-(3.47)
                eDisGo_OPF.variable_bus_voltage(pm, nw=n)  # Eq. (3.8)
            end

            eDisGo_OPF.variable_slack_heat_pump_storage(pm, nw=n) # Eq. (3.44)-(3.47)
            eDisGo_OPF.variable_slack_gen(pm, nw=n)  # keine Bounds für Slack Generator

            # Version 5: ONLY §14a as flexibility
            # - HP power (php) is FIXED to pd/cop (current demand)
            # - CP power (pcp) is FIXED to pd (current demand from load)
            # - No battery storage, heat storage, or DSM flexibility
            # - Only p_hp14a and p_cp14a can change to reduce load
            if opf_version == 5
                eDisGo_OPF.variable_heat_pump_power_fixed(pm, nw=n)
                eDisGo_OPF.variable_cp_power_fixed(pm, nw=n)
                # No storage/DSM - create empty dicts for power balance compatibility
                PowerModels.var(pm, n)[:ps] = Dict{Int, JuMP.VariableRef}()
                PowerModels.var(pm, n)[:phs] = Dict{Int, JuMP.VariableRef}()
                PowerModels.var(pm, n)[:pdsm] = Dict{Int, JuMP.VariableRef}()

                # Fix ALL feasibility slacks to 0: if §14a alone is not sufficient,
                # the model must become infeasible (no hidden load shedding allowed)
                for i in PowerModels.ids(pm, :load, nw=n)
                    JuMP.fix(PowerModels.var(pm, n, :pds, i), 0.0; force=true)
                end
                for i in PowerModels.ids(pm, :heatpumps, nw=n)
                    JuMP.fix(PowerModels.var(pm, n, :phps, i), 0.0; force=true)
                end
                for i in PowerModels.ids(pm, :electromobility, nw=n)
                    JuMP.fix(PowerModels.var(pm, n, :pcps, i), 0.0; force=true)
                end
                for i in PowerModels.ids(pm, :gen, nw=n)
                    JuMP.fix(PowerModels.var(pm, n, :pgens, i), 0.0; force=true)
                end
                for i in PowerModels.ids(pm, :gen_nd, nw=n)
                    JuMP.fix(PowerModels.var(pm, n, :pgc, i), 0.0; force=true)
                end
            else
                eDisGo_OPF.variable_battery_storage(pm, nw=n)  # Eq. (3.11) und (3.12)
                eDisGo_OPF.variable_heat_storage(pm, nw=n)  # Eq. (3.24)
                eDisGo_OPF.variable_heat_pump_power(pm, nw=n)  # Eq. (3.20)
                eDisGo_OPF.variable_cp_power(pm, nw=n)  #  Eq. (3.27), (3.28)
                eDisGo_OPF.variable_dsm_storage_power(pm, nw=n)  # Eq. (3.34), (3.35)
            end

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

            if opf_version in(3, 4) # Nicht Teil der MA
                eDisGo_OPF.variable_slack_HV_requirements(pm, nw=n)
                if opf_version in(3)
                    eDisGo_OPF.variable_gen_power_curt(pm, nw=n)
                end
                for i in PowerModels.ids(pm, :HV_requirements, nw=n)
                    eDisGo_OPF.constraint_HV_requirements(pm, i, n)
                end
            end
        else
            throw(ArgumentError("OPF version $(opf_version) is not implemented! Choose between version 1 to 5."))
        end

        # CONSTRAINTS
        for i in PowerModels.ids(pm, :bus, nw=n)
            eDisGo_OPF.constraint_power_balance_bf(pm, i, nw=n) # Eq. (3.3ii), (3.4ii) für Version 1 und 3 bzw. (3.3iii), (3.4iii) für Version 2 und 4
        end
        for i in PowerModels.ids(pm, :branch, nw=n)
            eDisGo_OPF.constraint_voltage_magnitude_difference_radial(pm, i, nw=n) # Eq. (3.5)
        end
        eDisGo_OPF.constraint_model_current(pm, nw=n)  # Eq. (3.6) bzw. (3.6i) (je nachdem ob nicht-konvex oder konvex gelöst wird) und (3.6ii)

        # HP operation constraint only for versions with flexible HPs (not v5)
        if opf_version != 5
            for i in PowerModels.ids(pm, :heatpumps, nw=n)
                eDisGo_OPF.constraint_hp_operation(pm, i, n) # Eq. (3.19)
            end
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

    # Storage/DSM state constraints - only for versions with flexible storage (not v5)
    network_ids = sort(collect(PowerModels.nw_ids(pm)))

    if opf_version != 5
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
    end

    # §14a EnWG daily time budget constraints for heat pumps
    # Apply time budget constraint (max 2 hours/day) for each day in the optimization horizon
    # Groups timesteps into 24-hour periods based on time_elapsed and constrains z_hp14a binary
    if haskey(PowerModels.ref(pm, 1), :gen_hp_14a) && !isempty(PowerModels.ref(pm, 1, :gen_hp_14a))
        # Determine timesteps per day based on time_elapsed (in hours)
        # Example: time_elapsed = 1.0 → 24 timesteps/day, time_elapsed = 0.5 → 48 timesteps/day
        n_first = network_ids[1]
        time_elapsed = PowerModels.ref(pm, n_first, :time_elapsed)
        timesteps_per_day = Int(round(24.0 / time_elapsed))

        # Group network_ids into days and apply constraint per day per generator
        # This ensures the §14a time budget (typically 2 hours/day) is respected
        for day_start_idx in 1:timesteps_per_day:length(network_ids)
            day_end_idx = min(day_start_idx + timesteps_per_day - 1, length(network_ids))
            day_network_ids = network_ids[day_start_idx:day_end_idx]

            # Apply daily time budget constraint for each §14a generator
            # Constraint: sum(z_hp14a[t] * time_elapsed for t in day) <= max_hours_per_day
            for i in PowerModels.ids(pm, :gen_hp_14a, nw=network_ids[1])
                eDisGo_OPF.constraint_hp_14a_time_budget_daily(pm, day_network_ids[1], day_network_ids[end], i)
            end
        end
    end

    # §14a EnWG daily time budget constraints for charging points
    # Apply time budget constraint (max 2 hours/day) for each day in the optimization horizon
    # Groups timesteps into 24-hour periods based on time_elapsed and constrains z_cp14a binary
    if haskey(PowerModels.ref(pm, 1), :gen_cp_14a) && !isempty(PowerModels.ref(pm, 1, :gen_cp_14a))
        # Determine timesteps per day based on time_elapsed (in hours)
        # Example: time_elapsed = 1.0 → 24 timesteps/day, time_elapsed = 0.5 → 48 timesteps/day
        n_first = network_ids[1]
        time_elapsed = PowerModels.ref(pm, n_first, :time_elapsed)
        timesteps_per_day = Int(round(24.0 / time_elapsed))

        # Group network_ids into days and apply constraint per day per generator
        # This ensures the §14a time budget (typically 2 hours/day) is respected
        for day_start_idx in 1:timesteps_per_day:length(network_ids)
            day_end_idx = min(day_start_idx + timesteps_per_day - 1, length(network_ids))
            day_network_ids = network_ids[day_start_idx:day_end_idx]

            # Apply daily time budget constraint for each §14a CP generator
            # Constraint: sum(z_cp14a[t] * time_elapsed for t in day) <= max_hours_per_day
            for i in PowerModels.ids(pm, :gen_cp_14a, nw=network_ids[1])
                eDisGo_OPF.constraint_cp_14a_time_budget_daily(pm, day_network_ids[1], day_network_ids[end], i)
            end
        end
    end

    # OBJECTIVE FUNCTION
    if opf_version == 1
        #eDisGo_OPF.objective_min_losses(pm)
        eDisGo_OPF.objective_min_line_loading_max(pm) # Eq. (3.2 ii)
    elseif opf_version == 3 # Nicht Teil der MA
        eDisGo_OPF.objective_min_line_loading_max_OG(pm)
    elseif opf_version == 2
        eDisGo_OPF.objective_min_losses_slacks(pm)  # Eq. (3.2 iii)
    elseif opf_version == 4
        eDisGo_OPF.objective_min_losses_slacks_OG(pm)  # Nicht Teil der MA
    elseif opf_version == 5
        eDisGo_OPF.objective_min_losses_14a_only(pm)  # §14a als einzige Flexibilität, Feasibility-Slacks extrem bestraft
    end
end
