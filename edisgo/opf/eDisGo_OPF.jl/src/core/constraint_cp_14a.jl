"""
Constraints for §14a EnWG charging point curtailment using virtual generators.

This file implements §14a curtailment by modeling virtual generators at each
charging point bus. The virtual generator can reduce the net electrical load, 
simulating the effect of curtailment while maintaining a minimum power level.
"""

"""
    constraint_cp_14a_binary_coupling(pm, i, nw)

Couples binary variable with power variable for §14a support generator.
When binary variable is 0, power must be 0. When binary is 1, power can be between 0 and pmax.
This ensures time budget tracking works correctly.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels model
- `i::Int`: Generator index
- `nw::Int`: Network (timestep) index
"""
function constraint_cp_14a_binary_coupling(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    gen_cp14a = PowerModels.ref(pm, nw, :gen_cp_14a, i)
    p_cp14a = PowerModels.var(pm, nw, :p_cp14a, i)
    z_cp14a = PowerModels.var(pm, nw, :z_cp14a, i)
    
    # p ≤ pmax × z  (if z=0 then p=0, if z=1 then p can be 0..pmax)
    JuMP.@constraint(pm.model, p_cp14a <= gen_cp14a["pmax"] * z_cp14a)
end


"""
    constraint_cp_14a_min_net_load(pm, i, nw)

Ensures that the net electrical load (charging point load - virtual generator support)
stays above the §14a minimum power level (typically 4.2 kW = 0.0042 MW).

For flexible CPs (in electromobility dict): uses optimization variable `pcp`.
For fixed CPs (in load dict): uses fixed load parameter `load["pd"]`.

Big-M formulation:
  p_cp - p_cp14a >= p_min_14a - M * (1 - z_cp14a)
  When z=0 (inactive): constraint relaxed
  When z=1 (active):   p_cp - p_cp14a >= p_min_14a (net load >= 4.2 kW)

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels model
- `i::Int`: Virtual generator index
- `nw::Int`: Network (timestep) index
"""
function constraint_cp_14a_min_net_load(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    gen_cp14a = PowerModels.ref(pm, nw, :gen_cp_14a, i)
    cp_idx = gen_cp14a["cp_index"]

    # Virtual generator support variable
    p_cp14a = PowerModels.var(pm, nw, :p_cp14a, i)

    # §14a minimum power (per unit)
    p_min_14a = gen_cp14a["p_min_14a"]

    # Maximum support capacity
    p_max_support = gen_cp14a["pmax"]

    if p_max_support < 1e-6
        # Charging point too small for §14a curtailment, disable virtual generator
        JuMP.@constraint(pm.model, p_cp14a == 0.0)
        return
    end

    # Binary time variable 
    z_cp14a = PowerModels.var(pm, nw, :z_cp14a, i)
    M = p_max_support + p_min_14a

    # Check if CP is flexible (in electromobility dict) or simple load
    if haskey(PowerModels.ref(pm, nw), :electromobility) && haskey(PowerModels.ref(pm, nw, :electromobility), cp_idx)
        # Flexible CP: use optimization VARIABLE
        pcp = PowerModels.var(pm, nw, :pcp, cp_idx)
        JuMP.@constraint(pm.model, pcp - p_cp14a >= p_min_14a - M * (1 - z_cp14a))
    else
        # Non-flexible CP: use fixed load timeseries
        cp_name = gen_cp14a["cp_name"]
        load_found = false
        for (load_id, load) in PowerModels.ref(pm, nw, :load)
            if haskey(load, "name") && load["name"] == cp_name
                p_cp_load = load["pd"]
                load_found = true
                if p_cp_load > 1e-6
                    # Fixed load with Big-M: when z=1, enforce min net load
                    JuMP.@constraint(pm.model, p_cp_load - p_cp14a >= p_min_14a - M * (1 - z_cp14a))
                else
                    # CP is off, no support needed
                    JuMP.@constraint(pm.model, p_cp14a == 0.0)
                end
                break
            end
        end

        if !load_found
            @warn "Could not find load for charging point $(cp_name), skipping constraint"
            return
        end
    end
end


"""
    constraint_cp_14a_time_budget_daily(pm, day_start, day_end, i)

Limits the usage of §14a support generator to a maximum number of hours per day.
This is implemented by counting the number of timesteps where the binary variable is 1.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels model
- `day_start::Int`: First timestep of the day
- `day_end::Int`: Last timestep of the day
- `i::Int`: Virtual generator index
"""
function constraint_cp_14a_time_budget_daily(pm::AbstractBFModelEdisgo, day_start::Int, day_end::Int, i::Int)
    # Get time step duration in hours
    if haskey(PowerModels.ref(pm, day_start), :time_elapsed)
        time_elapsed = PowerModels.ref(pm, day_start, :time_elapsed)
    else
        Memento.warn(_LOGGER, "network data should specify time_elapsed, using 1.0 as default")
        time_elapsed = 1.0
    end
    
    gen_cp14a = PowerModels.ref(pm, day_start, :gen_cp_14a, i)
    max_hours = gen_cp14a["max_hours_per_day"]
    
    # Collect binary variables for all timesteps of the day
    z_cp14a_day = [PowerModels.var(pm, t, :z_cp14a, i) for t in day_start:day_end]
    
    # Maximum number of active timesteps
    max_active_steps = max_hours / time_elapsed
    
    # Sum of binary variables must not exceed budget
    JuMP.@constraint(pm.model, sum(z_cp14a_day) <= max_active_steps)
end


"""
    constraint_cp_14a_time_budget_total(pm, i, nws)

Alternative to daily budget: Limits total usage over entire optimization horizon.
Can be used instead of daily budget for simpler formulation.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels model
- `i::Int`: Virtual generator index
- `nws`: Network IDs (all timesteps)
"""
function constraint_cp_14a_time_budget_total(pm::AbstractBFModelEdisgo, i::Int, nws)
    # Get time step duration
    if haskey(PowerModels.ref(pm, first(nws)), :time_elapsed)
        time_elapsed = PowerModels.ref(pm, first(nws), :time_elapsed)
    else
        time_elapsed = 1.0
    end
    
    gen_cp14a = PowerModels.ref(pm, first(nws), :gen_cp_14a, i)
    max_hours_per_day = gen_cp14a["max_hours_per_day"]
    
    # Calculate total hours available (number of days × hours per day)
    num_timesteps = length(nws)
    num_days = ceil(num_timesteps * time_elapsed / 24.0)
    total_max_hours = max_hours_per_day * num_days
    
    # Collect all binary variables
    z_cp14a_all = [PowerModels.var(pm, t, :z_cp14a, i) for t in nws]
    
    # Total active timesteps
    max_active_steps = total_max_hours / time_elapsed
    
    JuMP.@constraint(pm.model, sum(z_cp14a_all) <= max_active_steps)
end
