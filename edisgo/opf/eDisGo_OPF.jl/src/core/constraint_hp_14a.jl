"""
Constraints for §14a EnWG heat pump curtailment using virtual generators.

This file implements §14a curtailment by modeling virtual generators at each
heat pump bus. The virtual generator can reduce the net electrical load, 
simulating the effect of curtailment while maintaining a minimum power level.
"""

"""
    constraint_hp_14a_binary_coupling(pm, i, nw)

Couples binary variable with power variable for §14a support generator.
When binary variable is 0, power must be 0. When binary is 1, power can be between 0 and pmax.
This ensures time budget tracking works correctly.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels model
- `i::Int`: Generator index
- `nw::Int`: Network (timestep) index
"""
function constraint_hp_14a_binary_coupling(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    gen_hp14a = PowerModels.ref(pm, nw, :gen_hp_14a, i)
    p_hp14a = PowerModels.var(pm, nw, :p_hp14a, i)
    z_hp14a = PowerModels.var(pm, nw, :z_hp14a, i)
    
    # p ≤ pmax × z  (if z=0 then p=0, if z=1 then p can be 0..pmax)
    JuMP.@constraint(pm.model, p_hp14a <= gen_hp14a["pmax"] * z_hp14a)
end


"""
    constraint_hp_14a_min_net_load(pm, i, nw)

Ensures that the net electrical load (heat pump load - virtual generator support)
stays above the §14a minimum power level (typically 4.2 kW = 0.0042 MW).

Uses the HP optimization variable `php` (not the fixed parameter) so the constraint
correctly tracks the actual HP electrical draw after heat storage optimization.

Big-M formulation:
  php - p_hp14a >= p_min_14a - M * (1 - z_hp14a)
  When z=0 (inactive): constraint relaxed (always satisfied)
  When z=1 (active):   php - p_hp14a >= p_min_14a (net load >= 4.2 kW)

If php < p_min_14a at a timestep, z is forced to 0 (no curtailment possible),
which in turn forces p_hp14a = 0 via binary coupling. This correctly handles
cases where the HP draws less than 4.2 kW.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels model
- `i::Int`: Virtual generator index
- `nw::Int`: Network (timestep) index
"""
function constraint_hp_14a_min_net_load(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    gen_hp14a = PowerModels.ref(pm, nw, :gen_hp_14a, i)
    hp_idx = gen_hp14a["hp_index"]

    # Get the actual HP electrical power VARIABLE (not the fixed parameter)
    php = PowerModels.var(pm, nw, :php, hp_idx)

    # Virtual generator support variable
    p_hp14a = PowerModels.var(pm, nw, :p_hp14a, i)

    # §14a minimum power (per unit)
    p_min_14a = gen_hp14a["p_min_14a"]

    # Maximum support capacity
    p_max_support = gen_hp14a["pmax"]

    if p_max_support < 1e-6
        # Heat pump too small for §14a curtailment, disable virtual generator
        JuMP.@constraint(pm.model, p_hp14a == 0.0)
    else
        # Big-M formulation: when z=1 (curtailment active), enforce min net load
        # when z=0 (curtailment inactive), constraint is relaxed
        z_hp14a = PowerModels.var(pm, nw, :z_hp14a, i)
        M = p_max_support + p_min_14a
        JuMP.@constraint(pm.model, php - p_hp14a >= p_min_14a - M * (1 - z_hp14a))
    end
end


"""
    constraint_hp_14a_time_budget_daily(pm, day_start, day_end, i)

Limits the usage of §14a support generator to a maximum number of hours per day.
This is implemented by counting the number of timesteps where the binary variable is 1.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels model
- `day_start::Int`: First timestep of the day
- `day_end::Int`: Last timestep of the day
- `i::Int`: Virtual generator index
"""
function constraint_hp_14a_time_budget_daily(pm::AbstractBFModelEdisgo, day_start::Int, day_end::Int, i::Int)
    # Get time step duration in hours
    if haskey(PowerModels.ref(pm, day_start), :time_elapsed)
        time_elapsed = PowerModels.ref(pm, day_start, :time_elapsed)
    else
        Memento.warn(_LOGGER, "network data should specify time_elapsed, using 1.0 as default")
        time_elapsed = 1.0
    end
    
    gen_hp14a = PowerModels.ref(pm, day_start, :gen_hp_14a, i)
    max_hours = gen_hp14a["max_hours_per_day"]
    
    # Collect binary variables for all timesteps of the day
    z_hp14a_day = [PowerModels.var(pm, t, :z_hp14a, i) for t in day_start:day_end]
    
    # Maximum number of active timesteps
    max_active_steps = max_hours / time_elapsed
    
    # Sum of binary variables must not exceed budget
    JuMP.@constraint(pm.model, sum(z_hp14a_day) <= max_active_steps)
end


"""
    constraint_hp_14a_time_budget_total(pm, i, nws)

Alternative to daily budget: Limits total usage over entire optimization horizon.
Can be used instead of daily budget for simpler formulation.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels model
- `i::Int`: Virtual generator index
- `nws`: Network IDs (all timesteps)
"""
function constraint_hp_14a_time_budget_total(pm::AbstractBFModelEdisgo, i::Int, nws)
    # Get time step duration
    if haskey(PowerModels.ref(pm, first(nws)), :time_elapsed)
        time_elapsed = PowerModels.ref(pm, first(nws), :time_elapsed)
    else
        time_elapsed = 1.0
    end
    
    gen_hp14a = PowerModels.ref(pm, first(nws), :gen_hp_14a, i)
    max_hours_per_day = gen_hp14a["max_hours_per_day"]
    
    # Calculate total hours available (number of days × hours per day)
    num_timesteps = length(nws)
    num_days = ceil(num_timesteps * time_elapsed / 24.0)
    total_max_hours = max_hours_per_day * num_days
    
    # Collect all binary variables
    z_hp14a_all = [PowerModels.var(pm, t, :z_hp14a, i) for t in nws]
    
    # Total active timesteps
    max_active_steps = total_max_hours / time_elapsed
    
    JuMP.@constraint(pm.model, sum(z_hp14a_all) <= max_active_steps)
end
