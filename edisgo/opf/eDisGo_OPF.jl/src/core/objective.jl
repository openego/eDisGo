"""
    objective_edisgo(pm::AbstractBFModelEdisgo; kwargs...)

Unified objective function for eDisGo optimization problems.
Flexibly combines different optimization goals based on configuration.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels optimization model

# Keyword Arguments
- `minimize_losses::Bool=true`: Minimize line losses
- `minimize_line_loading::Bool=false`: Minimize maximum line loading
- `minimize_slacks::Bool=false`: Minimize grid constraint violation slacks
- `minimize_hv_slacks::Bool=false`: Minimize high voltage requirement slacks
- `minimize_14a_curtailment::Bool=false`: Minimize §14a curtailment (HPs and CPs)
- `weight_losses::Float64`: Weight for line losses (auto-calculated if not provided)
- `weight_line_loading::Float64`: Weight for line loading (auto-calculated if not provided)
- `weight_slacks::Float64`: Weight for slacks (default: 0.6)
- `weight_hv_slacks::Float64`: Weight for HV slacks (auto-calculated if not provided)
- `weight_14a::Float64`: Weight for §14a curtailment (default: 0.5)
- `weight_heat_storage_violation::Float64`: Large penalty for heat storage violations (default: 1e4)

# Returns
- JuMP objective expression
"""
function objective_edisgo(pm::AbstractBFModelEdisgo;
    minimize_losses::Bool = true,
    minimize_line_loading::Bool = false,
    minimize_slacks::Bool = false,
    minimize_hv_slacks::Bool = false,
    minimize_14a_curtailment::Bool = false,
    weight_losses::Union{Float64,Nothing} = nothing,
    weight_line_loading::Union{Float64,Nothing} = nothing,
    weight_slacks::Float64 = 0.6,
    weight_hv_slacks::Union{Float64,Nothing} = nothing,
    weight_14a::Float64 = 0.5,
    weight_heat_storage_violation::Float64 = 1e4
)
    nws = PowerModels.nw_ids(pm)
    objective_terms = []
    
    # Extract common network parameters for automatic weight calculation
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    storage = Dict(i => get(branch, "storage", 1.0) for (i,branch) in PowerModels.ref(pm, 1, :branch))
    
    # Auto-calculate weights if not provided
    if weight_losses === nothing
        weight_losses = 1.0 - weight_slacks  # Complementary to slacks
    end
    
    if weight_line_loading === nothing
        weight_line_loading = 0.1  # Default for line loading
    end
    
    # 1. LINE LOSSES (almost always included)
    if minimize_losses
        ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
        
        loss_term = weight_losses * sum(
            sum(ccm[n][b] * r[n][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from) if storage[b] == 0) 
            for n in nws
        )
        push!(objective_terms, loss_term)
    end
    
    # 2. LINE LOADING (minimize maximum line loading)
    if minimize_line_loading
        ll = PowerModels.var(pm, 1, :ll)
        
        loading_term = weight_line_loading * sum(
            (ll[(b,i,j)] - 1) * c[1][b] * l[1][b] 
            for (b,i,j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0
        )
        push!(objective_terms, loading_term)
    end
    
    # 3. GRID SLACKS (curtailment and load shedding for flexible components)
    if minimize_slacks
        # Generator curtailment slacks
        if haskey(PowerModels.ref(pm, 1), :gen_nd) && length(PowerModels.ref(pm, 1, :gen_nd)) > 0
            pgc = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
            slack_gen_nd = weight_slacks * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm, 1, :gen_nd))) for n in nws)
            push!(objective_terms, slack_gen_nd)
        end
        
        if haskey(PowerModels.ref(pm, 1), :gen) && length(PowerModels.ref(pm, 1, :gen)) > 0
            pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
            slack_gen = weight_slacks * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm, 1, :gen))) for n in nws)
            push!(objective_terms, slack_gen)
        end
        
        # Load shedding slacks
        if haskey(PowerModels.ref(pm, 1), :load) && length(PowerModels.ref(pm, 1, :load)) > 0
            pds = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
            slack_load = weight_slacks * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm, 1, :load))) for n in nws)
            push!(objective_terms, slack_load)
        end
        
        # Charging point slacks
        if haskey(PowerModels.ref(pm, 1), :electromobility) && length(PowerModels.ref(pm, 1, :electromobility)) > 0
            pcps = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
            slack_cp = weight_slacks * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm, 1, :electromobility))) for n in nws)
            push!(objective_terms, slack_cp)
        end
        
        # Heat pump slacks
        if haskey(PowerModels.ref(pm, 1), :heatpumps) && length(PowerModels.ref(pm, 1, :heatpumps)) > 0
            phps = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
            slack_hp = weight_slacks * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) for n in nws)
            push!(objective_terms, slack_hp)
        end
    end
    
    # 4. §14a CURTAILMENT (minimize use of §14a support generators)
    if minimize_14a_curtailment
        # Heat pump §14a support
        p_hp14a = Dict(n => get(PowerModels.var(pm, n), :p_hp14a, Dict()) for n in nws)
        if any(length(p_hp14a[n]) > 0 for n in nws)
            hp14a_term = weight_14a * sum(sum(p_hp14a[n][i] for i in keys(p_hp14a[n])) for n in nws)
            push!(objective_terms, hp14a_term)
        end
        
        # Charging point §14a support
        p_cp14a = Dict(n => get(PowerModels.var(pm, n), :p_cp14a, Dict()) for n in nws)
        if any(length(p_cp14a[n]) > 0 for n in nws)
            cp14a_term = weight_14a * sum(sum(p_cp14a[n][i] for i in keys(p_cp14a[n])) for n in nws)
            push!(objective_terms, cp14a_term)
        end
    end
    
    # 5. HIGH VOLTAGE SLACKS (overlying grid requirements)
    if minimize_hv_slacks
        if haskey(PowerModels.ref(pm, 1), :HV_requirements)
            phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)
            
            # Auto-calculate HV slack weight if not provided
            if weight_hv_slacks === nothing
                parameters = [r[1][i] for i in keys(r[1]) if r[1][i] > 0]
                if !isempty(parameters)
                    if minimize_line_loading
                        # For line loading optimization, use different scaling
                        parameters2 = [l[1][i] * c[1][i] for i in keys(c[1]) if l[1][i] * c[1][i] > 0]
                        weight_hv_slacks = 7.5 * exp10(floor(log10(maximum(0.01 * parameters2))) + 1)
                    else
                        # For losses/slacks optimization
                        weight_hv_slacks = exp10(floor(log10(maximum(parameters))) + 1)
                    end
                else
                    weight_hv_slacks = 1.0
                end
            end
            
            # HV slacks for non-DSM components
            hv_slack_term = weight_hv_slacks * sum(
                sum(phvs[n][i]^2 * flex["count"] for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"] != "dsm") 
                for n in nws
            )
            push!(objective_terms, hv_slack_term)
            
            # HV slacks for DSM components (lower weight)
            hv_slack_dsm_term = weight_hv_slacks * 1e-1 * sum(
                sum(phvs[n][i]^2 * flex["count"] for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"] == "dsm") 
                for n in nws
            )
            push!(objective_terms, hv_slack_dsm_term)
        end
    end
    
    # 6. HEAT STORAGE VIOLATIONS (always included if heat pumps exist, high penalty)
    if haskey(PowerModels.ref(pm, 1), :heatpumps) && length(PowerModels.ref(pm, 1, :heatpumps)) > 0
        phss = Dict(n => PowerModels.var(pm, n, :phss) for n in nws)
        phps2 = Dict(n => PowerModels.var(pm, n, :phps2) for n in nws)
        
        hs_violation_term = weight_heat_storage_violation * sum(
            sum(phss[n][i] + phps2[n][i] for i in keys(PowerModels.ref(pm, 1, :heatpumps))) 
            for n in nws
        )
        push!(objective_terms, hs_violation_term)
    end
    
    # Combine all objective terms
    if isempty(objective_terms)
        @warn "No objective terms added! Using dummy objective."
        return JuMP.@objective(pm.model, Min, 0.0)
    end
    
    return JuMP.@objective(pm.model, Min, sum(objective_terms))
end


"""
    objective_by_version(pm::AbstractBFModelEdisgo, opf_version::Int)

Wrapper function for backward compatibility with old opf_version system.
Maps opf_version numbers to appropriate objective_edisgo configurations.

# Arguments
- `pm::AbstractBFModelEdisgo`: PowerModels optimization model
- `opf_version::Int`: Version number (1-4)

# OPF Versions
- Version 1: Minimize line losses and maximal line loading
- Version 2: Minimize line losses and grid related slacks (with §14a)
- Version 3: Minimize line losses, maximal line loading and HV slacks
- Version 4: Minimize line losses, HV slacks and grid related slacks

# Returns
- JuMP objective expression
"""
function objective_by_version(pm::AbstractBFModelEdisgo, opf_version::Int)
    if opf_version == 1
        # Version 1: Minimize line losses and line loading
        return objective_edisgo(pm,
            minimize_losses = true,
            minimize_line_loading = false,  # In original, line loading was commented out
            minimize_slacks = false,
            minimize_hv_slacks = false,
            minimize_14a_curtailment = false
        )
    elseif opf_version == 2
        # Version 2: Minimize line losses and slacks (with §14a support)
        return objective_edisgo(pm,
            minimize_losses = true,
            minimize_line_loading = false,
            minimize_slacks = true,
            minimize_hv_slacks = false,
            minimize_14a_curtailment = true,
            weight_slacks = 0.6,
            weight_14a = 0.5
        )
    elseif opf_version == 3
        # Version 3: Minimize line losses and line loading (with §14a support)
        return objective_edisgo(pm,
            minimize_losses = true,
            minimize_line_loading = true,
            minimize_slacks = false,
            minimize_hv_slacks = false,
            minimize_14a_curtailment = true,
            weight_line_loading = 0.1,
            weight_14a = 0.05
        )
    elseif opf_version == 4
        # Version 4: Minimize line losses, slacks and HV slacks
        return objective_edisgo(pm,
            minimize_losses = true,
            minimize_line_loading = false,
            minimize_slacks = true,
            minimize_hv_slacks = true,
            minimize_14a_curtailment = false,
            weight_slacks = 0.6
        )
    else
        error("Invalid opf_version: $opf_version. Must be 1, 2, 3, or 4.")
    end
end


# ==============================================================================
# LEGACY FUNCTIONS (kept for reference, use objective_by_version instead)
# ==============================================================================

# OPF Version 1: Minimize line losses and maximal line loading
function objective_min_losses(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    p = Dict(n => PowerModels.var(pm, n, :p) for n in nws)
    q = Dict(n => PowerModels.var(pm, n, :q) for n in nws)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    s_nom = Dict(n => Dict(i => get(branch, "rate_a", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    storage = Dict(i => get(branch, "storage", 1.0) for (i,branch) in PowerModels.ref(pm, 1, :branch))
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .>0]

    return JuMP.@objective(pm.model, Min,
        sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from) if storage[b] == 0) for n in nws) # minimize line losses
        #+ factor2 * sum(sum((p[n][(b,i,j)]^2+q[n][(b,i,j)]^2)/s_nom[n][b]^2 * c[1][b]*l[1][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)  # minimize line loading * c[n][b]*l[n][b]
    )
end

# OPF Version 2: Minimize line losses and grid related slacks
function objective_min_losses_slacks(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    pgc = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
    pds = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
    pcps = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
    phps = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    phps2 = Dict(n => PowerModels.var(pm, n, :phps2) for n in nws)
    phss = Dict(n => PowerModels.var(pm, n, :phss) for n in nws)
    
    # §14a virtual generators for HPs and CPs
    p_hp14a = Dict(n => get(PowerModels.var(pm, n), :p_hp14a, Dict()) for n in nws)
    p_cp14a = Dict(n => get(PowerModels.var(pm, n), :p_cp14a, Dict()) for n in nws)
    
    factor_slacks = 0.6
    factor_14a = 0.5  # Weight for §14a curtailment (between slacks and losses)
    
    return JuMP.@objective(pm.model, Min,
        (1-factor_slacks) * sum(sum(ccm[n][b] * r[n][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from) ) for n in nws) # minimize line losses incl. storage losses
        + factor_slacks  * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm,1 , :gen_nd))) for n in nws) # minimize non-dispatchable curtailment
        + factor_slacks  * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm,1 , :gen))) for n in nws) # minimize dispatchable curtailment
        + factor_slacks  * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm,1 , :load))) for n in nws) # minimize load shedding
        + factor_slacks  * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm,1 , :electromobility))) for n in nws) # minimize cp load sheddin
        + factor_slacks * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm,1 , :heatpumps))) for n in nws) # minimize hp load shedding
        + 1e4 * sum(sum(phss[n][i] + phps2[n][i] for i in keys(PowerModels.ref(pm, 1 , :heatpumps))) for n in nws)
        + factor_14a * sum(sum(p_hp14a[n][i] for i in keys(p_hp14a[n])) for n in nws)  # minimize §14a HP curtailment support
        + factor_14a * sum(sum(p_cp14a[n][i] for i in keys(p_cp14a[n])) for n in nws)  # minimize §14a CP curtailment support
    )
end

# OPF Version 3: Minimize line losses, maximal line loading and HV slacks
function objective_min_line_loading_max(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    ll = PowerModels.var(pm, 1, :ll)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    storage = Dict(i => get(branch, "storage", 1.0) for (i,branch) in PowerModels.ref(pm, 1, :branch))
    
    # §14a virtual generators for HPs and CPs
    p_hp14a = Dict(n => get(PowerModels.var(pm, n), :p_hp14a, Dict()) for n in nws)
    p_cp14a = Dict(n => get(PowerModels.var(pm, n), :p_cp14a, Dict()) for n in nws)
    
    factor_ll = 0.1
    factor_14a = 0.05  # Small penalty for §14a usage in line loading optimization
    
    return JuMP.@objective(pm.model, Min,
        (1-factor_ll) * sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        + factor_ll * sum((ll[(b,i,j)]-1) * c[1][b] * l[1][b]  for (b,i,j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0)  # minimize max line loading
        + factor_14a * sum(sum(p_hp14a[n][i] for i in keys(p_hp14a[n])) for n in nws)  # minimize §14a HP curtailment support
        + factor_14a * sum(sum(p_cp14a[n][i] for i in keys(p_cp14a[n])) for n in nws)  # minimize §14a CP curtailment support
    )
end


# OPF Version 4: Minimize line losses, HV slacks and grid related slacks (with overlying grid)
function objective_min_losses_slacks_OG(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    pgc = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
    pds = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
    pcps = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
    phps = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    phps2 = Dict(n => PowerModels.var(pm, n, :phps2) for n in nws)
    phss = Dict(n => PowerModels.var(pm, n, :phss) for n in nws)
    phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .>0]
    #factor_hv_slacks = length(nws) * exp10(floor(log10(maximum(parameters)))+2)
    factor_hv_slacks = exp10(floor(log10(maximum(parameters)))+1)
    #println(factor_hv_slacks)
    factor_slacks = 0.6
    return JuMP.@objective(pm.model, Min,
        (1-factor_slacks) * sum(sum(ccm[n][b]*r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        + factor_slacks  * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm,1 , :gen_nd))) for n in nws) # minimize non-dispatchable curtailment
        + factor_slacks  * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm,1 , :gen))) for n in nws) # minimize dispatchable curtailment
        + factor_slacks  * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm,1 , :load))) for n in nws) # minimize load shedding
        + factor_slacks  * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm,1 , :electromobility))) for n in nws) # minimize cp load shedding
        + factor_slacks * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm, 1 , :heatpumps))) for n in nws) # minimize hp load shedding
        + factor_hv_slacks * sum(sum(phvs[n][i]^2 * flex["count"] for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"]!= "dsm") for n in nws)  #
        + factor_hv_slacks * 1e-1 * sum(sum(phvs[n][i]^2 * flex["count"] for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"]== "dsm") for n in nws) #
        + 1e4 * sum(sum(phss[n][i] + phps2[n][i] for i in keys(PowerModels.ref(pm, 1 , :heatpumps))) for n in nws)
    )
end

# OPF Version 3 (alternative): Minimize line losses, maximal line loading and HV slacks (with overlying grid)
function objective_min_line_loading_max_OG(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    ll = PowerModels.var(pm, 1, :ll)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    storage = Dict(i => get(branch, "storage", 1.0) for (i,branch) in PowerModels.ref(pm, 1, :branch))
    phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .>0]
    parameters2 = [l[1][i]*c[1][i] for i in keys(c[1])]
    parameters2 = parameters2[parameters2 .>0]
    factor_ll = 0.01
    println(factor_ll)
    factor_hv_slacks = 7.5 *  exp10(floor(log10(maximum(0.01*parameters2)))+1)
    println(factor_hv_slacks)
    return JuMP.@objective(pm.model, Min,
        (1 - factor_ll) * sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        + factor_ll * sum((ll[(b,i,j)]-1) * c[1][b] * l[1][b]  for (b,i,j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0)  # minimize max line loading
        + factor_hv_slacks * sum(sum(phvs[n][i]^2 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"]!= "dsm") for n in nws)  #
        + factor_hv_slacks * 1e-1 * sum(sum(phvs[n][i]^2 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"]== "dsm") for n in nws) #
    )
end
