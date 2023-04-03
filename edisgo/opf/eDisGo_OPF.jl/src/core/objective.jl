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

function objective_min_losses_slacks(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    pgc = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
    pds = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
    pcps = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
    phps = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost_factor", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    storage = Dict(i => get(branch, "storage", 1.0) for (i,branch) in PowerModels.ref(pm, 1, :branch))
    parameters = [r[1][i] for i in keys(c[1])]
    parameters = parameters[parameters .>0]
    factor_slacks = 5 * maximum(parameters) # exp10(floor(log10(maximum(parameters)))-1)
    println(factor_slacks)
    return JuMP.@objective(pm.model, Min,
        sum(sum(ccm[n][b] * r[n][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from) if storage[b] == 0) for n in nws) # minimize line losses
        + factor_slacks  * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm,1 , :gen_nd))) for n in nws) # minimize non-dispatchable curtailment
        + factor_slacks  * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm,1 , :gen))) for n in nws) # minimize dispatchable curtailment
        + factor_slacks  * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm,1 , :load))) for n in nws) # minimize load shedding
        + factor_slacks  * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm,1 , :electromobility))) for n in nws) # minimize cp load sheddin
        + factor_slacks * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm,1 , :heatpumps))) for n in nws) # minimize hp load shedding
    )
end

function objective_min_line_loading_max(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    ll = PowerModels.var(pm, 1, :ll)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    storage = Dict(i => get(branch, "storage", 1.0) for (i,branch) in PowerModels.ref(pm, 1, :branch))
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .>0]
    factor_ll = exp10(floor(log10(maximum(parameters)))+1)
    println(factor_ll)
    return JuMP.@objective(pm.model, Min,
        sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from) if storage[b] == 0) for n in nws) # minimize line losses
        + factor_ll * sum(ll[(b,i,j)] * c[1][b] * l[1][b]  for (b,i,j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0)  # minimize max line loading
    )
end


# OPF with overlying grid
function objective_min_losses_slacks_OG(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    pgc = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
    pds = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
    pcps = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
    phps = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)
    storage = Dict(i => get(branch, "storage", 1.0) for (i,branch) in PowerModels.ref(pm, 1, :branch))
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    parameters = [r[1][i] for i in keys(c[1])]
    parameters = parameters[parameters .>0]
    factor_slacks = 2 * maximum(parameters) # exp10(floor(log10(maximum(parameters)))+2)

    return JuMP.@objective(pm.model, Min,
        sum(sum(ccm[n][b]*r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from) if storage[b] == 0) for n in nws) # minimize line losses
        + factor_slacks  * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm,1 , :gen_nd))) for n in nws) # minimize non-dispatchable curtailment
        + factor_slacks  * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm,1 , :gen))) for n in nws) # minimize dispatchable curtailment
        + factor_slacks  * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm,1 , :load))) for n in nws) # minimize load shedding
        + factor_slacks  * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm,1 , :electromobility))) for n in nws) # minimize cp load sheddin
        + factor_slacks * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm,1 , :heatpumps))) for n in nws) # minimize hp load shedding
        + sum(sum(phvs[n][i]^2 * factor_slacks * 1e2 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)) for n in nws)
    )
end

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
    factor_ll = exp10(floor(log10(maximum(parameters)))+1)
    println(factor_ll)
    return JuMP.@objective(pm.model, Min,
        sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from) if storage[b] == 0) for n in nws) # minimize line losses
        + factor_ll * sum(ll[(b,i,j)] * c[1][b] * l[1][b]  for (b,i,j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0)  # minimize max line loading
        + sum(sum(phvs[n][i]^2 * 1e4 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)) for n in nws)
    )
end
