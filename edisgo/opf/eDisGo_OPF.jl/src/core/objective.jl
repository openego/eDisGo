function objective_min_losses(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    p = Dict(n => PowerModels.var(pm, n, :p) for n in nws)
    q = Dict(n => PowerModels.var(pm, n, :q) for n in nws)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    s_nom = Dict(n => Dict(i => get(branch, "rate_a", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    parameters = [r[1][i] for i in keys(r[1])]
    parameters = parameters[parameters .>0]
    factor = 1
    while true
        if minimum(factor*parameters) > 1e0
            break
        else
            factor = 10*factor
        end
    end
    parameters2 = [c[1][b]*l[1][b]/s_nom[1][b]^2 for b in keys(c[1])]
    factor2 = 1
    while true
        if (maximum(factor2*parameters2) > maximum(factor*parameters)) &
            (minimum(factor2*parameters2) > -1e-5)
            break
        else
            factor2 = 10*factor2
        end
    end
    println(factor2)
    return JuMP.@objective(pm.model, Min,
        factor * sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        + factor2 * sum(sum((p[n][(b,i,j)]^2+q[n][(b,i,j)]^2)/s_nom[n][b]^2 * c[n][b]*l[n][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)  # minimize line loading * c[n][b]*l[n][b]
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
    c = Dict(n => Dict(i => get(branch, "cost_factor", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    p = Dict(n => PowerModels.var(pm, n, :p) for n in nws)
    q = Dict(n => PowerModels.var(pm, n, :q) for n in nws)
    s_nom = Dict(n => Dict(i => get(branch, "rate_a", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    parameters = [r[1][i] for i in keys(c[1])]
    parameters = parameters[parameters .>0]
    factor = 1
    while true
        if minimum(factor*parameters) > 1e-1
            break
        else
            factor = 10*factor
        end
    end
    #println(factor)


    parameters2 = [c[1][b]/s_nom[1][b]^2 for b in keys(c[1])]
    factor2 = 1
    while true
        if (maximum(factor2*parameters2) > maximum(factor*parameters))
            break
        else
            factor2 = 10*factor2
        end
    end
    println(factor2)
    factor_slacks = exp10(floor(log10(maximum(factor2*parameters2))))

    return JuMP.@objective(pm.model, Min,
        factor  * sum(sum(ccm[n][b]*r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        + factor2  * sum(sum((p[n][(b,i,j)]^2+q[n][(b,i,j)]^2)/s_nom[n][b]^2 * c[n][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)  # minimize line loading
        + factor_slacks  * sum(sum(pgc[n][i]^2 for i in keys(PowerModels.ref(pm,1 , :gen_nd))) for n in nws) # minimize non-dispatchable curtailment
        + factor_slacks  * sum(sum(pgens[n][i]^2 for i in keys(PowerModels.ref(pm,1 , :gen))) for n in nws) # minimize dispatchable curtailment
        + factor_slacks  * sum(sum(pds[n][i]^2 for i in keys(PowerModels.ref(pm,1 , :load))) for n in nws) # minimize load shedding
        + factor_slacks  * sum(sum(pcps[n][i]^2 for i in keys(PowerModels.ref(pm,1 , :electromobility))) for n in nws) # minimize cp load shedding
    )
end


function objective_min_hv_slacks(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)

    return JuMP.@objective(pm.model, Min,
        sum(sum(phvs[n][i]^2 * 1e5 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)) for n in nws) # minimize HV req. slack variables
    )
end