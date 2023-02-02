function objective_min_line_loading(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    p = Dict(n => PowerModels.var(pm, n, :p) for n in nws)
    l = Dict(n =>
            Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))
        for n in nws)
    c = Dict(n =>
            Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))
        for n in nws)
    s_nom = Dict(n =>
            Dict(i => get(branch, "rate_a", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))
        for n in nws)# p_max?
    phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)
    #qhvs = Dict(n => PowerModels.var(pm, n, :qhvs) for n in nws)
    phps = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    pgs = Dict(n => PowerModels.var(pm, n, :pgs) for n in nws)

    return JuMP.@objective(pm.model, Min,
        #sum(sum(phvs[n][i]^2 * 1e5 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)) for n in nws) # minimize HV req. slack variables
        #+ sum(sum(phps[n][i] * 1e2 for (i, hp) in PowerModels.ref(pm, n, :heatpumps)) for n in nws) # minimize heat pump slack variables
        sum(sum(ccm[n][b]*r[n][b]*1e2 for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        #+ sum(sum(pgs[n]) for n in nws) *1e5
        #+ sum(sum(p[n][(b,i,j)]/s_nom[n][b]*l[n][b]*c[n][b]*1e3 for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)  # minimize line loading
    )
end

function objective_min_losses(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    p = Dict(n => PowerModels.var(pm, n, :p) for n in nws)
    q = Dict(n => PowerModels.var(pm, n, :q) for n in nws)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    s_nom = Dict(n => Dict(i => get(branch, "rate_a", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)# p_max?
    bus = Dict(n => Dict(i => get(branch, "f_bus", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    #v_min = Dict(n => Dict(i => get(bus, "vmin", 1.0) for (i,bus) in PowerModels.ref(pm, n, :branch))  for n in nws)
    # PowerModels.ref(pm, n, :bus)[bus[n][b]]["vmin"]
    parameters = [r[1][i] for i in keys(c[1])]
    parameters = parameters[parameters .>0]
    factor = 1
    while true
        if minimum(factor*parameters) > 1
            break
        else
            factor = 10*factor
        end
    end
    println(factor)

    return JuMP.@objective(pm.model, Min,
        #100 * sum(sum((ccm[n][b] / s_nom[n][b]^2 * 0.81 -0.9)^2 * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses * c[n][b] * l[n][b]
        factor * sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        + sum(sum((p[n][(b,i,j)]^2+q[n][(b,i,j)]^2)/s_nom[n][b]^2 * c[n][b]*l[n][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)  # minimize line loading * c[n][b]*l[n][b]
    )
end

function objective_min_losses_slacks(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    pgc = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
    #phps = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
    pds = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
    pcps = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
    s_base = PowerModels.ref(pm, 1, :baseMVA)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost_factor", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    p = Dict(n => PowerModels.var(pm, n, :p) for n in nws)
    s_nom = Dict(n => Dict(i => get(branch, "rate_a", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)# p_max?
    parameters = [r[1][i]*c[1][i] for i in keys(c[1])]
    parameters = parameters[parameters .>0]
    factor = 1
    while true
        if minimum(factor*parameters) > 1e-3
            break
        else
            factor = 10*factor
        end
    end
    println(factor)
    factor_slacks = exp10(floor(log10(maximum(factor*parameters))) + 2)
    #factor_ll = factor_slacks * 1e-3
    println(factor_slacks)
    return JuMP.@objective(pm.model, Min,
        factor * s_base * sum(sum(ccm[n][b]*r[n][b]*c[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        #+ factor_ll * sum(sum(p[n][(b,i,j)]/s_nom[n][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws)  # minimize line loading
        + factor_slacks * s_base * sum(sum(pgc[n]) for n in nws) # minimize non-dispatchable curtailment
        + factor_slacks * s_base * sum(sum(pgens[n]) for n in nws) # minimize dispatchable curtailment
        #+ s_base * sum(sum(phps[n]) for n in nws) # minimize heatpump slack variables
        + factor_slacks * s_base * sum(sum(pds[n]) for n in nws) # minimize load shedding
        + factor_slacks * s_base * sum(sum(pcps[n]) for n in nws) # minimize cp load shedding
    )
end


function objective_min_hv_slacks(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    phvs = Dict(n => PowerModels.var(pm, n, :phvs) for n in nws)

    return JuMP.@objective(pm.model, Min,
        sum(sum(phvs[n][i]^2 * 1e5 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements)) for n in nws) # minimize HV req. slack variables
    )
end
