"""
Penalty weight per unit of the heat balance slacks `phss` and `phps2`, in every
objective.

The heat pump balance is `cop·(php + phps2) == pd + phss − phs`. `phps2` covers heat
demand that heat pump and heat storage cannot supply, `phss` takes up surplus heat
that neither the heat demand nor the heat storage can. Both are meant for
inconsistent heat pump data, not as a grid-restriction slack: without a price the
solver lowers `php` and covers the heat demand with `phps2`, which appears in no
power balance.

The value is fixed. In `opf_version` 1 and 2 it is far above the other terms for
realistic grids. In versions 3 and 4 the quadratic penalty on the overlying-grid
requirement slacks `phvs` can exceed it for a large deviation from the heat pump
requirement; the solver then uses the heat balance slacks instead.
"""
const HEAT_SLACK_PENALTY = 1e4

"""
Penalty term `HEAT_SLACK_PENALTY · Σₙ (Σᵢ phss[n][i] + Σᵢ phps2[n][i])`, shared by
the objectives of all `opf_version`s. `phss` is summed over the heat storage ids
and `phps2` over the heat pump ids of network 1.
"""
function heat_slack_penalty(pm::AbstractBFModelEdisgo)
    hs = keys(PowerModels.ref(pm, 1, :heat_storage))
    hp = keys(PowerModels.ref(pm, 1, :heatpumps))
    return JuMP.@expression(pm.model, HEAT_SLACK_PENALTY * sum(
        sum(PowerModels.var(pm, n, :phss, i) for i in hs)
        + sum(PowerModels.var(pm, n, :phps2, i) for i in hp)
        for n in PowerModels.nw_ids(pm)))
end

"""
Objective: minimise total ohmic line losses, summed over all time steps, as
`Σ ccm[b]·r[b]` over the non-storage branches. (Defined as an alternative
objective; the multi-network build uses `objective_min_line_loading_max` for
`opf_version` 1.)
"""
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

"""
Objective for `opf_version` 2 — Eq. (3.2 iii) of the eDisGo OPF formulation:
minimise a weighted sum of line losses and the grid-restriction slacks. Combines
ohmic losses (`Σ ccm·r`) with penalties on non-dispatchable curtailment (`pgc`),
dispatchable-generation curtailment (`pgens`), load shedding (`pds`),
charging-point shedding (`pcps`) and heat-pump shedding (`phps`), weighted by
`factor_slacks`, plus the penalty `heat_slack_penalty` on the heat balance
slacks (`phss`, `phps2`).
"""
function objective_min_losses_slacks(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    pgc = Dict(n => PowerModels.var(pm, n, :pgc) for n in nws)
    pgens = Dict(n => PowerModels.var(pm, n, :pgens) for n in nws)
    pds = Dict(n => PowerModels.var(pm, n, :pds) for n in nws)
    pcps = Dict(n => PowerModels.var(pm, n, :pcps) for n in nws)
    phps = Dict(n => PowerModels.var(pm, n, :phps) for n in nws)
    factor_slacks = 0.6
    return JuMP.@objective(pm.model, Min,
        (1-factor_slacks) * sum(sum(ccm[n][b] * r[n][b] for (b,i,j) in PowerModels.ref(pm, n, :arcs_from) ) for n in nws) # minimize line losses incl. storage losses
        + factor_slacks  * sum(sum(pgc[n][i] for i in keys(PowerModels.ref(pm,1 , :gen_nd))) for n in nws) # minimize non-dispatchable curtailment
        + factor_slacks  * sum(sum(pgens[n][i] for i in keys(PowerModels.ref(pm,1 , :gen))) for n in nws) # minimize dispatchable curtailment
        + factor_slacks  * sum(sum(pds[n][i] for i in keys(PowerModels.ref(pm,1 , :load))) for n in nws) # minimize load shedding
        + factor_slacks  * sum(sum(pcps[n][i] for i in keys(PowerModels.ref(pm,1 , :electromobility))) for n in nws) # minimize cp load sheddin
        + factor_slacks * sum(sum(phps[n][i] for i in keys(PowerModels.ref(pm,1 , :heatpumps))) for n in nws) # minimize hp load shedding
        + heat_slack_penalty(pm) # heat balance slacks
    )
end

"""
Objective for `opf_version` 1 — Eq. (3.2 ii) of the eDisGo OPF formulation:
minimise a weighted sum of line losses (`Σ ccm·r`) and the maximum line loading
`ll` (weighted by `factor_ll`, scaled by branch cost × length) over the
non-storage branches, plus the penalty `heat_slack_penalty` on the heat balance
slacks (`phss`, `phps2`).
"""
function objective_min_line_loading_max(pm::AbstractBFModelEdisgo)
    nws = PowerModels.nw_ids(pm)
    ccm = Dict(n => PowerModels.var(pm, n, :ccm) for n in nws)
    r = Dict(n => Dict(i => get(branch, "br_r", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))  for n in nws)
    ll = PowerModels.var(pm, 1, :ll)
    l = Dict(n => Dict(i => get(branch, "length", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    c = Dict(n => Dict(i => get(branch, "cost", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch)) for n in nws)
    storage = Dict(i => get(branch, "storage", 1.0) for (i,branch) in PowerModels.ref(pm, 1, :branch))
    factor_ll = 0.1
    return JuMP.@objective(pm.model, Min,
        (1-factor_ll) * sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        + factor_ll * sum((ll[(b,i,j)]-1) * c[1][b] * l[1][b]  for (b,i,j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0)  # minimize max line loading
        + heat_slack_penalty(pm) # heat balance slacks
    )
end


# OPF with overlying grid
"""
Objective for `opf_version` 4 (with overlying grid): like
`objective_min_losses_slacks`, but additionally penalises the overlying-grid (HV)
requirement slacks `phvs` (with a reduced weight for DSM requirements). The HV
penalty factor is scaled from the branch resistances.
"""
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
        + heat_slack_penalty(pm) # heat balance slacks
    )
end

"""
Objective for `opf_version` 3 (with overlying grid): like
`objective_min_line_loading_max` (losses + maximum line loading), but additionally
penalises the overlying-grid (HV) requirement slacks `phvs` (reduced weight for
DSM). The curtailment `pgc` has no cost of its own; the "curt" requirement and the
lower bound of its slack (`variable_slack_HV_requirements_real`) limit it. The
heat balance slacks carry the penalty `heat_slack_penalty`.
"""
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
    factor_hv_slacks = 7.5 *  exp10(floor(log10(maximum(0.01*parameters2)))+1)
    return JuMP.@objective(pm.model, Min,
        (1 - factor_ll) * sum(sum(ccm[n][b] * r[n][b]  for (b,i,j) in PowerModels.ref(pm, n, :arcs_from)) for n in nws) # minimize line losses
        + factor_ll * sum((ll[(b,i,j)]-1) * c[1][b] * l[1][b]  for (b,i,j) in PowerModels.ref(pm, 1, :arcs_from) if storage[b] == 0)  # minimize max line loading
        + factor_hv_slacks * sum(sum(phvs[n][i]^2 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"]!= "dsm") for n in nws)  #
        + factor_hv_slacks * 1e-1 * sum(sum(phvs[n][i]^2 for (i, flex) in PowerModels.ref(pm, n, :HV_requirements) if flex["name"]== "dsm") for n in nws) #
        + heat_slack_penalty(pm) # heat balance slacks
    )
end
