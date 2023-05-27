""
function variable_branch_current(pm::AbstractBFModel; kwargs...)
    eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm; kwargs...)
end

function variable_bus_voltage(pm::AbstractBFModel; kwargs...)
    eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm; kwargs...)
end

function variable_buspair_current_magnitude_sqr(pm::AbstractBFModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    branch = PowerModels.ref(pm, nw, :branch)

    ccm = PowerModels.var(pm, nw)[:ccm] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :branch)], base_name="$(nw)_ccm",
        lower_bound = 0.0,
        start = comp_start_value(branch[i], "ccm_start")
    )

    if bounded
        bus = PowerModels.ref(pm, nw, :bus)
        for (i, b) in branch
            rate_a = Inf
            if haskey(b, "rate_a")
                rate_a = b["rate_a"]
            end
            ub = ((rate_a*b["tap"])/(bus[b["f_bus"]]["vmin"]))^2

            if !isinf(ub)
                JuMP.set_upper_bound(ccm[i], ub)
            end
        end
    else
        bus = PowerModels.ref(pm, nw, :bus)
        for (i, b) in branch
            rate_a = Inf
            if haskey(b, "rate_a")
                rate_a = b["rate_a"]
            end
            ub = ((rate_a*b["tap"])/(bus[b["f_bus"]]["vmin"]))^2

            if !isinf(ub)&(b["storage"])
                JuMP.set_upper_bound(ccm[i], ub)
            end
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :branch, :ccm, PowerModels.ids(pm, nw, :branch), ccm)
end


function constraint_voltage_magnitude_difference(pm::AbstractBFModelEdisgo, n::Int, i, f_bus, t_bus, f_idx, t_idx, r, x, tm)
    p_fr = PowerModels.var(pm, n, :p, f_idx)
    q_fr = PowerModels.var(pm, n, :q, f_idx)
    w_fr = PowerModels.var(pm, n, :w, f_bus)
    w_to = PowerModels.var(pm, n, :w, t_bus)
    ccm =  PowerModels.var(pm, n, :ccm, i)

    JuMP.@constraint(pm.model, ((w_fr/tm^2)  - w_to ==  2*(r*p_fr + x*q_fr) - (r^2 + x^2)*ccm))

    if PowerModels.ref(pm, n, :bus)[f_bus]["bus_type"]==3
        JuMP.@constraint(pm.model, (w_fr ==  1))
    end

end


function constraint_model_current(pm::AbstractSOCBFModelEdisgo, n::Int) # Eq. (3.9)
    PowerModels._check_missing_keys(PowerModels.var(pm, n), [:p,:q,:w,:ccm], typeof(pm))

    p  = PowerModels.var(pm, n, :p)
    q  = PowerModels.var(pm, n, :q)
    w  = PowerModels.var(pm, n, :w)
    ccm = PowerModels.var(pm, n, :ccm)

    for (i,branch) in PowerModels.ref(pm, n, :branch)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        f_idx = (i, f_bus, t_bus)
        tm = branch["tap"]
        JuMP.@constraint(pm.model, p[f_idx]^2 + q[f_idx]^2 <= (w[f_bus]/tm^2)*ccm[i])
    end
end

function constraint_model_current(pm::AbstractNCBFModelEdisgo, n::Int) # Eq. (3.5)
    PowerModels._check_missing_keys(PowerModels.var(pm, n), [:p,:q,:w,:ccm], typeof(pm))

    p  = PowerModels.var(pm, n, :p)
    q  = PowerModels.var(pm, n, :q)
    w  = PowerModels.var(pm, n, :w)
    ccm = PowerModels.var(pm, n, :ccm)

    for (i,branch) in PowerModels.ref(pm, n, :branch)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        f_idx = (i, f_bus, t_bus)
        tm = branch["tap"]
        if !(branch["storage"])
            JuMP.@NLconstraint(pm.model, p[f_idx]^2 + q[f_idx]^2 == (w[f_bus]/tm^2)*ccm[i])
        end
    end
end


function constraint_max_line_loading(pm::AbstractSOCBFModelEdisgo, n::Int)
    p  = PowerModels.var(pm, n, :p)
    q  = PowerModels.var(pm, n, :q)
    ll = PowerModels.var(pm, 1, :ll)
    s_nom = Dict(i => get(branch, "rate_a", 1.0) for (i,branch) in PowerModels.ref(pm, n, :branch))

    for (i,branch) in PowerModels.ref(pm, n, :branch)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        f_idx = (i, f_bus, t_bus)
        if !(branch["storage"])
            JuMP.@constraint(pm.model, (p[f_idx]^2 + q[f_idx]^2)/s_nom[i]^2 <= ll[f_idx])
        end
    end
end


function constraint_power_balance(pm::AbstractBFModelEdisgo, n::Int, i, bus_gens, bus_gens_nd, bus_gens_slack, bus_loads, bus_arcs_to, bus_arcs_from, bus_lines_to, bus_storage, bus_pg, bus_qg, bus_pg_nd, bus_qg_nd, bus_pd, bus_qd, branch_r, branch_x, bus_dsm, bus_hps, bus_cps, bus_storage_pf, bus_dsm_pf, bus_hps_pf, bus_cps_pf, bus_gen_nd_pf, bus_gen_d_pf, bus_loads_pf, branch_strg_pf)
    pt   = get(PowerModels.var(pm, n),  :p, Dict()); PowerModels._check_var_keys(pt, bus_arcs_to, "active power", "branch")
    qt   = get(PowerModels.var(pm, n),  :q, Dict()); PowerModels._check_var_keys(qt, bus_arcs_to, "reactive power", "branch")
    pf   = get(PowerModels.var(pm, n),  :p, Dict()); PowerModels._check_var_keys(pf, bus_arcs_from, "active power", "branch")
    qf   = get(PowerModels.var(pm, n),  :q, Dict()); PowerModels._check_var_keys(qf, bus_arcs_from, "reactive power", "branch")
    ps   = get(PowerModels.var(pm, n),  :ps, Dict()); PowerModels._check_var_keys(ps, bus_storage, "active power", "storage")
    pgs  = get(PowerModels.var(pm, n),  :pgs, Dict()); PowerModels._check_var_keys(pgs, bus_gens_slack, "active power", "slack")
    qgs  = get(PowerModels.var(pm, n),  :qgs, Dict()); PowerModels._check_var_keys(qgs, bus_gens_slack, "reactive power", "slack")
    ccm  = get(PowerModels.var(pm, n),  :ccm, Dict()); PowerModels._check_var_keys(ccm, bus_lines_to, "active power", "branch")
    pdsm  = get(PowerModels.var(pm, n),  :pdsm, Dict()); PowerModels._check_var_keys(pdsm, bus_dsm, "active power", "dsm")
    php  = get(PowerModels.var(pm, n),  :php, Dict()); PowerModels._check_var_keys(php, bus_hps, "active power", "heatpumps")
    pcp  = get(PowerModels.var(pm, n),  :pcp, Dict()); PowerModels._check_var_keys(pcp, bus_cps, "active power", "electromobility")

    if PowerModels.ref(pm, 1, :opf_version) in(2, 4)  # Eq. (3.3iii), (3.4iii)
        pgens  = get(PowerModels.var(pm, n),  :pgens, Dict()); PowerModels._check_var_keys(pgens, bus_gens, "active power slack", "curtailment")
        pds  = get(PowerModels.var(pm, n),  :pds, Dict()); PowerModels._check_var_keys(pds, bus_loads, "active power slack", "load")
        pcps  = get(PowerModels.var(pm, n),  :pcps, Dict()); PowerModels._check_var_keys(pcps, bus_cps, "active power slack", "charging point")
        pgc  = get(PowerModels.var(pm, n),  :pgc, Dict()); PowerModels._check_var_keys(pgc, bus_gens_nd, "active power", "curtailment")
        phps  = get(PowerModels.var(pm, n),  :phps, Dict()); PowerModels._check_var_keys(phps, bus_hps, "active power slack", "heatpump")

        cstr_p = JuMP.@constraint(pm.model,
            sum(pt[a] for a in bus_arcs_to)
            ==
            sum(pf[a] for a in bus_arcs_from)
            + sum(ccm[a] * branch_r[a] for a in bus_lines_to)
            - sum(pgs[g] for g in bus_gens_slack)
            - sum(pg for pg in values(bus_pg))
            - sum(pg for pg in values(bus_pg_nd))
            - sum(ps[s] for s in bus_storage)
            + sum(pd for pd in values(bus_pd))
            - sum(pds[l] for l in bus_loads)
            + sum(pgens[g] for g in bus_gens)
            + sum(pgc[g] for g in bus_gens_nd)
            + sum(pdsm[dsm] for dsm in bus_dsm)
            + sum(php[hp] - phps[hp] for hp in bus_hps)
            + sum(pcp[cp] - pcps[cp] for cp in bus_cps)
        )
        cstr_q = JuMP.@constraint(pm.model,
            sum(qt[a] for a in bus_arcs_to)
            - sum(pt[a] * branch_strg_pf[a[1]] for a in bus_arcs_to)
            ==
            sum(qf[a] for a in bus_arcs_from)
            + sum(ccm[a] * branch_x[a] for a in bus_lines_to)
            - sum(qgs[g] for g in bus_gens_slack)
            - sum(qg for qg in values(bus_qg))
            - sum(qg for qg in values(bus_qg_nd))
            + sum(qd for qd in values(bus_qd))
            - sum(pds[l] * bus_loads_pf[l] for l in bus_loads)
            + sum(pgc[g] * bus_gen_nd_pf[g] for g in bus_gens_nd)
            + sum(pgens[g] * bus_gen_d_pf[g] for g in bus_gens)
            + sum(pdsm[dsm] * bus_dsm_pf[dsm] for dsm in bus_dsm)
            + sum((php[hp] - phps[hp]) * bus_hps_pf[hp] for hp in bus_hps)
            + sum((pcp[cp] - pcps[cp]) * bus_cps_pf[cp] for cp in bus_cps)
        )
    else  # Eq. (3.3ii), (3.4ii)
        cstr_p = JuMP.@constraint(pm.model,
            sum(pt[a] for a in bus_arcs_to)
            ==
            sum(pf[a] for a in bus_arcs_from)
            + sum(ccm[a] * branch_r[a] for a in bus_lines_to)
            - sum(pgs[g] for g in bus_gens_slack)
            - sum(pg for pg in values(bus_pg))
            - sum(pg for pg in values(bus_pg_nd))
            - sum(ps[s] for s in bus_storage)
            + sum(pd for pd in values(bus_pd))
            + sum(pdsm[dsm] for dsm in bus_dsm)
            + sum(php[hp] for hp in bus_hps)
            + sum(pcp[cp] for cp in bus_cps)
        )
        cstr_q = JuMP.@constraint(pm.model,
            sum(qt[a] for a in bus_arcs_to)
            - sum(pt[a] * branch_strg_pf[a[1]] for a in bus_arcs_to)
            ==
            sum(qf[a] for a in bus_arcs_from)
            + sum(ccm[a] * branch_x[a] for a in bus_lines_to)
            - sum(qgs[g] for g in bus_gens_slack)
            - sum(qg for qg in values(bus_qg))
            - sum(qg for qg in values(bus_qg_nd))
            + sum(qd for qd in values(bus_qd))
            + sum(pdsm[dsm] * bus_dsm_pf[dsm] for dsm in bus_dsm)
            + sum(php[hp] * bus_hps_pf[hp] for hp in bus_hps)
            + sum(pcp[cp] * bus_cps_pf[cp] for cp in bus_cps)
        )
    end

    if InfrastructureModels.report_duals(pm)
        PowerModels.sol(pm, n, :bus, i)[:lam_kcl_r] = cstr_p
        PowerModels.sol(pm, n, :bus, i)[:lam_kcl_i] = cstr_q
    end
end
