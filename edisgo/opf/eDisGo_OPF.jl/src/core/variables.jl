""
function variable_branch_power_radial(pm::AbstractPowerModel; kwargs...)
    variable_branch_power_real_radial(pm; kwargs...)
    variable_branch_power_imaginary_radial(pm; kwargs...)
end


"variable: `p[l,i,j]` for `(l,i,j)` in `arcs_from`"
function variable_branch_power_real_radial(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    p = PowerModels.var(pm, nw)[:p] = JuMP.@variable(pm.model,
        [(l,i,j) in PowerModels.ref(pm, nw, :arcs_from)], base_name="$(nw)_p",
        start = comp_start_value(PowerModels.ref(pm, nw, :branch, l), "p_start")
    )

    if bounded
        flow_lb, flow_ub = ref_calc_branch_flow_bounds(PowerModels.ref(pm, nw, :branch), PowerModels.ref(pm, nw, :bus))

        for arc in PowerModels.ref(pm, nw, :arcs_from)
            l,i,j = arc
            if !isinf(flow_lb[l])
                JuMP.set_lower_bound(p[arc], flow_lb[l])
            end
            if !isinf(flow_ub[l])
                JuMP.set_upper_bound(p[arc], flow_ub[l])
            end
        end
    end

    for (l,branch) in PowerModels.ref(pm, nw, :branch)
        if haskey(branch, "pf_start")
            f_idx = (l, branch["f_bus"], branch["t_bus"])
            JuMP.set_start_value(p[f_idx], branch["pf_start"])
        end
        if haskey(branch, "pt_start")
            t_idx = (l, branch["t_bus"], branch["f_bus"])
            JuMP.set_start_value(p[t_idx], branch["pt_start"])
        end
    end

    report && eDisGo_OPF.sol_component_value_radial(pm, nw, :branch, :pf, PowerModels.ref(pm, nw, :arcs_from), p)
end

"variable: `q[l,i,j]` for `(l,i,j)` in `arcs`"
function variable_branch_power_imaginary_radial(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    q = PowerModels.var(pm, nw)[:q] = JuMP.@variable(pm.model,
        [(l,i,j) in PowerModels.ref(pm, nw, :arcs_from)], base_name="$(nw)_q",
        start = comp_start_value(PowerModels.ref(pm, nw, :branch, l), "q_start")
    )

    if bounded
        flow_lb, flow_ub = ref_calc_branch_flow_bounds(PowerModels.ref(pm, nw, :branch), PowerModels.ref(pm, nw, :bus))

        for arc in PowerModels.ref(pm, nw, :arcs_from)
            l,i,j = arc
            if !isinf(flow_lb[l])
                JuMP.set_lower_bound(q[arc], flow_lb[l])
            end
            if !isinf(flow_ub[l])
                JuMP.set_upper_bound(q[arc], flow_ub[l])
            end
        end
    end

    for (l,branch) in PowerModels.ref(pm, nw, :branch)
        if haskey(branch, "qf_start")
            f_idx = (l, branch["f_bus"], branch["t_bus"])
            JuMP.set_start_value(q[f_idx], branch["qf_start"])
        end
        if haskey(branch, "qt_start")
            t_idx = (l, branch["t_bus"], branch["f_bus"])
            JuMP.set_start_value(q[t_idx], branch["qt_start"])
        end
    end

    report && eDisGo_OPF.sol_component_value_radial(pm, nw, :branch, :qf, PowerModels.ref(pm, nw, :arcs_from), q)
end

function variable_bus_voltage_magnitude_sqr(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    busses = [i for i in PowerModels.ids(pm, nw, :bus) if !(PowerModels.ref(pm, nw, :bus)[i]["storage"])]
    w = PowerModels.var(pm, nw)[:w] = JuMP.@variable(pm.model,
        [i in busses], base_name="$(nw)_w",
        lower_bound = 0,
        start = comp_start_value(PowerModels.ref(pm, nw, :bus, i), "w_start", 1.001)
    )

    if bounded
        for (i, bus) in PowerModels.ref(pm, nw, :bus)
            if i in busses
                JuMP.set_lower_bound(w[i], bus["vmin"]^2)
                JuMP.set_upper_bound(w[i], bus["vmax"]^2)
            end
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :bus, :w, busses, w)
end

function variable_max_line_loading(pm::AbstractPowerModel; kwargs...)
    variable_line_loading_max(pm; kwargs...)
end


"variable: `ll[l,i,j]` for `(l,i,j)` in `arcs_from`"
function variable_line_loading_max(pm::AbstractPowerModel; nw::Int=nw_id_default, report::Bool=true)
    branches = [(l, i, j) for (l, i, j) in PowerModels.ref(pm, nw, :arcs_from) if !PowerModels.ref(pm, 1, :branch)[l]["storage"]]
    ll = PowerModels.var(pm, nw)[:ll] = JuMP.@variable(pm.model,
        [(l,i,j) in branches], base_name="$(nw)_ll",
        start = comp_start_value(PowerModels.ref(pm, nw, :branch, l), "ll_start"),
        lower_bound = 1
    )

    report && eDisGo_OPF.sol_component_value_radial(pm, nw, :branch, :ll, branches, ll)
end


"generates variables for both `active` and `reactive` non-dispatchable power generation curtailment"
function variable_gen_power_curt(pm::AbstractPowerModel; kwargs...)
    variable_gen_power_curt_real(pm; kwargs...)
    #variable_gen_power_curt_imaginary(pm; kwargs...)
end


"variable: `pgc[j]` for `j` in `gen_nd`"
function variable_gen_power_curt_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pgc = PowerModels.var(pm, nw)[:pgc] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen_nd)], base_name="$(nw)_pgc",
        start = comp_start_value(PowerModels.ref(pm, nw, :gen_nd, i), "pgc_start")
    )

    if bounded
        for (i, gen) in PowerModels.ref(pm, nw, :gen_nd)
            JuMP.set_lower_bound(pgc[i], 0)
            JuMP.set_upper_bound(pgc[i], gen["pg"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :gen_nd, :pgc, PowerModels.ids(pm, nw, :gen_nd), pgc)
end

"variable: `qgc[j]` for `j` in `gen_nd`"
function variable_gen_power_curt_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qgc = PowerModels.var(pm, nw)[:qgc] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen_nd)], base_name="$(nw)_qgc"
    )

    if bounded
        for (i, gen) in PowerModels.ref(pm, nw, :gen_nd)
            JuMP.set_lower_bound(qgc[i], gen["qg"])
            JuMP.set_upper_bound(qgc[i], 0)
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :gen_nd, :qgc, PowerModels.ids(pm, nw, :gen_nd), qgc)
end


"variables for modeling storage units, includes grid injection and internal variables"
function variable_battery_storage(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_battery_storage_power_real(pm; kwargs...)
    PowerModels.variable_storage_energy(pm; kwargs...)
end

""
function variable_battery_storage_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    ps = PowerModels.var(pm, nw)[:ps] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :storage)], base_name="$(nw)_ps",
        start = comp_start_value(PowerModels.ref(pm, nw, :storage, i), "ps_start")
    )

    if bounded
        for (i, storage) in PowerModels.ref(pm, nw, :storage)
            JuMP.set_lower_bound(ps[i], storage["pmin"])
            JuMP.set_upper_bound(ps[i], storage["pmax"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :storage, :ps, PowerModels.ids(pm, nw, :storage), ps)
end

""
function variable_battery_storage_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qs = PowerModels.var(pm, nw)[:qs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :storage)], base_name="$(nw)_qs",
        start = comp_start_value(PowerModels.ref(pm, nw, :storage, i), "qs_start")
    )

    if bounded
        for (i, storage) in PowerModels.ref(pm, nw, :storage)
            JuMP.set_lower_bound(qs[i], storage["qmin"])
            JuMP.set_upper_bound(qs[i], storage["qmax"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :storage, :qs, PowerModels.ids(pm, nw, :storage), qs)
end


### Additional Flexibility Variables (DSM, HS, HP, CP)

"variables for modeling dsm storage units, includes grid injection and internal variables"
function variable_dsm_storage_power(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_dsm_storage_power_real(pm; kwargs...)
    # eDisGo_OPF.variable_dsm_storage_power_imaginary(pm; kwargs...)
    eDisGo_OPF.variable_dsm_storage_energy(pm; kwargs...)
end

""
function variable_dsm_storage_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pdsm = PowerModels.var(pm, nw)[:pdsm] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :dsm)], base_name="$(nw)_pdsm",
        start = comp_start_value(PowerModels.ref(pm, nw, :dsm, i), "pdsm_start")
    )

    if bounded
        dsm = PowerModels.ref(pm, nw, :dsm)
        for (i, s) in dsm
            JuMP.set_lower_bound(pdsm[i], s["p_min"])
            JuMP.set_upper_bound(pdsm[i], s["p_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :dsm, :pdsm, PowerModels.ids(pm, nw, :dsm), pdsm)
end

""
function variable_dsm_storage_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qdsm = PowerModels.var(pm, nw)[:qdsm] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :dsm)], base_name="$(nw)_qdsm",
    )
    if bounded
        dsm = PowerModels.ref(pm, nw, :dsm)
        for (i, s) in dsm
            JuMP.set_lower_bound(qdsm[i], s["q_min"])
            JuMP.set_upper_bound(qdsm[i], s["q_max"])
        end
    end
    report && PowerModels.sol_component_value(pm, nw, :dsm, :qdsm, PowerModels.ids(pm, nw, :dsm), qdsm)
end

""
function variable_dsm_storage_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    dsme = PowerModels.var(pm, nw)[:dsme] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :dsm)], base_name="$(nw)_dsme",
        start = comp_start_value(PowerModels.ref(pm, nw, :dsm, i), "dsme_start")

    )

    if bounded
        for (i, dsm) in PowerModels.ref(pm, nw, :dsm)
            JuMP.set_lower_bound(dsme[i], dsm["e_min"])
            JuMP.set_upper_bound(dsme[i], dsm["e_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :dsm, :dsme, PowerModels.ids(pm, nw, :dsm), dsme)
end

""

"variables for modeling heat storage units, includes grid injection and internal variables"
function variable_heat_storage(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_heat_storage_power(pm; kwargs...)
    eDisGo_OPF.variable_heat_storage_energy(pm; kwargs...)
end

""
function variable_heat_storage_power(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phs = PowerModels.var(pm, nw)[:phs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heat_storage)], base_name="$(nw)_phs",
        start = comp_start_value(PowerModels.ref(pm, nw, :heat_storage, i), "phs_start")
    )

    if bounded
        for (i, hs) in PowerModels.ref(pm, nw, :heat_storage)
            JuMP.set_lower_bound(phs[i], -hs["capacity"])
            JuMP.set_upper_bound(phs[i], hs["capacity"])
        end
    end
    report && PowerModels.sol_component_value(pm, nw, :heat_storage, :phs, PowerModels.ids(pm, nw, :heat_storage), phs)
end

""
function variable_heat_storage_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    hse = PowerModels.var(pm, nw)[:hse] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heat_storage)], base_name="$(nw)_hse",
        start = comp_start_value(PowerModels.ref(pm, nw, :heat_storage, i), "hse_start")
    )

    if bounded
        for (i, hs) in PowerModels.ref(pm, nw, :heat_storage)
            JuMP.set_lower_bound(hse[i], 0)
            JuMP.set_upper_bound(hse[i], hs["capacity"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :heat_storage, :hse, PowerModels.ids(pm, nw, :heat_storage), hse)
end

""

"variables for modeling heat pumps, includes grid injection and internal variables"
function variable_heat_pump_power(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_heat_pump_power_real(pm; kwargs...)
    # eDisGo_OPF.variable_heat_pump_power_imaginary(pm; kwargs...)
end

function variable_heat_pump_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    php = PowerModels.var(pm, nw)[:php] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)], base_name="$(nw)_php",
        start = comp_start_value(PowerModels.ref(pm, nw, :heatpumps, i), "php_start")

    )

    if bounded
        for (i, hp) in PowerModels.ref(pm, nw, :heatpumps)
            JuMP.set_lower_bound(php[i], hp["p_min"])
            JuMP.set_upper_bound(php[i], hp["p_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :heatpumps, :php, PowerModels.ids(pm, nw, :heatpumps), php)
end

function variable_heat_pump_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qhp = PowerModels.var(pm, nw)[:qhp] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)], base_name="$(nw)_qhp",
    )

    if bounded
        for (i, hp) in PowerModels.ref(pm, nw, :heatpumps)
            JuMP.set_lower_bound(qhp[i], hp["q_min"])
            JuMP.set_upper_bound(qhp[i], hp["q_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :heatpumps, :qhp, PowerModels.ids(pm, nw, :heatpumps), qhp)
end

"variables for modeling charging points, includes grid injection and internal variables"
function variable_cp_power(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_cp_power_real(pm; kwargs...)
    # eDisGo_OPF.variable_cp_power_imaginary(pm; kwargs...)
    eDisGo_OPF.variable_cp_energy(pm; kwargs...)
end

""
function variable_cp_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pcp = PowerModels.var(pm, nw)[:pcp] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :electromobility)], base_name="$(nw)_pcp",
        start = comp_start_value(PowerModels.ref(pm, nw, :electromobility, i), "pcp_start")

    )

    if bounded
        for (i, cp) in PowerModels.ref(pm, nw, :electromobility)
            JuMP.set_lower_bound(pcp[i], cp["p_min"])
            JuMP.set_upper_bound(pcp[i], cp["p_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :electromobility, :pcp, PowerModels.ids(pm, nw, :electromobility), pcp)
end

""
function variable_cp_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qcp = PowerModels.var(pm, nw)[:qcp] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :electromobility)], base_name="$(nw)_qcp",
    )

    if bounded
        for (i, cp) in PowerModels.ref(pm, nw, :electromobility)
            JuMP.set_lower_bound(qcp[i], cp["q_min"])
            JuMP.set_upper_bound(qcp[i], cp["q_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :electromobility, :qcp, PowerModels.ids(pm, nw, :electromobility), qcp)
end

""
function variable_cp_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    cpe = PowerModels.var(pm, nw)[:cpe] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :electromobility)], base_name="$(nw)_cpe",
        start = comp_start_value(PowerModels.ref(pm, nw, :electromobility, i), "cpe_start")

    )

    if bounded
        for (i, cp) in PowerModels.ref(pm, nw, :electromobility)
            JuMP.set_lower_bound(cpe[i], cp["e_min"])
            JuMP.set_upper_bound(cpe[i], cp["e_max"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :electromobility, :cpe, PowerModels.ids(pm, nw, :electromobility), cpe)
end

"slack variables for grid restrictions"
function variable_slack_grid_restrictions(pm::AbstractBFModelEdisgo; kwargs...)
    eDisGo_OPF.variable_hp_slack(pm; kwargs...)
    eDisGo_OPF.variable_load_slack(pm; kwargs...)
    eDisGo_OPF.variable_gen_slack(pm; kwargs...)
    eDisGo_OPF.variable_ev_slack(pm; kwargs...)
end

function variable_slack_heat_pump_storage(pm::AbstractBFModelEdisgo; kwargs...)
    eDisGo_OPF.variable_hs_slack(pm; kwargs...)
    eDisGo_OPF.variable_hp2_slack(pm; kwargs...)
end

"heat storage slack variable"
function variable_hs_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phss = PowerModels.var(pm, nw)[:phss] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heat_storage)], base_name="$(nw)_phss",
        lower_bound = 0.0
    )

    report && PowerModels.sol_component_value(pm, nw, :heat_storage, :phss, PowerModels.ids(pm, nw, :heat_storage), phss)
end

"heat pump operation slack variable"
function variable_hp2_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phps2 = PowerModels.var(pm, nw)[:phps2] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)], base_name="$(nw)_phps2",
        lower_bound = 0.0
    )

    report && PowerModels.sol_component_value(pm, nw, :heatpumps, :phps2, PowerModels.ids(pm, nw, :heatpumps), phps2)
end

"heat pump slack variable"
function variable_hp_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phps = PowerModels.var(pm, nw)[:phps] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)], base_name="$(nw)_phps",
        lower_bound = 0.0
    )
    if bounded
        for (i, hp) in PowerModels.ref(pm, nw, :heatpumps)
            JuMP.set_upper_bound(phps[i], max(hp["pd"]/hp["cop"], 0))
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :heatpumps, :phps, PowerModels.ids(pm, nw, :heatpumps), phps)
end

"load slack variable"
function variable_load_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pds = PowerModels.var(pm, nw)[:pds] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :load)], base_name="$(nw)_pds",
        lower_bound = 0.0,
    )

    if bounded
        for (i, load) in PowerModels.ref(pm, nw, :load)
            JuMP.set_upper_bound(pds[i], load["pd"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :load, :pds, PowerModels.ids(pm, nw, :load), pds)
end

"gen slack variable"
function variable_gen_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pgens = PowerModels.var(pm, nw)[:pgens] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen)], base_name="$(nw)_pgens",
        lower_bound = 0.0,
    )

    if bounded
        for (i, gen) in PowerModels.ref(pm, nw, :gen)
            JuMP.set_upper_bound(pgens[i], gen["pg"])
        end
    end

    report && PowerModels.sol_component_value(pm, nw, :gen, :pgens, PowerModels.ids(pm, nw, :gen), pgens)
end

"EV slack variable"
function variable_ev_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pcps = PowerModels.var(pm, nw)[:pcps] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :electromobility)], base_name="$(nw)_pcps",
        lower_bound = 0.0,
    )

    report && PowerModels.sol_component_value(pm, nw, :electromobility, :pcps, PowerModels.ids(pm, nw, :electromobility), pcps)
end

"slack generator variables"
function variable_slack_gen(pm::AbstractBFModelEdisgo; kwargs...)
    eDisGo_OPF.variable_slack_gen_real(pm; kwargs...)
    eDisGo_OPF.variable_slack_gen_imaginary(pm; kwargs...)
end

function variable_slack_gen_real(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, report::Bool=true)
    pgs = PowerModels.var(pm, nw)[:pgs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen_slack)], base_name="$(nw)_pgs"
    )
    report && PowerModels.sol_component_value(pm, nw, :gen_slack, :pgs, PowerModels.ids(pm, nw, :gen_slack), pgs)
end

function variable_slack_gen_imaginary(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, report::Bool=true)
    qgs = PowerModels.var(pm, nw)[:qgs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :gen_slack)], base_name="$(nw)_qgs"
    )
    report && PowerModels.sol_component_value(pm, nw, :gen_slack, :qgs, PowerModels.ids(pm, nw, :gen_slack), qgs)
end


"slack variables for HV requirement constraints"
function variable_slack_HV_requirements(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_slack_HV_requirements_real(pm; kwargs...)
    # eDisGo_OPF.variable_slack_HV_requirements_imaginary(pm; kwargs...)
end

""
function variable_slack_HV_requirements_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phvs = PowerModels.var(pm, nw)[:phvs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :HV_requirements)], base_name="$(nw)_phvs",
        #lower_bound = -100,
        #upper_bound = 100
    )

    report && PowerModels.sol_component_value(pm, nw, :HV_requirements, :phvs, PowerModels.ids(pm, nw, :HV_requirements), phvs)

end

""
function variable_slack_HV_requirements_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qhvs = PowerModels.var(pm, nw)[:qhvs] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :HV_requirements)], base_name="$(nw)_qhvs",
    )

    report && PowerModels.sol_component_value(pm, nw, :HV_requirements, :qhvs, PowerModels.ids(pm, nw, :HV_requirements), qhvs)

end

""
