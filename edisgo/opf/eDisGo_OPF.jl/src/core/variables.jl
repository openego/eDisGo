"generates variables for both `active` and `reactive` non-dispatchable power generation curtailment"
function variable_gen_power_curt(pm::AbstractPowerModel; kwargs...)
    variable_gen_power_curt_real(pm; kwargs...)
    #variable_gen_power_curt_imaginary(pm; kwargs...)
end


"variable: `pgc[j]` for `j` in `gen_nd`"
function variable_gen_power_curt_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pgc = var(pm, nw)[:pgc] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :gen_nd)], base_name="$(nw)_pgc",
        start = comp_start_value(ref(pm, nw, :gen_nd, i), "pgc_start")
    )

    if bounded
        for (i, gen) in ref(pm, nw, :gen_nd)
            JuMP.set_lower_bound(pgc[i], 0)
            JuMP.set_upper_bound(pgc[i], gen["pg"])
        end
    end

    report && sol_component_value(pm, nw, :gen_nd, :pgc, ids(pm, nw, :gen_nd), pgc)
end

"variable: `qgc[j]` for `j` in `gen_nd`"
function variable_gen_power_curt_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qgc = var(pm, nw)[:qgc] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :gen_nd)], base_name="$(nw)_qgc"
    )

    if bounded
        for (i, gen) in ref(pm, nw, :gen_nd)
            JuMP.set_lower_bound(qgc[i], gen["qg"])
            JuMP.set_upper_bound(qgc[i], 0)
        end
    end

    report && sol_component_value(pm, nw, :gen_nd, :qgc, ids(pm, nw, :gen_nd), qgc)
end


"variables for modeling storage units, includes grid injection and internal variables"
function variable_battery_storage_power(pm::AbstractPowerModel; kwargs...)
    variable_battery_storage_power_real(pm; kwargs...)  # Eq. (21)
    #variable_battery_storage_power_imaginary(pm; kwargs...) # Eq. (21)
    variable_storage_energy(pm; kwargs...)  # Eq. (22)
end

""
function variable_battery_storage_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    ps = var(pm, nw)[:ps] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :storage)], base_name="$(nw)_ps",
        start = comp_start_value(ref(pm, nw, :storage, i), "ps_start")
    )

    if bounded
        for (i, storage) in ref(pm, nw, :storage)
            JuMP.set_lower_bound(ps[i], storage["pmin"])
            JuMP.set_upper_bound(ps[i], storage["pmax"])
        end
    end

    report && sol_component_value(pm, nw, :storage, :ps, ids(pm, nw, :storage), ps)
end

""
function variable_battery_storage_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qs = var(pm, nw)[:qs] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :storage)], base_name="$(nw)_qs",
        start = comp_start_value(ref(pm, nw, :storage, i), "qs_start")
    )

    if bounded
        for (i, storage) in ref(pm, nw, :storage)
            JuMP.set_lower_bound(qs[i], storage["qmin"])
            JuMP.set_upper_bound(qs[i], storage["qmax"])
        end
    end

    report && sol_component_value(pm, nw, :storage, :qs, ids(pm, nw, :storage), qs)
end


### Additional Flexibility Variables (DSM, HS, HP, CP)

"variables for modeling dsm storage units, includes grid injection and internal variables"
function variable_dsm_storage_power(pm::AbstractPowerModel; kwargs...)
    variable_dsm_storage_power_real(pm; kwargs...)  # Eq. (34)
    #variable_dsm_storage_power_imaginary(pm; kwargs...)  # TODO: to add
    variable_dsm_storage_energy(pm; kwargs...)  # Eq. (35)
end

""
function variable_dsm_storage_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pdsm = var(pm, nw)[:pdsm] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :dsm)], base_name="$(nw)_pdsm",
        start = comp_start_value(ref(pm, nw, :dsm, i), "pdsm_start")
    )

    if bounded
        dsm = ref(pm, nw, :dsm)
        for (i, s) in dsm
            JuMP.set_lower_bound(pdsm[i], s["p_min"])
            JuMP.set_upper_bound(pdsm[i], s["p_max"])
        end
    end

    report && sol_component_value(pm, nw, :dsm, :pdsm, ids(pm, nw, :dsm), pdsm)
end

""
function variable_dsm_storage_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qdsm = var(pm, nw)[:qdsm] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :dsm)], base_name="$(nw)_qdsm",
    )
    if bounded
        dsm = ref(pm, nw, :dsm)
        for (i, s) in dsm
            JuMP.set_lower_bound(qdsm[i], s["q_min"])
            JuMP.set_upper_bound(qdsm[i], s["q_max"])
        end
    end
    report && sol_component_value(pm, nw, :dsm, :qdsm, ids(pm, nw, :dsm), qdsm)
end

""
function variable_dsm_storage_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    dsme = var(pm, nw)[:dsme] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :dsm)], base_name="$(nw)_dsme",
        start = comp_start_value(ref(pm, nw, :dsm, i), "dsme_start")

    )

    if bounded
        for (i, dsm) in ref(pm, nw, :dsm)
            JuMP.set_lower_bound(dsme[i], dsm["e_min"])
            JuMP.set_upper_bound(dsme[i], dsm["e_max"])
        end
    end

    report && sol_component_value(pm, nw, :dsm, :dsme, ids(pm, nw, :dsm), dsme)
end

""

"variables for modeling heat storage units, includes grid injection and internal variables"
function variable_heat_storage(pm::AbstractPowerModel; kwargs...)
    variable_heat_storage_power(pm; kwargs...)  # Eq. (34)
    variable_heat_storage_energy(pm; kwargs...)  # Eq. (35)
end

""
function variable_heat_storage_power(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phs = var(pm, nw)[:phs] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :heat_storage)], base_name="$(nw)_phs",
        start = comp_start_value(ref(pm, nw, :heat_storage, i), "phs_start")

    )

    if bounded
        for (i, hs) in ref(pm, nw, :heat_storage)
            JuMP.set_lower_bound(phs[i], -hs["capacity"])
            JuMP.set_upper_bound(phs[i], hs["capacity"])
        end
    end
    report && sol_component_value(pm, nw, :heat_storage, :phs, ids(pm, nw, :heat_storage), phs)
end

""
function variable_heat_storage_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    hse = var(pm, nw)[:hse] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :heat_storage)], base_name="$(nw)_hse",
        start = comp_start_value(ref(pm, nw, :heat_storage, i), "hse_start")
    )

    if bounded
        for (i, hs) in ref(pm, nw, :heat_storage)
            JuMP.set_lower_bound(hse[i], 0)
            JuMP.set_upper_bound(hse[i], hs["capacity"])
        end
    end

    report && sol_component_value(pm, nw, :heat_storage, :hse, ids(pm, nw, :heat_storage), hse)
end

""

"variables for modeling heat pumps, includes grid injection and internal variables"
function variable_heat_pump_power(pm::AbstractPowerModel; kwargs...)
    variable_heat_pump_power_real(pm; kwargs...)  # Eq. (34)
    #variable_heat_pump_power_imaginary(pm; kwargs...)  # TODO: to add
end

function variable_heat_pump_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    php = var(pm, nw)[:php] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :heatpumps)], base_name="$(nw)_php",
        start = comp_start_value(ref(pm, nw, :heatpumps, i), "php_start")

    )

    if bounded
        for (i, hp) in ref(pm, nw, :heatpumps)
            JuMP.set_lower_bound(php[i], hp["p_min"])
            JuMP.set_upper_bound(php[i], hp["p_max"])
        end
    end

    report && sol_component_value(pm, nw, :heatpumps, :php, ids(pm, nw, :heatpumps), php)
end

function variable_heat_pump_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qhp = var(pm, nw)[:qhp] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :heatpumps)], base_name="$(nw)_qhp",
    )

    if bounded
        for (i, hp) in ref(pm, nw, :heatpumps)
            JuMP.set_lower_bound(qhp[i], hp["q_min"])
            JuMP.set_upper_bound(qhp[i], hp["q_max"])
        end
    end

    report && sol_component_value(pm, nw, :heatpumps, :qhp, ids(pm, nw, :heatpumps), qhp)
end

"variables for modeling charging points, includes grid injection and internal variables"
function variable_cp_power(pm::AbstractPowerModel; kwargs...)
    variable_cp_power_real(pm; kwargs...)  # Eq. (34)
    #variable_cp_power_imaginary(pm; kwargs...)
    variable_cp_energy(pm; kwargs...)  # Eq. (35)
end

""
function variable_cp_power_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pcp = var(pm, nw)[:pcp] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :electromobility)], base_name="$(nw)_pcp",
        start = comp_start_value(ref(pm, nw, :electromobility, i), "pcp_start")

    )

    if bounded
        for (i, cp) in ref(pm, nw, :electromobility)
            JuMP.set_lower_bound(pcp[i], cp["p_min"])
            JuMP.set_upper_bound(pcp[i], cp["p_max"])
        end
    end

    report && sol_component_value(pm, nw, :electromobility, :pcp, ids(pm, nw, :electromobility), pcp)
end

""
function variable_cp_power_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qcp = var(pm, nw)[:qcp] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :electromobility)], base_name="$(nw)_qcp",
    )

    if bounded
        for (i, cp) in ref(pm, nw, :electromobility)
            JuMP.set_lower_bound(qcp[i], cp["q_min"])
            JuMP.set_upper_bound(qcp[i], cp["q_max"])
        end
    end

    report && sol_component_value(pm, nw, :electromobility, :qcp, ids(pm, nw, :electromobility), qcp)
end

""
function variable_cp_energy(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    cpe = var(pm, nw)[:cpe] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :electromobility)], base_name="$(nw)_cpe",
        start = comp_start_value(ref(pm, nw, :electromobility, i), "cpe_start")

    )

    if bounded
        for (i, cp) in ref(pm, nw, :electromobility)
            JuMP.set_lower_bound(cpe[i], cp["e_min"])
            JuMP.set_upper_bound(cpe[i], cp["e_max"])
        end
    end

    report && sol_component_value(pm, nw, :electromobility, :cpe, ids(pm, nw, :electromobility), cpe)
end

"slack variables for grid restrictions"
function variable_slack_grid_restrictions(pm::AbstractBFModelEdisgo; kwargs...)
    if ref(pm, 1, :opf_version) in(2,4)
        #variable_hp_slack(pm; kwargs...)
        variable_load_slack(pm; kwargs...)
        variable_gen_slack(pm; kwargs...)
        variable_ev_slack(pm; kwargs...)
    end
end

"heat pump slack variable"
function variable_hp_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phps = var(pm, nw)[:phps] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :heatpumps)], base_name="$(nw)_phps",
        lower_bound = 0.0
    )
    if bounded
        for (i, hp) in ref(pm, nw, :heatpumps)
            JuMP.set_upper_bound(phps[i], hp["pd"]/hp["cop"])
        end
    end

    report && sol_component_value(pm, nw, :heatpumps, :phps, ids(pm, nw, :heatpumps), phps)
end

"load slack variable"
function variable_load_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pds = var(pm, nw)[:pds] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :load)], base_name="$(nw)_pds",
        lower_bound = 0.0,
    )

    if bounded
        for (i, load) in ref(pm, nw, :load)
            JuMP.set_upper_bound(pds[i], load["pd"])
        end
    end

    report && sol_component_value(pm, nw, :load, :pds, ids(pm, nw, :load), pds)
end

"gen slack variable"
function variable_gen_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pgens = var(pm, nw)[:pgens] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :gen)], base_name="$(nw)_pgens",
        lower_bound = 0.0,
    )

    if bounded
        for (i, gen) in ref(pm, nw, :gen)
            JuMP.set_upper_bound(pgens[i], gen["pg"])
        end
    end

    report && sol_component_value(pm, nw, :gen, :pgens, ids(pm, nw, :gen), pgens)
end

"EV slack variable"
function variable_ev_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    pcps = var(pm, nw)[:pcps] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :electromobility)], base_name="$(nw)_pcps",
        lower_bound = 0.0,
    )

    report && sol_component_value(pm, nw, :electromobility, :pcps, ids(pm, nw, :electromobility), pcps)
end

"slack generator variables"
function variable_slack_gen(pm::AbstractBFModelEdisgo; kwargs...)
    variable_slack_gen_real(pm; kwargs...)
    variable_slack_gen_imaginary(pm; kwargs...)
end

function variable_slack_gen_real(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, report::Bool=true)
    pgs = var(pm, nw)[:pgs] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :gen_slack)], base_name="$(nw)_pgs"
    )
    report && sol_component_value(pm, nw, :gen_slack, :pgs, ids(pm, nw, :gen_slack), pgs)
end

function variable_slack_gen_imaginary(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, report::Bool=true)
    qgs = var(pm, nw)[:qgs] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :gen_slack)], base_name="$(nw)_qgs"
    )
    report && sol_component_value(pm, nw, :gen_slack, :qgs, ids(pm, nw, :gen_slack), qgs)
end


"slack variables for HV requirement constraints"
function variable_slack_HV_requirements(pm::AbstractPowerModel; kwargs...)
    if ref(pm, 1, :opf_version) in(1,2)
        variable_slack_HV_requirements_real(pm; kwargs...)
        #variable_slack_HV_requirements_imaginary(pm; kwargs...)
    end
end

""
function variable_slack_HV_requirements_real(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phvs = var(pm, nw)[:phvs] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :HV_requirements)], base_name="$(nw)_phvs",
        lower_bound = -1e5,
        upper_bound = 1e5
    )

    report && sol_component_value(pm, nw, :HV_requirements, :phvs, ids(pm, nw, :HV_requirements), phvs)

end

""
function variable_slack_HV_requirements_imaginary(pm::AbstractPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    qhvs = var(pm, nw)[:qhvs] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :HV_requirements)], base_name="$(nw)_qhvs",
    )

    report && sol_component_value(pm, nw, :HV_requirements, :qhvs, ids(pm, nw, :HV_requirements), qhvs)

end

""
