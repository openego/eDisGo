"generates variables for both `active` and `reactive` non-dispatchable power generation curtailment"
function variable_gen_power_curt(pm::AbstractPowerModel; kwargs...)
    variable_gen_power_curt_real(pm; kwargs...) # Eq. (3.30) für non-dispatchable Generators
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
function variable_battery_storage_power(pm::AbstractPowerModel; kwargs...)
    eDisGo_OPF.variable_battery_storage_power_real(pm; kwargs...)  # Eq. (3.13)
    PowerModels.variable_storage_energy(pm; kwargs...)  # Eq. (3.12)
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
    eDisGo_OPF.variable_dsm_storage_power_real(pm; kwargs...)  # Eq. (34)
    # eDisGo_OPF.variable_dsm_storage_power_imaginary(pm; kwargs...)  # TODO: to add
    eDisGo_OPF.variable_dsm_storage_energy(pm; kwargs...)  # Eq. (35)
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
    eDisGo_OPF.variable_heat_storage_power(pm; kwargs...)  # wird hier durch Kapazität des Speichers beschränkt (kein Schranke kann die
    # Lösungsgeschwindigkeit verringern), indirekte Beschränkung durch min/max Speicherfüllstand ist jedoch restriktiver
    # -> Bound wird nicht in MA aufgenommen
    eDisGo_OPF.variable_heat_storage_energy(pm; kwargs...)  # Eq. (3.19)
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
    eDisGo_OPF.variable_heat_pump_power_real(pm; kwargs...)  # Eq. (3.16)
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
    eDisGo_OPF.variable_cp_power_real(pm; kwargs...)  # Eq. (3.23)
    # eDisGo_OPF.variable_cp_power_imaginary(pm; kwargs...)
    eDisGo_OPF.variable_cp_energy(pm; kwargs...)  # Eq. (3.22)
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
    #variable_hp_slack(pm; kwargs...)
    eDisGo_OPF.variable_load_slack(pm; kwargs...) # Eq. (3.31)
    eDisGo_OPF.variable_gen_slack(pm; kwargs...) # Eq. (3.30) für dispatchable Generators
    eDisGo_OPF.variable_ev_slack(pm; kwargs...) # Eq. (3.32)
end

"heat pump slack variable"
function variable_hp_slack(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    phps = PowerModels.var(pm, nw)[:phps] = JuMP.@variable(pm.model,
        [i in PowerModels.ids(pm, nw, :heatpumps)], base_name="$(nw)_phps",
        lower_bound = 0.0
    )
    if bounded
        for (i, hp) in PowerModels.ref(pm, nw, :heatpumps)
            JuMP.set_upper_bound(phps[i], hp["pd"]/hp["cop"])
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
function eDisGo_OPF.variable_slack_gen(pm::AbstractBFModelEdisgo; kwargs...)
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
        lower_bound = -1e5,
        upper_bound = 1e5
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
