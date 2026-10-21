"""
Initial-time-step state constraint for a storage `kind` (`"storage"` battery,
`"heat_storage"`, or `"dsm"`).

Sets the stored energy after the first time step from the unit's initial state of
charge and its (dis)charging power, and pins the final state of charge to the
required end value — Eq. (3.9)/(3.10) for batteries, (3.22)/(3.23) for heat
storage and (3.32)/(3.33) for DSM in the eDisGo OPF formulation.
"""
function constraint_store_state_initial(pm::AbstractBFModelEdisgo, n::Int, i::Int, energy, charge_eff, discharge_eff, time_elapsed, kind, p_loss)
    if kind == "storage"
        ps_1 = PowerModels.var(pm, n, :ps, i)
        se = PowerModels.var(pm, n, :se, i)
        se_end = PowerModels.var(pm, length(PowerModels.nw_ids(pm)), :se, i)
        soc_initial = PowerModels.ref(pm, n, :storage)[i]["soc_initial"]
        soc_end = PowerModels.ref(pm, n, :storage)[i]["soc_end"]
        JuMP.@constraint(pm.model, se - soc_initial == - time_elapsed * ps_1)  # Eq. (3.10) i.V.m. Eq. (3.9) für t = 1 bzw. = 0
        JuMP.@constraint(pm.model, se_end == soc_end)  # Eq. (3.9) für t = tau
    elseif kind == "heat_storage"
        phs_1 = PowerModels.var(pm, n, :phs, i)
        hse = PowerModels.var(pm, n, :hse, i)
        hse_end = PowerModels.var(pm, length(PowerModels.nw_ids(pm)), :hse, i)
        soc_initial = PowerModels.ref(pm, n, :heat_storage)[i]["soc_initial"]
        soc_end = PowerModels.ref(pm, n, :heat_storage)[i]["soc_end"]
        JuMP.@constraint(pm.model, hse - soc_initial == - time_elapsed * phs_1)  # Eq. (3.23) i.V.m. Eq. (3.22) (t=1)
        JuMP.@constraint(pm.model, hse_end == soc_end)  # Eq. (3.22) für t = tau
    elseif kind == "dsm"
        dsme = PowerModels.var(pm, n, :dsme, i)
        dsme_end = PowerModels.var(pm, length(PowerModels.nw_ids(pm)), :dsme, i)
        pdsm_1 = PowerModels.var(pm, n, :pdsm, i)
        JuMP.@constraint(pm.model, dsme - energy ==  + time_elapsed * pdsm_1)  # Eq. (3.33) für t=1 (und Eq. (3.32) für t = 0, da energy=e(0) = 0)
        JuMP.@constraint(pm.model, dsme_end == 0)  # Eq. (3.32) für t = tau
    end
end


"""
Inter-time-step state-of-charge coupling for a storage `kind` between networks
`n_1` and `n_2`: the stored energy in `n_2` equals that in `n_1` plus the energy
(dis)charged over the elapsed time — Eq. (3.10), (3.23), (3.33) of the eDisGo OPF
formulation. For heat storage the previous energy is reduced by the standing-loss
factor.
"""
function constraint_store_state(pm::AbstractBFModelEdisgo, n_1::Int, n_2::Int, i::Int, charge_eff, discharge_eff, time_elapsed, kind, p_loss)
    if kind == "storage"
        ps_2 = PowerModels.var(pm, n_2, :ps, i)
        se_2 = PowerModels.var(pm, n_2, :se, i)
        se_1 = PowerModels.var(pm, n_1, :se, i)

        JuMP.@constraint(pm.model, se_2 - se_1 == - time_elapsed*ps_2)  # Eq. (3.10)
    elseif kind == "heat_storage"
        phs_2 = PowerModels.var(pm, n_2, :phs, i)
        hse_2 = PowerModels.var(pm, n_2, :hse, i)
        hse_1 = PowerModels.var(pm, n_1, :hse, i)

        JuMP.@constraint(pm.model, hse_2 - hse_1 * (1 - p_loss)^(time_elapsed/24) == - time_elapsed*phs_2)  # Eq. (3.23)
    elseif kind == "dsm"
        pdsm_2 = PowerModels.var(pm, n_2, :pdsm, i)
        dsme_2 = PowerModels.var(pm, n_2, :dsme, i)
        dsme_1 = PowerModels.var(pm, n_1, :dsme, i)

        JuMP.@constraint(pm.model, dsme_2 - dsme_1 == time_elapsed*pdsm_2)  # Eq. (3.33)
    end
end

"""
Initial-time-step state constraint for charging park `i`: sets the charged energy
after the first time step from the midpoint of the flexibility band
`(e_min+e_max)/2` and the charging power, scaled by the charging efficiency `eta`
— Eq. (3.25) of the eDisGo OPF formulation.
"""
function constraint_cp_state_initial(pm::AbstractBFModelEdisgo, n::Int, i::Int, eta)
    if haskey(PowerModels.ref(pm, n), :time_elapsed)
        time_elapsed = PowerModels.ref(pm, n, :time_elapsed)
    else
        Memento.warn(_LOGGER, "network data should specify time_elapsed, using 1.0 as a default")
        time_elapsed = 1.0
    end

    cp = PowerModels.ref(pm, n, :electromobility, i)
    cpe = PowerModels.var(pm, n, :cpe, i)
    pcp_1 = PowerModels.var(pm, n, :pcp, i)
    JuMP.@constraint(pm.model, cpe == 0.5*(cp["e_min"]+cp["e_max"]) + time_elapsed * eta * pcp_1)  # Eq. (3.25)

end

"""
Inter-time-step energy coupling for charging park `i` between networks `n_1` and
`n_2`: the charged energy in `n_2` equals that in `n_1` plus the energy charged
over the elapsed time (scaled by the efficiency `eta`) — Eq. (3.26). In the last
time step the energy is pinned back to the midpoint of the flexibility band
(Eq. (3.25)).
"""
function constraint_cp_state(pm::AbstractBFModelEdisgo, n_1::Int, n_2::Int, i::Int, eta)
    if haskey(PowerModels.ref(pm, n_1), :time_elapsed)
        time_elapsed = PowerModels.ref(pm, n_1, :time_elapsed)
    else
        Memento.warn(_LOGGER, "network data should specify time_elapsed, using 1.0 as a default")
        time_elapsed = 1.0
    end

    pcp_2 = PowerModels.var(pm, n_2, :pcp, i)
    cpe_2 = PowerModels.var(pm, n_2, :cpe, i)
    cpe_1 = PowerModels.var(pm, n_1, :cpe, i)

    JuMP.@constraint(pm.model, cpe_2 - cpe_1 == time_elapsed * eta * pcp_2)  # Eq. (3.26)

    if n_2 == length(collect(PowerModels.nw_ids(pm)))
        cp = PowerModels.ref(pm, length(collect(PowerModels.nw_ids(pm))), :electromobility, i)
        JuMP.@constraint(pm.model, cpe_2 == 0.5*(cp["e_min"]+cp["e_max"]))  # Eq. (3.25) für t=tau
    end
end

"""
Heat-pump energy balance for heat pump `i` in network `nw`: the electrical power
drawn (`php`, plus the operation slack `phps2`) times the coefficient of
performance `cop` must cover the heat demand `pd` plus the net heat-storage
(dis)charge (`phss - phs`) — Eq. (3.19) of the eDisGo OPF formulation.
"""
function constraint_hp_operation(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    hp = PowerModels.ref(pm, nw, :heatpumps, i)
    php = PowerModels.var(pm, nw, :php, i)
    phs = PowerModels.var(pm, nw, :phs, i)
    phss = PowerModels.var(pm, nw, :phss, i)
    phps2 = PowerModels.var(pm, nw, :phps2, i)


    JuMP.@constraint(pm.model, hp["cop"] * (php+phps2) == hp["pd"] + phss - phs)

end

"""
Overlying-grid requirement constraint `i` in network `nw` (`opf_version` 3 and 4):
forces the total dispatch of the addressed flexibility (DSM, curtailment, storage,
heat pumps or charging points) to meet the active power `P` requested by the higher
voltage level, up to the slack `phvs`.
"""
function constraint_HV_requirements(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    hv_req = PowerModels.ref(pm, nw, :HV_requirements, i)
    phvs = PowerModels.var(pm, nw, :phvs, i)

    if hv_req["name"] == "dsm"
        pflex = PowerModels.var(pm, nw, :pdsm)
    elseif hv_req["name"] == "curt"
        pflex = PowerModels.var(pm, nw, :pgc)
    elseif hv_req["name"] == "storage"  # ToDo: virtual branch p variable instead of ps
        # branch = PowerModels.ref(pm, nw, :branch)
        # for (i, b) in branch
        #     if b["storage"]
        #         Hier die pf Variablen (negativ!) aufsummieren
        #     end
        # end
        pflex = PowerModels.var(pm, nw, :ps)
    elseif hv_req["name"] == "hp"
        pflex = PowerModels.var(pm, nw, :php)
    elseif hv_req["name"] == "cp"
        pflex = PowerModels.var(pm, nw, :pcp)
    end
    JuMP.@constraint(pm.model, sum(pflex) + phvs == hv_req["P"])

end
