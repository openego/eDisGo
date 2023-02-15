""" Creates constraints for storage operations (battery, heat, DSM)"""

function constraint_store_state_initial(pm::AbstractBFModelEdisgo, n::Int, i::Int, energy, charge_eff, discharge_eff, time_elapsed, kind, p_loss)
    if kind == "storage"
        ps_1 = PowerModels.var(pm, n, :ps, i)
        se = PowerModels.var(pm, n, :se, i)
        se_end = PowerModels.var(pm, length(PowerModels.nw_ids(pm)), :se, i)
        JuMP.@constraint(pm.model, se - se_end == - time_elapsed * ps_1)
    elseif kind == "heat_storage"
        phs_1 = PowerModels.var(pm, n, :phs, i)
        hse = PowerModels.var(pm, n, :hse, i)
        hse_end = PowerModels.var(pm, length(PowerModels.nw_ids(pm)), :hse, i)
        JuMP.@constraint(pm.model, hse - hse_end * (1 - p_loss) == - time_elapsed * phs_1)
    elseif kind == "dsm"
        dsme = PowerModels.var(pm, n, :dsme, i)
        #dsme_end = var(pm, length(nw_ids(pm)), :dsme, i)
        pdsm_1 = PowerModels.var(pm, n, :pdsm, i)
        JuMP.@constraint(pm.model, dsme - energy ==  + time_elapsed * pdsm_1)
    end
end


function constraint_store_state(pm::AbstractBFModelEdisgo, n_1::Int, n_2::Int, i::Int, charge_eff, discharge_eff, time_elapsed, kind, p_loss)
    if kind == "storage"
        ps_2 = PowerModels.var(pm, n_2, :ps, i)
        se_2 = PowerModels.var(pm, n_2, :se, i)
        se_1 = PowerModels.var(pm, n_1, :se, i)

        JuMP.@constraint(pm.model, se_2 - se_1 == - time_elapsed*ps_2)
    elseif kind == "heat_storage"
        phs_2 = PowerModels.var(pm, n_2, :phs, i)
        hse_2 = PowerModels.var(pm, n_2, :hse, i)
        hse_1 = PowerModels.var(pm, n_1, :hse, i)

        JuMP.@constraint(pm.model, hse_2 - hse_1 * (1 - p_loss) == - time_elapsed*phs_2)
    elseif kind == "dsm"
        pdsm_2 = PowerModels.var(pm, n_2, :pdsm, i)
        dsme_2 = PowerModels.var(pm, n_2, :dsme, i)
        dsme_1 = PowerModels.var(pm, n_1, :dsme, i)

        JuMP.@constraint(pm.model, dsme_2 - dsme_1 == time_elapsed*pdsm_2)
        if n_2 == length(PowerModels.nw_ids(pm))
            JuMP.@constraint(pm.model, dsme_2 == 0)
        end
    end
end

""" Creates constraints for EV charging per charging park"""

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
    JuMP.@constraint(pm.model, cpe == 0.5*(cp["e_min"]+cp["e_max"]) + time_elapsed * eta * pcp_1)

end

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

    JuMP.@constraint(pm.model, cpe_2 - cpe_1 == time_elapsed * eta * pcp_2)

    if n_2 == length(collect(PowerModels.nw_ids(pm)))
        cp = PowerModels.ref(pm, length(collect(PowerModels.nw_ids(pm))), :electromobility, i)
        JuMP.@constraint(pm.model, cpe_2 == 0.5*(cp["e_min"]+cp["e_max"]))
    end
end

""" Creates constraints for heat pump operation"""

function constraint_hp_operation(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    hp = PowerModels.ref(pm, nw, :heatpumps, i)
    php = PowerModels.var(pm, nw, :php, i)
    phs = PowerModels.var(pm, nw, :phs, i)

    if PowerModels.ref(pm, 1, :opf_version) in(5)#in(2,4)
        phps = PowerModels.var(pm, nw, :phps, i)
        JuMP.@constraint(pm.model, hp["cop"] * (php+phps) == hp["pd"] - phs)
    else
        JuMP.@constraint(pm.model, hp["cop"] * php == hp["pd"] - phs)
    end
end

""" Creates constraints for high voltage grid requirements"""

function constraint_HV_requirements(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    if PowerModels.ref(pm, 1, :opf_version) in(1,2)
        hv_req = PowerModels.ref(pm, nw, :HV_requirements, i)
        phvs = PowerModels.var(pm, nw, :phvs, i)

        if hv_req["name"] == "dsm"
            pflex = PowerModels.var(pm, nw, :pdsm)
        elseif hv_req["name"] == "curt"
            pflex = PowerModels.var(pm, nw, :pgc)
        elseif hv_req["name"] == "storage"
            pflex = PowerModels.var(pm, nw, :ps)
        elseif hv_req["name"] == "hp"
            pflex = PowerModels.var(pm, nw, :php)
        elseif hv_req["name"] == "cp"
            pflex = PowerModels.var(pm, nw, :pcp)
        end
        JuMP.@constraint(pm.model, sum(pflex) + phvs == hv_req["P"])
    end
end
