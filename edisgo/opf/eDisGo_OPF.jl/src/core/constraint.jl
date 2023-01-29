""" Creates constraints for storage operations (battery, heat, DSM)"""

function constraint_store_state_initial(pm::AbstractBFModelEdisgo, n::Int, i::Int, energy, charge_eff, discharge_eff, time_elapsed, kind, p_loss)
    if kind == "storage"
        ps_1 = var(pm, n, :ps, i)
        se = var(pm, n, :se, i)
        se_end = var(pm, length(nw_ids(pm)), :se, i)
        JuMP.@constraint(pm.model, se - se_end == - time_elapsed * ps_1)
    elseif kind == "heat_storage"
        phs_1 = var(pm, n, :phs, i)
        hse = var(pm, n, :hse, i)
        hse_end = var(pm, length(nw_ids(pm)), :hse, i)
        JuMP.@constraint(pm.model, hse - hse_end * (1 - p_loss) == - time_elapsed * phs_1)
    elseif kind == "dsm"
        dsme = var(pm, n, :dsme, i)
        #dsme_end = var(pm, length(nw_ids(pm)), :dsme, i)
        pdsm_1 = var(pm, n, :pdsm, i)
        JuMP.@constraint(pm.model, dsme - energy ==  + time_elapsed * pdsm_1)
    end
end


function constraint_store_state(pm::AbstractBFModelEdisgo, n_1::Int, n_2::Int, i::Int, charge_eff, discharge_eff, time_elapsed, kind, p_loss)
    if kind == "storage"
        ps_2 = var(pm, n_2, :ps, i)
        se_2 = var(pm, n_2, :se, i)
        se_1 = var(pm, n_1, :se, i)

        JuMP.@constraint(pm.model, se_2 - se_1 == - time_elapsed*ps_2)
    elseif kind == "heat_storage"
        phs_2 = var(pm, n_2, :phs, i)
        hse_2 = var(pm, n_2, :hse, i)
        hse_1 = var(pm, n_1, :hse, i)

        JuMP.@constraint(pm.model, hse_2 - hse_1 * (1 - p_loss) == - time_elapsed*phs_2)
    elseif kind == "dsm"
        pdsm_2 = var(pm, n_2, :pdsm, i)
        dsme_2 = var(pm, n_2, :dsme, i)
        dsme_1 = var(pm, n_1, :dsme, i)

        JuMP.@constraint(pm.model, dsme_2 - dsme_1 == time_elapsed*pdsm_2)
        if n_2 == length(nw_ids(pm))
            JuMP.@constraint(pm.model, dsme_2 == 0)
        end
    end
end

""" Creates constraints for EV charging per charging park"""

function constraint_cp_state_initial(pm::AbstractBFModelEdisgo, n::Int, i::Int)
    if haskey(ref(pm, n), :time_elapsed)
        time_elapsed = ref(pm, n, :time_elapsed)
    else
        Memento.warn(_LOGGER, "network data should specify time_elapsed, using 1.0 as a default")
        time_elapsed = 1.0
    end

    cp = ref(pm, n, :electromobility, i)
    cpe = var(pm, n, :cpe, i)
    pcp_1 = var(pm, n, :pcp, i)
    JuMP.@constraint(pm.model, cpe == 0.5*(cp["e_min"]+cp["e_max"]) + time_elapsed * pcp_1)

end

function constraint_cp_state(pm::AbstractBFModelEdisgo, n_1::Int, n_2::Int, i::Int)
    if haskey(ref(pm, n_1), :time_elapsed)
        time_elapsed = ref(pm, n_1, :time_elapsed)
    else
        Memento.warn(_LOGGER, "network data should specify time_elapsed, using 1.0 as a default")
        time_elapsed = 1.0
    end

    pcp_2 = var(pm, n_2, :pcp, i)
    cpe_2 = var(pm, n_2, :cpe, i)
    cpe_1 = var(pm, n_1, :cpe, i)

    JuMP.@constraint(pm.model, cpe_2 - cpe_1 == time_elapsed*pcp_2)

    if n_2 == length(collect(nw_ids(pm)))
        cp = ref(pm, length(collect(nw_ids(pm))), :electromobility, i)
        JuMP.@constraint(pm.model, cpe_2 == 0.5*(cp["e_min"]+cp["e_max"]))
    end
end

""" Creates constraints for heat pump operation"""

function constraint_hp_operation(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    hp = ref(pm, nw, :heatpumps, i)
    php = var(pm, nw, :php, i)
    phs = var(pm, nw, :phs, i)

    if ref(pm, 1, :opf_version) in(5)#in(2,4)
        phps = var(pm, nw, :phps, i)
        JuMP.@constraint(pm.model, hp["cop"] * (php+phps) == hp["pd"] - phs)
    else
        JuMP.@constraint(pm.model, hp["cop"] * php == hp["pd"] - phs)
    end
end

""" Creates constraints for high voltage grid requirements"""

function constraint_HV_requirements(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    if ref(pm, 1, :opf_version) in(1,2)
        hv_req = ref(pm, nw, :HV_requirements, i)
        phvs = var(pm, nw, :phvs, i)

        if hv_req["name"] == "dsm"
            pflex = var(pm, nw, :pdsm)
        elseif hv_req["name"] == "curt"
            pflex = var(pm, nw, :pgc)
        elseif hv_req["name"] == "storage"
            pflex = var(pm, nw, :ps)
        elseif hv_req["name"] == "hp"
            pflex = var(pm, nw, :php)
        elseif hv_req["name"] == "cp"
            pflex = var(pm, nw, :pcp)
        end
        JuMP.@constraint(pm.model, sum(pflex) + phvs == hv_req["P"])
    end
end
