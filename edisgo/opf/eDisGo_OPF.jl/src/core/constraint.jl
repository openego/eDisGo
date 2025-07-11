""" Creates constraints for storage operations (battery, heat, DSM)"""
const C_RATE         = 0.5      # max. Lade‑/Entlade‑C‑Rate  (0.5 C → 2 h Vollzyklus)
const HV_SLACK_COST  = 1e4      # €/kW Strafkosten bei Nichterfüllung der HV‑Vorgabe

function constraint_store_state_initial(pm::AbstractBFModelEdisgo, n::Int, i::Int, energy, charge_eff, discharge_eff, time_elapsed, kind, p_loss)
    if kind == "storage"
        ps_1 = PowerModels.var(pm, n, :ps, i)
        se_1 = PowerModels.var(pm, n, :se, i)
        se_T = PowerModels.var(pm, length(PowerModels.nw_ids(pm)), :se, i)
        e_cap = PowerModels.var(pm, n, :e_cap, i)

        # C‑Rate‑Begrenzung für den ersten Zeitschritt
        @constraint(pm.model,  ps_1 <=  C_RATE * e_cap)
        @constraint(pm.model, -ps_1 <=  C_RATE * e_cap)

        # Start-SoC = 50 % der gekauften Kapazität
        @constraint(pm.model, se_1 == 0.5 * e_cap)

        # End-SoC = Start-SoC (zyklische Bedingung)
        @constraint(pm.model, se_T == se_1)

        # SOC darf nie die variable Kapazität überschreiten
        @constraint(pm.model, se_1 <= e_cap)
        @constraint(pm.model, se_1 >= 0.0)

        # ENTFERNEN Sie die redundante Bilanz-Constraint hier
        # Die zeitliche Bilanz wird in constraint_store_state gehandhabt
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

function _get_charging_point_lines(pm::AbstractBFModelEdisgo, nw::Int)
    line_indices = Int[]
    branches = PowerModels.ref(pm, nw, :branch)
    for (i, branch) in branches
        if startswith(branch["name"], "Line_bus_charging_point_")
            push!(line_indices, i)
        end
    end
    return line_indices
end


function constraint_charging_point_line_flow(pm::AbstractBFModelEdisgo; nw::Int=nw_id_default)
    line_indices = _get_charging_point_lines(pm, nw)
    max_power_kw = 11000 # Maximum allowed power flow in kW

    for i in line_indices
        pf = PowerModels.var(pm, nw, :pf, i)  # Active power flow variable
        JuMP.@constraint(pm.model, pf <= max_power_kw)   # Upper bound
        JuMP.@constraint(pm.model, pf >= -max_power_kw)  # Lower bound
    end
end


function constraint_store_state(pm::AbstractBFModelEdisgo, n_1::Int, n_2::Int, i::Int, charge_eff, discharge_eff, time_elapsed, kind, p_loss)
    if kind == "storage"
        ps_2 = PowerModels.var(pm, n_2, :ps, i)
        se_2 = PowerModels.var(pm, n_2, :se, i)
        se_1 = PowerModels.var(pm, n_1, :se, i)
        e_cap = PowerModels.var(pm, n_2, :e_cap, i)  # Variable Kapazität

        # Energiebilanz zwischen Zeitschritten
        @constraint(pm.model, se_2 - se_1 == -time_elapsed * ps_2)

        # WICHTIG: SOC darf variable Kapazität nie überschreiten
        @constraint(pm.model, se_2 <= e_cap)
        @constraint(pm.model, se_2 >= 0.0)
    elseif kind == "heat_storage"
        phs_2 = PowerModels.var(pm, n_2, :phs, i)
        hse_2 = PowerModels.var(pm, n_2, :hse, i)
        hse_1 = PowerModels.var(pm, n_1, :hse, i)

        JuMP.@constraint(pm.model, hse_2 - hse_1 * (1 - p_loss)^(1/24) == - time_elapsed*phs_2)  # Eq. (3.23)
    elseif kind == "dsm"
        pdsm_2 = PowerModels.var(pm, n_2, :pdsm, i)
        dsme_2 = PowerModels.var(pm, n_2, :dsme, i)
        dsme_1 = PowerModels.var(pm, n_1, :dsme, i)

        JuMP.@constraint(pm.model, dsme_2 - dsme_1 == time_elapsed*pdsm_2)  # Eq. (3.33)
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
    JuMP.@constraint(pm.model, cpe == 0.5*(cp["e_min"]+cp["e_max"]) + time_elapsed * eta * pcp_1)  # Eq. (3.25)

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

    JuMP.@constraint(pm.model, cpe_2 - cpe_1 == time_elapsed * eta * pcp_2)  # Eq. (3.26)

    if n_2 == length(collect(PowerModels.nw_ids(pm)))
        cp = PowerModels.ref(pm, length(collect(PowerModels.nw_ids(pm))), :electromobility, i)
        JuMP.@constraint(pm.model, cpe_2 == 0.5*(cp["e_min"]+cp["e_max"]))  # Eq. (3.25) für t=tau
    end
end

""" Creates constraints for heat pump operation"""

function constraint_hp_operation(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
    hp = PowerModels.ref(pm, nw, :heatpumps, i)
    php = PowerModels.var(pm, nw, :php, i)
    phs = PowerModels.var(pm, nw, :phs, i)
    phss = PowerModels.var(pm, nw, :phss, i)
    phps2 = PowerModels.var(pm, nw, :phps2, i)


    JuMP.@constraint(pm.model, hp["cop"] * (php+phps2) == hp["pd"] + phss - phs)

end

""" Creates constraints for high voltage grid requirements"""

# function constraint_HV_requirements(pm::AbstractBFModelEdisgo, i::Int, nw::Int=nw_id_default)
#     hv_req = PowerModels.ref(pm, nw, :HV_requirements, i)
#     phvs = PowerModels.var(pm, nw, :phvs, i)

#     if hv_req["name"] == "dsm"
#         pflex = PowerModels.var(pm, nw, :pdsm)
#     elseif hv_req["name"] == "curt"
#         pflex = PowerModels.var(pm, nw, :pgc)
#     # elseif hv_req["name"] == "storage"  # ToDo: virtual branch p variable instead of ps
#     #     # branch = PowerModels.ref(pm, nw, :branch)
#     #     # for (i, b) in branch
#     #     #     if b["storage"]
#     #     #         Hier die pf Variablen (negativ!) aufsummieren
#     #     #     end
#     #     # end
#     #     pflex = PowerModels.var(pm, nw, :ps)
#     elseif hv_req["name"] == "hp"
#         pflex = PowerModels.var(pm, nw, :php)
#     elseif hv_req["name"] == "cp"
#         pflex = PowerModels.var(pm, nw, :pcp)
#     end

#     # weiche Gleichung mit beidseitigem Slack
#     @variable(pm.model, hv_slack_pos[i] >= 0)   # i = Index des Anforderungsknotens
#     @variable(pm.model, hv_slack_neg[i] >= 0)
#     @constraint(pm.model, sum(pflex) + phvs + hv_slack_pos[i] - hv_slack_neg[i] == hv_req["P"])

# end
