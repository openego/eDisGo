"power balance for radial branch flow model"
function constraint_power_balance_bf(pm::AbstractBFModelEdisgo, i::Int; nw::Int=nw_id_default)
    bus_arcs_to = PowerModels.ref(pm, nw, :bus_arcs_to, i)
    bus_arcs_from = PowerModels.ref(pm, nw, :bus_arcs_from, i)
    bus_lines_to = PowerModels.ref(pm, nw, :bus_lines_to, i)
    bus_gens = PowerModels.ref(pm, nw, :bus_gens, i)
    bus_gens_nd = PowerModels.ref(pm, nw, :bus_gens_nd, i)
    bus_gens_slack = PowerModels.ref(pm, nw, :bus_gens_slack, i)
    bus_loads = PowerModels.ref(pm, nw, :bus_loads, i)
    bus_storage = PowerModels.ref(pm, nw, :bus_storage, i)
    bus_dsm = PowerModels.ref(pm, nw, :bus_dsm, i)
    bus_hps = PowerModels.ref(pm, nw, :bus_hps, i)
    bus_cps = PowerModels.ref(pm, nw, :bus_cps, i)


    branch_r = Dict(k => PowerModels.ref(pm, nw, :branch, k, "br_r") for k in bus_lines_to)
    branch_x = Dict(k => PowerModels.ref(pm, nw, :branch, k, "br_x") for k in bus_lines_to)
    branch_strg_pf = Dict(k => PowerModels.ref(pm, nw, :branch, k, "storage_pf") for k in bus_lines_to)

    bus_pg = Dict(k => PowerModels.ref(pm, nw, :gen, k, "pg") for k in bus_gens)
    bus_qg = Dict(k => PowerModels.ref(pm, nw, :gen, k, "qg") for k in bus_gens)

    bus_pg_nd = Dict(k => PowerModels.ref(pm, nw, :gen_nd, k, "pg") for k in bus_gens_nd)
    bus_qg_nd = Dict(k => PowerModels.ref(pm, nw, :gen_nd, k, "qg") for k in bus_gens_nd)

    bus_pd = Dict(k => PowerModels.ref(pm, nw, :load, k, "pd") for k in bus_loads)
    bus_qd = Dict(k => PowerModels.ref(pm, nw, :load, k, "qd") for k in bus_loads)

    bus_storage_pf = Dict(k => tan(acos(PowerModels.ref(pm, nw, :storage, k, "pf")))*PowerModels.ref(pm, nw, :storage, k, "sign") for k in bus_storage)
    bus_dsm_pf = Dict(k => tan(acos(PowerModels.ref(pm, nw, :dsm, k, "pf")))*PowerModels.ref(pm, nw, :dsm, k, "sign") for k in bus_dsm)
    bus_hps_pf = Dict(k => tan(acos(PowerModels.ref(pm, nw, :heatpumps, k, "pf")))*PowerModels.ref(pm, nw, :heatpumps, k, "sign") for k in bus_hps)
    bus_cps_pf = Dict(k => tan(acos(PowerModels.ref(pm, nw, :electromobility, k, "pf")))*PowerModels.ref(pm, nw, :electromobility, k, "sign") for k in bus_cps)
    bus_gen_nd_pf = Dict(k => tan(acos(PowerModels.ref(pm, nw, :gen_nd, k, "pf")))*PowerModels.ref(pm, nw, :gen_nd, k, "sign") for k in bus_gens_nd)
    bus_gen_d_pf = Dict(k => tan(acos(PowerModels.ref(pm, nw, :gen, k, "pf")))*PowerModels.ref(pm, nw, :gen, k, "sign") for k in bus_gens)
    bus_loads_pf = Dict(k => tan(acos(PowerModels.ref(pm, nw, :load, k, "pf")))*PowerModels.ref(pm, nw, :load, k, "sign") for k in bus_loads)

    constraint_power_balance(pm, nw, i, bus_gens, bus_gens_nd, bus_gens_slack, bus_loads, bus_arcs_to, bus_arcs_from, bus_lines_to, bus_storage, bus_pg, bus_qg, bus_pg_nd, bus_qg_nd, bus_pd, bus_qd, branch_r, branch_x, bus_dsm, bus_hps, bus_cps, bus_storage_pf, bus_dsm_pf, bus_hps_pf, bus_cps_pf, bus_gen_nd_pf, bus_gen_d_pf, bus_loads_pf, branch_strg_pf)
end

""
function constraint_voltage_magnitude_difference_radial(pm::AbstractBFModelEdisgo, i::Int; nw::Int=nw_id_default)
    branch = PowerModels.ref(pm, nw, :branch, i)
    f_bus = branch["f_bus"]
    t_bus = branch["t_bus"]
    f_idx = (i, f_bus, t_bus)
    t_idx = (i, t_bus, f_bus)

    r = branch["br_r"]
    x = branch["br_x"]
    tm = branch["tap"]
    if !(branch["storage"])
        constraint_voltage_magnitude_difference(pm, nw, i, f_bus, t_bus, f_idx, t_idx, r, x, tm)
    end
end

function constraint_store_state(pm::AbstractBFModelEdisgo, i::Int; nw::Int=nw_id_default, kind::String)
    storage = PowerModels.ref(pm, nw, Symbol(kind), i)

    if kind == "dsm"
        p_loss = 0
    elseif kind in("storage", "heat_storage")
        p_loss =  storage["p_loss"]
    end

    if haskey(PowerModels.ref(pm, nw), :time_elapsed)
        time_elapsed = PowerModels.ref(pm, nw, :time_elapsed)
    else
        Memento.warn(_LOGGER, "network data should specify time_elapsed, using 1.0 as a default")
        time_elapsed = 1.0
    end

    constraint_store_state_initial(pm, nw, i, storage["energy"], storage["charge_efficiency"], storage["discharge_efficiency"], time_elapsed, kind, p_loss)
end

""
function constraint_store_state(pm::AbstractBFModelEdisgo, i::Int, nw_1::Int, nw_2::Int, kind::String)
    storage = PowerModels.ref(pm, nw_2, Symbol(kind), i)

    if kind == "dsm"
        p_loss = 0
    elseif kind in("storage", "heat_storage")
        p_loss =  storage["p_loss"]
    end

    if haskey(PowerModels.ref(pm, nw_2), :time_elapsed)
        time_elapsed = PowerModels.ref(pm, nw_2, :time_elapsed)
    else
        Memento.warn(_LOGGER, "network $(nw_2) should specify time_elapsed, using 1.0 as a default")
        time_elapsed = 1.0
    end

    if haskey(PowerModels.ref(pm, nw_1, Symbol(kind)), i)
        constraint_store_state(pm, nw_1, nw_2, i, storage["charge_efficiency"], storage["discharge_efficiency"], time_elapsed, kind, p_loss)
    else
        # if the storage device has status=0 in nw_1, then the stored energy variable will not exist. Initialize storage from data model instead.
        Memento.warn(_LOGGER, "storage component $(i) was not found in network $(nw_1) while building constraint_storage_state between networks $(nw_1) and $(nw_2). Using the energy value from the storage component in network $(nw_2) instead")
        constraint_store_state_initial(pm, nw_2, i, storage["energy"], storage["charge_efficiency"], storage["discharge_efficiency"], time_elapsed, kind, p_loss)
    end
end

function constraint_model_current(pm::AbstractPowerModel; nw::Int=nw_id_default)
    eDisGo_OPF.constraint_model_current(pm, nw)
end
