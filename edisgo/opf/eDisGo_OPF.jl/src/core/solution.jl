function sol_component_value_radial(aim::AbstractPowerModel, n::Int, comp_name::Symbol, field_name_to::Symbol, comp_ids_to, variables)
    for (l, i, j) in comp_ids_to
        @assert !haskey(InfrastructureModels.sol(aim, pm_it_sym, n, comp_name, l), field_name_to)
        _IM.sol(aim, pm_it_sym, n, comp_name, l)[field_name_to] = variables[(l, i, j)]
    end
end


function check_SOC_equality(result, data_edisgo)
    timesteps = keys(result["solution"]["nw"])
    branches = keys(data_edisgo["branch"])
    branch_f_bus = Dict(k => string(data_edisgo["branch"][k]["f_bus"]) for k in branches)
    soc_eq_dict = Dict()
    soc_tight = true
    for t in timesteps
        eq_res = Dict(b => (result["solution"]["nw"][t]["branch"][b]["pf"]^2
        + result["solution"]["nw"][t]["branch"][b]["qf"]^2
        -result["solution"]["nw"][t]["branch"][b]["ccm"]*result["solution"]["nw"][t]["bus"][branch_f_bus[b]]["w"]) for b in branches)
        soc_eq_dict[t]= filter(((k,v),) ->  v <-1e-2, eq_res) # Bobo Second-order cone relaxations of the optimal power flow for active distribution grids: Comparison of methods
        if length(keys(soc_eq_dict[t])) > 0
            soc_tight = false
        end
    end
    return soc_tight, soc_eq_dict
end
