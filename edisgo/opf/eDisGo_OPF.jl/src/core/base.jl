""
function solve_model(data::Dict{String,<:Any}, model_type::Type, optimizer, build_method;
        ref_extensions=[], solution_processors=[], relax_integrality=false,
        multinetwork=false, multiconductor=false, kwargs...)

    start_time = time()
    pm = eDisGo_OPF.instantiate_model(data, model_type, build_method; ref_extensions=ref_extensions, kwargs...)
    #print(pm.model)
    Memento.debug(PowerModels._LOGGER, "pm model build time: $(time() - start_time)")
    start_time = time()
    result = optimize_model!(pm, relax_integrality=relax_integrality, optimizer=optimizer, solution_processors=solution_processors)
    Memento.debug(PowerModels._LOGGER, "pm model solve and solution time: $(time() - start_time)")
    return result, pm
end

function instantiate_model(data::Dict{String,<:Any}, model_type::Type, build_method; kwargs...)
    return InfrastructureModels.instantiate_model(data, model_type, build_method, eDisGo_OPF.ref_add_core!, _pm_global_keys, pm_it_sym; kwargs...)
end

"""
Returns a dict that stores commonly used pre-computed data from of the data dictionary,
primarily for converting data-types, filtering out deactivated components, and storing
system-wide values that need to be computed globally.

Some of the common keys include:

* `:off_angmin` and `:off_angmax` (see `calc_theta_delta_bounds(data)`),
* `:bus` -- the set `{(i, bus) in ref[:bus] : bus["bus_type"] != 4}`,
* `:gen` -- the set `{(i, gen) in ref[:gen] : gen["gen_status"] == 1 && gen["gen_bus"] in keys(ref[:bus])}`,
* `:branch` -- the set of branches that are active in the network (based on the component status values),
* `:arcs_from` -- the set `[(i,b["f_bus"],b["t_bus"]) for (i,b) in ref[:branch]]`,
* `:arcs_to` -- the set `[(i,b["t_bus"],b["f_bus"]) for (i,b) in ref[:branch]]`,
* `:arcs` -- the set of arcs from both `arcs_from` and `arcs_to`,
* `:bus_arcs` -- the mapping `Dict(i => [(l,i,j) for (l,i,j) in ref[:arcs]])`,
* `:bus_arcs_to` -- the mapping `Dict(j => [(l,i,j) for (l,i,j) in ref[:arcs_from]])`,
* `:bus_arcs_from` -- the mapping `Dict(i => [(l,i,j) for (l,i,j) in ref[:arcs_from]])`,
* `:bus_lines_to` -- the mapping `Dict(j => [l for (l,i,j) in ref[:arcs_from]])`,
* `:buspairs` -- (see `buspair_parameters(ref[:arcs_from], ref[:branch], ref[:bus])`),
* `:bus_gens` -- the mapping `Dict(i => [gen["gen_bus"] for (i,gen) in ref[:gen]])`.
* `:bus_gens_nd` -- the mapping `Dict(i => [gen_nd["gen_bus"] for (i,gen) in ref[:gen_nd]])`.
* `:bus_gens_slack` -- the mapping `Dict(i => [gen_slack["gen_bus"] for (i,gen) in ref[:gen_slack]])`.
* `:bus_loads` -- the mapping `Dict(i => [load["load_bus"] for (i,load) in ref[:load]])`.
* `:bus_shunts` -- the mapping `Dict(i => [shunt["shunt_bus"] for (i,shunt) in ref[:shunt]])`.
* `:bus_dsm` -- the mapping `Dict(i => [dsm["dsm_bus"] for (i,dsm) in ref[:dsm]])`.
* `:bus_cps` -- the mapping `Dict(i => [cp["cp_bus"] for (i,cp) in ref[:electromobility]])`.
* `:bus_hps` -- the mapping `Dict(i => [hp["hp_bus"] for (i,hp) in ref[:heatpumps]])`.
* `:arcs_from_dc` -- the set `[(i,b["f_bus"],b["t_bus"]) for (i,b) in ref[:dcline]]`,
* `:arcs_to_dc` -- the set `[(i,b["t_bus"],b["f_bus"]) for (i,b) in ref[:dcline]]`,
* `:arcs_dc` -- the set of arcs from both `arcs_from_dc` and `arcs_to_dc`,
* `:bus_arcs_dc` -- the mapping `Dict(i => [(l,i,j) for (l,i,j) in ref[:arcs_dc]])`, and
* `:buspairs_dc` -- (see `buspair_parameters(ref[:arcs_from_dc], ref[:dcline], ref[:bus])`),

If `:ne_branch` exists, then the following keys are also available with similar semantics:

* `:ne_branch`, `:ne_arcs_from`, `:ne_arcs_to`, `:ne_arcs`, `:ne_bus_arcs`, `:ne_buspairs`.


Added keys for eDisGo OPF include:

* `:bus_arcs_to` -- the mapping `Dict(j => [(l,i,j) for (l,i,j) in ref[:arcs_from]])`,
* `:bus_arcs_from` -- the mapping `Dict(i => [(l,i,j) for (l,i,j) in ref[:arcs_from]])`,
* `:bus_lines_to` -- the mapping `Dict(j => [l for (l,i,j) in ref[:arcs_from]])`,
* `:bus_gens_nd` -- the mapping `Dict(i => [gen_nd["gen_bus"] for (i,gen) in ref[:gen_nd]])`.
* `:bus_gens_slack` -- the mapping `Dict(i => [gen_slack["gen_bus"] for (i,gen) in ref[:gen_slack]])`.
* `:bus_dsm` -- the mapping `Dict(i => [dsm["dsm_bus"] for (i,dsm) in ref[:dsm]])`.
* `:bus_cps` -- the mapping `Dict(i => [cp["cp_bus"] for (i,cp) in ref[:electromobility]])`.
* `:bus_hps` -- the mapping `Dict(i => [hp["hp_bus"] for (i,hp) in ref[:heatpumps]])`.

"""
function ref_add_core!(ref::Dict{Symbol,Any})
    for (nw, nw_ref) in ref[:it][pm_it_sym][:nw]
        if !haskey(nw_ref, :conductor_ids)
            if !haskey(nw_ref, :conductors)
                nw_ref[:conductor_ids] = 1:1
            else
                nw_ref[:conductor_ids] = 1:nw_ref[:conductors]
            end
        end

        ### filter out inactive components ###
        nw_ref[:bus] = Dict(x for x in nw_ref[:bus] if (x.second["bus_type"] != pm_component_status_inactive["bus"]))
        nw_ref[:load] = Dict(x for x in nw_ref[:load] if (x.second["status"] != pm_component_status_inactive["load"] && x.second["load_bus"] in keys(nw_ref[:bus])))
        nw_ref[:shunt] = Dict(x for x in nw_ref[:shunt] if (x.second["status"] != pm_component_status_inactive["shunt"] && x.second["shunt_bus"] in keys(nw_ref[:bus])))
        nw_ref[:gen] = Dict(x for x in nw_ref[:gen] if (x.second["gen_status"] != pm_component_status_inactive["gen"] && x.second["gen_bus"] in keys(nw_ref[:bus])))
        nw_ref[:storage] = Dict(x for x in nw_ref[:storage] if (x.second["status"] != pm_component_status_inactive["storage"] && x.second["storage_bus"] in keys(nw_ref[:bus])))
        nw_ref[:switch] = Dict(x for x in nw_ref[:switch] if (x.second["status"] != pm_component_status_inactive["switch"] && x.second["f_bus"] in keys(nw_ref[:bus]) && x.second["t_bus"] in keys(nw_ref[:bus])))
        nw_ref[:branch] = Dict(x for x in nw_ref[:branch] if (x.second["br_status"] != pm_component_status_inactive["branch"] && x.second["f_bus"] in keys(nw_ref[:bus]) && x.second["t_bus"] in keys(nw_ref[:bus])))
        nw_ref[:dcline] = Dict(x for x in nw_ref[:dcline] if (x.second["br_status"] != pm_component_status_inactive["dcline"] && x.second["f_bus"] in keys(nw_ref[:bus]) && x.second["t_bus"] in keys(nw_ref[:bus])))


        ### setup arcs from edges ###
        nw_ref[:arcs_from] = [(i,branch["f_bus"],branch["t_bus"]) for (i,branch) in nw_ref[:branch]]
        nw_ref[:arcs_to]   = [(i,branch["t_bus"],branch["f_bus"]) for (i,branch) in nw_ref[:branch]]
        nw_ref[:arcs] = [nw_ref[:arcs_from]; nw_ref[:arcs_to]]

        nw_ref[:arcs_from_dc] = [(i,dcline["f_bus"],dcline["t_bus"]) for (i,dcline) in nw_ref[:dcline]]
        nw_ref[:arcs_to_dc]   = [(i,dcline["t_bus"],dcline["f_bus"]) for (i,dcline) in nw_ref[:dcline]]
        nw_ref[:arcs_dc]      = [nw_ref[:arcs_from_dc]; nw_ref[:arcs_to_dc]]

        nw_ref[:arcs_from_sw] = [(i,switch["f_bus"],switch["t_bus"]) for (i,switch) in nw_ref[:switch]]
        nw_ref[:arcs_to_sw]   = [(i,switch["t_bus"],switch["f_bus"]) for (i,switch) in nw_ref[:switch]]
        nw_ref[:arcs_sw] = [nw_ref[:arcs_from_sw]; nw_ref[:arcs_to_sw]]


        ### bus connected component lookups ###
        bus_loads = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i, load) in nw_ref[:load]
            push!(bus_loads[load["load_bus"]], i)
        end
        nw_ref[:bus_loads] = bus_loads

        bus_shunts = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i,shunt) in nw_ref[:shunt]
            push!(bus_shunts[shunt["shunt_bus"]], i)
        end
        nw_ref[:bus_shunts] = bus_shunts

        bus_gens = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i,gen) in nw_ref[:gen]
            push!(bus_gens[gen["gen_bus"]], i)
        end
        nw_ref[:bus_gens] = bus_gens

        bus_dsm = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i,dsm) in nw_ref[:dsm]
            push!(bus_dsm[dsm["dsm_bus"]], i)
        end
        nw_ref[:bus_dsm] = bus_dsm

        bus_cps = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i,cp) in nw_ref[:electromobility]
            push!(bus_cps[cp["cp_bus"]], i)
        end
        nw_ref[:bus_cps] = bus_cps

        bus_hps = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i,hp) in nw_ref[:heatpumps]
            push!(bus_hps[hp["hp_bus"]], i)
        end
        nw_ref[:bus_hps] = bus_hps

        bus_gens_nd = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i,gen) in nw_ref[:gen_nd]
            push!(bus_gens_nd[gen["gen_bus"]], i)
        end
        nw_ref[:bus_gens_nd] = bus_gens_nd

        bus_gens_slack = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i,gen) in nw_ref[:gen_slack]
            push!(bus_gens_slack[gen["gen_bus"]], i)
        end
        nw_ref[:bus_gens_slack] = bus_gens_slack

        bus_storage = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (i,strg) in nw_ref[:storage]
            push!(bus_storage[strg["storage_bus"]], i)
        end
        nw_ref[:bus_storage] = bus_storage

        bus_arcs = Dict((i, Tuple{Int,Int,Int}[]) for (i,bus) in nw_ref[:bus])
        for (l,i,j) in nw_ref[:arcs]
            push!(bus_arcs[i], (l,i,j))
        end
        nw_ref[:bus_arcs] = bus_arcs

        bus_arcs_to = Dict((i, Tuple{Int,Int,Int}[]) for (i,bus) in nw_ref[:bus])
        for (l,i,j) in nw_ref[:arcs_from]
            push!(bus_arcs_to[j], (l,i,j))
        end
        nw_ref[:bus_arcs_to] = bus_arcs_to

        bus_arcs_from = Dict((i, Tuple{Int,Int,Int}[]) for (i,bus) in nw_ref[:bus])
        for (l,i,j) in nw_ref[:arcs_from]
            push!(bus_arcs_from[i], (l,i,j))
        end
        nw_ref[:bus_arcs_from] = bus_arcs_from

        bus_lines_to = Dict((i, Int[]) for (i,bus) in nw_ref[:bus])
        for (l,i,j) in nw_ref[:arcs_from]
            push!(bus_lines_to[j], l)
        end
        nw_ref[:bus_lines_to] = bus_lines_to

        bus_arcs_dc = Dict((i, Tuple{Int,Int,Int}[]) for (i,bus) in nw_ref[:bus])
        for (l,i,j) in nw_ref[:arcs_dc]
            push!(bus_arcs_dc[i], (l,i,j))
        end
        nw_ref[:bus_arcs_dc] = bus_arcs_dc

        bus_arcs_sw = Dict((i, Tuple{Int,Int,Int}[]) for (i,bus) in nw_ref[:bus])
        for (l,i,j) in nw_ref[:arcs_sw]
            push!(bus_arcs_sw[i], (l,i,j))
        end
        nw_ref[:bus_arcs_sw] = bus_arcs_sw

        ### reference bus lookup (a set to support multiple connected components) ###
        ref_buses = Dict{Int,Any}()
        for (k,v) in nw_ref[:bus]
            if v["bus_type"] == 3
                ref_buses[k] = v
            end
        end

        nw_ref[:ref_buses] = ref_buses

        if length(ref_buses) > 1
            Memento.warn(_LOGGER, "multiple reference buses found, $(keys(ref_buses)), this can cause infeasibility if they are in the same connected component")
        end

        ### aggregate info for pairs of connected buses ###
        if !haskey(nw_ref, :buspairs)
            nw_ref[:buspairs] = calc_buspair_parameters(nw_ref[:bus], nw_ref[:branch], nw_ref[:conductor_ids], haskey(nw_ref, :conductors))
        end
    end
end
