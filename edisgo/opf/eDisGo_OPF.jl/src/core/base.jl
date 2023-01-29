"used for building ref without the need to build a initialize an AbstractPowerModel"
function build_ref(data::Dict{String,<:Any}; ref_extensions=[])
    return PowerModels.build_ref(data, eDisGo_OPF.ref_add_core!, _pm_global_keys, pm_it_name; ref_extensions=ref_extensions)
end

"""
Returns a dict that stores commonly used pre-computed data from of the data
dictionary, primarily for converting data-types, filtering out deactivated components,
and storing system-wide values that need to be computed globally.

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

        ### bus connected component lookups ###
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

    end
end

ref(pm::AbstractPowerModel, nw::Int=nw_id_default) = PowerModels.ref(pm, nw)
ref(pm::AbstractPowerModel, nw::Int, key::Symbol) = PowerModels.ref(pm, nw, key)
ref(pm::AbstractPowerModel, nw::Int, key::Symbol, idx) = PowerModels.ref(pm, nw, key, idx)
ref(pm::AbstractPowerModel, nw::Int, key::Symbol, idx, param::String) = PowerModels.ref(pm, nw, key, idx, param)
ref(pm::AbstractPowerModel, key::Symbol; nw::Int=nw_id_default) = PowerModels.ref(pm, key; nw = nw)
ref(pm::AbstractPowerModel, key::Symbol, idx; nw::Int=nw_id_default) = PowerModels.ref(pm, key, idx; nw = nw)
ref(pm::AbstractPowerModel, key::Symbol, idx, param::String; nw::Int=nw_id_default) = PowerModels.ref(pm, key, idx, param; nw = nw)
