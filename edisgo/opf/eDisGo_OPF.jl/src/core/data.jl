function set_ac_bf_start_values!(network::Dict{String,<:Any})
    for (i,bus) in network["bus"]
        bus["w_start"] = bus["w"]
    end

    for (i,gen) in network["gen_nd"]
        gen["pgc_start"] = gen["pgc"]
    end

    for (i,gen) in network["gen_slack"]
        gen["pgs_start"] = gen["pgs"]
        gen["qgs_start"] = gen["qgs"]
    end

    for (i,dsm) in network["dsm"]
        dsm["pdsm_start"] = dsm["pdsm"]
        dsm["dsme_start"] = dsm["dsme"]
    end

    for (i,s) in network["storage"]
        s["ps_start"] = s["ps"]
        s["se_start"] = s["se"]
    end

    for (i,cp) in network["electromobility"]
        cp["pcp_start"] = cp["pcp"]
    end

    for (i,hp) in network["heatpumps"]
        hp["php_start"] = hp["php"]
    end

    for (i,hs) in network["heat_storage"]
        hs["phs_start"] = hs["phs"]
        hs["hse_start"] = hs["hse"]
    end

end

"""
checks bus types are suitable for a power flow study, if not, fixes them.

the primary checks are that all type 2 buses (i.e., PV) have a connected and
active generator and there is a single type 3 bus (i.e., slack bus) with an
active connected generator.

assumes that the network is a single connected component
"""
function correct_bus_types!(data::Dict{String,<:Any})
    apply_pm!(eDisGo_OPF._correct_bus_types!, data)
end

""
function _correct_bus_types!(pm_data::Dict{String,<:Any})
    bus_gens = Dict(bus["index"] => [] for (i,bus) in pm_data["bus"])

    for (i,gen) in pm_data["gen"]
        if gen["gen_status"] != 0
            push!(bus_gens[gen["gen_bus"]], i)
        end
    end
    for (i,gen) in pm_data["gen_nd"]
        if gen["gen_status"] != 0
            push!(bus_gens[gen["gen_bus"]], i)
        end
    end
    for (i,gen) in pm_data["gen_slack"]
        if gen["gen_status"] != 0
            push!(bus_gens[gen["gen_bus"]], i)
        end
    end

    slack_found = false
    for (i, bus) in pm_data["bus"]
        idx = bus["index"]
        if bus["bus_type"] == 1
            if length(bus_gens[idx]) != 0 # PQ
                #Memento.warn(_LOGGER, "active generators found at bus $(bus["bus_i"]), updating to bus type from $(bus["bus_type"]) to 2")
                #bus["bus_type"] = 2
            end
        elseif bus["bus_type"] == 2 # PV
            if length(bus_gens[idx]) == 0
                Memento.warn(_LOGGER, "no active generators found at bus $(bus["bus_i"]), updating to bus type from $(bus["bus_type"]) to 1")
                bus["bus_type"] = 1
            end
        elseif bus["bus_type"] == 3 # Slack
            if length(bus_gens[idx]) != 0
                slack_found = true
            else
                Memento.warn(_LOGGER, "no active generators found at bus $(bus["bus_i"]), updating to bus type from $(bus["bus_type"]) to 1")
                bus["bus_type"] = 1
            end
        elseif bus["bus_type"] == 4 # inactive bus
            # do nothing
        else  # unknown bus type
            new_bus_type = 1
            if length(bus_gens[idx]) != 0
                new_bus_type = 2
            end
            Memento.warn(_LOGGER, "bus $(bus["bus_i"]) has an unrecongized bus_type $(bus["bus_type"]), updating to bus_type $(new_bus_type)")
            bus["bus_type"] = new_bus_type
        end
    end

    if !slack_found
        gen = _biggest_generator(pm_data["gen"])
        if length(gen) > 0
            gen_bus = gen["gen_bus"]
            ref_bus = pm_data["bus"]["$(gen_bus)"]
            ref_bus["bus_type"] = 3
            Memento.warn(_LOGGER, "no reference bus found, setting bus $(gen_bus) as reference based on generator $(gen["index"])")
        else
            Memento.error(_LOGGER, "no generators found in the given network data, correct_bus_types! requires at least one generator at the reference bus")
        end
    end

end
