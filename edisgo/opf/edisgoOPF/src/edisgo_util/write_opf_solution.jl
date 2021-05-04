#=
write_opf_solution:
- Julia version: 
- Author: RL-INSTITUT\jaap.pedersen
- Date: 2019-11-13
=#
# using JSON

function write_opf_solution(pm,status,sol_time,network_name="test_network")
    filename = "$(network_name)_opf_sol.json"
    bus_vars = [:w]
    # symbol :ne added afterwards with ne = I_max/I_lb represents line expansion factor
    branch_vars_static =[:I_max,:r,:x,:ne]
    branch_vars = [:cm,:p,:q]
    gen_vars = [:pg,:qg]
    load_vars = [:pd,:qd]
    strg_vars_static = [:emax]
    strg_vars = [:uc,:ud,:soc]

    sol = Dict(
        "name"=>pm.data["nw"]["1"]["name"],
        "solver"=>typeof(pm.model.solver),
        "status"=>status,
        "sol_time"=>sol_time,
        "obj"=>getvalue(pm.model.obj),
        "branch" => Dict("nw"=>Dict(),"static"=>Dict()),
        "bus" => Dict("nw"=>Dict()),
        "storage"=> Dict("nw"=>Dict(),"static"=>Dict()),
        "gen" =>  Dict("nw"=>Dict()),
        "load" =>  Dict("nw"=>Dict())
    )


    for (nw,net) in nws(pm)
        sol["branch"]["nw"][nw] = Dict()
        sol["bus"]["nw"][nw] = Dict()
        sol["gen"]["nw"][nw] = Dict()
        sol["load"]["nw"][nw] = Dict()
        sol["storage"]["nw"][nw] = Dict()
    end

    for sym in branch_vars_static
        sol["branch"]["static"][string(sym)] = Dict()
    end

    for sym in strg_vars_static
        sol["storage"]["static"][string(sym)] = Dict()
    end

    for (nw,net) in nws(pm)
        for sym in branch_vars
            sol["branch"]["nw"][nw][string(sym)]=Dict()
        end
        for sym in strg_vars
            sol["storage"]["nw"][nw][string(sym)]=Dict()
        end
        for sym in bus_vars
            sol["bus"]["nw"][nw][string(sym)]=Dict()
        end
        for sym in gen_vars
            sol["gen"]["nw"][nw][string(sym)]=Dict()
        end
        for sym in load_vars
            sol["load"]["nw"][nw][string(sym)]=Dict()
        end
    end

    # save static variables of branches and storages
    for (i,br) in ref(pm,:branch)
        for sym in branch_vars_static
            if sym==:ne
                sol["branch"]["static"]["ne"][i] =  getvalue(var(pm,:I_max,i))/ getlowerbound(var(pm,:I_max,i))
                continue
            end
            sol["branch"]["static"][string(sym)][i] = getvalue(var(pm,sym,i))
        end
    end
    for (i,s) in ref(pm,:storage)
        for sym in strg_vars_static
            if sym == :emax && pm.data["storage_operation_only"]
                sol["storage"]["static"][string(sym)][i] = ref(pm,:storage,i,"energy_rating")
            else
                sol["storage"]["static"][string(sym)][i] = getvalue(var(pm,sym,i))
            end
        end
    end

    # save variables for each time step
    for (nw,net) in nws(pm)
        step = nw
        #= Replace values with those of linked step for clustered data=#
        if haskey(pm.data["clusters"], nw)
            step = pm.data["clusters"][nw]
        end
        for (i,br) in ref(pm,:branch)
            idx = (i,br["f_bus"],br["t_bus"])
            for sym in branch_vars
                try
                    sol["branch"]["nw"][nw][string(sym)][i] = getvalue(var(pm,step,sym,i))
                catch
                    sol["branch"]["nw"][nw][string(sym)][idx[1]] = getvalue(var(pm,step,sym,idx))
                end
            end
        end
        for (i,b) in ref(pm,:bus)
            for sym in bus_vars
                sol["bus"]["nw"][nw][string(sym)][i] = getvalue(var(pm,step,sym,i))
            end
        end

        for (i,g) in ref(pm,:gen)
            for sym in gen_vars
                sol["gen"]["nw"][nw][string(sym)][i] = getvalue(var(pm,step,sym,i))
            end
        end

        for (i,g) in ref(pm,:load)
            for sym in load_vars
                sol["load"]["nw"][nw][string(sym)][i] = getvalue(var(pm,step,sym,i))
            end
        end

        for (i,s) in ref(pm,:storage)
            for sym in strg_vars
                #= Read SOC value from actual time step (as it differs), other variables from linked step=#
                if sym == :soc
                    sol["storage"]["nw"][nw][string(sym)][i] = getvalue(var(pm,nw,sym,i))
                else
                    sol["storage"]["nw"][nw][string(sym)][i] = getvalue(var(pm,step,sym,i))
                end
            end
        end
    end

    jdict = JSON.json(sol)
    f = open(filename,"w")
    JSON.print(f,sol)
    close(f)
    return sol

end
