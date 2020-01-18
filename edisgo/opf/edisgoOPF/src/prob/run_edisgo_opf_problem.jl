function run_edisgo_opf_problem(network_name::String,solution_file::String)
    data = read_edisgo_problem(network_name)
    pm = PowerModels.GenericPowerModel(data,SOCBFForm);

    post_method_edisgo(pm)

    JuMP.setsolver(pm.model,IpoptSolver(mumps_mem_percent=100))
    status,sol_time = @timed solve(pm.model)
    solution_file = "$(solution_file)_$(pm.data["scenario"])_$(pm.data["relaxation"])"
    write_opf_solution(pm,status,sol_time,solution_file)
end

function post_method_edisgo(pm)
    ## Preprocessing generated powermodel
    if pm.data["curtailment_requirement"] || pm.data["curtailment_allowance"]
        # Add key :fluct_gen to ref() containing generators which are fluctuating, i.e. RES
        nws_data = nws(pm)
        for (n, nw_data) in nws_data
            ref = nws_data[n]
            fluct_gen = Dict()
            for (k,g) in ref[:gen]
                if haskey(g,"fluctuating") && g["fluctuating"] == 1
                    fluct_gen[k] = g
                end
            end
            pm.ref[:nw][n][:fluct_gen] = fluct_gen
        end
    else
        println("let fluctuating generators be negative load, such that no variables are created")
        for (nw,net) in nws(pm)
            for (id,gen) in net[:gen]
                # if fluctuating, generator non-dispatchable and let generator be a negative load
                if gen["fluctuating"]==1
                    gen_bus = gen["gen_bus"]
                    gen_id = gen["index"]
                    if isempty(net[:bus_loads][gen_bus])
                        # if generator at bus without a load, create new load ID and add load to net[:load] dictionary
                        load_id = length(net[:load])+1
                        push!(net[:bus_loads][gen_bus],load_id)
                        net[:load][load_id] = Dict{String,Any}("load_bus"=>gen_bus,
                            "status"=>true, "index"=>load_id, "qd"=>-gen["qmax"],"pd"=> -gen["pmax"])
                    else
                        # subtract current value of generator to existing load
                        load_id = net[:bus_loads][gen_bus][1]
                        net[:load][load_id]["qd"] -= gen["qmax"]
                        net[:load][load_id]["pd"] -= gen["pmax"]
                    end
                    # delete generator from net[:gen] dictionary
                    delete!(net[:gen],gen_id)
                    # remove generator id from net[:bus_gens][gen_bus]
                    filter!(e->e!=gen_id,net[:bus_gens][gen_bus])
                end
            end
        end
    end



    ## Build model
    ## ==========================

    ### Variables
    #### Storage units
    if pm.data["storage_units"]
        add_var_energy_rating(pm)
        for (t,network) in nws(pm)
            add_var_charging_rate(pm,nw=t,bounded=false)
            add_var_soc(pm,nw=t,bounded=false)
        end
    end
    #### Network expansion variables
    add_var_max_current(pm)
    add_var_resistance(pm)
    constraint_network_expansion(pm)
    #### Power flow variables
    for (t,network) in nws(pm)
        add_var_sqr_voltage(pm,t)
        add_var_power_gen(pm,t)
        add_var_power_flow(pm,t)
        add_var_sqr_current_magnitude(pm,t)
        # unbound power injection variables for slack bus
        for (id,slack) in ref(pm,:ref_buses)
            for gen_id in ref(pm,:bus_gens,slack["bus_i"])
                if !haskey(ref(pm,:gen,gen_id),"gen_slack")
                    gen_id = ref(pm,:bus_gens,slack["bus_i"])[1]
                    @warn("generator $(gen_id) on bus $(slack["bus_i"]) is set to slack by unbounding lower and upper limits of injection")
                    setlowerbound(var(pm,t,:pg,gen_id),-Inf)
                    setupperbound(var(pm,t,:pg,gen_id),Inf)
                    setlowerbound(var(pm,t,:qg,gen_id),-Inf)
                    setupperbound(var(pm,t,:qg,gen_id),Inf)
                    @warn("voltage at bus $(slack["bus_i"]) is set to 1.0")
                    setlowerbound(var(pm,t,:w,slack["bus_i"]),1.0)
                    setupperbound(var(pm,t,:w,slack["bus_i"]),1.0)
                    break
                elseif ref(pm,:gen,gen_id)["gen_slack"]==1
                    setlowerbound(var(pm,t,:pg,gen_id),-Inf)
                    setupperbound(var(pm,t,:pg,gen_id),Inf)
                    setlowerbound(var(pm,t,:qg,gen_id),-Inf)
                    setupperbound(var(pm,t,:qg,gen_id),Inf)
                    @warn("voltage at bus $(slack["bus_i"]) is set to 1.0")
                    setlowerbound(var(pm,t,:w,slack["bus_i"]),1.0)
                    setupperbound(var(pm,t,:w,slack["bus_i"]),1.0)
                    break
                end
            end
        end    
    end
    maxexp = pm.data["max_exp"]
    set_ub_flows(pm,maxexp)
    ### Constraints
    #### Relaxation scheme
    if pm.data["relaxation"]=="none"
        socp = false
        cr =  false
    elseif pm.data["relaxation"]=="cr"
        socp = false
        cr = true
    elseif pm.data["relaxation"]=="soc"
        socp = true
        cr = false
    elseif pm.data["relaxation"]=="soc_cr"
        socp = true
        cr = true
    else
        @warn("relaxation scheme $(pm.data["relaxation"]) is not supported, 
        choose from 
            'none',
            'cr',
            'soc',
            'soc_cr'
        Set to default 'none'")
        socp = false
        cr =  false
    end
    #### Power flow equations
    for (t,network) in nws(pm)
        # current limit depending on maximal allowed current I_max variable
        if cr
            constraint_current_rating_relaxed(pm,t)    
        else
            constraint_current_rating(pm,t)
        end
    
        # adding constraint for branch flow model
        # Power Balance for each bus
        for i in ids(pm, :bus)
            constraint_power_balance(pm,i,t)
        end
        # Ohms Law and branch flow over each line
        for i in ids(pm, :branch)        
            constraint_branch_flow(pm,i,t,pm.ccnd,socp)
            constraint_ohms_law(pm,i,t)
        end
    end
    #### Storage constraints
    if pm.data["storage_units"]
        constraint_total_storage_capacity(pm)
        network_ids = sort(collect(nw_ids(pm)))
        for t in network_ids
            # periodic boundaries: T_final + 1 = T_0
            t_2 = t+1 in network_ids ? t+1 : (t+1) % length(network_ids)
            for i in ids(pm, nw=t,:storage)
                constraint_energy_rating(pm,i,nw=t)
                constraint_charge_rating(pm,i,nw=t)
                constraint_soc(pm,i,t,t_2)
            end
        end
    end
    #### Curtailment constraints
    # TODO check if requirement and allowance do not disagree, e.g. sum(curtailment_requirement)>curtailment_allowance
    if pm.data["curtailment_requirement"]
        if length(pm.data["curtailment_requirement_series"])!=pm.data["time_horizon"]
            println("length of curtailment requirement does not match considered time horizon")
        else
            for (t,network) in nws(pm)
                constraint_curtailment_single(pm,nw=t)
            end
        end
    end
    
    if pm.data["curtailment_allowance"]
        constraint_curtailment_allowed(pm)
    end

    ### Objective
    if pm.data["objective"]=="nep"
        add_objective_nep(pm)
    end
end


