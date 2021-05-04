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

    # allow curtailment of all generators
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

    ## Build model
    ## ==========================

    ### Variables
    #### Storage units
    if pm.data["storage_units"]
        add_var_energy_rating(pm)
        for (t,network) in nws(pm)
            add_var_soc(pm,nw=t,bounded=false)

            #= Only set up SOC, skip uc and ud for linked steps=#
            if haskey(pm.data["clusters"], t)
                continue
            end
            add_var_charging_rate(pm,nw=t,bounded=false)
        end
    end
    #### Network expansion variables
    add_var_max_current(pm)
    add_var_resistance(pm)
    constraint_network_expansion(pm)
    #### Power flow variables
    for (t,network) in nws(pm)
        #= Dont set up variables if this is a linked step=#
        if haskey(pm.data["clusters"], t)
            continue
        end
        add_var_sqr_voltage(pm,t)
        add_var_power_gen(pm,t)
        add_var_power_load(pm,t)
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
        #= Don't calculate power flow for linked steps=#
        if haskey(pm.data["clusters"], t)
            continue
        end
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
                constraint_soc(pm,i,t,t_2)

                #= Set up charge rating once per cluster=#
                if haskey(pm.data["clusters"], t)
                    continue
                end
                constraint_charge_rating(pm,i,nw=t)
            end
        end
    end
    #### Curtailment constraints
    for (t,network) in nws(pm)
        constraint_curtailment_single(pm,nw=t)
    end
    
    if pm.data["curtailment_allowance"]
        constraint_curtailment_allowed(pm)
    end

    ### Objective
    if pm.data["objective"]=="nep"
        add_objective_nep(pm)
    end

end


