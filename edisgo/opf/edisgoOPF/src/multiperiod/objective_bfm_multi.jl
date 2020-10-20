function add_objective(pm;ismultinetwork::Bool=true,cost_factor=1.,scenario="both")
    #     pg = var(pm,nw,cnd,:pg)
    cnd=pm.ccnd
    gen_cost = Dict()
    for (nw,nw_ref) in nws(pm)
        if haskey(pm.data["clusters"], nw)
            nw = pm.data["clusters"][nw]
            nw_ref = ref(pm, nw)
        end
        for (i,gen) in nw_ref[:gen]
            pg = sum(var(pm, nw, :pg, i))
    
            if length(gen["cost"]) == 1
                gen_cost[(nw,i)] = gen["cost"][1]
            elseif length(gen["cost"]) == 2
                gen_cost[(nw,i)] = gen["cost"][1]*pg + gen["cost"][2]
            elseif length(gen["cost"]) == 3
                gen_cost[(nw,i)] = gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
            else
                gen_cost[(nw,i)] = 0.0
            end
        end
    end
    

    gen_sum = 0
    for (nw, nw_ref) in nws(pm)
        if haskey(pm.data["clusters"], nw)
            nw = pm.data["clusters"][nw]
            nw_ref = ref(pm, nw)
        end
        gen_sum += sum(gen_cost[(nw,i)] for (i,gen) in nw_ref[:gen])
    end
    if scenario=="onlyGen"
#         @objective(pm.model,Min,sum(g["cost"][1]*pg[i] for (i,g) in ref(pm,nw,:gen) if !isempty(g["cost"])) )
        @objective(pm.model,Min,gen_sum) 
    elseif scenario=="onlyExp"
        I_max = var(pm,:I_max) 
        @objective(pm.model,Min,sum(I_max[i] for (i,b) in ref(pm,:branch)))
    else
        I_max = var(pm,:I_max) 
        @objective(pm.model,Min,
            gen_sum +
            cost_factor*sum(I_max[i] for (i,b) in ref(pm,:branch)))
#         @objective(pm.model,Min,sum(g["cost"][1]*pg[i] for (i,g) in ref(pm,nw,:gen) if !isempty(g["cost"])) + 
#                         sum(I_max[i] for (i,b) in ref(pm,nw,:branch)))
    end
end
                            
function add_objective_linear(pm;ismultinetwork::Bool=true,cost_factor=1,scenario="both")
    #     pg = var(pm,nw,cnd,:pg)
    cnd=pm.ccnd
    gen_cost = Dict()
    for (nw,nw_ref) in nws(pm)
        for (i,gen) in nw_ref[:gen]
            pg = sum(var(pm, nw, :pg, i))
    
            if length(gen["cost"]) == 1
                gen_cost[(nw,i)] = gen["cost"][1]
            elseif length(gen["cost"]) == 2
                gen_cost[(nw,i)] = gen["cost"][1]*pg + gen["cost"][2]
            elseif length(gen["cost"]) == 3
                gen_cost[(nw,i)] = gen["cost"][2]*pg + gen["cost"][3]
            else
                gen_cost[(nw,i)] = 0.0
            end
        end
    end
    

    if scenario=="onlyGen"
        # @objective(pm.model,Min,sum(g["cost"][1]*pg[i] for (i,g) in ref(pm,nw,:gen) if !isempty(g["cost"])) )
        @objective(pm.model,Min, sum(
            sum(gen_cost[(nw,i)] for (i,gen) in nw_ref[:gen])
            for (nw, nw_ref) in nws(pm)))
    elseif scenario=="onlyExp"
        I_max = var(pm,:I_max) 
        @objective(pm.model,Min,sum(I_max[i] for (i,b) in ref(pm,:branch)))
    else
        I_max = var(pm,:I_max) 
        @objective(pm.model,Min,sum(
            sum(gen_cost[(nw,i)] for (i,gen) in nw_ref[:gen])
            for (nw, nw_ref) in nws(pm)) + 
            cost_factor*sum(I_max[i] for (i,b) in ref(pm,:branch)))
        #   @objective(pm.model,Min,sum(g["cost"][1]*pg[i] for (i,g) in ref(pm,nw,:gen) if !isempty(g["cost"])) + 
        #                  sum(I_max[i] for (i,b) in ref(pm,nw,:branch)))
    end
end 

function add_objective_nep(pm, cost_factor=0.001)
    branch_cost = Dict()
    I_max = var(pm,:I_max)
    for (i,branch) in ref(pm,:branch)
        if length(branch["cost"])==1
            branch_cost[i] = branch["cost"][1]*(I_max[i]/getlowerbound(I_max[i])-1)
        elseif length(branch["cost"])==2
            branch_cost[i] = branch["cost"][1]*(I_max[i]/getlowerbound(I_max[i])-1) + branch["cost"][2]
        else
            println("no costs for branch $i: $(branch["cost"])")
        end
    end

    RES = ids(pm,:fluct_gen)
    sum_curtail = 0
    for (nw,net) in nws(pm)
        pg = var(pm,nw,:pg)
        sum_curtail += sum(getupperbound(pg[i]) - pg[i] for i in RES)
    end

    sum_curtail_load = 0
    for (nw,nw_ref) in nws(pm)

        look_up_load_ids = Dict()
        for (load_ids, val) in nw_ref[:load]
           look_up_load_ids[val["load_bus"]] = load_ids
        end
        loads = Set(values(look_up_load_ids))

        pd = var(pm,nw,:pd)
        sum_curtail_load += sum(getupperbound(pd[i]) - pd[i] for i in loads)
    end

    @objective(pm.model,Min, ((sum(branch_cost[i] for (i,br) in ref(pm,:branch))+cost_factor*sum_curtail+cost_factor*sum_curtail_load)))
    #@objective(pm.model,Min, sum((getupperbound(pg[i]) - pg[i])*0.01 for i in RES))
end
