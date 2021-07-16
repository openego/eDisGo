function add_objective_ne(pm,ismultinetwork::Bool=true,cost_factor::Int=1,scenario="both")
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
                gen_cost[(nw,i)] = gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
            else
                gen_cost[(nw,i)] = 0.0
            end
        end
    end
    


    if scenario=="onlyGen"
#         @objective(pm.model,Min,sum(g["cost"][1]*pg[i] for (i,g) in ref(pm,nw,:gen) if !isempty(g["cost"])) )
        @objective(pm.model,Min, sum(
            sum(gen_cost[(nw,i)] for (i,gen) in nw_ref[:gen])
            for (nw, nw_ref) in nws(pm)))
    elseif scenario=="onlyExp"
        ne = var(pm,:ne) 
        cm = var(pm,pm.cnw,:cm)
        @objective(pm.model,Min,sum(cost_factor*sum(ne[i]*getupperbound(cm[i])^0.5 for (i,b) in ref(pm,:branch))))
    else
        ne = var(pm,:ne) 
        cm = var(pm,pm.cnw,:cm)
        @objective(pm.model,Min,sum(
            sum(gen_cost[(nw,i)] for (i,gen) in nw_ref[:gen])
            for (nw, nw_ref) in nws(pm)) + 
            cost_factor*sum(ne[i]*getupperbound(cm[i])^0.5 for (i,b) in ref(pm,:branch)))
#         @objective(pm.model,Min,sum(g["cost"][1]*pg[i] for (i,g) in ref(pm,nw,:gen) if !isempty(g["cost"])) + 
#                         sum(I_max[i] for (i,b) in ref(pm,nw,:branch)))
    end
    end
                            
        