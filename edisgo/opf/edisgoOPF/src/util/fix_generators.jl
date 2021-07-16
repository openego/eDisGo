#using PowerModels
#using JuMP
#using Random

function fix_generators(pm,scalefactor,gen2fix)
    network_ids = sort(collect(nw_ids(pm)))
    Random.seed!(1234)
    rand_vals = append!([1.0],rand(5:scalefactor,length(network_ids)-1)/10);
    #new_obj = pm.model.obj
    for n in network_ids
        for g in gen2fix
            pg_g = var(pm,n,:pg,g)
            ub_pg =  getupperbound(pg_g)
            JuMP.fix(pg_g,rand_vals[n]*ub_pg)
            
            #lin_arr = pm.model.obj.aff.vars.==pg_g
            #aff_coeff = pm.model.obj.aff.coeffs[lin_arr][1]
            #new_obj -= aff_coeff*pg_g
            #if !isempty(pm.model.obj.qvars1)
            #    quad_arr = pm.model.obj.qvars1.==pg_g
            #    quad_coeff = pm.model.obj.qcoeffs[quad_arr][1]
            #    new_obj -= quad_coeff*pg_g^2
            #end
        end
    end
    #@objective(pm.model,Min,new_obj)
    return rand_vals
end
