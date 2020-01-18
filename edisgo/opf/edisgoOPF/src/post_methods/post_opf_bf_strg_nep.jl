"""
# MPOPF including NEP and storages
## Arguments
- `pm::GenericPowerModel{T} where T <: PowerModels.AbstractBFForm`
### optional
- `maxexp::Integer` maximal allowed expansion of lines, DEFAULT = `10`
- `obj::String` DEFAULT = `both`, choose objective function between `both`,`onlyGen`,`onlyExp` 
    for minimizing generation and expansion or one of them
- `relaxation::String` relaxation scheme that is used, DEFAULT = `none`, options `cr`,`soc`,`soc_cr`

### Setup:

`` \\begin{equation}
\\begin{aligned}
& \\displaystyle p_k =  \\sum_{m:k\\rightarrow m}P_{km} - \\sum_{i:i\\rightarrow k}\\left(P_{ik} - r_{ik}\\ell_{ik}  \\right) + g_kv_k ,&&\\forall k \\in N\\newline
& \\displaystyle q_k =  \\sum_{m:k\\rightarrow m}Q_{km} - \\sum_{i:i\\rightarrow k}\\left(Q_{ik} - x_{ik}\\ell_{ik}  \\right) - b_kv_k ,&&\\forall k \\in N \\newline
& v_k = v_i - 2\\left(r_{ik}P_{ik} + x_{ik}Q_{ik} \\right) + \\left(r_{ik}^2 + x_{ik}^2 \\right) \\ell_{ik}, && \\forall \\left(i,k \\right)  \\in E \\newline
& v_i \\ell_{ik} = P_{ik}^2 + Q_{ik}^2,&&\\forall \\left(i,k \\right)  \\in E\\label{BF}\\newline
& \\ell_{ik} \\leq \\left| I_{ik}^{max}\\right|^2 &&\\forall \\left(i,k \\right)  \\in E\\newline
& r_{ik}I_{ik}^{max} = \\boldsymbol{r_{ik}^0I_{ik}^{max,0}}  &&\\forall 		\\left(i,k \\right)  \\in E\\newline
& x_{ik}I_{ik}^{max} = \\boldsymbol{x_{ik}^0 I_{ik}^{max,0}}  &&\\forall \\left(i,k \\right)  \\in E\\newline
& T_s \\left( \\eta_{c,i} u_{c,i}^t - \\frac{u_{d,i}^t}{\\eta_{d,i}}\\right)=e_i^{t+1} - e_i^t && \\forall i \\in S\\newline
&e_i^0 = e_i^{T+1} &&\\forall i \\in S
\\end{aligned}
\\end{equation} ``

### Relaxation schemes:
#### `cr` relaxation:
`` \\begin{equation}
\\begin{aligned}
& \\quad \\ell_{ik} \\leq \\left(I_{ik}^{max}\\right)^2 & \\newline
\\Rightarrow & \\quad \\ell_{ik} \\leq \\left(\\overline{I_{ik}^{max}} + \\underline{I_{ik}^{max}}\\right)I_{ik}^{max}-\\overline{I_{ik}^{max}}\\underline{I_{ik}^{max}} & \\newline
\\end{aligned}
\\end{equation} ``

#### `soc` relaxation:
`` \\begin{equation}
\\begin{aligned}
& \\quad v_i \\ell_{ik} = P_{ik}^2 + Q_{ik}^2& \\newline
\\Rightarrow & \\quad v_i \\ell_{ik} \\geq P_{ik}^2 + Q_{ik}^2& \\newline
\\end{aligned}
\\end{equation} ``

"""
function post_opf_bf_strg_nep(pm::GenericPowerModel{T};maxexp::Integer=10,obj::String="both",relaxation::String="none") where T <: PowerModels.AbstractBFForm
    # cost for network expansion
    costfactor = 10
    if relaxation=="none"
        socp = false
        cr =  false
    elseif relaxation=="cr"
        socp = false
        cr = true
    elseif relaxation=="soc"
        socp = true
        cr = false
    elseif relaxation=="soc_cr"
        socp = true
        cr = true
    else
        @error("this relaxation scheme is not supported, 
        choose from 
            'none',
            'cr',
            'soc',
            'soc_cr'")
        return
    end
    
    # add expansion variables and their constraints for the whole timehorizon
    add_var_max_current(pm)
    add_var_resistance(pm)
    constraint_network_expansion(pm)
    
    # add voltage and flow variebls for each time step in timehorizon
    for (t,network) in nws(pm)
        add_var_storage(pm,t)
        for i in ids(pm,t,:storage)
            constraint_complementary_approx(pm,i,nw=t)
        end
        add_var_sqr_voltage(pm,t)
        add_var_power_gen(pm,t)
        add_var_power_flow(pm,t)
        add_var_sqr_current_magnitude(pm,t)
        # unbound power injection variables for slack bus
#         for (id,slack) in ref(pm,:ref_buses)
#             gen_id = ref(pm,:bus_gens,slack["bus_i"])[1]
# #             JuMP.fix(var(pm,t,:w,slack["bus_i"]),1.0)
#             setlowerbound(var(pm,t,:pg,gen_id),-Inf)
#             setupperbound(var(pm,t,:pg,gen_id),Inf)
#             setlowerbound(var(pm,t,:qg,gen_id),-Inf)
#             setupperbound(var(pm,t,:qg,gen_id),Inf)
#         end
        for (id,slack) in ref(pm,:ref_buses)
            for gen_id in ref(pm,:bus_gens,slack["bus_i"])
                # find slack generator if key not in gens than first gen on slack bus is set
                # to slack
                if !haskey(ref(pm,:gen,gen_id),"gen_slack")
                    gen_id = ref(pm,:bus_gens,slack["bus_i"])[1]
                    @warn("generator $(gen_id) on bus $(slack["bus_i"]) is set to slack by unbounding lower and upper limits of injection")
                    setlowerbound(var(pm,t,:pg,gen_id),-Inf)
                    setupperbound(var(pm,t,:pg,gen_id),Inf)
                    setlowerbound(var(pm,t,:qg,gen_id),-Inf)
                    setupperbound(var(pm,t,:qg,gen_id),Inf)
                    break
                elseif ref(pm,:gen,gen_id)["gen_slack"]==1
                    setlowerbound(var(pm,t,:pg,gen_id),-Inf)
                    setupperbound(var(pm,t,:pg,gen_id),Inf)
                    setlowerbound(var(pm,t,:qg,gen_id),-Inf)
                    setupperbound(var(pm,t,:qg,gen_id),Inf)
                    break
                end
            end
        end
    end

    # set estimated upper limit for flow variables with maxexp
    set_ub_flows(pm,maxexp)

    # set constraint for power flow equations and technical ones for each time step in timehorizon
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
    network_ids = sort(collect(nw_ids(pm)))
    n_1 = network_ids[1]
    #for i in ids(pm, :storage, nw=n_1)
    #    JuMP.fix(var(pm,n_1,:soc,i),ref(pm,n_1,:storage,i,"energy"))
    #end
    for n_2 in network_ids[2:end]
        for i in ids(pm, :storage, nw=n_2)
            constraint_soc(pm, i, n_1, n_2)
        end
        n_1 = n_2
    end

    n_0 = network_ids[1]
    n_final = network_ids[end]
    for i in ids(pm,:storage,nw=n_final)
        constraint_soc(pm,i,n_final,n_0)
    end

    #n_final = network_ids[end]
    #for i in ids(pm,:storage,nw=n_final)
    #    constraint_soc_final(pm,i,n_final)
    #end

    # add objective scenario chosen with obj
    add_objective(pm,ismultinetwork=ismultinetwork(pm),cost_factor=costfactor,scenario=obj) 
end

