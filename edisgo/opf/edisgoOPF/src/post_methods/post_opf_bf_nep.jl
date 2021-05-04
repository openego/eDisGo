#export post_opf_bf_nep
"""
multiperiod optimal power flow as branch flow model including network expansion
## Arguments
- `pm::GenericPowerModel{T} where T <: PowerModels.AbstractBFForm`
### optional
- `maxexp::Integer` maximal allowed expansion of lines, DEFAULT = `10`
- `obj::String` choose objective function between `both`,`onlyGen`,`onlyExp` 
    for minimizing generation and expansion or one of them
### setup:
`` \\begin{equation}
\\begin{aligned}
& \\displaystyle p_k =  \\sum_{m:k\\rightarrow m}P_{km} - \\sum_{i:i\\rightarrow k}\\left(P_{ik} - r_{ik}\\ell_{ik}  \\right) + g_kv_k ,&&\\forall k \\in N\\newline
& \\displaystyle q_k =  \\sum_{m:k\\rightarrow m}Q_{km} - \\sum_{i:i\\rightarrow k}\\left(Q_{ik} - x_{ik}\\ell_{ik}  \\right) - b_kv_k ,&&\\forall k \\in N \\newline
& v_k = v_i - 2\\left(r_{ik}P_{ik} + x_{ik}Q_{ik} \\right) + \\left(r_{ik}^2 + x_{ik}^2 \\right) \\ell_{ik}, && \\forall \\left(i,k \\right)  \\in E \\newline
& v_i \\ell_{ik} = P_{ik}^2 + Q_{ik}^2,&&\\forall \\left(i,k \\right)  \\in E\\label{BF}\\newline
& \\ell_{ik} \\leq \\left| I_{ik}^{max}\\right|^2 &&\\forall \\left(i,k \\right)  \\in E\\newline
& r_{ik}I_{ik}^{max} = \\boldsymbol{r_{ik}^0I_{ik}^{max,0}}  &&\\forall 		\\left(i,k \\right)  \\in E\\newline
& x_{ik}I_{ik}^{max} = \\boldsymbol{x_{ik}^0 I_{ik}^{max,0}}  &&\\forall \\left(i,k \\right)  \\in E
\\end{aligned}
\\end{equation} ``
"""
function post_opf_bf_nep(pm::GenericPowerModel{T};maxexp::Integer=10,obj::String="both") where T <: PowerModels.AbstractBFForm
        # cost factor for network expansion
                costfactor = 10
        # add expansion variables and their constraints for the whole timehorizon
        add_var_max_current(pm)
        add_var_resistance(pm)
        constraint_network_expansion(pm)
        
        # add voltage and flow variebls for each time step in timehorizon
        for (t,network) in nws(pm)
                add_var_sqr_voltage(pm,t)
                add_var_power_gen(pm,t)
                add_var_power_flow(pm,t)
                add_var_sqr_current_magnitude(pm,t)
                # unbound power injection variables for slack bus
#                 for (id,slack) in ref(pm,:ref_buses)
#                         gen_id = ref(pm,:bus_gens,slack["bus_i"])[1]
#         #             JuMP.fix(var(pm,t,:w,slack["bus_i"]),1.0)
#                         setlowerbound(var(pm,t,:pg,gen_id),-Inf)
#                         setupperbound(var(pm,t,:pg,gen_id),Inf)
#                         setlowerbound(var(pm,t,:qg,gen_id),-Inf)
#                         setupperbound(var(pm,t,:qg,gen_id),Inf)
#                 end
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
                constraint_current_rating(pm,t)
                
                # adding constraint for branch flow model
                # Power Balance for each bus
                for i in ids(pm, :bus)
                constraint_power_balance(pm,i,t)
                end
                # Ohms Law and branch flow over each line
                for i in ids(pm, :branch)        
                constraint_branch_flow(pm,i,t)
                constraint_ohms_law(pm,i,t)
                end
        end
        # add objective scenario chosen with obj
        add_objective(pm,ismultinetwork=ismultinetwork(pm),cost_factor=costfactor,scenario=obj)
end

