"""
# cMIP
Multiperiod optimal power flow as branch flow model including network expansion 
with continuous variable representing number of lines
## Arguments
- `pm::GenericPowerModel{T} where T <: PowerModels.AbstractBFForm`
### optional
- `maxexp::Integer` maximal allowed expansion of lines, DEFAULT = `10`
- `obj::String` choose objective function between `both`,`onlyGen`,`onlyExp` 
    for minimizing generation and expansion or one of them
### setup:
`` \\begin{equation}
\\begin{aligned}
\\min_{\\substack{p_g^t,q_g^t,v^t,\\ell^t,P^t,Q^t,n}} &\\quad \\sum_{\\left(i,k \\right)\\in E}c_{ik}n_{ik} +\\sum_{t\\in \\mathcal{T}} \\sum_{i \\in N} C_g\\left(p_{g,i}^t\\right)&& \\label{obj}\\newline
\\text{subject to } \\left( \\forall t \\in \\mathcal{T} \\right)&&&\\nonumber\\newline
& \\displaystyle p_k =  \\sum_{m:k\\rightarrow m}n_{km}P_{km} - \\sum_{i:i\\rightarrow k}n_{ik}\\left(P_{ik} - \\boldsymbol{r_{ik}} \\ell_{ik}  \\right) + g_kv_k ,&&\\forall k \\in N \\label{realPB}\\newline
& \\displaystyle q_k =  \\sum_{m:k\\rightarrow m}n_{km}Q_{km} - \\sum_{i:i\\rightarrow k}n_{ik}\\left(Q_{ik} - \\boldsymbol{x_{ik}} \\ell_{ik}  \\right) - b_kv_k ,&&\\forall k \\in N \\label{imagPB}\\newline
& v_k = v_i - n_{ik}\\left(2\\left(\\boldsymbol{r_{ik}}P_{ik} + \\boldsymbol{x_{ik}}Q_{ik} \\right) + \\left(\\boldsymbol{r_{ik}}^2 + \\boldsymbol{x_{ik}}^2 \\right) \\ell_{ik}\\right), && \\forall \\left(i,k \\right)  \\in E \\label{ohm}\\newline
& v_i \\ell_{ik} = P_{ik}^2 + Q_{ik}^2,&&\\forall \\left(i,k \\right)  \\in E\\label{BF}\\newline
& \\boldsymbol{\\underline{v_i}}\\leq v_i \\leq\\boldsymbol{\\overline{v_i}},\\quad \\boldsymbol{\\underline{p_i}}\\leq p_i \\leq\\boldsymbol{\\overline{p_i}},\\quad \\boldsymbol{\\underline{q_i}}\\leq q_i \\leq\\boldsymbol{\\overline{q_i}}&& \\forall i \\in N\\newline
& \\ell_{ik} \\leq \\boldsymbol{\\overline{\\ell_{ik}}},\\quad-\\boldsymbol{\\overline{Q_{ik}}}\\leq Q_{ik} \\leq\\boldsymbol{\\overline{Q_{ik}}},\\quad-\\boldsymbol{\\overline{P_{ik}}}\\leq P_{ik} \\leq\\boldsymbol{\\overline{P_{ik}}}  &&\\forall \\left(i,k \\right)  \\in E\\label{Imax}\\newline
&&&\\newline
&1\\leq n_{ik} \\leq \\boldsymbol{N_{max}}, \\quad n_{ik} \\in \\mathbb{R}&&\\forall \\left(i,k \\right)  \\in E
\\end{aligned}
\\end{equation} ``
"""
function post_opf_bf_nep_cmip(pm::GenericPowerModel{T};maxexp::Integer=10,obj::String="both") where T <: PowerModels.AbstractBFForm
    mip = false
    # costfactor for network expansion
    costfactor = 10
    # add variables for network expansion
    
    add_var_ne(pm,mip,maxexp=maxexp)
    for (t,network) in nws(pm)
        add_var_sqr_voltage(pm,t)
        add_var_power_gen(pm,t);
        add_var_sqr_current_magnitude_ne(pm,t)
        add_var_power_flow_ne(pm,t)
        
#         for (id,slack) in ref(pm,:ref_buses)
# #           JuMP.fix(var(pm,t,:w,slack["bus_i"]),1.0)
#             gen_nr = ref(pm,:bus_gens,slack["bus_i"])[1]
#             setlowerbound(var(pm,t,:pg,gen_nr),-Inf)
#             setupperbound(var(pm,t,:pg,gen_nr),Inf)
#             setlowerbound(var(pm,t,:qg,gen_nr),-Inf)
#             setupperbound(var(pm,t,:qg,gen_nr),Inf)
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
    
    for (t,network) in nws(pm)            
        # adding constraint for branch flow model
        # Power Balance
        for i in ids(pm, :bus)
            constraint_power_balance_ne(pm,i,t)
        end
        # Ohms Law and branch flow over each line
        for i in ids(pm, :branch) 
            constraint_branch_flow_ne(pm,i,t)
            constraint_ohms_law_ne(pm,i,t)
        end
    end
    add_objective_ne(pm,ismultinetwork(pm),costfactor,obj);
end

