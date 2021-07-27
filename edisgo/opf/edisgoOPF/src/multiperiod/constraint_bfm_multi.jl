include("constraints/storage_constraints.jl")
include("constraints/curtailment_constraints.jl")
"""
branch flow constraint:

``v_i \\ell_{ik} = P_{ik}^2 + Q_{ik}^2``
"""
function constraint_branch_flow(pm, i, nw::Int=pm.cnw, cnd::Int=pm.ccnd,relaxed::Bool=false)
    branch = ref(pm,nw,:branch,i)
    f_bus = branch["f_bus"]
    t_bus = branch["t_bus"]
    idx = (i,f_bus,t_bus)
    p_fr = var(pm, nw,:p, idx)
    q_fr = var(pm, nw,:q, idx)
    w_fr = var(pm,nw,:w,f_bus)
    cm = var(pm,nw,:cm,i)
    if !relaxed
        @constraint(pm.model, p_fr^2 + q_fr^2 == w_fr*cm)
    else
        @constraint(pm.model, p_fr^2 + q_fr^2 <= w_fr*cm)
    end
end
"""
Power balance constraint:

``p_k =  \\sum_{m:k\\rightarrow m}P_{km} - \\sum_{i:i\\rightarrow k}\\left(P_{ik} - r_{ik}\\ell_{ik}  \\right) + g_kv_k ,\\forall k \\in N``

``q_k =  \\sum_{m:k\\rightarrow m}Q_{km} - \\sum_{i:i\\rightarrow k}\\left(Q_{ik} - x_{ik}\\ell_{ik}  \\right) - b_kv_k ,\\forall k \\in N``
"""
function constraint_power_balance(pm, i,nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    n_branches = length(ref(pm,:branch))
    p = var(pm, nw, :p)
    q = var(pm, nw, :q)
    w = var(pm,nw,:w)
    cm = var(pm,nw,:cm)
    pg = var(pm, nw, :pg)
    qg = var(pm, nw, :qg)
    pd = var(pm, nw, :pd)
    qd = var(pm, nw, :qd)

    bus_gens = ref(pm,nw,:bus_gens,i)
    bus_loads = ref(pm,nw,:bus_loads,i)
    bus_gens = ref(pm, nw, :bus_gens, i)
    bus_loads = ref(pm, nw, :bus_loads, i)
    bus_shunts = ref(pm, nw, :bus_shunts, i)
    
    if haskey(var(pm,nw),:uc)
        bus_storage = ref(pm,nw,:bus_storage,i)
        uc = var(pm,nw,:uc)
        ud = var(pm,nw,:ud)
    else
        bus_storage = []
    end

    bus_pd = Dict(k => ref(pm, nw, :load, k, "pd", cnd) for k in bus_loads)
    bus_qd = Dict(k => ref(pm, nw, :load, k, "qd", cnd) for k in bus_loads)

    bus_gs = Dict(k => ref(pm, nw, :shunt, k, "gs", cnd) for k in bus_shunts)
    bus_bs = Dict(k => ref(pm, nw, :shunt, k, "bs", cnd) for k in bus_shunts)
    
    incoming_arcs = [idx for idx in ref(pm)[:arcs][1:n_branches] if idx[3]==i]
    outgoing_arcs = [idx for idx in ref(pm)[:arcs][1:n_branches] if idx[2]==i]
    
    branch = ref(pm,:branch)
    if haskey(var(pm),:r)
        r = var(pm, :r)
    else
        r = Dict(k => b["br_r"] for (k,b) in branch)
    end

    if haskey(var(pm),:x)
        x = var(pm,:x)
    else
        x = Dict(k => b["br_x"] for (k,b) in branch)
    end

    @constraint(pm.model, sum(pg[g] for g in bus_gens)-sum(pd[l] for l in bus_loads) +
                        sum(ud[s] - uc[s] for s in bus_storage) ==
                        sum(p[idx] for idx in outgoing_arcs)-
                        sum(p[idx]-r[idx[1]]*cm[idx[1]] for idx in incoming_arcs) +
                        sum(gs for gs in values(bus_gs))*w[i])

    @constraint(pm.model, sum(qg[g] for g in bus_gens)-sum(qd[l] for l in bus_loads) ==
                        sum(q[idx] for idx in outgoing_arcs)-
                        sum(q[idx]-x[idx[1]]*cm[idx[1]] for idx in incoming_arcs)-
                        sum(bs for bs in values(bus_bs))*w[i])
                         
end
        
function constraint_ohms_law(pm, i, nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    branch = ref(pm,:branch,i)
    f_bus = branch["f_bus"]
    t_bus = branch["t_bus"]
    idx = (i,f_bus,t_bus)
    p_fr = var(pm, nw, :p, idx)
    q_fr = var(pm, nw,:q, idx)
    w_fr = var(pm,nw,:w,f_bus)
    w_to = var(pm,nw,:w,t_bus)
    cm = var(pm,nw, :cm,i)

    if haskey(var(pm),:r)
        r = var(pm, :r, i)
        r_sqr= var(pm)[:r_sqr][i]
    else
        r = branch["br_r"]
        r_sqr = r^2
    end

    if haskey(var(pm),:x)
        x = var(pm,:x,i)
        x_sqr = var(pm)[:x_sqr][i]
    else
        x = branch["br_x"]
        x_sqr = x^2
    end
    @constraint(pm.model, w_to == w_fr - 2(r*p_fr + x*q_fr) + (r_sqr+x_sqr)*cm)
end

function constraint_current_rating(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    """
    adds constraints for network expansion for dependencies of resistance, maximal allowed current and squared current
    input: 
        pm:: GenericPowerModel
    
    """
    I_max =  PowerModels.var(pm, :I_max)
    I_ub = getupperbound(I_max)
    I_lb = getlowerbound(I_max)
    cm = PowerModels.var(pm,nw,:cm)
    for i in ids(pm, pm.cnw, :branch)
        @constraint(pm.model,cm[i]<=I_max[i]^2)
    end
end

function constraint_current_rating_relaxed(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    """
    adds constraints in network expansion between maximal allowed current and squared current flow as linear relaxation 
    with a point-slope approximation between the lower bound and upper bound of the maximal allowed currrent
    input: 
        pm:: GenericPowerModel
    
    """
    I_max =  PowerModels.var(pm, :I_max)
    I_ub = getupperbound(I_max)
    I_lb = getlowerbound(I_max)
    cm = PowerModels.var(pm,nw,:cm)
    for i in ids(pm, pm.cnw, :branch)
        @constraint(pm.model,cm[i]<=(I_lb[i]+I_ub[i])*I_max[i] -I_lb[i]*I_ub[i])
    end
end


function constraint_network_expansion(pm)
    I_max =  PowerModels.var(pm, :I_max)
    if haskey(var(pm),:r)
        r =  PowerModels.var(pm, :r)
        for i in ids(pm, pm.cnw, :branch)
            @constraint(pm.model,r[i]*I_max[i] == getupperbound(r[i])*getlowerbound(I_max[i]))    
        end
    end

    if haskey(var(pm),:x)
        x = PowerModels.var(pm,:x)
        for i in ids(pm,pm.cnw,:branch)
            @constraint(pm.model,x[i]*I_max[i] == getupperbound(x[i])*getlowerbound(I_max[i]))
        end
    end
end

function constraint_soc(pm,i,nw::Int=pm.cnw)
    constraint_soc_initial(pm,i,nw)
end

function constraint_soc_initial(pm,i,nw::Int=pm.cnw)
    if haskey(pm.data,"time_elapsed")
        Ts  = pm.data["time_elapsed"]
    else
        println("network data should specify time_elapsed, using 1.0 as a default")
        Ts = 1.0
    end
    uc = var(pm,nw,:uc,i)
    ud = var(pm,nw,:ud,i)
    soc_nw = var(pm,nw,:soc,i)
    soc_0 = ref(pm,nw,:storage,i,"energy")
    eta_c = ref(pm,nw,:storage,i,"charge_efficiency")
    eta_d = ref(pm,nw,:storage,i,"discharge_efficiency")
    @constraint(pm.model, soc_nw - soc_0 == Ts*(eta_c * uc - ud/eta_d))
end
function constraint_soc_final(pm,i,nw)
    if haskey(pm.data,"time_elapsed")
        Ts  = pm.data["time_elapsed"]
    else
        println("network data should specify time_elapsed, using 1.0 as a default")
        Ts = 1.0
    end
    uc = var(pm,nw,:uc,i)
    ud = var(pm,nw,:ud,i)
    soc_nw = var(pm,nw,:soc,i)
    soc_final = ref(pm,nw,:storage,i,"energy")
    eta_c = ref(pm,nw,:storage,i,"charge_efficiency")
    eta_d = ref(pm,nw,:storage,i,"discharge_efficiency")
    soc_ub = getupperbound(var(pm,nw,:soc,i))
    @constraint(pm.model, soc_nw + Ts*(eta_c * uc - ud/eta_d) <=soc_ub)
    @constraint(pm.model, soc_nw + Ts*(eta_c * uc - ud/eta_d) >=0)
   #@constraint(pm.model, soc_final - soc_nw == Ts*(eta_c * uc - ud/eta_d))
end