function constraint_branch_flow_ne(pm, i, nw::Int=pm.cnw, cnd::Int=pm.ccnd,relaxed::Bool=false)
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

function constraint_power_balance_ne(pm, i,nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    n_branches = length(ref(pm,:branch))
    p = var(pm, nw, :p)
    q = var(pm, nw, :q)
    w = var(pm,nw,:w)
    cm = var(pm,nw,:cm)
    pg = var(pm, nw, :pg)
    qg = var(pm, nw, :qg)

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
    r = Dict(k => b["br_r"] for (k,b) in branch)
    x = Dict(k => b["br_x"] for (k,b) in branch)

    if haskey(var(pm),:ne)
        ne = var(pm,:ne)
    else
        ne = ones(n_branches)
    end
    

    @constraint(pm.model, sum(pg[g] for g in bus_gens)-sum(pd for pd in values(bus_pd)) + 
                        sum(ud[s] - uc[s] for s in bus_storage) ==
                        sum(ne[idx[1]]*p[idx] for idx in outgoing_arcs)-
                        sum(ne[idx[1]]*(p[idx]-r[idx[1]]*cm[idx[1]]) for idx in incoming_arcs) +
                        sum(gs for gs in values(bus_gs))*w[i])
    
    @constraint(pm.model, sum(qg[g] for g in bus_gens)-sum(qd for qd in values(bus_qd)) ==
                        sum(ne[idx[1]]*q[idx] for idx in outgoing_arcs)-
                        sum(ne[idx[1]]*(q[idx]-x[idx[1]]*cm[idx[1]]) for idx in incoming_arcs)-
                        sum(bs for bs in values(bus_bs))*w[i])
                         
end
        
function constraint_ohms_law_ne(pm, i, nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    branch = ref(pm,:branch,i)
    f_bus = branch["f_bus"]
    t_bus = branch["t_bus"]
    idx = (i,f_bus,t_bus)
    p_fr = var(pm, nw, :p, idx)
    q_fr = var(pm, nw,:q, idx)
    w_fr = var(pm,nw,:w,f_bus)
    w_to = var(pm,nw,:w,t_bus)
    cm = var(pm,nw, :cm,i)

    r = branch["br_r"]
    r_sqr = r^2

    x = branch["br_x"]
    x_sqr = x^2

    if haskey(var(pm),:ne)
        ne = var(pm,:ne,i)
    else
        ne = 1
    end


    @constraint(pm.model, w_to == w_fr - 2*ne*(r*p_fr + x*q_fr) + ne*(r_sqr+x_sqr)*cm)
end