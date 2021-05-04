# using statements should be used in global context...
# using JuMP
# using PowerModels

# using a reformulation-linearization technique (rlt) 
# from Bigane2012 "Tight-and-Cheap Conic Relaxation for the AC OPF problem"
function constraint_power_balance_rlt(pm, i,nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    n_branches = length(ref(pm,:branch))
    p = var(pm, nw, :p)
    q = var(pm, nw, :q)
    w = var(pm,nw,:w)
    cm = var(pm,nw,:cm)
    pg = var(pm, nw, :pg)
    qg = var(pm, nw, :qg)
    zr = var(pm, nw, :zr)
    zx = var(pm, nw, :zx)

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

    @constraint(pm.model, sum(pg[g] for g in bus_gens)-sum(pd for pd in values(bus_pd)) + 
                        sum(ud[s] - uc[s] for s in bus_storage) ==
                        sum(p[idx] for idx in outgoing_arcs)-
                        sum(p[idx]- zr[idx[1]] for idx in incoming_arcs) +
                        sum(gs for gs in values(bus_gs))*w[i])
    
    @constraint(pm.model, sum(qg[g] for g in bus_gens)-sum(qd for qd in values(bus_qd)) ==
                        sum(q[idx] for idx in outgoing_arcs)-
                        sum(q[idx]-zx[idx[1]] for idx in incoming_arcs)-
                        sum(bs for bs in values(bus_bs))*w[i])
                         
end

function constraint_z_rlt(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    cm = PowerModels.var(pm,nw, :cm)
    I_max =  PowerModels.var(pm, :I_max)
    I_ub = getupperbound(I_max)
    I_ub =[4 for i in 1:length(ids(pm,nw,:branch))]
    r = PowerModels.var(pm, :r)
    r_ub = getupperbound(r)
    x = PowerModels.var(pm, :x)
    x_ub = getupperbound(x)
    zr = var(pm, nw,:zr)
    zx = var(pm, nw,:zx)
    for i in ids(pm,nw,:branch)
        @constraint(pm.model, zr[i]<=r_ub[i] * cm[i])
        @constraint(pm.model, zr[i]<=r[i] * I_ub[i]^2)
        @constraint(pm.model, zr[i]>=r[i] * I_ub[i]^2 + r_ub[i] * cm[i] - r_ub[i] * I_ub[i]^2)
        @constraint(pm.model, zx[i]<=x_ub[i] * cm[i])
        @constraint(pm.model, zx[i]<=x[i] * I_ub[i]^2)
        @constraint(pm.model, zx[i]>=x[i] * I_ub[i]^2 + x_ub[i] * cm[i] - x_ub[i] * I_ub[i]^2)
    end
end


function add_variables_z_rlt(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd)
    buses = ref(pm,nw)[:bus]
    branch = ref(pm,nw)[:branch]
    var(pm, nw)[:zr] = @variable(pm.model,
        [i in ids(pm, nw, :branch)], basename="zr_$(nw)",
        lowerbound = 0)
    var(pm, nw)[:zx] = @variable(pm.model,
        [i in ids(pm, nw, :branch)], basename="zx_$(nw)",
        lowerbound = 0)
end