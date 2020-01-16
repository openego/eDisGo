include("variables/storage_variables.jl")
# Voltage
"variable: sqr voltage magnitude `w[i]` for `i` in `bus`es"
function add_var_sqr_voltage(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd,bounded::Bool=true)
    """
    adds variable of the squared voltage for all buses to a power model
    input: 
        pm:: GenericPowerModel
    
    """
    if bounded
        var(pm,nw)[:w]=@variable(pm.model,[i in ids(pm,nw,:bus)],basename="w_$(nw)",
            lowerbound = ref(pm, nw, :bus, i, "vmin", cnd)^2,
            upperbound = ref(pm, nw, :bus, i, "vmax", cnd)^2)
    else
        var(pm,nw)[:w]=@variable(pm.model,[i in ids(pm,nw,:bus)],basename="w_$(nw)",lowerbound = 0.0)
    end
    #     start = PowerModels.getval(ref(pm, nw, :bus, i), "w_start", cnd, 1.001))
end


# Power Generation
"variable: active and reactive power injection `pg[i]`, `qg[i]` for `i` in `gen`'s"
function add_var_power_gen(pm, nw::Int=pm.cnw, cnd::Int=pm.ccnd, bounded::Bool=true)
    """
    adds variable of acitve and reactive power generation for all generator in power model to a power model
    input: 
        pm:: GenericPowerModel
    
    """
    if bounded 
        var(pm, nw)[:pg] = @variable(pm.model,
                [i in ids(pm, nw, :gen)], basename="pg_$(nw)",
                lowerbound = ref(pm, nw, :gen, i, "pmin", cnd),
                upperbound = ref(pm, nw, :gen, i, "pmax", cnd)#,
        #         start = PowerModels.getval(ref(pm, nw, :gen, i), "pg_start", cnd)
            )
        var(pm, nw)[:qg] = @variable(pm.model,
            [i in ids(pm, nw, :gen)], basename="qg_$(nw)",
            lowerbound = ref(pm, nw, :gen, i, "qmin", cnd),
            upperbound = ref(pm, nw, :gen, i, "qmax", cnd)#,
        #     start = PowerModels.getval(ref(pm, nw, :gen, i), "qg_start", cnd)
            )
    else
        var(pm, nw)[:pg] = @variable(pm.model,
                [i in ids(pm, nw, :gen)], basename="pg_$(nw)"
            )
        var(pm, nw)[:qg] = @variable(pm.model,
            [i in ids(pm, nw, :gen)], basename="qg_$(nw)"
            )
    end
end

# Power flow
# flow_lb, flow_ub = PowerModels.calc_branch_flow_bounds(ref(pm, nw, :branch), ref(pm, nw, :bus), cnd)
"variable: active and reactive power flow `pg[(l,i,j)]`, `q[(l,i,j)]` for `(l,i,j)` in `arcs`'s"
function add_var_power_flow(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd,bounded::Bool=false)
    """
    adds variable of active and reactive power flow for each branch to a power model
    input: 
        pm:: GenericPowerModel
    
    """
    n_branches = length(ref(pm,:branch))
    var(pm, nw)[:p] = @variable(pm.model,
        [(l,i,j) in ref(pm, nw, :arcs)[1:n_branches]], basename="p_$(nw)"#,
    #     lowerbound = flow_lb[l],
    #     upperbound = flow_ub[l],
    #     start = PowerModels.getval(ref(pm, nw, :branch, l), "p_start", cnd)
    )


    var(pm, nw)[:q] = @variable(pm.model,
        [(l,i,j) in ref(pm, nw, :arcs)[1:n_branches]], basename="q_$(nw)"#,
    #     lowerbound = flow_lb[l],
    #     upperbound = flow_ub[l],
    #     start = PowerModels.getval(ref(pm, nw, :branch, l), "q_start", cnd)
    )
end

"variable: squared current magnitude `cm[i]`for `i` in `branch`'s"
function add_var_sqr_current_magnitude(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd, bounded::Bool=false)  
    """
    adds variable of squared magnitude of current for each branch to a power model, can be bounded or not
    input: 
        pm:: GenericPowerModel
    
    """
    buses = ref(pm,nw)[:bus]
    branch = ref(pm,nw)[:branch]
    var(pm, nw)[:cm] = @variable(pm.model,
        [i in ids(pm, nw, :branch)], basename="cm_$(nw)",
        lowerbound = 0)
    if bounded
        ub = Dict()
        for (i, b) in branch
            ub[i] = ((b["rate_a"][cnd]*b["tap"][cnd])/(buses[b["f_bus"]]["vmin"][cnd]))^2
            setupperbound(var(pm, nw)[:cm][i],ub[i])
        end
        
    end
end

# Adding variables for resistance of lines
function add_var_resistance(pm)
    """
    adds variable of resistance for each branch to a power model - used in case of network expansion
    input: 
        pm:: GenericPowerModel
    
    """
    branch = ref(pm)[:branch]
    r_init = Dict(k => b["br_r"] for (k,b) in branch)
    x_init = Dict(k => b["br_x"] for (k,b) in branch)
    var(pm)[:r]=@variable(pm.model,
        [i in ids(pm, :branch)], basename="r",
        lowerbound=0, upperbound=r_init[i])
        #lowerbound=r_init[i]/4, upperbound=r_init[i])

    var(pm)[:x]=@variable(pm.model,
        [i in ids(pm, :branch)], basename="x",
        lowerbound=0, upperbound=x_init[i])
        #lowerbound=x_init[i]/4, upperbound=x_init[i])
    

    r = var(pm)[:r]
    x = var(pm)[:x]
    r_sqr= var(pm)[:r_sqr]=@variable(pm.model,
        [i in ids(pm, :branch)], basename="r_sqr",
        lowerbound=0, upperbound=r_init[i]^2)
        #lowerbound=(r_init[i]/4)^2, upperbound=r_init[i]^2)
    @constraint(pm.model,[i in ids(pm,:branch)],r[i]^2==r_sqr[i])

    x_sqr = var(pm)[:x_sqr] = @variable(pm.model,
        [i in ids(pm,:branch)], basename="x_sqr",
        lowerbound=0, upperbound=x_init[i]^2)
        #lowerbound=(x_init[i]/4)^2, upperbound=x_init[i]^2)
    @constraint(pm.model,[i in ids(pm,:branch)],x[i]^2==x_sqr[i]) 

end

function add_var_max_current(pm)
    """
    adds variable of maximal current for each branch to a power model - used in case of network expansion
    input: 
        pm:: GenericPowerModel
    
    """
    buses = ref(pm)[:bus]
    branch = ref(pm)[:branch]
    ub = Dict()
    for (i, b) in branch
        ub[i] = ((b["rate_a"]*b["tap"])/(buses[b["f_bus"]]["vmin"]))
    end

    var(pm)[:I_max] = @variable(pm.model,
        [i in ids(pm, :branch)], basename="I_max",
        lowerbound = ub[i])
        #lowerbound = ub[i], upperbound = 4*ub[i])
    #@warn("Upperbound of variable maximal current rating I_max is not set")
end
# add_var_storage() only used in post_opf_bf_strg_nep.jl, for edisgo scenario look at storage_variables.jl
function add_var_storage(pm,nw::Int=pm.cnw,cnd::Int=pm.ccnd)
    var(pm,nw)[:uc] = @variable(pm.model,[i in ids(pm,nw,:storage)],basename="uc_$(nw)",
        lowerbound = 0,
        upperbound = ref(pm,nw,:storage,i,"charge_rating"))
    
    var(pm,nw)[:ud] = @variable(pm.model,[i in ids(pm,nw,:storage)], basename="ud_$(nw)",
        lowerbound = 0,
        upperbound = ref(pm,nw,:storage,i,"discharge_rating"))  
    
    var(pm,nw)[:soc] = @variable(pm.model,[i in ids(pm,nw,:storage)], basename="soc_$(nw)",
        lowerbound = 0,
        upperbound = ref(pm,nw,:storage,i,"energy_rating"))    
end

"""
Set upper bounds for the branch flow variables such as power flow, current flow
If I_max is not a variable, get upperbounds from power rating Rate A in ref(pm), transform power rating into current rating

### Arguments
- `pm:: GenericPowerModel`, containing an JuMP model with the variables for the Branch Flow Model
- `maxexp` maximal allowed line expansion
- `br::Array` List of branch numbers which are candidates for line expansion, optional, Default = [ ]

"""
function set_ub_flows(pm,maxexp,br_list::Array=[])
    #@warn("set upperbounds on flow variables")
    if isempty(br_list)
        br_list = [i for i in 1:length(ids(pm,:branch))]
    end
    
    for (i,br) in ref(pm,:branch)
        f_bus = br["f_bus"]
        t_bus = br["t_bus"]
        idx = (i,f_bus,t_bus)
        ub_voltage = ref(pm,:bus,f_bus)["vmax"]
        lb_voltage = ref(pm,:bus,f_bus)["vmin"]
        if :I_max in keys(var(pm))
            if i in br_list
                ub_current_rating = maxexp*getlowerbound(var(pm,:I_max,i))
                ub_power_rating = ub_voltage*ub_current_rating
            else
                ub_current_rating = getlowerbound(var(pm,:I_max,i))
                ub_power_rating = ub_voltage*ub_current_rating
            end
            setupperbound(var(pm,:I_max,i),ub_current_rating)
        else
            @warn("variable maximal current rating I_max is not defined, data from ref(pm) is used for rate_a")
            if i in br_list
                ub_power_rating = maxexp * ref(pm,:branch,i)["rate_a"]
                ub_current_rating = ub_power_rating * lb_voltage
            else
                ub_power_rating = ref(pm,:branch,i)["rate_a"]
                ub_current_rating = ref(pm,:branch,i)["rate_a"] * lb_voltage
            end
        end
        
        for (nw,network) in nws(pm)
            setupperbound(var(pm,nw,:cm,i),ub_current_rating^2)
            setupperbound(var(pm,nw,:p,idx),ub_power_rating)
            setlowerbound(var(pm,nw,:p,idx),-ub_power_rating)
            setupperbound(var(pm,nw,:q,idx),ub_power_rating)
            setlowerbound(var(pm,nw,:q,idx),-ub_power_rating)
        end
    end
end