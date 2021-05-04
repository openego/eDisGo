include("variables_bfm_multi.jl")
# Power flow

function add_var_ne(pm, mip=true;maxexp=10)
    #     maxexp = ref(pm,:maxexp)
    if mip
        var(pm)[:ne] = @variable(pm.model,[i in ids(pm,:branch)],Int, basename="ne",
            lowerbound = 1, upperbound=maxexp)
    else
        var(pm)[:ne] = @variable(pm.model,[i in ids(pm,:branch)], basename="ne",
            lowerbound = 1, upperbound=maxexp)
    end
end 

function add_var_ne_continuous(pm)
    max_exp = ref(pm,:maxexp)
    var(pm)[:ne] = @variable(pm.model,[i in ids(pm,:branch)], basename="ne",
        lowerbound = 1, upperbound=max_exp)
end 

function add_var_power_flow_ne(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd,bounded::Bool=false)
    """
    adds variable of active and reactive power flow for each branch to a power model
    input: 
        pm:: GenericPowerModel
    
    """
    flow_lb, flow_ub = PowerModels.calc_branch_flow_bounds(ref(pm, nw, :branch), ref(pm, nw, :bus), cnd)
    n_branches = length(ref(pm,:branch))
    var(pm, nw)[:p] = @variable(pm.model,
        [(l,i,j) in ref(pm, nw, :arcs)[1:n_branches]], basename="p_$(nw)",
        lowerbound = flow_lb[l],
        upperbound = flow_ub[l],
    #     start = PowerModels.getval(ref(pm, nw, :branch, l), "p_start", cnd)
    )


    var(pm, nw)[:q] = @variable(pm.model,
        [(l,i,j) in ref(pm, nw, :arcs)[1:n_branches]], basename="q_$(nw)",
        lowerbound = flow_lb[l],
        upperbound = flow_ub[l],
    #     start = PowerModels.getval(ref(pm, nw, :branch, l), "q_start", cnd)
    )
end


function add_var_sqr_current_magnitude_ne(pm,nw::Int=pm.cnw, cnd::Int=pm.ccnd, bounded::Bool=false)  
    """
    adds variable of squared magnitude of current for each branch to a power model, can be bounded or not
    input: 
        pm:: GenericPowerModel
    
    """
    buses = ref(pm,nw)[:bus]
    b = ref(pm,nw)[:branch]
    #maxexp = ref(pm,:maxexp)
    var(pm, nw)[:cm] = @variable(pm.model,
        [i in ids(pm, nw, :branch)], basename="cm_$(nw)",
        upperbound = ((b[i]["rate_a"][cnd]*b[i]["tap"][cnd])/(buses[b[i]["f_bus"]]["vmin"][cnd]))^2,
        lowerbound = 0)
end

function set_ub_flows_ne(pm,max_exp::Int,br_list::Array=[])
    """
        Function that sets upper bounds for the branch flow variables such as power flow, current flow
        If I_max is not a variable, get upperbounds from power rating Rate A in ref(pm), transform power rating into current rating
        
        ------Input-----
        pm: GenericPowerModel, containing an JuMP model with the variables for the Branch Flow Model
        max_exp: maximal allowed line expansion
        br: List of branch numbers which are candidates for line expansion
    
    """
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
                ub_current_rating = max_exp*getlowerbound(var(pm,:I_max,i))
                ub_power_rating = ub_voltage*ub_current_rating
            else
                ub_current_rating = getlowerbound(var(pm,:I_max,i))
                ub_power_rating = ub_voltage*ub_current_rating
            end
            setupperbound(var(pm,:I_max,i),ub_current_rating)
        else
            @warn("variable maximal current rating I_max is not defined, data from ref(pm) is used for rate_a")
            if i in br_list
                ub_power_rating = max_exp * ref(pm,:branch,i)["rate_a"]
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