using PowerModels
using JuMP

function branch_flow_equality_is_violated(pm)
    violation=false
    for t_i in 1:length(nws(pm))
        for (i,br) in ref(pm,:branch)
            idx = (i,br["f_bus"],br["t_bus"])
            p = getvalue(var(pm,t_i,:p,idx))
            q = getvalue(var(pm,t_i,:q,idx))
            cm = getvalue(var(pm,t_i,:cm,i))
            v = getvalue(var(pm,t_i,:w,br["f_bus"]))
            r = getvalue(var(pm,:r,i))
            x = getvalue(var(pm,:x,i))
            I_max = getvalue(var(pm,:I_max,i))
            I_lb = getlowerbound(var(pm,:I_max,i))
            val = cm*v -(p^2 + q^2)
            if round(val,digits=5)!=0
                violation=true
                return violation
            end
        end
    end
    return violation
end

function current_limit_is_violated(pm)
    violation= false
    for t_i in 1:length(nws(pm))
        for (i,br) in ref(pm,:branch)
            cm = getvalue(var(pm,t_i,:cm,i))
            I_max = getvalue(var(pm,:I_max,i))
#             I_lb = getlowerbound(var(pm,:I_max,i))
            val = I_max^2 - cm
            if round(val,digits=5)<0
                violation=true
#                 println("violation on branch $(i) = $(val)")
#                 println
                return violation
            end
        end
    end
    return violation
end