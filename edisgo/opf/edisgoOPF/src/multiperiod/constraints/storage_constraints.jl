"""
constraint for total storage capacity in system

``\\sum_{\\forall i \\in S}  \\overline{e_{i}} = \\boldsymbol{E_{Sys}}``
"""
function constraint_total_storage_capacity(pm;equality::Bool=true)
    if haskey(pm.data,"total_storage_capacity")
        e_total = pm.data["total_storage_capacity"]
    else
        println("network data should specify total storage capacity in system, using 0.0 as a default")
        e_total = 0.0
    end
    emax = var(pm,:emax)
    if equality
        @constraint(pm.model,sum(emax)==e_total)
    else
        @constraint(pm.model,sum(emax)<=e_total)
    end
end

"""
upper bound constraint for state of charge if capacity of storages is variable

- ``soc_i \\leq \\overline{e_i}``
"""
function constraint_energy_rating(pm,i;nw::Int=pm.cnw,cnd::Int=pm.ccnd)
    soc_nw = var(pm,nw,:soc,i)
    emax = var(pm,:emax,i)
    @constraint(pm.model,soc_nw <= emax)       
end
"""
upper bound constraint on charging and discharging rate with C-rate = 1
- ``uc_i \\leq \\frac{\\overline{e_i}}{T_s}``
- ``ud_i \\leq \\frac{\\overline{e_i}}{T_s}``

and relaxed complementary constraint
- ``uc_i + ud_i \\leq \\frac{\\overline{e_i}}{T_s}``
"""
function constraint_charge_rating(pm,i;nw::Int=pm.cnw,cnd::Int=pm.ccnd)
    Ts = pm.data["time_elapsed"]
    uc_nw = var(pm,nw,:uc,i)
    ud_nw = var(pm,nw,:ud,i)
    emax = var(pm,:emax,i)
    # upper bounds of charging and discarging rate with a C-rate of 1
    @constraint(pm.model, uc_nw <= emax/Ts)
    @constraint(pm.model, ud_nw <= emax/Ts)
    # relaxed complementary constraint with a C-rate of 1
    @constraint(pm.model, uc_nw + ud_nw <= emax/Ts)
end
"""
State of charge constraint, temporal coupling
- `` \\begin{equation}T_s \\left( \\eta_{c,i} u_{c,i}^t - \\frac{u_{d,i}^t}{\\eta_{d,i}}\\right)=e_i^{t+1} - e_i^t \\end{equation}``

If no charge/discarge efficiency is given in data, set efficiency to :
- `` \\eta_{c,i} = \\eta_{d,i} = 1.0``
"""
function constraint_soc(pm,i,nw_1::Int,nw_2::Int)
    if haskey(pm.data,"time_elapsed")
        Ts = pm.data["time_elapsed"]
    else
        println("network data should specify time_elapsed, using 1.0 as a default")
        Ts = 1.0
    end
    uc_1 = var(pm,nw_1,:uc,i)
    ud_1 = var(pm,nw_1,:ud,i)
    soc_nw_2 = var(pm,nw_2,:soc,i)
    soc_nw_1 = var(pm,nw_1,:soc,i)
    # if no charge or discarge efficiency is given in data, Default: 1.0
    eta_c = haskey(ref(pm,nw_2,:storage,i),"charge_efficiency") ? ref(pm,nw_2,:storage,i,"charge_efficiency") : 1.0
    eta_d = haskey(ref(pm,nw_2,:storage,i),"discharge_efficiency") ? ref(pm,nw_2,:storage,i,"discharge_efficiency") : 1.0
    #eta_c = ref(pm,nw_2,:storage,i,"charge_efficiency")
    #eta_d = ref(pm,nw_2,:storage,i,"discharge_efficiency")
    
    @constraint(pm.model, soc_nw_2 - soc_nw_1 == Ts*(eta_c * uc_1 - ud_1/eta_d))
end

"""
complementary constraint, cite Marley
- ``uc_i ud_i  = 0``
"""
function constraint_complementary(pm,i;nw::Int=pm.cnw,cnd::Int=pm.ccnd)
    uc = var(pm,nw,:uc,i)
    ud = var(pm,nw,:ud,i)
    @constraint(pm.model,uc*ud==0)
end
"""
relaxed complementary constraint, cite Marley
- ``u_{c,i}^t \\leq -\\left(\\frac{R^{max}_{c,i}}{R^{max}_{d,i}}\\right)u_{d,i}^t + R^{max}_{c,i}``
"""
function constraint_complementary_approx(pm,i;nw::Int=pm.cnw,cnd::Int=pm.ccnd)
    Rmax_discharging = ref(pm,nw,:storage,i,"discharge_rating")
    Rmax_charging = ref(pm,nw,:storage,i,"charge_rating")
    uc = var(pm,nw,:uc,i)
    ud = var(pm,nw,:ud,i)
    @constraint(pm.model,uc <= -(Rmax_charging/Rmax_discharging)*ud + Rmax_charging)
end