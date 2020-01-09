"""
variables for charge and discharging rate

optional argument: `bounded` DEFAULT: `true`, if `false` no upper bounds
- sym: `:uc`, `:ud`
- ``0\\leq u_c\\leq \\overline{u_c}``: charging rate
- ``0\\leq u_d\\leq \\overline{u_c}``: discharge_rating
"""
function add_var_charging_rate(pm;nw::Int=pm.cnw,cnd::Int=pm.ccnd,bounded=true)
    if bounded
        var(pm,nw)[:uc] = @variable(pm.model,[i in ids(pm,nw,:storage)],basename="uc_$(nw)",
            lowerbound = 0,
            upperbound = ref(pm,nw,:storage,i,"charge_rating"))

        var(pm,nw)[:ud] = @variable(pm.model,[i in ids(pm,nw,:storage)], basename="ud_$(nw)",
            lowerbound = 0,
            upperbound = ref(pm,nw,:storage,i,"discharge_rating"))  
    else
        var(pm,nw)[:uc] = @variable(pm.model,[i in ids(pm,nw,:storage)],basename="uc_$(nw)",
            lowerbound = 0)

        var(pm,nw)[:ud] = @variable(pm.model,[i in ids(pm,nw,:storage)], basename="ud_$(nw)",
            lowerbound = 0)
    end
        
end
"""
variables for state of starge `soc`

optional argument: `bounded` DEFAULT: `true`, if `false` no upper bounds
- sym: `:soc`
- ``0\\leq soc \\leq soc^{max}``: state of charge 

"""
function add_var_soc(pm;nw::Int=pm.cnw,cnd::Int=pm.ccnd,bounded=false)
    if bounded
        var(pm,nw)[:soc] = @variable(pm.model,[i in ids(pm,nw,:storage)], basename="soc_$(nw)",
            lowerbound = 0, upperbound =ref(pm,nw,:storage,i,"energy_rating"))
    else
        var(pm,nw)[:soc] = @variable(pm.model,[i in ids(pm,nw,:storage)], basename="soc_$(nw)",
            lowerbound = 0)
    end
end

"""
variable for energy rating, i.e. maximal state-of-charge, with lower bound 0
    
- sym: `:emax`

``0\\leq \\overline{e_i}``
"""
function add_var_energy_rating(pm)
    var(pm)[:emax] = @variable(pm.model,[i in ids(pm,:storage)],basename="emax",
        lowerbound = 0)
end