"""
constraint for curtailment requirements for a single timestep `t` 
for collection of fluctuating generetors `RES`

- ``\\sum_{\\forall i \\in RES}\\left(\\boldsymbol{p_{i,g}}^t-p_{i,g}^t\\right)= \\boldsymbol{P^t_{curtail}}``

### Arguments

- `pm::GenericPowerModel{T} where T <: PowerModels.AbstractBFForm` PowerModel
- `nw::Int` timestep
"""
function constraint_curtailment_single(pm;nw::Int=pm.cnw)
    # get collection of fluctuating generators
    RES = ids(pm,:fluct_gen)
    pg = var(pm,nw,:pg)
    # make RES controllable
    for i in RES
        setlowerbound(pg[i],0)
    end
    # get requirement for curtailment in timestep nw
    # P_curtail = ref(pm,nw,:curtailment)
    #P_curtail = pm.data["curtailment_requirement_series"][nw]
    #@constraint(pm.model, sum(getupperbound(pg[i]) - pg[i] for i in RES)==P_curtail)
end

"""
constraint for acceptable curtailment over entire time horizon `T`
for collection of fluctuating generetors `RES`

- ``\\sum_{\\forall t \\in T}\\left[\\sum_{\\forall i \\in E}\\left(\\boldsymbol{p_{i,g}}^t-p_{i,g}^t\\right)\\right]\\leq \\boldsymbol{P_{curtail,tot}}``

### Arguments

- `pm::GenericPowerModel{T} where T <: PowerModels.AbstractBFForm` PowerModel
"""
function constraint_curtailment_allowed(pm)
    RES = ids(pm,:fluct_gen)
    sum_curtail = 0
    for (nw,net) in nws(pm)
        pg = var(pm,nw,:pg)
        # make RES controllable
        for i in RES
            setlowerbound(pg[i],0)
        end
        sum_curtail += sum(getupperbound(pg[i]) - pg[i] for i in RES)
    end
    P_curtail_tot = pm.data["curtailment_total"]
    @constraint(pm.model,sum_curtail<=P_curtail_tot)
    return sum_curtail
end