using PowerModels
using JuMP
using LightGraphs
using LinearAlgebra
"""
Calculate matrix A_i = A_ij as described in Gan et. al "Exact Convex Relaxation of Optimal Power Flow
in Radial Networks", 2013

input node i, a powermodel pm, and the powermodel as tree graph T,
and bool para. if para=true, initial parameter values for resistance and reactance are taken for calculating A_i
if para=false, optimized values of resistance and reactance are taken  for calculating A_i

returns A_i, u_i, S_i = [P_ij, Q_ij], v_min, comp: node labels of injections

"""
function calculate_A_i(i,pm,T,para)
    eye = Matrix{Float64}(I, 2, 2)
    Tcopy = copy(T)
    j = Tcopy.badjlist[i][1]
    rem_edge!(Tcopy,j,i)
    c1,c2 = connected_components(Tcopy)
    comp = i in c1 ? c1 : c2
    pg,pc,qg,qc = 0,0,0,0
    # sum upperbounds of power injection of all nodes downstream of node i
    for b in comp
        load_nr = isempty(ref(pm,:bus_loads)[b]) ? [] : ref(pm,:bus_loads)[b]
        gen_nr = isempty(ref(pm,:bus_gens)[b]) ? [] : ref(pm,:bus_gens)[b]
        for load in load_nr
            pc += ref(pm,:load)[load]["pd"]
            qc += ref(pm,:load)[load]["qd"]
        end
        for gen in gen_nr
            pg += getupperbound(var(pm,1,:pg,gen))#ref(pm,:gen)[gen]["pmax"]
            qg += getupperbound(var(pm,1,:qg,gen))#ref(pm,:gen)[gen]["qmax"]
        end
    end
    P̂ = max(pg-pc,0)
    Q̂ = max(qg-qc,0)
    br_nr = (i,j) in keys(ref(pm,:buspairs)) ? ref(pm,:buspairs,(i,j))["branch"] : ref(pm,:buspairs,(j,i))["branch"]
    # get resistance and reactance values of upstream branch from bus i, either initial parameter values or
    # solution value from optimization
    if para
        r_ij = ref(pm,:branch,br_nr)["br_r"]
        x_ij = ref(pm,:branch,br_nr)["br_x"]
    else
        r_ij = getvalue(var(pm,:r,br_nr))  
        x_ij = getvalue(var(pm,:x,br_nr))
    end
    # get lowerbound of voltage at bus i
    v_min = ref(pm,:bus,i)["vmin"]
    
    # calculate A_i
    u_i = [r_ij; x_ij]
    S_i = [P̂ Q̂]
    A_i = eye - 2/v_min*(u_i*S_i)
    return A_i,u_i,S_i,v_min,comp
end     

"""
calculate condition C1 in Theorem 1 in Gan et. al "Exact Convex Relaxation of Optimal Power Flow
in Radial Networks", 2013

### Input: 
- `G`: a graph representing a powermodel
- `pm`: generic powermodel
- `para`: bool -  inital values for resistance/reactance are used, Default: true

C1 is calculated for any node in powermodel, all violations, i.e.
    A_{ls}...A_{lt-1}u_{t}<=0, are stored in a Dict "violations" 

returns the violations dict
"""
function check_condition_C1(G,pm,para::Bool=true)
    violations = Dict()
    slack = 1
    T = bfs_tree(G,slack)
    for l in 2:length(ref(pm,:bus))
        bus_list = Int[]
        path2leaf = a_star(T,1,l)[2:end]
        path2slack = reverse(path2leaf)
        A_i = Matrix{Float64}(I, 2, 2)
        _,u_l = calculate_A_i(l,pm,T,para)
        for e in path2slack
            push!(bus_list,src(e))
            A_k,u_k = calculate_A_i(dst(e),pm,T,para)
            # A_g,u_g = calc_A_i(dst(e),pm,T,para)
            A_i = A_k*A_i 
            res = A_i*u_l
            if any(res.<=0)
                println("t = $(l)")
                println(e)
                println("A$(reverse(bus_list)) u_$(l) = $(res) <= 0")
                violations[e] =res
            end
        end
    end
    return violations
end