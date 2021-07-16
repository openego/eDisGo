function find_next_bus(G,node_pairs,old,new,list,i,n)
"""
find surrounding nodes, starting at node "i" with path depth of "n" in a graph "G"
save edge number of path taken found in "node_pairs"["branch"] in "list"

G::SimpleGraph{Int64}, simple graph containing network with a tree structure
node_pairs::Dict{Tuple{Int64,Int64},Dict{String,Real}}, Dict containing the node pairs of a GenericPowerModel, generated with PowerModels.jl
old::Int64, last node visited
new::Int64, current node visitid
list::Array, list containing branch indices already visited
i::Int64 position of path taken
n::Int64 maximal path depth

"""
    if n==0
        return
    elseif i==n
        # maximal path depth, push latest edge to list
        pair = (old,new)
        br_nr = if haskey(node_pairs,pair)
            node_pairs[pair]["branch"]
        else node_pairs[reverse(pair)]["branch"]
            end
        if !(br_nr in list)
            push!(list,br_nr)
        end

        return 
    else       
        # increment path position i
        i+=1
        old = new
        for new in G.fadjlist[old]
            find_next_bus(G,node_pairs,old,new,list,i,n)
        end
        # decrement maximal path depth n to force back stepping after reaching end of path
        n -=1
    end
end

function find_path_to(G,s,t,br_list,bus_pairs)
    """ Find path from s to t in a tree graph G and save traversed edge in br_list

    G::SimpleGraph{Int64}, simple graph containing network with a tree structure
    s::Int64 starting node
    t::Int64 ending node
    br_list::Array, list containing branch indices already visited
    bus_pairs::Dict{Tuple{Int64,Int64},Dict{String,Real}}, Dict containing the node pairs of a GenericPowerModel, generated with PowerModels.jl
    """

    path = a_star(G,s,t)
    key_pairs = keys(bus_pairs)
    for e in path
        pair = (src(e),dst(e)) in key_pairs ? (src(e),dst(e)) : (dst(e),src(e))
        br_nr = bus_pairs[pair]["branch"]
        if !(br_nr in br_list)
            push!(br_list,br_nr)
        end
    end
end
