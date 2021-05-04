using PowerModels
using Colors
using LightGraphs
using GraphPlot
using Compose

function plot_pm(pm,label=true)
    
    ub_Imax = Dict()
    for (i, b) in ref(pm)[:branch]
        ub_Imax[i] = ((b["rate_a"]*b["tap"])/(ref(pm)[:bus][b["f_bus"]]["vmin"]))
    end
    if haskey(var(pm),:I_max)
        varVal = getvalue(var(pm,:I_max,1))
        solved=true
        if isnan(varVal)
            solved = false
        end
    else
        solved=false
    end
    n_vertices = length(ref(pm,:bus))
    n_branches = length(ref(pm,:branch))
    G = Graph(n_vertices)
    membership = create_node_membership(pm)
    #membership = []
    #for i in 1:n_vertices
    #    bus = ref(pm,:bus,i)
    #    if bus["bus_type"] == 3 || bus["bus_type"]==2
    #        bus_type = bus["bus_type"]
    #    elseif length(ref(pm, :bus_loads, i)) > 0
    #        bus_type = 4
    #    else
    #        bus_type = 1
    #    end
    #    push!(membership, bus_type)
    #end
    # nodecolor: bus, generator, reference bus, load
    #println("the nodecolor is defined as
    #bus: $(:lightseagreen)
    #bus with gen: $(:lightblue)
    #ref bus: $(:yellow)
    #bus with load: $(:grey)\n")
    nodecolor = [colorant"lightseagreen", colorant"lightblue",colorant"yellow",colorant"grey"]
    # membership color
    nodefillc = nodecolor[membership]
    
    cmVal = Int[]
    label_edge = Int[]
    for i in 1:n_branches
        branch = ref(pm,:branch,i)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        add_edge!(G,f_bus,t_bus) 
    end
    # iterate over all edges, set labels and color category
    bus_keys = keys(ref(pm,:buspairs))
    for (e_idx, e) in enumerate(edges(G))
        pair = (src(e),dst(e)) in bus_keys ? (src(e),dst(e)) : (dst(e),src(e))
        br = ref(pm,:buspairs)[pair]["branch"]
        push!(label_edge,br)
        if haskey(var(pm),:I_max)       
            if solved
                cmi = ceil(Int,round(getvalue(var(pm,:I_max,br))/ub_Imax[br]*100)/100)#getlowerbound(var(pm,:I_max,br))
                if cmi >= 4
                    cmi = 5
                end
            else
                    if getlowerbound(var(pm,:I_max,br)) == getupperbound(var(pm,:I_max,br))
                        cmi = ceil(Int,round(getlowerbound(var(pm,:I_max,br))/ub_Imax[br]*1000)/1000)
                    else
                        cmi = 5
                    end   
            end
            push!(cmVal,cmi) 
        end
    end
    # edgecolor: no expansion, 2x, 3x, 4x, edge is a free variable
    #println("the edge color describes nature of expansion as
    #no expansion: $(:black)
    #2x expansion: $(:green)
    #3x expansion: $(:blue)
    #4x expansion: $(:orange)
    #free expansion variable/ >4x expansion: $(:red)
    #if no optimization model: $(:grey)")
    edgecolor = [colorant"black", colorant"green",colorant"blue",colorant"orange",colorant"red"]
    if !isempty(cmVal)
        if all(cmVal.==cmVal[1])
            edgestrokec = edgecolor[cmVal[1]]
        else
            edgestrokec =edgecolor[cmVal]
        end
    else
        # set color to grey for all edges if opt model is not set up
        edgestrokec = colorant"grey"
    end
    
    xcor = zeros(n_vertices)
    ycor = zeros(n_vertices)
    visit_nodes = []
    xcor[1] = 0
    ycor[1] = 0
    # fill xcor and ycor with default values for a tree layout
    coor_next_node(G,xcor,ycor,1,1,visit_nodes,n_vertices)
    # correct y coordinate of all buses except the first one
    ycor[2:end].-=1
    if label
        gplt = gplot(G,xcor,ycor,nodelabel=1:n_vertices,nodefillc=nodefillc,edgelabel=label_edge,edgestrokec=edgestrokec)
    else
        gplt = gplot(G,xcor,ycor,nodefillc=nodefillc,edgestrokec=edgestrokec)
    end
    return G,gplt,xcor,ycor,nodefillc,label_edge,edgestrokec,cmVal
end

function coor_next_node(G,xcor,ycor,current_node,last_node,visit_nodes,n_vertices)
    if length(visit_nodes)==n_vertices
        return
    elseif current_node in visit_nodes
        return     
    else
        push!(visit_nodes,current_node)

        current_y = ycor[current_node]
        current_x = xcor[current_node]
        for next_node in G.fadjlist[current_node]
            if next_node in visit_nodes
                continue
            end
            current_x +=1
            xcor[next_node] = current_x   
            while length(findall((xcor.==current_x).&(ycor.==current_y)))>=1
                current_y +=1
            end
            ycor[next_node] = current_y
            
            coor_next_node(G,xcor,ycor,next_node,current_node,visit_nodes,n_vertices)
            current_x = xcor[last_node] + 1
            current_y +=1    
        end
   
    end
end
function create_node_membership(pm)
    membership = []
    n_vertices = length(ref(pm,:bus))
    for i in 1:n_vertices
        bus = ref(pm,:bus,i)
        if bus["bus_type"] == 3 || bus["bus_type"]==2
            bus_type = bus["bus_type"]
        elseif length(ref(pm, :bus_loads, i)) > 0
            bus_type = 4
        else
            bus_type = 1
        end
        push!(membership, bus_type)
    end
    #names = ["bus","gen","slack","load"]
    #nodelabs = names[membership]
    return membership#nodelabs
end

function create_node_membership(pm,fixGen::Bool)
    membership = []
    n_vertices = length(ref(pm,:bus))
    for i in 1:n_vertices
        bus = ref(pm,:bus,i)
        if bus["bus_type"] == 3 || bus["bus_type"]==2
            g = ref(pm,:bus_gens,i)[1]
            if getlowerbound(var(pm,pm.cnw,:pg,g))==getupperbound(var(pm,pm.cnw,:pg,g))
                bus_type = 5
            else
                bus_type = bus["bus_type"]
            end    
        elseif length(ref(pm, :bus_loads, i)) > 0
            bus_type = 4
        else
            bus_type = 1
        end
        push!(membership, bus_type)
    end
    #names = ["bus","gen","slack","load"]
    #nodelabs = names[membership]
    return membership#nodelabs
end


function create_legend(G,xcor,ycor,nodefillc)
    G_copy = copy(G)
    xcor_copy = copy(xcor)
    ycor_copy = copy(ycor)
    nodefillc_copy = copy(nodefillc)
    n_vertices = length(xcor_copy)
    ycor_max = maximum(ycor_copy)
    # add four vertices for legend nodes
    add_vertices!(G_copy,4)
    # append coordinates for the new nodes to xcor and ycor
    append!(xcor_copy,[0,0,0,0])
    append!(ycor_copy,[ycor_max-3,ycor_max-2,ycor_max-1,ycor_max])
    # append associated color to nodefillc
    append!(nodefillc_copy,[colorant"lightseagreen", colorant"lightblue",colorant"yellow",colorant"grey"])
    # create new nodelabels, empty for nodes of graph and label name for legend nodes
    nodelab=[]
    for i in 1:n_vertices
        append!(nodelab,[""])
    end
    append!(nodelab,["bus","gen","slack","load"])
    return G_copy,xcor_copy,ycor_copy,nodefillc_copy,nodelab
end

function create_legend(G,xcor,ycor,fixGen::Bool,membership)

    nodecolor = [colorant"lightseagreen", colorant"lightblue",colorant"yellow",colorant"grey",colorant"darkblue"]
    nodefillc = nodecolor[membership]


    G_copy = copy(G)
    xcor_copy = copy(xcor)
    ycor_copy = copy(ycor)
    nodefillc_copy = copy(nodefillc)
    n_vertices = length(xcor_copy)
    ycor_max = maximum(ycor_copy)
    # add four vertices for legend nodes
    add_vertices!(G_copy,5)
    # append coordinates for the new nodes to xcor and ycor
    append!(xcor_copy,[0,0,0,0,0])
    append!(ycor_copy,[ycor_max-4,ycor_max-3,ycor_max-2,ycor_max-1,ycor_max])
    # append associated color to nodefillc
    append!(nodefillc_copy,[colorant"lightseagreen", colorant"lightblue",colorant"yellow",colorant"grey",colorant"darkblue"])
    # create new nodelabels, empty for nodes of graph and label name for legend nodes
    nodelab=[]
    for i in 1:n_vertices
        append!(nodelab,[""])
    end
    append!(nodelab,["bus","gen","slack","load","fixGen"])
    return G_copy,xcor_copy,ycor_copy,nodefillc_copy,nodelab
end

