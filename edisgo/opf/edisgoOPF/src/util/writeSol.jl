using JuMP
using Printf

function varname_given(m::Model,col::Integer)
    name = getname(m, col)
    for (pat, sub) in [("[", "_"), ("]", ""),(", ", "_"),(",", "_"),("(",""),(")","")]
        name = replace(name, pat => sub)
    end
    name
end
function writeSol(m::Model,filename::String, set_obj::Bool=false)
    fname = "$(filename)_solution.sol"
    f = open(fname,"w")
    @printf(f, "solution status: optimal\n")
    if set_obj
        @printf(f,"objective value: %s\n",getvalue(m.obj))   
    else
        @printf(f,"objective value: \n")#%s\n",getvalue(m.obj))
    end

    for i in 1:length(m.colVal)
        @printf(f,"%s %s \n",varname_given(m,i),m.colVal[i])
    end
    if !isempty(m.obj.qvars1)
        quadobjvar = 0
        for i in 1:length(m.obj.qvars1)
            quadobjvar +=m.obj.qcoeffs[i]*getvalue(m.obj.qvars1)[i]^2
        end
        #quadobjvar = quadobjvar/2
        @printf(f,"quadobjvar %s \n",quadobjvar)
    end
    close(f)
end