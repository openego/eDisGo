"""Read solution of an optimization problem from .sol file """

using CSV
using DataFrames
using JuMP
using PowerModels

function readSol(filename::String)
    @warn("There is still a bug in function readSol")
    df = CSV.read(filename,delim=" ", datarow=3,ignorerepeated=true, 
        silencewarnings=true,types=[String,Float64,String,String,String])[1:2]
    names!(df,[:vars,:val])
    df_new = DataFrame()
    df_new.vars = []
    df_new.index = []
    df_new.from_bus = []
    df_new.to_bus = []
    df_new.vals = []
    for i in 1:length(df[:vars])
        varSymbol = Symbol()
        idx = []
        f_bus = 0
        t_bus = 0
        arr = split(df[:vars][i],"_")
        try
            if length(arr) == 1
                varSymbol = Symbol(arr[1])
            elseif length(arr) == 2
                varSymbol = Symbol(arr[1])
                idx = parse(Int,arr[2])
                f_bus = NaN
                t_bus = NaN
            elseif length(arr)==3
                varString = "$(arr[1])_$(arr[2])"
                varSymbol = Symbol(varString)
                idx = parse(Int,arr[3])
                f_bus = NaN
                t_bus = NaN
            elseif length(arr)==4
                varSymbol = Symbol(arr[1])
                idx = parse(Int,arr[2])
                f_bus = parse(Int,arr[3])
                t_bus = parse(Int,arr[4])
            end
            catch err

            continue
        end
        val = df.val[i]
        push!(df_new.vars,varSymbol)
        push!(df_new.index,idx)
        push!(df_new.from_bus,f_bus)
        push!(df_new.to_bus,t_bus)
        push!(df_new.vals,val)
    end
    n_vars = findfirst(df_new.vars.==Symbol(""))-1      
    df_new = sort(df_new[1:n_vars,:])
    return df_new
end

function fix_var_values_in_pm(pm::GenericPowerModel, df::DataFrame)
    for j in 1:length(df[:vars])
        varSymbol = df.vars[j]
        if varSymbol == :quadobjvar
            quadobjvar = df.vals[j]
    
        elseif haskey(PowerModels.var(pm),varSymbol)
            i = df.index[j]
            if isnan(df.from_bus[j])
                idx = i
            else
                idx = (i,df.from_bus[j],df.to_bus[j])
            end
            val = df.vals[j]
            JuMP.fix(PowerModels.var(pm,varSymbol,idx),val)
        else
            println("variable $(varSymbol) doesnt exist in model")
        end
    end
    
    for (varSym, varVal) in var(pm)
        for idx in varVal.indexsets[1]
            val = getvalue(var(pm,varSym,idx))
            if isnan(val)
                val = 0
                JuMP.fix(PowerModels.var(pm,varSym,idx),val)
            end
        end
    end
end