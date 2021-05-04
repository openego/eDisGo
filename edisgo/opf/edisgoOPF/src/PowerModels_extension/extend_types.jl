#using PowerModels

export 
    BFPowerModel, StandardBFForm


abstract type StandardBFForm <: PowerModels.AbstractBFForm end 

const BFPowerModel = GenericPowerModel{StandardBFForm}

"default BF constructor"
BFPowerModel(data::Dict{String,<:Any}; kwargs...) = GenericPowerModel(data, StandardBFForm; kwargs...)

