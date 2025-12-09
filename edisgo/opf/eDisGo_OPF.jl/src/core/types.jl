""

abstract type AbstractBFModelEdisgo <: AbstractBFQPModel end

"""
Radial branch flow model (eDisGo implementation)
Applicable to problem formulations with `_bf` in the name.
"""
mutable struct BFPowerModelEdisgo <: AbstractBFModelEdisgo @pm_fields end

abstract type AbstractSOCBFModelEdisgo <: AbstractBFModelEdisgo end

"""
Second-order cone relaxation of radial branch flow model (eDisGo implementation).
Applicable to problem formulations with `_bf` in the name.
"""
mutable struct SOCBFPowerModelEdisgo <: AbstractSOCBFModelEdisgo @pm_fields end


abstract type AbstractNCBFModelEdisgo <: AbstractBFModelEdisgo end

"""
Non convex radial branch flow model (eDisGo implementation).
Applicable to problem formulations with `_bf` in the name.
"""
mutable struct NCBFPowerModelEdisgo <: AbstractNCBFModelEdisgo @pm_fields end
