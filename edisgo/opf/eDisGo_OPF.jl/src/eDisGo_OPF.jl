module eDisGo_OPF

using PowerModels
using InfrastructureModels
using Memento
using JuMP
using Ipopt
using JSON
using Compat
using Gurobi

const _pm_global_keys = Set(["time_series", "per_unit"])
const pm_it_name = "pm"
const pm_it_sym = Symbol(pm_it_name)

# include functions extending PowerModels functions
include("core/types.jl")
include("core/base.jl")
include("core/constraint.jl")
include("core/constraint_template.jl")
include("core/data.jl")
include("core/objective.jl")
include("core/solution.jl")
include("core/variables.jl")
include("form/bf.jl")
include("prob/opf_bf.jl")
include("io/common.jl")
include("io/json.jl")
#include("../test/opf_test_case.jl")

# export new types of PowerModels
export BFPowerModelEdisgo, SOCBFPowerModelEdisgo, NCBFPowerModelEdisgo

end
