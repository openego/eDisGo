module eDisGo_OPF

using PowerModels
using JuMP
using Ipopt
using JSON
using Compat

# include functions extending PowerModels core functions
include("core/types.jl")
include("core/base.jl")
include("core/constraint.jl")
include("core/constraint_template.jl")
include("core/data.jl")
include("core/objective.jl")
include("core/solution.jl")
include("core/variables.jl")

# include functions extending PowerModels from functions
include("form/bf.jl")

# include functions extending PowerModels prob functions
include("prob/opf.jl")
include("prob/opf_bf.jl")
#include("../test/opf_test_case.jl")

export BFPowerModelEdisgo, SOCBFPowerModelEdisgo, NCBFPowerModelEdisgo

end # module
