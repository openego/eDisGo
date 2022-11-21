module edisgoOPF

using PowerModels
using JuMP
using Ipopt
using JSON
using Compat

# include util files for reader edisgo data and write opf solution
include("edisgo_util/read_edisgo_data.jl")
include("edisgo_util/write_opf_solution.jl")
#include("edisgo_util/add_storage_to_network.jl")

# include variables, constraints and objectives
include("multiperiod/variables_bfm_multi.jl")
include("multiperiod/constraint_bfm_multi.jl")
include("multiperiod/objective_bfm_multi.jl")


# include implemented post_methods
include("post_methods/post_opf_bf_nep_cmip.jl")
include("post_methods/post_opf_bf_nep_mip.jl")
include("post_methods/post_opf_bf_nep_relaxation.jl")
include("post_methods/post_opf_bf_nep.jl")
include("post_methods/post_opf_bf_strg_nep.jl")
# include problem formulation
include("prob/run_edisgo_opf_problem.jl")
# util functions
include("util/jump_extension/writeLP.jl")
# still missing: graph operators, plotting, gan conditions
# ...

include("../test/opf_test_case.jl")

greet() = print("Hello World!")

end # module
