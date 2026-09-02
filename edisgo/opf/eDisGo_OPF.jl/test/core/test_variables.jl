# Tests for `src/core/variables.jl`.
#
# With 40 functions this is by far the largest source file of the package, so its
# tests are split into three files under `test/core/variables/`, grouped the way
# the source file itself is: the branch-flow / curtailment block, the flexibility
# variables, and the slack variables. This file only wires them together, so that
# `test/core/test_variables.jl` stays the entry point the naming convention
# expects.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

for file in ("test_flow_and_curtailment.jl", "test_flexibilities.jl",
             "test_slacks.jl")
    @testset "$file" begin
        include(joinpath(@__DIR__, "variables", file))
    end
end
