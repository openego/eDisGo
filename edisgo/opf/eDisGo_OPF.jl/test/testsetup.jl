# Common preamble for every test file of the eDisGo_OPF test suite.
#
# Each test file starts with
#
#     include(joinpath(@__DIR__, "..", "testsetup.jl"))
#
# (drop the "..") for files directly in `test/`), which makes the file runnable on
# its own — `julia --project=. test/core/test_variables.jl` — as well as through
# `test/runtests.jl`. The include is idempotent: TestUtils is only defined once per
# session.

using Test
using eDisGo_OPF
using PowerModels
using InfrastructureModels
using JuMP
using JSON
using Ipopt

if !isdefined(Main, :TestUtils)
    include(joinpath(@__DIR__, "testutils", "TestUtils.jl"))
end
using .TestUtils

TestUtils.silence_powermodels!()
