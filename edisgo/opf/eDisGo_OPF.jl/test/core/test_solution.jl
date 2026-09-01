# Tests for `src/core/solution.jl`.
#
# The file holds the two ends of the solution pipeline:
#
# * `sol_component_value_radial` is a *solution processor*. It runs while the
#   model is being built (from `variable_branch_power_real_radial` and friends)
#   and registers arc-indexed JuMP variables under a branch index in
#   `pm.sol`, so that PowerModels can turn them into `result["solution"]` after
#   the solve. The tests therefore look at `pm.sol` directly, using a model
#   whose build method adds nothing.
# * `check_SOC_equality` runs after the solve, on the plain `result` dict. It is
#   pure arithmetic on that dict, so the tests hand it hand-built results with
#   exactly the SOC gaps they want to see reported.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

"""
An instantiated model on [`TestUtils.small_network`](@ref) whose build method adds
nothing, so `pm.sol` starts out empty and every entry a test observes was written
by the function under test.
"""
solution_bare_model() = eDisGo_OPF.instantiate_model(
    TestUtils.multinetwork(TestUtils.small_network()), SOCBFPowerModelEdisgo,
    _pm -> nothing)

"The solution entry of branch `l` in network `n` of `pm`."
solution_branch(pm, n::Int, l::Int) =
    InfrastructureModels.sol(pm, eDisGo_OPF.pm_it_sym, n, :branch, l)

@testset "sol_component_value_radial" begin
    pm = solution_bare_model()
    @test isempty(PowerModels.sol(pm, 1))

    # Arc `(l, i, j)` in, branch index `l` out: the two bus indices of the arc
    # only select the value, they do not appear in the solution structure.
    arcs = [(1, 1, 2), (2, 2, 3)]
    pf = Dict((1, 1, 2) => 1.5, (2, 2, 3) => -0.25, (3, 3, 4) => 99.0)
    @test eDisGo_OPF.sol_component_value_radial(pm, 1, :branch, :pf, arcs, pf) ===
          nothing
    @test solution_branch(pm, 1, 1)[:pf] == 1.5
    @test solution_branch(pm, 1, 2)[:pf] == -0.25
    # Only the requested arcs are written, even though `pf` knows a third one …
    @test sort(collect(keys(PowerModels.sol(pm, 1, :branch)))) == [1, 2]
    # … and only the requested network.
    @test isempty(PowerModels.sol(pm, 2))

    # A second field lands next to the first one, on the same branch entries.
    qf = Dict((1, 1, 2) => -0.5, (2, 2, 3) => 0.125)
    eDisGo_OPF.sol_component_value_radial(pm, 1, :branch, :qf, arcs, qf)
    @test sort(collect(keys(solution_branch(pm, 1, 1)))) == [:pf, :qf]
    @test solution_branch(pm, 1, 1)[:qf] == -0.5
    @test solution_branch(pm, 1, 1)[:pf] == 1.5

    # Writing the same field twice is a programming error and is caught by the
    # `@assert`; the guard is per branch, so it fires on the very first arc.
    @test_throws AssertionError eDisGo_OPF.sol_component_value_radial(
        pm, 1, :branch, :pf, arcs, pf)
    @test solution_branch(pm, 1, 1)[:pf] == 1.5

    # The same field in a different network is fine — the check is per network.
    @test eDisGo_OPF.sol_component_value_radial(pm, 2, :branch, :pf, arcs, pf) ===
          nothing
    @test solution_branch(pm, 2, 1)[:pf] == 1.5

    # Nothing about the function is branch-specific: any component name works,
    # and an empty arc list is a no-op.
    eDisGo_OPF.sol_component_value_radial(pm, 1, :storage, :ps, [(1, 4, 3)],
                                          Dict((1, 4, 3) => 0.4))
    @test InfrastructureModels.sol(pm, eDisGo_OPF.pm_it_sym, 1, :storage,
                                   1)[:ps] == 0.4
    @test eDisGo_OPF.sol_component_value_radial(pm, 1, :branch, :ll, (), Dict()) ===
          nothing
    @test !haskey(solution_branch(pm, 1, 1), :ll)

    # An arc without a value is a `KeyError` — the arc list and the variable
    # container have to agree.
    @test_throws KeyError eDisGo_OPF.sol_component_value_radial(
        pm, 1, :branch, :new, [(3, 3, 4)], Dict((1, 1, 2) => 0.0))

    # In the real model the function is fed `ref(pm, nw, :arcs_from)` and the
    # JuMP variable containers, and every branch — including the virtual storage
    # branch 3 — gets its `pf` and `qf` variable registered.
    built = TestUtils.build_small_model()
    for l in 1:3, field in (:pf, :qf)
        @test haskey(solution_branch(built, 1, l), field)
        @test solution_branch(built, 1, l)[field] isa JuMP.VariableRef
    end
    # Re-reporting a variable that the build already reported trips the assert.
    arcs_from = PowerModels.ref(built, 1, :arcs_from)
    @test_throws AssertionError eDisGo_OPF.sol_component_value_radial(
        built, 1, :branch, :pf, arcs_from, PowerModels.var(built, 1, :p))
end

