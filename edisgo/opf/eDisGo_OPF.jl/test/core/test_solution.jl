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

"""
    solution_soc_result(data, gaps; pf=0.0, qf=0.0)

Build a `result` dict of the shape `check_SOC_equality` reads, in which branch
`b` of time step `t` has the SOC gap `gaps[t][b]`.

`check_SOC_equality` evaluates `pf² + qf² - ccm·w`, so with `w = 1` on every
from-bus the requested gap is obtained by setting `ccm = pf² + qf² - gap`. Only
the fields the function actually reads are filled in; `pt`/`qt` and the to-bus
voltage are added as decoys, since the check is a purely from-side one.
"""
function solution_soc_result(data::Dict{String,<:Any}, gaps; pf::Float64=0.0,
                             qf::Float64=0.0)
    nw = Dict{String,Any}()
    for (t, branch_gaps) in gaps
        branches = Dict{String,Any}()
        buses = Dict{String,Any}()
        for (b, gap) in branch_gaps
            f_bus = string(data["branch"][b]["f_bus"])
            t_bus = string(data["branch"][b]["t_bus"])
            buses[f_bus] = Dict{String,Any}("w" => 1.0)
            get!(buses, t_bus, Dict{String,Any}("w" => 1e6))
            branches[b] = Dict{String,Any}("pf" => pf, "qf" => qf,
                                           "pt" => -pf - 1e6, "qt" => -qf - 1e6,
                                           "ccm" => pf^2 + qf^2 - gap)
        end
        nw[t] = Dict{String,Any}("branch" => branches, "bus" => buses)
    end
    return Dict{String,Any}("solution" => Dict{String,Any}("nw" => nw))
end

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

@testset "check_SOC_equality" begin
    data = TestUtils.small_network()
    # Branches 1 and 2 are physical lines, branch 3 is the virtual storage branch.
    @test [data["branch"][b]["storage"] for b in ("1", "2", "3")] ==
          [false, false, true]

    # A tight solution: every gap is exactly zero.
    tight = solution_soc_result(data, Dict("1" => Dict("1" => 0.0, "2" => 0.0),
                                           "2" => Dict("1" => 0.0, "2" => 0.0)))
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(tight, data)
    @test soc_tight === true
    @test sort(collect(keys(soc_eq_dict))) == ["1", "2"]
    @test all(isempty(v) for v in values(soc_eq_dict))

    # The same, with non-zero flows: `pf² + qf²` is what is compared, so
    # 0.6² + 0.8² = 1.0 against `ccm·w` = 1.0 is tight as well.
    tight_flow = solution_soc_result(data, Dict("1" => Dict("1" => 0.0, "2" => 0.0));
                                     pf=0.6, qf=0.8)
    @test tight_flow["solution"]["nw"]["1"]["branch"]["1"]["ccm"] == 1.0
    @test eDisGo_OPF.check_SOC_equality(tight_flow, data)[1] === true

    # A violated solution: only branch 2 of time step 2 is loose, and only that
    # one is reported.
    loose = solution_soc_result(data, Dict("1" => Dict("1" => 0.0, "2" => 0.0),
                                           "2" => Dict("1" => 0.0, "2" => -0.5)))
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(loose, data)
    @test soc_tight === false
    @test isempty(soc_eq_dict["1"])
    @test collect(keys(soc_eq_dict["2"])) == ["2"]
    @test soc_eq_dict["2"]["2"] ≈ -0.5 atol = 1e-12
    # A single loose branch anywhere makes the whole result non-tight.
    @test eDisGo_OPF.check_SOC_equality(
        solution_soc_result(data, Dict("1" => Dict("1" => -0.5, "2" => 0.0),
                                       "2" => Dict("1" => 0.0, "2" => 0.0))),
        data)[1] === false

    # Every loose branch of a time step is listed, not just the first.
    both = solution_soc_result(data, Dict("1" => Dict("1" => -0.3, "2" => -0.7)))
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(both, data)
    @test soc_tight === false
    @test sort(collect(keys(soc_eq_dict["1"]))) == ["1", "2"]
    @test soc_eq_dict["1"]["1"] ≈ -0.3 atol = 1e-12
    @test soc_eq_dict["1"]["2"] ≈ -0.7 atol = 1e-12

    # Virtual storage branches are skipped: branch 3 is wildly loose and is
    # neither reported nor allowed to spoil the verdict.
    with_storage = solution_soc_result(
        data, Dict("1" => Dict("1" => 0.0, "2" => 0.0, "3" => -5.0)))
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(with_storage, data)
    @test soc_tight === true
    @test isempty(soc_eq_dict["1"])
    # The converse: clear the flag on the same branch and it is checked.
    physical = deepcopy(data)
    physical["branch"]["3"]["storage"] = false
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(with_storage, physical)
    @test soc_tight === false
    @test collect(keys(soc_eq_dict["1"])) == ["3"]
    # Storage branches are skipped before they are looked up, so the result does
    # not even have to contain them.
    without = solution_soc_result(data, Dict("1" => Dict("1" => 0.0, "2" => 0.0)))
    @test !haskey(without["solution"]["nw"]["1"]["branch"], "3")
    @test eDisGo_OPF.check_SOC_equality(without, data)[1] === true

    # The threshold is a strict `< -1e-1`: a gap of exactly -0.1 still counts as
    # tight, the next representable value below it does not.
    boundary = solution_soc_result(data, Dict("1" => Dict("1" => -1e-1, "2" => 0.0)))
    @test boundary["solution"]["nw"]["1"]["branch"]["1"]["ccm"] == 1e-1
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(boundary, data)
    @test soc_tight === true
    @test isempty(soc_eq_dict["1"])

    below = solution_soc_result(data,
                                Dict("1" => Dict("1" => prevfloat(-1e-1), "2" => 0.0)))
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(below, data)
    @test soc_tight === false
    @test collect(keys(soc_eq_dict["1"])) == ["1"]
    # Tolerances just inside and outside the band, for good measure.
    @test eDisGo_OPF.check_SOC_equality(
        solution_soc_result(data, Dict("1" => Dict("1" => -0.09, "2" => 0.0))),
        data)[1] === true
    @test eDisGo_OPF.check_SOC_equality(
        solution_soc_result(data, Dict("1" => Dict("1" => -0.11, "2" => 0.0))),
        data)[1] === false

    # Only *negative* gaps are flagged. A positive one means the point violates
    # the cone rather than relaxing it, which this check deliberately ignores.
    violating = solution_soc_result(data, Dict("1" => Dict("1" => 0.5, "2" => 0.0));
                                    pf=0.6, qf=0.8)
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(violating, data)
    @test soc_tight === true
    @test isempty(soc_eq_dict["1"])

    # Degenerate inputs: no time steps at all, and a grid whose branches are all
    # virtual, both come out tight with nothing to report.
    empty_result = solution_soc_result(data, Dict{String,Dict{String,Float64}}())
    @test eDisGo_OPF.check_SOC_equality(empty_result, data) == (true, Dict())

    all_virtual = deepcopy(data)
    for b in keys(all_virtual["branch"])
        all_virtual["branch"][b]["storage"] = true
    end
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(tight, all_virtual)
    @test soc_tight === true
    @test all(isempty(v) for v in values(soc_eq_dict))

    # The from-bus voltage is the one that matters: the decoy `w` of 1e6 on the
    # to-buses would swamp every gap if the wrong end were used.
    @test tight["solution"]["nw"]["1"]["bus"]["2"]["w"] == 1.0   # from bus of branch 2
    @test tight["solution"]["nw"]["1"]["bus"]["3"]["w"] == 1e6   # its to bus

    # The real eDisGo test grid: 142 branches, one of them virtual, so 141 are
    # checked and the single loose one among them is found.
    real = TestUtils.pm_network()
    physical_branches = [b for (b, br) in real["branch"] if !br["storage"]]
    @test length(real["branch"]) == 142
    @test length(physical_branches) == 141
    gaps = Dict("1" => Dict(b => 0.0 for b in physical_branches))
    gaps["1"][first(sort(physical_branches))] = -0.25
    real_result = solution_soc_result(real, gaps)
    soc_tight, soc_eq_dict = eDisGo_OPF.check_SOC_equality(real_result, real)
    @test soc_tight === false
    @test collect(keys(soc_eq_dict["1"])) == [first(sort(physical_branches))]

    # A missing branch, bus or field in the result is not tolerated.
    incomplete = solution_soc_result(data, Dict("1" => Dict("1" => 0.0, "2" => 0.0)))
    delete!(incomplete["solution"]["nw"]["1"]["branch"], "2")
    @test_throws KeyError eDisGo_OPF.check_SOC_equality(incomplete, data)
    no_w = solution_soc_result(data, Dict("1" => Dict("1" => 0.0, "2" => 0.0)))
    delete!(no_w["solution"]["nw"]["1"]["bus"]["1"], "w")
    @test_throws KeyError eDisGo_OPF.check_SOC_equality(no_w, data)
end
