# Tests for `src/core/base.jl`.
#
# The file holds the three entry points that turn an eDisGo data dict into a
# solved optimisation model:
#
#   * `solve_model`      — instantiate, optimise, return `(result, pm)`,
#   * `instantiate_model`— build the JuMP model without solving it,
#   * `ref_add_core!`    — the reference extension both of them run: it drops the
#                          inactive components and pre-computes the lookup tables
#                          (`:arcs*`, `:bus_*`, `:ref_buses`, `:buspairs`, …) the
#                          constraint templates index into.
#
# `ref_add_core!` is tested directly on a `ref` built with
# `InfrastructureModels.ref_initialize` / `build_ref`, which is exactly the dict
# `instantiate_model` hands it, but without a JuMP model in the way.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

# Ipopt handles the non-convex formulation and is freely available; `print_level`
# 0 keeps the solver log out of the test output.
const BASE_OPTIMIZER = JuMP.optimizer_with_attributes(Ipopt.Optimizer,
                                                      "print_level" => 0)

"Sorted keys of `d` — Julia dicts are unordered, comparisons have to be."
base_sorted_keys(d) = sort(collect(keys(d)))

"""
The `ref` dict as it looks *before* `ref_add_core!` ran: the plain conversion of
`data` performed by `InfrastructureModels`, with the global keys hoisted out of
the per-network dicts. Same call `instantiate_model` makes via
`InitializeInfrastructureModel`.
"""
base_raw_ref(data) = InfrastructureModels.ref_initialize(
    data, eDisGo_OPF.pm_it_name, eDisGo_OPF._pm_global_keys)

"""
Build a full `ref` for `data` and return the reference dict of network `nw`
(`0` for single-network data, `1…n` for multi-network data).
"""
function base_nw_ref(data; nw::Int=0, ref_extensions=[])
    ref = InfrastructureModels.build_ref(data, eDisGo_OPF.ref_add_core!,
                                         eDisGo_OPF._pm_global_keys,
                                         eDisGo_OPF.pm_it_name;
                                         ref_extensions=ref_extensions)
    return ref[:it][eDisGo_OPF.pm_it_sym][:nw][nw]
end

"Trivial build method: one bounded variable and an objective, solvable in no time."
function base_trivial_build(pm)
    JuMP.@variable(pm.model, 0 <= x_trivial <= 1)
    JuMP.@objective(pm.model, Min, x_trivial)
end

"Build method with a binary variable — only solvable by Ipopt once relaxed."
function base_binary_build(pm)
    JuMP.@variable(pm.model, z_binary, Bin)
    JuMP.@objective(pm.model, Max, z_binary)
end

# Recorders for the pass-through tests. Kept at file scope (rather than as
# closures) so the call signature the caller uses is exactly the documented one.
const BASE_REF_EXT_CALLS = NamedTuple{(:nws, :data_is_mn, :core_ran),
                                      Tuple{Int,Bool,Bool}}[]
const BASE_SOL_PROC_CALLS = Any[]

"""
`ref_extensions` entry: records how it was called and stamps `:test_marker` into
every network of `ref`. `core_ran` captures whether `ref_add_core!` has already
run when the extension is invoked.
"""
function base_record_ref_ext!(ref::Dict{Symbol,Any}, data::Dict{String,<:Any})
    nw_refs = ref[:it][eDisGo_OPF.pm_it_sym][:nw]
    push!(BASE_REF_EXT_CALLS, (nws=length(nw_refs), data_is_mn=haskey(data, "nw"),
                               core_ran=all(haskey(r, :bus_cps) for r in values(nw_refs))))
    for (n, nw_ref) in nw_refs
        nw_ref[:test_marker] = 10 * n
    end
end

"""
`solution_processors` entry: records the model type it was handed and stamps a
`"test_marker"` into every network of the solution being assembled. At this point
the solution is still in `InfrastructureModels`' nested `"it"` form.
"""
function base_record_sol_proc!(pm, solution::Dict{String,Any})
    push!(BASE_SOL_PROC_CALLS, typeof(pm))
    for (n, nw_sol) in solution["it"][eDisGo_OPF.pm_it_name]["nw"]
        nw_sol["test_marker"] = 100 * parse(Int, n)
    end
end

@testset "solve_model" begin
    data = TestUtils.multinetwork(TestUtils.small_network())
    result, pm = eDisGo_OPF.solve_model(data, eDisGo_OPF.NCBFPowerModelEdisgo,
                                        BASE_OPTIMIZER,
                                        eDisGo_OPF.build_mn_opf_bf_flex;
                                        multinetwork=true)

    # Unlike PowerModels' `solve_model`, which returns the result dict only, this
    # one hands the model back as well — `Main.jl` reads variable values off it.
    @test result isa Dict{String,Any}
    @test pm isa eDisGo_OPF.NCBFPowerModelEdisgo

    @testset "result dict" begin
        @test issubset(["termination_status", "solve_time", "objective", "solution"],
                       keys(result))
        @test base_sorted_keys(result) == ["dual_status", "objective", "objective_lb",
                                           "optimizer", "primal_status", "solution",
                                           "solve_time", "termination_status"]
        # Ipopt reaches a local optimum of the non-convex model.
        @test result["termination_status"] == MOI.LOCALLY_SOLVED
        @test result["primal_status"] == MOI.FEASIBLE_POINT
        @test result["optimizer"] == "Ipopt"
        # Objective of version 2 is losses plus slack penalties: small, positive
        # and definitely a finite number.
        @test result["objective"] isa Float64
        @test isfinite(result["objective"])
        @test result["objective"] > 0
        @test result["objective"] < 1
        @test result["solve_time"] isa Float64
        @test result["solve_time"] > 0
    end

    @testset "solution" begin
        sol = result["solution"]
        @test sol["multinetwork"] == true
        @test sol["multiinfrastructure"] == false
        @test base_sorted_keys(sol["nw"]) == ["1", "2"]
        nw1 = sol["nw"]["1"]
        # One solution entry per component type of the fixture …
        @test issubset(["bus", "branch", "gen", "gen_nd", "gen_slack", "load",
                        "storage", "heatpumps", "heat_storage", "dsm",
                        "electromobility"], keys(nw1))
        # … covering all three branches, the virtual storage branch included.
        @test base_sorted_keys(nw1["branch"]) == ["1", "2", "3"]
        @test issubset(["pf", "qf", "ccm"], keys(nw1["branch"]["1"]))
        # The storage bus 4 gets no voltage variable in the radial branch-flow
        # formulation, so the bus solution covers buses 1 to 3.
        @test base_sorted_keys(nw1["bus"]) == ["1", "2", "3"]
        # Voltage magnitudes stay inside the [0.9, 1.1] band of the fixture
        # (`w` is the squared magnitude).
        @test all(0.81 - 1e-6 <= b["w"] <= 1.21 + 1e-6 for b in values(nw1["bus"]))
    end

    @testset "returned model" begin
        # `pm` is the solved model, not a fresh one.
        @test JuMP.termination_status(pm.model) == result["termination_status"]
        @test JuMP.num_variables(pm.model) > 0
        @test TestUtils.num_constraints_total(pm.model) > 0
        @test JuMP.objective_value(pm.model) ≈ result["objective"] atol = 1e-8
        # `optimize_model!` writes the solution back onto the model.
        @test pm.solution === result["solution"]
        @test pm.data === data
        @test base_sorted_keys(PowerModels.nws(pm)) == [1, 2]
    end

    @testset "model_type" begin
        # `model_type` is passed straight through to `instantiate_model`; with a
        # build method that needs no formulation-specific constraint even the base
        # type works.
        for T in (eDisGo_OPF.BFPowerModelEdisgo, eDisGo_OPF.SOCBFPowerModelEdisgo,
                  eDisGo_OPF.NCBFPowerModelEdisgo)
            res_t, pm_t = eDisGo_OPF.solve_model(data, T, BASE_OPTIMIZER,
                                                 base_trivial_build)
            @test pm_t isa T
            @test typeof(pm_t) === T
            @test res_t["termination_status"] == MOI.LOCALLY_SOLVED
            @test res_t["objective"] ≈ 0.0 atol = 1e-4
        end
    end

    @testset "ref_extensions and solution_processors" begin
        empty!(BASE_REF_EXT_CALLS)
        empty!(BASE_SOL_PROC_CALLS)
        res_x, pm_x = eDisGo_OPF.solve_model(
            data, eDisGo_OPF.NCBFPowerModelEdisgo, BASE_OPTIMIZER,
            eDisGo_OPF.build_mn_opf_bf_flex; multinetwork=true,
            ref_extensions=[base_record_ref_ext!],
            solution_processors=[base_record_sol_proc!])

        # The reference extension is called once, with the whole `ref` and the
        # data dict, and after `ref_add_core!` — so it can build on its keys.
        @test length(BASE_REF_EXT_CALLS) == 1
        @test BASE_REF_EXT_CALLS[1] == (nws=2, data_is_mn=true, core_ran=true)
        @test PowerModels.ref(pm_x, 1, :test_marker) == 10
        @test PowerModels.ref(pm_x, 2, :test_marker) == 20

        # The solution processor is called once, with the model and the solution.
        @test length(BASE_SOL_PROC_CALLS) == 1
        @test BASE_SOL_PROC_CALLS[1] === eDisGo_OPF.NCBFPowerModelEdisgo
        @test res_x["solution"]["nw"]["1"]["test_marker"] == 100
        @test res_x["solution"]["nw"]["2"]["test_marker"] == 200

        # Negative case: without the two hooks neither marker shows up.
        @test !haskey(PowerModels.ref(pm, 1), :test_marker)
        @test !haskey(result["solution"]["nw"]["1"], "test_marker")

        # Solving the same data again leaves the recorders untouched — the hooks
        # really are the ones passed in, not something registered globally.
        eDisGo_OPF.solve_model(data, eDisGo_OPF.NCBFPowerModelEdisgo,
                               BASE_OPTIMIZER, base_trivial_build)
        @test length(BASE_REF_EXT_CALLS) == 1
        @test length(BASE_SOL_PROC_CALLS) == 1
    end

    @testset "relax_integrality" begin
        # Ipopt cannot handle integrality itself, so the flag is observable: the
        # binary variable is relaxed away before the solve.
        res_r, pm_r = eDisGo_OPF.solve_model(data, eDisGo_OPF.SOCBFPowerModelEdisgo,
                                             BASE_OPTIMIZER, base_binary_build;
                                             relax_integrality=true)
        @test res_r["termination_status"] == MOI.LOCALLY_SOLVED
        @test res_r["objective"] ≈ 1.0 atol = 1e-4
        @test JuMP.num_constraints(pm_r.model, JuMP.VariableRef, MOI.ZeroOne) == 0

        # Default (`false`) keeps the binary and Ipopt rejects the model.
        @test_throws MOI.UnsupportedConstraint eDisGo_OPF.solve_model(
            data, eDisGo_OPF.SOCBFPowerModelEdisgo, BASE_OPTIMIZER,
            base_binary_build)
    end

    @testset "multinetwork and multiconductor" begin
        # BUG (deviation from PowerModels): PowerModels' `solve_model` errors when
        # these flags disagree with the data ("attempted to build a single-network
        # model with multi-network data"). This version dropped both checks, so
        # the flags are accepted and silently ignored — a wrong flag no longer
        # catches a wrongly shaped data dict.
        res_f, pm_f = eDisGo_OPF.solve_model(data, eDisGo_OPF.SOCBFPowerModelEdisgo,
                                             BASE_OPTIMIZER, base_trivial_build;
                                             multinetwork=false, multiconductor=true)
        @test res_f["termination_status"] == MOI.LOCALLY_SOLVED
        @test base_sorted_keys(PowerModels.nws(pm_f)) == [1, 2]
        @test_broken (try
            eDisGo_OPF.solve_model(data, eDisGo_OPF.SOCBFPowerModelEdisgo,
                                   BASE_OPTIMIZER, base_trivial_build;
                                   multinetwork=false)
            false
        catch
            true
        end)
    end

    @testset "kwargs pass-through" begin
        # Everything except the named keywords goes on to `instantiate_model` and
        # from there to `InitializeInfrastructureModel`.
        res_k, pm_k = eDisGo_OPF.solve_model(
            data, eDisGo_OPF.SOCBFPowerModelEdisgo, BASE_OPTIMIZER,
            base_trivial_build;
            setting=Dict{String,Any}("output" => Dict("duals" => true)),
            ext=Dict{Symbol,Any}(:answer => 42))
        @test pm_k.setting["output"]["duals"] == true
        @test pm_k.ext[:answer] == 42
        # `multinetwork` / `multiconductor` are consumed here and must *not* reach
        # `InitializeInfrastructureModel`, which would reject them.
        @test res_k["termination_status"] == MOI.LOCALLY_SOLVED
    end
end
