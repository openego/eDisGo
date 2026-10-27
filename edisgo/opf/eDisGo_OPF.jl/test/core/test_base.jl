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

@testset "instantiate_model" begin
    data = TestUtils.multinetwork(TestUtils.small_network())
    pm = eDisGo_OPF.instantiate_model(data, eDisGo_OPF.SOCBFPowerModelEdisgo,
                                      eDisGo_OPF.build_mn_opf_bf_flex)

    @testset "builds but does not solve" begin
        @test pm isa eDisGo_OPF.SOCBFPowerModelEdisgo
        @test pm.model isa JuMP.Model
        # The model is fully assembled …
        @test JuMP.num_variables(pm.model) > 0
        @test TestUtils.num_constraints_total(pm.model) > 0
        @test JuMP.objective_sense(pm.model) == MOI.MIN_SENSE
        # … but never handed to a solver.
        @test JuMP.termination_status(pm.model) == MOI.OPTIMIZE_NOT_CALLED
        @test JuMP.primal_status(pm.model) == MOI.NO_SOLUTION
        @test JuMP.result_count(pm.model) == 0
        @test isempty(pm.solution)

        # The build method is what fills the model: with one that adds nothing the
        # model stays empty, everything else is identical.
        pm_empty = eDisGo_OPF.instantiate_model(data, eDisGo_OPF.SOCBFPowerModelEdisgo,
                                                _pm -> nothing)
        @test JuMP.num_variables(pm_empty.model) == 0
        @test TestUtils.num_constraints_total(pm_empty.model) == 0
        @test base_sorted_keys(PowerModels.nws(pm_empty)) == [1, 2]
    end

    @testset "ref_add_core! is registered" begin
        # The eDisGo-specific lookup tables only exist because `instantiate_model`
        # hardwires `eDisGo_OPF.ref_add_core!` as the reference extension.
        for n in 1:2
            nw_ref = PowerModels.ref(pm, n)
            for key in (:bus_gens_nd, :bus_gens_slack, :bus_cps, :bus_hps, :bus_dsm,
                        :bus_storage, :bus_loads, :bus_shunts, :bus_gens,
                        :bus_arcs, :bus_arcs_to, :bus_arcs_from, :bus_lines_to,
                        :bus_arcs_dc, :bus_arcs_sw, :arcs, :arcs_from, :arcs_to,
                        :ref_buses, :buspairs, :conductor_ids)
                @test haskey(nw_ref, key)
            end
        end
        # Concrete content, not just presence: bus 3 carries the non-dispatchable
        # generator, the charging point, the heat pump and the DSM load.
        @test PowerModels.ref(pm, 1, :bus_gens_nd)[3] == [1]
        @test PowerModels.ref(pm, 1, :bus_cps)[3] == [1]
        @test PowerModels.ref(pm, 1, :bus_hps)[3] == [1]
        @test PowerModels.ref(pm, 1, :bus_dsm)[3] == [1]
        @test PowerModels.ref(pm, 1, :bus_gens_slack)[1] == [1]
        @test PowerModels.ref(pm, 1, :bus_storage)[4] == [1]

        # None of them is present before the extension runs.
        raw = base_raw_ref(data)[:it][eDisGo_OPF.pm_it_sym][:nw][1]
        for key in (:bus_gens_nd, :bus_cps, :bus_hps, :bus_dsm, :arcs, :ref_buses,
                    :buspairs, :conductor_ids)
            @test !haskey(raw, key)
        end
        # …while the raw component dicts are already keyed by Int.
        @test base_sorted_keys(raw[:bus]) == [1, 2, 3, 4]
    end

    @testset "number of networks" begin
        for steps in (1, 2, 4)
            pm_n = eDisGo_OPF.instantiate_model(
                TestUtils.multinetwork(TestUtils.small_network(num_steps=steps)),
                eDisGo_OPF.SOCBFPowerModelEdisgo, _pm -> nothing)
            @test base_sorted_keys(PowerModels.nws(pm_n)) == collect(1:steps)
            @test base_sorted_keys(pm_n.data["nw"]) == [string(i) for i in 1:steps]
            @test InfrastructureModels.ismultinetwork(pm_n, eDisGo_OPF.pm_it_sym)
            # Every network gets its own, complete reference dict.
            for n in 1:steps
                @test length(PowerModels.ref(pm_n, n, :bus)) == 4
                @test length(PowerModels.ref(pm_n, n, :branch)) == 3
            end
        end

        # Single-network data is kept as one network under the default id 0.
        pm_s = eDisGo_OPF.instantiate_model(TestUtils.small_network(),
                                            eDisGo_OPF.SOCBFPowerModelEdisgo,
                                            _pm -> nothing)
        @test base_sorted_keys(PowerModels.nws(pm_s)) == [0]
        @test !InfrastructureModels.ismultinetwork(pm_s, eDisGo_OPF.pm_it_sym)
    end

    @testset "infrastructure type and global keys" begin
        # Everything lives below the single infrastructure type `:pm`.
        @test eDisGo_OPF.pm_it_sym === :pm
        @test eDisGo_OPF.pm_it_name == "pm"
        @test collect(keys(pm.ref[:it])) == [:pm]

        # `_pm_global_keys` are hoisted out of the per-network refs.
        @test eDisGo_OPF._pm_global_keys == Set(["time_series", "per_unit"])
        pm_s = eDisGo_OPF.instantiate_model(TestUtils.small_network(),
                                            eDisGo_OPF.SOCBFPowerModelEdisgo,
                                            _pm -> nothing)
        it_ref = pm_s.ref[:it][eDisGo_OPF.pm_it_sym]
        @test it_ref[:per_unit] == true
        @test haskey(it_ref, :time_series)
        @test !haskey(it_ref[:nw][0], :per_unit)
        @test !haskey(it_ref[:nw][0], :time_series)
        # Non-global scalars stay per-network.
        @test it_ref[:nw][0][:opf_version] == 2
        @test it_ref[:nw][0][:baseMVA] == 1
    end

    @testset "ref_extensions" begin
        empty!(BASE_REF_EXT_CALLS)
        pm_x = eDisGo_OPF.instantiate_model(data, eDisGo_OPF.SOCBFPowerModelEdisgo,
                                            _pm -> nothing;
                                            ref_extensions=[base_record_ref_ext!])
        @test length(BASE_REF_EXT_CALLS) == 1
        @test BASE_REF_EXT_CALLS[1] == (nws=2, data_is_mn=true, core_ran=true)
        @test PowerModels.ref(pm_x, 1, :test_marker) == 10
        @test PowerModels.ref(pm_x, 2, :test_marker) == 20

        # Several extensions run in the given order, all of them after the core.
        order = Symbol[]
        first! = (ref, _d) -> push!(order, :first)
        second! = (ref, _d) -> push!(order, :second)
        eDisGo_OPF.instantiate_model(data, eDisGo_OPF.SOCBFPowerModelEdisgo,
                                     _pm -> nothing;
                                     ref_extensions=[first!, second!])
        @test order == [:first, :second]

        # Negative case: no extension, no marker.
        @test !haskey(PowerModels.ref(pm, 1), :test_marker)
    end

    @testset "kwargs pass-through" begin
        model = JuMP.Model()
        pm_k = eDisGo_OPF.instantiate_model(
            data, eDisGo_OPF.NCBFPowerModelEdisgo, _pm -> nothing;
            jump_model=model, setting=Dict{String,Any}("foo" => "bar"),
            ext=Dict{Symbol,Any}(:answer => 42))
        @test pm_k.model === model
        @test pm_k.setting["foo"] == "bar"
        @test pm_k.ext[:answer] == 42
        @test pm_k isa eDisGo_OPF.NCBFPowerModelEdisgo
    end
end

@testset "ref_add_core!" begin
    @testset "return value and scope" begin
        ref = base_raw_ref(TestUtils.small_network())
        # Mutates `ref` in place and returns nothing.
        @test eDisGo_OPF.ref_add_core!(ref) === nothing
        @test haskey(ref[:it][eDisGo_OPF.pm_it_sym][:nw][0], :bus_arcs)
        # Only the `:pm` branch of the ref is touched; the top level is untouched.
        @test base_sorted_keys(ref) == [:it]

        # Every network of a multi-network ref is processed.
        mn_ref = base_raw_ref(TestUtils.multinetwork(
            TestUtils.small_network(num_steps=3)))
        eDisGo_OPF.ref_add_core!(mn_ref)
        for n in 1:3
            nw_ref = mn_ref[:it][eDisGo_OPF.pm_it_sym][:nw][n]
            @test nw_ref[:bus_cps][3] == [1]
            @test length(nw_ref[:arcs]) == 6
        end
    end

    @testset "conductor_ids" begin
        nw_ref = base_nw_ref(TestUtils.small_network())
        # No `conductors` entry — the single-conductor default.
        @test nw_ref[:conductor_ids] == 1:1

        # With a `conductors` entry the range follows it. `:buspairs` is pre-set
        # here because `calc_buspair_parameters` expects per-conductor branch data,
        # which the single-conductor fixture does not have.
        data = TestUtils.small_network()
        data["conductors"] = 3
        ref = base_raw_ref(data)
        ref[:it][eDisGo_OPF.pm_it_sym][:nw][0][:buspairs] = Dict{Any,Any}()
        eDisGo_OPF.ref_add_core!(ref)
        @test ref[:it][eDisGo_OPF.pm_it_sym][:nw][0][:conductor_ids] == 1:3

        # An existing `:conductor_ids` is never overwritten (a single-element
        # range, because `calc_buspair_parameters` insists on one conductor).
        ref2 = base_raw_ref(TestUtils.small_network())
        ref2[:it][eDisGo_OPF.pm_it_sym][:nw][0][:conductor_ids] = 5:5
        eDisGo_OPF.ref_add_core!(ref2)
        @test ref2[:it][eDisGo_OPF.pm_it_sym][:nw][0][:conductor_ids] == 5:5
    end

    @testset "filtering inactive components" begin
        # Baseline: nothing is filtered out of the untouched fixture.
        nw_ref = base_nw_ref(TestUtils.small_network())
        @test base_sorted_keys(nw_ref[:bus]) == [1, 2, 3, 4]
        @test base_sorted_keys(nw_ref[:branch]) == [1, 2, 3]
        @test base_sorted_keys(nw_ref[:load]) == [1]
        @test base_sorted_keys(nw_ref[:gen]) == [1]
        @test base_sorted_keys(nw_ref[:storage]) == [1]

        # The inactive values `ref_add_core!` compares against.
        @test pm_component_status_inactive["bus"] == 4
        @test pm_component_status_inactive["load"] == 0
        @test pm_component_status_inactive["gen"] == 0
        @test pm_component_status_inactive["branch"] == 0
        @test pm_component_status_inactive["storage"] == 0

        @testset "inactive bus" begin
            # `bus_type == 4` removes the bus itself, both branches touching it and
            # the dispatchable generator sitting on it.
            data = TestUtils.small_network()
            data["bus"]["2"]["bus_type"] = 4
            r = base_nw_ref(data)
            @test base_sorted_keys(r[:bus]) == [1, 3, 4]
            @test base_sorted_keys(r[:branch]) == [3]
            @test isempty(r[:gen])
            # Components on the surviving buses stay.
            @test base_sorted_keys(r[:load]) == [1]
            @test base_sorted_keys(r[:storage]) == [1]
            # And all lookups are keyed by the surviving buses only.
            @test base_sorted_keys(r[:bus_gens]) == [1, 3, 4]
            @test r[:bus_gens] == Dict(1 => Int[], 3 => Int[], 4 => Int[])
            @test r[:arcs_from] == [(3, 3, 4)]
            @test base_sorted_keys(r[:buspairs]) == [(3, 4)]
        end

        @testset "inactive storage bus" begin
            # Bus 4 only carries the battery and the virtual storage branch.
            data = TestUtils.small_network()
            data["bus"]["4"]["bus_type"] = 4
            r = base_nw_ref(data)
            @test base_sorted_keys(r[:bus]) == [1, 2, 3]
            @test base_sorted_keys(r[:branch]) == [1, 2]
            @test isempty(r[:storage])
            @test r[:bus_storage] == Dict(1 => Int[], 2 => Int[], 3 => Int[])
        end

        @testset "component status" begin
            # Each component type is filtered by its own status field.
            data = TestUtils.small_network()
            data["load"]["1"]["status"] = 0
            data["gen"]["1"]["gen_status"] = 0
            data["storage"]["1"]["status"] = 0
            data["branch"]["2"]["br_status"] = 0
            r = base_nw_ref(data)
            @test isempty(r[:load])
            @test isempty(r[:gen])
            @test isempty(r[:storage])
            @test base_sorted_keys(r[:branch]) == [1, 3]
            # Buses are untouched by component status …
            @test base_sorted_keys(r[:bus]) == [1, 2, 3, 4]
            # … but the lookups no longer list the removed components.
            @test r[:bus_loads] == Dict(1 => Int[], 2 => Int[], 3 => Int[], 4 => Int[])
            @test r[:bus_gens][2] == Int[]
            @test r[:bus_storage][4] == Int[]
            @test sort(r[:arcs_from]) == [(1, 1, 2), (3, 3, 4)]

            # Positive counterpart: any non-inactive status keeps the component.
            data2 = TestUtils.small_network()
            data2["branch"]["2"]["br_status"] = 1.0
            data2["load"]["1"]["status"] = true
            r2 = base_nw_ref(data2)
            @test base_sorted_keys(r2[:branch]) == [1, 2, 3]
            @test base_sorted_keys(r2[:load]) == [1]
        end

        @testset "shunt, switch and dcline" begin
            data = TestUtils.small_network()
            data["shunt"]["1"] = Dict{String,Any}("index" => 1, "shunt_bus" => 3,
                                                  "status" => 1, "gs" => 0.0,
                                                  "bs" => 0.1)
            data["shunt"]["2"] = Dict{String,Any}("index" => 2, "shunt_bus" => 2,
                                                  "status" => 0, "gs" => 0.0,
                                                  "bs" => 0.1)
            data["switch"]["1"] = Dict{String,Any}("index" => 1, "f_bus" => 2,
                                                   "t_bus" => 3, "status" => 1)
            data["switch"]["2"] = Dict{String,Any}("index" => 2, "f_bus" => 1,
                                                   "t_bus" => 2, "status" => 0)
            data["dcline"]["1"] = Dict{String,Any}("index" => 1, "f_bus" => 1,
                                                   "t_bus" => 3, "br_status" => 1)
            data["dcline"]["2"] = Dict{String,Any}("index" => 2, "f_bus" => 1,
                                                   "t_bus" => 3, "br_status" => 0)
            r = base_nw_ref(data)
            @test base_sorted_keys(r[:shunt]) == [1]
            @test base_sorted_keys(r[:switch]) == [1]
            @test base_sorted_keys(r[:dcline]) == [1]
            @test r[:bus_shunts] == Dict(1 => Int[], 2 => Int[], 3 => [1], 4 => Int[])
            @test r[:arcs_from_sw] == [(1, 2, 3)]
            @test r[:arcs_to_sw] == [(1, 3, 2)]
            @test r[:arcs_sw] == [(1, 2, 3), (1, 3, 2)]
            @test r[:bus_arcs_sw] == Dict(1 => [], 2 => [(1, 2, 3)],
                                          3 => [(1, 3, 2)], 4 => [])
            @test r[:arcs_from_dc] == [(1, 1, 3)]
            @test r[:arcs_to_dc] == [(1, 3, 1)]
            @test r[:arcs_dc] == [(1, 1, 3), (1, 3, 1)]
            @test r[:bus_arcs_dc] == Dict(1 => [(1, 1, 3)], 2 => [],
                                          3 => [(1, 3, 1)], 4 => [])
        end

        @testset "eDisGo component types" begin
            # `gen_nd`, `gen_slack`, `heatpumps`, `electromobility`, `dsm` and
            # `heat_storage` are eDisGo additions. They are filtered like the
            # PowerModels components, by their own status field …
            data = TestUtils.small_network()
            data["gen_nd"]["1"]["gen_status"] = 0
            data["gen_slack"]["1"]["gen_status"] = 0
            data["heatpumps"]["1"]["status"] = 0
            data["electromobility"]["1"]["status"] = 0
            data["dsm"]["1"]["status"] = false
            r = base_nw_ref(data)
            for comp in (:gen_nd, :gen_slack, :heatpumps, :electromobility, :dsm)
                @test isempty(r[comp])
            end
            # … and the lookups no longer list them.
            @test r[:bus_gens_nd][3] == Int[]
            @test r[:bus_gens_slack][1] == Int[]
            @test r[:bus_hps][3] == Int[]
            @test r[:bus_cps][3] == Int[]
            @test r[:bus_dsm][3] == Int[]

            # A missing status field counts as active: the fixture has none on
            # `heatpumps`, `electromobility` and `dsm`.
            r_ok = base_nw_ref(TestUtils.small_network())
            for comp in (:gen_nd, :gen_slack, :heatpumps, :electromobility, :dsm,
                         :heat_storage)
                @test base_sorted_keys(r_ok[comp]) == [1]
            end
            # Any non-inactive status keeps the component.
            data_on = TestUtils.small_network()
            data_on["heatpumps"]["1"]["status"] = 1
            data_on["dsm"]["1"]["status"] = true
            r_on = base_nw_ref(data_on)
            @test base_sorted_keys(r_on[:heatpumps]) == [1]
            @test base_sorted_keys(r_on[:dsm]) == [1]

            # A deactivated bus drops the eDisGo components on it, like the
            # PowerModels components. Bus 3 of the fixture carries one of each.
            data_b = TestUtils.small_network()
            data_b["bus"]["3"]["bus_type"] = 4
            r_b = base_nw_ref(data_b)
            for comp in (:gen_nd, :heatpumps, :electromobility, :dsm, :heat_storage,
                         :load)
                @test isempty(r_b[comp])
            end
            @test base_sorted_keys(r_b[:gen_slack]) == [1]
            @test base_sorted_keys(r_b[:bus_hps]) == [1, 2, 4]
        end

        @testset "heat storage" begin
            # Switched off: dropped, its heat pump stays (and runs without storage,
            # see `constraint_hp_operation`).
            data = TestUtils.small_network()
            data["heat_storage"]["1"]["status"] = false
            r = base_nw_ref(data)
            @test isempty(r[:heat_storage])
            @test base_sorted_keys(r[:heatpumps]) == [1]

            # Heat storage `i` belongs to heat pump `i`: a heat pump that is
            # dropped, by status or by its bus, takes its heat storage with it.
            data_hp = TestUtils.small_network()
            data_hp["heatpumps"]["1"]["status"] = 0
            @test isempty(base_nw_ref(data_hp)[:heat_storage])
            data_bus = TestUtils.small_network()
            data_bus["heatpumps"]["1"]["hp_bus"] = 2
            data_bus["bus"]["2"]["bus_type"] = 4
            r_bus = base_nw_ref(data_bus)
            @test isempty(r_bus[:heatpumps])
            @test isempty(r_bus[:heat_storage])

            # A heat storage on a deactivated bus is dropped as well.
            data_hs = TestUtils.small_network()
            data_hs["heat_storage"]["1"]["storage_bus"] = 2
            data_hs["bus"]["2"]["bus_type"] = 4
            r_hs = base_nw_ref(data_hs)
            @test isempty(r_hs[:heat_storage])
            @test base_sorted_keys(r_hs[:heatpumps]) == [1]
        end
    end

    @testset "arcs" begin
        r = base_nw_ref(TestUtils.small_network())
        # One arc per branch and direction; the virtual storage branch 3 included.
        @test sort(r[:arcs_from]) == [(1, 1, 2), (2, 2, 3), (3, 3, 4)]
        @test sort(r[:arcs_to]) == [(1, 2, 1), (2, 3, 2), (3, 4, 3)]
        @test r[:arcs] == [r[:arcs_from]; r[:arcs_to]]
        @test length(r[:arcs]) == 2 * length(r[:branch])
        @test Set(r[:arcs]) == Set([r[:arcs_from]; r[:arcs_to]])
        # `arcs_to` is `arcs_from` with the buses swapped.
        @test sort([(l, j, i) for (l, i, j) in r[:arcs_to]]) == sort(r[:arcs_from])

        # Without switches or DC lines the corresponding sets are empty.
        @test isempty(r[:arcs_from_dc])
        @test isempty(r[:arcs_to_dc])
        @test isempty(r[:arcs_dc])
        @test isempty(r[:arcs_from_sw])
        @test isempty(r[:arcs_to_sw])
        @test isempty(r[:arcs_sw])
    end

    @testset "bus component lookups" begin
        r = base_nw_ref(TestUtils.small_network())
        all_buses = [1, 2, 3, 4]

        # Every lookup is keyed by *all* active buses, including those without a
        # component of that type — the constraint templates rely on that.
        for key in (:bus_loads, :bus_shunts, :bus_gens, :bus_dsm, :bus_cps,
                    :bus_hps, :bus_gens_nd, :bus_gens_slack, :bus_storage,
                    :bus_arcs, :bus_arcs_to, :bus_arcs_from, :bus_lines_to,
                    :bus_arcs_dc, :bus_arcs_sw)
            @test base_sorted_keys(r[key]) == all_buses
        end

        # Component lookups — bus 3 is the flexibility bus of the fixture.
        @test r[:bus_loads] == Dict(1 => Int[], 2 => Int[], 3 => [1], 4 => Int[])
        @test r[:bus_shunts] == Dict(1 => Int[], 2 => Int[], 3 => Int[], 4 => Int[])
        @test r[:bus_gens] == Dict(1 => Int[], 2 => [1], 3 => Int[], 4 => Int[])
        @test r[:bus_dsm] == Dict(1 => Int[], 2 => Int[], 3 => [1], 4 => Int[])
        @test r[:bus_cps] == Dict(1 => Int[], 2 => Int[], 3 => [1], 4 => Int[])
        @test r[:bus_hps] == Dict(1 => Int[], 2 => Int[], 3 => [1], 4 => Int[])
        @test r[:bus_gens_nd] == Dict(1 => Int[], 2 => Int[], 3 => [1], 4 => Int[])
        @test r[:bus_gens_slack] == Dict(1 => [1], 2 => Int[], 3 => Int[], 4 => Int[])
        @test r[:bus_storage] == Dict(1 => Int[], 2 => Int[], 3 => Int[], 4 => [1])

        # Arc lookups. `bus_arcs` collects both directions, so every inner bus of
        # the radial feeder sees two arcs, the two end buses one each.
        @test sort(r[:bus_arcs][1]) == [(1, 1, 2)]
        @test sort(r[:bus_arcs][2]) == [(1, 2, 1), (2, 2, 3)]
        @test sort(r[:bus_arcs][3]) == [(2, 3, 2), (3, 3, 4)]
        @test sort(r[:bus_arcs][4]) == [(3, 4, 3)]
        @test sum(length, values(r[:bus_arcs])) == length(r[:arcs])

        # `bus_arcs_from` / `bus_arcs_to` split `arcs_from` by its two buses: the
        # outgoing arcs of a bus and the single incoming one (radial grid).
        @test r[:bus_arcs_from] == Dict(1 => [(1, 1, 2)], 2 => [(2, 2, 3)],
                                        3 => [(3, 3, 4)], 4 => [])
        @test r[:bus_arcs_to] == Dict(1 => [], 2 => [(1, 1, 2)],
                                      3 => [(2, 2, 3)], 4 => [(3, 3, 4)])
        # The slack bus has no upstream branch, every other bus has exactly one.
        @test isempty(r[:bus_arcs_to][1])
        @test all(length(r[:bus_arcs_to][i]) == 1 for i in (2, 3, 4))

        # `bus_lines_to` is `bus_arcs_to` reduced to the branch index.
        @test r[:bus_lines_to] == Dict(1 => Int[], 2 => [1], 3 => [2], 4 => [3])
        for (i, arcs) in r[:bus_arcs_to]
            @test r[:bus_lines_to][i] == [l for (l, _f, _t) in arcs]
        end

        @test r[:bus_arcs_dc] == Dict(i => [] for i in all_buses)
        @test r[:bus_arcs_sw] == Dict(i => [] for i in all_buses)
    end

    @testset "ref_buses" begin
        r = base_nw_ref(TestUtils.small_network())
        # Bus 1 is the only slack (`bus_type == 3`) of the fixture …
        @test base_sorted_keys(r[:ref_buses]) == [1]
        @test r[:ref_buses][1] === r[:bus][1]
        @test r[:ref_buses][1]["bus_type"] == 3
        # … and the entry really is the bus dict, not just its index.
        @test r[:ref_buses][1]["name"] == "Bus_1"

        # No slack bus at all: an empty (but present) lookup, no error.
        data = TestUtils.small_network()
        data["bus"]["1"]["bus_type"] = 1
        r0 = base_nw_ref(data)
        @test r0[:ref_buses] isa Dict{Int,Any}
        @test isempty(r0[:ref_buses])

        # A deactivated slack bus does not end up in `:ref_buses` either; the
        # slack generator sitting on it is dropped with it.
        data_off = TestUtils.small_network()
        data_off["bus"]["1"]["bus_type"] = 4
        r_off = base_nw_ref(data_off)
        @test isempty(r_off[:ref_buses])
        @test isempty(r_off[:gen_slack])

        # More than one slack bus: both are collected, with a warning that this
        # can make the problem infeasible.
        data_two = TestUtils.small_network()
        data_two["bus"]["2"]["bus_type"] = 3
        r2, warnings = TestUtils.capture_warnings(() -> base_nw_ref(data_two))
        @test base_sorted_keys(r2[:ref_buses]) == [1, 2]
        @test count(occursin("multiple reference buses found", w) for w in warnings) == 1
        # A single slack bus does not warn.
        _, warnings = TestUtils.capture_warnings(() -> base_nw_ref(TestUtils.small_network()))
        @test !any(occursin("multiple reference buses", w) for w in warnings)
    end

    @testset "buspairs" begin
        r = base_nw_ref(TestUtils.small_network())
        # One entry per branch of the radial feeder, keyed by (f_bus, t_bus).
        @test base_sorted_keys(r[:buspairs]) == [(1, 2), (2, 3), (3, 4)]
        bp = r[:buspairs][(1, 2)]
        @test bp["branch"] == 1
        @test bp["rate_a"] == 5.0
        @test bp["tap"] == 1.0
        @test bp["vm_fr_min"] == 0.9
        @test bp["vm_fr_max"] == 1.1
        @test bp["angmin"] ≈ -pi / 6 atol = 1e-12
        @test bp["angmax"] ≈ pi / 6 atol = 1e-12

        # Only branches that survived the filter are considered.
        data = TestUtils.small_network()
        data["branch"]["2"]["br_status"] = 0
        @test base_sorted_keys(base_nw_ref(data)[:buspairs]) == [(1, 2), (3, 4)]

        # `:buspairs` is only computed when it is not there yet — a ref that
        # already carries one (e.g. from a previous `ref_add_core!` run or a
        # custom extension) keeps it untouched.
        ref = base_raw_ref(TestUtils.small_network())
        sentinel = Dict{Any,Any}("sentinel" => true)
        ref[:it][eDisGo_OPF.pm_it_sym][:nw][0][:buspairs] = sentinel
        eDisGo_OPF.ref_add_core!(ref)
        @test ref[:it][eDisGo_OPF.pm_it_sym][:nw][0][:buspairs] === sentinel

        # Running the function twice is therefore stable for `:buspairs` …
        ref2 = base_raw_ref(TestUtils.small_network())
        eDisGo_OPF.ref_add_core!(ref2)
        first_pairs = ref2[:it][eDisGo_OPF.pm_it_sym][:nw][0][:buspairs]
        eDisGo_OPF.ref_add_core!(ref2)
        @test ref2[:it][eDisGo_OPF.pm_it_sym][:nw][0][:buspairs] === first_pairs
        # … and for the lookups, which are rebuilt from the already filtered data.
        @test ref2[:it][eDisGo_OPF.pm_it_sym][:nw][0][:bus_cps][3] == [1]
        @test length(ref2[:it][eDisGo_OPF.pm_it_sym][:nw][0][:arcs]) == 6
    end

    @testset "real test grid" begin
        # Reality check on the 143-bus eDisGo test grid: the invariants that hold
        # for the hand-written fixture have to hold here too.
        r = base_nw_ref(TestUtils.pm_network())
        @test length(r[:bus]) == 143
        @test length(r[:branch]) == 142
        @test length(r[:load]) == 50
        @test length(r[:gen_nd]) == 27
        @test length(r[:gen_slack]) == 1
        @test length(r[:storage]) == 1
        @test length(r[:electromobility]) == 1
        @test length(r[:heatpumps]) == 3
        @test length(r[:dsm]) == 2

        # Exactly one slack bus, and it is the bus of the slack generator.
        @test length(r[:ref_buses]) == 1
        slack_bus = only(keys(r[:ref_buses]))
        @test r[:bus_gens_slack][slack_bus] == [1]
        @test isempty(r[:bus_arcs_to][slack_bus])

        # Radial grid: one branch fewer than buses, one incoming arc per non-slack
        # bus, and one buspair per branch.
        @test length(r[:branch]) == length(r[:bus]) - 1
        @test length(r[:arcs]) == 2 * length(r[:branch])
        @test length(r[:buspairs]) == length(r[:branch])
        @test all(length(r[:bus_arcs_to][i]) == 1
                  for i in keys(r[:bus]) if i != slack_bus)

        # Every lookup covers every bus, and every component is listed exactly once.
        for (key, comp) in ((:bus_loads, :load), (:bus_gens, :gen),
                            (:bus_gens_nd, :gen_nd), (:bus_gens_slack, :gen_slack),
                            (:bus_storage, :storage), (:bus_cps, :electromobility),
                            (:bus_hps, :heatpumps), (:bus_dsm, :dsm),
                            (:bus_shunts, :shunt))
            @test base_sorted_keys(r[key]) == base_sorted_keys(r[:bus])
            @test sort(reduce(vcat, values(r[key]); init=Int[])) ==
                  base_sorted_keys(r[comp])
        end

        # And the buses referenced by the components are all active.
        @test all(l["load_bus"] in keys(r[:bus]) for l in values(r[:load]))
        @test all(b["f_bus"] in keys(r[:bus]) && b["t_bus"] in keys(r[:bus])
                  for b in values(r[:branch]))
    end

    @testset "§14a curtailment" begin
        # Missing in the data: empty, with an empty bus mapping.
        nw_ref = base_nw_ref(TestUtils.small_network())
        @test isempty(nw_ref[:curtailment_14a])
        @test all(isempty, values(nw_ref[:bus_curtailment_14a]))

        # Mapped to its bus; switched off or at an inactive bus it is dropped.
        data = TestUtils.add_curtailment_14a!(TestUtils.small_network())
        TestUtils.add_curtailment_14a!(data; i=2, bus=2)
        TestUtils.add_curtailment_14a!(data; i=3)
        data["curtailment_14a"]["3"]["status"] = false
        nw_ref = base_nw_ref(data)
        @test sort(collect(keys(nw_ref[:curtailment_14a]))) == [1, 2]
        @test nw_ref[:bus_curtailment_14a][3] == [1]
        @test nw_ref[:bus_curtailment_14a][2] == [2]
        data["bus"]["2"]["bus_type"] = 4
        @test sort(collect(keys(base_nw_ref(data)[:curtailment_14a]))) == [1]
    end
end
