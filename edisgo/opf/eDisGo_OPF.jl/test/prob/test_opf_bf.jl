# Tests for `src/prob/opf_bf.jl`.
#
# The file defines the eDisGo OPF problem itself:
#
#   * `solve_mn_opf_bf_flex` — `solve_model` with the build method below and
#     `multinetwork=true`; this is the function `Main.jl` calls for both the SOC
#     (Gurobi) and the non-convex (Ipopt) run.
#   * `build_mn_opf_bf_flex` — the build method: variables, constraints and
#     objective of the multi-period radial branch-flow OPF. What it creates
#     depends on `opf_version` (read from network 1):
#       - 1 and 3 lift the grid restrictions (`ccm`/`w` without upper bounds) and
#         minimise the maximum line loading `ll`, which exists in network 1 only,
#       - 2 and 4 keep the grid restrictions and add curtailment/shedding slacks,
#       - 3 and 4 additionally add the overlying-grid (HV) requirement slacks
#         `phvs` and constraints,
#       - anything else is rejected with an `ArgumentError`.
#
# `build_mn_opf_bf_flex` is tested on instantiated but unsolved models: which
# variables exist in which network, their bounds, how many constraints of which
# kind are created and which objective is installed — on the hand-written
# `small_network` with hand-counted numbers and on the real eDisGo test grid
# against a counting rule derived from the build. `solve_mn_opf_bf_flex` is tested
# end to end with Ipopt on `small_network`: status, and plausibility of the
# solution (slack voltage, exactness of the non-convex current equation, system
# power balance, end-of-horizon storage states) rather than exact solver numbers.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Ipopt handles both formulations (the SOC cone is a quadratic constraint for it);
# `print_level` 0 keeps the solver log out of the test output.
const OPF_IPOPT = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

"Whether `opf_version` `v` takes overlying-grid (HV) requirements into account."
opf_has_og(v::Int) = v in (3, 4)

"""
Run `f` with `stdout` redirected and return `(result of f, everything it printed)`.

The objective of `opf_version` 3 prints two debug lines on every build (known
defect, see `test/core/test_objective.jl`); this keeps them out of the test log.
"""
function opf_capture_stdout(f)
    original = stdout
    rd, wr = redirect_stdout()
    reader = @async read(rd, String)
    local value
    try
        value = f()
    finally
        redirect_stdout(original)
        close(wr)
    end
    return value, fetch(reader)
end

"""
Instantiate `build_mn_opf_bf_flex` on `TestUtils.small_network` without solving.
`overlying_grid` defaults to whether the version uses HV requirements.
"""
function opf_small_model(model_type::Type=eDisGo_OPF.NCBFPowerModelEdisgo;
                         opf_version::Int, overlying_grid::Bool=opf_has_og(opf_version),
                         num_steps::Int=2)
    pm, _ = opf_capture_stdout() do
        TestUtils.build_small_model(model_type; opf_version=opf_version,
                                    overlying_grid=overlying_grid, num_steps=num_steps)
    end
    return pm
end

"Solve the multi-network form of `small_network` with Ipopt via `solve_mn_opf_bf_flex`."
function opf_solve_small(model_type::Type=eDisGo_OPF.NCBFPowerModelEdisgo;
                         opf_version::Int=2, kwargs...)
    data = TestUtils.multinetwork(TestUtils.small_network(
        opf_version=opf_version, overlying_grid=opf_has_og(opf_version)))
    (result, pm), _ = opf_capture_stdout() do
        eDisGo_OPF.solve_mn_opf_bf_flex(data, model_type, OPF_IPOPT; kwargs...)
    end
    return result, pm, data
end

"Number of affine equality constraints of `pm`."
opf_n_eq(pm) = JuMP.num_constraints(pm.model, JuMP.AffExpr, MOI.EqualTo{Float64})
"Number of quadratic `≤` constraints of `pm` (SOC cone, maximum line loading)."
opf_n_quad(pm) = JuMP.num_constraints(pm.model, JuMP.QuadExpr, MOI.LessThan{Float64})
"Number of (legacy) nonlinear constraints of `pm` (non-convex current equation)."
opf_n_nl(pm) = JuMP.num_nonlinear_constraints(pm.model)

"Constraint types of `pm` apart from variable bounds."
opf_constraint_types(pm) = Set{Any}((F, S) for (F, S) in JuMP.list_of_constraint_types(pm.model)
                                    if F != JuMP.VariableRef)

"Index set of the variable container `x` (dense or sparse)."
opf_indices(x) = Set(x isa JuMP.Containers.SparseAxisArray ? keys(x.data) : axes(x, 1))

"""
All constraints of `pm` except variable bounds, rendered as strings. Unlike
`TestUtils.constraint_strings` this leaves out `x ≥ lb` / `x ≤ ub`, so a search
for a variable name finds only the constraints the variable really takes part in.
"""
function opf_constraint_strings(pm)
    strs = String[]
    for (F, S) in JuMP.list_of_constraint_types(pm.model)
        F == JuMP.VariableRef && continue
        append!(strs, string.(JuMP.all_constraints(pm.model, F, S)))
    end
    return strs
end

"Regex matching the JuMP variable `<nw>_<name>[<index>` as a whole name."
opf_var_regex(nw, name, index="") = Regex("\\b$(nw)_$(name)\\[" * index)

"Number of non-bound constraints of `pm` that contain the variable `<nw>_<name>[<index>`."
opf_count_with(pm, nw, name, index="") =
    count(s -> occursin(opf_var_regex(nw, name, index), s), opf_constraint_strings(pm))

# Variables every `opf_version` creates in every network.
const OPF_COMMON_VARS = [:p, :q, :ccm, :w, :phss, :phps2, :ps, :se, :phs, :hse,
                         :php, :pcp, :cpe, :pdsm, :dsme, :pgs, :qgs]
# Grid-restriction slacks of versions 2 and 4.
const OPF_SLACK_VARS = [:pgc, :phps, :pds, :pgens, :pcps]

"Sorted variable keys `build_mn_opf_bf_flex` should create in network `nw`."
function opf_expected_vars(v::Int, nw::Int)
    vars = copy(OPF_COMMON_VARS)
    v in (2, 4) && append!(vars, OPF_SLACK_VARS)
    v in (3, 4) && push!(vars, :phvs)
    v == 3 && push!(vars, :pgc)             # curtailment for the HV requirement only
    v in (1, 3) && nw == 1 && push!(vars, :ll)
    return sort(vars)
end

"""
Number of constraints `build_mn_opf_bf_flex` should create for the data of `pm`,
as `(eq, quad, nl)` — the counting rule of the build written out per component:

* per network: P and Q balance at every bus, the voltage-drop equation on every
  physical branch (plus `w == 1` at a slack sending end), the heat-pump balance,
  the HV requirements (versions 3/4), the current model (one NL equality per
  physical branch for NC, one SOC inequality per branch for SOC) and the maximum
  line loading on every physical branch (versions 1/3);
* over the horizon: two initial/final-state equations plus one coupling equation
  per further step for every storage, heat storage and DSM unit; for every
  charging point one initial equation, one coupling equation per further step and
  one end condition in the last step (only reached if there is a second step).
"""
function opf_expected_counts(pm)
    v = PowerModels.ref(pm, 1, :opf_version)
    nws = sort(collect(PowerModels.nw_ids(pm)))
    N = length(nws)
    nc = pm isa eDisGo_OPF.AbstractNCBFModelEdisgo
    eq = quad = nl = 0
    for n in nws
        buses = PowerModels.ref(pm, n, :bus)
        branches = collect(values(PowerModels.ref(pm, n, :branch)))
        phys = [b for b in branches if !b["storage"]]
        eq += 2 * length(buses)
        eq += length(phys)
        eq += count(b -> buses[b["f_bus"]]["bus_type"] == 3, phys)
        eq += length(PowerModels.ids(pm, n, :heatpumps))
        opf_has_og(v) && (eq += length(PowerModels.ids(pm, n, :HV_requirements)))
        nc ? (nl += length(phys)) : (quad += length(branches))
        v in (1, 3) && (quad += length(phys))
    end
    for kind in (:storage, :heat_storage, :dsm)
        eq += length(PowerModels.ids(pm, kind, nw=nws[1])) * (2 + (N - 1))
    end
    eq += length(PowerModels.ids(pm, :electromobility, nw=nws[1])) * (1 + (N - 1) + (N > 1))
    return (eq=eq, quad=quad, nl=nl)
end

"Actual constraint counts of `pm` in the form of [`opf_expected_counts`](@ref)."
opf_counts(pm) = (eq=opf_n_eq(pm), quad=opf_n_quad(pm), nl=opf_n_nl(pm))

"""
Check the solution `sol` of `small_network` (multi-network `data`) for physical
plausibility. Everything here follows from the constraints of the model, so it
must hold for any (locally) optimal point up to the solver tolerance.
"""
function opf_check_plausible(sol, data, v::Int; exact_current::Bool=true)
    tol = 1e-5
    nws = sort(collect(keys(sol["nw"])))
    last_nw = nws[end]
    for n in nws
        s = sol["nw"][n]
        d = data["nw"][n]
        w = Dict(i => b["w"] for (i, b) in s["bus"])
        # Voltage at the slack bus is fixed to 1 p.u. and all squared voltages are
        # positive; versions 2 and 4 also keep them inside the [0.9, 1.1] band.
        @test w["1"] ≈ 1.0 atol = tol
        @test all(x > 0 for x in values(w))
        if v in (2, 4)
            @test all(0.81 - tol <= x <= 1.21 + tol for x in values(w))
        end
        # Current model: p² + q² = w·ccm on the physical branches (NC) or ≤ (SOC).
        for (l, br) in s["branch"]
            d["branch"][l]["storage"] && continue
            s2 = br["pf"]^2 + br["qf"]^2
            wccm = w[string(d["branch"][l]["f_bus"])] * br["ccm"]
            if exact_current
                @test s2 ≈ wccm atol = tol
            else
                @test s2 <= wccm + tol
            end
        end
        # Losses are non-negative and small for this short feeder.
        losses = sum(br["ccm"] * d["branch"][l]["br_r"] for (l, br) in s["branch"])
        @test -tol <= losses < 0.05
        # System-wide active power balance: summing the nodal balances over all
        # buses cancels the branch flows, leaving losses = net injection.
        injection = sum(g["pgs"] for g in values(s["gen_slack"])) +
                    sum(g["pg"] for g in values(d["gen"])) +
                    sum(g["pg"] for g in values(d["gen_nd"])) +
                    sum(st["ps"] for st in values(s["storage"])) -
                    sum(l["pd"] for l in values(d["load"])) -
                    sum(x["pdsm"] for x in values(s["dsm"])) -
                    sum(x["php"] for x in values(s["heatpumps"])) -
                    sum(x["pcp"] for x in values(s["electromobility"]))
        if v in (2, 4)
            injection += sum(x["pds"] for x in values(s["load"])) -
                         sum(x["pgens"] for x in values(s["gen"])) -
                         sum(x["pgc"] for x in values(s["gen_nd"])) +
                         sum(x["phps"] for x in values(s["heatpumps"])) +
                         sum(x["pcps"] for x in values(s["electromobility"]))
        end
        @test injection ≈ losses atol = tol
        # Heat pump balance: cop·(php + phps2) = heat demand + phss − phs.
        hp, hs = s["heatpumps"]["1"], s["heat_storage"]["1"]
        @test d["heatpumps"]["1"]["cop"] * (hp["php"] + hp["phps2"]) ≈
              d["heatpumps"]["1"]["pd"] + hs["phss"] - hs["phs"] atol = tol
        if v in (2, 4)
            pgc = s["gen_nd"]["1"]["pgc"]
            @test -tol <= pgc <= d["gen_nd"]["1"]["pg"] + tol
        end
        if opf_has_og(v)
            @test sort(collect(keys(s["HV_requirements"]))) == ["1", "2", "3", "4", "5"]
            @test all(isfinite(h["phvs"]) for h in values(s["HV_requirements"]))
        end
    end
    if v in (1, 3)
        # One maximum line loading per physical branch, reported in network 1.
        ll = sol["nw"]["1"]["branch"]
        @test all(ll[l]["ll"] >= 1 - tol for l in ("1", "2"))
        @test !haskey(ll["3"], "ll")
    end
    # End of the horizon: storage, heat storage and DSM back to their end state,
    # the charging point back to the middle of its flexibility band.
    s = sol["nw"][last_nw]
    d = data["nw"][last_nw]
    @test s["storage"]["1"]["se"] ≈ d["storage"]["1"]["soc_end"] atol = tol
    @test s["heat_storage"]["1"]["hse"] ≈ d["heat_storage"]["1"]["soc_end"] atol = tol
    @test s["dsm"]["1"]["dsme"] ≈ 0.0 atol = tol
    cp = d["electromobility"]["1"]
    @test s["electromobility"]["1"]["cpe"] ≈ 0.5 * (cp["e_min"] + cp["e_max"]) atol = tol
end

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@testset "solve_mn_opf_bf_flex" begin
    nc_objective = Dict{Int,Float64}()

    @testset "NC model, opf_version $v" for v in 1:4
        result, pm, data = opf_solve_small(opf_version=v)
        @test pm isa eDisGo_OPF.NCBFPowerModelEdisgo
        @test result["termination_status"] == MOI.LOCALLY_SOLVED
        @test result["primal_status"] == MOI.FEASIBLE_POINT
        @test result["optimizer"] == "Ipopt"
        # Every objective is a sum of losses, slacks, (ll − 1) ≥ 0 and squares.
        @test isfinite(result["objective"])
        @test result["objective"] >= -1e-6
        nc_objective[v] = result["objective"]

        sol = result["solution"]
        @test sol["multinetwork"] == true
        @test sort(collect(keys(sol["nw"]))) == ["1", "2"]
        # The model was built by `build_mn_opf_bf_flex` for this version.
        @test TestUtils.variable_keys(pm, 1) == opf_expected_vars(v, 1)
        opf_check_plausible(sol, data, v)
    end

    @testset "SOC model" begin
        result, pm, data = opf_solve_small(eDisGo_OPF.SOCBFPowerModelEdisgo, opf_version=2)
        @test pm isa eDisGo_OPF.SOCBFPowerModelEdisgo
        @test result["termination_status"] == MOI.LOCALLY_SOLVED
        # Same order of magnitude as the NC optimum. It is *not* a lower bound
        # here: the SOC cone is also imposed on the virtual storage branch, which
        # the NC current equation skips (known defect, form/bf.jl:101 vs. 128), so
        # the SOC model is slightly tighter there and ends up marginally above NC.
        @test result["objective"] ≈ nc_objective[2] rtol = 1e-2
        @test result["objective"] >= -1e-6
        opf_check_plausible(result["solution"], data, 2; exact_current=false)
    end

    @testset "keyword arguments" begin
        # Everything else is passed on to `solve_model`; requesting duals makes the
        # power balance report its multipliers.
        result, _, _ = opf_solve_small(
            opf_version=2, setting=Dict{String,Any}("output" => Dict("duals" => true)))
        @test result["termination_status"] == MOI.LOCALLY_SOLVED
        for n in ("1", "2"), i in ("1", "2", "3", "4")
            bus = result["solution"]["nw"][n]["bus"][i]
            @test haskey(bus, "lam_kcl_r")
            @test haskey(bus, "lam_kcl_i")
        end
        # Negative case: not requested, not reported.
        plain, _, _ = opf_solve_small(opf_version=2)
        @test !haskey(plain["solution"]["nw"]["1"]["bus"]["1"], "lam_kcl_r")
    end

    @testset "model_type" begin
        data = TestUtils.multinetwork(TestUtils.small_network())
        # Only branch-flow models are accepted by the signature …
        @test_throws MethodError eDisGo_OPF.solve_mn_opf_bf_flex(
            data, PowerModels.ACPPowerModel, OPF_IPOPT)
        # … and only the eDisGo ones by the build method.
        @test_throws MethodError eDisGo_OPF.solve_mn_opf_bf_flex(
            data, PowerModels.SOCBFPowerModel, OPF_IPOPT)
    end
end

@testset "build_mn_opf_bf_flex" begin
    @testset "variables, opf_version $v" for v in 1:4
        pm = opf_small_model(opf_version=v)
        @test pm isa eDisGo_OPF.NCBFPowerModelEdisgo
        @test TestUtils.variable_keys(pm, 1) == opf_expected_vars(v, 1)
        @test TestUtils.variable_keys(pm, 2) == opf_expected_vars(v, 2)
        # Same variables for the SOC formulation.
        pm_soc = opf_small_model(eDisGo_OPF.SOCBFPowerModelEdisgo, opf_version=v)
        @test TestUtils.variable_keys(pm_soc, 1) == opf_expected_vars(v, 1)
        if v in (1, 3)
            # One maximum line loading for the whole horizon, physical branches only.
            @test opf_indices(PowerModels.var(pm, 1, :ll)) == Set([(1, 1, 2), (2, 2, 3)])
        end
    end

    @testset "grid restrictions" begin
        ccm_ub = (5.0 * 1.0 / 0.9)^2      # (rate_a · tap / vmin)²
        for v in 1:4
            pm = opf_small_model(opf_version=v)
            for n in 1:2
                ccm = PowerModels.var(pm, n, :ccm)
                w = PowerModels.var(pm, n, :w)
                @test all(JuMP.lower_bound(ccm[l]) == 0.0 for l in 1:3)
                # The virtual storage branch keeps its current limit in every version.
                @test JuMP.upper_bound(ccm[3]) ≈ ccm_ub
                if v in (1, 3)
                    # Lifted: no current limit on lines, no voltage band.
                    @test !JuMP.has_upper_bound(ccm[1])
                    @test !JuMP.has_upper_bound(ccm[2])
                    @test all(JuMP.lower_bound(w[i]) == 0.0 for i in 1:3)
                    @test all(!JuMP.has_upper_bound(w[i]) for i in 1:3)
                else
                    @test JuMP.upper_bound(ccm[1]) ≈ ccm_ub
                    @test JuMP.upper_bound(ccm[2]) ≈ ccm_ub
                    @test all(JuMP.lower_bound(w[i]) ≈ 0.81 for i in 1:3)
                    @test all(JuMP.upper_bound(w[i]) ≈ 1.21 for i in 1:3)
                end
                # Branch flows are never bounded (the current limit replaces that).
                for x in (PowerModels.var(pm, n, :p), PowerModels.var(pm, n, :q))
                    @test opf_indices(x) == Set([(1, 1, 2), (2, 2, 3), (3, 3, 4)])
                    @test all(!JuMP.has_lower_bound(x[arc]) && !JuMP.has_upper_bound(x[arc])
                              for arc in opf_indices(x))
                end
            end
        end
    end

    @testset "curtailment" begin
        for v in (2, 4)
            pm = opf_small_model(opf_version=v)
            for n in 1:2
                pgc = PowerModels.var(pm, n, :pgc)[1]
                @test JuMP.lower_bound(pgc) == 0.0
                @test JuMP.upper_bound(pgc) ≈ 0.6
                # P and Q balance of bus 3, plus the "curt" HV requirement in 4.
                @test opf_count_with(pm, n, "pgc", "1") == (v == 2 ? 2 : 3)
            end
        end
        @test !haskey(PowerModels.var(opf_small_model(opf_version=1), 1), :pgc)

        # BUG: version 3 creates `pgc` for the "curt" HV requirement, but neither
        # the power balance (curtailment slacks only for 2 and 4, form/bf.jl:213)
        # nor the objective use it. The requirement is met by a curtailment that
        # never reaches the grid, while the Python side (`from_powermodels`)
        # subtracts exactly this `pgc` from the generator feed-in.
        pm = opf_small_model(opf_version=3)
        for n in 1:2
            pgc = PowerModels.var(pm, n, :pgc)[1]
            @test JuMP.upper_bound(pgc) ≈ 0.6
            @test opf_count_with(pm, n, "pgc", "1") == 1           # HV constraint only
            @test_broken opf_count_with(pm, n, "pgc", "1") == 3    # plus P/Q balance
            @test JuMP.coefficient(JuMP.objective_function(pm.model).aff, pgc) == 0.0
        end
    end

    @testset "overlying grid requirements" begin
        for v in (3, 4)
            pm = opf_small_model(opf_version=v)
            for n in 1:2
                @test opf_indices(PowerModels.var(pm, n, :phvs)) == Set(1:5)
                # One constraint per requirement, each containing its own slack.
                @test all(opf_count_with(pm, n, "phvs", "$i\\]") == 1 for i in 1:5)
            end
        end
        # Versions 1 and 2 ignore HV requirements present in the data.
        for v in (1, 2)
            pm = opf_small_model(opf_version=v, overlying_grid=true)
            @test !haskey(PowerModels.var(pm, 1), :phvs)
            @test !any(occursin("phvs", s) for s in opf_constraint_strings(pm))
        end
        # Versions 3 and 4 still build without any HV requirement in the data.
        for v in (3, 4)
            pm = opf_small_model(opf_version=v, overlying_grid=false)
            @test isempty(PowerModels.var(pm, 1, :phvs))
            @test opf_counts(pm) == opf_expected_counts(pm)
        end
    end

    @testset "constraint counts" begin
        # Hand-counted for small_network (4 buses, 2 physical + 1 storage branch,
        # one unit of every flexibility, 2 steps):
        # eq   = 2 × (8 balance + 2 voltage drop + 1 slack voltage + 1 heat pump
        #        [+ 5 HV]) + 3 storage kinds × 3 + 3 charging point = 36 [46]
        # quad = 2 × 2 maximum line loading (versions 1 and 3)
        # nl   = 2 × 2 current equations (NC; SOC: 2 × 3 quadratic cones instead)
        expected = Dict(1 => (eq=36, quad=4, nl=4), 2 => (eq=36, quad=0, nl=4),
                        3 => (eq=46, quad=4, nl=4), 4 => (eq=46, quad=0, nl=4))
        for v in 1:4
            pm = opf_small_model(opf_version=v)
            @test opf_counts(pm) == expected[v]
            @test opf_counts(pm) == opf_expected_counts(pm)
            # No other constraint types than these (apart from variable bounds).
            types = Set{Any}([(JuMP.AffExpr, MOI.EqualTo{Float64})])
            v in (1, 3) && push!(types, (JuMP.QuadExpr, MOI.LessThan{Float64}))
            @test opf_constraint_types(pm) == types
        end
        soc_expected = Dict(1 => (eq=36, quad=10, nl=0), 2 => (eq=36, quad=6, nl=0))
        for v in (1, 2)
            pm = opf_small_model(eDisGo_OPF.SOCBFPowerModelEdisgo, opf_version=v)
            @test opf_counts(pm) == soc_expected[v]
            @test opf_counts(pm) == opf_expected_counts(pm)
        end
        # More time steps add one coupling equation per storage unit and step:
        # 3 × 12 + 3 × (2 + 2) + (1 + 2 + 1) = 52.
        pm = opf_small_model(opf_version=2, num_steps=3)
        @test opf_counts(pm) == (eq=52, quad=0, nl=6)
        @test opf_counts(pm) == opf_expected_counts(pm)
    end

    @testset "inter-temporal coupling" begin
        pm = opf_small_model(opf_version=1)
        strs = opf_constraint_strings(pm)
        # The maximum line loading of both networks is bounded by `ll` of network 1.
        @test count(s -> occursin(opf_var_regex(1, "ll"), s), strs) == 4
        @test count(s -> occursin(opf_var_regex(2, "p"), s) &&
                         occursin(opf_var_regex(1, "ll"), s), strs) == 2
        @test !any(occursin(opf_var_regex(2, "ll"), s) for s in strs)
        # Exactly one equation couples the state of each storage-like unit across
        # the two steps.
        for name in ("se", "hse", "dsme", "cpe")
            @test count(s -> occursin(opf_var_regex(1, name, "1\\]"), s) &&
                             occursin(opf_var_regex(2, name, "1\\]"), s), strs) == 1
        end
        # Final state: the storage state of the last step appears in its coupling
        # equation and in the end condition (battery, heat storage, DSM, charging
        # point alike).
        for name in ("se", "hse", "dsme", "cpe")
            @test opf_count_with(pm, 2, name, "1\\]") == 2
        end
    end

    @testset "single time step" begin
        pm = opf_small_model(opf_version=2, num_steps=1)
        # 12 per network + 3 × 2 storage-like + 1 charging point.
        @test opf_counts(pm) == (eq=19, quad=0, nl=2)
        @test opf_counts(pm) == opf_expected_counts(pm)
        # Battery, heat storage and DSM get their initial and their end condition
        # in the same network …
        for name in ("se", "hse", "dsme")
            @test opf_count_with(pm, 1, name, "1\\]") == 2
        end
        # BUG: … the charging point only the initial one. Its end condition
        # (cpe = middle of the band, Eq. (3.25) for t = tau) lives in
        # `constraint_cp_state`, which is only called from the second step on, so
        # a single-step OPF leaves the charging-point energy free in its band.
        @test opf_count_with(pm, 1, "cpe", "1\\]") == 1
        @test_broken opf_count_with(pm, 1, "cpe", "1\\]") == 2
    end

    @testset "objective, opf_version $v" for (v, objective) in (
            1 => eDisGo_OPF.objective_min_line_loading_max,
            2 => eDisGo_OPF.objective_min_losses_slacks,
            3 => eDisGo_OPF.objective_min_line_loading_max_OG,
            4 => eDisGo_OPF.objective_min_losses_slacks_OG)
        pm = opf_small_model(opf_version=v)
        @test JuMP.objective_sense(pm.model) == MOI.MIN_SENSE
        built = JuMP.objective_function(pm.model)
        # The installed objective is exactly the one of this version: building it
        # again on the same model gives the same expression.
        opf_capture_stdout(() -> objective(pm))
        @test JuMP.isequal_canonical(built, JuMP.objective_function(pm.model))
        # Quadratic only through the squared HV slacks of versions 3 and 4.
        @test (built isa JuMP.QuadExpr) == opf_has_og(v)
        aff = built isa JuMP.QuadExpr ? built.aff : built
        if v in (1, 3)
            @test JuMP.coefficient(aff, PowerModels.var(pm, 1, :ll)[(1, 1, 2)]) > 0
        else
            @test JuMP.coefficient(aff, PowerModels.var(pm, 1, :pds)[1]) > 0
        end
        if opf_has_og(v)
            phvs = PowerModels.var(pm, 2, :phvs)[1]
            @test JuMP.coefficient(built, phvs, phvs) > 0
        end
    end

    @testset "unknown opf_version" begin
        for v in (0, 5)
            err = try
                opf_small_model(opf_version=v)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("OPF version $v is not implemented", err.msg)
        end
    end

    @testset "model type" begin
        data = TestUtils.multinetwork(TestUtils.small_network())
        @test_throws MethodError eDisGo_OPF.instantiate_model(
            data, PowerModels.SOCBFPowerModel, eDisGo_OPF.build_mn_opf_bf_flex)
    end

    @testset "eDisGo test grid, opf_version $v" for v in 1:4
        pm, _ = opf_capture_stdout() do
            TestUtils.build_model(eDisGo_OPF.NCBFPowerModelEdisgo;
                                  overlying_grid=opf_has_og(v), opf_version=v)
        end
        @test sort(collect(PowerModels.nw_ids(pm))) == 1:4
        @test TestUtils.variable_keys(pm, 1) == opf_expected_vars(v, 1)
        @test TestUtils.variable_keys(pm, 4) == opf_expected_vars(v, 4)
        @test length(PowerModels.var(pm, 1, :p)) == 142
        @test length(PowerModels.var(pm, 1, :pgs)) == 1
        if opf_has_og(v)
            @test !isempty(PowerModels.var(pm, 1, :phvs))
        end
        @test opf_counts(pm) == opf_expected_counts(pm)
    end
end
