# Tests for `src/core/objective.jl`.
#
# The file holds the five objective functions of the eDisGo OPF, one per
# `opf_version` plus the pure loss-minimising variant that is not wired into
# `build_mn_opf_bf_flex`:
#
#   * `objective_min_losses`            — Σ ccm·r over the non-storage branches,
#   * `objective_min_losses_slacks`     — version 2, Eq. (3.2 iii),
#   * `objective_min_line_loading_max`  — version 1, Eq. (3.2 ii),
#   * `objective_min_losses_slacks_OG`  — version 4 (overlying grid),
#   * `objective_min_line_loading_max_OG` — version 3 (overlying grid).
#
# All of them are `JuMP.@objective` calls, so what is observable is the sense of
# the model, the type of the objective expression (affine or quadratic) and the
# coefficient every single variable ends up with. The tests therefore pin the
# coefficients exactly and — just as importantly — assert that the variables the
# objective must *not* touch (storage branches, slack generators, flexibility
# variables of another version) carry a coefficient of zero.
#
# The four objectives used by the build are tested on the model
# `TestUtils.build_small_model` produces for the corresponding `opf_version`;
# `objective_min_losses` gets a build method of its own (`objective_losses_build`)
# that only creates the variables it reads.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"""
The affine part of the objective of `model`, no matter whether the objective is
an `AffExpr` or a `QuadExpr`.
"""
objective_affine(model::JuMP.Model) =
    (f = JuMP.objective_function(model); f isa JuMP.QuadExpr ? f.aff : f)

"Coefficient of `v` in the linear part of the objective (`0.0` if absent)."
objective_coeff(model::JuMP.Model, v) = JuMP.coefficient(objective_affine(model), v)

"Coefficient of `v^2` in the objective (`0.0` if absent or the objective is affine)."
function objective_quad_coeff(model::JuMP.Model, v)
    f = JuMP.objective_function(model)
    return f isa JuMP.QuadExpr ? JuMP.coefficient(f, v, v) : 0.0
end

"Number of linear terms of the objective."
objective_num_linear(model::JuMP.Model) = length(objective_affine(model).terms)

"Number of quadratic terms of the objective."
function objective_num_quadratic(model::JuMP.Model)
    f = JuMP.objective_function(model)
    return f isa JuMP.QuadExpr ? length(f.terms) : 0
end

"Constant term of the objective."
objective_constant(model::JuMP.Model) = objective_affine(model).constant

"""
Run `f` with `stdout` redirected and return `(result of f, everything it printed)`.

`objective_min_line_loading_max_OG` writes two debug lines to `stdout` on every
call (see the `@testset` below); this keeps them out of the test output and makes
them assertable.
"""
function objective_capture_stdout(f)
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
Build method that only creates the variables `objective_min_losses` reads: the
branch flows `p`/`q` and the squared branch currents `ccm`. The objective is not
used by any `opf_version`, so it cannot be tested through `build_mn_opf_bf_flex`.
"""
function objective_losses_build(pm)
    for (n, _network) in PowerModels.nws(pm)
        eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false)
        eDisGo_OPF.variable_branch_current(pm, nw=n)
    end
end

"Instantiate [`objective_losses_build`](@ref) on the multi-network form of `data`."
objective_losses_model(data) = eDisGo_OPF.instantiate_model(
    TestUtils.multinetwork(data), eDisGo_OPF.SOCBFPowerModelEdisgo,
    objective_losses_build)

"""
Instantiate `build_mn_opf_bf_flex` on the multi-network form of `data` — the same
model `TestUtils.build_small_model` returns, but for a network that the test
modified first.
"""
objective_flex_model(data) = eDisGo_OPF.instantiate_model(
    TestUtils.multinetwork(data), eDisGo_OPF.SOCBFPowerModelEdisgo,
    eDisGo_OPF.build_mn_opf_bf_flex)

"HV requirement name → its index in network `nw` of `pm`."
objective_hv_ids(pm, nw::Int=1) =
    Dict(flex["name"] => i for (i, flex) in PowerModels.ref(pm, nw, :HV_requirements))

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@testset "objective_min_losses" begin
    data = TestUtils.small_network(num_steps=2)
    # Distinct resistances so a coefficient can only be right if it was taken
    # from the matching branch.
    data["branch"]["1"]["br_r"] = 0.02
    data["branch"]["2"]["br_r"] = 0.05
    data["branch"]["3"]["br_r"] = 0.07   # the virtual storage branch
    pm = objective_losses_model(data)
    obj = eDisGo_OPF.objective_min_losses(pm)

    @testset "sense, type and return value" begin
        @test JuMP.objective_sense(pm.model) == MOI.MIN_SENSE
        # Σ ccm·r is linear in the model variables.
        @test JuMP.objective_function_type(pm.model) == JuMP.AffExpr
        # The function hands back exactly the expression it installed.
        @test obj == JuMP.objective_function(pm.model)
        @test obj isa JuMP.AffExpr
        @test objective_constant(pm.model) == 0.0
    end

    @testset "coefficients" begin
        for n in 1:2
            ccm = PowerModels.var(pm, n, :ccm)
            @test objective_coeff(pm.model, ccm[1]) == 0.02
            @test objective_coeff(pm.model, ccm[2]) == 0.05
            # Negative case: branch 3 is a virtual storage branch and is skipped
            # (`if storage[b] == 0`), even though its resistance is the largest.
            @test objective_coeff(pm.model, ccm[3]) == 0.0
            @test !haskey(objective_affine(pm.model).terms, ccm[3])
        end
        # Two branches × two time steps and nothing else — in particular none of
        # the `p`/`q` flow variables the function looks up but never uses.
        @test objective_num_linear(pm.model) == 4
        @test objective_num_quadratic(pm.model) == 0
        for n in 1:2, arc in PowerModels.ref(pm, n, :arcs_from)
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :p)[arc]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :q)[arc]) == 0.0
        end
    end

    @testset "sum over all time steps" begin
        # Every network contributes its own `ccm` variables; the number of terms
        # grows with the number of time steps, the coefficients do not.
        for steps in (1, 3)
            d = TestUtils.small_network(num_steps=steps)
            pm_s = objective_losses_model(d)
            eDisGo_OPF.objective_min_losses(pm_s)
            @test objective_num_linear(pm_s.model) == 2 * steps
            @test sort(collect(PowerModels.nw_ids(pm_s))) == collect(1:steps)
            for n in 1:steps
                ccm = PowerModels.var(pm_s, n, :ccm)
                @test objective_coeff(pm_s.model, ccm[1]) == 0.01
                @test objective_coeff(pm_s.model, ccm[2]) == 0.01
                @test objective_coeff(pm_s.model, ccm[3]) == 0.0
            end
        end
    end

    @testset "per-network resistances" begin
        # `r` is read per network (`ref(pm, n, :branch)`), so a resistance that
        # differs between time steps shows up in the respective coefficient.
        d = TestUtils.small_network(num_steps=2)
        mn = TestUtils.multinetwork(d)
        mn["nw"]["2"]["branch"]["1"]["br_r"] = 0.5
        pm_r = eDisGo_OPF.instantiate_model(mn, eDisGo_OPF.SOCBFPowerModelEdisgo,
                                            objective_losses_build)
        eDisGo_OPF.objective_min_losses(pm_r)
        @test objective_coeff(pm_r.model, PowerModels.var(pm_r, 1, :ccm)[1]) == 0.01
        @test objective_coeff(pm_r.model, PowerModels.var(pm_r, 2, :ccm)[1]) == 0.5
    end

    @testset "defaults for missing fields" begin
        # `br_r` is read with `get(branch, "br_r", 1.0)` …
        d = TestUtils.small_network(num_steps=1)
        delete!(d["branch"]["1"], "br_r")
        pm_d = objective_losses_model(d)
        eDisGo_OPF.objective_min_losses(pm_d)
        @test objective_coeff(pm_d.model, PowerModels.var(pm_d, 1, :ccm)[1]) == 1.0
        @test objective_coeff(pm_d.model, PowerModels.var(pm_d, 1, :ccm)[2]) == 0.01

        # … and `storage` with `get(branch, "storage", 1.0)`, so a branch without
        # the flag is treated like a storage branch and dropped from the sum.
        d2 = TestUtils.small_network(num_steps=1)
        delete!(d2["branch"]["2"], "storage")
        pm_d2 = objective_losses_model(d2)
        eDisGo_OPF.objective_min_losses(pm_d2)
        @test objective_coeff(pm_d2.model, PowerModels.var(pm_d2, 1, :ccm)[1]) == 0.01
        @test objective_coeff(pm_d2.model, PowerModels.var(pm_d2, 1, :ccm)[2]) == 0.0
        @test objective_num_linear(pm_d2.model) == 1
    end
end

@testset "objective_min_losses_slacks" begin
    # `opf_version` 2 — the objective `build_mn_opf_bf_flex` installs.
    pm = TestUtils.build_small_model(opf_version=2, num_steps=2)
    factor_slacks = 0.6

    @testset "sense, type and return value" begin
        @test JuMP.objective_sense(pm.model) == MOI.MIN_SENSE
        # Losses and slacks are all linear.
        @test JuMP.objective_function_type(pm.model) == JuMP.AffExpr
        @test objective_constant(pm.model) == 0.0
        @test objective_num_quadratic(pm.model) == 0
        # Re-running the objective returns exactly the expression it sets.
        obj = eDisGo_OPF.objective_min_losses_slacks(pm)
        @test obj == JuMP.objective_function(pm.model)
    end

    @testset "loss term" begin
        # Unlike `objective_min_losses` this one sums over *all* branches — the
        # virtual storage branch 3 included ("incl. storage losses").
        for n in 1:2, b in 1:3
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :ccm)[b]) ≈
                  (1 - factor_slacks) * 0.01 atol = 1e-12
        end
    end

    @testset "slack terms" begin
        for n in 1:2
            # One component of each type in the fixture, all weighted with
            # `factor_slacks`.
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pgc)[1]) == factor_slacks
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pgens)[1]) == factor_slacks
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pds)[1]) == factor_slacks
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pcps)[1]) == factor_slacks
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phps)[1]) == factor_slacks
            # The heat-pump / heat-storage operation slacks get the big penalty.
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phss)[1]) == 1e4
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phps2)[1]) == 1e4
        end
        # 3 branches + 7 slacks, per time step.
        @test objective_num_linear(pm.model) == 2 * (3 + 7)
    end

    @testset "variables outside the objective" begin
        # Everything the model carries but the objective must not price.
        for n in 1:2
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pgs)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :qgs)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :php)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pcp)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pdsm)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :ps)[1]) == 0.0
            # `w` exists for every non-storage bus.
            w = PowerModels.var(pm, n, :w)
            @test Set(w.axes[1]) == Set([1, 2, 3])
            for i in w.axes[1]
                @test objective_coeff(pm.model, w[i]) == 0.0
            end
        end
        # `ll` only exists for versions 1 and 3.
        @test !haskey(PowerModels.var(pm, 1), :ll)
        @test !haskey(PowerModels.var(pm, 1), :phvs)
    end

    @testset "sum over all time steps" begin
        for steps in (1, 3)
            pm_s = TestUtils.build_small_model(opf_version=2, num_steps=steps)
            @test objective_num_linear(pm_s.model) == steps * (3 + 7)
            for n in 1:steps
                @test objective_coeff(pm_s.model, PowerModels.var(pm_s, n, :pds)[1]) ==
                      factor_slacks
                @test objective_coeff(pm_s.model, PowerModels.var(pm_s, n, :phss)[1]) == 1e4
            end
        end
    end

    @testset "per-branch resistances" begin
        d = TestUtils.small_network(opf_version=2, num_steps=1)
        d["branch"]["1"]["br_r"] = 0.02
        d["branch"]["2"]["br_r"] = 0.05
        d["branch"]["3"]["br_r"] = 0.07
        pm_r = objective_flex_model(d)
        for (b, r) in ((1, 0.02), (2, 0.05), (3, 0.07))
            @test objective_coeff(pm_r.model, PowerModels.var(pm_r, 1, :ccm)[b]) ≈
                  (1 - factor_slacks) * r atol = 1e-12
        end
    end

    # NOTE (production code): the `phss` term is indexed by the *heat pump* ids
    # (`for i in keys(ref(pm, 1, :heatpumps))`) although `phss` is created per
    # `heat_storage`. Both fixtures happen to use identical ids, so the mismatch
    # is invisible here — a grid whose heat storages are numbered differently
    # from its heat pumps would raise a `KeyError` in the objective.
end
