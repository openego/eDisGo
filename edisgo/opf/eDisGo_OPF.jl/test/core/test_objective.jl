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

Used to check that building a model and the objectives print nothing.
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

    # NOTE (production code): the `phss` term is summed over the heat storage
    # ids and `phps2` over the heat pump ids. Heat pump `i` and heat storage `i`
    # are paired by id without a check (#807); both fixtures use identical ids.
end

@testset "objective_min_line_loading_max" begin
    # `opf_version` 1 — losses plus the maximum line loading `ll`.
    pm = TestUtils.build_small_model(opf_version=1, num_steps=2)
    factor_ll = 0.1

    @testset "sense, type and return value" begin
        @test JuMP.objective_sense(pm.model) == MOI.MIN_SENSE
        @test JuMP.objective_function_type(pm.model) == JuMP.AffExpr
        @test objective_num_quadratic(pm.model) == 0
        obj = eDisGo_OPF.objective_min_line_loading_max(pm)
        @test obj == JuMP.objective_function(pm.model)
    end

    @testset "loss term" begin
        # As in version 2 the loss sum runs over all branches, storage included.
        for n in 1:2, b in 1:3
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :ccm)[b]) ≈
                  (1 - factor_ll) * 0.01 atol = 1e-12
        end
    end

    @testset "line loading term" begin
        ll = PowerModels.var(pm, 1, :ll)
        # `cost * length` of the fixture is 10 * 1 for every branch.
        for arc in ((1, 1, 2), (2, 2, 3))
            @test objective_coeff(pm.model, ll[arc]) ≈ factor_ll * 10.0 * 1.0 atol = 1e-12
        end
        # The `-1` of `(ll - 1)` lands in the constant term.
        @test objective_constant(pm.model) ≈ -factor_ll * 2 * 10.0 * 1.0 atol = 1e-12

        # `ll` exists only for the storage-free branches and only in network 1 —
        # the maximum line loading is shared across all time steps.
        @test Set(ll.axes[1]) == Set([(1, 1, 2), (2, 2, 3)])
        @test !haskey(PowerModels.var(pm, 2), :ll)

        # 3 branches × 2 networks plus the two `ll` terms.
        @test objective_num_linear(pm.model) == 2 * 3 + 2
    end

    @testset "cost and length per branch" begin
        # The weight of a branch is its own `cost * length` …
        d = TestUtils.small_network(opf_version=1, num_steps=1)
        d["branch"]["1"]["cost"] = 20.0
        d["branch"]["2"]["length"] = 3.0
        d["branch"]["3"]["cost"] = 1000.0  # storage branch, must not show up
        pm_c = objective_flex_model(d)
        ll = PowerModels.var(pm_c, 1, :ll)
        @test objective_coeff(pm_c.model, ll[(1, 1, 2)]) ≈ factor_ll * 20.0 atol = 1e-12
        @test objective_coeff(pm_c.model, ll[(2, 2, 3)]) ≈ factor_ll * 3 * 10.0 atol = 1e-12
        @test objective_constant(pm_c.model) ≈ -factor_ll * (20.0 + 30.0) atol = 1e-12
        # … and the storage branch has no `ll` variable at all.
        @test Set(ll.axes[1]) == Set([(1, 1, 2), (2, 2, 3)])
    end

    @testset "sum over all time steps" begin
        for steps in (1, 3)
            pm_s = TestUtils.build_small_model(opf_version=1, num_steps=steps)
            # Losses accumulate over the time steps, the two `ll` terms do not.
            @test objective_num_linear(pm_s.model) == 3 * steps + 2
            @test objective_constant(pm_s.model) ≈ -2.0 atol = 1e-12
            for n in 1:steps
                @test objective_coeff(pm_s.model, PowerModels.var(pm_s, n, :ccm)[1]) ≈
                      0.009 atol = 1e-12
            end
        end
    end

    @testset "variables outside the objective" begin
        # Version 1 knows no grid-restriction slacks; the heat-pump storage
        # slacks do exist but are not priced by this objective.
        @test !haskey(PowerModels.var(pm, 1), :pgc)
        @test !haskey(PowerModels.var(pm, 1), :pds)
        for n in 1:2
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phss)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phps2)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pgs)[1]) == 0.0
        end
    end
end

@testset "objective_min_losses_slacks_OG" begin
    # `opf_version` 4 — version 2 plus the overlying-grid (HV) slacks.
    pm = TestUtils.build_small_model(opf_version=4, num_steps=2, overlying_grid=true)
    factor_slacks = 0.6
    hv = objective_hv_ids(pm)
    # `factor_hv_slacks` is derived from the branch resistances of the fixture:
    # one order of magnitude above the largest resistance.
    resistances = [b["br_r"] for b in values(PowerModels.ref(pm, 1, :branch))]
    factor_hv_slacks = exp10(floor(log10(maximum(resistances))) + 1)
    @test factor_hv_slacks == 0.1

    @testset "sense, type and return value" begin
        @test JuMP.objective_sense(pm.model) == MOI.MIN_SENSE
        # The squared HV slacks make the objective quadratic.
        @test JuMP.objective_function_type(pm.model) == JuMP.QuadExpr
        @test objective_constant(pm.model) == 0.0
        obj = eDisGo_OPF.objective_min_losses_slacks_OG(pm)
        @test obj == JuMP.objective_function(pm.model)
    end

    @testset "linear part equals objective_min_losses_slacks" begin
        # Everything but the HV terms is the version 2 objective, term for term.
        pm2 = TestUtils.build_small_model(opf_version=2, num_steps=2)
        @test objective_num_linear(pm.model) == objective_num_linear(pm2.model)
        for n in 1:2
            for b in 1:3
                @test objective_coeff(pm.model, PowerModels.var(pm, n, :ccm)[b]) ≈
                      (1 - factor_slacks) * 0.01 atol = 1e-12
            end
            for key in (:pgc, :pgens, :pds, :pcps, :phps)
                @test objective_coeff(pm.model, PowerModels.var(pm, n, key)[1]) ==
                      factor_slacks
            end
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phss)[1]) == 1e4
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phps2)[1]) == 1e4
        end
    end

    @testset "HV slack terms" begin
        # `count` of every requirement of the fixture is the number of time steps.
        count = PowerModels.ref(pm, 1, :HV_requirements, hv["cp"])["count"]
        @test count == 2
        for n in 1:2
            phvs = PowerModels.var(pm, n, :phvs)
            for name in ("curt", "cp", "hp", "storage")
                @test objective_quad_coeff(pm.model, phvs[hv[name]]) ≈
                      factor_hv_slacks * count atol = 1e-12
            end
            # DSM requirements are weighted down by a factor of 1e-1.
            @test objective_quad_coeff(pm.model, phvs[hv["dsm"]]) ≈
                  factor_hv_slacks * 1e-1 * count atol = 1e-12
            # The slacks enter squared only, never linearly.
            for i in PowerModels.ids(pm, n, :HV_requirements)
                @test objective_coeff(pm.model, phvs[i]) == 0.0
            end
        end
        # Five requirements per time step, and no cross terms.
        @test objective_num_quadratic(pm.model) == 5 * 2
    end

    @testset "count enters per requirement" begin
        d = TestUtils.small_network(opf_version=4, num_steps=2, overlying_grid=true)
        for req in values(d["HV_requirements"])
            req["count"] = req["name"] == "cp" ? 7 : 3
        end
        pm_c = objective_flex_model(d)
        hv_c = objective_hv_ids(pm_c)
        phvs = PowerModels.var(pm_c, 1, :phvs)
        @test objective_quad_coeff(pm_c.model, phvs[hv_c["cp"]]) ≈ 0.1 * 7 atol = 1e-12
        @test objective_quad_coeff(pm_c.model, phvs[hv_c["hp"]]) ≈ 0.1 * 3 atol = 1e-12
        @test objective_quad_coeff(pm_c.model, phvs[hv_c["dsm"]]) ≈
              0.1 * 1e-1 * 3 atol = 1e-12
    end

    @testset "factor_hv_slacks scales with the resistances" begin
        # `exp10(floor(log10(max r)) + 1)`: only the order of magnitude of the
        # largest branch resistance matters, and the storage branch counts too.
        for (r1, r2, r3, expected) in ((0.02, 0.3, 0.01, 1.0),
                                       (0.02, 0.01, 0.3, 1.0),
                                       (0.001, 0.002, 0.003, 0.01))
            d = TestUtils.small_network(opf_version=4, num_steps=1, overlying_grid=true)
            d["branch"]["1"]["br_r"] = r1
            d["branch"]["2"]["br_r"] = r2
            d["branch"]["3"]["br_r"] = r3
            pm_f = objective_flex_model(d)
            hv_f = objective_hv_ids(pm_f)
            phvs = PowerModels.var(pm_f, 1, :phvs)
            @test exp10(floor(log10(max(r1, r2, r3))) + 1) ≈ expected atol = 1e-12
            @test objective_quad_coeff(pm_f.model, phvs[hv_f["cp"]]) ≈
                  expected * 1 atol = 1e-12
            @test objective_quad_coeff(pm_f.model, phvs[hv_f["dsm"]]) ≈
                  expected * 1e-1 * 1 atol = 1e-12
        end
    end

    @testset "sum over all time steps" begin
        for steps in (1, 3)
            pm_s = TestUtils.build_small_model(opf_version=4, num_steps=steps,
                                               overlying_grid=true)
            @test objective_num_quadratic(pm_s.model) == 5 * steps
            @test objective_num_linear(pm_s.model) == steps * (3 + 7)
            hv_s = objective_hv_ids(pm_s)
            for n in 1:steps
                @test objective_quad_coeff(pm_s.model,
                                           PowerModels.var(pm_s, n, :phvs)[hv_s["hp"]]) ≈
                      0.1 * steps atol = 1e-12
            end
        end
    end

    @testset "no line loading term" begin
        # Version 4 has no `ll` variable — the maximum line loading is only part
        # of versions 1 and 3.
        @test !haskey(PowerModels.var(pm, 1), :ll)
        for n in 1:2
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pgs)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pdsm)[1]) == 0.0
        end
    end
end

@testset "objective_min_line_loading_max_OG" begin
    # `opf_version` 3 — version 1 plus the overlying-grid (HV) slacks. Building
    # the model already calls the objective; the build output is checked below.
    pm, build_output = objective_capture_stdout(
        () -> TestUtils.build_small_model(opf_version=3, num_steps=2,
                                          overlying_grid=true))
    factor_ll = 0.01
    hv = objective_hv_ids(pm)
    # `factor_hv_slacks` is derived from `0.01 * cost * length` over *all*
    # branches (the storage branch included), scaled by 7.5.
    weights = [b["cost"] * b["length"] for b in values(PowerModels.ref(pm, 1, :branch))]
    factor_hv_slacks = 7.5 * exp10(floor(log10(maximum(0.01 .* weights))) + 1)
    @test factor_hv_slacks == 7.5

    @testset "no output" begin
        # Neither building the model nor the objective itself prints anything
        # (it used to print `factor_ll` and `factor_hv_slacks`, #803).
        @test build_output == ""
        _obj, out = objective_capture_stdout(
            () -> eDisGo_OPF.objective_min_line_loading_max_OG(pm))
        @test out == ""
    end

    @testset "sense, type and return value" begin
        @test JuMP.objective_sense(pm.model) == MOI.MIN_SENSE
        @test JuMP.objective_function_type(pm.model) == JuMP.QuadExpr
        # The function hands back exactly the expression it installed.
        obj = eDisGo_OPF.objective_min_line_loading_max_OG(pm)
        @test obj isa JuMP.QuadExpr
        @test obj == JuMP.objective_function(pm.model)
    end

    @testset "loss and line loading terms" begin
        # Same shape as version 1, but with `factor_ll` 0.01 instead of 0.1.
        for n in 1:2, b in 1:3
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :ccm)[b]) ≈
                  (1 - factor_ll) * 0.01 atol = 1e-12
        end
        ll = PowerModels.var(pm, 1, :ll)
        for arc in ((1, 1, 2), (2, 2, 3))
            @test objective_coeff(pm.model, ll[arc]) ≈ factor_ll * 10.0 atol = 1e-12
        end
        @test objective_constant(pm.model) ≈ -factor_ll * 2 * 10.0 atol = 1e-12
        @test Set(ll.axes[1]) == Set([(1, 1, 2), (2, 2, 3)])
        @test !haskey(PowerModels.var(pm, 2), :ll)
        @test objective_num_linear(pm.model) == 2 * 3 + 2
    end

    @testset "HV slack terms" begin
        for n in 1:2
            phvs = PowerModels.var(pm, n, :phvs)
            for name in ("curt", "cp", "hp", "storage")
                @test objective_quad_coeff(pm.model, phvs[hv[name]]) ≈
                      factor_hv_slacks atol = 1e-12
            end
            @test objective_quad_coeff(pm.model, phvs[hv["dsm"]]) ≈
                  factor_hv_slacks * 1e-1 atol = 1e-12
            for i in PowerModels.ids(pm, n, :HV_requirements)
                @test objective_coeff(pm.model, phvs[i]) == 0.0
            end
        end
        @test objective_num_quadratic(pm.model) == 5 * 2

        # Unlike `objective_min_losses_slacks_OG` this objective ignores the
        # `count` of a requirement.
        d = TestUtils.small_network(opf_version=3, num_steps=2, overlying_grid=true)
        for req in values(d["HV_requirements"])
            req["count"] = 9
        end
        pm_c = objective_flex_model(d)
        hv_c = objective_hv_ids(pm_c)
        @test objective_quad_coeff(pm_c.model,
                                   PowerModels.var(pm_c, 1, :phvs)[hv_c["cp"]]) ≈
              7.5 atol = 1e-12
    end

    @testset "factor_hv_slacks scales with cost × length" begin
        # `7.5 * exp10(floor(log10(max 0.01·cost·length)) + 1)`, over all
        # branches — the storage branch included.
        for (costs, expected) in (((500.0, 10.0, 10.0), 75.0),
                                  # the virtual storage branch counts as well
                                  ((10.0, 10.0, 500.0), 75.0),
                                  ((0.5, 0.5, 0.5), 7.5 * 0.01))
            d = TestUtils.small_network(opf_version=3, num_steps=1, overlying_grid=true)
            for (b, cost) in zip(("1", "2", "3"), costs)
                d["branch"][b]["cost"] = cost
            end
            pm_f = objective_flex_model(d)
            hv_f = objective_hv_ids(pm_f)
            phvs = PowerModels.var(pm_f, 1, :phvs)
            @test 7.5 * exp10(floor(log10(0.01 * maximum(costs))) + 1) ≈
                  expected atol = 1e-12
            @test objective_quad_coeff(pm_f.model, phvs[hv_f["cp"]]) ≈
                  expected atol = 1e-12
            @test objective_quad_coeff(pm_f.model, phvs[hv_f["dsm"]]) ≈
                  expected * 1e-1 atol = 1e-12
        end
    end

    @testset "sum over all time steps" begin
        for steps in (1, 3)
            pm_s = TestUtils.build_small_model(opf_version=3, num_steps=steps,
                                               overlying_grid=true)
            @test objective_num_quadratic(pm_s.model) == 5 * steps
            # Losses per time step plus the two time-step independent `ll` terms.
            @test objective_num_linear(pm_s.model) == 3 * steps + 2
            @test objective_constant(pm_s.model) ≈ -0.2 atol = 1e-12
        end
    end

    @testset "variables outside the objective" begin
        # Version 3 creates the curtailment variable `pgc` (for the HV
        # requirement constraint) but does not price it.
        @test haskey(PowerModels.var(pm, 1), :pgc)
        for n in 1:2
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pgc)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phss)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :phps2)[1]) == 0.0
            @test objective_coeff(pm.model, PowerModels.var(pm, n, :pgs)[1]) == 0.0
        end
        # No grid-restriction slacks in version 3.
        @test !haskey(PowerModels.var(pm, 1), :pds)
        @test !haskey(PowerModels.var(pm, 1), :pcps)
    end
end
