# Tests for `src/core/constraint.jl`.
#
# The file holds the six constraint functions that are *not* dispatched on the
# formulation: the storage/DSM state-of-charge recursions, the charging-park
# energy recursion, the heat-pump energy balance and the overlying-grid
# requirement. They all take an already instantiated model, look their operands up
# in `PowerModels.var`/`PowerModels.ref` and append one or two `JuMP` constraints.
#
# They are therefore tested in isolation: a model is instantiated with
# `cst_build_vars` below, which creates *only* the variables the six functions
# touch and no constraint at all, so every constraint appearing afterwards is one
# the function under test added. The added constraints are pulled out with
# `cst_new_constraints` and compared coefficient by coefficient — string
# comparisons would not tell `-0.25 * ps` from `-0.25000001 * ps`.
#
# Known production bug (see also test/FUNKTIONSABDECKUNG.md): `_LOGGER` is not
# defined in `eDisGo_OPF`, so the `time_elapsed` fallback paths of
# `constraint_cp_state_initial`/`constraint_cp_state` throw an `UndefVarError`
# instead of warning. The tests below pin the actual behaviour and keep the
# intended one as `@test_broken`.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

"""
Build method creating only the variables `src/core/constraint.jl` operates on:
`ps`/`se` (battery), `phs`/`hse` (heat storage), `pdsm`/`dsme` (DSM), `pcp`/`cpe`
(charging parks), `php` (heat pumps), `phss`/`phps2` (heat slacks), `phvs` (HV
requirement slack) and `pgc` (curtailment). No constraint is created, only the
variable bounds — those live in a different `(F, S)` bucket than the affine
equalities the constraint functions add and never get in the way.
"""
function cst_build_vars(pm)
    for (n, _) in PowerModels.nws(pm)
        eDisGo_OPF.variable_battery_storage(pm, nw=n)          # ps, se
        eDisGo_OPF.variable_heat_storage(pm, nw=n)             # phs, hse
        eDisGo_OPF.variable_dsm_storage_power(pm, nw=n)        # pdsm, dsme
        eDisGo_OPF.variable_cp_power(pm, nw=n)                 # pcp, cpe
        eDisGo_OPF.variable_heat_pump_power(pm, nw=n)          # php
        eDisGo_OPF.variable_slack_heat_pump_storage(pm, nw=n)  # phss, phps2
        eDisGo_OPF.variable_gen_power_curt(pm, nw=n)           # pgc
        eDisGo_OPF.variable_slack_HV_requirements(pm, nw=n)    # phvs
    end
end

"Fresh model on [`TestUtils.small_network`](@ref) carrying only `cst_build_vars`' variables."
cst_model(; kwargs...) = TestUtils.build_small_model(
    eDisGo_OPF.NCBFPowerModelEdisgo; build_method=cst_build_vars, kwargs...)

"Fresh model on the real eDisGo test grid carrying only `cst_build_vars`' variables."
cst_big_model(; kwargs...) = TestUtils.build_model(
    eDisGo_OPF.NCBFPowerModelEdisgo; build_method=cst_build_vars, kwargs...)

"All affine equality constraints of `model`, in the order they were added."
cst_eq_constraints(model) = JuMP.all_constraints(model, JuMP.AffExpr,
                                                 MOI.EqualTo{Float64})

"""
    cst_new_constraints(f, model)

Run `f()` and return the affine equality constraints it appended to `model`, in
order. Everything the constraint functions of `src/core/constraint.jl` create is
an affine equality, so this isolates exactly their effect.
"""
function cst_new_constraints(f, model)
    before = length(cst_eq_constraints(model))
    f()
    return cst_eq_constraints(model)[(before + 1):end]
end

"""
    cst_coeffs(con)

The linear terms of the affine equality `con` as a `variable => coefficient` dict.
Every variable that does *not* appear in the constraint is absent from the dict,
which makes it a complete description: comparing against an expected dict asserts
the presence of the wanted terms and the absence of all others at once.
"""
function cst_coeffs(con)
    func = JuMP.constraint_object(con).func
    return Dict{JuMP.VariableRef,Float64}(v => JuMP.coefficient(func, v)
                                          for (_, v) in JuMP.linear_terms(func))
end

"Right-hand side of the affine equality `con` in JuMP's normalised form."
cst_rhs(con) = JuMP.normalized_rhs(con)

"""
    cst_check(con, expected, rhs)

Assert that the affine equality `con` is exactly `sum(coeff * var) == rhs`:
the same variables, the same coefficients and the same constant term.
"""
function cst_check(con, expected::Dict, rhs::Real; atol::Real=1e-12)
    coeffs = cst_coeffs(con)
    @test Set(keys(coeffs)) == Set(keys(expected))
    for (v, c) in expected
        @test haskey(coeffs, v) && isapprox(coeffs[v], c; atol=atol)
    end
    @test isapprox(cst_rhs(con), rhs; atol=atol)
end

@testset "constraint_store_state_initial" begin
    # Three time steps: the "end" constraint refers to network 3, so it is
    # distinguishable from the network the function is called for (1).
    @testset "storage" begin
        pm = cst_model(num_steps=3)
        # Distinct, non-zero values so a swapped/ignored parameter is visible.
        PowerModels.ref(pm, 1, :storage)[1]["soc_initial"] = 0.15
        PowerModels.ref(pm, 1, :storage)[1]["soc_end"] = 0.05
        # The end state is read from network `n`, not from the last network — a
        # different value there must not show up in the constraint.
        PowerModels.ref(pm, 3, :storage)[1]["soc_end"] = 0.99

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state_initial(pm, 1, 1, 0.7, 0.9, 0.8, 0.25,
                                                      "storage", 0.04)
        end
        @test length(cons) == 2

        # Eq. (3.10) for t = 1: se - soc_initial == -time_elapsed * ps
        #                  ⇔   se + 0.25 * ps == 0.15
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 1, :se, 1) => 1.0,
                       PowerModels.var(pm, 1, :ps, 1) => 0.25),
                  0.15)
        # Eq. (3.9) for t = tau: se[nw 3] == soc_end
        cst_check(cons[2], Dict(PowerModels.var(pm, 3, :se, 1) => 1.0), 0.05)
    end

    @testset "storage ignores energy, efficiencies and p_loss" begin
        # The battery branch takes its state from the `ref`, so the `energy`,
        # `charge_eff`, `discharge_eff` and `p_loss` arguments must not appear.
        pm = cst_model(num_steps=2)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state_initial(pm, 1, 1, 42.0, 0.1, 0.2, 1.0,
                                                      "storage", 0.5)
        end
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 1, :se, 1) => 1.0,
                       PowerModels.var(pm, 1, :ps, 1) => 1.0),
                  0.0)
        cst_check(cons[2], Dict(PowerModels.var(pm, 2, :se, 1) => 1.0), 0.0)
    end

    @testset "heat_storage" begin
        pm = cst_model(num_steps=3)
        PowerModels.ref(pm, 1, :heat_storage)[1]["soc_initial"] = 1.25
        PowerModels.ref(pm, 1, :heat_storage)[1]["soc_end"] = 2.5
        PowerModels.ref(pm, 3, :heat_storage)[1]["soc_end"] = 0.99

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state_initial(pm, 1, 1, 0.7, 1.0, 1.0, 0.5,
                                                      "heat_storage", 0.04)
        end
        @test length(cons) == 2

        # Eq. (3.23) for t = 1: hse + time_elapsed * phs == soc_initial. The
        # standing loss `p_loss` is deliberately absent from the first step.
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 1, :hse, 1) => 1.0,
                       PowerModels.var(pm, 1, :phs, 1) => 0.5),
                  1.25)
        # Eq. (3.22) for t = tau.
        cst_check(cons[2], Dict(PowerModels.var(pm, 3, :hse, 1) => 1.0), 2.5)
    end

    @testset "dsm" begin
        pm = cst_model(num_steps=3)
        # Unlike the two storage branches, DSM takes the initial energy from the
        # `energy` argument — the `ref` entry is not consulted.
        PowerModels.ref(pm, 1, :dsm)[1]["energy"] = 99.0

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state_initial(pm, 1, 1, 0.12, 1.0, 1.0, 0.25,
                                                      "dsm", 0)
        end
        @test length(cons) == 2

        # Eq. (3.33) for t = 1: dsme - energy == +time_elapsed * pdsm
        #                  ⇔   dsme - 0.25 * pdsm == 0.12
        # Note the opposite sign of the power term compared with the two storage
        # branches: DSM counts a positive `pdsm` as *charging* the shift energy.
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 1, :dsme, 1) => 1.0,
                       PowerModels.var(pm, 1, :pdsm, 1) => -0.25),
                  0.12)
        # Eq. (3.32) for t = tau: the shifted energy has to be balanced out; the
        # end value is hard-coded to 0 rather than read from the data.
        cst_check(cons[2], Dict(PowerModels.var(pm, 3, :dsme, 1) => 1.0), 0.0)
    end

    @testset "unknown kind" begin
        pm = cst_model(num_steps=2)
        before = TestUtils.num_constraints_total(pm.model)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state_initial(pm, 1, 1, 0.0, 1.0, 1.0, 1.0,
                                                      "battery", 0)
        end
        # No branch matches, so the call is silently a no-op — no constraint of any
        # kind is added and no error is raised.
        @test isempty(cons)
        @test TestUtils.num_constraints_total(pm.model) == before
    end
end

@testset "constraint_store_state" begin
    # Four time steps, coupling 2 → 3: neither network is the first or the last
    # one, so a constraint that silently used `1` or `end` would stand out.
    @testset "storage" begin
        pm = cst_model(num_steps=4)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state(pm, 2, 3, 1, 0.9, 0.8, 0.25, "storage",
                                              0.04)
        end
        @test length(cons) == 1
        # Eq. (3.10): se_2 - se_1 == -time_elapsed * ps_2
        #        ⇔    se[nw3] - se[nw2] + 0.25 * ps[nw3] == 0
        # The efficiencies and the standing loss are not part of the battery
        # recursion.
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 3, :se, 1) => 1.0,
                       PowerModels.var(pm, 2, :se, 1) => -1.0,
                       PowerModels.var(pm, 3, :ps, 1) => 0.25),
                  0.0)
    end

    @testset "heat_storage" begin
        pm = cst_model(num_steps=4)
        p_loss = 0.04
        # Standing loss over one hour of a daily loss rate: (1 - p_loss)^(1/24).
        loss_factor = (1 - p_loss)^(1 / 24)
        @test loss_factor ≈ 0.9983005 atol = 1e-7

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state(pm, 2, 3, 1, 1.0, 1.0, 0.5,
                                              "heat_storage", p_loss)
        end
        @test length(cons) == 1
        # Eq. (3.23): hse_2 - hse_1 * (1 - p_loss)^(1/24) == -time_elapsed * phs_2
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 3, :hse, 1) => 1.0,
                       PowerModels.var(pm, 2, :hse, 1) => -loss_factor,
                       PowerModels.var(pm, 3, :phs, 1) => 0.5),
                  0.0)
        # Nailing the number down independently of the expression above.
        @test cst_coeffs(cons[1])[PowerModels.var(pm, 2, :hse, 1)] ≈ -0.9983005 atol = 1e-7
    end

    @testset "heat_storage without standing loss" begin
        # p_loss = 0 collapses the loss factor to exactly 1, i.e. the same shape as
        # the battery recursion.
        pm = cst_model(num_steps=2)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state(pm, 1, 2, 1, 1.0, 1.0, 1.0,
                                              "heat_storage", 0)
        end
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 2, :hse, 1) => 1.0,
                       PowerModels.var(pm, 1, :hse, 1) => -1.0,
                       PowerModels.var(pm, 2, :phs, 1) => 1.0),
                  0.0)
    end

    @testset "dsm" begin
        pm = cst_model(num_steps=4)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state(pm, 2, 3, 1, 1.0, 1.0, 0.25, "dsm", 0)
        end
        @test length(cons) == 1
        # Eq. (3.33): dsme_2 - dsme_1 == time_elapsed * pdsm_2 — again the sign of
        # the power term is opposite to the two storage branches.
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 3, :dsme, 1) => 1.0,
                       PowerModels.var(pm, 2, :dsme, 1) => -1.0,
                       PowerModels.var(pm, 3, :pdsm, 1) => -0.25),
                  0.0)
    end

    @testset "efficiencies are ignored" begin
        # None of the three branches uses `charge_eff`/`discharge_eff`; extreme
        # values must leave the coefficients untouched.
        pm = cst_model(num_steps=2)
        for (kind, e, p) in (("storage", :se, :ps), ("dsm", :dsme, :pdsm))
            cons = cst_new_constraints(pm.model) do
                eDisGo_OPF.constraint_store_state(pm, 1, 2, 1, 0.1, 0.2, 1.0, kind, 0)
            end
            sign_p = kind == "dsm" ? -1.0 : 1.0
            cst_check(cons[1],
                      Dict(PowerModels.var(pm, 2, e, 1) => 1.0,
                           PowerModels.var(pm, 1, e, 1) => -1.0,
                           PowerModels.var(pm, 2, p, 1) => sign_p),
                      0.0)
        end
    end

    @testset "unknown kind" begin
        pm = cst_model(num_steps=2)
        before = TestUtils.num_constraints_total(pm.model)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state(pm, 1, 2, 1, 1.0, 1.0, 1.0, "battery", 0)
        end
        @test isempty(cons)
        @test TestUtils.num_constraints_total(pm.model) == before
    end
end
