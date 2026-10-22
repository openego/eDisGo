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
        time_elapsed = 0.5
        # Standing loss of a daily loss rate over one time step of `time_elapsed`
        # hours: (1 - p_loss)^(time_elapsed/24).
        loss_factor = (1 - p_loss)^(time_elapsed / 24)
        @test loss_factor ≈ 0.9991499 atol = 1e-7

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_store_state(pm, 2, 3, 1, 1.0, 1.0, time_elapsed,
                                              "heat_storage", p_loss)
        end
        @test length(cons) == 1
        # Eq. (3.23): hse_2 - hse_1 * (1 - p_loss)^(time_elapsed/24)
        #                 == -time_elapsed * phs_2
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 3, :hse, 1) => 1.0,
                       PowerModels.var(pm, 2, :hse, 1) => -loss_factor,
                       PowerModels.var(pm, 3, :phs, 1) => time_elapsed),
                  0.0)
        # Nailing the number down independently of the expression above.
        @test cst_coeffs(cons[1])[PowerModels.var(pm, 2, :hse, 1)] ≈ -0.9991499 atol = 1e-7
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

"Return the exception `f()` raises, or `nothing` if it returns normally."
function cst_thrown(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

@testset "constraint_cp_state_initial" begin
    @testset "constraint" begin
        pm = cst_model(num_steps=3)
        # Asymmetric flexibility band so the midpoint is neither 0 nor an endpoint,
        # and a different band in the last network to show that the parameters are
        # read from network `n`.
        PowerModels.ref(pm, 1, :electromobility)[1]["e_min"] = -0.2
        PowerModels.ref(pm, 1, :electromobility)[1]["e_max"] = 0.8
        PowerModels.ref(pm, 3, :electromobility)[1]["e_min"] = 5.0
        PowerModels.ref(pm, 3, :electromobility)[1]["e_max"] = 7.0

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_cp_state_initial(pm, 1, 1, 0.9)
        end
        @test length(cons) == 1
        # Eq. (3.25): cpe == 0.5*(e_min+e_max) + time_elapsed * eta * pcp
        #        ⇔    cpe - 1.0 * 0.9 * pcp == 0.3
        # `time_elapsed` comes from the `ref` (1 in the fixture), `eta` from the
        # argument.
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 1, :cpe, 1) => 1.0,
                       PowerModels.var(pm, 1, :pcp, 1) => -0.9),
                  0.3)
    end

    @testset "time_elapsed from ref" begin
        pm = cst_model(num_steps=2)
        # 15-minute time steps: the power term is scaled down accordingly.
        PowerModels.ref(pm, 1)[:time_elapsed] = 0.25
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_cp_state_initial(pm, 1, 1, 0.8)
        end
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 1, :cpe, 1) => 1.0,
                       PowerModels.var(pm, 1, :pcp, 1) => -0.2),  # -0.25 * 0.8
                  0.5)                                            # 0.5*(0.0+1.0)
    end

    @testset "eta" begin
        # `eta` only scales the power term; the midpoint stays untouched.
        pm = cst_model(num_steps=2)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_cp_state_initial(pm, 1, 1, 1.0)
        end
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 1, :cpe, 1) => 1.0,
                       PowerModels.var(pm, 1, :pcp, 1) => -1.0),
                  0.5)
    end

    @testset "missing time_elapsed" begin
        pm = cst_model(num_steps=2)
        delete!(PowerModels.ref(pm, 1), :time_elapsed)
        @test !haskey(PowerModels.ref(pm, 1), :time_elapsed)
        before = length(cst_eq_constraints(pm.model))

        # BUG: the fallback branch warns via `Memento.warn(_LOGGER, …)`, but
        # `_LOGGER` is not defined in `eDisGo_OPF` — the warning is not the only
        # thing that fails, the whole constraint is never created.
        @test !isdefined(eDisGo_OPF, :_LOGGER)
        @test_throws UndefVarError eDisGo_OPF.constraint_cp_state_initial(pm, 1, 1, 0.9)
        exception = cst_thrown(() -> eDisGo_OPF.constraint_cp_state_initial(pm, 1, 1, 0.9))
        @test exception isa UndefVarError
        @test exception.var === :_LOGGER
        @test length(cst_eq_constraints(pm.model)) == before

        # Intended behaviour: warn once and continue with time_elapsed = 1.0, i.e.
        # produce the very same constraint as the fixture's `time_elapsed == 1`.
        @test_broken (eDisGo_OPF.constraint_cp_state_initial(pm, 1, 1, 0.9);
                      length(cst_eq_constraints(pm.model)) == before + 1)
    end
end

@testset "constraint_cp_state" begin
    @testset "intermediate time step" begin
        pm = cst_model(num_steps=4)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_cp_state(pm, 2, 3, 1, 0.9)
        end
        # 3 is not the last network, so only the recursion is added.
        @test length(cons) == 1
        # Eq. (3.26): cpe_2 - cpe_1 == time_elapsed * eta * pcp_2
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 3, :cpe, 1) => 1.0,
                       PowerModels.var(pm, 2, :cpe, 1) => -1.0,
                       PowerModels.var(pm, 3, :pcp, 1) => -0.9),  # -1.0 * 0.9
                  0.0)
    end

    @testset "last time step" begin
        pm = cst_model(num_steps=4)
        # Band of the last network, used by the additional fixing constraint …
        PowerModels.ref(pm, 4, :electromobility)[1]["e_min"] = 0.4
        PowerModels.ref(pm, 4, :electromobility)[1]["e_max"] = 1.0
        # … and a different one earlier on, which must not be picked up.
        PowerModels.ref(pm, 3, :electromobility)[1]["e_min"] = -9.0
        PowerModels.ref(pm, 3, :electromobility)[1]["e_max"] = -7.0

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_cp_state(pm, 3, 4, 1, 0.5)
        end
        @test length(cons) == 2
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 4, :cpe, 1) => 1.0,
                       PowerModels.var(pm, 3, :cpe, 1) => -1.0,
                       PowerModels.var(pm, 4, :pcp, 1) => -0.5),
                  0.0)
        # Eq. (3.25) for t = tau: the charged energy returns to the midpoint of the
        # flexibility band, 0.5*(0.4+1.0) = 0.7.
        cst_check(cons[2], Dict(PowerModels.var(pm, 4, :cpe, 1) => 1.0), 0.7)
    end

    @testset "two time steps" begin
        # With only two networks the second one is already the last, so a single
        # call produces both constraints.
        pm = cst_model(num_steps=2)
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_cp_state(pm, 1, 2, 1, 1.0)
        end
        @test length(cons) == 2
        cst_check(cons[2], Dict(PowerModels.var(pm, 2, :cpe, 1) => 1.0), 0.5)
    end

    @testset "time_elapsed is taken from n_1" begin
        pm = cst_model(num_steps=3)
        PowerModels.ref(pm, 1)[:time_elapsed] = 0.25
        PowerModels.ref(pm, 2)[:time_elapsed] = 4.0
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_cp_state(pm, 1, 2, 1, 0.8)
        end
        # 0.25 * 0.8, not 4.0 * 0.8 — the first network's value wins.
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 2, :cpe, 1) => 1.0,
                       PowerModels.var(pm, 1, :cpe, 1) => -1.0,
                       PowerModels.var(pm, 2, :pcp, 1) => -0.2),
                  0.0)
    end

    @testset "missing time_elapsed" begin
        pm = cst_model(num_steps=3)
        # Only network 1 loses the entry; network 2 keeps it. The lookup is done on
        # `n_1`, so the fallback is still taken.
        delete!(PowerModels.ref(pm, 1), :time_elapsed)
        @test haskey(PowerModels.ref(pm, 2), :time_elapsed)
        before = length(cst_eq_constraints(pm.model))

        # BUG: same undefined `_LOGGER` as in `constraint_cp_state_initial`.
        @test_throws UndefVarError eDisGo_OPF.constraint_cp_state(pm, 1, 2, 1, 0.9)
        exception = cst_thrown(() -> eDisGo_OPF.constraint_cp_state(pm, 1, 2, 1, 0.9))
        @test exception isa UndefVarError
        @test exception.var === :_LOGGER
        @test length(cst_eq_constraints(pm.model)) == before

        # Intended behaviour: warn and continue with time_elapsed = 1.0.
        @test_broken (eDisGo_OPF.constraint_cp_state(pm, 1, 2, 1, 0.9);
                      length(cst_eq_constraints(pm.model)) == before + 1)
    end
end

@testset "constraint_hp_operation" begin
    @testset "constraint" begin
        pm = cst_model(num_steps=2)
        hp = PowerModels.ref(pm, 1, :heatpumps, 1)
        @test hp["cop"] == 5.0 && hp["pd"] == 1.0

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_hp_operation(pm, 1, 1)
        end
        @test length(cons) == 1
        # Eq. (3.19): cop * (php + phps2) == pd + phss - phs
        #        ⇔    5*php + 5*phps2 + phs - phss == 1.0
        # Both the electrical power and the operation slack are multiplied by the
        # COP; the heat-storage terms are not.
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 1, :php, 1) => 5.0,
                       PowerModels.var(pm, 1, :phps2, 1) => 5.0,
                       PowerModels.var(pm, 1, :phs, 1) => 1.0,
                       PowerModels.var(pm, 1, :phss, 1) => -1.0),
                  1.0)
    end

    @testset "cop and pd from the ref of nw" begin
        pm = cst_model(num_steps=2)
        # Network 2 gets different parameters; network 1 keeps the fixture's.
        PowerModels.ref(pm, 2, :heatpumps, 1)["cop"] = 3.0
        PowerModels.ref(pm, 2, :heatpumps, 1)["pd"] = 0.25

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_hp_operation(pm, 1, 2)
        end
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 2, :php, 1) => 3.0,
                       PowerModels.var(pm, 2, :phps2, 1) => 3.0,
                       PowerModels.var(pm, 2, :phs, 1) => 1.0,
                       PowerModels.var(pm, 2, :phss, 1) => -1.0),
                  0.25)
    end

    @testset "zero heat demand" begin
        # A vanishing `pd` leaves the balance homogeneous — the right-hand side is
        # the demand and nothing else.
        pm = cst_model(num_steps=2)
        PowerModels.ref(pm, 1, :heatpumps, 1)["pd"] = 0.0
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_hp_operation(pm, 1, 1)
        end
        @test cst_rhs(cons[1]) == 0.0
    end

    @testset "one constraint per heat pump" begin
        # The real grid has several heat pumps; each constraint must reference the
        # variables of its own unit and no other.
        pm = cst_big_model()
        hp_ids = sort(collect(PowerModels.ids(pm, :heatpumps, nw=1)))
        @test length(hp_ids) == 3
        for i in hp_ids
            hp = PowerModels.ref(pm, 1, :heatpumps, i)
            cons = cst_new_constraints(pm.model) do
                eDisGo_OPF.constraint_hp_operation(pm, i, 1)
            end
            @test length(cons) == 1
            cst_check(cons[1],
                      Dict(PowerModels.var(pm, 1, :php, i) => hp["cop"],
                           PowerModels.var(pm, 1, :phps2, i) => hp["cop"],
                           PowerModels.var(pm, 1, :phs, i) => 1.0,
                           PowerModels.var(pm, 1, :phss, i) => -1.0),
                      hp["pd"]; atol=1e-9)
        end
    end
end

"""
All variables of type `sym` in network `nw` — the set a HV requirement sums over.
Iterating the `JuMP` container yields its values, in no particular order; every
comparison below is order independent.
"""
function cst_flex_vars(pm, nw, sym)
    vars = JuMP.VariableRef[]
    for v in PowerModels.var(pm, nw, sym)
        push!(vars, v)
    end
    return vars
end

@testset "constraint_HV_requirements" begin
    # `small_network(overlying_grid=true)` carries one requirement per flexibility,
    # in this order: 1 curt, 2 cp, 3 hp, 4 dsm, 5 storage.
    "Requirement id, `hv_req[\"name\"]`, variable summed over and requested power."
    cst_hv_cases = ((1, "curt", :pgc, 0.0), (2, "cp", :pcp, 0.4),
                    (3, "hp", :php, 1.0), (4, "dsm", :pdsm, 0.1),
                    (5, "storage", :ps, -0.35))

    @testset "$name" for (idx, name, sym, p) in cst_hv_cases
        pm = cst_model(num_steps=2, overlying_grid=true)
        @test PowerModels.ref(pm, 1, :HV_requirements, idx)["name"] == name
        @test PowerModels.ref(pm, 1, :HV_requirements, idx)["P"] == p

        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_HV_requirements(pm, idx, 1)
        end
        @test length(cons) == 1
        # sum(pflex) + phvs == P — every flexibility variable of the addressed type
        # enters with coefficient 1, and so does the requirement's own slack.
        expected = Dict{JuMP.VariableRef,Float64}(
            v => 1.0 for v in cst_flex_vars(pm, 1, sym))
        expected[PowerModels.var(pm, 1, :phvs, idx)] = 1.0
        cst_check(cons[1], expected, p)

        # Negative case: the variables of the *other* flexibilities stay out.
        coeffs = cst_coeffs(cons[1])
        for other_sym in (:pgc, :pcp, :php, :pdsm, :ps)
            other_sym === sym && continue
            @test all(!haskey(coeffs, v) for v in cst_flex_vars(pm, 1, other_sym))
        end
        # … as do the slacks of the other requirements.
        for j in 1:5
            j == idx && continue
            @test !haskey(coeffs, PowerModels.var(pm, 1, :phvs, j))
        end
    end

    @testset "sums over all components" begin
        # The real grid has more than one unit of most flexibility types, so the
        # `sum` is actually exercised.
        pm = cst_big_model(overlying_grid=true)
        counts = Dict(1 => length(PowerModels.var(pm, 1, :pgc)),
                      2 => length(PowerModels.var(pm, 1, :pcp)),
                      3 => length(PowerModels.var(pm, 1, :php)),
                      4 => length(PowerModels.var(pm, 1, :pdsm)),
                      5 => length(PowerModels.var(pm, 1, :ps)))
        @test counts[3] == 3   # heat pumps
        @test counts[4] == 2   # DSM loads
        @test counts[1] > 1    # curtailable generators

        for (idx, _, sym, _) in cst_hv_cases
            cons = cst_new_constraints(pm.model) do
                eDisGo_OPF.constraint_HV_requirements(pm, idx, 1)
            end
            coeffs = cst_coeffs(cons[1])
            # One term per component plus the slack, all with coefficient 1.
            @test length(coeffs) == counts[idx] + 1
            @test all(isapprox(c, 1.0; atol=1e-12) for c in values(coeffs))
            @test Set(keys(coeffs)) ==
                  Set(vcat(cst_flex_vars(pm, 1, sym),
                           PowerModels.var(pm, 1, :phvs, idx)))
            @test cst_rhs(cons[1]) ==
                  PowerModels.ref(pm, 1, :HV_requirements, idx)["P"]
        end
    end

    @testset "nw argument" begin
        # The whole constraint — requirement, flexibility variables and slack — is
        # taken from the network passed in.
        pm = cst_model(num_steps=3, overlying_grid=true)
        PowerModels.ref(pm, 3, :HV_requirements, 4)["P"] = 0.75
        cons = cst_new_constraints(pm.model) do
            eDisGo_OPF.constraint_HV_requirements(pm, 4, 3)
        end
        cst_check(cons[1],
                  Dict(PowerModels.var(pm, 3, :pdsm, 1) => 1.0,
                       PowerModels.var(pm, 3, :phvs, 4) => 1.0),
                  0.75)
    end

    @testset "unknown requirement name" begin
        pm = cst_model(num_steps=2, overlying_grid=true)
        PowerModels.ref(pm, 1, :HV_requirements, 1)["name"] = "wind"
        before = length(cst_eq_constraints(pm.model))

        # BUG: no `else` branch — `pflex` is never assigned for an unrecognised
        # name, so the `sum(pflex)` below dies with an `UndefVarError` instead of
        # rejecting the input with a meaningful message.
        @test_throws UndefVarError eDisGo_OPF.constraint_HV_requirements(pm, 1, 1)
        exception = cst_thrown(() -> eDisGo_OPF.constraint_HV_requirements(pm, 1, 1))
        @test exception isa UndefVarError
        @test exception.var === :pflex
        @test length(cst_eq_constraints(pm.model)) == before

        # Intended behaviour: a self-explanatory error (or a documented fallback)
        # rather than an internal name leaking out of the function.
        @test_broken cst_thrown(
            () -> eDisGo_OPF.constraint_HV_requirements(pm, 1, 1)) isa ArgumentError
    end
end
