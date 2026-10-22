# Tests for `src/core/constraint_template.jl`.
#
# The templates in this file sit between the `ref` dict and the formulation
# specific constraint methods of `src/core/constraint.jl` and `src/form/bf.jl`:
# they look the per-bus/per-branch data up, derive the reactive-power factors
# `tan(acos(pf))·sign` and forward everything to the low-level constraint. What
# can be observed is therefore the constraint that ends up in the JuMP model.
#
# Calling a template in isolation needs a model whose *variables* already exist
# but whose constraints do not. `ct_variables_only!` is exactly the variable
# section of `build_mn_opf_bf_flex`, so a freshly built model contains no
# constraint other than the variable bounds; every constraint seen afterwards is
# the one the template under test produced.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

"""
Build method that creates every variable `build_mn_opf_bf_flex` creates — and
nothing else: no constraint, no objective. Mirrors the variable section of
`build_mn_opf_bf_flex` including its `opf_version` case distinction (versions 1
and 3 have no grid restrictions and hence no slack variables, but a line-loading
variable).
"""
function ct_variables_only!(pm)
    if PowerModels.ref(pm, 1, :opf_version) in (1, 3)
        eDisGo_OPF.variable_max_line_loading(pm, nw=1)
    end
    for (n, _) in PowerModels.nws(pm)
        eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false)
        if PowerModels.ref(pm, 1, :opf_version) in (1, 3)
            eDisGo_OPF.variable_branch_current(pm, nw=n, bounded=false)
            eDisGo_OPF.variable_bus_voltage(pm, nw=n, bounded=false)
        else
            eDisGo_OPF.variable_branch_current(pm, nw=n)
            eDisGo_OPF.variable_gen_power_curt(pm, nw=n)
            eDisGo_OPF.variable_slack_grid_restrictions(pm, nw=n)
            eDisGo_OPF.variable_bus_voltage(pm, nw=n)
        end
        eDisGo_OPF.variable_slack_heat_pump_storage(pm, nw=n)
        eDisGo_OPF.variable_battery_storage(pm, nw=n)
        eDisGo_OPF.variable_heat_storage(pm, nw=n)
        eDisGo_OPF.variable_heat_pump_power(pm, nw=n)
        eDisGo_OPF.variable_cp_power(pm, nw=n)
        eDisGo_OPF.variable_dsm_storage_power(pm, nw=n)
        eDisGo_OPF.variable_slack_gen(pm, nw=n)
    end
end

"""
Instantiate `data` (single-network form) as a multi-network model that holds all
variables but no constraints — see [`ct_variables_only!`](@ref).
"""
function ct_model(data::Dict{String,<:Any};
                  model_type::Type=eDisGo_OPF.SOCBFPowerModelEdisgo)
    return eDisGo_OPF.instantiate_model(TestUtils.multinetwork(data), model_type,
                                        ct_variables_only!)
end

"Convenience wrapper: `ct_model` on an unmodified [`TestUtils.small_network`](@ref)."
ct_small_model(; model_type::Type=eDisGo_OPF.SOCBFPowerModelEdisgo, kwargs...) =
    ct_model(TestUtils.small_network(; kwargs...); model_type=model_type)

"""
`small_network` with a distinct power factor per component type, so that a
mixed-up `pf` cannot pass unnoticed. The DSM load additionally gets `sign = -1`
(the fixture default is `+1`), which makes the tests sensitive to the `sign`
factor of `tan(acos(pf))·sign` as well.
"""
function ct_pf_network(; kwargs...)
    data = TestUtils.small_network(; kwargs...)
    data["gen"]["1"]["pf"] = 0.90          # sign -1, bus 2
    data["gen_nd"]["1"]["pf"] = 0.93       # sign -1, bus 3
    data["load"]["1"]["pf"] = 0.95         # sign +1, bus 3
    data["dsm"]["1"]["pf"] = 0.96
    data["dsm"]["1"]["sign"] = -1
    data["heatpumps"]["1"]["pf"] = 0.98    # sign +1, bus 3
    data["electromobility"]["1"]["pf"] = 0.99  # sign +1, bus 3
    data["storage"]["1"]["pf"] = 0.91      # sign -1, bus 4
    return data
end

"All affine equality constraints of `model`, in the order they were created."
ct_eq_constraints(model) = JuMP.all_constraints(model, JuMP.AffExpr,
                                                MOI.EqualTo{Float64})

"All quadratic inequality constraints of `model`, in the order they were created."
ct_quad_constraints(model) = JuMP.all_constraints(model, JuMP.QuadExpr,
                                                  MOI.LessThan{Float64})

"Coefficient of variable `v` in constraint `c` (0.0 if `v` does not appear)."
ct_coef(c, v) = JuMP.coefficient(JuMP.constraint_object(c).func, v)

"Coefficient of the bilinear term `v1·v2` in the quadratic constraint `c`."
ct_coef2(c, v1, v2) = JuMP.coefficient(JuMP.constraint_object(c).func, v1, v2)

"""
The affine constraint `c` as `(Dict(variable name => coefficient), rhs)` — a
representation that can be compared between two separately built models, where
the `VariableRef`s differ but their names do not.
"""
function ct_terms(c)
    func = JuMP.constraint_object(c).func
    terms = Dict{String,Float64}(JuMP.name(v) => coef for (v, coef) in func.terms)
    return terms, JuMP.normalized_rhs(c)
end

"Set of the given variables' names, to be compared with [`ct_varnames`](@ref)."
ct_nameset(vars...) = Set(JuMP.name(v) for v in vars)

"Names of the variables appearing in the affine constraint `c`."
ct_varnames(c) = Set(JuMP.name(v) for v in keys(JuMP.constraint_object(c).func.terms))

"Number of constraints `f()` adds to `model`."
function ct_added(f, model)
    before = TestUtils.num_constraints_total(model)
    f()
    return TestUtils.num_constraints_total(model) - before
end

"[`TestUtils.small_network`](@ref) with a tap ratio ≠ 1 on branch 1."
function ct_tap_network()
    data = TestUtils.small_network()
    data["branch"]["1"]["tap"] = 2.0
    return data
end

# The three predicates below describe what the fallback paths of
# `constraint_store_state` are *meant* to do. They are used with `@test_broken`:
# every one of these paths first logs through `Memento.warn(_LOGGER, …)`, and
# `_LOGGER` is not defined in `eDisGo_OPF` (only `PowerModels._LOGGER` is), so
# they all raise an `UndefVarError` instead of falling back.

"Initial-state constraint built with the documented default `time_elapsed = 1.0`."
function ct_initial_falls_back_to_unit_time_step()
    pm = ct_small_model()
    delete!(PowerModels.ref(pm, 1), :time_elapsed)
    eDisGo_OPF.constraint_store_state(pm, 1, nw=1, kind="storage")
    cs = ct_eq_constraints(pm.model)
    return length(cs) == 2 &&
           ct_coef(cs[1], PowerModels.var(pm, 1, :ps, 1)) == 1.0
end

"Coupling constraint built with the documented default `time_elapsed = 1.0`."
function ct_coupled_falls_back_to_unit_time_step()
    pm = ct_small_model()
    delete!(PowerModels.ref(pm, 2), :time_elapsed)
    eDisGo_OPF.constraint_store_state(pm, 1, 1, 2, "storage")
    cs = ct_eq_constraints(pm.model)
    return length(cs) == 1 &&
           ct_coef(cs[1], PowerModels.var(pm, 2, :ps, 1)) == 1.0
end

"""
Fallback to `constraint_store_state_initial` on `nw_2` for a unit that does not
exist in `nw_1` — the two initial-state constraints instead of the coupling one.
"""
function ct_coupled_falls_back_to_initial()
    pm = ct_small_model()
    delete!(PowerModels.ref(pm, 1, :storage), 1)
    eDisGo_OPF.constraint_store_state(pm, 1, 1, 2, "storage")
    cs = ct_eq_constraints(pm.model)
    return length(cs) == 2 &&
           ct_coef(cs[1], PowerModels.var(pm, 1, :se, 1)) == 0.0
end

@testset "constraint_power_balance_bf" begin
    @testset "one balance equation per bus" begin
        pm = ct_small_model()
        # Nothing but variable bounds so far — the template is the only source of
        # equality constraints in this model.
        @test isempty(ct_eq_constraints(pm.model))
        for i in PowerModels.ids(pm, 1, :bus)
            @test ct_added(() -> eDisGo_OPF.constraint_power_balance_bf(pm, i, nw=1),
                           pm.model) == 2
        end
        # Two equations (active and reactive power) for each of the four buses,
        # the storage bus 4 — which has no voltage variable — included.
        @test length(ct_eq_constraints(pm.model)) == 8
        @test sort(collect(PowerModels.ids(pm, 1, :bus))) == [1, 2, 3, 4]
    end

    @testset "bus without connected components" begin
        # Bus 5 is a dead end: one incoming line, no generator, load or
        # flexibility. It still gets a balance equation, reduced to "what flows in
        # equals the ohmic loss of the incoming line".
        data = TestUtils.small_network()
        data["bus"]["5"] = TestUtils._bus(5)
        data["branch"]["4"] = TestUtils._branch(4, 3, 5)
        pm = ct_model(data)
        for key in (:bus_gens, :bus_gens_nd, :bus_gens_slack, :bus_loads,
                    :bus_storage, :bus_dsm, :bus_hps, :bus_cps)
            @test isempty(PowerModels.ref(pm, 1, key, 5))
        end

        @test ct_added(() -> eDisGo_OPF.constraint_power_balance_bf(pm, 5, nw=1),
                       pm.model) == 2
        cstr_p, cstr_q = ct_eq_constraints(pm.model)
        p = PowerModels.var(pm, 1, :p, (4, 3, 5))
        q = PowerModels.var(pm, 1, :q, (4, 3, 5))
        ccm = PowerModels.var(pm, 1, :ccm, 4)
        # p_to == ccm·r  and  q_to == ccm·x, with r = x = 0.01 of the fixture.
        @test ct_terms(cstr_p) == (Dict(JuMP.name(p) => 1.0,
                                        JuMP.name(ccm) => -0.01), 0.0)
        @test ct_terms(cstr_q) == (Dict(JuMP.name(q) => 1.0,
                                        JuMP.name(ccm) => -0.01), 0.0)
    end

    @testset "power factor terms" begin
        pm = ct_model(ct_pf_network())
        # Bus 3 carries the non-dispatchable generator, the load, the DSM load,
        # the heat pump and the charging point.
        eDisGo_OPF.constraint_power_balance_bf(pm, 3, nw=1)
        _, cstr_q = ct_eq_constraints(pm.model)

        # `tan(acos(pf))·sign` as the template derives it. The terms sit on the
        # right-hand side of the equation, so JuMP's normalised coefficients carry
        # the opposite sign (`+` for the loads that are *subtracted*).
        pf_gen_nd = tan(acos(0.93)) * -1
        pf_load = tan(acos(0.95)) * 1
        pf_dsm = tan(acos(0.96)) * -1
        pf_hp = tan(acos(0.98)) * 1
        pf_cp = tan(acos(0.99)) * 1

        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :pgc, 1)) ≈ -pf_gen_nd
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :pds, 1)) ≈ pf_load
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :pdsm, 1)) ≈ -pf_dsm
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :php, 1)) ≈ -pf_hp
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :phps, 1)) ≈ pf_hp
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :pcp, 1)) ≈ -pf_cp
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :pcps, 1)) ≈ pf_cp
        # The five factors really are five different numbers …
        @test length(unique(round.([pf_gen_nd, pf_load, pf_dsm, pf_hp, pf_cp],
                                   digits=12))) == 5
        # … and none of them leaks into the active-power equation, where all these
        # variables enter with a plain ±1.
        cstr_p, _ = ct_eq_constraints(pm.model)
        for (sym, coef) in ((:pgc, -1.0), (:pds, 1.0), (:pdsm, -1.0), (:php, -1.0),
                            (:phps, 1.0), (:pcp, -1.0), (:pcps, 1.0))
            @test ct_coef(cstr_p, PowerModels.var(pm, 1, sym, 1)) == coef
        end

        @testset "dispatchable generator" begin
            # The dispatchable generator sits at bus 2, alone.
            pm2 = ct_model(ct_pf_network())
            eDisGo_OPF.constraint_power_balance_bf(pm2, 2, nw=1)
            cstr_p2, cstr_q2 = ct_eq_constraints(pm2.model)
            pf_gen = tan(acos(0.90)) * -1
            @test ct_coef(cstr_q2, PowerModels.var(pm2, 1, :pgens, 1)) ≈ -pf_gen
            @test ct_coef(cstr_p2, PowerModels.var(pm2, 1, :pgens, 1)) == -1.0
            # `pg`/`qg` of the generator are constants, they end up in the rhs:
            # -pg = -0.3 and -qg = +0.145 plus nothing else at this bus.
            @test JuMP.normalized_rhs(cstr_p2) ≈ -0.3
            @test JuMP.normalized_rhs(cstr_q2) ≈ 0.145
        end

        @testset "battery storage" begin
            # The battery hangs off bus 4 through the *virtual* branch 3. Its
            # reactive power is taken from that branch's `storage_pf`, which the
            # template forwards verbatim — no `tan(acos(...))` is applied to it.
            data = ct_pf_network()
            data["branch"]["3"]["storage_pf"] = 0.42
            pm3 = ct_model(data)
            eDisGo_OPF.constraint_power_balance_bf(pm3, 4, nw=1)
            cstr_p3, cstr_q3 = ct_eq_constraints(pm3.model)
            p34 = PowerModels.var(pm3, 1, :p, (3, 3, 4))
            @test ct_coef(cstr_q3, p34) ≈ -0.42
            @test ct_coef(cstr_p3, PowerModels.var(pm3, 1, :ps, 1)) == 1.0
            # The storage's own `pf` is collected by the template (`bus_storage_pf`)
            # but never used by `constraint_power_balance` — neither
            # tan(acos(0.91))·(-1) nor 0.91 shows up anywhere.
            for c in (cstr_p3, cstr_q3), coef in values(JuMP.constraint_object(c).func.terms)
                @test !isapprox(abs(coef), abs(tan(acos(0.91))), atol=1e-12)
                @test !isapprox(abs(coef), 0.91, atol=1e-12)
            end
            # It is collected all the same, so a storage unit without `pf` makes
            # the template fail even though the value is dead.
            pm4 = ct_model(ct_pf_network())
            delete!(PowerModels.ref(pm4, 1, :storage, 1), "pf")
            @test_throws KeyError eDisGo_OPF.constraint_power_balance_bf(pm4, 4, nw=1)
        end
    end

    @testset "forwarded to constraint_power_balance" begin
        # Two identically built models: the template on the one, the low-level
        # method with hand-collected arguments on the other. Identical constraints
        # prove that the template passes exactly these values on.
        pm_a = ct_model(ct_pf_network())
        pm_b = ct_model(ct_pf_network())
        i = 3
        eDisGo_OPF.constraint_power_balance_bf(pm_a, i, nw=1)

        r = PowerModels.ref(pm_b, 1)
        bus_lines_to = r[:bus_lines_to][i]
        bus_gens, bus_gens_nd = r[:bus_gens][i], r[:bus_gens_nd][i]
        bus_loads, bus_storage = r[:bus_loads][i], r[:bus_storage][i]
        bus_dsm, bus_hps, bus_cps = r[:bus_dsm][i], r[:bus_hps][i], r[:bus_cps][i]
        pf_of(sym, ks) = Dict(k => tan(acos(r[sym][k]["pf"])) * r[sym][k]["sign"]
                              for k in ks)
        eDisGo_OPF.constraint_power_balance(
            pm_b, 1, i, bus_gens, bus_gens_nd, r[:bus_gens_slack][i], bus_loads,
            r[:bus_arcs_to][i], r[:bus_arcs_from][i], bus_lines_to, bus_storage,
            Dict(k => r[:gen][k]["pg"] for k in bus_gens),
            Dict(k => r[:gen][k]["qg"] for k in bus_gens),
            Dict(k => r[:gen_nd][k]["pg"] for k in bus_gens_nd),
            Dict(k => r[:gen_nd][k]["qg"] for k in bus_gens_nd),
            Dict(k => r[:load][k]["pd"] for k in bus_loads),
            Dict(k => r[:load][k]["qd"] for k in bus_loads),
            Dict(k => r[:branch][k]["br_r"] for k in bus_lines_to),
            Dict(k => r[:branch][k]["br_x"] for k in bus_lines_to),
            bus_dsm, bus_hps, bus_cps,
            pf_of(:storage, bus_storage), pf_of(:dsm, bus_dsm),
            pf_of(:heatpumps, bus_hps), pf_of(:electromobility, bus_cps),
            pf_of(:gen_nd, bus_gens_nd), pf_of(:gen, bus_gens),
            pf_of(:load, bus_loads),
            Dict(k => r[:branch][k]["storage_pf"] for k in bus_lines_to))

        cs_a, cs_b = ct_eq_constraints(pm_a.model), ct_eq_constraints(pm_b.model)
        @test length(cs_a) == length(cs_b) == 2
        @test ct_terms(cs_a[1]) == ct_terms(cs_b[1])
        @test ct_terms(cs_a[2]) == ct_terms(cs_b[2])
        # Not vacuously equal: both equations do carry terms.
        @test length(first(ct_terms(cs_a[1]))) > 3
        @test length(first(ct_terms(cs_a[2]))) > 3
    end

    @testset "opf_version without grid restrictions" begin
        # Versions 1 and 3 know no curtailment/slack variables, so the balance
        # reduces to the flexibilities themselves — the power factors of DSM, heat
        # pump and charging point stay, those of generators and loads have no
        # variable left to multiply.
        pm = ct_model(ct_pf_network(opf_version=1))
        @test ct_added(() -> eDisGo_OPF.constraint_power_balance_bf(pm, 3, nw=1),
                       pm.model) == 2
        cstr_p, cstr_q = ct_eq_constraints(pm.model)
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :pdsm, 1)) ≈ -tan(acos(0.96)) * -1
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :php, 1)) ≈ -tan(acos(0.98))
        @test ct_coef(cstr_q, PowerModels.var(pm, 1, :pcp, 1)) ≈ -tan(acos(0.99))
        # No slack variable of any kind — in this version they do not even exist.
        for sym in (:pgc, :pgens, :pds, :phps, :pcps)
            @test !haskey(PowerModels.var(pm, 1), sym)
        end
        @test ct_varnames(cstr_p) == ct_nameset(
            PowerModels.var(pm, 1, :p, (2, 2, 3)), PowerModels.var(pm, 1, :p, (3, 3, 4)),
            PowerModels.var(pm, 1, :ccm, 2), PowerModels.var(pm, 1, :php, 1),
            PowerModels.var(pm, 1, :pcp, 1), PowerModels.var(pm, 1, :pdsm, 1))
        @test ct_varnames(cstr_q) == ct_nameset(
            PowerModels.var(pm, 1, :q, (2, 2, 3)), PowerModels.var(pm, 1, :q, (3, 3, 4)),
            PowerModels.var(pm, 1, :ccm, 2), PowerModels.var(pm, 1, :php, 1),
            PowerModels.var(pm, 1, :pcp, 1), PowerModels.var(pm, 1, :pdsm, 1))
    end

    @testset "network selection" begin
        # `nw` picks both the data and the variables of that time step.
        pm = ct_model(ct_pf_network())
        eDisGo_OPF.constraint_power_balance_bf(pm, 3, nw=2)
        cstr_p, _ = ct_eq_constraints(pm.model)
        @test ct_coef(cstr_p, PowerModels.var(pm, 2, :pdsm, 1)) == -1.0
        @test ct_coef(cstr_p, PowerModels.var(pm, 1, :pdsm, 1)) == 0.0
        @test all(startswith(name, "2_") for name in ct_varnames(cstr_p))
    end
end

@testset "constraint_voltage_magnitude_difference_radial" begin
    @testset "physical branch" begin
        # Branch 2 runs from bus 2 to bus 3, neither of which is the slack bus, so
        # it contributes the voltage-drop equation and nothing else. `br_r`/`br_x`
        # and `tap` are the parameters that must be read — note that the fixture
        # also carries a differing `"r"` entry, which must *not* be used.
        data = TestUtils.small_network()
        data["branch"]["2"]["br_r"] = 0.03
        data["branch"]["2"]["br_x"] = 0.04
        data["branch"]["2"]["r"] = 99.0
        data["branch"]["2"]["tap"] = 2.0
        pm = ct_model(data)

        @test ct_added(
            () -> eDisGo_OPF.constraint_voltage_magnitude_difference_radial(pm, 2, nw=1),
            pm.model) == 1
        c = only(ct_eq_constraints(pm.model))
        # w_fr/tm² - w_to == 2(r·p + x·q) - (r² + x²)·ccm
        @test ct_coef(c, PowerModels.var(pm, 1, :w, 2)) ≈ 1 / 2.0^2
        @test ct_coef(c, PowerModels.var(pm, 1, :w, 3)) == -1.0
        @test ct_coef(c, PowerModels.var(pm, 1, :p, (2, 2, 3))) ≈ -2 * 0.03
        @test ct_coef(c, PowerModels.var(pm, 1, :q, (2, 2, 3))) ≈ -2 * 0.04
        @test ct_coef(c, PowerModels.var(pm, 1, :ccm, 2)) ≈ 0.03^2 + 0.04^2
        @test JuMP.normalized_rhs(c) == 0.0
        @test length(ct_varnames(c)) == 5
    end

    @testset "branch at the slack bus" begin
        # Branch 1 starts at bus 1 (`bus_type == 3`), which pins its squared
        # voltage to 1 p.u. — a second equation.
        pm = ct_small_model()
        @test PowerModels.ref(pm, 1, :bus, 1)["bus_type"] == 3
        @test ct_added(
            () -> eDisGo_OPF.constraint_voltage_magnitude_difference_radial(pm, 1, nw=1),
            pm.model) == 2
        cs = ct_eq_constraints(pm.model)
        c_slack = cs[2]
        @test ct_terms(c_slack) == (Dict(JuMP.name(PowerModels.var(pm, 1, :w, 1)) => 1.0),
                                    1.0)
    end

    @testset "virtual storage branch" begin
        # Branch 3 is the virtual branch of the battery: no line, no equation.
        # (It could not even be written down — bus 4 has no `w` variable.)
        pm = ct_small_model()
        @test PowerModels.ref(pm, 1, :branch, 3)["storage"] == true
        @test ct_added(
            () -> eDisGo_OPF.constraint_voltage_magnitude_difference_radial(pm, 3, nw=1),
            pm.model) == 0
        @test isempty(ct_eq_constraints(pm.model))
        @test !(4 in axes(PowerModels.var(pm, 1, :w), 1))

        # Flipping the flag on a physical branch skips it just the same, so it is
        # really the flag and not the topology that decides.
        data = TestUtils.small_network()
        data["branch"]["2"]["storage"] = true
        pm2 = ct_model(data)
        @test ct_added(
            () -> eDisGo_OPF.constraint_voltage_magnitude_difference_radial(pm2, 2, nw=1),
            pm2.model) == 0
    end

    @testset "network selection" begin
        pm = ct_small_model()
        eDisGo_OPF.constraint_voltage_magnitude_difference_radial(pm, 2, nw=2)
        c = only(ct_eq_constraints(pm.model))
        @test all(startswith(name, "2_") for name in ct_varnames(c))
    end
end

@testset "constraint_store_state" begin
    @testset "initial time step" begin
        @testset "storage" begin
            data = TestUtils.small_network()
            data["time_elapsed"] = 0.25
            data["storage"]["1"]["soc_initial"] = 0.15
            data["storage"]["1"]["soc_end"] = 0.2
            pm = ct_model(data)

            @test ct_added(
                () -> eDisGo_OPF.constraint_store_state(pm, 1, nw=1, kind="storage"),
                pm.model) == 2
            c_state, c_end = ct_eq_constraints(pm.model)
            # se₁ - soc_initial == -Δt·ps₁
            @test ct_terms(c_state) == (
                Dict(JuMP.name(PowerModels.var(pm, 1, :se, 1)) => 1.0,
                     JuMP.name(PowerModels.var(pm, 1, :ps, 1)) => 0.25), 0.15)
            # The end-of-horizon state is pinned in the *last* network.
            @test ct_terms(c_end) == (
                Dict(JuMP.name(PowerModels.var(pm, 2, :se, 1)) => 1.0), 0.2)
        end

        @testset "heat_storage" begin
            data = TestUtils.small_network()
            data["time_elapsed"] = 0.5
            data["heat_storage"]["1"]["soc_initial"] = 0.6
            data["heat_storage"]["1"]["soc_end"] = 0.7
            pm = ct_model(data)

            @test ct_added(
                () -> eDisGo_OPF.constraint_store_state(pm, 1, nw=1, kind="heat_storage"),
                pm.model) == 2
            c_state, c_end = ct_eq_constraints(pm.model)
            @test ct_terms(c_state) == (
                Dict(JuMP.name(PowerModels.var(pm, 1, :hse, 1)) => 1.0,
                     JuMP.name(PowerModels.var(pm, 1, :phs, 1)) => 0.5), 0.6)
            @test ct_terms(c_end) == (
                Dict(JuMP.name(PowerModels.var(pm, 2, :hse, 1)) => 1.0), 0.7)
        end

        @testset "dsm" begin
            data = TestUtils.small_network()
            data["time_elapsed"] = 0.25
            data["dsm"]["1"]["energy"] = 0.1
            pm = ct_model(data)

            @test ct_added(
                () -> eDisGo_OPF.constraint_store_state(pm, 1, nw=1, kind="dsm"),
                pm.model) == 2
            c_state, c_end = ct_eq_constraints(pm.model)
            # dsme₁ - energy == +Δt·pdsm₁ — the DSM band is signed the other way
            # round than the storage power.
            @test ct_terms(c_state) == (
                Dict(JuMP.name(PowerModels.var(pm, 1, :dsme, 1)) => 1.0,
                     JuMP.name(PowerModels.var(pm, 1, :pdsm, 1)) => -0.25), 0.1)
            # DSM has to return to a neutral state, hence a hard 0 rather than a
            # `soc_end` from the data.
            @test ct_terms(c_end) == (
                Dict(JuMP.name(PowerModels.var(pm, 2, :dsme, 1)) => 1.0), 0.0)
        end

        @testset "p_loss" begin
            # For "dsm" the standing loss is hard-coded to 0 — and it has to be:
            # a DSM component carries no `p_loss` at all.
            pm = ct_small_model()
            @test !haskey(PowerModels.ref(pm, 1, :dsm, 1), "p_loss")
            @test ct_added(
                () -> eDisGo_OPF.constraint_store_state(pm, 1, nw=1, kind="dsm"),
                pm.model) == 2

            # For the two real storages it is read off the component …
            for kind in ("storage", "heat_storage")
                pm_k = ct_small_model()
                @test haskey(PowerModels.ref(pm_k, 1, Symbol(kind), 1), "p_loss")
                delete!(PowerModels.ref(pm_k, 1, Symbol(kind), 1), "p_loss")
                @test_throws KeyError eDisGo_OPF.constraint_store_state(
                    pm_k, 1, nw=1, kind=kind)
            end

            # … but `constraint_store_state_initial` ignores it: the first time
            # step has no previous state that could have decayed.
            data = TestUtils.small_network()
            data["heat_storage"]["1"]["p_loss"] = 0.5
            pm_loss = ct_model(data)
            eDisGo_OPF.constraint_store_state(pm_loss, 1, nw=1, kind="heat_storage")
            pm_ref = ct_small_model()
            eDisGo_OPF.constraint_store_state(pm_ref, 1, nw=1, kind="heat_storage")
            @test ct_terms.(ct_eq_constraints(pm_loss.model)) ==
                  ct_terms.(ct_eq_constraints(pm_ref.model))
        end

        @testset "time_elapsed" begin
            # Read from the `ref` of the requested network, not from network 1.
            pm = ct_small_model()
            PowerModels.ref(pm, 1)[:time_elapsed] = 0.25
            PowerModels.ref(pm, 2)[:time_elapsed] = 4.0
            eDisGo_OPF.constraint_store_state(pm, 1, nw=2, kind="storage")
            c_state, _ = ct_eq_constraints(pm.model)
            @test ct_coef(c_state, PowerModels.var(pm, 2, :ps, 1)) == 4.0
        end

        @testset "missing time_elapsed" begin
            # Actual behaviour: the fallback path logs through `_LOGGER`, which is
            # not defined in `eDisGo_OPF` — the model build dies instead.
            pm = ct_small_model()
            delete!(PowerModels.ref(pm, 1), :time_elapsed)
            @test !haskey(PowerModels.ref(pm, 1), :time_elapsed)
            @test_throws UndefVarError eDisGo_OPF.constraint_store_state(
                pm, 1, nw=1, kind="storage")
            @test isempty(ct_eq_constraints(pm.model))

            # Intended behaviour: warn and fall back to `time_elapsed = 1.0`.
            @test_broken ct_initial_falls_back_to_unit_time_step()
        end
    end

    @testset "two time steps" begin
        @testset "storage" begin
            data = TestUtils.small_network()
            data["time_elapsed"] = 0.25
            pm = ct_model(data)
            @test ct_added(
                () -> eDisGo_OPF.constraint_store_state(pm, 1, 1, 2, "storage"),
                pm.model) == 1
            # se₂ - se₁ == -Δt·ps₂
            @test ct_terms(only(ct_eq_constraints(pm.model))) == (
                Dict(JuMP.name(PowerModels.var(pm, 2, :se, 1)) => 1.0,
                     JuMP.name(PowerModels.var(pm, 1, :se, 1)) => -1.0,
                     JuMP.name(PowerModels.var(pm, 2, :ps, 1)) => 0.25), 0.0)
        end

        @testset "heat_storage" begin
            # The only place where `p_loss` actually enters a constraint: the
            # previous state decays by (1 - p_loss)^(time_elapsed/24) per time step.
            data = TestUtils.small_network()
            data["time_elapsed"] = 0.5
            data["heat_storage"]["1"]["p_loss"] = 0.1
            pm = ct_model(data)
            @test ct_added(
                () -> eDisGo_OPF.constraint_store_state(pm, 1, 1, 2, "heat_storage"),
                pm.model) == 1
            c = only(ct_eq_constraints(pm.model))
            @test ct_coef(c, PowerModels.var(pm, 2, :hse, 1)) == 1.0
            @test ct_coef(c, PowerModels.var(pm, 1, :hse, 1)) ≈ -(1 - 0.1)^(0.5 / 24)
            @test ct_coef(c, PowerModels.var(pm, 2, :phs, 1)) == 0.5
            @test JuMP.normalized_rhs(c) == 0.0
            # The fixture's own p_loss = 0.04 gives a different decay factor, so
            # the value really comes from the component (fixture time_elapsed = 1).
            pm_ref = ct_model(TestUtils.small_network())
            eDisGo_OPF.constraint_store_state(pm_ref, 1, 1, 2, "heat_storage")
            @test ct_coef(only(ct_eq_constraints(pm_ref.model)),
                          PowerModels.var(pm_ref, 1, :hse, 1)) ≈ -(1 - 0.04)^(1 / 24)
        end

        @testset "dsm" begin
            data = TestUtils.small_network()
            data["time_elapsed"] = 0.25
            pm = ct_model(data)
            # `p_loss` is 0 for DSM, and the component has none to read.
            @test !haskey(PowerModels.ref(pm, 2, :dsm, 1), "p_loss")
            @test ct_added(() -> eDisGo_OPF.constraint_store_state(pm, 1, 1, 2, "dsm"),
                           pm.model) == 1
            # dsme₂ - dsme₁ == +Δt·pdsm₂
            @test ct_terms(only(ct_eq_constraints(pm.model))) == (
                Dict(JuMP.name(PowerModels.var(pm, 2, :dsme, 1)) => 1.0,
                     JuMP.name(PowerModels.var(pm, 1, :dsme, 1)) => -1.0,
                     JuMP.name(PowerModels.var(pm, 2, :pdsm, 1)) => -0.25), 0.0)
        end

        @testset "time_elapsed" begin
            # Taken from `nw_2`, the network the new state belongs to.
            pm = ct_small_model()
            PowerModels.ref(pm, 1)[:time_elapsed] = 0.25
            PowerModels.ref(pm, 2)[:time_elapsed] = 4.0
            eDisGo_OPF.constraint_store_state(pm, 1, 1, 2, "storage")
            @test ct_coef(only(ct_eq_constraints(pm.model)),
                          PowerModels.var(pm, 2, :ps, 1)) == 4.0
        end

        @testset "missing time_elapsed" begin
            pm = ct_small_model()
            delete!(PowerModels.ref(pm, 2), :time_elapsed)
            @test_throws UndefVarError eDisGo_OPF.constraint_store_state(
                pm, 1, 1, 2, "storage")
            @test isempty(ct_eq_constraints(pm.model))
            # `nw_1` is not consulted for the elapsed time: removing it there
            # changes nothing.
            pm_ok = ct_small_model()
            delete!(PowerModels.ref(pm_ok, 1), :time_elapsed)
            eDisGo_OPF.constraint_store_state(pm_ok, 1, 1, 2, "storage")
            @test length(ct_eq_constraints(pm_ok.model)) == 1

            @test_broken ct_coupled_falls_back_to_unit_time_step()
        end

        @testset "component missing in the first network" begin
            # A unit that is inactive in `nw_1` has no stored-energy variable
            # there; the code means to fall back to the initial-state constraint
            # built from `nw_2`'s data — but warns through the undefined `_LOGGER`
            # before it gets there.
            pm = ct_small_model()
            delete!(PowerModels.ref(pm, 1, :storage), 1)
            @test !haskey(PowerModels.ref(pm, 1, :storage), 1)
            @test haskey(PowerModels.ref(pm, 2, :storage), 1)
            @test_throws UndefVarError eDisGo_OPF.constraint_store_state(
                pm, 1, 1, 2, "storage")
            @test isempty(ct_eq_constraints(pm.model))

            # Intended behaviour: the two initial-state constraints from `nw_2`.
            @test_broken ct_coupled_falls_back_to_initial()

            # Present in `nw_1`: the regular coupling constraint, no fallback.
            pm_ok = ct_small_model()
            @test ct_added(
                () -> eDisGo_OPF.constraint_store_state(pm_ok, 1, 1, 2, "storage"),
                pm_ok.model) == 1
        end
    end
end

@testset "constraint_model_current" begin
    # `ct_tap_network` gives branch 1 a tap ratio ≠ 1, so that the `1/tm²` factor
    # is visible in the constraint.
    @testset "AbstractSOCBFModelEdisgo" begin
        pm = ct_model(ct_tap_network(); model_type=eDisGo_OPF.SOCBFPowerModelEdisgo)
        # The convex relaxation covers *every* branch, the virtual storage branch
        # included, and produces no nonlinear constraint.
        @test ct_added(() -> eDisGo_OPF.constraint_model_current(pm, nw=1),
                       pm.model) == 3
        cs = ct_quad_constraints(pm.model)
        @test length(cs) == 3
        @test JuMP.num_nonlinear_constraints(pm.model) == 0

        for (idx, arc, f_bus, tm) in ((1, (1, 1, 2), 1, 2.0), (2, (2, 2, 3), 2, 1.0),
                                      (3, (3, 3, 4), 3, 1.0))
            c = only(filter(x -> occursin("ccm[$idx]", string(x)), cs))
            p = PowerModels.var(pm, 1, :p, arc)
            q = PowerModels.var(pm, 1, :q, arc)
            w = PowerModels.var(pm, 1, :w, f_bus)
            ccm = PowerModels.var(pm, 1, :ccm, idx)
            # p² + q² ≤ (w/tm²)·ccm
            @test ct_coef2(c, p, p) == 1.0
            @test ct_coef2(c, q, q) == 1.0
            @test ct_coef2(c, w, ccm) ≈ -1 / tm^2
            @test JuMP.constraint_object(c).set == MOI.LessThan(0.0)
        end
    end

    @testset "AbstractNCBFModelEdisgo" begin
        pm = ct_model(ct_tap_network(); model_type=eDisGo_OPF.NCBFPowerModelEdisgo)
        # `@NLconstraint` lands in JuMP's legacy nonlinear model, which
        # `num_constraints` does not see.
        @test ct_added(() -> eDisGo_OPF.constraint_model_current(pm, nw=1),
                       pm.model) == 0
        @test isempty(ct_quad_constraints(pm.model))
        # The exact equality is written for the two physical branches only …
        @test JuMP.num_nonlinear_constraints(pm.model) == 2
        strs = string.(JuMP.all_nonlinear_constraints(pm.model))
        @test length(strs) == 2
        @test count(s -> occursin("1_ccm[1]", s), strs) == 1
        @test count(s -> occursin("1_ccm[2]", s), strs) == 1
        # … the virtual storage branch 3 is skipped.
        @test !any(s -> occursin("1_ccm[3]", s), strs)
        # Equality, tap ratio and the squared flows are all in there.
        c1 = only(filter(s -> occursin("1_ccm[1]", s), strs))
        @test occursin("= 0", c1)
        @test occursin("2.0 ^ 2.0", c1)
        @test occursin("1_p[(1, 1, 2)] ^ 2.0", c1)
        @test occursin("1_q[(1, 1, 2)] ^ 2.0", c1)
    end

    @testset "BFPowerModelEdisgo" begin
        # The base type implements neither formulation; there is nothing to
        # forward to.
        pm = ct_model(ct_tap_network(); model_type=eDisGo_OPF.BFPowerModelEdisgo)
        @test pm isa eDisGo_OPF.AbstractBFModelEdisgo
        @test !(pm isa eDisGo_OPF.AbstractSOCBFModelEdisgo)
        @test !(pm isa eDisGo_OPF.AbstractNCBFModelEdisgo)
        before = TestUtils.num_constraints_total(pm.model)
        @test_throws MethodError eDisGo_OPF.constraint_model_current(pm, nw=1)
        @test TestUtils.num_constraints_total(pm.model) == before
        @test isempty(ct_quad_constraints(pm.model))
        @test JuMP.num_nonlinear_constraints(pm.model) == 0
    end

    @testset "network selection" begin
        pm = ct_model(TestUtils.small_network();
                      model_type=eDisGo_OPF.SOCBFPowerModelEdisgo)
        eDisGo_OPF.constraint_model_current(pm, nw=2)
        cs = ct_quad_constraints(pm.model)
        @test length(cs) == 3
        @test all(occursin("2_", string(c)) && !occursin("1_", string(c)) for c in cs)
    end
end
