# Tests for `src/form/bf.jl`.
#
# The file holds the branch-flow specific building blocks of the eDisGo OPF: the
# squared current (`ccm`) and squared voltage (`w`) variables, the voltage-drop
# equation, the current/power/voltage coupling of the SOC and the non-convex (NC)
# formulation, the maximum-line-loading constraint and the low-level nodal power
# balance.
#
# Everything is called in isolation on a model whose build method creates only
# what the function under test needs (`bf_nothing!`, `bf_flow_variables!`,
# `bf_all_variables!` below), so every variable or constraint seen afterwards is
# the one the function under test added. The low-level constraint functions are
# called directly with hand-picked arguments; how the templates in
# `src/core/constraint_template.jl` collect these arguments from the `ref` is
# tested in `test/core/test_constraint_template.jl`.
#
# Known production defects documented here (see test/FUNKTIONSABDECKUNG.md):
# - `constraint_model_current` treats the virtual storage branches differently in
#   the SOC (included) and the NC variant (skipped).
# - For `opf_version` 3 `constraint_power_balance` leaves the curtailment `pgc`
#   out of the balance although the variable exists and is counted towards the
#   overlying-grid requirement.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

"Build method that creates nothing: the model only carries the `ref` data."
bf_nothing!(pm) = nothing

"""
Build method creating the flow variables the branch-flow constraints operate on —
`p`, `q`, `ccm` and `w` in every network, all unbounded — plus the line-loading
variable `ll` in network 1 for `opf_version` 1 and 3, as `build_mn_opf_bf_flex`
does. No constraint is created apart from the variable bounds.
"""
function bf_flow_variables!(pm)
    if PowerModels.ref(pm, 1, :opf_version) in (1, 3)
        eDisGo_OPF.variable_max_line_loading(pm, nw=1)
    end
    for (n, _) in PowerModels.nws(pm)
        eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false)
        eDisGo_OPF.variable_branch_current(pm, nw=n, bounded=false)
        eDisGo_OPF.variable_bus_voltage(pm, nw=n, bounded=false)
    end
end

"Build method like [`bf_flow_variables!`](@ref) but without the current variable `ccm`."
function bf_flow_variables_without_ccm!(pm)
    for (n, _) in PowerModels.nws(pm)
        eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false)
        eDisGo_OPF.variable_bus_voltage(pm, nw=n, bounded=false)
    end
end

"""
Build method creating every variable `build_mn_opf_bf_flex` creates — and no
constraint. Mirrors its variable section including the `opf_version` case
distinction (slack variables for 2 and 4, line loading for 1 and 3, HV requirement
slacks for 3 and 4 and the curtailment variable for 3).
"""
function bf_all_variables!(pm)
    version = PowerModels.ref(pm, 1, :opf_version)
    if version in (1, 3)
        eDisGo_OPF.variable_max_line_loading(pm, nw=1)
    end
    for (n, _) in PowerModels.nws(pm)
        eDisGo_OPF.variable_branch_power_radial(pm, nw=n, bounded=false)
        if version in (1, 3)
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
        if version in (3, 4)
            eDisGo_OPF.variable_slack_HV_requirements(pm, nw=n)
            version == 3 && eDisGo_OPF.variable_gen_power_curt(pm, nw=n)
        end
    end
end

"""
Instantiate `data` (single-network form) as a multi-network model with
`build_method`. Further keyword arguments (e.g. `setting`) go to
`eDisGo_OPF.instantiate_model`.
"""
function bf_model(data::Dict{String,<:Any}, build_method;
                  model_type::Type=eDisGo_OPF.SOCBFPowerModelEdisgo, kwargs...)
    return eDisGo_OPF.instantiate_model(TestUtils.multinetwork(data), model_type,
                                        build_method; kwargs...)
end

"Model on an unmodified [`TestUtils.small_network`](@ref) with `build_method`."
bf_small_model(build_method; model_type::Type=eDisGo_OPF.SOCBFPowerModelEdisgo,
               kwargs...) =
    bf_model(TestUtils.small_network(; kwargs...), build_method; model_type=model_type)

"Number of constraints `f()` adds to `model` (variable bounds included)."
function bf_added(f, model)
    before = TestUtils.num_constraints_total(model)
    f()
    return TestUtils.num_constraints_total(model) - before
end

"All affine equality constraints of `model`, in the order they were created."
bf_eq_constraints(model) = JuMP.all_constraints(model, JuMP.AffExpr, MOI.EqualTo{Float64})

"All quadratic `≤` constraints of `model`, in the order they were created."
bf_quad_constraints(model) = JuMP.all_constraints(model, JuMP.QuadExpr,
                                                  MOI.LessThan{Float64})

"Coefficient of the (affine) variable `v` in constraint `c` (0.0 if absent)."
bf_coef(c, v) = JuMP.coefficient(JuMP.constraint_object(c).func, v)

"Coefficient of the quadratic term `v1·v2` in constraint `c` (0.0 if absent)."
bf_coef2(c, v1, v2) = JuMP.coefficient(JuMP.constraint_object(c).func, v1, v2)

"""
The affine constraint `c` as `(Dict(variable name => coefficient), rhs)`, so that
constraints can be compared by value.
"""
function bf_terms(c)
    func = JuMP.constraint_object(c).func
    return Dict{String,Float64}(JuMP.name(v) => coef for (v, coef) in func.terms),
           JuMP.normalized_rhs(c)
end

"Name of the variable `var(pm, n, sym)[idx]`."
bf_name(pm, n, sym, idx) = JuMP.name(PowerModels.var(pm, n, sym, idx))

"Value of the function of constraint `c` with the variables set as in `vals` (0 otherwise)."
bf_eval(c::JuMP.ConstraintRef, vals) =
    JuMP.value(v -> get(vals, v, 0.0), JuMP.constraint_object(c).func)
bf_eval(c::JuMP.NonlinearConstraintRef, vals) = JuMP.value(v -> get(vals, v, 0.0), c)

"Index set of the JuMP container `x`."
bf_index_set(x) = Set(axes(x, 1))

"""
Indices `i` of the branches whose current variable `ccm[i]` appears in one of the
current-model constraints of `pm` (quadratic and legacy nonlinear ones).
"""
function bf_current_model_branches(pm)
    strs = vcat(string.(bf_quad_constraints(pm.model)),
                string.(JuMP.all_nonlinear_constraints(pm.model)))
    return Set(parse(Int, m[1]) for s in strs for m in eachmatch(r"_ccm\[(\d+)\]", s))
end

"[`TestUtils.small_network`](@ref) with a tap ratio of 2 on branch 1."
function bf_tap_network(; kwargs...)
    data = TestUtils.small_network(; kwargs...)
    data["branch"]["1"]["tap"] = 2.0
    return data
end

"""
Arguments of the low-level `constraint_power_balance` after `(pm, n, i)` as a
`NamedTuple` in the order of the signature. The defaults connect one component of
every kind with a distinct parameter value, so that every term can be told apart;
keyword arguments replace single entries.
"""
function bf_balance_args(; kwargs...)
    defaults = (
        bus_gens=[1], bus_gens_nd=[1], bus_gens_slack=[1], bus_loads=[1],
        bus_arcs_to=[(2, 2, 3)], bus_arcs_from=[(3, 3, 4)], bus_lines_to=[2],
        bus_storage=[1],
        bus_pg=Dict(1 => 0.3), bus_qg=Dict(1 => -0.1),
        bus_pg_nd=Dict(1 => 0.6), bus_qg_nd=Dict(1 => -0.2),
        bus_pd=Dict(1 => 0.2), bus_qd=Dict(1 => 0.07),
        branch_r=Dict(2 => 0.03), branch_x=Dict(2 => 0.04),
        bus_dsm=[1], bus_hps=[1], bus_cps=[1],
        bus_storage_pf=Dict(1 => 0.11), bus_dsm_pf=Dict(1 => 0.12),
        bus_hps_pf=Dict(1 => 0.13), bus_cps_pf=Dict(1 => 0.14),
        bus_gen_nd_pf=Dict(1 => 0.15), bus_gen_d_pf=Dict(1 => 0.16),
        bus_loads_pf=Dict(1 => 0.17), branch_strg_pf=Dict(2 => 0.18),
    )
    return merge(defaults, values(kwargs))
end

"Call `constraint_power_balance` on bus `i` of network `n` with [`bf_balance_args`](@ref)."
bf_power_balance(pm, n=1, i=3; kwargs...) =
    eDisGo_OPF.constraint_power_balance(pm, n, i, values(bf_balance_args(; kwargs...))...)

"""
Shared checks of `constraint_max_line_loading` — the SOC and the NC method are
identical and are both run through this function.
"""
function bf_check_max_line_loading(model_type::Type)
    @testset "one constraint per physical branch" begin
        data = TestUtils.small_network(opf_version=1)
        data["branch"]["1"]["rate_a"] = 2.0
        pm = bf_model(data, bf_flow_variables!; model_type=model_type)
        ll = PowerModels.var(pm, 1, :ll)
        # `ll` exists for the physical branches only …
        @test bf_index_set(ll) == Set([(1, 1, 2), (2, 2, 3)])
        # … and so does the constraint: the virtual storage branch 3 is skipped.
        @test bf_added(() -> eDisGo_OPF.constraint_max_line_loading(pm, 1),
                       pm.model) == 2
        cs = bf_quad_constraints(pm.model)
        @test length(cs) == 2
        @test JuMP.num_nonlinear_constraints(pm.model) == 0

        for (idx, arc, rate_a) in ((1, (1, 1, 2), 2.0), (2, (2, 2, 3), 5.0))
            c = only(filter(x -> occursin("ll[$arc]", string(x)), cs))
            p = PowerModels.var(pm, 1, :p, arc)
            q = PowerModels.var(pm, 1, :q, arc)
            # (p² + q²)/s_nom² ≤ ll with s_nom = rate_a
            @test bf_coef2(c, p, p) ≈ 1 / rate_a^2
            @test bf_coef2(c, q, q) ≈ 1 / rate_a^2
            @test bf_coef(c, ll[arc]) == -1.0
            @test JuMP.constraint_object(c).set == MOI.LessThan(0.0)
            func = JuMP.constraint_object(c).func
            @test length(func.terms) == 2
            @test length(func.aff.terms) == 1
            @test func.aff.constant == 0.0
        end
        # No constraint touches the storage branch's flows.
        @test !any(occursin("p[(3, 3, 4)]", string(c)) for c in cs)
    end

    @testset "missing rate_a" begin
        # Without `rate_a` the nominal power falls back to 1.0.
        pm = bf_small_model(bf_flow_variables!; model_type=model_type, opf_version=1)
        delete!(PowerModels.ref(pm, 1, :branch, 2), "rate_a")
        eDisGo_OPF.constraint_max_line_loading(pm, 1)
        c = only(filter(x -> occursin("ll[(2, 2, 3)]", string(x)),
                        bf_quad_constraints(pm.model)))
        p = PowerModels.var(pm, 1, :p, (2, 2, 3))
        @test bf_coef2(c, p, p) == 1.0
    end

    @testset "storage flag decides" begin
        # Flagging the physical branch 2 as storage branch after `ll` was created
        # removes its constraint — it is the flag, not the topology.
        pm = bf_small_model(bf_flow_variables!; model_type=model_type, opf_version=1)
        PowerModels.ref(pm, 1, :branch, 2)["storage"] = true
        @test bf_added(() -> eDisGo_OPF.constraint_max_line_loading(pm, 1),
                       pm.model) == 1
        @test occursin("1_ll[(1, 1, 2)]", string(only(bf_quad_constraints(pm.model))))
    end

    @testset "network selection" begin
        # The flows are those of network `n`, but `ll` always comes from network 1:
        # one maximum loading for the whole horizon.
        pm = bf_small_model(bf_flow_variables!; model_type=model_type, opf_version=1)
        @test !haskey(PowerModels.var(pm, 2), :ll)
        eDisGo_OPF.constraint_max_line_loading(pm, 2)
        cs = bf_quad_constraints(pm.model)
        @test length(cs) == 2
        for c in cs
            s = string(c)
            @test occursin("2_p[", s) && occursin("2_q[", s) && occursin("1_ll[", s)
            @test !occursin("1_p[", s) && !occursin("2_ll[", s)
        end
    end

    @testset "missing line-loading variable" begin
        # Versions 2 and 4 create no `ll`; the constraint cannot be built.
        pm = bf_small_model(bf_flow_variables!; model_type=model_type, opf_version=2)
        @test !haskey(PowerModels.var(pm, 1), :ll)
        @test_throws KeyError eDisGo_OPF.constraint_max_line_loading(pm, 1)
        @test isempty(bf_quad_constraints(pm.model))
    end
end

@testset "variable_branch_current" begin
    @testset "creates ccm" begin
        pm = bf_small_model(bf_nothing!)
        @test isempty(PowerModels.var(pm, 1))
        eDisGo_OPF.variable_branch_current(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:ccm]
        # One variable per branch, the virtual storage branch 3 included.
        @test bf_index_set(PowerModels.var(pm, 1, :ccm)) == Set([1, 2, 3])
        @test isempty(PowerModels.var(pm, 2))
    end

    @testset "keyword arguments are forwarded" begin
        # Wrapper and wrapped function produce the same variables for every
        # combination of `bounded` and `report`.
        for bounded in (true, false), report in (true, false)
            pm_a = bf_small_model(bf_nothing!)
            pm_b = bf_small_model(bf_nothing!)
            eDisGo_OPF.variable_branch_current(pm_a, nw=2, bounded=bounded, report=report)
            eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm_b, nw=2, bounded=bounded,
                                                             report=report)
            ccm_a = PowerModels.var(pm_a, 2, :ccm)
            ccm_b = PowerModels.var(pm_b, 2, :ccm)
            @test bf_index_set(ccm_a) == bf_index_set(ccm_b)
            for i in bf_index_set(ccm_a)
                @test JuMP.name(ccm_a[i]) == JuMP.name(ccm_b[i]) == "2_ccm[$i]"
                @test JuMP.has_upper_bound(ccm_a[i]) == JuMP.has_upper_bound(ccm_b[i])
                if JuMP.has_upper_bound(ccm_a[i])
                    @test JuMP.upper_bound(ccm_a[i]) == JuMP.upper_bound(ccm_b[i])
                end
            end
            # Unbounded: only the storage branch keeps its upper bound.
            @test count(i -> JuMP.has_upper_bound(ccm_a[i]), bf_index_set(ccm_a)) ==
                  (bounded ? 3 : 1)
            @test haskey(PowerModels.sol(pm_a, 2), :branch) == report
            @test !haskey(PowerModels.var(pm_a, 1), :ccm)
        end
    end
end

@testset "variable_bus_voltage" begin
    @testset "creates w for non-storage buses" begin
        pm = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_bus_voltage(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:w]
        # Bus 4 is the storage bus of the virtual branch and has no voltage.
        @test PowerModels.ref(pm, 1, :bus, 4)["storage"] == true
        @test bf_index_set(PowerModels.var(pm, 1, :w)) == Set([1, 2, 3])
        @test isempty(PowerModels.var(pm, 2))
    end

    @testset "bounded" begin
        data = TestUtils.small_network()
        data["bus"]["2"]["vmin"] = 0.95
        data["bus"]["2"]["vmax"] = 1.05
        pm = bf_model(data, bf_nothing!)
        eDisGo_OPF.variable_bus_voltage(pm, nw=1)
        w = PowerModels.var(pm, 1, :w)
        for (i, vmin, vmax) in ((1, 0.9, 1.1), (2, 0.95, 1.05), (3, 0.9, 1.1))
            @test JuMP.lower_bound(w[i]) ≈ vmin^2
            @test JuMP.upper_bound(w[i]) ≈ vmax^2
        end
    end

    @testset "unbounded" begin
        pm = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_bus_voltage(pm, nw=1, bounded=false)
        w = PowerModels.var(pm, 1, :w)
        for i in 1:3
            @test JuMP.lower_bound(w[i]) == 0.0
            @test !JuMP.has_upper_bound(w[i])
        end
    end

    @testset "start values" begin
        data = TestUtils.small_network()
        data["bus"]["2"]["w_start"] = 0.98
        pm = bf_model(data, bf_nothing!)
        eDisGo_OPF.variable_bus_voltage(pm, nw=1)
        w = PowerModels.var(pm, 1, :w)
        @test JuMP.start_value(w[2]) == 0.98
        # Without `w_start` the start is slightly above 1 p.u.
        @test JuMP.start_value(w[1]) == 1.001
        @test JuMP.start_value(w[3]) == 1.001
    end

    @testset "report" begin
        pm = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_bus_voltage(pm, nw=1)
        w = PowerModels.var(pm, 1, :w)
        for i in 1:3
            @test PowerModels.sol(pm, 1, :bus, i)[:w] == w[i]
        end
        @test !haskey(PowerModels.sol(pm, 1, :bus, 4), :w)

        pm_off = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_bus_voltage(pm_off, nw=1, report=false)
        @test haskey(PowerModels.var(pm_off, 1), :w)
        @test !haskey(PowerModels.sol(pm_off, 1), :bus)
    end

    @testset "network selection" begin
        pm = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_bus_voltage(pm, nw=2)
        @test !haskey(PowerModels.var(pm, 1), :w)
        @test all(startswith(JuMP.name(v), "2_w[") for v in PowerModels.var(pm, 2, :w))
    end
end

@testset "variable_buspair_current_magnitude_sqr" begin
    @testset "index set and lower bound" begin
        for bounded in (true, false)
            pm = bf_small_model(bf_nothing!)
            eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=1, bounded=bounded)
            ccm = PowerModels.var(pm, 1, :ccm)
            # Every branch, the virtual storage branch 3 included.
            @test bf_index_set(ccm) == Set([1, 2, 3])
            for i in 1:3
                @test JuMP.has_lower_bound(ccm[i])
                @test JuMP.lower_bound(ccm[i]) == 0.0
                @test JuMP.name(ccm[i]) == "1_ccm[$i]"
            end
        end
    end

    @testset "bounded" begin
        # ub = (rate_a·tap / vmin(f_bus))². Branch 1 gets a tap of 2, the sending
        # bus 2 of branch 2 a lower vmin. Bus 3 — the *receiving* bus of branch 2
        # and the sending bus of the storage branch 3 — gets an even lower vmin,
        # which must only show up in branch 3's bound.
        data = bf_tap_network()
        data["bus"]["2"]["vmin"] = 0.8
        data["bus"]["3"]["vmin"] = 0.5
        pm = bf_model(data, bf_nothing!)
        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=1)
        ccm = PowerModels.var(pm, 1, :ccm)
        @test JuMP.upper_bound(ccm[1]) ≈ (5.0 * 2.0 / 0.9)^2
        @test JuMP.upper_bound(ccm[2]) ≈ (5.0 / 0.8)^2
        @test JuMP.upper_bound(ccm[3]) ≈ (5.0 / 0.5)^2
    end

    @testset "missing or infinite rate_a" begin
        # No finite rating, no upper bound — the other branches are unaffected.
        data = TestUtils.small_network()
        delete!(data["branch"]["1"], "rate_a")
        data["branch"]["2"]["rate_a"] = Inf
        pm = bf_model(data, bf_nothing!)
        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=1)
        ccm = PowerModels.var(pm, 1, :ccm)
        @test !JuMP.has_upper_bound(ccm[1])
        @test !JuMP.has_upper_bound(ccm[2])
        @test JuMP.upper_bound(ccm[3]) ≈ (5.0 / 0.9)^2
    end

    @testset "unbounded" begin
        # Only the virtual storage branches keep their upper bound.
        pm = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=1, bounded=false)
        ccm = PowerModels.var(pm, 1, :ccm)
        @test !JuMP.has_upper_bound(ccm[1])
        @test !JuMP.has_upper_bound(ccm[2])
        @test JuMP.upper_bound(ccm[3]) ≈ (5.0 / 0.9)^2

        # A storage branch without rating stays unbounded as well.
        data = TestUtils.small_network()
        delete!(data["branch"]["3"], "rate_a")
        pm2 = bf_model(data, bf_nothing!)
        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm2, nw=1, bounded=false)
        @test !any(JuMP.has_upper_bound(v) for v in PowerModels.var(pm2, 1, :ccm))
    end

    @testset "start values" begin
        data = TestUtils.small_network()
        data["branch"]["2"]["ccm_start"] = 0.7
        pm = bf_model(data, bf_nothing!)
        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=1)
        ccm = PowerModels.var(pm, 1, :ccm)
        @test JuMP.start_value(ccm[2]) == 0.7
        @test JuMP.start_value(ccm[1]) == 0.0
        @test JuMP.start_value(ccm[3]) == 0.0
    end

    @testset "report" begin
        pm = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=1)
        ccm = PowerModels.var(pm, 1, :ccm)
        for i in 1:3
            @test PowerModels.sol(pm, 1, :branch, i)[:ccm] == ccm[i]
        end

        pm_off = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm_off, nw=1, report=false)
        @test haskey(PowerModels.var(pm_off, 1), :ccm)
        @test !haskey(PowerModels.sol(pm_off, 1), :branch)
    end

    @testset "network selection" begin
        pm = bf_small_model(bf_nothing!)
        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=2)
        @test !haskey(PowerModels.var(pm, 1), :ccm)
        @test all(startswith(JuMP.name(v), "2_ccm[") for v in PowerModels.var(pm, 2, :ccm))
    end

    @testset "real test grid" begin
        pm = eDisGo_OPF.instantiate_model(TestUtils.multinetwork(TestUtils.pm_network()),
                                          eDisGo_OPF.SOCBFPowerModelEdisgo, bf_nothing!)
        branches = PowerModels.ref(pm, 1, :branch)
        storage_ids = Set(i for (i, b) in branches if b["storage"])
        rated_ids = Set(i for (i, b) in branches if isfinite(get(b, "rate_a", Inf)))
        @test !isempty(storage_ids)
        @test length(storage_ids) < length(branches)

        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=1)
        ccm = PowerModels.var(pm, 1, :ccm)
        @test bf_index_set(ccm) == Set(keys(branches))
        @test Set(i for i in keys(branches) if JuMP.has_upper_bound(ccm[i])) == rated_ids

        eDisGo_OPF.variable_buspair_current_magnitude_sqr(pm, nw=2, bounded=false)
        ccm2 = PowerModels.var(pm, 2, :ccm)
        @test Set(i for i in keys(branches) if JuMP.has_upper_bound(ccm2[i])) ==
              intersect(storage_ids, rated_ids)
    end
end

@testset "constraint_voltage_magnitude_difference" begin
    @testset "voltage drop equation" begin
        # Branch 2 (bus 2 → bus 3), neither end is the slack bus. `r`, `x` and `tm`
        # are taken from the arguments, not from the data (br_r = br_x = 0.01,
        # tap = 1 there); `t_idx` is not used at all.
        pm = bf_small_model(bf_flow_variables!)
        @test bf_added(() -> eDisGo_OPF.constraint_voltage_magnitude_difference(
            pm, 1, 2, 2, 3, (2, 2, 3), (99, 99, 99), 0.03, 0.04, 2.0), pm.model) == 1
        c = only(bf_eq_constraints(pm.model))
        # w_fr/tm² - w_to == 2(r·p + x·q) - (r² + x²)·ccm
        @test bf_terms(c) == (Dict(
            bf_name(pm, 1, :w, 2) => 1 / 2.0^2,
            bf_name(pm, 1, :w, 3) => -1.0,
            bf_name(pm, 1, :p, (2, 2, 3)) => -2 * 0.03,
            bf_name(pm, 1, :q, (2, 2, 3)) => -2 * 0.04,
            bf_name(pm, 1, :ccm, 2) => 0.03^2 + 0.04^2), 0.0)
    end

    @testset "slack bus" begin
        # Branch 1 leaves the slack bus 1 (`bus_type == 3`): the squared voltage
        # there is additionally pinned to 1 p.u.
        pm = bf_small_model(bf_flow_variables!)
        @test PowerModels.ref(pm, 1, :bus, 1)["bus_type"] == 3
        @test bf_added(() -> eDisGo_OPF.constraint_voltage_magnitude_difference(
            pm, 1, 1, 1, 2, (1, 1, 2), (1, 2, 1), 0.01, 0.01, 1.0), pm.model) == 2
        c_drop, c_slack = bf_eq_constraints(pm.model)
        @test JuMP.coefficient(JuMP.constraint_object(c_drop).func,
                               PowerModels.var(pm, 1, :w, 1)) == 1.0
        @test bf_terms(c_slack) == (Dict(bf_name(pm, 1, :w, 1) => 1.0), 1.0)
    end

    @testset "two branches at the slack bus" begin
        # Only the sending bus is checked, once per call: every branch leaving the
        # slack bus repeats the (redundant, but harmless) `w == 1` equation.
        data = TestUtils.small_network()
        data["bus"]["5"] = TestUtils._bus(5)
        data["branch"]["4"] = TestUtils._branch(4, 1, 5)
        pm = bf_model(data, bf_flow_variables!)
        eDisGo_OPF.constraint_voltage_magnitude_difference(
            pm, 1, 1, 1, 2, (1, 1, 2), (1, 2, 1), 0.01, 0.01, 1.0)
        eDisGo_OPF.constraint_voltage_magnitude_difference(
            pm, 1, 4, 1, 5, (4, 1, 5), (4, 5, 1), 0.01, 0.01, 1.0)
        cs = bf_eq_constraints(pm.model)
        @test length(cs) == 4
        slack = Dict(bf_name(pm, 1, :w, 1) => 1.0)
        @test count(c -> bf_terms(c) == (slack, 1.0), cs) == 2
    end

    @testset "virtual storage branch" begin
        # The storage bus 4 has no voltage variable, so the low-level function
        # cannot be applied to the storage branch — the template
        # `constraint_voltage_magnitude_difference_radial` skips it beforehand.
        pm = bf_small_model(bf_flow_variables!)
        @test_throws KeyError eDisGo_OPF.constraint_voltage_magnitude_difference(
            pm, 1, 3, 3, 4, (3, 3, 4), (3, 4, 3), 0.01, 0.01, 1.0)
        @test isempty(bf_eq_constraints(pm.model))
    end

    @testset "network selection" begin
        pm = bf_small_model(bf_flow_variables!)
        eDisGo_OPF.constraint_voltage_magnitude_difference(
            pm, 2, 1, 1, 2, (1, 1, 2), (1, 2, 1), 0.01, 0.01, 1.0)
        cs = bf_eq_constraints(pm.model)
        @test length(cs) == 2
        @test all(startswith(name, "2_") for c in cs for name in keys(first(bf_terms(c))))
    end
end
