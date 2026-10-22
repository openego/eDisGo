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

@testset "constraint_model_current" begin
    @testset "AbstractSOCBFModelEdisgo" begin
        @testset "second-order cone on every branch" begin
            pm = bf_model(bf_tap_network(), bf_flow_variables!;
                          model_type=eDisGo_OPF.SOCBFPowerModelEdisgo)
            @test bf_added(() -> eDisGo_OPF.constraint_model_current(pm, 1),
                           pm.model) == 3
            cs = bf_quad_constraints(pm.model)
            @test length(cs) == 3
            @test JuMP.num_nonlinear_constraints(pm.model) == 0
            # The virtual storage branch 3 is included.
            @test bf_current_model_branches(pm) == Set([1, 2, 3])

            for (idx, arc, f_bus, tm) in ((1, (1, 1, 2), 1, 2.0), (2, (2, 2, 3), 2, 1.0),
                                          (3, (3, 3, 4), 3, 1.0))
                c = only(filter(x -> occursin("1_ccm[$idx]", string(x)), cs))
                p = PowerModels.var(pm, 1, :p, arc)
                q = PowerModels.var(pm, 1, :q, arc)
                w = PowerModels.var(pm, 1, :w, f_bus)
                ccm = PowerModels.var(pm, 1, :ccm, idx)
                # p² + q² ≤ (w_fr/tm²)·ccm
                @test bf_coef2(c, p, p) == 1.0
                @test bf_coef2(c, q, q) == 1.0
                @test bf_coef2(c, w, ccm) ≈ -1 / tm^2
                @test length(JuMP.constraint_object(c).func.terms) == 3
                @test isempty(JuMP.constraint_object(c).func.aff.terms)
                @test JuMP.constraint_object(c).set == MOI.LessThan(0.0)
                vals = Dict(p => 0.3, q => 0.4, w => 0.81, ccm => 0.5)
                @test bf_eval(c, vals) ≈ 0.3^2 + 0.4^2 - 0.81 / tm^2 * 0.5
            end
        end

        @testset "missing variables" begin
            pm = bf_small_model(bf_flow_variables_without_ccm!;
                                model_type=eDisGo_OPF.SOCBFPowerModelEdisgo)
            @test_throws ErrorException eDisGo_OPF.constraint_model_current(pm, 1)
            @test isempty(bf_quad_constraints(pm.model))
        end

        @testset "network selection" begin
            pm = bf_small_model(bf_flow_variables!;
                                model_type=eDisGo_OPF.SOCBFPowerModelEdisgo)
            eDisGo_OPF.constraint_model_current(pm, 2)
            cs = bf_quad_constraints(pm.model)
            @test length(cs) == 3
            @test all(occursin("2_", string(c)) && !occursin("1_", string(c)) for c in cs)
        end
    end

    @testset "AbstractNCBFModelEdisgo" begin
        @testset "exact equality on physical branches" begin
            pm = bf_model(bf_tap_network(), bf_flow_variables!;
                          model_type=eDisGo_OPF.NCBFPowerModelEdisgo)
            # `@NLconstraint` ends up in JuMP's legacy nonlinear model, which
            # `num_constraints` does not count.
            @test bf_added(() -> eDisGo_OPF.constraint_model_current(pm, 1),
                           pm.model) == 0
            @test isempty(bf_quad_constraints(pm.model))
            @test JuMP.num_nonlinear_constraints(pm.model) == 2
            # The virtual storage branch 3 is skipped.
            @test bf_current_model_branches(pm) == Set([1, 2])

            cs = JuMP.all_nonlinear_constraints(pm.model)
            for (idx, arc, f_bus, tm) in ((1, (1, 1, 2), 1, 2.0), (2, (2, 2, 3), 2, 1.0))
                c = only(filter(x -> occursin("1_ccm[$idx]", string(x)), cs))
                s = string(c)
                @test occursin("= 0", s) && !occursin("<", s) && !occursin(">", s)
                p = PowerModels.var(pm, 1, :p, arc)
                q = PowerModels.var(pm, 1, :q, arc)
                w = PowerModels.var(pm, 1, :w, f_bus)
                ccm = PowerModels.var(pm, 1, :ccm, idx)
                # p² + q² - (w_fr/tm²)·ccm, evaluated at two points to pin both the
                # quadratic flow terms and the bilinear coupling term.
                @test bf_eval(c, Dict(p => 0.3, q => 0.4, w => 0.81, ccm => 0.5)) ≈
                      0.3^2 + 0.4^2 - 0.81 / tm^2 * 0.5
                @test bf_eval(c, Dict(p => -0.2, w => 1.1, ccm => 2.0)) ≈
                      0.2^2 - 1.1 / tm^2 * 2.0
            end
        end

        @testset "storage flag decides" begin
            # Flagging the physical branch 2 as storage branch drops its equation.
            pm = bf_small_model(bf_flow_variables!;
                                model_type=eDisGo_OPF.NCBFPowerModelEdisgo)
            PowerModels.ref(pm, 1, :branch, 2)["storage"] = true
            eDisGo_OPF.constraint_model_current(pm, 1)
            @test bf_current_model_branches(pm) == Set([1])
        end

        @testset "missing variables" begin
            pm = bf_small_model(bf_flow_variables_without_ccm!;
                                model_type=eDisGo_OPF.NCBFPowerModelEdisgo)
            @test_throws ErrorException eDisGo_OPF.constraint_model_current(pm, 1)
            @test JuMP.num_nonlinear_constraints(pm.model) == 0
        end

        @testset "network selection" begin
            pm = bf_small_model(bf_flow_variables!;
                                model_type=eDisGo_OPF.NCBFPowerModelEdisgo)
            eDisGo_OPF.constraint_model_current(pm, 2)
            strs = string.(JuMP.all_nonlinear_constraints(pm.model))
            @test length(strs) == 2
            @test all(occursin("2_", s) && !occursin("1_", s) for s in strs)
        end
    end

    @testset "SOC and NC cover the same branches" begin
        # Known defect: the SOC relaxation is written for the virtual storage
        # branch, the exact NC equality is not. Intended: both formulations treat
        # the storage branches alike.
        pm_soc = bf_small_model(bf_flow_variables!;
                                model_type=eDisGo_OPF.SOCBFPowerModelEdisgo)
        pm_nc = bf_small_model(bf_flow_variables!;
                               model_type=eDisGo_OPF.NCBFPowerModelEdisgo)
        eDisGo_OPF.constraint_model_current(pm_soc, 1)
        eDisGo_OPF.constraint_model_current(pm_nc, 1)
        soc_branches = bf_current_model_branches(pm_soc)
        nc_branches = bf_current_model_branches(pm_nc)
        @test setdiff(soc_branches, nc_branches) == Set([3])
        @test_broken soc_branches == nc_branches
    end
end

@testset "constraint_max_line_loading" begin
    @testset "AbstractSOCBFModelEdisgo" begin
        bf_check_max_line_loading(eDisGo_OPF.SOCBFPowerModelEdisgo)
    end

    @testset "AbstractNCBFModelEdisgo" begin
        # Plain quadratic `@constraint`, not `@NLconstraint`, in the NC model too.
        bf_check_max_line_loading(eDisGo_OPF.NCBFPowerModelEdisgo)
    end
end

@testset "constraint_power_balance" begin
    @testset "with grid restriction slacks" begin
        # opf_version 2: the balance includes the load-shedding and curtailment
        # slack variables. One component of each kind, each with its own
        # parameter value (see `bf_balance_args`).
        pm = bf_small_model(bf_all_variables!; opf_version=2)
        @test bf_added(() -> bf_power_balance(pm), pm.model) == 2
        cstr_p, cstr_q = bf_eq_constraints(pm.model)
        v(sym, idx) = bf_name(pm, 1, sym, idx)

        # Σpt == Σpf + Σccm·r - Σpgs - Σpg - Σpg_nd - Σps + Σpd - Σpds + Σpgens
        #        + Σpgc + Σpdsm + Σ(php - phps) + Σ(pcp - pcps),
        # constants moved to the right-hand side: -(0.3 + 0.6 - 0.2).
        @test bf_terms(cstr_p)[1] == Dict(
            v(:p, (2, 2, 3)) => 1.0, v(:p, (3, 3, 4)) => -1.0, v(:ccm, 2) => -0.03,
            v(:pgs, 1) => 1.0, v(:ps, 1) => 1.0, v(:pds, 1) => 1.0,
            v(:pgens, 1) => -1.0, v(:pgc, 1) => -1.0, v(:pdsm, 1) => -1.0,
            v(:php, 1) => -1.0, v(:phps, 1) => 1.0, v(:pcp, 1) => -1.0,
            v(:pcps, 1) => 1.0)
        @test bf_terms(cstr_p)[2] ≈ -(0.3 + 0.6 - 0.2)

        # Σqt - Σpt·strg_pf == Σqf + Σccm·x - Σqgs - Σqg - Σqg_nd + Σqd
        #   - Σpds·pf_load + Σpgc·pf_gen_nd + Σpgens·pf_gen_d + Σpdsm·pf_dsm
        #   + Σ(php - phps)·pf_hp + Σ(pcp - pcps)·pf_cp
        @test bf_terms(cstr_q)[1] == Dict(
            v(:q, (2, 2, 3)) => 1.0, v(:p, (2, 2, 3)) => -0.18,
            v(:q, (3, 3, 4)) => -1.0, v(:ccm, 2) => -0.04, v(:qgs, 1) => 1.0,
            v(:pds, 1) => 0.17, v(:pgc, 1) => -0.15, v(:pgens, 1) => -0.16,
            v(:pdsm, 1) => -0.12, v(:php, 1) => -0.13, v(:phps, 1) => 0.13,
            v(:pcp, 1) => -0.14, v(:pcps, 1) => 0.14)
        @test bf_terms(cstr_q)[2] ≈ -(-0.1 - 0.2 - 0.07)
    end

    @testset "without grid restriction slacks" begin
        # opf_version 1: no slack variables exist, none are used; the power
        # factors of generators and loads have nothing left to multiply.
        pm = bf_small_model(bf_all_variables!; opf_version=1)
        for sym in (:pgc, :pgens, :pds, :phps, :pcps)
            @test !haskey(PowerModels.var(pm, 1), sym)
        end
        @test bf_added(() -> bf_power_balance(pm), pm.model) == 2
        cstr_p, cstr_q = bf_eq_constraints(pm.model)
        v(sym, idx) = bf_name(pm, 1, sym, idx)
        @test bf_terms(cstr_p)[1] == Dict(
            v(:p, (2, 2, 3)) => 1.0, v(:p, (3, 3, 4)) => -1.0, v(:ccm, 2) => -0.03,
            v(:pgs, 1) => 1.0, v(:ps, 1) => 1.0, v(:pdsm, 1) => -1.0,
            v(:php, 1) => -1.0, v(:pcp, 1) => -1.0)
        @test bf_terms(cstr_p)[2] ≈ -(0.3 + 0.6 - 0.2)
        @test bf_terms(cstr_q)[1] == Dict(
            v(:q, (2, 2, 3)) => 1.0, v(:p, (2, 2, 3)) => -0.18,
            v(:q, (3, 3, 4)) => -1.0, v(:ccm, 2) => -0.04, v(:qgs, 1) => 1.0,
            v(:pdsm, 1) => -0.12, v(:php, 1) => -0.13, v(:pcp, 1) => -0.14)
        @test bf_terms(cstr_q)[2] ≈ -(-0.1 - 0.2 - 0.07)
    end

    @testset "storage power factor is unused" begin
        # `bus_storage_pf` is part of the signature but enters neither equation:
        # the battery's reactive power comes in via `branch_strg_pf` of its
        # virtual branch instead.
        pm_a = bf_small_model(bf_all_variables!)
        pm_b = bf_small_model(bf_all_variables!)
        bf_power_balance(pm_a; bus_storage_pf=Dict(1 => 0.11))
        bf_power_balance(pm_b; bus_storage_pf=Dict(1 => 0.99))
        @test bf_terms.(bf_eq_constraints(pm_a.model)) ==
              bf_terms.(bf_eq_constraints(pm_b.model))
        _, cstr_q = bf_eq_constraints(pm_a.model)
        @test bf_coef(cstr_q, PowerModels.var(pm_a, 1, :ps, 1)) == 0.0
    end

    @testset "bus with lines only" begin
        # Empty component collections reduce the balance to the line flows and
        # losses; the empty sums do not break the constraint.
        pm = bf_small_model(bf_all_variables!)
        empty_ids = Int[]
        no_values = Dict{Int,Float64}()
        @test bf_added(() -> bf_power_balance(pm;
            bus_gens=empty_ids, bus_gens_nd=empty_ids, bus_gens_slack=empty_ids,
            bus_loads=empty_ids, bus_storage=empty_ids, bus_dsm=empty_ids,
            bus_hps=empty_ids, bus_cps=empty_ids,
            bus_pg=no_values, bus_qg=no_values, bus_pg_nd=no_values,
            bus_qg_nd=no_values, bus_pd=no_values, bus_qd=no_values,
            branch_strg_pf=Dict(2 => 0.0)), pm.model) == 2
        cstr_p, cstr_q = bf_eq_constraints(pm.model)
        v(sym, idx) = bf_name(pm, 1, sym, idx)
        @test bf_terms(cstr_p) == (Dict(v(:p, (2, 2, 3)) => 1.0,
                                        v(:p, (3, 3, 4)) => -1.0,
                                        v(:ccm, 2) => -0.03), 0.0)
        # A zero storage power factor leaves at most a vanishing `pt` term.
        q_terms, q_rhs = bf_terms(cstr_q)
        @test get(q_terms, v(:p, (2, 2, 3)), 0.0) == 0.0
        @test filter(kv -> kv[2] != 0.0, q_terms) == Dict(v(:q, (2, 2, 3)) => 1.0,
                                                          v(:q, (3, 3, 4)) => -1.0,
                                                          v(:ccm, 2) => -0.04)
        @test q_rhs == 0.0
    end

    @testset "missing variables" begin
        # `_check_var_keys` only compares counts: asking for two storage units
        # when only one `ps` exists fails before anything is added.
        pm = bf_small_model(bf_all_variables!)
        @test_throws ErrorException bf_power_balance(pm; bus_storage=[1, 2])
        @test isempty(bf_eq_constraints(pm.model))
        # The same for a slack variable of opf_version 2 that was never created.
        pm2 = bf_small_model(bf_flow_variables!)
        @test_throws ErrorException bf_power_balance(pm2)
        @test isempty(bf_eq_constraints(pm2.model))
    end

    @testset "opf_version is read from network 1" begin
        # The case distinction looks at network 1 only, whatever `n` is.
        pm = bf_small_model(bf_all_variables!; opf_version=2)
        PowerModels.ref(pm, 2)[:opf_version] = 1
        bf_power_balance(pm, 2)
        cstr_p, _ = bf_eq_constraints(pm.model)
        @test bf_coef(cstr_p, PowerModels.var(pm, 2, :pds, 1)) == 1.0
        @test all(startswith(name, "2_") for name in keys(first(bf_terms(cstr_p))))
    end

    @testset "opf_version 4" begin
        # Version 4 takes the same branch as version 2.
        pm = bf_small_model(bf_all_variables!; opf_version=4, overlying_grid=true)
        bf_power_balance(pm)
        cstr_p, cstr_q = bf_eq_constraints(pm.model)
        @test bf_coef(cstr_p, PowerModels.var(pm, 1, :pds, 1)) == 1.0
        @test bf_coef(cstr_p, PowerModels.var(pm, 1, :pgc, 1)) == -1.0
        @test bf_coef(cstr_q, PowerModels.var(pm, 1, :pgc, 1)) ≈ -0.15
    end

    @testset "opf_version 3 ignores curtailment" begin
        # Version 3 creates the curtailment `pgc` and counts it towards the "curt"
        # requirement of the overlying grid, but the balance takes the branch of
        # version 1 and leaves `pgc` out: curtailment that serves the overlying
        # grid does not reduce the feed-in the grid has to carry.
        pm = bf_small_model(bf_all_variables!; opf_version=3, overlying_grid=true)
        pgc = PowerModels.var(pm, 1, :pgc, 1)
        bf_power_balance(pm)
        cstr_p, cstr_q = bf_eq_constraints(pm.model)
        @test bf_coef(cstr_p, pgc) == 0.0
        @test bf_coef(cstr_q, pgc) == 0.0

        i_curt = only(i for (i, req) in PowerModels.ref(pm, 1, :HV_requirements)
                      if req["name"] == "curt")
        eDisGo_OPF.constraint_HV_requirements(pm, i_curt, 1)
        c_hv = bf_eq_constraints(pm.model)[end]
        @test bf_coef(c_hv, pgc) == 1.0

        # Intended: curtailment enters the balance as in versions 2 and 4.
        @test_broken bf_coef(cstr_p, pgc) == -1.0
    end

    @testset "duals" begin
        # With `output => duals` in the setting both equations are stored in the
        # solution of the bus …
        setting = Dict{String,Any}("output" => Dict{String,Any}("duals" => true))
        pm = bf_model(TestUtils.small_network(), bf_all_variables!; setting=setting)
        @test InfrastructureModels.report_duals(pm)
        bf_power_balance(pm, 1, 3)
        cstr_p, cstr_q = bf_eq_constraints(pm.model)
        @test PowerModels.sol(pm, 1, :bus, 3)[:lam_kcl_r] == cstr_p
        @test PowerModels.sol(pm, 1, :bus, 3)[:lam_kcl_i] == cstr_q

        # … without it, nothing is reported.
        pm_off = bf_small_model(bf_all_variables!)
        @test !InfrastructureModels.report_duals(pm_off)
        bf_power_balance(pm_off, 1, 3)
        @test !haskey(PowerModels.sol(pm_off, 1, :bus, 3), :lam_kcl_r)
        @test !haskey(PowerModels.sol(pm_off, 1, :bus, 3), :lam_kcl_i)
    end
end
