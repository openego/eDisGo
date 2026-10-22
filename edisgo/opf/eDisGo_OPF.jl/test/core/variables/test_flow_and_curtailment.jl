# Tests for the branch-flow, bus-voltage, line-loading, curtailment, battery and
# DSM variables of `src/core/variables.jl` — the first block of the file, from
# `variable_branch_power_radial` through `variable_dsm_storage_energy`.
#
# Every function under test creates one `JuMP` variable container, stores it in
# `PowerModels.var(pm, nw)` and, with `report=true`, links its entries into the
# solution dict. The tests therefore run each function on a model whose build
# method creates *nothing* (`vfc_build_nothing`), so every variable in the model
# afterwards is one the function under test added. For each function they check
#
#   * the key in `PowerModels.var` and that no other key appears,
#   * the index set — including the components that are deliberately left out
#     (storage buses for `w`, the virtual storage branch for `ll`),
#   * the bounds for `bounded=true` and `bounded=false`,
#   * the start values taken from the `*_start` fields (and their defaults),
#   * that `report=true` writes into the solution dict and `report=false` does not,
#   * that the `nw` keyword is honoured (variables of network 2 read network 2's
#     data and are named `2_<name>`).
#
# The wrappers (`variable_branch_power_radial`, `variable_gen_power_curt`, ...)
# are checked for which inner functions they call and that they forward their
# keyword arguments.
#
# Production defects documented here (actual behaviour tested, intended
# behaviour kept as `@test_broken`):
#
#   * `variable_branch_power_real_radial`/`_imaginary_radial` index `p`/`q` by
#     `arcs_from` only, but still try to set a start value on the to-arc for
#     `pt_start`/`qt_start` — that throws a `KeyError`.
#   * `variable_line_loading_max` decides which branches are storage branches by
#     looking at network 1 (`ref(pm, 1, :branch)`) instead of network `nw`.

include(joinpath(@__DIR__, "..", "..", "testsetup.jl"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"Build method that creates no variable and no constraint at all."
vfc_build_nothing(pm) = nothing

"Empty (variable-free) model on the given single-network data dict."
vfc_model_from(data::Dict) = eDisGo_OPF.instantiate_model(
    TestUtils.multinetwork(data), eDisGo_OPF.SOCBFPowerModelEdisgo, vfc_build_nothing)

"Empty model on [`TestUtils.small_network`](@ref); keywords go to `small_network`."
vfc_model(; kwargs...) = vfc_model_from(TestUtils.small_network(; kwargs...))

"Empty model on the real eDisGo test grid; keywords go to `TestUtils.pm_network`."
vfc_big_model(; kwargs...) = TestUtils.build_model(
    eDisGo_OPF.SOCBFPowerModelEdisgo; build_method=vfc_build_nothing, kwargs...)

"Index set of the one-dimensional variable container `v`."
vfc_index(v) = Set(axes(v, 1))

"`(lower, upper)` bound of `x`, with `nothing` for a missing bound."
vfc_bounds(x::JuMP.VariableRef) = (
    JuMP.has_lower_bound(x) ? JuMP.lower_bound(x) : nothing,
    JuMP.has_upper_bound(x) ? JuMP.upper_bound(x) : nothing,
)

"Whether no entry of the variable container `v` carries a lower or upper bound."
vfc_unbounded(v) = all(vfc_bounds(x) == (nothing, nothing) for x in v)

"Solution dict of network `nw` of `pm` (read without creating entries)."
vfc_sol(pm, nw::Int) = pm.sol[:it][PowerModels.pm_it_sym][:nw][nw]

"""
    vfc_reported(pm, nw, comp, field)

All entries of component type `comp` in the solution dict of network `nw` that
carry `field`, as an `id => value` dict (empty if nothing was reported).
"""
function vfc_reported(pm, nw::Int, comp::Symbol, field::Symbol)
    sol = vfc_sol(pm, nw)
    haskey(sol, comp) || return Dict{Int,Any}()
    return Dict{Int,Any}(i => e[field] for (i, e) in sol[comp] if haskey(e, field))
end

"Arcs `(l, f_bus, t_bus)` of the small network's three branches."
const VFC_ARCS = Set([(1, 1, 2), (2, 2, 3), (3, 3, 4)])

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@testset "variable_branch_power_radial" begin
    @testset "creates p and q" begin
        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_radial(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:p, :q]
        @test vfc_index(PowerModels.var(pm, 1, :p)) == VFC_ARCS
        @test vfc_index(PowerModels.var(pm, 1, :q)) == VFC_ARCS
        # Nothing but the two flow variables, one per arc each.
        @test JuMP.num_variables(pm.model) == 2 * length(VFC_ARCS)
        # Both are reported.
        @test Set(keys(vfc_reported(pm, 1, :branch, :pf))) == Set([1, 2, 3])
        @test Set(keys(vfc_reported(pm, 1, :branch, :qf))) == Set([1, 2, 3])
        # Other networks are left alone.
        @test isempty(PowerModels.var(pm, 2))
    end

    @testset "forwards keyword arguments" begin
        # This is the call `build_mn_opf_bf_flex` makes.
        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_radial(pm, nw=2, bounded=false, report=false)
        @test isempty(PowerModels.var(pm, 1))
        p = PowerModels.var(pm, 2, :p)
        q = PowerModels.var(pm, 2, :q)
        @test vfc_unbounded(p)
        @test vfc_unbounded(q)
        @test startswith(JuMP.name(p[(1, 1, 2)]), "2_p[")
        @test startswith(JuMP.name(q[(1, 1, 2)]), "2_q[")
        @test isempty(vfc_reported(pm, 2, :branch, :pf))
        @test isempty(vfc_reported(pm, 2, :branch, :qf))

        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_radial(pm, nw=1, bounded=true)
        @test vfc_bounds(PowerModels.var(pm, 1, :p)[(1, 1, 2)]) == (-5.0, 5.0)
        @test vfc_bounds(PowerModels.var(pm, 1, :q)[(1, 1, 2)]) == (-5.0, 5.0)
    end
end

@testset "variable_branch_power_real_radial" begin
    @testset "index set" begin
        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:p]
        p = PowerModels.var(pm, 1, :p)
        # One variable per *from*-arc — the radial model has no to-side flow —
        # and the virtual storage branch 3 is included.
        @test vfc_index(p) == VFC_ARCS
        @test Set(vfc_index(p)) == Set(PowerModels.ref(pm, 1, :arcs_from))
        @test !any(arc in vfc_index(p) for arc in PowerModels.ref(pm, 1, :arcs_to))
        @test JuMP.name(p[(2, 2, 3)]) == "1_p[(2, 2, 3)]"
    end

    @testset "real grid" begin
        pm = vfc_big_model()
        eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1)
        p = PowerModels.var(pm, 1, :p)
        @test length(p) == length(PowerModels.ref(pm, 1, :branch)) == 142
        @test vfc_index(p) == Set(PowerModels.ref(pm, 1, :arcs_from))
    end

    @testset "bounded=true" begin
        pm = vfc_model()
        # Branch 1: `c_rating_a * max(vmax)` is tighter than `rate_a` (5.0).
        PowerModels.ref(pm, 1, :branch, 1)["c_rating_a"] = 2.0
        # Branch 2: no rating at all, so there is nothing to bound it with.
        delete!(PowerModels.ref(pm, 1, :branch, 2), "rate_a")
        eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1, bounded=true)
        p = PowerModels.var(pm, 1, :p)
        lb, ub = vfc_bounds(p[(1, 1, 2)])
        @test lb ≈ -2.2
        @test ub ≈ 2.2
        @test vfc_bounds(p[(2, 2, 3)]) == (nothing, nothing)
        @test vfc_bounds(p[(3, 3, 4)]) == (-5.0, 5.0)
    end

    @testset "bounded=false" begin
        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1, bounded=false)
        @test vfc_unbounded(PowerModels.var(pm, 1, :p))
    end

    @testset "start values" begin
        pm = vfc_model()
        PowerModels.ref(pm, 1, :branch, 1)["p_start"] = 0.11
        PowerModels.ref(pm, 1, :branch, 2)["p_start"] = 0.22
        # `pf_start` takes precedence over `p_start` on the from-arc.
        PowerModels.ref(pm, 1, :branch, 2)["pf_start"] = 0.33
        eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1)
        p = PowerModels.var(pm, 1, :p)
        @test JuMP.start_value(p[(1, 1, 2)]) == 0.11
        @test JuMP.start_value(p[(2, 2, 3)]) == 0.33
        # Default start value when no field is given.
        @test JuMP.start_value(p[(3, 3, 4)]) == 0.0
    end

    @testset "pt_start (defect)" begin
        # Actual: `p` only exists for the from-arcs, but a `pt_start` field still
        # makes the function set a start value on the to-arc `(1, 2, 1)` — which
        # is not part of the container, so the call throws.
        pm = vfc_model()
        PowerModels.ref(pm, 1, :branch, 1)["pt_start"] = 0.2
        @test_throws KeyError eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1)
        # Intended: the field is ignored (or rejected with a clear message).
        pm = vfc_model()
        PowerModels.ref(pm, 1, :branch, 1)["pt_start"] = 0.2
        @test_broken try
            eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1)
            true
        catch
            false
        end
    end

    @testset "report" begin
        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1, report=true)
        p = PowerModels.var(pm, 1, :p)
        reported = vfc_reported(pm, 1, :branch, :pf)
        @test reported == Dict(1 => p[(1, 1, 2)], 2 => p[(2, 2, 3)], 3 => p[(3, 3, 4)])
        # No to-side field is written.
        @test isempty(vfc_reported(pm, 1, :branch, :pt))

        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_real_radial(pm, nw=1, report=false)
        @test isempty(vfc_reported(pm, 1, :branch, :pf))
    end

    @testset "nw keyword" begin
        pm = vfc_model()
        PowerModels.ref(pm, 2, :branch, 1)["p_start"] = 0.7
        PowerModels.ref(pm, 2, :branch, 1)["rate_a"] = 3.0
        eDisGo_OPF.variable_branch_power_real_radial(pm, nw=2)
        @test isempty(PowerModels.var(pm, 1))
        p = PowerModels.var(pm, 2, :p)
        @test JuMP.name(p[(1, 1, 2)]) == "2_p[(1, 1, 2)]"
        @test JuMP.start_value(p[(1, 1, 2)]) == 0.7
        @test vfc_bounds(p[(1, 1, 2)]) == (-3.0, 3.0)
        @test length(vfc_reported(pm, 2, :branch, :pf)) == 3
        @test isempty(vfc_reported(pm, 1, :branch, :pf))
    end
end

@testset "variable_branch_power_imaginary_radial" begin
    @testset "index set" begin
        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:q]
        q = PowerModels.var(pm, 1, :q)
        # Like `p`, despite the docstring speaking of `arcs`: from-arcs only,
        # storage branch included.
        @test vfc_index(q) == VFC_ARCS
        @test !any(arc in vfc_index(q) for arc in PowerModels.ref(pm, 1, :arcs_to))
        @test JuMP.name(q[(2, 2, 3)]) == "1_q[(2, 2, 3)]"
    end

    @testset "bounded=true" begin
        # The reactive flow is bounded by the same apparent-power rating as the
        # active flow.
        pm = vfc_model()
        PowerModels.ref(pm, 1, :branch, 1)["c_rating_a"] = 2.0
        delete!(PowerModels.ref(pm, 1, :branch, 2), "rate_a")
        eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=1, bounded=true)
        q = PowerModels.var(pm, 1, :q)
        lb, ub = vfc_bounds(q[(1, 1, 2)])
        @test lb ≈ -2.2
        @test ub ≈ 2.2
        @test vfc_bounds(q[(2, 2, 3)]) == (nothing, nothing)
        @test vfc_bounds(q[(3, 3, 4)]) == (-5.0, 5.0)
    end

    @testset "bounded=false" begin
        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=1, bounded=false)
        @test vfc_unbounded(PowerModels.var(pm, 1, :q))
    end

    @testset "start values" begin
        pm = vfc_model()
        PowerModels.ref(pm, 1, :branch, 1)["q_start"] = -0.11
        PowerModels.ref(pm, 1, :branch, 2)["q_start"] = -0.22
        PowerModels.ref(pm, 1, :branch, 2)["qf_start"] = -0.33
        # The active-power fields must not leak into `q`.
        PowerModels.ref(pm, 1, :branch, 3)["p_start"] = 0.5
        eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=1)
        q = PowerModels.var(pm, 1, :q)
        @test JuMP.start_value(q[(1, 1, 2)]) == -0.11
        @test JuMP.start_value(q[(2, 2, 3)]) == -0.33
        @test JuMP.start_value(q[(3, 3, 4)]) == 0.0
    end

    @testset "qt_start (defect)" begin
        # Same defect as `pt_start` for the active flow: the to-arc does not exist.
        pm = vfc_model()
        PowerModels.ref(pm, 1, :branch, 1)["qt_start"] = 0.2
        @test_throws KeyError eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=1)
        pm = vfc_model()
        PowerModels.ref(pm, 1, :branch, 1)["qt_start"] = 0.2
        @test_broken try
            eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=1)
            true
        catch
            false
        end
    end

    @testset "report" begin
        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=1, report=true)
        q = PowerModels.var(pm, 1, :q)
        @test vfc_reported(pm, 1, :branch, :qf) ==
              Dict(1 => q[(1, 1, 2)], 2 => q[(2, 2, 3)], 3 => q[(3, 3, 4)])
        @test isempty(vfc_reported(pm, 1, :branch, :qt))

        pm = vfc_model()
        eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=1, report=false)
        @test isempty(vfc_reported(pm, 1, :branch, :qf))
    end

    @testset "nw keyword" begin
        pm = vfc_model()
        PowerModels.ref(pm, 2, :branch, 1)["q_start"] = -0.7
        eDisGo_OPF.variable_branch_power_imaginary_radial(pm, nw=2)
        @test isempty(PowerModels.var(pm, 1))
        q = PowerModels.var(pm, 2, :q)
        @test JuMP.name(q[(1, 1, 2)]) == "2_q[(1, 1, 2)]"
        @test JuMP.start_value(q[(1, 1, 2)]) == -0.7
        @test length(vfc_reported(pm, 2, :branch, :qf)) == 3
    end
end

@testset "variable_bus_voltage_magnitude_sqr" begin
    @testset "index set excludes storage buses" begin
        pm = vfc_model()
        eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:w]
        w = PowerModels.var(pm, 1, :w)
        # Bus 4 is the storage bus behind the virtual branch; it has no voltage.
        @test vfc_index(w) == Set([1, 2, 3])
        @test !(4 in vfc_index(w))
        @test JuMP.name(w[2]) == "1_w[2]"
    end

    @testset "real grid" begin
        pm = vfc_big_model()
        eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm, nw=1)
        w = PowerModels.var(pm, 1, :w)
        buses = PowerModels.ref(pm, 1, :bus)
        storage_buses = Set(i for (i, b) in buses if b["storage"])
        @test length(storage_buses) == 1
        @test vfc_index(w) == setdiff(Set(keys(buses)), storage_buses)
        @test length(w) == 142
    end

    @testset "bounded=true" begin
        pm = vfc_model()
        PowerModels.ref(pm, 1, :bus, 2)["vmin"] = 0.95
        PowerModels.ref(pm, 1, :bus, 2)["vmax"] = 1.05
        eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm, nw=1, bounded=true)
        w = PowerModels.var(pm, 1, :w)
        # Bounds are the squared voltage limits.
        for i in (1, 3)
            lb, ub = vfc_bounds(w[i])
            @test lb ≈ 0.9^2
            @test ub ≈ 1.1^2
        end
        lb, ub = vfc_bounds(w[2])
        @test lb ≈ 0.95^2
        @test ub ≈ 1.05^2
    end

    @testset "bounded=false" begin
        # Without bounds only the physical lower bound 0 of a squared quantity
        # remains.
        pm = vfc_model()
        eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm, nw=1, bounded=false)
        w = PowerModels.var(pm, 1, :w)
        @test all(vfc_bounds(w[i]) == (0.0, nothing) for i in (1, 2, 3))
    end

    @testset "start values" begin
        pm = vfc_model()
        PowerModels.ref(pm, 1, :bus, 2)["w_start"] = 1.02
        eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm, nw=1)
        w = PowerModels.var(pm, 1, :w)
        @test JuMP.start_value(w[2]) == 1.02
        # Default start value is slightly above 1, not PowerModels' 0.
        @test JuMP.start_value(w[1]) == 1.001
        @test JuMP.start_value(w[3]) == 1.001
    end

    @testset "report" begin
        pm = vfc_model()
        eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm, nw=1, report=true)
        w = PowerModels.var(pm, 1, :w)
        @test vfc_reported(pm, 1, :bus, :w) == Dict(1 => w[1], 2 => w[2], 3 => w[3])

        pm = vfc_model()
        eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm, nw=1, report=false)
        @test isempty(vfc_reported(pm, 1, :bus, :w))
    end

    @testset "nw keyword" begin
        pm = vfc_model()
        PowerModels.ref(pm, 2, :bus, 3)["vmax"] = 1.05
        PowerModels.ref(pm, 2, :bus, 3)["w_start"] = 0.98
        eDisGo_OPF.variable_bus_voltage_magnitude_sqr(pm, nw=2)
        @test isempty(PowerModels.var(pm, 1))
        w = PowerModels.var(pm, 2, :w)
        @test JuMP.name(w[3]) == "2_w[3]"
        @test vfc_bounds(w[3])[2] ≈ 1.05^2
        @test JuMP.start_value(w[3]) == 0.98
        @test length(vfc_reported(pm, 2, :bus, :w)) == 3
    end
end

@testset "variable_max_line_loading" begin
    @testset "creates ll" begin
        pm = vfc_model()
        eDisGo_OPF.variable_max_line_loading(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:ll]
        @test vfc_index(PowerModels.var(pm, 1, :ll)) == Set([(1, 1, 2), (2, 2, 3)])
        @test length(vfc_reported(pm, 1, :branch, :ll)) == 2
    end

    @testset "forwards keyword arguments" begin
        pm = vfc_model()
        eDisGo_OPF.variable_max_line_loading(pm, nw=2, report=false)
        @test isempty(PowerModels.var(pm, 1))
        @test startswith(JuMP.name(PowerModels.var(pm, 2, :ll)[(1, 1, 2)]), "2_ll[")
        @test isempty(vfc_reported(pm, 2, :branch, :ll))
    end

    @testset "no bounded keyword" begin
        # Unlike the other wrappers, `bounded` is not accepted: the inner function
        # has no such keyword (the lower bound of 1 is part of the definition).
        pm = vfc_model()
        @test_throws MethodError eDisGo_OPF.variable_max_line_loading(pm, nw=1,
                                                                      bounded=false)
    end
end

@testset "variable_line_loading_max" begin
    @testset "index set excludes storage branches" begin
        pm = vfc_model()
        eDisGo_OPF.variable_line_loading_max(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:ll]
        ll = PowerModels.var(pm, 1, :ll)
        # Branch 3 is the virtual storage branch.
        @test vfc_index(ll) == Set([(1, 1, 2), (2, 2, 3)])
        @test !((3, 3, 4) in vfc_index(ll))
        @test JuMP.name(ll[(1, 1, 2)]) == "1_ll[(1, 1, 2)]"
    end

    @testset "real grid" begin
        pm = vfc_big_model()
        eDisGo_OPF.variable_line_loading_max(pm, nw=1)
        ll = PowerModels.var(pm, 1, :ll)
        branches = PowerModels.ref(pm, 1, :branch)
        @test vfc_index(ll) == Set((l, b["f_bus"], b["t_bus"])
                                   for (l, b) in branches if !b["storage"])
        @test length(ll) == 141
    end

    @testset "bounds" begin
        # The loading factor is bounded below by 1 (= no line reinforcement
        # needed) and unbounded above; there is no `bounded` switch.
        pm = vfc_model()
        eDisGo_OPF.variable_line_loading_max(pm, nw=1)
        ll = PowerModels.var(pm, 1, :ll)
        @test all(vfc_bounds(x) == (1.0, nothing) for x in ll)
    end

    @testset "start values" begin
        pm = vfc_model()
        PowerModels.ref(pm, 1, :branch, 1)["ll_start"] = 1.5
        eDisGo_OPF.variable_line_loading_max(pm, nw=1)
        ll = PowerModels.var(pm, 1, :ll)
        @test JuMP.start_value(ll[(1, 1, 2)]) == 1.5
        # The default start value 0 lies below the lower bound 1.
        @test JuMP.start_value(ll[(2, 2, 3)]) == 0.0
    end

    @testset "report" begin
        pm = vfc_model()
        eDisGo_OPF.variable_line_loading_max(pm, nw=1, report=true)
        ll = PowerModels.var(pm, 1, :ll)
        @test vfc_reported(pm, 1, :branch, :ll) ==
              Dict(1 => ll[(1, 1, 2)], 2 => ll[(2, 2, 3)])

        pm = vfc_model()
        eDisGo_OPF.variable_line_loading_max(pm, nw=1, report=false)
        @test isempty(vfc_reported(pm, 1, :branch, :ll))
    end

    @testset "nw keyword" begin
        pm = vfc_model()
        PowerModels.ref(pm, 2, :branch, 2)["ll_start"] = 2.5
        eDisGo_OPF.variable_line_loading_max(pm, nw=2)
        @test isempty(PowerModels.var(pm, 1))
        ll = PowerModels.var(pm, 2, :ll)
        @test JuMP.name(ll[(2, 2, 3)]) == "2_ll[(2, 2, 3)]"
        @test JuMP.start_value(ll[(2, 2, 3)]) == 2.5
        @test length(vfc_reported(pm, 2, :branch, :ll)) == 2
    end

    @testset "storage flag read from network 1 (defect)" begin
        # In network 2, branch 3 is turned into an ordinary line. The function
        # nevertheless looks the `storage` flag up in network 1, so for `nw=2`
        # branch 3 is still skipped.
        pm = vfc_model()
        PowerModels.ref(pm, 2, :branch, 3)["storage"] = false
        eDisGo_OPF.variable_line_loading_max(pm, nw=2)
        ll = PowerModels.var(pm, 2, :ll)
        @test !((3, 3, 4) in vfc_index(ll))
        # Intended: network 2's own branch data decides.
        @test_broken (3, 3, 4) in vfc_index(ll)
    end
end

@testset "variable_gen_power_curt" begin
    @testset "creates pgc only" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt(pm, nw=1)
        # The reactive curtailment `qgc` is commented out in the wrapper.
        @test TestUtils.variable_keys(pm, 1) == [:pgc]
        @test !haskey(PowerModels.var(pm, 1), :qgc)
        @test JuMP.num_variables(pm.model) == 1
        @test length(vfc_reported(pm, 1, :gen_nd, :pgc)) == 1
    end

    @testset "forwards keyword arguments" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt(pm, nw=2, bounded=false, report=false)
        @test isempty(PowerModels.var(pm, 1))
        pgc = PowerModels.var(pm, 2, :pgc)
        @test vfc_unbounded(pgc)
        @test JuMP.name(pgc[1]) == "2_pgc[1]"
        @test isempty(vfc_reported(pm, 2, :gen_nd, :pgc))
    end
end

@testset "variable_gen_power_curt_real" begin
    @testset "index set" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:pgc]
        pgc = PowerModels.var(pm, 1, :pgc)
        # Only the non-dispatchable generators, not `gen` or `gen_slack`.
        @test vfc_index(pgc) == Set([1])
        @test JuMP.name(pgc[1]) == "1_pgc[1]"
    end

    @testset "real grid" begin
        pm = vfc_big_model()
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1)
        pgc = PowerModels.var(pm, 1, :pgc)
        @test vfc_index(pgc) == Set(PowerModels.ids(pm, 1, :gen_nd))
        @test length(pgc) == 27
    end

    @testset "bounded=true" begin
        # At most the whole time-step-specific feed-in `pg` can be curtailed.
        data = TestUtils.small_network(num_steps=2)
        data["time_series"]["gen_nd"]["1"]["pg"] = [0.6, 0.45]
        pm = vfc_model_from(data)
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1, bounded=true)
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=2, bounded=true)
        @test vfc_bounds(PowerModels.var(pm, 1, :pgc)[1]) == (0.0, 0.6)
        @test vfc_bounds(PowerModels.var(pm, 2, :pgc)[1]) == (0.0, 0.45)
    end

    @testset "bounded=false" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1, bounded=false)
        @test vfc_unbounded(PowerModels.var(pm, 1, :pgc))
    end

    @testset "start values" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :pgc)[1]) == 0.0

        pm = vfc_model()
        # The field `set_ac_bf_start_values!` writes for the warm start.
        PowerModels.ref(pm, 1, :gen_nd, 1)["pgc_start"] = 0.25
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :pgc)[1]) == 0.25
    end

    @testset "report" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1, report=true)
        @test vfc_reported(pm, 1, :gen_nd, :pgc) ==
              Dict(1 => PowerModels.var(pm, 1, :pgc)[1])

        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1, report=false)
        @test isempty(vfc_reported(pm, 1, :gen_nd, :pgc))
    end

    @testset "nw keyword" begin
        pm = vfc_model()
        PowerModels.ref(pm, 2, :gen_nd, 1)["pgc_start"] = 0.1
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=2)
        @test isempty(PowerModels.var(pm, 1))
        pgc = PowerModels.var(pm, 2, :pgc)
        @test JuMP.name(pgc[1]) == "2_pgc[1]"
        @test JuMP.start_value(pgc[1]) == 0.1
        @test length(vfc_reported(pm, 2, :gen_nd, :pgc)) == 1
    end

    @testset "no generators" begin
        data = TestUtils.small_network()
        data["gen_nd"] = Dict{String,Any}()
        data["time_series"]["gen_nd"] = Dict{String,Any}()
        pm = vfc_model_from(data)
        eDisGo_OPF.variable_gen_power_curt_real(pm, nw=1)
        @test isempty(PowerModels.var(pm, 1, :pgc))
        @test isempty(vfc_reported(pm, 1, :gen_nd, :pgc))
    end
end

@testset "variable_gen_power_curt_imaginary" begin
    @testset "index set" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_imaginary(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:qgc]
        qgc = PowerModels.var(pm, 1, :qgc)
        @test vfc_index(qgc) == Set([1])
        @test JuMP.name(qgc[1]) == "1_qgc[1]"
    end

    @testset "bounded=true" begin
        # Generators feed in with a negative (inductive) `qg`; curtailment can
        # reduce it down to zero.
        data = TestUtils.small_network(num_steps=2)
        data["time_series"]["gen_nd"]["1"]["qg"] = [-0.29, -0.2]
        pm = vfc_model_from(data)
        eDisGo_OPF.variable_gen_power_curt_imaginary(pm, nw=1, bounded=true)
        eDisGo_OPF.variable_gen_power_curt_imaginary(pm, nw=2, bounded=true)
        @test vfc_bounds(PowerModels.var(pm, 1, :qgc)[1]) == (-0.29, 0.0)
        @test vfc_bounds(PowerModels.var(pm, 2, :qgc)[1]) == (-0.2, 0.0)
    end

    @testset "bounded=false" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_imaginary(pm, nw=1, bounded=false)
        @test vfc_unbounded(PowerModels.var(pm, 1, :qgc))
    end

    @testset "no start value" begin
        # Unlike `pgc`, `qgc` gets no start value at all — not even from a field.
        pm = vfc_model()
        PowerModels.ref(pm, 1, :gen_nd, 1)["qgc_start"] = -0.1
        eDisGo_OPF.variable_gen_power_curt_imaginary(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :qgc)[1]) === nothing
    end

    @testset "report" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_imaginary(pm, nw=1, report=true)
        @test vfc_reported(pm, 1, :gen_nd, :qgc) ==
              Dict(1 => PowerModels.var(pm, 1, :qgc)[1])

        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_imaginary(pm, nw=1, report=false)
        @test isempty(vfc_reported(pm, 1, :gen_nd, :qgc))
    end

    @testset "nw keyword" begin
        pm = vfc_model()
        eDisGo_OPF.variable_gen_power_curt_imaginary(pm, nw=2)
        @test isempty(PowerModels.var(pm, 1))
        @test JuMP.name(PowerModels.var(pm, 2, :qgc)[1]) == "2_qgc[1]"
        @test length(vfc_reported(pm, 2, :gen_nd, :qgc)) == 1
    end
end

@testset "variable_battery_storage" begin
    @testset "creates ps and se" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage(pm, nw=1)
        # Active power and energy; the reactive power `qs` is not created (it is
        # derived from the power factor of the virtual branch instead).
        @test TestUtils.variable_keys(pm, 1) == [:ps, :se]
        @test !haskey(PowerModels.var(pm, 1), :qs)
        @test JuMP.num_variables(pm.model) == 2
        ps = PowerModels.var(pm, 1, :ps)
        se = PowerModels.var(pm, 1, :se)
        @test vfc_index(ps) == vfc_index(se) == Set([1])
        @test vfc_bounds(ps[1]) == (-0.4, 0.4)
        # `se` comes from PowerModels: bounded by [0, energy_rating], start 1.
        @test vfc_bounds(se[1]) == (0.0, 0.4)
        @test JuMP.start_value(se[1]) == 1.0
        @test vfc_reported(pm, 1, :storage, :ps) == Dict(1 => ps[1])
        @test vfc_reported(pm, 1, :storage, :se) == Dict(1 => se[1])
    end

    @testset "forwards keyword arguments" begin
        pm = vfc_model()
        PowerModels.ref(pm, 2, :storage, 1)["se_start"] = 0.2
        eDisGo_OPF.variable_battery_storage(pm, nw=2, bounded=false, report=false)
        @test isempty(PowerModels.var(pm, 1))
        ps = PowerModels.var(pm, 2, :ps)
        se = PowerModels.var(pm, 2, :se)
        @test vfc_unbounded(ps)
        @test vfc_unbounded(se)
        @test JuMP.name(ps[1]) == "2_ps[1]"
        @test JuMP.name(se[1]) == "2_se[1]"
        @test JuMP.start_value(se[1]) == 0.2
        @test isempty(vfc_reported(pm, 2, :storage, :ps))
        @test isempty(vfc_reported(pm, 2, :storage, :se))
    end
end

@testset "variable_battery_storage_power_real" begin
    @testset "index set" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:ps]
        ps = PowerModels.var(pm, 1, :ps)
        @test vfc_index(ps) == Set([1])
        @test JuMP.name(ps[1]) == "1_ps[1]"
    end

    @testset "inactive storage excluded" begin
        # A second, inactive unit is filtered out by `ref_add_core!` and so gets
        # no variable.
        data = TestUtils.small_network()
        s2 = deepcopy(data["storage"]["1"])
        s2["index"] = 2
        s2["status"] = false
        data["storage"]["2"] = s2
        pm = vfc_model_from(data)
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=1)
        @test vfc_index(PowerModels.var(pm, 1, :ps)) == Set([1])
    end

    @testset "bounded=true" begin
        pm = vfc_model()
        PowerModels.ref(pm, 1, :storage, 1)["pmin"] = -0.3
        PowerModels.ref(pm, 1, :storage, 1)["pmax"] = 0.25
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=1, bounded=true)
        @test vfc_bounds(PowerModels.var(pm, 1, :ps)[1]) == (-0.3, 0.25)
    end

    @testset "bounded=false" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=1, bounded=false)
        @test vfc_unbounded(PowerModels.var(pm, 1, :ps))
    end

    @testset "start values" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :ps)[1]) == 0.0

        pm = vfc_model()
        PowerModels.ref(pm, 1, :storage, 1)["ps_start"] = -0.15
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :ps)[1]) == -0.15
    end

    @testset "report" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=1, report=true)
        @test vfc_reported(pm, 1, :storage, :ps) ==
              Dict(1 => PowerModels.var(pm, 1, :ps)[1])

        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=1, report=false)
        @test isempty(vfc_reported(pm, 1, :storage, :ps))
    end

    @testset "nw keyword" begin
        pm = vfc_model()
        PowerModels.ref(pm, 2, :storage, 1)["pmax"] = 0.1
        PowerModels.ref(pm, 2, :storage, 1)["ps_start"] = 0.05
        eDisGo_OPF.variable_battery_storage_power_real(pm, nw=2)
        @test isempty(PowerModels.var(pm, 1))
        ps = PowerModels.var(pm, 2, :ps)
        @test JuMP.name(ps[1]) == "2_ps[1]"
        @test vfc_bounds(ps[1]) == (-0.4, 0.1)
        @test JuMP.start_value(ps[1]) == 0.05
        @test length(vfc_reported(pm, 2, :storage, :ps)) == 1
    end
end

@testset "variable_battery_storage_power_imaginary" begin
    @testset "index set" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_imaginary(pm, nw=1)
        @test TestUtils.variable_keys(pm, 1) == [:qs]
        qs = PowerModels.var(pm, 1, :qs)
        @test vfc_index(qs) == Set([1])
        @test JuMP.name(qs[1]) == "1_qs[1]"
    end

    @testset "bounded=true" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_imaginary(pm, nw=1, bounded=true)
        @test vfc_bounds(PowerModels.var(pm, 1, :qs)[1]) == (-0.19, 0.19)
    end

    @testset "bounded=false" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_imaginary(pm, nw=1, bounded=false)
        @test vfc_unbounded(PowerModels.var(pm, 1, :qs))
    end

    @testset "start values" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_imaginary(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :qs)[1]) == 0.0

        pm = vfc_model()
        PowerModels.ref(pm, 1, :storage, 1)["qs_start"] = 0.07
        # The active-power start value must not be picked up.
        PowerModels.ref(pm, 1, :storage, 1)["ps_start"] = 0.3
        eDisGo_OPF.variable_battery_storage_power_imaginary(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :qs)[1]) == 0.07
    end

    @testset "report" begin
        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_imaginary(pm, nw=1, report=true)
        @test vfc_reported(pm, 1, :storage, :qs) ==
              Dict(1 => PowerModels.var(pm, 1, :qs)[1])

        pm = vfc_model()
        eDisGo_OPF.variable_battery_storage_power_imaginary(pm, nw=1, report=false)
        @test isempty(vfc_reported(pm, 1, :storage, :qs))
    end

    @testset "nw keyword" begin
        pm = vfc_model()
        PowerModels.ref(pm, 2, :storage, 1)["qmin"] = -0.05
        eDisGo_OPF.variable_battery_storage_power_imaginary(pm, nw=2)
        @test isempty(PowerModels.var(pm, 1))
        qs = PowerModels.var(pm, 2, :qs)
        @test JuMP.name(qs[1]) == "2_qs[1]"
        @test vfc_bounds(qs[1]) == (-0.05, 0.19)
        @test length(vfc_reported(pm, 2, :storage, :qs)) == 1
    end
end

@testset "variable_dsm_storage_power" begin
    @testset "creates pdsm and dsme" begin
        pm = vfc_model()
        eDisGo_OPF.variable_dsm_storage_power(pm, nw=1)
        # The reactive DSM power `qdsm` is commented out in the wrapper.
        @test TestUtils.variable_keys(pm, 1) == [:dsme, :pdsm]
        @test !haskey(PowerModels.var(pm, 1), :qdsm)
        @test JuMP.num_variables(pm.model) == 2
        @test length(vfc_reported(pm, 1, :dsm, :pdsm)) == 1
        @test length(vfc_reported(pm, 1, :dsm, :dsme)) == 1
    end

    @testset "forwards keyword arguments" begin
        pm = vfc_model()
        eDisGo_OPF.variable_dsm_storage_power(pm, nw=2, bounded=false, report=false)
        @test isempty(PowerModels.var(pm, 1))
        pdsm = PowerModels.var(pm, 2, :pdsm)
        dsme = PowerModels.var(pm, 2, :dsme)
        @test vfc_unbounded(pdsm)
        @test vfc_unbounded(dsme)
        @test JuMP.name(pdsm[1]) == "2_pdsm[1]"
        @test JuMP.name(dsme[1]) == "2_dsme[1]"
        @test isempty(vfc_reported(pm, 2, :dsm, :pdsm))
        @test isempty(vfc_reported(pm, 2, :dsm, :dsme))
    end

    @testset "no DSM units" begin
        data = TestUtils.small_network()
        data["dsm"] = Dict{String,Any}()
        data["time_series"]["dsm"] = Dict{String,Any}()
        pm = vfc_model_from(data)
        eDisGo_OPF.variable_dsm_storage_power(pm, nw=1)
        @test isempty(PowerModels.var(pm, 1, :pdsm))
        @test isempty(PowerModels.var(pm, 1, :dsme))
        @test JuMP.num_variables(pm.model) == 0
    end
end
