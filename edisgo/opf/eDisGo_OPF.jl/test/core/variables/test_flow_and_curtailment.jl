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
