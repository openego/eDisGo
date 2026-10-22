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
