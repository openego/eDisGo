# Tests for the slack variables of `src/core/variables.jl` — grid restrictions,
# heat pump / heat storage, slack generator and overlying-grid requirements, from
# `variable_slack_grid_restrictions` to the end of the file.
#
# The block consists of four wrappers and ten functions that each create one
# family of JuMP variables:
#
#   * `variable_slack_grid_restrictions` → `phps`, `pds`, `pgens`, `pcps`
#     (only built for `opf_version` 2 and 4, i.e. with grid restrictions),
#   * `variable_slack_heat_pump_storage` → `phss`, `phps2` (every version),
#   * `variable_slack_gen`               → `pgs`, `qgs` (every version),
#   * `variable_slack_HV_requirements`   → `phvs` (`opf_version` 3 and 4); the
#     reactive counterpart `qhvs` is commented out of the wrapper.
#
# Every function is tested on a *bare* model — `TestUtils.small_network`
# instantiated with a build method that creates nothing — so each variable,
# bound and solution entry a test observes was created by the call under test.
# The fixture gets time series that differ between the two time steps, which
# makes per-time-step bounds (`pd`, `pg`, `pd/cop`) distinguishable, and extra
# components where an index set has to exclude (or, notably, does *not* exclude)
# something. The wrappers are additionally checked against the full
# `build_mn_opf_bf_flex` model of every `opf_version`.
#
# Known production bug (see also test/FUNCTION_COVERAGE.md):
# `variable_slack_gen_real`/`_imaginary` ignore the `pgs_start`/`qgs_start`
# fields that `set_ac_bf_start_values!` writes for the warm start, so the slack
# generator is never warm-started. The tests pin the actual behaviour and keep
# the intended one as `@test_broken`.

include(joinpath(@__DIR__, "..", "..", "testsetup.jl"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"""
    vsl_network(; overlying_grid=false)

`TestUtils.small_network` with two time steps whose time series differ, so that
bounds taken from the per-time-step data can be told apart:

| component     | field | step 1 | step 2 |
| ------------- | ----- | ------ | ------ |
| load 1        | `pd`  | 0.2    | 0.25   |
| gen 1         | `pg`  | 0.3    | 0.35   |
| heat pump 1   | `pd`  | 1.0    | 0.9    |
| heat pump 1   | `cop` | 4.0    | 3.0    |
"""
function vsl_network(; overlying_grid::Bool=false)
    data = TestUtils.small_network(; num_steps=2, overlying_grid=overlying_grid)
    ts = data["time_series"]
    ts["load"]["1"]["pd"] = [0.2, 0.25]
    ts["gen"]["1"]["pg"] = [0.3, 0.35]
    ts["heatpumps"]["1"]["pd"] = [1.0, 0.9]
    ts["heatpumps"]["1"]["cop"] = [4.0, 3.0]
    return data
end

"""
    vsl_add!(data, comp, id; fields...)

Add component `id` of type `comp` to `data` as a copy of component `"1"` with
`"index" => id` and the given `fields` overridden. The copy has no time series,
so it keeps its static values in every time step.
"""
function vsl_add!(data::Dict{String,<:Any}, comp::String, id::Int; fields...)
    entry = deepcopy(data[comp]["1"])
    entry["index"] = id
    for (k, v) in fields
        entry[string(k)] = v
    end
    data[comp][string(id)] = entry
    return data
end

"""
Instantiate the multi-network form of `data` with a build method that creates
nothing, so `PowerModels.var` and the solution dict start out empty.
"""
vsl_model(data=vsl_network(); model_type::Type=SOCBFPowerModelEdisgo) =
    eDisGo_OPF.instantiate_model(TestUtils.multinetwork(data), model_type,
                                 _pm -> nothing)

"Sorted index set of the JuMP container `v`."
vsl_ids(v) = sort(collect(axes(v, 1)))

"Lower bound of `x`, or `nothing` if it has none."
vsl_lb(x) = JuMP.has_lower_bound(x) ? JuMP.lower_bound(x) : nothing

"Upper bound of `x`, or `nothing` if it has none."
vsl_ub(x) = JuMP.has_upper_bound(x) ? JuMP.upper_bound(x) : nothing

"""
    vsl_reported(pm, nw, comp, field)

`id => value` of every entry of component type `comp` in the solution dict of
network `nw` that carries `field`. Unlike `PowerModels.sol(pm, nw, comp, id)`
this does not create missing entries, so it can be used to assert that nothing
was reported.
"""
function vsl_reported(pm, nw::Int, comp::Symbol, field::Symbol)
    s = PowerModels.sol(pm, nw)
    haskey(s, comp) || return Dict{Int,Any}()
    return Dict{Int,Any}(i => e[field] for (i, e) in s[comp] if haskey(e, field))
end

"""
    vsl_check_reporting(f, comp, field)

Call `f(pm; nw=1, report=...)` on fresh bare models and check that `report=true`
registers exactly the created variables under `comp`/`field` of network 1 and
`report=false` registers nothing (but still creates the variables).
"""
function vsl_check_reporting(f, comp::Symbol, field::Symbol)
    pm = vsl_model()
    f(pm; nw=1, report=true)
    v = PowerModels.var(pm, 1, field)
    reported = vsl_reported(pm, 1, comp, field)
    @test sort(collect(keys(reported))) == vsl_ids(v)
    @test all(reported[i] === v[i] for i in vsl_ids(v))
    @test isempty(vsl_reported(pm, 2, comp, field))

    pm = vsl_model()
    f(pm; nw=1, report=false)
    @test haskey(PowerModels.var(pm, 1), field)
    @test isempty(vsl_reported(pm, 1, comp, field))
end

"""
    vsl_check_nw(f, field)

Call `f(pm; nw=2)` on a fresh bare model and check that the variables are created
in network 2 only, with the network number in their base name.
"""
function vsl_check_nw(f, field::Symbol)
    pm = vsl_model()
    f(pm; nw=2)
    @test !haskey(PowerModels.var(pm, 1), field)
    v = PowerModels.var(pm, 2, field)
    @test all(startswith(JuMP.name(v[i]), "2_$(field)[") for i in vsl_ids(v))
end

"Run `f` with `stdout` discarded (opf_version 3 prints debug lines, see defect 7)."
vsl_quiet(f) = redirect_stdout(f, devnull)

# Not `const`: the file may be included more than once per session.
vsl_built_models = Dict{Int,Any}()

"""
The full `build_mn_opf_bf_flex` model of `TestUtils.small_network` for
`opf_version` `v` (3 and 4 with overlying-grid requirements), built once and
shared by the wrapper tests, which only read it.
"""
vsl_built(v::Int) = get!(vsl_built_models, v) do
    vsl_quiet(() -> TestUtils.build_small_model(; opf_version=v,
                                                overlying_grid=v in (3, 4)))
end

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@testset "variable_slack_grid_restrictions" begin
    grid_keys = [:pcps, :pds, :pgens, :phps]

    @testset "creates the four grid-restriction slacks" begin
        for model_type in (SOCBFPowerModelEdisgo, NCBFPowerModelEdisgo)
            pm = vsl_model(; model_type=model_type)
            eDisGo_OPF.variable_slack_grid_restrictions(pm; nw=1)
            @test TestUtils.variable_keys(pm, 1) == grid_keys
            @test isempty(PowerModels.var(pm, 2))
            @test JuMP.num_variables(pm.model) == 4   # one component of each type
        end
    end

    @testset "forwards nw, bounded and report" begin
        pm = vsl_model()
        eDisGo_OPF.variable_slack_grid_restrictions(pm; nw=2, bounded=false,
                                                    report=false)
        @test isempty(PowerModels.var(pm, 1))
        @test TestUtils.variable_keys(pm, 2) == grid_keys
        # bounded=false reaches the three bounded members ...
        for key in (:phps, :pds, :pgens)
            @test vsl_ub(PowerModels.var(pm, 2, key)[1]) === nothing
        end
        # ... and report=false all four.
        @test isempty(PowerModels.sol(pm, 2))

        pm = vsl_model()
        eDisGo_OPF.variable_slack_grid_restrictions(pm; nw=1)
        for key in (:phps, :pds, :pgens)
            @test vsl_ub(PowerModels.var(pm, 1, key)[1]) !== nothing
        end
        @test Set(keys(PowerModels.sol(pm, 1))) ==
              Set([:heatpumps, :load, :gen, :electromobility])
    end

    @testset "used by build_mn_opf_bf_flex for opf_version 2 and 4 only" begin
        for v in 1:4
            pm = vsl_built(v)
            for n in PowerModels.nw_ids(pm), key in grid_keys
                @test haskey(PowerModels.var(pm, n), key) == (v in (2, 4))
            end
        end
    end

    @testset "real grid: one slack per component in every time step" begin
        pm = TestUtils.build_model(; opf_version=2)
        for n in PowerModels.nw_ids(pm)
            for (key, comp) in ((:phps, :heatpumps), (:pds, :load),
                                (:pgens, :gen), (:pcps, :electromobility))
                @test vsl_ids(PowerModels.var(pm, n, key)) ==
                      sort(collect(PowerModels.ids(pm, n, comp)))
            end
        end
    end
end

@testset "variable_slack_heat_pump_storage" begin
    @testset "creates phss and phps2" begin
        for model_type in (SOCBFPowerModelEdisgo, NCBFPowerModelEdisgo)
            pm = vsl_model(; model_type=model_type)
            eDisGo_OPF.variable_slack_heat_pump_storage(pm; nw=1)
            @test TestUtils.variable_keys(pm, 1) == [:phps2, :phss]
            @test isempty(PowerModels.var(pm, 2))
            @test JuMP.num_variables(pm.model) == 2
        end
    end

    @testset "forwards nw and report, accepts bounded" begin
        pm = vsl_model()
        eDisGo_OPF.variable_slack_heat_pump_storage(pm; nw=2, bounded=false,
                                                    report=false)
        @test isempty(PowerModels.var(pm, 1))
        @test TestUtils.variable_keys(pm, 2) == [:phps2, :phss]
        @test isempty(PowerModels.sol(pm, 2))

        pm = vsl_model()
        eDisGo_OPF.variable_slack_heat_pump_storage(pm; nw=1)
        @test Set(keys(PowerModels.sol(pm, 1))) == Set([:heat_storage, :heatpumps])
    end

    @testset "used by build_mn_opf_bf_flex for every opf_version" begin
        for v in 1:4
            pm = vsl_built(v)
            for n in PowerModels.nw_ids(pm)
                @test haskey(PowerModels.var(pm, n), :phss)
                @test haskey(PowerModels.var(pm, n), :phps2)
            end
        end
    end
end

@testset "variable_hs_slack" begin
    # A second heat storage with an id no heat pump has, and switched off:
    # `phss` follows the heat-storage ids, and `ref_add_core!` does not filter
    # heat storages by status.
    data = vsl_network()
    vsl_add!(data, "heat_storage", 7; status=false)

    @testset "key, index set and bounds" begin
        for bounded in (true, false)
            pm = vsl_model(data)
            eDisGo_OPF.variable_hs_slack(pm; nw=1, bounded=bounded)
            phss = PowerModels.var(pm, 1, :phss)
            @test vsl_ids(phss) == [1, 7]
            @test vsl_ids(phss) == sort(collect(PowerModels.ids(pm, 1, :heat_storage)))
            @test vsl_ids(phss) != sort(collect(PowerModels.ids(pm, 1, :heatpumps)))
            for i in (1, 7)
                # Non-negative and otherwise free; `bounded` has no effect.
                @test vsl_lb(phss[i]) == 0.0
                @test vsl_ub(phss[i]) === nothing
                @test JuMP.start_value(phss[i]) === nothing
                @test JuMP.name(phss[i]) == "1_phss[$i]"
            end
            @test JuMP.num_variables(pm.model) == 2
        end
    end

    @testset "nw" begin
        vsl_check_nw(eDisGo_OPF.variable_hs_slack, :phss)
    end

    @testset "report" begin
        vsl_check_reporting(eDisGo_OPF.variable_hs_slack, :heat_storage, :phss)
    end
end

@testset "variable_hp2_slack" begin
    # A second heat pump with its own id on another bus.
    data = vsl_network()
    vsl_add!(data, "heatpumps", 5; hp_bus=2)

    @testset "key, index set and bounds" begin
        for bounded in (true, false)
            pm = vsl_model(data)
            eDisGo_OPF.variable_hp2_slack(pm; nw=1, bounded=bounded)
            phps2 = PowerModels.var(pm, 1, :phps2)
            @test vsl_ids(phps2) == [1, 5]
            @test vsl_ids(phps2) == sort(collect(PowerModels.ids(pm, 1, :heatpumps)))
            for i in (1, 5)
                # Non-negative and otherwise free; `bounded` has no effect.
                @test vsl_lb(phps2[i]) == 0.0
                @test vsl_ub(phps2[i]) === nothing
                @test JuMP.start_value(phps2[i]) === nothing
                @test JuMP.name(phps2[i]) == "1_phps2[$i]"
            end
            @test JuMP.num_variables(pm.model) == 2
        end
    end

    @testset "nw" begin
        vsl_check_nw(eDisGo_OPF.variable_hp2_slack, :phps2)
    end

    @testset "report" begin
        vsl_check_reporting(eDisGo_OPF.variable_hp2_slack, :heatpumps, :phps2)
    end
end

@testset "variable_hp_slack" begin
    # Heat pump 5 has a negative demand (no time series, so in both steps).
    data = vsl_network()
    vsl_add!(data, "heatpumps", 5; hp_bus=2, pd=-0.5, cop=2.0)

    @testset "key and index set" begin
        pm = vsl_model(data)
        eDisGo_OPF.variable_hp_slack(pm; nw=1)
        phps = PowerModels.var(pm, 1, :phps)
        @test vsl_ids(phps) == [1, 5]
        @test vsl_ids(phps) == sort(collect(PowerModels.ids(pm, 1, :heatpumps)))
        @test JuMP.num_variables(pm.model) == 2
        for i in (1, 5)
            @test JuMP.start_value(phps[i]) === nothing
            @test JuMP.name(phps[i]) == "1_phps[$i]"
        end
    end

    @testset "bounded=true: 0 ≤ phps ≤ max(pd/cop, 0) of the time step" begin
        pm = vsl_model(data)
        for n in (1, 2)
            eDisGo_OPF.variable_hp_slack(pm; nw=n)
        end
        # Heat pump 1: pd/cop = 1.0/4.0 in step 1 and 0.9/3.0 in step 2.
        @test vsl_lb(PowerModels.var(pm, 1, :phps)[1]) == 0.0
        @test vsl_ub(PowerModels.var(pm, 1, :phps)[1]) ≈ 0.25
        @test vsl_lb(PowerModels.var(pm, 2, :phps)[1]) == 0.0
        @test vsl_ub(PowerModels.var(pm, 2, :phps)[1]) ≈ 0.3
        # Heat pump 5: pd/cop = -0.25 is clamped to 0, the slack is fixed at 0.
        for n in (1, 2)
            @test vsl_lb(PowerModels.var(pm, n, :phps)[5]) == 0.0
            @test vsl_ub(PowerModels.var(pm, n, :phps)[5]) == 0.0
        end
    end

    @testset "bounded=false: only the lower bound" begin
        pm = vsl_model(data)
        eDisGo_OPF.variable_hp_slack(pm; nw=1, bounded=false)
        for i in (1, 5)
            @test vsl_lb(PowerModels.var(pm, 1, :phps)[i]) == 0.0
            @test vsl_ub(PowerModels.var(pm, 1, :phps)[i]) === nothing
        end
    end

    @testset "nw" begin
        vsl_check_nw(eDisGo_OPF.variable_hp_slack, :phps)
    end

    @testset "report" begin
        vsl_check_reporting(eDisGo_OPF.variable_hp_slack, :heatpumps, :phps)
    end
end

@testset "variable_load_slack" begin
    # Load 2 is switched off and must be filtered out, load 3 is active with a
    # static demand.
    data = vsl_network()
    vsl_add!(data, "load", 2; status=0)
    vsl_add!(data, "load", 3; load_bus=2, pd=0.15)

    @testset "key and index set" begin
        pm = vsl_model(data)
        eDisGo_OPF.variable_load_slack(pm; nw=1)
        pds = PowerModels.var(pm, 1, :pds)
        @test vsl_ids(pds) == [1, 3]
        @test !(2 in vsl_ids(pds))
        @test vsl_ids(pds) == sort(collect(PowerModels.ids(pm, 1, :load)))
        @test JuMP.num_variables(pm.model) == 2
        for i in (1, 3)
            @test JuMP.start_value(pds[i]) === nothing
            @test JuMP.name(pds[i]) == "1_pds[$i]"
        end
    end

    @testset "bounded=true: 0 ≤ pds ≤ pd of the time step" begin
        pm = vsl_model(data)
        for n in (1, 2)
            eDisGo_OPF.variable_load_slack(pm; nw=n)
        end
        @test vsl_lb(PowerModels.var(pm, 1, :pds)[1]) == 0.0
        @test vsl_ub(PowerModels.var(pm, 1, :pds)[1]) == 0.2
        @test vsl_lb(PowerModels.var(pm, 2, :pds)[1]) == 0.0
        @test vsl_ub(PowerModels.var(pm, 2, :pds)[1]) == 0.25
        for n in (1, 2)
            @test vsl_ub(PowerModels.var(pm, n, :pds)[3]) == 0.15
        end
    end

    @testset "bounded=false: only the lower bound" begin
        pm = vsl_model(data)
        eDisGo_OPF.variable_load_slack(pm; nw=1, bounded=false)
        for i in (1, 3)
            @test vsl_lb(PowerModels.var(pm, 1, :pds)[i]) == 0.0
            @test vsl_ub(PowerModels.var(pm, 1, :pds)[i]) === nothing
        end
    end

    @testset "nw" begin
        vsl_check_nw(eDisGo_OPF.variable_load_slack, :pds)
    end

    @testset "report" begin
        vsl_check_reporting(eDisGo_OPF.variable_load_slack, :load, :pds)
    end
end

@testset "variable_gen_slack" begin
    # Gen 2 is switched off and must be filtered out, gen 4 is active with a
    # static set point. The non-dispatchable and slack generators (ids 1) are
    # not part of the index set.
    data = vsl_network()
    vsl_add!(data, "gen", 2; gen_status=0)
    vsl_add!(data, "gen", 4; gen_bus=3, pg=0.1)
    vsl_add!(data, "gen_nd", 6)

    @testset "key and index set" begin
        pm = vsl_model(data)
        eDisGo_OPF.variable_gen_slack(pm; nw=1)
        pgens = PowerModels.var(pm, 1, :pgens)
        @test vsl_ids(pgens) == [1, 4]
        @test !(2 in vsl_ids(pgens))
        @test !(6 in vsl_ids(pgens))       # gen_nd ids are not used
        @test vsl_ids(pgens) == sort(collect(PowerModels.ids(pm, 1, :gen)))
        @test JuMP.num_variables(pm.model) == 2
        for i in (1, 4)
            @test JuMP.start_value(pgens[i]) === nothing
            @test JuMP.name(pgens[i]) == "1_pgens[$i]"
        end
    end

    @testset "bounded=true: 0 ≤ pgens ≤ pg of the time step" begin
        pm = vsl_model(data)
        for n in (1, 2)
            eDisGo_OPF.variable_gen_slack(pm; nw=n)
        end
        @test vsl_lb(PowerModels.var(pm, 1, :pgens)[1]) == 0.0
        @test vsl_ub(PowerModels.var(pm, 1, :pgens)[1]) == 0.3
        @test vsl_lb(PowerModels.var(pm, 2, :pgens)[1]) == 0.0
        @test vsl_ub(PowerModels.var(pm, 2, :pgens)[1]) == 0.35
        for n in (1, 2)
            @test vsl_ub(PowerModels.var(pm, n, :pgens)[4]) == 0.1
        end
    end

    @testset "bounded=false: only the lower bound" begin
        pm = vsl_model(data)
        eDisGo_OPF.variable_gen_slack(pm; nw=1, bounded=false)
        for i in (1, 4)
            @test vsl_lb(PowerModels.var(pm, 1, :pgens)[i]) == 0.0
            @test vsl_ub(PowerModels.var(pm, 1, :pgens)[i]) === nothing
        end
    end

    @testset "nw" begin
        vsl_check_nw(eDisGo_OPF.variable_gen_slack, :pgens)
    end

    @testset "report" begin
        vsl_check_reporting(eDisGo_OPF.variable_gen_slack, :gen, :pgens)
    end
end

@testset "variable_ev_slack" begin
    data = vsl_network()
    vsl_add!(data, "electromobility", 3; cp_bus=2, p_max=0.7)

    @testset "key, index set and bounds" begin
        for bounded in (true, false)
            pm = vsl_model(data)
            eDisGo_OPF.variable_ev_slack(pm; nw=1, bounded=bounded)
            pcps = PowerModels.var(pm, 1, :pcps)
            @test vsl_ids(pcps) == [1, 3]
            @test vsl_ids(pcps) ==
                  sort(collect(PowerModels.ids(pm, 1, :electromobility)))
            for i in (1, 3)
                # Non-negative and otherwise free — unlike the other grid
                # restriction slacks not even `bounded=true` caps it (e.g. at
                # `p_max`).
                @test vsl_lb(pcps[i]) == 0.0
                @test vsl_ub(pcps[i]) === nothing
                @test JuMP.start_value(pcps[i]) === nothing
                @test JuMP.name(pcps[i]) == "1_pcps[$i]"
            end
            @test JuMP.num_variables(pm.model) == 2
        end
    end

    @testset "nw" begin
        vsl_check_nw(eDisGo_OPF.variable_ev_slack, :pcps)
    end

    @testset "report" begin
        vsl_check_reporting(eDisGo_OPF.variable_ev_slack, :electromobility, :pcps)
    end
end

@testset "variable_slack_gen" begin
    @testset "creates pgs and qgs" begin
        for model_type in (SOCBFPowerModelEdisgo, NCBFPowerModelEdisgo)
            pm = vsl_model(; model_type=model_type)
            eDisGo_OPF.variable_slack_gen(pm; nw=1)
            @test TestUtils.variable_keys(pm, 1) == [:pgs, :qgs]
            @test isempty(PowerModels.var(pm, 2))
            @test JuMP.num_variables(pm.model) == 2
        end
    end

    @testset "forwards nw and report" begin
        pm = vsl_model()
        eDisGo_OPF.variable_slack_gen(pm; nw=2, report=false)
        @test isempty(PowerModels.var(pm, 1))
        @test TestUtils.variable_keys(pm, 2) == [:pgs, :qgs]
        @test isempty(PowerModels.sol(pm, 2))

        pm = vsl_model()
        eDisGo_OPF.variable_slack_gen(pm; nw=1)
        @test Set(keys(PowerModels.sol(pm, 1, :gen_slack, 1))) == Set([:pgs, :qgs])
    end

    @testset "has no bounded keyword" begin
        # Unlike the other wrappers, the two members do not take `bounded`, so
        # passing it through the wrapper fails. `build_mn_opf_bf_flex` never does.
        pm = vsl_model()
        @test_throws MethodError eDisGo_OPF.variable_slack_gen(pm; nw=1,
                                                                bounded=false)
    end

    @testset "used by build_mn_opf_bf_flex for every opf_version" begin
        for v in 1:4
            pm = vsl_built(v)
            for n in PowerModels.nw_ids(pm)
                @test haskey(PowerModels.var(pm, n), :pgs)
                @test haskey(PowerModels.var(pm, n), :qgs)
            end
        end
    end
end

"""
Shared body of the `variable_slack_gen_real` / `_imaginary` tests: `f` creates
the variable `field` with start field `start_field` over the slack generators.
"""
function vsl_check_slack_gen(f, field::Symbol, start_field::String)
    # A second slack generator on another bus, switched off: `ref_add_core!`
    # does not filter `gen_slack` by `gen_status`, so it gets a variable too.
    data = vsl_network()
    vsl_add!(data, "gen_slack", 2; gen_bus=2, gen_status=0)

    @testset "key, index set and bounds" begin
        pm = vsl_model(data)
        f(pm; nw=1)
        v = PowerModels.var(pm, 1, field)
        @test vsl_ids(v) == [1, 2]
        @test vsl_ids(v) == sort(collect(PowerModels.ids(pm, 1, :gen_slack)))
        @test JuMP.num_variables(pm.model) == 2
        for i in (1, 2)
            # Completely free, in both directions.
            @test vsl_lb(v[i]) === nothing
            @test vsl_ub(v[i]) === nothing
            @test JuMP.name(v[i]) == "1_$(field)[$i]"
        end
    end

    @testset "start value" begin
        # `set_ac_bf_start_values!` writes `pgs_start`/`qgs_start` for the warm
        # start, but the variable is created without `start = ...`.
        data_s = vsl_network()
        data_s["gen_slack"]["1"][start_field] = 0.42
        pm = vsl_model(data_s)
        f(pm; nw=1)
        x = PowerModels.var(pm, 1, field)[1]
        @test PowerModels.ref(pm, 1, :gen_slack, 1)[start_field] == 0.42
        @test JuMP.start_value(x) === nothing          # actual: ignored
        @test_broken JuMP.start_value(x) == 0.42       # intended: warm start
    end

    @testset "has no bounded keyword" begin
        pm = vsl_model()
        @test_throws MethodError f(pm; nw=1, bounded=true)
    end

    @testset "nw" begin
        vsl_check_nw(f, field)
    end

    @testset "report" begin
        vsl_check_reporting(f, :gen_slack, field)
    end
end

@testset "variable_slack_gen_real" begin
    vsl_check_slack_gen(eDisGo_OPF.variable_slack_gen_real, :pgs, "pgs_start")

    @testset "pgs_start as written by set_ac_bf_start_values!" begin
        # End to end: the field the warm-start helper writes is the one that
        # would have to be read.
        net = vsl_network()
        for (comp, fields) in (("gen_nd", ("pgc",)), ("gen_slack", ("pgs", "qgs")),
                               ("dsm", ("pdsm", "dsme")), ("storage", ("ps", "se")),
                               ("electromobility", ("pcp",)), ("heatpumps", ("php",)),
                               ("heat_storage", ("phs", "hse")))
            for (_, c) in net[comp], fld in fields
                c[fld] = 0.0
            end
        end
        net["gen_slack"]["1"]["pgs"] = -0.7
        eDisGo_OPF.set_ac_bf_start_values!(net)
        @test net["gen_slack"]["1"]["pgs_start"] == -0.7
        pm = vsl_model(net)
        eDisGo_OPF.variable_slack_gen_real(pm; nw=1)
        @test_broken JuMP.start_value(PowerModels.var(pm, 1, :pgs)[1]) == -0.7
    end
end

@testset "variable_slack_gen_imaginary" begin
    vsl_check_slack_gen(eDisGo_OPF.variable_slack_gen_imaginary, :qgs, "qgs_start")
end

@testset "variable_slack_HV_requirements" begin
    @testset "creates phvs only" begin
        pm = vsl_model(vsl_network(; overlying_grid=true))
        eDisGo_OPF.variable_slack_HV_requirements(pm; nw=1)
        # The reactive slack `qhvs` is commented out of the wrapper.
        @test TestUtils.variable_keys(pm, 1) == [:phvs]
        @test isempty(PowerModels.var(pm, 2))
        @test JuMP.num_variables(pm.model) == 5   # five HV requirements
    end

    @testset "forwards nw, bounded and report" begin
        pm = vsl_model(vsl_network(; overlying_grid=true))
        eDisGo_OPF.variable_slack_HV_requirements(pm; nw=2, bounded=false,
                                                  report=false)
        @test isempty(PowerModels.var(pm, 1))
        @test TestUtils.variable_keys(pm, 2) == [:phvs]
        @test isempty(PowerModels.sol(pm, 2))

        pm = vsl_model(vsl_network(; overlying_grid=true))
        eDisGo_OPF.variable_slack_HV_requirements(pm; nw=1)
        @test collect(keys(PowerModels.sol(pm, 1))) == [:HV_requirements]
    end

    @testset "used by build_mn_opf_bf_flex for opf_version 3 and 4 only" begin
        for v in 1:4
            pm = vsl_built(v)
            for n in PowerModels.nw_ids(pm)
                @test haskey(PowerModels.var(pm, n), :phvs) == (v in (3, 4))
                @test !haskey(PowerModels.var(pm, n), :qhvs)
            end
        end
    end

    @testset "real grid: one slack per HV requirement in every time step" begin
        pm = TestUtils.build_model(; overlying_grid=true, opf_version=4)
        for n in PowerModels.nw_ids(pm)
            @test !isempty(PowerModels.ids(pm, n, :HV_requirements))
            @test vsl_ids(PowerModels.var(pm, n, :phvs)) ==
                  sort(collect(PowerModels.ids(pm, n, :HV_requirements)))
        end
    end
end

"""
Shared body of the `variable_slack_HV_requirements_real` / `_imaginary` tests: `f`
creates the variable `field` over the overlying-grid requirements.
"""
function vsl_check_hv_slack(f, field::Symbol)
    @testset "key, index set and bounds" begin
        for bounded in (true, false)
            pm = vsl_model(vsl_network(; overlying_grid=true))
            f(pm; nw=1, bounded=bounded)
            v = PowerModels.var(pm, 1, field)
            @test vsl_ids(v) == [1, 2, 3, 4, 5]
            @test vsl_ids(v) ==
                  sort(collect(PowerModels.ids(pm, 1, :HV_requirements)))
            @test JuMP.num_variables(pm.model) == 5
            for i in 1:5
                # Free in both directions (the bounds are commented out), and
                # `bounded` has no effect.
                @test vsl_lb(v[i]) === nothing
                @test vsl_ub(v[i]) === nothing
                @test JuMP.start_value(v[i]) === nothing
                @test JuMP.name(v[i]) == "1_$(field)[$i]"
            end
        end
    end

    @testset "no HV requirements" begin
        # Without an overlying grid the index set is empty.
        pm = vsl_model()
        f(pm; nw=1)
        @test isempty(PowerModels.var(pm, 1, field))
        @test JuMP.num_variables(pm.model) == 0
    end

    @testset "nw" begin
        pm = vsl_model(vsl_network(; overlying_grid=true))
        f(pm; nw=2)
        @test !haskey(PowerModels.var(pm, 1), field)
        @test JuMP.name(PowerModels.var(pm, 2, field)[1]) == "2_$(field)[1]"
    end

    @testset "report" begin
        pm = vsl_model(vsl_network(; overlying_grid=true))
        f(pm; nw=1, report=true)
        v = PowerModels.var(pm, 1, field)
        reported = vsl_reported(pm, 1, :HV_requirements, field)
        @test sort(collect(keys(reported))) == [1, 2, 3, 4, 5]
        @test all(reported[i] === v[i] for i in 1:5)

        pm = vsl_model(vsl_network(; overlying_grid=true))
        f(pm; nw=1, report=false)
        @test haskey(PowerModels.var(pm, 1), field)
        @test isempty(vsl_reported(pm, 1, :HV_requirements, field))
    end
end

@testset "variable_slack_HV_requirements_real" begin
    vsl_check_hv_slack(eDisGo_OPF.variable_slack_HV_requirements_real, :phvs)
end

@testset "variable_slack_HV_requirements_imaginary" begin
    vsl_check_hv_slack(eDisGo_OPF.variable_slack_HV_requirements_imaginary, :qhvs)

    @testset "not created by build_mn_opf_bf_flex" begin
        for v in 1:4
            pm = vsl_built(v)
            @test all(!haskey(PowerModels.var(pm, n), :qhvs)
                      for n in PowerModels.nw_ids(pm))
        end
    end
end
