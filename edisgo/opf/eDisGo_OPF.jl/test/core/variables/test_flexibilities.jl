# Tests for the flexibility variables of `src/core/variables.jl` — heat storage,
# heat pumps and charging points, from `variable_heat_storage` through
# `variable_cp_energy`.
#
# The ten functions come in two shapes:
#
#   * three wrappers (`variable_heat_storage`, `variable_heat_pump_power`,
#     `variable_cp_power`) that only forward their keyword arguments to the
#     variable functions of one component type — note that the reactive-power
#     calls `variable_heat_pump_power_imaginary`/`variable_cp_power_imaginary`
#     are commented out in them, so `qhp`/`qcp` never exist in the OPF;
#   * seven functions that each create one JuMP variable per component of one
#     type (`heat_storage`, `heatpumps` or `electromobility`), optionally bound
#     it by fields of the component and register it in the solution.
#
# Every function is called on a model whose build method adds nothing
# (`vfx_bare_model`), so each variable, bound and solution entry observed
# afterwards was created by the function under test. The hand-written
# `TestUtils.small_network` is used for the unit tests, because its single
# component of every type can be manipulated in isolation; the real eDisGo test
# grid (`TestUtils.pm_network`, 3 heat pumps with heat storage, 1 charging point,
# 4 time steps with varying charging-point time series) confirms the index sets
# and the per-time-step bounds on representative data.
#
# None of the eDisGo flexibility types is filtered by `ref_add_core!`. For heat
# storage this means that the `"status"` field carried by every unit is ignored:
# an inactive unit still gets its variables. The tests pin that behaviour and
# keep the intended exclusion as `@test_broken` (see test/FUNKTIONSABDECKUNG.md).

include(joinpath(@__DIR__, "..", "..", "testsetup.jl"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"""
    vfx_bare_model(data=TestUtils.small_network(); model_type=NCBFPowerModelEdisgo)

Instantiate a model on `data` (single-network form, expanded with
`TestUtils.multinetwork`) whose build method adds nothing, so `pm.model` holds no
variable and `pm.sol` no entry before the function under test runs.
"""
function vfx_bare_model(data::Dict{String,<:Any}=TestUtils.small_network();
                        model_type::Type=NCBFPowerModelEdisgo)
    return eDisGo_OPF.instantiate_model(TestUtils.multinetwork(data), model_type,
                                        _pm -> nothing)
end

"Index set of the variable container `key` in network `nw` of `pm`."
vfx_indices(pm, nw::Int, key::Symbol) = Set(only(axes(PowerModels.var(pm, nw, key))))

"`(lower, upper)` bound of `v`, with `nothing` standing for a missing bound."
vfx_bounds(v::JuMP.VariableRef) =
    (JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : nothing,
     JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : nothing)

"""
    vfx_reported(pm, nw, comp)

The solution entries registered for component type `comp` in network `nw`, or
`nothing` if there are none. Reads `pm.sol` without side effects —
`PowerModels.sol(pm, nw, comp)` would `get!` an empty entry and thereby hide a
missing report.
"""
vfx_reported(pm, nw::Int, comp::Symbol) = get(PowerModels.sol(pm, nw), comp, nothing)

"Run `f()` and return the number of JuMP variables it added to `pm.model`."
function vfx_added_variables(f, pm)
    before = JuMP.num_variables(pm.model)
    f()
    return JuMP.num_variables(pm.model) - before
end

"Assert that every variable of the container `key` in network `nw` is free."
function vfx_check_unbounded(pm, nw::Int, key::Symbol)
    vars = PowerModels.var(pm, nw, key)
    @test !isempty(vars)
    for v in vars
        @test vfx_bounds(v) == (nothing, nothing)
    end
end

"""
    vfx_check_report(f, comp, fields)

Check the `report` keyword of the variable function `f`: `f(pm; nw=1, kwargs...)` must
register the variables `fields` (each stored under its own name as field) for
every `comp` in network 1 by default and with `report=true`, and must register
nothing — while still creating the variables — with `report=false`.
"""
function vfx_check_report(f, comp::Symbol, fields::Vector{Symbol})
    for kwargs in ((;), (; report=true))
        pm = vfx_bare_model()
        f(pm; nw=1, kwargs...)
        entries = vfx_reported(pm, 1, comp)
        ids = sort(collect(PowerModels.ids(pm, 1, comp)))
        @test entries !== nothing
        entries === nothing && continue
        @test sort(collect(keys(entries))) == ids
        for i in ids
            @test sort(collect(keys(entries[i]))) == sort(fields)
            for key in fields
                @test entries[i][key] == PowerModels.var(pm, 1, key, i)
            end
        end
        # Only the network the variables were created for is reported.
        @test vfx_reported(pm, 2, comp) === nothing
    end

    pm = vfx_bare_model()
    f(pm; nw=1, report=false)
    @test vfx_reported(pm, 1, comp) === nothing
    for key in fields
        @test haskey(PowerModels.var(pm, 1), key)
    end
end

"""
    vfx_add_heat_storage!(data, i; capacity, status=true)

Add heat storage unit `i` (without a heat pump of the same index) to the
single-network `data` of `TestUtils.small_network`.
"""
function vfx_add_heat_storage!(data::Dict{String,<:Any}, i::Int; capacity::Real,
                               status::Bool=true)
    hs = deepcopy(data["heat_storage"]["1"])
    hs["index"] = i
    hs["name"] = "Heat_Storage_$i"
    hs["capacity"] = capacity
    hs["status"] = status
    data["heat_storage"][string(i)] = hs
    return data
end

"""
    vfx_add_component!(data, comp, i; fields...)

Copy unit 1 of component type `comp` (and its time series, if any) of the
single-network `data` to index `i`, overriding the given `fields`.
"""
function vfx_add_component!(data::Dict{String,<:Any}, comp::String, i::Int; fields...)
    unit = deepcopy(data[comp]["1"])
    unit["index"] = i
    for (k, v) in fields
        unit[string(k)] = v
    end
    data[comp][string(i)] = unit
    ts = get(data["time_series"], comp, Dict())
    haskey(ts, "1") && (ts[string(i)] = deepcopy(ts["1"]))
    return data
end

"Small network whose heat pump has a non-degenerate reactive power range."
function vfx_hp_q_network()
    data = TestUtils.small_network()
    data["heatpumps"]["1"]["q_min"] = -0.3
    data["heatpumps"]["1"]["q_max"] = 0.2
    return data
end

"Small network whose charging point has a non-degenerate reactive power range."
function vfx_cp_q_network()
    data = TestUtils.small_network()
    data["electromobility"]["1"]["q_min"] = -0.25
    data["electromobility"]["1"]["q_max"] = 0.15
    return data
end

"Integer ids of the components in the (single-network, string-keyed) dict `comps`."
vfx_int_ids(comps::Dict) = Set(parse(Int, i) for i in keys(comps))

# ---------------------------------------------------------------------------
# heat storage
# ---------------------------------------------------------------------------

@testset "variable_heat_storage" begin
    @testset "creates the power and the energy variables" begin
        pm = vfx_bare_model()
        added = vfx_added_variables(pm) do
            eDisGo_OPF.variable_heat_storage(pm, nw=1)
        end
        # One `phs` and one `hse` per heat storage unit, nothing else.
        @test added == 2
        @test TestUtils.variable_keys(pm, 1) == [:hse, :phs]
        @test vfx_indices(pm, 1, :phs) == Set([1])
        @test vfx_indices(pm, 1, :hse) == Set([1])
        @test isempty(PowerModels.var(pm, 2))
        # Bounds and reports of both parts (default keywords).
        @test vfx_bounds(PowerModels.var(pm, 1, :phs, 1)) == (-4.0, 4.0)
        @test vfx_bounds(PowerModels.var(pm, 1, :hse, 1)) == (0.0, 4.0)
        @test sort(collect(keys(vfx_reported(pm, 1, :heat_storage)[1]))) == [:hse, :phs]
    end

    @testset "forwards nw, bounded and report" begin
        pm = vfx_bare_model()
        eDisGo_OPF.variable_heat_storage(pm, nw=2, bounded=false, report=false)
        @test isempty(PowerModels.var(pm, 1))
        @test TestUtils.variable_keys(pm, 2) == [:hse, :phs]
        vfx_check_unbounded(pm, 2, :phs)
        vfx_check_unbounded(pm, 2, :hse)
        @test vfx_reported(pm, 2, :heat_storage) === nothing
    end

    @testset "in the full OPF build" begin
        for model_type in (SOCBFPowerModelEdisgo, NCBFPowerModelEdisgo)
            pm = TestUtils.build_small_model(model_type)
            for n in 1:2
                @test vfx_indices(pm, n, :phs) == Set([1])
                @test vfx_indices(pm, n, :hse) == Set([1])
                @test vfx_bounds(PowerModels.var(pm, n, :phs, 1)) == (-4.0, 4.0)
                @test vfx_bounds(PowerModels.var(pm, n, :hse, 1)) == (0.0, 4.0)
            end
        end
    end
end

@testset "variable_heat_storage_power" begin
    @testset "key and index set" begin
        pm = vfx_bare_model()
        added = vfx_added_variables(pm) do
            eDisGo_OPF.variable_heat_storage_power(pm, nw=1)
        end
        @test added == 1
        @test TestUtils.variable_keys(pm, 1) == [:phs]
        @test vfx_indices(pm, 1, :phs) == Set(PowerModels.ids(pm, 1, :heat_storage))
        @test vfx_indices(pm, 1, :phs) == Set([1])
        @test JuMP.name(PowerModels.var(pm, 1, :phs, 1)) == "1_phs[1]"
        # Network 2 is untouched.
        @test isempty(PowerModels.var(pm, 2))
    end

    @testset "indexed by heat storage, not by heat pump" begin
        # Unit 7 has no heat pump of the same index; it still gets its variable,
        # and no variable is created for the heat pumps.
        data = vfx_add_heat_storage!(TestUtils.small_network(), 7; capacity=2.5)
        pm = vfx_bare_model(data)
        @test sort(collect(PowerModels.ids(pm, 1, :heatpumps))) == [1]
        eDisGo_OPF.variable_heat_storage_power(pm, nw=1)
        @test vfx_indices(pm, 1, :phs) == Set([1, 7])
        @test vfx_bounds(PowerModels.var(pm, 1, :phs, 1)) == (-4.0, 4.0)
        @test vfx_bounds(PowerModels.var(pm, 1, :phs, 7)) == (-2.5, 2.5)
    end

    @testset "no heat storage" begin
        data = TestUtils.small_network()
        empty!(data["heat_storage"])
        pm = vfx_bare_model(data)
        added = vfx_added_variables(pm) do
            eDisGo_OPF.variable_heat_storage_power(pm, nw=1)
        end
        # The (empty) container is registered all the same, so later lookups of
        # `var(pm, nw, :phs)` do not fail.
        @test added == 0
        @test haskey(PowerModels.var(pm, 1), :phs)
        @test isempty(PowerModels.var(pm, 1, :phs))
    end

    @testset "inactive heat storage" begin
        data = vfx_add_heat_storage!(TestUtils.small_network(), 2; capacity=2.5,
                                     status=false)
        pm = vfx_bare_model(data)
        eDisGo_OPF.variable_heat_storage_power(pm, nw=1)
        # Actual behaviour: `ref_add_core!` does not filter `heat_storage` by its
        # `status` field (unlike battery `storage`), so the inactive unit 2 gets a
        # variable — and, bounded by ±capacity, may be dispatched.
        @test vfx_indices(pm, 1, :phs) == Set([1, 2])
        # Intended behaviour: inactive units are left out of the model.
        @test_broken vfx_indices(pm, 1, :phs) == Set([1])
    end

    @testset "bounded" begin
        # Bounded by ±capacity on every network; a capacity changed in the `ref`
        # of network 2 only shows up there.
        pm = vfx_bare_model()
        PowerModels.ref(pm, 2, :heat_storage, 1)["capacity"] = 6.0
        eDisGo_OPF.variable_heat_storage_power(pm, nw=1)
        eDisGo_OPF.variable_heat_storage_power(pm, nw=2, bounded=true)
        @test vfx_bounds(PowerModels.var(pm, 1, :phs, 1)) == (-4.0, 4.0)
        @test vfx_bounds(PowerModels.var(pm, 2, :phs, 1)) == (-6.0, 6.0)
        @test JuMP.name(PowerModels.var(pm, 2, :phs, 1)) == "2_phs[1]"
    end

    @testset "unbounded" begin
        pm = vfx_bare_model()
        eDisGo_OPF.variable_heat_storage_power(pm, nw=1, bounded=false)
        vfx_check_unbounded(pm, 1, :phs)
    end

    @testset "start values" begin
        # Without `phs_start` the start value falls back to 0.0 ...
        pm = vfx_bare_model()
        eDisGo_OPF.variable_heat_storage_power(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :phs, 1)) == 0.0

        # ... otherwise it is taken from the unit, also when unbounded.
        data = vfx_add_heat_storage!(TestUtils.small_network(), 2; capacity=2.5)
        data["heat_storage"]["1"]["phs_start"] = 1.5
        data["heat_storage"]["2"]["phs_start"] = -0.75
        pm = vfx_bare_model(data)
        eDisGo_OPF.variable_heat_storage_power(pm, nw=1)
        eDisGo_OPF.variable_heat_storage_power(pm, nw=2, bounded=false)
        for n in 1:2
            @test JuMP.start_value(PowerModels.var(pm, n, :phs, 1)) == 1.5
            @test JuMP.start_value(PowerModels.var(pm, n, :phs, 2)) == -0.75
        end
        # The energy start value is not mixed up with the power start value.
        data = TestUtils.small_network()
        data["heat_storage"]["1"]["hse_start"] = 3.0
        pm = vfx_bare_model(data)
        eDisGo_OPF.variable_heat_storage_power(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :phs, 1)) == 0.0
    end

    @testset "report" begin
        vfx_check_report(eDisGo_OPF.variable_heat_storage_power, :heat_storage, [:phs])
    end

    @testset "real eDisGo test grid" begin
        data = TestUtils.pm_network()
        pm = vfx_bare_model(TestUtils.pm_network())
        for n in sort(collect(PowerModels.nw_ids(pm)))
            eDisGo_OPF.variable_heat_storage_power(pm, nw=n)
            @test vfx_indices(pm, n, :phs) == vfx_int_ids(data["heat_storage"])
            for (i, hs) in data["heat_storage"]
                @test vfx_bounds(PowerModels.var(pm, n, :phs, parse(Int, i))) ==
                      (-hs["capacity"], hs["capacity"])
            end
        end
        @test JuMP.num_variables(pm.model) == 4 * length(data["heat_storage"])
    end
end

@testset "variable_heat_storage_energy" begin
    @testset "key and index set" begin
        pm = vfx_bare_model()
        added = vfx_added_variables(pm) do
            eDisGo_OPF.variable_heat_storage_energy(pm, nw=1)
        end
        @test added == 1
        @test TestUtils.variable_keys(pm, 1) == [:hse]
        @test vfx_indices(pm, 1, :hse) == Set([1])
        @test JuMP.name(PowerModels.var(pm, 1, :hse, 1)) == "1_hse[1]"
        @test isempty(PowerModels.var(pm, 2))
    end

    @testset "indexed by heat storage, not by heat pump" begin
        data = vfx_add_heat_storage!(TestUtils.small_network(), 7; capacity=2.5)
        pm = vfx_bare_model(data)
        eDisGo_OPF.variable_heat_storage_energy(pm, nw=1)
        @test vfx_indices(pm, 1, :hse) == Set([1, 7])
        @test vfx_bounds(PowerModels.var(pm, 1, :hse, 7)) == (0.0, 2.5)
    end

    @testset "no heat storage" begin
        data = TestUtils.small_network()
        empty!(data["heat_storage"])
        pm = vfx_bare_model(data)
        @test vfx_added_variables(() -> eDisGo_OPF.variable_heat_storage_energy(pm, nw=1),
                                  pm) == 0
        @test haskey(PowerModels.var(pm, 1), :hse)
        @test isempty(PowerModels.var(pm, 1, :hse))
    end

    @testset "inactive heat storage" begin
        data = vfx_add_heat_storage!(TestUtils.small_network(), 2; capacity=2.5,
                                     status=false)
        pm = vfx_bare_model(data)
        eDisGo_OPF.variable_heat_storage_energy(pm, nw=1)
        # Same as for `phs`: the `status` field is ignored.
        @test vfx_indices(pm, 1, :hse) == Set([1, 2])
        @test_broken vfx_indices(pm, 1, :hse) == Set([1])
    end

    @testset "bounded" begin
        # Between 0 (empty) and the capacity; the lower bound does not depend on
        # the capacity and is not taken from `soc_initial`/`energy`.
        data = TestUtils.small_network()
        data["heat_storage"]["1"]["soc_initial"] = 1.0
        data["heat_storage"]["1"]["energy"] = 2.0
        pm = vfx_bare_model(data)
        PowerModels.ref(pm, 2, :heat_storage, 1)["capacity"] = 6.0
        eDisGo_OPF.variable_heat_storage_energy(pm, nw=1)
        eDisGo_OPF.variable_heat_storage_energy(pm, nw=2, bounded=true)
        @test vfx_bounds(PowerModels.var(pm, 1, :hse, 1)) == (0.0, 4.0)
        @test vfx_bounds(PowerModels.var(pm, 2, :hse, 1)) == (0.0, 6.0)
        @test JuMP.name(PowerModels.var(pm, 2, :hse, 1)) == "2_hse[1]"
    end

    @testset "unbounded" begin
        # Not even the physical lower bound 0 is kept.
        pm = vfx_bare_model()
        eDisGo_OPF.variable_heat_storage_energy(pm, nw=1, bounded=false)
        vfx_check_unbounded(pm, 1, :hse)
    end

    @testset "start values" begin
        pm = vfx_bare_model()
        eDisGo_OPF.variable_heat_storage_energy(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :hse, 1)) == 0.0

        data = TestUtils.small_network()
        data["heat_storage"]["1"]["hse_start"] = 3.25
        data["heat_storage"]["1"]["phs_start"] = -1.0
        pm = vfx_bare_model(data)
        eDisGo_OPF.variable_heat_storage_energy(pm, nw=1)
        eDisGo_OPF.variable_heat_storage_energy(pm, nw=2, bounded=false)
        @test JuMP.start_value(PowerModels.var(pm, 1, :hse, 1)) == 3.25
        @test JuMP.start_value(PowerModels.var(pm, 2, :hse, 1)) == 3.25
    end

    @testset "report" begin
        vfx_check_report(eDisGo_OPF.variable_heat_storage_energy, :heat_storage, [:hse])
    end

    @testset "real eDisGo test grid" begin
        data = TestUtils.pm_network()
        pm = vfx_bare_model(TestUtils.pm_network())
        for n in sort(collect(PowerModels.nw_ids(pm)))
            eDisGo_OPF.variable_heat_storage_energy(pm, nw=n)
            @test vfx_indices(pm, n, :hse) == vfx_int_ids(data["heat_storage"])
            for (i, hs) in data["heat_storage"]
                @test vfx_bounds(PowerModels.var(pm, n, :hse, parse(Int, i))) ==
                      (0.0, hs["capacity"])
            end
        end
    end
end

# ---------------------------------------------------------------------------
# heat pumps
# ---------------------------------------------------------------------------

@testset "variable_heat_pump_power" begin
    @testset "creates only the active power variable" begin
        pm = vfx_bare_model()
        added = vfx_added_variables(pm) do
            eDisGo_OPF.variable_heat_pump_power(pm, nw=1)
        end
        # The reactive power call is commented out: no `qhp`, reactive heat pump
        # power is derived from the power factor in the power balance instead.
        @test added == 1
        @test TestUtils.variable_keys(pm, 1) == [:php]
        @test !haskey(PowerModels.var(pm, 1), :qhp)
        @test vfx_bounds(PowerModels.var(pm, 1, :php, 1)) == (0.0, 2.0)
        @test sort(collect(keys(vfx_reported(pm, 1, :heatpumps)[1]))) == [:php]
        @test isempty(PowerModels.var(pm, 2))
    end

    @testset "forwards nw, bounded and report" begin
        pm = vfx_bare_model()
        eDisGo_OPF.variable_heat_pump_power(pm, nw=2, bounded=false, report=false)
        @test isempty(PowerModels.var(pm, 1))
        @test TestUtils.variable_keys(pm, 2) == [:php]
        vfx_check_unbounded(pm, 2, :php)
        @test vfx_reported(pm, 2, :heatpumps) === nothing
    end

    @testset "in the full OPF build" begin
        for model_type in (SOCBFPowerModelEdisgo, NCBFPowerModelEdisgo)
            pm = TestUtils.build_small_model(model_type)
            for n in 1:2
                @test vfx_indices(pm, n, :php) == Set([1])
                @test vfx_bounds(PowerModels.var(pm, n, :php, 1)) == (0.0, 2.0)
                @test !haskey(PowerModels.var(pm, n), :qhp)
            end
        end
    end
end

@testset "variable_heat_pump_power_real" begin
    @testset "key and index set" begin
        data = vfx_add_component!(TestUtils.small_network(), "heatpumps", 5;
                                  p_max=0.8)
        pm = vfx_bare_model(data)
        added = vfx_added_variables(pm) do
            eDisGo_OPF.variable_heat_pump_power_real(pm, nw=1)
        end
        @test added == 2
        @test TestUtils.variable_keys(pm, 1) == [:php]
        @test vfx_indices(pm, 1, :php) == Set(PowerModels.ids(pm, 1, :heatpumps))
        @test vfx_indices(pm, 1, :php) == Set([1, 5])
        @test JuMP.name(PowerModels.var(pm, 1, :php, 5)) == "1_php[5]"
        @test isempty(PowerModels.var(pm, 2))
    end

    @testset "no heat pumps" begin
        data = TestUtils.small_network()
        empty!(data["heatpumps"])
        empty!(data["time_series"]["heatpumps"])
        pm = vfx_bare_model(data)
        @test vfx_added_variables(() -> eDisGo_OPF.variable_heat_pump_power_real(pm, nw=1),
                                  pm) == 0
        @test isempty(PowerModels.var(pm, 1, :php))
    end

    @testset "bounded" begin
        # Bounded by `p_min`/`p_max` — not by the time-varying heat demand `pd`,
        # which only enters through `constraint_hp_operation`.
        data = TestUtils.small_network()
        data["heatpumps"]["1"]["p_min"] = 0.1
        data["time_series"]["heatpumps"]["1"]["pd"] = [0.4, 1.6]
        pm = vfx_bare_model(data)
        PowerModels.ref(pm, 2, :heatpumps, 1)["p_max"] = 1.25
        eDisGo_OPF.variable_heat_pump_power_real(pm, nw=1)
        eDisGo_OPF.variable_heat_pump_power_real(pm, nw=2, bounded=true)
        @test vfx_bounds(PowerModels.var(pm, 1, :php, 1)) == (0.1, 2.0)
        @test vfx_bounds(PowerModels.var(pm, 2, :php, 1)) == (0.1, 1.25)
        @test JuMP.name(PowerModels.var(pm, 2, :php, 1)) == "2_php[1]"
    end

    @testset "unbounded" begin
        pm = vfx_bare_model()
        eDisGo_OPF.variable_heat_pump_power_real(pm, nw=1, bounded=false)
        vfx_check_unbounded(pm, 1, :php)
    end

    @testset "start values" begin
        pm = vfx_bare_model()
        eDisGo_OPF.variable_heat_pump_power_real(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :php, 1)) == 0.0

        data = TestUtils.small_network()
        data["heatpumps"]["1"]["php_start"] = 1.2
        pm = vfx_bare_model(data)
        eDisGo_OPF.variable_heat_pump_power_real(pm, nw=1)
        eDisGo_OPF.variable_heat_pump_power_real(pm, nw=2, bounded=false)
        @test JuMP.start_value(PowerModels.var(pm, 1, :php, 1)) == 1.2
        @test JuMP.start_value(PowerModels.var(pm, 2, :php, 1)) == 1.2
    end

    @testset "report" begin
        vfx_check_report(eDisGo_OPF.variable_heat_pump_power_real, :heatpumps, [:php])
    end

    @testset "real eDisGo test grid" begin
        data = TestUtils.pm_network()
        pm = vfx_bare_model(TestUtils.pm_network())
        for n in sort(collect(PowerModels.nw_ids(pm)))
            eDisGo_OPF.variable_heat_pump_power_real(pm, nw=n)
            @test vfx_indices(pm, n, :php) == vfx_int_ids(data["heatpumps"])
            for (i, hp) in data["heatpumps"]
                @test vfx_bounds(PowerModels.var(pm, n, :php, parse(Int, i))) ==
                      (hp["p_min"], hp["p_max"])
            end
        end
    end
end

@testset "variable_heat_pump_power_imaginary" begin
    @testset "key and index set" begin
        data = vfx_add_component!(vfx_hp_q_network(), "heatpumps", 5)
        pm = vfx_bare_model(data)
        added = vfx_added_variables(pm) do
            eDisGo_OPF.variable_heat_pump_power_imaginary(pm, nw=1)
        end
        @test added == 2
        @test TestUtils.variable_keys(pm, 1) == [:qhp]
        @test vfx_indices(pm, 1, :qhp) == Set([1, 5])
        @test JuMP.name(PowerModels.var(pm, 1, :qhp, 1)) == "1_qhp[1]"
        @test isempty(PowerModels.var(pm, 2))
    end

    @testset "bounded" begin
        pm = vfx_bare_model(vfx_hp_q_network())
        PowerModels.ref(pm, 2, :heatpumps, 1)["q_max"] = 0.5
        eDisGo_OPF.variable_heat_pump_power_imaginary(pm, nw=1)
        eDisGo_OPF.variable_heat_pump_power_imaginary(pm, nw=2, bounded=true)
        @test vfx_bounds(PowerModels.var(pm, 1, :qhp, 1)) == (-0.3, 0.2)
        @test vfx_bounds(PowerModels.var(pm, 2, :qhp, 1)) == (-0.3, 0.5)
        # Not to be confused with the active power bounds.
        @test vfx_bounds(PowerModels.var(pm, 1, :qhp, 1)) !=
              (PowerModels.ref(pm, 1, :heatpumps, 1)["p_min"],
               PowerModels.ref(pm, 1, :heatpumps, 1)["p_max"])
    end

    @testset "unbounded" begin
        pm = vfx_bare_model(vfx_hp_q_network())
        eDisGo_OPF.variable_heat_pump_power_imaginary(pm, nw=1, bounded=false)
        vfx_check_unbounded(pm, 1, :qhp)
    end

    @testset "no start values" begin
        # Unlike `php`, `qhp` is created without a `start`, so a `qhp_start` in
        # the data is ignored and the variable has no start value at all.
        data = vfx_hp_q_network()
        data["heatpumps"]["1"]["qhp_start"] = 0.1
        pm = vfx_bare_model(data)
        eDisGo_OPF.variable_heat_pump_power_imaginary(pm, nw=1)
        @test JuMP.start_value(PowerModels.var(pm, 1, :qhp, 1)) === nothing
    end

    @testset "report" begin
        vfx_check_report(eDisGo_OPF.variable_heat_pump_power_imaginary, :heatpumps, [:qhp])
    end

    @testset "real eDisGo test grid" begin
        data = TestUtils.pm_network()
        pm = vfx_bare_model(TestUtils.pm_network())
        for n in sort(collect(PowerModels.nw_ids(pm)))
            eDisGo_OPF.variable_heat_pump_power_imaginary(pm, nw=n)
            @test vfx_indices(pm, n, :qhp) == vfx_int_ids(data["heatpumps"])
            for (i, hp) in data["heatpumps"]
                @test vfx_bounds(PowerModels.var(pm, n, :qhp, parse(Int, i))) ==
                      (hp["q_min"], hp["q_max"])
            end
        end
    end
end
