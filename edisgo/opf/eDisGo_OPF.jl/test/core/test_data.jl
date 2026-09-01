# Tests for `src/core/data.jl`.
#
# Two independent concerns live in this file:
#
# * `set_ac_bf_start_values!` seeds the warm start of the non-convex solve by
#   copying the optimised values of every flexibility into the `*_start` fields
#   the `variable_*` functions read (see `comp_start_value` in
#   `src/core/variables.jl`). It is a pure dict transformation, so the tests
#   compare the mutated dict field by field against the input.
# * `correct_bus_types!` / `_correct_bus_types!` repair the `bus_type` markers
#   before a power flow. Both are currently *dead code* — the only call site,
#   in `correct_network_data!` (`src/io/common.jl`), is commented out — and the
#   worker is broken: it references `_LOGGER` and `_biggest_generator`, neither
#   of which is in scope inside `eDisGo_OPF` (both are internal, non-exported
#   names of `PowerModels`). Every branch that wants to *change* something
#   therefore raises an `UndefVarError` instead. The tests below pin down that
#   actual behaviour and mark the intended behaviour `@test_broken`.

include(joinpath(@__DIR__, "..", "testsetup.jl"))

"""
The flexibility groups `set_ac_bf_start_values!` walks, mapped to the result
fields it copies into the corresponding `<field>_start` field. Listed in the
order the function processes them, which is also the order in which a missing
group makes it fail.
"""
const DATA_START_FIELDS = [
    "gen_nd" => ["pgc"],
    "gen_slack" => ["pgs", "qgs"],
    "dsm" => ["pdsm", "dsme"],
    "storage" => ["ps", "se"],
    "electromobility" => ["pcp"],
    "heatpumps" => ["php"],
    "heat_storage" => ["phs", "hse"],
]

"Total number of `*_start` fields a complete run has to produce."
const DATA_NUM_START_FIELDS = sum(length(fields) for (_, fields) in DATA_START_FIELDS)

"""
[`TestUtils.small_network`](@ref) with one *result* value per field of
[`DATA_START_FIELDS`](@ref), as `PowerModels.update_data!` would leave it after a
solve. Returns the network and a `(group, field) => value` map of the values that
were injected; every value is distinct, so a test notices when a value ends up in
the wrong `*_start` field.
"""
function data_solved_network()
    net = TestUtils.small_network()
    expected = Dict{Tuple{String,String},Float64}()
    value = 0.0
    for (group, fields) in DATA_START_FIELDS, field in fields
        value += 0.125
        net[group]["1"][field] = value
        expected[(group, field)] = value
    end
    return net, expected
end

"The exception `f` throws, or `nothing` when `f` returns normally."
function data_thrown(f)
    try
        f()
    catch exception
        return exception
    end
    return nothing
end

@testset "set_ac_bf_start_values!" begin
    net, expected = data_solved_network()
    before = deepcopy(net)

    @test eDisGo_OPF.set_ac_bf_start_values!(net) === nothing

    # Every expected `*_start` field exists and carries the matching result value.
    for ((group, field), value) in expected
        component = net[group]["1"]
        @test haskey(component, field * "_start")
        @test component[field * "_start"] == value
    end
    # … and no others: exactly one `*_start` field per copied result field.
    @test sum(count(endswith("_start"), keys(net[group]["1"]))
              for (group, _) in DATA_START_FIELDS) == DATA_NUM_START_FIELDS
    @test DATA_NUM_START_FIELDS == 11

    # Everything that was already there survives untouched — the function only
    # ever adds `*_start` keys.
    for (group, _) in DATA_START_FIELDS
        @test filter(p -> !endswith(p.first, "_start"), net[group]["1"]) ==
              before[group]["1"]
    end
    # Component types the function does not know about are not visited at all:
    # in particular the dispatchable generators (`gen`), whose set point comes
    # from the time series rather than from the optimisation.
    for group in ("gen", "load", "bus", "branch", "time_series")
        @test net[group] == before[group]
    end

    # Values are copied by assignment, not deep-copied: a mutable result value is
    # shared between the field and its `*_start` twin.
    vector_net, _ = data_solved_network()
    vector_net["dsm"]["1"]["dsme"] = [1.0, 2.0]
    eDisGo_OPF.set_ac_bf_start_values!(vector_net)
    @test vector_net["dsm"]["1"]["dsme_start"] === vector_net["dsm"]["1"]["dsme"]

    # Re-running after a new solve overwrites the previous start values.
    net["storage"]["1"]["ps"] = -0.375
    net["heatpumps"]["1"]["php"] = 1.75
    eDisGo_OPF.set_ac_bf_start_values!(net)
    @test net["storage"]["1"]["ps_start"] == -0.375
    @test net["heatpumps"]["1"]["php_start"] == 1.75
    @test net["gen_nd"]["1"]["pgc_start"] == expected[("gen_nd", "pgc")]

    # Empty flexibility groups are fine — nothing to copy, nothing added.
    empty_net, _ = data_solved_network()
    for (group, _) in DATA_START_FIELDS
        empty_net[group] = Dict{String,Any}()
    end
    @test eDisGo_OPF.set_ac_bf_start_values!(empty_net) === nothing
    @test all(isempty(empty_net[group]) for (group, _) in DATA_START_FIELDS)

    # A missing component group is not tolerated: the function indexes into the
    # dict directly, so every one of the seven groups has to be present.
    for (group, _) in DATA_START_FIELDS
        incomplete, _ = data_solved_network()
        delete!(incomplete, group)
        exception = data_thrown(() -> eDisGo_OPF.set_ac_bf_start_values!(incomplete))
        @test exception isa KeyError
        @test exception.key == group
    end

    # Likewise for an unsolved network: the result fields have to be in place,
    # which is what `PowerModels.update_data!` does before the call in `Main.jl`.
    unsolved = TestUtils.small_network()
    exception = data_thrown(() -> eDisGo_OPF.set_ac_bf_start_values!(unsolved))
    @test exception isa KeyError
    @test exception.key == "pgc"

    # The call site in `Main.jl` passes a single sub-network of the multi-network
    # dict, so only that one gains start values.
    solved, _ = data_solved_network()
    mn = TestUtils.multinetwork(solved)
    eDisGo_OPF.set_ac_bf_start_values!(mn["nw"]["1"])
    @test mn["nw"]["1"]["heat_storage"]["1"]["hse_start"] ==
          expected[("heat_storage", "hse")]
    @test !haskey(mn["nw"]["2"]["heat_storage"]["1"], "hse_start")
end

