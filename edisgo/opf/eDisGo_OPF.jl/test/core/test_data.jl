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

@testset "correct_bus_types!" begin
    # Single network: the worker is applied to the data dict itself. The fixture
    # is already consistent (bus 1 is the slack and carries `gen_slack` 1), so
    # nothing changes.
    net = TestUtils.small_network()
    @test eDisGo_OPF.correct_bus_types!(net) === nothing
    @test [net["bus"][i]["bus_type"] for i in ["1", "2", "3", "4"]] == [3, 1, 1, 1]

    # It really dispatches into `_correct_bus_types!`: a probe that makes the
    # worker fail (a missing generator group it indexes into) surfaces as the
    # worker's own `KeyError`.
    probe = TestUtils.small_network()
    delete!(probe, "gen_nd")
    exception = data_thrown(() -> eDisGo_OPF.correct_bus_types!(probe))
    @test exception isa KeyError
    @test exception.key == "gen_nd"

    # Multi-network: `apply_pm!` descends into every sub-network …
    mn = TestUtils.multinetwork(TestUtils.small_network())
    @test eDisGo_OPF.correct_bus_types!(mn) === nothing
    for n in ("1", "2")
        @test [mn["nw"][n]["bus"][i]["bus_type"] for i in ["1", "2", "3", "4"]] ==
              [3, 1, 1, 1]
    end
    # … each of them, not just the first: the probe fires wherever it is planted.
    for n in ("1", "2")
        probe_mn = TestUtils.multinetwork(TestUtils.small_network())
        delete!(probe_mn["nw"][n], "gen_nd")
        exception = data_thrown(() -> eDisGo_OPF.correct_bus_types!(probe_mn))
        @test exception isa KeyError
        @test exception.key == "gen_nd"
    end
    # … and *not* into the multi-network wrapper itself, which has no "bus" key
    # of its own — otherwise the clean run above would have raised
    # `KeyError("bus")`.
    @test !haskey(mn, "bus")

    # A correction inside a sub-network hits the same `_LOGGER` bug as the
    # single-network case (see the `_correct_bus_types!` test set below): bus 2
    # of sub-network 2 is marked PV but its generator is switched off.
    broken_mn = TestUtils.multinetwork(TestUtils.small_network())
    for n in ("1", "2")
        broken_mn["nw"][n]["gen"]["1"]["gen_status"] = 0
    end
    broken_mn["nw"]["2"]["bus"]["2"]["bus_type"] = 2
    exception = data_thrown(() -> eDisGo_OPF.correct_bus_types!(broken_mn))
    @test exception isa UndefVarError
    @test exception.var === :_LOGGER
    @test broken_mn["nw"]["2"]["bus"]["2"]["bus_type"] == 2
    # BUG: intended behaviour is the demotion to PQ in the affected sub-network.
    @test_broken (eDisGo_OPF.correct_bus_types!(broken_mn);
                  broken_mn["nw"]["2"]["bus"]["2"]["bus_type"] == 1)
end

@testset "_correct_bus_types!" begin
    # --- paths that work ---------------------------------------------------
    # A consistent network is left alone: bus 1 is the slack and has an active
    # `gen_slack`, the remaining buses are PQ.
    net = TestUtils.small_network()
    @test eDisGo_OPF._correct_bus_types!(net) === nothing
    @test [net["bus"][i]["bus_type"] for i in ["1", "2", "3", "4"]] == [3, 1, 1, 1]
    # Buses 2 and 3 carry active generators (`gen` and `gen_nd`) and stay PQ all
    # the same: the promotion of generator buses to PV is commented out in the
    # source.
    @test net["gen"]["1"]["gen_bus"] == 2
    @test net["gen_nd"]["1"]["gen_bus"] == 3

    # Generators of all three groups count towards a bus. `gen_slack` is covered
    # by the run above (bus 1 has no other generator); `gen` and `gen_nd` keep a
    # PV bus a PV bus.
    for (bus, group) in (("2", "gen"), ("3", "gen_nd"))
        pv = TestUtils.small_network()
        pv["bus"][bus]["bus_type"] = 2
        @test pv[group]["1"]["gen_bus"] == parse(Int, bus)
        @test eDisGo_OPF._correct_bus_types!(pv) === nothing
        @test pv["bus"][bus]["bus_type"] == 2
    end
    # A `gen` at the slack bus is enough to validate it, without any `gen_slack`.
    gen_slack_bus = TestUtils.small_network()
    gen_slack_bus["gen_slack"] = Dict{String,Any}()
    gen_slack_bus["gen"]["1"]["gen_bus"] = 1
    @test eDisGo_OPF._correct_bus_types!(gen_slack_bus) === nothing
    @test gen_slack_bus["bus"]["1"]["bus_type"] == 3

    # Inactive buses (type 4) are never touched, with or without a generator.
    for bus in ("2", "4")
        inactive = TestUtils.small_network()
        inactive["bus"][bus]["bus_type"] = 4
        @test eDisGo_OPF._correct_bus_types!(inactive) === nothing
        @test inactive["bus"][bus]["bus_type"] == 4
    end

    # The real eDisGo test grid (143 buses, one slack, one `gen`, 27 `gen_nd`)
    # passes unchanged as well.
    real = TestUtils.pm_network()
    real_before = Dict(i => bus["bus_type"] for (i, bus) in real["bus"])
    @test eDisGo_OPF._correct_bus_types!(real) === nothing
    @test Dict(i => bus["bus_type"] for (i, bus) in real["bus"]) == real_before
    @test count(bus["bus_type"] == 3 for bus in values(real["bus"])) == 1

    # --- paths that are broken --------------------------------------------
    # BUG: `_LOGGER` and `_biggest_generator` are internal names of PowerModels
    # and are not imported into `eDisGo_OPF`, so every branch that logs a warning
    # or looks for a replacement slack dies with an `UndefVarError` before it can
    # apply its fix. Each case below asserts the actual behaviour and marks the
    # documented behaviour `@test_broken`.

    # PV bus (type 2) without an active generator should become PQ (type 1).
    # `gen_status == 0` means the generator does not count.
    pv_off = TestUtils.small_network()
    pv_off["bus"]["2"]["bus_type"] = 2
    pv_off["gen"]["1"]["gen_status"] = 0
    exception = data_thrown(() -> eDisGo_OPF._correct_bus_types!(pv_off))
    @test exception isa UndefVarError
    @test exception.var === :_LOGGER
    # The warning precedes the assignment, so the bus keeps its wrong type.
    @test pv_off["bus"]["2"]["bus_type"] == 2
    @test_broken (eDisGo_OPF._correct_bus_types!(pv_off);
                  pv_off["bus"]["2"]["bus_type"] == 1)

    # Slack bus (type 3) without an active generator should become PQ, too —
    # after which no slack is left and the fallback below would take over.
    slack_off = TestUtils.small_network()
    slack_off["gen_slack"]["1"]["gen_status"] = 0
    exception = data_thrown(() -> eDisGo_OPF._correct_bus_types!(slack_off))
    @test exception isa UndefVarError
    @test exception.var === :_LOGGER
    @test slack_off["bus"]["1"]["bus_type"] == 3
    @test_broken (eDisGo_OPF._correct_bus_types!(slack_off);
                  slack_off["bus"]["1"]["bus_type"] != 3)

    # Unrecognised bus types become PV when generators hang off the bus (bus 2)
    # and PQ when none do (bus 4). Both negative and out-of-range codes are
    # "unknown".
    for (bus, code, intended) in (("2", 7, 2), ("4", 7, 1), ("2", 0, 2), ("4", -1, 1))
        unknown = TestUtils.small_network()
        unknown["bus"][bus]["bus_type"] = code
        exception = data_thrown(() -> eDisGo_OPF._correct_bus_types!(unknown))
        @test exception isa UndefVarError
        @test exception.var === :_LOGGER
        @test unknown["bus"][bus]["bus_type"] == code
        @test_broken (eDisGo_OPF._correct_bus_types!(unknown);
                      unknown["bus"][bus]["bus_type"] == intended)
    end

    # No slack bus at all: the bus of the biggest generator should be promoted.
    # Note that only `gen` is searched — `gen_nd` and `gen_slack` do not take
    # part in the fallback — so bus 2 would become the reference bus here.
    no_slack = TestUtils.small_network()
    no_slack["bus"]["1"]["bus_type"] = 1
    exception = data_thrown(() -> eDisGo_OPF._correct_bus_types!(no_slack))
    @test exception isa UndefVarError
    @test exception.var === :_biggest_generator
    @test all(bus["bus_type"] == 1 for bus in values(no_slack["bus"]))
    @test_broken (eDisGo_OPF._correct_bus_types!(no_slack);
                  no_slack["bus"]["2"]["bus_type"] == 3)

    # Same fallback, but with several dispatchable generators: the one with the
    # largest `pmax` wins.
    biggest = TestUtils.small_network()
    biggest["bus"]["1"]["bus_type"] = 1
    biggest["gen"]["2"] = merge(deepcopy(biggest["gen"]["1"]),
                                Dict{String,Any}("index" => 2, "gen_bus" => 4,
                                                 "pmax" => 5.0))
    exception = data_thrown(() -> eDisGo_OPF._correct_bus_types!(biggest))
    @test exception isa UndefVarError
    @test exception.var === :_biggest_generator
    @test_broken (eDisGo_OPF._correct_bus_types!(biggest);
                  biggest["bus"]["4"]["bus_type"] == 3)

    # No generators at all: the documented behaviour is a `Memento.error` about
    # the missing reference generator. Actually the `UndefVarError` above fires
    # first, so that message is unreachable.
    no_gens = TestUtils.small_network()
    no_gens["bus"]["1"]["bus_type"] = 1
    for group in ("gen", "gen_nd", "gen_slack")
        no_gens[group] = Dict{String,Any}()
    end
    exception = data_thrown(() -> eDisGo_OPF._correct_bus_types!(no_gens))
    @test exception isa UndefVarError
    @test exception.var === :_biggest_generator
    @test_broken occursin("no generators found",
                          sprint(showerror,
                                 data_thrown(() -> eDisGo_OPF._correct_bus_types!(no_gens))))

    # --- structural requirements -------------------------------------------
    # All three generator groups and the bus table have to exist.
    for group in ("bus", "gen", "gen_nd", "gen_slack")
        incomplete = TestUtils.small_network()
        delete!(incomplete, group)
        exception = data_thrown(() -> eDisGo_OPF._correct_bus_types!(incomplete))
        @test exception isa KeyError
        @test exception.key == group
    end
    # A generator pointing at a bus that does not exist is not caught either.
    dangling = TestUtils.small_network()
    dangling["gen"]["1"]["gen_bus"] = 99
    exception = data_thrown(() -> eDisGo_OPF._correct_bus_types!(dangling))
    @test exception isa KeyError
    @test exception.key == 99
end
