# Tests for `src/io/common.jl`.
#
# The file defines a single function, `correct_network_data!` — eDisGo's variant of
# PowerModels' function of the same name. It validates and normalises the parsed
# network data *in place* before the OPF is built. Every individual check it runs
# is PowerModels' own, so what is tested here is the composition: that the dict is
# mutated in place, that a well-formed eDisGo network survives it untouched, that
# each deliberately broken input is either repaired or rejected, and — just as
# important — which of PowerModels' checks eDisGo deliberately leaves out
# (`check_reference_bus`, `correct_bus_types!`, both commented out in the source).

include(joinpath(@__DIR__, "..", "testsetup.jl"))

"""
Build a fresh `TestUtils.small_network()`, apply `mutate!` to it and return the
dict *without* validating — the starting point of every case below.
"""
function common_network(mutate!::Function=identity; kwargs...)
    data = TestUtils.small_network(; kwargs...)
    mutate!(data)
    return data
end

"As [`common_network`](@ref), but run `correct_network_data!` before returning."
function common_validated(mutate!::Function=identity; kwargs...)
    data = common_network(mutate!; kwargs...)
    eDisGo_OPF.correct_network_data!(data)
    return data
end

# Default the angle limits of `small_network()` are tightened *to*: PowerModels'
# `default_pad` of 1.0472 rad (60 deg.).
const COMMON_DEFAULT_PAD = 1.0472

@testset "correct_network_data!" begin

    @testset "in-place mutation" begin
        # The function is a `!` function through and through: it hands back
        # nothing and communicates exclusively through the dict it was given.
        data = common_network(d -> d["branch"]["1"]["rate_a"] = 0.0)
        branch = data["branch"]["1"]
        @test eDisGo_OPF.correct_network_data!(data) === nothing
        # The correction is visible in the very object that was passed in — not in
        # a copy — down to the nested component dicts.
        @test !haskey(data["branch"]["1"], "rate_a")
        @test !haskey(branch, "rate_a")
        @test data["branch"]["1"] === branch
    end

    @testset "well-formed network is left untouched" begin
        # `small_network()` is what the Python side is supposed to send: already in
        # per unit, angles in radians, limits positive. Validation must be a no-op
        # on it, otherwise the fixture (and the input format) would be wrong.
        data = TestUtils.small_network()
        before = deepcopy(data)
        eDisGo_OPF.correct_network_data!(data)
        @test data == before

        # And it is idempotent: a second pass over already-corrected data changes
        # nothing either.
        corrected = common_validated(d -> d["branch"]["1"]["angmin"] = 0.0)
        once = deepcopy(corrected)
        eDisGo_OPF.correct_network_data!(corrected)
        @test corrected == once
    end

    @testset "make_per_unit!" begin
        # `per_unit` is set unconditionally …
        @test TestUtils.small_network()["per_unit"] == true
        @test common_validated(d -> delete!(d, "per_unit"))["per_unit"] == true
        @test common_validated(d -> d["per_unit"] = false)["per_unit"] == true

        # … but the rescaling only runs when the data was *not* already per-unit.
        # With `baseMVA = 2` every power quantity of a PowerModels component is
        # halved.
        data = common_validated(function (d)
            d["per_unit"] = false
            d["baseMVA"] = 2
        end)
        @test data["load"]["1"]["pd"] ≈ 0.1
        @test data["load"]["1"]["qd"] ≈ 0.0657 / 2
        @test data["gen"]["1"]["pmax"] ≈ 0.25
        @test data["storage"]["1"]["energy_rating"] ≈ 0.2
        @test data["branch"]["1"]["rate_a"] ≈ 2.5
        # Branch angles are treated as degrees on the way in and converted to
        # radians — which is why eDisGo data, whose angles are radians already,
        # must arrive with `per_unit = true`.
        @test data["branch"]["1"]["angmin"] ≈ deg2rad(-pi / 6)

        # PowerModels only knows its own component types: the eDisGo-specific ones
        # are not rescaled at all.
        @test data["gen_nd"]["1"]["pmax"] == 1.0
        @test data["heatpumps"]["1"]["p_max"] == 2.0
        @test data["electromobility"]["1"]["p_max"] == 1.0
        @test data["dsm"]["1"]["p_max"] == 0.3

        # Data that says it is per-unit is passed through unscaled, `baseMVA`
        # notwithstanding.
        untouched = common_validated(d -> d["baseMVA"] = 2)
        @test untouched["load"]["1"]["pd"] == 0.2
        @test untouched["branch"]["1"]["angmin"] == -pi / 6
    end

    @testset "correct_transformer_parameters!" begin
        # A tap of 0.0 would produce NaNs in the branch equations; missing tap /
        # shift values get their neutral defaults.
        data = common_validated(function (d)
            d["branch"]["1"]["tap"] = 0.0
            delete!(d["branch"]["2"], "shift")
            delete!(d["branch"]["3"], "tap")
        end)
        @test data["branch"]["1"]["tap"] == 1.0
        @test data["branch"]["2"]["shift"] == 0.0
        @test data["branch"]["3"]["tap"] == 1.0
        # A valid tap is kept as it is.
        @test common_validated(d -> d["branch"]["1"]["tap"] = 1.05)["branch"]["1"]["tap"] == 1.05
    end

    @testset "correct_voltage_angle_differences!" begin
        data = common_validated(function (d)
            # Degenerate: no angle difference allowed at all — widened to ±60 deg.
            d["branch"]["1"]["angmin"] = 0.0
            d["branch"]["1"]["angmax"] = 0.0
            # Wider than ±90 deg. — tightened to ±60 deg.
            d["branch"]["2"]["angmin"] = -pi
            d["branch"]["2"]["angmax"] = pi
        end)
        @test data["branch"]["1"]["angmin"] == -COMMON_DEFAULT_PAD
        @test data["branch"]["1"]["angmax"] == COMMON_DEFAULT_PAD
        @test data["branch"]["2"]["angmin"] == -COMMON_DEFAULT_PAD
        @test data["branch"]["2"]["angmax"] == COMMON_DEFAULT_PAD
        # The untouched branch keeps its (valid) ±30 deg.
        @test data["branch"]["3"]["angmin"] == -pi / 6
        @test data["branch"]["3"]["angmax"] == pi / 6

        # Exactly ±90 deg. is already out of range and gets tightened; anything
        # strictly inside is kept.
        limit = common_validated(function (d)
            d["branch"]["1"]["angmin"] = -pi / 2
            d["branch"]["1"]["angmax"] = pi / 2
            d["branch"]["2"]["angmin"] = -pi / 2 + 1e-6
            d["branch"]["2"]["angmax"] = pi / 2 - 1e-6
        end)
        @test limit["branch"]["1"]["angmin"] == -COMMON_DEFAULT_PAD
        @test limit["branch"]["1"]["angmax"] == COMMON_DEFAULT_PAD
        @test limit["branch"]["2"]["angmin"] == -pi / 2 + 1e-6
        @test limit["branch"]["2"]["angmax"] == pi / 2 - 1e-6
    end

    @testset "correct_thermal_limits!" begin
        # A zero rating means "unlimited" in Matpower and is removed rather than
        # enforced as a hard zero.
        data = common_validated(function (d)
            d["branch"]["1"]["rate_a"] = 0.0
            d["branch"]["2"]["rate_b"] = 0.0
        end)
        @test !haskey(data["branch"]["1"], "rate_a")
        @test !haskey(data["branch"]["2"], "rate_b")
        # Only the zero rating goes; the other limits of the same branch stay.
        @test data["branch"]["1"]["rate_b"] == 250.0
        @test data["branch"]["1"]["rate_c"] == 250.0
        @test data["branch"]["2"]["rate_a"] == 5.0
        # Positive ratings are untouched everywhere, virtual storage branch too.
        @test data["branch"]["3"]["rate_a"] == 5.0

        # A negative rating is not correctable and is rejected.
        for key in ("rate_a", "rate_b", "rate_c")
            @test_throws ErrorException eDisGo_OPF.correct_network_data!(
                common_network(d -> d["branch"]["1"][key] = -1.0))
        end
    end

    @testset "correct_current_limits!" begin
        # Same rule for the (optional) current ratings — eDisGo does not send
        # them, so they have to be added to the fixture to be seen at all.
        data = common_validated(function (d)
            d["branch"]["1"]["c_rating_a"] = 0.0
            d["branch"]["2"]["c_rating_a"] = 3.0
        end)
        @test !haskey(data["branch"]["1"], "c_rating_a")
        @test data["branch"]["2"]["c_rating_a"] == 3.0

        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["branch"]["1"]["c_rating_a"] = -1.0))
    end

    @testset "correct_branch_directions!" begin
        # Two parallel branches between the same pair of buses, entered with
        # opposite orientation: one of them is flipped so that both point the same
        # way. *Which* one is flipped depends on dict iteration order, so only the
        # invariant is asserted.
        data = common_validated(function (d)
            d["branch"]["4"] = deepcopy(d["branch"]["2"])
            d["branch"]["4"]["index"] = 4
            d["branch"]["4"]["f_bus"] = 3
            d["branch"]["4"]["t_bus"] = 2
        end)
        orientations = Set((data["branch"][i]["f_bus"], data["branch"][i]["t_bus"])
                           for i in ("2", "4"))
        @test length(orientations) == 1
        @test only(orientations) in ((2, 3), (3, 2))
        # The non-parallel branches keep the orientation they were given.
        @test (data["branch"]["1"]["f_bus"], data["branch"]["1"]["t_bus"]) == (1, 2)
        @test (data["branch"]["3"]["f_bus"], data["branch"]["3"]["t_bus"]) == (3, 4)

        # Without a parallel partner nothing is reordered at all.
        plain = common_validated()
        for (i, (f, t)) in (("1", (1, 2)), ("2", (2, 3)), ("3", (3, 4)))
            @test (plain["branch"][i]["f_bus"], plain["branch"][i]["t_bus"]) == (f, t)
        end
    end

    @testset "check_branch_loops" begin
        # A branch whose two ends are the same bus is a modelling error, not
        # something that can be repaired.
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["branch"]["2"]["t_bus"] = 2))
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["branch"]["3"]["f_bus"] = 4))
    end

    @testset "check_connectivity" begin
        # Every component must reference a bus that exists. Note that this is a
        # *reference* check, not a graph-connectivity check.
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["load"]["1"]["load_bus"] = 42))
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["gen"]["1"]["gen_bus"] = 42))
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["storage"]["1"]["storage_bus"] = 42))
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["branch"]["1"]["f_bus"] = 42))
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["branch"]["1"]["t_bus"] = 42))

        # A genuinely disconnected grid — bus 4 and its branch cut loose from the
        # feeder — is *not* caught here: the buses all exist, so the data passes
        # validation and the problem only shows up later, in the OPF itself.
        island = common_validated(function (d)
            delete!(d["branch"], "3")
            delete!(d["storage"], "1")
        end)
        @test haskey(island["bus"], "4")
        @test !any(b["f_bus"] == 4 || b["t_bus"] == 4 for b in values(island["branch"]))

        # The eDisGo-only components are outside PowerModels' component list and
        # are therefore not connectivity-checked at all.
        @test common_validated(d -> d["heatpumps"]["1"]["hp_bus"] = 42
                              )["heatpumps"]["1"]["hp_bus"] == 42
        @test common_validated(d -> d["dsm"]["1"]["dsm_bus"] = 42
                              )["dsm"]["1"]["dsm_bus"] == 42
    end

    @testset "check_status" begin
        # An active component on a deactivated (`bus_type == 4`) bus only produces
        # a warning — the data is passed through unchanged.
        data = common_validated(d -> d["bus"]["2"]["bus_type"] = 4)
        @test data["bus"]["2"]["bus_type"] == 4
        @test data["gen"]["1"]["gen_status"] == 1
        @test data["gen"]["1"]["gen_bus"] == 2
    end

    @testset "check_storage_parameters" begin
        # Negative ratings/energies are rejected …
        for key in ("energy", "energy_rating", "charge_rating", "discharge_rating",
                    "charge_efficiency", "discharge_efficiency", "r", "x",
                    "thermal_rating")
            @test_throws ErrorException eDisGo_OPF.correct_network_data!(
                common_network(d -> d["storage"]["1"][key] = -1.0))
        end
        # … while an efficiency above 1.0 is merely warned about and kept.
        data = common_validated(d -> d["storage"]["1"]["charge_efficiency"] = 1.5)
        @test data["storage"]["1"]["charge_efficiency"] == 1.5
    end

    @testset "check_conductors" begin
        # The package is single-conductor only. A `conductors` entry below 1 is a
        # hard error; the key is otherwise not part of eDisGo data at all.
        @test !haskey(TestUtils.small_network(), "conductors")
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["conductors"] = 0))
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(d -> d["conductors"] = -1))
    end

    @testset "cost functions" begin
        # eDisGo generators carry no cost model, so both cost passes are no-ops on
        # the fixture …
        data = common_validated()
        @test !haskey(data["gen"]["1"], "cost")
        @test !haskey(data["gen"]["1"], "model")

        # … but if a polynomial cost model is present, leading zero terms are
        # trimmed and `ncost` is corrected to match (`simplify_cost_terms!`).
        trimmed = common_validated(function (d)
            d["gen"]["1"]["model"] = 2
            d["gen"]["1"]["ncost"] = 3
            d["gen"]["1"]["cost"] = [0.0, 0.0, 5.0]
        end)
        @test trimmed["gen"]["1"]["cost"] == [5.0]
        @test trimmed["gen"]["1"]["ncost"] == 1

        # A cost model whose `ncost` does not match the number of coefficients is
        # rejected (`correct_cost_functions!`).
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(
            common_network(function (d)
                d["gen"]["1"]["model"] = 2
                d["gen"]["1"]["ncost"] = 2
                d["gen"]["1"]["cost"] = [1.0, 2.0, 3.0]
            end))
    end

    @testset "checks eDisGo leaves out" begin
        # `check_reference_bus` and `correct_bus_types!` are commented out in
        # `correct_network_data!`, so — unlike PowerModels' namesake — the bus
        # types are never inspected or repaired. A grid without a slack bus passes
        # validation silently …
        data = common_validated(d -> d["bus"]["1"]["bus_type"] = 1)
        @test data["bus"]["1"]["bus_type"] == 1
        @test !any(bus["bus_type"] == 3 for bus in values(data["bus"]))

        # … and a PV bus without a generator keeps its type instead of being
        # demoted to PQ.
        pv = common_validated(d -> d["bus"]["4"]["bus_type"] = 2)
        @test pv["bus"]["4"]["bus_type"] == 2
        @test !any(gen["gen_bus"] == 4 for gen in values(pv["gen"]))
        # For contrast: the PowerModels check that eDisGo omits does demote it.
        pm_fixed = common_network(d -> d["bus"]["4"]["bus_type"] = 2)
        PowerModels.correct_bus_types!(pm_fixed)
        @test pm_fixed["bus"]["4"]["bus_type"] == 1
    end

    @testset "multinetwork data is not supported" begin
        # Validation happens on the single-network dict, before
        # `make_multinetwork`; handing it a multinetwork fails in
        # `correct_voltage_angle_differences!`.
        mn = TestUtils.multinetwork(TestUtils.small_network())
        @test InfrastructureModels.ismultinetwork(mn)
        @test_throws ErrorException eDisGo_OPF.correct_network_data!(mn)
    end

    @testset "real test grid" begin
        # The checked-in fixture is what `to_powermodels` actually produces:
        # validating it must not change a single value.
        raw = TestUtils.pm_network(validate=false)
        validated = deepcopy(raw)
        eDisGo_OPF.correct_network_data!(validated)
        @test validated == raw
        # Which is also what `parse_json(validate=true)` returns.
        @test validated == TestUtils.pm_network(validate=true)

        # Same for the overlying-grid variant, HV requirements included.
        og_raw = TestUtils.pm_network(overlying_grid=true, validate=false)
        og = deepcopy(og_raw)
        eDisGo_OPF.correct_network_data!(og)
        @test og == og_raw
        @test !isempty(og["HV_requirements"])
    end
end
