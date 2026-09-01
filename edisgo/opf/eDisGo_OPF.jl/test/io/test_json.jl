# Tests for `src/io/json.jl`.
#
# `parse_json` is the entry point of the whole package: `Main.jl` reads the JSON
# that the Python side writes to stdin and turns it into the network dict the OPF
# is built from. It is a thin wrapper — `JSON.parse`, PowerModels'
# `_jsonver2juliaver!`, a warning for multi-conductor data and, unless switched
# off, `correct_network_data!` — so the tests cover the wrapper's contract: both
# accepted input types, the return type, the version conversion, and the effect of
# the `validate` keyword. They also pin down a bug in the `conductors` branch (see
# the corresponding testset).

include(joinpath(@__DIR__, "..", "testsetup.jl"))

"JSON text of a fresh `TestUtils.small_network()` with `mutate!` applied to it."
function json_text(mutate!::Function=identity; kwargs...)
    data = TestUtils.small_network(; kwargs...)
    mutate!(data)
    return JSON.json(data)
end

@testset "parse_json" begin

    @testset "input types" begin
        text = json_text()
        from_string = eDisGo_OPF.parse_json(text)
        # An `IO` is accepted just as well — this is the form `Main.jl` uses, which
        # reads the network from stdin.
        from_io = eDisGo_OPF.parse_json(IOBuffer(text))
        @test from_string == from_io
        # And so is a real file handle.
        from_file = open(io -> eDisGo_OPF.parse_json(io),
                         TestUtils.fixture_path("pm_test_network.json"))
        @test from_file == TestUtils.pm_network()

        # The stream is consumed exactly once; a second parse of the same buffer
        # sees an empty stream and fails.
        buffer = IOBuffer(text)
        eDisGo_OPF.parse_json(buffer)
        @test eof(buffer)
        @test_throws Exception eDisGo_OPF.parse_json(buffer)
    end

    @testset "return type" begin
        data = eDisGo_OPF.parse_json(json_text())
        # The signature declares `::Dict{String,Any}` — not just any `AbstractDict`.
        @test typeof(data) === Dict{String,Any}
        # Nested component dicts have the same type, which is what the `ref`
        # building and the `Dict{String,<:Any}` signatures downstream expect.
        for key in ("bus", "branch", "gen", "gen_nd", "load", "storage",
                    "heatpumps", "electromobility", "dsm", "time_series")
            @test typeof(data[key]) === Dict{String,Any}
        end
        @test typeof(data["branch"]["1"]) === Dict{String,Any}
        # JSON arrays become vectors, and numbers keep their JSON-side kind.
        @test data["time_series"]["gen"]["1"]["pg"] isa AbstractVector
        @test data["flexibilities"] isa AbstractVector
        @test data["opf_version"] isa Integer
        @test data["branch"]["1"]["br_r"] isa AbstractFloat
        @test data["bus"]["1"]["storage"] === false
    end

    @testset "round trip" begin
        # Serialising a network dict and parsing it back must reproduce it exactly
        # — the fixtures in `test/data/` and the Python side rely on it.
        network = TestUtils.small_network(opf_version=4, num_steps=3,
                                          overlying_grid=true)
        @test eDisGo_OPF.parse_json(JSON.json(network); validate=false) == network
        @test eDisGo_OPF.parse_json(JSON.json(network)) == network
    end

    @testset "validate" begin
        # `rate_a = 0.0` is one of the values `correct_network_data!` acts on: it
        # deletes the key. That makes the difference between the two modes
        # directly observable.
        text = json_text(d -> d["branch"]["1"]["rate_a"] = 0.0)

        validated = eDisGo_OPF.parse_json(text; validate=true)
        @test !haskey(validated["branch"]["1"], "rate_a")

        unvalidated = eDisGo_OPF.parse_json(text; validate=false)
        @test unvalidated["branch"]["1"]["rate_a"] == 0.0

        # Validation is on by default …
        @test eDisGo_OPF.parse_json(text) == validated
        @test eDisGo_OPF.parse_json(text) != unvalidated
        # … and the two differ *only* in what the validation corrected.
        patched = deepcopy(unvalidated)
        patched["branch"]["1"] = validated["branch"]["1"]
        @test patched == validated

        # A second correction on the same input: whatever `validate=false` returns
        # becomes the validated result once `correct_network_data!` is applied.
        again = eDisGo_OPF.parse_json(text; validate=false)
        eDisGo_OPF.correct_network_data!(again)
        @test again == validated

        # Degenerate angle limits are the other direction — validation *adds*
        # information instead of removing it.
        angles = json_text(function (d)
            d["branch"]["2"]["angmin"] = 0.0
            d["branch"]["2"]["angmax"] = 0.0
        end)
        @test eDisGo_OPF.parse_json(angles)["branch"]["2"]["angmax"] == 1.0472
        @test eDisGo_OPF.parse_json(angles; validate=false)["branch"]["2"]["angmax"] == 0.0

        # Broken data is rejected with validation on and passed through with it
        # off — `validate=false` is the only way to get at the raw input.
        broken = json_text(d -> d["load"]["1"]["load_bus"] = 42)
        @test_throws ErrorException eDisGo_OPF.parse_json(broken)
        @test eDisGo_OPF.parse_json(broken; validate=false)["load"]["1"]["load_bus"] == 42

        # The keyword is read out of `kwargs` with `get`, so unrelated keywords are
        # accepted and ignored, and `validate` still defaults to `true`.
        @test eDisGo_OPF.parse_json(text; import_all=true) == validated
        @test eDisGo_OPF.parse_json(text; import_all=true, validate=false) == unvalidated
    end

    @testset "_jsonver2juliaver!" begin
        # PowerModels writes `source_version` as a semver *object*; it is folded
        # back into the "major.minor.patch" string form on the way in.
        data = eDisGo_OPF.parse_json(json_text(
            d -> d["source_version"] = Dict("major" => 1, "minor" => 2,
                                            "patch" => 3)); validate=false)
        @test data["source_version"] == "1.2.3"
        @test data["source_version"] isa String

        # Anything that is not a dict — such as the plain integer eDisGo sends — is
        # left exactly as it is.
        @test eDisGo_OPF.parse_json(json_text(); validate=false)["source_version"] == 2
        @test TestUtils.pm_network()["source_version"] == 2
        @test eDisGo_OPF.parse_json(json_text(d -> d["source_version"] = "0.9.1");
                                    validate=false)["source_version"] == "0.9.1"
        # A missing entry is not invented either.
        @test !haskey(eDisGo_OPF.parse_json(json_text(d -> delete!(d, "source_version"));
                                            validate=false), "source_version")
    end

    @testset "conductors" begin
        # Multi-conductor data is not supported and is supposed to produce a
        # warning here. It does not: the logger `_LOGGER` referenced by the
        # `Memento.warn` call is never defined in `eDisGo_OPF` (it is an internal
        # of PowerModels and therefore not brought in by `using PowerModels`), so
        # the branch raises `UndefVarError` instead of warning. Any input
        # containing the key — with or without validation — is thus rejected
        # outright.
        text = json_text(d -> d["conductors"] = 1)
        @test_throws UndefVarError eDisGo_OPF.parse_json(text; validate=false)
        @test_throws UndefVarError eDisGo_OPF.parse_json(text)
        err = try
            eDisGo_OPF.parse_json(text; validate=false)
        catch e
            e
        end
        @test err.var === :_LOGGER
        @test !isdefined(eDisGo_OPF, :_LOGGER)

        # What *should* happen: a warning, and the parsed data handed back with the
        # key intact. Marked broken until the missing logger is fixed in `src/`.
        @test_broken (parsed = eDisGo_OPF.parse_json(text; validate=false);
                      parsed["conductors"] == 1)

        # Without the key the same input parses fine, so the key really is what
        # trips the call.
        @test eDisGo_OPF.parse_json(json_text(); validate=false) isa Dict{String,Any}
    end

    @testset "invalid input" begin
        # Malformed JSON comes back as a parser error rather than silently
        # producing an empty network.
        @test_throws Exception eDisGo_OPF.parse_json("{not json"; validate=false)
        @test_throws Exception eDisGo_OPF.parse_json(""; validate=false)
        @test_throws Exception eDisGo_OPF.parse_json("{\"bus\": }"; validate=false)
        # Valid JSON that is not an object cannot be a network dict either.
        @test_throws Exception eDisGo_OPF.parse_json("[1, 2, 3]"; validate=false)
        @test_throws Exception eDisGo_OPF.parse_json("42"; validate=false)
        # An object that is not a network fails in the validation, not in parsing.
        @test eDisGo_OPF.parse_json("{}"; validate=false) == Dict{String,Any}()
        @test_throws Exception eDisGo_OPF.parse_json("{}")
    end

    @testset "real test grid" begin
        # The fixtures are read through `parse_json`, so this is the path the whole
        # suite depends on.
        data = eDisGo_OPF.parse_json(read(
            TestUtils.fixture_path("pm_test_network.json"), String))
        @test length(data["bus"]) == 143
        @test length(data["branch"]) == 142
        @test data["per_unit"] == true
        @test data["time_series"]["num_steps"] == 4
        @test data["baseMVA"] == 1
        # Validation leaves the real data untouched, so both modes agree here.
        @test data == eDisGo_OPF.parse_json(read(
            TestUtils.fixture_path("pm_test_network.json"), String); validate=false)

        # Each call returns an independent dict; mutating one must not affect the
        # next parse (the fixtures are shared by every test file).
        data["bus"]["1"]["vmax"] = 42.0
        @test TestUtils.pm_network()["bus"]["1"]["vmax"] != 42.0
    end
end
