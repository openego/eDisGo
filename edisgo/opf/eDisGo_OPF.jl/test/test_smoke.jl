# Smoke tests: the package and every one of its dependencies must load, and the
# checked-in fixtures must be readable. If anything here fails, the environment is
# broken and the rest of the suite cannot be trusted.

include(joinpath(@__DIR__, "testsetup.jl"))

@testset "package loads" begin
    @test isdefined(Main, :eDisGo_OPF)
    # The three model types are the package's public API.
    @test isdefined(eDisGo_OPF, :BFPowerModelEdisgo)
    @test isdefined(eDisGo_OPF, :SOCBFPowerModelEdisgo)
    @test isdefined(eDisGo_OPF, :NCBFPowerModelEdisgo)
end

@testset "dependencies load" begin
    # Main.jl loads these in its own `using` block; a missing one only surfaces at
    # OPF runtime otherwise.
    for dep in (:Compat, :InfrastructureModels, :Ipopt, :JSON, :JuMP, :Memento,
                :PowerModels, :Gurobi)
        @eval using $dep
        @test isdefined(Main, dep)
    end
end

@testset "Gurobi" begin
    # Gurobi is commercial and needs a licence, so its absence must not fail the
    # suite — but it is worth reporting, since method="soc" cannot run without it.
    if TestUtils.gurobi_available()
        @test TestUtils.gurobi_available()
    else
        @info "Gurobi is not available (no licence?) — SOC tests will be skipped"
        @test_skip TestUtils.gurobi_available()
    end
end

@testset "fixtures" begin
    for name in ("pm_test_network.json", "pm_test_network_og.json")
        @test isfile(TestUtils.fixture_path(name))
    end

    data = TestUtils.pm_network()
    @test data isa Dict{String,Any}
    @test data["opf_version"] == 2
    @test length(data["bus"]) == 143
    @test isempty(data["HV_requirements"])

    data_og = TestUtils.pm_network(overlying_grid=true)
    @test data_og["opf_version"] == 4
    @test length(data_og["HV_requirements"]) == 5

    small = TestUtils.small_network()
    @test length(small["bus"]) == 4
    @test length(small["branch"]) == 3
end
