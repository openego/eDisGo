# Tests for `Main.jl`.
#
# `Main.jl` is not part of the package module but the command-line script the
# Python side launches as a Julia subprocess (`_pm_optimize_single` in
# `edisgo/opf/powermodels_opf.py`). Its only function, `optimize_edisgo`, reads
# globals the script sets up from its environment, so it is tested exactly the
# way Python uses it:
#
#   * spawn `julia Main.jl <ding0_grid> <results_path> <method> <silence_moi>
#     <warm_start> <threads>` (with the flags of the running Julia, `Base.julia_cmd()`, so
#     the subprocess reuses the precompiled package images),
#   * write the network as one JSON line to its stdin and close it,
#   * pick the result out of stdout as Python does — the line starting with
#     `{"name"` — and check status, solver and the plausibility of the schedules,
#     including every value `from_powermodels` reads back.
#
# The input is the real eDisGo test grid (`test/data/pm_test_network.json`,
# `opf_version` 2, 4 time steps), i.e. what `to_powermodels` produces. Every run
# loads the whole package in a fresh process, so the number of runs is kept
# minimal: one non-convex Ipopt run (`method="nc"`), which always runs (CI), and
# three SOC runs with Gurobi (`method="soc"`, without warm start, and with warm
# start for `opf_version` 2 and 1), which need a Gurobi licence and are skipped
# without one (TESTING_CONVENTIONS.md, section 4).

include(joinpath(@__DIR__, "testsetup.jl"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"Absolute path of the script under test."
const MAIN_SCRIPT = normpath(joinpath(@__DIR__, "..", "Main.jl"))

"""
The network `fixture` as the single JSON line Python writes to the subprocess
(`json.dumps(pm)`): the checked-in fixture is pretty-printed, so it is re-encoded.
"""
main_json_line(fixture::AbstractString="pm_test_network.json") =
    JSON.json(JSON.parse(read(TestUtils.fixture_path(fixture), String)))

"""
    main_run(json_line, method; ding0_grid, silence_moi="True", warm_start="False",
             threads="8")

Run `Main.jl` as a subprocess the way `powermodels_opf.py` does and return a
named tuple with `exitcode`, `stdout`, `stderr`, the `results_path` handed to the
script (a fresh temporary directory) and the wall time in seconds.

`JULIA_LOAD_PATH` and `JULIA_PROJECT` are removed from the environment: inside
`Pkg.test()` they point to the temporary test environment, while Python starts
`julia` with the default load path and leaves it to `Main.jl` to activate its own
project.
"""
function main_run(json_line::AbstractString, method::AbstractString;
                  ding0_grid::AbstractString="ding0_1_t_4",
                  silence_moi::AbstractString="True",
                  warm_start::AbstractString="False",
                  threads::AbstractString="8")
    dir = mktempdir()
    results_path = mkdir(joinpath(dir, "results"))
    stdin_file = joinpath(dir, "stdin.json")
    stdout_file = joinpath(dir, "stdout.txt")
    stderr_file = joinpath(dir, "stderr.txt")
    # Python writes the JSON without a trailing newline and closes stdin.
    write(stdin_file, json_line)

    env = copy(ENV)
    delete!(env, "JULIA_LOAD_PATH")
    delete!(env, "JULIA_PROJECT")
    cmd = `$(Base.julia_cmd()) $MAIN_SCRIPT $ding0_grid $results_path $method $silence_moi $warm_start $threads`
    cmd = setenv(ignorestatus(cmd), env)

    elapsed = @elapsed proc = run(pipeline(cmd; stdin=stdin_file, stdout=stdout_file,
                                           stderr=stderr_file))
    return (exitcode=proc.exitcode, stdout=read(stdout_file, String),
            stderr=read(stderr_file, String), results_path=results_path,
            elapsed=elapsed)
end

"Lines of `stdout` Python recognises as the result: `rstrip(line)` starts with `{\"name\"`."
main_result_lines(stdout::AbstractString) =
    [l for l in split(stdout, '\n') if startswith(rstrip(l), "{\"name\"")]

"""
Parse the result of a run: the line Python would pick, or — so the remaining
checks still say something if that contract is broken — the first JSON object
line. `nothing` if there is none.
"""
function main_parse_result(stdout::AbstractString)
    lines = main_result_lines(stdout)
    if isempty(lines)
        lines = [l for l in split(stdout, '\n') if startswith(lstrip(l), "{")]
    end
    return isempty(lines) ? nothing : JSON.parse(first(lines))
end

"""
`(component, variable)` pairs `from_powermodels` reads from every network of the
result, for the given `opf_version` and list of `flexibilities` (the battery's
schedule is read from its virtual branch, hence `branch`/`pf`).
"""
function main_python_reads(opf_version::Int, flexibilities)
    reads = Set([("gen_slack", "pgs"), ("gen_slack", "qgs"),
                 ("storage", "ps"), ("storage", "se"),
                 ("heat_storage", "phs"), ("heat_storage", "hse"), ("heat_storage", "phss"),
                 ("branch", "pf"), ("branch", "qf"), ("branch", "ccm")])
    flex_vars = Dict("curt" => ("gen_nd", "pgc"), "hp" => ("heatpumps", "php"),
                     "cp" => ("electromobility", "pcp"), "storage" => ("branch", "pf"),
                     "dsm" => ("dsm", "pdsm"))
    for flex in flexibilities
        push!(reads, flex_vars[flex])
    end
    if opf_version in (2, 4)
        union!(reads, [("gen", "pgens"), ("gen_nd", "pgc"), ("load", "pds"),
                       ("electromobility", "pcps"), ("heatpumps", "phps"),
                       ("heatpumps", "phps2")])
    end
    opf_version in (3, 4) && push!(reads, ("HV_requirements", "phvs"))
    return reads
end

"""
Checks shared by every successful run: the result is the optimised multi-network
of the input grid and carries every value the Python side reads back.
"""
function main_check_result(out, input)
    # `make_multinetwork` renames the network after the number of time steps.
    @test out["name"] == "4 replicates of " * input["name"]
    @test out["multinetwork"] == true
    @test sort(collect(keys(out["nw"]))) == ["1", "2", "3", "4"]
    @test out["solve_time"] isa Real
    @test 0 < out["solve_time"] < 3600
    nw1 = out["nw"]["1"]
    @test nw1["opf_version"] == input["opf_version"]
    @test nw1["flexibilities"] == input["flexibilities"]
    # `update_data!` merges the solution into the input: the grid is all there.
    for comp in ("bus", "branch", "gen_nd", "load", "storage", "heatpumps")
        @test length(nw1[comp]) == length(input[comp])
    end
    # Every value `from_powermodels` reads exists and is a finite number.
    reads = main_python_reads(nw1["opf_version"], nw1["flexibilities"])
    missing_values = [(t, comp, var) for t in keys(out["nw"]) for (comp, var) in reads
                      for c in values(out["nw"][t][comp])
                      if !(get(c, var, nothing) isa Real && isfinite(c[var]))]
    @test isempty(missing_values)
end

"""
Run `Main.jl` with `method="soc"` and `warm_start="True"` on `json_line` and check
the result. If the SOC relaxation is tight, the Gurobi result is polished by a
warm-started Ipopt run, which then provides the result. Otherwise the warm start
is skipped and the Gurobi result is returned.
"""
function main_check_warm_start(json_line, input)
    run_ws = main_run(json_line, "soc"; warm_start="True")
    @info "Main.jl subprocess, method soc + warm start" opf_version = input["opf_version"] elapsed_s = round(run_ws.elapsed, digits=1)
    if run_ws.exitcode != 0
        println(stderr, "Main.jl stderr:\n", run_ws.stderr)
    end
    @test run_ws.exitcode == 0
    @test occursin("Starting convex SOC AC-OPF with Gurobi.", run_ws.stdout)
    @test length(main_result_lines(run_ws.stdout)) == 1
    out = main_parse_result(run_ws.stdout)
    @test out isa Dict
    out isa Dict || return
    main_check_result(out, input)
    if isempty(readdir(run_ws.results_path))
        # Tight relaxation: the Gurobi result is polished by a warm-started
        # Ipopt run.
        @test occursin("Starting warm-start non-convex AC-OPF with IPOPT.", run_ws.stdout)
        @test out["solver"] == "Ipopt"
        @test out["status"] == "LOCALLY_SOLVED"
    else
        # Not tight: no warm start, the Gurobi result is returned.
        @test occursin("SOC solution is not tight!", run_ws.stdout)
        @test !occursin("Starting warm-start", run_ws.stdout)
        @test out["solver"] == "Gurobi"
    end
end

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@testset "optimize_edisgo" begin
    json_line = main_json_line()
    @test !occursin('\n', json_line)
    input = JSON.parse(json_line)
    @test input["opf_version"] == 2

    @testset "method nc (Ipopt)" begin
        run_nc = main_run(json_line, "nc")
        @info "Main.jl subprocess, method nc" elapsed_s = round(run_nc.elapsed, digits=1)
        if run_nc.exitcode != 0
            println(stderr, "Main.jl stderr:\n", run_nc.stderr)
        end
        @test run_nc.exitcode == 0
        @test occursin("Starting cold-start non-convex AC-OPF with IPOPT.", run_nc.stdout)
        # Python takes the result from the one line starting with `{"name"`.
        @test length(main_result_lines(run_nc.stdout)) == 1
        out = main_parse_result(run_nc.stdout)
        @test out isa Dict
        if out isa Dict
            @test out["status"] == "LOCALLY_SOLVED"
            @test out["solver"] == "Ipopt"
            main_check_result(out, input)

            tol = 1e-5
            for (t, nw) in out["nw"]
                # Version 2 keeps the voltage band; the slack bus is fixed to 1 p.u.
                w = Dict(i => b["w"] for (i, b) in nw["bus"] if haskey(b, "w"))
                @test !isempty(w)
                @test all(0.81 - tol <= x <= 1.21 + tol for x in values(w))
                slack = [i for (i, b) in nw["bus"] if b["bus_type"] == 3]
                @test length(slack) == 1
                @test w[only(slack)] ≈ 1.0 atol = tol
                # Schedules stay within the limits of their components.
                @test all(-tol <= g["pgc"] <= g["pg"] + tol for g in values(nw["gen_nd"]))
                @test all(h["p_min"] - tol <= h["php"] <= h["p_max"] + tol
                          for h in values(nw["heatpumps"]))
                @test all(c["p_min"] - tol <= c["pcp"] <= c["p_max"] + tol
                          for c in values(nw["electromobility"]))
                @test all(s["energy_rating"] + tol >= s["se"] >= -tol
                          for s in values(nw["storage"]))
                @test all(b["ccm"] >= -tol for b in values(nw["branch"]))
            end
            # The battery ends the horizon at its required state of charge.
            @test all(isapprox(s["se"], s["soc_end"]; atol=1e-4)
                      for s in values(out["nw"]["4"]["storage"]))
        end
        # Only a non-tight SOC solution is written to `results_path`.
        @test isempty(readdir(run_nc.results_path))
    end

    @testset "method soc (Gurobi)" begin
        if TestUtils.gurobi_available()
            # Python's default: SOC with Gurobi, no warm start.
            run_soc = main_run(json_line, "soc"; warm_start="False")
            @info "Main.jl subprocess, method soc" elapsed_s = round(run_soc.elapsed, digits=1)
            if run_soc.exitcode != 0
                println(stderr, "Main.jl stderr:\n", run_soc.stderr)
            end
            @test run_soc.exitcode == 0
            @test occursin("Starting convex SOC AC-OPF with Gurobi.", run_soc.stdout)
            @test !occursin("Starting warm-start", run_soc.stdout)
            @test length(main_result_lines(run_soc.stdout)) == 1
            out = main_parse_result(run_soc.stdout)
            @test out isa Dict
            if out isa Dict
                main_check_result(out, input)
                @test out["solver"] == "Gurobi"
                @test out["status"] in ("OPTIMAL", "ALMOST_OPTIMAL", "SUBOPTIMAL")
                files = readdir(run_soc.results_path)
                if isempty(files)
                    @test !occursin("SOC solution is not tight!", run_soc.stdout)
                else
                    # Not tight: the violations are saved under
                    # `<ding0_grid>_<flexibilities>.json`.
                    @test files == [input["name"] * "_" * join(input["flexibilities"]) * ".json"]
                    @test JSON.parsefile(joinpath(run_soc.results_path, only(files))) isa Dict
                    @test occursin("SOC solution is not tight!", run_soc.stdout)
                end
            end
        else
            @test_skip false
        end
    end

    @testset "method soc (Gurobi) with warm start" begin
        if TestUtils.gurobi_available()
            main_check_warm_start(json_line, input)
        else
            @test_skip false
        end
    end

    @testset "method soc (Gurobi) with warm start, opf_version 1" begin
        if TestUtils.gurobi_available()
            # Version 1 is the default of `EDisGo.pm_optimize`. It has no
            # curtailment `pgc`, so Python leaves "curt" out of the flexibilities.
            input_v1 = deepcopy(input)
            input_v1["opf_version"] = 1
            filter!(!=("curt"), input_v1["flexibilities"])
            main_check_warm_start(JSON.json(input_v1), input_v1)
        else
            @test_skip false
        end
    end
end
