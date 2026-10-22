# Entry point of the eDisGo_OPF test suite — run with
#
#     julia --project=. -e 'using Pkg; Pkg.test()'
#
# from the `eDisGo_OPF.jl` directory, or run a single file directly:
#
#     julia --project=. test/core/test_variables.jl
#
# A subset of the suite can be selected by name, either as a command line argument
# or via the EDISGO_OPF_TESTS environment variable (comma-separated), matched as a
# substring of the paths listed in TEST_FILES below:
#
#     julia --project=. test/runtests.jl variables objective
#     EDISGO_OPF_TESTS=core julia --project=. -e 'using Pkg; Pkg.test()'
#
# See test/TESTING_CONVENTIONS.md for the conventions this suite follows.

using Test

include(joinpath(@__DIR__, "testsetup.jl"))

"""
Test files in the order they are run. The layout mirrors `src/`: a source file
`src/<dir>/<name>.jl` is tested by `test/<dir>/test_<name>.jl`.
"""
const TEST_FILES = [
    "test_smoke.jl",
    "core/test_types.jl",
    "core/test_base.jl",
    "core/test_data.jl",
    "core/test_solution.jl",
    "core/test_variables.jl",
    "core/test_constraint.jl",
    "core/test_constraint_template.jl",
    "core/test_objective.jl",
    "form/test_bf.jl",
    "io/test_json.jl",
    "io/test_common.jl",
    "prob/test_opf_bf.jl",
    "test_main.jl",
]

"""
Return the subset of `TEST_FILES` selected by the command line arguments or by
`ENV["EDISGO_OPF_TESTS"]`; all of them when no selection is given.
"""
function selected_test_files()
    patterns = isempty(ARGS) ? split(get(ENV, "EDISGO_OPF_TESTS", ""), ",", keepempty=false) : ARGS
    isempty(patterns) && return TEST_FILES
    selected = [f for f in TEST_FILES if any(occursin(strip(p), f) for p in patterns)]
    isempty(selected) && error("no test file matches $(patterns); available: $(TEST_FILES)")
    return selected
end

@testset verbose = true "eDisGo_OPF" begin
    for file in selected_test_files()
        @testset verbose = true "$file" begin
            include(joinpath(@__DIR__, file))
        end
    end
end
