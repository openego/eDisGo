# Line coverage report for `src/`, built from the `.cov` files that
# `Pkg.test(coverage=true)` leaves behind. Run from `edisgo/opf/eDisGo_OPF.jl`
# after the test suite:
#
#     julia test/coverage_report.jl [min_percent]
#
# Prints the coverage per file and every uncovered line, appends the same report
# to the GitHub Actions job summary when run in CI, and exits with status 1 if the
# total is below `min_percent` (default 100). CoverageTools is installed into a
# temporary environment, so it does not become a dependency of the package.

using Pkg
Pkg.activate(; temp=true, io=devnull)
Pkg.add(name="CoverageTools", version="1"; io=devnull)

using CoverageTools
using Logging
using Printf

const MIN_PERCENT = isempty(ARGS) ? 100.0 : parse(Float64, ARGS[1])

"Coverage of `src/`, without CoverageTools' per-file progress messages."
function src_coverage()
    with_logger(SimpleLogger(stderr, Logging.Warn)) do
        process_folder("src")
    end
end

percent(covered, total) = total == 0 ? 100.0 : 100 * covered / total

function report(coverage)
    covered, total = get_summary(coverage)
    lines = String[]
    push!(lines, @sprintf("## Julia OPF line coverage: %.1f %% (%d of %d lines)",
                          percent(covered, total), covered, total), "")
    push!(lines, "| File | Covered | Lines | % |", "| --- | ---: | ---: | ---: |")
    for fc in sort(coverage; by=fc -> fc.filename)
        c, t = get_summary(fc)
        t == 0 && continue
        push!(lines, @sprintf("| `%s` | %d | %d | %.1f |", fc.filename, c, t, percent(c, t)))
    end
    uncovered = ["$(fc.filename):$i" for fc in coverage
                 for (i, hits) in enumerate(fc.coverage) if hits == 0]
    if !isempty(uncovered)
        push!(lines, "", "Uncovered lines:", "")
        append!(lines, ["- `$u`" for u in uncovered])
    end
    return join(lines, "\n") * "\n", covered, total
end

text, covered, total = report(src_coverage())
print(text)
summary = get(ENV, "GITHUB_STEP_SUMMARY", "")
isempty(summary) || open(io -> print(io, text), summary, "a")

if percent(covered, total) < MIN_PERCENT
    @error @sprintf("Line coverage %.1f %% is below the required %.1f %%",
                    percent(covered, total), MIN_PERCENT)
    exit(1)
end
