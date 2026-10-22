# Notes for coding agents

AI coding tools read this file automatically (the `AGENTS.md` convention).

## Julia OPF package (`edisgo/opf/eDisGo_OPF.jl`)

For **all** work on the test suite of the Julia package, these files are binding:

- [`edisgo/opf/eDisGo_OPF.jl/test/TESTING_CONVENTIONS.md`](edisgo/opf/eDisGo_OPF.jl/test/TESTING_CONVENTIONS.md)
  — layout, structure of a test file, fixtures, running the tests, commit
  conventions.
- [`edisgo/opf/eDisGo_OPF.jl/test/FUNCTION_COVERAGE.md`](edisgo/opf/eDisGo_OPF.jl/test/FUNCTION_COVERAGE.md)
  — checklist of all functions and the production defects found so far. The goal
  is complete coverage; one commit per ticked function.

In short: test files mirror `src/`, one `@testset` per function, everything in
English. Run the tests with `julia --project=. -e 'using Pkg; Pkg.test()'` from
`edisgo/opf/eDisGo_OPF.jl`.

## Commits

Commits and pull request descriptions carry only the author from the git
configuration. **No** `Co-Authored-By:` or `Claude-Session:` trailers, no
"Generated with" note, no mention of an AI tool — neither in the subject nor in
the body.

## Language

Code, docstrings, code comments, commit messages and Markdown documentation:
**English**.
