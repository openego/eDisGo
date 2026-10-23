# Testing conventions for `eDisGo_OPF.jl`

This file is binding for everyone who writes tests for the Julia package
`eDisGo_OPF`. Changing it changes the rules for every later contribution.

The goal is complete test coverage: **every** function and type of the package
gets tests. Progress is tracked in [`FUNCTION_COVERAGE.md`](FUNCTION_COVERAGE.md).

## 1. Directory layout

The test layout mirrors `src/` one to one:

| Source file                      | Test file                             |
| -------------------------------- | ------------------------------------- |
| `Main.jl`                        | `test/test_main.jl`                   |
| `src/core/base.jl`               | `test/core/test_base.jl`              |
| `src/core/constraint.jl`         | `test/core/test_constraint.jl`        |
| `src/core/constraint_template.jl`| `test/core/test_constraint_template.jl` |
| `src/core/data.jl`               | `test/core/test_data.jl`              |
| `src/core/objective.jl`          | `test/core/test_objective.jl`         |
| `src/core/solution.jl`           | `test/core/test_solution.jl`          |
| `src/core/types.jl`              | `test/core/test_types.jl`             |
| `src/core/variables.jl`          | `test/core/test_variables.jl`         |
| `src/form/bf.jl`                 | `test/form/test_bf.jl`                |
| `src/io/common.jl`               | `test/io/test_common.jl`              |
| `src/io/json.jl`                 | `test/io/test_json.jl`                |
| `src/prob/opf_bf.jl`             | `test/prob/test_opf_bf.jl`            |

In addition:

```
test/
├── runtests.jl             # entry point, list TEST_FILES, test selection
├── testsetup.jl            # common preamble of every test file
├── testutils/TestUtils.jl  # fixtures and helper functions
├── data/                   # checked-in JSON fixtures + generator script
├── test_smoke.jl           # load package, dependencies and fixtures
├── coverage_report.jl      # line coverage of src/ after a coverage run
├── TESTING_CONVENTIONS.md  # this file
└── FUNCTION_COVERAGE.md    # checklist of all functions
```

**New source file ⇒ new test file**: create the test file, add it to
`TEST_FILES` in `runtests.jl` and add its functions to `FUNCTION_COVERAGE.md`.

**Splitting very large test files**: when a test file becomes unwieldy (rule of
thumb: more than about 20 tested functions), it may be split into a subdirectory
of the same name, with the main file only including the parts. This is how
`src/core/variables.jl` (40 functions) is handled: `test/core/test_variables.jl`
includes `test/core/variables/test_*.jl`. The grouping follows the order of the
source file, not a scheme of its own.

## 2. Structure of a test file

Every test file runs on its own **and** can be included from `runtests.jl`. It
therefore always starts with the common preamble:

```julia
# Tests for `src/core/variables.jl`.

include(joinpath(@__DIR__, "..", "testsetup.jl"))   # without ".." for files in test/

@testset "variable_branch_power_real_radial" begin
    ...
end
```

Rules:

- **One `@testset` per function**, named exactly like the function. If there are
  several methods (e.g. `constraint_model_current` for SOC and NC), they are
  separated into nested `@testset`s named after the dispatch type:
  `@testset "AbstractSOCBFModelEdisgo" begin ... end`.
- The `@testset`s appear in the same order as the functions in the source file.
- Everything is written **in English**: code, comments, docstrings and the
  Markdown documentation (this file, `FUNCTION_COVERAGE.md`).

## 3. What a good test checks here

The package builds optimisation models; most functions return nothing but add
variables or constraints to a `JuMP.Model`. The tests therefore check the
observable effect:

1. **Variable functions** (`variable_*`)
   - The variable is stored under the expected key in `PowerModels.var(pm, nw)`.
   - The index set is right (which components/arcs are included and which are
     excluded — e.g. virtual storage branches are skipped on purpose).
   - Bounds: `JuMP.has_lower_bound` / `JuMP.lower_bound` etc., for both
     `bounded=true` and `bounded=false`.
   - Start values from `*_start` fields are applied.
   - `report=true` writes the entry to the solution structure, `report=false`
     does not.
2. **Constraint functions** (`constraint_*`)
   - The number of constraints grows by the expected amount
     (`TestUtils.num_constraints_total`).
   - The constraint has the expected terms/coefficients
     (`TestUtils.constraint_strings` or direct access via
     `PowerModels.con(pm, nw, :name)`).
   - Special cases: virtual storage branches, missing `time_elapsed`, component
     missing in the previous time step.
3. **Objective functions** (`objective_*`)
   - `JuMP.objective_sense` and the terms of `JuMP.objective_function`.
4. **Data functions** (`correct_*`, `set_*`, `parse_*`, `check_*`)
   - Compare the return value or the mutated dict exactly.
   - Check error cases with `@test_throws`.
   - Check logged warnings with `TestUtils.capture_warnings`.
5. **End to end** (`solve_*`, `build_*`, `Main.jl`)
   - Solve a small network with Ipopt and check the status and the order of
     magnitude of the solution. Don't pin exact solver numbers to many decimal
     places — use `atol`/`rtol` (rule of thumb `atol=1e-4` for p.u. quantities).

Principles:

- **No test without a claim.** `@test true` after a `using` is not a test — the
  deliberate smoke tests in `test_smoke.jl` are the exception.
- **Positive and negative case.** If a function filters components, check that
  the expected ones are in *and* the others are out.
- **No dependence on ordering.** Julia `Dict`s are unordered; always compare via
  `sort(...)` or as sets (`Set`).
- **Tests must not influence each other.** `TestUtils.pm_network()` and
  `TestUtils.small_network()` return fresh dicts on every call.

## 4. Fixtures

Two networks are available (see the docstrings in `testutils/TestUtils.jl`):

- `TestUtils.small_network(; opf_version, num_steps, overlying_grid)` — a
  hand-written 4-bus network with exactly one component of every flexibility
  type. **First choice for unit tests**: fast, easy to follow, easy to
  manipulate.
- `TestUtils.pm_network(; overlying_grid, opf_version, validate)` — the real
  eDisGo test grid (143 buses, 4 time steps) from
  `tests/data/ding0_test_network_1`. **First choice for integration tests** and
  wherever realistic data is needed.

Further helpers:

- `TestUtils.multinetwork(data)` — single network → multi-network (as in
  `Main.jl`).
- `TestUtils.build_model(...)` / `TestUtils.build_small_model(...)` —
  instantiate a model without solving it.
- `TestUtils.capture_warnings(f)` — run `f` with the PowerModels logger raised to
  `"warn"` and return its result together with the logged warnings.
- `TestUtils.gurobi_available()` — Gurobi is commercial and not available in CI.
  Tests that need Gurobi **must** be guarded with it:

  ```julia
  if TestUtils.gurobi_available()
      ...
  else
      @test_skip false
  end
  ```

  Use Ipopt for everything else — it is freely available and covers the
  non-convex formulation.

The JSON fixtures in `test/data/` are generated by
`test/data/generate_fixtures.py`. If the input schema on the Python side
(`edisgo/io/powermodels_io.to_powermodels`) changes, the fixtures have to be
regenerated and committed together with the code change.

## 5. Running the tests

From the directory `edisgo/opf/eDisGo_OPF.jl`:

```bash
# full suite
julia --project=. -e 'using Pkg; Pkg.test()'

# single file (fast, without the Pkg.test overhead)
julia --project=. test/core/test_variables.jl

# subset via runtests.jl
julia --project=. test/runtests.jl variables objective
EDISGO_OPF_TESTS=core julia --project=. -e 'using Pkg; Pkg.test()'

# with coverage, then the line coverage report of src/
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'
julia test/coverage_report.jl
```

**Line coverage of `src/` must stay at 100 %.** CI runs `test/coverage_report.jl`
after the suite; it lists every uncovered line and fails the job below 100 %. The
Julia coverage is not sent to Coveralls, which only holds the Python coverage.

The test dependencies are deliberately listed in `[extras]`/`[targets]` of
`Project.toml` and **not** in a separate `test/Project.toml`: that way the suite
runs against the checked-in `Manifest.toml`, i.e. against exactly the package
versions used in production.

## 6. Workflow and commits

Per function:

1. Pick the function in `FUNCTION_COVERAGE.md` (top to bottom).
2. Read the source of the function — including the places that call it.
3. Add the `@testset` to the matching test file (in source order).
4. Run the test file on its own, then the full suite.
5. Tick the box in `FUNCTION_COVERAGE.md`.
6. **One commit per ticked function.** Commit message:

   ```
   test(opf-julia): add tests for <function name>
   ```

   If several methods of one function are tested together, one commit for all
   methods of that function is enough.

   **Commit messages in English**, like the rest of the repository history.

If you find a **bug in production code** while writing tests: write the test so
that it documents the actual behaviour, mark the intended behaviour with
`@test_broken`, and record the finding in the commit message and in
`FUNCTION_COVERAGE.md`. Don't change production code in a test commit. A fix is a
commit of its own (`fix(opf-julia): …`) that also turns the affected
`@test_broken` into regular tests and marks the defect as fixed in
`FUNCTION_COVERAGE.md`.
