# Function coverage of `eDisGo_OPF.jl`

Checklist of all functions and types of the package. An entry is ticked only
once there is a meaningful `@testset` for it (see
[`TESTING_CONVENTIONS.md`](TESTING_CONVENTIONS.md), section 3) and the full suite
passes.

The suffix after the name is the dispatch type of the first argument — it
distinguishes functions with several methods (e.g. `constraint_model_current`
and `constraint_store_state`).

Legend: `[ ]` open · `[x]` tested · `[~]` partly tested (give the reason after
the entry)

## Status (as of 2026-10-23)

**84 of 84 entries done**, suite green: 3620 passing assertions, 6
`@test_broken`, 0 failures, about 6 to 7 min with a Gurobi licence (about 4 min of
that are the `Main.jl` subprocesses; without Gurobi, as in CI, the two SOC runs
are skipped). Line coverage of `src/`: 902 of 902 lines (100 %).

| Source file | Status |
| --- | --- |
| `src/core/types.jl` | done (6/6) |
| `src/core/base.jl` | done (3/3) |
| `src/core/data.jl` | done (3/3) |
| `src/core/solution.jl` | done (2/2) |
| `src/core/constraint.jl` | done (6/6) |
| `src/core/constraint_template.jl` | done (5/5) |
| `src/core/objective.jl` | done (5/5) |
| `src/io/json.jl` | done (1/1) |
| `src/io/common.jl` | done (1/1) |
| `src/core/variables.jl` | done (40/40) — split into `test/core/variables/test_flow_and_curtailment.jl` (16), `test_flexibilities.jl` (10), `test_slacks.jl` (14) |
| `src/form/bf.jl` | done (9/9) |
| `src/prob/opf_bf.jl` | done (2/2) |
| `Main.jl` | done (1/1) |

The only open item is cleanup: several test files have similar local helpers
(an empty model without build content, reading bounds, checking `report` without
side effects on `pm.sol`, capturing stdout) that belong in `TestUtils`.

## Defects found in production code

Found while writing the tests. Unless marked as fixed, they are **deliberately
not fixed** here (a fix is a commit of its own, see `TESTING_CONVENTIONS.md`,
section 6); the tests record the actual behaviour and mark the intended one with
`@test_broken`. Every open defect has a GitHub issue; the overview is #810.

1. ~~**`_LOGGER` is not defined in the module `eDisGo_OPF`**~~ — **fixed**: the
   module now defines `const _LOGGER = PowerModels._LOGGER`. Before, all 13
   `Memento.warn`/`Memento.error` calls raised `UndefVarError(:_LOGGER)` instead
   of warning, among them all `time_elapsed` fallbacks. The tests now check the
   warnings with `TestUtils.capture_warnings`.
2. ~~**`_biggest_generator` is not visible either**~~ — **fixed**:
   `_correct_bus_types!` calls `PowerModels._biggest_generator`. Remaining issue:
   if `gen` is empty, PowerModels already throws its own error ("generator list
   passed to _biggest_generator was empty. please report this bug."); the
   package's own message "no generators found" only appears when all generators
   are switched off. When promoting a reference bus, the fallback also only
   looks at `gen`, not at `gen_slack`, where eDisGo puts the slack generator.
   Still without effect in production, because `correct_network_data!` has the
   call commented out (defect 9). Tracked in #806.
3. ~~**`ref_add_core!` throws `KeyError` when a deactivated bus carries an
   eDisGo-specific component**~~ — **fixed**: `ref_add_core!` now drops `dsm`,
   `heatpumps`, `electromobility`, `gen_nd`, `gen_slack` and `heat_storage`
   together with their deactivated bus (`bus_type == 4`). Before, the bus lookups
   indexed the removed bus and threw `KeyError`. (#805)
4. **`solve_model` silently ignores `multinetwork`/`multiconductor`.**
   PowerModels throws an error for a contradicting flag; here a malformed data
   structure goes unnoticed. (#806)
5. **`constraint_model_current` treats virtual storage branches
   inconsistently**: the NC variant skips them (`form/bf.jl:128`), the SOC
   variant does not (`form/bf.jl:101`). In NC, `ccm` of the virtual branch is
   then only limited by its bounds, so the battery losses `ccm·r` are a free
   variable that usually ends up at 0. `check_SOC_equality` also skips storage
   branches. (#802)
6. **`constraint_HV_requirements` has no `else` branch** — for an unknown
   `hv_req["name"]`, `pflex` stays undefined and `sum(pflex)` throws
   `UndefVarError(:pflex)`. (#806)
7. ~~**Debug `println`s in `objective_min_line_loading_max_OG`**~~ — **fixed**:
   the two `println` calls that wrote `factor_ll` and `factor_hv_slacks` to
   stdout on every model build with `opf_version` 3 are removed, and so are the
   stdout captures in the tests that only existed because of them. (#803)
8. **The implicit 1:1 pairing of heat pump and heat storage**:
   `constraint_hp_operation` reads `phs`/`phss` with the heat pump index, so heat
   pump `i` is coupled to heat storage `i`. Since the fix of defect 15, a heat
   pump without a heat storage of the same id runs without storage (needed for a
   switched-off heat storage), and the `phss` term of
   `objective_min_losses_slacks`/`_OG` sums over the heat storage ids. With
   different numbering, heat pumps and heat storages are therefore silently
   decoupled instead of raising a `KeyError`. Nothing checks the pairing at
   input. (#807)
9. **`correct_network_data!` has `check_reference_bus` and `correct_bus_types!`
   commented out** — a network without a slack bus passes without a warning.
   `check_connectivity` does not check the eDisGo-specific components for valid
   bus references at all. (#806)
10. ~~**The SOC→NC warm start in `Main.jl` never works.**~~ — **fixed**:
    `Main.jl` calls `eDisGo_OPF.set_ac_bf_start_values!` with the module prefix,
    and `set_ac_bf_start_values!` skips `pgc` when the result has none
    (`opf_version` 1). Before, the script died with `UndefVarError` after the
    Gurobi solve of a tight SOC relaxation (exit code 1, no result JSON), and
    with the prefix alone `opf_version` 1 failed with `KeyError("pgc")`. (#800)
11. **In `opf_version` 3, the curtailment `pgc` is decoupled from the power
    balance.** `build_mn_opf_bf_flex` creates `pgc` for version 3
    (`prob/opf_bf.jl:36-38`), `constraint_HV_requirements` counts it towards the
    requirement `"curt"` (`core/constraint.jl:145`), but
    `constraint_power_balance` only includes `pgc` in the nodal balance for
    versions 2/4 (`form/bf.jl:211`). The HV requirement is thus met by a
    curtailment that has no effect in the grid; `from_powermodels` still
    subtracts exactly this `pgc` from the feed-in for versions 3/4
    (`edisgo/io/powermodels_io.py`), so the OPF power flows don't match the
    results written back. Side note: `TestUtils.pm_network(opf_version=3)`
    removes `"curt"` from `flexibilities`, while Python sets it for 2, 3 and 4
    (`powermodels_io.py:149`). This affects production: the runner presets
    `overlying_grid_opf.yaml` and `overlying_grid_opf_spatial.yaml` run
    `opf_version: 3`. The two `@test_broken` assume the fix "add `pgc` to the
    balance"; another domain decision means rewriting them. (#801)
12. ~~**`variable_slack_gen_real`/`_imaginary` ignore the start values**~~ —
    **fixed**: both read `pgs_start`/`qgs_start`, `set_ac_bf_start_values!`
    also writes `cpe_start` and the grid start values (`w_start`, `pf_start`,
    `qf_start`, `ccm_start`, `ll_start`), and `Main.jl` seeds every time step,
    not only the first. Before, the warm start left the slack generator, the
    charging energy, the grid variables and all later time steps at their
    defaults. (#804)
13. ~~**`variable_branch_power_real_radial`/`_imaginary_radial` throw a
    `KeyError` for `pt_start`/`qt_start`**~~ — **fixed**: the radial model has
    no to-arc variables, so both fields are ignored. Before, the start value was
    set on the to-arc, which is not in the container. (#804)
14. ~~**`variable_line_loading_max` reads the `storage` flag from network 1
    instead of `nw`**~~ — **fixed**: the flag is read from network `nw`, and the
    default start value is the lower bound 1 instead of 0. The defect had no
    effect in production, where the function is only called with `nw=1`. (#808)
15. ~~**`ref_add_core!` does not filter `heat_storage` by `status`**~~ —
    **fixed**: `ref_add_core!` filters the eDisGo components by their status
    (`gen_status` for `gen_nd`/`gen_slack`, `status` for the others; a missing
    field counts as active). A heat storage is also dropped with its heat pump;
    a heat pump whose heat storage is switched off runs without storage
    (`constraint_hp_operation`). Before, a switched-off heat storage got
    `phs`/`hse` with full bounds and could be dispatched. (#805)
16. ~~**With a single time step, the final-state condition of charging points is
    missing**~~ — **fixed**: `constraint_cp_state_initial` adds the end condition
    (Eq. 3.25 for t = τ) when its network is also the last one, like
    `constraint_store_state_initial` does for battery, heat storage and DSM.
    Before, it only lived in `constraint_cp_state`, which `build_mn_opf_bf_flex`
    calls from the second time step on. Python still rejects fewer than two
    snapshots. (#809)

Minor findings without `@test_broken`: for an unknown `method`, `Main.jl` prints
the unsolved network without `status`/`solver`/`solve_time` and exit code 0;
depending on whether Python picks up that line, it then misleadingly reports
"Julia process failed!" or silently writes no results (#806). Because of defect 5,
the SOC model is not a true relaxation of the NC model; the SOC optimum is
measurably above the NC optimum.

### `Main.jl` → `test/test_main.jl`

- [x] `optimize_edisgo` (the three SOC runs, one without and two with warm start,
  need Gurobi and are skipped without a licence; CI only runs the NC run)

### `src/core/types.jl` → `test/core/test_types.jl`

- [x] `AbstractBFModelEdisgo` — abstract type
- [x] `BFPowerModelEdisgo` — mutable struct
- [x] `AbstractSOCBFModelEdisgo` — abstract type
- [x] `SOCBFPowerModelEdisgo` — mutable struct
- [x] `AbstractNCBFModelEdisgo` — abstract type
- [x] `NCBFPowerModelEdisgo` — mutable struct

### `src/core/base.jl` → `test/core/test_base.jl`

- [x] `solve_model` — Dict
- [x] `instantiate_model` — Dict
- [x] `ref_add_core!` — Dict

### `src/core/data.jl` → `test/core/test_data.jl`

- [x] `set_ac_bf_start_values!` — Dict
- [x] `correct_bus_types!` — Dict
- [x] `_correct_bus_types!` — Dict (note: for an empty `gen`, PowerModels' error
  message applies instead of the package's own, see defect 2)

### `src/core/solution.jl` → `test/core/test_solution.jl`

- [x] `sol_component_value_radial` — AbstractPowerModel
- [x] `check_SOC_equality`

### `src/core/variables.jl` → `test/core/test_variables.jl`

- [x] `variable_branch_power_radial` — AbstractPowerModel
- [x] `variable_branch_power_real_radial` — AbstractPowerModel
- [x] `variable_branch_power_imaginary_radial` — AbstractPowerModel
- [x] `variable_bus_voltage_magnitude_sqr` — AbstractPowerModel
- [x] `variable_max_line_loading` — AbstractPowerModel (note: the only wrapper
  that does not accept `bounded`, `MethodError` — pinned with `@test_throws`)
- [x] `variable_line_loading_max` — AbstractPowerModel
- [x] `variable_gen_power_curt` — AbstractPowerModel
- [x] `variable_gen_power_curt_real` — AbstractPowerModel
- [x] `variable_gen_power_curt_imaginary` — AbstractPowerModel
- [x] `variable_battery_storage` — AbstractPowerModel
- [x] `variable_battery_storage_power_real` — AbstractPowerModel
- [x] `variable_battery_storage_power_imaginary` — AbstractPowerModel
- [x] `variable_dsm_storage_power` — AbstractPowerModel
- [x] `variable_dsm_storage_power_real` — AbstractPowerModel
- [x] `variable_dsm_storage_power_imaginary` — AbstractPowerModel
- [x] `variable_dsm_storage_energy` — AbstractPowerModel
- [x] `variable_heat_storage` — AbstractPowerModel
- [x] `variable_heat_storage_power` — AbstractPowerModel
- [x] `variable_heat_storage_energy` — AbstractPowerModel
- [x] `variable_heat_pump_power` — AbstractPowerModel
- [x] `variable_heat_pump_power_real` — AbstractPowerModel
- [x] `variable_heat_pump_power_imaginary` — AbstractPowerModel (note: created
  without `start`, `qhp_start` is ignored; never called in the OPF, because the
  call in `variable_heat_pump_power` is commented out)
- [x] `variable_cp_power` — AbstractPowerModel
- [x] `variable_cp_power_real` — AbstractPowerModel
- [x] `variable_cp_power_imaginary` — AbstractPowerModel (note: without `start`,
  `qcp_start` is ignored; never called in the OPF, the call in
  `variable_cp_power` is commented out)
- [x] `variable_cp_energy` — AbstractPowerModel
- [x] `variable_slack_grid_restrictions` — AbstractBFModelEdisgo
- [x] `variable_slack_heat_pump_storage` — AbstractBFModelEdisgo
- [x] `variable_hs_slack` — AbstractBFModelEdisgo
- [x] `variable_hp2_slack` — AbstractBFModelEdisgo
- [x] `variable_hp_slack` — AbstractBFModelEdisgo
- [x] `variable_load_slack` — AbstractBFModelEdisgo
- [x] `variable_gen_slack` — AbstractBFModelEdisgo
- [x] `variable_ev_slack` — AbstractBFModelEdisgo
- [x] `variable_slack_gen` — AbstractBFModelEdisgo
- [x] `variable_slack_gen_real` — AbstractBFModelEdisgo
- [x] `variable_slack_gen_imaginary` — AbstractBFModelEdisgo
- [x] `variable_slack_HV_requirements` — AbstractPowerModel
- [x] `variable_slack_HV_requirements_real` — AbstractPowerModel
- [x] `variable_slack_HV_requirements_imaginary` — AbstractPowerModel

### `src/core/constraint.jl` → `test/core/test_constraint.jl`

- [x] `constraint_store_state_initial` — AbstractBFModelEdisgo
- [x] `constraint_store_state` — AbstractBFModelEdisgo
- [x] `constraint_cp_state_initial` — AbstractBFModelEdisgo
- [x] `constraint_cp_state` — AbstractBFModelEdisgo
- [x] `constraint_hp_operation` — AbstractBFModelEdisgo
- [x] `constraint_HV_requirements` — AbstractBFModelEdisgo (bug: there is no
  `else` branch for an unknown `hv_req["name"]`; `pflex` stays undefined and the
  function throws `UndefVarError(:pflex)` instead of a meaningful error message
  — documented as `@test_broken`)

### `src/core/constraint_template.jl` → `test/core/test_constraint_template.jl`

- [x] `constraint_power_balance_bf` — AbstractBFModelEdisgo (note: the power
  factor term `bus_storage_pf` is computed and passed on, but never used by
  `constraint_power_balance` — the battery's reactive power comes from
  `storage_pf` of the virtual branch; a storage without `pf` still makes the
  template fail)
- [x] `constraint_voltage_magnitude_difference_radial` — AbstractBFModelEdisgo
- [x] `constraint_store_state` — AbstractBFModelEdisgo, `(pm, i; nw, kind)`
- [x] `constraint_store_state` — AbstractBFModelEdisgo, `(pm, i, nw_1, nw_2, kind)`
- [x] `constraint_model_current` — AbstractPowerModel

### `src/core/objective.jl` → `test/core/test_objective.jl`

- [x] `objective_min_losses` — AbstractBFModelEdisgo
- [x] `objective_min_losses_slacks` — AbstractBFModelEdisgo (note: heat pump and
  heat storage are paired by id without a check, see defect 8)
- [x] `objective_min_line_loading_max` — AbstractBFModelEdisgo
- [x] `objective_min_losses_slacks_OG` — AbstractBFModelEdisgo
- [x] `objective_min_line_loading_max_OG` — AbstractBFModelEdisgo

### `src/form/bf.jl` → `test/form/test_bf.jl`

- [x] `variable_branch_current` — AbstractBFModel
- [x] `variable_bus_voltage` — AbstractBFModel
- [x] `variable_buspair_current_magnitude_sqr` — AbstractBFModel
- [x] `constraint_voltage_magnitude_difference` — AbstractBFModelEdisgo
- [x] `constraint_model_current` — AbstractSOCBFModelEdisgo (defect 5: the SOC
  variant also writes the constraint for the virtual storage branch — actual
  behaviour pinned, the same branch set as NC as `@test_broken`)
- [x] `constraint_model_current` — AbstractNCBFModelEdisgo (counterpart of
  defect 5: the virtual storage branch is skipped)
- [x] `constraint_max_line_loading` — AbstractSOCBFModelEdisgo
- [x] `constraint_max_line_loading` — AbstractNCBFModelEdisgo
- [x] `constraint_power_balance` — AbstractBFModelEdisgo (bug: for
  `opf_version` 3 the curtailment `pgc` is missing from the balance, see defect
  11 — documented as `@test_broken`; note: `bus_storage_pf` is passed on but
  never used — pinned by a test)

### `src/io/json.jl` → `test/io/test_json.jl`

- [x] `parse_json` — Union{IO

### `src/io/common.jl` → `test/io/test_common.jl`

- [x] `correct_network_data!` — Dict

### `src/prob/opf_bf.jl` → `test/prob/test_opf_bf.jl`

- [x] `solve_mn_opf_bf_flex`
- [x] `build_mn_opf_bf_flex` — AbstractBFModelEdisgo (bug: in version 3 `pgc`
  has no effect on the power balance, see defect 11 — documented as
  `@test_broken`)

<!-- Total number of entries: 84 -->
