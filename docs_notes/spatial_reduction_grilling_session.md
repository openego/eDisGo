# Spatial complexity reduction — pipeline integration design (grilling session)

**Status:** DESIGN CLOSED. All questions resolved 2026-07-14. Next: implement, and decide
the two ADR candidates below.
**Date:** 2026-07-09 (started), closed 2026-07-14.
**Repo/branch:** `/storage/MS/ego/eDisGo`, branch `edisgo_run_edisgo`.
**Companion files:** `CONTEXT.md` (glossary, at repo root), this file.

---

## Goal

Integrate eDisGo's **spatial complexity reduction** into the run pipeline, as a
counterpart to the temporal complexity reduction (timestep selection) already added.

Spatial reduction merges nearby buses into representative buses to shrink the grid so
the **optimization (OPF)** runs faster. **Reinforcement must run on the FULL grid**
(full topology) — but on the *reduced* time index (temporal reduction still applies).

---

## How spatial reduction works (established from code)

- `EDisGo.spatial_complexity_reduction()` (edisgo.py:3439) wraps
  `tools/spatial_complexity_reduction.py::spatial_complexity_reduction()` (line 1830).
- It builds a **busmap** (original bus → clustered `new_bus`) + **linemap**, then mutates
  the Topology in place (or on a copy if `copy_edisgo=True`). Returns `(edisgo, busmap_df,
  linemap_df)`.
- Clustering uses coordinates / grid-graph distance — **independent of the time index**.
  Works on a full-year grid too. `apply_pseudo_coordinates=True` (default) fills missing
  coords.
- **Storage is never aggregated** (only bus-relabeled) — keeps its name/identity even
  with `aggregation_mode=True` (spatial_complexity_reduction.py:1756–1760).
- **Loads/generators ARE merged** when `aggregation_mode=True` (lines 1699–1754), grouped
  by bus(+type+sector). Originals recorded in an **`old_name`** column on the reduced
  component rows; their series summed via `aggregate_timeseries`.
- `reduction_factor_not_focused` uses `find_buses_of_interest` which runs a worst-case
  power flow — but on an **internal deepcopy** (line 93), so it does NOT disturb the
  working object's time series.

### Legacy eGo reference (the pattern we are re-implementing cleanly)
`eGo/ego/tools/edisgo_integration.py::_run_edisgo_task_optimisation` (~line 1632):
- `edisgo_copy = deepcopy(edisgo_grid)` (full grid stays in `edisgo_grid`)
- temporal-reduce copy → spatial-reduce copy → `pm_optimize` on copy
- write dispatch back onto the full grid **by component name** (lines 1763–1795):
  loads/generators/storage active+reactive power, sliced by `time_steps`
- `edisgo_grid.timeseries.timeindex = timeindex` (union of optimized intervals)
- reinforce runs on the full grid. Both grids were resident in memory simultaneously.

---

## Decisions made (confirmed with user)

1. **Two bracketing tasks (Option A), NOT inside `pm_optimize`.**
   - `spatial_reduce` (before `optimize`) and `spatial_restore` (after `optimize`).
   - Rationale: reduce/restore are topology operations, not OPF concerns; must be
     visible/toggleable in YAML; enables stopping after the reduced-grid OPF when only
     the derived time series matter.
   - This deliberately breaks the "all logic inside pm_optimize" principle we used for
     the multi-interval split — justified by the stop-early capability. **ADR candidate.**

2. **`spatial_restore` is OPTIONAL** — a run may stop after optimizing on the reduced
   grid. (But when present it must follow optimize; see validator note.)

3. **Full-grid stash kept IN MEMORY on `ctx`** (matches legacy, which held both grids
   resident). Disk-artifact persistence to cut peak memory is a deferred optimization,
   not v1. (Runner does support disk reload via `save` + `stage_artifacts` + `load_from`,
   but we are not using it here.)

4. **Aggregation support:** implement `aggregation_mode=False` FIRST, then design so
   `aggregation_mode=True` follows.

5. **Map-back only touches components the OPF rewrites:** flexible charging points, heat
   pumps, DSM loads, and storage. **Inflexible loads/generators are SKIPPED** — the OPF
   doesn't change them; the full grid already holds their correct series.
   - `aggregation_mode=False`: write back **by component name**.
   - `aggregation_mode=True`: disaggregate the representative's series onto its `old_name`
     members.

6. **Disaggregation rule (aggregation_mode=True):** split the representative's optimized
   series onto original members **per time step**, weighted by each member's own **pre-OPF
   flexibility envelope** (a known input, never the optimized result):
   - charging points → `electromobility.flexibility_bands["upper_power"][cp_name]` (a CP
     with no connected vehicle has `upper_power(t)=0`, so it receives no charge that step
     — physically correct);
   - heat pumps → `weight(t) = min(heat_demand_df[hp_name][t] / cop_df[hp_name][t],
     loads_df.p_set[hp_name])`. **Refined during implementation (2026-07-14):** the
     original "heat-demand/thermal envelope" phrasing was underspecified. Investigated
     `edisgo/io/powermodels_io.py::_build_heatpump` (~lines 1259-1282): the OPF's actual
     per-unit electrical cap is the CONSTANT rated power `loads_df.p_set`, not a
     time-varying series — the genuinely time-varying pre-OPF quantity is
     `heat_demand_df` (thermal, MW) divided by `cop_df` (electrical-equivalent demand).
     Capping that ratio at each member's own `p_set` mirrors charging points exactly (a
     CP's `upper_power(t)` is already a capped bound, not raw uncapped vehicle demand) and
     ensures no member is ever assigned a share exceeding what it could physically draw;
   - DSM → `dsm.p_max[load_name][t]` band (`edisgo/network/dsm.py:63`).
   - All three sources share the same shape: rows = timestamps matching
     `TimeSeries.timeindex`, columns = component names matching `Topology.loads_df.index`.
   - Sums back to the representative exactly at each step; equal split as zero-envelope
     fallback. (User explicitly preferred a time-series-based split over a static scalar,
     because these envelopes are known pre-OPF and reflect actual flexibility-relevant
     events, e.g. a connected vehicle or nonzero heat demand.)

7. **Ordering:** `select_timesteps` → `spatial_reduce` → `optimize` → `spatial_restore`
   → `reinforce`.
   - The two reduction *mechanisms* commute (orthogonal: topology vs time index), BUT the
     pipeline pins `select_timesteps` before `spatial_reduce` so the stashed full grid
     (hence reinforce) inherits the **reduced** time index. Result: reinforce on full
     **topology** × reduced **time index**.
   - `spatial_restore` does **no time-index surgery** — only writes flexible dispatch back.

8. **Tasks stay thin; computation lives in eDisGo core** (same principle as the
   timestep-selection refactor). See open question for how this applies to restore.

---

## Where we paused — OPEN QUESTION (resume here)

Applying "tasks are thin wrappers" to the restore half. Established:
- **Reduction half is already correct:** `spatial_complexity_reduction()` is already a
  self-contained core function + EDisGo method. `spatial_reduce` task just needs to
  deepcopy+stash the full grid and call it. Nothing to extract.
- **Restore half is the gap:** there is **NO** existing core function that maps reduced
  OPF results back onto a full grid (the legacy logic lived inline in eGo's private
  method; `_restore_pristine_inputs` in powermodels_opf.py:225 is unrelated — it's the
  multi-interval snapshot/restore).

**Proposal put to the user (awaiting confirmation):**
- (a) Create a NEW core function, e.g.
  `tools/spatial_complexity_reduction.py::apply_reduced_results_to_full_grid(full_grid,
  reduced_grid, *, flexible_cps, flexible_hps, flexible_loads, flexible_storage_units)`
  + a thin `EDisGo` method wrapper (mirroring `spatial_complexity_reduction`). The task
  `spatial_restore` just reads the stashed full grid + flexible sets from `ctx` and calls
  it. Used **outside** the pipeline, a caller passes `full_grid`, `reduced_grid`, and the
  flexible sets directly (no `ctx`). This gives symmetry: both halves = core fn + method
  wrapper + thin task; both usable standalone; disaggregation rule lives/tested in core.
- (b) **`old_name`** carried on the reduced grid's `loads_df`/`generators_df` is
  sufficient provenance for disaggregation — the reduced grid self-describes its origins,
  so **no busmap needs stashing**. Full grid needed as the write target (holds individual
  members + their pre-OPF weighting envelopes); reduced grid supplies optimized series +
  `old_name`. Both grids are required args.

**User's last message (the prompt to answer):** agrees restore logic should NOT live in
the task and should become its own eDisGo function; flexible-component names stored in
`ctx` and passed to the function, or passed differently when used outside the pipeline;
reduction is already an independent eDisGo function.

→ So (a) and (b) are essentially aligned with the user's view; next step is to CONFIRM the
signature details (both grids as args; `old_name` sufficient, no busmap) and then move on.

**RESOLVED (2026-07-14):**
- Core function signature:
  `apply_reduced_results_to_full_grid(full_grid, reduced_grid, *, flexible_cps=None,
  flexible_hps=None, flexible_loads=None, flexible_storage_units=None)` — four separate
  kwargs, one per flexible-component type, each defaulting to `None`/skip.
- `EDisGo` method wrapper name: `map_reduced_results_to_full_grid` (full symmetry with the
  core function name, no abbreviation).
- `old_name` on the reduced grid's `loads_df`/`generators_df` is CONFIRMED sufficient
  provenance for disaggregation. No busmap/linemap stash on `ctx`.

---

## Remaining questions still to grill (not yet discussed)

- ~~Validator ordering~~ — **RESOLVED (2026-07-14).** Investigated
  `edisgo/run/validator.py` (`validate()`, lines 47-139) + `edisgo/run/registry.py`
  (`TaskMeta`, `register_task()`): ordering today is capability-based (`requires`/
  `provides` sets accumulated linearly across the pipeline, `validator.py:96,118-130`),
  NOT a dependency graph and NOT named task-to-task precedence. The one existing hardcoded
  exception is `reactive_power` must be last among `ts_altering` tasks
  (`validator.py:110-116`). `select_timesteps`'s optional dual-position behavior
  (`timeseries.py:328-403`) is NOT validator-enforced — it's a runtime-only check inside
  the task, so it was not usable as a precedent.
  - **Decision:** extend the existing capability system rather than add a new validator
    concept or fall back to runtime-only checking (matches the pipeline's existing
    mechanism everywhere else):
    - `spatial_reduce` declares `provides={"reduced_grid"}`.
    - `optimize` declares `provides={"optimized_dispatch", ...}` (in addition to its
      existing provides).
    - `spatial_restore` declares `requires={"reduced_grid", "optimized_dispatch"}`.
  - This closes the gap where presence-only capability accumulation would otherwise let
    `spatial_restore` validate successfully even if placed before `optimize` (both
    `reduced_grid`-derived requirements would already be "satisfied" from
    `spatial_reduce` alone) — requiring `optimized_dispatch` too means `spatial_restore`
    cannot pass validation until `optimize` has actually appeared earlier in the
    pipeline.
  - **Correction (2026-07-14, during implementation):** `register_task`'s `requires`/
    `provides` (`edisgo/run/registry.py:52-58`) are fixed at decoration time (module
    load), NOT evaluated per-run — so "optimize requires `reduced_grid` only when spatial
    reduction is configured for THIS run" is not expressible and was dropped.
    `optimize`'s `requires` stays exactly `{"timeseries", "flex"}`, unchanged — it does
    not need to know spatial reduction exists. Ordering is fully enforced from
    `spatial_restore`'s side alone; adding `reduced_grid` to `optimize`'s `requires`
    unconditionally would have broken every existing preset that runs `optimize` without
    `spatial_reduce` (uc2, uc4, uc5_select_timesteps).
- ~~YAML config surface~~ — **RESOLVED (2026-07-14).** Top-level `spatial_reduction:`
  block, mirroring `timeseries_selection:`. Read via
  `ctx.raw_config.get("spatial_reduction", {})` inside the `spatial_reduce` task, same
  pattern as `select_timesteps` (`timeseries.py:415`). Holds `mode`, `cluster_area`,
  `reduction_factor`, `reduction_factor_not_focused`, `aggregation_mode`, aggregation
  sub-modes.
- ~~eGo injection~~ — **RESOLVED (2026-07-14).** Verified `timeseries_selection`'s actual
  injection in `EDisGoNetworks._build_run_edisgo_config()`,
  `eGo/ego/tools/edisgo_integration.py:675-685`: global `timeseries_selection` default +
  `timeseries_selection_per_grid` dict keyed by `str(mv_grid_id)`
  (`edisgo_integration.py:681-684`), **whole-block replacement** (not field-level merge),
  key omitted from `cfg` entirely if both are unset (line 684: `if ts_selection is not
  None`), letting the eDisGo preset's own default apply. Granularity is truly per
  individual MV grid (`mv_grid_id`, looped in `run_all`, `edisgo_integration.py:597-626`).
  - **Decision:** `spatial_reduction` replicates this exactly — global `spatial_reduction`
    default + `spatial_reduction_per_grid` dict keyed by `str(mv_grid_id)`, whole-block
    replacement, omitted if unset (same as `timeseries_selection`, not the simpler
    `overlying_grid` hardcoded-fallback pattern).
- ~~Reactive power~~ — **RESOLVED (2026-07-14).** Investigated existing convention:
  `pm_optimize`'s results-writer (`edisgo/io/powermodels_io.py::from_powermodels`,
  lines 283-352) writes ONLY active power for flex components (heat pumps, CPs, DSM,
  storage) into `_generators_active_power`/`_loads_active_power`/
  `_storage_units_active_power`; reactive power is untouched there. Immediately after
  (line 354-355), it calls the plain `edisgo_object.set_time_series_reactive_power_control()`
  — same generic fixed-cosphi default (`network/timeseries.py::fixed_cosphi`,
  `flex_opt/q_control.py`) used everywhere else in eDisGo, applied blanket over the whole
  object, not scoped to flex components. Confirmed the existing `reactive_power` pipeline
  task (`edisgo/run/tasks/timeseries.py:584-629`) is just a thin wrapper around the exact
  same call — no special-casing for OPF-derived components anywhere in the codebase today.
  - Considered alternative: split reactive power proportionally to each `old_name`
    member's share of the representative's active power (mirroring the active-power
    disaggregation rule) instead of recomputing. **Verified mathematically equivalent**
    under fixed-cosphi: all `old_name` members of one representative share the same
    `type` → same `power_factor`, so `Q = P · tan(φ)` per member gives an identical result
    whether derived by proportional split or by recomputing from each member's
    disaggregated P directly.
  - **Decision:** `spatial_restore` writes active power for flexible components onto the
    full grid, then calls `set_time_series_reactive_power_control()` itself — mirrors
    `pm_optimize`'s own convention exactly (write P, then blanket-recompute Q). Reuses the
    existing method with no new reactive-power math anywhere, and makes `spatial_restore`
    correct standalone even in pipelines with no downstream `reactive_power` task.
- ~~Testing strategy~~ — **RESOLVED (2026-07-14).** Scope for this first pass: core
  function only (`apply_reduced_results_to_full_grid`), NOT pipeline/task-level
  integration tests (deferred). Cover both aggregation modes:
  - `aggregation_mode=False`: by-name write-back correctness.
  - `aggregation_mode=True`: disaggregation math — per-step envelope-weighted split,
    exact-sum-back-to-representative check, and the equal-split zero-envelope fallback.
  - Stub/fake OPF results as fixtures; no real `pm_optimize` call, no real pipeline run
    through `ctx`/validator.
- ~~uc5 preset~~ — **RESOLVED (2026-07-14).** New standalone preset
  `edisgo/run/presets/uc5_spatial_reduction.yaml` (full copy of
  `uc5_select_timesteps.yaml` + the spatial bracket, NOT an `extends` overlay — tasks
  don't exist in code yet so this is documentation/example, and a standalone file matches
  `uc5_select_timesteps.yaml`'s own self-contained style). Disable switch: explicit
  `spatial_reduction.enabled` flag (mirrors `overlying_grid.enabled`, NOT
  `timeseries_selection`'s absent-block-is-the-toggle style) — lets params stay in the
  YAML while toggling on/off with one flag. Bracket placement: `spatial_reduce` right
  before `optimize`, `spatial_restore` right after; `reactive_power` stays where it already
  is (pre-OPF full-series cosphi on the reduced index), unaffected by the spatial bracket.
  Final order: `select_timesteps(post_grid) → reactive_power → spatial_reduce → optimize →
  spatial_restore → reinforce`.

---

## Implementation (2026-07-14)

Implemented and tested end-to-end (real venv, python3.10, `pip install -e ".[dev]"`,
real ding0 test grid `tests/data/ding0_test_network_1`):

- `apply_reduced_results_to_full_grid` +
  `EDisGo.map_reduced_results_to_full_grid` —
  `edisgo/tools/spatial_complexity_reduction.py`, `edisgo/edisgo.py`.
- `spatial_reduce` / `spatial_restore` tasks — new file `edisgo/run/tasks/spatial.py`,
  registered in `edisgo/run/tasks/__init__.py`.
- `RunContext.full_grid_stash` — new field, `edisgo/run/context.py`.
- `task_optimize` writes `flexible_cps`/`flexible_hps`/`flexible_loads`/
  `flexible_storage_units` to `ctx.flags`; `@register_task("optimize", ...)` gained
  `provides={"optimized_dispatch"}` — `edisgo/run/tasks/analysis.py`.
- New preset `edisgo/run/presets/uc5_spatial_reduction.yaml` (already covered above).
- New tests `tests/tools/test_spatial_complexity_reduction.py::TestApplyReducedResultsToFullGrid`
  (6 tests: by-name write-back, multi-member disaggregation sum check, singleton-rename
  regression, time-index-mismatch error, storage-unit by-name path, reactive-power
  recompute). Full existing suite (`tests/tools/`, `tests/run/`, `tests/opf/`) reverified
  green: 99 passed, 1 unrelated skip.

**Two real bugs found by a dedicated code-review agent (dispatched because no import-
capable env existed initially) and fixed before landing:**

1. **Singleton-rename data corruption (serious).** Original code took a `_write_by_name`
   fast path whenever a flexible-component set had NO multi-member merged group,
   assuming an unmerged representative's name always equals its member's name. FALSE
   under `aggregation_mode=True`: `spatial_complexity_reduction` renames **every**
   group's representative, including singleton groups (a bus with exactly one flexible
   load of a type/sector) — confirmed empirically on the real test grid (11 such
   singletons exist in `ding0_test_network_1` alone). `_write_by_name` using the
   representative's (renamed) name against `full_grid` (which only has the original,
   un-renamed name) silently created a phantom column via pandas' `.loc[]` auto-vivify
   behavior, leaving the real target column stale — a silent, hard-to-detect data
   corruption, not a crash. **Fix:** removed the `_write_by_name` fast path for CPs/HPs/
   DSM loads entirely; always route through `_disaggregate`, which was proven correct for
   every case (matching name, mismatched singleton, multi-member) by direct test.
   `_write_by_name` now only serves storage units, which are genuinely never renamed.
2. **`flexible_* or []` crashes on numpy-array input (real, hit on first live
   end-to-end run).** `task_optimize` derives `flexible_loads` as
   `edisgo.dsm.p_min.columns.values` (`analysis.py:357`) — a numpy array — unlike the
   other three `flexible_*` lists, which use `.tolist()`. `array or []` raises
   `ValueError: The truth value of an array with more than one element is ambiguous...`
   for any such array with 2+ elements. This surfaced immediately on the very first real
   pipeline run (`run_example_06.py`, `uc6_spatial_reduction.yaml`, `aggregation_mode:
   false`) — the OPF/Gurobi solve completed successfully, `spatial_restore` crashed on
   the very next line. **Fix:** replaced `flexible_x = flexible_x or []` with
   `flexible_x = list(flexible_x) if flexible_x is not None else []` for all four
   parameters in `apply_reduced_results_to_full_grid` — `is None` is the correct
   emptiness check for an optional list-like argument that may be a list, tuple, or numpy
   array (the codebase's own docstrings elsewhere already document these params as
   accepting `numpy.ndarray or None`). Added regression test
   `test_accepts_numpy_array_flexible_component_lists`.
3. **Unguarded `KeyError` on time-index mismatch (robustness).** If a flexibility-band/
   DSM/heat-pump attribute on `full_grid` doesn't cover `full_grid.timeseries.timeindex`
   (e.g. the full-grid stash was taken before time-index selection ran — a real risk for
   any pipeline not following the `select_timesteps` → `spatial_reduce` convention, since
   nothing in the registry enforces that order), the disaggregation envelope lookup would
   raise a bare `KeyError` deep inside a `.loc` call. **Fix:** added
   `_require_full_timeindex`, a defensive check before each envelope lookup that raises a
   clear `ValueError` naming the mismatch and the likely cause. Decided against also
   adding a validator `requires={"timeseries"}` declaration — the pipeline is already
   constructed so a time index is guaranteed set before `spatial_reduce` runs (decision 7,
   above), so the defensive check alone is sufficient without touching registry metadata.

## ADR candidates (offer at end of session)

1. "Spatial reduction as two bracketing pipeline tasks, restore optional" — hard to
   reverse, surprising (breaks the pm_optimize-owns-its-logic principle), real trade-off
   (consistency vs stop-early). → write when design settles.
2. Possibly: "Disaggregation by pre-OPF flexibility envelope per time step" — a modeling
   choice with alternatives (static scalar, proportional-to-original-series). Borderline;
   decide at end.
