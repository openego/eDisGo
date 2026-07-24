# Integrate spatial complexity reduction into the run pipeline

## Goal

Add **spatial complexity reduction** to the eDisGo run pipeline as a
counterpart to the existing temporal complexity reduction (timestep
selection, `select_timesteps`). Spatial reduction merges nearby buses into
representative buses to shrink the grid so the **optimization (OPF)** runs
faster. **Reinforcement must run on the FULL grid** (full topology) — but on
the *reduced* time index (temporal reduction still applies).

## What's done

- **Two bracketing pipeline tasks**, `spatial_reduce` (before `optimize`)
  and `spatial_restore` (after, optional): `spatial_reduce` deepcopies and
  stashes the full grid on the run context, then spatially reduces the
  working object so `optimize` runs on a smaller grid; `spatial_restore`
  writes the optimized flexible-component dispatch back onto the stashed
  full grid and makes it active again for `reinforce`. Both are no-ops when
  `spatial_reduction.enabled` is false, so a pipeline carrying the bracket
  behaves identically to one without it when disabled.
- **New core function** `apply_reduced_results_to_full_grid(full_grid,
  reduced_grid, *, flexible_cps, flexible_hps, flexible_loads,
  flexible_storage_units)` in `edisgo/tools/spatial_complexity_reduction.py`,
  plus a thin `EDisGo.map_reduced_results_to_full_grid` wrapper — mirroring
  how `spatial_complexity_reduction`/`EDisGo.spatial_complexity_reduction`
  already pair up for the reduction half.
- **Map-back only touches components the OPF rewrites**: flexible charging
  points, heat pumps, DSM loads, and storage units. Inflexible
  loads/generators are skipped since the OPF never changes their series.
  With `aggregation_mode=False` (see "What's still open" below), every
  component keeps its own name, so restore is a plain by-name write-back.
- **Reactive power on restore**: `spatial_restore` writes active power only,
  then calls `EDisGo.set_time_series_reactive_power_control()` itself —
  mirroring exactly how the OPF's own results-writer
  (`io/powermodels_io.py::from_powermodels`) handles it (write P, then
  blanket-recompute Q). No new reactive-power logic anywhere.
- **Validator integration**: extended the existing `requires`/`provides`
  capability system rather than adding new validator machinery.
  `spatial_reduce` provides `reduced_grid`; `optimize` additionally provides
  `optimized_dispatch`; `spatial_restore` requires both — so a misordered
  pipeline (e.g. `spatial_restore` before `optimize`) fails static
  validation instead of crashing mid-run.
- **Config surface**: top-level `spatial_reduction:` YAML block (`enabled`,
  `mode`, `cluster_area`, `reduction_factor`, `reduction_factor_not_focused`,
  `aggregation_mode`), mirroring `timeseries_selection:` exactly, read via
  `ctx.raw_config.get("spatial_reduction", {})`.
- **New preset** `edisgo/run/presets/uc6_spatial_reduction.yaml` wiring the
  bracket into a real pipeline (`select_timesteps → reactive_power →
  spatial_reduce → optimize → spatial_restore → reinforce → save`), disabled
  by default via `spatial_reduction.enabled: false`.
- **eGo integration**: `EDisGoNetworks._build_run_edisgo_config` injects
  `spatial_reduction`/`spatial_reduction_per_grid` exactly like
  `timeseries_selection`/`timeseries_selection_per_grid` (global default +
  per-grid override keyed by MV grid id, whole-block replacement). New
  `scenario_setting_uc6_example.json`, plus unit tests for the injection
  logic.
- **Tests**: `tests/tools/test_spatial_complexity_reduction.py`
  (`TestApplyReducedResultsToFullGrid`) covers by-name write-back, the
  numpy-array-input regression, the time-index-mismatch guard, and the
  reactive-power recompute, with stub OPF results (no real Julia/Gurobi
  dependency in CI).
- **Verified end-to-end** against a real grid (32377) through both eDisGo
  directly and through eGo: `spatial_reduce → optimize (Gurobi) →
  spatial_restore → reinforce → save` all completed successfully, and a
  dedicated notebook (`analyse_spatial_reduction.ipynb`) confirms the
  reduced grid's OPF output exactly matches the full grid's post-restore
  value for every flexible component (0 mismatches across 388 components on
  the test run).
- **Two bugs found and fixed during implementation** (unrelated to the
  design, surfaced by actually running the code): a `flexible_* or []`
  crash on numpy-array input (`task_optimize` derives `flexible_loads` as an
  array, not a list); and `electromobility.flexibility_bands` not being
  trimmed to the active time index after `build_flexibility_bands`, causing
  a `KeyError` deep inside the OPF's charging-point constraint builder.

## What's still open

### `aggregation_mode=True` is not usable with charging points present

`spatial_complexity_reduction` (`aggregation_mode=True`) merges loads at the
same bus and correctly aggregates their **time series**
(`loads_active_power`/`loads_reactive_power`). This works for DSM loads and
heat pumps, whose OPF constraints are read directly from
`loads_df`/`dsm.p_max`/`heat_pump.heat_demand_df`.

Charging points are different: the OPF's constraint builder
(`_build_electromobility` in `edisgo/io/powermodels_io.py`) reads its
upper-power bound from `electromobility.flexibility_bands["upper_power"]`, a
separate DataFrame keyed by charging-point name — and
`spatial_complexity_reduction` never aggregates `flexibility_bands` during
load merging. So a merged representative has a correctly-summed
`loads_active_power` entry but **no** entry in `flexibility_bands`, and
`optimize` crashes with a `KeyError` for the representative's name before
`spatial_restore` ever runs.

This is a gap in the reduction side (`spatial_complexity_reduction`/
`apply_busmap`), not in `spatial_restore`/`apply_reduced_results_to_full_grid`
— reachable via `EDisGo.spatial_complexity_reduction` + `EDisGo.pm_optimize`
directly, no pipeline involved. It only manifests under
`aggregation_mode=True` with charging points present at a bus with 2+ of
them; `aggregation_mode=False` is unaffected. Full writeup with the exact
traceback and a suggested fix (aggregate `flexibility_bands` in
`apply_busmap`'s load-merging step, the same way `loads_active_power`
already is) is in
`docs_notes/issue_aggregation_mode_flexibility_bands_not_aggregated.md`.

**Until this is fixed, `aggregation_mode` should stay `False`** — that is
the default in the new preset and the only mode covered by the disaggregation
tests and the end-to-end verification above.

### Related, deferred design gap (not blocking, tracked separately)

`build_flexibility_bands` builds bands over whatever date range the
underlying SimBEV data spans and only resamples to the target *frequency*,
not the target *date range* — the fix applied here trims them after the
fact (`reduce_timeseries_data_to_given_timeindex`). Building them
pre-scoped to the selected window in the first place would be cleaner and
more efficient; see
`docs_notes/issue_temporal_reduction_flexibility_bands.md`.

### Not yet done

- Reactive-power write-back has no dedicated integration test against a
  real OPF run (covered by unit test with stub dispatch only).
- No pipeline/task-level integration test for `spatial_reduce`/
  `spatial_restore` (current test scope is core-function-only, per an
  explicit scoping decision — see `docs_notes/spatial_reduction_grilling_session.md`).
- Two ADR candidates flagged during design, not yet written: "spatial
  reduction as two bracketing pipeline tasks, restore optional" (breaks the
  established "logic lives inside `pm_optimize`" convention, trading
  consistency for the ability to stop early after the reduced-grid OPF), and
  "disaggregation by pre-OPF flexibility envelope" (a modeling choice with
  real alternatives).

## References

- Full design session record: `docs_notes/spatial_reduction_grilling_session.md`
- `aggregation_mode=True` + charging points gap: `docs_notes/issue_aggregation_mode_flexibility_bands_not_aggregated.md`
- Flexibility-bands time-index gap (fixed): `docs_notes/issue_temporal_reduction_flexibility_bands.md`
