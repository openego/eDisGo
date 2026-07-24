# Context / Glossary

Domain vocabulary for the eDisGo run pipeline. Glossary only — no implementation
details, no decisions (those live in `docs/adr/`).

## Complexity reduction

- **Spatial complexity reduction** — merging nearby buses into a smaller set of
  representative buses (clustering *along the grid*, keeping it radial) to shrink the
  grid for faster power flow / optimization. Implemented by
  `EDisGo.spatial_complexity_reduction`, which builds a *busmap* + *linemap* and mutates
  the Topology. Used only to accelerate the optimization; reinforcement runs on the
  **full grid**.
- **Temporal complexity reduction** — keeping only grid-critical time steps/intervals
  instead of the full year. See the `select_timesteps` task.
- **Busmap** — DataFrame mapping each original bus to its clustered *new_bus* (with new
  coordinates). Index = original bus names.
- **Linemap** — DataFrame mapping original line names to *new_line_name* after lines are
  recalculated/merged.
- **Reduced grid** — the spatially-reduced EDisGo object the OPF runs on.
- **Full grid** — the original, unreduced EDisGo object; reinforcement always runs here.
- **Map-back / restore** — writing the OPF flexibility dispatch from the reduced grid
  onto the full grid. Only the components the OPF *rewrites* are mapped back — flexible
  charging points, heat pumps, DSM loads, and storage. Inflexible loads/generators are
  **skipped** (the OPF does not change their series; the full grid already holds them
  correctly). Implemented by core function
  `tools/spatial_complexity_reduction.py::apply_reduced_results_to_full_grid(full_grid,
  reduced_grid, *, flexible_cps=None, flexible_hps=None, flexible_loads=None,
  flexible_storage_units=None)` + thin wrapper `EDisGo.map_reduced_results_to_full_grid`.
  Provenance for disaggregation comes solely from `old_name` on the reduced grid's
  `loads_df`/`generators_df` — no busmap/linemap stash needed on `ctx`.
  - `aggregation_mode=False`: components keep their names → write back **by component
    name**.
  - `aggregation_mode=True`: loads/generators may be merged into a representative
    (originals recorded in `old_name`; storage is never aggregated). The representative's
    optimized series is **disaggregated** onto its `old_name` members.
- **Disaggregation rule** — split a merged representative's optimized series onto its
  original members **per time step**, weighted by each member's own *pre-OPF flexibility
  envelope* (a known input, never the optimized result): `upper_power(t)` band for
  charging points, heat-demand/thermal envelope for heat pumps, `p_max(t)` band for DSM.
  Per-step weighting means a charging point only receives power at steps where it has a
  connected vehicle (`upper_power(t) > 0`). Sums back to the representative exactly at
  every step; equal split as the zero-envelope fallback.
- **Reactive power on restore** — `spatial_restore` writes active power only, then calls
  `EDisGo.set_time_series_reactive_power_control()` itself (plain fixed-cosphi default),
  mirroring exactly how `pm_optimize`'s own results-writer
  (`io/powermodels_io.py::from_powermodels`) handles it: write P, then blanket-recompute Q.
  No bespoke reactive-power logic — proportional-split and recompute are mathematically
  identical under fixed-cosphi since all `old_name` members of one representative share
  the same `power_factor`.

## Pipeline tasks (spatial reduction)

- **spatial_reduce** — task run *before* `optimize`: deepcopy the full grid, stash it,
  and spatially reduce the working object so `optimize` runs on the reduced grid.
- **spatial_restore** — task run *after* `optimize`: write the optimized dispatch time
  series back onto the stashed full grid and make it active again. **Optional** — a run
  may legitimately stop after the reduced-grid optimization when only the derived time
  series matter.
- **Full-grid stash** — the deepcopied full grid kept in memory on `ctx` between
  `spatial_reduce` and `spatial_restore`. Held in-memory (matching legacy eGo, which kept
  both full and reduced grids resident during optimize). Persisting it to a disk artifact
  to lower peak memory is a possible later optimization, not the initial design.

## Config surface (spatial reduction)

- Top-level YAML block `spatial_reduction:` (mirrors `timeseries_selection:`), holding
  `mode`, `cluster_area`, `reduction_factor`, `reduction_factor_not_focused`,
  `aggregation_mode`, and aggregation sub-modes. Read inside the `spatial_reduce` task via
  `ctx.raw_config.get("spatial_reduction", {})`.
- eGo injects it exactly like `timeseries_selection`: a global `spatial_reduction` default
  plus a `spatial_reduction_per_grid` dict keyed by `str(mv_grid_id)`, whole-block
  replacement (not a field-level merge). If neither is set the key is omitted entirely and
  the eDisGo preset's own default applies.

## Ordering (spatial reduction)

Pipeline order: `select_timesteps` → `spatial_reduce` → `optimize` → `spatial_restore`
→ `reinforce`.

- Spatial reduction touches only **topology**; temporal reduction touches only the
  **time index** — orthogonal operations that commute as *mechanisms*. Spatial reduction
  works on any time index, including full-year (clustering depends on coordinates/graph
  distance, not the time series; the `reduction_factor_not_focused` worst-case power flow
  runs on an internal deepcopy and does not disturb the working series).
- But the **pipeline** pins `select_timesteps` before `spatial_reduce`: the full-grid
  stash inherits whatever time index exists at deepcopy time, and reinforce runs on that
  stash. Reducing timesteps first means the stash (and therefore reinforce) carries the
  reduced index — full **topology**, reduced **time index**. Reversing the two would
  force separately re-reducing the stash's index.
- `spatial_restore` does **no time-index surgery** — it only writes flexible-component
  dispatch back onto the stashed full grid.
