# Flexibility bands should be built scoped to the selected time index, not built-then-trimmed

**Type:** design gap / bug
**Found:** 2026-07-15, while testing spatial complexity reduction against a
manual-mode `select_timesteps` run (`uc6_spatial_reduction.yaml`).
**Affects:** `task_build_flexibility_bands` (`edisgo/run/tasks/flex.py`),
`Electromobility.get_flexibility_bands` (`edisgo/network/electromobility.py`),
and by extension any other per-component band/envelope built after
`select_timesteps` (heat pump `heat_demand_df`/`cop_df`, DSM `p_max`/`p_min`).

## Problem

`select_timesteps` (manual mode, `position: pre_import`) fixes
`edisgo.timeseries.timeindex` to an arbitrary, possibly short window (e.g. 24
hours starting `2035-01-15`) early in the pipeline, before electromobility
data is even imported.

`build_flexibility_bands` runs later and calls
`Electromobility.get_flexibility_bands(edisgo, use_case=...)`. Per that
method's own docstring, `resample=True` only "resamples the bands to the same
**frequency** as time series data in the `TimeSeries` object" — it does
*not* clip the bands to the same *date range*. The bands are built from the
raw SimBEV charging-process data and keep whatever date range that data
spans, matching the target row spacing (e.g. hourly) but not the target
window.

Nothing downstream re-trims `electromobility.flexibility_bands` to the
selected 24-hour window afterward. The mismatch went unnoticed until a piece
of code tried to actually index `flexibility_bands` using
`edisgo.timeseries.timeindex` and hit a `KeyError`-shaped failure (missing
time steps) — in this case, the new
`apply_reduced_results_to_full_grid`/`spatial_restore` disaggregation logic,
which reads `flexibility_bands["upper_power"]` as a per-charging-point,
per-time-step weighting envelope.

This is a **pre-existing gap**, not something introduced by spatial
reduction — it already exists in `uc5_select_timesteps.yaml` (the preset
`uc6_spatial_reduction.yaml`/`uc5_spatial_reduction.yaml` were both derived
from). It simply had no consumer that indexed `flexibility_bands` by the
active time index before now.

## Immediate fix applied (unblocks spatial-reduction testing)

`task_build_flexibility_bands` now calls
`reduce_timeseries_data_to_given_timeindex(edisgo, edisgo.timeseries.timeindex,
electromobility=True, timeseries=False, heat_pump=False, dsm=False,
overlying_grid=False)` right after `get_flexibility_bands`, trimming
`flexibility_bands` down to the active index. See
`edisgo/run/tasks/flex.py::task_build_flexibility_bands`.

This is a workaround (build full, then trim), not the better design below.

## Suggested proper fix (not yet implemented — this issue)

Build flexibility bands (and, likely, heat-pump/DSM bands) scoped to the
already-selected time index from the start, rather than building over the
full/native data range and trimming afterward:

- `Electromobility.get_flexibility_bands` (or its caller) should accept/use
  the target time index *before* running the difference-array band
  construction, so the SimBEV charging-process data outside that window is
  never even considered.
- Audit whether `HeatPump`/`DSM` band construction (wherever their
  time-varying bounds are first populated — likely in the `import_heat_pumps`
  / `import_dsm` tasks or their underlying `edisgo/io/*` importers) has the
  same built-on-full-range-then-maybe-trimmed pattern, since
  `reduce_timeseries_data_to_given_timeindex` already has `heat_pump=True`/
  `dsm=True` flags suggesting this was anticipated but may not be
  consistently invoked at the right point in every pipeline path.
- Consider whether this should be a single, explicit "finalize time index"
  pipeline hook that every band-producing task can rely on having already
  run, rather than each task needing to remember to trim itself.

## Reproduction

Run `uc6_spatial_reduction.yaml` (or `uc5_select_timesteps.yaml`) with
`timeseries_selection: {mode: manual, start: "2035-01-15 00:00", periods:
24, freq: h}` and `overlying_grid.enabled: true`, then inspect
`edisgo.electromobility.flexibility_bands["upper_power"].index` after
`build_flexibility_bands` runs — before the fix above, its date range does
not match `edisgo.timeseries.timeindex`.
