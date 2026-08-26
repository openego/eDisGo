# OEP test support

This directory collects documentation and, where necessary, small local datasets for
decoupling the regular test suite from the live Open Energy Platform (OEP).

The detailed list of affected tests, tables, columns, data volumes and planned
treatments is maintained in
[`oep_test_inventory.md`](oep_test_inventory.md). General instructions for running
regular and live tests are in [`tests/README.md`](../../README.md).

## Testing approach

- Tests that intentionally access the live OEP are marked with `pytest.mark.oep` and
  use the lazy `oep_engine` fixture.
- Regular tests run without an OEP token and are guarded against unmarked OEP access.
- Offline tests keep the production logic real and mock only the external retrieval or
  query boundary.
- Synthetic DataFrames use the smallest useful set of rows, columns and timestamps.
- Functions containing both a query and transformation are split into a small live
  contract test and a detailed offline transformation test.
- Test names use `_live` and `_offline` where both variants exist.

The usual offline call chain is:

```text
pytest fixture or monkeypatch
    -> replaces the external retrieval function
    -> returns synthetic OEP-shaped data
    -> real production function transforms the data
    -> assertions verify values, shape and function calls
```

`monkeypatch` changes the reference in the loaded Python module only for the duration
of a test. It does not modify the source file, and pytest restores the original
function afterwards.

## Implemented changes

### Test infrastructure

- Registered the `oep` marker.
- Replaced the global OEP engine with a lazy `oep_engine` fixture.
- Added a runtime guard that reports unmarked tests attempting live OEP access.

### `test_edisgo.py`

- Converted both predefined OEDB time-series tests to offline tests.
- Mocked load and feed-in retrieval while retaining branch selection, assignment and
  automatic time-index handling.
- Converted the legacy generator API test to an offline delegation test and avoided
  creating an unused default engine for legacy grids.
- Converted the OEDB electromobility API test to three synthetic retrieval results
  while retaining charging-demand allocation and grid integration.
- Converted heat-pump import orchestration to synthetic components, heat demand and
  COP profiles while retaining scenario validation, leap-year handling and profile
  mapping.

### `test_timeseries_import.py`

- Converted heat- and electricity-demand orchestration tests to use one shared mock
  fixture with minimal synthetic profiles.
- Split feed-in, COP and district-heating imports into live contract and offline
  transformation tests.
- Extracted the existing SQL code into the private query helpers
  `_query_feedin_oedb`, `_query_feedin_oedb_legacy`, `_query_cop_oedb` and
  `_query_district_heating_heat_demand_profiles`.
- Split legacy feed-in into live and offline tests and made its database engine
  injectable while retaining the legacy connection as the default.
- Reduced the live COP contract to one weather cell and two returned timestamps.
- Converted cross-grid CTS profile collection to an offline test and split per-grid
  CTS queries from offline disaggregation and heat-scaling checks.
- Split industrial and residential electricity profiles into small live query
  contracts and offline aggregation/scaling tests.
- Split residential heat profiles into a live input-schema contract and an offline
  profile-construction test using two synthetic days.

### `test_dsm_import.py`

- Converted `dsm_import.oedb` orchestration to an offline test with mocked CTS and
  industrial retrieval.
- Added offline coverage for array pivoting, CTS distribution and empty industrial-ID
  input.
- Extracted CTS distribution into the private helper
  `_distribute_dsm_profiles_to_cts_loads`.
- Reduced the live CTS result from 85 load columns to one; the industrial live
  contract retains one ID for each distinct source table.

Focused runs of the implemented offline tests pass. Renamed live contracts are
discoverable with `pytest -m oep --collect-only`, but have not yet been executed
against the live OEP.

## Remaining work

- Apply the selected treatment to the remaining tests in the inventory.
- Configure separate regular and live OEP jobs in CI, including serialization and
  finite timeouts.
- Run and verify the retained live contracts with an OEP token supplied through the
  approved secret mechanism.
- Update the inventory status as each group is completed.

No OEP-derived datasets are currently committed here; the implemented offline tests
generate synthetic data in code. If local extracted data is added later, document its
source query, extraction date, scenario, licence, reduction steps and consuming tests
in this file.
