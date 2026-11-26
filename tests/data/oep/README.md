# OEP test support

This directory documents how the regular test suite was decoupled from the live Open
Energy Platform (OEP). It contains no local datasets: offline tests build their
minimal synthetic inputs directly in the test modules.

The detailed list of affected tests, tables, columns, data volumes and planned
treatments is maintained in
[`oep_test_inventory.md`](oep_test_inventory.md). General instructions for running
regular and live tests are in [`tests/README.md`](../../README.md).

## Testing approach

- Tests that intentionally access the live OEP are marked with `pytest.mark.oep`.
  Tests requiring an explicit engine use the lazy `oep_engine` fixture; legacy
  connection paths can use `session_scope()` instead.
- OEP tests are skipped before fixture setup unless `--runoep` is supplied.
  Local regular runs therefore need no `-m "not oep"` filter; it is optional and
  reports OEP tests as deselected instead of skipped.
  The CI coverage entry supplies `--runoep` automatically; its push trigger is unchanged.
- Regular tests run without an OEP token and are guarded against unmarked OEP access.
- Offline tests keep the production logic real and mock only the external retrieval or
  query boundary.
- Synthetic DataFrames use the smallest useful set of rows, columns and timestamps.
- Functions containing both a query and transformation are split into a small live
  contract test and a detailed offline transformation test.
- Test names use `_live` and `_offline` where both variants exist.
- The three basic CI entries select `not oep` without an OEP token. The coverage
  entry runs regular and live tests together and uploads their combined coverage
  to Coveralls. Only OEP-enabled coverage entries are serialized across branches.

The usual offline call chain is:

```text
pytest fixture or monkeypatch
    -> replaces the external retrieval function
    -> returns synthetic OEP-shaped data
    -> real production function transforms the data
    -> assertions verify values, shape and function calls
```

The shared mock fixtures are small installers rather than sources of test data. A
test first creates the synthetic DataFrame or dictionary that represents the result
of an OEP import. It then calls the fixture's installer with the module, function
name and synthetic result. The installer creates a
`Mock(return_value=synthetic_result)` and uses pytest's `monkeypatch` fixture to
replace that import function in the module where the production code looks it up.

The high-level call can therefore still use an option such as `"oedb"`. This selects
and exercises the real OEDB processing branch, but its external retrieval call now
returns the synthetic result instead of opening a database connection. Such tests
pass `engine=None` deliberately because the replacement does not require an engine.
The returned `Mock` object also records its calls, allowing the test to verify that
the production code forwarded the expected scenario, time index, IDs and engine.

`monkeypatch` changes the reference in the loaded Python module only for the duration
of a test. It does not modify the source file, and pytest restores the original
function afterwards.

## Implemented changes

### Test infrastructure

- Registered the `oep` marker.
- Replaced the global OEP engine with a lazy `oep_engine` fixture.
- The fixture checks the token before engine creation, using `OEP_TOKEN` first and
  otherwise the package token file. Missing, unreadable or malformed tokens fail
  without displaying their values. Legacy sessions use eDisGo's token-aware OEP
  engine but bypass this fixture, as do notebook kernels. CI separately checks
  token presence before setup.
- Added a runtime guard that reports unmarked tests attempting live OEP access.
- Configured the three basic matrix entries to exclude live OEP tests.
- Kept the Linux/Python 3.12 coverage entry and its 140-minute timeout. It now runs
  regular and live OEP tests together under coverage.py and uploads the combined
  result directly to Coveralls after all selected tests pass. An OEP failure
  prevents that upload. There is no separate OEP or coverage-merging job.
- Moved the early nonempty-token check and shared `edisgo-live-oep` concurrency
  queue to the coverage entry. Its entire execution is serialized across branches;
  basic entries stay independent. It uses no parallel pytest workers or automatic
  retries. Push triggers are unchanged; superseded runs on the same branch are
  still canceled.
- Dependabot runs keep all four matrix entries enabled but execute only regular
  tests. No OEP token is required and no partial coverage is uploaded. Other actors
  still require the configured token for the coverage entry.
- Added `--runoep` opt-in in `tests/conftest.py`. `-m oep` selects marked tests,
  but does not enable them without the flag; slow live tests also need `--runslow`.
- Database/request timeouts are not added. The ordinary suite is OEP-free, not
  completely network-free.

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
- Added a three-hour heat-demand test in February with time-varying annual inputs.
  It checks exact timestamps and values, residential-plus-CTS aggregation and the
  mapping of individual and district-heating demand to heat pumps.
- Split feed-in, COP and district-heating imports into live contract and offline
  transformation tests.
- The ERA5 feed-in live test calls `_query_feedin_oedb` directly for one weather
  cell, avoiding a repeated spatial lookup. It checks required cell/carrier pairs,
  unique keys and finite annual arrays; reshaping and time slicing remain offline.
- Extracted the existing SQL code into the private query helpers
  `_query_feedin_oedb`, `_query_feedin_oedb_legacy`, `_query_cop_oedb` and
  `_query_district_heating_heat_demand_profiles`.
- Split legacy feed-in into a one-weather-cell live query contract and an offline
  transformation test, and made its database engine injectable while retaining the
  legacy connection as the default.
- Reduced the live COP contract to one weather cell and two returned timestamps.
- Converted cross-grid CTS profile collection to an offline test and split per-grid
  CTS queries from offline disaggregation and heat-scaling checks.
- Split industrial and residential electricity profiles into small live query
  contracts and offline aggregation/scaling tests.
- Restored the original 8760-hour requirement for industrial electricity profiles.
  The offline test uses two synthetic annual arrays and checks every hourly value;
  additional cases reject short and 8784-hour leap-year inputs. Supporting arbitrary
  profile lengths is not part of the OEP decoupling.
- Residential electricity queries now return raw profile arrays and mappings;
  the public importer expands the arrays, aggregates profiles, selects scenario
  factors and applies scaling/unit conversion. `eGon2021` sets factor 1.0 without
  a factor query. Three offline scenario cases check exact results and factor-query
  calls; the live contract checks raw inputs and both future factor columns.
- Split residential heat profiles into a live input-schema contract and an offline
  profile-construction test using two synthetic days.

### `test_dsm_import.py`

- Converted `dsm_import.oedb` orchestration to an offline test with mocked CTS and
  industrial retrieval.
- Added a three-hour February selection for all four DSM profiles. Distinct,
  time-varying CTS and industrial inputs check the selected hours, values and
  industrial-building-to-load renaming. Production time selection is not mocked.
- Added offline coverage for array pivoting, CTS distribution and empty industrial-ID
  input.
- Extracted CTS distribution into the private helper
  `_distribute_dsm_profiles_to_cts_loads`.
- Reduced the live CTS result from 85 load columns to one; the industrial live
  contract retains one ID for each distinct source table.

### `test_electromobility_import.py`

- Split SimBEV metadata, potential charging parks and charging processes into
  private OEP query helpers and offline processing tests.
- The offline tests use one shared query-mock fixture and small synthetic metadata,
  geometry, EV-pool and trip datasets.
- Retained calculation of simulated days, CRS conversion, AGS assignment, duplicate
  EV handling, zero-based parking times, duration calculation and dtype conversion.
- Added bounded live query contracts: one metadata row, at most two charging parks,
  and one EV-pool entry with at most two trips.

### `test_storage_import.py`

- Extracted the home-battery OEP query into `_query_home_batteries_oedb` and added
  an optional result limit for the live contract.
- Converted the detailed test to an offline test using three synthetic batteries and
  three local PV generators with matching building IDs.
- Retained the real public import and grid-integration logic, including placement
  with and without matching PV systems and the corresponding log messages.
- Replaced the former grid-wide battery and generator queries with a live contract
  returning at most two home-battery records.

### `test_heat_pump_import.py`

- Separated heat-pump retrieval from validation and grid integration in the public
  `oedb` importer and added an optional query limit for the live contract.
- Converted the detailed import test to synthetic individual, central and resistive-
  heater data while retaining capacity validation, `import_types` selection,
  technology mapping, co-location and voltage-level integration.
- Moved `import_types` selection into `oedb` and split retrieval into technology-
  specific query helpers. Fixed-return mocks no longer implement selection logic.
  Nine offline cases cover the default, an empty selection and every technology
  combination; they check both integrated components and omitted query calls.
- The individual-capacity query and validation run only when individual heat pumps
  are selected, avoiding an unrelated query and warning for central-only imports.
- Central query helpers return raw plant and spatial tables. The importer calls
  `_prepare_central_heat_sources` for CRS conversion, weather/district assignment
  and location selection (in-grid CHP, heat bus, then HV/MV station).
  These steps now run with synthetic spatial inputs in the offline import test.
- New focused offline assertions check reprojection, spatial matches/nonmatches,
  all location fallbacks and empty input. Before decoupling, the live import test
  exercised spatial processing indirectly but asserted component counts,
  capacities and grid integration rather than each spatial result.
- Moved building deduplication out of the query helper into `oedb`, before
  capacity validation and grid integration. The offline import test now supplies
  a duplicate building with a different weather-cell ID
  and checks that only the first record is retained, without double-counting
  capacity or changing its weather-cell assignment.
- Separated retrieval of the scenario heat-parameter record from extraction of the
  resistive-heater efficiency dictionary and tested that extraction offline.
- Added two live query contracts: at most two records per heat-pump technology and
  one scenario heat-parameter record.

### `test_generators_import.py`

- Extracted the three eGon generator-table queries into the private helper
  `_query_generator_data_oedb` and added a per-query result limit for the live
  contract.
- Converted the high-level import test to one synthetic PV rooftop plant, one other
  power plant and one CHP plant while retaining the real public import and grid
  integration.
- Moved carrier normalization, subtype assignment (including rooftop PV) and
  `source_id` extraction into `oedb`, outside the query mock. The offline test now
  checks raw wind/solar records, missing and non-dictionary source IDs, and an empty
  power-plant table. The live contract checks only the columns returned by the queries.
- Replaced the former 677-generator live integration with one bounded contract that
  checks the returned schema and retrieves at most two records from each table.
- Extracted the conventional, renewable-MV and renewable-LV legacy queries into
  `_query_generator_data_oedb_legacy`, which can use the explicit lazy OEP engine.
- Converted the three legacy import/time-series tests and the target-capacity test to
  small synthetic generator parks using one shared query mock.
- The shared legacy data helper includes an MV wind plant (1500 kW) and an MV gas
  plant (0.7 MW), alongside the LV records. Offline assertions check their import,
  MV connection and active/reactive power profiles for worst-case and technology-
  based time series. The normalization test modifies these shared records to test
  numeric strings and missing subtypes.
- Legacy query mocks return renewable capacities in kW. Conversion to numeric MW
  and replacement of missing subtypes with `unknown` run in `oedb_legacy`, outside
  the mocked helper. An offline regression test verifies both renewable voltage
  levels, preserved existing subtypes and unchanged conventional capacities in MW.
- Retained the real public import, generator updates, target-capacity scaling and both
  worst-case and technology-based time-series assignment paths.
- Added one bounded live contract that returns at most two rows from each legacy query.
- Legacy queries sort by ID only when a result limit is requested. Unrestricted
  production imports retain their earlier query behaviour without explicit sorting;
  the pre-existing eGon sorting is unchanged. Offline SQL checks cover all three
  legacy queries with and without a limit.

### `test_tools.py`

- Retained separate live contracts for the legacy COSMO-CLM and eGon ERA5
  weather-cell queries.
- Both live tests temporarily clear generator weather-cell assignments so local
  IDs cannot hide an empty database result. They require a nonempty set of finite,
  positive integer-valued IDs instead of fixed counts or snapshot IDs. The local/
  database ID union remains covered by the existing offline test.
- The eGon ERA5 contract uses the lazy `oep_engine` fixture. The legacy COSMO-CLM
  contract does not request an engine and uses the legacy `session_scope()`
  connection instead. Both retain the `oep` marker; their names identify the data
  source and live scope.

### `test_timeseries.py`

- Converted the legacy and eGon predefined fluctuating-generator tests to use one
  shared feed-in mock fixture with two-hour synthetic profiles.
- Retained the real generator selection, weather-cell mapping, nominal-power scaling,
  raw-profile storage and overwrite behaviour.
- Removed the `oep` markers and the legacy test's `slow` marker; lower-level feed-in
  and weather-cell live contracts cover the external interfaces.

### `test_heat.py`

- Converted COP and heat-demand assignment from OEDB to offline tests using one
  shared heat-import mock fixture and synthetic hourly profiles.
- Retained weather-cell validation and mapping, missing-data warnings, resistive-
  heater efficiencies, heat-pump selection and returned time-index handling.
- Removed both `oep` markers; the lower-level COP, efficiency and heat-demand tests
  cover the corresponding import and live-query boundaries.

### `test_examples.py`

- The plotting notebook uses the checked-out `ding0_test_network_1` grid when run
  from the repository root or `examples` directory. Its test sets the kernel's
  working directory explicitly; fast regression checks verify both local paths
  without downloading. Standalone notebook use retains the GitHub fallback.
- Retained the simple and electromobility notebooks as slow live end-to-end smoke
  tests so the documented OEP workflows remain covered.
- Excluded both notebooks from the three basic entries and retained them in the
  coverage entry with finite notebook and job timeouts.
- Notebook completion without a cell exception is the assertion; the electromobility
  input is local, while its renewable feed-in setup still uses the live OEP.

Focused runs of the implemented offline tests pass. The slow and non-slow groups of
the retained live contracts have also been executed successfully against the OEP.
