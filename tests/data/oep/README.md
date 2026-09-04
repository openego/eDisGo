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
- Split legacy feed-in into a one-weather-cell live query contract and an offline
  transformation test, and made its database engine injectable while retaining the
  legacy connection as the default.
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
- Replaced the former 677-generator live integration with one bounded contract that
  checks the returned schema and retrieves at most two records from each table.
- Extracted the conventional, renewable-MV and renewable-LV legacy queries into
  `_query_generator_data_oedb_legacy`, which can use the explicit lazy OEP engine.
- Converted the three legacy import/time-series tests and the target-capacity test to
  small synthetic generator parks using one shared query mock.
- Retained the real public import, generator updates, target-capacity scaling and both
  worst-case and technology-based time-series assignment paths.
- Added one bounded live contract that returns at most two rows from each legacy query.

### `test_tools.py`

- Retained separate live contracts for the legacy COSMO-CLM and eGon ERA5
  weather-cell queries.
- Both contracts now use the explicit lazy `oep_engine` fixture; their names identify
  the data source and live scope.

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

- Retained the simple and electromobility notebooks as slow live end-to-end smoke
  tests so the documented OEP workflows remain covered.
- Exclude both notebooks from the regular matrix and run them only in the dedicated,
  serialized OEP job with finite notebook and job timeouts.
- Notebook completion without a cell exception is the assertion; the electromobility
  input is local, while its renewable feed-in setup still uses the live OEP.

Focused runs of the implemented offline tests pass. Renamed live contracts are
discoverable with `pytest -m oep --collect-only`, but have not yet been executed
against the live OEP.

## Remaining work

#### 1. Vollständige Offline-Suite ohne OEP_TOKEN ausführen:
**Testsammlung prüfen:**
```
  python -m pytest --collect-only -q -m "not oep"
  python -m pytest --collect-only -q -m oep
```
**Schneller Offline-Testlauf**: Ohne OEP-Token und ohne Slow-Tests:
```
  env -u OEP_TOKEN python -m pytest -m "not oep" -x
```

**Vollständiger Offline-Testlauf:**
```
  env -u OEP_TOKEN python -m pytest \
  --runslow \
  -m "not oep" \
  --durations=20
```

#### 2. Alle Live-Contracts einmal mit Token ausführen

**Kleiner Live Contract:**
```
python -m pytest \
  -m "oep and not slow" \
  -x \
  --durations=20
```

**Langsamer OEP-Test:**
```
python -m pytest \
  --runslow \
  -m "oep and slow" \
  -x \
  --durations=20
```

Danach:
```unset OEP_TOKEN```

#### 3. Reguläre CI-Matrix auf -m "not oep" umstellen
- reguläre Matrix: -m "not oep", kein OEP-Token;
- Coverage-Job: ebenfalls -m "not oep";
- eigener OEP-Job: Linux, eine Python-Version, --runslow -m oep;

#### 4. Einen dedizierten, serialisierten OEP-Job anlegen
#### 5. Job- und Datenbank-Timeouts ergänzen
#### 6. Coverage prüfen und mit dem bisherigen Stand vergleichen
```
env -u OEP_TOKEN python -m coverage run \
  --source=edisgo \
  -m pytest \
  --runslow \
  -m "not oep"

python -m coverage report -m
```
- Configure separate regular and live OEP jobs in CI, including serialization and
  finite timeouts.
- Run and verify the retained live contracts with an OEP token supplied through the
  approved secret mechanism.
- Update the inventory status as each group is completed.

No OEP-derived datasets are currently committed here; the implemented offline tests
generate synthetic data in code. If local extracted data is added later, document its
source query, extraction date, scenario, licence, reduction steps and consuming tests
in this file.
