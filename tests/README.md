# Tests

## Installing test dependencies

Install eDisGo with its development dependencies:

```bash
python -m pip install -e '.[dev]'
```

## Running tests

Run these commands from the repository root with your test environment activated.

Run the regular test suite (live OEP tests are skipped by default):

```bash
python -m pytest
```

Include regular tests marked as slow:

```bash
python -m pytest --runslow
```

Run only live OEP tests, including those marked as slow:

```bash
python -m pytest --runoep --runslow -m oep
```

The `oep` marker identifies live tests. `--runoep` permits them to execute, while
pytest's built-in `-m oep` filter selects only those tests. Without the filter,
`--runoep --runslow` runs both regular and live tests.

`-m "not oep"` is optional for regular runs: without `--runoep`, OEP tests are
already skipped before fixture setup. The filter reports them as deselected
instead of skipped; neither approach executes their test bodies or fixtures.
Both approaches still collect tests and import their modules.

## CI test execution

Every push starts the four-entry `build` matrix:

- The three basic Linux/Windows entries select `-m "not oep"` and receive no
  OEP token.
- The Linux/Python 3.12 coverage entry runs regular and live OEP tests together
  under coverage.py using `--runoep --runslow --runonlinux`, without a pytest
  marker filter. After the complete selection passes, it uploads one combined
  report to Coveralls. There is no separate OEP or coverage-merging job.
- Only the coverage entry receives the `OEP_TOKEN` repository secret. Its first
  step fails clearly if the token is missing, empty or whitespace-only. This
  checks presence, not whether the server accepts the token.
- The coverage entry retains its 140-minute timeout and runs without parallel
  pytest workers or automatic retries.
- OEP-enabled coverage entries share the `edisgo-live-oep` concurrency group
  across branches. This serializes the entire coverage entry, including regular
  tests and setup, not just individual OEP requests. Basic entries use distinct
  groups and can run in parallel. `queue: max` lets pending coverage entries wait.
- Dependabot runs execute only regular tests in all four entries. The coverage
  entry does not require an OEP token or upload a partial report to Coveralls.
- Workflow-level concurrency still cancels an outdated run when a newer commit
  is pushed to the same branch.

An OEP failure now fails the coverage entry and prevents its Coveralls upload;
the basic jobs remain independent. This workflow does not change repository
branch-protection rules.

The basic selection does not block all network access. The plotting notebook
uses the checked-out grid in `tests/data/ding0_test_network_1` rather than
downloading it from GitHub. Standalone use retains a download fallback.

In `python -m pytest -m "not oep"`, Python's first `-m` launches pytest as a
module; the second is pytest's marker selection.

The push trigger is unchanged. CI explicitly supplies `--runoep` in the coverage
entry; ordinary local pytest runs still skip OEP tests before fixture setup.
Neither `-m oep` nor `--runslow` alone enables live OEP access.

## Test markers

- `oep`: intentionally accesses the live Open Energy Platform; skipped unless
  `--runoep` is supplied.
- `slow`: excluded unless `--runslow` is supplied.
- `runonlinux`: requires `--runonlinux`.

Tests without the `oep` marker are prevented from connecting to the live OEP by an
automatic runtime guard.

Tests that explicitly require an OEP engine must:

1. have the `pytest.mark.oep` marker;
2. request the `oep_engine` fixture.

```python
@pytest.mark.oep
def test_oep_import(oep_engine):
    result = import_from_oep(engine=oep_engine)
```

Do not commit an OEP token. Live OEP tests read it from `edisgo/config/OEP_TOKEN.txt`
or the `OEP_TOKEN` environment variable, see "OEP token" in
`doc/userguide/data_sources.rst`.

Before creating an engine, the `oep_engine` fixture checks that a token is present
and consists of exactly 40 lowercase hexadecimal characters. It uses `OEP_TOKEN`
when set; otherwise it reads `edisgo/config/OEP_TOKEN.txt` from the loaded package.
An empty or malformed environment variable is an error even if the file exists.
Errors never include the token value. This validates format, not server acceptance.

The CI presence check still runs first, before dependency installation. The fixture
check also protects local runs and checks the token format. Neither check makes a
database request. Legacy tests using `session_scope()` do not request this fixture,
but their connection now uses eDisGo's token-aware OEP engine. Notebook tests
create their engines in a separate kernel and also bypass the fixture precheck.
CI checks token presence for the whole live job; local legacy/notebook runs do not
have the additional fixture precheck.

## OEP test inventory and local data

The detailed inventory of OEP-dependent tests, accessed tables, columns, dimensions
and treatments is available in
[`data/oep/oep_test_inventory.md`](data/oep/oep_test_inventory.md). Documentation
for committed OEP-derived test datasets is stored in
[`data/oep/README.md`](data/oep/README.md).
