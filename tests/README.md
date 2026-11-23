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

## CI test separation

Every push starts the existing four-entry matrix. A separate live OEP job runs
unless the triggering actor is `dependabot[bot]`:

- The three basic Linux/Windows entries and the Linux/Python 3.12 coverage entry
  all select `-m "not oep"` and receive no OEP token. CI retains this optional
  filter to make the intended exclusion explicit.
- The coverage entry uploads only non-OEP coverage to Coveralls. It does not wait
  for the live job, so an OEP failure does not prevent that report from uploading.
- The dedicated Linux/Python 3.12 OEP job selects
  `--runoep --runslow --runonlinux -m oep`,
  including the OEP example notebooks. It has a 140-minute timeout and runs
  without parallel pytest workers or automatic retries.
- Only the OEP job receives the `OEP_TOKEN` repository secret. Its first step fails
  with a clear message if the token is missing, empty or whitespace-only. This
  checks presence, not whether the server accepts the token.
- Dependabot runs skip only the OEP job; the regular matrix remains enabled.
  For other actors, a missing token still causes the explicit failure above.
- A fixed `edisgo-live-oep` concurrency group serializes OEP jobs across branches
  in this repository. `queue: max` allows waiting jobs to queue instead of replacing
  the previous pending job. The existing workflow-level rule still cancels an
  outdated run when a newer commit is pushed to the same branch.

When the OEP job runs, it remains a failing workflow check if a live test fails;
it is not marked as optional. This separation does not change repository
branch-protection rules.
This separation does not block all network access in regular tests. The plotting
notebook test uses the checked-out grid in `tests/data/ding0_test_network_1` rather
than downloading it from GitHub. Standalone notebook use retains a download
fallback when the repository data is unavailable.

In `coverage run --source=edisgo -m pytest -m "not oep"`, the first `-m` launches
pytest as a Python module; the second is pytest's marker selection.

The push trigger is unchanged. CI explicitly supplies `--runoep` in the dedicated
OEP job; ordinary local pytest runs skip OEP tests before fixture setup. Neither
`-m oep` nor `--runslow` alone enables live OEP access.

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
