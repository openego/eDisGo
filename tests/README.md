# Tests

## Installing test dependencies

Install eDisGo with its development dependencies:

```bash
python -m pip install -e '.[dev]'
```

## Running tests

Run the regular test suite without live OEP tests:

```bash
python -m pytest -m "not oep"
```

Run only live OEP tests:

```bash
python -m pytest -m oep
```

Include tests marked as slow:

```bash
python -m pytest --runslow
```

Some live OEP tests are also marked slow. To execute all live OEP tests:

```bash
python -m pytest --runslow -m oep
```

## CI test separation

The CI matrix runs four jobs in parallel. Three regular Linux and Windows entries
run `pytest -m "not oep"` without an OEP token. The Linux/Python 3.12 coverage
entry runs the complete test suite, including live OEP tests, with the token
supplied through the GitHub secret. It is the only matrix entry with live OEP
access, has a 140-minute job timeout and uploads its result to Coveralls.

The coverage command has no pytest marker expression: `coverage run -m pytest`
uses `-m pytest` to launch the Python module and does not select a pytest marker.

## Test markers

- `oep`: intentionally accesses the live Open Energy Platform.
- `slow`: excluded unless `--runslow` is supplied.
- `local`: requires the local test configuration and `--runlocal`.
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

Do not commit an OEP token. See the contributing documentation for local token
configuration.

## OEP test inventory and local data

The detailed inventory of OEP-dependent tests, accessed tables, columns, dimensions
and treatments is available in
[`data/oep/oep_test_inventory.md`](data/oep/oep_test_inventory.md). Documentation
for committed OEP-derived test datasets is stored in
[`data/oep/README.md`](data/oep/README.md).
