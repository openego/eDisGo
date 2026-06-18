import json
import os

# Set matplotlib backend to non-interactive for tests (prevents TclError on Windows CI)
import matplotlib
import pytest

matplotlib.use("Agg")

from edisgo.io.db import engine


def pytest_configure(config):
    # small self constructed ding0 grid with only 9 LV grids used for general testing
    pytest.ding0_test_network_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "data/ding0_test_network_1"
    )
    # real ding0 grid without georeference in LV used to test import of open_ego data
    # from oedb
    pytest.ding0_test_network_2_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "data/ding0_test_network_2"
    )
    # real ding0 grid from newer version of ding0 with georeferenced LV used to test
    # import of egon_data data
    pytest.ding0_test_network_3_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "data/ding0_test_network_3"
    )

    pytest.simbev_example_scenario_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "data/simbev_example_scenario"
    )

    pytest.tracbev_example_scenario_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "data/tracbev_example_scenario"
    )

    # Path to the egon-data database configuration YAML used by the local
    # (SSH-tunneled) backend tests. Overridable via the EGON_DATA_CONFIG
    # environment variable so it does not have to live in a fixed location.
    pytest.egon_data_config_yml = os.environ.get(
        "EGON_DATA_CONFIG",
        os.path.expanduser("~/.ssh/egon-data.configuration.yaml"),
    )

    pytest.engine = engine()

    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "local: mark test as local to run")
    config.addinivalue_line("markers", "runonlinux: mark test to run only on linux")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runonlinux",
        action="store_true",
        default=False,
        help="run tests that only work on linux",
    )
    parser.addoption(
        "--runlocal",
        action="store_true",
        default=False,
        help="run tests that only work locally",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--runlocal"):
        skip_local = pytest.mark.skip(reason="need --runlocal option to run")
        for item in items:
            if "local" in item.keywords:
                item.add_marker(skip_local)
    if not config.getoption("--runonlinux"):
        skip_windows = pytest.mark.skip(reason="need --runonlinux option to run")
        for item in items:
            if "runonlinux" in item.keywords:
                item.add_marker(skip_windows)

    _reorder_slowest_first(config, items)


def _reorder_slowest_first(config, items):
    """
    Reorder collected tests so the slowest run first ("longest processing time
    first" scheduling). With ``pytest-xdist`` the controller dispatches tests in
    collection order, so putting the long-running tests up front keeps them from
    ending up as a tail that a single worker runs alone while the others idle.

    Test durations are read from the ``.test_durations`` file produced by
    ``pytest-split``'s ``--store-durations`` option. Without that file (e.g. a
    first run) the collection order is left untouched, so durations are simply
    measured and stored for the next run. Tests with no recorded duration (new
    tests) are treated as slow so they run early rather than risk a long tail.

    The reordering is deterministic (it depends only on the shared durations
    file and the already-identical collection order), which xdist requires —
    all workers must collect tests in the same order.
    """
    durations_path = config.rootpath / ".test_durations"
    if not durations_path.is_file():
        return

    try:
        durations = json.loads(durations_path.read_text())
    except (json.JSONDecodeError, OSError):
        return

    if not durations:
        return

    # Unknown tests sort as the slowest known test -> run early.
    fallback = max(durations.values())
    items.sort(key=lambda item: durations.get(item.nodeid, fallback), reverse=True)


@pytest.fixture(scope="session")
def engine_local():
    """
    SQLAlchemy engine for the local egon-data database via an SSH tunnel.

    Built from ``pytest.egon_data_config_yml`` with ``ssh=True``, i.e. it
    exercises the local egon-data backend (credential parsing -> SSH tunnel ->
    ``postgresql+psycopg2``) rather than the remote OpenEnergyPlatform. Only
    requested by tests marked ``@pytest.mark.local`` (run with ``--runlocal``),
    so it is never created in CI, which has no local DB / SSH access.
    """
    # sshtunnel 0.4.0 still references ``paramiko.DSSKey`` (DSA), which paramiko
    # >= 3 removed. Alias it to a present key class so importing/using the SSH
    # tunnel works with modern paramiko; ed25519/RSA keys are loaded by their
    # own key classes, so the alias is never used to parse a real key.
    import paramiko

    if not hasattr(paramiko, "DSSKey"):
        paramiko.DSSKey = paramiko.RSAKey

    eng, tunnel = engine(
        path=pytest.egon_data_config_yml, ssh=True, return_tunnel=True
    )
    yield eng

    # Stop the SSH tunnel at session teardown so its paramiko keepalive thread
    # terminates before pytest closes its output streams. Otherwise the still
    # running thread keeps logging at interpreter shutdown and floods the output
    # with "ValueError: I/O operation on closed file" tracebacks.
    eng.dispose()
    tunnel.stop()
