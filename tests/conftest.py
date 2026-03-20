import logging
import os

# Set matplotlib backend to non-interactive for tests (prevents TclError on Windows CI)
import matplotlib
import pytest

matplotlib.use("Agg")

# Suppress paramiko DEBUG keepalive messages that cause "I/O operation on
# closed file" logging errors when pytest workers shut down while the SSH
# tunnel background thread is still running.
logging.getLogger("paramiko").setLevel(logging.WARNING)

from edisgo.io.db import engine  # noqa: E402


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
        help="also run DB tests against local egon-data database",
    )
    parser.addoption(
        "--egon-data-config",
        action="store",
        default=None,
        help="path to egon-data YAML configuration file for local DB tests",
    )


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

    # OEP engine (always created)
    config._oep_engine = engine()
    pytest.engine = config._oep_engine  # backward compatibility

    # Local engine (only when --runlocal is passed)
    if config.getoption("--runlocal"):
        egon_config = (
            config.getoption("--egon-data-config")
            or os.environ.get("EGON_DATA_CONFIG")
            or os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "egon-data.configuration.yaml",
            )
        )
        config._local_engine = engine(path=egon_config)
    else:
        config._local_engine = None

    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "runonlinux: mark test to run only on linux")
    config.addinivalue_line("markers", "local: mark test that requires local DB")


@pytest.fixture
def db_engine(request):
    """Database engine fixture, parametrized by pytest_generate_tests."""
    return request.param


def pytest_generate_tests(metafunc):
    if "db_engine" in metafunc.fixturenames:
        engines = [metafunc.config._oep_engine]
        ids = ["oep"]
        if metafunc.config._local_engine is not None:
            engines.append(metafunc.config._local_engine)
            ids.append("local")
        metafunc.parametrize("db_engine", engines, ids=ids, indirect=True)


def pytest_sessionfinish(session, exitstatus):
    """Dispose engines and stop SSH tunnels after all tests."""
    for attr in ("_local_engine", "_oep_engine"):
        eng = getattr(session.config, attr, None)
        if eng is not None:
            ssh_server = getattr(eng, "_ssh_server", None)
            eng.dispose()
            if ssh_server is not None:
                ssh_server.stop()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--runonlinux"):
        skip_windows = pytest.mark.skip(reason="need --runonlinux option to run")
        for item in items:
            if "runonlinux" in item.keywords:
                item.add_marker(skip_windows)
