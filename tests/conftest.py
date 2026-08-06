import os

# Set matplotlib backend to non-interactive for tests (prevents TclError on Windows CI)
import matplotlib
import pytest

from sqlalchemy.engine import Engine

matplotlib.use("Agg")

from edisgo.io.db import default_config_path, engine


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

    pytest.egon_data_config_yml = default_config_path()

    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "local: mark test as local to run")
    config.addinivalue_line("markers", "runonlinux: mark test to run only on linux")
    config.addinivalue_line(
        "markers", "oep: mark test as intentionally accessing the live OEP"
    )

    if config.getoption("--runlocal"):
        pytest.engine_local = engine(path=pytest.egon_data_config_yml, ssh=True)


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


def _is_oep_engine(db_engine):
    return (
        db_engine.url.host == "openenergyplatform.org"
        or db_engine.url.drivername.endswith("+oedialect")
    )


@pytest.fixture
def oep_engine(request):
    """Create an OEP engine only for a test that explicitly requests it."""

    if request.node.get_closest_marker("oep") is None:
        pytest.fail(
            "The oep_engine fixture may only be used by tests marked with "
            "@pytest.mark.oep.",
            pytrace=False,
        )

    # OEP tests assert against live OEP data. Force the OEP instead of
    # auto-selecting a potentially different local eGon database.
    db_engine = engine(ssh=False)

    yield db_engine

    db_engine.dispose()


@pytest.fixture(autouse=True)
def prevent_unmarked_oep_access(request, monkeypatch):
    """Prevent tests without the oep marker from connecting to the OEP."""

    oep_access_allowed = request.node.get_closest_marker("oep") is not None

    original_connect = Engine.connect
    original_raw_connection = Engine.raw_connection

    def check_access(engine):
        if _is_oep_engine(engine) and not oep_access_allowed:
            pytest.fail(
                "Unmarked test attempted to connect to the live OEP: "
                f"{request.node.nodeid}",
                pytrace=False,
            )

    def guarded_connect(engine, *args, **kwargs):
        check_access(engine)
        return original_connect(engine, *args, **kwargs)

    def guarded_raw_connection(engine, *args, **kwargs):
        check_access(engine)
        return original_raw_connection(engine, *args, **kwargs)

    monkeypatch.setattr(Engine, "connect", guarded_connect)
    monkeypatch.setattr(Engine, "raw_connection", guarded_raw_connection)
