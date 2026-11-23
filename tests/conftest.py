import importlib.util
import os
import re

from pathlib import Path

# Set matplotlib backend to non-interactive for tests (prevents TclError on Windows CI)
import matplotlib
import pytest

from oedialect.dialect import OEDialect
from sqlalchemy import event
from sqlalchemy.engine import Engine

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

    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "runonlinux: mark test to run only on linux")
    config.addinivalue_line(
        "markers", "oep: mark test as intentionally accessing the live OEP"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--runoep",
        action="store_true",
        default=False,
        help="allow tests that access the live OEP",
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runonlinux",
        action="store_true",
        default=False,
        help="run tests that only work on linux",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runoep"):
        skip_oep = pytest.mark.skip(reason="need --runoep option to run live OEP tests")
        for item in items:
            if item.get_closest_marker("oep") is not None:
                item.add_marker(skip_oep)
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


def _is_oep_connection(dialect, connect_params):
    return (
        isinstance(dialect, OEDialect)
        or connect_params.get("host") == "openenergyplatform.org"
    )


@pytest.fixture
def oep_engine(request):
    """Require a token before creating an explicitly requested OEP engine."""

    if request.node.get_closest_marker("oep") is None:
        pytest.fail(
            "The oep_engine fixture may only be used by tests marked with "
            "@pytest.mark.oep.",
            pytrace=False,
        )

    _require_oep_test_token()

    # OEP tests assert against live OEP data. Force the OEP instead of
    # auto-selecting a potentially different local eGon database.
    db_engine = engine(ssh=False)

    yield db_engine

    db_engine.dispose()


def _require_oep_test_token():
    """Check the same token sources and precedence as edisgo.io.db.engine."""
    if "OEP_TOKEN" in os.environ:
        token = os.environ["OEP_TOKEN"]
    else:
        spec = importlib.util.find_spec("edisgo")
        token_path = Path(spec.origin).resolve().parent / "config" / "OEP_TOKEN.txt"
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            pytest.fail(
                "The oep_engine fixture requires an OEP token. Set OEP_TOKEN or "
                "provide a readable edisgo/config/OEP_TOKEN.txt file.",
                pytrace=False,
            )

    if re.fullmatch(r"[a-f0-9]{40}", token) is None:
        pytest.fail(
            "Invalid OEP token for the oep_engine fixture: expected exactly 40 "
            "lowercase hexadecimal characters. OEP_TOKEN takes precedence over "
            "the token file, even when empty. The token value is not displayed.",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def prevent_unmarked_oep_access(request):
    """Prevent tests without the oep marker from connecting to the OEP.

    The guard rejects each new DBAPI connection to the OEP before it is opened.
    That covers every way SQLAlchemy connects, including table reflection, which
    bypasses Engine.connect. OEP engines use a NullPool, so no OEP connection is
    reused from the pool without passing the guard.
    """

    if request.node.get_closest_marker("oep") is not None:
        yield
        return

    def reject_oep_connection(dialect, connection_record, cargs, connect_params):
        if _is_oep_connection(dialect, connect_params):
            pytest.fail(
                "Unmarked test attempted to connect to the live OEP: "
                f"{request.node.nodeid}",
                pytrace=False,
            )

    event.listen(Engine, "do_connect", reject_oep_connection)
    try:
        yield
    finally:
        event.remove(Engine, "do_connect", reject_oep_connection)
