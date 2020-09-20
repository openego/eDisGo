import pytest
import os


def pytest_configure(config):
    pytest.ding0_test_network_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        "ding0_test_network_1")

    pytest.ding0_test_network_2_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        "ding0_test_network_2")

    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
