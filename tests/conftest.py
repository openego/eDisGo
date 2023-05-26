import os

import pytest

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

    pytest.egon_data_config_yml = os.path.join(
        os.path.realpath(os.path.dirname(os.path.dirname(__file__))),
        "egon-data.configuration.yaml",
    )

    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "local: mark test as local to run")

    if config.getoption("--runlocal"):
        pytest.engine = engine(path=pytest.egon_data_config_yml, ssh=True)


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
