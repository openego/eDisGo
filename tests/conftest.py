import pytest
import os


def pytest_configure():
    pytest.ding0_test_network_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        "ding0_test_network")
