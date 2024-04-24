import logging
import os

import pytest

from edisgo.tools.logger import setup_logger


def check_file_output(filename, output):
    with open(filename) as file:
        last_line = file.readlines()[-1].split(" ")[3:]
        last_line = " ".join(last_line)
    assert last_line == output


def reset_loggers(filename):
    logger = logging.getLogger("edisgo")
    logger.handlers.clear()
    logger.propagate = True
    # try removing file - when run on github for Windows removing the file leads
    # to a PermissionError
    try:
        os.remove(filename)
    except PermissionError:
        pass


class TestClass:
    def test_setup_logger(self):
        filename = os.path.join(
            os.path.expanduser("~"), ".edisgo", "log", "test_log.log"
        )
        if os.path.exists(filename):
            os.remove(filename)

        setup_logger(
            loggers=[
                {"name": "root", "file_level": "debug", "stream_level": "debug"},
                {"name": "edisgo", "file_level": "debug", "stream_level": "debug"},
            ],
            file_name="test_log.log",
            log_dir="default",
        )

        logger = logging.getLogger("edisgo")
        # Test that edisgo logger writes to file.
        logger.debug("root")
        check_file_output(filename, "edisgo - DEBUG: root\n")
        # Test that root logger writes to file.
        logging.debug("root")
        check_file_output(filename, "root - DEBUG: root\n")

        reset_loggers(filename)

    @pytest.mark.runonlinux
    def test_setup_logger_2(self):
        """
        This test is only run on linux, as the log file is written to the user
        home directory, which is not allowed when tests are run on github.

        """

        # delete any existing log files
        log_files = [_ for _ in os.listdir(os.getcwd()) if ".log" in _]
        for log_file in log_files:
            os.remove(log_file)

        setup_logger(
            loggers=[
                {"name": "edisgo", "file_level": "debug", "stream_level": "debug"},
            ],
            reset_loggers=True,
            debug_message=True,
        )
        logger = logging.getLogger("edisgo")

        filename = [_ for _ in os.listdir(os.getcwd()) if ".log" in _][0]
        # Test that edisgo logger writes to file.
        logger.debug("edisgo")
        check_file_output(filename, "edisgo - DEBUG: edisgo\n")
        # Test that root logger doesn't write to file.
        logging.debug("root")
        check_file_output(filename, "edisgo - DEBUG: edisgo\n")

        reset_loggers(filename)
