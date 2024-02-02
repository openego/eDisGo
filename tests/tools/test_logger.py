import logging
import os

from edisgo.tools.logger import setup_logger


class TestClass:
    def test_setup_logger(self):
        def check_file_output(filename, output):
            with open(filename) as file:
                last_line = file.readlines()[-1].split(" ")[3:]
                last_line = " ".join(last_line)
            assert last_line == output

        def reset_loggers():
            logger = logging.getLogger("edisgo")
            logger.handlers.clear()
            logger.propagate = True

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

        reset_loggers()
        os.remove(filename)

        setup_logger(
            loggers=[
                {"name": "edisgo", "file_level": "debug", "stream_level": "debug"},
            ],
            reset_loggers=True,
            debug_message=True,
        )
        logger = logging.getLogger("edisgo")

        filename = [_ for _ in os.listdir(os.getcwd()) if ".log" in _]
        # if not 1 there are several log files, which shouldn't be the case and could
        # mess up the following tests
        assert len(filename) == 1
        filename = filename[0]
        # Test that edisgo logger writes to file.
        logger.debug("edisgo")
        check_file_output(filename, "edisgo - DEBUG: edisgo\n")
        # Test that root logger doesn't write to file.
        logging.debug("root")
        check_file_output(filename, "edisgo - DEBUG: edisgo\n")

        reset_loggers()
        os.remove(filename)
