import logging
import os

from edisgo.tools.logger import setup_logger


class TestClass:
    def test_setup_logger(self):
        def check_file_output(output):
            with open("edisgo.log") as file:
                last_line = file.readlines()[-1].split(" ")[3:]
                last_line = " ".join(last_line)
            assert last_line == output

        def reset_loggers():
            logger = logging.getLogger("edisgo")
            logger.propagate = True
            logger.handlers.clear()
            logger = logging.getLogger()
            logger.handlers.clear()

        if os.path.exists("edisgo.log"):
            os.remove("edisgo.log")

        setup_logger(
            loggers=[
                {"name": "root", "file_level": "debug", "stream_level": "debug"},
                {"name": "edisgo", "file_level": "debug", "stream_level": "debug"},
            ],
            file_name="edisgo.log",
        )

        logger = logging.getLogger("edisgo")
        # Test that edisgo logger writes to file.
        logger.debug("root")
        check_file_output("edisgo - DEBUG: root\n")
        # Test that root logger writes to file.
        logging.debug("root")
        check_file_output("root - DEBUG: root\n")

        # reset_loggers()

        setup_logger(
            loggers=[
                {"name": "edisgo", "file_level": "debug", "stream_level": "debug"},
            ],
            file_name="edisgo.log",
            reset_loggers=True,
            debug_message=True,
        )
        logger = logging.getLogger("edisgo")
        # Test that edisgo logger writes to file.
        logger.debug("edisgo")
        check_file_output("edisgo - DEBUG: edisgo\n")
        # Test that root logger doesn't writes to file.
        logging.debug("edisgo")
        check_file_output("edisgo - DEBUG: edisgo\n")

    @classmethod
    def teardown_class(cls):
        logger = logging.getLogger("edisgo")
        logger.handlers.clear()
        logger.propagate = True

        os.remove("edisgo.log")
