import logging
import os
import sys

from datetime import datetime

from edisgo.tools import config as cfg_edisgo


def setup_logger(
    file_name=None,
    log_dir=None,
    loggers=None,
    stream_output=sys.stdout,
    debug_message=False,
    reset_loggers=False,
):
    """
    Setup different loggers with individual logging levels and where to write output.

    The following table from python 'Logging Howto' shows you when which logging level
    is used.

    .. tabularcolumns:: |l|L|

    +--------------+---------------------------------------------+
    | Level        | When it's used                              |
    +==============+=============================================+
    | ``DEBUG``    | Detailed information, typically of interest |
    |              | only when diagnosing problems.              |
    +--------------+---------------------------------------------+
    | ``INFO``     | Confirmation that things are working as     |
    |              | expected.                                   |
    +--------------+---------------------------------------------+
    | ``WARNING``  | An indication that something unexpected     |
    |              | happened, or indicative of some problem in  |
    |              | the near future (e.g. 'disk space low').    |
    |              | The software is still working as expected.  |
    +--------------+---------------------------------------------+
    | ``ERROR``    | Due to a more serious problem, the software |
    |              | has not been able to perform some function. |
    +--------------+---------------------------------------------+
    | ``CRITICAL`` | A serious error, indicating that the program|
    |              | itself may be unable to continue running.   |
    +--------------+---------------------------------------------+

    Parameters
    ----------
    file_name : str or None
        Specifies file name of file logging information is written to. Possible options
        are:

        * None (default)
            Saves log file with standard name `%Y_%m_%d-%H:%M:%S_edisgo.log`.
        * str
            Saves log file with the specified file name.

    log_dir : str or None
        Specifies directory log file is saved to. Possible options are:

        * None (default)
            Saves log file in current working directory.
        * "default"
            Saves log file into directory configured in the configs.
        * str
            Saves log file into the specified directory.

    loggers : None or list(dict)

        * None
            Configuration as shown in the example below is used. Configures root logger
            with file and stream level warning and the edisgo logger with file and
            stream level debug.
        * list(dict)
            List of dicts with the logger configuration. Each dictionary must contain
            the following keys and corresponding values:

            * 'name'
                Specifies name of the logger as string, e.g. 'root' or 'edisgo'.
            * 'file_level'
                Specifies file logging level. Possible options are:

                * "debug"
                    Logs logging messages with logging level logging.DEBUG and above.
                * "info"
                    Logs logging messages with logging level logging.INFO and above.
                * "warning"
                    Logs logging messages with logging level logging.WARNING and above.
                * "error"
                    Logs logging messages with logging level logging.ERROR and above.
                * "critical"
                    Logs logging messages with logging level logging.CRITICAL.
                * None
                    No logging messages are logged.
            * 'stream_level'
                Specifies stream logging level. Possible options are the same as for
                `file_level`.

    stream_output : stream
        Default sys.stdout is used. sys.stderr is also possible.

    debug_message : bool
        If True the handlers of every configured logger is printed.

    reset_loggers : bool
        If True the handlers of all loggers are cleared before configuring the loggers.
        Only use if you know what you do, it could be dangerous.

    Examples
    --------
    >>> setup_logger(
    >>>     loggers=[
    >>>         {"name": "root", "file_level": "warning", "stream_level": "warning"},
    >>>         {"name": "edisgo", "file_level": "info", "stream_level": "info"}
    >>>     ]
    >>> )

    """

    def create_dir(dir_path):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    def get_default_root_dir():
        dir_path = str(cfg_edisgo.get("user_dirs", "root_dir"))
        return os.path.join(os.path.expanduser("~"), dir_path)

    def create_home_dir():
        dir_path = get_default_root_dir()
        create_dir(dir_path)

    cfg_edisgo.load_config("config_system.cfg")

    if file_name is None:
        now = datetime.now()
        file_name = now.strftime("%Y_%m_%d-%H:%M:%S_edisgo.log")

    if log_dir == "default":
        create_home_dir()
        log_dir = os.path.join(
            get_default_root_dir(), cfg_edisgo.get("user_dirs", "log_dir")
        )
    create_dir(log_dir)

    if log_dir is not None:
        file_name = os.path.join(log_dir, file_name)

    if reset_loggers:
        existing_loggers = [logging.getLogger()]  # get the root logger
        existing_loggers = existing_loggers + [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]

        for logger in existing_loggers:
            for handler in logger.handlers:
                if not isinstance(handler, logging.NullHandler):
                    if debug_message:
                        print(f"Removed {handler} of Logger: {logger}")
                    logger.removeHandler(handler)

    loglevel_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
        None: logging.CRITICAL + 1,
    }

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
    )
    stream_formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")

    if loggers is None:
        loggers = [
            {"name": "root", "file_level": "warning", "stream_level": "warning"},
            {"name": "edisgo", "file_level": "info", "stream_level": "info"},
        ]

    for logger_config in loggers:
        logger_name = logger_config["name"]
        logger_file_level = loglevel_dict[logger_config["file_level"]]
        logger_stream_level = loglevel_dict[logger_config["stream_level"]]

        if logger_name == "root":
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(logger_name)
            logger.propagate = False

        # clear existing handlers for the logger
        logger.handlers.clear()

        if logger_file_level < logger_stream_level:
            logger.setLevel(logger_file_level)
        else:
            logger.setLevel(logger_stream_level)

        if logger_file_level < logging.CRITICAL + 1:
            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(logger_file_level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        if logger_stream_level < logging.CRITICAL + 1:
            console_handler = logging.StreamHandler(stream=stream_output)
            console_handler.setLevel(logger_stream_level)
            console_handler.setFormatter(stream_formatter)
            logger.addHandler(console_handler)

        if debug_message:
            print(f"Handlers of logger {logger_name}: {logger.handlers}")
