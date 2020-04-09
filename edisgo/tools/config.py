"""This file is part of eDisGo, a python package for distribution network
analysis and optimization.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

eDisGo lives on github: https://github.com/openego/edisgo/
The documentation is available on RTD: http://edisgo.readthedocs.io

Based on code by oemof developing group

This module provides a highlevel layer for reading and writing config files.

"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/edisgo/blob/master/LICENSE"
__author__ = "nesnoj, gplssm"


import os
from glob import glob
import shutil
import edisgo
import logging
import datetime

logger = logging.getLogger("edisgo")

try:
    import configparser as cp
except:
    # to be compatible with Python2.7
    import ConfigParser as cp

cfg = cp.RawConfigParser()
_loaded = False

# load config dirs
package_path = edisgo.__path__[0]
internal_config_file = os.path.join(
    package_path, "config", "config_system.cfg"
)
try:
    cfg.read(internal_config_file)
except:
    logger.exception(
        "Internal config {} file not found.".format(internal_config_file)
    )


class Config:
    """
    Container for all configurations.

    Parameters
    -----------
    config_path : None or :obj:`str` or :obj:`dict`
        Path to the config directory. Options are:

        * None
          If `config_path` is None configs are loaded from the edisgo
          default config directory ($HOME$/.edisgo). If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.
        * :obj:`str`
          If `config_path` is a string configs will be loaded from the
          directory specified by `config_path`. If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.
        * :obj:`dict`
          A dictionary can be used to specify different paths to the
          different config files. The dictionary must have the following
          keys:
          * 'config_db_tables'
          * 'config_grid'
          * 'config_grid_expansion'
          * 'config_timeseries'

          Values of the dictionary are paths to the corresponding
          config file. In contrast to the other two options the directories
          and config files must exist and are not automatically created.

        Default: None.

    Notes
    -----
    The Config object can be used like a dictionary. See example on how to use
    it.

    Examples
    --------
    Create Config object from default config files

    >>> from edisgo.tools.config import Config
    >>> config = Config()

    Get reactive power factor for generators in the MV network

    >>> config['reactive_power_factor']['mv_gen']

    """

    def __init__(self, **kwargs):
        self._data = self._load_config(kwargs.get("config_path", None))

    @staticmethod
    def _load_config(config_path=None):
        """
        Load config files.

        Parameters
        -----------
        config_path : None or :obj:`str` or dict
            See class definition for more information.

        Returns
        -------
        :obj:`collections.OrderedDict`
            eDisGo configuration data from config files.

        """

        config_files = [
            "config_db_tables",
            "config_grid",
            "config_grid_expansion",
            "config_timeseries",
            "config_opf_julia",
        ]

        # load configs
        if isinstance(config_path, dict):
            for conf in config_files:
                load_config(
                    filename="{}.cfg".format(conf),
                    config_dir=config_path[conf],
                    copy_default_config=False,
                )
        else:
            for conf in config_files:
                load_config(
                    filename="{}.cfg".format(conf), config_dir=config_path
                )

        config_dict = cfg._sections

        # convert numeric values to float
        for sec, subsecs in config_dict.items():
            for subsec, val in subsecs.items():
                # try str -> float conversion
                try:
                    config_dict[sec][subsec] = float(val)
                except:
                    pass

        # convert to time object
        config_dict["demandlib"]["day_start"] = datetime.datetime.strptime(
            config_dict["demandlib"]["day_start"], "%H:%M"
        )
        config_dict["demandlib"]["day_start"] = datetime.time(
            config_dict["demandlib"]["day_start"].hour,
            config_dict["demandlib"]["day_start"].minute,
        )
        config_dict["demandlib"]["day_end"] = datetime.datetime.strptime(
            config_dict["demandlib"]["day_end"], "%H:%M"
        )
        config_dict["demandlib"]["day_end"] = datetime.time(
            config_dict["demandlib"]["day_end"].hour,
            config_dict["demandlib"]["day_end"].minute,
        )

        return config_dict

    def __getitem__(self, key1, key2=None):
        if key2 is None:
            try:
                return self._data[key1]
            except:
                raise KeyError(
                    "Config does not contain section {}.".format(key1)
                )
        else:
            try:
                return self._data[key1][key2]
            except:
                raise KeyError(
                    "Config does not contain value for {} or "
                    "section {}.".format(key2, key1)
                )

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def load_config(filename, config_dir=None, copy_default_config=True):
    """
    Loads the specified config file.

    Parameters
    -----------
    filename : :obj:`str`
        Config file name, e.g. 'config_grid.cfg'.
    config_dir : :obj:`str`, optional
        Path to config file. If None uses default edisgo config directory
        specified in config file 'config_system.cfg' in section 'user_dirs'
        by subsections 'root_dir' and 'config_dir'. Default: None.
    copy_default_config : Boolean
        If True copies a default config file into `config_dir` if the
        specified config file does not exist. Default: True.

    """
    if not config_dir:
        config_file = os.path.join(get_default_config_path(), filename)
    else:
        config_file = os.path.join(config_dir, filename)

        # config file does not exist -> copy default
        if not os.path.isfile(config_file):
            if copy_default_config:
                logger.info(
                    "Config file {} not found, I will create a "
                    "default version".format(config_file)
                )
                make_directory(config_dir)
                shutil.copy(
                    os.path.join(
                        package_path,
                        "config",
                        filename.replace(".cfg", "_default.cfg"),
                    ),
                    config_file,
                )
            else:
                message = "Config file {} not found.".format(config_file)
                logger.error(message)
                raise FileNotFoundError(message)

    if len(cfg.read(config_file)) == 0:
        message = "Config file {} not found or empty.".format(config_file)
        logger.error(message)
        raise FileNotFoundError(message)
    global _loaded
    _loaded = True


def get(section, key):
    """
    Returns the value of a given key of a given section of the main
    config file.

    Parameters
    -----------
    section : :obj:`str`
    key : :obj:`str`

    Returns
    --------
    float or int or Boolean or str
        The value which will be casted to float, int or boolean.
        If no cast is successful, the raw string is returned.

    """
    if not _loaded:
        pass
    try:
        return cfg.getfloat(section, key)
    except:
        try:
            return cfg.getint(section, key)
        except:
            try:
                return cfg.getboolean(section, key)
            except:
                return cfg.get(section, key)


def get_default_config_path():
    """
    Returns the basic edisgo config path. If it does not yet exist it creates
    it and copies all default config files into it.

    Returns
    --------
    :obj:`str`
        Path to default edisgo config directory specified in config file
        'config_system.cfg' in section 'user_dirs' by subsections 'root_dir'
        and 'config_dir'.

    """
    config_dir = get("user_dirs", "config_dir")
    root_dir = get("user_dirs", "root_dir")
    root_path = os.path.join(os.path.expanduser("~"), root_dir)
    config_path = os.path.join(root_path, config_dir)

    # root directory does not exist
    if not os.path.isdir(root_path):
        # create it
        logger.info(
            "eDisGo root path {} not found, I will create it.".format(
                root_path
            )
        )
        make_directory(root_path)

    # config directory does not exist
    if not os.path.isdir(config_path):
        # create it
        config_path = os.path.join(root_path, config_dir)
        make_directory(config_path)

        # copy default config files
        logger.info(
            "eDisGo config path {} not found, I will create it.".format(
                config_path
            )
        )

    # copy default config files if they don't exist
    internal_config_dir = os.path.join(package_path, "config")
    for file in glob(os.path.join(internal_config_dir, "*.cfg")):
        filename = os.path.join(
            config_path, os.path.basename(file).replace("_default", "")
        )
        if not os.path.isfile(filename):
            logger.info(
                "I will create a default config file {} in {}".format(
                    file, config_path
                )
            )
            shutil.copy(file, filename)
    return config_path


def make_directory(directory):
    """
    Makes directory if it does not exist.

    Parameters
    -----------
    directory : :obj:`str`
        Directory path

    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
        logger.info("Path {} not found, I will create it.".format(directory))
