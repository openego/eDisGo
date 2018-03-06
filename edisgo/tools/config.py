"""This file is part of eDisGo, a python package for distribution grid
analysis and optimization.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

eDisGo lives on github: https://github.com/openego/edisgo/
The documentation is available on RTD: http://edisgo.readthedocs.io

Based on code by oemof developing group

This module provides a highlevel layer for reading and writing config files.

"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/edisgo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


import os
from glob import glob
import shutil
import edisgo
import logging

logger = logging.getLogger('edisgo')

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
    package_path, 'config', 'config_system.cfg')
try:
    cfg.read(internal_config_file)
except:
    logger.exception('Internal config {} file not found.'.format(
        internal_config_file))


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
                logger.info('Config file {} not found, I will create a '
                            'default version'.format(config_file))
                make_directory(config_dir)
                shutil.copy(os.path.join(package_path, 'config', filename.
                                         replace('.cfg', '_default.cfg')),
                            config_file)
            else:
                message = 'Config file {} not found.'.format(config_file)
                logger.error(message)
                raise FileNotFoundError(message)

    if len(cfg.read(config_file)) == 0:
        message = 'Config file {} not found or empty.'.format(config_file)
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
    config_dir = get('user_dirs', 'config_dir')
    root_dir = get('user_dirs', 'root_dir')
    root_path = os.path.join(os.path.expanduser('~'), root_dir)
    config_path = os.path.join(root_path, config_dir)

    # root directory does not exist
    if not os.path.isdir(root_path):
        # create it
        logger.info('eDisGo root path {} not found, I will create it.'
                    .format(root_path))
        make_directory(root_path)

    # config directory does not exist
    if not os.path.isdir(config_path):
        # create it
        config_path = os.path.join(root_path, config_dir)
        make_directory(config_path)

        # copy default config files
        logger.info('eDisGo config path {} not found, I will create it.'
                    .format(config_path))

    # copy default config files if they don't exist
    internal_config_dir = os.path.join(package_path, 'config')
    for file in glob(os.path.join(internal_config_dir, '*.cfg')):
        filename = os.path.join(config_path,
                                os.path.basename(file).replace('_default', ''))
        if not os.path.isfile(filename):
            logger.info('I will create a default config file {} in {}'
                        .format(file, config_path))
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
        os.mkdir(directory)
        logger.info('Path {} not found, I will create it.'
                    .format(directory))

