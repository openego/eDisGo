"""Setup"""

import os
import sys

from setuptools import find_packages, setup

if sys.version_info[:2] < (3, 9):
    error = (
        "eDisGo requires Python 3.9 or later (%d.%d detected)." % sys.version_info[:2]
    )
    sys.stderr.write(error + "\n")
    sys.exit(1)


def read(fname):
    """
    Read a text file.

    Parameters
    ----------
    fname : str or PurePath
        Path to file

    Returns
    -------
    str
        File content

    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


requirements = [
    "contextily < 1.7.0",
    "dash < 3.1.0",
    "demandlib < 0.3.0",
    "descartes < 1.2.0",
    "egoio >= 0.4.7, < 0.5.0",
    "geoalchemy2 < 0.7.0",
    "geopandas >= 0.12.0, < 1.1.0",
    "geopy >= 2.0.0, < 2.5.0",
    "jupyterlab < 4.5.0",
    "jupyter_dash < 0.5.0",
    "matplotlib >= 3.3.0, < 3.11.0",
    "multiprocess < 0.71.0",
    "networkx >= 2.5.0, < 3.5.0",
    # newer pandas versions don't work with specified sqlalchemy versions, but upgrading
    # sqlalchemy leads to new errors.. should be fixed at some point
    "numpy ==1.26.4",
    "pandas >= 1.4.0, < 2.2.0",
    "plotly < 6.0",
    "pydot < 4.1.0",
    "pygeos < 0.15.0",
    "pypower < 5.2.0",
    "pyproj >= 3.0.0, < 3.8.0",
    "pypsa == 0.26.2",
    "pyyaml < 6.1.0",
    "saio < 0.3.0",
    "scikit-learn < 1.3.0",
    "shapely >= 1.7.0, < 2.1.0",
    "sqlalchemy < 1.4.0",
    "sshtunnel < 0.5.0",
    "urllib3 < 2.6.0",
    "workalendar < 17.1.0",
    "astroid == 3.3.11",
]

dev_requirements = [
    "black < 25.2.0",
    "flake8 < 7.4.0",
    "isort < 6.1.0",
    "pre-commit < 4.3.0",
    "pylint < 3.4.0",
    "pytest < 8.5.0",
    "pytest-notebook < 0.11.0",
    "pyupgrade < 3.21.0",
    "sphinx < 8.3.0",
    "sphinx_rtd_theme >=0.5.2, < 3.1.0",
    "sphinx-autodoc-typehints < 3.2.0",
    "sphinx-autoapi < 3.7.0",
    "astroid == 3.3.11",
]

extras = {"dev": dev_requirements}

setup(
    name="eDisGo",
    version="0.3.0dev",
    packages=find_packages(),
    url="https://github.com/openego/eDisGo",
    license="GNU Affero General Public License v3.0",
    author=(
        "birgits, AnyaHe, khelfen, mltja, gplssm, nesnoj, jaappedersen, Elias, "
        "boltbeard"
    ),
    author_email="anya.heider@rl-institut.de",
    description="A python package for distribution network analysis and optimization",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require=extras,
    package_data={
        "edisgo": [
            os.path.join("config", "*.cfg"),
            os.path.join("equipment", "*.csv"),
        ]
    },
)
