"""Setup"""
import os
import sys

from setuptools import find_packages, setup

if sys.version_info[:2] < (3, 8):
    error = (
        "eDisGo requires Python 3.8 or later (%d.%d detected)." % sys.version_info[:2]
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
    "demandlib",
    "networkx >= 2.5.0",
    "geopy >= 2.0.0",
    "pandas >= 1.2.0",
    "geopandas >= 0.9.0",
    "pyproj >= 3.0.0",
    "shapely >= 1.7.0",
    "pypsa >= 0.17.0, <= 0.20.1",
    "pyomo >= 6.0",
    "multiprocess",
    "workalendar",
    "sqlalchemy < 1.4.0",
    "geoalchemy2 < 0.7.0",
    "egoio >= 0.4.7",
    "matplotlib >= 3.3.0",
    "pypower",
    "scikit-learn",
    "pydot",
    "pygeos",
    "beautifulsoup4",
    "contextily",
    "descartes",
    "jupyter",
    "jupyterlab",
    "plotly",
    "dash==2.6.0",
    "jupyter_dash",
    "werkzeug==2.2.0",
]

dev_requirements = [
    "pytest",
    "pytest-notebook",
    "sphinx_rtd_theme",
    "sphinx-autodoc-typehints",
    "pre-commit",
    "black",
    "isort",
    "pyupgrade",
    "flake8",
    "pylint",
]

extras = {"dev": dev_requirements}

setup(
    name="eDisGo",
    version="0.2.0",
    packages=find_packages(),
    url="https://github.com/openego/eDisGo",
    license="GNU Affero General Public License v3.0",
    author="birgits, AnyaHe, khelfen, gplssm, nesnoj, jaappedersen, Elias, boltbeard, "
    "mltja",
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
