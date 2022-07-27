"""Setup"""
import os
import sys

from setuptools import find_packages, setup

if sys.version_info[:2] < (3, 7):
    error = (
        "eDisGo requires Python 3.7 or later (%d.%d detected)." % sys.version_info[:2]
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
    "pypsa >= 0.17.0",
    "pyomo >= 6.0",
    "multiprocess",
    "workalendar",
    "sqlalchemy < 1.4.0",
    "geoalchemy2 < 0.7.0",
    "egoio >= 0.4.7",
    "matplotlib >= 3.3.0",
    "pypower",
    "sklearn",
    "pydot",
    "pygeos",
]

geo_plot_requirements = [
    "contextily",
    "descartes",
    "plotly",
    "dash==2.0.0",
    "werkzeug==2.0.3",
]
examples_requirements = [
    "jupyter",
    "jupyterlab",
    "plotly",
    "dash==2.0.0",
    "jupyter_dash",
    "werkzeug==2.0.3",
]
dev_requirements = [
    "pytest",
    "jupyter_contrib_nbextensions",
    "sphinx_rtd_theme",
    "sphinx-autodoc-typehints",
    "pre-commit",
    "black",
    "isort",
    "pyupgrade",
    "flake8",
    "pylint",
]
full_requirements = list(
    set(geo_plot_requirements + examples_requirements + dev_requirements)
)

extras = {
    "geoplot": geo_plot_requirements,
    "examples": examples_requirements,
    "dev": dev_requirements,
    "full": full_requirements,
}

setup(
    name="eDisGo",
    version="0.2.0dev",
    packages=find_packages(),
    url="https://github.com/openego/eDisGo",
    license="GNU Affero General Public License v3.0",
    author="birgits, AnyaHe, khelfen, gplssm, nesnoj, jaappedersen, Elias, boltbeard",
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
