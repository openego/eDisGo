# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Setup"""

import os

from setuptools import find_packages, setup


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
    "contextily < 1.8.0",
    "dash < 4.5.0",
    # eDisGo builds the standard load profiles with get_scaled_power_profiles(),
    # which demandlib added in 0.2.2
    "demandlib >= 0.2.2, < 0.3.0",
    "descartes < 1.2.0",
    "egoio >= 0.4.7, < 0.5.0",
    "geoalchemy2 >= 0.17.0, < 0.21.0",
    "geopandas >= 0.12.0, < 1.2.0",
    "geopy >= 2.0.0, < 2.6.0",
    "jupyterlab < 4.7.0",
    "jupyter_dash < 0.5.0",
    "matplotlib >= 3.3.0, < 3.12.0",
    "multiprocess < 0.71.0",
    "networkx >= 2.5.0, < 3.7.0",
    "numpy >= 2.0.0, < 2.5.0",
    "pandas >= 3.0.0, < 3.1.0",
    # sshtunnel 0.4.0 - the latest release - reads paramiko.DSSKey, which
    # paramiko removed in version 4
    "paramiko < 4.0",
    "plotly >= 6.0, < 8.0",
    "pydot < 4.1.0",
    "pypower < 5.2.0",
    "pyproj >= 3.0.0, < 3.8.0",
    # eDisGo uses PyPSA internals (Network.static/dynamic and the component
    # defaults) to export its grids, so the range is kept narrow
    "pypsa >= 1.3.0, < 1.4.0",
    "pyyaml < 6.1.0",
    "scikit-learn < 1.10.0",
    "scipy < 1.18.0",
    "shapely >= 1.7.0, < 2.2.0",
    "sqlalchemy >= 2.0.0, < 2.1.0",
    "sshtunnel < 0.5.0",
    "urllib3 < 2.8.0",
    "workalendar < 17.1.0",
    "astroid == 4.0.3",
]

dev_requirements = [
    "ruff < 0.16.0",
    "pre-commit < 4.7.0",
    "pylint < 4.1.0",
    "pytest < 9.2.0",
    "nbclient < 0.12.0",
    "pytest-xdist < 4.0.0",
    "pytest-rerunfailures < 17.0.0",
    "pytest-html < 5.0.0",
    "pytest-metadata < 4.0.0",
    "anyio < 5.0.0",
    "dash < 4.5.0",
    "pluggy < 2.0.0",
    "pyupgrade < 3.22.0",
    "sphinx < 9.2.0",
    "sphinx_rtd_theme >= 0.5.2, < 3.2.0",
    "sphinx-autodoc-typehints < 3.13.0",
    "sphinx-autoapi < 3.9.0",
    "astroid == 4.0.3",
]

extras = {"dev": dev_requirements}

setup(
    name="eDisGo",
    version="0.3.0",
    packages=find_packages(),
    url="https://github.com/openego/eDisGo",
    project_urls={
        "Documentation": "https://edisgo.readthedocs.io/",
        "Source": "https://github.com/openego/eDisGo",
        "Changelog": "https://edisgo.readthedocs.io/en/dev/whatsnew.html",
        "Issues": "https://github.com/openego/eDisGo/issues",
    },
    license="GNU Affero General Public License v3.0",
    # Maintainers first, then the largest contributors. The complete list is in
    # AUTHORS.md and CITATION.cff.
    author=(
        "Jonas Danke, Moritz Schlösser, Maike Held, birgits, AnyaHe, khelfen, "
        "mltja, gplssm, nesnoj, jaappedersen, Elias, boltbeard"
    ),
    author_email="jonas.danke@rl-institut.de",
    maintainer="Jonas Danke, Moritz Schlösser",
    maintainer_email="jonas.danke@rl-institut.de",
    description="A python package for distribution network analysis and optimization",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    # PyPSA 1.3 requires Python 3.11; the test matrix covers 3.11 to 3.13.
    python_requires=">=3.11,<3.14",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later "
        "(AGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=requirements,
    extras_require=extras,
    package_data={
        "edisgo.io.oedialect": ["LICENSE"],
        "edisgo": [
            os.path.join("config", "*.cfg"),
            os.path.join("equipment", "*.csv"),
            os.path.join("run", "presets", "*.yaml"),
        ],
        # The Julia OPF sources need their own key: edisgo/opf is a package of its
        # own, and package_data patterns are resolved relative to the package they
        # are listed under, so they cannot be reached from the "edisgo" entry above.
        # Without this, pm_optimize() is broken in any non-editable install (e.g.
        # eGo's "edisgo @ git+..."), because Main.jl never ships.
        "edisgo.opf": [
            os.path.join("eDisGo_OPF.jl", "**", "*"),
        ],
    },
)
