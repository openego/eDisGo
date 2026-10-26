.. _installation:

Installation
============

eDisGo is tested with **Python 3.10–3.12** (the conda environment files pin
``>=3.9, <=3.11``). The documentation is built with Python 3.11, which is a good
default choice.

User installation
-----------------

Install the latest release from PyPI. We highly recommend using a virtual
environment and its ``pip``:

.. code-block:: bash

    python -m pip install edisgo

Linux
~~~~~

The command above works out of the box on most Linux systems. If installation of
the geospatial stack (``geopandas``, ``shapely``, ``pyproj``, ``fiona``, ...)
fails, install those packages from ``conda-forge`` first (see the Windows
instructions) and then run ``pip install edisgo``.

Windows
~~~~~~~

On Windows we recommend `Anaconda <https://www.anaconda.com/download>`_ and
installing the geospatial stack from the ``conda-forge`` channel **before**
installing eDisGo. You can use the provided
`eDisGo_env.yml <https://github.com/openego/eDisGo/blob/dev/eDisGo_env.yml>`_:

.. code-block:: bash

    conda env create -f path/to/eDisGo_env.yml
    conda activate eDisGo_env

macOS
~~~~~

We have little experience with eDisGo on macOS. If you try it, please let us know
how it went.

Developer installation
----------------------

To work on the source code, clone the repository and install it in editable mode
with the development extras (linters, test tools, documentation tools):

.. code-block:: bash

    git clone https://github.com/openego/eDisGo.git
    cd eDisGo
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    pre-commit install   # install the git pre-commit hooks

See :ref:`contributing` for code standards, tests and how to build these docs.

.. _opf-requirements:

Additional requirements for the optimal power flow
--------------------------------------------------

The multi-period optimal power flow used for the flexibility optimisation
(:ref:`flexibility-opf`) is solved in `Julia <https://julialang.org>`_ via
`PowerModels.jl <https://lanl-ansi.github.io/PowerModels.jl/stable/>`_ with the
`Gurobi <https://www.gurobi.com>`_ and `Ipopt <https://github.com/jump-dev/Ipopt.jl>`_
solvers. These are **only** needed if you call
:meth:`~edisgo.edisgo.EDisGo.pm_optimize`; the rest of eDisGo works without them.

#. **Julia** — install Julia (the bundled ``eDisGo_OPF.jl`` package targets Julia
   ``1.12``) from the `Julia downloads page <https://julialang.org/downloads/>`_ and
   make sure ``julia`` is on your system ``PATH``.
#. **Solvers** — install the **Gurobi** solver and a valid licence following the
   `Gurobi getting-started guide
   <https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer>`_.
   Gurobi covers the default SOC method; **Ipopt** (a dependency of the Julia package)
   is additionally required for the non-convex method (``method="nc"``) and for
   ``warm_start=True``.

.. _grid-data-prerequisite:

Grid data
---------

Beyond a working eDisGo installation you need **grid topology data**. eDisGo uses
synthetic grids generated with `ding0 <https://github.com/openego/ding0>`_. You can

* download ready-made example grids from
  `Zenodo <https://zenodo.org/records/10405129>`_ (choose the latest record), or
* generate grids yourself following the
  `ding0 documentation <https://dingo.readthedocs.io/en/dev/usage_details.html#examples>`_.

See :ref:`data-sources` for details on grid data and on the scenario data obtained
from the OpenEnergy Platform.

License
-------

eDisGo is licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). Copyright (C) Reiner Lemoine Institut gGmbH. This program is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. See the
`GNU Affero General Public License <https://www.gnu.org/licenses/>`_ for details.
